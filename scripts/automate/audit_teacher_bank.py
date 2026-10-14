#!/usr/bin/env python3
"""Compare routed recurrent teachers with the original RL-Games player."""

import argparse
import json
from pathlib import Path

from teacher_bank import TeacherBank
import gym
import numpy as np
import torch
import yaml
from rl_games.algos_torch.players import PpoPlayerContinuous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teachers', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(917)
    # Deliberately shuffled routing; unequal group sizes; asynchronous resets.
    routing = torch.arange(27) % 10
    routing = routing[torch.randperm(len(routing))]
    bank = TeacherBank(args.teachers, routing)
    refs, statistics = {}, {}
    for i, model in bank.models.items():
        entry = bank.manifest['teachers'][i]
        cfg = yaml.safe_load(Path(entry['agent_config']).read_text())['params']
        cfg['config'].update({'device_name':'cpu', 'env_info':{
            'action_space':gym.spaces.Box(-1.,1.,shape=(6,),dtype=np.float32),
            'observation_space':gym.spaces.Box(-np.inf,np.inf,shape=(24,),dtype=np.float32),
            'agents':1, 'value_size':1}, 'player':{'deterministic':True}})
        player = PpoPlayerContinuous(cfg)
        player.restore(entry['checkpoint'])
        player.has_batch_dimension = True
        player.batch_size = len(bank.slots[i])
        player.init_rnn()
        refs[i] = player
        statistics[i] = {k:v.clone() for k,v in model.running_mean_std.state_dict().items()}
    max_error = 0.
    for step in range(12):
        obs = torch.empty((len(routing),24))
        reset = torch.zeros(len(routing),dtype=torch.bool)
        if step in (0, 4, 9):
            reset[step % 3::3] = True
        for i, slots in bank.slots.items():
            norm = bank.models[i].running_mean_std
            obs[slots] = norm.running_mean.float() + .3*norm.running_var.sqrt().float()*torch.randn(len(slots),24)
        actual = bank.act(obs, reset)
        for i, slots in bank.slots.items():
            for state in refs[i].states:
                state[:,reset[slots],:] = 0
            bound = bank.clip_obs[i]
            expected = refs[i].get_action(obs[slots].clamp(-bound,bound), is_deterministic=True)
            error = float((actual[slots]-expected).abs().max())
            max_error = max(max_error,error)
            if error > 1e-6:
                raise AssertionError(f'Teacher {i} routing/action/reset mismatch: {error}')
    for i, model in bank.models.items():
        for k,v in model.running_mean_std.state_dict().items():
            if not torch.equal(v, statistics[i][k]):
                raise AssertionError('Teacher normalizer mutated during labeling')
    try:
        bank.act(torch.zeros((len(routing),24)), torch.tensor([0,1]))
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid reset-mask shape/dtype accepted')
    result = {'status':'PASS','teachers':10,'env_slots':27,'steps':12,
              'max_action_error_vs_original_player':max_error,
              'checks':['shuffled task routing','unequal task batch sizes','asynchronous RNN resets',
                        'frozen per-teacher observation normalizers','deterministic clipped mean parity',
                        'reject malformed reset mask'],
              'limitation':'CPU checkpoint/interface parity only; does not establish simulator routing or DAgger training quality'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
