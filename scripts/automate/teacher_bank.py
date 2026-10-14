"""Frozen recurrent specialists for labeling student-visited observations.

This module deliberately does not import SRSA/Isaac: checkpoint auditing and
player parity can run without a simulator. Task routing is fixed per env slot.
"""

import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'rl_games_sil'))

import torch
import yaml
from rl_games.algos_torch.model_builder import ModelBuilder


def checked_file(path, expected_hash):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f'File changed since teacher manifest: {path}')
    return path


class TeacherBank:
    """Return each slot's teacher action while advancing its own RNN history.

    Call act on every step, including when only the student action is executed.
    reset_before is a bool mask marking the slots whose previous episode ended.
    Input is the raw 24D policy observation before observation normalization.
    Output is the clipped mean action before environment scaling and EMA.
    """

    def __init__(self, manifest_path, task_indices, device='cpu'):
        self.device = torch.device(device)
        self.manifest = json.loads(Path(manifest_path).read_text())
        if self.manifest.get('schema') != 'automate10.teacher_bank.v1':
            raise ValueError('Unsupported teacher bank schema')
        self.task_indices = torch.as_tensor(task_indices, dtype=torch.long, device=self.device).clone()
        if self.task_indices.ndim != 1 or self.task_indices.numel() == 0:
            raise ValueError('task_indices must be a nonempty one-dimensional vector')
        teachers = self.manifest['teachers']
        if len(teachers) != len(self.manifest['tasks']):
            raise ValueError('Incomplete teacher manifest')
        if self.task_indices.min() < 0 or self.task_indices.max() >= len(teachers):
            raise ValueError('Unknown task index')
        self.models, self.states, self.slots, self.clip_obs = {}, {}, {}, {}
        for index, entry in enumerate(teachers):
            if entry['task_index'] != index or entry['assembly_id'] != self.manifest['tasks'][index]:
                raise ValueError('Inconsistent task identity/order in manifest')
            slots = (self.task_indices == index).nonzero(as_tuple=True)[0]
            if not slots.numel():
                continue
            checkpoint = checked_file(entry['checkpoint'], entry['checkpoint_sha256'])
            agent = checked_file(entry['agent_config'], entry['agent_config_sha256'])
            cfg = yaml.safe_load(agent.read_text())['params']
            if entry['observation_dim'] != 24 or entry['action_dim'] != 6:
                raise ValueError('Teacher bank requires 24D observations and 6D actions')
            model = ModelBuilder().load(cfg).build({
                'actions_num':6, 'input_shape':(24,), 'num_seqs':slots.numel(),
                'value_size':1, 'normalize_input':cfg['config']['normalize_input'],
                'normalize_value':cfg['config']['normalize_value'],
            }).to(self.device)
            weights = torch.load(checkpoint, map_location=self.device, weights_only=False)
            model.load_state_dict(weights['model'], strict=True)
            if model.normalize_input and 'running_mean_std' in weights:
                model.running_mean_std.load_state_dict(weights['running_mean_std'], strict=True)
            model.eval().requires_grad_(False)
            self.models[index] = model
            self.slots[index] = slots
            self.states[index] = tuple(s.to(self.device) for s in model.get_default_rnn_state())
            self.clip_obs[index] = float(cfg['env'].get('clip_observations', math.inf))

    @torch.inference_mode()
    def act(self, observations, reset_before=None):
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        if obs.shape != (self.task_indices.numel(), 24) or not torch.isfinite(obs).all():
            raise ValueError('Expected finite raw policy observations [num_envs, 24]')
        if reset_before is None:
            reset = torch.zeros(len(obs), dtype=torch.bool, device=self.device)
        else:
            reset = torch.as_tensor(reset_before, device=self.device)
            if reset.dtype != torch.bool or reset.shape != (len(obs),):
                raise ValueError('reset_before must be a boolean mask [num_envs]')
        actions = torch.empty((len(obs), 6), dtype=obs.dtype, device=self.device)
        for index, model in self.models.items():
            slots = self.slots[index]
            for state in self.states[index]:
                state[:, reset[slots], :] = 0
            bound = self.clip_obs[index]
            result = model({'is_train':False, 'prev_actions':None,
                            'obs':obs[slots].clamp(-bound, bound),
                            'rnn_states':self.states[index]})
            self.states[index] = tuple(result['rnn_states'])
            actions[slots] = result['mus'].clamp(-1., 1.)
        if not torch.isfinite(actions).all():
            raise RuntimeError('Non-finite teacher action')
        return actions
