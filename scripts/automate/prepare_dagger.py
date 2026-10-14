#!/usr/bin/env python3
"""Audit final specialists and export a reproducible DAgger preparation bundle."""

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TASKS = ['00062', '00186', '00256', '00271', '00726', '01079', '01029', '01092', '01102', '01125']


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    radius = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return [max(0., center-radius), min(1., center+radius)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--queue', type=Path, required=True)
    ap.add_argument('--eval-dirs', type=Path, nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    queue = args.queue.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((queue/'manifest.json').read_text())
    status = json.loads((queue/'status.json').read_text())
    if manifest['tasks'] != TASKS:
        raise ValueError('Not the historical ten-task set containing 00271')
    jobs = {j['assembly_id']: j for j in status['jobs']}
    evaluations = {}
    for directory in args.eval_dirs:
        result = json.loads((directory/'result.json').read_text())
        if result['failures']:
            raise ValueError(f'Evaluation failures in {directory}')
        for j in result['jobs']:
            key = (j['task'], j['seed'])
            if key in evaluations or j['state'] != 'completed':
                raise ValueError(f'Duplicate or incomplete evaluation {key}')
            evaluations[key] = j

    rows, teachers = [], []
    for task_index, task in enumerate(TASKS):
        job = jobs[task]
        if job['state'] != 'completed' or job['exit_code'] != 0:
            raise ValueError(f'Incomplete training {task}')
        run = ROOT/'logs/rl_games/Assembly'/job['experiment_name']
        candidates = [p for p in (run/'nn').glob('last_Assembly_ep_1500*.pth') if '_rew__' not in p.name]
        if not candidates:
            raise ValueError(f'No final checkpoint for {task}')
        checkpoint = max(candidates, key=lambda p: p.stat().st_mtime)
        checksum = sha256(checkpoint)
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if ckpt['epoch'] != 1500 or ckpt['frame'] != 12_000_000:
            raise ValueError(f'Unexpected final budget for {task}')
        weights = ckpt['model']
        if weights['running_mean_std.running_mean'].numel() != 24 or weights['a2c_network.mu.weight'].shape[0] != 6:
            raise ValueError(f'Incompatible teacher dimensions {task}')
        if any(not torch.isfinite(v).all() for v in weights.values() if torch.is_floating_point(v)):
            raise ValueError(f'Non-finite teacher weights {task}')
        counts = {'episodes': 0, 'official': 0, 'relaxed_stable': 0, 'relaxed_episode': 0, 'terminal_depth_gt_085': 0, 'jam_final': 0}
        sources = []
        for seed in (5101, 5102):
            j = evaluations[(task, seed)]
            if j['checkpoint_sha256'] != checksum:
                raise ValueError(f'Checkpoint mismatch {task} seed {seed}')
            path = Path(j['output']).resolve()
            with h5py.File(path, 'r') as hf:
                if hf.attrs.get('checkpoint_sha256') != checksum or int(hf.attrs.get('eval_seed', -1)) != seed:
                    raise ValueError(f'HDF5 provenance mismatch {path}')
                if hf.attrs.get('policy_action') != 'deterministic mean' or 'depth_fraction > 0.85' not in hf.attrs.get('relaxed_protocol', ''):
                    raise ValueError(f'HDF5 evaluation protocol mismatch {path}')
                fields = {'official':'success', 'relaxed_stable':'relaxed_success_stable', 'relaxed_episode':'relaxed_success_episode', 'jam_final':'relaxed_jam_final'}
                observed = {'episodes': 100}
                for name, dataset in fields.items():
                    a = hf[dataset][:].reshape(-1)
                    if len(a) != 100 or not np.isin(a, [0, 1]).all():
                        raise ValueError(f'Invalid episode flags {path}:{dataset}')
                    observed[name] = int(a.sum())
                depth = hf['relaxed_depth_fraction_final'][:].reshape(-1)
                if len(depth) != 100 or not np.isfinite(depth).all():
                    raise ValueError(f'Invalid depth values {path}')
                observed['terminal_depth_gt_085'] = int((depth > .85).sum())
                for name, value in j['metrics'].items():
                    if observed[name] != value:
                        raise ValueError(f'JSON/HDF5 metric mismatch {path}:{name}')
                for name in counts:
                    counts[name] += observed[name]
            sources.append({'seed':seed, 'path':str(path), 'sha256':sha256(path), 'metrics':observed})
        k, n = counts['relaxed_stable'], counts['episodes']
        quality = 'zero_success_observed' if k == 0 else ('below_80pct' if k/n < .8 else 'at_least_80pct')
        row = {'task':task, 'epoch':ckpt['epoch'], 'training_transitions':ckpt['frame'], **counts,
               'relaxed_rate':k/n, 'relaxed_ci95':wilson(k,n), 'quality':quality}
        rows.append(row)
        teacher = {'task_index':task_index, 'assembly_id':task, 'checkpoint':str(checkpoint.resolve()),
                   'checkpoint_sha256':checksum, 'agent_config':str((run/'params/agent.yaml').resolve()),
                   'agent_config_sha256':sha256(run/'params/agent.yaml'),
                   'env_config':str((run/'params/env.yaml').resolve()), 'env_config_sha256':sha256(run/'params/env.yaml'),
                   'train_seed':manifest['seed'], 'epoch':ckpt['epoch'], 'frame':ckpt['frame'],
                   'observation_dim':24, 'action_dim':6, 'normalizer':'checkpoint model.running_mean_std; frozen',
                   'metrics':row, 'evaluations':sources}
        teachers.append(teacher)
    summary = {'schema':'automate10.results.v1', 'created_at':datetime.now().astimezone().isoformat(),
               'method':'AutoMate scratch specialists (dense reward + SBC)', 'queue':str(queue),
               'selection':'final epoch 1500; no post-hoc best-checkpoint selection',
               'evaluation':'deterministic; seeds 5101/5102 x100; depth>0.85, lateral/no-jam, 3 terminal stable steps',
               'training_seed_count':1, 'total_training_transitions':sum(r['training_transitions'] for r in rows),
               'macro_relaxed_rate':sum(r['relaxed_rate'] for r in rows)/len(rows),
               'macro_official_rate':sum(r['official']/r['episodes'] for r in rows)/len(rows),
               'zero_success_tasks':[r['task'] for r in rows if r['relaxed_stable']==0], 'tasks':rows}
    save(output/'summary.json', summary)
    with (output/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    bank = {'schema':'automate10.teacher_bank.v1', 'tasks':TASKS, 'selection':summary['selection'],
            'action_semantics':'clipped deterministic mean in [-1,1], before environment action scaling/EMA',
            'frozen':True, 'teachers':teachers}
    save(output/'teacher_bank.json', bank)
    weak = [r['task'] for r in rows if r['relaxed_rate'] < .8]
    pilot = [r['task'] for r in sorted(rows, key=lambda r:(-r['relaxed_rate'],r['task']))[:2]]
    cfg = {'schema':'automate10.dagger_preparation.v1', 'stage':'PREPARED_NOT_TRAINING',
           'teacher_bank':str(output/'teacher_bank.json'), 'teacher_bank_sha256':sha256(output/'teacher_bank.json'),
           'tasks':TASKS, 'train_seed':1,
           'student':{'observation_dim':24, 'geometry_dim':64, 'action_dim':6, 'lstm_hidden':256, 'lstm_layers':2,
                      'mlp':[512,256,128,64], 'normalizer':'own training-data statistics; never borrow one teacher normalizer',
                      'geometry_checkpoint':None, 'generalization_claim':'one policy over these ten training tasks'},
           'collection':{'num_envs':250, 'envs_per_task':25, 'physical_grasp':True, 'sbc':False,
                         'geometry_scaling':False, 'force_observations':False, 'teacher_action':'deterministic clipped mean',
                         'reset_contract':'same frozen nominal evaluation distribution; log actual reset parameters'},
           'bc':{'pilot_successful_episodes_per_task':100, 'full_successful_episodes_per_task':5000,
                 'pilot_max_attempts_per_task':2000, 'full_max_attempts_per_task':50000,
                 'success_filter':'relaxed_success_stable_depth_gt_085', 'split_unit':'whole episode',
                 'validation_fraction':.1, 'task_balanced':True, 'failure_policy':'report missing quota; do not loop indefinitely or silently drop task'},
           'dagger':{'rounds':5, 'episodes_per_task_per_round':256, 'keep_failed_episodes':True,
                     'student_executes':True, 'teacher_queries_every_step':True, 'teacher_beta':0.,
                     'teacher_rnn':'advance on the same student-visited observation history; reset only done slots',
                     'aggregation':'BC plus all prior DAgger rounds, task-balanced complete sequences',
                     'train_on':'teacher_action labels; store executed student action separately',
                     'optimizer':'Adam', 'learning_rate':.0001, 'grad_norm':1., 'max_updates_per_round':10000,
                     'freeze_student_per_collection_round':True, 'selection':'independent simulator validation, retain Final and Selected'},
           'gpu_schedule':{'gpu0':'250-env collection with frozen student version', 'gpu1':'offline fitting or independent validation after collection barrier'},
           'evaluation':{'validation_seeds':[4101,4102], 'historical_test_seeds':[5101,5102],
                         'fresh_test_seeds':[6101,6102], 'episodes_per_seed_per_task':100,
                         'primary':'relaxed stable; depth>0.85; lateral/no-jam/3 steps unchanged'},
           'readiness':{'all_ten_quality_ready':not weak, 'provisional_teacher_quality_threshold':.8,
                        'below_threshold_tasks':weak, 'zero_success_tasks':summary['zero_success_tasks'],
                        'suggested_engineering_pilot_tasks':pilot, 'pilot_selection_uses_observed_results':True,
                        'remaining':['geometry encoder and cache', 'BC/DAgger collectors and student trainer',
                                     'simulator teacher-routing parity and recurrent resets', 'weak-teacher diagnosis']}}
    save(output/'dagger_preparation.json', cfg)
    lines = ['# AutoMate 十任务队列结果与 DAgger 准备', '', '## Material Passport', '',
             '- Origin: academic-research-suite / experiment-agent',
             '- Verification: final checkpoint epoch/frame/shape/finiteness and raw evaluation HDF5 audited.',
             '- Scope: one training seed; evaluation seeds are not training repeats.', '',
             f"训练完成：10/10；每任务 1500 轮、1200 万步；总计 1.2 亿训练步。", '',
             '| 任务 | relaxed 成功 | 成功率 | Wilson 95% 区间 | official 成功率 |',
             '|---|---:|---:|---:|---:|']
    for r in rows:
        lo, hi = r['relaxed_ci95']
        lines.append(f"| {r['task']} | {r['relaxed_stable']}/{r['episodes']} | {r['relaxed_rate']:.1%} | {lo:.1%}–{hi:.1%} | {r['official']/r['episodes']:.1%} |")
    lines += ['', f"十任务宏平均 relaxed 成功率：{summary['macro_relaxed_rate']:.2%}。", '',
              '上述区间只描述给定模型的评估回合不确定性，不包含训练 seed 方差。测试结果已观察，用它选出的 pilot 只用于工程验证。', '',
              '## DAgger 准备', '',
              f"建议先以 {'、'.join(pilot)} 验证采集与教师标注链路；完整配置保留全部十任务。",
              f"零成功任务：{'、'.join(summary['zero_success_tasks']) or '无'}。DAgger 的监督依赖教师在 student 访问状态上的动作质量；当前低分教师不支持承诺十任务高成功率。", '',
              '训练路线为 BC → student 执行、对应 teacher 标注 → 聚合全部轮次数据 → 更新同一个 student；使用 relaxed 成功示范做 BC，DAgger 保留成功和失败回合。',
              '教师 bank、student 的几何编码和训练器准备状态详见 teacher_bank.json 与 dagger_preparation.json。80% 为工程准备参考线，并非论文规定或用户验收门槛。', '',
              '[AutoMate 方法来源](https://arxiv.org/html/2407.08028v1#S5.SS4)；现有 docs/automate_dagger_10task_plan.md 中的旧预算和 official 主指标由本次准备协议更新。', '']
    (output/'report.md').write_text('\n'.join(lines))
    print(json.dumps({'output':str(output), 'macro_relaxed':summary['macro_relaxed_rate'],
                      'zero_success':summary['zero_success_tasks'], 'pilot_tasks':pilot}, ensure_ascii=False))


if __name__ == '__main__':
    main()
