#!/usr/bin/env python3
"""Compare official, original relaxed, and one-sided relaxed success on paired rollouts."""

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

TASKS = ('00062','00186','00256','00271','00726','01079','01029','01092','01102','01125')


def interval(k, n):
    z = 1.959963984540054
    p = k/n
    d = 1+z*z/n
    c = (p+z*z/(2*n))/d
    r = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [max(0.,c-r), min(1.,c+r)]


def load(path):
    with h5py.File(path,'r') as f:
        needed = {'success','relaxed_original_success_stable','relaxed_success_stable',
                  'relaxed_original_success_episode','relaxed_success_episode',
                  'relaxed_depth_fraction_final','relaxed_lateral_error_final','relaxed_jam_final',
                  'relaxed_original_depth_ok_final','relaxed_lateral_ok_final','relaxed_contact_final'}
        if not needed.issubset(f.keys()):
            raise ValueError(f'Missing paired metrics in {path}: {needed-set(f.keys())}')
        if 'depth_fraction > 0.85' not in f.attrs.get('relaxed_protocol','') or 'height_window' not in f.attrs.get('original_relaxed_protocol',''):
            raise ValueError(f'Metric definition mismatch in {path}')
        arrays = {name:f[name][:].reshape(-1) for name in needed}
        attrs = dict(f.attrs)
        poses = {name:f[name][:] for name in ('fixed_asset_pose','held_asset_pose')}
    if any(len(a)!=100 for a in arrays.values()):
        raise ValueError(f'Expected 100 complete episodes in {path}')
    return arrays, attrs, poses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired',type=Path,required=True)
    parser.add_argument('--previous-evals',type=Path,nargs='*',default=[])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    paired=args.paired.resolve()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    jobs=json.loads((paired/'result.json').read_text())
    if jobs['failures'] or len(jobs['jobs'])!=20:
        raise ValueError('Paired evaluator is incomplete')
    previous={}
    for folder in args.previous_evals:
        result=json.loads((folder/'result.json').read_text())
        if result['failures']:
            raise ValueError(f'Previous evaluation incomplete: {folder}')
        previous.update({(j['task'],j['seed']):j for j in result['jobs']})
    rows=[];seed_rows=[];cross_run=[]
    for task in TASKS:
        collected={k:[] for k in ('official','original','modified','original_episode','modified_episode','depth',
                                   'lateral','jam','original_depth_ok','lateral_ok','contact')}
        for seed in (5101,5102):
            j=next(x for x in jobs['jobs'] if x['task']==task and x['seed']==seed)
            if j['state']!='completed':raise ValueError(f'Incomplete {task} seed {seed}')
            a,attrs,poses=load(Path(j['output']))
            if int(attrs.get('eval_seed',-1))!=seed or attrs.get('checkpoint_sha256')!=j['checkpoint_sha256']:
                raise ValueError(f'Provenance mismatch {task} seed {seed}')
            m={'official':'success','original':'relaxed_original_success_stable','modified':'relaxed_success_stable',
               'original_episode':'relaxed_original_success_episode','modified_episode':'relaxed_success_episode',
               'depth':'relaxed_depth_fraction_final','lateral':'relaxed_lateral_error_final','jam':'relaxed_jam_final',
               'original_depth_ok':'relaxed_original_depth_ok_final','lateral_ok':'relaxed_lateral_ok_final',
               'contact':'relaxed_contact_final'}
            for key,name in m.items():collected[key].append(a[name])
            seed_rows.append({'task':task,'seed':seed,'official':int(a['success'].sum()),
                              'original_relaxed':int(a['relaxed_original_success_stable'].sum()),
                              'modified_relaxed':int(a['relaxed_success_stable'].sum())})
            old=previous.get((task,seed))
            if old:
                with h5py.File(old['output'],'r') as f:
                    same_success=np.array_equal(f['success'][:].reshape(-1),a['success'])
                    same_modified=np.array_equal(f['relaxed_success_stable'][:].reshape(-1),a['relaxed_success_stable'])
                    max_pose_delta=max(float(np.abs(f[n][:]-poses[n]).max()) for n in poses)
                cross_run.append({'task':task,'seed':seed,'same_official':same_success,
                                  'same_modified_relaxed':same_modified,'max_pose_delta_m':max_pose_delta})
        v={key:np.concatenate(parts) for key,parts in collected.items()}
        flags={key:v[key].astype(bool) for key in ('official','original','modified','original_episode','modified_episode',
                                                  'jam','original_depth_ok','lateral_ok','contact')}
        depth_gt=v['depth']>.85
        within=flags['original_depth_ok']
        lateral=flags['lateral_ok']
        process=depth_gt & lateral & ~flags['jam']
        if np.any(flags['original'] & ~flags['modified']):
            raise ValueError(f'Original relaxed not a subset of modified for {task}')
        rows.append({'task':task,'episodes':200,
          'official':int(flags['official'].sum()), 'original_relaxed':int(flags['original'].sum()),
          'modified_relaxed':int(flags['modified'].sum()),
          'original_episode':int(flags['original_episode'].sum()),
          'modified_episode':int(flags['modified_episode'].sum()),
          'newly_counted':int((flags['modified'] & ~flags['original']).sum()),
          'modified_success_depth_ge_1':int((flags['modified'] & (v['depth']>=1.)).sum()),
          'official_only_vs_modified':int((flags['official'] & ~flags['modified']).sum()),
          'modified_only_vs_official':int((flags['modified'] & ~flags['official']).sum()),
          'terminal_depth_gt_085':int(depth_gt.sum()),
          'terminal_original_depth_window':int(within.sum()),
          'terminal_lateral_ok':int(lateral.sum()),
          'terminal_contact':int(flags['contact'].sum()),
          'terminal_jam':int(flags['jam'].sum()),
          'terminal_depth_fail':int((~depth_gt).sum()),
          'terminal_lateral_fail_after_depth':int((depth_gt & ~lateral).sum()),
          'terminal_stability_fail_after_process':int((process & ~flags['modified']).sum()),
          'terminal_depth_ge_1':int((v['depth']>=1.).sum()),
          'depth_median':float(np.median(v['depth'])),
          'depth_p10':float(np.quantile(v['depth'],.1)),
          'depth_p90':float(np.quantile(v['depth'],.9)),
          'lateral_mm_median':float(np.median(v['lateral'])*1000),
          'official_ci95':interval(int(flags['official'].sum()),200),
          'original_ci95':interval(int(flags['original'].sum()),200),
          'modified_ci95':interval(int(flags['modified'].sum()),200),
        })
    aggregate={'schema':'automate10.success_definition_comparison.v1','paired_eval':str(paired),
               'training_seeds':1,'eval_seeds':[5101,5102],'episodes_per_task':200,
               'metrics':{'official':'AutoMate episode-latched success',
                          'original_relaxed':'FACA terminal stable; height window 0<height<target; depth>=0.85; lateral/no jam; 3 steps',
                          'modified_relaxed':'FACA terminal stable; depth>0.85; lateral/no jam; 3 steps'},
               'overall':{key:sum(row[key] for row in rows) for key in ('official','original_relaxed','modified_relaxed','newly_counted')},
               'cross_run_identical':all(x['same_official'] and x['same_modified_relaxed'] and x['max_pose_delta_m']<1e-6 for x in cross_run),
               'cross_run_checks':cross_run,'tasks':rows,'by_seed':seed_rows}
    (output/'comparison.json').write_text(json.dumps(aggregate,indent=2,ensure_ascii=False)+'\n')
    lines=['# 十任务专家成功口径配对分析','',
           '同一最终 checkpoint、同一 seed 5101/5102，各 100 回合；一个 rollout 同时记录三种口径。训练仅 seed 1。','',
           '| 任务 | 原始 AutoMate | 原版 relaxed | 修改后 relaxed | 新增成功 | 深度 >0.85 | 原版深度窗口 | 横向合格 | 终态 jam |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['task']} | {row['official']}/200 ({row['official']/2:.1f}%) | {row['original_relaxed']}/200 ({row['original_relaxed']/2:.1f}%) | {row['modified_relaxed']}/200 ({row['modified_relaxed']/2:.1f}%) | {row['newly_counted']} | {row['terminal_depth_gt_085']} | {row['terminal_original_depth_window']} | {row['terminal_lateral_ok']} | {row['terminal_jam']} |")
    lines += ['',f"十任务平均：AutoMate {aggregate['overall']['official']/20:.1f}%；原版 relaxed {aggregate['overall']['original_relaxed']/20:.1f}%；修改后 relaxed {aggregate['overall']['modified_relaxed']/20:.1f}%。",'',
              f"与之前修改后 relaxed 评测逐回合一致：{aggregate['cross_run_identical']}（对照 {len(cross_run)}/20 个评测文件）。",'',
              '“新增成功”是同一回合修改后成功而原版失败；深度、横向、jam 三列为终态快照，不能单独替代连续 3 步成功判定。',
              '每任务 Wilson 区间、按 seed 拆分、深度分位数、终态失败分解、逐评测文件配对验证详见 comparison.json。','',
              '原版 relaxed 依 FACA tdmpc2/envs/isaaclab.py：height_window_ok 且 depth_fraction>=0.85；修改后深度严格 >0.85 且无高度上限；横向、无 jam 和连续 3 步规则完全一致。','']
    (output/'comparison.md').write_text('\n'.join(lines))
    print(json.dumps(aggregate['overall']))


if __name__=='__main__':main()
