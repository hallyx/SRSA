"""Describe terminal depth for episodes failing full one-sided relaxed success."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


FIELDS = {
    'stable': 'relaxed_success_stable',
    'ever_stable': 'relaxed_success_episode',
    'official': 'success',
    'depth': 'relaxed_depth_fraction_final',
    'lateral_m': 'relaxed_lateral_error_final',
    'lateral_ok': 'relaxed_lateral_ok_final',
    'jam': 'relaxed_jam_final',
}


def stats(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {'n': 0, **{k: None for k in ('mean', 'sample_std', 'min', 'p10', 'p25', 'median', 'p75', 'p90', 'max')}}
    return dict(n=len(values), mean=float(values.mean()),
                sample_std=float(values.std(ddof=1)) if len(values) > 1 else None,
                min=float(values.min()), max=float(values.max()),
                **dict(zip(('p10', 'p25', 'median', 'p75', 'p90'),
                           map(float, np.quantile(values, [.1, .25, .5, .75, .9], method='linear')))))


def summarize(rows, task, seed):
    failed = [r for r in rows if not r['stable']]
    successful = [r for r in rows if r['stable']]
    reasons = {k: sum(r['failure_category'] == k for r in failed) for k in
               ('depth_not_met', 'lateral_after_depth', 'jam_after_depth_lateral', 'terminal_process_not_stable')}
    assert sum(reasons.values()) == len(failed)
    depth = np.array([r['depth'] for r in failed])
    bands = dict(le_0p5=int((depth <= .5).sum()),
                 gt_0p5_le_0p85=int(((depth > .5) & (depth <= .85)).sum()),
                 gt_0p85_lt_1=int(((depth > .85) & (depth < 1)).sum()),
                 ge_1=int((depth >= 1).sum()))
    assert sum(bands.values()) == len(failed)
    return dict(task=task, seed=seed, episodes=len(rows), successes=len(successful), failures=len(failed),
                failed_depth=stats(depth), successful_depth=stats([r['depth'] for r in successful]),
                all_depth=stats([r['depth'] for r in rows]), depth_bands=bands, failure_categories=reasons,
                failed_lateral_mm=stats([1000*r['lateral_m'] for r in failed]),
                failed_official_success=sum(r['official'] for r in failed),
                failed_ever_stable=sum(r['ever_stable'] for r in failed),
                failed_jam=sum(r['jam'] for r in failed),
                failed_depth_gt_2=sum(r['depth'] > 2 for r in failed))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    directory = args.paired.resolve()
    manifest = json.loads((directory/'manifest.json').read_text())
    result = json.loads((directory/'result.json').read_text())
    if result['failures'] or len(result['jobs']) != 20:
        raise ValueError('Expected 20 completed paired evaluation jobs')
    records, sources, task_summaries, seed_summaries = [], [], [], []
    for task in manifest['tasks']:
        task_records = []
        for seed in (5101, 5102):
            matches = [j for j in result['jobs'] if j['task'] == task and j['seed'] == seed]
            assert len(matches) == 1
            job = matches[0]
            assert job['state'] == 'completed'
            path = Path(job['output'])
            with h5py.File(path, 'r') as f:
                assert int(f.attrs['eval_seed']) == seed
                assert f.attrs['checkpoint_sha256'] == job['checkpoint_sha256']
                assert 'depth_fraction > 0.85' in f.attrs['relaxed_protocol']
                arrays = {k: f[v][:].reshape(-1) for k, v in FIELDS.items()}
            assert all(len(a) == 100 and np.isfinite(a).all() for a in arrays.values())
            for k in ('stable', 'ever_stable', 'official', 'lateral_ok', 'jam'):
                assert np.isin(arrays[k], [0, 1]).all()
            assert (arrays['depth'] >= 0).all()
            seed_records = []
            for index in range(100):
                row = dict(task=task, seed=seed, episode_row=index,
                           **{k: float(a[index]) for k, a in arrays.items()})
                process = row['depth'] > .85 and row['lateral_ok'] and not row['jam']
                assert not row['stable'] or (process and row['ever_stable'])
                if row['stable']:
                    category = 'success'
                elif row['depth'] <= .85:
                    category = 'depth_not_met'
                elif not row['lateral_ok']:
                    category = 'lateral_after_depth'
                elif row['jam']:
                    category = 'jam_after_depth_lateral'
                else:
                    category = 'terminal_process_not_stable'
                row.update(failure_category=category, source=str(path), checkpoint_sha256=job['checkpoint_sha256'])
                seed_records.append(row)
            assert sum(r['stable'] for r in seed_records) == job['metrics']['relaxed_stable']
            seed_summaries.append(summarize(seed_records, task, seed))
            task_records.extend(seed_records)
            sources.append(dict(task=task, seed=seed, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        task_summaries.append(summarize(task_records, task, 'pooled'))
        records.extend(task_records)
    overall = summarize(records, 'all_tasks', 'pooled')
    assert overall['episodes'] == 2000 and overall['successes'] == 310 and overall['failures'] == 1690
    summary = dict(definition='Failure iff terminal relaxed_success_stable=0; progress=raw terminal depth fraction, not clipped.',
                   quantile_method='numpy linear; sample standard deviation ddof=1',
                   aggregation='Pooled episodes; different tasks have different failure counts. Not task-macro progress.',
                   sources=sources, tasks=task_summaries, by_seed=seed_summaries, overall=overall)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    write_csv(out/'episodes.csv', records)
    extremes = sorted([r for r in records if not r['stable'] and r['depth'] > 2],
                      key=lambda r: r['depth'], reverse=True)
    if extremes:
        write_csv(out/'failed_depth_gt_200pct.csv', extremes)
    flat = []
    for r in task_summaries + seed_summaries + [overall]:
        flat.append(dict(task=r['task'], seed=r['seed'], episodes=r['episodes'], failures=r['failures'],
                         **{f'failed_depth_{k}': v for k, v in r['failed_depth'].items()},
                         **r['depth_bands'], **r['failure_categories'],
                         failed_lateral_median_mm=r['failed_lateral_mm']['median'],
                         failed_official_success=r['failed_official_success'], failed_ever_stable=r['failed_ever_stable']))
    write_csv(out/'task_and_seed_statistics.csv', flat)
    lines = ['## 失败回合的 final progress 详细统计', '',
             '失败按完整新版 `relaxed_success_stable=0` 筛选，共 **1,690/2,000 回合**。final progress 为 `relaxed_depth_fraction_final`：末态插入深度/目标深度；下表以百分比展示，保留超过 100% 的值，不裁剪。不是回合最大深度，也不是毫米深度。', '',
             '### 逐任务失败深度分布', '',
             '均值、中位数和分位数仅在该任务失败回合内计算；标准差为样本标准差，分位数采用线性插值。', '',
             '| 任务 | 失败数 | 均值 | 标准差 pp | 最小 | P10 | P25 | 中位数 | P75 | P90 | 最大 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in task_summaries + [overall]:
        d = r['failed_depth']
        cells = [f"{100*d[k]:.2f}" for k in ('mean', 'sample_std', 'min', 'p10', 'p25', 'median', 'p75', 'p90', 'max')]
        lines.append('| '+('全部失败回合' if r['task'] == 'all_tasks' else r['task'])+' | '+str(r['failures'])+' | '+' | '.join(cells)+' |')
    lines += ['', '全部失败回合一行按回合合并，失败多的任务权重较高，不能当作十任务等权宏平均。深度比例为坐标计算量，超过 100% 不足以单独证明物理穿透或正确就位。', '',
              '### 失败深度区间计数', '', '每格为计数及占该行失败回合比例；四区间互斥且穷尽，恰好 85% 属于不达标。', '',
              '| 任务 | ≤50% | >50% 且 ≤85% | >85% 且 <100% | ≥100% |', '|---|---:|---:|---:|---:|']
    for r in task_summaries + [overall]:
        cells = [f"{n} ({100*n/r['failures']:.1f}%)" for n in r['depth_bands'].values()]
        lines.append('| '+('全部失败回合' if r['task'] == 'all_tasks' else r['task'])+' | '+' | '.join(cells)+' |')
    lines += ['', '### 终态失败条件分解', '',
              '按顺序分类：深度≤85%；深度>85%但横向不合格；深度和横向均合格但 jam；末步条件全部合格却未满足终态连续三步。类别互斥且穷尽，是末态条件分解，不是故障根因。末步条件全合格但 stable=0 的分类直接使用已保存的 stable 值，不重建缺失轨迹。', '',
              '| 任务 | 深度不达标 | 达深度但横向失败 | 深度横向合格但 jam | 末步合格但稳定不足 | 失败横向中位数 mm |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in task_summaries + [overall]:
        cells = list(map(str, r['failure_categories'].values()))
        lines.append('| '+('全部失败回合' if r['task'] == 'all_tasks' else r['task'])+' | '+' | '.join(cells)+f" | {r['failed_lateral_mm']['median']:.2f} |")
    lines += ['', 'jam 与横向失败可以同时出现。先分横向的表不重复计 jam；所以 jam 列为零不表示原始 jam 标志全为零。所有失败回合的原始 jam 总数另存 JSON。', '',
              '### 评估 seed 和曾经成功对照', '',
              '各 seed 单独对失败回合求深度均值/中位数，单位为目标深度百分比。两 seed 是同一训练模型的评估重复。曾稳定成功指 `relaxed_success_episode=1` 但终态 stable=0；官方曾成功使用独立官方锁存标签。', '',
              '| 任务 | 5101 失败数 | 5101 均值/中位数 | 5102 失败数 | 5102 均值/中位数 | 失败中曾稳定成功 | 失败中官方曾成功 |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in task_summaries:
        a,b = [s for s in seed_summaries if s['task'] == r['task']]
        values = [f"{100*s['failed_depth']['mean']:.2f}/{100*s['failed_depth']['median']:.2f}" for s in (a,b)]
        lines.append(f"| {r['task']} | {a['failures']} | {values[0]} | {b['failures']} | {values[1]} | {int(r['failed_ever_stable'])} | {int(r['failed_official_success'])} |")
    lines += ['', '成功回合、全部回合与失败回合的独立深度统计，以及逐 seed 的完整分位数，均保存在 `summary.json`；没有成功样本的任务用 null 表示成功深度统计，不能填 0。', '']
    lines += ['### 极端深度值及解读', '',
              f"失败回合中有 **{len(extremes)} 个**末态深度比例严格超过 200%。这是用于检查长尾的描述性阈值，不是成功或安全门槛。原始值不删除、不截断；全部明细见 `failed_depth_gt_200pct.csv`。", '',
              '00186、00271、00726 的失败深度中位数均超过 85%，但仍可能横向不合格或终态稳定不足。01125 的失败均值接近 95%，中位数只有约 63%，说明均值被极端尾部拉高；不能把高均值解读为普遍接近完成。', '',
              '| 任务 | seed | HDF5 行索引从 0 起 | final progress % | 横向误差 mm |', '|---|---:|---:|---:|---:|']
    for r in extremes[:10]:
        lines.append(f"| {r['task']} | {r['seed']} | {r['episode_row']} | {100*r['depth']:.2f} | {1000*r['lateral_m']:.2f} |")
    lines += ['', '上表列最大十个异常值。由 plug/socket 根位姿计算的深度可能因离轴、物体位置异常等产生大值；仅凭这些末态记录不能确定原因，也不能把 >100% 等同于实际正确插入或碰撞穿透。', '']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    y = np.arange(len(task_summaries))
    med = np.array([r['failed_depth']['median']*100 for r in task_summaries])
    low = np.array([r['failed_depth']['p10']*100 for r in task_summaries])
    high = np.array([r['failed_depth']['p90']*100 for r in task_summaries])
    axes[0].errorbar(med, y, xerr=[med-low, high-med], fmt='o', color='#235789', capsize=3)
    axes[0].axvline(85, color='#c14953', linestyle='--', label='Depth threshold 85%')
    axes[0].set_yticks(y, [r['task'] for r in task_summaries])
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Final depth / target depth (%)')
    axes[0].set_title('Failed episodes: median and P10-P90')
    axes[0].legend(loc='lower left', fontsize=8)
    left = np.zeros(len(y))
    for key, label, color in zip(task_summaries[0]['depth_bands'],
                                 ['<=50%', '>50% to <=85%', '>85% to <100%', '>=100%'],
                                 ['#dce5ef', '#a4bdd3', '#efba76', '#c14953']):
        values = np.array([100*r['depth_bands'][key]/r['failures'] for r in task_summaries])
        axes[1].barh(y, values, left=left, color=color, label=label)
        left += values
    axes[1].set_yticks(y, [f"{r['task']} (n={r['failures']})" for r in task_summaries])
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Share of failed episodes (%)')
    axes[1].set_title('Depth bands within failures')
    axes[1].legend(loc='upper center', bbox_to_anchor=(.5, -.13), ncol=2, fontsize=8)
    fig.savefig(out/'failed_final_progress.png', dpi=180)
    fig.savefig(out/'failed_final_progress.svg')
    plt.close(fig)
    (out/'report_section.md').write_text('\n'.join(lines))
    print(json.dumps(overall, ensure_ascii=False))


if __name__ == '__main__':
    main()
