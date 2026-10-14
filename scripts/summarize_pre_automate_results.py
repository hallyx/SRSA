"""Read historical FACA artifacts; export results without running simulation."""
import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--faca-root', type=Path, default=Path('/media/gpuserver/data2t/hx/github/FACA'))
    parser.add_argument('--output-dir', type=Path, default=Path('docs/pre_automate_results_20261010'))
    args = parser.parse_args()
    root, out = args.faca_root, args.output_dir
    sources = {}

    def read(path):
        data = path.read_bytes()
        sources[str(path.relative_to(root))] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    size = read(root / 'logs/size_condition_v3/20260912_01079_gpu0/campaign/summary.json')
    size_rows = []
    for group in ('train', 'interpolation', 'clearance_holdout', 'extrapolation'):
        for arm in ('A', 'B', 'C'):
            values = [seed['arms'][arm][group]['macro_success'] for seed in size['seeds'].values()]
            assert len(values) == 3
            size_rows.append(dict(group=group, arm=arm, training_seeds=3,
                                  mean_pct=100 * statistics.mean(values),
                                  sample_std_pp=100 * statistics.stdev(values)))
    ood = read(root / 'logs/size_condition_v3/20260917_01079_size_ood_parameter_eval/eval/summary.json')
    tight = read(root / 'logs/size_condition_v3/20260916_01079_tight_vs_mixed/summary.json')
    campaign = root / 'logs/wrist_wrench_v3/tasksize10_v2_1_m0_m3_mt_true_seed1_matched_20260920_1710'
    protocol = read(campaign / 'campaign/protocol.json')
    task_order = protocol['task_order']
    arm_rows, task_rows = [], []
    episode_files = set()
    unique_episodes = 0
    for split in ('seen', 'held_out'):
        for role in ('selected', 'final'):
            for arm in ('m0', 'm1', 'm2', 'm3'):
                directory = campaign / 'evaluation/test' / split / role / arm
                files = sorted(directory.glob('seed_*/episodes.jsonl'))
                reused = False
                if not files and arm == 'm2' and role == 'final':
                    directory = campaign / 'evaluation/test' / split / 'selected' / arm
                    files = sorted(directory.glob('seed_*/episodes.jsonl'))
                    reused = True
                assert len(files) == 2, (split, role, arm, files)
                counts = defaultdict(lambda: [0, 0])
                depths = defaultdict(lambda: [0, 0])
                checkpoint_hashes = set()
                for path in files:
                    data = path.read_bytes()
                    sources[str(path.relative_to(root))] = hashlib.sha256(data).hexdigest()
                    records = [json.loads(line) for line in data.splitlines() if line.strip()]
                    assert len(records) == 1000, path
                    if path not in episode_files:
                        unique_episodes += len(records)
                        episode_files.add(path)
                    result = read(path.with_name('result.json'))
                    local_counts = defaultdict(lambda: [0, 0])
                    for row in records:
                        task = row['assembly_id']
                        success = row['metrics']['relaxed_success_stable']
                        assert success in (0, 1), row
                        counts[task][0] += int(success)
                        counts[task][1] += 1
                        local_counts[row['condition_id']][0] += int(success)
                        local_counts[row['condition_id']][1] += 1
                        depths[task][0] += int(row['metrics']['depth_fraction'] > .85)
                        depths[task][1] += 1
                        checkpoint_hashes.add(row['checkpoint_sha256'])
                    for condition, (successes, episodes) in local_counts.items():
                        assert result['condition_counts'][condition] == episodes
                        assert result['condition_relaxed_success'][condition] == successes
                assert len(checkpoint_hashes) == 1
                assert set(counts) == set(task_order)
                rates = []
                depth_rates = []
                for task in task_order:
                    successes, episodes = counts[task]
                    assert episodes == 200
                    rates.append(successes / episodes)
                    depth_rates.append(depths[task][0] / episodes)
                    task_rows.append(dict(split=split, role=role, arm=arm, task=task,
                                          successes=successes, episodes=episodes,
                                          relaxed_bounded_pct=100 * rates[-1],
                                          terminal_depth_only_pct=100 * depth_rates[-1]))
                arm_rows.append(dict(split=split, role=role, arm=arm, episodes=2000,
                                     macro_pct=100 * statistics.mean(rates),
                                     worst_task_pct=100 * min(rates),
                                     tasks_at_least_80pct=sum(rate >= .8 for rate in rates),
                                     terminal_depth_only_pct=100 * statistics.mean(depth_rates),
                                     checkpoint_sha256=next(iter(checkpoint_hashes)),
                                     final_reused_selected=reused))
    summary = dict(scope='Historical pre-AutoMate training; original bounded relaxed evaluation',
                   unique_tasksize10_episode_files=len(episode_files),
                   unique_tasksize10_episodes=unique_episodes,
                   size_abc=size_rows, size_ood=ood['primary_aggregate'], tight_vs_mixed=tight,
                   tasksize10=arm_rows,
                   terminal_depth_only_warning='Depth > .85 only; excludes lateral, no-jam and 3-step stability. Not full unbounded relaxed success.',
                   source_sha256=sources)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'verified_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    for name, rows in [('size_abc.csv', size_rows), ('tasksize10_arms.csv', arm_rows), ('tasksize10_tasks.csv', task_rows)]:
        with (out / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(dict(output=str(out), unique_episode_files=len(episode_files),
                          unique_episodes=unique_episodes, task_rows=len(task_rows)), ensure_ascii=False))


if __name__ == '__main__':
    main()
