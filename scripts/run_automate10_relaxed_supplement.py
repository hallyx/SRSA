#!/usr/bin/env python3
"""Queue follow-up training for experts with zero modified relaxed success."""

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "logs/rl_games/Assembly/faca00271_srsa_queue_20260925_113344_250env_numba1"
COMPARISON = ROOT / "runs/automate10/dagger_prepare_20260927/comparison.json"
SOURCE_FILES = (
    "scripts/run_automate10_relaxed_supplement.py",
    "source/SRSA/SRSA/tasks/direct/srsa/assembly_direct_env.py",
    "source/SRSA/SRSA/tasks/direct/srsa/relaxed_eval.py",
)


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def source_checkpoint(experiment):
    directory = ROOT / "logs/rl_games/Assembly" / experiment / "nn"
    files = sorted(directory.glob("last_Assembly_ep_1500_rew_*.pth"))
    files = [p for p in files if re.search(r"rew_-?[0-9.]+\.pth$", p.name)]
    if len(files) != 1:
        raise ValueError(f"expected exactly one epoch-1500 checkpoint in {directory}: {files}")
    return files[0].resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra-iterations", type=int, default=300)
    parser.add_argument("--num-envs", type=int, default=250)
    parser.add_argument("--gpus", nargs=2, type=int, default=[0, 1])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.extra_iterations < 1 or args.num_envs != 250 or len(set(args.gpus)) != 2:
        parser.error("positive extra iterations, 250 envs, and two distinct GPUs are required")

    comparison = json.loads(COMPARISON.read_text())
    tasks = [row["task"] for row in comparison["tasks"] if row["modified_relaxed"] == 0]
    if tasks != ["00062", "00256", "01079", "01029", "01092", "01102", "01125"]:
        raise ValueError(f"unexpected zero-success task set: {tasks}")
    base_jobs = {job["assembly_id"]: job for job in json.loads((BASE / "result.json").read_text())["jobs"]}
    if any(base_jobs[task]["state"] != "completed" for task in tasks):
        raise ValueError("source queue is incomplete")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "logs/rl_games/Assembly" / f"faca00271_relaxed_supplement_queue_{stamp}"
    jobs = []
    for task in tasks:
        ckpt = source_checkpoint(base_jobs[task]["experiment_name"])
        jobs.append({"assembly_id": task, "checkpoint": str(ckpt),
                     "checkpoint_sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                     "experiment_name": f"faca00271_relaxed_supplement_{task}_seed1_{stamp}",
                     "state": "queued", "gpu": None})
    if args.dry_run:
        print(json.dumps({"run_dir": str(run_dir), "jobs": jobs}, ensure_ascii=False, indent=2))
        return

    run_dir.mkdir(parents=True)
    manifest = {
        "started_at": now(), "source_queue": str(BASE), "comparison": str(COMPARISON),
        "criterion": "modified relaxed terminal success == 0/200",
        "training_metric": "depth_fraction > 0.85, lateral <= clamp(2*radial_clearance,1,3mm), no jam, 3 consecutive steps",
        "reward": "AutoMate SDF + imitation + modified relaxed stable success + capped depth/lateral alignment aid",
        "alignment_reward": "clamp(depth_fraction/0.85,0,1)*clamp(1-lateral_error/0.015,0,1)*(not jam)",
        "initialization": "actor weights and observation normalizer from original epoch-1500; critic and optimizer reset",
        "extra_iterations": args.extra_iterations, "num_envs_per_gpu": args.num_envs,
        "training_seed": 1, "gpus": args.gpus, "tasks": tasks,
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES},
    }
    save_json(run_dir / "manifest.json", manifest)
    save_json(run_dir / "status.json", {"updated_at": now(), "jobs": jobs})
    lock = (ROOT / "logs/rl_games/Assembly/automate10_queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(f"pid={os.getpid()} run_dir={run_dir}\n")
    lock.flush()
    pending = queue.Queue()
    for i in range(len(jobs)):
        pending.put(i)
    status_lock = threading.Lock()

    def update(index, **fields):
        with status_lock:
            jobs[index].update(fields)
            save_json(run_dir / "status.json", {"updated_at": now(), "jobs": jobs})

    def worker(gpu):
        while True:
            try:
                index = pending.get_nowait()
            except queue.Empty:
                return
            job = jobs[index]
            command = [sys.executable, str(ROOT / "source/SRSA/SRSA/tasks/direct/srsa/run_w_id.py"),
                       "--assembly_id", job["assembly_id"], "--train", "--num_envs", str(args.num_envs),
                       "--headless", "--device", "cuda:0", "--seed", "1",
                       "--max_iterations", str(args.extra_iterations), "--minibatch_size", "2000",
                       "--central_value_minibatch_size", "500", "--experiment_name", job["experiment_name"],
                       "--checkpoint", job["checkpoint"], "--load_mode", "actor"]
            env = {key: value for key, value in os.environ.items() if not key.startswith(("SRSA_", "VISION_"))}
            conda_prefix = str(Path(sys.executable).resolve().parent.parent)
            env.update({"CONDA_PREFIX": conda_prefix, "CONDA_DEFAULT_ENV": Path(conda_prefix).name,
                        "CUDA_VISIBLE_DEVICES": str(gpu), "NUMBA_NUM_THREADS": "1",
                        "FULL_PATH_TO_ISAACLAB": "/home/gpuserver/IsaacLab", "TERM": "xterm",
                        "OMNI_KIT_ACCEPT_EULA": "YES", "SRSA_NEWT_OBS": "0",
                        "SRSA_ENABLE_AXIAL_TASK_PARAM_SAMPLER": "0", "SRSA_TASK_PARAM_GEOMETRY_SCALE": "0",
                        "SRSA_ENABLE_FLANGE_FORCE_SENSOR": "1", "SRSA_ENABLE_HELD_ASSET_NET_CONTACT_FORCE": "1",
                        "SRSA_FLANGE_FORCE_SENSOR_OBS": "0", "SRSA_HELD_ASSET_NET_CONTACT_FORCE_OBS": "0",
                        "SRSA_GRASP_CONSTRAINT_MODE": "physical_grasp", "SRSA_SUCCESS_METRIC": "official",
                        "SRSA_TRAIN_RELAXED_GT085": "1"})
            env["PATH"] = os.pathsep.join(
                path for path in (str(Path(conda_prefix) / "bin") + os.pathsep + env.get("PATH", "")).split(os.pathsep)
                if path and not (Path(path) / "nvcc").exists()
            )
            log_path = run_dir / f"{job['assembly_id']}.log"
            try:
                update(index, state="running", gpu=gpu, started_at=now(), command=command, log=str(log_path))
                with log_path.open("w") as log:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    update(index, pid=process.pid)
                    code = process.wait()
                update(index, state="completed" if code == 0 else "failed", exit_code=code, ended_at=now())
            except Exception as exc:
                update(index, state="failed", error=repr(exc), ended_at=now())
            finally:
                pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in args.gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failures = sum(job["state"] != "completed" for job in jobs)
    save_json(run_dir / "result.json", {"ended_at": now(), "failures": failures, "jobs": jobs})
    print(f"Supplement queue finished: {len(jobs)-failures}/{len(jobs)} completed; {run_dir}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
