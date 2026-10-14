#!/usr/bin/env python3
"""Evaluate completed AutoMate specialists with the FACA relaxed metric."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
from datetime import datetime

import h5py


ROOT = Path(__file__).resolve().parents[1]
TRAIN_QUEUE = ROOT / "logs/rl_games/Assembly/faca00271_srsa_queue_20260925_113344_250env_numba1"
OUTPUT = ROOT / "logs/rl_games/Assembly/faca00271_srsa_relaxed_eval_20260926"
TASKS = ("00062", "00186", "00256", "00271", "00726", "01079")
SEEDS = (5101, 5102)


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def save_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def read_metrics(path):
    with h5py.File(path, "r") as hf:
        protocol = hf.attrs.get("relaxed_protocol", "")
        if "depth_fraction > 0.85" not in protocol:
            raise ValueError(f"wrong relaxed protocol in {path}")
        official = hf["success"][:].reshape(-1)
        relaxed = hf["relaxed_success_stable"][:].reshape(-1)
        episode = hf["relaxed_success_episode"][:].reshape(-1)
        depth = hf["relaxed_depth_fraction_final"][:].reshape(-1)
        if len(official) != 100 or len(relaxed) != 100:
            raise ValueError(f"expected 100 episodes in {path}")
        return {"episodes": 100, "official": int(official.sum()),
                "relaxed_stable": int(relaxed.sum()), "relaxed_episode": int(episode.sum()),
                "terminal_depth_gt_085": int((depth > 0.85).sum())}


def main():
    global OUTPUT, TRAIN_QUEUE, TASKS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', nargs='+', default=TASKS)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--train-queue', type=Path, default=TRAIN_QUEUE)
    parser.add_argument('--checkpoint-epoch', type=int, default=1500)
    args = parser.parse_args()
    if args.checkpoint_epoch < 1:
        parser.error('--checkpoint-epoch must be positive')
    TASKS = tuple(args.tasks)
    if len(set(TASKS)) != len(TASKS):
        parser.error('duplicate task IDs')
    OUTPUT = args.output.expanduser().resolve()
    TRAIN_QUEUE = args.train_queue.expanduser().resolve()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train_status = json.loads((TRAIN_QUEUE / "status.json").read_text())
    train_jobs = {job["assembly_id"]: job for job in train_status["jobs"]}
    jobs = []
    for task in TASKS:
        train = train_jobs[task]
        if train["state"] != "completed" or train.get("exit_code") != 0:
            raise RuntimeError(f"training task {task} is not complete")
        directory = ROOT / "logs/rl_games/Assembly" / train["experiment_name"] / "nn"
        checkpoints = [path for path in directory.glob(f"last_Assembly_ep_{args.checkpoint_epoch}_rew_*.pth")
                       if "_rew__" not in path.name]
        if not checkpoints:
            raise RuntimeError(f"missing final checkpoint for {task}")
        checkpoint = max(checkpoints, key=lambda path: path.stat().st_mtime).resolve()
        checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        for seed in SEEDS:
            jobs.append({"task": task, "seed": seed, "checkpoint": str(checkpoint),
                         "checkpoint_sha256": checksum,
                         "output": str(OUTPUT / f"{task}_seed{seed}.h5"), "state": "queued"})

    save_json(OUTPUT / "manifest.json", {
        "started_at": now(), "training_queue": str(TRAIN_QUEUE), "tasks": TASKS,
        "seeds": SEEDS, "episodes_per_seed": 100,
        "checkpoint_selection": f"final epoch {args.checkpoint_epoch}",
        "num_envs": 100, "policy_action": "deterministic mean", "eval_sbc": False,
        "relaxed_protocol": "FACA relaxed_success_stable: depth_fraction > 0.85, lateral/no-jam/3-step gates",
        "jobs": [{"task": j["task"], "seed": j["seed"], "checkpoint": j["checkpoint"],
                  "checkpoint_sha256": j["checkpoint_sha256"]} for j in jobs],
    })
    status_lock = threading.Lock()

    def update(job, **fields):
        with status_lock:
            job.update(fields)
            save_json(OUTPUT / "status.json", {"updated_at": now(), "jobs": jobs})

    save_json(OUTPUT / "status.json", {"updated_at": now(), "jobs": jobs})
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)

    def worker(gpu):
        while True:
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            path = Path(job["output"])
            try:
                if path.exists():
                    metrics = read_metrics(path)
                    with h5py.File(path, "r") as hf:
                        old_hash = hf.attrs.get("checkpoint_sha256")
                        if old_hash != job["checkpoint_sha256"]:
                            raise ValueError(f"checkpoint hash mismatch in {path}")
                        if int(hf.attrs.get('eval_seed', -1)) != job['seed']:
                            raise ValueError(f'evaluation seed mismatch in {path}')
                        if hf.attrs.get('policy_action') != 'deterministic mean':
                            raise ValueError(f'policy action mismatch in {path}')
                    update(job, state="completed", gpu=gpu, metrics=metrics,
                           reused_existing=True, ended_at=now())
                    continue
                command = [sys.executable, str(ROOT / "source/SRSA/SRSA/tasks/direct/srsa/run_w_id.py"),
                           "--assembly_id", job["task"], "--checkpoint", job["checkpoint"],
                           "--num_envs", "100", "--device", "cuda:0", "--seed", str(job["seed"]),
                           "--deterministic", "--log_eval", "--num_eval_trials", "100", "--headless"]
                env = {key: value for key, value in os.environ.items()
                       if not key.startswith(("SRSA_", "VISION_"))}
                prefix = str(Path(sys.executable).resolve().parent.parent)
                env["CONDA_PREFIX"] = prefix
                env["CONDA_DEFAULT_ENV"] = Path(prefix).name
                env["PATH"] = os.pathsep.join(
                    part for part in env.get("PATH", "").split(os.pathsep)
                    if part and not (Path(part) / "nvcc").exists()
                )
                env.update({
                    "CUDA_VISIBLE_DEVICES": str(gpu), "NUMBA_NUM_THREADS": "1",
                    "FULL_PATH_TO_ISAACLAB": "/home/gpuserver/IsaacLab", "TERM": "xterm",
                    "OMNI_KIT_ACCEPT_EULA": "YES", "SRSA_NEWT_OBS": "0",
                    "SRSA_ENABLE_AXIAL_TASK_PARAM_SAMPLER": "0", "SRSA_TASK_PARAM_GEOMETRY_SCALE": "0",
                    "SRSA_ENABLE_FLANGE_FORCE_SENSOR": "1",
                    "SRSA_ENABLE_HELD_ASSET_NET_CONTACT_FORCE": "1",
                    "SRSA_FLANGE_FORCE_SENSOR_OBS": "0",
                    "SRSA_HELD_ASSET_NET_CONTACT_FORCE_OBS": "0",
                    "SRSA_GRASP_CONSTRAINT_MODE": "physical_grasp", "SRSA_SUCCESS_METRIC": "official",
                    "SRSA_RELAXED_DEPTH_UNBOUNDED": "1", "SRSA_EVAL_FILENAME": str(path),
                })
                log_path = path.with_suffix(".log")
                update(job, state="running", gpu=gpu, started_at=now(), log=str(log_path))
                with log_path.open("w") as log:
                    result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                            stderr=subprocess.STDOUT, timeout=900)
                if result.returncode:
                    raise RuntimeError(f"evaluation exited {result.returncode}")
                metrics = read_metrics(path)
                with h5py.File(path, "a") as hf:
                    hf.attrs["eval_seed"] = job["seed"]
                    hf.attrs["checkpoint_sha256"] = job["checkpoint_sha256"]
                    hf.attrs["policy_action"] = "deterministic mean"
                update(job, state="completed", metrics=metrics, ended_at=now())
            except Exception as exc:
                update(job, state="failed", error=repr(exc), ended_at=now())
            finally:
                pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failures = sum(job["state"] != "completed" for job in jobs)
    save_json(OUTPUT / "result.json", {"ended_at": now(), "failures": failures, "jobs": jobs})
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
