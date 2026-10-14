#!/usr/bin/env python3
"""Run the ten AutoMate specialist jobs, one process per physical GPU."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("00062", "00186", "00256", "00271", "00726", "01079", "01029", "01092", "01102", "01125")
FACA_PROTOCOL = Path("/media/gpuserver/data2t/hx/github/FACA/reports/phase_force_contact/faca_factorized_v2_force_regime/formal_10task_protocol.json")
CONFIG_FILES = (
    "scripts/run_automate10_specialist_queue.py",
    "scripts/rl_games/train.py",
    "source/SRSA/SRSA/tasks/direct/srsa/run_w_id.py",
    "source/SRSA/SRSA/tasks/direct/srsa/agents/rl_games_ppo_cfg.yaml",
)


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def save_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=250)
    parser.add_argument("--max-iterations", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpus", nargs=2, type=int, default=(0, 1))
    parser.add_argument("--wait-for-pid", action="append", default=[], metavar="GPU:PID",
                        help="Keep a GPU idle until an existing parent process exits")
    parser.add_argument("--allow-busy-gpu", action="append", type=int, default=[], metavar="GPU",
                        help="Share this GPU with an existing low-utilization job")
    parser.add_argument("--numba-threads", type=int, default=1,
                        help="Threads per Soft-DTW call; one call processes one environment")
    parser.add_argument("--resume-queue-dir", type=Path, default=None,
                        help="Resume unfinished specialists from full-state checkpoints in an earlier queue")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.num_envs <= 0 or args.max_iterations <= 0 or args.seed < 0
            or args.numba_threads <= 0 or len(set(args.gpus)) != 2):
        parser.error("num-envs, max-iterations, numba-threads must be positive; seed nonnegative; GPUs distinct")
    wait_pids = {}
    if any(gpu not in args.gpus for gpu in args.allow_busy_gpu):
        parser.error("--allow-busy-gpu must name one of --gpus")
    for entry in args.wait_for_pid:
        try:
            gpu_text, pid_text = entry.split(":", 1)
            gpu, pid = int(gpu_text), int(pid_text)
        except ValueError:
            parser.error(f"invalid --wait-for-pid {entry!r}; expected GPU:PID")
        if gpu not in args.gpus or pid <= 0:
            parser.error(f"invalid wait GPU or PID: {entry}")
        wait_pids[gpu] = pid

    horizon = 32  # agents/rl_games_ppo_cfg.yaml
    batch = args.num_envs * horizon
    if batch % 4:
        parser.error("rollout batch must divide into four recurrent minibatches")
    minibatch = batch // 4
    central_minibatch = batch // 16
    if minibatch % 4:
        parser.error("minibatch must be divisible by the recurrent sequence length (4)")
    if batch % 16:
        parser.error("rollout batch must divide into 16 central value minibatches")
    for task in TASKS:
        for path in (ROOT / "data/mesh" / task / "plug.obj", ROOT / "data/mesh" / task / "socket.obj", ROOT / "data/disassembly_paths" / f"asset_{task}_disassembly_traj.json"):
            if not path.is_file():
                parser.error(f"missing task asset: {path}")
    if FACA_PROTOCOL.is_file():
        reference_tasks = json.loads(FACA_PROTOCOL.read_text())["tasks"]
        if reference_tasks != list(TASKS):
            parser.error("task list differs from the original FACA ten-task protocol that includes 00271")

    resume_info = {}
    if args.resume_queue_dir is not None:
        previous_dir = args.resume_queue_dir.resolve()
        previous_manifest = json.loads((previous_dir / "manifest.json").read_text())
        previous_status = json.loads((previous_dir / "status.json").read_text())
        if (previous_manifest["tasks"] != list(TASKS)
                or previous_manifest["num_envs_per_gpu"] != args.num_envs
                or previous_manifest["seed"] != args.seed
                or previous_manifest["max_iterations"] != args.max_iterations):
            parser.error("resume queue protocol differs from this run")
        if any(job["state"] == "completed" for job in previous_status["jobs"]):
            parser.error("resume queue already has completed jobs; this recovery path expects unfinished jobs")
        import torch

        for previous_job in previous_status["jobs"]:
            task = previous_job["assembly_id"]
            if previous_job["state"] == "queued":
                continue
            checkpoint_dir = ROOT / "logs/rl_games/Assembly" / previous_job["experiment_name"] / "nn"
            candidates = []
            for checkpoint in checkpoint_dir.glob("*.pth"):
                try:
                    epoch = int(torch.load(checkpoint, map_location="cpu", weights_only=False)["epoch"])
                except (OSError, EOFError, KeyError, RuntimeError, ValueError) as exc:
                    print(f"Ignoring unreadable checkpoint {checkpoint}: {exc}", file=sys.stderr)
                    continue
                candidates.append((epoch, checkpoint))
            if not candidates:
                parser.error(f"no full-state checkpoint for previously started task {task}")
            epoch, checkpoint = max(candidates, key=lambda pair: (pair[0], pair[1].stat().st_mtime))
            resume_info[task] = {"checkpoint": str(checkpoint.resolve()), "epoch": epoch}

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"faca00271_srsa_queue_{stamp}_{args.num_envs}env_numba{args.numba_threads}"
    run_dir = ROOT / "logs/rl_games/Assembly" / run_name
    experiments = [f"faca00271_srsa_{task}_seed{args.seed}_{stamp}_{args.num_envs}env_numba{args.numba_threads}" for task in TASKS]
    jobs = [{"assembly_id": task, "experiment_name": exp, "state": "queued", "gpu": None,
             "resume": resume_info.get(task)}
            for task, exp in zip(TASKS, experiments)]
    if args.dry_run:
        print(json.dumps({"run_dir": str(run_dir), "num_envs": args.num_envs,
                          "minibatch_size": minibatch, "central_value_minibatch_size": central_minibatch,
                          "max_iterations": args.max_iterations,
                          "gpus": args.gpus, "wait_pids": wait_pids, "numba_threads": args.numba_threads,
                          "resume_queue_dir": str(args.resume_queue_dir) if args.resume_queue_dir else None,
                          "allow_busy_gpus": args.allow_busy_gpu, "jobs": jobs}, indent=2))
        return

    run_dir.mkdir(parents=True, exist_ok=False)
    lock_path = ROOT / "logs/rl_games/Assembly/automate10_queue.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another AutoMate specialist queue is already running")
        lock.write(f"pid={os.getpid()} run_dir={run_dir}\n")
        lock.flush()
        hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in CONFIG_FILES}
        manifest = {"started_at": now(), "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": hashes, "num_envs_per_gpu": args.num_envs,
            "minibatch_size": minibatch, "central_value_minibatch_size": central_minibatch,
            "max_iterations": args.max_iterations,
            "seed": args.seed, "physical_gpus": args.gpus, "wait_pids": wait_pids,
            "numba_threads": args.numba_threads,
            "resume_queue_dir": str(args.resume_queue_dir.resolve()) if args.resume_queue_dir else None,
            "resume_info": resume_info,
            "allow_busy_gpus": args.allow_busy_gpu, "tasks": list(TASKS),
            "faca_reference_protocol": str(FACA_PROTOCOL),
            "faca_reference_sha256": hashlib.sha256(FACA_PROTOCOL.read_bytes()).hexdigest() if FACA_PROTOCOL.is_file() else None,
            "training_protocol": "AutoMate dense reward, SBC, physical_grasp, official reward success",
            "soft_dtw_backend": "CPU fallback (nvcc hidden from PATH; matches earlier SRSA specialist runs)",
            "evaluation_protocol": "FACA relaxed success with relaxed_depth_unbounded=true: depth_fraction > 0.85, lateral/no-jam/3-step gates retained"}
        save_json(run_dir / "manifest.json", manifest)
        pending = queue.Queue()
        for index in range(len(jobs)):
            pending.put(index)
        status_lock = threading.Lock()
        save_json(run_dir / "status.json", {"updated_at": now(), "jobs": jobs})

        def update(index, **fields):
            with status_lock:
                jobs[index].update(fields)
                save_json(run_dir / "status.json", {"updated_at": now(), "jobs": jobs})

        gpu_rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
        ).strip().splitlines()
        gpu_uuids = {int(row.split(",", 1)[0].strip()): row.split(",", 1)[1].strip() for row in gpu_rows}
        if any(gpu not in gpu_uuids for gpu in args.gpus):
            raise SystemExit(f"requested GPU not present: {args.gpus}")

        def gpu_blockers(gpu):
            output = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True
            )
            return [int(parts[0]) for row in output.splitlines()
                    if len(parts := [part.strip() for part in row.split(",")]) == 2
                    and parts[1] == gpu_uuids[gpu] and parts[0].isdigit()]

        def wait_until_free(gpu, index):
            if gpu in args.allow_busy_gpu:
                return
            while True:
                parent = wait_pids.get(gpu)
                parent_alive = parent is not None and Path(f"/proc/{parent}").exists()
                blockers = gpu_blockers(gpu)
                if not parent_alive and not blockers:
                    return
                update(index, state="waiting_for_gpu", gpu=gpu, wait_parent_pid=parent if parent_alive else None,
                       blocking_pids=blockers)
                time.sleep(30)

        def worker(gpu):
            while True:
                try:
                    index = pending.get_nowait()
                except queue.Empty:
                    return
                job = jobs[index]
                wait_until_free(gpu, index)
                command = [sys.executable, str(ROOT / "source/SRSA/SRSA/tasks/direct/srsa/run_w_id.py"),
                           "--assembly_id", job["assembly_id"], "--train", "--num_envs", str(args.num_envs),
                           "--headless", "--device", "cuda:0", "--seed", str(args.seed),
                           "--max_iterations", str(args.max_iterations), "--minibatch_size", str(minibatch),
                           "--central_value_minibatch_size", str(central_minibatch),
                           "--experiment_name", job["experiment_name"]]
                if job["resume"] is not None:
                    command.extend(["--checkpoint", job["resume"]["checkpoint"], "--load_mode", "all"])
                env = {key: value for key, value in os.environ.items()
                       if not key.startswith(("SRSA_", "VISION_"))}
                conda_prefix = str(Path(sys.executable).resolve().parent.parent)
                env["CONDA_PREFIX"] = conda_prefix
                env["CONDA_DEFAULT_ENV"] = Path(conda_prefix).name
                env["PATH"] = str(Path(conda_prefix) / "bin") + os.pathsep + env.get("PATH", "")
                # The local CUDA 12.8 nvcc selects AutoMate's Numba Soft-DTW path,
                # which fails in the simulator. The historic SRSA runs used CPU DTW.
                env["PATH"] = os.pathsep.join(
                    path for path in env.get("PATH", "").split(os.pathsep)
                    if path and not (Path(path) / "nvcc").exists()
                )
                env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "NUMBA_NUM_THREADS": str(args.numba_threads),
                            "FULL_PATH_TO_ISAACLAB": "/home/gpuserver/IsaacLab",
                            "TERM": "xterm", "OMNI_KIT_ACCEPT_EULA": "YES",
                            "SRSA_NEWT_OBS": "0", "SRSA_ENABLE_AXIAL_TASK_PARAM_SAMPLER": "0",
                            "SRSA_TASK_PARAM_GEOMETRY_SCALE": "0", "SRSA_ENABLE_FLANGE_FORCE_SENSOR": "0",
                            "SRSA_ENABLE_HELD_ASSET_NET_CONTACT_FORCE": "0",
                            "SRSA_FLANGE_FORCE_SENSOR_OBS": "0", "SRSA_HELD_ASSET_NET_CONTACT_FORCE_OBS": "0",
                            "SRSA_GRASP_CONSTRAINT_MODE": "physical_grasp", "SRSA_SUCCESS_METRIC": "official"})
                log_path = run_dir / f"{job['assembly_id']}.log"
                update(index, state="running", gpu=gpu, started_at=now(), log=str(log_path), command=command)
                try:
                    with log_path.open("w") as log:
                        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                        update(index, pid=process.pid)
                        exit_code = process.wait()
                    update(index, state="completed" if exit_code == 0 else "failed",
                           exit_code=exit_code, ended_at=now())
                except Exception as exc:
                    update(index, state="failed", error=repr(exc), ended_at=now())
                finally:
                    pending.task_done()

        threads = [threading.Thread(target=worker, args=(gpu,), name=f"gpu{gpu}") for gpu in args.gpus]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        failures = sum(job["state"] != "completed" for job in jobs)
        save_json(run_dir / "result.json", {"ended_at": now(), "failures": failures, "jobs": jobs})
        print(f"Queue finished: {len(jobs) - failures} completed, {failures} failed. See {run_dir}", flush=True)
        if failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
