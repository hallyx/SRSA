#!/usr/bin/env python3
"""Evaluate the seven follow-up experts after their training queue completes."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue", type=Path)
    parser.add_argument("--timeout-hours", type=float, default=36)
    args = parser.parse_args()
    train_queue = args.queue.resolve()
    deadline = time.monotonic() + args.timeout_hours * 3600
    result_path = train_queue / "result.json"
    while not result_path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"training queue did not complete: {train_queue}")
        time.sleep(60)
    result = json.loads(result_path.read_text())
    if result["failures"]:
        raise RuntimeError(f"training queue has {result['failures']} failed jobs; skipping evaluation")
    tasks = json.loads((train_queue / "manifest.json").read_text())["tasks"]
    output = train_queue.parent / f"{train_queue.name}_eval"
    command = [sys.executable, str(ROOT / "scripts/run_srsa_specialist_relaxed_eval.py"),
               "--train-queue", str(train_queue), "--output", str(output),
               "--checkpoint-epoch", "300", "--tasks", *tasks]
    print("Starting paired final evaluation:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
