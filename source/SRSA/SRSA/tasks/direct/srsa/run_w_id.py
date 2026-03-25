# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import re
import subprocess


def update_task_param(task_cfg, assembly_id, if_sbc, if_log_eval):
    # Read the file lines.
    with open(task_cfg) as f:
        lines = f.readlines()

    updated_lines = []

    # Regex patterns to capture the assignment lines
    assembly_pattern = re.compile(r"^(.*assembly_id\s*=\s*).*$")
    if_sbc_pattern = re.compile(r"^(.*if_sbc\s*:\s*bool\s*=\s*).*$")
    if_log_eval_pattern = re.compile(r"^(.*if_logging_eval\s*:\s*bool\s*=\s*).*$")
    eval_file_pattern = re.compile(r"^(.*eval_filename\s*:\s*str\s*=\s*).*$")

    for line in lines:
        if "assembly_id =" in line:
            line = assembly_pattern.sub(fr"\1'{assembly_id}'", line)
        elif "if_sbc: bool =" in line:
            line = if_sbc_pattern.sub(fr"\1{str(if_sbc)}", line)
        elif "if_logging_eval: bool =" in line:
            line = if_log_eval_pattern.sub(fr"\1{str(if_log_eval)}", line)
        elif "eval_filename: str = " in line:
            line = eval_file_pattern.sub(r"\1'{}'".format(f"evaluation_{assembly_id}.h5"), line)

        updated_lines.append(line)

    # Write the modified lines back to the file.
    with open(task_cfg, "w") as f:
        f.writelines(updated_lines)


def main():
    parser = argparse.ArgumentParser(description="Update assembly_id and run training script.")
    parser.add_argument(
        "--cfg_path",
        type=str,
        help="Path to the file containing assembly_id.",
        default="../isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/automate/assembly_tasks_cfg.py",
    )
    parser.add_argument("--assembly_id", type=str, help="New assembly ID to set.")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path.")
    parser.add_argument("--load_mode", type=str, default="actor", help="Load checkpoint for fine-tuning.")
    parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel environment.")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed.")
    parser.add_argument("--train", action="store_true", help="Run training mode.")
    parser.add_argument("--sil", action="store_true", help="Use self-imitation learning.")
    parser.add_argument("--sparse", action="store_true", help="Use sparse reward.")
    parser.add_argument("--no_sbc", action="store_true", help="Apply curriculum difficulty update")
    parser.add_argument("--log_eval", action="store_true", help="Log evaluation results.")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode.")
    args = parser.parse_args()

    if not args.train:
        args.no_sbc = True

    update_task_param(args.cfg_path, args.assembly_id, not args.no_sbc, args.log_eval)

    if args.sil:
        if args.sparse:
            task = "Assembly-Sparse-Sil-v0"
        else:
            task = "Assembly-Direct-Sil-v0"
    else:
        if args.sparse:
            task = "Assembly-Sparse-v0"
        else:
            task = "Assembly-Direct-v0"

    bash_command = None
    if args.train:
        bash_command = "${{FULL_PATH_TO_ISAACLAB}}/isaaclab.sh -p scripts/rl_games/train.py --task={} --load_mode={}".format(
            task,
            args.load_mode,
        )
        bash_command += f" --seed={str(args.seed)}"
    else:
        if not args.checkpoint:
            raise ValueError("No checkpoint provided for evaluation.")
        bash_command = "${FULL_PATH_TO_ISAACLAB}/isaaclab.sh -p scripts/rl_games/play.py --task=%s" % task

    bash_command += f" --num_envs={str(args.num_envs)}"

    if args.checkpoint:
        bash_command += f" --checkpoint={args.checkpoint}"

    if args.headless:
        bash_command += " --headless"

    # Run the bash command
    subprocess.run(bash_command, shell=True, check=True)


if __name__ == "__main__":
    main()
