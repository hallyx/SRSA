# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import sys

import numpy as np
import torch

sys.path.append("..")
from dynamics.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--disassembly_root_path", type=str, default="data/disassembly_paths/", help="path to disassembly paths"
    )
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ckpt_root", type=str, default="ckpt")
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    from dynamics.dataset import TransitionData
    from model import ContextAction

    os.makedirs(args.ckpt_root, exist_ok=True)
    train(model_cls=ContextAction, dataset_cls=TransitionData, args=args)


if __name__ == "__main__":
    main()
