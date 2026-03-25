# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm


def train(model_cls, dataset_cls, args):

    device = torch.device("cuda:0")

    train_data = dataset_cls(history_length=10, disassembly_root_path=args.disassembly_root_path)
    model = model_cls(
        obs_dim=train_data.obs_dim,
        act_dim=train_data.act_dim,
        history_length=10,
        context_out_dim=32,
        margin=1.0,
        device=device,
    )

    optimizer = optim.Adam(model.get_parameters(), lr=0.0005, weight_decay=0.0)

    train_loader = DataLoader(
        dataset=train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    i = 0
    for e in tqdm(range(args.num_epoch)):

        model.set_train()
        for cp_obs, cp_act, obs, act, next_obs, other_cp_obs, other_cp_act, target in train_loader:
            cp_obs, cp_act, obs, act, next_obs, other_cp_obs, other_cp_act, target = (
                cp_obs.to(device),
                cp_act.to(device),
                obs.to(device),
                act.to(device),
                next_obs.to(device),
                other_cp_obs.to(device),
                other_cp_act.to(device),
                target.to(device),
            )
            loss = model(cp_obs, cp_act, obs, act, next_obs, other_cp_obs, other_cp_act, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            i += 1
            if i % 1000 == 0:
                print("e: %d, i: %d, loss: %g" % (e, i, loss.item()))

        if (e + 1) % args.save_interval == 0:
            model.save(args.ckpt_root + "/%d.pth" % (e + 1))


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

    from dataset import TransitionData
    from model import ContextDynamics

    os.makedirs(args.ckpt_root, exist_ok=True)
    train(model_cls=ContextDynamics, dataset_cls=TransitionData, args=args)


if __name__ == "__main__":
    main()
