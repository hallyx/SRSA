# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
import os

import numpy as np
import torch

from dataset import PointCloudsData
from torch.utils.data import DataLoader

from tqdm import tqdm
from trainer import Trainer


def train(args):
    data = PointCloudsData(args.asset_root_path, num_points=2000, seed=args.seed)
    device = torch.device("cuda:0")

    num_steps = args.num_epoch * (len(data) // args.batch_size)
    model = Trainer(num_steps, num_points=2000, device=device)

    text = "e: {0}, i: {1}, loss: {2:.3f}"

    train_loader = DataLoader(dataset=data, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    i = 0
    for e in tqdm(range(args.num_epoch)):

        model.network.train()
        for x in train_loader:

            x = x.to(device)
            loss = model.train_step(x)

            i += 1
            log = text.format(e, i, loss)
            print(log)

        if (e + 1) % args.save_interval == 0:
            model.save(args.ckpt_root + "/%d.pth" % (e + 1))


def eval(args):
    data = PointCloudsData(args.asset_root_path, num_points=2000, seed=args.seed)
    device = torch.device("cuda:0")

    num_steps = args.num_epoch * (len(data) // args.batch_size)
    model = Trainer(num_steps, num_points=2000, device=device)

    model.load(args.ckpt_root + "/%d.pth" % args.num_epoch)

    model.network.eval()
    eval_loader = DataLoader(dataset=data, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)

    asset_ids = sorted(os.listdir(args.asset_root_path))
    encodings = {}
    total_loss = 0
    for i, x in enumerate(eval_loader):
        x = x.to(device)
        loss, x_restored, encoding = model.evaluate(x)
        print("loss", loss)
        total_loss += loss
        # append the encoding for plug and socket as task embedding
        encodings[asset_ids[i]] = encoding.cpu().numpy().reshape(-1).tolist()

    print("average loss", total_loss / i)
    with open(args.ckpt_root + "/embedding.json", "w") as f:
        json.dump(encodings, f, indent=6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_root_path", type=str, default="SRSA_data/mesh/", help="path to asset")
    parser.add_argument("--batch_size", type=int, default=64, help="input batch size for training (default: 64)")
    parser.add_argument("--num_epoch", type=int, default=24000)
    parser.add_argument("--ckpt_root", type=str, default="ckpt")
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.ckpt_root, exist_ok=True)
    if args.eval:
        eval(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
