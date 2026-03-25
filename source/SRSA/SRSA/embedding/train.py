# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
import os

import numpy as np
import torch
import torch.optim as optim
from dataset import PairData
from model import SuccPreNetwork
from task_file import SOURCE_TASKS, TEST_TASKS
from torch.optim.lr_scheduler import StepLR


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (
        sockets_1,
        plugs_1,
        assets_1,
        cp_obs_1,
        cp_act_1,
        obs_1,
        act_1,
        next_obs_1,
        sockets_2,
        plugs_2,
        assets_2,
        cp_obs_2,
        cp_act_2,
        targets,
    ) in enumerate(train_loader):
        (
            sockets_1,
            plugs_1,
            assets_1,
            cp_obs_1,
            cp_act_1,
            obs_1,
            act_1,
            next_obs_1,
            sockets_2,
            plugs_2,
            assets_2,
            cp_obs_2,
            cp_act_2,
            targets,
        ) = (
            sockets_1.to(device),
            plugs_1.to(device),
            assets_1.to(device),
            cp_obs_1.to(device),
            cp_act_1.to(device),
            obs_1.to(device),
            act_1.to(device),
            next_obs_1.to(device),
            sockets_2.to(device),
            plugs_2.to(device),
            assets_2.to(device),
            cp_obs_2.to(device),
            cp_act_2.to(device),
            targets.to(device),
        )
        optimizer.zero_grad()

        loss, _ = model(
            [sockets_1, plugs_1, assets_1],
            [cp_obs_1, cp_act_1],
            [sockets_2, plugs_2, assets_2],
            [cp_obs_2, cp_act_2],
            targets,
        )

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(sockets_1),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def retrieve(model, device, dataset, num_skills=5, retrieval_file="retrieval.json", transfer_success_dir=""):
    model.eval()
    num_samples = 100
    all_retrieved_success = []
    all_retrieved_tasks = {}
    with torch.no_grad():
        for target in dataset.target_asset_ids:
            retrieved_task = ""
            task_pred = {}
            for source in dataset.source_asset_ids:
                pred = 0
                for _ in range(num_samples):
                    socket_2, plug_2, asset_2 = dataset.get_asset_pcd(target)
                    sockets_2, plugs_2, assets_2 = (
                        socket_2.unsqueeze(0).to(device),
                        plug_2.unsqueeze(0).to(device),
                        asset_2.unsqueeze(0).to(device),
                    )
                    cp_obs_2, cp_act_2, _, _, _ = dataset.get_sample_from_asset(target)
                    cp_obs_2, cp_act_2 = cp_obs_2.unsqueeze(0).to(device), cp_act_2.unsqueeze(0).to(device)
                    socket_1, plug_1, asset_1 = dataset.get_asset_pcd(source)
                    sockets_1, plugs_1, assets_1 = (
                        socket_1.unsqueeze(0).to(device),
                        plug_1.unsqueeze(0).to(device),
                        asset_1.unsqueeze(0).to(device),
                    )
                    cp_obs_1, cp_act_1, _, _, _ = dataset.get_sample_from_asset(source)
                    cp_obs_1, cp_act_1 = cp_obs_1.unsqueeze(0).to(device), cp_act_1.unsqueeze(0).to(device)
                    _, output = model(
                        [sockets_1, plugs_1, assets_1],
                        [cp_obs_1, cp_act_1],
                        [sockets_2, plugs_2, assets_2],
                        [cp_obs_2, cp_act_2],
                    )
                    pred += output.item()
                pred = pred / num_samples
                task_pred[source] = pred
            task_pred = {k: v for k, v in sorted(task_pred.items(), key=lambda x: x[1], reverse=True)}
            source_tasks = [k for k, v in list(task_pred.items())[:num_skills]]
            all_retrieved_tasks[target] = source_tasks

            retrieved_task = source_tasks[0]
            print("target", target, "retrieved task", retrieved_task)
            eval_file = transfer_success_dir + f"/eval{retrieved_task}_task{target}.txt"
            if not os.path.exists(eval_file):
                line = "0.0"
            else:
                with open(eval_file) as f:
                    lines = f.readlines()
                line = lines[0]
            true_success = float(line)
            print("true success", true_success)
            all_retrieved_success.append(true_success)
    print(all_retrieved_success)
    with open(retrieval_file, "w") as f:
        json.dump(all_retrieved_tasks, f)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--asset_root_path", type=str, default="data/mesh/", help="path to asset")
    parser.add_argument(
        "--disassembly_root_path", type=str, default="data/disassembly_paths/", help="path to disassembly paths"
    )
    parser.add_argument(
        "--transfer_success_path", type=str, default="data/eval_transfer/", help="path to data of transfer success"
    )
    parser.add_argument(
        "--geometry_model_path",
        type=str,
        default="geometry/ckpt/23000.pth",
        help="path to well-trained geometry embedding model",
    )
    parser.add_argument(
        "--dynamics_model_path",
        type=str,
        default="dynamics/ckpt/200.pth",
        help="path to well-trained dynamics embedding model",
    )
    parser.add_argument(
        "--action_model_path",
        type=str,
        default="action/ckpt/200.pth",
        help="path to well-trained action embedding model",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="input batch size for training (default: 64)")
    parser.add_argument("--num_epoch", type=int, default=100, help="number of epochs to train (default: 14)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument(
        "--log_interval", type=int, default=10, help="how many batches to wait before logging training status"
    )
    parser.add_argument("--fc", type=int, default=128)
    parser.add_argument("--retrieve", action="store_true", default=False, help="For reconstructing the samples")
    parser.add_argument("--ckpt_root", type=str, default="ckpt")
    args = parser.parse_args()

    os.makedirs(args.ckpt_root, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda:0")

    train_kwargs = {"batch_size": args.batch_size, "num_workers": 1, "pin_memory": True, "shuffle": True}

    source_asset_ids = SOURCE_TASKS
    test_asset_ids = TEST_TASKS

    train_dataset = PairData(
        asset_root_path=args.asset_root_path,
        disassembly_root_path=args.disassembly_root_path,
        source_asset_ids=source_asset_ids,
        target_asset_ids=source_asset_ids,
        transfer_success_dir=args.transfer_success_path,
        seed=args.seed,
    )

    model = SuccPreNetwork(
        fc_dim=args.fc,
        obs_dim=train_dataset.obs_dim,
        act_dim=train_dataset.act_dim,
        history_length=train_dataset.history_length,
        seed=args.seed,
        device=device,
    ).to(device)
    model.load_embedding_models(
        geometry_model_path=args.geometry_model_path,
        dynamics_model_path=args.dynamics_model_path,
        action_model_path=args.action_model_path,
    )

    if args.retrieve:
        model.load_state_dict(torch.load(args.ckpt_root + "/ep%d" % args.num_epoch + ".pt", weights_only=True))
        test_dataset = PairData(
            asset_root_path=args.asset_root_path,
            disassembly_root_path=args.disassembly_root_path,
            source_asset_ids=source_asset_ids,
            target_asset_ids=test_asset_ids,
            transfer_success_dir=args.transfer_success_path,
        )
        retrieve(
            model,
            device,
            test_dataset,
            retrieval_file=args.ckpt_root + "/ep%d" % args.num_epoch + "_retrieval.json",
            transfer_success_dir=args.transfer_success_path,
        )
        return

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    for epoch in range(1, args.num_epoch + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        scheduler.step()

        if epoch % args.log_interval == 0:
            torch.save(model.state_dict(), args.ckpt_root + "/ep%d" % epoch + ".pt")


if __name__ == "__main__":
    main()
