# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import json
import os

import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset


class PCDData(Dataset):

    def __init__(self, asset_root_path, num_pts_sample=2000, seed=0):
        self.asset_root_path = asset_root_path
        self.num_pts_sample = num_pts_sample
        self.seed = seed

        self.asset_ids = sorted(os.listdir(self.asset_root_path))[:100]

    def __len__(self):
        return len(self.asset_ids) * 2

    def get_asset_mesh(self, asset_id):
        socket_mesh = trimesh.load(self.asset_root_path + "/" + asset_id + "/socket.obj")
        plug_mesh = trimesh.load(self.asset_root_path + "/" + asset_id + "/plug.obj")
        return socket_mesh, plug_mesh

    def get_asset_pcd(self, asset_id):
        socket_mesh, plug_mesh = self.get_asset_mesh(asset_id)
        socket_pts_sample, _ = trimesh.sample.sample_surface(socket_mesh, self.num_pts_sample, seed=self.seed)
        socket_pcd = torch.FloatTensor(socket_pts_sample).permute(1, 0)
        plug_pts_sample, _ = trimesh.sample.sample_surface(plug_mesh, self.num_pts_sample, seed=self.seed)
        plug_pcd = torch.FloatTensor(plug_pts_sample).permute(1, 0)

        socket_pts_sample, _ = trimesh.sample.sample_surface(socket_mesh, self.num_pts_sample // 2, seed=self.seed)
        plug_pts_sample, _ = trimesh.sample.sample_surface(plug_mesh, self.num_pts_sample // 2, seed=self.seed)
        asset_pts_sample = np.concatenate((socket_pts_sample, plug_pts_sample), axis=0)
        asset_pcd = torch.FloatTensor(asset_pts_sample).permute(1, 0)
        return socket_pcd, plug_pcd, asset_pcd

    def __getitem__(self, index):

        socket_pcd, plug_pcd, asset_pcd = self.get_asset_pcd(self.asset_ids[index // 2])
        return socket_pcd, plug_pcd, asset_pcd


class TransitionData(Dataset):
    def __init__(self, history_length, disassembly_root_path):
        self.trajs = {}
        self.num_trajs = 0
        self.files = sorted(glob.glob(disassembly_root_path + "/asset_*_disassembly_traj.json"))
        for fi in self.files:
            asset_id = fi.split("/")[-1].split("_")[1]
            with open(fi) as f:
                tmp = json.load(f)
            tmp = list(tmp[:])
            n = len(tmp)
            self.num_trajs += n
            self.trajs[asset_id] = tmp[:]

        self.history_length = history_length
        traj = tmp[0]
        self.obs_dim = (
            len(traj["fingertip_centered_pos"][0])
            + len(traj["fingertip_centered_quat"][0])
            + len(traj["plug_pos"][0])
            + len(traj["plug_quat"][0])
        )
        self.act_dim = len(traj["actions"][0])

    def __len__(self):
        return self.num_trajs * 100

    def get_obs(self, traj, start_idx, end_idx):
        obs = [
            traj["fingertip_centered_pos"][start_idx:end_idx],
            traj["fingertip_centered_quat"][start_idx:end_idx],
            traj["plug_pos"][start_idx:end_idx],
            traj["plug_quat"][start_idx:end_idx],
        ]
        obs = np.concatenate(obs, 1)
        return obs

    def get_sample_from_asset(self, asset_id):
        if asset_id not in self.trajs.keys():
            asset_id = list(self.trajs.keys())[0]
        trajs = self.trajs[asset_id]
        traj_idx = np.random.randint(low=0, high=len(trajs))
        traj = trajs[traj_idx]
        length = len(traj["actions"])

        state_idx = np.random.randint(low=self.history_length, high=length - 2)
        cp_act = torch.FloatTensor(traj["actions"][state_idx - self.history_length : state_idx]).flatten()
        cp_obs = torch.FloatTensor(self.get_obs(traj, state_idx - self.history_length, state_idx)).flatten()

        act = torch.FloatTensor(traj["actions"][state_idx])
        obs = torch.FloatTensor(self.get_obs(traj, state_idx, state_idx + 1)[0])
        next_obs = torch.FloatTensor(self.get_obs(traj, state_idx + 1, state_idx + 2)[0])
        return cp_obs, cp_act, obs, act, next_obs

    def __getitem__(self, i):

        asset_ids = list(self.trajs.keys())
        idx = np.random.randint(low=0, high=len(asset_ids))
        cp_obs, cp_act, obs, act, next_obs = self.get_sample_from_asset(asset_ids[idx])

        return cp_obs, cp_act, obs, act, next_obs


class PairData(PCDData, TransitionData):
    def __init__(
        self,
        asset_root_path,
        disassembly_root_path,
        num_pts_sample=2000,
        history_length=10,
        source_asset_ids=[],
        target_asset_ids=[],
        transfer_success_dir="",
        seed=0,
    ):
        PCDData.__init__(self, asset_root_path=asset_root_path, num_pts_sample=num_pts_sample, seed=seed)
        TransitionData.__init__(self, disassembly_root_path=disassembly_root_path, history_length=history_length)
        self.source_asset_ids = []
        self.target_asset_ids = []
        for i, a in enumerate(self.asset_ids):
            if a in source_asset_ids:
                self.source_asset_ids.append(a)
            if a in target_asset_ids:
                self.target_asset_ids.append(a)

        source_tasks = []
        source_transfers = []
        self.transfer_eval = {}
        for source in self.source_asset_ids:
            source_success = 0
            for target in self.target_asset_ids:
                eval_file = transfer_success_dir + f"/eval{source}_task{target}.txt"
                if not os.path.exists(eval_file):
                    line = "0.0"
                else:
                    with open(eval_file) as f:
                        lines = f.readlines()
                    line = lines[0]
                tmp = float(line)
                self.transfer_eval[source + "-" + target] = tmp
                source_success += tmp
            print("source", source, "training success", source_success / len(self.target_asset_ids))
            source_tasks.append(source)
            source_transfers.append(source_success / len(self.target_asset_ids))
        print("source tasks", source_tasks)
        print("source transfers", source_transfers)

    def __len__(self):
        return len(self.source_asset_ids) * len(self.target_asset_ids)

    def __getitem__(self, index):

        selected_id = index // len(self.target_asset_ids)

        asset_id_1 = self.source_asset_ids[selected_id]
        socket_1, plug_1, asset_1 = self.get_asset_pcd(asset_id_1)
        cp_obs_1, cp_act_1, obs_1, act_1, next_obs_1 = self.get_sample_from_asset(asset_id_1)

        other_selected_id = index % len(self.target_asset_ids)

        asset_id_2 = self.target_asset_ids[other_selected_id]
        socket_2, plug_2, asset_2 = self.get_asset_pcd(asset_id_2)
        cp_obs_2, cp_act_2, obs_2, act_2, next_obs_2 = self.get_sample_from_asset(asset_id_2)
        target = torch.tensor(self.transfer_eval[asset_id_1 + "-" + asset_id_2], dtype=torch.float)
        return (
            socket_1,
            plug_1,
            asset_1,
            cp_obs_1,
            cp_act_1,
            obs_1,
            act_1,
            next_obs_1,
            socket_2,
            plug_2,
            asset_2,
            cp_obs_2,
            cp_act_2,
            target,
        )
