# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import json

import numpy as np
import torch
from torch.utils.data import Dataset


class TransitionData(Dataset):

    def __init__(self, history_length, disassembly_root_path):

        self.trajs = []
        self.num_files = 0
        self.num_trajs = 0
        self.files = sorted(glob.glob(disassembly_root_path + "/asset_*_disassembly_traj.json"))

        for fi in self.files:
            self.num_files += 1
            with open(fi) as f:
                tmp = json.load(f)
            tmp = list(tmp[:])
            n = len(tmp)

            self.trajs.append(tmp[:])
            self.num_trajs += n

        self.history_length = history_length
        self.obs_dim = (
            len(self.trajs[0][0]["fingertip_centered_pos"][0])
            + len(self.trajs[0][0]["fingertip_centered_quat"][0])
            + len(self.trajs[0][0]["plug_pos"][0])
            + len(self.trajs[0][0]["plug_quat"][0])
        )
        self.act_dim = len(self.trajs[0][0]["actions"][0])

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

    def get_sample_from_file(self, file_idx):
        trajs = self.trajs[file_idx]
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

        file_idx = np.random.randint(low=0, high=self.num_files)
        cp_obs, cp_act, obs, act, next_obs = self.get_sample_from_file(file_idx)

        if np.random.rand() < 0.5 and self.num_files > 1:
            other_file_idx = np.random.randint(low=0, high=self.num_files)
            while other_file_idx == file_idx:
                other_file_idx = np.random.randint(low=0, high=self.num_files)
            other_cp_obs, other_cp_act, _, _, _ = self.get_sample_from_file(other_file_idx)
            target = 1
        else:
            other_cp_obs, other_cp_act, _, _, _ = self.get_sample_from_file(file_idx)
            target = 0

        return cp_obs, cp_act, obs, act, next_obs, other_cp_obs, other_cp_act, target
