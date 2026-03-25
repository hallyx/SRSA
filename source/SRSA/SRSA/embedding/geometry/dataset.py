# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import torch
import trimesh
from torch.utils.data import Dataset


class PointCloudsData(Dataset):

    def __init__(self, asset_root_path, num_points, seed=0):
        """
        Arguments:
            asset_root_path(str): the root path where stores all asset meshes.
            num_pts_sample(int): number of points sampled on each asset mesh.
        """

        self.asset_root_path = asset_root_path
        self.num_points = num_points
        self.seed = seed

        self.asset_paths = []

        assembly_ids = sorted(os.listdir(self.asset_root_path))[:100]
        for assembly_id in assembly_ids:
            self.asset_paths.append(self.asset_root_path + "/" + assembly_id + "/socket.obj")
            self.asset_paths.append(self.asset_root_path + "/" + assembly_id + "/plug.obj")

    def __len__(self):
        return len(self.asset_paths)

    def __getitem__(self, i):
        """
        Returns:
            sampled_pts_tensor: a float tensor with shape [3, num_points].
        """

        mesh_path = self.asset_paths[i]
        mesh = trimesh.load(mesh_path)

        sampled_pts, _ = trimesh.sample.sample_surface(mesh, self.num_points, seed=self.seed)
        sampled_pts_tensor = torch.FloatTensor(sampled_pts).permute(1, 0)

        return sampled_pts_tensor
