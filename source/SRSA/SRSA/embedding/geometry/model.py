# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy

import torch
import torch.nn as nn
from pointnet_utils import PointBackbone, build_backbone, masked_average, masked_max


class SimplePointNetV0(PointBackbone):
    def __init__(
        self,
        conv_cfg,
        mlp_cfg,
        stack_frame=1,
        subtract_mean_coords=False,
        max_mean_mix_aggregation=False,
        with_activation=False,
    ):
        """
        PointNet that processes multiple consecutive frames of pcd data.
        :param conv_cfg: configuration for building point feature extractor
        :param mlp_cfg: configuration for building global feature extractor
        :param stack_frame: num of stacked frames in the input
        :param subtract_mean_coords: subtract_mean_coords trick
            subtract the mean of xyz from each point's xyz, and then concat the mean to the original xyz;
            we found concatenating the mean pretty crucial
        :param max_mean_mix_aggregation: max_mean_mix_aggregation trick
        """
        super().__init__()
        conv_cfg = deepcopy(conv_cfg)
        conv_cfg["mlp_spec"][0] += int(subtract_mean_coords) * 3
        self.conv_mlp = build_backbone(conv_cfg)
        self.stack_frame = stack_frame
        self.max_mean_mix_aggregation = max_mean_mix_aggregation
        self.subtract_mean_coords = subtract_mean_coords
        self.global_mlp = build_backbone(mlp_cfg)
        self.with_activation = with_activation
        if with_activation:
            self.activation = nn.Sigmoid()

    def forward_raw(self, pcd, mask=None):
        """
        :param pcd: point cloud with states
        :param mask: [B, N] ([batch size, n_points]) provides which part of point cloud should be considered
        :return: [B, F] ([batch size, final output dim])
        """

        if self.subtract_mean_coords:
            # Use xyz - mean xyz instead of original xyz
            mask = torch.ones_like(pcd[..., :1])
            xyz = pcd[:, :, :3]
            mean_xyz = masked_average(xyz, 1, mask=mask, keepdim=True)  # [B, 1, 3]
            pcd = torch.cat((mean_xyz.repeat(1, xyz.shape[1], 1), xyz - mean_xyz, pcd[:, :, 3:]), dim=2)

        B, N = pcd.shape[:2]

        point_feature = self.conv_mlp(pcd.transpose(2, 1)).transpose(2, 1)  # [B, N, CF]
        # [B, K, N / K, CF]
        point_feature = point_feature.view(B, self.stack_frame, N // self.stack_frame, point_feature.shape[-1])
        mask = torch.ones_like(pcd[..., :1])
        mask = mask.view(B, self.stack_frame, N // self.stack_frame, 1)  # [B, K, N / K, 1]

        if self.max_mean_mix_aggregation:
            sep = point_feature.shape[-1] // 2
            max_feature = masked_max(point_feature[..., :sep], 2, mask=mask)  # [B, K, CF / 2]
            mean_feature = masked_average(point_feature[..., sep:], 2, mask=mask)  # [B, K, CF / 2]
            global_feature = torch.cat([max_feature, mean_feature], dim=-1)  # [B, K, CF]

        else:
            global_feature = masked_max(point_feature, 2, mask=mask)  # [B, K, CF]

        global_feature = global_feature.reshape(B, -1)

        if self.with_activation:
            f = self.global_mlp(global_feature)
            return self.activation(f)
        return self.global_mlp(global_feature)


def getPointNet(cfg):
    stack_frame = 1
    nn_cfg = dict(
        conv_cfg=dict(
            type="ConvMLP",
            norm_cfg=None,
            mlp_spec=[cfg["input_feature_dim"], 128, 256],
            bias="auto",
            inactivated_output=False,
            conv_init_cfg=dict(
                type="xavier_init",
                gain=1,
                bias=0,
            ),
        ),
        mlp_cfg=dict(
            type="LinearMLP",
            norm_cfg=None,
            mlp_spec=[256 * stack_frame, 256, cfg["feat_dim"]],
            bias="auto",
            inactivated_output=False,
            linear_init_cfg=dict(
                type="xavier_init",
                gain=1,
                bias=0,
            ),
        ),
        subtract_mean_coords=True,
        max_mean_mix_aggregation=True,
        stack_frame=stack_frame,
    )

    pointnet = SimplePointNetV0(**nn_cfg)

    return pointnet


class PointNetBackbone(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super().__init__()

        cfg = dict({})
        cfg["input_feature_dim"] = input_dim
        cfg["feat_dim"] = feature_dim

        self.transpn = getPointNet(cfg)

    def forward(self, input_pc):

        output = self.transpn.forward_raw(input_pc.permute(0, 2, 1))

        return output


class AutoencoderTransPN(nn.Module):

    def __init__(self, num_points):
        """
        Arguments:
            num_points: an integer.
        """
        super().__init__()

        feature_dim = 32
        # ENCODER
        self.backbone = PointNetBackbone(input_dim=3, feature_dim=feature_dim)

        # DECODER
        self.decoder = nn.Sequential(
            nn.Conv1d(feature_dim, 256, kernel_size=1, bias=False),
            # nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=1, bias=False),
            # nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, num_points * 3, kernel_size=1),
        )

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, num_points].
        Returns:
            encoding: a float tensor with shape [b, k].
            restoration: a float tensor with shape [b, 3, num_points].
        """

        b, _, num_points = x.size()
        encoding = self.backbone(x).unsqueeze(2)  # shape [b, k, 1]
        x = self.decoder(encoding)  # shape [b, num_points * 3, 1]
        restoration = x.view(b, 3, num_points)

        return encoding, restoration
