# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.init as init
from dynamics.model import RunningMeanStd
from geometry.model import PointNetBackbone


class SuccPreNetwork(nn.Module):
    def __init__(self, fc_dim=128, num_points=2000, obs_dim=17, act_dim=14, history_length=10, seed=0, device="cuda:0"):
        super().__init__()
        self.num_points = num_points

        # set dynamics model
        self.norm_obs = RunningMeanStd(obs_dim, device=device)
        self.norm_act = RunningMeanStd(act_dim, device=device)
        self.norm_delta = RunningMeanStd(obs_dim, device=device)
        self.norm_cp_obs = RunningMeanStd(obs_dim * history_length, device=device)
        self.norm_cp_act = RunningMeanStd(act_dim * history_length, device=device)

        cp_hidden_sizes = (256, 128, 64)
        context_out_dim = 32

        self.context_model = nn.Sequential()
        _cp_hidden_sizes = [
            history_length * (obs_dim + act_dim),
        ] + list(cp_hidden_sizes)
        for i in range(len(_cp_hidden_sizes) - 1):
            self.context_model.add_module("dense%d" % i, nn.Linear(_cp_hidden_sizes[i], _cp_hidden_sizes[i + 1]))
            self.context_model.add_module("act%d" % i, nn.ReLU())
        self.context_model.add_module("output", nn.Linear(_cp_hidden_sizes[-1], context_out_dim))

        # set action model
        self.action_context_model = nn.Sequential()
        _cp_hidden_sizes = [
            history_length * (obs_dim + act_dim),
        ] + list(cp_hidden_sizes)
        for i in range(len(_cp_hidden_sizes) - 1):
            self.action_context_model.add_module("dense%d" % i, nn.Linear(_cp_hidden_sizes[i], _cp_hidden_sizes[i + 1]))
            self.action_context_model.add_module("act%d" % i, nn.ReLU())
        self.action_context_model.add_module("output", nn.Linear(_cp_hidden_sizes[-1], context_out_dim))

        # set geometry model
        feature_dim = 32
        self.backbone = PointNetBackbone(input_dim=3, feature_dim=feature_dim)

        # add linear layers to compare between the features of the two inputs
        self.fc = nn.Sequential(
            nn.Linear(feature_dim * 10, fc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dim, 1),
        )

        # initialize the weights
        self.fc.apply(self.init_weights)
        self.loss = nn.MSELoss()

    def load_embedding_models(self, geometry_model_path, dynamics_model_path, action_model_path):
        state_dict = torch.load(geometry_model_path, weights_only=True)
        backbone_state = self.backbone.state_dict()
        for name, param in state_dict.items():
            if "backbone" in name:
                print("copy", name)
                backbone_state[name[9:]].copy_(param)

        checkpoint = torch.load(dynamics_model_path, weights_only=True)
        self.context_model.load_state_dict(checkpoint["context"])
        self.norm_cp_obs.set_params(checkpoint["cp_obs"])
        self.norm_cp_act.set_params(checkpoint["cp_act"])
        self.norm_obs.set_params(checkpoint["obs"])
        self.norm_act.set_params(checkpoint["act"])
        self.norm_delta.set_params(checkpoint["delta"])

        checkpoint = torch.load(action_model_path, weights_only=True)
        self.action_context_model.load_state_dict(checkpoint["action_context"])

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if isinstance(m, nn.Conv1d):
            init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)
        if isinstance(m, nn.BatchNorm1d):
            init.ones_(m.weight)
            init.zeros_(m.bias)

    def forward(self, input1, cp_input1, input2, cp_input2, target=None):
        # get two images' features
        socket_1 = self.backbone(input1[0])
        plug_1 = self.backbone(input1[1])
        asset_1 = self.backbone(input1[2])

        socket_2 = self.backbone(input2[0])
        plug_2 = self.backbone(input2[1])
        asset_2 = self.backbone(input2[2])

        # get two context feature
        context_1 = self.context(cp_input1[0], cp_input1[1])
        context_2 = self.context(cp_input2[0], cp_input2[1])

        # get two action context features
        action_context_1, _ = self.action_context(cp_input1[0], cp_input1[1])
        action_context_2, _ = self.action_context(cp_input2[0], cp_input2[1])

        # concatenate both images' features
        output = torch.cat(
            (
                socket_1,
                plug_1,
                asset_1,
                context_1,
                action_context_1,
                socket_2,
                plug_2,
                asset_2,
                context_2,
                action_context_2,
            ),
            1,
        )

        # pass the concatenation to the linear layers
        output = self.fc(output)
        if target is None:
            return None, output
        else:
            return self.loss(output.squeeze(), target), output.squeeze()

    def context(self, cp_obs, cp_act, training=False):
        normalized_cp_obs = self.norm_cp_obs(cp_obs, training=training)
        normalized_cp_act = self.norm_cp_act(cp_act, training=training)
        normalized_cp_x = torch.cat([normalized_cp_obs, normalized_cp_act], dim=1)

        cp_output = self.context_model(normalized_cp_x)
        return cp_output

    def action_context(self, cp_obs, cp_act, training=False):
        normalized_cp_obs = self.norm_cp_obs(cp_obs, training=training)
        normalized_cp_act = self.norm_cp_act(cp_act, training=training)
        normalized_cp_x = torch.cat([normalized_cp_obs, normalized_cp_act], dim=1)

        cp_output = self.action_context_model(normalized_cp_x)
        return cp_output, normalized_cp_act
