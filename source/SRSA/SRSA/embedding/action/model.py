# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("..")
from dynamics.model import RunningMeanStd


class ContextAction(nn.Module):

    def __init__(
        self,
        obs_dim,
        act_dim,
        history_length,
        cp_hidden_sizes=(256, 128, 64),
        context_out_dim=32,
        hidden_sizes=(200, 200, 200, 200),
        margin=0,
        device="cuda:0",
    ):

        super().__init__()

        self.norm_cp_obs = RunningMeanStd(obs_dim * history_length, device=device)
        self.norm_cp_act = RunningMeanStd(act_dim * history_length, device=device)

        self.action_context_model = nn.Sequential()
        cp_hidden_sizes = [
            history_length * (obs_dim + act_dim),
        ] + list(cp_hidden_sizes)
        for i in range(len(cp_hidden_sizes) - 1):
            self.action_context_model.add_module("dense%d" % i, nn.Linear(cp_hidden_sizes[i], cp_hidden_sizes[i + 1]))
            self.action_context_model.add_module("act%d" % i, nn.ReLU())
        self.action_context_model.add_module("output", nn.Linear(cp_hidden_sizes[-1], context_out_dim))
        self.action_context_model.to(device)

        self.action_model = nn.Sequential()
        hidden_sizes = [
            context_out_dim,
        ] + list(hidden_sizes)
        for i in range(len(hidden_sizes) - 1):
            self.action_model.add_module("dense%d" % i, nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.action_model.add_module("act%d" % i, nn.ReLU())
        self.action_model.add_module("output", nn.Linear(hidden_sizes[-1], act_dim * history_length))
        self.action_model.to(device)

        self.loss = nn.MSELoss()
        self.margin = margin

    def set_train(self):
        self.action_context_model.train()
        self.action_model.train()

    def get_parameters(self):
        return list(self.action_context_model.parameters()) + list(self.action_model.parameters())

    def embed(self, cp_obs, cp_act, training=True):
        normalized_cp_obs = self.norm_cp_obs(cp_obs, training=training)
        normalized_cp_act = self.norm_cp_act(cp_act, training=training)
        normalized_cp_x = torch.cat([normalized_cp_obs, normalized_cp_act], dim=1)

        cp_output = self.action_context_model(normalized_cp_x)
        return cp_output, normalized_cp_act

    def forward(self, cp_obs, cp_act, obs, act, next_obs, other_cp_obs, other_cp_act, target, training=True):

        cp_output, normalized_cp_act = self.embed(cp_obs, cp_act, training=training)

        output = self.action_model(cp_output)
        action_loss = self.loss(output, normalized_cp_act)

        other_cp_output, _ = self.embed(other_cp_obs, other_cp_act, training=training)
        euclidean_distance = F.pairwise_distance(cp_output, other_cp_output)
        contrastive_loss = torch.mean(
            (1 - target) * torch.pow(euclidean_distance, 2)
            + (target) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        loss = action_loss + contrastive_loss
        return loss

    def save(self, path):
        params = dict({})
        params["action_context"] = self.action_context_model.state_dict()
        params["action"] = self.action_model.state_dict()
        params["cp_obs"] = self.norm_cp_obs.get_params()
        params["cp_act"] = self.norm_cp_act.get_params()
        torch.save(params, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.action_context_model.load_state_dict(checkpoint["action_context"])
        self.action_model.load_state_dict(checkpoint["action"])
        self.norm_cp_obs.set_params(checkpoint["cp_obs"])
        self.norm_cp_act.set_params(checkpoint["cp_act"])
