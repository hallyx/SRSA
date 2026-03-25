# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.functional as F


class RunningMeanStd(nn.Module):
    def __init__(self, insize, device, epsilon=1e-05, per_channel=False, norm_only=False):
        super().__init__()
        print("RunningMeanStd: ", insize)
        self.insize = insize
        self.epsilon = epsilon

        self.norm_only = norm_only
        self.per_channel = per_channel
        if per_channel:
            if len(self.insize) == 3:
                self.axis = [0, 2, 3]
            if len(self.insize) == 2:
                self.axis = [0, 2]
            if len(self.insize) == 1:
                self.axis = [0]
            in_size = self.insize[0]
        else:
            self.axis = [0]
            in_size = insize

        self.register_buffer("running_mean", torch.zeros(in_size, device=device, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(in_size, device=device, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), device=device, dtype=torch.float64))

    def _update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    def forward(self, input, denorm=False, mask=None, training=True):
        if training:
            if mask is not None:
                mean, var = torch_ext.get_mean_std_with_masks(input, mask)
            else:
                mean = input.mean(self.axis)  # along channel axis
                var = input.var(self.axis)
            self.running_mean, self.running_var, self.count = self._update_mean_var_count_from_moments(
                self.running_mean, self.running_var, self.count, mean, var, input.size()[0]
            )

        # change shape
        if self.per_channel:
            if len(self.insize) == 3:
                current_mean = self.running_mean.view([1, self.insize[0], 1, 1]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0], 1, 1]).expand_as(input)
            if len(self.insize) == 2:
                current_mean = self.running_mean.view([1, self.insize[0], 1]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0], 1]).expand_as(input)
            if len(self.insize) == 1:
                current_mean = self.running_mean.view([1, self.insize[0]]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0]]).expand_as(input)
        else:
            current_mean = self.running_mean
            current_var = self.running_var
        # get output

        if denorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = torch.sqrt(current_var.float() + self.epsilon) * y + current_mean.float()
        else:
            if self.norm_only:
                y = input / torch.sqrt(current_var.float() + self.epsilon)
            else:
                y = (input - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
                y = torch.clamp(y, min=-5.0, max=5.0)
        return y

    def get_params(self):
        return {"running_mean": self.running_mean, "running_var": self.running_var, "count": self.count}

    def set_params(self, x):
        self.running_mean = x["running_mean"]
        self.running_var = x["running_var"]
        self.count = x["count"]


class ContextDynamics(nn.Module):

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

        self.norm_obs = RunningMeanStd(obs_dim, device=device)
        self.norm_act = RunningMeanStd(act_dim, device=device)
        self.norm_delta = RunningMeanStd(obs_dim, device=device)
        self.norm_cp_obs = RunningMeanStd(obs_dim * history_length, device=device)
        self.norm_cp_act = RunningMeanStd(act_dim * history_length, device=device)

        self.context_model = nn.Sequential()
        cp_hidden_sizes = [
            history_length * (obs_dim + act_dim),
        ] + list(cp_hidden_sizes)
        for i in range(len(cp_hidden_sizes) - 1):
            self.context_model.add_module("dense%d" % i, nn.Linear(cp_hidden_sizes[i], cp_hidden_sizes[i + 1]))
            self.context_model.add_module("act%d" % i, nn.ReLU())
        self.context_model.add_module("output", nn.Linear(cp_hidden_sizes[-1], context_out_dim))
        self.context_model.to(device)

        self.forward_model = nn.Sequential()
        hidden_sizes = [
            context_out_dim + obs_dim + act_dim,
        ] + list(hidden_sizes)
        for i in range(len(hidden_sizes) - 1):
            self.forward_model.add_module("dense%d" % i, nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.forward_model.add_module("act%d" % i, nn.ReLU())
        self.forward_model.add_module("output", nn.Linear(hidden_sizes[-1], obs_dim))
        self.forward_model.to(device)

        self.forward_loss = nn.MSELoss()
        self.margin = margin

    def embed(self, cp_obs, cp_act, training=True):
        normalized_cp_obs = self.norm_cp_obs(cp_obs, training=training)
        normalized_cp_act = self.norm_cp_act(cp_act, training=training)
        normalized_cp_x = torch.cat([normalized_cp_obs, normalized_cp_act], dim=1)

        cp_output = self.context_model(normalized_cp_x)
        return cp_output

    def set_train(self):
        self.context_model.train()
        self.forward_model.train()

    def get_parameters(self):
        return list(self.context_model.parameters()) + list(self.forward_model.parameters())

    def forward(self, cp_obs, cp_act, obs, act, next_obs, other_cp_obs, other_cp_act, target, training=True):

        delta = next_obs - obs
        normalized_obs = self.norm_obs(obs, training=training)
        normalized_act = self.norm_act(act, training=training)
        normalized_delta = self.norm_delta(delta, training=training)

        cp_output = self.embed(cp_obs, cp_act, training=training)

        x = torch.cat([normalized_obs, normalized_act, cp_output], dim=1)
        output = self.forward_model(x)
        forward_loss = self.forward_loss(output, normalized_delta)

        other_cp_output = self.embed(other_cp_obs, other_cp_act, training=False)
        euclidean_distance = F.pairwise_distance(cp_output, other_cp_output)
        contrastive_loss = torch.mean(
            (1 - target) * torch.pow(euclidean_distance, 2)
            + (target) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        loss = forward_loss + contrastive_loss
        return loss

    def save(self, path):
        params = dict({})
        params["context"] = self.context_model.state_dict()
        params["forward"] = self.forward_model.state_dict()
        params["cp_obs"] = self.norm_cp_obs.get_params()
        params["cp_act"] = self.norm_cp_act.get_params()
        params["delta"] = self.norm_delta.get_params()
        params["obs"] = self.norm_obs.get_params()
        params["act"] = self.norm_act.get_params()
        torch.save(params, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.context_model.load_state_dict(checkpoint["context"])
        self.forward_model.load_state_dict(checkpoint["forward"])
        self.norm_cp_obs.set_params(checkpoint["cp_obs"])
        self.norm_cp_act.set_params(checkpoint["cp_act"])
        self.norm_obs.set_params(checkpoint["obs"])
        self.norm_act.set_params(checkpoint["act"])
        self.norm_delta.set_params(checkpoint["delta"])
