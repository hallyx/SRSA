# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from model import AutoencoderTransPN
from torch.optim.lr_scheduler import CosineAnnealingLR


class ChamferDistance(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        """
        The inputs are sets of d-dimensional points:
        x = {x_1, ..., x_n} and y = {y_1, ..., y_m}.

        Arguments:
            x: a float tensor with shape [b, d, n].
            y: a float tensor with shape [b, d, m].
        Returns:
            a float tensor with shape [].
        """
        x = x.unsqueeze(3)  # shape [b, d, n, 1]
        y = y.unsqueeze(2)  # shape [b, d, 1, m]

        # compute pairwise l2-squared distances
        d = torch.pow(x - y, 2)  # shape [b, d, n, m]
        d = d.sum(1)  # shape [b, n, m]

        min_for_each_x_i, _ = d.min(dim=2)  # shape [b, n]
        min_for_each_y_j, _ = d.min(dim=1)  # shape [b, m]

        distance = min_for_each_x_i.sum(1) + min_for_each_y_j.sum(1)  # shape [b]
        return distance.mean(0)


class Trainer:

    def __init__(self, num_steps, num_points, device):

        def weights_init(m):
            if isinstance(m, nn.Conv1d):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                init.ones_(m.weight)
                init.zeros_(m.bias)

        network = AutoencoderTransPN(num_points=num_points)
        self.network = network.apply(weights_init).to(device)

        self.loss = ChamferDistance()
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-3, weight_decay=1e-5)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_steps, eta_min=1e-7)

    def train_step(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, num_points].
        Returns:
            a float number.
        """

        encoding, x_restored = self.network(x)
        loss = self.loss(x, x_restored)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint)

    def evaluate(self, x):

        with torch.no_grad():
            encoding, x_restored = self.network(x)
            loss = self.loss(x, x_restored)

        return loss.item(), x_restored, encoding
