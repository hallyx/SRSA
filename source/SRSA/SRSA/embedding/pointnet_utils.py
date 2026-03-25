# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Masked aggregation utilities
# ---------------------------------------------------------------------------

def masked_average(x, axis, mask=None, keepdim=False):
    if mask is None:
        return torch.mean(x, dim=axis, keepdim=keepdim)
    else:
        return torch.sum(x * mask, dim=axis, keepdim=keepdim) / (torch.sum(mask, dim=axis, keepdim=keepdim) + 1e-6)


def masked_max(x, axis, mask=None, keepdim=False, empty_value=0):
    if mask is None:
        return torch.max(x, dim=axis, keepdim=keepdim).values
    else:
        value_with_inf = torch.max(x * mask + -1e18 * (1 - mask), dim=axis, keepdim=keepdim).values
        return torch.where(value_with_inf > -1e17, value_with_inf, torch.ones_like(value_with_inf) * empty_value)


# ---------------------------------------------------------------------------
# Backbone registry and MLP builders
# ---------------------------------------------------------------------------

_BACKBONES = {}


def _register(cls):
    _BACKBONES[cls.__name__] = cls
    return cls


def build_backbone(cfg):
    cfg = cfg.copy()
    cls_name = cfg.pop("type")
    return _BACKBONES[cls_name](**cfg)


def _xavier_init(module, gain=1, bias=0, **kwargs):
    nn.init.xavier_normal_(module.weight, gain=gain)
    if module.bias is not None:
        nn.init.constant_(module.bias, bias)


class _ConvAct(nn.Module):
    """Single Conv1d + optional ReLU, structured to match original state dict keys."""

    def __init__(self, in_channels, out_channels, bias, with_act):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        if with_act:
            self.activate = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if hasattr(self, "activate"):
            x = self.activate(x)
        return x


@_register
class ConvMLP(nn.Module):
    """1D conv MLP. State dict keys: mlp.layer{i}.conv.{weight,bias}"""

    def __init__(self, mlp_spec, norm_cfg=None, bias="auto", inactivated_output=True,
                 conv_init_cfg=None, norm_init_cfg=None, pretrained=None):
        super().__init__()
        self.mlp = nn.Sequential()
        for i in range(len(mlp_spec) - 1):
            with_norm = norm_cfg is not None
            use_bias = (not with_norm) if bias == "auto" else bias
            is_last = (i == len(mlp_spec) - 2) and inactivated_output
            self.mlp.add_module(f"layer{i}", _ConvAct(mlp_spec[i], mlp_spec[i + 1], use_bias, not is_last))
            if with_norm:
                self.mlp.add_module(f"norm{i}", nn.BatchNorm1d(mlp_spec[i + 1]))

        if conv_init_cfg is not None and conv_init_cfg.get("type") == "xavier_init":
            kw = {k: v for k, v in conv_init_cfg.items() if k != "type"}
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    _xavier_init(m, **kw)

    def forward(self, x):
        return self.mlp(x)


@_register
class LinearMLP(nn.Module):
    """Linear MLP. State dict keys: mlp.linear{i}.{weight,bias}"""

    def __init__(self, mlp_spec, norm_cfg=None, bias="auto", inactivated_output=True,
                 linear_init_cfg=None, norm_init_cfg=None, pretrained=None):
        super().__init__()
        self.mlp = nn.Sequential()
        for i in range(len(mlp_spec) - 1):
            with_norm = norm_cfg is not None
            use_bias = (not with_norm) if bias == "auto" else bias
            is_last = (i == len(mlp_spec) - 2) and inactivated_output
            self.mlp.add_module(f"linear{i}", nn.Linear(mlp_spec[i], mlp_spec[i + 1], bias=use_bias))
            if with_norm:
                self.mlp.add_module(f"norm{i}", nn.BatchNorm1d(mlp_spec[i + 1]))
            if not is_last:
                self.mlp.add_module(f"act{i}", nn.ReLU(inplace=True))

        if linear_init_cfg is not None and linear_init_cfg.get("type") == "xavier_init":
            kw = {k: v for k, v in linear_init_cfg.items() if k != "type"}
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    _xavier_init(m, **kw)

    def forward(self, x):
        return self.mlp(x)


# ---------------------------------------------------------------------------
# PointNet backbone base class
# ---------------------------------------------------------------------------

class PointBackbone(nn.Module):
    def __init__(self):
        super().__init__()

    def forward_raw(self, pcd, mask=None):
        raise NotImplementedError
