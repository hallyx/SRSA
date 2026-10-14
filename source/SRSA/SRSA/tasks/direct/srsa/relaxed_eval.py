"""FACA relaxed insertion check used for SRSA checkpoint evaluation."""

import torch


def relaxed_step(depth_fraction, lateral_error, radial_clearance, contact, current_official, streak):
    """Return the one-sided relaxed process, stable result, jam, and next streak."""
    lateral_tol = (radial_clearance * 2.0).clamp(min=0.0010, max=0.0030)
    jam = contact & ~current_official & (lateral_error > torch.maximum(radial_clearance, lateral_tol))
    process = (depth_fraction > 0.85) & (lateral_error <= lateral_tol) & ~jam
    next_streak = torch.where(process, streak + 1, torch.zeros_like(streak))
    stable = process & (next_streak >= 3)
    return process, stable, jam, next_streak
