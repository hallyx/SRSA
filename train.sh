#!/usr/bin/env bash
set -euo pipefail

SRSA_NEWT_OBS=0 \
SRSA_ENABLE_AXIAL_TASK_PARAM_SAMPLER=1 \
SRSA_ENABLE_FLANGE_FORCE_SENSOR=1 \
SRSA_FLANGE_FORCE_SENSOR_SOURCE=held_sensor \
SRSA_TASK_PARAM_OBS_MODE=task_vec \
SRSA_AXIAL_FIXED_PLUG_SCALE=1 \
SRSA_AXIAL_CLEARANCE_BASE=0.000114 \
SRSA_AXIAL_CLEARANCE_DEPTH_TEMPLATES="0.5:0.5;0.5:1.0;1.0:1.0;2.0:1.5;4.0:2.0" \
SRSA_AXIAL_CLEARANCE_JITTER_RATIO=0.10 \
SRSA_AXIAL_DEPTH_BASE=0.015 \
SRSA_AXIAL_DEPTH_JITTER_RATIO=0.10 \
SRSA_AXIAL_INIT_ERROR_XY_RANGE=0.005,0.0010 \
SRSA_AXIAL_INIT_ERROR_Z_RANGE=0.00,0.005 \
SRSA_AXIAL_INIT_ERROR_YAW_RANGE=-0.15,0.15 \
SRSA_AXIAL_VISUAL_NOISE_XY_RANGE=0.0,0.001 \
SRSA_AXIAL_VISUAL_NOISE_Z_RANGE=0.0,0.0005 \
/home/gpuserver/miniconda3/envs/isaac51/bin/python source/SRSA/SRSA/tasks/direct/srsa/run_w_id.py \
  --sparse \
  --no_sbc \
  --task_param_obs \
  --task_param_obs_mode task_vec \
  --assembly_id 01125 \
  --train \
  --num_envs 256 \
  --headless \
  --device cuda:1 \
  --seed 0
