#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${CHECKPOINT:-logs/rl_games/Assembly/test/nn/Assembly.pth}
UNSEEN_TEMPLATES=("0.75:1.5" "1.5:0.75" "3.0:1.5")

for template in "${UNSEEN_TEMPLATES[@]}"; do
  safe_template=${template/:/_}
  SRSA_NEWT_OBS=0 \
  SRSA_ENABLE_AXIAL_TASK_PARAM_SAMPLER=1 \
  SRSA_ENABLE_FLANGE_FORCE_SENSOR=1 \
  SRSA_FLANGE_FORCE_SENSOR_OBS=1 \
  SRSA_FLANGE_FORCE_SENSOR_SOURCE=held_sensor \
  SRSA_TASK_PARAM_OBS_MODE=task_vec \
  SRSA_AXIAL_FIXED_PLUG_SCALE=1 \
  SRSA_AXIAL_CLEARANCE_BASE=0.000114 \
  SRSA_AXIAL_CLEARANCE_DEPTH_TEMPLATES="${template}" \
  SRSA_AXIAL_CLEARANCE_JITTER_RATIO=0.0 \
  SRSA_AXIAL_DEPTH_BASE=0.015 \
  SRSA_AXIAL_DEPTH_JITTER_RATIO=0.0 \
  SRSA_AXIAL_INIT_ERROR_XY_RANGE=0.005,0.0010 \
  SRSA_AXIAL_INIT_ERROR_Z_RANGE=0.00,0.005 \
  SRSA_AXIAL_INIT_ERROR_YAW_RANGE=-0.15,0.15 \
  SRSA_AXIAL_VISUAL_NOISE_XY_RANGE=0.0,0.001 \
  SRSA_AXIAL_VISUAL_NOISE_Z_RANGE=0.0,0.0005 \
  SRSA_EVAL_FILENAME="evaluation_01125_unseen_${safe_template}.h5" \
  python source/SRSA/SRSA/tasks/direct/srsa/run_w_id.py \
    --sparse \
    --no_sbc \
    --task_param_obs \
    --task_param_obs_mode task_vec \
    --assembly_id 01125 \
    --checkpoint "${CHECKPOINT}" \
    --headless \
    --num_envs 128 \
    --log_eval \
    --num_eval_trials 1024 \
    --device cuda:0
done
