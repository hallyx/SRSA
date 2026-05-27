# SRSA 任务参数实验记录

本文档记录 SRSA task parameterization 相关修改、当前训练/评估参数、验证结果和脚本使用方式。后续实验继续在这里追加，避免关键信息散落在聊天记录和临时命令里。

## 当前目标

核心实验目标：

- 固定 plug 横截面尺寸。
- 通过改变 socket 横截面尺寸来控制 clearance。
- 通过改变目标插入深度/有效套合长度来控制 depth。
- depth 只影响 `insertion_depth / target_insertion_depth`，不改变 plug/socket 横截面和 clearance。
- 在训练中使用少量联合模板 `(gamma_c, gamma_H)`，并在模板附近加入连续扰动。
- 在评估中测试未见联合组合，并关闭扰动，得到精确测试点。

当前训练模板：

```text
(gamma_c, gamma_H)
(0.5, 0.5)
(0.5, 1.0)
(1.0, 1.0)
(2.0, 1.5)
(4.0, 2.0)
```

当前未见评估组合：

```text
(0.75, 1.5)
(1.5, 0.75)
(3.0, 1.5)
```

当前基准值：

```text
clearance_base = 0.000114 m
depth_base     = 0.015 m
```

## 修改进度

已完成：

- 从官方任务参数中导出 `outputs/srsa_task_params_all.csv`。
- 确认官方 cfg 中 plug/socket 直径为同一套 nominal 值，不能直接反映每个 id 的真实几何尺寸。
- 增加 mesh-derived 几何参数导出脚本，用 OBJ 估计每个 id 的几何 proxy。
- 增加 clearance 分析绘图脚本。
- 增加 reset-time 轴向任务参数 sampler。
- 支持固定 plug、只改变 socket 以控制 clearance。
- 支持 depth 独立采样，且只改变目标插入深度/有效套合长度。
- 支持 clearance/depth 联合模板采样。
- 支持 `task_param_obs_mode=task_vec`，使用 6D task vector。
- 更新 `train.sh` 使用 5 个训练联合模板。
- 更新 `eval.sh` 循环测试 3 个未见联合组合。
- 增加轻量 sampler debug 脚本，不启动 Isaac 即可验证参数分布。
- 增加默认录制相机和 `run_w_id.py --video` 视频录制透传。

待确认或后续可做：

- 在有可用 CUDA GPU 和完整 Isaac/Omniverse 资产缓存的环境下，运行完整小规模训练/评估 smoke test。
- 根据第一轮训练结果调整模板权重或 jitter 大小。
- 如需做 per-id 几何归一化，可以用 mesh-derived proxy 改造 `clearance_base`。

## 关键代码位置

任务参数配置：

```text
source/SRSA/SRSA/tasks/direct/srsa/assembly_task_param_cfg.py
```

运行时环境变量解析和 sampler 接线：

```text
source/SRSA/SRSA/tasks/direct/srsa/assembly_runtime_env.py
```

参数解析、effective task params、reset-time sampler：

```text
source/SRSA/SRSA/tasks/direct/srsa/task_param_utils.py
```

训练/评估 launcher：

```text
source/SRSA/SRSA/tasks/direct/srsa/run_w_id.py
```

当前训练入口：

```text
train.sh
```

当前评估入口：

```text
eval.sh
```

## 环境变量说明

核心几何参数：

```text
SRSA_AXIAL_FIXED_PLUG_SCALE=1
SRSA_AXIAL_CLEARANCE_BASE=0.000114
SRSA_AXIAL_DEPTH_BASE=0.015
```

联合模板采样：

```text
SRSA_AXIAL_CLEARANCE_DEPTH_TEMPLATES="0.5:0.5;0.5:1.0;1.0:1.0;2.0:1.5;4.0:2.0"
SRSA_AXIAL_CLEARANCE_JITTER_RATIO=0.10
SRSA_AXIAL_DEPTH_JITTER_RATIO=0.10
```

模板格式：

```text
"gamma_c:gamma_H;gamma_c:gamma_H;..."
```

可选模板权重：

```text
SRSA_AXIAL_CLEARANCE_DEPTH_TEMPLATE_WEIGHTS="1,1,1,1,1"
```

评估精确点时关闭扰动：

```text
SRSA_AXIAL_CLEARANCE_JITTER_RATIO=0.0
SRSA_AXIAL_DEPTH_JITTER_RATIO=0.0
```

初始误差和视觉噪声：

```text
SRSA_AXIAL_INIT_ERROR_XY_RANGE=0.005,0.0010
SRSA_AXIAL_INIT_ERROR_Z_RANGE=0.00,0.005
SRSA_AXIAL_INIT_ERROR_YAW_RANGE=-0.15,0.15
SRSA_AXIAL_VISUAL_NOISE_XY_RANGE=0.0,0.001
SRSA_AXIAL_VISUAL_NOISE_Z_RANGE=0.0,0.0005
```

任务参数观测：

```text
SRSA_TASK_PARAM_OBS_MODE=task_vec
```

`task_vec` 维度为 6：

```text
task_type_id_float
log_scale
clearance_abs_norm
clearance_rel_norm
depth_abs_norm
yaw_requirement_float
```

视频相机覆盖：

```text
SRSA_CAMERA_EYE=1.05,-0.85,0.45
SRSA_CAMERA_LOOKAT=0.55,0.0,0.18
SRSA_CAMERA_RESOLUTION=1280,720
SRSA_CAMERA_ENV_INDEX=0
```

兼容原始 SRSA checkpoint：

- `Assembly-Sparse-v0` / `Assembly-Direct-v0` 默认保持原始 24 维 policy observation。
- `SRSA_ENABLE_AXIAL_TASK_PARAM_SAMPLER=1` 才启用 reset-time 轴向任务参数 sampler。
- `SRSA_ENABLE_FLANGE_FORCE_SENSOR=1` 只启用力/接触诊断采集。
- `SRSA_FLANGE_FORCE_SENSOR_OBS=1` 才把 3 维 flange force observation 拼到 policy observation；对原始 `00783.pth` 做诊断时用 `0`。
- 因此原始 `checkpoints/00783.pth` 可直接用 `run_w_id.py --video` 录制；task-param/force-sensor 新模型继续用 `train.sh`、`eval.sh` 中的显式环境变量。

## 训练脚本

当前训练命令：

```bash
bash train.sh
```

当前 `train.sh` 关键配置：

```bash
SRSA_ENABLE_AXIAL_TASK_PARAM_SAMPLER=1
SRSA_ENABLE_FLANGE_FORCE_SENSOR=1
SRSA_FLANGE_FORCE_SENSOR_OBS=1
SRSA_AXIAL_FIXED_PLUG_SCALE=1
SRSA_AXIAL_CLEARANCE_BASE=0.000114
SRSA_AXIAL_CLEARANCE_DEPTH_TEMPLATES="0.5:0.5;0.5:1.0;1.0:1.0;2.0:1.5;4.0:2.0"
SRSA_AXIAL_CLEARANCE_JITTER_RATIO=0.10
SRSA_AXIAL_DEPTH_BASE=0.015
SRSA_AXIAL_DEPTH_JITTER_RATIO=0.10
```

含义：

- 每次 reset 先随机抽一个联合模板。
- 在该模板的 `gamma_c` 附近乘以 `[0.9, 1.1]` 连续扰动。
- 在该模板的 `gamma_H` 附近乘以 `[0.9, 1.1]` 连续扰动。
- plug scale 固定为 1。
- socket diameter = plug diameter + sampled diametral clearance。
- insertion depth = `depth_base * sampled gamma_H`。

## 评估脚本

当前评估命令：

```bash
bash eval.sh
```

`eval.sh` 会循环跑 3 个未见组合：

```text
0.75:1.5
1.5:0.75
3.0:1.5
```

评估时 jitter 关闭：

```bash
SRSA_AXIAL_CLEARANCE_JITTER_RATIO=0.0
SRSA_AXIAL_DEPTH_JITTER_RATIO=0.0
```

每个组合会写独立 eval 文件：

```text
evaluation_01125_unseen_0.75_1.5.h5
evaluation_01125_unseen_1.5_0.75.h5
evaluation_01125_unseen_3.0_1.5.h5
```

如果要指定 checkpoint：

```bash
CHECKPOINT=logs/rl_games/Assembly/your_run/nn/Assembly.pth bash eval.sh
```

## 00783 力和卡滞诊断

`--force_diagnostics` 会启用 flange/held-asset 接触力采集，但保持原始 24 维 policy observation，适合直接加载 `checkpoints/00783.pth`：

```bash
python source/SRSA/SRSA/tasks/direct/srsa/run_w_id.py \
  --assembly_id 00783 \
  --checkpoint checkpoints/00783.pth \
  --sparse \
  --headless \
  --log_eval \
  --num_eval_trials 1024 \
  --force_diagnostics \
  --flange_force_source held_sensor \
  --flange_force_threshold 1.0
```

生成的 `evaluation_00783.h5` 除原有 `held_asset_pose`、`fixed_asset_pose`、`success` 外，还包含：

```text
force_max
force_mean
force_final
force_world_final
force_socket_final
contact
contact_steps
jam
jam_steps
lateral_error_max
lateral_error_final
depth_fraction_max
depth_fraction_final
current_depth_final
target_depth_final
orientation_error_final
yaw_error_final
keypoint_error_final
```

## 视频录制

SRSA 默认 viewer 相机已固定到 env0 的装配区域。通过 `run_w_id.py` 加 `--video` 即可录制实验过程视频；底层会自动启用 IsaacLab camera rendering。默认入口兼容原始 24 维 SRSA checkpoint。视频保存不依赖任务成功；脚本会在失败、提前结束或内部 `exit(0)` 时尽量关闭 `RecordVideo` 并 flush 已录制帧。

评估录制示例：

```bash
python source/SRSA/SRSA/tasks/direct/srsa/run_w_id.py \
  --assembly_id 00783 \
  --checkpoint checkpoints/00783.pth \
  --sparse \
  --headless \
  --video \
  --video_length 300
```

训练录制示例：

```bash
python source/SRSA/SRSA/tasks/direct/srsa/run_w_id.py \
  --assembly_id 01125 \
  --train \
  --sparse \
  --headless \
  --video \
  --video_length 300 \
  --video_interval 2000
```

可选相机覆盖：

```bash
--camera_eye 1.05 -0.85 0.45 \
--camera_lookat 0.55 0.0 0.18 \
--camera_resolution 1280 720 \
--camera_env_index 0
```

视频输出路径遵循 IsaacLab/RL-Games 默认目录：

```text
logs/rl_games/Assembly/<run>/videos/play
logs/rl_games/Assembly/<run>/videos/train
```

如果 checkpoint 直接来自仓库内 `checkpoints/` 目录，当前 `play.py` 会把视频写到：

```text
videos/play
```

play 模式视频文件名使用当前 assembly id，例如：

```text
00783.mp4
```

train 模式可能会按间隔保存多段视频，文件名前缀同样使用当前 assembly id，例如 `00783-step-2000.mp4`。

## 轻量参数 Debug

脚本：

```text
scripts/debug_srsa_task_param_sampler.py
```

用途：

- 不启动 Isaac Sim。
- 直接读取 `SRSA_AXIAL_*` 环境变量。
- 调用真实 `AxialTaskParamSampler`。
- 检查 plug 是否固定、clearance/depth multiplier 是否正确、联合模板是否覆盖、jitter 是否落在范围内、`task_vec` 是否一致。

验证当前训练参数：

```bash
SRSA_AXIAL_FIXED_PLUG_SCALE=1 \
SRSA_AXIAL_CLEARANCE_BASE=0.000114 \
SRSA_AXIAL_CLEARANCE_DEPTH_TEMPLATES="0.5:0.5;0.5:1.0;1.0:1.0;2.0:1.5;4.0:2.0" \
SRSA_AXIAL_CLEARANCE_JITTER_RATIO=0.10 \
SRSA_AXIAL_DEPTH_BASE=0.015 \
SRSA_AXIAL_DEPTH_JITTER_RATIO=0.10 \
/home/gpuserver/miniconda3/envs/isaac51/bin/python scripts/debug_srsa_task_param_sampler.py --samples 50000
```

当前验证结果摘要：

```text
plug_scale_xy: min=1 max=1
clearance_multiplier: min=0.4500 max=4.3999
depth_multiplier: min=0.4500 max=2.1999
template_counts=[9940, 10111, 9708, 10084, 10157]
all checks OK
```

验证未见评估点：

```bash
for template in "0.75:1.5" "1.5:0.75" "3.0:1.5"; do
  SRSA_AXIAL_FIXED_PLUG_SCALE=1 \
  SRSA_AXIAL_CLEARANCE_BASE=0.000114 \
  SRSA_AXIAL_CLEARANCE_DEPTH_TEMPLATES="${template}" \
  SRSA_AXIAL_CLEARANCE_JITTER_RATIO=0.0 \
  SRSA_AXIAL_DEPTH_BASE=0.015 \
  SRSA_AXIAL_DEPTH_JITTER_RATIO=0.0 \
  /home/gpuserver/miniconda3/envs/isaac51/bin/python scripts/debug_srsa_task_param_sampler.py --samples 2048
done
```

当前验证结果：

```text
(0.75, 1.5): clearance=0.0000855, depth=0.0225
(1.5, 0.75): clearance=0.000171,  depth=0.01125
(3.0, 1.5):  clearance=0.000342,  depth=0.0225
```

## Runtime 接线 Debug

已用 stub 配置直接调用：

```text
AssemblyRuntimeEnvMixin._apply_runtime_task_overrides(cfg)
```

验证结果：

```text
runtime_parse_ok
task_param_obs_mode task_vec dim 6
use_task_param True enable_sampler True
fixed_plug True
clearance_base 0.000114 depth_base 0.015
templates [[0.5, 0.5], [0.5, 1.0], [1.0, 1.0], [2.0, 1.5], [4.0, 2.0]]
jitters 0.1 0.1
```

说明训练脚本中的环境变量能够被真实 runtime 解析成预期配置。

## 完整 Isaac Smoke Test 状态

已尝试运行极小环境测试：

```bash
TERM=xterm \
CONDA_PREFIX=/home/gpuserver/miniconda3/envs/isaac51 \
/home/gpuserver/IsaacLab/isaaclab.sh -p scripts/test_flange_force_sensor.py \
  --task Assembly-Sparse-v0 \
  --assembly_id 01125 \
  --num_envs 2 \
  --steps 0 \
  --task_param_obs \
  --task_param_obs_mode task_vec \
  --headless \
  --device cuda:0
```

当前会话中 `cuda:0` 失败：

```text
RuntimeError: No CUDA GPUs are available
```

改用 CPU 后进入 Isaac 场景创建，但缺远程 ground plane USD 资源：

```text
FileNotFoundError:
Unable to open the usd file at path:
https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Grid/default_environment.usd
```

因此当前已经验证参数逻辑和 runtime 解析，但完整 Isaac 环境实例化需要：

- 可见 CUDA GPU。
- Isaac/Omniverse 远程资产可访问，或本地缓存完整。

## Clearance 分析脚本

脚本：

```text
scripts/analyze_srsa_clearance.py
```

用途：

- 读取 `outputs/srsa_task_params_all.csv`。
- 按 assembly id 和 task family 汇总 clearance。
- 生成 summary CSV、pivot CSV 和图。

默认命令：

```bash
python scripts/analyze_srsa_clearance.py
```

常用参数：

```bash
python scripts/analyze_srsa_clearance.py \
  --input_csv outputs/srsa_task_params_all.csv \
  --output_dir outputs/clearance_analysis \
  --metric diametral_clearance
```

可选 metric：

```text
diametral_clearance
radial_clearance
clearance_ratio
```

输出目录：

```text
outputs/clearance_analysis/
```

## Mesh 几何参数导出脚本

脚本：

```text
scripts/export_srsa_mesh_geometry_params.py
```

用途：

- 读取每个 assembly id 的 `plug.obj` 和 `socket.obj`。
- 导出 mesh-derived 几何 proxy。
- 这些值不是官方标注直径，而是从 mesh 估计出来的几何 proxy。

默认命令：

```bash
python scripts/export_srsa_mesh_geometry_params.py
```

指定 mesh root 和输出目录：

```bash
python scripts/export_srsa_mesh_geometry_params.py \
  --mesh_root /home/gpuserver/hx/github/SRSA_restore/data/mesh \
  --output_dir outputs/mesh_geometry_params \
  --sample_count 2048
```

只导出指定 id：

```bash
python scripts/export_srsa_mesh_geometry_params.py \
  --assembly_ids 01125,00783,00426
```

主要输出：

```text
outputs/mesh_geometry_params/srsa_mesh_geometry_params.csv
outputs/mesh_geometry_params/mesh_geometry_report.txt
```

## 任务参数 CSV

当前任务参数总表：

```text
outputs/srsa_task_params_all.csv
```

用途：

- 检查官方 task family 参数。
- 和 mesh-derived proxy 对比。
- 作为 clearance 分析脚本的默认输入。

注意：

- 官方 task cfg 中的 diameter/clearance 对所有 id 使用 nominal 值，不能直接说明每个 mesh 的真实几何差异。
- 每个 id 的真实几何差异应优先参考 mesh-derived 输出。

## 常用检查命令

语法检查：

```bash
python -m py_compile \
  scripts/debug_srsa_task_param_sampler.py \
  source/SRSA/SRSA/tasks/direct/srsa/task_param_utils.py \
  source/SRSA/SRSA/tasks/direct/srsa/assembly_runtime_env.py \
  source/SRSA/SRSA/tasks/direct/srsa/assembly_task_param_cfg.py \
  source/SRSA/SRSA/tasks/direct/srsa/run_w_id.py
```

shell 脚本检查：

```bash
bash -n train.sh eval.sh
```

查看当前改动：

```bash
git status --short
git diff -- train.sh eval.sh scripts/debug_srsa_task_param_sampler.py
git diff -- source/SRSA/SRSA/tasks/direct/srsa/task_param_utils.py
git diff -- source/SRSA/SRSA/tasks/direct/srsa/assembly_runtime_env.py
```

## 当前结论

截至本记录：

- 参数训练配置是正确接线的。
- 训练会从 5 个联合模板中采样，并在模板附近加 10% 连续扰动。
- plug 横截面保持固定。
- clearance 通过 socket 横截面变化实现。
- depth 只改变目标插入深度/有效套合长度。
- 评估脚本会测试 3 个未见组合，且关闭 jitter。
- 完整 Isaac smoke test 受当前运行环境限制，尚未在本会话完成。
