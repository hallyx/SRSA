# SRSA Franka Force Sensor and Debug Notes

本文件记录当前仓库中针对 Franka AutoMate/SRSA 装配环境添加力传感器与调试脚本的改动，便于后期维护和复现实验。

## 修改范围

只修改当前仓库中的 SRSA Franka task-param 环境，不迁移 UR10e、Robotiq、gripper lock、UR reset、UR force sensor 相关逻辑。

主要文件：

- `source/SRSA/SRSA/tasks/direct/srsa/assembly_task_param_cfg.py`
- `source/SRSA/SRSA/tasks/direct/srsa/assembly_runtime_env.py`
- `scripts/test_flange_force_sensor.py`

参考目录：

- `/home/gpuserver/hx/github/SRSA_restore`

## 力传感器设计

当前环境使用 Franka Panda/AutoMate Franka 资产，机器人 body/joint 名包括 `panda_hand`、`panda_leftfinger`、`panda_rightfinger`、`panda_fingertip_centered`。

最初尝试将 ContactSensor 挂在：

```text
/World/envs/env_.*/Robot/panda_hand
```

该传感器可成功创建，但在装配接触中读数通常为 0。原因是视觉上看到的接触主要发生在 held plug 与 socket 之间，而不是 `panda_hand` 刚体本身直接碰撞。

最终有效力源为 held plug 上的标准 IsaacLab `ContactSensor`：

```text
/World/envs/env_.*/HeldAsset/.*
```

默认配置：

```text
enable_flange_force_sensor = True
flange_force_sensor_source = "held_sensor"
flange_force_sensor_obs_frame = "socket"
flange_force_sensor_obs_scale = 50.0
flange_force_sensor_force_threshold = 1.0
```

观测拼接：

- 原始 policy obs: 24 维
- 力观测: 3 维
- 任务参数观测: 9 维，可选

因此：

- 不启用任务参数观测时，`policy_obs_shape[-1] = 27`
- 启用任务参数观测时，`policy_obs_shape[-1] = 36`

## Runtime 输出

`AssemblyRuntimeEnvMixin` 中新增的关键张量：

```text
flange_force_world
flange_force_socket
flange_force_obs
flange_force_norm
flange_force_flag
flange_body_contact_force_world
held_sensor_contact_force_world
held_asset_contact_force_world
```

含义：

- `flange_force_world`: 当前被选为 policy obs 来源的 world-frame 力。
- `flange_force_socket`: 将 `flange_force_world` 转到 socket/fixed frame。
- `flange_force_obs`: 缩放后的 3 维力观测，默认除以 50。
- `flange_force_norm`: world-frame 力范数。
- `flange_force_flag`: `flange_force_norm > threshold`。
- `flange_body_contact_force_world`: `panda_hand` ContactSensor 输出，通常为 0。
- `held_sensor_contact_force_world`: held plug ContactSensor 输出，当前有效力源。
- `held_asset_contact_force_world`: 通过 `root_physx_view.get_net_contact_forces()` 读取的 held asset 力，在当前资产结构下验证为 0，不建议作为主要力源。

## Debug 脚本

综合调试脚本：

```bash
conda run -n isaac51 python scripts/test_flange_force_sensor.py \
  --headless \
  --assembly_id 00783 \
  --num_envs 4 \
  --steps 120 \
  --print_every 10 \
  --action_mode down \
  --require_nonzero \
  --expected_policy_obs_dim 27
```

如果需要同时验证任务参数观测：

```bash
conda run -n isaac51 python scripts/test_flange_force_sensor.py \
  --headless \
  --assembly_id 00783 \
  --task_family_name normal_fit \
  --task_param_obs \
  --num_envs 4 \
  --steps 120 \
  --print_every 10 \
  --action_mode down \
  --require_nonzero \
  --expected_policy_obs_dim 36
```

脚本默认使用：

```text
--step_mode physics
```

该模式只推进物理和 scene/sensor update，不调用原始 reward，因此可避开 AutoMate imitation reward 的 SoftDTW CUDA/Numba 路径。若使用：

```text
--step_mode env
```

则会走完整 `env.step()`，可能触发 SoftDTW CUDA 相关问题。

## Debug 输出判读

关键力相关行：

```text
world
body_sensor_world
held_sensor_world
held_asset_world
socket
obs
norm
flag_count
```

当前验证结果：

- `held_sensor_world` 非零，力传感链路正常。
- `world/socket/obs` 与 `held_sensor_world` 一致，说明默认力源选取正确。
- `body_sensor_world` 为 0，符合预期，因为 `panda_hand` 未直接接触。
- `held_asset_world` 为 0，说明 `root_physx_view.get_net_contact_forces()` 在当前 held asset 结构下不可靠。
- `flag_count=4` 表示 4 个 env 都超过阈值。
- `PASS` 表示 `--require_nonzero` 条件满足。

任务参数检查重点：

```text
use_task_family=True
use_task_param=True
task_param_obs=True/False
current_task_param_tensor_shape=(num_envs, 9)
policy_obs_shape=(num_envs, 36)  # task_param_obs=True
policy_obs_shape=(num_envs, 27)  # task_param_obs=False
```

误差检查重点：

```text
tcp_goal_error_norm
tcp_goal_error_xy
tcp_goal_error_z
held_fixed_socket_xy
held_fixed_socket_z
keypoint_dist
ctrl_target_error_norm
```

在 `--action_mode down` 测试中，期望看到 plug/TCP 向 socket/goal 靠近，同时接触力上升。若 `ctrl_target_error_norm` 稳定在较大值，通常说明动作持续要求继续下压，但接触/限幅/控制器阻止实际 TCP 继续跟随；这不代表力传感失败。

性能检查重点：

```text
first_nonzero_step
first_flag_step
nonzero_step_count
flag_step_count
mean_force_norm
rms_force_norm
avg_step_ms
max_step_ms
```

`avg_step_ms` 和 `max_step_ms` 来自 Python debug 脚本和大量打印，不代表训练吞吐。

## 已验证样例

命令：

```bash
conda run -n isaac51 python scripts/test_flange_force_sensor.py \
  --headless \
  --assembly_id 00783 \
  --task_family_name normal_fit \
  --task_param_obs \
  --num_envs 4 \
  --steps 120 \
  --print_every 10 \
  --action_mode down \
  --require_nonzero \
  --expected_policy_obs_dim 36
```

验证结果摘要：

```text
policy_obs_shape=(4, 36)
held_sensor_world != 0
flag_count=4 after contact
PASS
max_norm_seen ~= 21.69 N
mean_force_norm ~= 4.74 N
first_nonzero_step=0
first_flag_step=5
```

## 维护注意事项

1. 如果更换 robot asset 或 body 名，先确认 `panda_hand` 是否仍存在；但默认有效力源是 `HeldAsset/.*`，不依赖 `panda_hand` 直接接触。
2. 如果更换 held asset USD 层级，确认 `/World/envs/env_.*/HeldAsset/.*` 是否仍能匹配到带 `PhysxContactReportAPI` 的 rigid body。
3. 不建议重新启用 FixedAsset filter，之前出现过：

   ```text
   Filter pattern '/World/envs/env_*/FixedAsset/*' did not match the correct number of entries
   ```

   原因是每个 env 的 FixedAsset 匹配到多个 rigid body。当前只需要 net force，因此不使用 filter。

4. 如果训练脚本加载旧 checkpoint，注意 policy observation 维度变化：

   - 旧策略若按 24 维训练，启用 force obs 后不能直接加载到同结构网络。
   - force obs 默认增加 3 维。
   - task param obs 额外增加 9 维。

5. 若要真机 FR3/Panda 使用该力观测，需要额外实现真机力源适配：

   - 外置 F/T 传感器，或
   - 通过关节力矩估计 TCP 外力，或
   - 机器人控制器提供的 external wrench。

   当前 `held_sensor` 是仿真中的 plug/socket 接触力代理，不是直接可用于真机的硬件接口。
