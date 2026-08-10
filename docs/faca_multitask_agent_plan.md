# SRSA × FACA M0：单环境异构多任务训练

本文描述当前 `multitask-trainning` 分支已经实现的环境端方案。目标是让一个
Isaac Lab 向量环境同时承载 50–60 个真实 AutoMate assembly，并由一个
task-conditioned TD-MPC2 M0 agent 共同训练。

## 1. 已实现的结构

一个训练进程只创建一次 Isaac Sim 和一次 `Assembly-Direct-v0`：

```text
one Isaac process
  └── one vector environment (num_envs = E)
        ├── env_0  -> assembly task 0
        ├── env_1  -> assembly task 1
        ├── ...
        ├── env_59 -> assembly task 59
        └── env_i  -> task (i mod N)
```

这里的“一个 env”指一个 Isaac Lab vector environment；其中的 replica 加载不同
socket/plug USD。任务分配在场景创建时固定，reset 只重置该 replica 的状态，不更换任务。

- `E >= N`，否则启动时报错，避免有任务没有并行环境。
- `E == N` 时每个任务一个 replica；`E == 2N` 时每个任务两个 replica。
- 建议 `E` 是 `N` 的整数倍，以获得严格均衡的在线采样。
- 异构 USD 要求 `scene.replicate_physics = False`。
- 实际 assembly geometry 与 task vector 静态绑定，因此多任务模式会关闭 reset-time
  geometry scaling 和 axial task sampler。
- 当前只支持 `physical_grasp`；`rigid_weld` 是审计模式，多任务下会直接拒绝启动。

## 2. FACA M0 环境合同

SRSA 暴露以下稳定接口：

```python
inputs = env.unwrapped.get_faca_m0_inputs()
state = inputs["state"]       # [num_envs, 14]
task = inputs["task"]         # [num_envs, 6]
task_ids = env.unwrapped.get_faca_m0_task_indices()
assembly_ids = env.unwrapped.get_faca_m0_assembly_ids()
metadata = env.unwrapped.get_faca_m0_contract()
```

14D state 的字段顺序为：

```text
tcp_pos_socket[3]
tcp_quat_socket_wxyz_canonical[4]
tcp_linvel_socket[3]
tcp_angvel_socket[3]
gripper_width[1]
```

6D task vector 的字段顺序为：

```text
task_type_id
log_scale
clearance_abs_norm
clearance_rel_norm
depth_abs_norm
yaw_requirement
```

task vector 不拼进 14D state，而是单独交给 FACA 的 `AxialTaskEncoder`。每个 env 还会使用
对应 assembly 的 grasp、插入深度、成功阈值、gripper width、Warp mesh、SDF sample points 和
imitation trajectory。因此当前实现不是只更换观测标签，而是 geometry、reset target、reward
和 success 都沿同一份 env-to-task mapping。

## 3. Manifest

训练模式必须提供含完整 `task_vec_6` 的 JSON manifest。最小格式如下：

```json
{
  "tasks": [
    {
      "assembly_id": "00062",
      "task_vec_6": [0, 0.1, 0.2, 0.3, 1.0, 0],
      "srsa_params": {
        "insertion_depth": 0.015,
        "success_pos_tol": 0.015
      }
    }
  ]
}
```

可以直接从 FACA 的 mesh CSV 和 axial template 生成 60 任务 manifest：

```bash
cd /home/gpuserver/hx/github/srsa

/home/gpuserver/miniconda3/envs/isaac51/bin/python \
  scripts/build_faca_multitask_manifest.py \
  --mesh_csv /media/gpuserver/data2t/hx/github/FACA/data/srsa_mesh_geometry_params.csv \
  --template_fp /media/gpuserver/data2t/hx/github/FACA/data/srsa_axial_task_templates.json \
  --template_id 2 \
  --limit 60 \
  --reference_anchor_assembly_id 01125 \
  --output /tmp/srsa_faca_m0_60_tasks.json
```

`--assembly_ids` 可显式指定任务集合和顺序；省略时按 CSV 顺序选择，`--limit` 截断数量。

## 4. 启动前验证

两任务快速验证：

```bash
/home/gpuserver/miniconda3/envs/isaac51/bin/python \
  scripts/smoke_multitask_env.py \
  --headless --device cuda:0 \
  --manifest /tmp/srsa_faca_m0_60_tasks.json \
  --assembly_ids 00062,00186 \
  --require_task_vectors \
  --num_envs 4 --steps 1
```

60 任务目标规模验证：

```bash
/home/gpuserver/miniconda3/envs/isaac51/bin/python \
  scripts/smoke_multitask_env.py \
  --headless --device cuda:0 \
  --manifest /tmp/srsa_faca_m0_60_tasks.json \
  --require_task_vectors \
  --num_envs 60 --steps 1
```

smoke 会检查所有任务都被分配、mapping 正确、state/task shape 正确且有限，以及至少一步
Direct reward 有限。

## 5. 接入 FACA TD-MPC2 M0

在启动 FACA 之前导出以下变量：

```bash
export SRSA_MULTITASK_MANIFEST=/tmp/srsa_faca_m0_60_tasks.json
export SRSA_MULTITASK_REQUIRE_TASK_VECS=1
export SRSA_ENABLE_AXIAL_TASK_PARAM_SAMPLER=0
export SRSA_TASK_PARAM_GEOMETRY_SCALE=0
export SRSA_GRASP_CONSTRAINT_MODE=physical_grasp
```

然后基于 FACA 的 `configs/train/srsa_01125_faca_m0.yaml` 启动，并至少覆盖：

```text
num_envs=60
faca_variant=m0
task_conditioning=axial_params
srsa_use_runtime_task_vec=true
isaaclab_use_canonical_obs=true
isaaclab_canonical_append_task_params=false
```

FACA 当前 `Trainer._runtime_task_vec()` 已能逐 env 读取 SRSA 的
`current_task_vec[num_envs, 6]`，并将同一个 runtime task tensor 同时用于 agent action 和 replay
transition；因此 model/replay 的任务条件不再是单个 assembly 的广播值。

需要注意：FACA 当前的 `_tasks`、per-task evaluation 名称和 episode-length 表仍由 FACA config
维护。它们不影响 M0 读取逐 env task vector，但要得到严格正确的 60 任务分项日志，下一步应在
FACA adapter 中读取 `get_faca_m0_task_indices()` 和 `get_faca_m0_assembly_ids()`，替换旧的单任务
label bookkeeping。

## 6. 当前验证结果

2026-08-10 在 Isaac Sim 5.1 / RTX 4090 上完成：

- CPU contract tests：`16 passed`；
- 两任务：4 replicas，mapping `[0, 1, 0, 1]`，一步 reward 有限；
- 60 任务：60 replicas，mapping `0..59`，60 个不同 socket/plug USD 均成功实例化；
- M0 state shape：`(60, 14)`；
- M0 task shape：`(60, 6)`；
- 一步 dense reward mean：约 `3.1120`，无 NaN/Inf；
- 60 任务场景创建约 `14.46 s`，simulation start 约 `1.59 s`。

Isaac Lab 会对 instanced USD 的 collision-property override 打印 warning，也会提示
`replicate_physics=False` 时不能自动启用 clone collision filtering。当前 2 m env spacing 的 smoke
已正常运行；正式长训练仍应监控跨 env 接触和吞吐。缺少 `nvcc` 时 SoftDTW 会使用现有的 torch/CPU
fallback，这不会阻止启动，但可能成为 60 任务训练的性能瓶颈。
