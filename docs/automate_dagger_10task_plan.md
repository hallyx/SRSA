# AutoMate：既有 10 个装配任务的专家训练与 DAgger 多任务蒸馏计划

## Material Passport

- Origin Skill / Mode: academic-research-suite / experiment-agent / plan
- Origin Date: 2026-09-10
- Verification Status: UNVERIFIED（完整训练尚未实施；下述本地文件和代码行为已做静态核查）
- Version Label: automate_dagger_10task_plan_v1
- 工作目录：SRSA 仓库；本文件为实施计划，不代表已新增训练器或已获得实验结果。
- 目标：为固定的 10 个 assembly 获取合格 AutoMate 专家，经 BC、DAgger 和 PPO+SBC 微调，产出一个可独立执行全部 10 个任务的 generalist。
- 实验定位：在现有 Isaac Lab / SRSA 运行时上实现 AutoMate 方法；资产集合、软件版本及下述工程参数与原论文不同，不能称为逐项精确复现。

## 1. 冻结任务集合和已有证据

顺序来自 FACA 的 `AGENT.md`、`scripts/run_srsa_faca_m0_hetero10.sh` 和两份正式十任务 protocol，三处一致：

| task_index | assembly_id | 本地 plug/socket mesh | 本地拆解路径 JSON | SRSA checkpoints/ 中对应专家 |
|---:|---|---|---|---|
| 0 | 00062 | 有 | 有 | 无 |
| 1 | 00186 | 有 | 有 | 无 |
| 2 | 00256 | 有 | 有 | 无 |
| 3 | 00271 | 有 | 有 | 无 |
| 4 | 00726 | 有 | 有 | 无 |
| 5 | 01079 | 有 | 有 | 无 |
| 6 | 01029 | 有 | 有 | 无 |
| 7 | 01092 | 有 | 有 | 无 |
| 8 | 01102 | 有 | 有 | 无 |
| 9 | 01125 | 有 | 有 | 无 |

本地 mesh 路径为 `data/mesh/<assembly_id>/{plug,socket}.obj`；路径数据为 `data/disassembly_paths/asset_<assembly_id>_disassembly_traj.json`。存在文件只证明数据可找到，尚需验证与仿真 USD 的坐标、尺度和路径格式一致。

现有 `checkpoints/` 只有 `00141、00211、00426、00638、00783`。这些可作为另外的迁移初始化候选，不能直接命名成上述任务的合格教师。未发现这 10 个任务的现成 AutoMate 教师银行，不排除其它未检索存储中存在权重。

历史 `FACA/data/offline_manifest_01125_family.json` 记录的是 17D 观测、3D 动作、每任务 10 回合的数据，且来源是同一个历史策略；其中部分成功率为零。它不构成 10 个已验收专家，也不应直接作为本计划的 BC 标签库。这里的历史成功率不能用于判断任务本身不可学。

本次会话已实际通过 `00783、01125` 两任务环境一步 smoke；之前文档记录过 60 任务环境 smoke。两者都不证明本清单的专家训练、DAgger 或多任务 PPO 已跑通。

本次只复用 FACA 的任务集合与历史协议作为参照。AutoMate 训练实现和产物放在 SRSA 的独立命名空间。

## 2. 方法边界：哪些来自论文，哪些是本计划的选择

AutoMate 用任务专属策略作为教师，以插头/插座 PointNet 几何表示作为 generalist 条件，先 BC，再由 student 执行、对应教师在其访问状态上标注动作，最后 RL+SBC 微调。专家阶段使用带轨迹模仿奖励的 RL；generalist 微调采用 RL-only 基础奖励。论文 BC 每专家收集 5000 个成功回合，DAgger 文段描述采集 256 个成功或失败回合；后者未在该段明确给出本文需要的“每任务每轮”调度。以上依据 [AutoMate §V-D](https://arxiv.org/html/2407.08028v1#S5.SS4)。

下文的任务均衡、256 回合/任务/轮、轮数、验收阈值、32D 单物体编码、GPU 调度、学习率和预算均为本项目的起始方案，不冒充论文原始超参数。训练数值属于计划，不是结果。

## 3. 先冻结环境、观测和动作协议（阶段 A）

### 3.1 主实验协议

- 使用原生 assembly 几何、Franka、`physical_grasp`；不进行 reset-time 轴向任务采样和几何缩放。
- 专家使用 `Assembly-Direct-v0` 对 AutoMate 的封装；关闭 sparse、SIL、NEWT 观测、task_vec 拼接和所有力观测。
- 教师 actor 采用当前 AutoMate 的 24D 观测；动作保持 6D 位姿增量。critic 基础状态为 44D。实际启动时再次核对张量维度、字段顺序和配置哈希。
- student 使用同一 24D actor 观测加 64D 几何编码，计划输入 88D，输出同样的 6D 动作。PPO critic 使用 44D 状态加 64D 几何编码，计划输入 108D。
- 几何编码为 `[z_plug(32), z_socket(32)]`；`task_index` 用于教师路由、数据分桶和统计，不作为主实验的 one-hot 输入。
- 动作标签位于统一的归一化 action 接口，明确裁剪规则，位于环境 EMA、物理尺度变换和低层控制器之前。保存执行动作和教师标签两个字段，不能把实际 student 动作误当监督标签。
- 教师的观测归一化统计逐 checkpoint 恢复并冻结；student 使用自己的统一统计。禁止把某个教师的归一化统计套到其它教师或 student。
- student/teacher 共用一次观测采样；不要为两者分别调用带随机噪声的观测函数，以免标签对应不同输入。
- 从实际加载后的 cfg 冻结 reset 分布、噪声、控制频率、action scale、阻抗增益、终止规则及 episode 上限。不要搬用历史 FACA 的 74 步或其 3D 动作约束。

### 3.2 成功标准

每回合同时输出：AutoMate 官方 success 是否曾发生、结束时官方 success、SRSA terminal-process success、最终插入深度和位置/姿态误差。精确阈值和实现版本写入 protocol。

主 AutoMate 协议以冻结的官方 success 为主指标；terminal success 和过程指标作为必报诊断。如果与 FACA 做正式比较，必须在同一个冻结 evaluator、相同物理动作能力和 reset/noise 协议下重评；“只使用同一组任务”不构成公平比较。

若出现官方成功率很高但结束状态大量脱出或失败，在阶段 A 校验成功定义与几何目标；不能在看到最终结果后改阈值，也不能静默把过程成功率替换为官方成功率。

### 3.3 必做的环境检查

1. 十任务资产、抓取元数据、插入深度、SDF mesh 和拆解路径逐项一致；远程 USD 缓存和本地 OBJ 的来源可追溯。
2. 先每任务单独 reset/step，再 10 任务混合运行。比较相同物理状态下单任务/异构环境的观测、动作变换、成功判断、奖励分量。
3. 每任务至少 10 回合初步场景检查；选择失败、成功边界、夹爪滑落状态检查指标；几何条件交换只改编码，不能顺带改资产。
4. 性能 profile：10、80、160 个环境，记录初始化时间、env transitions/s、最大显存、DTW/SDF/推理耗时。160 不适合时降到 80，保持每任务同样的 replica 数。
5. GPU 映射以 `CUDA_VISIBLE_DEVICES` 决定物理卡，进程内部用逻辑 `cuda:0`，避免硬编码在不同机器上错卡。

## 4. 十个 specialist 的获取和验收（阶段 B）

### 4.1 训练方式

每任务一套独立 PPO 专家，使用 AutoMate dense reward + DTW 模仿奖励 + SBC。先训练 `01125` 和 `00062` 打通两条链路，然后按固定清单补齐其余 8 个；顺序是工程选择，不代表已经确认任务难度。

两个 GPU 各运行一个单任务训练进程，每进程 128 个环境。每个 run 使用独立目录、配置和日志。优先使用 SRSA 的运行时 assembly override，不使用会改写共享 `assembly_tasks_cfg.py` 的旧 AutoMate `run_w_id.py`，避免并发任务互相覆盖配置。

现有单任务启动入口可复用，下面是未来训练阶段的命令示例，本次不执行：

```bash
conda activate isaac51
export FULL_PATH_TO_ISAACLAB=/home/gpuserver/IsaacLab
CUDA_VISIBLE_DEVICES=1 \
SRSA_NEWT_OBS=0 \
SRSA_ENABLE_AXIAL_TASK_PARAM_SAMPLER=0 \
SRSA_TASK_PARAM_GEOMETRY_SCALE=0 \
SRSA_ENABLE_FLANGE_FORCE_SENSOR=0 \
SRSA_ENABLE_HELD_ASSET_NET_CONTACT_FORCE=0 \
SRSA_FLANGE_FORCE_SENSOR_OBS=0 \
SRSA_GRASP_CONSTRAINT_MODE=physical_grasp \
SRSA_SUCCESS_METRIC=official \
python source/SRSA/SRSA/tasks/direct/srsa/run_w_id.py \
  --assembly_id 01125 --train --num_envs 128 \
  --headless --device cuda:0 --seed 1 --max_iterations 1500 \
  --experiment_name automate10_01125_seed1_RUNSTAMP
```

正式 launcher 要另外清理继承的 `SRSA_MULTITASK_*`、force/task-family/noise 等覆盖变量，生成有效配置快照，并拒绝复用已有 run 目录。示例中的 RUNSTAMP 必须替换为唯一运行标识。这里不加 `--no_sbc`、`--sparse` 或 `--sil`。

### 4.2 预算与阶段出口

- 首个检查点：500 PPO iterations；基于 128 env × 32 horizon，为每专家约 204.8 万 transitions。
- 常规首轮预算：1500 iterations，即约 614.4 万 transitions/专家；十任务合计约 6144 万，不含 reset 动作和评估。
- 在 500、1000、1500 iterations 做冻结验证，成功率平台或下降时检查 reward/初始化/轨迹和 checkpoint，而不是只看训练 reward。
- 教师选型：每任务先 seed 1；不合格的任务再试 seed 2、3或延长到 3000 iterations。新增预算单独计账；不能把一次低分直接归因于任务不可学。
- 建议教师准入：独立验证 1000 回合，官方成功率目标 ≥90%；80%–90% 标为弱教师并继续修复；<80% 不进入正式十任务蒸馏。若某任务最终只有弱教师，十任务集合仍保留，结果明确标注，不能悄悄删任务提高均值。
- 同时报告 terminal/process 指标。用相同判据计算 teacher 和 student 差距；教师资格检查与最终 student 测试使用不同的固定初始状态集。
- 验证 deterministic mean-action 和历史 stochastic player 两种模式，选择并冻结标签策略。当前 PPO YAML 的 player 默认 `deterministic: False`，不能假定加载权重就会输出 mean action。

交付 `teachers.json`：assembly_id、checkpoint 路径/SHA256、训练 seed、网络结构、观测顺序及 normalizer、动作协议、RNN 类型/层数、环境协议哈希、逐项验证指标。教师在 BC/DAgger 全阶段冻结。

## 5. 几何编码和 BC 数据（阶段 C）

### 5.1 几何表示

复用 `embedding/geometry` 的 PointNet autoencoder 结构，先核对数据加载、物体次序和训练脚本。当前实现每个物体为 32D 表征，可形成 64D 配对编码；现有 dataset 以 socket→plug 顺序读取，不能直接假设输出就是本计划的 plug→socket。

以 20 个目标 mesh 为最低数据集，优先使用本地已有的完整 mesh 集训练无监督编码器；使用哪些 mesh 在运行前冻结。即使只用未见任务的无标签 mesh，也必须在泛化报告中披露。保留米制尺寸或统一全局缩放，不能逐物体归一化后无意丢掉绝对尺寸。

起始设置：每物体采样 2000 点，保存点云 seed、坐标变换、单位、mesh hash 和编码器 hash。输出 `geometry_embeddings.pt` 及显式命名字段，冻结编码器后进入 BC。现有 Chamfer 实现显式构造点对差值，先做显存 profile；若分块实现，验证损失及梯度等价。

### 5.2 专家示范数据

- 工程 pilot：每任务 500 个成功回合，共 5000 个。
- 完整计划：每任务 5000 个成功回合，共 50000 个；专家实际尝试次数和失败率也记录。
- 每回合存原始 actor 观测、规范化动作标签、实际执行动作、episode/task ID、终止类型、成功各指标、teacher/env/geometry hash。
- 几何编码按 assembly_id 引用，不必每一步重复存 64D；RNN 状态用于诊断，不作为离线 student 固定标签。
- 训练/验证以完整 episode 划分，建议 90%/10%。同一轨迹的相邻帧不能分到两侧；最终仿真测试使用另一个独立初始状态集。
- 数据按 task 分桶，采样时先等概率选任务，再选 episode/序列；避免某任务多采了数据就主导损失。

初始采集用 deterministic 专家动作，标签与执行动作对齐；若改为随机执行、均值监督，作为明确的变体登记。

## 6. 共享 generalist 的 BC 初始化（阶段 D）

- 结构起点：88D actor 输入，LSTM hidden 256，MLP `[512,256,128,64]`，6D action mean。沿用 RL-Games 可兼容的 recurrent actor 定义；确切 LSTM 层数、前后次序和激活函数写进配置，不凭权重 shape 猜测。
- 优化起点：Adam，lr=1e-4，gradient clipping=1.0；按 10 个任务均衡的 masked action MSE 优化。平移、旋转在其各自归一化动作维度上监督，同时分别记录误差和裁剪饱和率。
- 首选完整 episode 的 padding+mask 训练，每 batch 100 条序列、每任务 10 条；显存不足则降到每任务 2 条并累计梯度。按任务分别做有效步归一化。
- 若改用截断序列，必须有 burn-in 并从有效历史重算 hidden state；不能随机打散单帧训练 LSTM，也不能用旧 student 的 hidden state 当作永久真值。
- 每 5 epochs 做固定仿真验证；pilot 最多 100 epochs，无持续改善后结束。完整训练上限 1000 epochs，使用验证 checkpoint 选择，不能用训练 MSE 选择最终模型。
- BC 出口是一个在两任务乃至十任务上可以完整 rollout 的 student；不要求这一阶段已经达到最终 80% 成功率。

保存网络、student normalizer、optimizer、RNG、geometry/env hash 和数据版本，为 DAgger 与后续 PPO 保持相同 actor 实现。

## 7. 多任务 DAgger 主循环（阶段 E）

### 7.1 一轮如何运行

```text
冻结本轮 student θ_r 和 10 个 teacher
    ↓
十任务环境：env_i 的任务固定为 i % 10
    ↓
同一观测 → student 给执行动作
         → 按 assembly_id 路由的 teacher 给监督标签
    ↓
只执行 student 动作，记录成功和失败完整轨迹
    ↓
十任务均达到本轮采集配额 → 聚合并校验数据
    ↓
按任务均衡监督更新 student → 独立仿真验证 → θ_(r+1)
```

默认纯 student rollout；不混入专家接管动作，不根据“这一回合最后失败”删除 DAgger 标签。如果后续试 teacher/student 混合执行，单独命名为变体并记录 mixing schedule。

每个教师在自己负责的 env 槽位持有独立 LSTM 状态。教师虽然不执行动作，仍须每步沿 student 实际访问的观测历史推进。episode reset 时，只清零对应 env 的 teacher 和 student 状态。教师映射使用 `get_faca_m0_task_indices()` / `get_faca_m0_assembly_ids()` 暴露的 assembly 映射，不用 `current_task_id` 的轴向类型字段替代装配编号。

边界要求：在 `env.step(student_action)` 之前生成本步教师标签；auto-reset 后的新观测不能配到前一个 episode 的动作。训练和采集切换时采用 episode 边界或重算 student 隐状态，不能携带已更新权重之前的不一致 hidden state。

### 7.2 配额与更新

- 先两任务工程 pilot：`00062、01125`，每任务 32 回合，2 轮。用于检查路由/数据/梯度，不作为正式实验成绩。
- 十任务 pilot：每任务每轮 64 回合，3 轮；使用全部成功和失败回合。
- 完整初始计划：每任务每轮 256 回合，5 轮，即新增 12800 回合；是否延长到 10 轮依据预先冻结的验证平台规则。
- 始终保留 BC 库；建议 minibatch 数据来源为 BC 25%、历史 DAgger 25%、最新一轮 DAgger 50%，每个来源内部任务均衡。首轮无历史 DAgger 时将该份额并入最新数据。该比例是工程选择。
- lr 初始 1e-4；每轮最多 10000 次 optimizer updates，按验证 action error 与仿真成功率决定 checkpoint；同时记录每条样本被重复使用次数。
- 连续 3 轮宏平均验证成功率改善不足 1 个百分点且最弱任务没有改善，或达到轮数上限，结束 DAgger 并进入 RL 微调。这是平台规则，不是统计显著性结论。
- 任一任务较上轮下降超过 10 个百分点，先检查数据路由/normalizer/动作饱和/RNN reset，再增加预算。回滚以验证 checkpoint 为依据，不能删除失败任务的数据。

### 7.3 本阶段的验收

数据正确性必须先通过：十个任务各占约 10%、全部标签能追溯至对应 teacher、teacher 参数不变、失败回合存在、student 实际动作确实驱动环境、全链路可断点继续。

性能验收报告 BC→DAgger 的逐任务成功率变化及对 teacher 的差距。DAgger 不保证单独达到高成功率；教师在 student 访问的离轨状态上也可能给不出有效恢复动作。因此“监督 loss 降低但成功率不升”需要检查闭环，而不是无限追加 BC epochs。

## 8. PPO + 逐任务 SBC 微调（阶段 F）

将 DAgger actor、student normalizer 和 geometry encoder 原样载入 PPO。新建共享的、以几何为条件的 critic；不给新 student 套用任一 specialist 的 value normalizer。对 action mean、输入归一化和初始 hidden state 做迁移前后数值一致性验证；新增策略探索方差，记录确定值。

可先只做短暂 critic warm-up，再解冻 actor 进行低学习率 PPO。主方案 lr 起点 1e-5，确认 KL 和成功率稳定后才考虑升至 3e-5。微调阶段采用基础 RL reward，关闭 DTW imitation 的贡献；能跳过计算时不继续做无用 DTW。这个阶段性切换必须进入配置和运行快照。

**现有代码的两个明确缺口：**

1. `assembly_runtime_env.py::_get_rewards()` 在多任务模式仍以全局 `ep_succeeded` 均值推进所有环境的 SBC；需按 assembly 聚合刚结束回合的成功率，维护 10 份课程状态。简单任务不能替困难任务升级，未结束回合不能重复计数。
2. `automate_algo_utils.get_curriculum_reward_scale()` 返回均值；多任务场景需审计并按任务应用奖励缩放，避免不同任务课程难度相互影响。

课程沿用 AutoMate 的高度采样语义和边界，按任务推进；结合当前坐标和 `curr_max_disp` 的定义验证“更难”的实际物理含义。SBC 参数、回合计数器和统计窗口必须能断点恢复。

初始单卡配置：160 env（每任务16），horizon=32，PPO rollout batch=5120，minibatch=1280，recurrent sequence length=4，保持整除。现有默认 minibatch=4096，不能在更改环境数后原样套用。若调至80或320 env，重新核对所有 actor/critic batch 与序列约束。

微调预算：500 iterations 的 pilot 为 256 万 transitions；完整初始预算 1500 iterations 为 768 万 transitions，均匀分配时每任务约76.8万。checkpoint/验证每100 iterations一次；需要续训时以原协议继续并单独报告追加预算。

验证关闭课程更新，在统一冻结的完整目标初始分布上进行；不能使用各任务当前的简单课程分布报告最终成功率。

## 9. 软件模块和交付顺序

下表是待实现文件，不是当前已有命令。公共策略结构只保留一份，供 BC、DAgger、PPO 共用。

| 优先级 | 拟新增或修改位置 | 职责与通过标准 |
|---|---|---|
| P0 | `configs/automate/ten_task_protocol.yaml` | 固定任务顺序、资产/观测/动作/成功/预算；未知字段或不兼容 hash fail fast |
| P0 | `scripts/automate/audit_assets.py` | 十任务资产及路径审计，导出状态和错误清单 |
| P0 | `scripts/automate/train_specialists.py` | 使用现有单任务入口排队训练；每 GPU 一个进程；唯一 run 目录 |
| P0 | `scripts/automate/evaluate.py` | 逐任务完整回合评估、独立 reset 集、全部成功定义、教师和 student 通用 |
| P1 | `source/SRSA/SRSA/automate_distill/teacher_bank.py` | 专家加载、obs normalizer、动作接口、每 env RNN 状态与精确任务路由 |
| P1 | `source/SRSA/SRSA/automate_distill/geometry.py` | 明确 plug/socket 次序、缓存与尺度、encoder hash |
| P1 | `source/SRSA/SRSA/automate_distill/policy.py` | BC/DAgger/PPO 共享 recurrent actor 与 geometry-conditioned critic |
| P1 | `source/SRSA/SRSA/automate_distill/dataset.py` | 按 episode 存储、task-balanced sampler、mask/burn-in、数据版本 |
| P1 | `scripts/automate/collect_bc.py`、`train_bc.py` | 成功专家 rollout 与序列监督训练 |
| P2 | `scripts/automate/train_dagger.py` | 冻结轮次采集、teacher 标注、聚合更新、恢复与验证 |
| P2 | `assembly_runtime_env.py` 与独立 AutoMate 配置 | 加入几何观测接口、逐任务日志及阶段奖励开关，保持旧默认行为 |
| P3 | `scripts/automate/finetune_ppo.py`、SBC 实现 | 共享 actor 迁移、逐任务课程、基础奖励微调 |
| P3 | `scripts/automate/report.py` | 成功率表、种子统计、预算、失败分类、复现信息 |

必要测试：教师单独 player 与 bank 输出等价；混合任务路由不会串号；RNN done mask 只影响目标槽位；归一化/动作尺度一致；teacher 冻结；DAgger 时序标签正确；跨 episode padding 不参与损失；SBC 单任务升级不改其它任务；DAgger→PPO actor 输出一致；恢复后的下一批采样及更新与未中断参考一致或误差可解释。

产物目录建议 `runs/automate10/<run_id>/`，包含 `protocol/`、`teachers/`、`geometry/`、`bc_data/`、`dagger_data/round_*/`、`models/`、`eval/`、`logs/`。所有大数据/权重使用已有 ignored runs 目录；轻量协议与代码进入版本控制。

## 10. GPU、数据量和进度安排

| 阶段 | 默认执行方式 | 可扩展方式 |
|---|---|---|
| 专家获取 | 两 GPU 各一个128-env单任务进程，五批完成十任务首轮 | 不合格任务追加预算；不在一张卡堆多个仿真进程 |
| 几何/BC | 单 GPU 离线学习，另一卡采集或评估 | 先修正高显存 Chamfer，再考虑提 batch |
| DAgger | 单 GPU 160-env异构采集，轮末暂停采集后更新 student | 两采集进程各暴露全部任务，固定同一 student hash；全轮结束后中央更新，不先引入 DDP |
| PPO微调 | 单 GPU 160 env，另一卡冻结评估 | 验证容量/吞吐后扩至320 env；DDP作为独立后续优化 |

单个 student seed 的完整数据计划为：BC=50000成功回合，DAgger=10×256×5=12800回合，合计62800回合。三个 student seeds 共用BC库、各自采集DAgger，实际存储共88400回合。transition 数按各数据源真实序列长度求和，环境 episode 上限以运行时值为准。

十专家各一个训练seed，加一个 student seed 的1500-iteration微调，合计约6912万训练 transitions（6144万+768万）。三个 student seeds 共用同一教师银行时，这部分合计约8448万（6144万+3×768万）；另加BC/DAgger采集、验证、reset成本和可能追加的专家训练。数据压缩、RNN训练和物理仿真的成本分开报告。

不预先承诺若干小时完成：先对每任务测量稳定的 transitions/s。专家阶段时间按两张卡的实际任务队列最大累积耗时计算，再加采集、离线更新和评估时间；报告瓶颈任务及DTW/SDF占比。当前 smoke 中出现无 nvcc 的提示，须记录实际启用的DTW实现并做吞吐测试。

建议执行顺序：

1. 协议与资产审计 → 两任务专家入口及评估闭环。
2. 十任务专家训练、筛选；可同时开发几何编码和教师银行。
3. 两任务 BC/DAgger 工程 pilot → 十任务 pilot。
4. 完整 BC 数据 → 三个 student seeds 的 BC/DAgger。
5. 逐任务 SBC 审计 → 三个 student seeds 的 PPO 微调。
6. 冻结测试集逐任务评估、条件敏感性和失败报告。

## 11. 评估、对照和最终验收

主对照必须保留四类 checkpoint：十专家 bank、BC student、BC+DAgger student、BC+DAgger+PPO+SBC student。条件允许再增加“不使用SBC的PPO微调”，并单独计预算。

- 初始工程 seed=1；正式 student seeds=1、2、3，共用冻结教师银行和BC库，每个 seed 独立进行 DAgger/RL。由此得到的是“给定该教师银行”的稳定性，不宣称覆盖专家训练随机性。
- 验证集用于选择 checkpoint 与判断继续训练；最终测试集只在方案/模型冻结后使用。GPU/vectorization 变化可能改变随机数消耗，应保存实际 reset 参数，不只保存整数 seed。
- 教师准入验证与最终测试建议每任务1000回合。最终每个 student seed 得到10行结果；报告每任务success、10任务宏平均、最差任务、终态成功和teacher差距。
- 单任务成功率报告二项比例区间（如Wilson）；跨训练seed报告均值和标准差。十任务是固定实验集合，不能把大量并行episode当作大量独立训练重复。
- 建议最终目标：宏平均官方成功率≥80%、最弱任务≥60%、宏平均较对应teacher下降≤10个百分点；terminal/process指标必须完整披露。这些是工程目标，不保证可达到。
- 以冻结模型做正确/错配/零几何编码的配对测试，记录动作差异及逐任务success变化；某些几何近似任务不敏感不自动判失败，但整个模型完全无差异需要排查条件是否接入。
- 故障分类至少包括抓取滑落、未对齐、插入不足、超时、仿真异常；force诊断仅在信号语义已经验证时额外加入，不改主策略输入。
- 十任务训练内的高分仅说明多任务能力；未见几何的泛化要另设任务和协议，不能从训练任务成绩推断。
- 若与FACA比较，另列信息输入、6D/3D动作、episode时长、reset/noise、成功定义、教师训练与标注总成本；既报告student在线成本，也报告完整专家蒸馏总成本。

## 12. 启动正式长训练前的退出条件

阶段A通过且十任务专家全部验收后，才进入正式十任务BC/DAgger；两任务开发pilot可提前进行。DAgger正确性测试通过后才扩大采集量。逐任务SBC及actor迁移测试通过后才进入完整PPO微调。

当前最先实施的工作是：冻结AutoMate十任务协议，完成专家评估器和教师银行原型，然后获得`00062、01125`两个合格教师。首个可交付闭环应是“两个教师 → BC → student执行并由教师标注 → DAgger更新 → 两任务独立评估”。

### 本地核查入口

- `source/SRSA/SRSA/tasks/direct/srsa/run_w_id.py`：任务选择、运行时环境变量和训练入口。
- `source/SRSA/SRSA/tasks/direct/srsa/multitask_assembly.py`：静态env-to-task映射。
- `source/SRSA/SRSA/tasks/direct/srsa/assembly_runtime_env.py`：任务资产、reward、success、SBC、观测扩展。
- `source/SRSA/SRSA/tasks/direct/srsa/agents/rl_games_ppo_cfg.yaml`：当前PPO/RNN/player配置。
- `scripts/rl_games/play.py`：既有RL-Games推理和RNN reset路径。
- `source/SRSA/SRSA/embedding/geometry/{model,dataset,trainer}.py`：32D编码、socket/plug读取次序、Chamfer实现。
- `/home/gpuserver/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/automate/assembly_env_cfg.py`：24D actor / 44D critic / 6D action基础协议。
- `/media/gpuserver/data2t/hx/github/FACA/AGENT.md` 与 `scripts/run_srsa_faca_m0_hetero10.sh`：历史10任务清单。
- [AutoMate原论文](https://arxiv.org/html/2407.08028v1#S5.SS4)、[NVIDIA方法介绍](https://developer.nvidia.com/blog/training-sim-to-real-transferable-robotic-assembly-skills-over-diverse-geometries/)：方法来源。
