# FACA 和 SRSA 历史训练及 AutoMate 十任务专家结果汇总

整理日期：2026-10-10。范围包括 2026 年 5 月至 9 月的前期 FACA/SRSA 训练，以及 9 月 25 日启动、9 月 27 日配对评估的 AutoMate 十任务 scratch specialist 队列。文件名保留以兼容已有引用；本报告已包含用户补充要求的十任务专家结果。BC/DAgger、之后的 residual adapter、jam 校准和辅助头新实验不计入本次训练汇总。

前期结果说明：单任务与近邻两任务可以学会，但扩大任务集合后仍有弱任务、遗忘和 checkpoint 退化；01079 的多尺寸物理覆盖有收益，真实任务向量相对固定向量未显示稳定优势。十任务尺寸 M0–M3 在旧 relaxed 口径下 M0 宏平均最高，但深度上限会改变臂间排序，不能据此断言力输入或辅助监督无效。

AutoMate 十任务专家已完成 10/10，但官方锁存成功率 76.85% 与新版 relaxed 终态稳定成功率 15.50% 差距明显；新版 relaxed 有七任务为 0/200。目录名称中的 `srsa` 不代表这批训练执行了 SRSA 检索迁移算法。

本次读取了训练入口、模型、replay 和成功判据代码，复算尺寸 A/B/C 汇总、十任务尺寸原始逐回合记录，以及 AutoMate 十任务配对 HDF5。早期部分记录仅剩交接文档或报告索引，表中明确列为历史转录；这些结果没有获得与当前主表相同的原始文件复核。

## 结果口径和代码对应

FACA 是在 SRSA/IsaacLab 环境上使用 TD-MPC2 世界模型和 MPPI 的训练路线，主动作是 3D 平移。后来 AutoMate 专家路线使用 PPO 与 6D 位姿动作。两者不能仅因任务编号相同而直接比较成功率。

| 概念 | 代码实际行为 | 结果解读 |
|---|---|---|
| 任务参数 | `task_vec_6` 独立经 AxialTaskEncoder 形成条件；WorldModel 将条件送入模型路径 | 参数被保存和接入，不代表模型从正确参数获得性能收益 |
| family replay | 当前任务负责新增数据；采样时按配置混合当前、anchor 和历史 replay | 目录含 `online_family_replay` 不代表实际启用了历史混合，要看 resolved 配置 |
| V3 力观测 | 14D TCP 状态加 6D 去 tare 的 TCP 腕部 wrench；力/力矩分别除以 50 N/5 Nm | 与早期 17D held-asset net-contact proxy 不同，不能混载或混算 |
| 旧 relaxed | `0.85 <= depth_fraction < 1`，横向门槛、配置要求的 no-jam、终态连续 3 个控制步 | 过插会失败；是下述主要历史表的口径 |
| 完整新版 relaxed | 显式 `relaxed_depth_unbounded=true`，深度严格 `>0.85`，保留横向、no-jam 和终态连续 3 步 | 默认开关仍为 false；必须用完整轨迹或闭环评估，不能只改旧表头 |
| 末态深度单条件 | 只要求末步 `depth_fraction >0.85` | 9 月 24 日重算采用这一口径，缺少横向和稳定条件，另表报告 |
| official | 官方成功可曾经命中并锁存，和末态 stable 分开 | official=100% 不能代替终态插入成功 |

深度由 plug/socket 刚体根位姿计算：相对位置旋转到 socket 坐标系后，`depth=max(0, target_depth-relative_z)`；不是 TCP 的累计位移。当前 relaxed 不把姿态、yaw、keypoint 加入布尔 gate，虽然这些诊断量仍被计算。代码中的 jam 布尔量不能自动解释为独立验证的机械卡死标签。

代码入口：[成功判据](/media/gpuserver/data2t/hx/github/FACA/tdmpc2/envs/isaaclab.py)、[配置默认值](/media/gpuserver/data2t/hx/github/FACA/tdmpc2/config.py)、[WorldModel](/media/gpuserver/data2t/hx/github/FACA/tdmpc2/common/world_model.py)、[family replay](/media/gpuserver/data2t/hx/github/FACA/tdmpc2/common/online_family_replay.py)。链接指向本次读取的 FACA 实际路径。

## 训练路线总表

| 阶段 | 训练路线 | 代表结果 | 证据和结论 |
|---|---|---|---|
| 5 月至 6 月 | 同一 01125 起点独立 direct fine-tune | 00256 保存 best 95.56%，00186 41.33%，00062/00271 0% | 历史转录，来自训练期保存指标，不是当前统一冻结测试 |
| 5 月至 6 月 | 串行 latest 权重交接 | 新任务部分获取，旧任务 retention 大幅下降 | 历史 official/strict/process 混合表，保留原列名 |
| 6 月 | 01125→00256 获取后 family retention | 01125 75%，00256 90%，均值 82.5% | 历史近邻两任务结果；不能推成十任务泛化 |
| 6 月至 7 月 | FiLM 与任务编码修复 | raw-task FiLM 对任务输入更敏感，却造成两任务 relaxed 都为 0%；后续编码修复改善表示 | 表示变化与闭环增益需要分别报告 |
| 7 月 | 第三任务 00186 与 policy-only elite 适配 | source 40%，elite-only 20%，anchor λ=3 为 25% | 小样本冻结闭环否定了该适配方案的收益 |
| 7 月 | Phase 4 初始化及连续参数预训练 | 索引分类分别为 MIXED_OR_UNRESOLVED 和 PARAMETRIC_PRETRAINING_NO_TRANSFER | 主报告缺失，不补造具体分数 |
| 7 月 | Phase 5 三任务 simultaneous 与 matched conditioning | correct reference 两训练 seed；family-mean 和 shuffled 各两 seed 训练完成 | 最新 interim 文件仍为冻结评估待完成；不能宣布训练期参数收益 |
| 8 月至 9 月 | V3 单任务获取和力输入对照 | 01125 延长训练后 89.5%；01079 单卡有力/无力 77.5%/57.5% | 单 seed、旧 relaxed；力输入收益随任务变化 |
| 9 月 10 日 | 力惩罚前的获取准入 | 原 reward 01125 为 78.91%，未过每 seed 80% 门槛 | 没有进入正式 force-refinement，不能称为力奖励有效 |
| 9 月 14 日至 15 日 | V3 01125→01079 current-only 获取 | 新任务 86.5%，旧任务 27.5%，均值 57% | 获取成功且遗忘明显；历史 replay 混合实际关闭 |
| 9 月 12 日起 | 01079 尺寸 A/B/C 与独立 OOD | OOD A 96.22%，B 97.37%，A−B −1.15 pp | 三个共同专家起点的续训 seed；多尺寸覆盖有收益，正确向量未胜出 |
| 9 月 15 日至 17 日 | 初版异构 joint | 曾报告两任务 Selected 90%/99% | 资产叠加和 PhysX 路由错误，不能作为有效多任务结果 |
| 9 月 17 日至 18 日 | 修复后 nominal 十任务 | 带力 Selected/Final 65.85%/64.00%；M0 60.70%/62.95% | 正确路由的单 seed pilot，原任务集仍有弱任务 |
| 9 月 19 日起 | F0–F3 周期内力信息 | F3 两任务 Final 98.5%/97.0%；F1 22.5%/31.0% | 独立单任务 campaign，不能与 M0–M3 横比 |
| 9 月 20 日起 | tasksize10-v2.1 共享 M0–M3 | held-out Selected：84.45/63.70/80.80/57.45% | 四臂均 MT_True，一个 scratch 训练 seed；完整结果见下表 |
| 9 月 25 日至 27 日 | AutoMate 十任务 scratch PPO 专家 | 官方 76.85%，旧 relaxed 13.80%，新版 relaxed 15.50% | 十个独立策略，每任务一个训练 seed；不是 SRSA 检索迁移，也不是共享 generalist |

历史证据入口：[早期状态](/media/gpuserver/data2t/hx/github/FACA/docs/multitask_status_brief.md)、[Phase 3 交接](/media/gpuserver/data2t/hx/github/FACA/docs/srsa_phase3_handoff_20260715.md)、[训练组织对比](/media/gpuserver/data2t/hx/github/FACA/docs/multitask_training_strategy_comparison_20260922.md)。早期状态文件多次追加，不能仅按文件标题日期判断所有段落的实验日期。

## SRSA 目录中的 AutoMate 十任务专家

这批目录实际是 AutoMate 从零训练的十个 PPO 专家，无 SRSA 检索迁移或 SIL。10/10 完成，每任务 1200 万 transitions，总计 1.2 亿；官方/旧 relaxed/完整新版 relaxed 宏平均分别为 76.85%/13.80%/15.50%。

逐任务成绩、两个评估 seed 的成功数、20 个 HDF5 路径和完整新版 relaxed 的计算代码已拆到 [AutoMate 十任务专家独立报告](automate10_specialist_training_results_20261010.md)。该报告说明成功值如何在闭环逐步计算、终态保存，再按任务聚合；这里保留路线索引。

## 早期获取和任务参数路线

direct fine-tune 每个目标独立从同一来源开始；串行 continual 则把 latest 权重继续传到下一任务，不保留旧在线 replay。早期保存 best 的 95.56% 等分数与后来冻结 test 不同。

| direct 目标 | 保存 best episode_success | 所在步数 |
|---|---:|---:|
| 00186 | 41.33% | 600,000 |
| 00256 | 95.56% | 600,000 |
| 00062 | 0% | 150,000 |
| 00271 | 0% | 150,000 |

串行路线在完成 00004 后，01125 的 official terminal 仅 0.5%、strict/process 均 0%；完成 00062 后，01125 official terminal 为 1%、strict/process 仍为 0%。这些是原来的诊断口径，不能换名为 relaxed。离线 family 初始数据只有十任务各十回合，总计 7,400 transitions，且含大量终态失败；没有证据支持把它当作十专家示范库。

01125→00256 历史获取在 299,520 steps 达到监控成功率 90.23%，之后 retention 为 75%/90%。同一 00256 策略正常/强制 01125 向量都为 90%，零向量为 75%；真实向量 swap 的离线 action/Q/reward/dynamics 差异接近数值噪声。该阶段的 replay 标签正确和闭环任务区分能力是两个问题。

6 月 learned-context FiLM 的上下文仍塌缩，训练后闭环退化；raw-task FiLM 增大 swap 差异，却使 final 00256 为 0%，family 的两任务 relaxed 都为 0%。7 月编码修复使 01125/00256 context L2 从约 1.36e−8 增至 0.01298、真实任务 reconstruction R² 达 0.983；这是表示修复，不是任意任务上的性能保证。

Phase 3 的 00186 初次 acquisition 为 15%；后续 rescue checkpoint 上的 policy-only 100 次更新比较，每任务仅 20 回合：

| 模型 | 00186 relaxed | 01125 relaxed/process | 00256 relaxed/process |
|---|---:|---:|---:|
| source | 40% | 90% / 30% | 100% / 10% |
| elite-only | 20% | 90% / 10% | 100% / 10% |
| elite + anchor λ=3 | 25% | 90% / 20% | 95% / 5% |
| elite + anchor λ=10 | 10% | 80% / 30% | 100% / 5% |

离线 proposal-regret 改善没有转化为闭环提高。Phase 4 两项实验目前只有索引，所指主报告在当前 FACA 路径不存在，因此只保留其分类。Phase 5 以较新的 [interim status](/media/gpuserver/data2t/hx/github/FACA/reports/phase5_five_task_generalist/phase5_9_2_interim_status.md) 为准：family-mean 和 episode-shuffle 各两个训练 seed 均完成，每 run 299,700 transitions/22,477 updates，每任务 99,900 transitions；旧 README 的“shuffle 未正式执行”已过时。推理期任务向量贡献已有报告，但训练期 True 对 Mean/Shuffle 的最终冻结收益仍不能从采集曲线判定。

## V3 单任务腕力和训练退化

修正后的 wrench 使用 `incoming_joint_frame_v2`；旧参考系曾产生约 45° 纯力方向误差。以下主要对照均为单帧输入，每臂一个训练 seed，64 env 到 299,520 steps，再单卡 250 env 到 699,020 steps，测试 5101/5102 各 100 回合。

| 任务 | 20D 有力 Final | 14D 无力 Final | 差值 | 腕部力峰值 P95 有力/无力 |
|---|---:|---:|---:|---:|
| 01079 | 77.5% | 57.5% | +20 pp | 99.63 / 112.56 N |
| 00256 | 25.0% | 0.0% | +25 pp | 76.95 / 187.36 N |
| 01125 | 89.5% | 98.5% | −9 pp | 15.97 / 38.38 N |

01079 早先双卡 500-env 有力模型为 88.5%，不能替换单卡 77.5% 的主表。两个测试 seed 不增加独立训练重复数；比较 JSON 自身也限定为描述性，不能把观察到的差值全部归因于力输入。

00256 第一阶段 best 冻结有力/无力分别 80%/85%，到 299,520 steps 为 79%/35.5%，666,020 steps 为 0.5%/66.5%，最终 25%/0%。无力路径在切换并行环境数之前已经退化，所以环境数变化不能解释全部问题。冻结有力起点 wrench 置零后由 79% 降至 18.5%，只说明该 checkpoint 依赖该输入。

terminal-value 诊断对无力 good/bad checkpoint 的原 MPPI 为 85%/35.5%，λ=0.25 为 64.5%/53%，λ=0 为 5%/10.5%，actor mean 为 47.5%/49.5%。调低权重改善 bad 却损害 good；不能宣布删除 terminal value 已解决退化。

V3 01125 初始 scratch 的监控峰值 61.33%、Final 43.36% 未过门槛，后来延长预算后的冻结模型为 89.5%。另一个 9 月 10 日获取 pilot 到 142,080 steps，标准冻结两个 seed 为 79.69%/78.12%，均未达每 seed 80%；该目录虽然名含 force_reward，正式 force-refinement 并未准入。

current-only 01125→01079 从 699,020-step 模型出发，新增 129,500 steps，在总 828,520 steps 达目标 88%/85%。回测 01125 为 24%/31%，平均 27.5%，较来源 89.5% 下降 62 pp。这是一次新任务获取伴随遗忘，不能混作 00256 单任务内部退化。

来源：[单任务主表与 checkpoint 轨迹](/media/gpuserver/data2t/hx/github/FACA/docs/force_mainline_status_data_plan_20260914.md)、[阶段获取结果](/media/gpuserver/data2t/hx/github/FACA/docs/wrist_mainline_and_terminal_results_20260915.md)、[获取准入原始汇总](/media/gpuserver/data2t/hx/github/FACA/logs/wrist_wrench_v3_force_reward/20260910_acquisition1/summary.md)。

## 01079 尺寸参数和间隙对照

A 使用十种物理尺寸及真实任务向量；B 使用相同物理尺寸但固定基准向量；C 只训练基准尺寸及基准向量。十种条件为五个 plug XY 倍率 0.90/0.95/1.00/1.05/1.10 与两种间隙倍率 0.75/1.25，目标深度固定 35.282 mm。三臂从同一修正 V3 专家独立续训，共三个训练 seed，不是三个独立 scratch 起点；日程由 A 的门槛驱动，匹配环境步不等于完全相同更新次数。

原冻结测试 25 条件，每条件每 seed 256 回合。以下为条件宏平均，再跨训练 seed 取均值，括号为样本标准差 pp：

| 测试组 | A 真实参数 | B 固定参数 | C 仅基准训练 | B−A |
|---|---:|---:|---:|---:|
| 已见十条件 | 96.29 (1.14) | 97.47 (0.53) | 92.92 (3.93) | +1.18 pp |
| 尺寸插值八条件 | 96.37 (0.92) | 97.85 (0.56) | 93.31 (3.85) | +1.48 pp |
| 未见间隙两条件 | 96.61 (0.30) | 98.37 (0.60) | 93.68 (3.68) | +1.76 pp |
| 尺寸外推四条件 | 95.28 (1.23) | 95.21 (1.29) | 91.18 (4.54) | −0.07 pp |

B−C 约 +4.56/+4.54/+4.69/+4.04 pp，支持这个家族内多尺寸物理训练的收益。独立 OOD 使用 reset seed 97031、八插值加四外推条件，每臂每训练 seed 3,072 回合；总体 A=96.22%、B=97.37%、A−B=−1.15 pp，三个训练 seed 差值分别 −2.21/−0.39/−0.85 pp。正确向量未胜过固定向量，不等于参数输入完全不起作用。

固定输入的仅紧间隙 T 与紧宽混合 M，每臂每 seed 998,400 steps：

| 训练 seed | T 宏平均 | M 宏平均 | T−M |
|---|---:|---:|---:|
| 1 | 92.28% | 92.34% | −0.06 pp |
| 2 | 93.13% | 86.64% | +6.48 pp |
| 3 | 95.14% | 95.30% | −0.16 pp |

方向不一致，不能宣布仅紧间隙与混合训练等效。三组原始 JSON 已复算，来源路径与 SHA256 在 [verified_summary.json](pre_automate_results_20261010/verified_summary.json)。更完整的协议见 [已有尺寸报告](/media/gpuserver/data2t/hx/github/FACA/docs/size_parameter_training_results_20260926.md)。

## 异构 joint 和 nominal 十任务

初版异构场景存在首任务资产继承叠加、PhysX view 绑定首任务及 basename mesh 路由错误。旧两任务 Selected=90%/99%、Final=99.5%/60.5% 仅作故障历史，不能证明两个真实物理任务的联合学习。完整运行时修复后，同一个冻结旧模型对 01125/01079 的独立复核为 91.5%/99.5%；这是环境修复证据，不是修复后重新 joint 训练的结果。

修复后 nominal 十任务均为一个训练 seed、1,398,600 transitions/103,230 updates；每任务 139,860 transitions，测试每任务 200 回合。

| 任务 | 有力 Selected | 有力 Final | M0 Selected | M0 Final |
|---|---:|---:|---:|---:|
| 00062 | 95.0% | 71.5% | 71.5% | 96.0% |
| 00186 | 9.5% | 10.0% | 9.5% | 12.5% |
| 00256 | 90.0% | 93.5% | 83.5% | 92.0% |
| 00271 | 0.0% | 0.0% | 0.0% | 0.0% |
| 00726 | 77.5% | 98.5% | 83.5% | 92.0% |
| 01079 | 89.5% | 80.0% | 82.5% | 55.5% |
| 01029 | 97.5% | 97.0% | 93.0% | 95.0% |
| 01092 | 2.0% | 0.0% | 1.0% | 0.5% |
| 01102 | 98.5% | 90.0% | 96.5% | 98.0% |
| 01125 | 99.0% | 99.5% | 86.0% | 88.0% |
| 十任务宏平均 | 65.85% | 64.00% | 60.70% | 62.95% |

有力 Selected step=1,221,000，M0 Selected step=777,000。有力模型 Selected/Final 均只有 6/10 任务≥80%。资产路由正确不等于所有任务的夹持/reset/mating 区域都有效，00271/01092/00186 的弱结果需要保留。

来源：[修复后有力原始报告](/media/gpuserver/data2t/hx/github/FACA/logs/wrist_wrench_v3/joint10_nominal_seed1_corrected_v2_20260917/report.md)、[M0 报告](/media/gpuserver/data2t/hx/github/FACA/logs/wrist_wrench_v3/joint10_nominal_m0_seed1_20260918/report.md)、[弱任务审计](/media/gpuserver/data2t/hx/github/FACA/docs/joint10_weak_task_audit_20260918.md)。宏平均在本次按原表十行等权复算；这里没有复核全部 nominal 原始回合。

## F0–F3 周期内力信息实验

01079 和 01125 各自单任务 scratch 四臂：F0 无力；F1 周期末帧重复八次；F2 末帧、均值、population std 和 signed peak；F3 有序八点完整序列。F1–F3 共用相同 48→64→64 力编码分支，F1 与旧 20D 直接拼接不是同一模型。物理采样 120 Hz，策略 15 Hz，不是提高反馈控制频率。

每 run 一个 seed、150 env、699,300 transitions/51,615 updates，Selected 与 Final 独立测试各 2×100 回合：

| 任务 | F0 Selected/Final | F1 Selected/Final | F2 Selected/Final | F3 Selected/Final |
|---|---:|---:|---:|---:|
| 01079 | 99.5% / 99.5% | 72.5% / 22.5% | 91.5% / 91.5% | 97.0% / 98.5% |
| 01125 | 94.0% / 91.5% | 97.5% / 31.0% | 94.5% / 56.0% | 99.0% / 97.0% |

F3 在本次 seed 的 Final 较稳，但 01079 F0 仍最高；不能宣称全序列力在所有任务优于无力，也不能把低分全部归因于力。旧深度上限仍影响这些结果。来源：[完成报告](/media/gpuserver/data2t/hx/github/FACA/logs/force_cycle_v1/mechanism_seed1_20260919/report.md)、[实际实现合同](/media/gpuserver/data2t/hx/github/FACA/docs/force_cycle_implementation_20260919.md)。

## tasksize10-v2.1 共享尺寸训练

任务集由 nominal 的 00271/01092/01102 更换为 00649/00831/00659，其余七任务保留。原始十任务尺寸方案曾因几何准入而阻塞；不能把该阻塞方案与后来 v2.1 的完成结果混为一批。

每任务十个预烘焙尺寸，五 seen、五 held-out，共 50+50 条件；相对尺寸为 0.875/0.900/0.925/0.950/0.975/1.000/1.025/1.050/1.075/1.125，间隙固定为任务专属基准的 0.75 倍。各臂 scratch seed1，300 env，每 seen 条件六 env，1,398,600 transitions/103,230 updates，无历史 checkpoint/replay。

| 臂 | 观测与结构 | 参数条件 |
|---|---|---|
| M0 | 14D TCP，无力、无执行历史 | MT_True |
| M1 | 20D TCP+wrench，四步 state/wrench/前序动作历史 | MT_True |
| M2 | M1 加 factorized belief 和零权重辅助头 | MT_True |
| M3 | M2 加 alignment/contact 损失权重 6.5/0.072 | MT_True |

validation 72011/72012 仅 seen，用于选 checkpoint；test 82011/82012 独立报告 seen/held-out，各 split/臂/checkpoint 2,000 回合，每任务 200，每尺寸 40。M2 Selected 与 Final 相同，独立结果文件只有 28 份、28,000 回合；把角色展开为 32,000 行不会产生新评估。

| 臂 | Selected step | seen Selected | seen Final | held-out Selected | held-out Final | held-out Selected 最弱任务 |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 1,332,000 | 86.30% | 83.95% | 84.45% | 83.95% | 41.5% |
| M1 | 999,000 | 64.40% | 68.75% | 63.70% | 68.20% | 35.0% |
| M2 | 1,398,600 | 79.35% | 79.35% | 80.80% | 80.80% | 46.0% |
| M3 | 1,110,000 | 60.70% | 64.90% | 57.45% | 62.55% | 2.0% |

以下为 Selected held-out，全部从逐回合记录复算：

| 任务 | M0 | M1 | M2 | M3 |
|---|---:|---:|---:|---:|
| 00062 | 88.5% | 54.0% | 76.0% | 64.0% |
| 00186 | 41.5% | 45.5% | 59.5% | 4.5% |
| 00256 | 100.0% | 35.0% | 94.0% | 9.0% |
| 00649 | 44.5% | 65.0% | 46.0% | 2.0% |
| 00726 | 79.0% | 56.5% | 80.0% | 66.5% |
| 01079 | 98.5% | 98.0% | 94.0% | 98.0% |
| 01029 | 98.5% | 49.5% | 89.5% | 71.0% |
| 00831 | 100.0% | 93.5% | 96.0% | 97.0% |
| 00659 | 95.0% | 59.0% | 81.5% | 69.5% |
| 01125 | 99.0% | 81.0% | 91.5% | 93.0% |

M0 旧口径宏平均最高，M2 最差任务略高，但各臂都没有实现所有任务稳定成功。M0→M1 同时改变力、历史、输入维度和 burn-in，无法单独归因于腕力。四臂都是真实参数条件，不是十任务 MT_True 对 MT_Fixed 消融。

### 末态深度单条件重算

9 月 24 日已有重算仅判末态 `depth_fraction>0.85`。本次从原 episodes 重新计算得到：

| Selected 测试集 | M0 | M1 | M2 | M3 |
|---|---:|---:|---:|---:|
| seen 深度单条件 | 94.15% | 92.50% | 91.00% | 96.70% |
| held-out 深度单条件 | 94.35% | 92.60% | 91.90% | 96.30% |

臂序变化说明旧深度上限影响很大。这张表未检查横向、no-jam 和最后三步稳定，所以不能称为完整新版 relaxed。旧回合末值不足以重建连续三步；完整新版成绩需要单列闭环评估。本次范围之外已有 10 月 9 日重采集产物，因此“9 月 26 日报告当时待重评”不能被扩写成“目前仍没有新评估”。

来源：[原评估报告](/media/gpuserver/data2t/hx/github/FACA/logs/wrist_wrench_v3/tasksize10_v2_1_m0_m3_mt_true_seed1_matched_20260920_1710/evaluation/report.md)、[深度单条件重算合同](/media/gpuserver/data2t/hx/github/FACA/reports/phase_force_contact/wrist_wrench_v3/tasksize10_depth_gt_0p85_unbounded_20260924/report.md)。

## 可用于后续报告的结论

1. 单任务可学性、旧任务 retention、共享多任务能力和参数输入收益分别需要证据。获取 gate 成功不能替代 retention 或泛化。
2. 01079 三 seed 支持多尺寸物理训练优于只训练基准尺寸；未证明真实参数优于固定参数。结论限定于该家族、共同专家初始化和旧 relaxed。
3. V3 力输入收益随任务变化；F3 单 seed 的稳定表现值得保留，但当前证据不支持“力始终有益”或“力始终有害”。
4. 修复后 nominal 十任务与 v2.1 的物理任务集不同，宏平均不能直接相减。初版错误场景 joint 高分排除出有效多任务结论。
5. 旧 relaxed 下 M0 在 v2.1 宏平均最高；末态深度单条件下 M3 排序最高。两种口径必须并列命名，不能据后一张表覆盖前一张或冒充完整新版。
6. 早期 sync-rescue/M0–M3 的旧报告未形成可统一比较的冻结证据，保留历史状态；预算或方法方案不作为已完成结果。
7. SRSA 命名目录中的十任务队列实际是 AutoMate 从零 PPO 专家训练。官方宏平均 76.85% 不能替代新版 relaxed 15.50%；专家训练完成不代表十任务 generalist 或 DAgger 已完成。

## 复核产物和范围

FACA 实际读取根目录为 `/media/gpuserver/data2t/hx/github/FACA`。本汇总保存在 SRSA `docs/`，原有尺寸报告和训练代码未修改。

- [机器可读汇总及源 SHA256](pre_automate_results_20261010/verified_summary.json)：A/B/C 跨 seed 统计、OOD、紧/混合间隙及 M0–M3 全角色结果。
- [尺寸 A/B/C CSV](pre_automate_results_20261010/size_abc.csv)。
- [十任务臂级 CSV](pre_automate_results_20261010/tasksize10_arms.csv)。
- [十任务逐任务 CSV](pre_automate_results_20261010/tasksize10_tasks.csv)：含整数成功数/分母与分开的深度单条件。
- [只读复算脚本](../scripts/summarize_pre_automate_results.py)。
- [AutoMate 十任务 HDF5 复算结果](pre_automate_results_20261010/automate10_recheck/comparison.json)：官方、原版 relaxed、新版 relaxed 的整数计数、逐 seed 分解及 Wilson 区间。

复算命令：

```bash
python scripts/summarize_pre_automate_results.py \
  --faca-root /media/gpuserver/data2t/hx/github/FACA \
  --output-dir docs/pre_automate_results_20261010
```

AutoMate 十任务配对复算命令（使用安装了 h5py/numpy 的 Isaac 环境）：

```bash
/home/gpuserver/miniconda3/envs/isaac51/bin/python \
  scripts/automate/analyze_success_definitions.py \
  --paired logs/rl_games/Assembly/faca00271_automate_paired_relaxed_eval_20260927 \
  --output docs/pre_automate_results_20261010/automate10_recheck
```

复核已核对十任务原始 JSONL 与 result.json 的逐条件整数计数、每任务 200 回合、每文件 1,000 回合和每单元 checkpoint hash 一致性。未重新训练、未启动仿真、未按新版口径重选模型；原始训练时源码字节版本不能仅由今天的工作区代码证明。缺失的早期日志和 Phase 4 主报告仅作历史转录或索引分类，不能与本次原始回合复算混称为同级验证。
