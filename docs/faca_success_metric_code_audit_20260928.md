# FACA 成功率定义的代码审计与正文替换稿

## 范围与状态

2026-09-28，只读核查本地 Isaac Lab AutoMate、SRSA runtime、FACA TD-MPC2 评估入口与当前 jam 校准协议。**没有改动任何成功率数字或冻结 benchmark 代码。** 用户提到的正文文件尚未定位；下文是可直接移入正文的方法定义，并把未实现的真机和安全事件规则单列。

本次 jam development_v2 的 import_audit 确认实际导入的是运行目录中的 SRSA 快照（assembly_runtime_env.py SHA256: f767df6a4cf7550ae048a1ddea497eea9a3df8fbeacaebcb6668ae4c71883bdc），不是当前工作区文件。其 resolved.yaml 记录了 relaxed_depth_unbounded=true、relaxed_success_steps=3、eval_terminate_on_success=false。source_lock.json 锁定采集器和该 SRSA 快照，但**未列入 FACA/tdmpc2/envs/isaaclab.py 与 FACA/tdmpc2/config.py**；因此仅凭现有 source lock 不能证明采集时这两个评价文件的精确字节版本。后续正式评估必须将其 SHA256 纳入冻结协议，同时保存 resolved.yaml、import_audit、原始逐步轨迹及 checkpoint/资产哈希。当前代码和逐步记录支持以下语义审计，但不能把缺失的源锁说成已经补齐。\n\n这几种指标必须分名：`official_success_latched` 是 AutoMate benchmark 兼容口径；`relaxed_success_stable` 是 FACA 当前选定主口径，且本轮必须显式开启 `relaxed_depth_unbounded=true`；`strict_success_stable` 是现有更严格的辅助指标。三者不可都写成“depth 到了且 $T_i\le T_{max}$”。

## 六个问题：代码核查结果

### 1. $d_i$ 是什么？

对 FACA 修改后的 relaxed，令第 $i$ 回合第 $t$ 个**控制步结束后**的 plug（`held_asset`）刚体根位置为 $p^W_{P,i,t}$，socket（`fixed_asset`）刚体根位置及姿态为 $(p^W_{S,i,t},R^W_{S,i,t})$，任务给定的插入目标深度为 $D_i>0$。代码计算

\[
r^S_{i,t}=(R^W_{S,i,t})^\top(p^W_{P,i,t}-p^W_{S,i,t}),\qquad
d_{i,t}=\max(0,D_i-r^S_{i,t,z}),\qquad
\rho_{i,t}=d_{i,t}/\max(D_i,10^{-6}{\rm m}).
\]

这里的 $d_{i,t}$ 是**实际 plug/socket 根位姿的轴向相对位置与名义目标 $D_i$ 组合得到的深度**；轴是 socket 局部 $z$ 轴。它不是 TCP 的位移，不是从 reset 时刻起累积的位移，也不是 keypoint 推导的深度。`held_pos/fixed_pos/fixed_quat` 来自仿真刚体状态；keypoint 只在官方和 strict 判据中单独使用。若 socket 姿态非恒等，官方 benchmark 的世界坐标 $z$ 高度测试和这里的 socket 局部 $z$ 投影并不等价。

代码入口：`FACA/tdmpc2/envs/isaaclab.py:651-679`；plug/socket 状态来源：`IsaacLab/.../automate/assembly_env.py:273-283`；SRSA 同义实现：`source/SRSA/SRSA/tasks/direct/srsa/assembly_runtime_env.py:1984-1987,2092-2119`。

### 2. final depth 还是 max depth？

**现行主指标用终止采样点的稳定成功状态，不用 $\max_t d_{i,t}$。** 对每步形成布尔条件 $g_{i,t}$，维护连续满足计数 $c_{i,t}=g_{i,t}(c_{i,t-1}+1)$（不满足即归零）。终止时 `relaxed_success_stable = (g_{i,T_i} \land c_{i,T_i}\ge3)`。因此早先达到深度、末步退出容差或失去连续 3 步条件的回合，在 stable 指标下失败。`relaxed_success_episode = \max_{t\le T_i}\mathbf1[c_{i,t}\ge3]` 是另一个**曾经稳定成功**的诊断字段，不能替换主指标。`depth_fraction_max` 若被记录，仅为诊断量，也不能直接算成功率。

AutoMate 官方 `ep_succeeded` 则按每步官方成功取逻辑 OR；**一次命中即锁存到回合末**，无需终态保持。要复现 benchmark 表，读取 `official_success_latched`，不能用 `official_success_terminal` 或 `relaxed_success_stable` 顶替。

代码入口：`FACA/tdmpc2/envs/isaaclab.py:768-784,1628-1641`、`IsaacLab/.../automate/assembly_env.py:525-540`。

### 3. 有没有 lateral validity？孔旁下压能否算成功？

新版 relaxed 的每步几何条件为

\[
g_{i,t}=\underbrace{[\rho_{i,t}>0.85]}_{\text{无 }\rho<1\text{ 上限}}
\land\underbrace{[\|r^S_{i,t,xy}\|_2\le\tau^{rel}_{i}]}_{\tau^{rel}_{i}=\operatorname{clip}(2c_i,1,3)\,\mathrm{mm}}
\land[\neg J_{i,t}],
\]

其中 $c_i$ 为任务的**径向**间隙；若只有直径间隙，代码用其一半。默认 relaxed_success_require_official=false。所以只在孔旁边使 $z$ 降低、但横向误差超过 $\tau_i^{rel}$ 的回合**不会**计为 relaxed 成功。但 relaxed **不要求姿态、yaw 或 keypoint 合格**；即使几何中心满足这两个标量门槛，也不能单凭代码宣称已物理就位、全长穿入或未发生碰撞。特别是取消 $\rho<1$ 上限后，过插入可能被计为成功，需另报其发生率和力/抓取漂移。

现有 `J` 定义为 `contact & ~current_official & (lateral_error > max(radial_clearance, lateral_tol))`。由于 relaxed 已要求 `lateral_error <= lateral_tol`，**在同一采样点，no-jam 条件对通过横向门槛的状态数学上没有额外筛选作用**；不能把它写成独立的卡滞安全保证。此外，当前 jam 校准 `development_v2` 的参考机械卡滞标签不足、阈值未验证，不能把任何候选力阈值当作真值或 force-stop。

在核查的 development_v2 运行配置中，srsa_enable_held_asset_net_contact_force=false 和 srsa_force_semantics_audit_enabled=false，**但这并不关闭运行时接触审计**：采集器设置 srsa_force_eval_metrics_enabled=true，FACA wrapper 据此设置 SRSA_FORCE_SEMANTICS_AUDIT_ENABLED=true。受控 policy 的 wave_000 轨迹在 14,800 个控制步中记录 relaxed_jam=272 次，但这 272 次与 relaxed lateral 合格的交集为 0；因此该项存在真实非零信号，却不额外改变当前 relaxed 成功结果。腕部 wrench、接触标志和 force-stop 是不同概念，不能互相替代。代码在 socket 坐标旋转出错时会退回世界坐标差；正式复现应核实运行时依赖及坐标变换没有这种退回。

AutoMate 官方 benchmark 使用 `plug_z > socket_z`、`plug_z < socket_z + D_i`，再要求 plug/socket keypoint 的平均欧氏距离 **严格小于** `close_error_thresh`。这个 keypoint 检查专门防止 plug 在 socket 外侧但高度合格；官方没有独立的显式横向标量门槛、3 步 hold 或 force-stop。现有 `strict_success_stable` 还要求深度比例 `>=0.90` 且 `0<r_z^S<D_i`、更紧的 lateral、姿态、必要时 yaw、keypoint 和连续 10 步。**strict 仍有深度上限**，不能冒充新版 relaxed 的无上限版本。

代码入口：`FACA/tdmpc2/envs/isaaclab.py:665-825`；官方判据：`IsaacLab/.../automate/automate_algo_utils.py:201-246`；当前 jam 证据：`FACA/logs/jam_calibration_v1_20260928/evidence_review_v2/report.md`。

### 4. 是否需要 hold？

FACA 新版 relaxed 需要**连续 3 个控制步**同时满足 $g_{i,t}$，且在终止采样点仍满足。现有配置和 `development_v2/resolved.yaml` 均为 `relaxed_success_steps: 3`。AutoMate 官方指标只需单步，之后回合锁存；现有 strict 为连续 10 个控制步。这里的 3 步不是“0.2 秒保持”的代码实现。AutoMate 基础配置 `dt=1/120 s, decimation=8`，名义控制周期约 `1/15 s`；从第一个满足样本到第三个满足样本跨 **2 个控制周期，约 0.133 s**。若正文想要求 0.2 s，必须独立实现按时间戳累计的 hold 并重新评估，不能改文字冒充已有结果。

代码入口：`FACA/tdmpc2/envs/isaaclab.py:768-774`、`FACA/tdmpc2/config.py:195-218`、`IsaacLab/.../automate/assembly_env_cfg.py:72,108`。

### 5. sim 和 real 各怎样计算 $d_i$？

**仿真**：直接读取 PhysX 中 plug/socket 刚体根位姿，按第 1 节公式计算。TCP 观测、腕力与相机不是该深度的来源；目标 (D_i) 来自当前任务的 `insertion_depth`，在 SRSA runtime 中由 `current_insertion_depth_tensor` 保存。

**现有真机入口**：FACA `eval_real_closed_loop` 只接收真机侧 canonical TCP observation、生成并发送 action；遇到消息 `done=true` 或达到 `eval_real_steps` 就退出。该函数**没有读取 plug 位姿、没有按上述公式计算真机深度，也没有计算真机成功率**，返回的 `episode_success: 0.0` 是占位值。真机 HIL 采集中的 `done/success` 是真机侧消息或人工 episode marker 的外部标签，不能与仿真指标视为同一算法。`eval_real_socket_pos` 和标定的 socket 坐标系可用于 TCP 观测/命令或距离 reward，但**TCP-to-socket 位置不能无条件代替 plug-to-socket 深度**，尤其 `physical_grasp` 下可能发生抓取漂移。没有找到可复现的真机 plug/socket 位姿测量、TCP-to-plug 外参、误差界和同步时钟协议。

若正文包含真机成功率，应补充独立真机测量协议：固定并公布 socket 坐标系标定 $T^B_S$、plug 位姿来源（视觉/动捕，或在经过漂移验证后用机器人正运动学 $T^B_{TCP}$ 与标定 $T^{TCP}_P$ 推算）、传感器时间同步、丢帧规则、位置/角度误差界与人工复核规则。以同一公式重建 $r^S$ 和 $d$，并明确是否有独立的视觉就位确认。**在这些证据具备前，不应写“sim 与 real 用相同深度判据”，也不能报告由真机闭环函数返回的 0.0 为实测失败率。**

代码入口：`FACA/tdmpc2/eval.py:760-880`、`FACA/tdmpc2/zmq_action_publisher.py:385-386`；真机文档：`FACA/docs/eval.md:927-1026,1128-1138`。

### 6. 同一步 success、force stop、timeout 的优先级

**当前仿真代码实际行为**：物理步后先运行 `_get_dones()`；AutoMate `_get_dones()` 只按 `episode_length_buf >= max_episode_length-1` 返回超时（两个返回位均为 timeout）。随后算 reward 和 success，再允许可选的 `eval_terminate_on_success` 与 done 作 OR，最后将**同一步算得的 success** 写入 `final_info`，reset 环境；wrapper 对所有 done 返回 `terminated=false, truncated=true`。当前 `development_v2` 显式 `eval_terminate_on_success=false`，74 步同步评估。代码中**没有 force-stop 终止分支或已验证的强制停机阈值**；力峰值统计和 reward penalty 不等于 force stop。

因此现行可复现的同一步情形是：第 74 步同时 timeout 与 relaxed 3 步稳定成功，`final_info.relaxed_success_stable=1`，该回合计为成功；若此前成功但第 74 步未保持，stable=0，而 `relaxed_success_episode` 仍可为 1。官方 `official_success_latched` 只要此前或该步曾命中即为 1。现有代码**没有**可回答“force stop 与 success 同步时谁优先”的行为；正文不能声称已有确定顺序。

| 同一步事件 | 当前 `final_info` 与计数 | 如新增独立 `safe_success`，建议的终止原因 |
| --- | --- | --- |
| stable 成功 + timeout | stable=1；该回合主指标成功 | `success_at_timeout`，成功 |
| 先前曾成功，末步不满足 + timeout | stable=0；episode 指标可为 1 | `timeout`，不计终态稳定成功 |
| 官方单步成功 + timeout | 官方锁存=1 | 不改变官方 benchmark |
| force stop + stable 成功（同控制步） | 当前没有 force-stop 事件，无法从现有日志裁决 | `safety_stop`，`safe_success=0`；原始几何布尔量另存 |
| force stop + timeout | 当前没有 force-stop 事件 | `safety_stop` 优先于 timeout |

如论文另定义安全调整后的指标，建议先冻结并单独实现如下**拟议规则**，不得回写旧 benchmark：在每次物理子步记录硬件/仿真安全事件，形成 `safety_stop_at_or_before_t`；控制步结束时先判安全停机，其次判终态稳定成功，最后判 timeout。若安全事件与成功在同一控制步内发生且子步先后无法证实，安全停机优先，`safe_success=0`；若无安全事件且同一步成功与 timeout 并存，`safe_success=1`。同时保存互不覆盖的 `official_success_latched`、`relaxed_success_stable`、`safety_stop`、`timeout`、`terminal_reason`，使 benchmark 结果和安全调整结果可分别复算。这个顺序是**待实现的预注册建议**，不是当前实现事实。

代码入口：`FACA/tdmpc2/envs/isaaclab.py:1826-1914`、`IsaacLab/.../automate/assembly_env.py:519-523`；当前 jam 校准状态：`FACA/logs/jam_calibration_v1_20260928/development_v2/status.json`。

## 可直接替换正文的方法段落

> 我们分别报告 AutoMate 官方 benchmark、FACA relaxed terminal stable 和 strict terminal stable 成功率。官方 benchmark 在每个控制步检验 plug 世界坐标高度是否位于 socket 高度与目标插入高度之间，并检验 plug/socket 对应 keypoint 的平均欧氏距离是否小于任务阈值；任一步满足即在该回合锁存为官方成功。FACA 主指标在每个控制步使用仿真 plug/socket 刚体根位姿：将 plug 相对 socket 的位置旋转到 socket 坐标系，令插入深度为目标深度减去相对轴向高度（下限截为零），再除以目标深度。一步有效需要深度比例严格大于 0.85（不设上限）且横向距离不超过 `clip(2×径向间隙, 1 mm, 3 mm)`；代码包含 no-jam 项，但在现行阈值下该项不增加通过条件。主指标要求该条件连续满足 3 个控制步，且在回合终止步仍成立。成功率以固定评估回合数为分母，读取终止 `final_info.relaxed_success_stable`；达到过门槛但终态未保持的回合只计入单列的 `relaxed_success_episode`。现有 strict 指标另要求更高深度比例、原高度窗口、姿态、必要时 yaw、keypoint 和连续 10 步，故与主指标不等价。所有结果同时披露 `relaxed_depth_unbounded=true`、评估步数、checkpoint、测试 seed 和任务条件。真机结果使用独立记录的外部 success 标签；在 plug/socket 位姿测量与时间同步协议确立前，不称其与仿真深度成功率同口径。

**对原公式的处理**：若 $T_i\le T_{max}$ 只是回合有效性约束，应另写为测试窗口，不作为空间成功的充分条件；$d_i\ge d_{target}$ 必须替换成上述逐步深度、横向及连续稳定定义。保留现有 benchmark 数字与原列名，增加新版 relaxed/strict 列或表，不能用旧逐回合终态深度直接换算。对新安全停机规则，在代码、协议和逐子步日志落实之前只列“拟议”，不写成既有实验方法。
