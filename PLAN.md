# HGVTP-PLGAFormer 项目减法与四大实验主线整合计划

状态：待实施、待审核（本文件仅规划，不代表已经完成）  
日期：2026-08-09  
目标仓库：`HGVTP_PLGAformer-main/`  
实施者：Luna Max 5.6 子进程  
复核者：Codex（实施完成后另行审核）

## 0. 任务目标与执行边界

本轮不是增加模型、数据、实验场景或论文结论，而是通过合并职责、收拢入口和隔离历史资产，把项目整理成一条可解释、可运行、可审计的 HGV 论文主线。

实施期间必须遵守：

1. 只在本仓库运行 Git 命令，保留当前工作树中已有修改和未跟踪的 formal-v3 记录。
2. 不运行训练、数据生成、完整评测或论文数值更新；验证限于静态检查、单元测试、CLI dry-run 和 fixture。
3. 不修改模型结构、损失、数据划分、归一化、指标、种子、预测长度或冻结协议。
4. 不把 legacy、quick、smoke、pilot 或旧协议结果提升为正式论文证据。
5. 不直接写入 `../Init_Submit_TAES/`；论文产物只能先进入仓库内 staging。
6. 不立即删除历史代码和结果。先解除依赖、标记和提出归档清单；不可恢复删除须另行获批。
7. 不提交、不推送，除非用户另行明确授权。

## 1. 唯一论文主线

```text
完整 HGV 仿真轨迹
  -> 按完整 trajectory_id 划分 train/validation/test
  -> 各 split 内生成 256 s 历史到 256 s 未来的监督窗口
  -> 在同一任务、数据和训练预算下比较分析模型与学习模型
  -> PLGAFormer 用观测窗在线辨识的旋转地球 3-DOF 先验辅助长时预测
  -> 通过机理消融和物理一致性解释收益
  -> 通过动力学偏移和观测扰动检验泛化边界
  -> 通过参数量、FLOPs、时延和吞吐说明代价
  -> 由 manifest 将运行记录、表图和论文声明串成证据链
```

项目身份始终是“物理感知的 HGV 长时轨迹预测”，不是通用时间序列预测库。TSLib 只作为工程职责分层参考。

## 2. 冻结的科学与证据合同

- 协议：`hgv_multiregime_state_v2_1`；
- 数据：1,800 条完整轨迹，每条 1,000 点，1 Hz；
- 分层：两类垂向状态与 longitudinal/turning/weaving 组成六个联合层；
- 完整轨迹划分：1,260/180/360；先按 trajectory ID 划分，再生成窗口；
- 输入/输出：`seq_len=256`、`pred_len=256`、ECEF Cartesian 三维位置；
- 报告时域：32/64/128/256 s，均来自同一次 256 步直接预测；
- 正式种子：42/123/456；验证集选模，冻结测试集只作一次最终评估；
- 正式配置：`configs/formal_v3.json`；
- 证据链：claim -> table/figure -> manifest -> formal result -> run/checkpoint/seed -> code/data；
- 证据边界：纯仿真，不得写成飞行试验、实测雷达或部署验证。

以下路径已经进入配置、记录或清单。第一轮整理不得移动、重命名或重写：

```text
experiments/exp1_sota/results/formal_v3/hgv_multiregime_state_v2_1/
experiments/exp1_sota/trained_models/formal_v3/hgv_multiregime_state_v2_1/
experiments/exp2_ablation/results/formal_v3/hgv_multiregime_state_v2_1/
experiments/exp2_ablation/trained_models/formal_v3/hgv_multiregime_state_v2_1/
experiments/taes_submission_artifacts/generated/formal_v3/hgv_multiregime_state_v2_1/
```

## 3. 现有结构诊断

`exp1`–`exp7` 混合了论文核心比较、机理解释、压力测试和开发调参。编号增长反映历史开发过程，并不等于论文需要七个独立大实验。

| 现有单元 | 实际研究问题 | 当前问题 | 决策 |
|---|---|---|---|
| `exp1_sota` | 谁预测得更准 | 2,500+ 行单体兼做训练、评测、统计、绘图和效率 | 保留为大实验一；拆职责，不移动正式证据 |
| `exp2_ablation` | 为什么有效 | 四代历史消融混在 2,500+ 行文件；旧 full 身份存在冲突 | phase4 为大实验二核心；phase1–3 降为历史 |
| `exp3_robustness` | 扰动下是否稳定 | 噪声常量重复；噪声、缺失、短历史混合；与 exp5/7 重叠 | 提取 noise/history evaluator，并入大实验三 |
| `exp4_physics_consistency` | 是否物理合理 | 本质是评价指标，不应独占大实验；旧结果协议不兼容 | evaluator 并入大实验二 |
| `exp5_missing_data` | 缺测表现 | 每种场景重训、单种子、与贡献弱且和 exp3 重复 | 移出主线，保留 legacy，不补跑 |
| `exp5_ood_dynamics` | 动力学偏移泛化 | HGV 特异且有论文价值，但编号重复、入口位于 scripts | 作为大实验三核心 |
| `exp6_efficiency` | 代价多大 | 合理，但配置和 checkpoint 来源需统一 | 保留为大实验四 |
| `exp7_longterm` | 超参数如何选择 | 多为开发期搜索，与 exp2/3 重复 | 移出主线，保留 legacy |

关键依赖事实：

- `run.py` 仍公开 exp1–exp7 和 `all`，造成“七个正式实验”的错觉。
- `scripts/formal_pipeline.py`、`configs/formal_v3.json`、`utils/formal_evidence.py` 已形成 formal-v3 证据骨架，应扩展而不是另建一套。
- `scripts/run_formal_sota_unit.py` 和 `scripts/run_formal_ablation_unit.py` 直接导入两个大单体；解除依赖前禁止移动或删除它们。
- OOD 动力学偏移的真实入口是 `scripts/run_taes_dynamics_shift.py`，输出使用 `exp5_ood_dynamics` 名称。
- `utils/final_plgaformer.py`、`utils/experiment_io.py`、`utils/experiment_summary.py` 仍含 exp 编号、formal-v2 或 partial-run fallback。
- 新旧 paper generator、run-all 和 run-exp wrapper 并存，形成多条执行和产物路线。
- `paper_artifacts*`、formal/formal_v2、partial/quick/smoke 都是历史资产，不得进入 formal-v3 证据。

## 4. TSLib 参考边界

参考 `D:/OneDrive/02_Study/TSLib/Time-Series-Library` 的职责分离，不照搬其通用协议：

| TSLib 思路 | 本项目对应 | 明确不照搬 |
|---|---|---|
| 一个根入口 | 一个 HGV formal CLI | 不保留七个平级 task 作为主线 |
| `data_provider/` | 现有完整轨迹 loader/validation | 不复制通用 CSV split |
| model registry | `models/model_factory.py` | 不改变 final PLGAFormer 身份 |
| experiment abstraction | 四类 study + 共享 evaluator | 不制造巨型通用基类 |
| scripts | 声明式调用或薄兼容层 | scripts 不再承载第二套业务逻辑 |

## 5. 四大实验合同

### 5.1 Overall Prediction Performance

回答：在相同数据、输入、预测长度、训练预算和统计单位下，PLGAFormer 是否优于分析模型与学习基线？

- trainable：Transformer baseline、PLGAFormer full、PIT、DLinear、PatchTST、iTransformer、AF-CILN；
- analytical：kinematic、rotating-Earth 3-DOF；
- trainable 使用 42/123/456；分析模型按配置保留一个确定性记录；
- trajectory-level ADE、FDE、Cartesian RMSE，报告 32/64/128/256 s；
- 可按 maneuver/vertical regime 分层，但统计单位仍为完整测试轨迹；
- 配对统计必须基于同一 test trajectory IDs。

效率、物理违反率、扰动测试和硬编码论文数字不属于本实验。完整性只由 formal config 和 bundle validator 判断；缺失项只能形成 blocker，不得用 pilot 或旧协议补齐。

### 5.2 Mechanism Ablation and Physical Consistency

回答：收益是否来自旋转地球动力学先验、融合调度及其组合，并且是否改善物理行为？

正式主表只保留 formal-v3 phase4：baseline、spherical_prior、schedule_only、full，使用三种子；baseline/full 从大实验一同身份复用。

原 exp4 只作为 evaluator：约束违反率、轨迹平滑性以及物理单位 ADE/FDE/RMSE。阈值必须有公式、单位和来源；旧 seq64/pred128 或其他协议结果不得复用。

论文若继续声称“逐一移除每个 gate input”，则与现有 config 不一致。由于本计划要求不增加实验，默认决策是删除该声明，而不是增加新变体。phase1–3 只保留为方法演化记录。

### 5.3 Generalization and Robustness

回答：冻结模型在训练分布外的 HGV 动力学和合理观测扰动下是否保持相对优势，边界在哪里？

核心是 OOD dynamics：nominal、CL -10%/CD +10%、mass +15%/reference area -10%，比较 Transformer、PLGAFormer、rotating_3dof，并只使用大实验一通过 bundle 选出的冻结 checkpoint。

辅助压力测试仅保留：

- 物理单位和生成机制明确的 sensor noise；
- shortened input history，前提是不重新训练且输入接口一致。

默认舍弃 random missing；`exp5_missing_data` 和 `exp7_longterm` 中的 input-length 不再单独存在。所有场景必须记录单位、随机种子、受扰变量、`retrain=false`、checkpoint hash 和 test IDs。scaled MSE 不作为论文主指标。

### 5.4 Efficiency and Computational Cost

回答：精度与物理解释性需要多少额外代价？

报告参数量、batch-1 端到端 256 步时延、固定 batch 吞吐、峰值显存和 FLOPs；记录硬件、软件、warm-up、repeat、同步方式、dtype 和 batch size，并注明 profiler 对不支持算子的遗漏。shape 和 checkpoint 必须来自 formal bundle，不能从漂移的默认配置猜测。本实验不训练。

## 6. 目标结构与迁移原则

```text
experiments/
  overall_prediction/
  mechanism_analysis/
  generalization_robustness/
  efficiency/
  common/                       # 仅放两类以上 study 真正复用的逻辑
  legacy/                       # 兼容索引，不复制正式结果
  taes_submission_artifacts/
```

这是目标态，不允许一次性粗暴改名：

1. 先建立四类逻辑 study 注册表，旧目录继续作为实现路径和不可变 evidence root。
2. 所有 import、test、doc、manifest 解除硬编码后，再单独决定是否物理移动源码。
3. 禁止为了目录整齐复制代码。任何新文件必须替代重复职责，使活动入口、活动源码或重复 LOC 净减少。

论文 Experiments 最终映射为：Experimental Setup（不计大实验）、Overall Performance、Mechanism/Physics、Generalization/Robustness、Efficiency、Evidence Boundary。

## 7. 分阶段实施方案

### Phase 0：安全基线与依赖清单

1. 保存 `git status --short`、`git diff --stat` 和未跟踪清单，不得先清理工作树。
2. 生成“入口 -> import -> config -> output -> downstream test/generator/doc”依赖表。
3. 记录 formal-v3 records、checkpoints、config、manifest 的文件数和 SHA-256。
4. 标识此前 formal pipeline 已有修改，禁止覆盖。

验收：实施报告包含基线；formal-v3 内容未变化。无法区分用户修改时立即停止。

### Phase 1：建立四类逻辑注册表，不搬文件

1. 兼容扩展 `configs/formal_v3.json`，定义四类 study、允许场景、输入 bundle 和 artifact root；不改冻结字段。
2. `scripts/formal_pipeline.py` 公开 `main`、`mechanism`、`robustness`、`efficiency`；旧 sota/ablation 可作显式兼容别名。
3. `run.py` 默认只展示 formal 路线；exp1–exp7 移入显式 legacy 层，取消危险的默认 `all` 主路线。
4. 无参数运行只能显示帮助，正式命令不得隐式回落旧结果。

### Phase 2：收拢共享运行与评测职责

1. 从 exp1/2 识别共用的 dataset loading、model construction、checkpoint loading、trajectory metrics 和 serialization。
2. 优先复用现有 `data_provider/`、`model_factory.py`、`trajectory_metrics.py`、`formal_evidence.py`。
3. 只有两个以上 study 使用的逻辑才进入 `experiments/common/`，禁止创建没有第二使用者的抽象。
4. 先改 formal runner 的 import，再缩减旧单体；每一步保持兼容入口可导入。
5. 删除重复实现前用 `rg` 证明零调用；物理删除另列清单等待批准。

验收：formal runner 不依赖旧 CLI/绘图副作用；重复代码净减少；tensor shape、device、dtype、train/eval 和 causal flow 不变。

### Phase 3：整理大实验一

分离训练、冻结测试、统计、绘图和效率职责；模型、种子、时域和指标只来自 config；保留现有 artifact root；规范 record schema 和 bundle gate。当前缺 full/PIT 时只能输出 blocker。

### Phase 4：整理大实验二

正式入口只运行 phase4；phase1–3 需要显式 legacy 参数。baseline/full 只复用 main bundle。把 exp4 物理计算抽成无训练副作用 evaluator。消除“channel residual = final full”的当前命名，但不改历史记录。记录 gate-input 声明默认删除的决定。

### Phase 5：整理大实验三

将 dynamics shift 设为核心入口，从 main bundle 解析 checkpoint/test identity。从 exp3 提取 noise/history evaluator，删除重复 `NOISE_LEVELS`，将扰动定义放入正式配置。禁用 random missing 主线；统一 OOD/noise/history schema，同时保留 scenario_type、单位和随机性字段。

### Phase 6：整理大实验四

从 bundle 构造模型和 shape，去掉 exp1 的重复效率实现；固化环境、warm-up、repeat、CUDA synchronization 和显存协议；先输出 JSON/CSV，再由统一 generator 生成表图。

### Phase 7：统一论文产物路线

指定 `scripts/generate_taes_*` + validated bundle 为唯一正式路线。缺项只生成 blocker report，不生成带数字表图。旧 generator 降为 legacy。manifest 记录输入 hash、config hash、生成器版本和输出 hash。staging 不自动同步 submission workspace。

### Phase 8：文档与非破坏性清理

README 只讲协议、模型、四实验、正式命令、证据状态和产物路径。每个正式 study README 写清研究问题、输入、输出、指标、前提和论文位置。建立 legacy index。硬编码审计达到零调用、零正式文档引用、零 manifest 依赖后，才提出移至外层 `Archive/` 的清单；本阶段不执行删除。

### Phase 9：审核交付

实施者提交：changed-file list、逐文件理由、实际 keep/merge/legacy/archive 表、验证命令和退出码、formal-v3 前后 hash、未完成项、`git diff --check`、`git diff --stat`、`git status --short`，并明确是否运行训练（预期：否）。随后停止，不改论文、不归档、不提交，等待 Codex 审核。

## 8. 文件级修改顺序

第一优先级：

- `run.py`：七编号入口转为 formal 四 study + 显式 legacy；
- `configs/formal_v3.json`：扩展 study 合同，不改冻结语义；
- `scripts/formal_pipeline.py`：唯一正式编排入口；
- `utils/formal_evidence.py`：四类 bundle/schema/gate，禁止隐式 fallback；
- 两个 formal unit runner：逐步解除大单体副作用依赖。

第二优先级：

- `experiments/exp1_sota/SOTA_comparison.py`：拆训练/评测/统计/效率/绘图；
- `experiments/exp2_ablation/ablation_study.py`：phase4 与历史 phase1–3 分离；
- `experiments/exp3_robustness/robustness_analysis.py`：保留 noise/history，去重配置；
- `experiments/exp4_physics_consistency/physics_consistency.py`：提取物理 evaluator；
- `scripts/run_taes_dynamics_shift.py`：纳入大实验三；
- `experiments/exp6_efficiency/efficiency_experiment.py`：改用 bundle/config。

第三优先级：

- `utils/final_plgaformer.py`、`utils/experiment_io.py`、`utils/experiment_summary.py`：legacy 只可显式 opt-in；
- `scripts/generate_taes_*.py`：统一正式 bundle 和 manifest；
- 旧 generators、run-all、run-exp wrappers：降为 legacy 或列待删清单；
- README、scripts README、protocol registry、四个 study README：同步。

默认不修改：模型数学实现、数据生成和冻结 dataset、formal-v3 final 内容、submission workspace、ExternalCode、References 和任何历史结果内容。

## 9. 风险与停止条件

出现以下任一情况必须停止：

1. config hash 因非预期语义变化导致旧记录无法验证；
2. 必须移动或重写 formal-v3 final 才能继续；
3. 无法区分用户修改与实施者修改；
4. baseline/full 身份、checkpoint 或 test IDs 不一致；
5. 新结构导致代码复制而非净减少；
6. 测试会启动训练、写入正式结果或覆盖 artifact；
7. 需要新增模型、数据、场景或正式实验变体；
8. 需要修改论文数值或 submission snapshot；
9. 待删/归档对象仍被代码、文档、测试或 manifest 引用。

## 10. 验证策略

- 静态：Python AST/import、JSON/config、`git diff --check`；
- 单元：formal evidence、CLI、model identity、artifact immutability、legacy fallback rejection；
- fixture：完整 bundle 通过；缺 seed、重复 record、协议/hash/checkpoint 错误均失败；
- CLI：无参数帮助、四类 dry-run/status、legacy 显式告警；
- 不变性：formal-v3 final 文件数和 SHA-256 前后一致，final roots 无新增；
- 依赖：`rg` 确认正式文档和 generator 不再读取 formal_v2、partial_runs、paper_artifacts_v2；
- 差异：不夹带模型或协议更改。

不得运行完整训练、数据再生成、正式测试集重评、论文数值生成或 LaTeX 提交包同步。

## 11. 完成验收标准

- 对外恰好四个论文级大实验；Experimental Setup 不计数；
- exp5 missing、exp7、exp2 phase1–3 不再默认执行；
- exp4 是大实验二 evaluator；OOD 与选定 noise/history 组成大实验三；
- 一个 formal CLI、一个配置合同、一个正式 artifact 路线；
- 旧入口若保留，只是薄 wrapper 且明确 non-paper-facing；
- 活动重复代码和入口数量净下降，没有为改名复制代码；
- formal-v3 数据、结果、checkpoint、manifest 内容未改变；
- 缺失正式运行仍诚实显示 blocker；
- README、代码、配置、测试对四实验名称和职责一致；
- Codex 审核前不声称项目、实验或论文已经完成。

## 12. 留待后续讨论，当前不执行

1. 结构稳定后是否物理重命名旧实验目录；
2. 零引用历史代码是移到外层 Archive，还是依赖 Git 历史后删除；
3. noise/history 放主文、补充材料，还是只保留 OOD；
4. gate-input 声明是否按默认方案删除；
5. Main Results/Ablation 补齐后如何同步 TAES 稿；
6. 是否需要第五个大实验。当前结论是不需要，除非出现四类合同无法承载的新核心主张。

## 13. 给 Luna Max 的最终指令

先读本文件、外层与仓库 `AGENTS.md`、当前工作树和 formal config，再从 Phase 0 开始。严格按 Phase 0–9 推进，每阶段先验证再继续。优先减少入口、重复实现和概念数量，不要用“新框架”替换旧复杂度。若触碰冻结协议、要求新增实验或超出结构整理范围，立即停止并报告。完成后只交付实施报告和工作树差异，等待 Codex 审核。
