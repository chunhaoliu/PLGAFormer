# 项目与顶刊标准的差距及详细修改建议

本文档基于对项目代码的系统阅读，从 **NeurIPS/ICML/ICLR/TPAMI** 等顶刊的常见要求出发，列出差距并给出可执行的修改建议。

---

## 一、实验设计与对比完整性

### 1.1 SOTA 对比实验只跑了一个模型（严重）

**现状**  
- `experiments/exp1_sota/SOTA_comparison.py` 中 `COMPARISON_MODELS` 仅包含一项：`"Transformer (baseline)"`。  
- `create_model()` 只支持 `transformer` 和 `pit`，**没有 PLGAFormer、Informer、PatchTST、FEDformer 等**。  
- 结果：名为 “SOTA comparison”，实际只训练/评估了基线 Transformer，无法支撑“优于 SOTA”的结论。

**顶刊常见做法**  
- 主表至少包含：强基线（如 Standard Transformer）、2–3 个同任务/同领域 SOTA（如 PIT、Informer 等）、以及提出的方法（PLGAFormer）。  
- 每个方法同一配置下多轮运行，报告 mean±std 并做显著性检验。

**修改建议**  
1. **扩展 `COMPARISON_MODELS`**（在 `SOTA_comparison.py` 中），例如：
   - `"Transformer (baseline)"` → `create_baseline_model('transformer', ...)`
   - `"PLGAFormer (proposed)"` → 使用 `create_sota_model('plgaformer', ...)` 或你们现有的 PLGAFormer 工厂，并配物理损失
   - `"PIT"` → `create_pit_model(...)`
   - 若时间允许：`"Informer"`, `"PatchTST"` 等通过 `create_sota_model(...)` 加入  
2. **统一 `create_model(model_type)`**：根据 `model_type` 分别调用 `create_baseline_model`、`create_pit_model`、`create_sota_model`，保证每个 `COMPARISON_MODELS` 中的条目都有对应实现且输入/输出与评估脚本一致（如 `output_dim`、预测长度等）。  
3. **主表与附录**：主表放主要方法 + 主要指标（如 MSE/MAE/FDE 在 32/64/128 步）；完整结果（含 ADE、多 horizon、per-maneuver）可放附录或补充材料。

---

### 1.2 消融实验的“基线”与“完整模型”定义

**现状**  
- 消融已改为：基线 = Standard Transformer（`create_baseline_model('transformer')`），其余为 PLGAFormer 及其变体，这点已符合“只改创新点、其余一致”的诉求。  
- 需在正文或附录中**明确写出**：“baseline = Standard Transformer（与 PLGAFormer 同 encoder/decoder 配置：gelu, norm_first=True）；完整模型 = PLGAFormer（B+C 或你们最终采用的创新点组合）。”

**修改建议**  
- 在 `ablation_study.py` 或配套 README 中增加 1 段 “Ablation protocol”：列出 baseline 与各消融变体的定义（含 `model_type` 与创新点开关），便于审稿人复现。

---

## 二、可复现性（Reproducibility）

### 2.1 随机种子与确定性

**现状**  
- 已有 `set_random_seed(seed)`（含 `cudnn.deterministic`、`use_deterministic_algorithms`），且多次运行用不同 seed。  
- SOTA 脚本里部分路径仍用 `torch.manual_seed(42)` 等写死一次，与“每 run 一个 seed”的循环一致即可。

**修改建议**  
1. **单入口**：所有实验（SOTA + 消融）在“每轮运行”开始时只调用一次 `set_random_seed(seed)`，避免在数据加载、模型初始化、DataLoader 之外再改 seed。  
2. **文档化**：在 README 或 `docs/REPRODUCIBILITY.md` 中写明：  
   - 使用的 Python / PyTorch / CUDA 版本；  
   - 是否使用 `torch.use_deterministic_algorithms(True)` 及可能带来的性能影响；  
   - 若某些操作无法完全确定性（如部分 cuDNN），在文档中说明并给出“建议环境”。

### 2.2 依赖与环境

**现状**  
- 未发现项目根目录的 `requirements.txt` 或 `environment.yml`。

**修改建议**  
1. 增加 `requirements.txt`（或 `environment.yml`），固定主要版本，例如：  
   - `torch>=2.0,<2.3`  
   - `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`  
   - 若用 Optuna：`optuna`  
2. 在 README 中说明：  
   - 推荐 Python 版本（如 3.10）；  
   - 安装步骤：`pip install -r requirements.txt`；  
   - 数据生成与实验的先后顺序（先跑 data_generator 再跑 SOTA/消融）。

### 2.3 数据与路径

**现状**  
- 数据路径多为相对路径，如 `data_generation/data/processed/hgv_trajectory_dataset.npz`，依赖当前工作目录。

**修改建议**  
1. **项目根目录统一**：在配置或常量中定义 `PROJECT_ROOT`（例如用 `pathlib` 基于 `__file__` 解析），所有数据/结果路径基于 `PROJECT_ROOT` 拼接，避免因在 `experiments/exp1_sota/` 或根目录运行导致找不到文件。  
2. **数据版本**：若会多次重新生成数据，建议在 npz 或目录名中带版本/日期，或在 README 中说明“默认使用某次生成的数据”，便于复现。

---

## 三、训练与优化设置

### 3.1 学习率调度

**现状**  
- SOTA：`StepLR(step_size=1, gamma=lr_decay_factor)`，每 epoch 衰减。  
- 消融：Cosine Annealing + Linear Warmup（更接近 Informer/PatchTST 等顶刊设置）。

**修改建议**  
- **统一为 Cosine + Warmup**：在 SOTA_comparison 的 `train_model` 中采用与消融相同的 warmup + cosine 策略，并在 HGVConfig 或实验配置中统一 `warmup_epochs`、`total_epochs`，这样 SOTA 与消融的对比更公平，也符合顶刊常见写法。  
- 若保留 StepLR，需在正文中说明为何与消融不同，并做敏感性实验（如附录）证明结论不依赖调度器选择。

### 3.2 优化器

**现状**  
- SOTA：`Adam`；消融：`AdamW`。

**修改建议**  
- 统一为 **AdamW**（顶刊中更常见），并统一 `weight_decay`（如 1e-5 或 5e-5），避免“方法 A 用 Adam、方法 B 用 AdamW”的混淆。

### 3.3 超参数搜索与报告

**现状**  
- HGVConfig 中有 `from_trial()` 支持 Optuna，但未看到 SOTA/消融主流程中调用超参搜索。  
- 超参数多为默认或手工设定。

**修改建议**  
1. **主实验**：可保持固定一组超参，但在正文/附录中给出“超参数表”（学习率、batch size、层数、dropout、物理损失权重等），并注明“未做大规模搜索，与基线使用相同或对应设置”。  
2. **可选**：对 PLGAFormer 做小规模网格/随机搜索（如学习率、alpha/beta），在附录报告“敏感性”或“选参方式”，提升说服力。

---

## 四、评估与报告

### 4.1 指标与单位

**现状**  
- 已有 MSE、MAE、RMSE、FDE、ADE；FDE/ADE 在笛卡尔空间以米为单位，与顶刊轨迹预测一致。  
- `REPORT_ON_PHYSICAL_SCALE`、`REPORT_MSE_ON_SCALED` 控制是否在物理量/归一化空间报告。

**修改建议**  
1. **正文主表**：明确写清“MSE/MAE 是在标准化空间还是物理空间”，单位（若物理空间）写清楚（如 m²、m）。  
2. **FDE/ADE**：保持“米”并注明 “in meters” 或 “(m)”，与现有代码一致即可。  
3. **一致性**：若主表用“物理空间 MSE”，则全文统一，避免同一指标一处 scaled、一处 physical。

### 4.2 统计检验与报告格式

**现状**  
- 已做配对 t 检验、报告 p-value，并区分 * p<0.05、** p<0.01。  
- 多轮运行报告 mean±std。

**修改建议**  
1. **样本量**：若 `num_runs=3`，在文中说明“3 次独立运行”；若审稿人要求更高，可改为 5 次并报告 5 次的 mean±std 与 p-value。  
2. **多重比较**：若同时比较多个方法、多个指标，可考虑简单校正（如 Bonferroni）或在附录说明“未校正，仅作参考”，避免审稿人质疑 inflated Type I error。  
3. **表格**：主表用 “mean ± std” 且对显著优于/劣于基线的单元格加粗或标注 *，便于阅读。

### 4.3 Per-maneuver 与效率

**现状**  
- 已有 per-maneuver 评估和效率 benchmark（延迟、吞吐、参数量）。  
- 可视化中引用了 “LSTM”, “SS-DLSTM” 等，但 `COMPARISON_MODELS` 中可能没有对应项，可能导致绘图时报错或缺曲线。

**修改建议**  
1. **Per-maneuver**：若正文/附录要报 per-maneuver 结果，确保所有被报告的模型都已在 `COMPARISON_MODELS` 中训练并保存，且与主表一致。  
2. **效率**：保持“同一设备、同一 batch/seq 长度”下测延迟与显存，并在文中注明（如 “on NVIDIA V100, batch size 32”）。  
3. **可视化**：`generate_pit_style_trajectory_visualization` 等函数中，只绘制 `models_dict` 或 `COMPARISON_MODELS` 里实际存在的模型，避免写死 “LSTM”“SS-DLSTM” 导致 KeyError 或缺失。

---

## 五、数据与物理设定

### 5.1 数据集描述

**现状**  
- 数据由 `DCBNN_HGV_Simulator` 等生成，注释中提到了 PIT/DCBNN 参数；缺少面向论文读者的“数据集说明”。

**修改建议**  
1. 在 **README** 或 **docs/DATA.md** 中增加“数据集”小节：  
   - 输入/输出维度与含义（如 [r, λ, φ, V, γ, ψ] → [r, λ, φ]）；  
   - 序列长度、预测长度、划分比例（如 7:1:2 或 8:1:1）；  
   - 机动类型（longitudinal / turning / weaving）及大致比例；  
   - 是否做标准化、标准化方式（如 StandardScaler per feature）。  
2. 若与 PIT/DCBNN 论文一致，可写“仿真参数与 PIT/DCBNN 一致，见 …”，便于审稿人对照。

### 5.2 物理约束与损失

**现状**  
- HGVPhysicsLoss、物理常数、约束范围等在 HGVConfig 与 plgaformer 中已集中；HGVConfig 中已有 `output_dim`、物理约束等说明。

**修改建议**  
- 在方法部分或附录中给出“物理约束与损失”的简短公式或表格（如高度/纬度范围、损失中 α/β 的含义），与代码注释一致，便于审稿人理解“物理信息”如何被使用。

---

## 六、代码结构与工程实践

### 6.1 配置唯一真相源

**现状**  
- 已通过 HGVConfig 集中 num_runs、random_seeds、data_subset_ratio、output_dim 等；SOTA 与消融从同一处读取，方向正确。

**修改建议**  
- SOTA 脚本中若仍有“本地常量”（如预测长度列表、batch size）与 HGVConfig 重复，逐步改为从 HGVConfig 或 `get_experiment_config('sota')` 读取，减少两套逻辑。

### 6.2 长脚本拆分

**现状**  
- `SOTA_comparison.py` 超过 2200 行，包含数据加载、训练、评估、可视化、主流程等。

**修改建议**  
1. **按职责拆分**（在不影响你当前实验的前提下逐步做）：  
   - `data_loading.py`：加载与预处理；  
   - `training.py`：训练循环与验证；  
   - `evaluation.py`：指标计算与统计检验；  
   - `visualization.py`：绘图（或与现有 `visualize_results.py` 合并）；  
   - `run_sota.py`：只做配置 + 调用上述模块。  
2. 这样便于单元测试、复现单步、以及审稿人/后续维护者阅读。

### 6.3 文档与注释

**现状**  
- 关键函数有中文注释；缺少面向“复现者”的顶层说明。

**修改建议**  
1. **README.md**（项目根目录）：  
   - 一两句话概括项目（HGV 轨迹预测 + PLGAFormer）；  
   - 环境与安装；  
   - 数据生成命令；  
   - SOTA 与消融的运行命令（如 `python experiments/exp1_sota/SOTA_comparison.py`）；  
   - 结果所在目录（如 `experiments/exp1_sota/results/`）。  
2. **实验 README**：在 `experiments/exp1_sota/` 和 `experiments/exp2_ablation/` 下各放简短 README，说明该实验目的、入口命令、输出文件及与主文的对应关系。

---

## 七、与顶刊审稿要点的对应

| 审稿常见问题           | 当前状态           | 建议 |
|------------------------|--------------------|------|
| 对比是否全面？         | 仅 Transformer 基线 | 扩展 COMPARISON_MODELS，至少加入 PLGAFormer、PIT，可选 Informer 等 |
| 多次运行 + 显著性？    | 已有 mean±std + t 检验 | 保持；可考虑 5 次 run 与多重比较说明 |
| 超参是否一致/公平？    | 部分一致           | 统一优化器、学习率调度、数据与评估协议 |
| 可复现吗？             | 缺依赖与运行说明   | 增加 requirements + README 运行步骤 |
| 数据与任务定义清楚吗？ | 分散在代码注释     | 集中到 README/DATA.md + 方法/附录 |
| 消融基线是否合理？     | 已改为 Standard Transformer | 在文档中明确写出“消融协议” |
| 物理信息如何用？       | 有 HGVPhysicsLoss 与配置 | 方法/附录中简短公式或表格 |

---

## 八、建议的修改优先级

1. **P0（必做）**  
   - 扩展 SOTA 的 `COMPARISON_MODELS` 与 `create_model`，至少包含 **PLGAFormer (proposed)** 与 **PIT**，并跑通完整实验；  
   - 增加 **requirements.txt** 与 **README** 的安装/运行说明。

2. **P1（强烈建议）**  
   - 统一学习率调度（Cosine + Warmup）与优化器（AdamW）；  
   - 数据与结果路径基于项目根目录，避免依赖当前工作目录；  
   - 在 README 或 DATA.md 中写清数据集与训练/测试划分。

3. **P2（提升质量）**  
   - 消融协议与 SOTA 主表写入简短文档；  
   - 可视化只引用已训练模型，避免硬编码不存在的模型名；  
   - 主表注明指标单位与是否在物理/标准化空间。

4. **P3（长期）**  
   - 拆分 SOTA_comparison.py 为多模块；  
   - 可选：超参搜索与敏感性实验；  
   - REPRODUCIBILITY.md 中记录环境与确定性设置。

按上述顺序推进，可以在不改变方法本身的前提下，显著缩小与顶刊在实验规范与可复现性上的差距。
