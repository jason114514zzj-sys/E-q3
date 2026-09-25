# 提交材料对照表（第三问 · 可解释性多模态情感预测）

> **本表由仓库内实际文件逐字节计算生成，不是手写。**
> 一键核验：`python verify_manifest.py`（逐条校验 `manifest.json` 的字节数与 sha256）。

## 0. 先读：本仓库布局 vs 服务器交付树

本仓库采用**扁平布局**（`results/`、`models/`、`figures/`、`docs/`），服务器交付树采用**分层布局**（`output/`、`work/`、`handoff/`）。两者收纳的是**同一批文件**，已逐字节核对一致。

**为什么会是两种布局**：`src/q3_explain/` 里的代码与 `config.yaml` 是**服务器侧的冻结原件**，它们按服务器路径书写（`output/q3/…`、`work/q3/…`）。因此任何人 clone 本仓库后运行

```bash
export PYTHONPATH=$PWD/src
python -m q3_explain.run_all --epochs 60
```

会在**本仓库根目录**下新建 `output/` 与 `work/` 两棵树；这两棵树的内容应与随仓库附带的 `results/`、`models/`、`figures/`、`docs/` **逐字节一致**（`sha256` 可直接对比）。

> **我们刻意不搬运随仓库附带的冻结件**：提交件 `results/附件4_预测与解释结果.csv` 的 sha256 为 `311d9059b8b76bc450b316da17ad2bda6d2248d67616224a11dd540e56677867`，它被论文引用；搬运会引入字节风险而收益为零。

---

## 1. 段055「可复现核心材料」（问题三部分）

### 1.1 核心代码 / 说明文档 / 模型参数 / 配置文件 / 运行环境

题面原文：*问题2和问题3两类专项模型的核心代码、说明文档、模型参数文件、配置文件与运行环境说明。代码须附带详细运行说明、数据集处理规则与参数配置，确保实验结果可完整复现。*

| 题面依据 | 材料说明 | 服务器交付树路径 | 本仓库路径 | 字节 | sha256(前16) |
|---|---|---|---|---:|---|
| 段055「核心代码」 | 问题三核心代码（7 模块）：`__init__.py`、`run_all.cpython-313.pyc`、`shapley.cpython-313.pyc`、`data.py`、`evidence_time.py`、`model.py`、`model_explain.py`、`run_all.py`、`shapley.py`、`train.py` | `src/q3_explain/` | `src/q3_explain/`（10 文件） | 159844 | — |
| 段055「核心代码」 | 词级强制对齐最小依赖（与问题一解耦）：`__init__.py`、`monotonic_align.py` | `src/q1_alignment/` | `src/q1_alignment/`（2 文件） | 8908 | — |
| 段055「说明文档」 | 解题报告 | `output/evidence/问题三解题报告.md` | `docs/问题三解题报告.md` | 9049 | 489d7e3f93f7d4ac… |
| 段055「说明文档」 | 运行说明 + 数据集处理规则 + 参数配置 | `output/evidence/问题三_运行说明.md` | `docs/问题三_运行说明.md` | 8560 | 930c0cf20d719f81… |
| 段055「说明文档」 | 方案校验报告 | `output/evidence/问题三方案校验报告.md` | `docs/qa/问题三方案校验报告.md` | 35486 | 1eea98c4d24d52a9… |
| 段055「说明文档」 | 方法学说明（本仓库自有文档） | `—` | `method.md` | 8959 | 9aed5f7673679fd5… |
| 段055「模型参数文件」 | 模型权重（state_dict + dims + hidden） | `output/models/q3_model.pt` | `models/q3_model.pt` | 1493704 | deed0e1935c06282… |
| 段055「模型参数文件」 | 训练记录（超参 + 逐轮 history + 验证指标） | `output/models/q3_training.json` | `models/q3_training.json` | 4840 | 69b2182409428cc4… |
| 段055「配置文件」 | 冻结配置（与服务器逐字节同一份） | `config_q3.yaml` | `config.yaml` | 9047 | 714b3c03d7e7b01d… |
| 段055「运行环境说明」 | 服务器环境与两套环境警告 | `SERVER_ENV.md` | `SERVER_ENV.md` | 4643 | 848efcc14af03603… |
| 段055「运行环境说明」 | 依赖清单（V100/sm_70 锁定说明） | `requirements.txt` | `requirements.txt` | 1189 | 66b21665db2839af… |
| 段055「运行环境说明」 | 实测版本（Python/torch/numpy/matplotlib/CUDA） | `versions.txt（整题）` | `runtime_versions.json` | 532 | 59b524a2f7a04a17… |
| 段055「协议」 | 训练-验证-测试协议与评测口径（本仓库自有文档） | `—` | `protocol.json` | 7202 | fcee6b685f1d0464… |

---

## 2. 段056「专项测试结果文件」

### 2.1 专项测试结果 CSV

题面原文：*附件3测试集预测结果CSV文件、附件4测试集预测与解释结果CSV文件，文件命名清晰规范。*

| 题面依据 | 材料说明 | 服务器交付树路径 | 本仓库路径 | 字节 | sha256(前16) |
|---|---|---|---|---:|---|
| 段056「专项测试结果文件」 | **附件4 测试集预测与解释结果 CSV**（20 行 × 8 列） | `output/q3/附件4_预测与解释结果.csv` | `results/附件4_预测与解释结果.csv` | 1723 | 311d9059b8b76bc4… |
| 段056「专项测试结果文件」 | 附件3 预测结果 CSV —— **属第二问，不在本仓库**（见 §5） | `output/q2/附件3_预测结果.csv` | — | — | — |

---

## 3. 段049–053 论文正文所需素材（仓库内已备，正文待写）

### 3.1 论文第 7 章五个小节的对应素材

| 题面依据 | 材料说明 | 服务器交付树路径 | 本仓库路径 | 字节 | sha256(前16) |
|---|---|---|---|---:|---|
| 段050「典型样本解释卡」 | 解释卡（含预测、三模态作用程度、主要参考模态、关键证据定位） | `work/q3/解释卡.md` | `results/解释卡.md` | 4051 | 6e62f458cc7976ab… |
| 段052「全量预测与解释汇总」 | 附件4 全量解释（20 条） | `work/q3/explanations.json` | `results/explanations.json` | 24539 | eba51100247cffbd… |
| 段053「基础性能/错误归因」 | 验证集指标、忠实度、门控-LOMO、时间溯源 | `work/q3/q3_report.json` | `results/q3_report.json` | 14367 | f6b1f6b28ae5b7d7… |
| 段051「局部片段重要性可视化」 | 5 张图 | `output/figures/q3_{attention_profiles,faithfulness,gate_vs_lomo,modality_weights}.png` | `figures/q3_attention_profiles.png` | 125493 | 25ca057446366674… |
| 段051「局部片段重要性可视化」 | 5 张图 | `output/figures/q3_{attention_profiles,faithfulness,gate_vs_lomo,modality_weights}.png` | `figures/q3_faithfulness.png` | 50174 | b74879c7e1a1376b… |
| 段051「局部片段重要性可视化」 | 5 张图 | `output/figures/q3_{attention_profiles,faithfulness,gate_vs_lomo,modality_weights}.png` | `figures/q3_gate_vs_lomo.png` | 48283 | b71d2e48aea1c66f… |
| 段051「局部片段重要性可视化」 | 5 张图 | `output/figures/q3_{attention_profiles,faithfulness,gate_vs_lomo,modality_weights}.png` | `figures/q3_modality_weights.png` | 41752 | 6c9e6655127f9a6b… |
| 段051「局部片段重要性可视化」 | 5 张图 | `output/figures/q3_{attention_profiles,faithfulness,gate_vs_lomo,modality_weights}.png` | `figures/q3_shapley.png` | 87112 | fcc0dfd0959f36b6… |
| 段051「三模态作用差异对比」 | 跨测试集模态重要性对照 | `handoff/模态重要性_跨测试集对照.{md,csv}` | `results/模态重要性_跨测试集对照.csv` | 481 | f1e298e18afac17c… |
| 段051「三模态作用差异对比」 | 跨测试集模态重要性对照 | `handoff/模态重要性_跨测试集对照.{md,csv}` | `results/模态重要性_跨测试集对照.md` | 2882 | 2f72c5c7f47f1f2d… |
| 段029「关键证据定位」 | 20 条词级起止秒（缓存/溯源） | `work/q3/att4_time_map.json` | `results/att4_time_map.json` | 53315 | b1c348cc4a0a052d… |
| 段029「关键证据定位」 | 特征位置 → 词 → 秒 的平面表（564 行） | `handoff/att4_position_time_map/position_time_map.csv` | `results/position_time_map.csv` | 23395 | 5d78a62bc7710634… |
| 段028「模态作用差异」 | **精确 Shapley 分解**（三模态 8 子集穷举，含 Σφ = f(M) − f(∅) 逐样本校验） | `work/q3/shapley_attribution.json` | `results/shapley_attribution.json` | 16967 | 237f24a3ec66ec99… |

---

## 4. 一键复现与核验

```bash
# 0) 依赖：见 requirements.txt；实测版本见 runtime_versions.json
# 1) 核验随仓库附带的冻结件是否完整（逐条字节数 + sha256）
python verify_manifest.py

# 2) 从原始数据重跑全流程（V100 上约 1 分钟；会生成 output/ 与 work/）
export PYTHONPATH=$PWD/src
python -m q3_explain.run_all --epochs 60

# 3) 重打包（自动重建 manifest.json 与 submission_q3.zip，并做 <=50MB 守门）
python package.py
```

**第 2 步与冻结件的对应关系**（跑完后逐字节应相同）：

| 运行产物 | 随仓库附带的冻结件 |
|---|---|
| `output/q3/附件4_预测与解释结果.csv` | `results/附件4_预测与解释结果.csv` |
| `output/models/q3_model.pt` | `models/q3_model.pt` |
| `output/models/q3_training.json` | `models/q3_training.json` |
| `output/figures/q3_*.png` | `figures/q3_*.png` |
| `work/q3/{解释卡.md,explanations.json,q3_report.json,att4_time_map.json}` | `results/` 下的同名件 |

---

## 5. 边界与诚实说明

1. **附件3 的预测结果 CSV 不在本仓库。** 它属第二问（由队友以冻结 BERT 编码器 + 观测门控实现），交付在第二问的代码包中；本仓库只负责第三问。

2. **三问的文本输入接口并不相同**（本问用附件2 官方预计算的 `text`；第二问因附件3 无 `text` 列而改用 `text_bert`）。同一问内的「训练↔推理」接口一致，符合段018；跨问接口差异属已知边界，论文中已交代。

3. **关键证据的秒数来自词级约束式单调强制对齐**（20/20 覆盖率 1.000），不是把时长按 50 等分的均匀切分——均匀切分的偏差均值 1.968 s、最大 3.627 s，已被否决。

4. **附件4 是「同源留出样本」**，不是独立第三方测试集。

5. **附件4 样本 13 的视觉通道在「对齐版」中全零**，与题面段017「三模态信息完整」矛盾；本仓库保留了这一事实（`results/q3_report.json` 内有记录），未做掩饰。

6. **推理不重新训练**：`run_all.py` 会重新训练以获得可复现的确认，但附件4 的推理只使用训练得到的冻结权重，附件4 不参与任何参数或阈值选择。

---

## 6. 段057 合规：身份信息与体积

- **身份信息**：`package.py` 内置硬守门 `assert_no_identity()`，打包前扫描全部收录的文本文件。已知身份串**不写在源码里**，而是从仓库根的 `identity_blocklist.txt` 读取 ——因为 `package.py` 自身也是提交材料，若把身份串写进源码，守门自己就成了泄漏源。命中即**中止打包并返回码 2**。

- 另有「结构性通用模式」单列为**提示**（IPv4 地址、`/home/<账号>/` 路径、`账号@主机`、邮箱、身份字段标签）：交付文档会**合法地引用禁令原文**（如「严禁出现参赛单位…」），一律阻断会造成自伤，故只提示、需人工确认。

- `identity_blocklist.txt` 既不在 `INCLUDE_FILES` 也不在 `INCLUDE_DIRS` 中，**永不进入提交包**，仅供打包守门读取。

- **体积**：`package.py` 对产出的 zip 断言 ≤ 50 MB（竞赛对整题全部附件的限制）。

- **命名**：提交件命名为 `附件4_预测与解释结果.csv`，与题面段056 的表述一致；压缩包内根目录即为本仓库根，解压即可对上本表路径。

- **若与问题一/二的材料合并提交**：建议把本包解压到 `问题三/` 目录下，避免与其它问题的 `src/`、`results/` 等同名目录冲突。

