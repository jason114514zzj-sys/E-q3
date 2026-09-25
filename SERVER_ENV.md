# ==========================================================
# E题 多模态情感识别 —— 实验环境说明
# 实验服务器: Linux 工作站（Ubuntu 24.04.5 LTS / x86_64）
# 搭建日期: 2026-09-23      最后更新: 2026-09-25
# ==========================================================
# 按竞赛「结果与提交说明」要求，提交材料中不出现参赛单位、
# 队员姓名、队伍编号等身份信息。本文件中的主机名、IP 地址与
# 账号名字段均已隐去；除这些字段外，环境信息未作任何删减。
# ==========================================================

## 一、环境信息

| 项目 | 值 |
|---|---|
| 服务器 | Linux 工作站（x86_64）；主机名与 IP 已隐去 |
| 系统 | Ubuntu 24.04.5 LTS, 内核 7.0.0-31 |
| CPU / 内存 | 48 核 / 125 GB |
| 磁盘 | /dev/sda2 1.5T (已用 131G, 可用 1.3T) |
| GPU | Tesla V100-PCIE-16GB, 16GB 显存, **compute capability 7.0 (sm_70)** |
| 驱动 | 580.178.04 (nvidia-smi 显示 CUDA 13.0，但系统无 nvcc) |
| Conda | miniconda3, conda 26.3.2 |
| **项目环境** | **conda env `mosei`** (Python 3.10.21) |
| 项目目录 | `<项目根>/`（绝对路径中的用户名段已隐去） |

## 二、环境激活

```bash
conda activate mosei
```

## 三、⚠️ 关键技术约束

### 1. V100 必须用 PyTorch ≤ 2.4.x
V100 = Volta = sm_70。PyTorch 2.5+ 官方 wheel **已移除 sm_70 内核**，
使用会报：
```
CUDA error: no kernel image is available for execution on the device
```
当前安装 **torch 2.4.1+cu121**，已验证 `sm_70` 在内核列表中。

### 2. 安装源必须绕开 pypi.nvidia.com
实验服务器**无法访问 `pypi.nvidia.com`**（连接超时）。
PyTorch 官方源 `download.pytorch.org` 会从该域名拉取 CUDA 依赖 → 必然卡死。

**解决**：使用清华源安装（已同步完整 CUDA 轮子）：
```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```
不要加 `--index-url https://download.pytorch.org/whl/cu121`。

### 3. 长任务必须用 setsid
SSH 断开时普通 `nohup ... &` 仍可能被杀：
```bash
setsid nohup bash script.sh > log.txt 2>&1 < /dev/null &
```

## 四、验证结果

```
torch      : 2.4.1+cu121
cuda avail : True
device     : Tesla V100-PCIE-16GB
capability : (7, 0)
arch list  : ['sm_50','sm_60','sm_70','sm_75','sm_80','sm_86','sm_90']
cudnn      : 90100
matmul 4000x4000 x10 : 0.134 s
backward OK, grad norm: 714.07
SMOKE TEST PASSED
```

## 五、目录结构

```
<项目根>/
├── src/              源代码
├── data/             数据（附件3/4 等）
├── output/
│   ├── figures/      论文图表
│   └── models/       模型权重
├── notebooks/        探索性分析
├── logs/             运行日志
└── scripts/          工具脚本
```

## 六、问题三（可解释性情感预测）运行环境与验证

问题三与本文件所述环境**同一套**（conda `mosei` / Python 3.10.21 / torch 2.4.1+cu121 / V100 sm_70）。

```bash
conda activate mosei
export PYTHONPATH=$PWD/src          # 必需：src/q3_explain 是顶层包
python -m q3_explain.run_all --epochs 60      # V100 上约 1 分钟
```

**实测验证（2026-09-25）**

| 项 | 值 |
|---|---|
| 运行设备 | `cuda`（Tesla V100-PCIE-16GB） |
| 最佳 epoch / 早停 | 8 / 第 20 轮 |
| 验证集指标 | Accuracy 0.6442 · Macro-F1 0.6118 · MAE 0.6074 · Pearson 0.6750（q3 v1.1.0 重训后） |
| 训练耗时 | 约 1 分钟（60 轮上限，实际 20 轮早停） |
| 附带的 ffmpeg / ffprobe | conda 环境内 `mosei/bin/`（由 `imageio-ffmpeg` / `opencv-python-headless` 提供），用于附件4 的 16 kHz 解码与真实时长探测 |
| 中文字体 | `Noto Sans CJK JP`（缺失时 matplotlib 只发 Warning，图内中文会静默变成方框） |

**详细运行说明、数据集处理规则与参数配置**见 `docs/问题三_运行说明.md` 与 `config.yaml`。

> ⚠️ **本项目是两套计算环境**：问题一、问题三在**本实验服务器（V100 + torch 2.4.1+cu121）**完成；
> 问题二在**另一台机器（RTX 4090 + Python 3.13 + torch 2.10.0+cu130）**完成。
> 论文的可复现性说明必须分别交代，否则复现步骤不成立。

---

## 七、数据与材料边界

- 赛题原始数据（附件1–4）**只读**，派生结果一律写入 `output/`。
- 全流程只使用赛题提供的 CMU-MOSEI 系列数据，**未引入任何外部情感数据集**。
- 附件4 只在模型结构与决策规则**冻结之后**用于推理，不参与任何参数或阈值选择。
