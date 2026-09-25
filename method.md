# 第三问建模与实验定义

对应题面段028–030。本文件给出数学定义与实验协议，结果见 `README.md` 与 `results/q3_report.json`。

---

## 1. 输入与边界

用附件2 官方训练集学习参数、验证集选模；附件4 只做**冻结后**的推理与解释。

采用 aligned_50 接口，三模态按官方对应的序列位置建模，最大位置数 `T = 50`，其中
**位置 0 为 `[CLS]` 占位、末位为 `[SEP]`，可用内容槽位 `K = 48`**。

设第 m 个模态的输入为 $x_m \in \mathbb{R}^{T \times d_m}$，$m \in \{\text{text},\text{audio},\text{vision}\}$，
$d_{\text{text}}=768,\ d_{\text{audio}}=74,\ d_{\text{vision}}=35$。

**有效性掩码**（因三模态的填充约定不同，必须分开定义）：

- 文本：$p^{\text{text}}_t = \mathbb{1}\!\left[\texttt{text\_bert}[1,t] = 1\right]$
  —— 取自注意力掩码通道。**不能用 $x_t = 0$ 判断**，因为 `text` 的填充值非零。
- 音视频：$p^m_t = \mathbb{1}\!\left[\lnot\,\text{all-zero}(x_{m,t})\right]$。
- 记 $o_{m,t} = p^m_t$，即"该位置是否携带可用观测"。
- 某模态在整条样本上完全无有效位置时记 $\varnothing_m$（如附件4 样本 13 的 vision）。

掩码外位置在送入编码器前一律置零；该约定使"填充 = 0"成立，避免编码器把填充当信号。

## 2. 预处理

- 文本：不做数值标准化（它是预训练表示，各维不是物理量纲）。
- 音视频：**逐维 z-score**，统计量**只用 train 的有效步**估计：

$$
\mu_m^{(j)}=\frac{1}{|S_m|}\sum_{t\in S_m} x^{(j)}_{m,t},\qquad
\sigma_m^{(j)}=\max\!\left(\sqrt{\frac{1}{|S_m|}\sum_{t\in S_m}\big(x_{m,t}^{(j)}-\mu_m^{(j)}\big)^2},\ \epsilon\right)
$$

其中 $S_m=\{(i,t): o_{m,t}=1,\ i\in\text{train}\}$，$\epsilon=10^{-6}$。变换后掩码外位置仍置零。

> 为什么必须逐维：`audio` 第 0 维取值域为 $[0,500]$，其余 73 维为 $\pm 50$，
> 标准差异比达 **394×**。整表统一 z-score 会让 dim0 主导距离与融合表示。

## 3. 每模态独立编码器

$$
h_m = \mathrm{LN}\Big(\mathrm{TransformerEncoder}_1\big(W_m x_m + e_{\text{pos}}\big)\Big),\qquad h_m\in\mathbb{R}^{T\times H}
$$

- $W_m\in\mathbb{R}^{H\times d_m}$，$H=96$；$e_{\text{pos}}\in\mathbb{R}^{1\times T\times H}$ 为可学习位置编码，初始化 $\mathcal{N}(0,0.02^2)$。
- 单层、4 头、`dim_feedforward` $=2H$、`activation=gelu`、`norm_first=True`、dropout 0.1，输出再过一个 `LayerNorm`。
- **每模态必须独立编码**，否则无法分离各模态的贡献，也就无法给出"作用程度"。

## 4. 时间注意力 —— 关键证据定位

$$
s_{m,t}=w_2^{\top}\tanh\!\big(W_1 h_{m,t}\big),\qquad
a_{m,t}=\frac{\exp(s_{m,t})\,\cdot\, o_{m,t}}{\sum_{t'}\exp(s_{m,t'})\,\cdot\, o_{m,t'}}
$$

$$
z_m=\sum_{t=1}^{T} a_{m,t}\, h_{m,t}
$$

等价地：把无效位置的 logits 置 $-\infty$ 后再 softmax。**先 softmax 再置零会破坏归一化**，必须取前置。

- 分布 $a_m\in\Delta^{T-1}$ 即"该模态中与判断密切相关的局部片段"（段028 第三项）。
- 取 $a_m$ 的 top-$k$（本方案 $k=3$）连续位置，经第 8 节映射为秒，即为**关键证据定位**（段029）。
- 若 $\varnothing_m$，则该模态无注意力分布，$z_m\leftarrow \mathbf{0}$。

## 5. 模态门控 —— 作用程度与主要参考模态

$$
u=\big[z_{\text{text}};z_{\text{audio}};z_{\text{vision}}\big]\in\mathbb{R}^{3H},\qquad
\ell = W_4\,\mathrm{GELU}\!\big(W_3 u\big)\in\mathbb{R}^{3}
$$

$$
g_m=\frac{\exp(\ell_m)\cdot \mathbb{1}[\lnot \varnothing_m]}{\sum_{m'}\exp(\ell_{m'})\cdot \mathbb{1}[\lnot \varnothing_{m'}]}
$$

- $g$ 即**模态作用程度**（段028 第一项 / 段029），三项非负且和为 1。
- $\arg\max_m g_m$ 即**主要参考模态**（段028 第二项 / 段029）。
- **不可用模态的 logits 硬置 $-\infty$**：这是**结构约束**而非正则化。
  若不加该约束，模型会给"没有数据"的模态赋权 —— 在附件2 验证集上实测曾有 15 条 vision 整条缺失的样本
  被赋平均 0.2351 的 vision 权重（高于全体平均 0.1745），输出"主要参考模态 = vision"这种荒谬解释。
- 融合表示 $z=\sum_m g_m z_m$，再过 `Dropout(0.1)` 与 `LayerNorm`。

## 6. 双头与目标函数

$$
\hat{p}=\mathrm{softmax}(W_c z + b_c)\in\Delta^{2},\qquad
\hat{y}=W_r z + b_r\in\mathbb{R}
$$

$$
\mathcal{L}=\underbrace{-\sum_{c=1}^{3} \omega_c\, y_c \log \hat{p}_c}_{\text{CE}_w}
\;+\;\lambda\,\underbrace{(\hat{y}-y)^2}_{\text{MSE}},\qquad \lambda=1.0
$$

$\omega_c \propto 1/n_c$ 为**逆频率类权重**（$n_c$ 为 train 上第 $c$ 类样本数），
用于抵消中性类样本偏少（train 正向占 49.2%）带来的偏置。

## 7. 训练与选模

- 优化器 AdamW，$\text{lr}=1.5\times10^{-3}$，weight decay $10^{-4}$，`CosineAnnealingLR(T_max=epochs)`。
- batch 128，最多 60 轮，**按验证集 Macro-F1 选最佳 epoch**，连续 12 轮无改善即早停（实测最佳第 4 轮，第 16 轮停）。
- **决策阈值固定为 argmax**（等价于阈值为 0 的对称判决）：本方案不存在需要调优的阈值，
  因而没有阈值过拟合的空间 —— 这是对段031"在验证集上选择决策阈值"的一种更强的满足。
- 随机种子固定为 `20260924`。

> 实验性正则：训练时以概率 0.1 随机屏蔽某一模态（`modality_dropout`）。
> 它与第 5 节的 $-\infty$ 硬约束是**两件不同的事**：前者提高鲁棒性，后者保证解释不自相矛盾。

## 8. 关键证据的时间定位（段029 末句 / 段017）

模型只给出**位置序号**，而段029 要求证据可对应至"原始文本片段、语音时段或视觉关键帧"，
因此必须把位置映射到真实时间。

$$
\text{位置 } j \;\longrightarrow\; \text{词 } w(j) \;\longrightarrow\; [\text{start}_{w},\ \text{end}_{w}]
$$

**(a) 词级时间由约束式单调强制对齐给出**：对附件4 的 mp4 用 ffmpeg 解码为 16 kHz 单声道，
以 10 ms 一跳求 RMS 能量包络，再用 `raw_text` 的逐词序列求解边界 $b_0<\dots<b_N$，约束为

$$
b_0=0,\quad b_N=D,\quad b_{i}<b_{i+1},\quad \frac{N}{D}\in[1,6]\ \text{词/秒},
$$

并让边界在能量谷附近吸附（停顿感知）。实测 20/20 条覆盖率 1.000、语速 1.80–4.01 词/秒。

**(b) 位置→子词→词为精确映射（v1.1.0）**：用 bert-base-uncased 词表（$\texttt{vocab/bert-base-uncased\_vocab.txt}$）
重建转写的 WordPiece 序列，与数据自带 `text_bert[0]` 逐位比对（附件4 20/20、附件2 抽样 400/400 一致），
于是位置 $j$ 对应哪个子词、该子词属于哪个词都有确定答案；旧版的比例式
$w(j)=\mathrm{round}((j-1)/n\cdot N)$ 仅作为词表缺失时的回退。

**(c) 禁止均匀切分**。曾误用 $j\cdot D/50$，实测该式**根本不成立**：

> 这 50 个位置是**词片段轴**而不是等长时间片。20/20 条满足
> 「`text_bert` 掩码长度 = 音频有效位置数 + 2」（偏移恒为 2，corr = 1.000），
> 音频有效位置数在 11~48 间变化而非恒为 50，上限 48 与内容槽位 $K=48$ 吻合。
> 两种做法的差异为**均值 1.587 s、最大 3.595 s**；
> 最差样本 15 号把真实的 5.680 s 指成 2.053 s，**偏差 2.77 倍** —— 点开视频一眼即可看出错误。

## 9. 解释的可复核性（段028「可量化、可复核」）

注意力本身不是解释。因此附加两项**行为学检验**：

**(A) 三组忠实度（遮蔽检验）**：分别遮蔽 $a_m$ 最高的 3 个位置（峰值组）、最低的 3 个位置
（低谷组，与峰值组不相交）与随机的 3 个**有效**位置（随机组，重复 5 次），比较预测强度的变化量。
若注意力峰值确实重要，应满足「峰值组 > 随机组 > 低谷组」的严格三段序。
实测只有 text 满足（73.9%；vision 24.0%、audio 27.5%），且 vision/audio 的低谷组略高于随机组，置换检验（把注意力随机重排构造经验零分布）给出 text 的单侧 p=0.0002、audio 0.0012、vision 0.1362，其中 audio 的显著来自极小的绝对效应量（峰值−随机仅 0.0051，为 text 的约 1/43），
故本文**只对 text 主张片段级解释效力**，vision/audio 只作参考展示。

**(B) 留一模态扰动（LOMO）**：整条移除某模态（等价于置 $\varnothing_m$ 并重新归一化门控），
测量预测变化量。**门控是模型的自我报告，LOMO 是实测行为**；两者不一致时如实标注而非掩饰。

## 10. 输出约定与边界

**输出**：极性、强度、主要参考模态、模态作用程度、关键证据定位（秒 + 模态）。

**边界**：

1. 证据时间的秒级定位来自**自建词级强制对齐**，不是官方时间戳；子词→词为**精确映射**（bert-base-uncased 词表重建 token 序列并与数据自带 `text_bert[0]` 逐位比对通过），但词自身的起止时间仍是估计量，交付中表述为「近似定位、需回看确认」。
2. 附件4 与附件2 的 test 划分存在内容重叠（5/20 逐值一致），属**同源留出样本**而非独立测试集；
   落在 valid 的为 0 条，选模无泄漏。
3. 附件4 样本 13 的 vision 整条为零，与题面段017「三模态信息完整」**矛盾**；门控硬约束已处理。
4. 验证集参与早停与最佳 epoch 选择，故报告的是**开发评价**。
5. 本方案已用 **5 个随机种子**重跑并报告均值±标准差（验证集 Macro-F1 0.6117±0.0093）；模态作用的**视听两路相对次序在 5 个种子中有 1 次反转**，故只主张文本明显高于另两路。
6. 附件4 无标签，不能报告任何需要真值的指标。
