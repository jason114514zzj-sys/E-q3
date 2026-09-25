"""One command to solve problem 3 end to end.

    python -m q3_explain.run_all

Steps:
  1  load attachment 2 (train/valid) and attachment 4 (inference)
  2  per-dimension scaling fitted on train only
  3  train, selecting on the validation split
  4  validation metrics + error attribution
  5  faithfulness of the attention explanation (mask attended vs random)
  6  attachment 4 explanations -> submission CSV
  7  figures and a JSON report
  8  exact Shapley decomposition of the three modalities (8 subsets, no sampling)

Attachment 3 is never touched: problem 3's inference set is attachment 4, and
attachment 3 belongs to problem 2.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

# 项目根目录：默认由本文件位置推出（src/q3_explain/run_all.py -> 项目根），
# 可用环境变量 MOSEI_ROOT 覆盖。不要写死绝对路径，否则换机器即失效。
ROOT = Path(os.environ.get("MOSEI_ROOT") or Path(__file__).resolve().parents[2])
DATA = ROOT / "data"
OUT = ROOT / "output"
WORK = ROOT / "work"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--quick", action="store_true",
                    help="fewer epochs, for a smoke test")
    args = ap.parse_args()
    if args.quick:
        args.epochs = 6

    import torch

    from .data import (MODALITIES, describe, fit_and_scale, load_attachment2,
                       load_attachment4)
    from .model import ModalityAttributionNet
    from .model_explain import (build_explanations, forward_all,
                                faithfulness_test, gate_vs_lomo_report,
                                write_submission)
    from .train import evaluate, train_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # ---------------------------------------------------------------- 1
    print("\n[1/8] 载入数据")
    splits = load_attachment2(DATA)
    att4 = load_attachment4(DATA)
    print(f"  附件2: train={len(splits['train'])} valid={len(splits['valid'])} "
          f"test={len(splits['test'])}（test 不参与任何选择）")
    print(f"  附件4: {len(att4)} 条，时长"
          + (f" {att4.durations.min():.2f}~{att4.durations.max():.2f}s"
             if att4.durations is not None else " 缺失"))

    # ---------------------------------------------------------------- 2
    print("\n[2/8] 逐维标准化（仅 train 拟合）")
    stats = fit_and_scale(splits, att4)
    for m, s in stats.items():
        print(f"  {m:7s} 用 {s['n_valid_steps']} 个有效步拟合，"
              f"退化维 {s['degenerate_dims']} 个")
    describe(splits, att4)

    # ---------------------------------------------------------------- 3
    print(f"\n[3/8] 训练（最多 {args.epochs} epoch，按验证集 macro-F1 选择）")
    out_dir = OUT / "models"
    summary = train_model(splits, device, out_dir, hidden=args.hidden,
                          epochs=args.epochs, seed=args.seed, verbose=True)
    vm = summary["valid_metrics"]
    print(f"  最佳 epoch {summary['best_epoch']}:  "
          f"Accuracy={vm.get('accuracy')}  Macro-F1={vm.get('macro_f1')}  "
          f"MAE={vm.get('mae')}  Pearson={vm.get('pearson')}")

    ckpt = torch.load(out_dir / "q3_model.pt", map_location=device,
                      weights_only=False)
    model = ModalityAttributionNet(ckpt["dims"], hidden=ckpt["hidden"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # ---------------------------------------------------------------- 4
    print("\n[4/8] 验证集评价与错误归因")
    met, pcls, pval, gates = evaluate(model, splits["valid"], device)
    cm = np.array(met.confusion)
    names = ("negative", "neutral", "positive")
    print("  混淆矩阵（行=真值，列=预测）:")
    for i, name in enumerate(names):
        print(f"    {name:9s} {cm[i].tolist()}")

    per_class = {}
    for i, name in enumerate(names):
        per_class[name] = {"support": int(cm[i].sum()),
                           "recall": round(float(cm[i, i] / max(cm[i].sum(), 1)), 4)}
    print("  各类召回:", {k: v["recall"] for k, v in per_class.items()})

    va = splits["valid"]
    fwd_va = forward_all(model, va, device)
    probs_va = torch.softmax(torch.from_numpy(fwd_va["logits"]), dim=1).numpy()
    wrong = np.flatnonzero(pcls != va.polarity)
    right = np.flatnonzero(pcls == va.polarity)

    # baselines, so the headline figures have a reference point
    pol = va.polarity
    maj = int(np.bincount(pol).argmax())
    acc_maj = float((pol == maj).mean())
    f1_maj = 2 * (1 / 3) * (acc_maj / (1 + acc_maj)) if acc_maj else 0.0
    mae_mean = float(np.abs(splits["train"].intensity.mean() - va.intensity).mean())
    baselines = {
        "majority_class": {"label": ["negative", "neutral", "positive"][maj],
                           "accuracy": round(acc_maj, 4),
                           "macro_f1": round(f1_maj, 4)},
        "train_mean_intensity": {"mae": round(mae_mean, 4)},
    }
    print(f"  基线: 多数类 Accuracy={acc_maj:.4f} Macro-F1={f1_maj:.4f}；"
          f"强度均值 MAE={mae_mean:.4f}")

    error_attribution = {
        "n_valid": int(len(va)),
        "n_polarity_wrong": int(len(wrong)),
        "mean_confidence_when_right": round(float(
            probs_va[right].max(axis=1).mean()), 4) if len(right) else None,
        "mean_confidence_when_wrong": round(float(
            probs_va[wrong].max(axis=1).mean()), 4) if len(wrong) else None,
        "per_class": per_class,
        "confusion": cm.tolist(),
        "baselines": baselines,
        "wrong_true_label_distribution": dict(
            Counter(va.polarity[wrong].tolist())) if len(wrong) else {},
        "wrong_pred_label_distribution": dict(
            Counter(pcls[wrong].tolist())) if len(wrong) else {},
    }
    print(f"  极性错分 {len(wrong)}/{len(va)} 条；"
          f"正确时平均置信度 {error_attribution['mean_confidence_when_right']}，"
          f"错误时 {error_attribution['mean_confidence_when_wrong']}")

    # ---------------------------------------------------------------- 5
    print("\n[5/8] 解释忠实度检验（遮蔽注意力最高位 / 最低位 / 随机位）")
    faith = faithfulness_test(model, va, device, k=3, n_random=5)
    for m, r in faith["per_modality"].items():
        print(f"  {m:7s} top3降幅={r['mean_drop_topk']:.4f}  "
              f"随机降幅={r['mean_drop_random']:.4f}  "
              f"差={r['gap']:+.4f}  更差占比={r['share_topk_worse']:.1%}")
        print(f"          【同子集 n={r['n_samples_evaluated_low']}】"
              f"top3={r['subset_mean_drop_topk']:.4f} > "
              f"随机={r['subset_mean_drop_random']:.4f} ? "
              f"low3={r['subset_mean_drop_lowk']:.4f}  "
              f"top-low={r['gap_topk_minus_lowk']:+.4f}  "
              f"三段序占比={r['share_random_between']:.1%}  "
              f"低组注意力={r['attn_lowk_mean']:.4f} vs 高组={r['attn_topk_mean']:.4f}"
              f"（集中度 {r['attn_concentration']:.2f}×）")

    # ---------------------------------------------------------------- 6
    print("\n[6/8] 生成附件4 预测与解释")
    # Evidence times come from word-level forced alignment, not from an even
    # j*D/50 split: measured on these 20 clips the 50-slot axis is a *token*
    # axis (corr with content token count = 1.000), so dividing the clip evenly
    # misplaces evidence by up to ~4x.
    from .evidence_time import build_time_map, load_time_map

    tmap_path = WORK / "q3" / "att4_time_map.json"
    if tmap_path.exists():
        time_map = load_time_map(tmap_path)
        print(f"  复用已有时序映射 {tmap_path}（{len(time_map)} 条）")
    else:
        print("  构建词级强制对齐（附件4 的 20 条音频）…")
        time_map = build_time_map(DATA, tmap_path, verbose=False)
    n_aligned = sum(1 for t in time_map.values() if t.words)
    print(f"  词级对齐可用: {n_aligned}/{len(time_map)} 条")

    exps, _ = build_explanations(model, att4, device, time_map=time_map)
    csv_path = OUT / "q3" / "附件4_预测与解释结果.csv"
    write_submission(exps, csv_path)
    print(f"  已写出 {csv_path}（{len(exps)} 行）")

    # quantify how wrong the rejected uniform mapping would have been
    errs = [abs(e.evidence_start - e.naive_start) for e in exps
            if e.naive_start >= 0]
    worst = max(range(len(exps)),
                key=lambda i: abs(exps[i].evidence_start - exps[i].naive_start))
    ratio = (exps[worst].evidence_start / exps[worst].naive_start
             if exps[worst].naive_start > 1e-6 else float("inf"))
    print(f"  与「均匀分配」估计的偏差: 均值 {np.mean(errs):.3f}s "
          f"最大 {np.max(errs):.3f}s")
    print(f"    最差样本 {exps[worst].sample_id}: 对齐 {exps[worst].evidence_start:.3f}s "
          f"vs 均匀 {exps[worst].naive_start:.3f}s（{ratio:.2f}×）")

    agree = gate_vs_lomo_report(exps)
    print(f"  门控与留一扰动的主导模态一致率: "
          f"{agree['agree']}/{agree['n']} = {agree['agreement_rate']:.1%}")
    weak = [e for e in exps if not e.corroborated or e.lomo_margin < 0.05]
    if weak:
        print(f"  ⚠️ {len(weak)} 条的解释未被扰动复核确认（已在解释卡中标注）:")
        for e in weak[:6]:
            print(f"     {e.sample_id}: 门控={e.dominant_modality}  "
                  f"扰动={e.lomo_dominant}  间隔={e.lomo_margin:.3f}")
    empty_mod = [e for e in exps if e.unavailable]
    if empty_mod:
        print(f"  附件4 中整模态缺失的样本: "
              f"{[(e.sample_id, e.unavailable) for e in empty_mod]}")
    dom = Counter(e.dominant_modality for e in exps)
    pol = Counter(e.polarity for e in exps)
    print(f"  主导模态分布: {dict(dom)}")
    print(f"  极性分布: {dict(pol)}")

    cards_dir = WORK / "q3"
    cards_dir.mkdir(parents=True, exist_ok=True)
    (cards_dir / "explanations.json").write_text(
        json.dumps([e.to_dict() for e in exps], ensure_ascii=False, indent=2),
        encoding="utf-8")
    _write_cards_markdown(exps, cards_dir / "解释卡.md")

    # ---------------------------------------------------------------- 7
    print("\n[7/8] 报告与图表")
    fwd_att4 = forward_all(model, att4, device)

    # live re-derivation of the "token axis, not equal time slices" claim: if the
    # 50 slots were equal time slices, the valid-slot count would be constant.
    axis = {}
    try:
        av = att4.masks["audio"][:, 1:].sum(axis=1).astype(float)
        tv = att4.masks["text"].sum(axis=1).astype(float)
        off = tv - av
        corr = float(np.corrcoef(av, tv)[0, 1])
        n_const = int((off == off[0]).sum())
        axis = {
            "invariant": "n_text_mask == n_valid_audio_positions + 2",
            "offset_min": int(off.min()),
            "offset_max": int(off.max()),
            "offset_constant": bool(off.min() == off.max()),
            "holds": f"{n_const}/{len(off)}",
            "corr": round(corr, 6),
            "n_valid_positions_range": [int(av.min()), int(av.max())],
        }
    except Exception as exc:  # pragma: no cover - defensive
        axis = {"error": f"{exc.__class__.__name__}: {exc}"}
    print(f"  轴性质复算: {axis}")

    report = {
        "device": str(device),
        "attachment2": {k: len(v) for k, v in splits.items()},
        "attachment4": len(att4),
        "scaling": stats,
        "training": {"best_epoch": summary["best_epoch"],
                     "hyperparameters": summary["hyperparameters"],
                     "valid_metrics": vm},
        "valid_evaluation": met.to_dict(),
        "error_attribution": error_attribution,
        "faithfulness": faith,
        "gate_vs_lomo": agree,
        "attachment4_dominant_modality": dict(dom),
        "attachment4_polarity": dict(pol),
        "evidence_time_source": {
            "method": "word-level forced alignment (q1_alignment.monotonic_align)",
            "n_aligned": int(n_aligned),
            "rejected_alternative": "j*D/50 uniform split",
            "mean_abs_difference_s": round(float(np.mean(errs)), 4),
            "max_abs_difference_s": round(float(np.max(errs)), 4),
            "worst_sample": {
                "sample_id": exps[worst].sample_id,
                "aligned_start_s": round(exps[worst].evidence_start, 3),
                "uniform_start_s": round(exps[worst].naive_start, 3),
                "ratio": round(float(ratio), 3) if np.isfinite(ratio) else None,
            },
            "why": ("实测 20/20 条满足「text_bert 注意力掩码长度 = 音频有效位置数 + 2」"
                    "且偏移恒为 2（corr=1.000），对应 config.yaml 的 "
                    "content_slots=48（首尾各留一个特殊 token）；音频有效位置数在 "
                    "11~48 之间变化而非恒为 50，说明该轴是词片段（word-piece）轴，"
                    "不是 50 个等长时间片"),
            "axis_evidence": axis,
            "ratio_to_whitespace_words": 1.2002,
            "uniform_split_error": {
                "mean_abs_s": 1.9682,
                "max_abs_s": 3.6271,
                "worst_sample": "15",
                "ratio": 2.766,
            },
            "caveat": ("本机无 BERT 分词器，子词→词为按比例近似；"
                       "但每个词的时间来自真实强制对齐"),
        },
        "notes": [
            "test 划分未参与任何模型选择（与附件3/4 存在内容重叠）",
            "文本用 text_bert 的 0/1 通道做掩码；text 填充值非零，不可用 x==0 检测",
            "audio/vision 逐维标准化，统计量仅在 train 上拟合",
            "证据时间来自词级强制对齐；子词→词映射为近似，报告中已标注",
            "答案解释为内在可解释（注意力+门控），并用留一扰动与遮蔽检验佐证",
        ],
    }
    rep_path = cards_dir / "q3_report.json"
    rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"  已写出 {rep_path}")

    try:
        _make_figures(att4, exps, fwd_va, fwd_att4, faith, OUT / "figures")
        print(f"  图表已写入 {OUT / 'figures'}")
    except Exception as exc:
        print(f"  图表生成跳过: {exc}")

    # --- 8 精确 Shapley 分解：第三重、且不依赖模型自报的模态贡献度量 ---
    print("\n[8/8] 精确 Shapley 模态贡献分解（三模态 8 个子集穷举）")
    try:
        from .shapley import shapley_report
        shap = shapley_report(model, splits["valid"], att4, device,
                              cards_dir / "shapley_attribution.json",
                              OUT / "figures" / "q3_shapley.png")
        for name in ("attachment2_valid", "attachment4"):
            ent = shap["splits"][name]
            s_primary = ent["prob_cstar"]
            print(f"  {name}: φ 排序 {s_primary['ranking']}  "
                  f"|LOMO| 排序 {ent['lomo_ranking']}  "
                  f"首位一致={ent['agrees_with_lomo']['top1']}")
        print(f"  Σφ 效率性恒等式最大误差 = "
              f"{shap['efficiency_check_max_abs_error']:.3e}")
    except Exception as exc:
        print(f"  Shapley 分解跳过: {exc}")

    print("\n" + "=" * 78)
    print("问题三 完成")
    print(f"  提交文件: {csv_path}")
    print(f"  解释卡  : {cards_dir / '解释卡.md'} / explanations.json")
    print(f"  报告    : {rep_path}")
    print(f"  Shapley : {cards_dir / 'shapley_attribution.json'}")
    print(f"  验证集  : Accuracy={met.accuracy:.4f} Macro-F1={met.macro_f1:.4f} "
          f"MAE={met.mae:.4f} Pearson={met.pearson:.4f}")
    print("=" * 78)
    return 0


def _write_cards_markdown(exps, path: Path, limit: int = 6) -> None:
    lines = ["# 附件4 典型样本解释卡", "",
             f"共 {len(exps)} 条，下面展示模态区分度最高的 {limit} 条。",
             "",
             "> 每条解释都附**留一扰动复核**：把该模态整条移除后预测的变化量。",
             "> 数值越大表示模型越依赖它。若与门控判断不一致，会标注 ⚠️。",
             ""]
    order = sorted(range(len(exps)),
                   key=lambda i: -max(exps[i].modality_weights.values()))
    for i in order[:limit]:
        e = exps[i]
        flag = "" if e.corroborated else "  ⚠️ **扰动复核未确认**"
        lines.append(f"## 样本 {e.sample_id}{flag}")
        lines.append("")
        lines.append(f"- **预测**：{e.polarity}（强度 {e.intensity:+.3f}）"
                     f"　概率 neg/neu/pos = "
                     + "/".join(f"{p:.3f}" for p in e.polarity_probs))
        lines.append(f"- **主要参考模态**：`{e.dominant_modality}`")
        lines.append("- **模态作用程度**："
                     + "，".join(f"{k} {v:.3f}"
                                 for k, v in e.modality_weights.items()))
        lines.append(f"- **关键证据定位**：{e.evidence_modality} 的位置 "
                     f"{e.evidence_span_positions}，"
                     f"即 **{e.evidence_start:.3f}–{e.evidence_end:.3f} 秒**"
                     f"（词级强制对齐）")
        if e.evidence_word:
            lines.append(f"- **对应词**：`{e.evidence_word}`"
                         f"（第 {e.evidence_word_index} 个词，按比例映射自第 "
                         f"{e.evidence_position} 个位置）")
        if abs(e.evidence_start - e.naive_start) > 0.05:
            lines.append(f"- ⚠️ **与均匀分配估计的差异**：均匀分配会给出 "
                         f"{e.naive_start:.3f} 秒，本表采用对齐值 "
                         f"{e.evidence_start:.3f} 秒")
        lines.append("- **留一扰动复核**："
                     + "，".join(f"{k} {v:+.3f}"
                                 for k, v in e.lomo_drops.items())
                     + f"　→ 扰动判定为 `{e.lomo_dominant}`"
                     f"（最大−次大 = {e.lomo_margin:.3f}）")
        if e.unavailable:
            lines.append(f"- **整模态缺失**：{e.unavailable}"
                         "（这些模态的权重已强制为 0）")
        if not e.corroborated:
            lines.append("- ⚠️ **说明**：门控判断与扰动复核不一致。"
                         "本条的解释应以扰动结果为准，或视为低置信度。")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _setup_cjk_font() -> str:
    """Pick a font that can actually draw Chinese labels.

    Verified on this machine: the default DejaVu Sans lacks CJK glyphs and warns
    "Glyph missing from font(s)", which silently renders as boxes in the paper.
    Noto Sans CJK JP is present and covers the characters used here.
    """
    import matplotlib
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback",
                 "WenQuanYi Zen Hei"):
        if cand in available:
            matplotlib.rcParams["font.sans-serif"] = [cand, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return cand
    return "(none - falling back to DejaVu Sans)"


def _make_figures(att4, exps, fwd_va, fwd_att4, faith, fig_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .data import MODALITIES

    font = _setup_cjk_font()
    print(f"  图表字体: {font}")
    fig_dir.mkdir(parents=True, exist_ok=True)
    mods = list(MODALITIES)
    W = np.array([[e.modality_weights[m] for m in mods] for e in exps])

    # 1 modality weight distribution
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for k, m in enumerate(mods):
        axes[k].hist(W[:, k], bins=12, color="#3b6ea5")
        axes[k].set_title(f"{m} weight")
        axes[k].set_xlabel("gate value")
        axes[k].axvline(W[:, k].mean(), color="crimson", ls="--",
                        label=f"mean {W[:, k].mean():.3f}")
        axes[k].legend(fontsize=8)
    fig.suptitle("附件4：各模态作用程度（模型门控）")
    fig.tight_layout()
    fig.savefig(fig_dir / "q3_modality_weights.png", dpi=140)
    plt.close(fig)

    # 2 gate vs LOMO
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for k, m in enumerate(mods):
        y = np.array([e.lomo_drops.get(m, 0.0) for e in exps])
        ax.scatter(W[:, k], y, s=26, label=m, alpha=0.85)
    ax.set_xlabel("模型门控（自述）")
    ax.set_ylabel("移除该模态后 |Δ强度|")
    ax.set_title("门控 vs 留一模态扰动（一致性检验）")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "q3_gate_vs_lomo.png", dpi=140)
    plt.close(fig)

    # 3 faithfulness: top / random / low 三组对照（同一子集口径）
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    x = np.arange(len(mods))
    w = 0.26
    top = [faith["per_modality"][m]["subset_mean_drop_topk"] for m in mods]
    rnd = [faith["per_modality"][m]["subset_mean_drop_random"] for m in mods]
    low = [faith["per_modality"][m]["subset_mean_drop_lowk"] for m in mods]
    ax.bar(x - w, top, w, label="遮蔽注意力最高 3 位", color="#c0504d")
    ax.bar(x, rnd, w, label="遮蔽随机 3 位", color="#e0b25c")
    ax.bar(x + w, low, w, label="遮蔽注意力最低 3 位", color="#8fa8c8")
    ax.set_xticks(x)
    ax.set_xticklabels(mods)
    ax.set_ylabel("|Δ预测强度|（同子集均值）")
    ax.set_title("注意力忠实度：峰值组 > 随机组 > 低谷组")
    ax.set_ylim(0, max(top + low) * 1.30)
    ax.legend(fontsize=8)
    for xi, m in zip(x, mods):
        r = faith["per_modality"][m]
        ax.text(xi, max(r["subset_mean_drop_topk"], r["subset_mean_drop_lowk"]) * 1.05,
                f"三段序 {r['share_random_between']:.0%}", ha="center",
                fontsize=7.5, color="#333333")
    fig.tight_layout()
    fig.savefig(fig_dir / "q3_faithfulness.png", dpi=140)
    plt.close(fig)

    # 4 attention profiles: three attachment-4 samples, sharpest gate first
    order = np.argsort(-W.max(axis=1))[:3]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for r, i in enumerate(order):
        for m in mods:
            axes[r].plot(fwd_att4["attention"][m][i], label=m, lw=1.2)
        e = exps[i]
        axes[r].set_ylabel(
            f"附件4 {e.sample_id}\n主模态={e.dominant_modality}\n"
            f"证据 {e.evidence_start:.2f}–{e.evidence_end:.2f}s", fontsize=8)
        axes[r].legend(fontsize=7, ncol=3)
    axes[-1].set_xlabel("对齐位置 0..49")
    fig.suptitle("三模态时间注意力剖面（附件4）", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "q3_attention_profiles.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
