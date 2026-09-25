"""Exact Shapley decomposition of the three modalities (problem 3).

Why exact
---------
Three modalities give only 2**3 = 8 subsets, so the Shapley value can be
enumerated rather than sampled.  The numbers below are the definition itself,
not an estimate of it -- there is no Monte-Carlo error to report.

The identity that makes it checkable
------------------------------------
    sum_m phi_m  ==  f(M) - f(empty)
holds exactly for any f and any reference point.  The pipeline recomputes it on
every sample and records the maximum absolute residual, so the decomposition is
a verifiable claim rather than an asserted one.  This is what "可量化、可复核"
means in practice.

Which target to attribute
-------------------------
* ``prob_cstar`` (primary) -- the probability the model gives to the class it
  chose from the full input.  Non-negative, so phi reads as "how much each
  modality contributed to the evidence for this decision".
* ``intensity`` (secondary) -- the regression output.  Intensity is *signed*,
  so a modality that pushes the prediction downward earns a negative phi.  On
  this data phi_text averages -0.138 on attachment 4 (median +0.218): the mean
  is dragged down by the negative-emotion samples.  A signed target therefore
  measures *directional push*, not importance, and must not be used to rank
  modalities.  This is exactly why the leave-one-out perturbation uses |delta|;
  both are reported so the difference is visible rather than hidden.

Reference point
---------------
f(empty) is the model's output with all three modalities masked out.
For the intensity target that value is a per-sample constant (the fused vector
collapses onto the LayerNorm bias) and the code verifies it.  For the
probability target the masked *logits* are constant but each sample selects a
different class, so f(empty) varies with the selected class -- expected, and
reported as such.  Either way f(empty) is a *reference point for attribution*,
not the "all three modalities missing" prediction rule of section 3.5.

Honest caveat
-------------
Any attribution needs a reference point and phi depends on which one is taken.
The efficiency identity survives that choice; the individual phi values do not.
That is why phi is reported next to the reference-independent leave-one-out
perturbation and the masking test, and why the agreement between them is stated
instead of assumed.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

from .data import MODALITIES

MODS = tuple(MODALITIES)                       # ("text", "audio", "vision")
_ORDER = {m: i for i, m in enumerate(MODS)}
SUBSET_ORDER: tuple[tuple[str, ...], ...] = tuple(
    tuple(c) for r in range(len(MODS) + 1) for c in itertools.combinations(MODS, r)
)
# Shapley weights for 3 players: |S|=0 -> 1/3, |S|=1 -> 1/6, |S|=2 -> 1/3
SHAPLEY_W = {0: 1.0 / 3.0, 1: 1.0 / 6.0, 2: 1.0 / 3.0}

PRIMARY_TARGET = "prob_cstar"
SECONDARY_TARGET = "intensity"
# Two modalities whose |phi| differ by less than this fraction of the largest
# are reported as indistinguishable rather than ranked.
TIE_FRACTION = 0.01


def _key(subset) -> tuple[str, ...]:
    return tuple(sorted(subset, key=lambda m: _ORDER[m]))


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@torch.no_grad()
def subset_outputs(model, sp, device, batch_size: int = 128) -> dict:
    """Evaluate f(S) for all 8 subsets of M on one split."""
    model.eval()
    n = len(sp)
    store = {S: {"intensity": np.zeros(n, dtype=np.float32),
                 "logits": np.zeros((n, 3), dtype=np.float32)}
             for S in SUBSET_ORDER}

    for start in range(0, n, batch_size):
        idx = np.arange(start, min(start + batch_size, n))
        base = {m: torch.from_numpy(np.asarray(getattr(sp, m))[idx]).to(device)
                for m in MODS}
        base_masks = {m: torch.from_numpy(sp.masks[m][idx]).to(device)
                      for m in MODS}
        for S in SUBSET_ORDER:
            keep = set(S)
            masks = {m: (base_masks[m] if m in keep else torch.zeros_like(base_masks[m]))
                     for m in MODS}
            batch = {m: (base[m] if m in keep else torch.zeros_like(base[m]))
                     for m in MODS}
            out = model(batch, masks)
            store[S]["intensity"][idx] = out.intensity.cpu().numpy()
            store[S]["logits"][idx] = out.polarity_logits.cpu().numpy()
    return store


def targets_from(store: dict) -> dict:
    """The two attribution targets, both as functions of the subset S."""
    full = _key(MODS)
    probs_full = _softmax(store[full]["logits"])
    cstar = probs_full.argmax(axis=1)
    return {
        "prob_cstar": {S: _softmax(store[S]["logits"])[np.arange(len(cstar)), cstar]
                       for S in SUBSET_ORDER},
        "intensity": {S: store[S]["intensity"] for S in SUBSET_ORDER},
    }, cstar


def shapley_values(f: dict) -> tuple[dict, np.ndarray]:
    """phi_m per sample, plus the per-sample identity residual."""
    n = len(f[SUBSET_ORDER[0]])
    phi = {m: np.zeros(n, dtype=np.float64) for m in MODS}
    for m in MODS:
        others = [x for x in MODS if x != m]
        for r in range(len(others) + 1):
            for combo in itertools.combinations(others, r):
                S = _key(combo)
                Su = _key(combo + (m,))
                marginal = f[Su].astype(np.float64) - f[S].astype(np.float64)
                phi[m] += SHAPLEY_W[r] * marginal
    total = np.sum([phi[m] for m in MODS], axis=0)
    residual = total - (f[_key(MODS)].astype(np.float64)
                        - f[_key(())].astype(np.float64))
    return phi, residual


def _rank(means: dict, max_abs: float) -> tuple[list[str], list[list[str]]]:
    """Strict ranking, and a ranking that merges statistically indistinguishable
    neighbours (|phi| within TIE_FRACTION of the largest)."""
    strict = sorted(MODS, key=lambda m: -means[m])
    floor = TIE_FRACTION * max_abs
    groups: list[list[str]] = []
    for m in strict:
        if groups and abs(means[m] - means[groups[-1][0]]) <= floor:
            groups[-1].append(m)
        else:
            groups.append([m])
    return strict, groups


def _summarize(name: str, phi: dict, residuals: np.ndarray,
               f: dict, extra: dict) -> dict:
    n = len(f[SUBSET_ORDER[0]])
    stack = np.stack([phi[m] for m in MODS], axis=1)
    win = stack.argmax(axis=1)
    means = {m: float(phi[m].mean()) for m in MODS}
    max_abs = max(abs(v) for v in means.values()) or 1.0
    strict, groups = _rank(means, max_abs)
    return {
        "split": name,
        "n": n,
        "f_empty_is_constant": bool(np.allclose(f[_key(())], f[_key(())][0],
                                                atol=1e-6)),
        "f_empty_mean": round(float(f[_key(())].mean()), 6),
        "f_full_mean": round(float(f[_key(MODS)].mean()), 4),
        "mean_phi": {m: round(means[m], 4) for m in MODS},
        "median_phi": {m: round(float(np.median(phi[m])), 4) for m in MODS},
        "argmax_count": {m: int((win == k).sum()) for k, m in enumerate(MODS)},
        "argmax_share": {m: round(float((win == k).mean()), 4)
                         for k, m in enumerate(MODS)},
        "ranking": strict,
        "ranking_merged_ties": groups,
        "sum_phi_mean": round(float(np.sum([phi[m] for m in MODS], axis=0).mean()), 4),
        "f_full_minus_empty_mean": round(
            float((f[_key(MODS)] - f[_key(())]).mean()), 4),
        "efficiency_max_abs_error": float(np.abs(residuals).max()),
        **extra,
    }


@torch.no_grad()
def shapley_report(model, valid, att4, device, json_path: Path,
                   fig_path: Path | None = None) -> dict:
    """Exact decomposition on the validation split and attachment 4."""
    report: dict = {
        "method": "exact enumeration of the 8 modality subsets (no sampling)",
        "formula": ("phi_m = sum_{S subset M\\{m}}"
                    " |S|!(3-|S|-1)!/3! * [f(S u {m}) - f(S)]"),
        "primary_target": PRIMARY_TARGET,
        "secondary_target": SECONDARY_TARGET,
        "target_definition": {
            "prob_cstar": ("模型对「全输入预测类别 c*」给出的概率；非负，"
                           "phi 可读作「该模态为这一判断贡献了多少证据」"),
            "intensity": ("回归强度；**带符号**，phi 可读作「该模态把预测推向哪个方向、"
                          "推了多少」，不能用于排序模态重要性"),
        },
        "reference_point": ("f(empty) = 三模态全部遮罩时的模型输出。"
                            "强度目标下该值为与样本无关的常数（已在结果中校验）；"
                            "概率目标下遮罩后的 logits 为常数，但各样本选中的类别不同，"
                            "故 f(empty) 随所选类别变化。两者都是**归因参考点**，"
                            "不是 3.5 节关于全空样本的预测约定。"),
        "tie_fraction": TIE_FRACTION,
        "splits": {},
    }

    for name, sp in (("attachment2_valid", valid), ("attachment4", att4)):
        store = subset_outputs(model, sp, device)
        tg, cstar = targets_from(store)
        f_full_i = tg["intensity"][_key(MODS)]
        lomo = {m: float(np.abs(f_full_i.astype(np.float64)
                                - tg["intensity"][_key([x for x in MODS if x != m])]
                                .astype(np.float64)).mean())
                for m in MODS}
        lomo_strict, lomo_groups = _rank(lomo, max(abs(v) for v in lomo.values()) or 1.0)

        entry: dict = {"cstar_distribution": {str(k): int((cstar == k).sum())
                                              for k in range(3)},
                       "lomo_intensity_mean": {m: round(lomo[m], 4) for m in MODS},
                       "lomo_ranking": lomo_strict,
                       "lomo_ranking_merged_ties": lomo_groups}
        for tname in (PRIMARY_TARGET, SECONDARY_TARGET):
            phi, resid = shapley_values(tg[tname])
            summ = _summarize(name, phi, resid, tg[tname], {})
            entry[tname] = summ
            if name == "attachment4":
                # 注意：phi 是按模态索引的字典，样本数是每个数组的长度
                n_samp = len(phi[MODS[0]])
                ids = [str(x) for x in getattr(sp, "ids", [""] * n_samp)]
                summ["per_sample"] = [
                    {"sample_id": ids[i],
                     "phi": {m: round(float(phi[m][i]), 4) for m in MODS},
                     "sum_phi": round(float(sum(phi[m][i] for m in MODS)), 4),
                     "f_full_minus_empty": round(
                         float(tg[tname][_key(MODS)][i]
                               - tg[tname][_key(())][i]), 4)}
                    for i in range(n_samp)]
        # agreement between the primary Shapley ranking and the |LOMO| ranking
        entry["agrees_with_lomo"] = {
            "top1": entry[PRIMARY_TARGET]["ranking"][0] == lomo_strict[0],
            "strict_list_equal": entry[PRIMARY_TARGET]["ranking"] == lomo_strict,
            "shapley": entry[PRIMARY_TARGET]["ranking"],
            "lomo": lomo_strict,
        }
        report["splits"][name] = entry

    report["efficiency_check_max_abs_error"] = float(max(
        report["splits"][s][t]["efficiency_max_abs_error"]
        for s in report["splits"] for t in (PRIMARY_TARGET, SECONDARY_TARGET)))

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"  已写出 {json_path}")

    if fig_path is not None:
        _make_figure(report, fig_path)
    return report


def _make_figure(report: dict, fig_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .run_all import _setup_cjk_font
    font = _setup_cjk_font()
    print(f"  Shapley 图表字体: {font}")

    va = report["splits"]["attachment2_valid"][PRIMARY_TARGET]
    a4 = report["splits"]["attachment4"][PRIMARY_TARGET]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))

    x = np.arange(len(MODS))
    w = 0.38
    axes[0].bar(x - w / 2, [va["mean_phi"][m] for m in MODS], w,
                label=f"附件2 验证集 (n={va['n']})", color="#3b6ea5")
    axes[0].bar(x + w / 2, [a4["mean_phi"][m] for m in MODS], w,
                label=f"附件4 (n={a4['n']})", color="#c0504d")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(MODS)
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_ylabel("平均 Shapley 值 $\\phi_m$")
    axes[0].set_title("精确 Shapley 模态贡献（概率目标）")
    axes[0].legend(fontsize=8)

    per = a4["per_sample"]
    ids = [p["sample_id"] for p in per]
    xx = np.arange(len(per))
    bottom_pos = np.zeros(len(per))
    bottom_neg = np.zeros(len(per))
    for m, color in zip(MODS, ("#3b6ea5", "#9bbb59", "#c0504d")):
        vals = np.array([p["phi"][m] for p in per])
        pos = np.clip(vals, 0, None)
        neg = np.clip(vals, None, 0)
        axes[1].bar(xx, pos, 0.72, bottom=bottom_pos, label=m, color=color)
        axes[1].bar(xx, neg, 0.72, bottom=bottom_neg, color=color)
        bottom_pos += pos
        bottom_neg += neg
    axes[1].set_xticks(xx)
    axes[1].set_xticklabels(ids, fontsize=7)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xlabel("附件4 样本编号")
    axes[1].set_ylabel("$\\phi_m$")
    axes[1].set_title("逐样本 Shapley 分解（概率目标）")
    axes[1].legend(fontsize=8, ncol=3)

    # the signed-target caveat, shown rather than asserted
    sec = report["splits"]["attachment4"][SECONDARY_TARGET]
    axes[2].bar(x - w / 2, [sec["mean_phi"][m] for m in MODS], w,
                label="均值", color="#8f8f8f")
    axes[2].bar(x + w / 2, [sec["median_phi"][m] for m in MODS], w,
                label="中位数", color="#c9a227")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(MODS)
    axes[2].axhline(0, color="k", lw=0.8)
    axes[2].set_ylabel("$\\phi_m$（强度目标）")
    axes[2].set_title("带符号目标的陷阱：\n均值与中位数方向相反")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    print(f"  已写出 {fig_path}")


def main() -> int:
    """Standalone entry point: python -m q3_explain.shapley"""
    import argparse
    import os
    from pathlib import Path as _P

    from .data import fit_and_scale, load_attachment2, load_attachment4
    from .model import ModalityAttributionNet

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    root = _P(os.environ.get("MOSEI_ROOT") or _P(__file__).resolve().parents[2])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"设备: {device}")

    splits = load_attachment2(root / "data")
    att4 = load_attachment4(root / "data")
    fit_and_scale(splits, att4)

    ck = torch.load(root / "output" / "models" / "q3_model.pt",
                    map_location="cpu", weights_only=False)
    model = ModalityAttributionNet(ck["dims"], hidden=ck["hidden"])
    model.load_state_dict(ck["state_dict"])
    model.to(device)

    rep = shapley_report(model, splits["valid"], att4, device,
                         root / "work" / "q3" / "shapley_attribution.json",
                         root / "output" / "figures" / "q3_shapley.png")
    for split in ("attachment2_valid", "attachment4"):
        e = rep["splits"][split]
        s = e[PRIMARY_TARGET]
        print(f"\n[{split}] n={s['n']}")
        print(f"  概率目标排序   : {s['ranking']}  （并列合并后 {s['ranking_merged_ties']}）")
        print(f"  |LOMO| 排序    : {e['lomo_ranking']}")
        print(f"  与该排序一致   : {e['agrees_with_lomo']}")
        for m in MODS:
            print(f"    phi[{m:7s}] 均值={s['mean_phi'][m]:+.4f} "
                  f"中位={s['median_phi'][m]:+.4f} "
                  f"为最大者={s['argmax_count'][m]} ({s['argmax_share'][m]*100:.1f}%)")
        t = e[SECONDARY_TARGET]
        print(f"  强度目标(带符号，仅作对照): 排序 {t['ranking']}  "
              f"均值 {t['mean_phi']}  中位 {t['median_phi']}")
        print(f"  Σφ 效率性校验最大误差 = {s['efficiency_max_abs_error']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
