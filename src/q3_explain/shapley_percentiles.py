# -*- coding: utf-8 -*-
"""Shapley 值的分布形态与配对不确定性（回应审查意见 Q3-04）。

审查意见指出：只报有符号均值与 mean|φ| 不足以支撑"某两路作用不可区分"或"作用很小"的判断，
需要给出分位数、正负比例，以及"实质等效界限"的配对不确定性。本模块只读模型检查点与附件数据，
不改动任何既有产物，输出一份新的汇总 results/shapley_percentiles.json。

做法：
1. 对附件2 验证集与附件4 分别枚举 8 个模态子集，得到 f(S)；
2. 按同一目标（概率目标固定全输入预测类别 c*；强度目标）计算逐样本 φ_m；
3. 汇总每个模态的 均值 / 中位 / mean|φ| / 5、25、75、95 分位 / 正负零比例 / 逐样本最大者占比；
4. 对 mean|φ| 及其两两差值做按样本配对的 Bootstrap（1000 次重采样），给出 95% 区间，
   作为"实质等效界限"的经验不确定性；本模块不对该界限预设数值门槛，只给出区间。

用法：MOSEI_ROOT=~/MathModel PYTHONPATH=~/MathModel/src python -m q3_explain.shapley_percentiles
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from .data import MODALITIES, fit_and_scale, load_attachment2, load_attachment4
from .model import ModalityAttributionNet
from .shapley import (MODS, PRIMARY_TARGET, SECONDARY_TARGET, shapley_values,
                      subset_outputs, targets_from)

N_BOOT = 1000
SEED = 20260924


def _stats(v: np.ndarray) -> dict:
    a = np.abs(v)
    return {
        "n": int(v.size),
        "mean": round(float(v.mean()), 6),
        "median": round(float(np.median(v)), 6),
        "mean_abs": round(float(a.mean()), 6),
        "p05": round(float(np.percentile(v, 5)), 6),
        "p25": round(float(np.percentile(v, 25)), 6),
        "p75": round(float(np.percentile(v, 75)), 6),
        "p95": round(float(np.percentile(v, 95)), 6),
        "abs_p05": round(float(np.percentile(a, 5)), 6),
        "abs_p50": round(float(np.percentile(a, 50)), 6),
        "abs_p95": round(float(np.percentile(a, 95)), 6),
        "share_positive": round(float((v > 0).mean()), 4),
        "share_negative": round(float((v < 0).mean()), 4),
        "share_zero": round(float((v == 0).mean()), 4),
    }


def _paired_bootstrap(phi: dict) -> dict:
    rng = np.random.default_rng(SEED)
    abs_phi = {m: np.abs(phi[m]) for m in MODS}
    n = abs_phi[MODS[0]].size
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boot = {m: abs_phi[m][idx].mean(axis=1) for m in MODS}
    gap_ta = boot["text"] - np.maximum(boot["audio"], boot["vision"])
    gap_av = boot["audio"] - boot["vision"]          # 有符号：负值表示视觉 > 语音
    out = {
        "n_boot": N_BOOT,
        "seed": SEED,
        "resampling_unit": "按样本配对重采样（同一批样本索引在三路之间同步抽取）",
        "equivalence_margin": "本文未预设数值型实质等效界限；下表只给出按样本配对的 95% 区间，"
                              "区间含 0 即表示在该次训练的开发集上无法区分，区间不含 0 仅表示"
                              "本次训练的抽样不确定性下可区分，跨训练种子的稳定性另见结果文件的多种子对照。",
    }
    for m in MODS:
        out[f"mean_abs_phi_{m}"] = [round(float(np.percentile(boot[m], 2.5)), 6),
                                    round(float(np.percentile(boot[m], 97.5)), 6)]
    out["gap_text_minus_weaker"] = [round(float(np.percentile(gap_ta, 2.5)), 6),
                                    round(float(np.percentile(gap_ta, 97.5)), 6)]
    out["gap_text_minus_weaker_excludes_zero"] = bool(np.percentile(gap_ta, 2.5) > 0)
    out["gap_abs_audio_minus_abs_vision"] = [round(float(np.percentile(gap_av, 2.5)), 6),
                                             round(float(np.percentile(gap_av, 97.5)), 6)]
    lo = float(np.percentile(gap_av, 2.5))
    hi = float(np.percentile(gap_av, 97.5))
    out["gap_abs_audio_minus_abs_vision_verdict"] = (
        "视觉 > 语音（区间整体为负）" if hi < 0 else
        "语音 > 视觉（区间整体为正）" if lo > 0 else
        "两者不可区分（区间跨 0）")
    return out


def main() -> int:
    root = Path(os.environ.get("MOSEI_ROOT") or Path(__file__).resolve().parents[2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("设备", device)

    splits = load_attachment2(root / "data")
    att4 = load_attachment4(root / "data")
    fit_and_scale(splits, att4)

    ck = torch.load(root / "output" / "models" / "q3_model.pt",
                    map_location="cpu", weights_only=False)
    model = ModalityAttributionNet(ck["dims"], hidden=ck["hidden"])
    model.load_state_dict(ck["state_dict"])
    model.to(device)

    report = {
        "purpose": "Shapley 值的分布形态、正负比例与 mean|φ| 的配对 Bootstrap 区间（审查意见 Q3-04）",
        "seed": SEED,
        "primary_target": PRIMARY_TARGET,
        "secondary_target": SECONDARY_TARGET,
        "note": "两个目标的 φ 语义不同：概率目标固定全输入预测类别 c*（有符号，可作方向读法）；"
                "强度目标为带符号回归扰动。mean|φ| 用于汇总强弱，但绝对化后不再满足同一加法分解，"
                "因此 mean|φ| 只用来说明某一路的量级，不能与 Σφ=1 的效率性恒等式混用。",
        "splits": {},
    }
    for name, sp in (("attachment2_valid", splits["valid"]), ("attachment4", att4)):
        store = subset_outputs(model, sp, device)
        tg, cstar = targets_from(store)
        entry = {"n": int(len(sp)), "cstar_distribution":
                 {str(k): int(v) for k, v in zip(*np.unique(cstar, return_counts=True))},
                 "targets": {}}
        for tname in (PRIMARY_TARGET, SECONDARY_TARGET):
            phi, resid = shapley_values(tg[tname])
            entry["targets"][tname] = {
                "per_modality": {m: _stats(phi[m]) for m in MODS},
                "efficiency_max_abs_error": float(np.abs(resid).max()),
                "argmax_share": {m: round(float((np.stack([phi[x] for x in MODS]).argmax(axis=0)
                                                 == MODS.index(m)).mean()), 4) for m in MODS},
                "paired_bootstrap": _paired_bootstrap(phi),
            }
        report["splits"][name] = entry
        for tname in (PRIMARY_TARGET, SECONDARY_TARGET):
            t = entry["targets"][tname]
            print(f"[{name}/{tname}] n={entry['n']}")
            for m in MODS:
                s = t["per_modality"][m]
                print(f"   {m:7s} mean={s['mean']:+.4f} mean|φ|={s['mean_abs']:.4f} "
                      f"p05={s['p05']:+.4f} p95={s['p95']:+.4f} "
                      f"正/负={s['share_positive']:.3f}/{s['share_negative']:.3f}")
            pb = t["paired_bootstrap"]
            print(f"   gap(text−较弱) 95% 区间 {pb['gap_text_minus_weaker']} "
                  f"不含 0 = {pb['gap_text_minus_weaker_excludes_zero']}")
            print(f"   gap(|φ_audio|−|φ_vision|) 95% 区间 {pb['gap_abs_audio_minus_abs_vision']} -> {pb['gap_abs_audio_minus_abs_vision_verdict']}")

    out = root / "work" / "q3" / "shapley_percentiles.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已写出", out, f"{out.stat().st_size:,} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
