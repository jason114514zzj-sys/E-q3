# -*- coding: utf-8 -*-
"""连续窗口遮蔽：把「实际交付的证据区间」本身拿去做等预算对照（审查意见 Q3-05）。

为什么需要它
------------
论文的三组遮蔽检验（`model_explain.faithfulness_test`）遮蔽的是**注意力最高的 k 个候选位**，
这 k 个位置只按注意力排序、**不保证连续**；而交付给读者的证据是**峰位左右各一位的连续窗口**
（式 7-9）。删除一个集合能造成更大扰动，**不能自动证明另一个集合有效**。审查方因此要求：
直接遮蔽实际交付的每个证据区间，与等预算的「随机连续窗口」「最低注意力连续窗口」做配对比较。

本模块在同一批样本、同预算 k=3、同一互斥规则下比较四类位置集合：

* ``win_top``  —— **实际交付的证据窗口**（注意力峰值位在候选序上的左右各一位，端点截断，
  与式 7-9 的定义逐字一致）；
* ``win_low``  —— 注意力**最低**位为中心的同构连续窗口（与 ``win_top`` 不相交者才计入）；
* ``win_rand`` —— 候选序上随机起点的连续 k 位，取 R=5 次平均（与主协议同 R）；
* ``top_k``    —— 注意力最高的 k 个候选位（**非连续**，即原协议的峰值组），用于与 win_top 对照。

统计口径
--------
* 资格：候选位数 ≥ 2k，且 ``win_top ∩ win_low = ∅``（否则两个被删集合重合、比较退化）；
  被剔除的样本数如实报告。
* 配对区间：对逐样本效应量做 **1000 次按样本配对的重采样**，给出差距与胜率的 95% 分位区间。
  本问的输入接口不携带 video_id，故重采样单位是样本；该限制在输出里写明。
* 两种视角：**按模态**（每个模态各自的窗口）与**按交付**（每条样本实际交付的主导模态窗口）。

用法：
    python -m q3_explain.window_faithfulness
    python -m q3_explain.window_faithfulness --n-random 5 --n-boot 1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("MOSEI_ROOT") or Path(__file__).resolve().parents[2])
DATA = ROOT / "data"
OUT = ROOT / "output"
RESULTS = ROOT / "work" / "q3"
MODS = ("text", "audio", "vision")


def _window_around(cand: np.ndarray, pos_index: int, k: int) -> np.ndarray:
    """候选序上以 pos_index 为中心、长度 k 的连续窗口（端点处截断）。"""
    lo = int(np.clip(pos_index - k // 2, 0, max(len(cand) - k, 0)))
    return cand[lo:lo + k]


def _drops(model, sp, m, picks, base, device, batch_size: int = 128) -> np.ndarray:
    """把每条样本 picks[i] 的位置从该模态可观测掩码上去掉，返回 |Δ强度|。"""
    import torch

    n = len(sp)
    out = np.zeros(n, dtype=np.float64)
    for start in range(0, n, batch_size):
        idx = np.arange(start, min(start + batch_size, n))
        batch = {mm: torch.from_numpy(np.asarray(getattr(sp, mm))[idx]).to(device)
                 for mm in MODS}
        masks = {mm: torch.from_numpy(sp.masks[mm][idx]).to(device) for mm in MODS}
        picked = masks[m].clone()
        for r, i in enumerate(idx):
            for p in picks[i]:
                picked[r, int(p)] = False
        ab = {mm: (torch.zeros_like(batch[mm]) if mm == m else batch[mm])
              for mm in MODS}
        ab[m] = batch[m] * picked.unsqueeze(-1).float()
        nm = dict(masks)
        nm[m] = picked
        with torch.no_grad():
            pred = model(ab, nm).intensity.cpu().numpy()
        out[idx] = np.abs(pred - base[idx])
    return out


def _boot_ci(x: np.ndarray, n_boot: int, rng: np.random.Generator,
             stat=np.mean) -> tuple[float, float]:
    """按样本配对重采样的 95% 分位区间。"""
    if x.size == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    vals = stat(x[idx], axis=1)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def window_faithfulness(model, sp, device, k: int = 3, n_random: int = 5,
                        n_boot: int = 1000, seed: int = 0,
                        batch_size: int = 128, tag: str = "") -> dict:
    from .model_explain import forward_all

    rng = np.random.default_rng(seed)
    fwd = forward_all(model, sp, device)
    base = fwd["intensity"]
    n = len(sp)

    rep: dict = {
        "tag": tag,
        "n_samples": n,
        "k": k,
        "n_random_trials": n_random,
        "n_bootstrap": n_boot,
        "bootstrap_unit": "样本（本问接口不携带 video_id；未按原视频分组）",
        "window_rule": ("交付窗口 = 注意力峰值位在候选序上左右各一位（共 k 个连续候选位，"
                        "端点截断），与论文式（7-9）一致；低谷窗口 = 注意力最低位为中心的同构窗口；"
                        "随机窗口 = 候选序上随机起点的连续 k 位，取 R 次平均"),
        "eligibility": "候选位数 ≥ 2k 且 win_top ∩ win_low = ∅",
        "per_modality": {},
    }

    gates = fwd["gate"]                       # (n, 3)
    dominant = np.array(MODS)[np.argmax(gates, axis=1)]
    rep["dominant_modality_counts"] = {m: int((dominant == m).sum()) for m in MODS}

    d_top = {m: np.full(n, np.nan) for m in MODS}
    d_low = {m: np.full(n, np.nan) for m in MODS}
    d_rand = {m: np.full(n, np.nan) for m in MODS}
    d_sort = {m: np.full(n, np.nan) for m in MODS}
    ok = {m: np.zeros(n, dtype=bool) for m in MODS}
    same_set = {m: np.zeros(n, dtype=bool) for m in MODS}

    for m in MODS:
        picks_top: list[np.ndarray] = []
        picks_low: list[np.ndarray] = []
        picks_sort: list[np.ndarray] = []
        for i in range(n):
            cand = np.flatnonzero(sp.content_of(m)[i])
            if cand.size < 2 * k:
                picks_top.append(cand[:0]); picks_low.append(cand[:0])
                picks_sort.append(cand[:0])
                continue
            a = np.asarray(fwd["attention"][m][i])[cand]
            j_hi = int(np.argmax(a)); j_lo = int(np.argmin(a))
            w_hi = _window_around(cand, j_hi, k)
            w_lo = _window_around(cand, j_lo, k)
            srt = cand[np.argsort(-a, kind="stable")][:k]
            picks_top.append(w_hi); picks_low.append(w_lo); picks_sort.append(srt)
            ok[m][i] = (np.intersect1d(w_hi, w_lo).size == 0)
            same_set[m][i] = set(w_hi.tolist()) == set(srt.tolist())

        # 随机连续窗口：R 次平均
        acc = np.zeros(n, dtype=np.float64)
        for _ in range(n_random):
            picks_rnd: list[np.ndarray] = []
            for i in range(n):
                cand = np.flatnonzero(sp.content_of(m)[i])
                if cand.size < k:
                    picks_rnd.append(cand[:0]); continue
                s = int(rng.integers(0, cand.size - k + 1))
                picks_rnd.append(cand[s:s + k])
            acc += _drops(model, sp, m, picks_rnd, base, device, batch_size)
        rand_mean = acc / n_random

        d_top[m] = _drops(model, sp, m, picks_top, base, device, batch_size)
        d_low[m] = _drops(model, sp, m, picks_low, base, device, batch_size)
        d_sort[m] = _drops(model, sp, m, picks_sort, base, device, batch_size)
        d_rand[m] = rand_mean

        sel = ok[m]
        n_sel = int(sel.sum())
        gt, gl, gr = d_top[m][sel], d_low[m][sel], d_rand[m][sel]
        gs = d_sort[m][sel]
        strict = ((gt > gr) & (gr > gl))
        gap_tr = gt - gr
        gap_rl = gr - gl
        g_tr_lo, g_tr_hi = _boot_ci(gap_tr, n_boot, rng)
        g_rl_lo, g_rl_hi = _boot_ci(gap_rl, n_boot, rng)
        s_lo, s_hi = _boot_ci(strict.astype(np.float64), n_boot, rng)
        rep["per_modality"][m] = {
            "n_candidates_mean": float(np.mean(
                [np.flatnonzero(sp.content_of(m)[i]).size for i in range(n)])),
            "n_eligible": n_sel,
            "n_excluded_too_few_candidates": int((~np.array(
                [np.flatnonzero(sp.content_of(m)[i]).size >= 2 * k for i in range(n)])).sum()),
            "n_excluded_window_overlap": int(((~sel) & np.array(
                [np.flatnonzero(sp.content_of(m)[i]).size >= 2 * k for i in range(n)])).sum()),
            "mean_drop_window_top": float(np.mean(gt)) if n_sel else float("nan"),
            "mean_drop_window_random": float(np.mean(gr)) if n_sel else float("nan"),
            "mean_drop_window_low": float(np.mean(gl)) if n_sel else float("nan"),
            "mean_drop_sorted_topk": float(np.mean(gs)) if n_sel else float("nan"),
            "gap_window_top_minus_random": float(np.mean(gap_tr)) if n_sel else float("nan"),
            "gap_window_top_minus_random_ci95": [g_tr_lo, g_tr_hi],
            "gap_window_random_minus_low": float(np.mean(gap_rl)) if n_sel else float("nan"),
            "gap_window_random_minus_low_ci95": [g_rl_lo, g_rl_hi],
            "share_strict_order_window": float(np.mean(strict)) if n_sel else float("nan"),
            "share_strict_order_window_ci95": [s_lo, s_hi],
            "share_topk_gt_random": float(np.mean(gt > gr)) if n_sel else float("nan"),
            "share_window_gt_sorted_topk": float(np.mean(gt > gs)) if n_sel else float("nan"),
            "sorted_topk_equals_window_share": float(np.mean(same_set[m][sel]))
            if n_sel else float("nan"),
            "mean_drop_sorted_topk_minus_window": float(np.mean(gs - gt)) if n_sel else float("nan"),
        }
        e = rep["per_modality"][m]
        print(f"  [{tag}|{m}] 资格 {n_sel} 条（候选不足剔除 "
              f"{e['n_excluded_too_few_candidates']}，窗口重合剔除 {e['n_excluded_window_overlap']}）"
              f"  交付窗口 {e['mean_drop_window_top']:.4f} / 随机 {e['mean_drop_window_random']:.4f}"
              f" / 低谷 {e['mean_drop_window_low']:.4f}  严格三段序 {e['share_strict_order_window']:.3f}"
              f"  窗口==sorted topk {e['sorted_topk_equals_window_share']:.3f}")

    # ---- 按交付：每条样本用**实际交付**的主导模态窗口做对照 ----
    sel_d = np.zeros(n, dtype=bool)
    d_del = np.zeros(n, dtype=np.float64)
    d_del_rand = np.zeros(n, dtype=np.float64)
    d_del_low = np.zeros(n, dtype=np.float64)
    for i in range(n):
        m = str(dominant[i])
        if ok[m][i]:
            sel_d[i] = True
            d_del[i] = d_top[m][i]
            d_del_rand[i] = d_rand[m][i]
            d_del_low[i] = d_low[m][i]
    nd = int(sel_d.sum())
    strict_d = ((d_del[sel_d] > d_del_rand[sel_d]) & (d_del_rand[sel_d] > d_del_low[sel_d]))
    gtr = d_del[sel_d] - d_del_rand[sel_d]
    grl = d_del_rand[sel_d] - d_del_low[sel_d]
    a_lo, a_hi = _boot_ci(gtr, n_boot, rng)
    b_lo, b_hi = _boot_ci(grl, n_boot, rng)
    c_lo, c_hi = _boot_ci(strict_d.astype(np.float64), n_boot, rng)
    rep["delivered_window"] = {
        "n_evaluated": nd,
        "dominant_modality_counts_on_subset": {
            m: int(((dominant == m) & sel_d).sum()) for m in MODS},
        "mean_drop_delivered_window": float(np.mean(d_del[sel_d])) if nd else float("nan"),
        "mean_drop_random_window": float(np.mean(d_del_rand[sel_d])) if nd else float("nan"),
        "mean_drop_low_window": float(np.mean(d_del_low[sel_d])) if nd else float("nan"),
        "gap_delivered_minus_random": float(np.mean(gtr)) if nd else float("nan"),
        "gap_delivered_minus_random_ci95": [a_lo, a_hi],
        "gap_random_minus_low": float(np.mean(grl)) if nd else float("nan"),
        "gap_random_minus_low_ci95": [b_lo, b_hi],
        "share_delivered_gt_random_and_gt_low": float(np.mean(
            (d_del[sel_d] > d_del_rand[sel_d]) & (d_del[sel_d] > d_del_low[sel_d]))) if nd else float("nan"),
        "share_strict_order": float(np.mean(strict_d)) if nd else float("nan"),
        "share_strict_order_ci95": [c_lo, c_hi],
    }
    print(f"  [{tag}|交付窗口] n={nd}  "
          f"交付 {rep['delivered_window']['mean_drop_delivered_window']:.4f} / "
          f"随机 {rep['delivered_window']['mean_drop_random_window']:.4f} / "
          f"低谷 {rep['delivered_window']['mean_drop_low_window']:.4f}  "
          f"gap(交付−随机) {rep['delivered_window']['gap_delivered_minus_random']:+.4f} "
          f"CI {rep['delivered_window']['gap_delivered_minus_random_ci95']}  "
          f"严格三段序 {rep['delivered_window']['share_strict_order']:.3f}")
    return rep


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from .data import fit_and_scale, load_attachment2, load_attachment4
    from .model import ModalityAttributionNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    splits = load_attachment2(DATA)
    att4 = load_attachment4(DATA)
    fit_and_scale(splits, att4)
    ck = torch.load(OUT / "models" / "q3_model.pt", map_location=device,
                    weights_only=False)
    model = ModalityAttributionNet(ck["dims"], hidden=ck["hidden"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"检查点: output/models/q3_model.pt  best_epoch={ck.get('best_epoch')}")

    rep = {
        "k": args.k,
        "n_random": args.n_random,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "checkpoint": "output/models/q3_model.pt",
        "attachment2_valid": window_faithfulness(
            model, splits["valid"], device, k=args.k, n_random=args.n_random,
            n_boot=args.n_boot, seed=args.seed, tag="att2-valid"),
        "attachment4": window_faithfulness(
            model, att4, device, k=args.k, n_random=args.n_random,
            n_boot=args.n_boot, seed=args.seed + 1, tag="att4"),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    dst = RESULTS / "window_faithfulness.json"
    dst.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
