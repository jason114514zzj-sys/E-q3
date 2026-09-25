# -*- coding: utf-8 -*-
"""打乱注意力的置换检验：三组遮蔽统计量的**经验零分布**与 p 值（审查意见 Q3-06 的最高要求）。

为什么需要它
------------
论文原来的做法是拿一个**理想**零分布（1/4 − 1/(12R) = 7/30 ≈ 23.3%）当参照，并声明
"没有预设拒绝阈值、不作显著性结论"。审查方要求的是：在固定长度、预算、互斥规则与重复次数 R
的条件下，把注意力打乱后**重算**统计量，得到经验零分布，再据此报告效应量与显著性。

置换方案（与原协议逐条对齐）
--------------------------
原协议（`model_explain.faithfulness_test`）对模态 m 的**候选位**做三组遮蔽，k=3：

* 峰值组：注意力最高的 k 个候选位
* 低谷组：注意力最低的 k 个候选位，且与峰值组**不相交**
* 随机组：均匀取 k 个候选位，重复 R=5 次取平均
* 资格：候选位数 ≥ 2k 的样本才计入（三组同子集）

把注意力在候选位上**随机重排**之后，"最高的 k 个"就是 K 的均匀随机 k 子集，因此等价于：

* 峰值组 = 均匀随机 k 子集 T
* 低谷组 = 从候选位去掉 T 后均匀随机的 k 子集 L（**这就是重排后"最低的 k 个"**）
* 随机组 = 均匀随机 k 子集，R 次取平均
* 资格规则不变

实现上把 (T, L) 作为**联合抽样**存进池（这样低谷组的条件分布与重排后一致，且在候选位只有
恰好 2k 个时"两个 k 子集必相交"的边界也天然正确），随机组另用一个单子集池。
随后独立抽 `n_rounds` 轮组合统计量；每轮的**边际分布**与逐轮重算完全一致，只是不再受
逐次 torch 调用开销限制（M=200 组池 ⇒ 每模态 600 次前向即可支撑上万轮）。

输出（写 `work/q3/permutation_test.json`）
----------------------------------------
每个模态：观测值、经验零分布的均值/分位数、单侧经验 p 值，以及效应量（峰值−随机、随机−低谷）
与它们的零分布。

用法：
    python -m q3_explain.permutation_test
    python -m q3_explain.permutation_test --n-pool 200 --n-rounds 4000
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


def _drops_for_picks(model, sp, m, picks, base, device, batch_size: int = 128):
    """把每条样本 `picks[i]` 里的位置从可观测掩码上去掉，返回 |Δ强度|。"""
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
        ab = {mm: (torch.zeros_like(batch[mm]) if mm == m else batch[mm]) for mm in MODS}
        ab[m] = batch[m] * picked.unsqueeze(-1).float()
        nm = dict(masks)
        nm[m] = picked
        with torch.no_grad():
            pred = model(ab, nm).intensity.cpu().numpy()
        out[idx] = np.abs(pred - base[idx])
    return out


def _rank_positions(att_row, cand_flat, k):
    """按注意力从高到低排出候选位（观测协议用）。"""
    a = np.asarray(att_row)[cand_flat]
    order = cand_flat[np.argsort(-a, kind="stable")]
    return order


def permutation_test(model, sp, device, k: int = 3, n_random: int = 5,
                     n_pool: int = 200, n_rounds: int = 4000, seed: int = 0,
                     batch_size: int = 128) -> dict:
    from .model_explain import forward_all

    fwd = forward_all(model, sp, device)
    base = fwd["intensity"]
    n = len(sp)
    # 观测统计量以**主流程已发布的那一份**为准（预注册口径），本脚本重算的值只作交叉核对：
    # 两者之差仅来自随机组 5 次重抽样的蒙特卡洛噪声，不是协议差异。
    canonical: dict = {}
    ref_path = RESULTS / "q3_report.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text(encoding="utf-8"))["faithfulness"]
        canonical = {m: {"share_strict_order": ref["per_modality"][m]["share_random_between"],
                         "gap_top_minus_rand": ref["per_modality"][m]["gap_topk_minus_random"],
                         "gap_rand_minus_low": ref["per_modality"][m]["gap_random_minus_lowk"],
                         "mean_top": ref["per_modality"][m]["subset_mean_drop_topk"],
                         "mean_low": ref["per_modality"][m]["subset_mean_drop_lowk"]}
                     for m in MODS}
    rep: dict = {"k": k, "n_random": n_random, "n_pool": n_pool,
                 "n_rounds": n_rounds, "seed": seed,
                 "observed_source": ("work/q3/q3_report.json 的 faithfulness 段"
                                     "（主流程预注册口径，R=5）" if canonical else "本脚本重算"),
                 "protocol": ("注意力在候选位上随机重排 ⇒ 峰值组=均匀随机 k 子集 T、"
                              "低谷组=从候选位去 T 后均匀随机 k 子集（与重排后的"
                              "『最低 k 个』同分布）、随机组=均匀随机 k 子集，R 次取平均；"
                              "资格仍是候选位数 ≥ 2k 的同一子集"),
                 "per_modality": {}}

    for mi, m in enumerate(MODS):
        cand = np.asarray(sp.content_of(m))
        cand_flat = [np.flatnonzero(cand[i]) for i in range(n)]
        n_cand = np.array([c.size for c in cand_flat])
        elig = n_cand >= 2 * k
        sel = np.flatnonzero(elig)
        print(f"[{m}] 候选位数 {n_cand.min()}~{n_cand.max()}，资格样本 {sel.size}/{n}"
              f"（剔除 {n - sel.size}）")

        # ---------------- 观测统计量（与 faithfulness_test 同协议） ----------------
        picks_top = [[] for _ in range(n)]
        picks_low = [[] for _ in range(n)]
        for i in sel:
            order = _rank_positions(fwd["attention"][m][i], cand_flat[i], k)
            picks_top[i] = order[:k]
            picks_low[i] = order[k:][::-1][:k]
        d_top = _drops_for_picks(model, sp, m, picks_top, base, device, batch_size)
        d_low = _drops_for_picks(model, sp, m, picks_low, base, device, batch_size)

        rng = np.random.default_rng(seed + 1013 * mi)
        acc = np.zeros(n, dtype=np.float64)
        for _ in range(n_random):
            picks = [[] for _ in range(n)]
            for i in sel:
                picks[i] = rng.choice(cand_flat[i], size=k, replace=False)
            acc += _drops_for_picks(model, sp, m, picks, base, device, batch_size)
        d_rand = acc / n_random

        obs = {
            "mean_top": float(d_top[sel].mean()),
            "mean_rand": float(d_rand[sel].mean()),
            "mean_low": float(d_low[sel].mean()),
            "gap_top_minus_rand": float(d_top[sel].mean() - d_rand[sel].mean()),
            "gap_rand_minus_low": float(d_rand[sel].mean() - d_low[sel].mean()),
            "share_strict_order": float(
                ((d_top[sel] > d_rand[sel]) & (d_rand[sel] > d_low[sel])).mean()),
        }
        print(f"  观测: top={obs['mean_top']:.4f} rand={obs['mean_rand']:.4f} "
              f"low={obs['mean_low']:.4f} 三段序={obs['share_strict_order']:.4f}")

        # ---------------- 池：(T,L) 联合 + 单子集 ----------------
        pool_tl_t = np.zeros((n_pool, n), dtype=np.float64)
        pool_tl_l = np.zeros((n_pool, n), dtype=np.float64)
        pool_r = np.zeros((n_pool, n), dtype=np.float64)
        for j in range(n_pool):
            pT = [[] for _ in range(n)]
            pL = [[] for _ in range(n)]
            for i in sel:
                T = rng.choice(cand_flat[i], size=k, replace=False)
                rest = np.setdiff1d(cand_flat[i], T, assume_unique=False)
                L = rng.choice(rest, size=k, replace=False)
                pT[i] = T
                pL[i] = L
            pool_tl_t[j] = _drops_for_picks(model, sp, m, pT, base, device, batch_size)
            pool_tl_l[j] = _drops_for_picks(model, sp, m, pL, base, device, batch_size)
            pR = [[] for _ in range(n)]
            for i in sel:
                pR[i] = rng.choice(cand_flat[i], size=k, replace=False)
            pool_r[j] = _drops_for_picks(model, sp, m, pR, base, device, batch_size)
            if (j + 1) % 50 == 0:
                print(f"    池 {j + 1}/{n_pool}")

        # ---------------- 经验零分布 ----------------
        sims = np.empty((n_rounds, 3), dtype=np.float64)   # strict / gap_tr / gap_rl
        for r in range(n_rounds):
            e = rng.integers(0, n_pool, size=n)            # 每样本独立抽一个 (T,L) 条目
            f = rng.integers(0, n_pool, size=(n_random, n))
            t = pool_tl_t[e, np.arange(n)]
            l = pool_tl_l[e, np.arange(n)]
            rr = pool_r[f, np.arange(n)].mean(axis=0)      # R 次取平均
            ts, rs, ls = t[sel], rr[sel], l[sel]
            sims[r] = [
                ((ts > rs) & (rs > ls)).mean(),
                ts.mean() - rs.mean(),
                rs.mean() - ls.mean(),
            ]

        def pval(obs_v, null_v):
            return float((1 + int((null_v >= obs_v).sum())) / (1 + len(null_v)))

        can = canonical.get(m, {})
        obs_used = {
            "share_strict_order": can.get("share_strict_order", obs["share_strict_order"]),
            "gap_top_minus_rand": can.get("gap_top_minus_rand", obs["gap_top_minus_rand"]),
            "gap_rand_minus_low": can.get("gap_rand_minus_low", obs["gap_rand_minus_low"]),
        }
        ent = {
            "n_eligible": int(sel.size),
            "n_excluded": int(n - sel.size),
            "observed_recomputed": obs,
            "observed_canonical": dict(obs_used),
            "recompute_vs_canonical_max_abs_diff": round(max(
                abs(obs["share_strict_order"] - obs_used["share_strict_order"]),
                abs(obs["gap_top_minus_rand"] - obs_used["gap_top_minus_rand"]),
                abs(obs["gap_rand_minus_low"] - obs_used["gap_rand_minus_low"])), 4),
            "null_strict_order": {
                "mean": round(float(sims[:, 0].mean()), 4),
                "p2.5": round(float(np.percentile(sims[:, 0], 2.5)), 4),
                "p50": round(float(np.percentile(sims[:, 0], 50)), 4),
                "p97.5": round(float(np.percentile(sims[:, 0], 97.5)), 4),
                "p_value_one_sided": round(pval(obs_used["share_strict_order"], sims[:, 0]), 4),
                "p_value_resolution": round(1.0 / (1 + n_rounds), 5),
            },
            "null_gap_top_minus_rand": {
                "mean": round(float(sims[:, 1].mean()), 4),
                "p2.5": round(float(np.percentile(sims[:, 1], 2.5)), 4),
                "p97.5": round(float(np.percentile(sims[:, 1], 97.5)), 4),
                "p_value_one_sided": round(pval(obs_used["gap_top_minus_rand"], sims[:, 1]), 4),
            },
            "null_gap_rand_minus_low": {
                "mean": round(float(sims[:, 2].mean()), 4),
                "p2.5": round(float(np.percentile(sims[:, 2], 2.5)), 4),
                "p97.5": round(float(np.percentile(sims[:, 2], 97.5)), 4),
                "p_value_one_sided": round(pval(obs_used["gap_rand_minus_low"], sims[:, 2]), 4),
            },
        }
        rep["per_modality"][m] = ent
        ns = ent["null_strict_order"]
        print(f"  零分布 三段序 {ns['mean']:.4f} [{ns['p2.5']:.4f}, {ns['p97.5']:.4f}]  "
              f"观测 {obs_used['share_strict_order']:.4f}  p={ns['p_value_one_sided']:.5f}  |  "
              f"gap(top-rand) 观测 {obs_used['gap_top_minus_rand']:+.4f} "
              f"零分布 [{ent['null_gap_top_minus_rand']['p2.5']:+.4f}, "
              f"{ent['null_gap_top_minus_rand']['p97.5']:+.4f}]  "
              f"p={ent['null_gap_top_minus_rand']['p_value_one_sided']:.5f}  |  "
              f"重算差 {ent['recompute_vs_canonical_max_abs_diff']:.4f}")

    # 交叉核对：本脚本重算的观测值 vs 主报告（差异只应来自随机组重抽样的噪声）
    if canonical:
        chk = {m: {"recomputed_share": rep["per_modality"][m]["observed_recomputed"]["share_strict_order"],
                   "canonical_share": rep["per_modality"][m]["observed_canonical"]["share_strict_order"],
                   "max_abs_diff": rep["per_modality"][m]["recompute_vs_canonical_max_abs_diff"],
                   "within_random_resampling_noise": bool(
                       rep["per_modality"][m]["recompute_vs_canonical_max_abs_diff"] <= 0.03)}
               for m in MODS}
        rep["self_check_vs_main_pipeline"] = chk
        print("\n自证（重算 vs 主报告，容差 0.03 = 随机组 5 次重抽样的量级）：" +
              json.dumps({m: chk[m]["within_random_resampling_noise"] for m in MODS},
                         ensure_ascii=False))
    return rep


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument("--n-pool", type=int, default=200)
    ap.add_argument("--n-rounds", type=int, default=4000)
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

    rep = permutation_test(model, splits["valid"], device, k=args.k,
                           n_random=args.n_random, n_pool=args.n_pool,
                           n_rounds=args.n_rounds, seed=args.seed)
    rep["checkpoint"] = "output/models/q3_model.pt"
    rep["split"] = "attachment2_valid(n=728)"
    RESULTS.mkdir(parents=True, exist_ok=True)
    dst = RESULTS / "permutation_test.json"
    dst.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
