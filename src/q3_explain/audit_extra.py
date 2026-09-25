# -*- coding: utf-8 -*-
"""问题三：补交复算脚本（回应审查意见 Q3-02 / 04 / 08 / 10 / 12）。

用法（服务器上，非交互）：
    source ~/miniconda3/etc/profile.d/conda.sh; conda activate mosei
    cd ~/MathModel && export PYTHONPATH=~/MathModel/src
    python -m q3_explain.audit_extra

本脚本**只做评价与导出**：不训练、不改任何既有结果文件，新增产物一律写到
results/ 下的新文件名，便于与既有交付件对照：

    results/q3_audit_extra.json              复算汇总
    results/附件4_8子集逐样本.csv              每样本 8 个子集的输出（160 行）
    results/附件4_预测与解释结果_全字段.csv     解释卡全字段（含不一致标记、映射置信）

覆盖的审查项
    Q3-10  附件2 测试划分（727 条，全程不参与任何选择）的指标 → 留出对照
    Q3-12  回归残差分布 / 分位数 / 分组 MAE → 补上回归侧的错误归因
    Q3-04  有符号 φ 与 mean|φ| 并列，供「正负相消」判断
    Q3-02  逐样本 8 个子集的可用掩码、概率向量、强度 → 空集分支可复算、可验有限性
    Q3-08  全量 CSV 补 gate/lomo 主导、是否一致、证据窗口、映射置信状态

尚未实现（已在报告中列为缺口，不做假占位）：
    Q3-06 打乱注意力的经验零分布。当前论文用理想零分布 1/4−1/(12R)=7/30 作参照，
    若要经验零分布，需要在 attention 上做置换后重跑遮蔽实验（faithfulness_test 不
    支持注意力覆盖，需另写循环）。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("MOSEI_ROOT") or Path(__file__).resolve().parents[2])
DATA = ROOT / "data"
OUT = ROOT / "output"
RESULTS = ROOT / "work" / "q3"   # 与 run_all 的结果目录一致（不另建 project/results）
WORK = ROOT / "work"
MODS = ("text", "audio", "vision")
POLARITY = ("negative", "neutral", "positive")


def _softmax1(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _residual_stats(pred: np.ndarray, true: np.ndarray,
                    cls_pred=None, cls_true=None) -> dict:
    r = pred.astype(np.float64) - true.astype(np.float64)
    at = np.abs(true.astype(np.float64))
    out = {
        "n": int(len(r)),
        "mae": float(np.abs(r).mean()),
        "mean_signed_residual": float(r.mean()),
        "median_signed_residual": float(np.median(r)),
        "signed_q10": float(np.quantile(r, 0.10)),
        "signed_q90": float(np.quantile(r, 0.90)),
        "abs_q50": float(np.quantile(np.abs(r), 0.50)),
        "abs_q90": float(np.quantile(np.abs(r), 0.90)),
        "abs_q99": float(np.quantile(np.abs(r), 0.99)),
        "abs_max": float(np.abs(r).max()),
        "share_abs_gt_1": float((np.abs(r) > 1.0).mean()),
    }
    a, b = np.abs(r), at
    out["corr_abs_resid_abs_true"] = (
        float(np.corrcoef(a, b)[0, 1])
        if len(r) > 2 and a.std() > 0 and b.std() > 0 else 0.0)
    out["by_true_polarity"] = {}
    for k, nm in enumerate(POLARITY):
        m = true.astype(np.int64) == k
        if m.any():
            out["by_true_polarity"][nm] = {
                "n": int(m.sum()),
                "mae": float(np.abs(r[m]).mean()),
                "mean_signed": float(r[m].mean()),
            }
    out["by_abs_true"] = {}
    for lo, hi in zip([0.0, 0.5, 1.0, 2.0], [0.5, 1.0, 2.0, 1e9]):
        m = (at >= lo) & (at < hi)
        if m.any():
            out["by_abs_true"][f"[{lo},{hi})"] = {
                "n": int(m.sum()), "mae": float(np.abs(r[m]).mean())}
    if cls_pred is not None and cls_true is not None and len(cls_pred) == len(r):
        m = np.asarray(cls_pred) == np.asarray(cls_true)
        out["mae_when_polarity_right"] = float(np.abs(r[m]).mean()) if m.any() else None
        out["mae_when_polarity_wrong"] = float(np.abs(r[~m]).mean()) if (~m).any() else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=str(OUT / "models" / "q3_model.pt"))
    args = ap.parse_args()

    import torch

    from .data import describe, fit_and_scale, load_attachment2, load_attachment4
    from .model import ModalityAttributionNet
    from .model_explain import build_explanations, faithfulness_test
    from .shapley import shapley_values, subset_outputs, targets_from
    from .train import evaluate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    splits = load_attachment2(DATA)
    att4 = load_attachment4(DATA)
    fit_and_scale(splits, att4)          # 统计量只在 train 上拟合
    describe(splits, att4)

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ModalityAttributionNet(ck["dims"], hidden=ck["hidden"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"检查点: {args.checkpoint}  hidden={ck['hidden']}  "
          f"best_epoch={ck.get('best_epoch')}")
    # 记录**相对路径**：绝对路径里含账号名，属身份信息，不能进交付件
    ck_rel = Path(args.checkpoint)
    try:
        ck_rel = ck_rel.resolve().relative_to(ROOT.resolve())
    except Exception:
        ck_rel = Path(ck_rel.name)
    rep: dict = {"checkpoint": str(ck_rel), "hidden": ck["hidden"],
                 "best_epoch": ck.get("best_epoch"), "device": str(device)}

    # ------------------------------------------------------- A 三个划分
    print("\n[A] 指标（附件2 三个划分；test 全程不参与选择）")
    rep["metrics"] = {}
    evals = {}
    for name in ("train", "valid", "test"):
        met, pcls, pval, gates = evaluate(model, splits[name], device)
        evals[name] = (met, pcls, pval, gates)
        d = dict(met.to_dict())
        d["confusion"] = met.confusion
        cm = np.array(met.confusion)
        d["per_class_recall"] = {
            nm: (float(cm[k, k] / cm[k].sum()) if cm[k].sum() else 0.0)
            for k, nm in enumerate(POLARITY)}
        d["role"] = ("参数学习" if name == "train" else
                     "开发集：选模与结构选择均在此集合" if name == "valid" else
                     "不参与任何选择的留出对照")
        rep["metrics"][name] = d
        print(f"  {name:5s} n={d['n']:5d} Acc={d['accuracy']:.4f} "
              f"MacroF1={d['macro_f1']:.4f} MAE={d['mae']:.4f} Pearson={d['pearson']:.4f}")

    # ------------------------------------------------------- B 回归残差
    print("\n[B] 回归残差分析")
    rep["residual_analysis"] = {}
    for name in ("valid", "test"):
        _m, pcls, pval, _g = evals[name]
        st = _residual_stats(pval, splits[name].intensity, pcls, splits[name].polarity)
        rep["residual_analysis"][name] = st
        print(f"  {name}: MAE={st['mae']:.4f}  平均有符号残差={st['mean_signed_residual']:+.4f}"
              f"  中位={st['median_signed_residual']:+.4f}")
        print(f"        |残差| q50/q90/q99/max = {st['abs_q50']:.4f}/{st['abs_q90']:.4f}/"
              f"{st['abs_q99']:.4f}/{st['abs_max']:.4f}  |残差|>1 占比={st['share_abs_gt_1']:.3f}"
              f"  corr(|残差|,|y|)={st['corr_abs_resid_abs_true']:+.4f}")
        print("        按真实极性 MAE: " + "  ".join(
            f"{k}={v['mae']:.4f}" for k, v in st["by_true_polarity"].items()))
        print("        按 |y| 分箱 MAE: " + "  ".join(
            f"{k}={v['mae']:.4f}" for k, v in st["by_abs_true"].items()))
        print(f"        分类判对/判错的回归 MAE = {st['mae_when_polarity_right']:.4f}"
              f" / {st['mae_when_polarity_wrong']:.4f}")

    # ------------------------------------------------------- C φ 与 mean|φ|
    print("\n[C] 精确 Shapley：有符号均值与平均绝对值")
    rep["shapley_abs"] = {}
    for name, sp in (("attachment2_valid", splits["valid"]), ("attachment4", att4)):
        store = subset_outputs(model, sp, device)
        f, cstar = targets_from(store)
        rep["shapley_abs"][name] = {
            "cstar_distribution": {str(k): int((cstar == k).sum()) for k in range(3)}}
        for tgt in ("prob_cstar", "intensity"):
            phi, resid = shapley_values(f[tgt])
            rep["shapley_abs"][name][tgt] = {
                m: {"mean": float(phi[m].mean()),
                    "mean_abs": float(np.abs(phi[m]).mean()),
                    "median": float(np.median(phi[m])),
                    "share_positive": float((phi[m] > 0).mean())}
                for m in MODS}
            rep["shapley_abs"][name][tgt]["efficiency_max_abs_error"] = \
                float(np.abs(resid).max())
            e = rep["shapley_abs"][name][tgt]
            print(f"  {name:18s} {tgt:11s} " + "  ".join(
                f"{m}: mean={e[m]['mean']:+.4f} mean|phi|={e[m]['mean_abs']:.4f}"
                for m in MODS))
        if name == "attachment4":
            ids = list(getattr(sp, "ids", [])) or [f"{i + 1:02d}" for i in range(len(sp))]
            rows = []
            for i in range(len(sp)):
                for S in store:
                    p = _softmax1(store[S]["logits"][i].astype(np.float64))
                    rows.append({
                        "sample_id": ids[i],
                        "subset": "+".join(S) if S else "empty",
                        "n_modalities": len(S),
                        "mask_text": int("text" in S),
                        "mask_audio": int("audio" in S),
                        "mask_vision": int("vision" in S),
                        "p_negative": round(float(p[0]), 6),
                        "p_neutral": round(float(p[1]), 6),
                        "p_positive": round(float(p[2]), 6),
                        "prob_sum": round(float(p.sum()), 9),
                        "intensity": round(float(store[S]["intensity"][i]), 6),
                        "fixed_cstar": int(cstar[i]),
                        "prob_cstar": round(float(p[cstar[i]]), 6),
                    })
            p8 = RESULTS / "附件4_8子集逐样本.csv"
            p8.parent.mkdir(parents=True, exist_ok=True)
            with p8.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            sums = np.array([r["prob_sum"] for r in rows])
            print(f"  已写出 {p8}（{len(rows)} 行 = 20 条 × 8 子集）；"
                  f"概率和最大偏差={np.abs(sums - 1).max():.2e}")
            rep["eight_subsets_att4"] = {
                "rows": len(rows),
                "prob_sum_max_abs_error": float(np.abs(sums - 1).max()),
                "intensity_finite": bool(np.isfinite(
                    np.array([r["intensity"] for r in rows])).all()),
            }

    # ------------------------------------------------------- D 全字段解释卡
    print("\n[D] 附件4 全字段解释卡")
    tmap = None
    tm_path = WORK / "q3" / "att4_time_map.json"
    try:
        from .evidence_time import load_time_map
        if tm_path.exists():
            tmap = load_time_map(tm_path)
            print(f"  复用时序映射 {tm_path}（{len(tmap)} 条）")
        else:
            print(f"  未找到 {tm_path}，时间字段将标记为近似来源缺失")
    except Exception as exc:
        print(f"  时序映射读取失败（继续）：{type(exc).__name__}: {exc}")

    exps, _fwd = build_explanations(model, att4, device, time_map=tmap)
    faith = faithfulness_test(model, att4, device, k=3, n_random=5)
    rows = []
    for e in exps:
        w = e.modality_weights
        order = sorted(w.values(), reverse=True)
        rows.append({
            "sample_id": e.sample_id,
            "polarity": e.polarity,
            "intensity": round(float(e.intensity), 4),
            "gate_dominant": e.dominant_modality,
            "gate_margin": round(order[0] - order[1], 4),
            "lomo_dominant": e.lomo_dominant,
            "lomo_margin": round(float(e.lomo_margin), 4),
            "gate_vs_lomo_agree": int(bool(e.corroborated)),
            "shapley_target_fixed": "prob_cstar(主) / intensity(次)，见 shapley_attribution.json",
            "evidence_position": int(e.evidence_position),
            "evidence_modality": e.evidence_modality,
            "evidence_span": "-".join(str(x) for x in e.evidence_span_positions),
            "evidence_start_s": round(float(e.evidence_start), 4),
            "evidence_end_s": round(float(e.evidence_end), 4),
            "evidence_word": e.evidence_word,
            "time_source": e.time_source,
            "time_confidence": ("近似定位，需回看确认"
                                if e.time_source == "forced_alignment"
                                else "未对齐，位置均分估计"),
            "fragment_level_claim": ("仅文本片段可主张忠实性"
                                     if e.evidence_modality == "text"
                                     else "视听片段不作单样本结论（见 7.3.3）"),
            "unavailable_modalities": "+".join(e.unavailable),
            "note": "" if e.corroborated else "模态区分度不足，不作强判断",
        })
    p = RESULTS / "附件4_预测与解释结果_全字段.csv"
    with p.open("w", newline="", encoding="utf-8-sig") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wtr.writeheader()
        wtr.writerows(rows)
    agree = sum(r["gate_vs_lomo_agree"] for r in rows)
    print(f"  已写出 {p}（{len(rows)} 行）；门控与留一扰动一致 {agree}/{len(rows)}")
    rep["explanations_all_fields"] = {
        "n": len(rows), "gate_vs_lomo_agree": int(agree),
        "unavailable": sum(1 for r in rows if r["unavailable_modalities"]),
        "time_source_counts": {k: sum(1 for r in rows if r["time_source"] == k)
                               for k in sorted({r["time_source"] for r in rows})},
    }
    rep["faithfulness_att4_recomputed"] = {
        m: {"share_random_between": float(faith["per_modality"][m]["share_random_between"]),
            "n_low": int(faith["per_modality"][m]["n_samples_evaluated_low"])}
        for m in MODS}

    # ------------------------------------------------------- E 描述性统计
    print("\n[E] 门控 / 留一扰动 / 时序的描述性统计（供论文表 24、图 17、7.3.4 引用）")
    from .model_explain import leave_one_modality_out
    rep["descriptive"] = {"gate": {}, "lomo": {}}
    for name, sp in (("attachment2_valid", splits["valid"]), ("attachment4", att4)):
        _m, _c, _v, g = evaluate(model, sp, device)
        rep["descriptive"]["gate"][name] = {
            m: {"mean": round(float(g[:, k].mean()), 4),
                "median": round(float(np.median(g[:, k])), 4),
                "max": round(float(g[:, k].max()), 4),
                "min": round(float(g[:, k].min()), 4)}
            for k, m in enumerate(MODS)}
        L = leave_one_modality_out(model, sp, device)
        am = L.argmax(axis=1)
        rep["descriptive"]["lomo"][name] = {
            m: {"mean": round(float(L[:, k].mean()), 4),
                "median": round(float(np.median(L[:, k])), 4),
                "n_argmax": int((am == k).sum()),
                "share_argmax": round(float((am == k).mean()), 4)}
            for k, m in enumerate(MODS)}
        print(f"  {name:18s} 门控均值 " + "  ".join(
            f"{m}={rep['descriptive']['gate'][name][m]['mean']:.4f}" for m in MODS))
        print(f"  {name:18s} 留一均值 " + "  ".join(
            f"{m}={rep['descriptive']['lomo'][name][m]['mean']:.4f}" for m in MODS)
            + "；逐样本居首 " + "  ".join(
            f"{m}={rep['descriptive']['lomo'][name][m]['n_argmax']}" for m in MODS))

    results_dir = RESULTS  # 与 run_all 一致：work/q3
    tm = WORK / "q3" / "att4_time_map.json"
    if tm.exists():
        raw = json.loads(tm.read_text(encoding="utf-8"))
        # 逐位置溯源表：位置 → 子词 → 词 → 秒区间（含特殊词元与填充位）
        rows_p = []
        for sid in sorted(raw):
            v = raw[sid]
            words = v.get("words", [])
            pieces = v.get("pieces", [])
            pw = v.get("piece_word", [])
            n_content = len(pieces)
            for p in range(50):
                if p == 0:
                    piece, wi = "[CLS]", ""
                elif 1 <= p <= n_content:
                    piece = pieces[p - 1]
                    wi = pw[p - 1] if p - 1 < len(pw) else ""
                elif p == n_content + 1:
                    piece, wi = "[SEP]", ""
                else:
                    piece, wi = "[PAD]", ""
                if isinstance(wi, int) and 0 <= wi < len(words):
                    w = words[wi]
                    ws, we = w["start"], w["end"]
                    wd = w["w"]
                else:
                    ws = we = ""
                    wd = ""
                rows_p.append({
                    "sample_id": sid, "position": p, "piece": piece,
                    "position_role": ("[CLS]" if p == 0 else
                                      "[SEP]" if p == n_content + 1 else
                                      "content" if 1 <= p <= n_content else "padding"),
                    "word_index": wi, "word": wd,
                    "start_sec": ws, "end_sec": we,
                    "n_positions": 50, "n_content_subwords": n_content,
                    "n_words": len(words), "duration_sec": v["duration"],
                    "mapping_verified": int(bool(v["quality"].get("token_map_verified"))),
                })
        p_csv = results_dir / "position_time_map.csv"
        with p_csv.open("w", newline="", encoding="utf-8") as fh:
            w2 = csv.DictWriter(fh, fieldnames=list(rows_p[0]))
            w2.writeheader()
            w2.writerows(rows_p)
        n_content_rows = sum(1 for r in rows_p if r["position_role"] == "content")
        print(f"  已写出 {p_csv}（{len(rows_p)} 行 = 20 条 × 50 位置，"
              f"其中内容位 {n_content_rows}）")

    if tm.exists():
        raw = json.loads(tm.read_text(encoding="utf-8"))
        dur = np.array([v["duration"] for v in raw.values()], dtype=float)
        wps = np.array([v["n_words"] / v["duration"] for v in raw.values()], dtype=float)
        cov = np.array([(v["quality"].get("coverage") if v["quality"].get("coverage")
                         is not None else np.nan) for v in raw.values()], dtype=float)
        nsub = np.array([len(v.get("pieces", [])) for v in raw.values()], dtype=float)
        nver = sum(1 for v in raw.values() if v["quality"].get("token_map_verified"))
        rep["descriptive"]["timing"] = {
            "n_aligned": len(raw),
            "coverage_min": float(np.nanmin(cov)) if np.isfinite(cov).any() else None,
            "coverage_median": float(np.nanmedian(cov)) if np.isfinite(cov).any() else None,
            "wps_min": round(float(wps.min()), 4),
            "wps_max": round(float(wps.max()), 4),
            "wps_median": round(float(np.median(wps)), 4),
            "duration_min": round(float(dur.min()), 4),
            "duration_max": round(float(dur.max()), 4),
            "n_subwords_min": int(nsub.min()),
            "n_subwords_max": int(nsub.max()),
            "token_map_verified": f"{nver}/{len(raw)}",
        }
        print(f"  附件4 时长 {dur.min():.3f}~{dur.max():.3f}s；语速 "
              f"{wps.min():.2f}~{wps.max():.2f}（中位 {np.median(wps):.2f}）词/秒；"
              f"覆盖率中位 {np.nanmedian(cov):.4f}；子词数 {int(nsub.min())}~{int(nsub.max())}；"
              f"映射核对 {nver}/{len(raw)}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    dst = RESULTS / "q3_audit_extra.json"
    dst.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {dst}")
    print("提示：本脚本不训练、不覆盖既有结果文件；论文引用请用本文件的数字并标注口径。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
