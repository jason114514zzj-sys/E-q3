# -*- coding: utf-8 -*-
"""问题三：多种子稳定性（回应审查意见 Q3-10 的"单一种子不支持稳定性结论"）。

做法：同一冻结配置下重训若干个随机种子，逐个在**验证集**（开发集）与**测试划分**（留出）
上评价，并重算忠实度三段序、留一扰动均值与门控均值，最后给出跨种子均值/标准差。

纪律：
* **绝不覆盖**任何既有交付件——检查点写进 `work/q3/_seed_tmp/seed{S}/`；
* 只把汇总写进 `work/q3/seed_stability.json`；
* 第一个种子就是论文正在用的 20260924，用于**自证**：它的数字必须与 q3_report.json 逐位一致，
  否则说明本脚本与主流程口径不同，必须停下来查。

用法：
    python -m q3_explain.seed_stability
    python -m q3_explain.seed_stability --seeds 20260924 20260925 20260926 1 2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("MOSEI_ROOT") or Path(__file__).resolve().parents[2])
DATA = ROOT / "data"
OUT = ROOT / "output"
RESULTS = ROOT / "work" / "q3"
TMP = RESULTS / "_seed_tmp"
MODS = ("text", "audio", "vision")
DEFAULT_SEEDS = [20260924, 20260925, 20260926, 1, 2]


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--keep-checkpoints", action="store_true")
    args = ap.parse_args()

    from .data import fit_and_scale, load_attachment2, load_attachment4
    from .model import ModalityAttributionNet
    from .model_explain import faithfulness_test, leave_one_modality_out
    from .train import evaluate, train_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    splits = load_attachment2(DATA)
    att4 = load_attachment4(DATA)
    fit_and_scale(splits, att4)      # 统计量只在 train 上拟合，全部种子共用（与主流程一致）

    rep: dict = {"seeds": [int(s) for s in args.seeds], "epochs": args.epochs,
                 "device": str(device), "per_seed": {}}
    for seed in args.seeds:
        d = TMP / f"seed{seed}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        summ = train_model(splits, device, d, hidden=96, epochs=args.epochs,
                           seed=int(seed), verbose=False)
        ck = torch.load(d / "q3_model.pt", map_location=device, weights_only=False)
        model = ModalityAttributionNet(ck["dims"], hidden=ck["hidden"]).to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()

        met_v, _pv, _val, gate_v = evaluate(model, splits["valid"], device)
        met_t, _pt, _vt, _gt = evaluate(model, splits["test"], device)
        faith = faithfulness_test(model, splits["valid"], device, k=3, n_random=5)
        lomo = leave_one_modality_out(model, splits["valid"], device)

        ent = {
            "best_epoch": summ["best_epoch"],
            "epochs_run": len(summ["history"]),
            "valid": met_v.to_dict(),
            "test": met_t.to_dict(),
            "faithfulness_strict_share": {
                m: round(faith["per_modality"][m]["share_random_between"], 4) for m in MODS},
            "faithfulness_gap_top_rand": {
                m: round(faith["per_modality"][m]["gap_topk_minus_random"], 4) for m in MODS},
            "faithfulness_gap_rand_low": {
                m: round(faith["per_modality"][m]["gap_random_minus_lowk"], 4) for m in MODS},
            "faithfulness_n_low": {
                m: faith["per_modality"][m]["n_samples_evaluated_low"] for m in MODS},
            "lomo_mean": {m: round(float(lomo[:, k].mean()), 4) for k, m in enumerate(MODS)},
            "gate_mean": {m: round(float(gate_v[:, k].mean()), 4) for k, m in enumerate(MODS)},
        }
        rep["per_seed"][str(seed)] = ent
        print(f"seed {seed}: best_ep={ent['best_epoch']:2d} run={ent['epochs_run']:2d} "
              f"valid Acc={ent['valid']['accuracy']:.4f} F1={ent['valid']['macro_f1']:.4f} "
              f"MAE={ent['valid']['mae']:.4f} r={ent['valid']['pearson']:.4f} | "
              f"test F1={ent['test']['macro_f1']:.4f} | "
              f"三段序 文本={ent['faithfulness_strict_share']['text']:.3f} "
              f"视觉={ent['faithfulness_strict_share']['vision']:.3f} "
              f"语音={ent['faithfulness_strict_share']['audio']:.3f}")

    # ---------------- 汇总：均值 / 标准差 / 极差 ----------------
    def agg(path):
        vals = np.array([_dig(rep["per_seed"][s], path) for s in rep["per_seed"]], dtype=float)
        return {"mean": round(float(vals.mean()), 4),
                "std": round(float(vals.std(ddof=1)), 4) if vals.size > 1 else 0.0,
                "min": round(float(vals.min()), 4), "max": round(float(vals.max()), 4),
                "values": [round(float(v), 4) for v in vals]}

    def _dig(d, key):
        cur = d
        for k in key.split("."):
            cur = cur[k]
        return cur

    keys = ["best_epoch", "epochs_run",
            "valid.accuracy", "valid.macro_f1", "valid.mae", "valid.pearson",
            "test.accuracy", "test.macro_f1", "test.mae", "test.pearson",
            "faithfulness_strict_share.text", "faithfulness_strict_share.vision",
            "faithfulness_strict_share.audio",
            "faithfulness_gap_top_rand.text", "faithfulness_gap_rand_low.text",
            "lomo_mean.text", "lomo_mean.vision", "lomo_mean.audio",
            "gate_mean.text", "gate_mean.vision", "gate_mean.audio"]
    rep["aggregate"] = {k: agg(k) for k in keys}

    # ---------------- 自证：第一个种子必须与主流程一致 ----------------
    primary = str(rep["seeds"][0])
    ref_path = RESULTS / "q3_report.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        mine = rep["per_seed"][primary]
        checks = {
            "valid_accuracy": (mine["valid"]["accuracy"], ref["training"]["valid_metrics"]["accuracy"]),
            "valid_macro_f1": (mine["valid"]["macro_f1"], ref["training"]["valid_metrics"]["macro_f1"]),
            "valid_mae": (mine["valid"]["mae"], ref["training"]["valid_metrics"]["mae"]),
            "valid_pearson": (mine["valid"]["pearson"], ref["training"]["valid_metrics"]["pearson"]),
            "best_epoch": (mine["best_epoch"], ref["training"]["best_epoch"]),
        }
        same = {k: bool(a == b) for k, (a, b) in checks.items()}
        rep["self_check_vs_main_pipeline"] = {
            "primary_seed": int(primary), "compare": {k: {"seed_run": a, "q3_report": b}
                                                      for k, (a, b) in checks.items()},
            "all_equal": all(same.values())}
        print("\n自证（主种子与 q3_report.json 对比）：" + json.dumps(same, ensure_ascii=False))
        if not all(same.values()):
            print("⚠️ 不一致：本脚本与原主流程口径不同，结果不可直接引用！")

    RESULTS.mkdir(parents=True, exist_ok=True)
    dst = RESULTS / "seed_stability.json"
    dst.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {dst}")

    if not args.keep_checkpoints:
        shutil.rmtree(TMP, ignore_errors=True)
        print(f"已清理临时检查点 {TMP}")
    print("\n=== 跨种子汇总（mean ± std）===")
    for k, v in rep["aggregate"].items():
        print(f"  {k:42s} {v['mean']:.4f} ± {v['std']:.4f}   [{v['min']:.4f}, {v['max']:.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
