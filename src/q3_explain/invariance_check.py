# -*- coding: utf-8 -*-
"""问题三：编码器掩码与无效位置不被读取的**验收测试**（审查意见 Q3-01 的验收项）。

审查方要求的四条验收，逐条在这里实现并可复算：

    A. 任意改变无效位置载荷不改变预测
    B. 增加无效填充不改变有效位置的响应
    C. 任何被屏蔽的位置不进入池化，也不进入证据
    D. 整路缺失仍产生有限值

另加两条自证：
    E. 关键证据/遮蔽检验的候选位不含 [CLS]/[SEP]
    F. 子词→词映射与数据自带 token id 逐位一致

用法：
    python -m q3_explain.invariance_check
输出：
    work/q3/invariance_check.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("MOSEI_ROOT") or Path(__file__).resolve().parents[2])
DATA = ROOT / "data"
OUT = ROOT / "output"
RESULTS = ROOT / "work" / "q3"


def main() -> int:
    import torch

    from .data import MODALITIES, fit_and_scale, load_attachment2, load_attachment4
    from .evidence_time import load_time_map
    from .model import ModalityAttributionNet
    from .model_explain import build_explanations, forward_all
    from .wordpiece import load_vocab, map_words_to_pieces

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    splits = load_attachment2(DATA)
    att4 = load_attachment4(DATA)
    fit_and_scale(splits, att4)
    ck = torch.load(OUT / "models" / "q3_model.pt", map_location=device,
                    weights_only=False)
    model = ModalityAttributionNet(ck["dims"], hidden=ck["hidden"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"设备 {device}；检查点 hidden={ck['hidden']} best_epoch={ck.get('best_epoch')}")

    ckpt_path = OUT / "models" / "q3_model.pt"
    rep: dict = {"checkpoint": ckpt_path.relative_to(ROOT).as_posix(),
                 "hidden": ck["hidden"], "best_epoch": ck.get("best_epoch"),
                 "checks": {}}
    va = splits["valid"]
    idx = np.arange(min(256, len(va)))
    batch = {m: torch.from_numpy(np.asarray(getattr(va, m))[idx]).to(device)
             for m in MODALITIES}
    masks = {m: torch.from_numpy(va.masks[m][idx]).to(device) for m in MODALITIES}

    with torch.no_grad():
        base = model(batch, masks)
        base_int = base.intensity.cpu().numpy()
        base_logits = base.polarity_logits.cpu().numpy()
        base_att = {m: base.attention[m].cpu().numpy() for m in MODALITIES}

    # ---------------------------------------------------------------- A
    worst_d = 0.0
    worst_l = 0.0
    for m in MODALITIES:
        pad = ~masks[m]
        if not bool(pad.any()):
            continue
        noisy = {mm: batch[mm].clone() for mm in MODALITIES}
        noise = torch.from_numpy(
            rng.normal(0, 10.0, size=tuple(batch[m].shape)).astype(np.float32)).to(device)
        noisy[m] = torch.where(pad.unsqueeze(-1), noise, batch[m])
        with torch.no_grad():
            o = model(noisy, masks)
        worst_d = max(worst_d, float(np.abs(o.intensity.cpu().numpy() - base_int).max()))
        worst_l = max(worst_l, float(np.abs(
            o.polarity_logits.cpu().numpy() - base_logits).max()))
    rep["checks"]["A_padding_payload_irrelevant"] = {
        "how": "把不可观测位置的特征整行换成 N(0,10) 噪声后重算（三个模态各一次）",
        "max_abs_intensity_change": worst_d,
        "max_abs_logit_change": worst_l,
        "passed": bool(worst_d < 1e-5 and worst_l < 1e-4),
    }
    print(f"[A] 无效位置载荷换噪声 → Δ强度 max={worst_d:.3e} Δlogit max={worst_l:.3e} "
          f"{'通过' if rep['checks']['A_padding_payload_irrelevant']['passed'] else '未通过'}")

    # ---------------------------------------------------------------- B
    # 把有效位之后的填充整段右移一位（保持掩码不变），有效位的隐藏状态不应改变
    shift_bad = 0.0
    for m in MODALITIES:
        shifted = {mm: batch[mm].clone() for mm in MODALITIES}
        x = shifted[m].clone()
        rolled = torch.roll(x, shifts=1, dims=1)
        keep = masks[m].unsqueeze(-1)
        shifted[m] = torch.where(keep, x, rolled)
        with torch.no_grad():
            o = model(shifted, masks)
        shift_bad = max(shift_bad, float(np.abs(
            o.intensity.cpu().numpy() - base_int).max()))
    rep["checks"]["B_padding_shift_irrelevant"] = {
        "how": "把不可观测位置整体循环右移一位（有效位保持原值）后重算",
        "max_abs_intensity_change": shift_bad,
        "passed": bool(shift_bad < 1e-5),
    }
    print(f"[B] 填充循环右移 → Δ强度 max={shift_bad:.3e} "
          f"{'通过' if rep['checks']['B_padding_shift_irrelevant']['passed'] else '未通过'}")

    # ---------------------------------------------------------------- C
    # 池化权重在被屏蔽位置上必须恒为 0；关键证据必须落在候选位上
    att_out = 0.0
    for m in MODALITIES:
        a = base_att[m]
        att_out = max(att_out, float(np.abs(a[~masks[m].cpu().numpy()]).max()))
    tmap_path = RESULTS / "att4_time_map.json"
    exps_none = build_explanations(model, att4, device,
                                   time_map=load_time_map(tmap_path) if tmap_path.exists()
                                   else None)[0]
    bad_pos: list = []
    for i, e in enumerate(exps_none):
        cand = att4.content_of(e.evidence_modality)[i]
        if not bool(cand[e.evidence_position]):
            bad_pos.append(e.sample_id)
        if any(not bool(cand[p]) for p in e.evidence_span_positions):
            bad_pos.append(e.sample_id + "(span)")
    rep["checks"]["C_masked_positions_never_enter_pooling_or_evidence"] = {
        "how": "检查池化注意力在不可观测位置上的最大值；检查每条解释的峰值与区间是否都落在候选位",
        "max_attention_on_masked_position": att_out,
        "evidence_outside_candidates": sorted(set(bad_pos)),
        "passed": bool(att_out < 1e-7 and not bad_pos),
    }
    print(f"[C] 被屏蔽位置的池化权重 max={att_out:.3e}；证据越界 {sorted(set(bad_pos))} "
          f"{'通过' if rep['checks']['C_masked_positions_never_enter_pooling_or_evidence']['passed'] else '未通过'}")

    # ---------------------------------------------------------------- D
    # 整路缺失 → 有限值；并把三种缺失形态（单路/双路/三路）都跑一遍
    finite_ok = True
    detail = {}
    for combo in [("audio", "vision"), ("text",), ("vision",)]:
        b2 = {m: (torch.zeros_like(batch[m]) if m in combo else batch[m])
              for m in MODALITIES}
        m2 = {m: (torch.zeros_like(masks[m]) if m in combo else masks[m])
              for m in MODALITIES}
        with torch.no_grad():
            o = model(b2, m2)
        f_ok = bool(torch.isfinite(o.intensity).all()
                    and torch.isfinite(o.polarity_logits).all())
        g = o.gate.cpu().numpy()
        zero_ok = bool((g[:, [MODALITIES.index(m) for m in combo]] == 0).all())
        detail["+".join(combo)] = {"all_finite": f_ok, "gate_zero_for_removed": zero_ok,
                                   "intensity_range": [float(o.intensity.min()),
                                                       float(o.intensity.max())]}
        finite_ok = finite_ok and f_ok and zero_ok
    # 三模态全空：门控走"全空兜底"（均匀），但池化向量全为零，故融合表示为 0，
    # 输出与门控取值无关——这一点必须单独验证，否则会把兜底当成缺陷。
    allzero_b = {m: torch.zeros_like(batch[m]) for m in MODALITIES}
    allzero_m = {m: torch.zeros_like(masks[m]) for m in MODALITIES}
    with torch.no_grad():
        o = model(allzero_b, allzero_m)
        pooled_zero = all(bool(torch.all(v == 0))
                          for v in model(allzero_b, allzero_m).pooled.values())
    finite_all = bool(torch.isfinite(o.intensity).all()
                      and torch.isfinite(o.polarity_logits).all())
    detail["all_three"] = {
        "all_finite": finite_all,
        "pooled_all_zero": pooled_zero,
        "gate_uniform_fallback": bool(np.allclose(
            o.gate.cpu().numpy(), 1.0 / len(MODALITIES), atol=1e-6)),
        "intensity_constant": float(o.intensity.mean()),
        "note": ("三路全空时门控走均匀兜底（避免全 -inf softmax），但三个池化向量"
                 "都是零向量，融合表示恒为零，因此输出与门控取值无关"),
    }
    rep["checks"]["D_missing_modalities_stay_finite"] = {
        "how": "整路置零（含三路同时置零）后前向，检查有限性、门控归零与全空时融合表示为 0",
        "detail": detail,
        "passed": bool(finite_ok and finite_all and pooled_zero),
    }
    print(f"[D] 整路缺失有限性: {detail} "
          f"{'通过' if rep['checks']['D_missing_modalities_stay_finite']['passed'] else '未通过'}")

    # ---------------------------------------------------------------- E
    spec_bad = []
    for m in ("text",):
        c = att4.content_of(m)
        mk = att4.masks[m]
        for i in range(len(att4)):
            if c[i].sum() != max(0, mk[i].sum() - 2):
                spec_bad.append(att4.ids[i])
    rep["checks"]["E_special_tokens_excluded"] = {
        "how": "文本候选位数必须等于可观测位数减 2（[CLS] 与 [SEP]）",
        "violations": spec_bad,
        "passed": not spec_bad,
    }
    print(f"[E] [CLS]/[SEP] 排除: 违规 {spec_bad} "
          f"{'通过' if rep['checks']['E_special_tokens_excluded']['passed'] else '未通过'}")

    # ---------------------------------------------------------------- F
    tmap_path = RESULTS / "att4_time_map.json"
    tm_ok = 0
    n_map = 0
    if tmap_path.exists():
        loaded = load_time_map(tmap_path)
        n_map = len(loaded)
        tm_ok = sum(1 for v in loaded.values() if v.quality.get("token_map_verified"))
    rep["checks"]["F_subword_to_word_verified"] = {
        "how": "附件4 逐条比对重建 token id 与 text_bert[0]（见 evidence_time 的 quality 字段）",
        "verified": f"{tm_ok}/{n_map}",
        "mapping_verified_in_explanations": sum(1 for e in exps_none if e.mapping_verified),
        "n_explanations": len(exps_none),
        "passed": bool(n_map > 0 and tm_ok == n_map),
    }
    print(f"[F] 子词→词映射: {rep['checks']['F_subword_to_word_verified']['verified']} "
          f"{'通过' if rep['checks']['F_subword_to_word_verified']['passed'] else '未通过'}")

    # ---------------------------------------------------------------- G
    # Shapley 的语义验收（回应审查意见 Q3-04）：
    #  (1) 手算反例：概率非负**不能**推出 Shapley 值非负；给一个所有子集输出都非负、
    #      但某个参与者 φ 为负的可手算博弈，与解析值逐位核对；
    #  (2) 常数平移不变性：对全部 f(S) 加同一常数，φ 必须不变（同一参考点平移）；
    #  (3) 真实 8 子集响应：全部有限、概率和为 1、效率性恒等式成立。
    def _shapley(vals: dict, players) -> dict:
        """n≤3 的精确 Shapley：φ_p = Σ_{S⊆N\\{p}} |S|!(n−|S|−1)!/n! · [f(S∪p) − f(S)]。

        n=3 时权重即「空集与两元素子集 1/3、单元素子集 1/6」，与论文正文一致。
        """
        import itertools
        import math
        n = len(players)
        out = {}
        for p in players:
            others = [q for q in players if q != p]
            tot = 0.0
            for r in range(n):
                for comb in itertools.combinations(others, r):
                    w = (math.factorial(r) * math.factorial(n - r - 1)
                         / math.factorial(n))
                    tot += w * (vals[frozenset(comb) | {p}] - vals[frozenset(comb)])
            out[p] = tot
        return out

    players = ("T", "A", "V")
    demo = {frozenset(): 0.7, frozenset(("T",)): 0.6, frozenset(("A",)): 0.7,
            frozenset(("V",)): 0.7, frozenset(("T", "A")): 0.6,
            frozenset(("T", "V")): 0.6, frozenset(("A", "V")): 0.7,
            frozenset(("T", "A", "V")): 0.6}
    phi_demo = _shapley(demo, players)
    shift = 3.0
    demo_shift = {k: v + shift for k, v in demo.items()}
    phi_shift = _shapley(demo_shift, players)
    all_nonneg = all(v >= 0 for v in demo.values())
    hand_ok = (abs(phi_demo["T"] + 0.1) < 1e-12 and abs(phi_demo["A"]) < 1e-12
               and abs(phi_demo["V"]) < 1e-12)
    shift_ok = all(abs(phi_demo[p] - phi_shift[p]) < 1e-12 for p in players)
    rep["checks"]["G_shapley_semantics"] = {
        "how": ("手算 3 参与者博弈 f(S)=0.7−0.1·1[T∈S]（全部子集输出非负）；核对可手算的 "
                "φ_T=−0.1 与常数平移（+3）不变性；再核对真实 8 子集响应的有限性与概率和"),
        "synthetic_all_values_nonnegative": all_nonneg,
        "synthetic_phi": {k: round(v, 6) for k, v in phi_demo.items()},
        "synthetic_hand_calculation_matches": bool(hand_ok),
        "constant_shift_invariance": bool(shift_ok),
    }
    # (3) 真实 8 子集响应
    sub_path = RESULTS / "附件4_8子集逐样本.csv"
    finite = sum_ok = 0
    rows = 0
    if sub_path.exists():
        import csv
        with sub_path.open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                rows += 1
                probs = []
                for k in ("p_negative", "p_neutral", "p_positive"):
                    if k in row and row[k] not in ("", None):
                        probs.append(float(row[k]))
                if probs and all(np.isfinite(probs)):
                    finite += 1
                    # 表内概率只保留 6 位小数，故和与 1 的偏差可达 ~1.5e-6
                    if abs(sum(probs) - 1.0) < 5e-6:
                        sum_ok += 1
    rep["checks"]["G_shapley_semantics"].update({
        "eight_subset_rows": rows,
        "rows_with_finite_probabilities": finite,
        "rows_with_probability_sum_one": sum_ok,
        "passed": bool(hand_ok and shift_ok and finite == rows and rows > 0
                       and sum_ok == rows),
    })
    print(f"[G] Shapley 语义: 手算 φ=(−0.1,0,0) {hand_ok}；常数平移不变 {shift_ok}；"
          f"8 子集 {rows} 行全部有限且概率和为 1: {finite == rows and sum_ok == rows} "
          f"{'通过' if rep['checks']['G_shapley_semantics']['passed'] else '未通过'}")

    rep["all_passed"] = all(v.get("passed") for v in rep["checks"].values())
    RESULTS.mkdir(parents=True, exist_ok=True)
    dst = RESULTS / "invariance_check.json"
    dst.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {dst}")
    print(f"全部通过: {rep['all_passed']}")
    return 0 if rep["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
