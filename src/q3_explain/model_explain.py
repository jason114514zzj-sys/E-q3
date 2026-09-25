"""Turn the model's own attention and gating into reviewable explanations.

The problem asks for three things, all of which this module produces:

* **主要参考模态** -- argmax of the modality gate
* **模态作用程度** -- the gate vector, corroborated by a leave-one-modality-out
  (LOMO) perturbation so the claim is not just the model's self-report
* **关键证据定位** -- the highest-attention position, converted to seconds via
  the clip's real duration (attachment 4 ships the 20 mp4 files)

Two honesty measures run alongside:

1. **Faithfulness of attention.**  Attention is not evidence by itself.  Three
   groups are masked in turn -- the ``k`` highest-attention valid positions, the
   ``k`` lowest-attention valid positions (disjoint from the top-k), and ``k``
   randomly chosen valid positions -- and the prediction change is compared
   across them.  A faithful explanation must order the damage as
   ``top > rand > low``.  The low group is what rules out the trivial reading
   "any three positions matter": masking the *least* attended positions should
   hurt the least.  This is the "可复核" (re-checkable) requirement made
   concrete: anyone can rerun the three perturbations.

2. **Position-to-word caveat.**  Verified on attachment 4: the text attention
   mask counts BERT *subword* slots (24 where raw_text has 21 words; 50 where it
   has 51, i.e. truncated).  So a text position maps to a word only
   approximately, and the code says so instead of implying an exact token index.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .data import MODALITIES, PROBLEM_MAX_POSITIONS, Split
from .model import ModalityAttributionNet


@dataclass
class Explanation:
    sample_id: str
    polarity: str
    intensity: float
    polarity_probs: list[float]
    dominant_modality: str
    modality_weights: dict[str, float]
    evidence_modality: str
    evidence_position: int
    evidence_start: float
    evidence_end: float
    evidence_span_positions: list[int]
    text_excerpt: str = ""
    word_estimate: str = ""
    lomo_drops: dict[str, float] = field(default_factory=dict)
    attention_peaks: dict[str, list[int]] = field(default_factory=dict)
    # Does the LOMO perturbation pick the same dominant modality as the gate?
    # Reported so a self-report is never presented as independently verified.
    corroborated: bool = True
    lomo_dominant: str = ""
    lomo_margin: float = 0.0
    unavailable: list[str] = field(default_factory=list)
    # ---- how the evidence time was obtained -------------------------------
    # "forced_alignment": word span from aligning raw_text to the clip audio
    # "uniform_estimate": the rejected j*D/50 mapping, kept only for comparison
    time_source: str = "uniform_estimate"
    evidence_word: str = ""
    evidence_word_index: int = -1
    naive_start: float = 0.0
    naive_end: float = 0.0

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "polarity": self.polarity,
            "intensity": round(float(self.intensity), 4),
            "polarity_probs": [round(float(x), 4) for x in self.polarity_probs],
            "dominant_modality": self.dominant_modality,
            "modality_weights": {k: round(float(v), 4)
                                 for k, v in self.modality_weights.items()},
            "evidence_modality": self.evidence_modality,
            "evidence_position": int(self.evidence_position),
            "evidence_start": round(float(self.evidence_start), 3),
            "evidence_end": round(float(self.evidence_end), 3),
            "evidence_span_positions": [int(x) for x in self.evidence_span_positions],
            "text_excerpt": self.text_excerpt,
            "word_estimate": self.word_estimate,
            "lomo_drops": {k: round(float(v), 4) for k, v in self.lomo_drops.items()},
            "attention_peaks": {k: [int(x) for x in v]
                                for k, v in self.attention_peaks.items()},
            "corroborated": bool(self.corroborated),
            "lomo_dominant": self.lomo_dominant,
            "lomo_margin": round(float(self.lomo_margin), 4),
            "unavailable": list(self.unavailable),
            "time_source": self.time_source,
            "evidence_word": self.evidence_word,
            "evidence_word_index": int(self.evidence_word_index),
            "naive_start": round(float(self.naive_start), 3),
            "naive_end": round(float(self.naive_end), 3),
        }


POLARITY_NAMES = ("negative", "neutral", "positive")


@torch.no_grad()
def forward_all(model: ModalityAttributionNet, sp: Split, device,
                batch_size: int = 256) -> dict:
    """Run the whole split, returning predictions, gates and attention."""
    model.eval()
    n = len(sp)
    out = {
        "logits": np.zeros((n, 3), dtype=np.float32),
        "intensity": np.zeros(n, dtype=np.float32),
        "gate": np.zeros((n, len(MODALITIES)), dtype=np.float32),
        "attention": {m: np.zeros((n, PROBLEM_MAX_POSITIONS), dtype=np.float32)
                      for m in MODALITIES},
    }
    for start in range(0, n, batch_size):
        idx = np.arange(start, min(start + batch_size, n))
        batch = {m: torch.from_numpy(np.asarray(getattr(sp, m))[idx]).to(device)
                 for m in MODALITIES}
        masks = {m: torch.from_numpy(sp.masks[m][idx]).to(device)
                 for m in MODALITIES}
        o = model(batch, masks)
        out["logits"][idx] = o.polarity_logits.cpu().numpy()
        out["intensity"][idx] = o.intensity.cpu().numpy()
        out["gate"][idx] = o.gate.cpu().numpy()
        for m in MODALITIES:
            out["attention"][m][idx] = o.attention[m].cpu().numpy()
    return out


@torch.no_grad()
def leave_one_modality_out(model: ModalityAttributionNet, sp: Split, device,
                           batch_size: int = 128) -> np.ndarray:
    """(N,3) drop in predicted intensity when each modality is removed.

    This is the empirical counterpart of the gate: if the gate says text matters
    most, zeroing text should hurt more than zeroing audio or vision.  Where the
    two disagree the explanation is called unreliable rather than reported as if
    it were solid.
    """
    model.eval()
    n = len(sp)
    base = np.zeros(n, dtype=np.float32)
    drops = np.zeros((n, len(MODALITIES)), dtype=np.float32)

    for start in range(0, n, batch_size):
        idx = np.arange(start, min(start + batch_size, n))
        batch = {m: torch.from_numpy(np.asarray(getattr(sp, m))[idx]).to(device)
                 for m in MODALITIES}
        masks = {m: torch.from_numpy(sp.masks[m][idx]).to(device)
                 for m in MODALITIES}
        base[idx] = model(batch, masks).intensity.cpu().numpy()
        for k, m in enumerate(MODALITIES):
            ablated_masks = dict(masks)
            ablated_masks[m] = torch.zeros_like(masks[m])
            ablated = {mm: (torch.zeros_like(batch[mm]) if mm == m else batch[mm])
                       for mm in MODALITIES}
            pred = model(ablated, ablated_masks).intensity.cpu().numpy()
            drops[idx, k] = np.abs(pred - base[idx])
    return drops


def _span_around(peak: int, length: int = 3) -> list[int]:
    half = length // 2
    lo = max(0, peak - half)
    hi = min(PROBLEM_MAX_POSITIONS - 1, lo + length - 1)
    lo = max(0, hi - length + 1)
    return list(range(lo, hi + 1))


def build_explanations(model: ModalityAttributionNet, sp: Split, device,
                       top_k_positions: int = 3,
                       time_map: dict | None = None) -> tuple[list[Explanation], dict]:
    """One explanation per sample, plus the raw arrays for reporting.

    ``time_map`` carries word-level forced alignments (sample_id -> ClipTimeMap).
    When it is supplied, an evidence position is converted to seconds through
    the *word* that position corresponds to, because the 50-slot axis was
    measured to be a token axis rather than an even time grid.  Without it the
    function falls back to the naive ``j*D/50`` estimate and labels the result
    ``time_source="uniform_estimate"`` so nobody mistakes it for a measurement.
    """
    fwd = forward_all(model, sp, device)
    lomo = leave_one_modality_out(model, sp, device)

    probs = torch.softmax(torch.from_numpy(fwd["logits"]), dim=1).numpy()
    labels = fwd["logits"].argmax(axis=1)

    exps: list[Explanation] = []
    for i in range(len(sp)):
        sid = sp.ids[i] if i < len(sp.ids) else str(i)
        w = {m: float(fwd["gate"][i, k]) for k, m in enumerate(MODALITIES)}
        dom = max(w, key=w.get)

        # evidence comes from the dominant modality's attention.  Masked-out
        # positions must never win, so restrict to that modality's valid steps.
        att = fwd["attention"][dom][i].copy()
        valid = sp.masks[dom][i]
        att = np.where(valid, att, -np.inf)
        peak = int(np.argmax(att)) if np.isfinite(att).any() else 0
        span = _span_around(peak)

        dur = float(sp.durations[i]) if sp.durations is not None else 1.0
        step = dur / PROBLEM_MAX_POSITIONS

        # ---- time: word-level alignment when available --------------------
        tmap = time_map.get(sid) if time_map else None
        n_tokens = int(sp.masks["text"][i].sum())   # includes [CLS] and [SEP]
        content_tokens = max(n_tokens - 2, 1)
        if tmap is not None and tmap.words:
            word_idx = tmap.word_for_token(peak, content_tokens)
            start_s, end_s = tmap.span(word_idx)
            evidence_word = tmap.words[word_idx].word
            time_source = "forced_alignment"
        else:
            word_idx = -1
            start_s = span[0] * step
            end_s = min(dur, (span[-1] + 1) * step)
            evidence_word = ""
            time_source = "uniform_estimate"

        if end_s <= start_s:                     # guard the submission contract
            end_s = min(dur, start_s + max(step, 1e-3))

        # keep the naive number too, so the report can quantify its error
        naive_start = span[0] * step
        naive_end = min(dur, (span[-1] + 1) * step)

        peaks = {}
        for m in MODALITIES:
            a = np.where(sp.masks[m][i], fwd["attention"][m][i], -np.inf)
            if np.isfinite(a).any():
                peaks[m] = np.argsort(-a)[:top_k_positions].tolist()

        raw = sp.raw_text[i] if i < len(sp.raw_text) else ""
        words = raw.split()
        mask_count = int(sp.masks["text"][i].sum())
        word_note = ""
        if words:
            # approximate: subword slot -> word by proportional index
            lo_w = int(span[0] * len(words) / max(mask_count, 1))
            hi_w = int((span[-1] + 1) * len(words) / max(mask_count, 1))
            lo_w = min(lo_w, len(words) - 1)
            hi_w = min(max(hi_w, lo_w + 1), len(words))
            word_note = " ".join(words[lo_w:hi_w])
        if evidence_word:
            word_note = evidence_word

        # Corroboration: the gate is the model's self-report, the LOMO argmax is
        # measured behaviour.  Where they differ the explanation is flagged
        # rather than quietly presented as solid.
        lomo_row = {m: float(lomo[i, k]) for k, m in enumerate(MODALITIES)}
        lomo_dom = max(lomo_row, key=lomo_row.get) if lomo_row else dom
        ordered = sorted(lomo_row.values(), reverse=True)
        lomo_margin = (ordered[0] - ordered[1]) if len(ordered) > 1 else 0.0
        unavailable = [m for m in MODALITIES if not bool(sp.masks[m][i].any())]

        exps.append(Explanation(
            sample_id=sid,
            polarity=POLARITY_NAMES[int(labels[i])],
            intensity=float(fwd["intensity"][i]),
            polarity_probs=probs[i].tolist(),
            dominant_modality=dom,
            modality_weights=w,
            evidence_modality=dom,
            evidence_position=peak,
            evidence_start=start_s,
            evidence_end=end_s,
            evidence_span_positions=span,
            text_excerpt=raw,
            word_estimate=word_note,
            lomo_drops=lomo_row,
            attention_peaks=peaks,
            corroborated=bool(lomo_dom == dom),
            lomo_dominant=lomo_dom,
            lomo_margin=lomo_margin,
            unavailable=unavailable,
            time_source=time_source,
            evidence_word=evidence_word,
            evidence_word_index=word_idx,
            naive_start=naive_start,
            naive_end=naive_end,
        ))

    return exps, {"logits": fwd["logits"], "intensity": fwd["intensity"],
                  "gate": fwd["gate"], "attention": fwd["attention"],
                  "lomo": lomo, "probs": probs, "labels": labels}


@torch.no_grad()
def faithfulness_test(model: ModalityAttributionNet, sp: Split, device,
                      k: int = 3, n_random: int = 5, seed: int = 0,
                      batch_size: int = 64) -> dict:
    """Does masking the attended positions hurt more than masking others?

    Three groups per modality, each masking ``k`` *valid* positions of that
    modality only:

    * ``top``  -- the ``k`` highest-attention valid positions, i.e. what the
      explanation points at;
    * ``low``  -- the ``k`` lowest-attention valid positions, drawn from the
      positions **outside** the top-k so the two sets are disjoint;
    * ``rand`` -- ``k`` uniformly drawn valid positions, averaged over
      ``n_random`` independent draws.

    A faithful attribution must order the damage as ``top > rand > low``.  The
    low group is what separates "the attention peaks matter" from "any three
    positions matter": if masking the *least* attended positions hurt as much as
    masking the peaks, the attribution would carry no information about which
    part of the input drove the prediction.

    Only samples with at least ``2k`` valid positions can enter the low-group
    statistics, because otherwise the two sets cannot be made disjoint.  Those
    samples are excluded from the triple comparison and the excluded count is
    reported rather than silently dropped.  The top/random means are therefore
    also recomputed on that same subset (``subset_*``) so the three figures are
    compared on one identical sample set.
    """
    rng = np.random.default_rng(seed)
    fwd = forward_all(model, sp, device)
    base = fwd["intensity"]
    n = len(sp)
    result: dict = {
        "k": k,
        "n_random_trials": n_random,
        "n_samples": n,
        "low_group_rule": "注意力最低的 k 个有效位置，且与最高 k 个位置不相交",
        "expected_ordering": "top > rand > low",
        "per_modality": {},
    }

    for m in MODALITIES:
        drops_top = np.zeros(n, dtype=np.float32)
        drops_low = np.zeros(n, dtype=np.float32)
        drops_rand = np.zeros(n, dtype=np.float32)
        attn_top = np.full(n, np.nan, dtype=np.float32)
        attn_low = np.full(n, np.nan, dtype=np.float32)
        has_low = np.zeros(n, dtype=bool)
        n_valid = np.zeros(n, dtype=np.int32)

        for start in range(0, n, batch_size):
            idx = np.arange(start, min(start + batch_size, n))
            batch = {mm: torch.from_numpy(np.asarray(getattr(sp, mm))[idx]).to(device)
                     for mm in MODALITIES}
            masks = {mm: torch.from_numpy(sp.masks[mm][idx]).to(device)
                     for mm in MODALITIES}

            top_pick: list[np.ndarray] = []
            low_pick: list[np.ndarray] = []
            for i in idx:
                valid = np.flatnonzero(sp.masks[m][i])
                n_valid[i] = valid.size
                if valid.size == 0:
                    top_pick.append(valid)
                    low_pick.append(valid)
                    continue
                a = np.asarray(fwd["attention"][m][i])[valid]
                order = valid[np.argsort(-a, kind="stable")]   # 注意力 高 -> 低
                top_pick.append(order[:k])
                rest = order[k:]                                # 与 top-k 不相交
                if rest.size >= k:
                    low_pick.append(rest[::-1][:k])             # 其中最低的 k 个
                    has_low[i] = True
                else:
                    low_pick.append(rest[::-1])

            def _apply(pick):
                picked = masks[m].clone()
                for r in range(len(idx)):
                    for p in pick[r]:
                        picked[r, int(p)] = False
                return picked

            def _forward_with(new_mask):
                ab = {mm: (torch.zeros_like(batch[mm]) if mm == m else batch[mm])
                      for mm in MODALITIES}
                ab[m] = batch[m] * new_mask.unsqueeze(-1).float()
                nm = dict(masks)
                nm[m] = new_mask
                return model(ab, nm)

            # --- top-k：解释所指向的位置 ---
            tm = _apply(top_pick)
            drops_top[idx] = np.abs(_forward_with(tm).intensity.cpu().numpy()
                                    - base[idx])

            # --- low-k：注意力最低的 k 个（与 top-k 不相交）---
            lm = _apply(low_pick)
            drops_low[idx] = np.abs(_forward_with(lm).intensity.cpu().numpy()
                                    - base[idx])

            # --- random：k 个随机有效位置，重复 n_random 次取平均 ---
            acc = np.zeros(len(idx), dtype=np.float32)
            for _ in range(n_random):
                rm = masks[m].clone()
                for r, i in enumerate(idx):
                    valid = np.flatnonzero(sp.masks[m][i])
                    if valid.size:
                        pick = rng.choice(valid, size=min(k, valid.size),
                                          replace=False)
                        rm[r, pick] = False
                acc += np.abs(_forward_with(rm).intensity.cpu().numpy()
                              - base[idx])
            drops_rand[idx] = acc / n_random

            # --- 记录被选位置的平均注意力（证明两组确实不同）---
            for i in idx:
                if top_pick[i - start].size:
                    attn_top[i] = float(np.mean(
                        np.asarray(fwd["attention"][m][i])[top_pick[i - start]]))
                if low_pick[i - start].size:
                    attn_low[i] = float(np.mean(
                        np.asarray(fwd["attention"][m][i])[low_pick[i - start]]))

        sel = has_low
        d_t, d_l, d_r = drops_top[sel], drops_low[sel], drops_rand[sel]
        n_sel = int(sel.sum())
        nan = float("nan")
        per: dict = {
            # 全样本口径（与历史版本可比）
            "mean_drop_topk": float(drops_top.mean()),
            "mean_drop_random": float(drops_rand.mean()),
            "gap": float(drops_top.mean() - drops_rand.mean()),
            "share_topk_worse": float((drops_top > drops_rand).mean()),
            # 同一子集口径（top / rand / low 三者可比）
            "n_samples_evaluated_low": n_sel,
            "n_samples_excluded_low": int(n - n_sel),
            "subset_mean_drop_topk": float(d_t.mean()) if n_sel else nan,
            "subset_mean_drop_random": float(d_r.mean()) if n_sel else nan,
            "subset_mean_drop_lowk": float(d_l.mean()) if n_sel else nan,
            "median_drop_topk": float(np.median(drops_top)),
            "median_drop_random": float(np.median(drops_rand)),
            "subset_median_drop_lowk": float(np.median(d_l)) if n_sel else nan,
            "gap_topk_minus_random": float(d_t.mean() - d_r.mean()) if n_sel else nan,
            "gap_random_minus_lowk": float(d_r.mean() - d_l.mean()) if n_sel else nan,
            "gap_topk_minus_lowk": float(d_t.mean() - d_l.mean()) if n_sel else nan,
            "share_topk_worse_than_lowk": float((d_t > d_l).mean()) if n_sel else nan,
            "share_random_between": float(((d_t > d_r) & (d_r > d_l)).mean())
            if n_sel else nan,
            "share_random_below_lowk": float((d_r < d_l).mean()) if n_sel else nan,
            "attn_topk_mean": float(np.nanmean(attn_top)),
            "attn_lowk_mean": float(np.nanmean(attn_low)),
            "attn_gap": float(np.nanmean(attn_top) - np.nanmean(attn_low)),
            # 注意力集中度：峰值位平均注意力 相对 均匀分布 1/n_valid 的倍数。
            # 1.0 = 注意力完全平坦（峰值位与低谷位无区别），越大越尖锐。
            # 这个量本身就解释了为什么弱势模态过不了严格三段序检验。
            "attn_concentration": float(np.nanmean(attn_top) * n_valid.mean()),
            "attn_topk_mass": float(np.nanmean(attn_top) * k),
            "attn_uniform_mass": float(k / max(float(n_valid.mean()), 1e-9)),
            "n_valid_positions_mean": float(n_valid.mean()),
        }
        result["per_modality"][m] = per
    return result


def write_submission(exps: list[Explanation], path: Path) -> None:
    """Write the attachment-4 CSV in the column layout the problem requires.

    Columns match src/q1_alignment/submission_schema.py's 'att4' contract, and
    the evidence span is guaranteed start < end so it can be replayed on video.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["sample_id", "polarity", "intensity", "dominant_modality",
              "modality_weights", "evidence_start", "evidence_end",
              "evidence_modality"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for e in exps:
            w.writerow({
                "sample_id": e.sample_id,
                "polarity": e.polarity,
                "intensity": f"{e.intensity:.4f}",
                "dominant_modality": e.dominant_modality,
                "modality_weights": ",".join(
                    f"{k}:{v:.3f}" for k, v in e.modality_weights.items()),
                "evidence_start": f"{e.evidence_start:.3f}",
                "evidence_end": f"{e.evidence_end:.3f}",
                "evidence_modality": e.evidence_modality,
            })


def gate_vs_lomo_report(exps: list[Explanation]) -> dict:
    """Agreement between the model's gate and the perturbation measurement.

    The gate is the model's self-report; the LOMO drop is measured behaviour.
    Reporting the agreement rate keeps us from presenting a self-report as if it
    were independently verified.
    """
    ok = 0
    for e in exps:
        gate_dom = max(e.modality_weights, key=e.modality_weights.get)
        if e.lomo_drops:
            lomo_dom = max(e.lomo_drops, key=e.lomo_drops.get)
        else:
            lomo_dom = gate_dom
        ok += int(gate_dom == lomo_dom)
    n = max(len(exps), 1)
    return {"n": len(exps), "agree": ok,
            "agreement_rate": round(ok / n, 4)}
