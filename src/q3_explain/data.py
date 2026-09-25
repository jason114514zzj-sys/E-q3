"""Problem-3 data preparation.

Design decisions, each tied to a verified fact about the data:

* **Text route.** Attachment 4 ships ``text`` (50x768), so P3 uses the
  precomputed BERT features directly.  Attachment 3's lack of ``text`` does not
  bind here because P3 never reads attachment 3.

* **Text validity comes from ``text_bert``, never from ``text == 0``.**
  Verified: all 50 rows of ``text`` are non-zero in 20/20 attachment-4 samples,
  so the padding is invisible to a zero test.  Channel 1 of ``text_bert`` is the
  0/1 attention mask.

* **Audio/vision validity comes from ``~all(x == 0)``.**  54.7% / 57.2% of
  attachment-2 train steps are all-zero padding and there is no length field.

* **Per-dimension scaling, fitted on train only.**  Audio dim 0 has ~394x the
  standard deviation of the other 73 dimensions, so a single global mean/std
  lets dim 0 dominate the fused representation.

* **Position is a BERT subword slot, not a word.**  Verified on attachment 4:
  the mask counts 24 tokens where ``raw_text`` has 21 words, and 50 tokens where
  it has 51 words (the encoder truncates at 50).  So a position can only be
  mapped back to a word approximately, and the code labels it as such.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MODALITIES = ("text", "audio", "vision")
PROBLEM_MAX_POSITIONS = 50


@dataclass
class Split:
    name: str
    text: np.ndarray            # (N, 50, 768) float32, invalid rows zeroed
    audio: np.ndarray           # (N, 50, 74)
    vision: np.ndarray          # (N, 50, 35)
    masks: dict[str, np.ndarray]  # modality -> (N, 50) bool
    polarity: np.ndarray        # (N,) int64 in {0,1,2} = neg/neu/pos
    intensity: np.ndarray       # (N,) float32 in [-3, 3]
    ids: list[str] = field(default_factory=list)
    raw_text: list[str] = field(default_factory=list)
    durations: np.ndarray | None = None   # (N,) seconds, attachment 4 only

    def __len__(self) -> int:
        return len(self.intensity)


def _text_mask_from_bert(bert: np.ndarray) -> np.ndarray:
    """(N, 50) bool from the 0/1 attention channel of text_bert."""
    if bert.ndim != 3:
        raise ValueError(f"text_bert must be (N,3,50), got {bert.shape}")
    for ch in range(bert.shape[1]):
        u = np.unique(bert[:, ch, :])
        if u.size <= 2 and set(np.round(u.astype(float), 6)).issubset({0.0, 1.0}):
            return bert[:, ch, :].astype(bool)
    raise ValueError("no 0/1 attention channel found in text_bert")


def _zero_pad_mask(x: np.ndarray) -> np.ndarray:
    """(N, 50) bool: True where the step carries signal."""
    return ~np.all(np.asarray(x) == 0, axis=-1)


def labels_from_regression(reg: np.ndarray) -> np.ndarray:
    """3-class polarity from the continuous label.

    Verified on attachment 2: ``0`` is exactly neutral, negatives run
    [-3, -0.3333] and positives [+0.1667, 3], with empty open intervals
    (-0.3333, 0) and (0, 0.1667).  So the sign of the label IS the class, and
    the boundary value +0.1667 is a legitimate positive -- it must not be
    absorbed into neutral by a near-zero threshold.
    """
    out = np.ones(len(reg), dtype=np.int64)      # default neutral
    out[reg < 0] = 0
    out[reg > 0] = 2
    return out


def load_attachment2(data_root: Path) -> dict[str, Split]:
    path = Path(data_root) / "附件2-数据集特征文件" / "aligned_50.pkl"
    with path.open("rb") as fh:
        d = pickle.load(fh)

    splits: dict[str, Split] = {}
    for name in ("train", "valid", "test"):
        if name not in d:
            continue
        s = d[name]
        text = np.asarray(s["text"], dtype=np.float32)
        bert = np.asarray(s["text_bert"])
        tm = _text_mask_from_bert(bert)
        # Text padding carries non-zero values, so zero the invalid rows.
        # Without this the model averages garbage into its text summary.
        text = np.where(tm[:, :, None], text, 0.0).astype(np.float32)

        audio = np.asarray(s["audio"], dtype=np.float32)
        vision = np.asarray(s["vision"], dtype=np.float32)

        reg = np.asarray(s["regression_labels"], dtype=np.float32)
        if "classification_labels" in s:
            pol = np.asarray(s["classification_labels"]).astype(np.int64).ravel()
        else:
            pol = labels_from_regression(reg)

        splits[name] = Split(
            name=name, text=text, audio=audio, vision=vision,
            masks={"text": tm,
                   "audio": _zero_pad_mask(audio),
                   "vision": _zero_pad_mask(vision)},
            polarity=pol, intensity=reg.ravel(),
            ids=[str(x) for x in s["id"]],
            raw_text=[str(x) for x in s.get("raw_text", [""] * len(reg))],
        )
    return splits


def load_attachment4(data_root: Path) -> Split:
    root = Path(data_root) / "附件4-可解释专项视频样本与特征文件"
    base = None
    for c in root.rglob("*"):
        if c.is_dir() and (c / "对齐版本").exists():
            base = c
            break
    if base is None:
        raise FileNotFoundError(f"attachment 4 layout not found under {root}")

    pkls = sorted((base / "对齐版本").glob("*.pkl"))
    text, audio, vision, tmask = [], [], [], []
    ids, raws = [], []
    for p in pkls:
        with p.open("rb") as fh:
            o = pickle.load(fh)
        bert = np.asarray(o["text_bert"])
        if bert.ndim == 3:
            bert = bert[0]
        tm = _text_mask_from_bert(bert[None, ...])[0]
        t = np.asarray(o["text"], dtype=np.float32)
        t = np.where(tm[:, None], t, 0.0).astype(np.float32)
        text.append(t)
        audio.append(np.asarray(o["audio"], dtype=np.float32))
        vision.append(np.asarray(o["vision"], dtype=np.float32))
        tmask.append(tm)
        ids.append(str(o.get("id", p.stem)))
        raws.append(str(o.get("raw_text", "")))

    audio = np.stack(audio)
    vision = np.stack(vision)

    durations = None
    dur_file = Path(data_root).parent / "work" / "att4_durations.json"
    if dur_file.exists():
        table = json.loads(dur_file.read_text(encoding="utf-8"))
        vals = []
        for p in pkls:
            key = f"{p.stem}.mp4"
            vals.append(float(table.get(key, np.nan)))
        if np.isfinite(vals).all():
            durations = np.asarray(vals, dtype=np.float32)

    return Split(
        name="att4",
        text=np.stack(text), audio=audio, vision=vision,
        masks={"text": np.stack(tmask),
               "audio": _zero_pad_mask(audio),
               "vision": _zero_pad_mask(vision)},
        polarity=np.full(len(pkls), -1, dtype=np.int64),   # unlabelled
        intensity=np.full(len(pkls), np.nan, dtype=np.float32),
        ids=ids, raw_text=raws, durations=durations,
    )


def fit_and_scale(splits: dict[str, Split], att4: Split) -> dict:
    """Per-dimension scaling fitted on train, applied to every split + att4.

    Text is left alone: it is already a BERT representation whose dimensions are
    not physical units, and the mask-pooling step is what protects it.
    """
    stats = {}
    for mod in ("audio", "vision"):
        tr = splits["train"]
        m = tr.masks[mod]
        valid = np.asarray(getattr(tr, mod), dtype=np.float64)[m]
        mean = valid.mean(axis=0)
        std = valid.std(axis=0)
        eps = 1e-6
        n_deg = int((std < eps).sum())
        std = np.maximum(std, eps)

        for sp in list(splits.values()) + [att4]:
            x = np.asarray(getattr(sp, mod), dtype=np.float32)
            mm = sp.masks[mod]
            z = (x - mean.astype(np.float32)) / std.astype(np.float32)
            z = np.where(mm[:, :, None], z, 0.0)
            setattr(sp, mod, z.astype(np.float32))

        stats[mod] = {"mean": mean.tolist(), "std": std.tolist(),
                      "n_valid_steps": int(m.sum()),
                      "degenerate_dims": n_deg,
                      "source_split": "train"}

    return stats


def describe(splits: dict[str, Split], att4: Split) -> None:
    print("=" * 78)
    print("数据准备结果")
    print("=" * 78)
    for name, sp in list(splits.items()) + [("att4", att4)]:
        print(f"\n[{name}] n={len(sp)}")
        for mod in MODALITIES:
            m = sp.masks[mod]
            counts = m.sum(axis=1)
            print(f"   {mod:7s} 有效位: 平均 {counts.mean():5.1f}/50  "
                  f"最少 {int(counts.min()):2d}  最多 {int(counts.max()):2d}")
        if name != "att4":
            from collections import Counter
            c = Counter(sp.polarity.tolist())
            print(f"   极性分布: neg={c[0]} neu={c[1]} pos={c[2]}   "
                  f"强度范围 [{sp.intensity.min():.2f}, {sp.intensity.max():.2f}]")
        else:
            print(f"   时长: "
                  + ("有" if sp.durations is not None else "缺")
                  + (f"  [{sp.durations.min():.2f}, {sp.durations.max():.2f}] s"
                     if sp.durations is not None else ""))
