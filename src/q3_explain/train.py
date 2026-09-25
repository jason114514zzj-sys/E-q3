"""Train the problem-3 model on attachment 2, select on its validation split.

Protocol follows the problem statement (段60): model parameters are learned on
attachment 2's **train** split, and the model structure / hyper-parameters are
chosen on the **validation** split.  Attachment 2's **test** split is never used
for selection, and attachments 3/4 are only ever fed forward.

Verified reason for that caution: by content matching, 4/30 attachment-3 and
5/20 attachment-4 samples are byte-identical to rows in attachment 2's test
split.  Touching test for selection would therefore leak into the reported
special-test performance.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import MODALITIES, Split
from .model import (
    ModalityAttributionNet,
    masked_modality_dropout,
    polarity_class_weights,
)


@dataclass
class Metrics:
    accuracy: float = 0.0
    macro_f1: float = 0.0
    mae: float = 0.0
    pearson: float = 0.0
    n: int = 0
    confusion: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"accuracy": round(self.accuracy, 4),
                "macro_f1": round(self.macro_f1, 4),
                "mae": round(self.mae, 4),
                "pearson": round(self.pearson, 4),
                "n": self.n}


def compute_metrics(pred_cls: np.ndarray, true_cls: np.ndarray,
                    pred_val: np.ndarray, true_val: np.ndarray) -> Metrics:
    """Accuracy / macro-F1 for polarity, MAE / Pearson for intensity."""
    m = Metrics(n=int(len(true_cls)))
    if m.n == 0:
        return m
    m.accuracy = float((pred_cls == true_cls).mean())

    # macro-F1 over the three classes, computed from the confusion matrix so the
    # result does not depend on an external library
    K = 3
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(true_cls.tolist(), pred_cls.tolist()):
        if 0 <= t < K and 0 <= p < K:
            cm[t, p] += 1
    f1s = []
    for k in range(K):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    m.macro_f1 = float(np.mean(f1s))
    m.confusion = cm.tolist()

    m.mae = float(np.abs(pred_val - true_val).mean())
    if len(true_val) > 1 and true_val.std() > 0 and pred_val.std() > 0:
        m.pearson = float(np.corrcoef(pred_val, true_val)[0, 1])
    else:
        m.pearson = 0.0
    return m


def _to_tensors(sp: Split, idx: np.ndarray, device) -> tuple[dict, dict, dict]:
    batch = {m: torch.from_numpy(np.asarray(getattr(sp, m))[idx]).to(device)
             for m in MODALITIES}
    masks = {m: torch.from_numpy(sp.masks[m][idx]).to(device) for m in MODALITIES}
    targets = {"polarity": torch.from_numpy(sp.polarity[idx]).to(device),
               "intensity": torch.from_numpy(sp.intensity[idx].astype(np.float32)).to(device)}
    return batch, masks, targets


@torch.no_grad()
def evaluate(model: ModalityAttributionNet, sp: Split, device,
             batch_size: int = 256) -> tuple[Metrics, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    n = len(sp)
    pcls = np.zeros(n, dtype=np.int64)
    pval = np.zeros(n, dtype=np.float32)
    gates = np.zeros((n, len(MODALITIES)), dtype=np.float32)

    for start in range(0, n, batch_size):
        idx = np.arange(start, min(start + batch_size, n))
        batch, masks, _ = _to_tensors(sp, idx, device)
        out = model(batch, masks)
        pcls[idx] = out.polarity_logits.argmax(dim=1).cpu().numpy()
        pval[idx] = out.intensity.cpu().numpy()
        gates[idx] = out.gate.cpu().numpy()

    met = compute_metrics(pcls, sp.polarity, pval, sp.intensity)
    return met, pcls, pval, gates


def train_model(splits: dict[str, Split], device, out_dir: Path,
                hidden: int = 96, epochs: int = 60, lr: float = 1.5e-3,
                weight_decay: float = 1e-4, batch_size: int = 128,
                lambda_reg: float = 1.0, mod_dropout: float = 0.10,
                seed: int = 20260924, patience: int = 12,
                verbose: bool = True) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    tr, va = splits["train"], splits["valid"]
    dims = {m: int(np.asarray(getattr(tr, m)).shape[-1]) for m in MODALITIES}
    model = ModalityAttributionNet(dims, hidden=hidden).to(device)

    cls_w = polarity_class_weights(tr.polarity).to(device)
    ce = nn.CrossEntropyLoss(weight=cls_w)
    mse = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    gen = torch.Generator(device="cpu").manual_seed(seed)
    best = {"macro_f1": -1.0, "epoch": -1, "state": None, "metrics": None}
    history = []
    n = len(tr)

    for ep in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(n)
        total = 0.0
        t0 = time.time()
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch, masks, tgt = _to_tensors(tr, idx, device)
            # auxiliary whole-modality dropout: the problem defines 局部缺失 as a
            # contiguous span, so this is a regulariser, not the main mechanism
            masks = masked_modality_dropout(masks, mod_dropout, generator=gen)
            batch = {m: batch[m] * masks[m].unsqueeze(-1).float() for m in MODALITIES}

            out = model(batch, masks)
            loss_cls = ce(out.polarity_logits, tgt["polarity"])
            loss_reg = mse(out.intensity, tgt["intensity"])
            loss = loss_cls + lambda_reg * loss_reg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss) * len(idx)
        sched.step()

        met, _, _, _ = evaluate(model, va, device)
        history.append({"epoch": ep, "train_loss": total / n,
                        "valid": met.to_dict()})
        if met.macro_f1 > best["macro_f1"]:
            best = {"macro_f1": met.macro_f1, "epoch": ep,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()},
                    "metrics": met}
        if verbose and (ep == 1 or ep % 5 == 0 or ep == epochs):
            print(f"  epoch {ep:3d}  loss={total/n:.4f}  "
                  f"valid F1={met.macro_f1:.4f} acc={met.accuracy:.4f} "
                  f"MAE={met.mae:.4f} r={met.pearson:.4f}  "
                  f"({time.time()-t0:.1f}s)")
        if ep - best["epoch"] >= patience:
            if verbose:
                print(f"  早停于 epoch {ep}（最佳 {best['epoch']}）")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "dims": dims, "hidden": hidden,
                "modalities": list(MODALITIES),
                "best_epoch": best["epoch"]},
               out_dir / "q3_model.pt")

    summary = {
        "best_epoch": best["epoch"],
        "valid_metrics": best["metrics"].to_dict() if best["metrics"] else {},
        "hyperparameters": {"hidden": hidden, "epochs": epochs, "lr": lr,
                            "weight_decay": weight_decay,
                            "batch_size": batch_size,
                            "lambda_reg": lambda_reg,
                            "modality_dropout": mod_dropout,
                            "seed": seed, "patience": patience},
        "history": history,
        "dims": dims,
    }
    (out_dir / "q3_training.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
