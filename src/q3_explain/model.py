"""A modality-attribution network that explains itself.

Why intrinsic rather than post-hoc
---------------------------------
The problem requires the explanation to be "可量化、可复核" (quantifiable and
re-checkable) and the key evidence to map back onto original text spans, speech
segments or video frames.  Post-hoc methods (SHAP/LIME) need a second mechanism
to obtain temporal localisation and introduce sampling randomness.  This model
emits the three required explanations directly from its own computation:

======================  ==========================================================
required output         where it comes from
======================  ==========================================================
主要参考模态            argmax of the modality gate ``g``
模态作用程度            the gate vector ``g`` itself (sums to 1)
关键证据定位            the temporal attention ``a_m`` over the 50 positions
======================  ==========================================================

Attention is not automatically faithful, so the pipeline additionally runs a
perturbation test (see :mod:`q3_explain.explain`) that masks the most-attended
positions and measures whether the prediction actually degrades more than when
masking random positions.  That test is what makes the explanation checkable.

Architecture notes
------------------
* **One encoder per modality.**  Sharing an encoder would prevent separating
  per-modality contributions, which is the whole point.
* **Masks are applied twice**: invalid steps are zeroed on input AND set to
  ``-inf`` before the attention softmax.  Zeroing alone is not enough -- a zero
  vector still produces a non-zero attention score.
* **Polarity and intensity share the fused vector but have separate heads**,
  matching the problem's split between classification and regression.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

MODALITIES = ("text", "audio", "vision")
N_POLARITY = 3


class ModalityEncoder(nn.Module):
    """Project one modality to a shared width and mix along time."""

    def __init__(self, in_dim: int, hidden: int, n_heads: int = 4,
                 max_positions: int = 50, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.pos = nn.Parameter(torch.zeros(1, max_positions, hidden))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 2,
            dropout=dropout, batch_first=True, activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x) + self.pos[:, : x.shape[1]]
        # no key_padding_mask here: padding is already zeroed and the attention
        # pooling below masks it explicitly
        h = self.encoder(h)
        return self.norm(h)


class TemporalAttention(nn.Module):
    """Additive attention over time, with a hard mask."""

    def __init__(self, hidden: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1)
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        """h: (B,T,H)  mask: (B,T) bool.  Returns (pooled (B,H), weights (B,T))."""
        e = self.score(h).squeeze(-1)                       # (B,T)
        neg = torch.finfo(e.dtype).min
        e = e.masked_fill(~mask, neg)
        # a sample with no valid step would produce NaN; give it uniform weights
        empty = ~mask.any(dim=1, keepdim=True)
        if empty.any():
            e = torch.where(empty.expand_as(e), torch.zeros_like(e), e)
        a = torch.softmax(e, dim=1)
        a = torch.where(empty.expand_as(a), torch.zeros_like(a), a)
        pooled = torch.einsum("bt,bth->bh", a, h)
        return pooled, a


@dataclass
class Output:
    polarity_logits: torch.Tensor     # (B,3)
    intensity: torch.Tensor           # (B,)
    gate: torch.Tensor                # (B,3)  modality contribution
    attention: dict[str, torch.Tensor]  # modality -> (B,50)
    pooled: dict[str, torch.Tensor]     # modality -> (B,H)

    def dominant_index(self) -> torch.Tensor:
        return self.gate.argmax(dim=1)


class ModalityAttributionNet(nn.Module):
    def __init__(self, dims: dict[str, int], hidden: int = 96,
                 dropout: float = 0.1, max_positions: int = 50):
        super().__init__()
        self.modalities = tuple(m for m in MODALITIES if m in dims)
        self.encoders = nn.ModuleDict({
            m: ModalityEncoder(dims[m], hidden, dropout=dropout,
                               max_positions=max_positions)
            for m in self.modalities
        })
        self.attentions = nn.ModuleDict({
            m: TemporalAttention(hidden) for m in self.modalities
        })
        # gating sees each modality summary plus their elementwise dispersion,
        # which helps it tell "all agree" from "one dominates"
        self.gate = nn.Sequential(
            nn.Linear(hidden * len(self.modalities), hidden),
            nn.GELU(),
            nn.Linear(hidden, len(self.modalities)),
        )
        self.fuse_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.head_polarity = nn.Linear(hidden, N_POLARITY)
        self.head_intensity = nn.Linear(hidden, 1)

    def forward(self, batch: dict[str, torch.Tensor],
                masks: dict[str, torch.Tensor]) -> Output:
        pooled: dict[str, torch.Tensor] = {}
        attn: dict[str, torch.Tensor] = {}
        for m in self.modalities:
            h = self.encoders[m](batch[m])
            z, a = self.attentions[m](h, masks[m])
            pooled[m] = z
            attn[m] = a

        cat = torch.cat([pooled[m] for m in self.modalities], dim=-1)
        logits = self.gate(cat)

        # A modality with no valid step must receive zero contribution.  Without
        # this the gate read a *zeroed-out* summary and, measured on attachment
        # 2's validation split, handed vision 0.287 on the 15 samples where
        # vision is entirely absent -- MORE than its 0.227 average.  Reporting
        # "主要参考模态 = vision" for a sample that has no vision is nonsense, so
        # availability is enforced rather than left for the optimiser to learn.
        avail = torch.stack([masks[m].any(dim=1) for m in self.modalities], dim=1)
        neg = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~avail, neg)
        # a sample with nothing available would softmax to NaN; fall back to
        # uniform over the modalities that at least exist
        any_avail = avail.any(dim=1, keepdim=True)
        if not bool(any_avail.all()):
            logits = torch.where(any_avail, logits, torch.zeros_like(logits))
            avail = torch.where(any_avail, avail, torch.ones_like(avail))
        gate = torch.softmax(logits, dim=-1)
        gate = torch.where(avail, gate, torch.zeros_like(gate))
        gate = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        stacked = torch.stack([pooled[m] for m in self.modalities], dim=1)
        fused = torch.einsum("bm,bmh->bh", gate, stacked)
        fused = self.fuse_norm(self.dropout(fused))

        return Output(
            polarity_logits=self.head_polarity(fused),
            intensity=self.head_intensity(fused).squeeze(-1),
            gate=gate,
            attention=attn,
            pooled=pooled,
        )


def masked_modality_dropout(masks: dict[str, torch.Tensor],
                            p: float, generator=None,
                            ) -> dict[str, torch.Tensor]:
    """Drop an entire modality for a fraction of samples.

    Included as an *auxiliary* regulariser only.  The problem is explicit that
    "局部缺失" means a contiguous span is unavailable, not that a whole modality
    vanishes, so this must not be the only augmentation -- it exists so the
    gating head sees both regimes during training.

    The random draw is made on CPU and moved, because a CPU ``generator`` cannot
    be used to sample directly onto a CUDA tensor.
    """
    if p <= 0:
        return masks
    out = {}
    for m, mask in masks.items():
        draw = torch.rand(mask.shape[0], 1, generator=generator)
        keep = (draw >= p).to(mask.device)
        out[m] = mask & keep
    return out


def polarity_class_weights(polarity) -> torch.Tensor:
    """Inverse-frequency weights so the neutral class is not ignored."""
    import numpy as np
    counts = np.bincount(np.asarray(polarity), minlength=N_POLARITY).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (N_POLARITY * counts)
    return torch.tensor(w, dtype=torch.float32)
