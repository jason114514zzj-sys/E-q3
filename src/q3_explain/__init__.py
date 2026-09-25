"""Problem 3: interpretable multimodal sentiment prediction.

Deliverables (problem statement section 四.4):
  (1) model principle / architecture / objective / training scheme / parameters
  (2) typical-sample explanation cards
  (3) per-modality importance visualisation and cross-modality comparison
  (4) full prediction + explanation results for attachment 4
  (5) validation-split evaluation, visualisation and error attribution
"""

from .data import Split, load_attachment2, load_attachment4, fit_and_scale
from .model import ModalityAttributionNet

__all__ = [
    "Split",
    "load_attachment2",
    "load_attachment4",
    "fit_and_scale",
    "ModalityAttributionNet",
]
