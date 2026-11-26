"""Lexide speech: small text-to-speech models trained from scratch, one per language.

The speech model loads no pretrained weights. A pretrained phoneme recogniser
is used only by the annotation step, and an ASR model only by evaluation.
"""

from .config import Config
from .model import TinyVoca

__all__ = ["Config", "TinyVoca"]
