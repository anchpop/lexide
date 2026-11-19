"""tiny-voca: a small two-voice TTS trained from scratch.

The speech model loads no pretrained weights. A pretrained phoneme recogniser
is used only by the annotation step, and an ASR model only by evaluation.
"""

from .config import Config
from .model import TinyVoca

__all__ = ["Config", "TinyVoca"]
