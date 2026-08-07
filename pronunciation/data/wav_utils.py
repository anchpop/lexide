"""Shared WAV helpers for the corpus importers.

Kept dependency-free (stdlib only) so any importer can pull it in without
dragging along network/audio libraries.
"""
import struct
from pathlib import Path


def repair_streamed_wav_header(path: Path) -> bool:
    """Replace pipe/stream-output sentinel sizes in an otherwise valid WAV.

    Some sources (ffmpeg streamed through stdout; certain TTS APIs) leave the
    RIFF and data chunk lengths at 0xffffffff. The PCM payload is intact, so a
    deterministic header repair avoids a lossy second encode. libsndfile reads
    such files correctly regardless, but Python's ``wave`` module and any tool
    that trusts the size field will report a multi-day clip. Returns whether a
    repair was made.
    """
    with path.open("r+b") as wav:
        raw = wav.read()
        if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
            return False
        data_marker = raw.find(b"data", 12)
        if data_marker < 0 or data_marker + 8 > len(raw):
            return False
        riff_size = struct.unpack_from("<I", raw, 4)[0]
        data_size = struct.unpack_from("<I", raw, data_marker + 4)[0]
        if riff_size != 0xFFFFFFFF and data_size != 0xFFFFFFFF:
            return False
        wav.seek(4)
        wav.write(struct.pack("<I", len(raw) - 8))
        wav.seek(data_marker + 4)
        wav.write(struct.pack("<I", len(raw) - data_marker - 8))
    return True
