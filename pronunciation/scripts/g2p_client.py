"""Client for the `g2p` binary (github.com/anchpop/g2p): our espeak-ng fork,
statically linked with its data embedded, plus the ported Hindi chain and the
model-label tokenizer. yap links the same crate, so the two projects cannot
drift apart on segmentation or on which G2P build they run.

One long-lived `g2p serve` process; requests stream through it as JSON lines,
exactly one utterance per line, so the clause-vs-line framing problem of
`espeak-ng --stdin` (a comma sentence emits two lines, an unpunctuated line
merges into the next) cannot occur.

Install: cargo install --git https://github.com/anchpop/g2p --locked
(needs cmake + a C compiler). G2P_BIN overrides the binary, e.g. a checkout's
target/release/g2p.
"""

from __future__ import annotations

import functools
import json
import os
import shutil
import subprocess
import threading

_proc: subprocess.Popen[str] | None = None
_pid: int | None = None
# The server answers requests in order on one pipe pair, so a request's
# write and its readline must not interleave with another thread's (the
# Groq audit phonemizes from a thread pool; unlocked, replies cross over).
_lock = threading.Lock()


class Unlabelable(RuntimeError):
    """The backend refused the text rather than emit labels that silently
    omit part of what is spoken (e.g. Hindi text with digits). `reason` is a
    stable `reason:detail` code suitable for a sidecar `exclude_reason`."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def binary() -> str:
    override = os.environ.get("G2P_BIN")
    if override:
        return override
    found = shutil.which("g2p")
    if found is None:
        raise RuntimeError(
            "g2p binary not found. Install it with "
            "`cargo install --git https://github.com/anchpop/g2p --locked` "
            "(needs cmake + a C compiler), or point G2P_BIN at a build."
        )
    return found


def _server() -> subprocess.Popen[str]:
    """The shared `g2p serve` child, (re)started if missing, dead, or
    inherited across a fork (a forked child must not share the pipes)."""
    global _proc, _pid
    if _proc is None or _proc.poll() is not None or _pid != os.getpid():
        _proc = subprocess.Popen(
            [binary(), "serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8",
        )
        _pid = os.getpid()
    return _proc


@functools.cache
def identity() -> str:
    """Which phonemizer build produces this process's labels: g2p crate
    version plus a digest of the fork's sources. Stamp anything that
    persists phonemes with it."""
    return subprocess.run(
        [binary(), "identity"], capture_output=True, text=True, check=True,
    ).stdout.strip()


def request(**req) -> dict:
    """One raw request (`text` plus `voice` or `lang`, optional `canon`)."""
    with _lock:
        proc = _server()
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
    if not line:
        raise RuntimeError(f"g2p exited (code {proc.poll()}) on request {req!r}")
    result = json.loads(line)
    if "error" in result:
        if result.get("unlabelable"):
            raise Unlabelable(result["unlabelable"], f"g2p: {result['error']} ({req!r})")
        raise RuntimeError(f"g2p: {result['error']} ({req!r})")
    return result


def phonemize(text: str, espeak_lang: str) -> tuple[list[str], list[int], list[tuple[int, int]]]:
    """Phonemize one utterance with espeak voice `espeak_lang`.

    Returns (phonemes, stress, word_spans): the model-label tokenization
    (continuation diacritics folded onto the previous token, ʲ onto a
    preceding consonant, language-switch markers stripped) with stress codes
    0/1/2, and word_spans[i] = (start, end) into `phonemes` for the i-th word
    espeak emitted. Embedded newlines are spaces — a sentence is one
    utterance.
    """
    r = request(text=text, voice=espeak_lang)
    return r["phonemes"], r["stress"], [tuple(s) for s in r["word_spans"]]
