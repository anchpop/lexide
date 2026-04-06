"""Pronunciation grader using Wav2Vec2Phoneme + Viterbi forced alignment.

Usage:
    from grader import PronunciationGrader

    grader = PronunciationGrader()
    result = grader.grade("path/to/audio.wav", "bonjour le monde", language="fr")
    for phoneme in result["phonemes"]:
        print(f"{phoneme['ipa']}  score={phoneme['score']:.2f}  {'⚠' if phoneme['flagged'] else '✓'}")
"""

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from phonemizer.backend import EspeakBackend

from .alignment import (
    compute_phoneme_scores,
    filter_silence_frames,
    viterbi_align,
)

# eSpeak language codes for phonemizer
LANG_TO_ESPEAK = {
    "en": "en-us",
    "de": "de",
    "es": "es",
    "fr": "fr-fr",
    "it": "it",
    "pt": "pt",
    "nl": "nl",
    "ru": "ru",
    "zh": "cmn",
    "ja": "ja",
    "el": "el",
    "hi": "hi",
    "ta": "ta",
    "te": "te",
    "bn": "bn",
}

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"


class PronunciationGrader:
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(self.device)
        self.model.eval()

        # Build token-to-id mapping from the model's vocab
        self.vocab = self.processor.tokenizer.get_vocab()
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.blank_id = self.processor.tokenizer.pad_token_id

        # Cache eSpeak backends per language
        self._g2p_cache: dict[str, EspeakBackend] = {}

    def _get_g2p(self, language: str) -> EspeakBackend:
        if language not in self._g2p_cache:
            espeak_lang = LANG_TO_ESPEAK.get(language, language)
            self._g2p_cache[language] = EspeakBackend(
                language=espeak_lang,
                preserve_punctuation=False,
                with_stress=True,
                language_switch="remove-flags",
            )
        return self._g2p_cache[language]

    def text_to_ipa(self, text: str, language: str) -> str:
        """Convert text to IPA using eSpeak G2P."""
        backend = self._get_g2p(language)
        result = backend.phonemize([text], strip=True)
        return result[0] if result else ""

    def ipa_to_token_ids(self, ipa: str) -> list[int]:
        """Convert IPA string to model token IDs.

        The model's vocab uses individual IPA characters. Characters not
        in the vocab are skipped.
        """
        ids = []
        for ch in ipa:
            if ch in self.vocab:
                ids.append(self.vocab[ch])
            # Skip characters not in the model's vocabulary (spaces, unknown symbols)
        return ids

    def get_ctc_log_probs(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Run the model and return CTC log-probability matrix.

        Args:
            audio: 1D float32 audio array at 16kHz.
            sample_rate: Audio sample rate.

        Returns:
            (T, V) numpy array of log-probabilities.
        """
        inputs = self.processor(
            audio, sampling_rate=sample_rate, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values).logits  # (1, T, V)

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs[0].cpu().numpy()

    def grade(
        self,
        audio_path: str,
        text: str,
        language: str,
        confidence_threshold: float = -0.5,
    ) -> dict:
        """Grade pronunciation of an audio file against expected text.

        Args:
            audio_path: Path to audio file (WAV, FLAC, MP3, etc.)
            text: The text the speaker was supposed to say.
            language: Language code (e.g. "fr", "en", "de").
            confidence_threshold: Log-prob below which a phoneme is flagged.

        Returns:
            Dict with:
                "text": original text
                "ipa": expected IPA transcription
                "phonemes": list of per-phoneme results with scores
                "overall_score": average score across phonemes
                "flagged_phonemes": list of poorly pronounced phonemes
        """
        import soundfile as sf

        # Load audio
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            import torchaudio
            waveform = torch.from_numpy(audio).unsqueeze(0)
            audio = torchaudio.functional.resample(waveform, sr, 16000).squeeze(0).numpy()

        # Get expected IPA
        ipa = self.text_to_ipa(text, language)
        target_ids = self.ipa_to_token_ids(ipa)

        if not target_ids:
            return {
                "text": text,
                "ipa": ipa,
                "phonemes": [],
                "overall_score": 0.0,
                "flagged_phonemes": [],
                "error": "No valid IPA tokens found for this text",
            }

        # Get CTC output
        log_probs = self.get_ctc_log_probs(audio)

        # Filter silence
        log_probs, frame_indices = filter_silence_frames(log_probs, self.blank_id)

        # Viterbi forced alignment
        segments = viterbi_align(log_probs, target_ids, self.blank_id)

        # Score phonemes
        segments = compute_phoneme_scores(segments, confidence_threshold)

        # Build results with IPA characters
        ipa_chars = [ch for ch in ipa if ch in self.vocab]
        phoneme_results = []
        for i, seg in enumerate(segments):
            ipa_char = ipa_chars[i] if i < len(ipa_chars) else "?"
            phoneme_results.append({
                "ipa": ipa_char,
                "score": seg["score"],
                "confidence": seg["confidence"],
                "flagged": seg["flagged"],
                "start_frame": seg["start_frame"],
                "end_frame": seg["end_frame"],
            })

        scores = [p["score"] for p in phoneme_results]
        overall = sum(scores) / len(scores) if scores else 0.0
        flagged = [p for p in phoneme_results if p["flagged"]]

        return {
            "text": text,
            "ipa": ipa,
            "phonemes": phoneme_results,
            "overall_score": overall,
            "flagged_phonemes": flagged,
        }

    def grade_from_array(
        self,
        audio: np.ndarray,
        text: str,
        language: str,
        sample_rate: int = 16000,
        confidence_threshold: float = -0.5,
    ) -> dict:
        """Grade pronunciation from a numpy audio array.

        Same as grade() but takes audio directly instead of a file path.
        """
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16000:
            import torchaudio
            waveform = torch.from_numpy(audio).unsqueeze(0)
            audio = torchaudio.functional.resample(waveform, sample_rate, 16000).squeeze(0).numpy()

        ipa = self.text_to_ipa(text, language)
        target_ids = self.ipa_to_token_ids(ipa)

        if not target_ids:
            return {
                "text": text, "ipa": ipa, "phonemes": [],
                "overall_score": 0.0, "flagged_phonemes": [],
                "error": "No valid IPA tokens found",
            }

        log_probs = self.get_ctc_log_probs(audio)
        log_probs, _ = filter_silence_frames(log_probs, self.blank_id)
        segments = viterbi_align(log_probs, target_ids, self.blank_id)
        segments = compute_phoneme_scores(segments, confidence_threshold)

        ipa_chars = [ch for ch in ipa if ch in self.vocab]
        phoneme_results = []
        for i, seg in enumerate(segments):
            phoneme_results.append({
                "ipa": ipa_chars[i] if i < len(ipa_chars) else "?",
                "score": seg["score"],
                "confidence": seg["confidence"],
                "flagged": seg["flagged"],
                "start_frame": seg["start_frame"],
                "end_frame": seg["end_frame"],
            })

        scores = [p["score"] for p in phoneme_results]
        overall = sum(scores) / len(scores) if scores else 0.0

        return {
            "text": text,
            "ipa": ipa,
            "phonemes": phoneme_results,
            "overall_score": overall,
            "flagged_phonemes": [p for p in phoneme_results if p["flagged"]],
        }
