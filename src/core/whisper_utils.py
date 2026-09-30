"""
whisper_utils.py — Mel spectrogram and greedy decoder helpers for QNN Whisper.

These utilities replicate the preprocessing done by the Whisper tokenizer and
audio frontend, needed when running Whisper encoder+decoder as separate ONNX
graphs on the NPU.
"""

import numpy as np
from typing import Optional

# Whisper audio constants
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
CHUNK_LENGTH = 30    # seconds
N_SAMPLES = CHUNK_LENGTH * 16000
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3000

# Whisper token IDs (english, no_timestamps mode)
SOT_TOKEN = 50258
EOT_TOKEN = 50257
NO_TIMESTAMPS = 50364
TRANSCRIBE = 50359
ENGLISH = 50259
PAD_TOKEN = 0


def log_mel_spectrogram(
    audio: np.ndarray,
    n_mels: int = N_MELS,
    padding: int = 0,
) -> np.ndarray:
    """
    Compute log-Mel spectrogram from float32 audio (16kHz, mono).
    Returns array of shape (n_mels, n_frames).
    """
    try:
        import librosa
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=16000,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=n_mels,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        # Normalize to [-1, 1]
        log_mel = (log_mel + 40) / 40
        # Pad or trim to N_FRAMES
        if log_mel.shape[1] < N_FRAMES:
            pad = N_FRAMES - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode="constant")
        else:
            log_mel = log_mel[:, :N_FRAMES]
        return log_mel.astype(np.float32)
    except ImportError:
        # Fallback: simple FFT-based approximation without librosa
        return _simple_mel_spectrogram(audio, n_mels)


def _simple_mel_spectrogram(audio: np.ndarray, n_mels: int = 80) -> np.ndarray:
    """Minimal mel spectrogram without librosa (lower quality, no external deps)."""
    # Pad audio to N_SAMPLES
    if len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)))
    else:
        audio = audio[:N_SAMPLES]

    frames = []
    window = np.hanning(N_FFT)
    for i in range(0, len(audio) - N_FFT, HOP_LENGTH):
        frame = audio[i:i + N_FFT] * window
        fft = np.abs(np.fft.rfft(frame)) ** 2
        frames.append(fft)

    spec = np.array(frames).T   # (N_FFT//2+1, T)

    # Simple triangular mel filters
    freq_bins = np.linspace(0, 8000, spec.shape[0])
    mel_bins = np.linspace(0, 8000, n_mels + 2)
    filters = np.zeros((n_mels, spec.shape[0]))
    for m in range(1, n_mels + 1):
        f_m_minus = mel_bins[m - 1]
        f_m = mel_bins[m]
        f_m_plus = mel_bins[m + 1]
        for k in range(spec.shape[0]):
            f = freq_bins[k]
            if f_m_minus <= f <= f_m:
                filters[m - 1, k] = (f - f_m_minus) / (f_m - f_m_minus)
            elif f_m <= f <= f_m_plus:
                filters[m - 1, k] = (f_m_plus - f) / (f_m_plus - f_m)

    mel_spec = filters @ spec   # (n_mels, T)
    log_mel = np.log1p(mel_spec * 1e4)
    log_mel = log_mel / (np.max(log_mel) + 1e-8)

    # Pad/trim
    if log_mel.shape[1] < N_FRAMES:
        log_mel = np.pad(log_mel, ((0, 0), (0, N_FRAMES - log_mel.shape[1])))
    else:
        log_mel = log_mel[:, :N_FRAMES]

    return log_mel.astype(np.float32)


def greedy_decode(
    decoder_session,   # ort.InferenceSession
    encoder_output: np.ndarray,
    max_tokens: int = 224,
) -> tuple[str, float]:
    """
    Greedy decode from Whisper encoder output using the QNN decoder ONNX session.
    
    This is a simplified greedy decoder — production would use beam search.
    Returns (text, approx_confidence).
    """
    try:
        # Lazy import tokenizer
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            def decode_tokens(tokens): return enc.decode(tokens)
        except ImportError:
            def decode_tokens(tokens): return " ".join(str(t) for t in tokens)

        tokens = [SOT_TOKEN, ENGLISH, TRANSCRIBE, NO_TIMESTAMPS]
        log_probs = []

        for _ in range(max_tokens):
            tokens_array = np.array([tokens], dtype=np.int64)
            outputs = decoder_session.run(
                None,
                {
                    "tokens": tokens_array,
                    "audio_features": encoder_output,
                }
            )
            logits = outputs[0][0, -1, :]   # Last token logits (vocab_size,)
            probs = softmax(logits)
            next_token = int(np.argmax(probs))
            log_probs.append(float(np.log(probs[next_token] + 1e-8)))
            if next_token == EOT_TOKEN:
                break
            tokens.append(next_token)

        text_tokens = [t for t in tokens[4:] if t != EOT_TOKEN]   # Skip prompt tokens
        text = decode_tokens(text_tokens)
        confidence = float(np.exp(np.mean(log_probs))) if log_probs else 0.0
        return text.strip(), confidence

    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"greedy_decode error: {e}")
        return "", 0.0


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
