"""
asr_engine.py — Automatic Speech Recognition using Whisper.

Primary backend: faster-whisper (CTranslate2) on CPU/CUDA — runs on any machine.
NPU backend: Whisper-tiny.en compiled to QNN/ONNX via Qualcomm AI Hub.
             Requires onnxruntime-qnn on Snapdragon X Elite / 8 Elite hardware.

Qualcomm AI Hub model reference:
  https://aihub.qualcomm.com/models/whisper_tiny_en
  Model: WhisperTinyEn, optimized for Hexagon HTP (NPU), INT8 quantized.
  Published benchmark (Snapdragon X Elite, NPU): ~150ms/s of audio  [NOT MEASURED LOCALLY]
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .audio_capture import SpeechSegment, SAMPLE_RATE

logger = logging.getLogger(__name__)

WHISPER_MODEL_SIZE = "tiny.en"   # Default: fastest; switch to "base.en" for better accuracy


@dataclass
class Transcript:
    """Output from a single ASR pass."""
    text: str
    confidence: float          # 0.0–1.0 (approximated from log-prob)
    language: str
    latency_ms: float          # Wall-clock ASR latency
    segment_duration_s: float
    backend: str               # "faster-whisper" | "onnx-qnn" | "injected"
    timestamp: float = 0.0

    @property
    def words_per_minute(self) -> float:
        words = len(self.text.split())
        if self.segment_duration_s > 0:
            return words / self.segment_duration_s * 60
        return 0.0


class WhisperCPUBackend:
    """
    Whisper inference via faster-whisper (CTranslate2).
    Works on CPU (and CUDA if available). No special hardware required.
    """

    def __init__(self, model_size: str = WHISPER_MODEL_SIZE, device: str = "auto"):
        self._model_size = model_size
        self._device = device
        self._model = None
        self._load()

    def _load(self):
        try:
            from faster_whisper import WhisperModel
            compute = "int8" if self._device == "cpu" else "float16"
            device = self._device if self._device != "auto" else "cpu"
            logger.info(f"Loading Whisper-{self._model_size} ({device}/{compute})…")
            self._model = WhisperModel(self._model_size, device=device, compute_type=compute)
            logger.info("Whisper CPU backend ready.")
        except ImportError:
            logger.error("faster-whisper not installed. Run: pip install faster-whisper")
        except Exception as e:
            logger.error(f"Whisper load error: {e}")

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        """Return (text, avg_log_prob)."""
        if self._model is None:
            return "", -1.0
        segments, info = self._model.transcribe(
            audio,
            language="en",
            beam_size=1,         # Fast; increase to 5 for accuracy
            vad_filter=False,    # VAD done upstream
            condition_on_previous_text=True,
        )
        texts = []
        avg_log_prob = 0.0
        count = 0
        for seg in segments:
            texts.append(seg.text.strip())
            avg_log_prob += seg.avg_logprob
            count += 1
        text = " ".join(texts).strip()
        confidence = float(np.exp(avg_log_prob / max(count, 1)))
        return text, confidence

    @property
    def name(self) -> str:
        return "faster-whisper"


class WhisperQNNBackend:
    """
    Whisper-tiny.en via ONNX Runtime + QNN Execution Provider (Snapdragon NPU).

    Requirements (NOT VERIFIED locally — requires Snapdragon hardware):
      pip install onnxruntime-qnn
      Download whisper_tiny_en_qnn.onnx from Qualcomm AI Hub:
        qai_hub.get_model("WhisperTinyEn").download("models/")

    Status: ARCHITECTURE READY — NPU verification pending hardware access.
    """

    def __init__(self, model_path: Optional[str] = None):
        self._model_path = model_path or str(
            Path(__file__).parents[2] / "models" / "whisper_tiny_en_qnn.onnx"
        )
        self._encoder_session = None
        self._decoder_session = None
        self._load()

    def _load(self):
        model_p = Path(self._model_path)
        if not model_p.exists():
            logger.warning(
                f"QNN Whisper model not found at {self._model_path}.\n"
                "Download from Qualcomm AI Hub:\n"
                "  pip install qai_hub\n"
                "  python scripts/download_qnn_models.py"
            )
            return
        try:
            import onnxruntime as ort

            qnn_opts = {
                "backend_path": "QnnHtp.dll",
                "profiling_level": "off",
                "htp_performance_mode": "burst",       # Max NPU throughput
                "rpc_control_latency": "100",
            }
            providers = [("QNNExecutionProvider", qnn_opts), "CPUExecutionProvider"]
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_opts.enable_mem_pattern = False

            # Whisper has separate encoder + decoder ONNX graphs
            enc_path = str(model_p).replace(".onnx", "_encoder.onnx")
            dec_path = str(model_p).replace(".onnx", "_decoder.onnx")

            if Path(enc_path).exists() and Path(dec_path).exists():
                self._encoder_session = ort.InferenceSession(
                    enc_path, sess_options=sess_opts, providers=providers
                )
                self._decoder_session = ort.InferenceSession(
                    dec_path, sess_options=sess_opts, providers=providers
                )
                active = self._encoder_session.get_providers()
                logger.info(f"QNN Whisper loaded. Active providers: {active}")
                if "QNNExecutionProvider" not in active:
                    logger.warning("QNN EP not active — running on CPU fallback.")
            else:
                # Single combined ONNX
                self._encoder_session = ort.InferenceSession(
                    str(model_p), sess_options=sess_opts, providers=providers
                )
        except ImportError:
            logger.error("onnxruntime-qnn not installed. pip install onnxruntime-qnn")
        except Exception as e:
            logger.error(f"QNN Whisper load error: {e}")

    @property
    def is_available(self) -> bool:
        return self._encoder_session is not None

    def transcribe(self, audio: np.ndarray) -> tuple[str, float]:
        """
        Run Whisper encoder → greedy decoder on NPU.
        Returns (text, confidence_approx).
        NOTE: Full beam-search decoder not ported to QNN — greedy decode only.
        """
        if not self.is_available:
            return "", 0.0
        try:
            from .whisper_utils import log_mel_spectrogram, greedy_decode
            mel = log_mel_spectrogram(audio)                    # (80, T)
            mel_input = mel[np.newaxis, :, :].astype(np.float32)  # (1, 80, T)

            enc_out = self._encoder_session.run(
                None, {"mel": mel_input}
            )[0]   # (1, T//2, 384)

            text, confidence = greedy_decode(self._decoder_session, enc_out)
            return text, confidence
        except Exception as e:
            logger.error(f"QNN transcribe error: {e}")
            return "", 0.0

    @property
    def name(self) -> str:
        return "onnx-qnn"


class ASREngine:
    """
    Auto-selects the best available ASR backend.
    Falls back gracefully: QNN → faster-whisper → stub.
    """

    def __init__(
        self,
        use_npu: bool = False,
        model_size: str = WHISPER_MODEL_SIZE,
        device: str = "cpu",
    ):
        self._backend = None
        self._use_npu = use_npu

        if use_npu:
            qnn = WhisperQNNBackend()
            if qnn.is_available:
                self._backend = qnn
                logger.info("ASR: QNN/NPU backend selected.")
            else:
                logger.warning("ASR: QNN backend unavailable, falling back to CPU.")

        if self._backend is None:
            self._backend = WhisperCPUBackend(model_size=model_size, device=device)

    def process_segment(self, segment: SpeechSegment) -> Optional[Transcript]:
        """Transcribe a SpeechSegment; return Transcript or None."""

        # Demo mode: text injected by SyntheticAudioCapture
        if hasattr(segment, "_injected_text") and segment._injected_text:
            return Transcript(
                text=segment._injected_text,
                confidence=1.0,
                language="en",
                latency_ms=0.0,
                segment_duration_s=segment.duration_s,
                backend="injected",
                timestamp=segment.end_time,
            )

        audio = segment.numpy_array
        if len(audio) == 0:
            return None

        t0 = time.monotonic()
        text, confidence = self._backend.transcribe(audio)
        latency_ms = (time.monotonic() - t0) * 1000

        if not text:
            return None

        t = Transcript(
            text=text,
            confidence=max(0.0, min(1.0, confidence)),
            language="en",
            latency_ms=latency_ms,
            segment_duration_s=segment.duration_s,
            backend=self._backend.name,
            timestamp=segment.end_time,
        )
        logger.debug(
            f"[ASR|{t.backend}] {t.latency_ms:.0f}ms | "
            f"conf={t.confidence:.2f} | '{t.text[:60]}'"
        )
        return t

    @property
    def backend_name(self) -> str:
        return self._backend.name if self._backend else "none"
