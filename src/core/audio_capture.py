"""
audio_capture.py — Real-time audio capture with Voice Activity Detection.

Captures system/microphone audio in 30ms frames, applies WebRTC VAD to
detect speech segments, and emits audio chunks for ASR processing.

On Snapdragon hardware: the VAD silero model can be loaded via ONNX Runtime
with the QNN Execution Provider (HTP backend) for NPU-accelerated VAD.
CPU fallback is provided for non-NPU environments.
"""

import threading
import queue
import time
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional, Generator

import numpy as np

logger = logging.getLogger(__name__)

# Audio constants
SAMPLE_RATE = 16000          # Whisper requires 16 kHz
FRAME_DURATION_MS = 30       # 30ms frames (VAD requirement)
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 480 samples
CHANNELS = 1
DTYPE = np.int16

# VAD settings
VAD_AGGRESSIVENESS = 2       # 0–3; higher = more aggressive filtering
SPEECH_BUFFER_MS = 300       # Buffer after speech ends before cut
MAX_SEGMENT_DURATION_S = 15  # Max speech segment before forced flush


@dataclass
class AudioFrame:
    """Single 30ms audio frame with metadata."""
    data: bytes               # Raw PCM bytes (int16, 16kHz mono)
    timestamp: float          # Wall-clock time when captured
    is_speech: bool = False   # VAD result


@dataclass
class SpeechSegment:
    """A complete speech segment ready for ASR."""
    frames: list = field(default_factory=list)  # list[AudioFrame]
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def audio_bytes(self) -> bytes:
        return b"".join(f.data for f in self.frames)

    @property
    def numpy_array(self) -> np.ndarray:
        raw = np.frombuffer(self.audio_bytes, dtype=DTYPE)
        return raw.astype(np.float32) / 32768.0   # Normalize to [-1, 1]

    @property
    def duration_s(self) -> float:
        return self.end_time - self.start_time


class VoiceActivityDetector:
    """
    VAD using webrtcvad (CPU) with optional NPU override.

    NPU path: load silero-vad-quantized.onnx via onnxruntime with
    QNN Execution Provider when --npu flag is set.
    """

    def __init__(self, aggressiveness: int = VAD_AGGRESSIVENESS, use_npu: bool = False):
        self._use_npu = use_npu
        self._vad = None
        self._npu_session = None
        self._init_vad(aggressiveness)

    def _init_vad(self, aggressiveness: int):
        if self._use_npu:
            self._init_npu_vad()
            if self._npu_session is not None:
                logger.info("VAD: NPU backend (QNN/HTP) active.")
                return
            logger.warning("VAD: NPU init failed, falling back to CPU WebRTC VAD.")

        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(aggressiveness)
            logger.info("VAD: WebRTC CPU backend active.")
        except ImportError:
            logger.warning("webrtcvad not installed; VAD disabled (all frames treated as speech).")

    def _init_npu_vad(self):
        """
        Load silero-vad via ONNX Runtime with QNN Execution Provider.

        Requires:
          - onnxruntime-qnn package (Qualcomm QNN EP for Windows)
          - models/silero_vad_quantized.onnx
          - Snapdragon X-series or 8 Elite hardware with HTP driver stack

        Status: NOT VERIFIED — requires physical Snapdragon device.
        """
        try:
            import onnxruntime as ort
            from pathlib import Path

            model_path = Path(__file__).parents[2] / "models" / "silero_vad_quantized.onnx"
            if not model_path.exists():
                logger.warning(f"NPU VAD model not found at {model_path}.")
                return

            # QNN Execution Provider — targets Hexagon HTP (NPU)
            providers = [
                ("QNNExecutionProvider", {
                    "backend_path": "QnnHtp.dll",   # Windows path; Linux: libQnnHtp.so
                    "profiling_level": "off",
                }),
                "CPUExecutionProvider",             # Fallback
            ]
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._npu_session = ort.InferenceSession(
                str(model_path), sess_options=opts, providers=providers
            )
            active = self._npu_session.get_providers()
            logger.info(f"ONNX RT providers active: {active}")
            if "QNNExecutionProvider" not in active:
                logger.warning("QNN EP not loaded; running on CPU fallback.")
        except Exception as e:
            logger.error(f"NPU VAD init error: {e}")
            self._npu_session = None

    def is_speech(self, frame: bytes) -> bool:
        """Return True if the 30ms PCM frame contains speech."""
        if self._npu_session is not None:
            return self._npu_vad_infer(frame)
        if self._vad is not None:
            try:
                return self._vad.is_speech(frame, SAMPLE_RATE)
            except Exception:
                return True   # Default: treat as speech on error
        return True   # No VAD — pass all frames

    def _npu_vad_infer(self, frame: bytes) -> bool:
        """Run Silero VAD inference on NPU."""
        try:
            audio = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
            audio = audio[np.newaxis, :]   # (1, T)
            outputs = self._npu_session.run(None, {"input": audio})
            prob = float(outputs[0].squeeze())
            return prob > 0.5
        except Exception as e:
            logger.debug(f"NPU VAD inference error: {e}")
            return True


class AudioCapture:
    """
    Captures microphone audio, applies VAD, and yields SpeechSegments.

    Usage:
        cap = AudioCapture(use_npu=False)
        for segment in cap.stream_segments():
            # segment.numpy_array is float32 16kHz audio
            asr_engine.transcribe(segment)
    """

    def __init__(
        self,
        device_index: Optional[int] = None,
        use_npu: bool = False,
        on_frame: Optional[Callable[[AudioFrame], None]] = None,
    ):
        self._device_index = device_index
        self._use_npu = use_npu
        self._on_frame = on_frame
        self._vad = VoiceActivityDetector(use_npu=use_npu)
        self._segment_queue: queue.Queue[SpeechSegment] = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("AudioCapture started.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("AudioCapture stopped.")

    def stream_segments(self) -> Generator[SpeechSegment, None, None]:
        """Generator — yields complete speech segments."""
        self.start()
        try:
            while self._running:
                try:
                    seg = self._segment_queue.get(timeout=0.5)
                    yield seg
                except queue.Empty:
                    continue
        finally:
            self.stop()

    def _capture_loop(self):
        """Main capture loop — reads PyAudio stream, applies VAD."""
        try:
            import pyaudio
        except ImportError:
            logger.error("pyaudio not installed. Run: pip install pyaudio")
            return

        pa = pyaudio.PyAudio()
        stream = pa.open(
            rate=SAMPLE_RATE,
            channels=CHANNELS,
            format=pyaudio.paInt16,
            input=True,
            input_device_index=self._device_index,
            frames_per_buffer=FRAME_SIZE,
        )

        in_speech = False
        speech_frames: list[AudioFrame] = []
        silence_frames = 0
        max_silence_frames = int(SPEECH_BUFFER_MS / FRAME_DURATION_MS)
        max_speech_frames = int(MAX_SEGMENT_DURATION_S * 1000 / FRAME_DURATION_MS)

        logger.info("AudioCapture: streaming started.")
        try:
            while self._running:
                raw = stream.read(FRAME_SIZE, exception_on_overflow=False)
                ts = time.monotonic()
                is_speech = self._vad.is_speech(raw)
                frame = AudioFrame(data=raw, timestamp=ts, is_speech=is_speech)

                if self._on_frame:
                    self._on_frame(frame)

                if is_speech:
                    silence_frames = 0
                    if not in_speech:
                        in_speech = True
                        speech_frames = []
                    speech_frames.append(frame)

                    # Force flush if segment is too long
                    if len(speech_frames) >= max_speech_frames:
                        self._emit_segment(speech_frames)
                        speech_frames = []
                        in_speech = False
                else:
                    if in_speech:
                        silence_frames += 1
                        speech_frames.append(frame)  # Keep trailing silence
                        if silence_frames >= max_silence_frames:
                            self._emit_segment(speech_frames)
                            speech_frames = []
                            in_speech = False
                            silence_frames = 0
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    def _emit_segment(self, frames: list[AudioFrame]):
        if not frames:
            return
        seg = SpeechSegment(
            frames=frames,
            start_time=frames[0].timestamp,
            end_time=frames[-1].timestamp,
        )
        if seg.duration_s > 0.3:   # Filter very short noise bursts
            self._segment_queue.put(seg)
            logger.debug(f"Emitted segment: {seg.duration_s:.2f}s")


# ── Synthetic audio capture for testing (no microphone needed) ──────────────

class SyntheticAudioCapture(AudioCapture):
    """
    Plays back pre-recorded WAV/MP3 files or synthetic scam scripts
    through the same interface as AudioCapture.
    Useful for CI testing and demo without a microphone.
    """

    def __init__(self, audio_file: Optional[str] = None, script_text: Optional[str] = None):
        super().__init__()
        self._audio_file = audio_file
        self._script_text = script_text

    def stream_segments(self) -> Generator[SpeechSegment, None, None]:
        """Yield segments from a WAV file or synthetic silence + text."""
        if self._audio_file:
            yield from self._stream_from_file()
        elif self._script_text:
            yield from self._stream_synthetic()

    def _stream_from_file(self) -> Generator[SpeechSegment, None, None]:
        import wave
        try:
            with wave.open(self._audio_file, "rb") as wf:
                assert wf.getframerate() == SAMPLE_RATE, "WAV must be 16kHz"
                assert wf.getnchannels() == 1, "WAV must be mono"
                while True:
                    raw = wf.readframes(FRAME_SIZE)
                    if not raw:
                        break
                    frames = [AudioFrame(data=raw, timestamp=time.monotonic(), is_speech=True)]
                    seg = SpeechSegment(
                        frames=frames,
                        start_time=time.monotonic(),
                        end_time=time.monotonic() + FRAME_DURATION_MS / 1000,
                    )
                    yield seg
                    time.sleep(FRAME_DURATION_MS / 1000)   # Real-time pacing
        except Exception as e:
            logger.error(f"SyntheticAudioCapture file error: {e}")

    def _stream_synthetic(self) -> Generator[SpeechSegment, None, None]:
        """
        For demo mode: generate silent audio segments labelled with the
        script text. The ASR stage is skipped and text injected directly.
        """
        lines = self._script_text.strip().split("\n")
        for line in lines:
            silence = np.zeros(SAMPLE_RATE * 2, dtype=np.int16)  # 2s silence
            frames = [AudioFrame(
                data=silence.tobytes(),
                timestamp=time.monotonic(),
                is_speech=True,
            )]
            seg = SpeechSegment(
                frames=frames,
                start_time=time.monotonic(),
                end_time=time.monotonic() + 2.0,
            )
            seg._injected_text = line.strip()   # Bypass ASR in demo mode
            yield seg
            time.sleep(1.0)
