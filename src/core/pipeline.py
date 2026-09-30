"""
pipeline.py — Main orchestrator for ScamShield Live.

Wires together: AudioCapture → ASR → ScamDetector → Alert UI
Supports both live microphone mode and demo/file mode.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .audio_capture import AudioCapture, SyntheticAudioCapture, SpeechSegment
from .asr_engine import ASREngine, Transcript
from .scam_detector import ScamDetector, DetectionResult, Alert, RiskLevel

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Runtime configuration for the pipeline."""
    use_npu: bool = False               # Enable QNN/NPU backends
    asr_model_size: str = "tiny.en"     # Whisper model size
    asr_device: str = "cpu"             # cpu | cuda
    use_llm: bool = False               # Enable optional SLM layer
    alert_threshold: RiskLevel = RiskLevel.MEDIUM
    audio_device_index: Optional[int] = None
    demo_mode: bool = False             # Run with scripted scenarios
    demo_script: Optional[str] = None
    demo_audio_file: Optional[str] = None
    log_transcripts: bool = True
    session_id: str = field(default_factory=lambda: str(int(time.time())))


@dataclass
class PipelineStats:
    """Running statistics for the current session."""
    segments_processed: int = 0
    transcripts_generated: int = 0
    alerts_fired: int = 0
    session_start: float = field(default_factory=time.monotonic)
    total_asr_latency_ms: float = 0.0
    total_detect_latency_ms: float = 0.0

    @property
    def avg_asr_latency_ms(self) -> float:
        if self.transcripts_generated == 0:
            return 0.0
        return self.total_asr_latency_ms / self.transcripts_generated

    @property
    def avg_detect_latency_ms(self) -> float:
        if self.segments_processed == 0:
            return 0.0
        return self.total_detect_latency_ms / self.segments_processed

    @property
    def session_duration_s(self) -> float:
        return time.monotonic() - self.session_start

    def report(self) -> str:
        return (
            f"Session: {self.session_duration_s:.1f}s | "
            f"Segments: {self.segments_processed} | "
            f"Transcripts: {self.transcripts_generated} | "
            f"Alerts: {self.alerts_fired} | "
            f"ASR avg: {self.avg_asr_latency_ms:.0f}ms | "
            f"Detect avg: {self.avg_detect_latency_ms:.1f}ms"
        )


class ScamShieldPipeline:
    """
    Full end-to-end pipeline:
      AudioCapture → VAD → ASREngine → ScamDetector → Alert callbacks

    Usage (live mic):
        pipeline = ScamShieldPipeline(PipelineConfig())
        pipeline.on_alert = my_alert_handler
        pipeline.run()   # blocks; Ctrl+C to stop

    Usage (demo mode):
        config = PipelineConfig(demo_mode=True, demo_script=SCAM_SCRIPT)
        pipeline = ScamShieldPipeline(config)
        pipeline.run()
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.stats = PipelineStats()

        # Callbacks — set these before calling run()
        self.on_transcript: Optional[Callable[[Transcript], None]] = None
        self.on_detection: Optional[Callable[[DetectionResult], None]] = None
        self.on_alert: Optional[Callable[[Alert], None]] = None
        self.on_segment: Optional[Callable[[SpeechSegment], None]] = None

        self._running = False
        self._last_alert_level = RiskLevel.SAFE

        # Build components
        self._asr = ASREngine(
            use_npu=config.use_npu,
            model_size=config.asr_model_size,
            device=config.asr_device,
        )
        self._detector = ScamDetector(use_llm=config.use_llm)

        # Build audio source
        if config.demo_mode:
            if config.demo_audio_file:
                self._audio = SyntheticAudioCapture(audio_file=config.demo_audio_file)
            else:
                self._audio = SyntheticAudioCapture(script_text=config.demo_script)
        else:
            self._audio = AudioCapture(
                device_index=config.audio_device_index,
                use_npu=config.use_npu,
            )

        logger.info(
            f"Pipeline ready | NPU={config.use_npu} | "
            f"ASR={self._asr.backend_name} | Demo={config.demo_mode}"
        )

    def run(self):
        """Run the pipeline. Blocks until stopped."""
        self._running = True
        self._detector.reset_session()
        logger.info("ScamShield pipeline started. Press Ctrl+C to stop.")
        try:
            for segment in self._audio.stream_segments():
                if not self._running:
                    break
                self._process_segment(segment)
        except KeyboardInterrupt:
            logger.info("Pipeline stopped by user.")
        finally:
            self._running = False
            logger.info(f"Final stats — {self.stats.report()}")

    def stop(self):
        self._running = False

    def _process_segment(self, segment: SpeechSegment):
        self.stats.segments_processed += 1

        if self.on_segment:
            self.on_segment(segment)

        # ── ASR ──────────────────────────────────────────────────────────────
        t0 = time.monotonic()
        transcript = self._asr.process_segment(segment)
        asr_ms = (time.monotonic() - t0) * 1000
        self.stats.total_asr_latency_ms += asr_ms

        if transcript is None:
            return

        self.stats.transcripts_generated += 1
        if self.config.log_transcripts:
            logger.info(f"[ASR] {transcript.text}")

        if self.on_transcript:
            self.on_transcript(transcript)

        # ── Scam Detection ───────────────────────────────────────────────────
        t1 = time.monotonic()
        result = self._detector.analyze(transcript.text, timestamp=transcript.timestamp)
        detect_ms = (time.monotonic() - t1) * 1000
        self.stats.total_detect_latency_ms += detect_ms

        if self.on_detection:
            self.on_detection(result)

        # ── Alert Logic ──────────────────────────────────────────────────────
        if result.risk_level.value in (
            RiskLevel.MEDIUM.value,
            RiskLevel.HIGH.value,
            RiskLevel.CRITICAL.value,
        ):
            # Only fire alert if risk level increased or new evidence
            if (
                result.risk_level != self._last_alert_level
                or result.matches  # New patterns just matched
            ):
                alert = self._detector.get_alert()
                if alert and self.on_alert:
                    self.stats.alerts_fired += 1
                    self._last_alert_level = result.risk_level
                    self.on_alert(alert)

    @property
    def detector(self) -> ScamDetector:
        return self._detector

    @property
    def asr(self) -> ASREngine:
        return self._asr
