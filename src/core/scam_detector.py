"""
scam_detector.py — Pattern-based scam detection with stage tracking.

Architecture:
  1. Rule-based keyword/phrase matching (fast, deterministic, explainable)
  2. LLM-assisted intent classification (optional, runs on-device via local model)
  3. Scam stage progression tracker (detects escalation patterns over time)

The rule engine runs entirely on CPU in microseconds; the LLM layer is optional
and can be replaced with a QNN-compiled SLM (e.g., Phi-3-mini-4k-instruct on
Qualcomm AI Hub) for NPU acceleration on Snapdragon devices.

Qualcomm AI Hub reference for SLM:
  https://aihub.qualcomm.com/models/phi_3_mini_4k_instruct
  Status: Architecture designed for QNN SLM — NOT VERIFIED on hardware.
"""

from __future__ import annotations

import re
import time
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

logger = logging.getLogger(__name__)


# ── Risk Levels ──────────────────────────────────────────────────────────────

class RiskLevel(Enum):
    SAFE = "SAFE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def color(self) -> str:
        return {
            "SAFE": "green",
            "LOW": "yellow",
            "MEDIUM": "orange",
            "HIGH": "red",
            "CRITICAL": "darkred",
        }[self.value]

    @property
    def emoji(self) -> str:
        return {
            "SAFE": "✅",
            "LOW": "🟡",
            "MEDIUM": "🟠",
            "HIGH": "⚠️",
            "CRITICAL": "🚨",
        }[self.value]


# ── Scam Stages ───────────────────────────────────────────────────────────────

class ScamStage(Enum):
    """
    Stages of a social-engineering scam, modelled on FTC/AARP fraud research.
    Scams often progress through these stages sequentially.
    """
    INITIAL_CONTACT = auto()       # Caller establishes presence
    AUTHORITY_CLAIM = auto()       # Claims to be bank/gov/police/tech support
    PROBLEM_CREATION = auto()      # Creates a (fake) crisis
    URGENCY_ESCALATION = auto()    # Time pressure + threats
    ISOLATION = auto()             # "Don't tell anyone / don't hang up"
    CREDENTIAL_HARVEST = auto()    # Asks for OTP, password, SSN
    PAYMENT_DEMAND = auto()        # Gift cards, wire, crypto
    REMOTE_ACCESS = auto()         # Asks to install AnyDesk / TeamViewer
    DEEPENING = auto()             # Re-contacts victim for more


# ── Pattern Definitions ───────────────────────────────────────────────────────

@dataclass
class ScamPattern:
    name: str
    stage: ScamStage
    patterns: list[str]              # Regex patterns (case-insensitive)
    weight: float = 1.0             # Contribution to risk score
    description: str = ""
    mitigation: str = ""            # Advice for the victim


SCAM_PATTERNS: list[ScamPattern] = [
    ScamPattern(
        name="authority_impersonation_bank",
        stage=ScamStage.AUTHORITY_CLAIM,
        weight=2.0,
        patterns=[
            r"\b(calling from|this is|speaking from).{0,30}(bank|sbi|hdfc|icici|axis|rbi|reserve bank)",
            r"\b(fraud|suspicious|unauthorized).{0,30}(transaction|activity|access).{0,30}account",
            r"\bbank.{0,20}(security|fraud|investigation)\b",
            r"\byour account (has been|is|will be) (blocked|frozen|suspended)",
        ],
        description="Caller claims to be from a bank or financial regulator.",
        mitigation="Hang up and call your bank's official number directly.",
    ),
    ScamPattern(
        name="authority_impersonation_government",
        stage=ScamStage.AUTHORITY_CLAIM,
        weight=2.5,
        patterns=[
            r"\b(irs|cbi|ed|enforcement directorate|income tax|cyber crime|police|officer|department of|government of)",
            r"\byour (aadhaar|pan|ssn|social security).{0,30}(misused|fraudulent|linked to)",
            r"\b(arrest warrant|legal action|court order|fir|case registered)",
            r"\bnarcotics|money laundering|drug trafficking.{0,30}your name",
        ],
        description="Caller impersonates law enforcement or government agency.",
        mitigation="Legitimate agencies do not demand payment over phone. Contact the agency directly.",
    ),
    ScamPattern(
        name="authority_impersonation_tech",
        stage=ScamStage.AUTHORITY_CLAIM,
        weight=1.5,
        patterns=[
            r"\b(microsoft|apple|google|amazon|windows|tech support).{0,30}(detected|found|identified).{0,30}(virus|hack|breach|malware)",
            r"\byour computer (has been|is) (hacked|infected|compromised)",
            r"\bwe detected.{0,20}(suspicious|unusual|unauthorized).{0,20}activity.{0,20}(device|computer|account)",
        ],
        description="Tech support scam — fake virus/hack alert.",
        mitigation="Microsoft/Apple never cold-call users. Hang up immediately.",
    ),
    ScamPattern(
        name="urgency_time_pressure",
        stage=ScamStage.URGENCY_ESCALATION,
        weight=1.5,
        patterns=[
            r"\b(immediately|right now|within.{0,10}(minutes|hours|today)|urgent|emergency|last chance)",
            r"\b(if you (don't|do not|fail to).{0,20}(act|respond|comply|transfer))",
            r"\b(expire|expired|deadline|before it's too late|limited time)",
            r"\b(your account will be|you will be).{0,20}(closed|blocked|arrested|prosecuted)",
        ],
        description="Artificial urgency / time pressure tactics.",
        mitigation="Scammers create urgency to prevent rational thinking. Take time to verify.",
    ),
    ScamPattern(
        name="threat_arrest_legal",
        stage=ScamStage.URGENCY_ESCALATION,
        weight=3.0,
        patterns=[
            r"\b(arrest|arrested|arrested|warrant|jail|prison|prosecute|sued|legal action).{0,30}(you|your name|against you)",
            r"\b(police|officers).{0,20}(on their way|coming|dispatched).{0,20}(to your|your address|your home)",
            r"\b(non-bailable|fir|case filed).{0,20}(your|against)",
        ],
        description="Threatening arrest or legal consequences.",
        mitigation="No legitimate agency threatens arrest over phone for payment. This is a scam.",
    ),
    ScamPattern(
        name="isolation_secrecy",
        stage=ScamStage.ISOLATION,
        weight=2.0,
        patterns=[
            r"\b(don't tell|do not tell|don't inform|do not inform).{0,30}(anyone|family|spouse|friend|relative)",
            r"\b(keep this (confidential|secret|between us))",
            r"\b(don't hang up|do not disconnect|do not hang up|stay on the line|do not call back)",
            r"\b(hang up|disconnect).{0,20}(call|phone|line).{0,20}(not|don't|cannot|do not)",
            r"\b(this is a confidential|classified|sensitive).{0,20}(matter|case|investigation)",
        ],
        description="Attempting to isolate victim from support network.",
        mitigation="Legitimate institutions never ask you to hide calls from family. Alert someone now.",
    ),
    ScamPattern(
        name="credential_request",
        stage=ScamStage.CREDENTIAL_HARVEST,
        weight=3.5,
        patterns=[
            r"\b(otp|one.time.password|verification code|pin|password|cvv|card number|account number|routing number)",
            r"\b(share|tell|give|provide|read out|dictate).{0,20}(otp|code|pin|password)",
            r"\b(aadhaar|pan|ssn|social security|date of birth).{0,20}(number|details|verify)",
            r"\bwhat is your.{0,20}(password|pin|otp|code)",
        ],
        description="Requesting credentials, OTPs, or sensitive personal information.",
        mitigation="NEVER share OTPs, passwords, or PINs with anyone — including bank employees.",
    ),
    ScamPattern(
        name="payment_demand_gift_card",
        stage=ScamStage.PAYMENT_DEMAND,
        weight=4.0,
        patterns=[
            r"\b(gift card|google play|amazon gift|itunes|steam card|voucher)",
            r"\b(buy|purchase|get).{0,30}(gift card|google play|amazon|itunes|voucher)",
            r"\b(pay|payment|transfer|send).{0,30}(fine|fee|bail|charge|penalty)",
            r"\b(cryptocurrency|bitcoin|crypto|btc|ethereum|usdt|tether)",
        ],
        description="Payment demanded via gift cards, crypto, or wire transfer.",
        mitigation="No legitimate agency or business collects payment via gift cards. This is ALWAYS a scam.",
    ),
    ScamPattern(
        name="payment_demand_wire",
        stage=ScamStage.PAYMENT_DEMAND,
        weight=3.5,
        patterns=[
            r"\b(wire transfer|rtgs|neft|imps|upi|transfer.{0,10}amount)",
            r"\b(send|deposit|transfer).{0,20}(money|amount|rupees|dollars|funds)",
            r"\b(account number|beneficiary|recipient).{0,20}(transfer|deposit|send)",
        ],
        description="Direct bank transfer demanded.",
        mitigation="Verify any transfer request through an official channel before proceeding.",
    ),
    ScamPattern(
        name="remote_access_request",
        stage=ScamStage.REMOTE_ACCESS,
        weight=4.5,
        patterns=[
            r"\b(anydesk|teamviewer|quicksupport|remote|screen share|desk\s*365|ultraviewer|zoho assist)",
            r"\b(download|install|run).{0,30}(software|application|app|program|tool).{0,30}(computer|device|phone)",
            r"\b(allow|give|grant).{0,20}(access|control|permission).{0,20}(computer|screen|device)",
            r"\bscreen sharing.{0,20}(enable|allow|start|turn on)",
        ],
        description="Remote desktop / screen access requested.",
        mitigation="NEVER allow remote access to your device. This gives scammers full control.",
    ),
    ScamPattern(
        name="lottery_prize_offer",
        stage=ScamStage.INITIAL_CONTACT,
        weight=2.0,
        patterns=[
            r"\b(won|winner|selected|chosen|congratulations).{0,30}(prize|lottery|lucky draw|reward|cash|gift)",
            r"\b(claim your|collect your).{0,20}(prize|reward|winnings)",
            r"\byou have been.{0,20}(selected|chosen|picked).{0,20}(receive|win|get)",
        ],
        description="Advance fee fraud / lottery scam.",
        mitigation="You cannot win a lottery you didn't enter. This is a scam.",
    ),
    ScamPattern(
        name="digital_arrest_threat",
        stage=ScamStage.URGENCY_ESCALATION,
        weight=5.0,
        patterns=[
            r"\bdigital arrest\b",
            r"\b(you are under|placed under).{0,20}(digital|online|cyber).{0,20}arrest",
            r"\bcannot leave.{0,20}(home|house|premises).{0,30}(investigation|case|pending)",
        ],
        description="'Digital Arrest' — a scam specific to India using fake CBI/police calls.",
        mitigation="'Digital arrest' is NOT a legal concept in India. This is 100% a scam. Hang up.",
    ),
]


# ── Detection Result ──────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """Output of the scam detector for one transcript."""
    text: str
    matches: list[dict] = field(default_factory=list)   # [{pattern_name, stage, evidence}]
    risk_score: float = 0.0          # 0.0–10.0
    risk_level: RiskLevel = RiskLevel.SAFE
    timestamp: float = field(default_factory=time.monotonic)

    @property
    def top_threat(self) -> Optional[str]:
        if self.matches:
            return self.matches[0]["pattern_name"]
        return None

    def summary_lines(self) -> list[str]:
        lines = []
        for m in self.matches:
            lines.append(f"• {m['description']}")
        return lines


@dataclass
class EvidenceEvent:
    """A single piece of evidence added to the timeline."""
    timestamp: float
    pattern_name: str
    stage: ScamStage
    evidence_text: str        # Matched text excerpt
    description: str
    weight: float
    full_utterance: str       # Full sentence where match was found


@dataclass
class Alert:
    """A risk alert generated when threshold is crossed."""
    risk_level: RiskLevel
    risk_score: float
    active_stages: list[ScamStage]
    evidence_timeline: list[EvidenceEvent]
    mitigation: str
    timestamp: float = field(default_factory=time.monotonic)

    def format_console(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"  {self.risk_level.emoji}  {self.risk_level.value} RISK DETECTED",
            f"{'='*60}",
            f"  Risk Score: {self.risk_score:.1f}/10",
            f"  Active Scam Stages:",
        ]
        for stage in self.active_stages:
            lines.append(f"    ✗ {stage.name.replace('_', ' ').title()}")
        lines.append(f"\n  Evidence Timeline:")
        for ev in self.evidence_timeline[-5:]:
            lines.append(f'    [{ev.stage.name}] "{ev.evidence_text}"')
        lines.append(f"\n  ⚡ ACTION: {self.mitigation}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ── Scam Detector ─────────────────────────────────────────────────────────────

class ScamDetector:
    """
    Multi-layered scam detection engine.

    Layer 1: Compiled regex pattern matching — O(n) per transcript.
    Layer 2: Scam stage progression model — detects multi-step escalation.
    Layer 3 (optional): SLM intent classifier — runs on NPU if available.

    The stage progression model awards extra weight when multiple stages
    are observed in sequence, matching the behavioral fingerprint of real scams.
    """

    # Risk level thresholds (tunable)
    THRESHOLDS = {
        RiskLevel.LOW: 1.5,
        RiskLevel.MEDIUM: 3.0,
        RiskLevel.HIGH: 6.0,
        RiskLevel.CRITICAL: 9.0,
    }

    # Stage progression bonus — consecutive stages multiply base score
    STAGE_PROGRESSION_BONUS = 1.5

    def __init__(self, use_llm: bool = False):
        self._compiled_patterns = self._compile_patterns()
        self._use_llm = use_llm
        self._llm_classifier = None
        self._evidence_timeline: list[EvidenceEvent] = []
        self._stage_history: list[ScamStage] = []
        self._cumulative_score: float = 0.0
        self._session_start = time.monotonic()

        if use_llm:
            self._init_llm()

        logger.info(
            f"ScamDetector ready: {len(SCAM_PATTERNS)} patterns, "
            f"LLM={'enabled' if use_llm else 'disabled'}"
        )

    def _compile_patterns(self) -> list[tuple[ScamPattern, list[re.Pattern]]]:
        """Pre-compile all regex patterns at init time."""
        compiled = []
        for sp in SCAM_PATTERNS:
            patterns = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in sp.patterns]
            compiled.append((sp, patterns))
        return compiled

    def _init_llm(self):
        """
        Optional: load a local SLM (Phi-3-mini or similar) for intent classification.
        On Snapdragon with QNN: replace transformers pipeline with ONNX Runtime + QNN EP.
        """
        try:
            from transformers import pipeline
            self._llm_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",  # CPU fallback
                device=-1,   # CPU
            )
            logger.info("LLM intent classifier loaded (CPU).")
        except ImportError:
            logger.warning("transformers not installed; LLM layer disabled.")
        except Exception as e:
            logger.warning(f"LLM init error: {e}")

    def analyze(self, transcript_text: str, timestamp: Optional[float] = None) -> DetectionResult:
        """
        Analyze a transcript string.
        Returns a DetectionResult; updates internal evidence timeline.
        """
        if timestamp is None:
            timestamp = time.monotonic()

        matches = []
        new_score = 0.0

        for scam_pattern, compiled in self._compiled_patterns:
            for regex in compiled:
                match = regex.search(transcript_text)
                if match:
                    evidence_text = transcript_text[
                        max(0, match.start() - 20): min(len(transcript_text), match.end() + 20)
                    ].strip()

                    # Stage bonus: multiply weight if stage is new in this session
                    bonus = 1.0
                    if scam_pattern.stage in self._stage_history:
                        bonus = 0.5   # Seen before — reduce duplicate weight
                    elif len(self._stage_history) > 0:
                        bonus = self.STAGE_PROGRESSION_BONUS   # New stage detected — escalation!

                    weighted = scam_pattern.weight * bonus
                    new_score += weighted

                    ev = EvidenceEvent(
                        timestamp=timestamp,
                        pattern_name=scam_pattern.name,
                        stage=scam_pattern.stage,
                        evidence_text=evidence_text,
                        description=scam_pattern.description,
                        weight=weighted,
                        full_utterance=transcript_text,
                    )
                    self._evidence_timeline.append(ev)

                    if scam_pattern.stage not in self._stage_history:
                        self._stage_history.append(scam_pattern.stage)

                    matches.append({
                        "pattern_name": scam_pattern.name,
                        "stage": scam_pattern.stage,
                        "evidence": evidence_text,
                        "description": scam_pattern.description,
                        "weight": weighted,
                    })
                    break   # One match per pattern is enough

        # Optional LLM layer
        if self._use_llm and self._llm_classifier and not matches:
            llm_score = self._llm_analyze(transcript_text)
            new_score += llm_score

        self._cumulative_score = self._cumulative_score + new_score  # Uncapped for progression tracking
        risk_level = self._score_to_level(self._cumulative_score)

        result = DetectionResult(
            text=transcript_text,
            matches=matches,
            risk_score=self._cumulative_score,
            risk_level=risk_level,
            timestamp=timestamp,
        )

        if matches:
            logger.info(
                f"[DETECT] score={self._cumulative_score:.1f} | "
                f"level={risk_level.value} | "
                f"patterns={[m['pattern_name'] for m in matches]}"
            )

        return result

    def _llm_analyze(self, text: str) -> float:
        """
        Zero-shot intent classification via local SLM.
        Returns a score contribution 0.0–2.0.
        """
        try:
            labels = ["scam", "fraud", "social engineering", "legitimate"]
            result = self._llm_classifier(text, candidate_labels=labels)
            scam_idx = result["labels"].index("scam") if "scam" in result["labels"] else -1
            if scam_idx >= 0:
                prob = result["scores"][scam_idx]
                return prob * 2.0   # Scale to [0, 2]
        except Exception as e:
            logger.debug(f"LLM analyze error: {e}")
        return 0.0

    def _score_to_level(self, score: float) -> RiskLevel:
        if score >= self.THRESHOLDS[RiskLevel.CRITICAL]:
            return RiskLevel.CRITICAL
        elif score >= self.THRESHOLDS[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        elif score >= self.THRESHOLDS[RiskLevel.MEDIUM]:
            return RiskLevel.MEDIUM
        elif score >= self.THRESHOLDS[RiskLevel.LOW]:
            return RiskLevel.LOW
        return RiskLevel.SAFE

    def get_alert(self) -> Optional[Alert]:
        """
        Return an Alert if current risk >= MEDIUM, else None.
        Used by the UI layer to trigger notifications.
        """
        level = self._score_to_level(self._cumulative_score)
        if level in (RiskLevel.SAFE, RiskLevel.LOW):
            return None

        # Best mitigation from highest-weight evidence
        best_mitigation = "Trust your instincts. Hang up and verify through official channels."
        if self._evidence_timeline:
            # Sort by weight descending
            top_ev = sorted(self._evidence_timeline, key=lambda e: e.weight, reverse=True)
            for ev in top_ev:
                for sp in SCAM_PATTERNS:
                    if sp.name == ev.pattern_name and sp.mitigation:
                        best_mitigation = sp.mitigation
                        break
                break

        return Alert(
            risk_level=level,
            risk_score=self._cumulative_score,
            active_stages=list(self._stage_history),
            evidence_timeline=list(self._evidence_timeline),
            mitigation=best_mitigation,
        )

    def reset_session(self):
        """Reset for a new call session."""
        self._evidence_timeline.clear()
        self._stage_history.clear()
        self._cumulative_score = 0.0
        self._session_start = time.monotonic()
        logger.info("ScamDetector: session reset.")

    @property
    def evidence_timeline(self) -> list[EvidenceEvent]:
        return list(self._evidence_timeline)

    @property
    def cumulative_score(self) -> float:
        return self._cumulative_score

    @property
    def active_stages(self) -> list[ScamStage]:
        return list(self._stage_history)

    @property
    def session_duration_s(self) -> float:
        return time.monotonic() - self._session_start
