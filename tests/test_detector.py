"""
tests/test_detector.py — Unit tests for ScamDetector.
Run: pytest tests/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest
from core.scam_detector import ScamDetector, RiskLevel, ScamStage


@pytest.fixture
def detector():
    return ScamDetector(use_llm=False)


# ── Safe inputs should score low ──────────────────────────────────────────────

def test_safe_greeting(detector):
    r = detector.analyze("Hello, how can I help you today?")
    assert r.risk_level == RiskLevel.SAFE
    assert r.risk_score == 0.0


def test_safe_order_enquiry(detector):
    r = detector.analyze("When will my order arrive? I placed it last Tuesday.")
    assert r.risk_level == RiskLevel.SAFE


def test_safe_appointment(detector):
    r = detector.analyze("I'd like to reschedule my appointment to Friday afternoon.")
    assert r.risk_level == RiskLevel.SAFE


# ── Authority impersonation ───────────────────────────────────────────────────

def test_cbi_impersonation(detector):
    r = detector.analyze("This is CBI officer Sharma calling regarding your case.")
    assert r.risk_level != RiskLevel.SAFE
    patterns = [m["pattern_name"] for m in r.matches]
    assert "authority_impersonation_government" in patterns


def test_bank_impersonation(detector):
    r = detector.analyze("I'm calling from SBI fraud department about suspicious transactions.")
    assert r.risk_level != RiskLevel.SAFE
    patterns = [m["pattern_name"] for m in r.matches]
    assert "authority_impersonation_bank" in patterns


def test_microsoft_impersonation(detector):
    r = detector.analyze("Microsoft detected a virus on your computer. Your device has been hacked.")
    assert r.risk_level != RiskLevel.SAFE


# ── Credential harvesting ─────────────────────────────────────────────────────

def test_otp_request(detector):
    r = detector.analyze("Please share the OTP sent to your mobile number right now.")
    patterns = [m["pattern_name"] for m in r.matches]
    assert "credential_request" in patterns
    assert r.risk_score >= 3.0


def test_cvv_request(detector):
    r = detector.analyze("Please provide your card number and CVV for verification.")
    assert r.risk_level != RiskLevel.SAFE


# ── Payment demands ───────────────────────────────────────────────────────────

def test_gift_card_demand(detector):
    r = detector.analyze("Buy Google Play gift cards worth fifty thousand rupees immediately.")
    patterns = [m["pattern_name"] for m in r.matches]
    assert "payment_demand_gift_card" in patterns
    assert r.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)


def test_crypto_payment(detector):
    r = detector.analyze("Send USDT cryptocurrency to this wallet address to clear the case.")
    assert r.risk_level != RiskLevel.SAFE


def test_wire_transfer(detector):
    r = detector.analyze("Transfer the amount via RTGS to our secure account number.")
    patterns = [m["pattern_name"] for m in r.matches]
    assert "payment_demand_wire" in patterns


# ── Remote access ─────────────────────────────────────────────────────────────

def test_anydesk_request(detector):
    r = detector.analyze("Please download and install AnyDesk so we can fix your computer remotely.")
    patterns = [m["pattern_name"] for m in r.matches]
    assert "remote_access_request" in patterns
    assert r.risk_score >= 4.0


def test_teamviewer_request(detector):
    r = detector.analyze("Install TeamViewer and grant us access to your screen.")
    assert r.risk_level != RiskLevel.SAFE


# ── Isolation tactics ─────────────────────────────────────────────────────────

def test_isolation_family(detector):
    r = detector.analyze("Do not tell your family or anyone about this call.")
    patterns = [m["pattern_name"] for m in r.matches]
    assert "isolation_secrecy" in patterns


def test_dont_disconnect(detector):
    r = detector.analyze("Do not hang up this call or disconnect under any circumstances.")
    assert r.risk_level != RiskLevel.SAFE


# ── India-specific: Digital Arrest ───────────────────────────────────────────

def test_digital_arrest(detector):
    r = detector.analyze("You are under digital arrest and cannot leave your home.")
    patterns = [m["pattern_name"] for m in r.matches]
    assert "digital_arrest_threat" in patterns
    assert r.risk_score >= 5.0


def test_digital_arrest_keyword(detector):
    r = detector.analyze("This is a digital arrest situation under cyber crime law.")
    assert r.risk_level != RiskLevel.SAFE


# ── Stage progression ─────────────────────────────────────────────────────────

def test_stage_progression_escalates(detector):
    """Multi-stage scam should score significantly higher than single stage."""
    # Fresh detector for this test
    d = ScamDetector()

    r1 = d.analyze("This is CBI officer Sharma calling.")
    score_after_stage1 = d.cumulative_score

    r2 = d.analyze("An arrest warrant has been issued in your name.")
    score_after_stage2 = d.cumulative_score

    r3 = d.analyze("Do not tell your family about this call.")
    score_after_stage3 = d.cumulative_score

    r4 = d.analyze("Buy Google Play gift cards worth fifty thousand rupees now.")
    final_score = d.cumulative_score

    # Score should increase at each stage
    assert score_after_stage2 > score_after_stage1
    assert score_after_stage3 > score_after_stage2
    assert final_score > score_after_stage3

    # Multi-stage should reach at least HIGH
    assert d.active_stages.__len__() >= 2
    assert final_score >= 6.0


def test_session_reset(detector):
    """After reset, score should be zero."""
    detector.analyze("CBI officer calling, arrest warrant issued, buy gift cards now.")
    assert detector.cumulative_score > 0
    detector.reset_session()
    assert detector.cumulative_score == 0.0
    assert len(detector.active_stages) == 0
    assert len(detector.evidence_timeline) == 0


def test_alert_returned_on_high_risk(detector):
    """get_alert() should return an Alert when score >= MEDIUM."""
    assert detector.get_alert() is None   # No detections yet

    detector.analyze("CBI officer calling. Arrest warrant in your name. Buy gift cards NOW.")
    alert = detector.get_alert()
    assert alert is not None
    assert alert.risk_score > 0
    assert len(alert.active_stages) > 0
    assert alert.mitigation != ""
