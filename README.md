# ScamShield Live 🛡️

> **On-device, real-time scam detection for live phone calls — powered by Snapdragon NPU**  
> Qualcomm Snapdragon AI Lab Build & Present Challenge 2026

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Platform: Snapdragon X Elite](https://img.shields.io/badge/Platform-Snapdragon%20X%20Elite-red.svg)](https://www.qualcomm.com/snapdragon/ai-lab)

---

## The Problem

India loses **₹11,333 crore annually** to phone scams. Scammers follow a predictable psychological playbook — authority claim → fear creation → isolation → payment demand. No technology today warns the victim **while the call is happening**.

## The Solution

ScamShield Live listens passively during any phone call, detects social-engineering patterns as they unfold, and delivers **real-time, explainable alerts** — entirely on-device. No audio ever leaves the device.

```
🚨 CRITICAL RISK  ·  Score: 9.2/10
──────────────────────────────────────────
✗ Authority Impersonation  — "This is CBI Officer Sharma"
✗ Arrest Threat            — "warrant issued in your name"
✗ Isolation Tactic         — "do not tell your family"
✗ Gift Card Demand         — "buy Google Play cards NOW"
──────────────────────────────────────────
⚡ This is a scam. Hang up. Call 1930.
```

---

## Architecture

```
Microphone
    │  30ms PCM frames
    ▼
┌──────────────────┐
│  VAD             │  Silero VAD · webrtcvad (CPU fallback)
│  (NPU target)    │  QNN / Hexagon HTP via ONNX Runtime
└────────┬─────────┘
         │ speech segments only
         ▼
┌──────────────────┐
│  ASR             │  Whisper-tiny.en · faster-whisper (CPU)
│  (NPU target)    │  Qualcomm AI Hub → QNN HTP (Snapdragon)
└────────┬─────────┘
         │ transcript text
         ▼
┌──────────────────┐
│  Scam Detector   │  11 pattern categories · 9 scam stages
│  (CPU)           │  Cumulative + escalation scoring
│  + SLM optional  │  Phi-3-mini-4k (AI Hub, NPU target)
└────────┬─────────┘
         │ risk score + evidence
         ▼
┌──────────────────┐
│  Alert UI        │  Explainable · quoted evidence
│                  │  Terminal (Rich) + desktop notification
└──────────────────┘
```

### NPU Deployment Stack

| Component | CPU Backend | NPU Target | Qualcomm Path |
|---|---|---|---|
| VAD | webrtcvad | Silero VAD INT8 | ONNX Runtime + QNN EP |
| ASR | faster-whisper | Whisper-tiny.en | Qualcomm AI Hub → QNN HTP |
| Intent SLM | bart-large-mnli | Phi-3-mini-4k | Qualcomm AI Hub → QNN HTP |

> ⚠️ **NPU performance figures are from Qualcomm AI Hub published benchmarks, not locally measured.**  
> Full QNN inference requires Snapdragon X Elite / 8 Elite hardware with the QAIRT SDK.

---

## Quick Start

### Prerequisites

- Python 3.11+ (AMD64 — **not** ARM64 Python on Windows)
- Microphone (or use demo mode — no mic required)
- Windows 11 or Linux

### 1. Clone & install

```bash
git clone https://github.com/Abh-igyan/ScamShield.git
cd ScamShield
pip install -r requirements.txt
```

### 2. Run the demo (no microphone needed)

```bash
# Default: digital arrest scam scenario
python scripts/run_demo.py

# Choose a scenario
python scripts/run_demo.py --scenario bank_fraud
python scripts/run_demo.py --scenario tech_support
python scripts/run_demo.py --scenario lottery
python scripts/run_demo.py --scenario safe   # Control: should score ~0

# Run all scenarios
python scripts/run_demo.py --all

# List all scenarios
python scripts/run_demo.py --list
```

### 3. Run with live microphone

```bash
python main.py
```

### 4. Run benchmarks

```bash
python scripts/benchmark.py
python scripts/benchmark.py --output results/benchmark.json
```

---

## NPU Setup (Snapdragon Hardware)

> Requires: Snapdragon X Elite PC, QAIRT SDK, Qualcomm AI Hub account

```bash
# 1. Install AI Hub SDK
pip install qai-hub qai-hub-models

# 2. Configure with your API token (get from app.aihub.qualcomm.com)
qai-hub configure --api_token YOUR_TOKEN

# 3. Download and compile NPU models
python src/npu/qai_hub_setup.py

# 4. Run with NPU enabled
python main.py --npu
```

---

## Detected Pattern Categories

| Category | Scam Stage | Weight | Example |
|---|---|---|---|
| Authority — Bank/RBI | AUTHORITY_CLAIM | 2.0 | "This is SBI fraud department" |
| Authority — Government | AUTHORITY_CLAIM | 2.5 | "CBI officer calling" |
| Authority — Tech Support | AUTHORITY_CLAIM | 1.5 | "Microsoft detected virus" |
| Urgency / Time Pressure | URGENCY_ESCALATION | 1.5 | "Act within 10 minutes" |
| Threat — Arrest/Legal | URGENCY_ESCALATION | 3.0 | "Arrest warrant issued" |
| Isolation / Secrecy | ISOLATION | 2.0 | "Don't tell your family" |
| Credential Request | CREDENTIAL_HARVEST | 3.5 | "Share your OTP" |
| Gift Card / Crypto | PAYMENT_DEMAND | 4.0 | "Buy Google Play cards" |
| Wire Transfer | PAYMENT_DEMAND | 3.5 | "Transfer via RTGS" |
| Remote Access | REMOTE_ACCESS | 4.5 | "Install AnyDesk" |
| **Digital Arrest** 🇮🇳 | URGENCY_ESCALATION | **5.0** | "You are under digital arrest" |
| Lottery / Prize | INITIAL_CONTACT | 2.0 | "You have won ₹15 lakh" |

### Stage Progression Model

Risk compounds when multiple scam stages are observed in sequence:

```
Stage 1 — Authority claim      +2.0 pts
Stage 2 — Urgency/threat       +4.5 pts  (×1.5 new-stage bonus)
Stage 3 — Isolation            +3.0 pts  (×1.5 bonus)
Stage 4 — Payment demand       +6.0 pts  (×1.5 bonus)
                               ──────────
Cumulative:                    15.5 → CRITICAL 🚨
```

This models the attacker's psychological playbook — not just isolated keywords.

---

## Benchmark Results (CPU Baseline)

> Measured on Intel i7-1260P, Windows 11, CPU only.

| Metric | Result | Notes |
|---|---|---|
| VAD latency | ~2 ms/frame | webrtcvad, CPU |
| ASR latency | ~420 ms/seg | Whisper-tiny.en, int8, CPU |
| Pattern engine | < 1 ms | Deterministic regex |
| **End-to-end** | **~430 ms** | Audio end → alert displayed |
| Accuracy | 93.3% | 28/30 test cases |
| Precision | 95.0% | 19 TP / 1 FP |
| Recall | 95.0% | 19 TP / 1 FN |
| F1 Score | 95.0% | |

> 🎯 **NPU target (Snapdragon X Elite):** < 150ms end-to-end  
> Source: Qualcomm AI Hub published Whisper-tiny.en benchmark. Not locally verified.

---

## Project Structure

```
ScamShield/
├── main.py                      # Entry point (live mic mode)
├── requirements.txt
├── requirements-npu.txt         # Additional NPU deps
├── src/
│   ├── core/
│   │   ├── audio_capture.py     # VAD + audio streaming
│   │   ├── asr_engine.py        # Whisper CPU + QNN bridge
│   │   ├── whisper_utils.py     # Mel spectrogram + greedy decode
│   │   ├── scam_detector.py     # Pattern engine + stage model
│   │   └── pipeline.py          # End-to-end orchestrator
│   ├── npu/
│   │   └── qai_hub_setup.py     # AI Hub model download + QNN setup
│   └── ui/
│       └── terminal_ui.py       # Rich terminal display
├── scripts/
│   ├── run_demo.py              # Demo runner (no mic needed)
│   └── benchmark.py             # Evaluation + accuracy metrics
├── models/                      # Downloaded model artifacts (gitignored)
├── results/                     # Benchmark output JSON
└── docs/
    └── architecture.md          # Detailed architecture notes
```

---

## Limitations & Honest Disclosures

- **NPU not locally verified** — QNN backend is architecturally complete but requires Snapdragon X Elite hardware for end-to-end testing. CPU fallback is fully functional.
- **English only** — Whisper-tiny.en; multilingual support planned.
- **Pattern-based** — Novel scam scripts not matching existing patterns may be missed; the optional SLM layer addresses this.
- **No phone call integration** — Currently captures system/microphone audio; native telephony API integration is a roadmap item.
- **Benchmark sample size** — 30 manually scripted test cases; larger real-world dataset needed for production validation.

---

## Roadmap

| Phase | Target |
|---|---|
| **Now** | MVP on CPU · QNN architecture ready |
| +1 mo | Validate full QNN pipeline on Snapdragon X Elite |
| +2 mo | System tray daemon — always-on, any call |
| +3 mo | Hindi / Tamil / Telugu multilingual ASR |
| +4 mo | Community scam pattern dataset |

---

## License

Apache 2.0 — see [LICENSE](LICENSE)

---

## Competition Context

Built for the **Qualcomm Snapdragon AI Lab Build & Present Challenge 2026**.  
Uses **Qualcomm AI Hub** for model optimization targeting Snapdragon X-series NPU.  
Scam pattern database covers India-specific fraud types including "Digital Arrest" scams.

> For questions or contributions, open an issue or PR.
