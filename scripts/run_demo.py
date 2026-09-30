#!/usr/bin/env python
"""
run_demo.py — ScamShield Live demo runner.

Runs a scripted scam scenario through the full pipeline without a microphone.
Perfect for CI, presentations, and testing on machines without audio hardware.

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --scenario digital_arrest
    python scripts/run_demo.py --scenario bank_fraud
    python scripts/run_demo.py --scenario tech_support
    python scripts/run_demo.py --list
"""

import sys
import argparse
import logging
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from core.pipeline import ScamShieldPipeline, PipelineConfig
from core.scam_detector import Alert, DetectionResult, RiskLevel
from core.asr_engine import Transcript
from ui.terminal_ui import TerminalUI, print_banner

logging.basicConfig(level=logging.WARNING)  # Suppress debug noise in demo

# ── Demo scripts ──────────────────────────────────────────────────────────────

SCENARIOS = {
    "digital_arrest": {
        "title": "Digital Arrest Scam (India)",
        "description": "Fake CBI officer claims digital arrest, demands payment",
        "script": """
Hello, am I speaking with the account holder? This is Deputy Inspector Sharma calling from the Central Bureau of Investigation, Cyber Crime Division.
We have detected that your Aadhaar number has been misused in a money laundering case.
A non-bailable arrest warrant has been issued in your name by the Delhi High Court.
You are now under digital arrest. You cannot leave your home or disconnect this call.
Do not tell your family about this call. This is a classified government investigation.
To avoid immediate arrest, you must transfer a processing fee of rupees fifty thousand to our secure account.
Please do not share this with anyone. Stay on the line and cooperate.
""",
    },
    "bank_fraud": {
        "title": "Bank Impersonation + OTP Harvest",
        "description": "Fake bank employee tries to extract OTP and card details",
        "script": """
Good morning, this is calling from SBI Fraud Prevention Department.
We have detected unauthorized transactions on your account totaling rupees twenty two thousand.
Your account has been temporarily blocked for security reasons.
To immediately reverse this and unblock your account, I need to verify your identity.
Could you please share the OTP that has just been sent to your registered mobile number?
Also I will need your debit card number and the CVV on the back for verification purposes.
This is an urgent matter and must be resolved within the next ten minutes or the account will be permanently closed.
""",
    },
    "tech_support": {
        "title": "Tech Support + Remote Access Scam",
        "description": "Fake Microsoft support requests remote access",
        "script": """
Hello this is John from Microsoft Windows Security Team.
We have detected that your computer has been hacked and is sending out malware to other computers on the internet.
This is a serious security breach and your data is at risk right now.
You need to act immediately before your computer is completely compromised.
Please download and install AnyDesk on your computer right now so our engineers can fix this remotely.
Once you install it, share the access code with me and we will clean your computer within minutes.
If you do not act immediately, we will have to suspend your Windows license and you will lose all your data.
""",
    },
    "lottery": {
        "title": "Lottery / Advance Fee Fraud",
        "description": "Fake prize claim with processing fee demand",
        "script": """
Congratulations! You have been selected as the lucky winner of our national prize draw.
You have won rupees fifteen lakh in our annual lucky draw contest.
To claim your prize, you need to pay a small processing and tax fee of rupees five thousand.
This must be done immediately as the offer expires within two hours.
Please keep this confidential and do not tell anyone until the prize is credited to your account.
You can transfer the amount via Google Pay or buy an Amazon gift card and share the code with us.
""",
    },
    "safe": {
        "title": "Safe Conversation (Control)",
        "description": "Normal conversation — should score near zero",
        "script": """
Hi, I wanted to check on the delivery status for my order.
Can you tell me when it will arrive?
Thank you, I placed the order last Tuesday.
Okay that sounds good, I will be home on Friday.
Thank you for the help, goodbye.
""",
    },
}


def run_scenario(name: str, verbose: bool = False):
    if name not in SCENARIOS:
        print(f"Unknown scenario '{name}'. Use --list to see available scenarios.")
        sys.exit(1)

    scenario = SCENARIOS[name]
    print_banner()
    print(f"  Scenario: {scenario['title']}")
    print(f"  {scenario['description']}")
    print()

    ui = TerminalUI()
    ui.print_header()

    config = PipelineConfig(
        demo_mode=True,
        demo_script=scenario["script"],
        log_transcripts=verbose,
    )
    pipeline = ScamShieldPipeline(config)
    pipeline.on_transcript = ui.update_transcript
    pipeline.on_detection = ui.update_detection
    pipeline.on_alert = ui.fire_alert

    try:
        pipeline.run()
    except KeyboardInterrupt:
        pass
    finally:
        ui.print_stats(pipeline.stats.report())
        print()
        d = pipeline.detector
        print(f"  Final risk score : {d.cumulative_score:.1f}/10")
        print(f"  Active stages    : {[s.name for s in d.active_stages]}")
        print(f"  Evidence items   : {len(d.evidence_timeline)}")
        print()


def main():
    parser = argparse.ArgumentParser(description="ScamShield Live — Demo Runner")
    parser.add_argument(
        "--scenario", default="digital_arrest",
        help="Scenario to run (default: digital_arrest)"
    )
    parser.add_argument("--list", action="store_true", help="List available scenarios")
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    parser.add_argument("--verbose", action="store_true", help="Show debug logs")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if args.list:
        print("\nAvailable scenarios:")
        for k, v in SCENARIOS.items():
            print(f"  {k:<20} — {v['title']}")
        print()
        return

    if args.all:
        for name in SCENARIOS:
            print(f"\n{'='*60}")
            run_scenario(name, verbose=args.verbose)
            time.sleep(1)
    else:
        run_scenario(args.scenario, verbose=args.verbose)


if __name__ == "__main__":
    main()
