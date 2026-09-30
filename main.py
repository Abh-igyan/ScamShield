#!/usr/bin/env python
"""
main.py — ScamShield Live entry point.

Usage:
    python main.py                  # Live mic, CPU mode
    python main.py --npu            # Live mic, NPU mode (Snapdragon required)
    python main.py --demo           # Demo mode, no mic needed
    python main.py --demo --scenario digital_arrest
    python main.py --device 1       # Select audio input device index
    python main.py --list-devices   # Show available audio devices
"""

import argparse
import logging
import sys

from src.core.pipeline import ScamShieldPipeline, PipelineConfig
from src.core.scam_detector import RiskLevel
from src.ui.terminal_ui import TerminalUI, print_banner


def list_audio_devices():
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        print("\nAvailable audio input devices:")
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                print(f"  [{i}] {info['name']}")
        pa.terminate()
    except ImportError:
        print("pyaudio not installed. pip install pyaudio")


DEMO_SCRIPTS = {
    "digital_arrest": """
This is Deputy Inspector Sharma calling from the Central Bureau of Investigation.
Your Aadhaar number has been used in a money laundering case.
A non-bailable arrest warrant has been issued in your name.
You are now under digital arrest. Do not disconnect this call.
Do not tell your family about this investigation.
To avoid immediate arrest you must transfer fifty thousand rupees to our secure account.
""",
    "bank_fraud": """
Good morning, this is calling from SBI Fraud Prevention Department.
We have detected suspicious unauthorized transactions on your account.
Your account has been temporarily blocked. To unblock it immediately,
please share the OTP that was just sent to your registered mobile number.
Also provide your debit card number and the CVV for verification.
This must be done in the next ten minutes or your account will be permanently closed.
""",
    "tech_support": """
Hello this is John from Microsoft Windows Security Team.
We have detected that your computer has been hacked and is infected with malware.
You need to act immediately. Please download and install AnyDesk right now.
Share the access code with me so our engineers can fix your computer remotely.
If you do not cooperate your Windows license will be suspended and all data lost.
""",
}


def main():
    parser = argparse.ArgumentParser(
        description="ScamShield Live — On-Device Scam Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        # Live mic, CPU
  python main.py --npu                  # Live mic, NPU (Snapdragon required)
  python main.py --demo                 # Demo: digital_arrest scenario
  python main.py --demo --scenario bank_fraud
  python main.py --list-devices         # Show audio devices
        """
    )
    parser.add_argument("--npu", action="store_true",
                        help="Enable Snapdragon NPU backends (requires QAIRT SDK)")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo mode (no microphone required)")
    parser.add_argument("--scenario", default="digital_arrest",
                        choices=list(DEMO_SCRIPTS.keys()),
                        help="Demo scenario (default: digital_arrest)")
    parser.add_argument("--device", type=int, default=None,
                        help="Audio input device index")
    parser.add_argument("--list-devices", action="store_true",
                        help="List available audio input devices and exit")
    parser.add_argument("--model", default="tiny.en",
                        choices=["tiny.en", "base.en", "small.en"],
                        help="Whisper model size (default: tiny.en)")
    parser.add_argument("--llm", action="store_true",
                        help="Enable LLM intent classifier layer")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    if args.list_devices:
        list_audio_devices()
        return

    print_banner()

    if args.npu:
        print("  ⚡ NPU mode enabled (requires Snapdragon X Elite + QAIRT SDK)")
        print("  Run `python src/npu/qai_hub_setup.py` first to download models.")
        print()

    config = PipelineConfig(
        use_npu=args.npu,
        asr_model_size=args.model,
        use_llm=args.llm,
        demo_mode=args.demo,
        demo_script=DEMO_SCRIPTS.get(args.scenario) if args.demo else None,
        audio_device_index=args.device,
    )

    ui = TerminalUI()
    pipeline = ScamShieldPipeline(config)
    pipeline.on_transcript = ui.update_transcript
    pipeline.on_detection = ui.update_detection
    pipeline.on_alert = ui.fire_alert

    mode = "DEMO" if args.demo else "LIVE MIC"
    npu = "NPU" if args.npu else "CPU"
    print(f"  Mode    : {mode}")
    print(f"  Backend : {npu}")
    print(f"  Model   : Whisper-{args.model}")
    if args.demo:
        print(f"  Scenario: {args.scenario}")
    print()
    print("  Listening... (Ctrl+C to stop)")
    print()

    try:
        pipeline.run()
    except KeyboardInterrupt:
        pass
    finally:
        ui.print_stats(pipeline.stats.report())


if __name__ == "__main__":
    main()
