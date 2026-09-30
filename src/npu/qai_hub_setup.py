"""
qai_hub_setup.py — Qualcomm AI Hub model download and QNN deployment helper.

This script automates:
  1. Authenticating with Qualcomm AI Hub (qai-hub Python SDK)
  2. Downloading pre-optimized model artifacts for Snapdragon X Elite
  3. Placing them in the models/ directory for use by the ASR/VAD backends

Qualcomm AI Hub models targeted:
  - WhisperTinyEn   : Speech recognition (ASR)
  - (Phi-3-mini-4k  : Intent classification SLM — optional)

Setup:
  pip install qai-hub qai-hub-models
  qai-hub configure --api_token YOUR_TOKEN
  python src/npu/qai_hub_setup.py

Status: Architecture verified. Model download requires qai-hub account + API token.
        Full QNN inference requires Snapdragon X Elite hardware.

Qualcomm AI Hub docs: https://app.aihub.qualcomm.com
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parents[2] / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Target Snapdragon device for compilation
# Run `qai-hub list-devices` to see available devices
TARGET_DEVICE = "Snapdragon X Elite CRD"   # or "Snapdragon 8 Elite"

# Models to download from AI Hub
AI_HUB_MODELS = {
    "whisper_tiny_en": {
        "model_id": "mnjl9xovq",       # Qualcomm AI Hub model ID — verify at aihub.qualcomm.com
        "description": "Whisper Tiny English — speech recognition",
        "output_name": "whisper_tiny_en_qnn.onnx",
        "required": True,
    },
    "silero_vad": {
        "model_id": None,              # Must export manually — see export_silero_vad()
        "description": "Silero VAD — voice activity detection",
        "output_name": "silero_vad_quantized.onnx",
        "required": False,
    },
}


def check_qai_hub() -> bool:
    """Check if qai-hub is installed and configured."""
    try:
        import qai_hub
        # Try a lightweight API call to verify auth
        devices = qai_hub.get_devices()
        logger.info(f"qai-hub authenticated. Devices available: {len(devices)}")
        return True
    except ImportError:
        logger.error(
            "qai-hub not installed.\n"
            "Install: pip install qai-hub qai-hub-models\n"
            "Configure: qai-hub configure --api_token YOUR_TOKEN"
        )
        return False
    except Exception as e:
        logger.error(f"qai-hub auth error: {e}\n"
                     "Run: qai-hub configure --api_token YOUR_TOKEN")
        return False


def download_whisper_qnn() -> Path | None:
    """
    Download Whisper-tiny.en QNN artifacts from Qualcomm AI Hub.

    The AI Hub hosts pre-compiled, INT8-quantized versions of Whisper
    optimised for the Hexagon HTP (NPU) backend.

    Returns path to downloaded model or None on failure.
    """
    try:
        import qai_hub
        from qai_hub_models.models.whisper_tiny_en import Model as WhisperTinyEn

        logger.info("Downloading Whisper-tiny.en from Qualcomm AI Hub...")

        # Compile for target device (submits job to AI Hub cloud)
        model = WhisperTinyEn.from_pretrained()

        # Export to ONNX
        sample_input = model.sample_inputs()
        compile_job = qai_hub.submit_compile_job(
            model=model.convert_to_torchscript(sample_input),
            device=qai_hub.Device(TARGET_DEVICE),
            name="ScamShield-WhisperTinyEn",
            options="--target_runtime qnn_lib_aarch64_android",
        )

        logger.info(f"Compile job submitted: {compile_job.job_id}")
        logger.info("Waiting for compilation... (this may take a few minutes)")
        compile_job.wait()

        if compile_job.get_status().success:
            out_path = MODELS_DIR / "whisper_tiny_en_qnn.bin"
            compile_job.download_target_model(str(out_path))
            logger.info(f"Whisper QNN model saved to {out_path}")
            return out_path
        else:
            logger.error(f"Compile job failed: {compile_job.get_status()}")
            return None

    except ImportError:
        logger.error(
            "qai-hub-models not installed.\n"
            "Install: pip install qai-hub-models"
        )
        return None
    except Exception as e:
        logger.error(f"AI Hub download error: {e}")
        return None


def export_silero_vad_onnx() -> Path | None:
    """
    Export Silero VAD model to ONNX for QNN deployment.
    Silero VAD is not on AI Hub — export locally then submit for compilation.
    """
    out_path = MODELS_DIR / "silero_vad.onnx"
    if out_path.exists():
        logger.info(f"Silero VAD ONNX already exists: {out_path}")
        return out_path
    try:
        import torch
        logger.info("Exporting Silero VAD to ONNX...")
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        model.eval()
        dummy = torch.zeros(1, 512)  # 16kHz, 32ms frame
        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch", 1: "frames"}},
            opset_version=14,
        )
        logger.info(f"Silero VAD exported to {out_path}")
        return out_path
    except ImportError:
        logger.error("torch not installed. pip install torch")
        return None
    except Exception as e:
        logger.error(f"Silero VAD export error: {e}")
        return None


def profile_model_on_device(model_path: Path, model_name: str):
    """
    Submit a profiling job to AI Hub to measure latency on Snapdragon hardware.
    This is how we get real NPU benchmark numbers.
    """
    try:
        import qai_hub
        logger.info(f"Submitting profile job for {model_name} on {TARGET_DEVICE}...")
        profile_job = qai_hub.submit_profile_job(
            model=str(model_path),
            device=qai_hub.Device(TARGET_DEVICE),
            name=f"ScamShield-Profile-{model_name}",
        )
        logger.info(f"Profile job: {profile_job.job_id}")
        profile_job.wait()
        results = profile_job.download_profile()
        logger.info(f"Profile results for {model_name}:")
        logger.info(f"  Inference time: {results.get('inference_time_ms', 'N/A')} ms")
        return results
    except Exception as e:
        logger.error(f"Profile job error: {e}")
        return None


def setup_all():
    """Run full setup: check auth, download models, report status."""
    print("\n⚡ ScamShield Live — AI Hub Setup")
    print("=" * 50)

    if not check_qai_hub():
        print("\n❌ qai-hub not configured. Follow setup instructions above.")
        sys.exit(1)

    print("\n✅ qai-hub authenticated")
    print(f"📁 Models directory: {MODELS_DIR}")

    # Whisper
    print("\n1. Downloading Whisper-tiny.en (NPU-optimized)...")
    whisper_path = download_whisper_qnn()
    if whisper_path:
        print(f"   ✅ {whisper_path.name}")
    else:
        print("   ⚠️  Whisper QNN download failed — ASR will run on CPU")

    # Silero VAD
    print("\n2. Exporting Silero VAD to ONNX...")
    vad_path = export_silero_vad_onnx()
    if vad_path:
        print(f"   ✅ {vad_path.name}")
    else:
        print("   ⚠️  VAD export failed — will use webrtcvad (CPU)")

    print("\n" + "=" * 50)
    print("Setup complete. Run: python scripts/run_demo.py")
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    setup_all()
