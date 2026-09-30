# Architecture Notes

## Pipeline Design

The ScamShield Live pipeline is designed around three principles:
1. **NPU-first**: Every inference step targets the Hexagon HTP NPU via QNN
2. **Fallback-safe**: CPU backends exist for every NPU component
3. **Explainable output**: Every alert quotes exact evidence

## Why VAD-gated ASR?

Running ASR continuously wastes NPU cycles on silence. Silero VAD runs at ~2ms/frame on CPU (target: <1ms on NPU), gating ASR to only speech-containing frames. This reduces ASR load by ~60-70% in typical phone call scenarios.

## Scam Stage Model

Unlike keyword filters (which have high FP rates), ScamShield tracks the *sequence* of scam tactics. The stage progression bonus multiplies risk score when a new scam stage is observed after a prior one:

- Same stage repeat: 0.5× weight (deduplication)
- New stage after prior: 1.5× weight (escalation signal)

This means a caller who first claims authority, then threatens arrest, then demands payment scores dramatically higher than random keyword hits.

## QNN Deployment Path

```
PyTorch model
    │
    │ torch.onnx.export()
    ▼
ONNX model
    │
    │ qai-hub compile (cloud)
    ▼
QNN binary (.bin)
    │
    │ onnxruntime-qnn + QNNExecutionProvider
    ▼
Hexagon HTP (NPU on-device)
```

All three AI models (VAD, ASR, SLM) follow this same path via Qualcomm AI Hub.

## Qualcomm AI Hub Integration

The `qai_hub` Python SDK allows:
- `submit_compile_job()` — compile ONNX/TorchScript to QNN binary
- `submit_profile_job()` — measure real latency on Snapdragon hardware
- `submit_inference_job()` — run inference on cloud-connected device

For competition: profiling jobs give us legitimate NPU benchmark numbers without requiring local Snapdragon hardware.
