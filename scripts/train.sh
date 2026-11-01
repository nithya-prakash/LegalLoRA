#!/usr/bin/env bash
# Train the LoRA adapter. Run from the repo root.
set -euo pipefail
.venv/bin/python -m mlx_lm lora --config configs/lora_qwen3b.yaml
