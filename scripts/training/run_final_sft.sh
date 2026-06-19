#!/bin/bash
set -e

echo "Starting full 2000-step SFT training..."
./venv/bin/python -m mlx_lm.lora --config sft_config.yaml

echo "Training complete. Now generating checkpoint evaluations..."
./venv/bin/python scripts/9_evaluate_checkpoints.py

echo "All tasks complete!"
