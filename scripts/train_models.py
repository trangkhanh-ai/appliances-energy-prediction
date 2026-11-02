"""Train the documented search. Existing experiments are never overwritten."""

import argparse
from pathlib import Path
from src.appliances.experiment import run_experiment
from src.appliances.config import RELEASE, MODEL_DIR

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=RELEASE)
    p.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    a = p.parse_args()
    run_experiment(a.output, a.model_dir, a.device)
