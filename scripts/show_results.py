"""Read locked measurements without training a model."""

import json
from src.appliances.config import ROOT, RELEASE
from src.appliances.data import sha256

if __name__ == "__main__":
    lock = json.loads((RELEASE / "results_lock.json").read_text(encoding="utf-8"))
    for path, digest in lock.items():
        if sha256(ROOT / path) != digest:
            raise ValueError(f"Changed result {path}")
    print((RELEASE / "summary.json").read_text(encoding="utf-8"))
