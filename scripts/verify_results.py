import argparse, json
from pathlib import Path
from src.appliances.config import RELEASE, MODEL_DIR
from src.appliances.verification import verify
from src.appliances.data import write_json

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=RELEASE)
    p.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    p.add_argument("--save", type=Path)
    a = p.parse_args()
    result = verify(a.output, a.model_dir)
    if a.save:
        write_json(a.save, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
