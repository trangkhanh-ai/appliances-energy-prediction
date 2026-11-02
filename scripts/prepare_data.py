"""Download and validate the exact public dataset."""

import json
from src.appliances.data import download_data, load_data
from src.appliances.features import supervised, split_rows

if __name__ == "__main__":
    download_data()
    raw, audit = load_data()
    _, counts = split_rows(supervised(raw)[0])
    print(json.dumps({"audit": audit, "split_rows": counts}, ensure_ascii=False, indent=2))
