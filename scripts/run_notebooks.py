"""Execute and retain notebook outputs with the current Python environment."""

import json
import subprocess
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from src.appliances.config import ROOT
from src.appliances.data import write_json


def run_notebooks():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            "appliances-project",
            "--display-name",
            "Appliances project",
        ],
        check=True,
        capture_output=True,
    )
    records = []
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        NotebookClient(
            notebook,
            timeout=900,
            kernel_name="appliances-project",
            resources={"metadata": {"path": str(ROOT)}},
        ).execute()
        nbformat.write(notebook, path)
        counts = [c.execution_count for c in notebook.cells if c.cell_type == "code"]
        records.append(
            {
                "notebook": path.name,
                "code_cells": len(counts),
                "executed": all(n is not None for n in counts),
                "errors": sum(
                    o.output_type == "error"
                    for c in notebook.cells
                    if c.cell_type == "code"
                    for o in c.get("outputs", [])
                ),
            }
        )
        print(json.dumps(records[-1]), flush=True)
    write_json(ROOT / "outputs/appliances_20261102/notebook_verification.json", records)


if __name__ == "__main__":
    run_notebooks()
