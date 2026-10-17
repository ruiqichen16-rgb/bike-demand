"""PR builds use an explicitly non-production model, without cloud secrets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor

from app.features import COLUMNS


def make_fixture(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(np.zeros((4, len(COLUMNS))), columns=COLUMNS)
    for col in COLUMNS[:8]:
        frame[col] = frame[col].astype("int64")
    model = RandomForestRegressor(n_estimators=2, random_state=42, n_jobs=1)
    model.fit(frame, [0, 1, 2, 3])
    path = output / "model.pkl"
    joblib.dump(model, path)
    manifest = {
        "purpose": "ci-test",
        "features": COLUMNS,
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sklearn_version": sklearn.__version__,
        "run_id": "synthetic-fixture",
        "code_sha": "not-a-training-run",
        "dataset_version": "synthetic",
        "dataset_dvc_hash": "not-dvc-data",
        "validation_mae": 0.0,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/model"))
    make_fixture(parser.parse_args().output)
