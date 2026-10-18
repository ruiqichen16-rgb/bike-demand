"""Small synthetic checks; official UCI execution is a separate integration run."""

import importlib
from pathlib import Path

import pandas as pd
import pytest


def test_time_splits_are_disjoint_and_share_validation():
    assert importlib.util.find_spec("ml.data") is not None, "data CLI is not implemented"
    from ml.data import split_frame

    frame = pd.DataFrame(
        {
            "dteday": [
                "2011-01-01",
                "2011-06-30",
                "2011-07-01",
                "2011-12-31",
                "2012-01-01",
                "2012-03-31",
                "2012-04-01",
            ],
            "cnt": range(7),
        }
    )
    train1, val1 = split_frame(frame, "v1")
    train2, val2 = split_frame(frame, "v2")
    assert train1["cnt"].tolist() == [0, 1]
    assert train2["cnt"].tolist() == [0, 1, 2, 3]
    assert val1["cnt"].tolist() == [4, 5]
    pd.testing.assert_frame_equal(val1, val2)
    assert not set(train2["cnt"]) & set(val2["cnt"])
    with pytest.raises(ValueError):
        split_frame(frame, "v3")


def test_real_local_training_selection_and_provenance(tmp_path):
    import hashlib
    import json
    import os
    import shutil
    import subprocess
    import sys

    assert importlib.util.find_spec("ml.train") is not None, "training CLI is not implemented"
    source = Path(__file__).resolve().parents[1]
    assert (source / "app/features.py").exists(), "APP seat must provide real features.py"
    checkout = tmp_path / "checkout"
    shutil.copytree(
        source / "ml",
        checkout / "ml",
        ignore=shutil.ignore_patterns("__pycache__", "*.md", "registry"),
    )
    (checkout / "app").mkdir()
    shutil.copy2(source / "app/features.py", checkout / "app/features.py")
    (checkout / ".gitignore").write_text(
        "data/\nruntime/\nmodels/\n__pycache__/\nml/*.md\nml/registry/\n", encoding="utf-8"
    )
    (checkout / ".gitattributes").write_text("* text eol=lf\n", encoding="utf-8")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")

    def command(*args, ok=True):
        result = subprocess.run(
            args, cwd=checkout, env=env, capture_output=True, text=True, encoding="utf-8"
        )
        if ok:
            assert result.returncode == 0, result.stdout + result.stderr
        return result

    command("git", "init", "-q")
    command("git", "config", "core.autocrlf", "false")
    command("git", "config", "user.name", "Synthetic test")
    command("git", "config", "user.email", "synthetic@example.invalid")
    (checkout / "NOTES.md").write_text("Initial documentation.\n", encoding="utf-8")
    command("git", "add", ".")
    command("git", "commit", "-qm", "isolated synthetic test sources")
    head = command("git", "rev-parse", "HEAD").stdout.strip()
    (checkout / "NOTES.md").write_text(
        "Generated documentation may change without changing training source identity.\n",
        encoding="utf-8",
    )
    from app.features import COLUMNS

    for version, count in (("v1", 20), ("v2", 32)):
        folder = checkout / "data/processed" / version
        folder.mkdir(parents=True)
        details = {}
        for name, rows in (("train", count), ("validation", 8)):
            frame = pd.DataFrame(
                {
                    column: [((i + j) % 3) if j < 8 else (i % 5) / 5 for i in range(rows)]
                    for j, column in enumerate(COLUMNS)
                }
            )
            frame["cnt"] = [i * 2 + 1 for i in range(rows)]
            frame.to_csv(folder / f"{name}.csv", index=False, lineterminator="\n")
            details[name] = {
                "rows": rows,
                "sha256": hashlib.sha256((folder / f"{name}.csv").read_bytes()).hexdigest(),
            }
        (folder / "metadata.json").write_text(
            json.dumps({"version": version, "features": COLUMNS, "splits": details}) + "\n",
            encoding="utf-8",
        )
        entries = [
            {"md5": hashlib.md5(path.read_bytes()).hexdigest(), "relpath": path.name}
            for path in sorted(folder.iterdir())
        ]
        dvc_hash = hashlib.md5(json.dumps(entries, sort_keys=True).encode()).hexdigest() + ".dir"
        (folder.parent / f"{version}.dvc").write_text(
            f"outs:\n- md5: {dvc_hash}\n  hash: md5\n  path: {version}\n", encoding="utf-8"
        )

    records = []
    for name in ("baseline", "improved"):
        if name == "improved":
            # Real manual creates a new data-version commit before its second run.
            (checkout / "NOTES.md").write_text("Second data-version commit.\n", encoding="utf-8")
            command("git", "add", "NOTES.md")
            command("git", "commit", "-qm", "record second version separately")
            head = command("git", "rev-parse", "HEAD").stdout.strip()
        result = command(sys.executable, "-m", "ml.train", "--config", f"ml/configs/{name}.json")
        record = json.loads(result.stdout.splitlines()[-1])
        assert record["code_sha"] == head
        assert record["dataset_dvc_hash"].endswith(".dir")
        assert record["metrics"]["validation_mae"] >= 0
        assert (checkout / "runtime/runs" / record["run_id"] / "record.json").exists()
        records.append(record)
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri="sqlite:///" + (checkout / "runtime/mlflow.db").as_posix())
    for record in records:
        logged = client.get_run(record["run_id"])
        assert logged.info.status == "FINISHED"
        assert logged.data.metrics["validation_mae"] == record["metrics"]["validation_mae"]
        assert logged.data.params["dataset_dvc_hash"] == record["dataset_dvc_hash"]
        assert {"model.pkl", "record.json", "validation_scatter.png"} <= {
            item.path for item in client.list_artifacts(record["run_id"])
        }
    selected = command(sys.executable, "-m", "ml.select")
    exported = json.loads(selected.stdout.splitlines()[-1])
    winner = min(records, key=lambda record: record["metrics"]["validation_mae"])
    assert exported["run_id"] == winner["run_id"]
    manifest = json.loads((checkout / "models/manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == winner["run_id"]
    assert manifest["purpose"] == "selected"
    assert manifest["validation_mae"] == winner["metrics"]["validation_mae"]
    assert (
        hashlib.sha256((checkout / "models/model.pkl").read_bytes()).hexdigest()
        == winner["model_sha256"]
    )
    assert (
        client.search_runs(
            [
                client.get_experiment_by_name("BikeBaseline").experiment_id,
                client.get_experiment_by_name("BikeImproved").experiment_id,
            ]
        ).__len__()
        == 2
    )
    for filename in ("ml/README_experiments.md", "ml/registry/current_model.md"):
        raw = (checkout / filename).read_bytes()
        assert b"\r\n" not in raw
        assert winner["run_id"].encode() in raw
    report = (checkout / "ml/README_experiments.md").read_text(encoding="utf-8")
    assert "train RMSE" in report and "train R2" in report
    table_rows = [line for line in report.splitlines() if line.startswith("|")]
    assert len(table_rows) == 4
    assert all(
        line.startswith("|")
        for line in report.splitlines()[
            report.splitlines().index(table_rows[0]) : report.splitlines().index(table_rows[0]) + 4
        ]
    )
    folder = checkout / "data/processed/v1"
    with (folder / "train.csv").open("a", encoding="utf-8") as stream:
        stream.write("corrupt\n")
    bad = command(
        sys.executable, "-m", "ml.train", "--config", "ml/configs/baseline.json", ok=False
    )
    assert bad.returncode != 0 and "DVC directory content hash mismatch" in bad.stderr
    with (checkout / "ml/data.py").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write("# dirty\n")
    dirty = command(
        sys.executable, "-m", "ml.train", "--config", "ml/configs/baseline.json", ok=False
    )
    assert dirty.returncode != 0 and "byte-identical" in dirty.stderr
