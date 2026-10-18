"""Train one configured random forest and log to explicit local MLflow."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
import yaml
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from ml.data import ROOT, sha256, write_json


def code_identity():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], stderr=subprocess.PIPE)

    top = Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve()
    if top != ROOT:
        raise ValueError("run from the actual checkout root; borrowed Git identity rejected")
    if Path.cwd().resolve() != ROOT:
        raise ValueError("run the CLI from its checkout root")
    head = git("rev-parse", "HEAD").decode().strip()
    files = sorted(
        [
            *ROOT.joinpath("ml").glob("*.py"),
            *ROOT.joinpath("ml/configs").glob("*.json"),
            ROOT / "app/features.py",
        ]
    )
    if not files or not (ROOT / "app/features.py").is_file():
        raise ValueError("missing source files")
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        try:
            committed = git("show", f"{head}:{relative}")
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"source must be tracked in HEAD: {relative}") from exc
        if (
            path.read_bytes() != committed
            or git("diff", "--cached", "--name-only", "HEAD", "--", relative).strip()
        ):
            raise ValueError(f"source/config must be clean and byte-identical to HEAD: {relative}")
    return head


def load_dataset(version):
    from app.features import COLUMNS

    folder = ROOT / "data/processed" / version
    pointer = yaml.safe_load((folder.parent / f"{version}.dvc").read_text(encoding="utf-8"))
    outputs = pointer.get("outs", [])
    if len(outputs) != 1 or outputs[0].get("path") != version:
        raise ValueError("unexpected DVC output path")
    dvc_hash = outputs[0].get("md5", "")
    entries = [
        {
            "md5": hashlib.md5(path.read_bytes()).hexdigest(),
            "relpath": path.relative_to(folder).as_posix(),
        }
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    ]
    actual = hashlib.md5(json.dumps(entries, sort_keys=True).encode("utf-8")).hexdigest() + ".dir"
    if dvc_hash != actual:
        raise ValueError("DVC directory content hash mismatch")
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if metadata["version"] != version or metadata["features"] != list(COLUMNS):
        raise ValueError("metadata version/features mismatch")
    dtypes = {column: "int64" if index < 8 else "float64" for index, column in enumerate(COLUMNS)}
    dtypes["cnt"] = "int64"
    frames = []
    for name in ("train", "validation"):
        path = folder / f"{name}.csv"
        frame = pd.read_csv(path, dtype=dtypes)
        info = metadata["splits"][name]
        if (
            list(frame.columns) != list(COLUMNS) + ["cnt"]
            or len(frame) != info["rows"]
            or sha256(path) != info["sha256"]
        ):
            raise ValueError("dataset schema/rows/content mismatch")
        if frame.empty or not np.isfinite(frame.to_numpy()).all() or (frame["cnt"] < 0).any():
            raise ValueError("dataset must have finite features and nonnegative counts")
        frames.append(frame)
    return frames, dvc_hash, sha256(folder / "metadata.json")


def local_tracking():
    runtime = ROOT / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    uri = "sqlite:///" + (runtime / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)


def train(config_path):
    from app.features import COLUMNS

    code_sha = code_identity()
    config_path = Path(config_path).resolve()
    if config_path.parent != ROOT / "ml/configs":
        raise ValueError("config must be a tracked ml/configs JSON")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    name, version = config["name"], config["version"]
    if (name, version) not in {("baseline", "v1"), ("improved", "v2")}:
        raise ValueError("expected baseline/v1 or improved/v2")
    (training, validation), dvc_hash, metadata_hash = load_dataset(version)
    params = {key: config[key] for key in ("n_estimators", "max_depth", "random_state", "n_jobs")}
    model = RandomForestRegressor(**params)
    model.fit(training[list(COLUMNS)], training["cnt"])
    metrics = {}
    for label, frame in (("train", training), ("validation", validation)):
        predicted = model.predict(frame[list(COLUMNS)])
        metrics.update(
            {
                f"{label}_mae": float(mean_absolute_error(frame["cnt"], predicted)),
                f"{label}_rmse": float(root_mean_squared_error(frame["cnt"], predicted)),
                f"{label}_r2": float(r2_score(frame["cnt"], predicted)),
            }
        )
    client = local_tracking()
    experiment = client.get_experiment_by_name(config["experiment"])
    experiment_id = (
        experiment.experiment_id
        if experiment
        else client.create_experiment(
            config["experiment"],
            artifact_location=(ROOT / "runtime/mlartifacts" / config["experiment"]).as_uri(),
        )
    )
    with mlflow.start_run(experiment_id=experiment_id, run_name=name) as active:
        run_id = active.info.run_id
        folder = ROOT / "runtime/runs" / run_id
        folder.mkdir(parents=True, exist_ok=True)
        model_path = folder / "model.pkl"
        joblib.dump(model, model_path)
        record = {
            "run_id": run_id,
            "code_sha": code_sha,
            "code_root": str(ROOT),
            "config": config,
            "dataset_version": version,
            "dataset_dvc_hash": dvc_hash,
            "metadata_sha256": metadata_hash,
            "validation_sha256": sha256(ROOT / "data/processed" / version / "validation.csv"),
            "features": list(COLUMNS),
            "sklearn_version": sklearn.__version__,
            "metrics": metrics,
            "model_sha256": sha256(model_path),
            "model_path": str(model_path),
            "tracking_uri": mlflow.get_tracking_uri(),
        }
        write_json(folder / "record.json", record)
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        figure, axes = plt.subplots(figsize=(5, 4))
        axes.scatter(validation["cnt"], model.predict(validation[list(COLUMNS)]), s=7, alpha=0.4)
        axes.set(
            xlabel="Actual demand (rides/hour)",
            ylabel="Predicted demand (rides/hour)",
            title=f"{name}: validation",
        )
        figure.tight_layout()
        figure.savefig(folder / "validation_scatter.png", dpi=100)
        plt.close(figure)
        for key, value in {
            **config,
            "code_sha": code_sha,
            "dataset_dvc_hash": dvc_hash,
            "metadata_sha256": metadata_hash,
        }.items():
            mlflow.log_param(key, value)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        for filename in ("model.pkl", "record.json", "validation_scatter.png"):
            mlflow.log_artifact(str(folder / filename))
    write_json(ROOT / f"runtime/latest-{name}.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(train(args.config), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
