"""Select by shared validation MAE; copy, never retrain or register."""

import json
import math
import shutil

from ml.data import ROOT, sha256, write_json
from ml.train import code_identity, local_tracking


def select():
    code_identity()
    client = local_tracking()
    records = []
    for name in ("baseline", "improved"):
        record = json.loads((ROOT / f"runtime/latest-{name}.json").read_text(encoding="utf-8"))
        run_id = record["run_id"]
        folder = ROOT / "runtime/runs" / run_id
        if record != json.loads((folder / "record.json").read_text(encoding="utf-8")):
            raise ValueError("latest record differs from this run's immutable local record")
        if record["code_root"] != str(ROOT) or record["config"]["name"] != name:
            raise ValueError("record must belong to the current checkout")
        if record["tracking_uri"] != "sqlite:///" + (ROOT / "runtime/mlflow.db").as_posix():
            raise ValueError("record is not from this local tracking database")
        run = client.get_run(run_id)
        if run.info.status != "FINISHED" or run.data.params["code_sha"] != record["code_sha"]:
            raise ValueError("MLflow run must be finished with its recorded training code identity")
        mae = record["metrics"]["validation_mae"]
        if (
            not math.isfinite(mae)
            or mae < 0
            or run.data.metrics["validation_mae"] != mae
            or run.data.params["dataset_dvc_hash"] != record["dataset_dvc_hash"]
        ):
            raise ValueError("validation metric/data identity differs from actual MLflow run")
        model_path = folder / "model.pkl"
        if record["model_path"] != str(model_path):
            raise ValueError("model path must belong to this runtime run")
        if not model_path.exists():
            downloaded = client.download_artifacts(run_id, "model.pkl", str(folder))
            model_path = type(model_path)(downloaded)
        if sha256(model_path) != record["model_sha256"]:
            raise ValueError("run model SHA256 mismatch")
        records.append(record)
    if records[0]["validation_sha256"] != records[1]["validation_sha256"]:
        raise ValueError("Candidates must use identical validation bytes")
    winner = min(records, key=lambda record: record["metrics"]["validation_mae"])
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    shutil.copyfile(winner["model_path"], models / "model.pkl")
    manifest = {
        key: winner[key]
        for key in (
            "model_sha256",
            "features",
            "sklearn_version",
            "run_id",
            "code_sha",
            "dataset_version",
            "dataset_dvc_hash",
        )
    }
    manifest.update(purpose="selected", validation_mae=winner["metrics"]["validation_mae"])
    write_json(models / "manifest.json", manifest)
    lines = [
        "# Bike Sharing Experiment Report",
        "",
        "Metrics are read from actual local MLflow runs. Selection compares MAE on the same 2012Q1 validation set (rides/hour).",
        "",
        "|Configuration|run_id|Dataset|train MAE|train RMSE|train R2|validation MAE|validation RMSE|validation R2|",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        m = record["metrics"]
        lines.append(
            f"|{record['config']['name']}|{record['run_id']}|{record['dataset_version']}|{m['train_mae']:.6f}|{m['train_rmse']:.6f}|{m['train_r2']:.6f}|{m['validation_mae']:.6f}|{m['validation_rmse']:.6f}|{m['validation_r2']:.6f}|"
        )
    for record in records:
        lines.extend(
            [
                "",
                f"Parameters `{json.dumps(record['config'], ensure_ascii=False, sort_keys=True)}`",
                f"Code `{record['code_sha']}`; DVC `{record['dataset_dvc_hash']}`.",
                "",
            ]
        )
    lines.extend(
        [
            f"Selected **{winner['config']['name']}**, run_id `{winner['run_id']}`, validation MAE **{manifest['validation_mae']:.6f} rides/hour**. This run has the lower validation MAE; an exact tie selects baseline.",
            "",
            "Limitations: v1 uses 2011H1; v2 uses the full year 2011. Data size and parameters both change, so the difference cannot be attributed to one factor. Validation is used for selection, not as an independent test set. The remaining 2012 data is not evaluated; no claim is made about future generalization. Missing hours are not filled with zeros. cnt measures rentals per hour, not available inventory.",
            "",
            "Only the already-trained model is copied. Selection does not retrain, register a model, or start a service.",
        ]
    )
    (ROOT / "ml/README_experiments.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    registry = ROOT / "ml/registry"
    registry.mkdir(exist_ok=True)
    (registry / "current_model.md").write_text(
        "# Current Exported Model (Local Manifest, Not a Model Registry)\n\n"
        + "\n".join(f"- {key}: `{value}`" for key, value in manifest.items())
        + "\n\nvalidation_mae unit: rides/hour. The model is a byte-identical copy of the original run artifact. Load only trusted local model files.\n",
        encoding="utf-8",
        newline="\n",
    )
    if sha256(models / "model.pkl") != manifest["model_sha256"]:
        raise ValueError("export copy hash mismatch")
    return manifest


def main():
    print(json.dumps(select(), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
