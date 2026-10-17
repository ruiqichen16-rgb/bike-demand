"""Download verified UCI data and create two chronological versions."""

import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
ZIP_SHA256 = "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401"
CSV_SHA256 = "e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def split_frame(frame, version):
    if version not in {"v1", "v2"}:
        raise ValueError("version must be v1 or v2")
    dates = pd.to_datetime(frame["dteday"], format="%Y-%m-%d", errors="raise")
    stop = "2011-07-01" if version == "v1" else "2012-01-01"
    train = frame.loc[(dates >= "2011-01-01") & (dates < stop)].copy()
    validation = frame.loc[(dates >= "2012-01-01") & (dates < "2012-04-01")].copy()
    return train, validation


def download():
    folder = ROOT / "data/raw"
    folder.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=90) as response:
        raw = response.read()
    if hashlib.sha256(raw).hexdigest() != ZIP_SHA256:
        raise ValueError("official ZIP SHA256 mismatch")
    temporary = folder / "bike-sharing.zip"
    temporary.write_bytes(raw)
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            content = archive.read("hour.csv")
        if hashlib.sha256(content).hexdigest() != CSV_SHA256:
            raise ValueError("official hour.csv SHA256 mismatch")
        (folder / "hour.csv").write_bytes(content)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": "data/raw/hour.csv", "sha256": CSV_SHA256}


def prepare(version):
    from app.features import COLUMNS

    source = ROOT / "data/raw/hour.csv"
    if sha256(source) != CSV_SHA256:
        raise ValueError("source SHA256 mismatch; run download")
    frame = pd.read_csv(source)
    if frame.shape != (17379, 17) or frame.isna().any().any() or frame.duplicated().any():
        raise ValueError("unexpected UCI source dimensions, missing or duplicate rows")
    if (
        frame.duplicated(["dteday", "hr"]).any()
        or not (frame.cnt == frame.casual + frame.registered).all()
    ):
        raise ValueError("source time keys or count identity failed")
    train, validation = split_frame(frame, version)
    expected = 4250 if version == "v1" else 8645
    if len(train) != expected or len(validation) != 2176:
        raise ValueError("unexpected chronological split counts")
    folder = ROOT / "data/processed" / version
    folder.mkdir(parents=True, exist_ok=True)
    types = {key: "int64" if i < 8 else "float64" for i, key in enumerate(COLUMNS)}
    types["cnt"] = "int64"
    splits = {}
    for name, part in (("train", train), ("validation", validation)):
        output = folder / f"{name}.csv"
        part[list(COLUMNS) + ["cnt"]].astype(types).to_csv(output, index=False, lineterminator="\n")
        splits[name] = {
            "rows": len(part),
            "sha256": sha256(output),
            "start": str(part.dteday.min()),
            "end": str(part.dteday.max()),
        }
    metadata = {
        "source": URL,
        "source_sha256": CSV_SHA256,
        "source_rows": 17379,
        "source_columns": 17,
        "version": version,
        "features": list(COLUMNS),
        "dtypes": types,
        "splits": splits,
        "audit": "No imputation: 165 absent hours are not zero demand; cnt=casual+registered.",
    }
    write_json(folder / "metadata.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download")
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--version", choices=["v1", "v2"], required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            download() if args.command == "download" else prepare(args.version),
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
