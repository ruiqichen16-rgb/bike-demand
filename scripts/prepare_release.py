"""Restore the reviewed DVC model for an environment build, without retraining."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dvc.repo import Repo


def prepare(root: Path) -> None:
    environment = os.environ.get("APP_ENV", "")
    message = os.environ.get("APP_MESSAGE", "")
    account = os.environ.get("AZURE_STORAGE_ACCOUNT", "")
    token = os.environ.get("AZURE_STORAGE_SAS_TOKEN", "")
    if (
        environment not in ("staging", "production")
        or not message
        or not message.isascii()
        or "\n" in message
        or "\r" in message
    ):
        raise ValueError("Environment and one-line APP_MESSAGE are required")
    if not account or not token:
        raise ValueError("Azure credentials are required")
    # Do not pass the SAS in process arguments, stdout, YAML or the Git config.
    with Repo(root) as repo:
        with repo.config.edit(level="local") as config:
            config.setdefault("remote", {}).setdefault("myazure", {}).update(
                account_name=account, sas_token=token
            )
    subprocess.run(
        [sys.executable, "-m", "dvc", "pull", "models/model.pkl.dvc", "-r", "myazure"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    source = root / "models"
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest["purpose"] != "selected":
        raise ValueError("Only the selected real model may enter a release build")
    if hashlib.sha256((source / "model.pkl").read_bytes()).hexdigest() != manifest["model_sha256"]:
        raise ValueError("Restored model does not match the committed manifest")
    target = root / "build/model"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("model.pkl", "manifest.json"):
        shutil.copyfile(source / name, target / name)
    output = root / "runtime/deploy.env"
    output.parent.mkdir(exist_ok=True)
    output.write_text(
        f"APP_ENV={environment}\nAPP_MESSAGE={message}\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    try:
        prepare(Path.cwd())
    except Exception:
        # DVC errors can include signed locations; public job log stays redacted.
        raise SystemExit(
            "Release preparation failed. Check Azure access, model pointer and manifest."
        ) from None
    print("Selected DVC model restored; environment file created. No credentials printed.")
