"""Keep the minimal CI and release contract readable and executable."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.ensemble import RandomForestRegressor

from scripts.prepare_release import prepare

ROOT = Path(__file__).resolve().parents[1]


def test_ci_has_three_secret_free_required_jobs_and_separate_environments():
    flow = yaml.load((ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
    assert "pull_request" in flow["on"]
    assert flow["on"]["push"]["branches"] == ["dev", "staging", "main"]
    for name in ["tests", "quality", "image-build"]:
        assert "environment" not in flow["jobs"][name]
        assert "secrets." not in str(flow["jobs"][name])
    assert any("python -m pytest -q" in s.get("run", "") for s in flow["jobs"]["tests"]["steps"])
    for name, branch in [("staging", "staging"), ("production", "main")]:
        job = flow["jobs"][name]
        assert set(job["needs"]) == {"tests", "quality", "image-build"}
        assert job["environment"] == name
        assert f"refs/heads/{branch}" in job["if"] and "ENABLE_CD" in job["if"]
        assert any("prepare_release" in step.get("run", "") for step in job["steps"])
        assert any(step.get("if") == "always()" for step in job["steps"])
    docker = (ROOT / "Dockerfile").read_text()
    assert "python:3.12-slim" in docker and "USER 10001" in docker
    assert "secrets" not in docker.lower() and ".env" not in docker


def test_release_prepares_only_reviewed_model_and_runtime_env(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    columns = [
        "season",
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "temp",
        "hum",
        "windspeed",
    ]
    frame = pd.DataFrame(np.zeros((2, 11)), columns=columns)
    model = RandomForestRegressor(n_estimators=1, random_state=42).fit(frame, [0, 1])
    joblib.dump(model, model_dir / "model.pkl")
    manifest = {
        "purpose": "selected",
        "model_sha256": hashlib.sha256((model_dir / "model.pkl").read_bytes()).hexdigest(),
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest))
    for k, v in {
        "APP_ENV": "staging",
        "APP_MESSAGE": "English $literal",
        "AZURE_STORAGE_ACCOUNT": "syntheticaccount",
        "AZURE_STORAGE_SAS_TOKEN": "synthetic-token",
    }.items():
        monkeypatch.setenv(k, v)
    calls = []

    class Edit:
        def __enter__(self):
            self.data = {}
            return self.data

        def __exit__(self, *args):
            pass

    class Config:
        def edit(self, **kwargs):
            calls.append(("local_config", kwargs))
            return Edit()

    class FakeRepo:
        def __init__(self, root):
            self.config = Config()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("scripts.prepare_release.Repo", FakeRepo)
    monkeypatch.setattr(
        "scripts.prepare_release.subprocess.run", lambda *a, **k: calls.append((a, k))
    )
    prepare(tmp_path)
    assert (tmp_path / "build/model/model.pkl").read_bytes() == (
        model_dir / "model.pkl"
    ).read_bytes()
    assert (
        tmp_path / "runtime/deploy.env"
    ).read_text() == "APP_ENV=staging\nAPP_MESSAGE=English $literal\n"
    assert b"\r\n" not in (tmp_path / "runtime/deploy.env").read_bytes()
    assert calls[1][0][0][-3:] == ["models/model.pkl.dvc", "-r", "myazure"]


@pytest.mark.parametrize("purpose,wrong_hash", [("ci-test", False), ("selected", True)])
def test_release_refuses_ci_fixture_or_wrong_hash(tmp_path, monkeypatch, purpose, wrong_hash):
    (tmp_path / "models").mkdir()
    payload = b"not-loaded-in-this-negative-test"
    (tmp_path / "models/model.pkl").write_bytes(payload)
    manifest = {
        "purpose": purpose,
        "model_sha256": "0" * 64 if wrong_hash else hashlib.sha256(payload).hexdigest(),
    }
    (tmp_path / "models/manifest.json").write_text(json.dumps(manifest))
    for k, v in {
        "APP_ENV": "production",
        "APP_MESSAGE": "example",
        "AZURE_STORAGE_ACCOUNT": "syntheticaccount",
        "AZURE_STORAGE_SAS_TOKEN": "synthetic-token",
    }.items():
        monkeypatch.setenv(k, v)

    class RepoDouble:
        def __init__(self, root):
            self.config = self

        def __enter__(self):
            return {}

        def __exit__(self, *args):
            pass

        def edit(self, **kwargs):
            return self

    # Retain real function structure without any Azure write in this unit test.
    from contextlib import nullcontext

    double = type("D", (), {"config": type("C", (), {"edit": lambda *a, **k: nullcontext({})})()})()
    monkeypatch.setattr("scripts.prepare_release.Repo", lambda root: nullcontext(double))
    monkeypatch.setattr("scripts.prepare_release.subprocess.run", lambda *a, **k: None)
    with pytest.raises(ValueError):
        prepare(tmp_path)
    assert not (tmp_path / "build/model").exists()


def test_all_project_text_is_english_ascii():
    candidates = [
        ROOT / "Dockerfile",
        ROOT / "README.md",
        ROOT / "DEPLOYMENT.md",
        ROOT / "requirements.txt",
        ROOT / "requirements-dev.txt",
        ROOT / "pyproject.toml",
        ROOT / ".gitignore",
        ROOT / ".gitattributes",
        ROOT / ".dockerignore",
        ROOT / ".github/workflows/ci.yml",
    ]
    for folder in ("app", "ml", "scripts", "tests"):
        candidates.extend(
            p
            for p in (ROOT / folder).rglob("*")
            if p.is_file()
            and p.suffix in (".py", ".html", ".json", ".md")
            and "__pycache__" not in p.parts
        )
    assert candidates
    failures = [
        str(p.relative_to(ROOT)) for p in candidates if not p.read_text(encoding="utf-8").isascii()
    ]
    assert not failures, f"Project text must be English ASCII: {failures}"
