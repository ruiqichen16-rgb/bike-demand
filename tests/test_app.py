"""Contract tests use only local, real RandomForest artifacts."""

import importlib

import pytest

COLUMNS = [
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
SAMPLE = {
    "date": "2012-01-02",
    "season": 1,
    "hr": 8,
    "holiday": 0,
    "weathersit": 1,
    "temp": 0.3,
    "hum": 0.5,
    "windspeed": 0.1,
}


def features():
    try:
        return importlib.import_module("app.features")
    except ModuleNotFoundError:
        pytest.fail("Missing make_frame implementation")


def test_features_column_order_types_and_calendar():
    module = features()
    frame = module.make_frame(SAMPLE)
    assert module.COLUMNS == COLUMNS
    assert frame.columns.tolist() == COLUMNS
    assert frame.dtypes.astype(str).tolist() == ["int64"] * 8 + ["float64"] * 3
    assert frame.iloc[0].tolist() == [1, 1, 1, 8, 0, 1, 1, 1, 0.3, 0.5, 0.1]


@pytest.mark.parametrize(
    "date,holiday,expected",
    [
        ("2011-01-01", 0, [0, 1, 6, 0]),
        ("2012-02-29", 0, [1, 2, 3, 1]),
        ("2012-12-31", 1, [1, 12, 1, 0]),
        ("2011-01-02", 0, [0, 1, 0, 0]),
    ],
)
def test_features_calendar_boundaries(date, holiday, expected):
    frame = features().make_frame({**SAMPLE, "date": date, "holiday": holiday})
    assert frame[["yr", "mnth", "weekday", "workingday"]].iloc[0].tolist() == expected


@pytest.mark.parametrize(
    "changes",
    [
        {"date": "2010-12-31"},
        {"date": "2013-01-01"},
        {"date": "2011-02-29"},
        {"date": "2012-1-02"},
        {"date": 20120102},
        {"season": 0},
        {"season": 5},
        {"season": True},
        {"season": 1.0},
        {"hr": -1},
        {"hr": 24},
        {"hr": "8"},
        {"holiday": 2},
        {"holiday": False},
        {"weathersit": 0},
        {"weathersit": 5},
        {"temp": -0.01},
        {"temp": 1.01},
        {"hum": float("nan")},
        {"windspeed": float("inf")},
        {"temp": True},
        {"hum": "0.5"},
        {"unexpected": 1},
    ],
)
def test_features_reject_invalid_values(changes):
    with pytest.raises(ValueError):
        features().make_frame({**SAMPLE, **changes})


@pytest.mark.parametrize("field", list(SAMPLE))
def test_features_require_all_fields(field):
    payload = SAMPLE.copy()
    payload.pop(field)
    with pytest.raises(ValueError):
        features().make_frame(payload)


@pytest.mark.parametrize("value", [0, 1, 0.5])
def test_features_accept_normalized_number_boundaries(value):
    frame = features().make_frame({**SAMPLE, "temp": value, "hum": value, "windspeed": value})
    assert frame[["temp", "hum", "windspeed"]].iloc[0].tolist() == [value] * 3


def application():
    try:
        return importlib.import_module("app.main")
    except ModuleNotFoundError:
        pytest.fail("Missing FastAPI application")


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    import hashlib
    import json

    import joblib
    import pandas as pd
    import sklearn
    from sklearn.ensemble import RandomForestRegressor

    directory = tmp_path_factory.mktemp("app-model")
    rows = [features().make_frame({**SAMPLE, "hr": i, "temp": i / 23}) for i in range(24)]
    model = RandomForestRegressor(n_estimators=4, random_state=42, n_jobs=1).fit(
        pd.concat(rows), [i * 10 for i in range(24)]
    )
    joblib.dump(model, directory / "model.pkl")
    manifest = {
        "purpose": "selected",
        "model_sha256": hashlib.sha256((directory / "model.pkl").read_bytes()).hexdigest(),
        "features": COLUMNS,
        "sklearn_version": sklearn.__version__,
        "run_id": "local-real-test",
        "code_sha": "test-only",
        "dataset_version": "synthetic-test",
        "dataset_dvc_hash": "test-only",
        "validation_mae": 1.0,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8", newline="\n")
    return directory


@pytest.fixture
def client(artifact, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("APP_MESSAGE", "Local test")
    return TestClient(application().create_app(artifact))


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"hr": 0, "temp": 0, "hum": 0, "windspeed": 0},
        {
            "date": "2012-12-31",
            "hr": 23,
            "season": 4,
            "weathersit": 4,
            "temp": 1,
            "hum": 1,
            "windspeed": 1,
        },
        {"date": "2012-02-29", "holiday": 1},
    ],
)
def test_predict_real_model(client, changes):
    import math

    response = client.post("/predict", json={**SAMPLE, **changes})
    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"prediction", "unit", "model_sha256"}
    assert (
        type(data["prediction"]) is float
        and math.isfinite(data["prediction"])
        and data["prediction"] >= 0
    )
    assert data["unit"] == "rides/hour" and len(data["model_sha256"]) == 64


@pytest.mark.parametrize(
    "changes",
    [
        {"hr": 24},
        {"season": True},
        {"date": "2011-02-29"},
        {"temp": "0.2"},
        {"hum": -1},
        {"extra": 1},
    ],
)
def test_predict_invalid(client, changes):
    assert client.post("/predict", json={**SAMPLE, **changes}).status_code == 422


@pytest.mark.parametrize("a_hr,b_hr", [(0, 23), (23, 0), (8, 8)])
def test_compare_real_predictions_and_direction(client, a_hr, b_hr):
    a, b = {**SAMPLE, "hr": a_hr}, {**SAMPLE, "hr": b_hr}
    response = client.post("/compare", json={"a": a, "b": b})
    assert response.status_code == 200
    data = response.json()
    assert data == {
        "a": client.post("/predict", json=a).json()["prediction"],
        "b": client.post("/predict", json=b).json()["prediction"],
        "difference": data["b"] - data["a"],
        "unit": "rides/hour",
    }
    if a_hr == b_hr:
        assert data["difference"] == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"a": SAMPLE},
        {"a": SAMPLE, "b": {**SAMPLE, "hr": -1}},
        {"a": None, "b": SAMPLE},
        {"a": SAMPLE, "b": SAMPLE, "extra": 0},
    ],
)
def test_compare_invalid(client, payload):
    assert client.post("/compare", json=payload).status_code == 422


@pytest.mark.parametrize(
    "predicted,actual", [(0, 0), (12.5, 10), (10, 12), (0, 1000000), (1000001, 0)]
)
def test_error_arithmetic(client, predicted, actual):
    response = client.post("/error", json={"predicted": predicted, "actual": actual})
    assert response.status_code == 200
    assert response.json() == {
        "residual": actual - predicted,
        "absolute_error": abs(actual - predicted),
        "unit": "rides/hour",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"predicted": -1, "actual": 0},
        {"predicted": True, "actual": 0},
        {"predicted": "1", "actual": 0},
        {"predicted": 1, "actual": True},
        {"predicted": 1, "actual": 1.0},
        {"predicted": 1, "actual": -1},
        {"predicted": 1, "actual": 1000001},
        {"predicted": 1},
        {"predicted": 1, "actual": 1, "extra": 1},
    ],
)
def test_error_invalid(client, payload):
    assert client.post("/error", json=payload).status_code == 422


@pytest.mark.parametrize(
    "route,body",
    [
        ("/predict", '{"temp":NaN}'),
        ("/compare", '{"a":Infinity,"b":{}}'),
        ("/error", '{"predicted":NaN,"actual":0}'),
        ("/error", '{"predicted":Infinity,"actual":0}'),
        ("/error", '{"predicted":1,"actual":NaN}'),
        ("/error", "NaN"),
        ("/predict", "not-json"),
    ],
)
def test_nonfinite_and_malformed_json_are_422(client, route, body):
    response = client.post(route, content=body, headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert "NaN" not in response.text and "Infinity" not in response.text


def test_health_and_english_single_page(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["purpose"] == "selected"
    assert response.json()["environment"] == "test"
    assert response.json()["message"] == "Local test"
    page = client.get("/")
    assert page.status_code == 200
    assert 'lang="en"' in page.text
    for text in [
        "Hourly demand prediction",
        "Scenario comparison",
        "Error calculation",
        "/predict",
        "/compare",
        "/error",
        "B-A",
        "not causal",
        "normalized",
        "actual",
    ]:
        assert text in page.text
    assert page.text.count("<button") == 3
    assert "<script src=" not in page.text and "localStorage" not in page.text
    for code in range(1, 5):
        assert f'<option value="{code}">{code}</option>' in page.text


@pytest.mark.parametrize(
    "mutation", ["missing", "hash", "columns", "version", "purpose", "ci-test", "broken-pickle"]
)
def test_model_gates_fail_closed(artifact, tmp_path, monkeypatch, mutation):
    import hashlib
    import json
    import shutil

    from fastapi.testclient import TestClient

    directory = tmp_path / "model"
    shutil.copytree(artifact, directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        (directory / "model.pkl").unlink()
    elif mutation == "hash":
        manifest["model_sha256"] = "0" * 64
    elif mutation == "columns":
        manifest["features"] = list(reversed(COLUMNS))
    elif mutation == "version":
        manifest["sklearn_version"] = "0.0.0"
    elif mutation == "purpose":
        manifest["purpose"] = "unknown"
    elif mutation == "ci-test":
        manifest["purpose"] = "ci-test"
    elif mutation == "broken-pickle":
        (directory / "model.pkl").write_bytes(b"not-a-pickle")
        manifest["model_sha256"] = hashlib.sha256(
            (directory / "model.pkl").read_bytes()
        ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8", newline="\n")
    monkeypatch.setenv("APP_ENV", "production")
    failed = TestClient(application().create_app(directory))
    assert failed.get("/").status_code == 200
    assert failed.get("/health").status_code == 503
    assert failed.post("/predict", json=SAMPLE).status_code == 503
    assert failed.post("/compare", json={"a": SAMPLE, "b": SAMPLE}).status_code == 503
    assert failed.post("/error", json={"predicted": 1, "actual": 0}).status_code == 200


def test_ci_fixture_only_permitted_in_ci(artifact, tmp_path, monkeypatch):
    import json
    import shutil

    from fastapi.testclient import TestClient

    directory = tmp_path / "ci"
    shutil.copytree(artifact, directory)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["purpose"] = "ci-test"
    path.write_text(json.dumps(manifest), encoding="utf-8", newline="\n")
    monkeypatch.setenv("APP_ENV", "ci")
    ready = TestClient(application().create_app(directory))
    assert ready.get("/health").json()["purpose"] == "ci-test"
    assert ready.post("/predict", json=SAMPLE).status_code == 200


def test_model_dir_environment_and_no_model_boot(tmp_path, artifact, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "absent"))
    missing = TestClient(application().create_app())
    assert missing.get("/health").status_code == 503 and missing.get("/").status_code == 200
    monkeypatch.setenv("MODEL_DIR", str(artifact))
    assert TestClient(application().create_app()).get("/health").status_code == 200


def test_user_facing_content_is_english(client, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    page = client.get("/").text
    assert 'lang="en"' in page
    assert "Predict hourly demand" in page
    assert "Compare A and B" in page
    assert "Calculate error" in page
    assert "novalidate" in page
    assert page.isascii()
    invalid = client.post("/predict", json={**SAMPLE, "hr": 24})
    assert invalid.status_code == 422
    assert invalid.json()["detail"] == "hr must be an integer from 0 to 23"
    malformed = client.post("/error", content="not-json")
    assert malformed.json()["detail"] == "Request body must be a valid JSON object"
    missing = TestClient(application().create_app(tmp_path / "absent"))
    assert missing.get("/health").json()["detail"] == (
        "Model unavailable; restore the model and its matching manifest"
    )
    monkeypatch.setenv("APP_MESSAGE", "non-ASCII: \u4e2d")
    with pytest.raises(ValueError, match="APP_MESSAGE must contain English ASCII text"):
        application().create_app(tmp_path / "absent")
