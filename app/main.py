"""Minimal stateless bike-demand API; model files must be trusted locally."""

import hashlib
import io
import json
import math
import os
from pathlib import Path

import joblib
import sklearn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from sklearn.ensemble import RandomForestRegressor

from .features import COLUMNS, make_frame

UNIT = "rides/hour"


def create_app(model_dir: Path | None = None) -> FastAPI:
    api = FastAPI(title="Bike-Sharing Hourly Demand", docs_url=None, redoc_url=None)
    environment = os.getenv("APP_ENV", "local")
    message = os.getenv("APP_MESSAGE", "Local run")
    if not message.isascii():
        raise ValueError("APP_MESSAGE must contain English ASCII text")
    directory = (
        Path(model_dir)
        if model_dir is not None
        else Path(os.getenv("MODEL_DIR", str(Path(__file__).resolve().parents[1] / "models")))
    )
    model, manifest = None, {}
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        content = (directory / "model.pkl").read_bytes()
        allowed = manifest["purpose"] == "selected" or (
            manifest["purpose"] == "ci-test" and environment == "ci"
        )
        if (
            not allowed
            or manifest["features"] != COLUMNS
            or manifest["sklearn_version"] != sklearn.__version__
        ):
            raise ValueError("Model identity does not match")
        if hashlib.sha256(content).hexdigest() != manifest["model_sha256"]:
            raise ValueError("Model hash verification failed")
        loaded = joblib.load(io.BytesIO(content))
        if (
            not isinstance(loaded, RandomForestRegressor)
            or list(loaded.feature_names_in_) != COLUMNS
        ):
            raise ValueError("Model type or feature columns do not match")
        model = loaded
    except Exception:
        # Never substitute a synthetic CI artifact or leak local paths on failure.
        model, manifest = None, {}

    def require_model():
        if model is None:
            raise HTTPException(
                503, "Model unavailable; restore the model and its matching manifest"
            )
        return model

    async def payload(request: Request) -> dict:
        try:
            value = await request.json()
        except (ValueError, UnicodeError):
            raise HTTPException(422, "Request body must be a valid JSON object") from None
        if not isinstance(value, dict):
            raise HTTPException(422, "Request body must be a JSON object")
        return value

    def frame(value):
        try:
            return make_frame(value)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    def predict_frame(data):
        try:
            prediction = float(require_model().predict(data)[0])
            if not math.isfinite(prediction) or prediction < 0:
                raise ValueError("Prediction must be finite and nonnegative")
            return prediction
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(503, "Model could not produce a valid prediction") from None

    @api.get("/", response_class=HTMLResponse)
    def index():
        return Path(__file__).with_name("index.html").read_text(encoding="utf-8")

    @api.get("/health")
    def health():
        require_model()
        return {
            "status": "ok",
            "purpose": manifest["purpose"],
            "model_sha256": manifest["model_sha256"],
            "environment": environment,
            "message": message,
        }

    @api.post("/predict")
    async def predict(request: Request):
        data = frame(await payload(request))
        return {
            "prediction": predict_frame(data),
            "unit": UNIT,
            "model_sha256": manifest["model_sha256"],
        }

    @api.post("/compare")
    async def compare(request: Request):
        value = await payload(request)
        if set(value) != {"a", "b"}:
            raise HTTPException(422, "Provide exactly two complete scenarios named a and b")
        a_frame, b_frame = frame(value["a"]), frame(value["b"])
        a, b = predict_frame(a_frame), predict_frame(b_frame)
        return {"a": a, "b": b, "difference": b - a, "unit": UNIT}

    @api.post("/error")
    async def error(request: Request):
        value = await payload(request)
        if set(value) != {"predicted", "actual"}:
            raise HTTPException(422, "Provide exactly predicted and actual")
        predicted, actual = value["predicted"], value["actual"]
        try:
            valid = type(predicted) in (int, float) and predicted >= 0 and math.isfinite(predicted)
        except OverflowError:
            valid = False
        if not valid or type(actual) is not int or not 0 <= actual <= 1000000:
            raise HTTPException(
                422,
                "predicted must be finite and nonnegative; actual must be an integer from 0 to 1000000",
            )
        residual = actual - float(predicted)
        return {"residual": residual, "absolute_error": abs(residual), "unit": UNIT}

    return api


app = create_app()
