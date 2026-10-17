"""One strict, shared feature transformation for training and inference."""

import math
import re
from datetime import date

import pandas as pd

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
INPUTS = {"date", "season", "hr", "holiday", "weathersit", "temp", "hum", "windspeed"}


def make_frame(payload: dict) -> pd.DataFrame:
    if not isinstance(payload, dict) or set(payload) != INPUTS:
        raise ValueError(
            "Provide exactly the eight required input fields; extra fields are not allowed"
        )
    day = payload["date"]
    if not isinstance(day, str) or not re.fullmatch(r"201[12]-\d{2}-\d{2}", day):
        raise ValueError("date must use YYYY-MM-DD in 2011 or 2012")
    try:
        day = date.fromisoformat(day)
    except ValueError:
        raise ValueError("date is invalid") from None
    for key, low, high in [
        ("season", 1, 4),
        ("hr", 0, 23),
        ("holiday", 0, 1),
        ("weathersit", 1, 4),
    ]:
        value = payload[key]
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{key} must be an integer from {low} to {high}")
    for key in ["temp", "hum", "windspeed"]:
        value = payload[key]
        if type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number from 0 to 1")
    weekday = (day.weekday() + 1) % 7
    row = {
        **payload,
        "yr": day.year - 2011,
        "mnth": day.month,
        "weekday": weekday,
        "workingday": int(weekday not in (0, 6) and payload["holiday"] == 0),
    }
    return pd.DataFrame([row], columns=COLUMNS).astype(
        {name: "int64" if i < 8 else "float64" for i, name in enumerate(COLUMNS)}
    )
