
import argparse
import json
import math
import time
import urllib.error
import urllib.request

SAMPLE = {
    "date": "2012-02-29",
    "season": 1,
    "hr": 8,
    "holiday": 0,
    "weathersit": 2,
    "temp": 0.3,
    "hum": 0.5,
    "windspeed": 0.2,
}


def smoke(base: str) -> dict:
    def request(path, body=None):
        data = None if body is None else json.dumps(body, allow_nan=False).encode()
        req = urllib.request.Request(
            base.rstrip("/") + path,
            data=data,
            headers={"Content-Type": "application/json"} if data else {},
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            return json.load(response)

    deadline = time.monotonic() + 60
    while True:
        try:
            health = request("/health")
            if health["status"] == "ok":
                break
        except (OSError, KeyError):
            pass
        if time.monotonic() > deadline:
            raise RuntimeError("Application did not become healthy within 60 seconds")
        time.sleep(0.25)
    predicted = request("/predict", SAMPLE)
    assert math.isfinite(predicted["prediction"]) and predicted["prediction"] >= 0
    assert predicted["model_sha256"] == health["model_sha256"]
    compared = request("/compare", {"a": SAMPLE, "b": SAMPLE})
    assert compared["difference"] == 0 and compared["a"] == compared["b"]
    error = request("/error", {"predicted": 10.5, "actual": 0})
    assert error["residual"] == -10.5 and error["absolute_error"] == 10.5
    assert error["unit"] == "rides/hour"
    try:
        request("/predict", {**SAMPLE, "hr": 24})
    except urllib.error.HTTPError as exc:
        assert exc.code == 422
    else:
        raise AssertionError("Invalid hour was accepted")
    return {
        "status": "passed",
        "purpose": health["purpose"],
        "model_sha256": health["model_sha256"],
        "prediction": predicted["prediction"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8200")
    print(json.dumps(smoke(parser.parse_args().url), ensure_ascii=False))
