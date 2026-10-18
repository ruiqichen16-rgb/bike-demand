# Bike Sharing Experiment Report

Metrics are read from actual local MLflow runs. Selection compares MAE on the same 2012Q1 validation set (rides/hour).

|Configuration|run_id|Dataset|train MAE|train RMSE|train R2|validation MAE|validation RMSE|validation R2|
|---|---|---|---:|---:|---:|---:|---:|---:|
|baseline|d8e3c155f3c249578eed75f80760c126|v1|21.951651|34.238514|0.923681|82.964802|126.271986|0.366093|
|improved|8b37de33e2ed4ec8a73eb2c74fd03084|v2|14.882165|24.800733|0.965638|82.173463|121.853101|0.409684|

Parameters `{"experiment": "BikeBaseline", "max_depth": 8, "n_estimators": 32, "n_jobs": 1, "name": "baseline", "random_state": 42, "version": "v1"}`
Code `96a169ec67cc921392ff1b02ea1d89a02095e3bc`; DVC `6dca5337b2a44cd400f411145c3c3118.dir`.


Parameters `{"experiment": "BikeImproved", "max_depth": 12, "n_estimators": 64, "n_jobs": 1, "name": "improved", "random_state": 42, "version": "v2"}`
Code `fc2c759f31d7ebd5c6fe0c9333050287549f8dc9`; DVC `531a810a27543ccf6e981ecefd3ccc3f.dir`.

Selected **improved**, run_id `8b37de33e2ed4ec8a73eb2c74fd03084`, validation MAE **82.173463 rides/hour**. This run has the lower validation MAE; an exact tie selects baseline.

Limitations: v1 uses 2011H1; v2 uses the full year 2011. Data size and parameters both change, so the difference cannot be attributed to one factor. Validation is used for selection, not as an independent test set. The remaining 2012 data is not evaluated; no claim is made about future generalization. Missing hours are not filled with zeros. cnt measures rentals per hour, not available inventory.

Only the already-trained model is copied. Selection does not retrain, register a model, or start a service.
