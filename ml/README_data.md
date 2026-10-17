# Data source and dataset versions

Source: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
Bike Sharing data by Fanaee-T and Gama. Check and cite the current license on the
official page. The downloader verifies fixed ZIP and hour.csv hashes; use the
actual fixed file rather than a possibly different API summary count.

v1 uses January-June 2011 for training. v2 expands training to all of 2011.
Both use the identical January-March 2012 validation split. Training and
validation dates do not overlap. The target cnt is rentals per hour. Do not use
casual or registered as input features because they are components of the target.

| Version | Actual Git commit (read from its tag) | Actual DVC hash | Change |
|---|---|---|---|
| v1 | To be filled after execution | To be filled after execution | First usable training data |
| v2 | To be filled after execution | To be filled after execution | Add July-December 2011; validation unchanged |

DVC stores data in authorized MS Azure storage; Git stores only small pointers.
Ignore data/raw and do not upload private data or credentials. Do not commit
.dvc/config.local. Missing hours are not filled with zero demand. Preserve the
original season codes and normalized weather values. Training does not observe
yr=1; historical validation is not a guarantee of future operational performance.
