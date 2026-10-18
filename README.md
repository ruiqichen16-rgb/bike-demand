# Bike-Sharing Hourly Demand Estimator

Course project for DevOps and MLOps. The app estimates hourly bike rental demand
from historical date, hour, and weather inputs. It stores no business data.

## Functions

1.Hourly demand prediction: estimate rentals per hour for one scenario.

2.Scenario comparison: compare two scenarios that differ only in the hour.

3.Error evaluation: compute residual and absolute error for a prediction.

## Team

1.Chen ruiqi: application, HTTP interface, tests,CI/CD.

2.Yang qialun: data, training, CI/CD.



## Branches

feature/\* -> dev -> staging -> main

Direct pushes to dev, staging, and main are not allowed.



## CI/CD



1\. Pull requests and pushes to dev, staging, and main run the tests, Ruff

&#x20;  checks, and a Docker build with an HTTP smoke test.



2\. Merging requires a review approval and successful tests, quality, and

&#x20;  image-build checks.



3\. With ENABLE\_CD=true, a push to staging restores the selected model,

&#x20;  creates the environment file, builds the image, and runs a temporary

&#x20;  container for an HTTP smoke test.



4\. With ENABLE\_CD=true, a push to main restores the selected model,

&#x20;  creates the environment file, and builds the production image only.

## 

## Models



1\. Model: RandomForestRegressor.



2\. Data: UCI Bike Sharing. Version v1 uses January-June 2011, while v2

&#x20;  uses the full year 2011.



3\. Selection: the lowest MAE on the same 2012Q1 validation set.



4\. Selected run: <8b37de33e2ed4ec8a73eb2c74fd03084>.



5\. Validation MAE: <82.1735> rides/hour.



Data and model files are tracked with DVC and stored in Azure.



See ml/README\_data.md for data versions and ml/README\_experiments.md for

experiment results and model limitations.



### Code flow from commit to production



1\. Commit on a feature branch and open a pull request against dev; the request

&#x20;  runs tests, quality, and image-build in parallel, with no cloud secret.



2\. After one approval and three green checks, merge into dev, then promote dev

&#x20;  to staging and to main through separate pull requests.



3\. With ENABLE\_CD=true, a push to staging restores the selected model, writes

&#x20;  runtime/deploy.env, builds and runs the image, then removes the container; a

&#x20;  push to main restores the model and builds the production image only.



### Branches and pipeline triggers



1\. Any pull request triggers tests, quality, and image-build.



2\. A push to dev triggers the same three checks; a push to staging or main also

&#x20;  triggers the environment stage when ENABLE\_CD=true.



3\. A push to a feature branch triggers nothing; dev, staging, and main are

&#x20;  protected and need one approval plus three successful checks to merge.



