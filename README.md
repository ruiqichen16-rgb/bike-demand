# Bike-Sharing Hourly Demand Estimator

Course project for DevOps and MLOps. The app estimates hourly bike rental demand
from historical date, hour, and weather inputs. It stores no business data.

## Functions

1.Hourly demand prediction: estimate rentals per hour for one scenario.

2.Scenario comparison: compare two scenarios that differ only in the hour.

3.Error evaluation: compute residual and absolute error for a prediction.

## Team

1.Chen ruiqi: application, HTTP interface, tests.

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





