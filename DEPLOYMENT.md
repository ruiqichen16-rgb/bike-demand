# Environment builds and temporary staging

The only active workflow is .github/workflows/ci.yml. Pull requests do not read
cloud secrets. The required checks are tests, quality, and image-build.
ENABLE_CD remains unset during preparation; enable it after the selected model
exists in Azure and both GitHub Environments are configured. Final CD jobs must
actually succeed for the intended commit, not be skipped.

## Environment access

Allow the staging branch in the staging Environment and main in production.
Each Environment has AZURE_STORAGE_ACCOUNT, AZURE_STORAGE_SAS_TOKEN, and
APP_MESSAGE. The SAS permits authorized read/list access to the course container
with HTTPS and an appropriate expiry. Local uploads use separately authorized
write access. Never expose a write token to a pull request. APP_MESSAGE must be
single-line ASCII English text, such as bike-staging or bike-production.

## Current release sequence

scripts.prepare_release restores models/model.pkl.dvc from the committed myazure
remote without retraining. It verifies the selected purpose and model hash,
then copies model.pkl and manifest.json into build/model.

The script creates runtime/deploy.env with APP_ENV and APP_MESSAGE only. Azure
credentials stay in .dvc/config.local temporarily and never enter that file.
Do not COPY secrets or pass them through Docker ARG, ENV, or image labels.

- Staging: restore -> create env file -> build -> run with --env-file -> HTTP
  smoke for all three functions -> always remove the container and local config.
- Production: restore -> create env file -> build -> always remove local config.
  This job does not run a service or publish to Docker Hub.

The current Dockerfile does not consume runtime/deploy.env during docker build.
Staging consumes it at container startup. Production creates it before building
but does not otherwise consume it. Do not claim that environment settings have
been embedded in either image. An explicit build-configuration change is a
separate revision, not part of this English-text update.

A temporary staging runner is not a persistent public website. The final project
recommends environment builds and does not request production deployment.
If the instructor specifies additional acceptance criteria, confirm the scope
before adding a host, registry, or deployment platform.

## Failure handling

Check the failing step, DVC object availability, authorization, and manifest.
Do not delete the only model or garbage-collect shared Azure objects. Azure errors
can contain signed URLs; public release errors stay redacted. Enter credentials
only in authorized local terminals or platform secret fields. Never commit raw
data, model binaries, runtime databases, .env, or .dvc/config.local.
