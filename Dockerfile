# Lambda container image for the SentinelOps API.
#
# Built as a container rather than a .zip because the zip route caps at 250 MB
# unzipped and xgboost + shap + scikit-learn + pandas blow past that on their
# own. Container images get 10 GB, and this one lands around 1.2 GB.
#
# BUILD (from repo root — the build context must be the repo root, not deploy/):
#   docker build -t sentinelops-api .
#
# RUN LOCALLY (the base image bundles the Lambda Runtime Interface Emulator):
#   docker run --rm -p 9000:8080 sentinelops-api
#   curl -s "http://localhost:9000/2015-03-31/functions/function/invocations" \
#     -d '{"version":"2.0","rawPath":"/health","requestContext":{"http":{"method":"GET","path":"/health"}},"headers":{},"isBase64Encoded":false}'

FROM public.ecr.aws/lambda/python:3.12

# Build toolchain for any dependency without a manylinux wheel for this platform.
# Removed in the same layer so it never reaches the final image.
COPY deploy/requirements-lambda.txt ${LAMBDA_TASK_ROOT}/requirements-lambda.txt

RUN dnf install -y gcc gcc-c++ \
 && pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements-lambda.txt \
 && dnf remove -y gcc gcc-c++ \
 && dnf clean all \
 && rm -rf /var/cache/dnf

# The repo layout is preserved inside the image on purpose. features.py derives
# REPO_ROOT as `Path(__file__).parent.parent`, so src/ must sit one level under
# the root that holds data/, docs/ and models/ — exactly as it does locally.
# Flattening src/* into the task root would silently point REPO_ROOT at /var.
COPY src/    ${LAMBDA_TASK_ROOT}/src/

# Runtime data. The Engine builds its in-memory frame from the CSV at load time;
# /metrics and /playbook read these JSON files off disk.
COPY data/ai4i2020.csv        ${LAMBDA_TASK_ROOT}/data/ai4i2020.csv
COPY docs/model_metrics.json  ${LAMBDA_TASK_ROOT}/docs/model_metrics.json
COPY docs/playbook.json       ${LAMBDA_TASK_ROOT}/docs/playbook.json
COPY docs/cost_params.json    ${LAMBDA_TASK_ROOT}/docs/cost_params.json

# Trained artefacts, bundled by default (~7.5 MB). Set MODEL_S3_BUCKET on the
# function to pull them from S3 at cold start instead — see src/api/model_store.py.
COPY models/ ${LAMBDA_TASK_ROOT}/models/

# Lets the runtime resolve `api.lambda_handler` and lets that module's own
# imports (`predict`, `features`, `rules_engine`) work unchanged.
ENV PYTHONPATH="${LAMBDA_TASK_ROOT}/src:${LAMBDA_TASK_ROOT}/src/agent"

CMD [ "api.lambda_handler.handler" ]
