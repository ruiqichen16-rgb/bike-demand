ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}
WORKDIR /app
ARG PIP_INDEX_URL=https://pypi.org/simple
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" -r requirements.txt && useradd --uid 10001 --no-create-home bike
COPY app/ app/
COPY build/model/ models/
ENV MODEL_DIR=/app/models
USER 10001
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
