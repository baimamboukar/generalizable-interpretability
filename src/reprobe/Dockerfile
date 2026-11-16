# The agent's environment. Everything is installed here because the container has no network at run time.
FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

RUN apt-get update && apt-get install -y --no-install-recommends poppler-utils git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --root-user-action=ignore -r /tmp/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

RUN mkdir -p /work/output

WORKDIR /work
