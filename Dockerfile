FROM python:3.10.4-slim as builder

ARG SNOOSPOOF_API_PORT \
    SNOOSPOOF_API_HOST

LABEL maintainer="Shing Chan <chan.shing@protonmail.com>"

#Install Poetry
#https://github.com/max-pfeiffer/python-poetry/blob/main/build/Dockerfile

ENV PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=${POETRY_VERSION} \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_NO_INTERACTION=1 \
#Poetry fails to install packages(torch) with correct hashes with the new installer
#Remove the below line once python-poetry fixes the issue. This configuration
#is unstable and may be deprecated in the future
#See https://github.com/python-poetry/poetry/issues/6301
    POETRY_EXPERIMENTAL_NEW_INSTALLER=0


ENV PATH="$POETRY_HOME/bin:$PATH"

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        curl \
    && curl -sSL https://install.python-poetry.org | python - \
    && apt-get purge --auto-remove -y \
      build-essential \
      curl

COPY pyproject.toml ./

RUN poetry check

RUN poetry install --no-root --no-ansi --without dev

#Copy poetry's virtualenv and remove poetry as a running dependency
#https://stackoverflow.com/a/70999427
FROM python:3.10.4-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/.venv/bin:$PATH" \
    SNOOSPOOF_DIR="/snoospoof" 

#Move the Huggingface Cache to own directory
ENV HUGGINGFACE_HUB_CACHE="$SNOOSPOOF_DIR/.cache"

COPY --from=builder /.venv /.venv

#Configure application
#https://github.com/max-pfeiffer/uvicorn-poetry/blob/main/build/Dockerfile
RUN groupadd -g 1001 snoospoofapi && \
    useradd -r -u 1001 -g snoospoofapi snoospoofapi

RUN chown snoospoofapi:snoospoofapi /.venv

WORKDIR ${SNOOSPOOF_DIR}

COPY src/SnooSpoof/ .

RUN chown snoospoofapi:snoospoofapi . 

EXPOSE ${SNOOSPOOF_API_PORT}

USER 1001

#Assume praw.ini is a valid secret from docker compose
#Links a secret to the project directory for PRAW scrapper
RUN ln -s /run/secrets/praw.ini ./praw.ini

CMD uvicorn api.middleman:app --host ${SNOOSPOOF_API_HOST} --port ${SNOOSPOOF_API_PORT}