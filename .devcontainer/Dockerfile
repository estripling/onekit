FROM python:3.11.9-slim@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317

# configurations
ENV PYTHONUNBUFFERED=on \
    PYTHONFAULTHANDLER=on \
    PYTHONHASHSEED=random \
    PYTHONDONTWRITEBYTECODE=on \
    # pip:
    PIP_NO_CACHE_DIR=on \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=120 \
    # poetry:
    POETRY_VERSION=2.1.1 \
    POETRY_NO_ANSI=on \
    POETRY_NO_INTERACTION=on \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_HOME="/opt/poetry" \
    PATH="${POETRY_HOME}/bin:${PATH}"

# install additional system packages
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    gcc \
    make \
    pandoc \
    ## cleanup
    && apt-get clean -y \
    && apt-get autoclean -y \
    && apt-get autoremove --purge  -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces/onekit

COPY . .

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install poetry==${POETRY_VERSION} \
    && poetry install --no-interaction --all-extras \
    && rm -rf ~/.cache/pypoetry/{cache,artifacts}
