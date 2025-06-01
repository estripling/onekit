# Dev Guide

## Local Development Environment Setup

```shell
python playbook.py
```

```shell
python playbook.py --help
```

```shell
python playbook.py --create venv
```

```shell
python playbook.py --pre-commit --pytest
```

## Useful Poetry commands

```shell
poetry check --lock
```

```shell
poetry lock
```

```shell
poetry add --optional base \
toolz \
pytz
```

```shell
poetry install --extras "base"
```

```shell
poetry add --optional analytics \
pandas[compression,computation,excel,output-formatting,parquet,performance,plot] \
scikit-learn \
tqdm
```

```shell
poetry install --extras "analytics"
```

```shell
poetry add --optional pyspark \
pyspark==3.5.3
```

```shell
poetry install --extras "pyspark"
```

```shell
poetry install --all-extras
```

```shell
poetry add --group dev \
autoflake \
black[jupyter] \
isort \
flake8 \
pre-commit \
pre-commit-hooks \
pytest \
pytest-cov \
pytest-skip-slow
```

```shell
poetry add --group docs \
furo \
jupyterlab \
myst-parser \
nbsphinx \
sphinx-autoapi \
sphinx-copybutton \
time-machine
```

```shell
poetry add --group packaging \
python-semantic-release
```

```shell
poetry build
```

```shell
poetry publish
```

## Git Commit Guidelines

This project applies semantic versioning
with [Python Semantic Release](https://python-semantic-release.readthedocs.io/en/stable/)
using [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
