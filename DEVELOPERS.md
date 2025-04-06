# Developer Guide

## Development Environment: Dev Container

This project comes with a [development container (or dev container)](https://containers.dev) setup, which allows for a
reproducible development environment.
To get started with the development, make sure you have the following software installed on your machine:

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [VS Code editor](https://code.visualstudio.com)
- [VS Code extension: Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- [VS Code extension: Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)

Next,

1. Start Docker Desktop
2. Clone repository
3. Open repository with VS Code using the `Open Folder` option
4. Click `Reopen in Container` when VS Code prompts you

Once the dev container is created, you are all set and ready to code!

### Things To Try

The `playbook.py` file in the root directory provides a set of predefined commands.
In the dev container, open a new terminal and run the following command to see what is available:

```shell
python playbook.py --help
```

Use the following command to run `pre-commit` and `pytest` with one command to make sure everything works as intended:

```shell
python playbook.py --check
```

## Development Environment: Local Python Venv

To install a virtual Python environment locally, run the following commands from the root directory:

```shell
which python  # Bash-like shell
```

```shell
python playbook.py --create-venv
```

```shell
source .venv/onekit_on_windows/Scripts/activate  # Windows - Git Bash
```

```shell
source .venv/onekit_on_linux/bin/activate  # Linux
```

```shell
poetry install --no-interaction --all-extras; \
pre-commit install
```

```shell
python playbook.py --check
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
with [Python Semantic Release](https://python-semantic-release.readthedocs.io/en/stable/) using [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
