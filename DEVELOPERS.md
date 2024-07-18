# Developer Guide

## Development Environment: Dev Container

This project comes with a [development container (or dev container)](https://containers.dev) setup, which allows for a reproducible development environment.
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

The `c2.py` file in the root directory provides a collection of commands.
In the dev container, open a new terminal and run the following command to see what is available:

```shell
python c2.py --help
```

Use the following command to run `pre-commit` and `pytest` with one command to make sure everything works as intended:

```shell
python c2.py --check
```

## Development Environment: Local Python Venv

To install a virtual Python environment locally, run the following commands from the root directory:

```shell
which python
```

```shell
python c2.py --create-venv
```

```shell
source venv/onekit_on_windows/Scripts/activate  # Windows - Git Bash
```

```shell
source venv/onekit_on_linux/bin/activate  # Linux
```

```shell
poetry install --no-interaction; \
pre-commit install
```

```shell
python c2.py --check
```

## Useful Poetry commands

```shell
poetry check --lock
```

```shell
poetry lock --no-update
```

```shell
poetry add \
toolz \
pytz
```

```shell
poetry add --group precommit \
autoflake \
"black[jupyter]" \
isort \
"flake8>=5.0.4" \
pre-commit \
pre-commit-hooks
```

```shell
poetry add --group testing \
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
poetry add --group pandaskit \
"pandas>=0.23.2"
```

```shell
poetry add --group sklearnkit \
"scikit-learn>=1.3"
```

```shell
poetry add --group sparkkit \
pyspark==3.1.1
```

```shell
poetry add --group vizkit \
"matplotlib>=3.7.1"
```

```shell
poetry build
```

```shell
poetry publish
```

## Git Commit Guidelines

This project applies semantic versioning with [Python Semantic Release](https://python-semantic-release.readthedocs.io/en/stable/commit-parsing.html#built-in-commit-parsers) with the default [Commit Message Format](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#-commit-message-format) according to the Angular guidelines.
