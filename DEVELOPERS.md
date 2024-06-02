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

The `Makefile` in the root directory provides a collection of common commands.
In the dev container, open a new terminal and run the following command to see what is available:

```shell
make help
```

Use the following command to run `pre-commit` and `pytest` with one command to make sure everything works as intended:

```shell
make check
```

## Development Environment: Local Python VENV

To install a virtual Python environment locally, run the following commands from the root directory:

```shell
which python
```

```shell
python command.py --create-venv
```

```shell
source venv/Scripts/activate  # Windows - Git Bash
```

```shell
source venv/bin/activate  # Linux
```

```shell
poetry install --no-interaction; \
pre-commit install
```

```shell
python command.py --check
```

## Git Commit Guidelines

This project applies semantic versioning with [Python Semantic Release](https://python-semantic-release.readthedocs.io/en/stable/commit-parsing.html#built-in-commit-parsers) with the default [Commit Message Format](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#-commit-message-format) according to the Angular guidelines.
