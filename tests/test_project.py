import importlib
from importlib import metadata
from pathlib import Path


def test_project_name():
    project_name = get_project_name_from_toml()
    project_pkg = importlib.import_module(project_name)
    assert project_pkg.__name__ == project_name


def test_project_version():
    project_name = get_project_name_from_toml()
    project_pkg = importlib.import_module(project_name)
    assert project_pkg.__version__ == metadata.version(project_name)


def get_project_name_from_toml() -> str:
    path = get_root().joinpath("pyproject.toml").resolve()

    with open(file=str(path), mode="r") as lines:
        for line in lines:
            if line.startswith("name"):
                name = line.split("=")[1].strip().strip('"')
                break

    return name


def get_root() -> Path:
    """Get path to root directory."""
    return Path(__file__).parent.parent.resolve()
