import importlib
import platform
import shlex
import shutil
import subprocess
from argparse import (
    ArgumentParser,
    Namespace,
)
from functools import partial
from importlib import metadata
from pathlib import Path
from subprocess import (
    CalledProcessError,
    CompletedProcess,
)
from typing import (
    Iterable,
    NamedTuple,
)

Response = CompletedProcess | str


class Config(NamedTuple):
    commit_hash_length: int | None = 7
    poetry_version: str = "2.1.1"
    project: str = "onekit"


def main() -> None:
    args = get_arguments()
    cfg = Config()

    print(f" project - {cfg.project}")
    print(f" version - {get_project_version(cfg.project)}")
    print(f"    root - {get_root()}")
    print(f"  python - {which_python()}")
    print(f"packages - {get_num_packages_in_venv()}")
    print(f"  branch - {get_current_branch()}")
    print(f"  commit - {get_last_commit(max_length=cfg.commit_hash_length)}")
    print()

    if all(is_inactive(value) for value in args.__dict__.values()):
        print("see help:")
        print(f"python {get_current_filename()} -h")

    elif args.create_venv:
        print(f"  create - venv on {get_platform_name()}")
        run_create_venv(cfg.poetry_version)

    else:
        functions = [
            (run_pre_commit, args.pre_commit),
            (partial(run_pytest, cfg.project), args.pytest),
            (run_clear_cache, args.clear_cache),
            (run_remove_branches, args.remove_branches),
        ]

        for func, value in functions:
            if is_inactive(value):
                continue

            elif value is True:
                func()

            else:
                func(value)


def get_arguments() -> Namespace:
    parser = ArgumentParser(description="set of predefined commands")
    parser.add_argument(
        "--pre-commit",
        nargs="?",
        const=True,
        default=None,
        choices=["all"],
        help="run pre-commit",
    )
    parser.add_argument(
        "--pytest",
        nargs="?",
        const=True,
        default=None,
        choices=["slow"],
        help="run pytest",
    )
    parser.add_argument(
        "--create-venv",
        action="store_true",
        help="create virtual Python environment for project",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="clear cache files and directories",
    )
    parser.add_argument(
        "--remove-branches",
        action="store_true",
        help=(
            "remove both local and remote-tracking git branches "
            "except main and current"
        ),
    )
    return parser.parse_args()


def run_clear_cache() -> None:
    print("   clear - cache")
    root = get_root()
    directories = [
        "__pycache__",
        ".pytest_cache",
        ".ipynb_checkpoints",
        "spark-warehouse",
    ]
    file_extensions = [
        "*.py[co]",
        ".coverage",
        ".coverage.*",
    ]

    for directory in directories:
        for path in root.rglob(directory):
            if "venv" in str(path):
                continue
            shutil.rmtree(path.absolute(), ignore_errors=False)
            print(f" deleted - {path}")

    for file_extension in file_extensions:
        for path in root.rglob(file_extension):
            if "venv" in str(path):
                continue
            path.unlink()
            print(f" deleted - {path}")


def run_create_venv(poetry_version: str) -> None:
    commands = [
        f"{get_global_python()} -m venv {get_venv_path()} --clear",
        f"{get_local_python()} -m pip install --upgrade pip",
        f"{get_local_python()} -m pip install poetry=={poetry_version}",
        f"{get_local_python()} -m poetry install --no-interaction --all-extras",
        f"{get_local_precommit()} install",
    ]

    for command in commands:
        execute_subprocess(command, print_command=True)


def run_pre_commit(option: str | None = None) -> None:
    command = "pre-commit run"
    if option == "all":
        command += " --all-files"
    execute_subprocess(command, print_command=True)


def run_pytest(project: str, option: str | None = None) -> None:
    sparkkit_module = f"src/{project}/sparkkit.py"
    base_command = (
        f"{get_local_python()} -m pytest "
        f"--doctest-modules --ignore-glob={sparkkit_module} src/ "
        "--cov-report term-missing --cov=src/ "
        "tests/"
    )

    commands = (
        [
            f"{get_local_python()} -m pytest --doctest-modules {sparkkit_module}",
            f"{base_command} --slow",
        ]
        if option == "slow"
        else [base_command]
    )

    for command in commands:
        execute_subprocess(command, print_command=True)


def run_remove_branches() -> None:
    """Remove both local and remote-tracking git branches except main and current."""
    pipelines = {
        "delete local branches": {
            "title": "removing local git branches except main and current",
            "commands": (
                "git -P branch",
                f"grep --invert-match --word-regexp -E 'main|{get_current_branch()}'",
                "xargs git branch -D",
            ),
        },
        "delete remote branches": {
            "title": "removing remote-tracking git branches except main and current",
            "commands": (
                "git -P branch --all",
                f"grep --invert-match --word-regexp -E 'main|{get_current_branch()}'",
                r"sed 's/remotes\///g'",
                "xargs git branch -r -d",
            ),
        },
    }

    for i, pipeline in enumerate(pipelines.values(), start=1):
        print(pipeline["title"])
        try:
            print(execute_chained_subprocesses(*pipeline["commands"]))
        except CalledProcessError:
            print(None)
        if i < len(pipeline):
            print()


def get_current_branch() -> str:
    return execute_subprocess(
        "git rev-parse --abbrev-ref HEAD",
        capture_output=True,
        return_string=True,
    )


def get_current_filename() -> str:
    return Path(__file__).name


def get_global_python() -> str:
    return "python3"


def get_local(program: str, /) -> str:
    return str(Path().joinpath(get_venv_path()).joinpath("bin").joinpath(program))


def get_local_precommit() -> str:
    return get_local("pre-commit")


def get_local_python() -> str:
    return get_local("python")


def get_last_commit(max_length: int | None = None) -> str:
    """Returns hash of the most recent commit of current branch."""
    response = execute_subprocess(
        "git rev-parse HEAD",
        capture_output=True,
        return_string=True,
    )
    return response[:max_length]


def get_num_packages_in_venv() -> int:
    num_pkg = sum(1 for _ in metadata.distributions())  # count includes current project
    if num_pkg < 10:
        num_pkg = execute_chained_subprocesses(
            "pip list --format=freeze",
            "wc -l",
            print_command=False,
            return_string=True,
        )
    return int(num_pkg)


def get_platform_name() -> str:
    return platform.system().lower()


def get_project_version(project_name: str) -> str:
    try:
        project_pkg = importlib.import_module(project_name)

        pip_version = project_pkg.__version__
        toml_version = get_project_version_from_toml()
        is_stale = (
            f'(stale: version in pyproject.toml is "{toml_version}")'
            if pip_version != toml_version
            else ""
        )
        return f"{pip_version} {is_stale}".strip()

    except ImportError:
        return "project not installed"


def get_project_version_from_toml() -> str:
    path = get_root().joinpath("pyproject.toml").resolve()

    with open(file=str(path), mode="r") as lines:
        for line in lines:
            if line.startswith("version"):
                version = line.split("=")[1].strip().strip('"')
                break

    return version


def get_root() -> Path:
    """Get path to root directory."""
    return Path(__file__).parent.resolve()


def get_venv_path() -> str:
    return str(get_root().joinpath(".venv").resolve())


def is_inactive(value) -> bool:
    """Check if value signals an inactive flag."""
    return value in [None, False]


def which_python() -> str:
    return execute_subprocess("which python", capture_output=True, return_string=True)


def execute_chained_subprocesses(
    *commands: str | Iterable[str],
    print_command: bool = True,
    return_string: bool = True,
) -> Response:
    if print_command:
        print(" | ".join(commands))

    response = None
    for command in commands:
        response = execute_subprocess(command, capture_output=True, other=response)

    return cast_to_str(response) if return_string else response


def execute_subprocess(
    command: str,
    /,
    *,
    print_command: bool = False,
    capture_output: bool = False,
    return_string: bool = False,
    other: CompletedProcess | None = None,
) -> Response:
    cmd = shlex.split(command)

    if print_command:
        print(" ".join(cmd))

    response = subprocess.run(
        cmd,
        input=other.stdout if hasattr(other, "stdout") else None,
        capture_output=capture_output,
        timeout=None,
        check=True,
    )

    return cast_to_str(response) if return_string else response


def cast_to_str(response: CompletedProcess) -> str:
    return response.stdout.decode("utf-8").rstrip()


if __name__ == "__main__":
    main()
