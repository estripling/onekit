import os
import platform
import shlex
import shutil
import subprocess
from argparse import (
    ArgumentParser,
    Namespace,
)
from pathlib import Path
from subprocess import (
    CalledProcessError,
    CompletedProcess,
)
from typing import (
    Optional,
    Union,
)

Response = Union[CompletedProcess, CalledProcessError]


def main() -> None:
    args = get_arguments()

    print(f" branch - {get_current_branch()}")
    print(f"rootdir - {get_root()}")
    print(f"    cwd - {os.getcwd()}")

    if args.create_venv:
        print(" create - venv")
        process_argument__create_venv(poetry_version="1.8.3")

    else:
        process_argument__clear_cache(args.clear_cache)
        process_argument__pre_commit(args.run_pre_commit)
        process_argument__pytest(args.run_pytest)
        process_argument__pytest__slow(args.run_pytest_slow)
        process_argument__pytest__slow_doctests(args.run_pytest_slow_doctests)
        process_argument__create_docs(args.create_docs)
        process_argument__remove_docs(args.create_docs)
        process_argument__remove_branches(args.remove_branches)


def get_arguments() -> Namespace:
    parser = ArgumentParser(description="Execute command")
    parser.add_argument("--create-venv", action="store_true", help="create venv")
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="clear cache files and directories",
    )
    parser.add_argument(
        "--run-pre-commit",
        action="store_true",
        help="run pre-commit",
    )
    parser.add_argument(
        "--run-pytest",
        action="store_true",
        help="run pytest with coverage report",
    )
    parser.add_argument(
        "--run-pytest-slow",
        action="store_true",
        help="run slow tests with coverage report",
    )
    parser.add_argument(
        "--run-pytest-slow-doctests",
        action="store_true",
        help="run slow doctests",
    )
    parser.add_argument(
        "--create-docs",
        action="store_true",
        help="create local documentation files",
    )
    parser.add_argument(
        "--remove-docs",
        action="store_true",
        help="remove local documentation files",
    )
    parser.add_argument(
        "--remove-branches",
        action="store_true",
        help="remove local git branches, except main and current",
    )
    return parser.parse_args()


def process_argument__clear_cache(execute: bool):
    if execute:
        cwd = Path().cwd()
        file_extensions = ["*.py[co]", ".coverage", ".coverage.*"]
        directories = ["__pycache__", ".pytest_cache", ".ipynb_checkpoints"]

        for file_extension in file_extensions:
            for path in cwd.rglob(file_extension):
                if "venv" in str(path):
                    continue
                path.unlink()
                print(f"deleted - {path}")

        for directory in directories:
            for path in cwd.rglob(directory):
                if "venv" in str(path):
                    continue
                shutil.rmtree(path.absolute(), ignore_errors=False)
                print(f"deleted - {path}")


def process_argument__create_docs(execute: bool) -> None:
    if execute:
        print("create local documentation files")
        cwd = Path().cwd()
        os.chdir(get_root().joinpath("docs"))
        run_shell_command("make html")
        os.chdir(cwd)


def process_argument__create_venv(poetry_version: str) -> None:
    run_multiple_shell_commands(
        f"{get_python_exe()} -m venv {get_venv_path()} --clear --upgrade-deps",
        f"{get_python_venv_exe()} -m pip install poetry=={poetry_version}",
    )


def process_argument__pre_commit(execute: bool) -> None:
    if execute:
        run_shell_command("pre-commit run --all-files", print_cmd=True)


def process_argument__pytest(execute: bool) -> None:
    if execute:
        run_shell_command(
            [
                f"{get_python_venv_exe()} -m pytest",
                "--doctest-modules --ignore-glob=src/onekit/sparkkit.py src/",
                "--cov-report term-missing --cov=src/",
                "tests/",
            ],
            print_cmd=True,
        )


def process_argument__pytest__slow(execute: bool) -> None:
    if execute:
        run_shell_command(
            [
                f"{get_python_venv_exe()} -m pytest",
                "--slow",
                "--doctest-modules --ignore-glob=src/onekit/sparkkit.py src/",
                "--cov-report term-missing --cov=src/",
                "tests/",
            ],
            print_cmd=True,
        )


def process_argument__pytest__slow_doctests(execute: bool) -> None:
    if execute:
        run_shell_command(
            f"{get_python_venv_exe()} -m pytest --doctest-modules src/",
            print_cmd=True,
        )


def process_argument__remove_docs(execute: bool) -> None:
    if execute:
        pass


def process_argument__remove_branches(execute: bool) -> None:
    if execute:
        response__delete_local_branches = run_pipe_command(
            "git -P branch",
            "grep -v 'main'",
            f"grep -v '{get_current_branch()}'",
            "xargs git branch -D",
        )
        print(process(response__delete_local_branches))

        response__delete_remote_branch_reference = run_pipe_command(
            "git -P branch --all",
            "grep -v 'main'",
            f"grep -v '{get_current_branch()}'",
            r"sed 's/remotes\///g'",
            "xargs git branch -r -d",
        )
        print(process(response__delete_remote_branch_reference))


def get_current_branch() -> Optional[str]:
    response = run_shell_command("git rev-parse --abbrev-ref HEAD", capture_output=True)
    return process(response)


def decode(response: CompletedProcess) -> str:
    return response.stdout.decode("utf-8").rstrip()


def get_python_exe() -> str:
    return "python" if is_windows() else "python3"


def get_python_venv_exe() -> str:
    return str(
        Path()
        .joinpath(get_venv_path())
        .joinpath("Scripts" if is_windows() else "bin")
        .joinpath(get_python_exe())
        .as_posix()
    )


def get_root() -> Path:
    """Get path to root directory."""
    return Path(__file__).parent


def get_venv_path() -> str:
    return get_root().joinpath("venv").as_posix()


def has_command_run_successfully(response: CompletedProcess) -> bool:
    return response.returncode == 0


def is_linux() -> bool:
    return platform.system() == "Linux"


def is_windows() -> bool:
    return platform.system() == "Windows"


def process(response: Response) -> Optional[str]:
    if has_command_run_successfully(response):
        return decode(response)


def run_multiple_shell_commands(*commands: str, print_cmd: bool = True) -> None:
    """Execute a series of shell commands with Python."""
    for command in commands:
        run_shell_command(cmd=command, capture_output=False, print_cmd=print_cmd)


def run_pipe_command(*commands: str, print_cmd: bool = True) -> Response:
    """Execute a chain of shell commands with Python."""
    if print_cmd:
        print(" | ".join(commands))
    cmd, *cmds = commands
    response = run_shell_command(cmd, capture_output=True)
    for cmd_i in cmds:
        response = run_shell_command(cmd_i, other=response, capture_output=True)
    return response


def run_shell_command(
    cmd: str | list[str],
    capture_output: bool = False,
    print_cmd: bool = False,
    other: Optional[Response] = None,
) -> Response:
    """Execute a shell command with Python."""
    try:
        is_cmd_str = isinstance(cmd, str)
        cmd_to_run = shlex.split(cmd) if is_cmd_str else shlex.split(" ".join(cmd))
        if print_cmd:
            print(cmd if is_cmd_str else " ".join(cmd_to_run))
        stdout = other.stdout if isinstance(other, Response) else None
        return subprocess.run(
            cmd_to_run,
            input=stdout,
            capture_output=capture_output,
            timeout=None,
            check=True,
        )
    except CalledProcessError as error:
        return error


if __name__ == "__main__":
    main()
