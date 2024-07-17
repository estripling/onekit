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
    List,
    Optional,
    Union,
)

Response = Union[CompletedProcess, CalledProcessError]


def main() -> None:
    args = get_arguments()

    print(f" branch - {get_current_branch()}")
    print(f" commit - {get_last_commit(7)}")
    print(f"rootdir - {get_root().as_posix()}")
    print(f"    cwd - {Path.cwd().as_posix()}")

    if all(v is False for v in args.__dict__.values()):
        print()
        print("see help:")
        print(f"python {Path(__file__).name} -h")

    elif args.create_venv:
        print(" create - venv")
        run_create_venv(poetry_version="1.8.3")

    else:
        functions = [
            (run_check, args.check),
            (run_clear_cache, args.clear_cache),
            (run_pre_commit, args.run_pre_commit),
            (run_pytest, args.run_pytest),
            (run_pytest__slow, args.run_pytest_slow),
            (run_pytest__slow_doctests, args.run_pytest_slow_doctests),
            (run_create_docs, args.create_docs),
            (run_remove_docs, args.remove_docs),
            (run_remove_branches, args.remove_branches),
        ]
        for func, condition in functions:
            if condition:
                func()


def get_arguments() -> Namespace:
    parser = ArgumentParser(description="command and control")
    parser.add_argument(
        "-c",
        "--check",
        action="store_true",
        help="run sequence of commands to check code quality",
    )
    parser.add_argument(
        "--create-venv",
        action="store_true",
        help="create virtual Python environment",
    )
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


def run_check() -> None:
    run_pre_commit()
    run_pytest()


def run_clear_cache() -> None:
    print("  clear - cache")
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


def run_create_docs() -> None:
    print(" create - local documentation files")
    cwd = Path().cwd()
    os.chdir(get_root().joinpath("docs"))
    run_shell_command("make html")
    os.chdir(cwd)


def run_create_venv(poetry_version: str) -> None:
    run_multiple_shell_commands(
        f"{get_local_python()} -m venv {get_venv_path()} --clear",
        f"{get_python_exe()} -m pip install --upgrade pip",
        f"{get_python_exe()} -m pip install poetry=={poetry_version}",
    )


def run_pre_commit() -> None:
    run_shell_command("pre-commit run --all-files", print_cmd=True)


def run_pytest() -> None:
    run_shell_command(
        [
            f"{get_python_exe()} -m pytest",
            "--doctest-modules --ignore-glob=src/onekit/sparkkit.py src/",
            "--cov-report term-missing --cov=src/",
            "tests/",
        ],
        print_cmd=True,
    )


def run_pytest__slow() -> None:
    run_shell_command(
        [
            f"{get_python_exe()} -m pytest",
            "--slow",
            "--doctest-modules --ignore-glob=src/onekit/sparkkit.py src/",
            "--cov-report term-missing --cov=src/",
            "tests/",
        ],
        print_cmd=True,
    )


def run_pytest__slow_doctests() -> None:
    run_shell_command(
        f"{get_python_exe()} -m pytest --doctest-modules src/",
        print_cmd=True,
    )


def run_remove_docs() -> None:
    path = get_root().joinpath("docs").joinpath("_build").resolve()
    if path.exists():
        shutil.rmtree(path, ignore_errors=False)
        print(f"deleted - {path}")


def run_remove_branches() -> None:
    """Remove local git branches, except main and current."""
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


def get_last_commit(n: Optional[int] = None) -> Optional[str]:
    """Returns hash of the most recent commit of current branch."""
    response = run_shell_command("git rev-parse HEAD", capture_output=True)
    return process(response)[:n]


def decode(response: CompletedProcess) -> str:
    return response.stdout.decode("utf-8").rstrip()


def get_local_python() -> str:
    return "python.exe" if is_windows() else "python3"


def get_python_exe() -> str:
    return "python3" if is_docker_container() else get_python_venv_exe()


def get_python_venv_exe() -> str:
    return str(
        Path()
        .joinpath(get_venv_path())
        .joinpath("Scripts" if is_windows() else "bin")
        .joinpath(get_local_python())
        .as_posix()
    )


def get_root() -> Path:
    """Get path to root directory."""
    return Path(__file__).parent.resolve()


def get_venv_path() -> str:
    name = f"onekit_on_{platform.system().lower()}"
    return get_root().joinpath("venv").joinpath(name).resolve().as_posix()


def has_command_run_successfully(response: CompletedProcess) -> bool:
    return response.returncode == 0


def is_docker_container() -> bool:
    return str(get_root().resolve()).startswith("/workspaces/onekit")


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
    cmd: Union[str, List[str]],
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
        stdout = other.stdout if hasattr(other, "stdout") else None
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
