"""Script to remove cache files and directories."""

import pathlib
import shutil


def main():
    cwd = pathlib.Path(".")
    file_extensions = ["*.py[co]", ".coverage", ".coverage.*"]
    directories = ["__pycache__", ".pytest_cache", ".ipynb_checkpoints"]

    for file_extension in file_extensions:
        for path in cwd.rglob(file_extension):
            path.unlink()
            print(f"deleted {path}")

    for directory in directories:
        for path in cwd.rglob(directory):
            shutil.rmtree(path.absolute(), ignore_errors=False)
            print(f"deleted {path}")


if __name__ == "__main__":
    main()
