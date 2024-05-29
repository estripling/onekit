import shutil
from pathlib import Path


def main():
    cwd = Path().cwd()
    file_extensions = ["*.py[co]", ".coverage", ".coverage.*"]
    directories = ["__pycache__", ".pytest_cache", ".ipynb_checkpoints"]

    for file_extension in file_extensions:
        for path in cwd.rglob(file_extension):
            if "venv" in str(path):
                continue
            path.unlink()
            print(f"delete - {path}")

    for directory in directories:
        for path in cwd.rglob(directory):
            if "venv" in str(path):
                continue
            shutil.rmtree(path.absolute(), ignore_errors=False)
            print(f"delete - {path}")


if __name__ == "__main__":
    main()
