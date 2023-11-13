from importlib import metadata

import onekit


def test_version():
    assert onekit.__version__ == metadata.version("onekit")
