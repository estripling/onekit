from importlib import metadata

import onekit as ok


def test_version():
    assert ok.__version__ == metadata.version("onekit")
