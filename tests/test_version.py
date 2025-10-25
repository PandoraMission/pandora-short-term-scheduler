# First-party/Local
from shortschedule import __version__


def test_version_is_string():
    # Ensure package exposes a version string (CI may install a built package)
    assert isinstance(__version__, str)
    assert len(__version__) > 0
