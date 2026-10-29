"""Shared pytest fixtures."""

# Third-party
import numpy as np
import pytest
from pandoravisibility import Visibility


@pytest.fixture(autouse=True)
def empty_trace_field(monkeypatch):
    """Give every target an empty NIRDA star field.

    The trace check in the roll search (on by default since
    pandoravisibility v2.1.0) would otherwise read the gaiaoffline
    database, which CI does not have. With no neighbors every roll is
    clean, so the search returns what it did before the check existed.
    Mirrors the fixture of the same name in pandoravisibility's tests.
    """
    empty = {
        "target_source_id": 0,
        "target_jmag": np.nan,
        "xi": np.array([]),
        "eta": np.array([]),
        "jmag": np.array([]),
        "flux_ratio": np.array([]),
    }
    monkeypatch.setattr(
        Visibility, "_trace_field", lambda self, coord, time: empty
    )
