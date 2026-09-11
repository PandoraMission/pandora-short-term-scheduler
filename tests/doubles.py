"""Shared pieces for the visibility stand-ins the tests build."""

# Standard library
import functools

# Third-party
import numpy as np
from astropy import units as u


def visibility_result(visible, roll=None, optimize_roll=False):
    """The dict ``Visibility.get_visibility`` returns, built from a verdict.

    Mirrors pandoravisibility v2.0.0: every per-timestep field takes the
    shape of ``visible``. The boresight is taken to pass wherever the
    target is visible, a searched roll is always 0 deg, a given roll is
    echoed and the Sun-constrained attitude reports NaN.
    """
    visible = np.asarray(visible, dtype=bool)
    if optimize_roll:
        roll_deg = np.zeros(visible.shape)
    elif roll is None:
        roll_deg = np.full(visible.shape, np.nan)
    else:
        roll_deg = np.full(visible.shape, float(u.Quantity(roll, u.deg).value))
    return {
        "visible": visible,
        "boresight_visible": visible,
        "roll_deg": roll_deg,
        "n_visible": int(visible.sum()),
        "n_st_pass": visible.astype(int),
        "solar_power_frac": np.where(visible, 1.0, np.nan),
    }


def answers_visibility(method):
    """Adapt a double's boolean ``get_visibility`` to the v2.0.0 contract.

    The double keeps scripting the verdict as a boolean array (or a bare
    bool for a scalar time). The wrapper hands it back inside the result
    dict and accepts the roll-search options, so one double serves both a
    fixed-roll query and the roll sweep, which gets roll 0 with ``visible``
    unchanged.
    """

    @functools.wraps(method)
    def get_visibility(
        self,
        coord,
        times,
        roll=None,
        *,
        optimize_roll=False,
        roll_step=None,
        min_power_frac=None,
        weights=None
    ):
        return visibility_result(
            method(self, coord, times, roll), roll, optimize_roll
        )

    return get_visibility
