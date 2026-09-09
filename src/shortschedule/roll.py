"""One spacecraft roll per target per visit.

Every observation of a target within a visit flies the same roll; other
targets in that visit, and the same target in other visits, are solved on
their own. The search itself is ``pandoravisibility.Visibility.get_visibility``
with ``optimize_roll=True``. This module only decides which minutes to score,
how much each counts, and which keepout model applies.
"""

# Standard library
from typing import Any, Dict, Optional

# Third-party
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord


def get_best_roll_per_visit(
    visit: Any,
    visibility: Any,
    roll_step: float = 1.0,
    min_power_frac: Optional[float] = None,
    priority_0_visibility: Any = None,
    growth_margin_minutes: float = 0.0,
) -> Dict[str, dict]:
    """Choose one roll per target in ``visit`` and write it onto its sequences.

    The minutes scored are every scheduled minute of the target in this
    visit plus ``growth_margin_minutes`` on each side of each observation.
    Scheduled minutes weigh ``n_margin + 1`` and margin minutes 1, so the
    roll is first the one with the most visible scheduled minutes, then
    among those the one keeping the most margin minutes visible, then the
    best lit: growth minutes only ever break ties.

    Parameters
    ----------
    visit : Visit
        Its sequences get ``roll`` set in place, in degrees.
    visibility : pandoravisibility.Visibility
        Model whose ``get_visibility(optimize_roll=True)`` runs the search.
    roll_step : float, optional
        Sweep resolution in degrees.
    min_power_frac : float, optional
        Rolls whose mean solar power is below this are not considered.
    priority_0_visibility : pandoravisibility.Visibility, optional
        Stricter model used when every observation of a target in this
        visit is priority 0. A target with mixed priorities keeps
        ``visibility``, since one roll has to serve the whole visit and the
        strictest member would over-constrain the rest.
    growth_margin_minutes : float, optional
        Minutes on each side of every observation the roll should also try
        to keep visible, so growth has somewhere to go.

    Returns
    -------
    dict
        Target name to the ``get_visibility`` result, with ``roll_deg``
        reduced to the one roll held, and ``scheduled`` (which scored
        minutes were scheduled) and ``n_scheduled_visible`` added. Zero
        scheduled minutes visible means no roll can observe the target and
        the fallback attitude was written.
    """
    by_target: Dict[str, list] = {}
    for seq in visit.sequences:
        by_target.setdefault(seq.target, []).append(seq)

    margin = int(np.rint(growth_margin_minutes))
    results: Dict[str, dict] = {}
    for target, sequences in by_target.items():
        sequences = sorted(sequences, key=lambda s: s.start_time)
        origin = sequences[0].start_time
        offsets, scheduled = [], []
        for seq in sequences:
            n_minutes = max(1, int(np.rint(seq.duration.sec / 60.0)))
            local = np.arange(-margin, n_minutes + margin)
            offsets.append(
                (seq.start_time - origin).to_value(u.min) + local
            )
            scheduled.append((local >= 0) & (local < n_minutes))
        offsets = np.concatenate(offsets)
        scheduled = np.concatenate(scheduled)
        times = origin + offsets * u.min
        weights = np.where(scheduled, int((~scheduled).sum()) + 1, 1)

        model = visibility
        if priority_0_visibility is not None and all(
            seq.priority == 0 for seq in sequences
        ):
            model = priority_0_visibility
        coord = SkyCoord(
            sequences[0].ra, sequences[0].dec, unit="deg", frame="icrs"
        )
        result = model.get_visibility(
            coord,
            times,
            optimize_roll=True,
            roll_step=roll_step * u.deg,
            min_power_frac=min_power_frac,
            weights=weights,
        )
        # The search holds one roll over every scored minute and echoes it
        # per timestep; the visit rule wants that one number.
        result["roll_deg"] = float(np.atleast_1d(result["roll_deg"])[0])
        result["scheduled"] = scheduled
        result["n_scheduled_visible"] = int(
            np.asarray(result["visible"])[scheduled].sum()
        )
        for seq in sequences:
            seq.roll = result["roll_deg"]
        results[target] = result
    return results
