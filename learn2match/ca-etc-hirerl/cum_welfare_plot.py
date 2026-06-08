"""Social welfare figures for the batched CA-ETC baseline.

Two figures, three time-varying curves each (mean +/- 95% CI bands across
seeds):

* ``actual``  -- per-period ``history["social_welfare"]`` (already computed
  in the baseline under the ``effective = matched | tentative_matched`` gauge).
* ``DA ref``  -- worker-proposing DA welfare on the TRUE U. Derived from
  existing history fields via the identity
  ``SW_da_ref = social_welfare + regret_w_total + regret_f_total``
  (since regret_*_total is already measured against ``ref_f * U_true``).
* ``planner`` -- first-best linear-assignment welfare on the true
  U = x @ y.T. Computed host-side per (seed, period) via
  ``scipy.optimize.linear_sum_assignment``. Requires ``x_snap`` and
  ``y_snap`` in the history.

Both ``DA ref`` and ``planner`` are optional and silently skipped if the
required history fields are missing -- the actual SW curve always plots.

Two public entry points:

* ``make_per_period_welfare_figure`` -- raw per-period welfare.
* ``make_cumulative_welfare_figure`` -- cumsum across periods.
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


def _band(ax, t, mean, lo, hi, label, color=None, ls="-"):
    line, = ax.plot(t, mean, label=label, color=color, linewidth=1.6, linestyle=ls)
    ax.fill_between(t, lo, hi, color=line.get_color(), alpha=0.15)
    return line


def _stats(arr: np.ndarray):
    """arr: (N, T) -> (mean_T, ci_lo_T, ci_hi_T)."""
    N = arr.shape[0]
    m = arr.mean(axis=0)
    if N > 1:
        sem = arr.std(axis=0, ddof=1) / np.sqrt(N)
    else:
        sem = np.zeros_like(m)
    half = 1.96 * sem
    return m, m - half, m + half


def _planner_per_period(x_snap: np.ndarray, y_snap: np.ndarray) -> np.ndarray:
    """First-best welfare per (seed, period) via linear-assignment on true U.

    x_snap: (N, T, Nw, d), y_snap: (N, T, Nf, d). Returns (N, T) float array.

    Welfare = 2 * sum(M_opt * <x, y>) -- factor of 2 mirrors HireRL's
    ``social_welfare`` accounting (workers + firms each see <x, y>).
    """
    from scipy.optimize import linear_sum_assignment

    N, T, Nw, _ = x_snap.shape
    Nf = y_snap.shape[2]
    out = np.zeros((N, T), dtype=np.float64)
    for n in range(N):
        for s in range(T):
            U = x_snap[n, s] @ y_snap[n, s].T  # (Nw, Nf)
            row, col = linear_sum_assignment(-U)
            out[n, s] = 2.0 * U[row, col].sum()
    return out


def _compute_welfare_series(
    history: Dict[str, np.ndarray],
    with_da_ref: bool,
    with_planner: bool,
) -> Dict[str, np.ndarray]:
    """Returns per-period welfare series. Each value is (N, T).

    Always includes ``actual``; ``da_ref`` and ``planner`` only if their
    inputs are present and the corresponding flag is True.
    """
    out: Dict[str, np.ndarray] = {}
    if "social_welfare" not in history:
        return out
    sw = np.asarray(history["social_welfare"])
    if sw.ndim != 2:
        return out
    out["actual"] = sw
    if with_da_ref and {"regret_w_total", "regret_f_total"}.issubset(history):
        rw = np.asarray(history["regret_w_total"])
        rf = np.asarray(history["regret_f_total"])
        out["da_ref"] = sw + rw + rf
    if with_planner:
        if "social_welfare_planner" in history:
            planner = np.asarray(history["social_welfare_planner"])
            if planner.ndim == 1:  # (N,) -- broadcast to (N, T) for plotting
                planner = np.broadcast_to(planner[:, None], sw.shape).copy()
            if planner.shape == sw.shape:
                out["planner"] = planner
        elif {"x_snap", "y_snap"}.issubset(history):
            out["planner"] = _planner_per_period(
                np.asarray(history["x_snap"]),
                np.asarray(history["y_snap"]),
            )
    return out


_STYLE = [
    ("actual",  "actual (belief, effective)",            "C0", "-"),
    ("da_ref",  "DA ref (true U, worker-proposing)",     "C2", "--"),
    ("planner", "planner (first-best, linear assignment)", "C3", "-"),
]


def _build_welfare_figure(
    series: Dict[str, np.ndarray],
    t: np.ndarray,
    *,
    cumulative: bool,
) -> Optional[plt.Figure]:
    if "actual" not in series:
        return None

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for key, label, color, ls in _STYLE:
        if key not in series:
            continue
        arr = series[key]
        if cumulative:
            arr = np.cumsum(arr, axis=1)
        m, lo, hi = _stats(arr)
        _band(ax, t, m, lo, hi, label=label, color=color, ls=ls)

    N = int(series["actual"].shape[0])
    if cumulative:
        ax.set_title(f"Cumulative social welfare  (N={N} seeds, mean ± 95% CI)")
        ax.set_ylabel("cumulative SW")
    else:
        ax.set_title(f"Social welfare per market period  (N={N} seeds, mean ± 95% CI)")
        ax.set_ylabel("SW per period")
    ax.set_xlabel("market period")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def make_cumulative_welfare_figure(
    history: Dict[str, np.ndarray],
    *,
    with_da_ref: bool = True,
    with_planner: bool = True,
) -> Optional[plt.Figure]:
    """Cumulative-over-time SW figure (3 lines, mean ± 95% CI)."""
    series = _compute_welfare_series(history, with_da_ref, with_planner)
    return _build_welfare_figure(series, np.asarray(history["t"]), cumulative=True)


def make_per_period_welfare_figure(
    history: Dict[str, np.ndarray],
    *,
    with_da_ref: bool = True,
    with_planner: bool = True,
) -> Optional[plt.Figure]:
    """Per-market-period SW figure (3 lines, mean ± 95% CI). No cumsum."""
    series = _compute_welfare_series(history, with_da_ref, with_planner)
    return _build_welfare_figure(series, np.asarray(history["t"]), cumulative=False)


def make_welfare_figures(
    history: Dict[str, np.ndarray],
    *,
    with_da_ref: bool = True,
    with_planner: bool = True,
    include_per_period: bool = True,
    include_cumulative: bool = True,
) -> Dict[str, plt.Figure]:
    """Build per-period and cumulative SW figures sharing one series compute.

    Returns a dict suitable for the plot_batched save loop:

        {"social_welfare": fig_per_period,
         "cumulative_social_welfare": fig_cumulative}

    Either entry is omitted if the corresponding ``include_*`` flag is False
    or the actual social_welfare trace is missing from history.
    """
    series = _compute_welfare_series(history, with_da_ref, with_planner)
    if "actual" not in series:
        return {}
    t = np.asarray(history["t"])
    figs: Dict[str, plt.Figure] = {}
    if include_per_period:
        fig = _build_welfare_figure(series, t, cumulative=False)
        if fig is not None:
            figs["social_welfare"] = fig
    if include_cumulative:
        fig = _build_welfare_figure(series, t, cumulative=True)
        if fig is not None:
            figs["cumulative_social_welfare"] = fig
    return figs
