"""Shared plotting utilities for HireRL settled-state rollouts.

Single-seed consumers (``examples/eval_and_plot.py`` for RL policy rollouts,
``ca-etc-hirerl/plot_ca_etc_baseline.py`` for the CA-ETC baseline) and
batched (multi-seed) consumers (``ca-etc-hirerl/plot_batched_ca_etc_baseline.py``
and ``examples/eval_and_plot.py`` when ``--num_seeds > 1``) share this
module, so all per-period metrics use the same gauge:
``hirerl.metrics.compute_all_metrics`` for aggregates plus the matching
``per_*_regret`` / ``per_*_information_loss`` for per-agent breakdowns.

Single-seed ``history`` schema (T = number of settled market periods):

* Scalar series, shape ``(T,)``:
    ``t``, ``reward_w_total``, ``reward_f_total``,
    ``regret_w_total``, ``regret_f_total``,
    ``info_loss_w_total``, ``info_loss_f_total``,
    ``social_welfare``, ``friction_loss``, ``match_rate``.
* Per-agent series, shape ``(T, N)`` (N = Nw or Nf):
    ``reward_w_per_agent``, ``reward_f_per_agent``,
    ``regret_w_per_agent``, ``regret_f_per_agent``,
    ``info_loss_w_per_agent``, ``info_loss_f_per_agent``.
* Bipartite match matrix, shape ``(T, Nw, Nf)`` bool: ``matched_matrix``.

Batched (N seeds) ``history`` schema:

* ``t`` is still ``(T,)``; ``num_seeds``, ``Nw``, ``Nf`` are scalar metadata.
* Per-period scalar series are ``(N, T)``.
* ``matched_matrix`` is ``(N, T, Nw, Nf)`` bool.
* Per-agent series carry an extra leading seed axis but are unused by the
  batched figure suite (per-agent panels would be illegible at N seeds).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def stack_history(records: List[Dict]) -> Dict[str, np.ndarray]:
    """Convert a list of per-period record dicts into the array-based history
    schema consumed by ``make_figures`` / ``make_gantt_figure``.

    Each record must contain the same keys; scalars become 1D arrays of length
    T, per-agent arrays become 2D arrays of shape (T, N), and ``matched_matrix``
    becomes a 3D array (T, Nw, Nf).
    """
    if not records:
        raise ValueError("stack_history received an empty record list")
    history: Dict[str, np.ndarray] = {}
    for key in records[0]:
        values = [r[key] for r in records]
        if key in ("matched_matrix",) or (
            isinstance(values[0], np.ndarray) and values[0].ndim >= 1
        ):
            history[key] = np.stack(values, axis=0)
        else:
            history[key] = np.asarray(values)
    return history


def make_figures(history: Dict[str, np.ndarray], env_config) -> Dict[str, plt.Figure]:
    """Build the standard suite of figures from a settled-state history.

    Returned keys: ``aggregate``, ``per_worker``, ``per_firm``, ``aux``,
    ``cumulative``, ``gantt_worker``.
    """
    figs: Dict[str, plt.Figure] = {}
    t = history["t"]

    # 1) Aggregate per-period: reward / regret / info-loss with workers vs firms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in [
        (axes[0], "reward",    "Total reward per period"),
        (axes[1], "regret",    "Total regret per period (hirerl gauge)"),
        (axes[2], "info_loss", "Total information loss per period"),
    ]:
        ax.plot(t, history[f"{key}_w_total"], label="workers")
        ax.plot(t, history[f"{key}_f_total"], label="firms")
        ax.set_title(title)
        ax.set_xlabel("market period")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    figs["aggregate"] = fig

    # 2) Per-side breakdown: one panel per metric, one line per agent
    for side, prefix, N in [
        ("worker", "w", env_config.Nw),
        ("firm",   "f", env_config.Nf),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, key, title in [
            (axes[0], "reward",    f"Per-{side} reward"),
            (axes[1], "regret",    f"Per-{side} regret"),
            (axes[2], "info_loss", f"Per-{side} information loss"),
        ]:
            data = history[f"{key}_{prefix}_per_agent"]   # (T, N)
            for i in range(N):
                ax.plot(t, data[:, i], alpha=0.7, linewidth=1.0, label=f"{side} {i}")
            ax.set_title(title)
            ax.set_xlabel("market period")
            ax.grid(alpha=0.3)
            if N <= 12:
                ax.legend(fontsize=6, loc="best", ncol=2)
        fig.tight_layout()
        figs[f"per_{side}"] = fig

    # 3) Auxiliary aggregate: welfare / friction / match_rate
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(t, history["social_welfare"], label="actual (belief)")
    axes[0].set_title("Social welfare (belief)")
    if "social_welfare_da_ref" in history:
        ref_const = float(np.max(history["social_welfare_da_ref"]))
        axes[0].axhline(ref_const, ls="--", color="k",
                        label="worker-proposing DA (true)")
        axes[0].legend(fontsize=8, loc="best")
    axes[1].plot(t, history["friction_loss"]);  axes[1].set_title("Friction loss (DA gap)")
    axes[2].plot(t, history["match_rate"]);     axes[2].set_title("Match rate")
    for ax in axes:
        ax.set_xlabel("market period")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    figs["aux"] = fig

    # 4) Cumulative regret + cumulative friction loss (paper-style sublinearity check)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cum_reg_w = np.cumsum(history["regret_w_total"])
    cum_reg_f = np.cumsum(history["regret_f_total"])
    cum_fric  = np.cumsum(history["friction_loss"])
    axes[0].plot(t, cum_reg_w, label="workers")
    axes[0].plot(t, cum_reg_f, label="firms")
    axes[0].set_title("Cumulative regret")
    axes[0].set_xlabel("market period")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(t, cum_fric)
    axes[1].set_title("Cumulative friction loss")
    axes[1].set_xlabel("market period")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    figs["cumulative"] = fig

    # 5) Worker-side Gantt of who-matched-whom
    figs["gantt_worker"] = make_gantt_figure(history, env_config, side="worker")

    return figs


def save_cumulative_csvs(history: Dict[str, np.ndarray], out_dir) -> Dict[str, Path]:
    """Save cumulative regret/welfare/friction series as two CSVs.

    Files written under ``out_dir``:
    * ``cumulative_regret_welfare.csv`` — columns
      ``t, cum_regret_w, cum_regret_f, cum_social_welfare``
    * ``cumulative_friction.csv`` — columns ``t, cum_friction_loss``

    Returns a dict mapping logical name → written path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t = np.asarray(history["t"])
    cum_reg_w = np.cumsum(history["regret_w_total"])
    cum_reg_f = np.cumsum(history["regret_f_total"])
    cum_welfare = np.cumsum(history["social_welfare"])
    cum_friction = np.cumsum(history["friction_loss"])

    paths: Dict[str, Path] = {}

    rw_path = out_dir / "cumulative_regret_welfare.csv"
    with rw_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "cum_regret_w", "cum_regret_f", "cum_social_welfare"])
        for i in range(len(t)):
            w.writerow([t[i], cum_reg_w[i], cum_reg_f[i], cum_welfare[i]])
    paths["regret_welfare"] = rw_path

    fric_path = out_dir / "cumulative_friction.csv"
    with fric_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "cum_friction_loss"])
        for i in range(len(t)):
            w.writerow([t[i], cum_friction[i]])
    paths["friction"] = fric_path

    return paths


def save_batched_cumulative_csvs(history: Dict[str, np.ndarray], out_dir) -> Dict[str, Path]:
    """Batched (N seeds) counterpart to ``save_cumulative_csvs``.

    ``history`` follows the shape contract of
    ``plot_batched_ca_etc_baseline.make_batched_figures``: scalar per-period
    metrics are ``(N, T)`` arrays. For each cumulative quantity we report
    seed-aggregated mean / median / ci_lo / ci_hi at every market period —
    ci_lo, ci_hi define a 95% normal-approx CI band on the seed-mean
    (mean +/- 1.96 * std / sqrt(N)); median is reported separately so any
    skew (mean diverging from median) is visible in the CSV without needing
    to re-aggregate.

    Files written under ``out_dir``:
    * ``cumulative_regret_welfare.csv`` — columns
      ``t`` plus ``{stat}`` ∈ ``{mean, median, ci_lo, ci_hi}`` for each of
      ``cum_regret_w``, ``cum_regret_f``, ``cum_social_welfare``.
    * ``cumulative_friction.csv`` — columns
      ``t`` plus ``{stat}`` for ``cum_friction_loss``.

    Returns a dict mapping logical name → written path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t = np.asarray(history["t"])
    cum_reg_w = np.cumsum(history["regret_w_total"], axis=1)   # (N, T)
    cum_reg_f = np.cumsum(history["regret_f_total"], axis=1)
    cum_welfare = np.cumsum(history["social_welfare"], axis=1)
    cum_friction = np.cumsum(history["friction_loss"], axis=1)

    def _stats(arr_NT: np.ndarray) -> Dict[str, np.ndarray]:
        # 95% normal-approx CI on the seed-mean: mean +/- 1.96 * SEM.
        n = arr_NT.shape[0]
        m = arr_NT.mean(axis=0)
        sem = arr_NT.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(m)
        half = 1.96 * sem
        return {
            "mean":   m,
            "median": np.median(arr_NT, axis=0),
            "ci_lo":  m - half,
            "ci_hi":  m + half,
        }

    paths: Dict[str, Path] = {}

    rw_path = out_dir / "cumulative_regret_welfare.csv"
    series = [
        ("cum_regret_w",       _stats(cum_reg_w)),
        ("cum_regret_f",       _stats(cum_reg_f)),
        ("cum_social_welfare", _stats(cum_welfare)),
    ]
    # CA-ETC additionally reports committed-gauge regret (state.matched only,
    # matching PPO's gauge). Appended as extra columns when present so
    # compare_ppo_vs_caetc.py can pick gauge at plot time. PPO doesn't write
    # these because its single ``regret_*_total`` is already committed-gauge.
    if "regret_w_total_committed" in history:
        cum_reg_w_committed = np.cumsum(
            np.asarray(history["regret_w_total_committed"]), axis=1
        )
        series.append(("cum_regret_w_committed", _stats(cum_reg_w_committed)))
    if "regret_f_total_committed" in history:
        cum_reg_f_committed = np.cumsum(
            np.asarray(history["regret_f_total_committed"]), axis=1
        )
        series.append(("cum_regret_f_committed", _stats(cum_reg_f_committed)))
    header = ["t"] + [
        f"{name}_{stat}" for name, _ in series for stat in ("mean", "median", "ci_lo", "ci_hi")
    ]
    with rw_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(t)):
            row = [t[i]]
            for _, s in series:
                row.extend([s["mean"][i], s["median"][i], s["ci_lo"][i], s["ci_hi"][i]])
            w.writerow(row)
    paths["regret_welfare"] = rw_path

    fric_path = out_dir / "cumulative_friction.csv"
    fric_stats = _stats(cum_friction)
    with fric_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t",
            "cum_friction_loss_mean",  "cum_friction_loss_median",
            "cum_friction_loss_ci_lo", "cum_friction_loss_ci_hi",
        ])
        for i in range(len(t)):
            w.writerow([
                t[i],
                fric_stats["mean"][i],  fric_stats["median"][i],
                fric_stats["ci_lo"][i], fric_stats["ci_hi"][i],
            ])
    paths["friction"] = fric_path

    pp_fric_path = out_dir / "friction_loss.csv"
    pp_fric_stats = _stats(np.asarray(history["friction_loss"]))
    with pp_fric_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t",
            "friction_loss_mean",  "friction_loss_median",
            "friction_loss_ci_lo", "friction_loss_ci_hi",
        ])
        for i in range(len(t)):
            w.writerow([
                t[i],
                pp_fric_stats["mean"][i],  pp_fric_stats["median"][i],
                pp_fric_stats["ci_lo"][i], pp_fric_stats["ci_hi"][i],
            ])
    paths["friction_per_period"] = pp_fric_path

    # Per-period social welfare with optional DA-ref / planner upper bounds.
    # Used by learn2match/examples/compare_ppo_vs_caetc.py for the 4-line overlay.
    sw = np.asarray(history["social_welfare"])                 # (N, T)
    sw_path = out_dir / "social_welfare.csv"
    sw_series: list = [("actual", _stats(sw))]
    # CA-ETC writes a second SW gauge ("matched-only", i.e. post-RETENTION
    # state.matched -- the same gauge PPO eval uses). Stored as ``committed_*``
    # columns so compare_ppo_vs_caetc.py can pick it via --sw_gauge committed
    # for an apples-to-apples overlay against PPO. PPO eval doesn't write this
    # column because its single ``social_welfare`` value already IS the
    # committed gauge.
    if "social_welfare_committed" in history:
        committed = np.asarray(history["social_welfare_committed"])
        if committed.shape == sw.shape:
            sw_series.append(("committed", _stats(committed)))
    # DA ref via identity SW_da_ref = social_welfare + regret_w + regret_f
    # (mathematically equivalent to direct sum(ref_f * (U_true + U_true)) when
    # both runs compute regret against the same ref_f; see derivation in chat).
    if {"regret_w_total", "regret_f_total"}.issubset(history):
        rw = np.asarray(history["regret_w_total"])
        rf = np.asarray(history["regret_f_total"])
        sw_series.append(("da_ref", _stats(sw + rw + rf)))
    # Planner: PPO produces (N,), CA-ETC produces (N, T) -- broadcast to (N, T).
    if "social_welfare_planner" in history:
        planner = np.asarray(history["social_welfare_planner"])
        if planner.ndim == 1:
            planner = np.broadcast_to(planner[:, None], sw.shape).copy()
        if planner.shape == sw.shape:
            sw_series.append(("planner", _stats(planner)))
    sw_header = ["t"] + [
        f"{name}_{stat}" for name, _ in sw_series
        for stat in ("mean", "median", "ci_lo", "ci_hi")
    ]
    with sw_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(sw_header)
        for i in range(len(t)):
            row = [t[i]]
            for _, s in sw_series:
                row.extend([s["mean"][i], s["median"][i], s["ci_lo"][i], s["ci_hi"][i]])
            w.writerow(row)
    paths["social_welfare_per_period"] = sw_path

    # Cumulative-over-time twin of social_welfare.csv. Stats are computed on
    # cumsum'd per-seed traces (NOT cumsum of per-period stats) so the CI band
    # reflects the actual cross-seed variability of the cumulative series.
    cum_sw_path = out_dir / "cumulative_social_welfare.csv"
    cum_series: list = [("cum_actual", _stats(np.cumsum(sw, axis=1)))]
    if "social_welfare_committed" in history:
        committed = np.asarray(history["social_welfare_committed"])
        if committed.shape == sw.shape:
            cum_series.append(("cum_committed", _stats(np.cumsum(committed, axis=1))))
    if {"regret_w_total", "regret_f_total"}.issubset(history):
        rw = np.asarray(history["regret_w_total"])
        rf = np.asarray(history["regret_f_total"])
        cum_series.append(("cum_da_ref", _stats(np.cumsum(sw + rw + rf, axis=1))))
    if "social_welfare_planner" in history:
        planner = np.asarray(history["social_welfare_planner"])
        if planner.ndim == 1:
            planner = np.broadcast_to(planner[:, None], sw.shape).copy()
        if planner.shape == sw.shape:
            cum_series.append(("cum_planner", _stats(np.cumsum(planner, axis=1))))
    cum_sw_header = ["t"] + [
        f"{name}_{stat}" for name, _ in cum_series
        for stat in ("mean", "median", "ci_lo", "ci_hi")
    ]
    with cum_sw_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cum_sw_header)
        for i in range(len(t)):
            row = [t[i]]
            for _, s in cum_series:
                row.extend([s["mean"][i], s["median"][i], s["ci_lo"][i], s["ci_hi"][i]])
            w.writerow(row)
    paths["cumulative_social_welfare"] = cum_sw_path

    return paths


def _band(ax, t, mean, lo, hi, label, color=None):
    line, = ax.plot(t, mean, label=label, color=color, linewidth=1.6)
    ax.fill_between(t, lo, hi, color=line.get_color(), alpha=0.15)
    return line


def make_batched_figures(history: Dict[str, np.ndarray], env_config) -> Dict[str, plt.Figure]:
    """Build the figure suite for a (N, T, ...) batched history.

    Each panel shows the per-period mean across seeds with a shaded 95%
    normal-approx CI band on the seed-mean (mean +/- 1.96 * std / sqrt(N)).
    Returned keys: ``aggregate``, ``aux/social_welfare``, ``aux/friction_loss``,
    ``aux/match_rate``, ``cumulative_worker_regret``, ``cumulative_firm_regret``,
    ``cumulative_friction_loss``, ``cumulative_per_worker``,
    ``cumulative_per_firm``, ``per_seed_regret``, ``final_distribution``,
    ``gantt_seed0``, optional ``coverage_heatmap``.

    ``history`` must contain scalar metadata ``num_seeds``, ``Nw``, ``Nf``
    so the figure code does not need to peek into ``env_config`` for shapes.
    """
    figs: Dict[str, plt.Figure] = {}
    t = history["t"]                                                   # (T,)
    N = int(history["num_seeds"])
    Nw = int(history["Nw"])
    Nf = int(history["Nf"])

    def stats(arr):
        # arr: (N, T) -> (mean, ci_lo, ci_hi); 95% normal-approx CI on the seed-mean.
        m = arr.mean(axis=0)
        sem = arr.std(axis=0, ddof=1) / np.sqrt(N) if N > 1 else np.zeros_like(m)
        half = 1.96 * sem
        return m, m - half, m + half

    # 1) Aggregate per-period: reward / regret / info-loss with workers vs firms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in [
        (axes[0], "reward",    "Total reward / period"),
        (axes[1], "regret",    "Total regret / period"),
        (axes[2], "info_loss", "Total info-loss / period"),
    ]:
        for side, prefix, color in [("workers", "w", None), ("firms", "f", None)]:
            m, lo, hi = stats(history[f"{key}_{prefix}_total"])
            _band(ax, t, m, lo, hi, label=side, color=color)
        ax.set_title(f"{title}  (N={N} seeds, mean ± 95% CI)")
        ax.set_xlabel("market period")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    figs["aggregate"] = fig

    # 2) Aux: one figure per metric (social welfare, friction loss, match rate).
    # The social-welfare figure additionally overlays the worker-proposing DA
    # reference (green dashed) and the first-best planner welfare (red solid)
    # so the gap between the policy and these two upper-bound references is
    # directly visible. Slash-keyed so the saving loop nests them under
    # ``aux/`` locally and ``plot/aux/...`` in W&B.
    aux_specs = [
        ("aux/social_welfare", "social_welfare", "Social welfare"),
        ("aux/friction_loss",  "friction_loss",  "Friction loss"),
        ("aux/match_rate",     "match_rate",     "Match rate"),
    ]
    for slug, key, title in aux_specs:
        fig, ax = plt.subplots(figsize=(7, 4))
        m, lo, hi = stats(history[key])
        _band(ax, t, m, lo, hi, label="seed-mean (belief)")
        if key == "social_welfare":
            if "true_matched_welfare" in history:
                m_t, lo_t, hi_t = stats(history["true_matched_welfare"])
                _band(ax, t, m_t, lo_t, hi_t,
                      label="seed-mean (true x,y)", color="C1")
            if "social_welfare_da_ref" in history:
                per_seed_const = np.max(history["social_welfare_da_ref"], axis=1)
                ax.axhline(float(np.mean(per_seed_const)), ls="--", color="g",
                           label="DA ref mean")
            if "social_welfare_planner" in history:
                ax.axhline(float(np.mean(history["social_welfare_planner"])),
                           ls="-", color="r",
                           label="planner mean")
        ax.set_title(title)
        ax.set_xlabel("market period")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        figs[slug] = fig

    # 3) Cumulative regret (workers / firms / friction) -- one figure each so
    # they can be uploaded individually to W&B.
    cum_reg_w = np.cumsum(history["regret_w_total"], axis=1)            # (N, T)
    cum_reg_f = np.cumsum(history["regret_f_total"], axis=1)
    cum_fric  = np.cumsum(history["friction_loss"], axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    m, lo, hi = stats(cum_reg_w)
    _band(ax, t, m, lo, hi, label="workers", color="C0")
    ax.set_title("Cumulative worker regret (mean ± 95% CI)")
    ax.set_xlabel("market period")
    ax.set_ylabel("cumulative regret")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figs["cumulative_worker_regret"] = fig

    fig, ax = plt.subplots(figsize=(7, 4))
    m, lo, hi = stats(cum_reg_f)
    _band(ax, t, m, lo, hi, label="firms", color="C1")
    ax.set_title("Cumulative firm regret (mean ± 95% CI)")
    ax.set_xlabel("market period")
    ax.set_ylabel("cumulative regret")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figs["cumulative_firm_regret"] = fig

    fig, ax = plt.subplots(figsize=(7, 4))
    m, lo, hi = stats(cum_fric)
    _band(ax, t, m, lo, hi, label="friction", color="C2")
    ax.set_title("Cumulative friction loss (mean ± 95% CI)")
    ax.set_xlabel("market period")
    ax.set_ylabel("cumulative friction loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figs["cumulative_friction_loss"] = fig

    # 3b) Per-agent cumulative regret: one curve per worker / per firm,
    # averaged across seeds. Useful for spotting an outlier agent whose
    # individual regret dominates the aggregate.
    reg_w_per = history["regret_w_per_agent"]   # (N_seeds, T, Nw)
    reg_f_per = history["regret_f_per_agent"]   # (N_seeds, T, Nf)
    cum_per_w = np.cumsum(reg_w_per, axis=1)    # (N_seeds, T, Nw)
    cum_per_f = np.cumsum(reg_f_per, axis=1)    # (N_seeds, T, Nf)
    mean_cum_per_w = cum_per_w.mean(axis=0)     # (T, Nw)
    mean_cum_per_f = cum_per_f.mean(axis=0)     # (T, Nf)

    fig, ax = plt.subplots(figsize=(8, 4))
    cmap_w = plt.get_cmap("tab10", max(Nw, 10))
    for i in range(Nw):
        ax.plot(t, mean_cum_per_w[:, i], color=cmap_w(i % cmap_w.N),
                linewidth=1.4, label=f"worker {i}")
    ax.set_title("Cumulative per-worker regret (seed-mean)")
    ax.set_xlabel("market period")
    ax.set_ylabel("cumulative regret")
    ax.legend(loc="best", fontsize=8, ncol=min(Nw, 5))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figs["cumulative_per_worker"] = fig

    fig, ax = plt.subplots(figsize=(8, 4))
    cmap_f = plt.get_cmap("tab10", max(Nf, 10))
    for j in range(Nf):
        ax.plot(t, mean_cum_per_f[:, j], color=cmap_f(j % cmap_f.N),
                linewidth=1.4, label=f"firm {j}")
    ax.set_title("Cumulative per-firm regret (seed-mean)")
    ax.set_xlabel("market period")
    ax.set_ylabel("cumulative regret")
    ax.legend(loc="best", fontsize=8, ncol=min(Nf, 5))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figs["cumulative_per_firm"] = fig

    # 4) Per-seed cumulative regret w (thin lines + bold mean) -- variance vibes
    fig, ax = plt.subplots(figsize=(8, 4))
    for s in range(N):
        ax.plot(t, cum_reg_w[s], color="C0", alpha=min(0.3, 6.0 / max(N, 1)), linewidth=0.7)
    ax.plot(t, cum_reg_w.mean(axis=0), color="C0", linewidth=2.2, label=f"mean over {N} seeds")
    ax.set_title("Cumulative worker regret -- one curve per seed")
    ax.set_xlabel("market period")
    ax.set_ylabel("cumulative regret")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    figs["per_seed_regret"] = fig

    # 5) Final-period distribution: histograms of cum-regret and final friction
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    bin_count = min(20, max(5, N // 2))
    axes[0].hist(cum_reg_w[:, -1], bins=bin_count, color="C0", alpha=0.7)
    axes[0].set_title("Final cum-regret (workers)")
    axes[0].set_xlabel("cumulative regret")
    axes[1].hist(cum_reg_f[:, -1], bins=bin_count, color="C1", alpha=0.7)
    axes[1].set_title("Final cum-regret (firms)")
    axes[1].set_xlabel("cumulative regret")
    axes[2].hist(cum_fric[:, -1],  bins=bin_count, color="C2", alpha=0.7)
    axes[2].set_title("Final cum-friction-loss")
    axes[2].set_xlabel("cumulative friction loss")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    figs["final_distribution"] = fig

    # 6) Worker-side match Gantt for seed 0 only -- per-seed alias differences
    # make any cross-seed aggregation meaningless.
    matched_s0 = history["matched_matrix"][0]                          # (T, Nw, Nf)
    T = matched_s0.shape[0]
    partner = np.full((T, Nw), -1, dtype=np.int32)
    for ti in range(T):
        row_any = matched_s0[ti].any(axis=1)
        row_idx = matched_s0[ti].argmax(axis=1)
        partner[ti] = np.where(row_any, row_idx, -1)
    base_cmap = plt.get_cmap("tab20", max(Nf, 20))
    colors = np.zeros((Nf + 1, 4))
    colors[0] = (0.92, 0.92, 0.92, 1.0)
    for j in range(Nf):
        colors[j + 1] = base_cmap(j % base_cmap.N)
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    img = (partner + 1).T
    fig, ax = plt.subplots(figsize=(max(8, T * 0.10), max(3, Nw * 0.35)))
    ax.imshow(img, aspect="auto", cmap=cmap, vmin=0, vmax=Nf, interpolation="nearest",
              extent=[t[0] - 0.5, t[-1] + 0.5, Nw - 0.5, -0.5])
    ax.set_xlabel("market period")
    ax.set_ylabel("worker id")
    ax.set_title("Seed-0 worker-side match timeline")
    ax.set_yticks(range(Nw))
    if Nf <= 20:
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[0], label="unmatched")]
        handles += [plt.Rectangle((0, 0), 1, 1, color=colors[j + 1], label=f"firm {j}") for j in range(Nf)]
        ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.18, 1.0),
                  fontsize=7, ncol=1, frameon=False)
    fig.tight_layout()
    figs["gantt_seed0"] = fig

    # 7) Interview coverage + cumulative tenure heatmaps (final period, seed-mean).
    # Coverage panel: fraction of seeds where (i, j) was ever interviewed by the
    # end of the episode; cells far from 1.0 are pairs the policy systematically
    # leaves unobserved -- a direct cause of belief gaps and friction-loss floor.
    if "interviewed_matrix" in history and "cumulative_tenure" in history:
        interviewed_final = history["interviewed_matrix"][:, -1].astype(np.float32)  # (N, Nw, Nf)
        tenure_final = history["cumulative_tenure"][:, -1].astype(np.float32)        # (N, Nw, Nf)
        coverage_mean = interviewed_final.mean(axis=0)                                # (Nw, Nf)
        tenure_mean = tenure_final.mean(axis=0)                                       # (Nw, Nf)

        fig, axes = plt.subplots(1, 2, figsize=(11, max(3.5, 0.55 * Nw + 1.5)))

        im0 = axes[0].imshow(coverage_mean, cmap="viridis", vmin=0.0, vmax=1.0,
                             aspect="auto", interpolation="nearest")
        axes[0].set_title(f"Interview coverage (seed-mean, final period, N={N})")
        axes[0].set_xlabel("firm id")
        axes[0].set_ylabel("worker id")
        axes[0].set_xticks(range(Nf))
        axes[0].set_yticks(range(Nw))
        fig.colorbar(im0, ax=axes[0], label="frac of seeds with interviewed[i,j]")

        im1 = axes[1].imshow(tenure_mean, cmap="magma", aspect="auto", interpolation="nearest")
        axes[1].set_title(f"Cumulative tenure (seed-mean, final period, N={N})")
        axes[1].set_xlabel("firm id")
        axes[1].set_ylabel("worker id")
        axes[1].set_xticks(range(Nf))
        axes[1].set_yticks(range(Nw))
        fig.colorbar(im1, ax=axes[1], label="mean periods (i,j) was matched")

        if max(Nw, Nf) <= 12:
            for i in range(Nw):
                for j in range(Nf):
                    cov = coverage_mean[i, j]
                    axes[0].text(j, i, f"{cov:.2f}", ha="center", va="center",
                                 color="white" if cov < 0.5 else "black", fontsize=8)
                    ten = tenure_mean[i, j]
                    tnorm = ten / max(tenure_mean.max(), 1e-9)
                    axes[1].text(j, i, f"{ten:.1f}", ha="center", va="center",
                                 color="white" if tnorm < 0.5 else "black", fontsize=8)
        fig.tight_layout()
        figs["coverage_heatmap"] = fig

    return figs


def make_gantt_figure(history: Dict[str, np.ndarray], env_config, side: str = "worker"):
    """Worker-side (or firm-side) Gantt of bipartite matches over time.

    Each row is a worker (or firm); each cell is colored by the partner id at
    that period, or shown as background gray if unmatched. ``tab20`` is cycled
    when the partner side has more than 20 entries.
    """
    matched = history["matched_matrix"]   # (T, Nw, Nf) bool
    t = history["t"]
    T = matched.shape[0]
    Nw, Nf = env_config.Nw, env_config.Nf

    if side == "worker":
        rows, cols = Nw, Nf
        partner = np.full((T, Nw), -1, dtype=np.int32)
        for ti in range(T):
            row_any = matched[ti].any(axis=1)
            row_idx = matched[ti].argmax(axis=1)
            partner[ti] = np.where(row_any, row_idx, -1)
        ylabel, partner_label = "worker", "firm"
    else:
        rows, cols = Nf, Nw
        partner = np.full((T, Nf), -1, dtype=np.int32)
        for ti in range(T):
            col_any = matched[ti].any(axis=0)
            col_idx = matched[ti].argmax(axis=0)
            partner[ti] = np.where(col_any, col_idx, -1)
        ylabel, partner_label = "firm", "worker"

    base_cmap = plt.get_cmap("tab20", max(cols, 20))
    colors = np.zeros((cols + 1, 4))
    colors[0] = (0.92, 0.92, 0.92, 1.0)   # unmatched
    for j in range(cols):
        colors[j + 1] = base_cmap(j % base_cmap.N)
    cmap = plt.matplotlib.colors.ListedColormap(colors)

    img = (partner + 1).T    # (rows, T); +1 so unmatched=0
    fig, ax = plt.subplots(figsize=(max(8, T * 0.10), max(3, rows * 0.35)))
    ax.imshow(img, aspect="auto", cmap=cmap, vmin=0, vmax=cols, interpolation="nearest",
              extent=[t[0] - 0.5, t[-1] + 0.5, rows - 0.5, -0.5])
    ax.set_xlabel("market period")
    ax.set_ylabel(f"{ylabel} id")
    ax.set_title(f"{ylabel.capitalize()}-side match timeline ({partner_label} id encoded by color)")
    ax.set_yticks(range(rows))
    ax.grid(axis="x", which="both", color="white", linewidth=0.4, alpha=0.5)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[0], label="unmatched")]
    handles += [
        plt.Rectangle((0, 0), 1, 1, color=colors[j + 1], label=f"{partner_label} {j}")
        for j in range(cols)
    ]
    if cols <= 20:
        ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.18, 1.0),
                  fontsize=7, ncol=1, frameon=False)
    fig.tight_layout()
    return fig
