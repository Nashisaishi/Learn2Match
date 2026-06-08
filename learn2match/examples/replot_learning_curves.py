"""Replot multi-seed learning curves from saved npz, with smoothing options.

Reads the same npz files as ``plot_multiseed_learning_curves.py`` -- written
by ``train_hirerl_ippo.py:_save_learning_curve`` and containing the FULL
``(num_snapshots, num_episodes, num_periods)`` raw eval block per metric --
so you can re-aggregate any way you want without retraining.

Two extra knobs vs. the original plot script:

* ``--cumulative`` -- collapse each (episode, snapshot) eval block by SUMMING
  over the period axis (then averaging over episodes), yielding "cumulative
  metric over N eval periods" instead of "per-period mean". y-axis numbers
  get ~100x larger so the *relative* visual noise drops -- this matches the
  legacy 100x100 plot style ("cumulative worker regret over 600 periods").
* ``--smooth_span S`` -- exponential weighted moving average across snapshots,
  applied per-seed BEFORE the across-seed aggregation. Effective alpha =
  ``2/(S+1)``. Default 2 (gentle); set 1 or 0 to disable. With only ~10
  snapshots, span 2-3 is usually plenty.

Both knobs are orthogonal; combine freely.

Reads (per seed)
----------------
    {plots_dir}/{base_run_name}_s{i}_learning_curve.npz

Each npz keys: ``env_steps`` (S,) plus one ``(S, num_episodes, num_periods)``
ndarray per eval metric.

Writes (per metric, into ``--out_dir``)
---------------------------------------
    {slug}.pdf, {slug}.png  -- figure
    {slug}.csv              -- columns: env_step, ppo_mean, ppo_std,
                               ppo_ci_lo, ppo_ci_hi

W&B upload
----------
Each replot is uploaded as its own W&B run named
``multiseed_lc_replot_{cum|perperiod}_{smoothN|raw}_{base_run_name}_<ts>``
so different cumulative / smooth_span variants off the same training appear
as distinct rows in the UI instead of overwriting each other. Pass
``--no_wandb`` to skip the upload.

Example
-------
    python learn2match/examples/replot_learning_curves.py \\
        --plots_dir learn2match/examples/plots \\
        --base_run_name ppo_500x500 \\
        --num_seeds 5 \\
        --out_dir learn2match/graph/learning_curves/500x500_replot \\
        --cumulative --smooth_span 2
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size":         15,
    "axes.titlesize":    18,
    "axes.labelsize":    15,
    "xtick.labelsize":   13,
    "ytick.labelsize":   13,
    "legend.fontsize":   13,
})

# (slug, npz_metric_key, title)
METRICS = [
    ("worker_regret",  "regret_w_total", "Worker Regret"),
    ("firm_regret",    "regret_f_total", "Firm Regret"),
    ("social_welfare", "social_welfare", "Social Welfare"),
    ("friction_loss",  "friction_loss",  "Friction Loss"),
]


def _t_critical_975(n: int) -> float:
    """Two-sided 95% t critical value for ``df = n - 1``."""
    if n < 2:
        return float("nan")
    try:
        from scipy.stats import t  # type: ignore
        return float(t.ppf(0.975, n - 1))
    except Exception:
        table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
                 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
        return table.get(n, 1.96)


def _ewma(x: np.ndarray, span: int) -> np.ndarray:
    """Adjusted EWMA matching ``pandas.Series.ewm(span=span, adjust=True).mean()``.

    Adjusted form (rather than recursive) so the first few snapshots aren't
    pinned to ``x[0]`` -- gives a more honest smoothed value at the curve's
    left edge.
    """
    if span is None or span <= 1:
        return x.astype(float, copy=True)
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(x, dtype=float)
    weight_sum = 0.0
    value_sum = 0.0
    for i in range(x.shape[0]):
        weight_sum = 1.0 + (1.0 - alpha) * weight_sum
        value_sum = float(x[i]) + (1.0 - alpha) * value_sum
        out[i] = value_sum / weight_sum
    return out


def _load_per_seed(
    plots_dir: Path,
    base_run_name: str,
    num_seeds: int,
    metric_keys: Tuple[str, ...],
    cumulative: bool,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[int], int]:
    """Returns ``(env_steps, scalars_per_metric, used_seeds, num_periods)``.

    ``scalars_per_metric[key]`` has shape ``(N_used, S)``. Per (seed, snapshot)
    we collapse the eval block ``arr[s] : (num_episodes, num_periods)`` to one
    scalar:
        * cumulative=False -> ``arr[s].mean()``      (matches plot_multiseed_learning_curves)
        * cumulative=True  -> ``arr[s].sum(-1).mean()``  (cumulative over an episode)
    """
    rows: Dict[str, List[np.ndarray]] = {k: [] for k in metric_keys}
    used: List[int] = []
    env_steps_ref: Optional[np.ndarray] = None
    num_periods: int = -1

    print("Reading npz files:")
    for i in range(num_seeds):
        path = plots_dir / f"{base_run_name}_s{i}_learning_curve.npz"
        if not path.is_file():
            print(f"  [warn] seed {i}: missing {path}")
            continue
        print(f"  seed{i}: {path}")
        with np.load(path) as npz:
            steps = np.asarray(npz["env_steps"])
            if env_steps_ref is None:
                env_steps_ref = steps
            elif steps.shape != env_steps_ref.shape or not np.array_equal(steps, env_steps_ref):
                print(f"  [warn] seed{i}: env_steps mismatch vs seed{used[0]}; skipping")
                continue

            seed_ok = True
            seed_scalars: Dict[str, np.ndarray] = {}
            for key in metric_keys:
                if key not in npz.files:
                    print(f"  [warn] seed{i}: missing metric '{key}' in npz; skipping seed")
                    seed_ok = False
                    break
                arr = np.asarray(npz[key])  # (S, num_episodes, num_periods)
                if arr.ndim != 3:
                    print(f"  [warn] seed{i}: '{key}' has unexpected shape {arr.shape}; skipping")
                    seed_ok = False
                    break
                if num_periods < 0:
                    num_periods = int(arr.shape[-1])
                if cumulative:
                    # Cumulative over an eval episode: sum over periods, then
                    # mean over the few eval episodes.
                    seed_scalars[key] = arr.sum(axis=-1).mean(axis=-1)  # (S,)
                else:
                    seed_scalars[key] = arr.reshape(arr.shape[0], -1).mean(axis=1)  # (S,)

            if not seed_ok:
                continue
            for key in metric_keys:
                rows[key].append(seed_scalars[key])
            used.append(i)

    if env_steps_ref is None or not used:
        raise SystemExit(
            f"No usable npz files under {plots_dir} for base_run_name "
            f"'{base_run_name}' (seeds 0..{num_seeds-1}). Aborting."
        )

    stacked = {key: np.stack(rows[key], axis=0) for key in metric_keys}
    return env_steps_ref, stacked, used, num_periods


def _aggregate_t_ci(arr_NT: np.ndarray, smooth_span: int) -> Dict[str, np.ndarray]:
    """Smooth each seed's curve, then aggregate across seeds with t-CI.

    Smoothing each seed BEFORE the across-seed aggregation (instead of
    smoothing the mean line) gives a coherent CI band that contracts in lock
    step with the smoothed mean -- otherwise the band looks wider than the
    line wiggles, which is visually confusing.
    """
    n = arr_NT.shape[0]
    if smooth_span and smooth_span > 1:
        smoothed = np.stack([_ewma(arr_NT[i], smooth_span) for i in range(n)], axis=0)
    else:
        smoothed = arr_NT.astype(float)
    mean = smoothed.mean(axis=0)
    if n < 2:
        z = np.zeros_like(mean)
        return {"mean": mean, "std": z, "ci_lo": mean.copy(), "ci_hi": mean.copy()}
    std = smoothed.std(axis=0, ddof=1)
    half = _t_critical_975(n) * std / math.sqrt(n)
    return {"mean": mean, "std": std, "ci_lo": mean - half, "ci_hi": mean + half}


def _ylabel_for(slug: str, cumulative: bool, num_periods: int) -> str:
    pretty = slug.replace("_", " ")
    if cumulative:
        return f"{pretty} (cumulative over {num_periods} periods)"
    return f"{pretty} (per period)"


def _plot_one(
    env_steps: np.ndarray,
    stats: Dict[str, np.ndarray],
    n_used: int,
    title: str,
    ylabel: str,
    smooth_span: int,
    ylim_from_zero: bool,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    extra = f", EWMA span={smooth_span}" if smooth_span and smooth_span > 1 else ""
    line, = ax.plot(
        env_steps, stats["mean"],
        label=f"Learn2Match (Mean ± 95% t-CI across {n_used} train seeds{extra})",
        color="C1", linewidth=1.8,
    )
    ax.fill_between(
        env_steps, stats["ci_lo"], stats["ci_hi"],
        color=line.get_color(), alpha=0.18,
    )
    ax.set_title(title)
    ax.set_xlabel("env steps")
    ax.set_ylabel(ylabel)
    if ylim_from_zero:
        # Only force lower bound; keep matplotlib's auto upper.
        cur_lo, cur_hi = ax.get_ylim()
        ax.set_ylim(min(0.0, cur_lo), cur_hi)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--plots_dir", type=Path, required=True,
                        help="Directory containing {base_run_name}_s{i}_learning_curve.npz.")
    parser.add_argument("--base_run_name", type=str, required=True,
                        help="e.g. 'ppo_500x500' -- npz files expected as "
                             "'{base_run_name}_s{i}_learning_curve.npz'.")
    parser.add_argument("--num_seeds", type=int, default=5,
                        help="Number of train seeds (looks for s0..s{N-1}).")
    parser.add_argument("--out_dir", type=Path, required=True,
                        help="Where to write the per-metric PDF/PNG/CSV.")
    parser.add_argument("--cumulative", action="store_true",
                        help="Plot cumulative-over-eval-window instead of per-period mean.")
    parser.add_argument("--smooth_span", type=int, default=2,
                        help="EWMA span across snapshots, applied per seed before "
                             "across-seed aggregation. 0 or 1 disables. Default 2.")
    parser.add_argument("--ylim_from_zero", action="store_true",
                        help="Force y-axis lower bound to min(0, current). Visually "
                             "suppresses zoomed-in noise.")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Skip W&B upload; just write files locally.")
    parser.add_argument("--wandb_entity", default="haijingzong-university-of-washington")
    parser.add_argument("--wandb_project", default="hireRL")
    parser.add_argument("--wandb_group", default=None,
                        help="W&B group; default = base_run_name so the replots sit "
                             "alongside the original train + aggregated runs.")
    parser.add_argument("--run_name", default=None,
                        help="W&B run name; defaults to "
                             "multiseed_lc_replot_<mode>_<smooth>_<base>_<ts>.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    metric_keys = tuple(m[1] for m in METRICS)
    env_steps, stacked, used, num_periods = _load_per_seed(
        args.plots_dir, args.base_run_name, args.num_seeds, metric_keys,
        cumulative=args.cumulative,
    )
    mode = "cumulative-over-eval" if args.cumulative else "per-period"
    print(f"\nLoaded {len(used)} seeds {used}, "
          f"S={len(env_steps)} snapshots from env_step={env_steps[0]} to {env_steps[-1]}")
    print(f"Mode: {mode} | smooth_span = {args.smooth_span} | "
          f"eval window = {num_periods} periods")

    for slug, key, title in METRICS:
        arr_NT = stacked[key]
        stats = _aggregate_t_ci(arr_NT, args.smooth_span)
        ylabel = _ylabel_for(slug, args.cumulative, num_periods)

        fig = _plot_one(
            env_steps, stats, n_used=len(used),
            title=title, ylabel=ylabel,
            smooth_span=args.smooth_span,
            ylim_from_zero=args.ylim_from_zero,
        )
        pdf_path = args.out_dir / f"{slug}.pdf"
        png_path = args.out_dir / f"{slug}.png"
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        fig.savefig(png_path, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {pdf_path}")
        print(f"  saved {png_path}")

        csv_path = args.out_dir / f"{slug}.csv"
        pd.DataFrame({
            "env_step":  env_steps,
            "ppo_mean":  stats["mean"],
            "ppo_std":   stats["std"],
            "ppo_ci_lo": stats["ci_lo"],
            "ppo_ci_hi": stats["ci_hi"],
        }).to_csv(csv_path, index=False)
        print(f"  saved {csv_path}")

    print(f"\nAll outputs written to {args.out_dir}")

    if args.no_wandb:
        return
    try:
        import wandb
    except ImportError:
        print("  [warn] wandb not installed; skipping upload")
        return

    # Build a self-describing run_name so multiple replot variants
    # (different cumulative / smooth_span combos off the same training)
    # appear as distinct rows in the W&B UI instead of overwriting each other.
    smooth_tag = (
        f"smooth{args.smooth_span}"
        if args.smooth_span and args.smooth_span > 1 else "raw"
    )
    mode_tag = "cum" if args.cumulative else "perperiod"
    run_name = args.run_name or (
        f"multiseed_lc_replot_{mode_tag}_{smooth_tag}_"
        f"{args.base_run_name}_{datetime.now():%Y%m%d-%H%M%S}"
    )
    group = args.wandb_group or args.base_run_name

    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=group,
        name=run_name,
        job_type="replot_learning_curves",
        config={
            "kind":                "multiseed_learning_curves_replot",
            "plots_dir":           str(args.plots_dir),
            "base_run_name":       args.base_run_name,
            "num_seeds_requested": args.num_seeds,
            "num_seeds_used":      len(used),
            "snapshots":           int(len(env_steps)),
            "env_step_min":        int(env_steps[0]),
            "env_step_max":        int(env_steps[-1]),
            "cumulative":          bool(args.cumulative),
            "smooth_span":         int(args.smooth_span),
            "ylim_from_zero":      bool(args.ylim_from_zero),
            "eval_num_periods":    int(num_periods),
        },
    )
    for slug, _, _ in METRICS:
        png = args.out_dir / f"{slug}.png"
        if png.is_file():
            wandb.log({f"plot/{slug}": wandb.Image(str(png))})
            print(f"  uploaded image plot/{slug} <- {png.name}")
    # Also save PDFs/CSVs/PNGs as W&B Files (downloadable from the run's Files tab).
    for ext in ("pdf", "csv", "png"):
        for p in sorted(args.out_dir.glob(f"*.{ext}")):
            wandb.save(str(p), base_path=str(args.out_dir), policy="now")
    wandb.finish()
    print("W&B run finished.")


if __name__ == "__main__":
    main()
