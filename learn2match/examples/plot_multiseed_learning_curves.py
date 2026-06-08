"""Aggregate in-training PPO learning curves across multi-seed train runs.

For each of ``--num_seeds`` PPO training runs, reads the npz dropped by
``train_hirerl_ippo.py:_save_learning_curve`` (the ``env_steps`` array and the
``(num_snapshots, num_eval_episodes, num_eval_periods)`` raw arrays for each
eval metric). For each snapshot we collapse a seed's eval block to one scalar
(mean over all episode/period entries -- same scalar that ``_run_eval`` uses
to populate the per-snapshot W&B point), then aggregate across seeds with
**mean ± 95% t-CI** (training variance: re-train the same recipe with a
different RNG -> how much does the per-period eval reward shift?).

Reads (per seed)
----------------
    {plots_dir}/{base_run_name}_s{i}_learning_curve.npz

Each npz keys: ``env_steps`` (S,) plus one ``(S, num_episodes, num_periods)``
ndarray per eval metric.

Writes (per metric)
-------------------
    {out_dir}/{slug}.pdf
    {out_dir}/{slug}.csv  -- columns: env_step, ppo_mean, ppo_std, ppo_ci_lo, ppo_ci_hi

Where ``slug`` is one of: ``worker_regret``, ``firm_regret``,
``social_welfare``, ``friction_loss``.

The CSV columns mirror the format produced by ``plot_multiseed_ci.py`` so the
two output sets can be combined / compared if needed.

Example
-------
    python learn2match/examples/plot_multiseed_learning_curves.py \\
        --plots_dir learn2match/examples/plots \\
        --base_run_name ppo_multiseed_may6 \\
        --num_seeds 10 \\
        --out_dir learn2match/graph/learning_curves

Dependencies: numpy, pandas, matplotlib (scipy optional, used for exact t-CI).
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

# (slug, npz_metric_key, ylabel, title) -- matches the 4 panels the user asked for.
METRICS = [
    ("worker_regret",  "regret_w_total", "worker regret (per period)",  "Worker Regret"),
    ("firm_regret",    "regret_f_total", "firm regret (per period)",    "Firm Regret"),
    ("social_welfare", "social_welfare", "social welfare (per period)", "Social Welfare"),
    ("friction_loss",  "friction_loss",  "friction loss (per period)",  "Friction Loss"),
]


def _t_critical_975(n: int) -> float:
    """Two-sided 95% t critical value for ``df = n - 1``. scipy if available,
    else hard-coded table for small n / 1.96 fallback for n >= 30."""
    if n < 2:
        return float("nan")
    try:
        from scipy.stats import t  # type: ignore
        return float(t.ppf(0.975, n - 1))
    except Exception:
        table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
                 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228,
                 12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 20: 2.093,
                 25: 2.060, 30: 2.045}
        if n in table:
            return table[n]
        if n >= 30:
            return 1.96
        return table[min(table, key=lambda k: abs(k - n))]


def _load_per_seed(
    plots_dir: Path,
    base_run_name: str,
    num_seeds: int,
    metric_keys: Tuple[str, ...],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[int]]:
    """Returns ``(env_steps, scalars_per_metric, used_seeds)``.

    ``scalars_per_metric[key]`` has shape ``(N_used, S)`` where each row is one
    seed's per-snapshot scalar (= ``arr.mean()`` over the eval (episodes, periods)
    block, the same scalar W&B logs).
    """
    rows: Dict[str, List[np.ndarray]] = {k: [] for k in metric_keys}
    used: List[int] = []
    env_steps_ref: Optional[np.ndarray] = None

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
                seed_scalars[key] = arr.reshape(arr.shape[0], -1).mean(axis=1)  # (S,)

            if not seed_ok:
                continue
            for key in metric_keys:
                rows[key].append(seed_scalars[key])
            used.append(i)

    if env_steps_ref is None or not used:
        raise SystemExit(
            f"No usable npz files found under {plots_dir} for base_run_name "
            f"'{base_run_name}' (seeds 0..{num_seeds-1}). Aborting."
        )

    stacked = {key: np.stack(rows[key], axis=0) for key in metric_keys}
    return env_steps_ref, stacked, used


def _aggregate_t_ci(arr_NT: np.ndarray) -> Dict[str, np.ndarray]:
    n = arr_NT.shape[0]
    mean = arr_NT.mean(axis=0)
    if n < 2:
        z = np.zeros_like(mean)
        return {"mean": mean, "std": z, "ci_lo": mean.copy(), "ci_hi": mean.copy()}
    std = arr_NT.std(axis=0, ddof=1)
    half = _t_critical_975(n) * std / math.sqrt(n)
    return {"mean": mean, "std": std, "ci_lo": mean - half, "ci_hi": mean + half}


def _plot_one(
    env_steps: np.ndarray,
    stats: Dict[str, np.ndarray],
    n_used: int,
    title: str,
    ylabel: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    line, = ax.plot(
        env_steps, stats["mean"],
        label=f"Learn2Match (Mean ± 95% t-CI across {n_used} train seeds)",
        color="C1", linewidth=1.8,
    )
    ax.fill_between(
        env_steps, stats["ci_lo"], stats["ci_hi"],
        color=line.get_color(), alpha=0.18,
    )
    ax.set_title(title)
    ax.set_xlabel("env steps")
    ax.set_ylabel(ylabel)
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
                        help="Used to build the npz filenames; e.g. 'ppo_multiseed_may6'.")
    parser.add_argument("--num_seeds", type=int, default=10,
                        help="Number of seeds (looks for s0..s{N-1}).")
    parser.add_argument("--out_dir", type=Path, required=True,
                        help="Where to write the 4 PDF/PNG/CSV triples.")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Skip W&B upload; just save files locally.")
    parser.add_argument("--wandb_entity", default="haijingzong-university-of-washington")
    parser.add_argument("--wandb_project", default="hireRL")
    parser.add_argument("--wandb_group", default=None,
                        help="W&B group; default = base_run_name so the aggregated "
                             "learning curves sit alongside the per-seed train runs.")
    parser.add_argument("--run_name", default=None,
                        help="W&B run name; defaults to multiseed_lc_<base>_<ts>.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    metric_keys = tuple(m[1] for m in METRICS)
    env_steps, stacked, used = _load_per_seed(
        args.plots_dir, args.base_run_name, args.num_seeds, metric_keys,
    )
    print(f"\nAggregating {len(used)} seeds {used}, "
          f"S={len(env_steps)} snapshots from env_step={env_steps[0]} to {env_steps[-1]}")

    for slug, key, ylabel, title in METRICS:
        arr_NT = stacked[key]
        stats = _aggregate_t_ci(arr_NT)

        fig = _plot_one(env_steps, stats, n_used=len(used), title=title, ylabel=ylabel)
        # PDF for paper, PNG for W&B inline preview (wandb.Image needs raster).
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

    print(f"\nAll learning-curve outputs written to {args.out_dir}")

    if args.no_wandb:
        return
    try:
        import wandb
    except ImportError:
        print("  [warn] wandb not installed; skipping upload")
        return
    run_name = args.run_name or (
        f"multiseed_lc_{args.base_run_name}_{datetime.now():%Y%m%d-%H%M%S}"
    )
    group = args.wandb_group or args.base_run_name
    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        group=group,
        name=run_name,
        job_type="aggregate_learning_curves",
        config={
            "kind": "multiseed_learning_curves",
            "plots_dir":         str(args.plots_dir),
            "base_run_name":     args.base_run_name,
            "num_seeds_requested": args.num_seeds,
            "num_seeds_used":    len(used),
            "snapshots":         int(len(env_steps)),
            "env_step_min":      int(env_steps[0]),
            "env_step_max":      int(env_steps[-1]),
        },
    )
    for slug, _, _, _ in METRICS:
        png = args.out_dir / f"{slug}.png"
        if png.is_file():
            wandb.log({f"plot/{slug}": wandb.Image(str(png))})
            print(f"  uploaded image plot/{slug} <- {png.name}")
    # Upload PDFs and CSVs as W&B Files (downloadable from the run's Files tab).
    for ext in ("pdf", "csv", "png"):
        for p in sorted(args.out_dir.glob(f"*.{ext}")):
            wandb.save(str(p), base_path=str(args.out_dir), policy="now")
    wandb.finish()
    print("W&B run finished.")


if __name__ == "__main__":
    main()
