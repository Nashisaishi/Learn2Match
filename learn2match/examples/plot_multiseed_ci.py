"""Aggregate multi-seed PPO eval into 4 plots with training-variance CI.

For each of ``num_seeds`` independently trained PPO policies, reads its eval
CSVs from ``{ppo_eval_root}/seed{i}/``. The eval CSVs already store a 32-env
mean (the ``*_mean`` columns from ``save_batched_cumulative_csvs``), so each
seed contributes ONE point per market period; the aggregation here is across
the ``num_seeds`` train seeds, producing **mean ± 95% t-CI** that reflects
**training variance** (re-train the same recipe with a different RNG ->
how much does the resulting policy's metric change?).

Optionally overlays the CA-ETC baseline as a **mean line only** (no CI band)
read from ``--ca_etc_dir``. CA-ETC has no training-time RNG (it's a fixed
algorithm), so its CI would mean something different (env sampling variance
across the 32 eval envs); plotting only the mean keeps the figure's CI band
unambiguously about PPO training stability.

Outputs
-------
``{out_dir}/`` contains:
* ``cumulative_worker_regret.png``  + ``.csv``
* ``cumulative_firm_regret.png``    + ``.csv``
* ``cumulative_friction_loss.png``  + ``.csv``
* ``cumulative_social_welfare.png`` + ``.csv``

Each CSV has columns ``t, ppo_mean, ppo_ci_lo, ppo_ci_hi[, caetc_mean]``.

Inputs (per PPO seed)
---------------------
* ``cumulative_regret_welfare.csv``  -> ``cum_regret_{w,f}_mean``
* ``cumulative_friction.csv``        -> ``cum_friction_loss_mean``
* ``cumulative_social_welfare.csv``  -> ``cum_actual_mean``  (PPO native gauge
                                          is already committed-side)

Inputs (CA-ETC, optional)
-------------------------
* same files. For SW, picks ``cum_committed_mean`` if ``--gauge committed``
  and the column is present, else ``cum_actual_mean``. For regret w/f, picks
  ``cum_regret_{w,f}_committed_mean`` if available under committed gauge.
  Friction is gauge-independent.

Example
-------
    python learn2match/examples/plot_multiseed_ci.py \\
        --ppo_eval_root /home/hisaishi/hireRL/eval_runs/ppo_multiseed_may6 \\
        --num_seeds 10 \\
        --ca_etc_dir /home/hisaishi/hireRL/eval_runs/ca_etc_may6_1 \\
        --gauge committed \\
        --out_dir /home/hisaishi/hireRL/eval_runs/ppo_multiseed_may6/aggregated
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _t_critical_975(n: int) -> float:
    """Two-sided 95% t critical value for ``df = n - 1``.

    Uses scipy if available; otherwise falls back to a hard-coded table for
    small n and the normal-approx 1.96 for n >= 30.
    """
    if n < 2:
        return float("nan")
    try:
        from scipy.stats import t  # type: ignore
        return float(t.ppf(0.975, n - 1))
    except Exception:
        table = {
            2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
            7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228,
            12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 20: 2.093,
            25: 2.060, 30: 2.045,
        }
        if n in table:
            return table[n]
        if n >= 30:
            return 1.96
        keys = sorted(table)
        nearest = min(keys, key=lambda k: abs(k - n))
        return table[nearest]


def _aggregate_t_ci(arr_NT: np.ndarray) -> Dict[str, np.ndarray]:
    """Mean +/- 95% t-CI across the leading (N) axis. Returns dict of (T,)."""
    n = arr_NT.shape[0]
    mean = arr_NT.mean(axis=0)
    if n < 2:
        zeros = np.zeros_like(mean)
        return {"mean": mean, "std": zeros, "ci_lo": mean.copy(), "ci_hi": mean.copy()}
    std = arr_NT.std(axis=0, ddof=1)
    sem = std / math.sqrt(n)
    half = _t_critical_975(n) * sem
    return {"mean": mean, "std": std, "ci_lo": mean - half, "ci_hi": mean + half}


def _read_seed_column(
    eval_root: Path,
    num_seeds: int,
    csv_name: str,
    column: str,
) -> Optional[Tuple[np.ndarray, np.ndarray, List[int]]]:
    """Stack ``column`` across seed{0..N-1}/csv_name into an (N, T) array.

    Returns ``(t, arr_NT, used_seeds)`` where ``used_seeds`` lists the seed
    indices that actually contributed (missing files are skipped with a
    warning). Returns ``None`` if no seed has the file/column.
    """
    rows: List[np.ndarray] = []
    used: List[int] = []
    t_ref: Optional[np.ndarray] = None
    for i in range(num_seeds):
        p = eval_root / f"seed{i}" / csv_name
        if not p.is_file():
            print(f"  [warn] seed {i}: missing {p}")
            continue
        df = pd.read_csv(p)
        if column not in df.columns:
            print(f"  [warn] seed {i}: {p} lacks column {column}")
            continue
        if t_ref is None:
            t_ref = df["t"].to_numpy()
        elif len(df) != len(t_ref):
            print(f"  [warn] seed {i}: {p} has T={len(df)} != {len(t_ref)}; skipping")
            continue
        rows.append(df[column].to_numpy())
        used.append(i)
    if not rows or t_ref is None:
        return None
    return t_ref, np.stack(rows, axis=0), used


def _read_ppo_refs(
    eval_root: Path,
    num_seeds: int,
    columns: Tuple[str, ...] = ("cum_da_ref_mean", "cum_planner_mean"),
    paired_check_atol: float = 1e-6,
) -> Dict[str, np.ndarray]:
    """Read policy-independent reference series (DA ref / planner) from a PPO
    seed's ``cumulative_social_welfare.csv``.

    These series depend only on ``(x, y)``, which is identical across all train
    seeds under paired-eval design (single shared ``--eval_seed`` for every
    train policy). So one seed's values are authoritative; we additionally
    cross-check against the next available seed and warn on max-abs diff above
    ``paired_check_atol`` -- a sentinel for accidentally unpaired eval.

    Returns a dict mapping requested column name -> (T,) array, plus key ``"t"``
    if any column was found.
    """
    primary: Optional[pd.DataFrame] = None
    primary_seed = -1
    out: Dict[str, np.ndarray] = {}
    for i in range(num_seeds):
        p = eval_root / f"seed{i}" / "cumulative_social_welfare.csv"
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        if primary is None:
            primary = df
            primary_seed = i
            out["t"] = df["t"].to_numpy()
            for col in columns:
                if col in df.columns:
                    out[col] = df[col].to_numpy()
                else:
                    print(f"  [warn] refs: column {col} absent in seed{i} CSV")
            continue
        for col in columns:
            if col in df.columns and col in out:
                if len(df) != len(out[col]):
                    continue
                diff = float(np.abs(df[col].to_numpy() - out[col]).max())
                if diff > paired_check_atol:
                    print(f"  [warn] refs: {col} differs between seed{primary_seed} "
                          f"and seed{i} (max |Δ|={diff:.4g}); paired eval may be off")
        break
    return out


def _read_caetc_mean(
    caetc_dir: Path,
    csv_name: str,
    column: str,
    fallback_column: Optional[str] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    p = caetc_dir / csv_name
    if not p.is_file():
        print(f"  [warn] CA-ETC: missing {p}")
        return None
    df = pd.read_csv(p)
    chosen = column
    if column not in df.columns:
        if fallback_column and fallback_column in df.columns:
            print(f"  [warn] CA-ETC: column {column} absent in {p}; "
                  f"falling back to {fallback_column}")
            chosen = fallback_column
        else:
            print(f"  [warn] CA-ETC: column {column} absent in {p}")
            return None
    return df["t"].to_numpy(), df[chosen].to_numpy(), chosen


def _plot_single_metric(
    t: np.ndarray,
    ppo_stats: Dict[str, np.ndarray],
    n_used: int,
    title: str,
    ylabel: str,
    ppo_label: str,
    caetc: Optional[Tuple[np.ndarray, np.ndarray, str]] = None,
    refs: Optional[List[Tuple[np.ndarray, np.ndarray, str, str]]] = None,
) -> plt.Figure:
    """``refs``: list of ``(t, values, label, color)`` reference series drawn as
    dashed lines without a CI band. Use for policy-independent quantities (DA
    ref, planner) whose cross-train-seed variance is exactly 0 in paired eval.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    line, = ax.plot(t, ppo_stats["mean"], label=ppo_label, color="C1", linewidth=1.6)
    ax.fill_between(
        t, ppo_stats["ci_lo"], ppo_stats["ci_hi"],
        color=line.get_color(), alpha=0.18,
        label=f"_{ppo_label} 95% CI",
    )
    if caetc is not None:
        t_cae, mean_cae, label_cae = caetc
        ax.plot(t_cae, mean_cae, label=label_cae, color="C0",
                linewidth=1.6, linestyle="--")
    if refs:
        for t_ref, vals_ref, label_ref, color_ref in refs:
            ax.plot(t_ref, vals_ref, label=label_ref, color=color_ref,
                    linewidth=1.4, linestyle=":")
    ax.set_title(
        f"{title}\n(PPO band = mean ± 95% t-CI across {n_used} train seeds)"
    )
    ax.set_xlabel("market period")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    return fig


def _save_metric_csv(
    path: Path,
    t: np.ndarray,
    ppo_stats: Dict[str, np.ndarray],
    caetc: Optional[Tuple[np.ndarray, np.ndarray, str]],
    refs: Optional[List[Tuple[np.ndarray, np.ndarray, str, str]]] = None,
) -> None:
    cols = {
        "t": t,
        "ppo_mean":  ppo_stats["mean"],
        "ppo_std":   ppo_stats["std"],
        "ppo_ci_lo": ppo_stats["ci_lo"],
        "ppo_ci_hi": ppo_stats["ci_hi"],
    }
    if caetc is not None:
        t_cae, mean_cae, _ = caetc
        if len(t_cae) == len(t) and np.allclose(t_cae, t):
            cols["caetc_mean"] = mean_cae
        else:
            df_cae = pd.DataFrame({"t": t_cae, "caetc_mean": mean_cae})
            df_ppo = pd.DataFrame(cols)
            for t_ref, vals_ref, label_ref, _ in (refs or []):
                slug = label_ref.lower().replace(" ", "_").replace("(", "").replace(")", "")
                if len(t_ref) == len(t) and np.allclose(t_ref, t):
                    df_ppo[f"{slug}_mean"] = vals_ref
                else:
                    df_ppo = df_ppo.merge(
                        pd.DataFrame({"t": t_ref, f"{slug}_mean": vals_ref}),
                        on="t", how="outer",
                    )
            pd.merge(df_ppo, df_cae, on="t", how="outer").sort_values("t").to_csv(
                path, index=False
            )
            return
    for t_ref, vals_ref, label_ref, _ in (refs or []):
        slug = label_ref.lower().replace(" ", "_").replace("(", "").replace(")", "")
        if len(t_ref) == len(t) and np.allclose(t_ref, t):
            cols[f"{slug}_mean"] = vals_ref
    pd.DataFrame(cols).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ppo_eval_root", required=True, type=Path,
                        help="Root containing seed0/, seed1/, ... each with "
                             "the eval CSVs from eval_and_plot.py.")
    parser.add_argument("--num_seeds", type=int, default=10,
                        help="Number of seed{i} subdirs to look for.")
    parser.add_argument("--out_dir", required=True, type=Path,
                        help="Where to write the 4 PNG/CSV pairs.")
    parser.add_argument("--ca_etc_dir", type=Path, default=None,
                        help="Optional CA-ETC eval out_dir. If set, overlay "
                             "its mean (no CI) as a comparison line.")
    parser.add_argument("--gauge", choices=["effective", "committed"],
                        default="committed",
                        help="CA-ETC gauge for SW + cumulative regret. PPO is "
                             "always committed-side; 'committed' gives "
                             "apples-to-apples mean comparison.")
    parser.add_argument("--ppo_label", default="PPO",
                        help="Legend label for the PPO band line.")
    parser.add_argument("--caetc_label", default="CA-ETC",
                        help="Legend label for the CA-ETC mean line.")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Skip W&B upload; just save PNGs/CSVs locally.")
    parser.add_argument("--wandb_entity", default="haijingzong-university-of-washington")
    parser.add_argument("--wandb_project", default="hireRL")
    parser.add_argument("--run_name", default=None,
                        help="W&B run name; defaults to "
                             "multiseed_<out_dir basename>_<ts>.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # PPO-side aggregation specs.
    # Each item: (logical_name, csv_filename, column_in_csv, ylabel, title)
    metrics = [
        (
            "cumulative_worker_regret",
            "cumulative_regret_welfare.csv", "cum_regret_w_mean",
            "cumulative worker regret", "Cumulative worker regret",
        ),
        (
            "cumulative_firm_regret",
            "cumulative_regret_welfare.csv", "cum_regret_f_mean",
            "cumulative firm regret", "Cumulative firm regret",
        ),
        (
            "cumulative_friction_loss",
            "cumulative_friction.csv", "cum_friction_loss_mean",
            "cumulative friction loss", "Cumulative friction loss",
        ),
        (
            "cumulative_social_welfare",
            "cumulative_social_welfare.csv", "cum_actual_mean",
            "cumulative social welfare", "Cumulative social welfare",
        ),
    ]

    # CA-ETC column choice per metric (gauge-aware).
    def caetc_column_for(logical_name: str) -> Tuple[str, str, Optional[str]]:
        """Returns (csv_filename, column, fallback_column)."""
        if logical_name == "cumulative_worker_regret":
            if args.gauge == "committed":
                return ("cumulative_regret_welfare.csv",
                        "cum_regret_w_committed_mean",
                        "cum_regret_w_mean")
            return ("cumulative_regret_welfare.csv", "cum_regret_w_mean", None)
        if logical_name == "cumulative_firm_regret":
            if args.gauge == "committed":
                return ("cumulative_regret_welfare.csv",
                        "cum_regret_f_committed_mean",
                        "cum_regret_f_mean")
            return ("cumulative_regret_welfare.csv", "cum_regret_f_mean", None)
        if logical_name == "cumulative_friction_loss":
            return ("cumulative_friction.csv", "cum_friction_loss_mean", None)
        if logical_name == "cumulative_social_welfare":
            if args.gauge == "committed":
                return ("cumulative_social_welfare.csv",
                        "cum_committed_mean",
                        "cum_actual_mean")
            return ("cumulative_social_welfare.csv", "cum_actual_mean", None)
        raise KeyError(logical_name)

    print("=" * 72)
    print(f"Multi-seed CI plot | ppo_eval_root={args.ppo_eval_root}")
    print(f"  num_seeds   = {args.num_seeds}")
    print(f"  ca_etc_dir  = {args.ca_etc_dir}")
    print(f"  gauge       = {args.gauge}")
    print(f"  out_dir     = {args.out_dir}")
    print("=" * 72)

    figs: Dict[str, plt.Figure] = {}

    for logical_name, csv_name, ppo_col, ylabel, title in metrics:
        print(f"\n[{logical_name}]")
        ppo_data = _read_seed_column(
            args.ppo_eval_root, args.num_seeds, csv_name, ppo_col,
        )
        if ppo_data is None:
            print(f"  [skip] no seed produced {csv_name}:{ppo_col}")
            continue
        t, arr_NT, used = ppo_data
        n_used = len(used)
        print(f"  PPO: aggregated across {n_used} seeds {used}")
        ppo_stats = _aggregate_t_ci(arr_NT)

        caetc: Optional[Tuple[np.ndarray, np.ndarray, str]] = None
        if args.ca_etc_dir is not None:
            cae_csv, cae_col, cae_fallback = caetc_column_for(logical_name)
            cae = _read_caetc_mean(args.ca_etc_dir, cae_csv, cae_col, cae_fallback)
            if cae is not None:
                t_cae, mean_cae, used_col = cae
                tag = (f"{args.caetc_label} (committed)"
                       if args.gauge == "committed" and "committed" in used_col
                       else args.caetc_label)
                caetc = (t_cae, mean_cae, tag)
                print(f"  CA-ETC: read {cae_csv}:{used_col}")

        # Policy-independent reference lines (DA ref + planner) only make
        # sense on the cumulative SW figure. Identical across all train seeds
        # under paired eval, so we plot a single line each (no CI band).
        refs: Optional[List[Tuple[np.ndarray, np.ndarray, str, str]]] = None
        if logical_name == "cumulative_social_welfare":
            ref_dict = _read_ppo_refs(args.ppo_eval_root, args.num_seeds)
            refs_list: List[Tuple[np.ndarray, np.ndarray, str, str]] = []
            t_ref = ref_dict.get("t", t)
            if "cum_da_ref_mean" in ref_dict:
                refs_list.append((t_ref, ref_dict["cum_da_ref_mean"], "DA ref", "C2"))
                print("  refs: read cum_da_ref_mean from PPO seed CSV")
            if "cum_planner_mean" in ref_dict:
                refs_list.append((t_ref, ref_dict["cum_planner_mean"],
                                  "planner (first-best)", "C3"))
                print("  refs: read cum_planner_mean from PPO seed CSV")
            if refs_list:
                refs = refs_list

        fig = _plot_single_metric(
            t=t, ppo_stats=ppo_stats, n_used=n_used,
            title=title, ylabel=ylabel,
            ppo_label=f"{args.ppo_label} (n={n_used})",
            caetc=caetc, refs=refs,
        )
        png_path = args.out_dir / f"{logical_name}.png"
        fig.savefig(png_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        figs[logical_name] = png_path
        print(f"  saved {png_path}")

        csv_path = args.out_dir / f"{logical_name}.csv"
        _save_metric_csv(csv_path, t, ppo_stats, caetc, refs=refs)
        print(f"  saved {csv_path}")

    print(f"\nAll multi-seed CI figures written to {args.out_dir}")

    if not args.no_wandb and figs:
        try:
            import wandb
        except ImportError:
            print("  [warn] wandb not installed; skipping upload")
            return
        run_name = args.run_name or (
            f"multiseed_{args.out_dir.name}_{datetime.now():%Y%m%d-%H%M%S}"
        )
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            config={
                "kind": "multiseed_ci",
                "ppo_eval_root": str(args.ppo_eval_root),
                "num_seeds_requested": args.num_seeds,
                "ca_etc_dir": str(args.ca_etc_dir) if args.ca_etc_dir else None,
                "gauge": args.gauge,
            },
        )
        for name, png_path in figs.items():
            wandb.log({f"plot/{name}": wandb.Image(str(png_path))})
            print(f"  uploaded {png_path.name} -> wandb")
        for csv_path in sorted(args.out_dir.glob("*.csv")):
            wandb.save(str(csv_path), base_path=str(args.out_dir), policy="now")
        wandb.finish()
        print("W&B run finished.")


if __name__ == "__main__":
    main()
