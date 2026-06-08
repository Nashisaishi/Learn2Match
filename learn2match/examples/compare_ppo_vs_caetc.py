"""Overlay comparison plots: PPO eval vs CA-ETC baseline.

Reads CSVs from each run's output directory and produces five PNGs:

* ``social_welfare.png``           -- four lines: CA-ETC actual SW + PPO
                                       actual SW + DA ref (worker-proposing
                                       on true U) + planner (first-best linear
                                       assignment), each with 95% CI band.
                                       References default to the CA-ETC run's
                                       per-period values; with matching seeds
                                       and ``--unified_seeds`` on CA-ETC, both
                                       runs see bit-equal (x, y), so refs from
                                       either run are identical.
* ``cumulative_social_welfare.png`` -- cumulative-over-time twin of the above;
                                       reads ``cumulative_social_welfare.csv``
                                       whose stats come from cumsum on raw
                                       per-seed traces (proper CI band).
* ``cumulative_worker_regret.png`` -- two lines: CA-ETC vs PPO, with 95% CI
                                       bands. Reads ``cum_regret_w_*`` from
                                       each run's ``cumulative_regret_welfare.csv``.
* ``cumulative_firm_regret.png``   -- same with ``cum_regret_f_*``.
* ``friction_loss.png``            -- per-period friction loss; reads each run's
                                       ``friction_loss.csv`` (per-period CSV
                                       newly emitted by save_batched_cumulative_csvs).

Both source runs must have been produced *after* the per-period friction +
social_welfare CSVs were added; older runs only have the cumulative versions.
Missing CSVs cause the corresponding figure to be skipped with a warning.

Inputs are explicit run directories -- no auto-pairing -- to avoid mistaken
pairings when multiple runs share parameters. Expected to be invoked from the
machine that already has the runs; PNGs land in ``--out_dir`` for scp.

Dependencies: numpy, pandas, matplotlib. No JAX needed.
"""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _band(ax, t, mean, lo, hi, label, color=None, ls="-"):
    line, = ax.plot(t, mean, label=label, color=color, linewidth=1.6, linestyle=ls)
    ax.fill_between(t, lo, hi, color=line.get_color(), alpha=0.15)
    return line


def _read_regret_csv(run_dir: Path) -> Optional[pd.DataFrame]:
    """Returns DataFrame with t and cum_regret_w/f mean+CI columns; None if absent."""
    p = run_dir / "cumulative_regret_welfare.csv"
    if not p.is_file():
        print(f"  [warn] missing {p}")
        return None
    return pd.read_csv(p)


def _read_friction_per_period(run_dir: Path) -> Optional[pd.DataFrame]:
    p = run_dir / "friction_loss.csv"
    if not p.is_file():
        print(f"  [warn] missing {p} -- per-period friction not available; "
              f"re-run after the friction_loss.csv extension landed.")
        return None
    return pd.read_csv(p)


def _read_sw_per_period(run_dir: Path) -> Optional[pd.DataFrame]:
    p = run_dir / "social_welfare.csv"
    if not p.is_file():
        print(f"  [warn] missing {p} -- per-period SW not available; "
              f"re-run after the social_welfare.csv extension landed.")
        return None
    return pd.read_csv(p)


def _read_cum_sw(run_dir: Path) -> Optional[pd.DataFrame]:
    p = run_dir / "cumulative_social_welfare.csv"
    if not p.is_file():
        print(f"  [warn] missing {p} -- cumulative SW not available; "
              f"re-run after the cumulative_social_welfare.csv extension landed.")
        return None
    return pd.read_csv(p)


def _compile_overlay_csv(
    df_ppo: Optional[pd.DataFrame],
    df_cae: Optional[pd.DataFrame],
    out_path: Path,
    columns: Optional[set] = None,
) -> Optional[Path]:
    """Merge PPO and CA-ETC DataFrames into one self-contained CSV per figure.

    Output schema: ``t`` + every other column prefixed with ``ppo_`` or
    ``caetc_``. Lets the user download a single CSV per figure for offline
    re-plotting without having to merge the per-run CSVs by hand.

    ``columns``: if provided, only these column names from each side are kept
    (other than ``t``); otherwise all columns are kept.
    """
    if df_ppo is None or df_cae is None:
        return None

    def _prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        keep = ["t"] + [
            c for c in df.columns
            if c != "t" and (columns is None or c in columns)
        ]
        sub = df[keep].copy()
        sub = sub.rename(columns={c: f"{prefix}_{c}" for c in keep if c != "t"})
        return sub

    merged = _prefix(df_ppo, "ppo").merge(_prefix(df_cae, "caetc"), on="t", how="outer")
    merged.to_csv(out_path, index=False)
    return out_path


def _plot_two_run_overlay(
    series_a: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str],
    series_b: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str],
    title: str,
    ylabel: str,
) -> plt.Figure:
    """Each series tuple: (t, mean, ci_lo, ci_hi, label)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for t, m, lo, hi, label in (series_a, series_b):
        _band(ax, t, m, lo, hi, label=label)
    ax.set_title(f"{title}  (mean ± 95% CI across seeds)")
    ax.set_xlabel("market period")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay PPO eval vs CA-ETC baseline metrics.",
    )
    parser.add_argument("--ppo_dir", required=True, type=Path,
                        help="PPO eval output directory (contains "
                             "cumulative_regret_welfare.csv, friction_loss.csv).")
    parser.add_argument("--caetc_dir", required=True, type=Path,
                        help="CA-ETC output directory (same files plus "
                             "social_welfare.png from cum_welfare_plot).")
    parser.add_argument("--out_dir", required=True, type=Path,
                        help="Where to write the comparison PNGs.")
    parser.add_argument("--ppo_label", default="PPO",
                        help="Legend label for the PPO curve.")
    parser.add_argument("--caetc_label", default="CA-ETC",
                        help="Legend label for the CA-ETC curve.")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Skip W&B upload; just save PNGs locally.")
    parser.add_argument("--wandb_entity", default="haijingzong-university-of-washington")
    parser.add_argument("--wandb_project", default="hireRL")
    parser.add_argument("--run_name", default=None,
                        help="W&B run name; defaults to compare_<out_dir basename>_<ts>.")
    parser.add_argument("--refs_from", choices=["caetc", "ppo"], default="caetc",
                        help="Which run's per-period DA-ref / planner to draw "
                             "in social_welfare.png. With matching --seed and "
                             "CA-ETC --unified_seeds, both runs see identical "
                             "(x, y), so refs from either match exactly.")
    parser.add_argument("--gauge", choices=["effective", "committed"],
                        default="effective",
                        help="Which gauge to use for CA-ETC's metrics (SW + "
                             "cumulative regret_w/f). 'effective' (default) = "
                             "matched | tentative_matched, what CA-ETC reports "
                             "natively, nonzero during exploration via tentative "
                             "pairs. 'committed' = state.matched post-RETENTION "
                             "(the same gauge PPO eval uses), giving a strict "
                             "apples-to-apples overlay -- but 0 throughout "
                             "CA-ETC's exploration phase since RETENTION always "
                             "releases. PPO's lines are always its native "
                             "committed-gauge values. friction_loss is "
                             "gauge-independent and unaffected by this flag.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = not args.no_wandb
    wandb_run = None
    if use_wandb:
        import wandb
        run_name = args.run_name or (
            f"compare_{args.out_dir.name}_{datetime.now():%Y%m%d-%H%M%S}"
        )
        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            config={
                "kind": "ppo_vs_caetc_compare",
                "ppo_dir": str(args.ppo_dir),
                "caetc_dir": str(args.caetc_dir),
                "ppo_label": args.ppo_label,
                "caetc_label": args.caetc_label,
            },
        )

    # 1) Figure 1: per-period SW with 4 lines (CA-ETC actual + PPO actual +
    # DA ref + planner). Refs default to the CA-ETC run (configurable).
    sw_ppo = _read_sw_per_period(args.ppo_dir)
    sw_cae = _read_sw_per_period(args.caetc_dir)
    if sw_ppo is not None and sw_cae is not None:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        # CA-ETC actual: pick gauge per --gauge. Fall back to effective if
        # 'committed' is requested but the CSV pre-dates the committed_* columns.
        # Default --gauge effective leaves labels untagged (= prior behavior);
        # only tag with "(committed)" when explicitly committed-gauge mode.
        if args.gauge == "committed" and "committed_mean" in sw_cae.columns:
            cae_prefix = "committed"
            cae_label = f"{args.caetc_label} (committed)"
            ppo_label = f"{args.ppo_label} (committed)"
        else:
            if args.gauge == "committed":
                print("  [warn] --gauge committed requested but CA-ETC CSV "
                      "lacks committed_* columns (older run); falling back to "
                      "effective gauge.")
            cae_prefix = "actual"
            cae_label = args.caetc_label
            ppo_label = args.ppo_label
        for df, label, color, prefix in [
            (sw_cae, cae_label, "C0", cae_prefix),
            (sw_ppo, ppo_label, "C1", "actual"),
        ]:
            _band(
                ax,
                df["t"].values,
                df[f"{prefix}_mean"].values,
                df[f"{prefix}_ci_lo"].values,
                df[f"{prefix}_ci_hi"].values,
                label=label, color=color, ls="-",
            )
        # Refs from the chosen source run.
        ref_src = sw_cae if args.refs_from == "caetc" else sw_ppo
        ref_src_label = args.caetc_label if args.refs_from == "caetc" else args.ppo_label
        if "da_ref_mean" in ref_src.columns:
            _band(
                ax,
                ref_src["t"].values,
                ref_src["da_ref_mean"].values,
                ref_src["da_ref_ci_lo"].values,
                ref_src["da_ref_ci_hi"].values,
                label=f"DA ref ({ref_src_label})", color="C2", ls="--",
            )
        if "planner_mean" in ref_src.columns:
            _band(
                ax,
                ref_src["t"].values,
                ref_src["planner_mean"].values,
                ref_src["planner_ci_lo"].values,
                ref_src["planner_ci_hi"].values,
                label=f"planner ({ref_src_label})", color="C3", ls="-",
            )
        ax.set_title(
            f"Social welfare per market period  (mean ± 95% CI across seeds)"
        )
        ax.set_xlabel("market period")
        ax.set_ylabel("SW per period")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        path = args.out_dir / "social_welfare.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path}")
        csv_path = _compile_overlay_csv(
            sw_ppo, sw_cae, args.out_dir / "social_welfare.csv",
        )
        if csv_path is not None:
            print(f"  saved {csv_path}")
    elif sw_cae is not None:
        # Fallback: only CA-ETC has SW CSV -- copy its solo figure.
        sw_src = args.caetc_dir / "social_welfare.png"
        if sw_src.is_file():
            sw_dst = args.out_dir / "social_welfare.png"
            shutil.copyfile(sw_src, sw_dst)
            print(f"  [warn] PPO SW CSV missing -- copied CA-ETC solo figure.")

    # 1b) Cumulative-over-time twin of figure 1: same 4 lines, cumsum on
    # raw per-seed traces (proper CI bands, not just cumsum of period CIs).
    cum_sw_ppo = _read_cum_sw(args.ppo_dir)
    cum_sw_cae = _read_cum_sw(args.caetc_dir)
    if cum_sw_ppo is not None and cum_sw_cae is not None:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        # Mirror the per-period block's gauge logic, but using cum_* columns.
        if args.gauge == "committed" and "cum_committed_mean" in cum_sw_cae.columns:
            cae_prefix = "cum_committed"
            cae_label = f"{args.caetc_label} (committed)"
            ppo_label = f"{args.ppo_label} (committed)"
        else:
            if args.gauge == "committed":
                print("  [warn] --gauge committed requested but CA-ETC cum SW "
                      "CSV lacks cum_committed_* columns; falling back to "
                      "effective gauge.")
            cae_prefix = "cum_actual"
            cae_label = args.caetc_label
            ppo_label = args.ppo_label
        for df, label, color, prefix in [
            (cum_sw_cae, cae_label, "C0", cae_prefix),
            (cum_sw_ppo, ppo_label, "C1", "cum_actual"),
        ]:
            _band(
                ax,
                df["t"].values,
                df[f"{prefix}_mean"].values,
                df[f"{prefix}_ci_lo"].values,
                df[f"{prefix}_ci_hi"].values,
                label=label, color=color, ls="-",
            )
        cum_ref_src = cum_sw_cae if args.refs_from == "caetc" else cum_sw_ppo
        cum_ref_src_label = (args.caetc_label if args.refs_from == "caetc"
                             else args.ppo_label)
        if "cum_da_ref_mean" in cum_ref_src.columns:
            _band(
                ax,
                cum_ref_src["t"].values,
                cum_ref_src["cum_da_ref_mean"].values,
                cum_ref_src["cum_da_ref_ci_lo"].values,
                cum_ref_src["cum_da_ref_ci_hi"].values,
                label=f"DA ref ({cum_ref_src_label})", color="C2", ls="--",
            )
        if "cum_planner_mean" in cum_ref_src.columns:
            _band(
                ax,
                cum_ref_src["t"].values,
                cum_ref_src["cum_planner_mean"].values,
                cum_ref_src["cum_planner_ci_lo"].values,
                cum_ref_src["cum_planner_ci_hi"].values,
                label=f"planner ({cum_ref_src_label})", color="C3", ls="-",
            )
        ax.set_title(
            f"Cumulative social welfare  (mean ± 95% CI across seeds)"
        )
        ax.set_xlabel("market period")
        ax.set_ylabel("cumulative SW")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        path = args.out_dir / "cumulative_social_welfare.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path}")
        csv_path = _compile_overlay_csv(
            cum_sw_ppo, cum_sw_cae,
            args.out_dir / "cumulative_social_welfare.csv",
        )
        if csv_path is not None:
            print(f"  saved {csv_path}")

    # 2) and 3) Cumulative regret overlays.
    rw_ppo = _read_regret_csv(args.ppo_dir)
    rw_cae = _read_regret_csv(args.caetc_dir)

    if rw_ppo is not None and rw_cae is not None:
        for side, slug, ylabel in [
            ("w", "cumulative_worker_regret", "cumulative worker regret"),
            ("f", "cumulative_firm_regret",   "cumulative firm regret"),
        ]:
            # CA-ETC: prefer committed-gauge column under --gauge committed,
            # fall back to effective with a warning if the column is absent
            # (older runs pre-dating the regret_*_committed extension).
            cae_col = f"cum_regret_{side}"
            cae_label = args.caetc_label
            if args.gauge == "committed":
                committed_col = f"cum_regret_{side}_committed"
                if f"{committed_col}_mean" in rw_cae.columns:
                    cae_col = committed_col
                    cae_label = f"{args.caetc_label} (committed)"
                else:
                    print(f"  [warn] --gauge committed requested but CA-ETC "
                          f"regret CSV lacks {committed_col}_* columns "
                          f"(older run); falling back to effective gauge "
                          f"for {slug}.")
                    cae_label = f"{args.caetc_label} (effective)"
            ppo_label = (f"{args.ppo_label} (committed)"
                         if args.gauge == "committed" else args.ppo_label)
            ppo_series = (
                rw_ppo["t"].values,
                rw_ppo[f"cum_regret_{side}_mean"].values,
                rw_ppo[f"cum_regret_{side}_ci_lo"].values,
                rw_ppo[f"cum_regret_{side}_ci_hi"].values,
                ppo_label,
            )
            cae_series = (
                rw_cae["t"].values,
                rw_cae[f"{cae_col}_mean"].values,
                rw_cae[f"{cae_col}_ci_lo"].values,
                rw_cae[f"{cae_col}_ci_hi"].values,
                cae_label,
            )
            fig = _plot_two_run_overlay(
                cae_series, ppo_series,
                title=ylabel.capitalize(),
                ylabel=ylabel,
            )
            path = args.out_dir / f"{slug}.png"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {path}")
            stats = ("mean", "median", "ci_lo", "ci_hi")
            keep_cols = (
                {f"cum_regret_{side}_{s}" for s in stats}
                | {f"cum_regret_{side}_committed_{s}" for s in stats}
            )
            csv_path = _compile_overlay_csv(
                rw_ppo, rw_cae, args.out_dir / f"{slug}.csv",
                columns=keep_cols,
            )
            if csv_path is not None:
                print(f"  saved {csv_path}")

    # 4) Per-period friction loss overlay.
    fl_ppo = _read_friction_per_period(args.ppo_dir)
    fl_cae = _read_friction_per_period(args.caetc_dir)

    if fl_ppo is not None and fl_cae is not None:
        ppo_series = (
            fl_ppo["t"].values,
            fl_ppo["friction_loss_mean"].values,
            fl_ppo["friction_loss_ci_lo"].values,
            fl_ppo["friction_loss_ci_hi"].values,
            args.ppo_label,
        )
        cae_series = (
            fl_cae["t"].values,
            fl_cae["friction_loss_mean"].values,
            fl_cae["friction_loss_ci_lo"].values,
            fl_cae["friction_loss_ci_hi"].values,
            args.caetc_label,
        )
        fig = _plot_two_run_overlay(
            cae_series, ppo_series,
            title="Friction loss per market period",
            ylabel="friction loss",
        )
        path = args.out_dir / "friction_loss.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path}")
        csv_path = _compile_overlay_csv(
            fl_ppo, fl_cae, args.out_dir / "friction_loss.csv",
        )
        if csv_path is not None:
            print(f"  saved {csv_path}")

    print(f"All comparison figures written to {args.out_dir}")

    if use_wandb and wandb_run is not None:
        import wandb
        for png_path in sorted(args.out_dir.glob("*.png")):
            wandb.log({f"plot/{png_path.stem}": wandb.Image(str(png_path))})
            print(f"  uploaded {png_path.name} -> wandb")
        # Also upload the compiled per-figure CSVs so the user can download a
        # single self-contained CSV per figure for offline replotting. Files
        # appear under the run's "Files" tab on W&B.
        for csv_path in sorted(args.out_dir.glob("*.csv")):
            wandb.save(str(csv_path), base_path=str(args.out_dir), policy="now")
            print(f"  uploaded {csv_path.name} -> wandb")
        wandb.finish()
        print("W&B run finished.")


if __name__ == "__main__":
    main()
