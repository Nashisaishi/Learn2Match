"""Entry point for the batched (vmapped) CA-ETC baseline.

Drives :class:`batched_ca_etc_baseline.BatchedCAETCBaseline` across N seeds
in parallel on one GPU, then plots seed-aggregated metrics (median + IQR
band, plus a thin per-seed line for variance intuition).

Why a separate entry script
---------------------------
The single-seed entry (``plot_ca_etc_baseline.py``) consumes a 1-D-time
history via ``plot_helpers.make_figures``; the batched history has an extra
leading seed axis and is plotted by ``plot_helpers.make_batched_figures``.
This entry script just drives ``BatchedCAETCBaseline`` and dispatches to
the shared batched plotting / CSV helpers.

Usage
-----
    cd /home/hisaishi/hireRL
    conda activate jaxenv
    python learn2match/ca-etc-hirerl/plot_batched_ca_etc_baseline.py \
        --Nw 5 --Nf 5 --d 4 --horizon 200 \
        --num_seeds 32 --T0 5 --gamma 0.4 \
        --no_wandb

Skip flags:
    --no_plots    don't save PNGs (still runs + prints stats + optionally W&B)
    --no_wandb    don't upload to W&B (still saves PNGs locally)

Both can be combined for a stats-only dry run.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import jax
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)                       # learn2match/
_GRANDPARENT = os.path.dirname(_PARENT)                # repo root (jax_pbt)
_EXAMPLES = os.path.join(_PARENT, "examples")          # plot_helpers lives here
for _p in (_HERE, _PARENT, _GRANDPARENT, _EXAMPLES):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from batched_ca_etc_baseline import BatchedCAETCBaseline  # noqa: E402
from cum_welfare_plot import make_welfare_figures, _planner_per_period  # noqa: E402

from hirerl import HireRLConfig, HireRLEnv  # noqa: E402
from plot_helpers import (  # noqa: E402
    make_batched_figures,
    save_batched_cumulative_csvs,
)


HORIZON_BUFFER = 5


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--Nw", type=int, default=5)
    parser.add_argument("--Nf", type=int, default=5)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--num_seeds", type=int, default=32,
                        help="Number of parallel seeds (= leading vmap dim).")
    parser.add_argument("--T0", type=int, default=None,
                        help="Defaults to ceil(Nw/Nf)*Nf.")
    parser.add_argument("--gamma", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0,
                        help="Base PRNG seed (each parallel seed gets a derived sub-key).")
    parser.add_argument("--sigma_interview", type=float, default=0.3)
    parser.add_argument("--sigma_match", type=float, default=0.1)
    parser.add_argument("--lambda_reveal", type=float, default=1.0)
    parser.add_argument("--no_confidence_check", action="store_true")
    parser.add_argument(
        "--outside_option", type=float, default=0.0,
        help="Reservation utility threshold used by reference_match and "
             "belief_induced_match. With default 0.0, pairs with x.y < 0 are "
             "rejected by the reference matching but the algorithm still "
             "matches them (causing positive per-period regret in steady "
             "state). Pass a very negative value (e.g. -1e9) to disable, "
             "matching the original CA-ETC paper's no-outside-option setup.",
    )
    parser.add_argument(
        "--non_negative_features", action="store_true",
        help="Sample x and y as half-normal (|N(0, 1)|) so every pair has "
             "x.y >= 0. With this flag, outside_option=0 doesn't bite -- "
             "closest to the original CA-ETC paper where all mean rewards "
             "lie in [0, 0.99].",
    )
    parser.add_argument("--out_dir", type=str, default=None,
                        help="PNG output dir (default: timestamped subdir under plots/).")
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--no_cum_welfare", action="store_true",
                        help="Skip the cumulative_social_welfare.png plot.")
    parser.add_argument("--no_per_period_welfare", action="store_true",
                        help="Skip the per-period social_welfare.png plot.")
    parser.add_argument("--no_da_ref", action="store_true",
                        help="In SW plots, skip the DA-ref curve.")
    parser.add_argument("--no_planner", action="store_true",
                        help="In SW plots, skip the planner first-best "
                             "curve (host-side linear_sum_assignment).")
    parser.add_argument("--unified_seeds", action="store_true",
                        help="Use the same per-seed env.reset key derivation "
                             "as learn2match/examples/eval_and_plot.py "
                             "(PPO eval). With this flag and the same --seed / "
                             "--num_seeds, per-seed (x, y) are bit-equal "
                             "across the two scripts, so DA-ref / planner "
                             "upper bounds line up exactly when overlaying. "
                             "Off by default to preserve reproducibility of "
                             "older runs.")
    parser.add_argument("--wandb_entity", default="haijingzong-university-of-washington")
    parser.add_argument("--wandb_project", default="hireRL")
    parser.add_argument("--run_name", default=None)
    args = parser.parse_args()

    if args.T0 is None:
        num_batches = (args.Nw + args.Nf - 1) // args.Nf
        T0 = num_batches * args.Nf
    else:
        T0 = args.T0

    env_horizon = args.horizon + HORIZON_BUFFER
    config = HireRLConfig(
        Nw=args.Nw, Nf=args.Nf, d=args.d, horizon=env_horizon,
        sigma_interview=args.sigma_interview,
        sigma_match=args.sigma_match,
        lambda_reveal=args.lambda_reveal,
        outside_option=args.outside_option,
        non_negative_features=args.non_negative_features,
    )
    env = HireRLEnv(config)

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        run_name = args.run_name or (
            f"ca-etc-batched-{args.Nw}x{args.Nf}-h{args.horizon}-N{args.num_seeds}-"
            f"seed{args.seed}-{datetime.now():%Y%m%d-%H%M%S}"
        )
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            config={
                "algorithm": "ca_etc_batched",
                "num_seeds": args.num_seeds,
                "Nw": args.Nw, "Nf": args.Nf, "d": args.d,
                "baseline_horizon": args.horizon,
                "env_horizon": env_horizon,
                "T0": T0, "gamma": args.gamma, "seed": args.seed,
                "sigma_interview": args.sigma_interview,
                "sigma_match": args.sigma_match,
                "lambda_reveal": args.lambda_reveal,
                "use_confidence_check": not args.no_confidence_check,
                "outside_option": float(config.outside_option),
            },
        )

    print(
        f"Batched CA-ETC: N={args.num_seeds} seeds, Nw={args.Nw}, Nf={args.Nf}, "
        f"d={args.d}, horizon={args.horizon}, T0={T0}, gamma={args.gamma}"
    )

    base_rng = jax.random.PRNGKey(args.seed)
    baseline = BatchedCAETCBaseline(
        env=env,
        num_seeds=args.num_seeds,
        base_rng=base_rng,
        Nw=args.Nw, Nf=args.Nf, horizon=args.horizon,
        T0=T0, gamma=args.gamma,
        use_confidence_check=not args.no_confidence_check,
        unified_seeds=args.unified_seeds,
    )

    t0 = time.perf_counter()
    history = baseline.run()
    # Block until device is done so the timing reflects actual compute, not
    # async dispatch.
    jax.block_until_ready(history["regret_w_total"])
    elapsed = time.perf_counter() - t0
    T = int(history["regret_w_total"].shape[1])
    print(
        f"Completed in {elapsed:.1f}s (epochs={len(history['epoch_log'])}, "
        f"market_periods={T}, num_seeds={args.num_seeds}). "
        f"Throughput: {args.num_seeds * T / elapsed:.0f} seed-periods/sec."
    )

    # Planner welfare via host-side linear_sum_assignment on the per-snapshot
    # true U = x @ y.T. Stashed back into history so save_batched_cumulative_csvs
    # picks it up (PPO eval already populates this key in eval_and_plot.py:505)
    # and so cum_welfare_plot.make_welfare_figures can reuse it instead of
    # recomputing.
    if "social_welfare_planner" not in history and {"x_snap", "y_snap"}.issubset(history):
        history["social_welfare_planner"] = _planner_per_period(
            np.asarray(history["x_snap"]), np.asarray(history["y_snap"])
        )

    # Per-period scalar logging to W&B.
    if use_wandb:
        cum_reg_w = np.cumsum(history["regret_w_total"], axis=1)
        cum_reg_f = np.cumsum(history["regret_f_total"], axis=1)
        cum_fric  = np.cumsum(history["friction_loss"], axis=1)
        for idx, t in enumerate(history["t"]):
            log = {}
            for key in ("reward_w_total", "reward_f_total",
                        "regret_w_total", "regret_f_total",
                        "info_loss_w_total", "info_loss_f_total",
                        "social_welfare", "friction_loss", "match_rate"):
                arr = history[key][:, idx]
                log[f"agg/{key}_mean"] = float(arr.mean())
                log[f"agg/{key}_std"]  = float(arr.std())
            for key, arr in [("cum_regret_w", cum_reg_w),
                             ("cum_regret_f", cum_reg_f),
                             ("cum_friction_loss", cum_fric)]:
                slc = arr[:, idx]
                log[f"cum/{key}_mean"] = float(slc.mean())
                log[f"cum/{key}_std"]  = float(slc.std())
            wandb.log(log, step=int(t))

    if not args.no_plots:
        figs = make_batched_figures(history, config)
        # CA-ETC's social-welfare trace is dominated by the explore-vs-commit
        # schedule (large drops during round-robin epochs) and isn't a useful
        # baseline against the RL policy's smooth SW curve, so skip it here.
        if "aux/social_welfare" in figs:
            plt.close(figs.pop("aux/social_welfare"))
        if args.out_dir is None:
            out_dir = os.path.join(
                _HERE, "plots",
                f"ca_etc_batched_{args.Nw}x{args.Nf}_h{args.horizon}_"
                f"N{args.num_seeds}_seed{args.seed}_{datetime.now():%Y%m%d-%H%M%S}",
            )
        else:
            out_dir = args.out_dir
        os.makedirs(out_dir, exist_ok=True)
        for name, fig in figs.items():
            path = os.path.join(out_dir, f"{name}.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fig.savefig(path, dpi=120, bbox_inches="tight")
            print(f"  saved {path}")
            if use_wandb:
                wandb.log({f"plot/{name}": wandb.Image(fig)})
            plt.close(fig)

        welfare_figs = make_welfare_figures(
            history,
            with_da_ref=not args.no_da_ref,
            with_planner=not args.no_planner,
            include_per_period=not args.no_per_period_welfare,
            include_cumulative=not args.no_cum_welfare,
        )
        for wf_name, wf_fig in welfare_figs.items():
            wf_path = os.path.join(out_dir, f"{wf_name}.png")
            wf_fig.savefig(wf_path, dpi=120, bbox_inches="tight")
            print(f"  saved {wf_path}")
            if use_wandb:
                wandb.log({f"plot/{wf_name}": wandb.Image(wf_fig)})
            plt.close(wf_fig)

        csv_paths = save_batched_cumulative_csvs(history, out_dir)
        for name, path in csv_paths.items():
            print(f"  saved {path}")

        print(f"All figures written to {out_dir}")
    else:
        print("--no_plots set; skipped figure generation.")

    if use_wandb:
        cum_reg_w = np.cumsum(history["regret_w_total"], axis=1)
        cum_reg_f = np.cumsum(history["regret_f_total"], axis=1)
        cum_fric  = np.cumsum(history["friction_loss"], axis=1)
        wandb.run.summary["mean_final_cum_regret_w"] = float(cum_reg_w[:, -1].mean())
        wandb.run.summary["std_final_cum_regret_w"]  = float(cum_reg_w[:, -1].std())
        wandb.run.summary["mean_final_cum_regret_f"] = float(cum_reg_f[:, -1].mean())
        wandb.run.summary["std_final_cum_regret_f"]  = float(cum_reg_f[:, -1].std())
        wandb.run.summary["mean_final_cum_friction_loss"] = float(cum_fric[:, -1].mean())
        wandb.run.summary["std_final_cum_friction_loss"]  = float(cum_fric[:, -1].std())
        wandb.run.summary["wallclock_seconds"]    = float(elapsed)
        wandb.run.summary["seed_periods_per_sec"] = float(args.num_seeds * T / elapsed)
        wandb.run.summary["num_seeds"]            = int(args.num_seeds)
        wandb.run.summary["market_periods"]       = int(T)
        wandb.finish()
        print("W&B run finished.")


if __name__ == "__main__":
    main()
