"""Sole entry point for the CA-ETC baseline.

Drives a HireRL env with the CA-ETC baseline (``CAETCBaseline`` from
``ca_etc_baseline.py``) and produces per-period plots of regret, friction
loss, social welfare, match rate, per-agent reward / regret / information
loss, and a worker-side match Gantt. Every metric is computed through
``hirerl.metrics`` (the same module the env emits in
``info["settled_metrics"]`` and the RL eval in ``examples/eval_and_plot.py``
consumes), so CA-ETC and PPO plots share axes, units, and color schemes for
direct comparison.

Pass ``--no_plots`` to skip figure generation (still runs baseline + prints
stats + optionally logs scalars to wandb). Pass ``--no_wandb`` to skip the
upload (still saves PNGs locally). Both flags can be combined for a stats-only
dry run.

Note on the env horizon buffer
------------------------------
The env auto-resets at ``HireRLConfig.horizon``. If the baseline runs to its
own ``horizon`` and the env's horizon equals that value, the very last
post-RETENTION snapshot is the auto-reset state, polluting per-agent regret
and information loss for that single period. The aggregates in
``info["settled_metrics"]`` are still correct because the env captures them
*before* the reset, but the per-agent recomputation happens on the post-step
state which has been reset.

To keep all per-period values clean we set ``HireRLConfig.horizon =
baseline_horizon + HORIZON_BUFFER``; the baseline's own ``done_or_horizon_reached``
still stops at ``baseline_horizon`` so the env never auto-resets within the
recorded run.

Usage:
    cd /home/hisaishi/hireRL
    conda activate jaxenv
    python learn2match/ca-etc-hirerl/plot_ca_etc_baseline.py
    python learn2match/ca-etc-hirerl/plot_ca_etc_baseline.py --Nw 8 --Nf 8 --horizon 300
    python learn2match/ca-etc-hirerl/plot_ca_etc_baseline.py --no_wandb           # PNGs only
    python learn2match/ca-etc-hirerl/plot_ca_etc_baseline.py --no_plots           # wandb scalars only
    python learn2match/ca-etc-hirerl/plot_ca_etc_baseline.py --no_plots --no_wandb # stats only

Wandb integration mirrors examples/eval_and_plot.py: one ``wandb.log`` call per
settled market period (5 phase steps), keyed by ``state.market_t`` so RL and
CA-ETC runs can be overlaid on the same x-axis. Aggregated summary stats land
in ``wandb.run.summary``; the six matplotlib figures are uploaded as
``plot/<name>`` (skipped under ``--no_plots``).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import jax
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)              # learn2match/
_GRANDPARENT = os.path.dirname(_PARENT)       # repo root
_EXAMPLES = os.path.join(_PARENT, "examples") # plot_helpers lives here
for _p in (_HERE, _PARENT, _GRANDPARENT, _EXAMPLES):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ca_etc_baseline import CAETCBaseline  # noqa: E402

from hirerl import HireRLConfig, HireRLEnv  # noqa: E402
from plot_helpers import make_figures, save_cumulative_csvs, stack_history  # noqa: E402


HORIZON_BUFFER = 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--Nw", type=int, default=5)
    parser.add_argument("--Nf", type=int, default=5)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=200,
                        help="CA-ETC baseline horizon in market periods.")
    parser.add_argument("--T0", type=int, default=None,
                        help="Defaults to ceil(Nw/Nf)*Nf (paper-spec minimum).")
    parser.add_argument("--gamma", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
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
                        help="Directory for PNG outputs (only used when figures are "
                             "generated). Defaults to a timestamped subdir under "
                             "learn2match/ca-etc-hirerl/plots/.")
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip matplotlib figure generation, PNG saving, and "
                             "wandb image upload. Per-period scalars still go to "
                             "wandb (unless --no_wandb is also set).")
    parser.add_argument("--wandb_entity", default="haijingzong-university-of-washington")
    parser.add_argument("--wandb_project", default="hireRL")
    parser.add_argument("--run_name", default=None,
                        help="Wandb run name. Defaults to ca-etc-{Nw}x{Nf}-h{horizon}-seed{seed}-<timestamp>.")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Skip wandb upload; just save PNGs locally.")
    args = parser.parse_args()

    if args.T0 is None:
        num_batches = (args.Nw + args.Nf - 1) // args.Nf
        T0 = num_batches * args.Nf
    else:
        T0 = args.T0

    # Buffer env horizon above baseline horizon so the final settled snapshot
    # is not the auto-reset state. See module docstring.
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
    const = env.default_const

    rng = jax.random.PRNGKey(args.seed)
    rng, reset_rng, baseline_rng = jax.random.split(rng, 3)
    state, _obs = env.env_reset(reset_rng, const)

    baseline = CAETCBaseline(
        env=env, const=const, rng=baseline_rng,
        Nw=args.Nw, Nf=args.Nf, horizon=args.horizon,
        T0=T0, gamma=args.gamma,
        use_confidence_check=not args.no_confidence_check,
        seed=args.seed,
    )

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        run_name = args.run_name or (
            f"ca-etc-{args.Nw}x{args.Nf}-h{args.horizon}-seed{args.seed}-"
            f"{datetime.now():%Y%m%d-%H%M%S}"
        )
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            config={
                "algorithm": "ca_etc",
                "Nw": args.Nw, "Nf": args.Nf, "d": args.d,
                "baseline_horizon": args.horizon,
                "env_horizon": env_horizon,
                "T0": T0,
                "gamma": args.gamma,
                "seed": args.seed,
                "sigma_interview": args.sigma_interview,
                "sigma_match": args.sigma_match,
                "lambda_reveal": args.lambda_reveal,
                "use_confidence_check": not args.no_confidence_check,
                "outside_option": float(config.outside_option),
            },
        )

    print(
        f"Running CA-ETC: Nw={args.Nw}, Nf={args.Nf}, d={args.d}, "
        f"baseline_horizon={args.horizon}, env_horizon={env_horizon}, "
        f"T0={T0}, gamma={args.gamma}"
    )
    _final_state, metrics = baseline.run(state)
    history_records = metrics["settled_history"]
    print(
        f"Completed: epochs={len(metrics['epoch'])}, "
        f"market_periods={baseline.market_period}, "
        f"phase_steps={baseline.timestep}, "
        f"settled_records={len(history_records)}"
    )

    if not history_records:
        if use_wandb:
            wandb.finish(exit_code=1)
        raise SystemExit(
            "No settled-state records were captured. Check that env.step "
            "returned info['is_settled']=True at least once during the run."
        )

    history = stack_history(history_records)

    # Per-period scalar logging. One wandb.log call per settled market period
    # (5 phase steps), keyed by state.market_t so this aligns with the
    # examples/eval_and_plot.py x-axis. Also log the running cumulative
    # regret / friction loss because CA-ETC's intended view is sublinear
    # cumulative regret.
    if use_wandb:
        cum_reg_w = np.cumsum(history["regret_w_total"])
        cum_reg_f = np.cumsum(history["regret_f_total"])
        cum_fric  = np.cumsum(history["friction_loss"])
        for idx, t in enumerate(history["t"]):
            log = {
                "agg/reward_w_total":    float(history["reward_w_total"][idx]),
                "agg/reward_f_total":    float(history["reward_f_total"][idx]),
                "agg/regret_w_total":    float(history["regret_w_total"][idx]),
                "agg/regret_f_total":    float(history["regret_f_total"][idx]),
                "agg/info_loss_w_total": float(history["info_loss_w_total"][idx]),
                "agg/info_loss_f_total": float(history["info_loss_f_total"][idx]),
                "agg/social_welfare":    float(history["social_welfare"][idx]),
                "agg/friction_loss":     float(history["friction_loss"][idx]),
                "agg/match_rate":        float(history["match_rate"][idx]),
                "cum/regret_w":          float(cum_reg_w[idx]),
                "cum/regret_f":          float(cum_reg_f[idx]),
                "cum/friction_loss":     float(cum_fric[idx]),
            }
            for i in range(args.Nw):
                log[f"per_worker/reward_{i}"]    = float(history["reward_w_per_agent"][idx, i])
                log[f"per_worker/regret_{i}"]    = float(history["regret_w_per_agent"][idx, i])
                log[f"per_worker/info_loss_{i}"] = float(history["info_loss_w_per_agent"][idx, i])
            for j in range(args.Nf):
                log[f"per_firm/reward_{j}"]    = float(history["reward_f_per_agent"][idx, j])
                log[f"per_firm/regret_{j}"]    = float(history["regret_f_per_agent"][idx, j])
                log[f"per_firm/info_loss_{j}"] = float(history["info_loss_f_per_agent"][idx, j])
            wandb.log(log, step=int(t))

    if not args.no_plots:
        figs = make_figures(history, config)

        if args.out_dir is None:
            out_dir = os.path.join(
                _HERE, "plots",
                f"ca_etc_{args.Nw}x{args.Nf}_h{args.horizon}_seed{args.seed}_"
                f"{datetime.now():%Y%m%d-%H%M%S}",
            )
        else:
            out_dir = args.out_dir
        os.makedirs(out_dir, exist_ok=True)

        for name, fig in figs.items():
            path = os.path.join(out_dir, f"{name}.png")
            fig.savefig(path, dpi=120, bbox_inches="tight")
            print(f"  saved {path}")
            if use_wandb:
                wandb.log({f"plot/{name}": wandb.Image(fig)})
            plt.close(fig)

        csv_paths = save_cumulative_csvs(history, out_dir)
        for name, path in csv_paths.items():
            print(f"  saved {path}")

        print(f"All figures written to {out_dir}")
    else:
        print("--no_plots set; skipped figure generation, PNG save, and wandb image upload.")

    if use_wandb:
        wandb.run.summary["mean_reward_workers"]    = float(np.mean(history["reward_w_total"]))
        wandb.run.summary["mean_reward_firms"]      = float(np.mean(history["reward_f_total"]))
        wandb.run.summary["mean_regret_workers"]    = float(np.mean(history["regret_w_total"]))
        wandb.run.summary["mean_regret_firms"]      = float(np.mean(history["regret_f_total"]))
        wandb.run.summary["mean_info_loss_workers"] = float(np.mean(history["info_loss_w_total"]))
        wandb.run.summary["mean_info_loss_firms"]   = float(np.mean(history["info_loss_f_total"]))
        wandb.run.summary["final_cum_regret_w"]     = float(cum_reg_w[-1])
        wandb.run.summary["final_cum_regret_f"]     = float(cum_reg_f[-1])
        wandb.run.summary["final_cum_friction_loss"] = float(cum_fric[-1])
        wandb.run.summary["ca_etc_epochs"]          = int(len(metrics["epoch"]))
        wandb.run.summary["ca_etc_market_periods"]  = int(baseline.market_period)
        wandb.run.summary["ca_etc_phase_steps"]     = int(baseline.timestep)
        wandb.finish()
        print("W&B run finished.")


if __name__ == "__main__":
    main()
