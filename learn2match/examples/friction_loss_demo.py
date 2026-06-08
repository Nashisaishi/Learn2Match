"""Friction-loss curve under round-robin exploration.

Demonstrates that under V2 env semantics (interview-after-retention does not
overwrite hat_x for ever-matched pairs), sigma_match=0, and lambda_reveal
saturated, a round-robin policy that visits every (i, j) pair drives
friction_loss to exactly 0 once full coverage is achieved (~period 18 for a
10x10 market).

Round-robin schedule:
    market_period m -> round_idx = m // 2, is_form = (m % 2 == 0)
    Form period: worker i  ->  firm (i + round_idx) % Nf, both retain
    Dissolve period: NOOP everywhere; retention=dissolve

Run:
    python friction_loss_demo.py
    python friction_loss_demo.py --no_wandb
    python friction_loss_demo.py --num_periods 50 --seed 7
"""

import argparse
import os
import sys
from datetime import datetime

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))   # repo root for jax_pbt
sys.path.insert(0, os.path.join(_HERE, ".."))           # learn2match for hirerl

from jax_pbt.env.spaces import Action

from hirerl import HireRLConfig, HireRLEnv


def make_round_robin_policy(Nw: int, Nf: int):
    """Cyclic worker<->firm schedule that covers every pair in Nw rounds.

    The closure tracks phase_count (a Python int), so each call advances to the
    next env phase deterministically without reading state.phase from the obs.
    Form periods run interview->match->retain on the round's pairing; dissolve
    periods NOOP through everything except retention, which dissolves all matches.
    """
    if Nw != Nf:
        raise ValueError(f"round-robin requires Nw == Nf, got Nw={Nw}, Nf={Nf}")

    holder = {"phase_count": 0}
    arange_w = jnp.arange(Nw, dtype=jnp.int32)
    arange_f = jnp.arange(Nf, dtype=jnp.int32)
    z = jnp.zeros((Nw,), dtype=jnp.int32)
    ones = jnp.ones((Nw,), dtype=jnp.int32)

    def policy_fn(rng, obs_lst):
        pc = holder["phase_count"]
        market_period = pc // 5
        phase = pc % 5                       # 0..4 = propose/respond/match_propose/match_respond/retention
        round_idx = market_period // 2
        is_form = (market_period % 2 == 0)
        holder["phase_count"] = pc + 1

        if not is_form:
            # Dissolve period: NOOP for phases 0..3; in retention (phase 4),
            # action == 0 means worker_retain = (act == 1) is False -> dissolve.
            return z, z, rng

        if phase == 0 or phase == 2:         # INTERVIEW_PROPOSE / MATCH_PROPOSE
            target = (arange_w + round_idx) % Nf
            return target + 1, z, rng
        if phase == 1 or phase == 3:         # INTERVIEW_RESPOND / MATCH_RESPOND
            target = (arange_f - round_idx) % Nw
            return z, target + 1, rng
        # phase == 4 (RETENTION): both retain
        return ones, ones, rng

    return policy_fn


def rollout(env, num_periods, seed, policy_fn):
    """Step env until market_t >= num_periods. Record (period, friction_loss) per settled snapshot."""
    const = env.default_const
    rng = jax.random.key(seed)
    rng, rng_reset = jax.random.split(rng)
    state, obs_lst = env.reset(rng_reset, const)

    times, fls = [], []
    while int(state.market_t) < num_periods:
        rng, sub = jax.random.split(rng)
        w_choice, f_choice, rng = policy_fn(rng, obs_lst)
        action = [
            Action({"choice": jnp.asarray(w_choice, dtype=jnp.int32)}),
            Action({"choice": jnp.asarray(f_choice, dtype=jnp.int32)}),
        ]
        state, obs_lst, _, _, info = env.step(sub, const, state, action)
        if not bool(info["is_settled"]):
            continue
        times.append(int(state.market_t))
        fls.append(float(info["settled_metrics"]["friction_loss"]))

    return np.asarray(times), np.asarray(fls), state


def plot_friction_loss(times, fls, sigma_match, lambda_reveal, Nw, Nf):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, fls, marker="o", linewidth=1.5, markersize=4)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("market period")
    ax.set_ylabel("friction loss")
    ax.set_title(
        f"Round-robin policy: friction loss vs time "
        f"({Nw}x{Nf}, σ_match={sigma_match}, λ={lambda_reveal})"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Nw", type=int, default=10)
    parser.add_argument("--Nf", type=int, default=10)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--num_periods", type=int, default=100)
    parser.add_argument("--sigma_interview", type=float, default=0.5)
    parser.add_argument("--sigma_match", type=float, default=0.0)
    parser.add_argument("--lambda_reveal", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_entity", default="haijingzong-university-of-washington")
    parser.add_argument("--wandb_project", default="hireRL")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--no_wandb", action="store_true",
                        help="Skip W&B upload; just save PNG locally")
    args = parser.parse_args()

    config = HireRLConfig(
        Nw=args.Nw, Nf=args.Nf, d=args.d,
        # +10 buffer keeps env from auto-resetting at horizon mid-episode (env.py:260).
        horizon=args.num_periods + 10,
        sigma_interview=args.sigma_interview,
        sigma_match=args.sigma_match,
        lambda_reveal=args.lambda_reveal,
    )
    env = HireRLEnv(config)

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        run_name = args.run_name or f"roundrobin-demo-{datetime.now():%Y%m%d-%H%M%S}"
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            config={
                "policy": "round_robin",
                "Nw": args.Nw, "Nf": args.Nf, "d": args.d,
                "num_periods": args.num_periods,
                "sigma_interview": args.sigma_interview,
                "sigma_match": args.sigma_match,
                "lambda_reveal": args.lambda_reveal,
                "seed": args.seed,
            },
        )

    policy_fn = make_round_robin_policy(args.Nw, args.Nf)
    print(
        f"Rolling out {args.num_periods} periods (round-robin, "
        f"{args.Nw}x{args.Nf}, σ_M={args.sigma_match}, λ={args.lambda_reveal})..."
    )
    times, fls, final_state = rollout(env, args.num_periods, args.seed, policy_fn)
    print(f"Collected {len(times)} settled snapshots.")

    zero_threshold = 1e-5
    below = np.where(np.abs(fls) < zero_threshold)[0]
    periods_to_zero = int(times[below[0]]) if len(below) else None

    cum_ten = np.asarray(final_state.cumulative_tenure)
    final_coverage = (cum_ten > 0).sum() / (args.Nw * args.Nf)
    print(f"Final pair coverage: {(cum_ten > 0).sum()}/{args.Nw * args.Nf} = {final_coverage:.0%}")
    if periods_to_zero is not None:
        print(f"friction_loss first below {zero_threshold} at market period {periods_to_zero}")
    else:
        print(f"friction_loss never reached < {zero_threshold} (min={fls.min():.4f})")

    if use_wandb:
        # Match the key used by eval_and_plot.py so both runs overlay on the
        # same W&B panel.
        for t, fl in zip(times, fls):
            wandb.log({"agg/friction_loss": float(fl)}, step=int(t))

    fig = plot_friction_loss(times, fls, args.sigma_match, args.lambda_reveal, args.Nw, args.Nf)
    out_dir = os.path.join(_HERE, "eval_outputs")
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, "friction_loss_roundrobin.png")
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"saved {fig_path}")

    if use_wandb:
        wandb.log({"plot/friction_loss": wandb.Image(fig)})
        wandb.run.summary["periods_to_zero"] = periods_to_zero if periods_to_zero is not None else -1
        wandb.run.summary["final_coverage"] = float(final_coverage)
        wandb.run.summary["final_friction_loss"] = float(fls[-1])
        wandb.finish()
        print("W&B run finished.")
    plt.close(fig)


if __name__ == "__main__":
    main()
