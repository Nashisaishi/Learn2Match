"""Runner for the oracle-assisted Round-Robin ETC baseline.

Example:
    python run_rr_etc.py --Nw 5 --Nf 5 --d 5 --horizon 8000 \
        --sigma-interview 0.08 --seed 0 --outdir out_5x5_seed0

Writes into --outdir:
    metrics.csv     per-market-period effective-gauge metrics
    round_log.csv   per-checkpoint diagnostics (confidence / gate / commits)
    summary.json    config + outcome
    curves.png      instantaneous + cumulative regret, welfare, settlement
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "learn2match"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax

from hirerl.config import HireRLConfig
from hirerl.env import HireRLEnv

from rr_etc_baseline import RoundRobinETC, RRETCConfig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--Nw", type=int, default=5)
    ap.add_argument("--Nf", type=int, default=5)
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=8000)
    ap.add_argument("--sigma-interview", type=float, default=0.08)
    ap.add_argument("--sigma-match", type=float, default=0.02)
    ap.add_argument("--outside-option", type=float, default=0.0)
    ap.add_argument("--gaussian-features", action="store_true",
                    help="full Gaussian features (default: half-normal, "
                         "which keeps all utilities non-negative)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--L", type=float, default=1.0,
                    help="exploration granularity: reps/round = ceil(L*log2(T))")
    ap.add_argument("--c-radius", type=float, default=1.0)
    ap.add_argument("--eps-w", type=float, default=0.0)
    ap.add_argument("--eps-f", type=float, default=0.0)
    ap.add_argument("--no-gate", action="store_true",
                    help="disable the arm-readiness gate (ablation only)")
    ap.add_argument("--no-friction", action="store_true")
    ap.add_argument("--no-strict", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--outdir", type=str, default=None)
    args = ap.parse_args()

    env_cfg = HireRLConfig(
        Nw=args.Nw, Nf=args.Nf, d=args.d, horizon=args.horizon,
        sigma_interview=args.sigma_interview, sigma_match=args.sigma_match,
        outside_option=args.outside_option,
        non_negative_features=not args.gaussian_features,
    )
    env = HireRLEnv(env_cfg)
    rr_cfg = RRETCConfig(
        L=args.L, c_radius=args.c_radius, eps_w=args.eps_w, eps_f=args.eps_f,
        gate_enabled=not args.no_gate,
        strict_checks=not args.no_strict,
        compute_friction=not args.no_friction,
        verbose=not args.quiet,
    )
    algo = RoundRobinETC(env, env.default_const, jax.random.PRNGKey(args.seed), rr_cfg)

    print(f"[rr-etc] {args.Nw}x{args.Nf} d={args.d} horizon={args.horizon} "
          f"sigma_int={args.sigma_interview} seed={args.seed} L={args.L} "
          f"gate={'on' if rr_cfg.gate_enabled else 'OFF'}")
    summary = algo.run()

    n_periods = summary["market_periods"]
    print(f"[rr-etc] done in {summary['wall_seconds']:.1f}s "
          f"({n_periods / max(summary['wall_seconds'], 1e-9):.0f} periods/s)")
    print(f"[rr-etc] all_settled={summary['all_settled']} "
          f"matches_reference={summary['settled_matches_reference']} "
          f"rounds={summary['rounds']} mismatches={len(summary['commit_mismatches'])}")
    print(f"[rr-etc] final per-period true regret: "
          f"{summary['final_regret_true_total']:.6f} "
          f"(pos-part {summary['final_regret_true_pos_total']:.6f})")

    if args.outdir:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        _write_outputs(outdir, args, summary)
        print(f"[rr-etc] outputs written to {outdir}/")


def _write_outputs(outdir: Path, args, summary):
    import pandas as pd

    df = pd.DataFrame(summary["records"])
    df.to_csv(outdir / "metrics.csv", index=False)

    rl = pd.DataFrame(summary["round_log"])
    if len(rl):
        rl["committed"] = rl["committed"].apply(
            lambda c: ";".join(f"{i}->{j}" for i, j in c)
        )
    rl.to_csv(outdir / "round_log.csv", index=False)

    js = {
        "args": vars(args),
        "settled": {int(k): int(v) for k, v in summary["settled"].items()},
        "all_settled": bool(summary["all_settled"]),
        "settled_matches_reference": bool(summary["settled_matches_reference"]),
        "rounds": int(summary["rounds"]),
        "market_periods": int(summary["market_periods"]),
        "sched_misses": int(summary["sched_misses"]),
        "n_commit_mismatches": len(summary["commit_mismatches"]),
        "final_regret_true_total": float(summary["final_regret_true_total"]),
        "final_regret_true_pos_total": float(summary["final_regret_true_pos_total"]),
        "wall_seconds": float(summary["wall_seconds"]),
    }
    (outdir / "summary.json").write_text(json.dumps(js, indent=2))

    _plot(outdir, df)


def _plot(outdir: Path, df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    t = df["t"].to_numpy()

    ax = axes[0, 0]
    ax.plot(t, df["regret_true_pos_total"], lw=0.8)
    ax.set_title("Instantaneous player-optimal stable regret (true, pos-part)")
    ax.set_xlabel("market period")

    ax = axes[0, 1]
    ax.plot(t, np.cumsum(df["regret_true_pos_total"]), lw=1.2)
    ax.set_title("Cumulative regret")
    ax.set_xlabel("market period")

    ax = axes[1, 0]
    ax.plot(t, df["social_welfare_true"], lw=0.8, label="true")
    ax.plot(t, df["social_welfare"], lw=0.8, alpha=0.7, label="belief")
    ax.set_title("Social welfare (effective matching)")
    ax.set_xlabel("market period")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(t, df["n_settled"], lw=1.2, label="workers settled")
    ax.plot(t, df["match_rate"] * df["n_settled"].max() if df["n_settled"].max() else df["match_rate"],
            lw=0.6, alpha=0.5, label="match rate (scaled)")
    ax.set_title("Settlement progress")
    ax.set_xlabel("market period")
    ax.legend()

    fig.tight_layout()
    fig.savefig(outdir / "curves.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
