"""RR-ETC vs CA-ETC: cumulative player/arm regret on identical markets.

Runs both batched baselines with the SAME base_rng and num_seeds, which makes
the i-th seed's (x, y) market bit-identical across the two algorithms (both
use the derivation `split(base_rng)[1] -> BatchedEnv.reset`). The script
verifies this pairing explicitly before running.

Outputs (into --outdir):
    cumulative_worker_regret.csv   t, rr_mean, rr_std, ca_mean, ca_std
    cumulative_firm_regret.csv     same schema
    compare_cumregret.png          two panels (workers / firms), mean +- 1 std
    summary.json

Example:
    python compare_rr_vs_ca.py --num-seeds 10 --horizon 8000 --outdir compare_5x5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_CA_DIR = _REPO_ROOT / "learn2match" / "ca-etc-hirerl"
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "learn2match"), str(_HERE), str(_CA_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax

from hirerl.config import HireRLConfig
from hirerl.env import HireRLEnv

from rr_etc_baseline import RRETCConfig
from batched_rr_etc import BatchedRoundRobinETC
from batched_ca_etc_baseline import BatchedCAETCBaseline


class _LeanCAETC(BatchedCAETCBaseline):
    """CA-ETC with a light history materialization.

    The parent stacks every per-period record with one jnp.stack whose operand
    count equals the horizon — fine at 8k periods, pathological at 150k. The
    comparison only needs the two regret totals, so pull those to host
    record-by-record and drop the per-agent / matched-matrix payloads.
    """

    def _materialize_history(self):
        keys = ("regret_w_total", "regret_f_total")
        out = {
            k: np.stack([np.asarray(r[k]) for r in self._records], axis=1)
            for k in keys
        }
        out["t"] = np.asarray(self._record_t, dtype=np.int32)
        return out


def gap_floor_market(Nw: int, Nf: int):
    """Rank-1 shared-utility market U = a x b with hard gap floors.

    a = linspace(3.0, 1.0, Nw), b = linspace(2.0, 0.4, Nf). For 5x5:
    worker-side adjacent gaps a_i*0.4 >= 0.4, firm-side gaps b_j*0.5 >= 0.2 —
    the paper's Appendix-E gap floor. Globally-ranked preferences on both
    sides (maximal conflict: everyone wants the same top firm), unique stable
    matching = assortative (worker i <-> firm i). Realized in HireRL with
    d = 1: x_i = [a_i], y_j = [b_j], so sigma_eff scales with each agent's
    own utility scale and sigma_int/gap is uniform across workers.
    """
    a = np.linspace(3.0, 1.0, Nw)
    b = np.linspace(2.0, 0.4, Nf)
    x = a.reshape(Nw, 1).astype(np.float32)
    y = b.reshape(Nf, 1).astype(np.float32)
    return x, y


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--Nw", type=int, default=5)
    ap.add_argument("--Nf", type=int, default=5)
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=8000)
    ap.add_argument("--num-seeds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0, help="base PRNG key")
    ap.add_argument("--sigma-interview", type=float, default=0.08)
    ap.add_argument("--sigma-match", type=float, default=0.02)
    ap.add_argument("--outside-option", type=float, default=0.0)
    # RR-ETC knobs
    ap.add_argument("--L", type=float, default=1.0)
    ap.add_argument("--c-radius", type=float, default=1.0)
    ap.add_argument("--eps-w", type=float, default=0.0)
    ap.add_argument("--eps-f", type=float, default=0.0)
    # CA-ETC knobs (repo defaults: T0=ceil(Nw/Nf)*Nf, gamma=0.4, radius 0.5)
    ap.add_argument("--T0", type=int, default=None)
    ap.add_argument("--gamma", type=float, default=0.4)
    ap.add_argument("--radius-scale", type=float, default=0.5)
    ap.add_argument("--gap-floor", action="store_true",
                    help="use the constructed rank-1 market with hard gap "
                         "floors (paper Appendix-E style) instead of random "
                         "env-sampled features; forces d=1 and injects the "
                         "same market into both baselines")
    ap.add_argument("--outdir", type=str, default="compare_out")
    # W&B (same conventions as plot_batched_ca_etc_baseline.py: on by
    # default, --no-wandb to skip). Curves are downsampled to <= ~5000
    # logged steps so a 150k-period run uploads in seconds.
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-entity", default="haijingzong-university-of-washington")
    ap.add_argument("--wandb-project", default="hireRL")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    if args.gap_floor:
        args.d = 1

    env_cfg = HireRLConfig(
        Nw=args.Nw, Nf=args.Nf, d=args.d, horizon=args.horizon,
        sigma_interview=args.sigma_interview, sigma_match=args.sigma_match,
        outside_option=args.outside_option,
        non_negative_features=True,
    )
    env = HireRLEnv(env_cfg)
    S = args.num_seeds

    print(f"[compare] {args.Nw}x{args.Nf} d={args.d} horizon={args.horizon} "
          f"seeds={S} sigma_int={args.sigma_interview}")

    # --- RR-ETC (batched) --------------------------------------------------
    rr_cfg = RRETCConfig(
        L=args.L, c_radius=args.c_radius, eps_w=args.eps_w, eps_f=args.eps_f,
        compute_friction=False, verbose=True,
        # For the comparison we want the run to complete even if a tolerance
        # config ever mis-commits; mismatches are still counted and reported.
        strict_checks=False,
    )
    x_gf = y_gf = None
    if args.gap_floor:
        x_gf, y_gf = gap_floor_market(args.Nw, args.Nf)
        print("[compare] gap-floor market: worker gaps >= "
              f"{np.min(np.diff(np.sort((x_gf @ y_gf.T), axis=1), axis=1)):.2f}, "
              "unique stable matching = assortative")

    rr = BatchedRoundRobinETC(env, num_seeds=S,
                              base_rng=jax.random.PRNGKey(args.seed), cfg=rr_cfg,
                              x_override=x_gf, y_override=y_gf)

    # --- CA-ETC (batched, repo implementation) ------------------------------
    ca = _LeanCAETC(
        env=env, num_seeds=S, base_rng=jax.random.PRNGKey(args.seed),
        Nw=args.Nw, Nf=args.Nf, horizon=args.horizon,
        T0=args.T0, gamma=args.gamma, radius_scale=args.radius_scale,
    )
    if args.gap_floor:
        # Inject the same constructed market (state surgery at episode start,
        # before any interaction — the same hook BatchedRoundRobinETC exposes).
        import jax.numpy as jnp
        S_ = ca.state.x.shape[0]
        ca.state = ca.state.replace(
            x=jnp.broadcast_to(jnp.asarray(x_gf), (S_, *x_gf.shape)),
            y=jnp.broadcast_to(jnp.asarray(y_gf), (S_, *y_gf.shape)),
        )

    # Market pairing check: bit-identical latent features per seed.
    x_rr = np.stack([s._x for s in rr.seeds])
    x_ca = np.asarray(ca.state.x)
    assert np.allclose(x_rr, x_ca, atol=0), "seed markets are NOT paired!"
    print(f"[compare] market pairing verified (max|dx| = "
          f"{np.abs(x_rr - x_ca).max():.2e})")

    print("[compare] running RR-ETC ...")
    rr_hist = rr.run()
    print(f"[compare] RR-ETC done in {rr_hist['wall_seconds']:.1f}s — "
          f"{rr_hist['n_all_settled']}/{S} seeds fully settled, "
          f"{rr_hist['n_correct']}/{S} on the exact stable matching, "
          f"sched_misses={rr_hist['total_sched_misses']}")

    print("[compare] running CA-ETC ...")
    import time as _time
    t0 = _time.time()
    ca_hist = ca.run()
    print(f"[compare] CA-ETC done in {_time.time() - t0:.1f}s")

    # --- align + cumulate ----------------------------------------------------
    T = min(rr_hist["regret_w_total"].shape[1], ca_hist["regret_w_total"].shape[1])
    t_axis = np.arange(1, T + 1)

    def cum(h, key):
        return np.cumsum(h[key][:, :T], axis=1)          # (S, T)

    curves = {
        "worker": (cum(rr_hist, "regret_w_total"), cum(ca_hist, "regret_w_total")),
        "firm":   (cum(rr_hist, "regret_f_total"), cum(ca_hist, "regret_f_total")),
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    stats = {}
    for side, (rr_c, ca_c) in curves.items():
        df = pd.DataFrame({
            "t": t_axis,
            "rr_mean": rr_c.mean(axis=0), "rr_std": rr_c.std(axis=0),
            "ca_mean": ca_c.mean(axis=0), "ca_std": ca_c.std(axis=0),
        })
        df.to_csv(outdir / f"cumulative_{side}_regret.csv", index=False)
        stats[side] = {
            "rr_final_mean": float(rr_c[:, -1].mean()),
            "rr_final_std": float(rr_c[:, -1].std()),
            "ca_final_mean": float(ca_c[:, -1].mean()),
            "ca_final_std": float(ca_c[:, -1].std()),
        }

    _plot(outdir, t_axis, curves, args)

    js = {
        "args": vars(args),
        "rr": {
            "n_all_settled": int(rr_hist["n_all_settled"]),
            "n_correct": int(rr_hist["n_correct"]),
            "wall_seconds": float(rr_hist["wall_seconds"]),
        },
        "final_cumulative_regret": stats,
    }
    (outdir / "summary.json").write_text(json.dumps(js, indent=2))

    for side in ("worker", "firm"):
        s = stats[side]
        print(f"[compare] final cumulative {side} regret:  "
              f"RR-ETC {s['rr_final_mean']:.0f} ± {s['rr_final_std']:.0f}   "
              f"CA-ETC {s['ca_final_mean']:.0f} ± {s['ca_final_std']:.0f}")
    print(f"[compare] outputs in {outdir}/")

    if not args.no_wandb:
        _log_wandb(args, outdir, t_axis, curves, stats, rr_hist)


def _log_wandb(args, outdir: Path, t_axis, curves, stats, rr_hist):
    """Upload curves + figure + summary to W&B.

    Never lets a W&B failure (missing login, no network) destroy a finished
    run — local CSV/PNG outputs are already on disk by the time this runs.
    """
    from datetime import datetime
    try:
        import wandb

        run_name = args.run_name or (
            f"rr-vs-ca-{args.Nw}x{args.Nf}-h{args.horizon}-N{args.num_seeds}-"
            f"seed{args.seed}-{datetime.now():%Y%m%d-%H%M%S}"
        )
        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            config={"algorithm": "rr_etc_vs_ca_etc_paired", **vars(args)},
        )

        T = len(t_axis)
        stride = max(1, T // 5000)
        idxs = list(range(0, T, stride))
        if idxs[-1] != T - 1:
            idxs.append(T - 1)
        series = {}
        for side, (rr_c, ca_c) in curves.items():
            series[f"rr/cum_regret_{side}_mean"] = rr_c.mean(axis=0)
            series[f"rr/cum_regret_{side}_std"] = rr_c.std(axis=0)
            series[f"ca/cum_regret_{side}_mean"] = ca_c.mean(axis=0)
            series[f"ca/cum_regret_{side}_std"] = ca_c.std(axis=0)
        for i in idxs:
            wandb.log({k: float(v[i]) for k, v in series.items()},
                      step=int(t_axis[i]))

        wandb.log({"compare_cumregret": wandb.Image(
            str(outdir / "compare_cumregret.png"))})
        # Attach the full-resolution outputs to the run (Files tab) — the
        # logged curves above are downsampled, these are the exact data.
        for fname in ("cumulative_worker_regret.csv",
                      "cumulative_firm_regret.csv", "summary.json"):
            p = outdir / fname
            if p.exists():
                wandb.save(str(p), base_path=str(outdir), policy="now")
        run.summary.update({
            "rr_n_all_settled": int(rr_hist["n_all_settled"]),
            "rr_n_correct": int(rr_hist["n_correct"]),
            **{f"final/{side}_{algo}_{stat}": stats[side][f"{algo}_final_{stat}"]
               for side in ("worker", "firm")
               for algo in ("rr", "ca")
               for stat in ("mean", "std")},
        })
        wandb.finish()
        print(f"[compare] W&B run uploaded: {run_name}")
    except Exception as e:                                    # noqa: BLE001
        print(f"[compare] WARNING: W&B upload failed ({e!r}); "
              f"local outputs in {outdir}/ are unaffected. "
              f"Re-run with --no-wandb to silence this.")


def _plot(outdir: Path, t, curves, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    titles = {
        "worker": "Cumulative regret — players (workers)",
        "firm": "Cumulative regret — arms (firms)",
    }
    colors = {"rr": "#1f77b4", "ca": "#d62728"}
    labels = {"rr": "Round-Robin ETC (oracle-assisted)", "ca": "CA-ETC"}

    for ax, side in zip(axes, ("worker", "firm")):
        rr_c, ca_c = curves[side]
        for key, arr in (("rr", rr_c), ("ca", ca_c)):
            m, sd = arr.mean(axis=0), arr.std(axis=0)
            ax.plot(t, m, color=colors[key], lw=1.6, label=labels[key])
            ax.fill_between(t, m - sd, m + sd, color=colors[key], alpha=0.18, lw=0)
        ax.set_title(titles[side])
        ax.set_xlabel("market period")
        ax.set_ylabel("cumulative regret")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.25)

    fig.suptitle(
        f"{args.Nw}x{args.Nf} market, d={args.d}, "
        f"σ_int={args.sigma_interview}, {args.num_seeds} paired seeds "
        f"(mean ± 1 std)"
    )
    fig.tight_layout()
    fig.savefig(outdir / "compare_cumregret.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
