"""Self-tests for the oracle-assisted Round-Robin ETC baseline.

Run:  python test_rr_etc.py          (from this folder, with the repo venv)

Covers:
  1. Physical GS == worker_proposing_da reference under perfect knowledge.
  2. Structured 3x3 market (paper Example-1 preference pattern realized as a
     shared-utility matrix): converges to the unique true stable matching,
     with staged commits and clean invariants.
  3. Random 4x4 market (env-sampled half-normal features): all workers settle
     on the true stable matching; regret goes to ~0 after the last commit.
  4. Gate stress: big worker gaps + tiny firm gap. Workers become confident
     long before the firm can rank them; the gate must hold commits until the
     firm's empirical ranking is correct, and the final matching must be right.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "learn2match"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
import jax.numpy as jnp

from hirerl.config import HireRLConfig
from hirerl.env import HireRLEnv
from hirerl.metrics import worker_proposing_da

from rr_etc_baseline import RoundRobinETC, RRETCConfig


def _make(env_kwargs, rr_kwargs, seed=0):
    cfg = HireRLConfig(**env_kwargs)
    env = HireRLEnv(cfg)
    algo = RoundRobinETC(env, env.default_const, jax.random.PRNGKey(seed),
                         RRETCConfig(**rr_kwargs))
    return algo


def _check_invariants(summary, Nw, Nf):
    assert summary["sched_misses"] == 0, (
        f"exploration schedule produced {summary['sched_misses']} misses — "
        f"conflict-free scheduling is broken"
    )
    settled = summary["settled"]
    assert len(set(settled.values())) == len(settled), "two workers share a firm"
    n_settled_series = [r["n_settled"] for r in summary["records"]]
    assert all(a <= b for a, b in zip(n_settled_series, n_settled_series[1:])), (
        "n_settled must be monotone"
    )


# ---------------------------------------------------------------- test 1

def test_gs_equals_reference_under_perfect_knowledge():
    print("== test 1: physical GS == DA reference under perfect knowledge ==")
    rng = np.random.default_rng(7)
    for trial in range(3):
        Nw = Nf = 4
        U = rng.uniform(0.1, 1.0, size=(Nw, Nf))
        algo = _make(
            dict(Nw=Nw, Nf=Nf, d=Nf, horizon=400, sigma_interview=0.05,
                 sigma_match=0.01, non_negative_features=True),
            dict(L=0.1, verbose=False, compute_friction=False),
            seed=trial,
        )
        # Start an episode with constructed utilities: x = U rows, y = identity.
        algo.rng, r_reset = jax.random.split(algo.rng)
        state, _ = algo.env.reset(r_reset, algo.const)
        state = state.replace(x=jnp.asarray(U, jnp.float32),
                              y=jnp.asarray(np.eye(Nf), jnp.float32))
        algo.state = state
        algo._x = U.astype(np.float64)
        algo._y = np.eye(Nf)
        algo.U = U.copy()
        algo.sigma_w = np.linalg.norm(U, axis=1) * algo.sigma_interview
        algo.sigma_f = np.ones(Nf) * algo.sigma_interview
        algo.ref_matched = worker_proposing_da(
            jnp.asarray(U, jnp.float32), jnp.asarray(U, jnp.float32), outside=0.0)
        algo.ref_np = np.asarray(algo.ref_matched)
        algo.w_means = np.zeros((Nw, Nf)); algo.w_counts = np.zeros((Nw, Nf), np.int64)
        algo.f_means = np.zeros((Nf, Nw)); algo.f_counts = np.zeros((Nf, Nw), np.int64)
        algo.remaining = list(range(Nw)); algo.available = list(range(Nf))
        algo.index = {w: p for p, w in enumerate(algo.remaining)}
        algo.settled = {}; algo.market_period = 0; algo.round = 0
        algo.records = []; algo.round_log = []; algo.commit_mismatches = []
        algo._sched_misses = 0

        # One exploration rep to satisfy the env's interviewed[i, j] guard,
        # then overwrite the learned estimates with the truth.
        assert algo._exploration_block()
        algo.w_means = U.copy(); algo.w_counts[:] = 10**6
        algo.f_means = U.T.copy(); algo.f_counts[:] = 10**6

        assign = algo._gs_block()
        assert assign is not None
        got = np.zeros((Nw, Nf), bool)
        for i, j in assign.items():
            got[i, j] = True
        assert np.array_equal(got, algo.ref_np), (
            f"trial {trial}: GS {assign} != reference "
            f"{[(i, int(np.argmax(algo.ref_np[i]))) for i in range(Nw)]}"
        )
    print("   ok: 3/3 random utility matrices\n")


# ---------------------------------------------------------------- test 2

def test_example1_structured_market():
    print("== test 2: structured 3x3 (paper Example-1 pattern) ==")
    # Shared-utility matrix realizing Example 1's preference orders:
    #   rows  (workers): w0: f0>f1>f2 | w1: f2>f1>f0 | w2: f1>f2>f0
    #   cols  (firms):   f0: w0>w2>w1 | f1: w0>w2>w1 | f2: w2>w0>w1
    # Unique stable matching: w0-f0, w2-f1, w1-f2 (same as the paper's).
    U = np.array([
        [0.95, 0.85, 0.55],
        [0.20, 0.35, 0.45],
        [0.30, 0.75, 0.65],
    ])
    expected = {0: 0, 2: 1, 1: 2}
    for seed in range(3):
        algo = _make(
            dict(Nw=3, Nf=3, d=3, horizon=3000, sigma_interview=0.08,
                 sigma_match=0.01, non_negative_features=True),
            dict(L=0.5, verbose=False),
            seed=seed,
        )
        summary = algo.run(x_override=U, y_override=np.eye(3))
        assert summary["all_settled"], f"seed {seed}: not all settled"
        assert summary["settled"] == expected, (
            f"seed {seed}: settled {summary['settled']} != {expected}"
        )
        assert not summary["commit_mismatches"]
        _check_invariants(summary, 3, 3)
        assert summary["final_regret_true_total"] < 1e-6, (
            f"seed {seed}: final true regret {summary['final_regret_true_total']}"
        )
        # Staged exit is allowed but not required; commits must happen.
        n_commit_events = sum(1 for d in summary["round_log"] if d["committed"])
        assert n_commit_events >= 1
        print(f"   seed {seed}: settled {summary['settled']} in "
              f"{summary['rounds']} rounds, {summary['market_periods']} periods, "
              f"{n_commit_events} commit event(s)")
    print("   ok\n")


# ---------------------------------------------------------------- test 3

def test_random_market_env_features():
    print("== test 3: random 4x4 market (env-sampled features) ==")
    # sigma_interview=0.08: with half-normal d=4 features (||x|| ~ 2) the
    # effective sample noise is ~0.16, matched to the ~0.04 min gaps these
    # seeds draw. Seed 1 additionally exercises the staged-exit shrinkage:
    # its tightest pair involves a firm that stronger workers commit away,
    # so the residual market never needs to resolve it.
    for seed in [0, 1, 2]:
        algo = _make(
            dict(Nw=4, Nf=4, d=4, horizon=9000, sigma_interview=0.08,
                 sigma_match=0.02, non_negative_features=True),
            dict(L=1.0, verbose=False),
            seed=seed,
        )
        summary = algo.run()
        U = summary["U_true"]
        gaps_w = min(
            abs(U[i, a] - U[i, b])
            for i in range(4) for a in range(4) for b in range(a + 1, 4)
        )
        print(f"   seed {seed}: min worker gap {gaps_w:.4f}, "
              f"settled={summary['all_settled']} in {summary['rounds']} rounds "
              f"({summary['market_periods']} periods), "
              f"final true regret {summary['final_regret_true_total']:.5f}")
        assert summary["all_settled"], f"seed {seed}: horizon too short to settle"
        assert summary["settled_matches_reference"], (
            f"seed {seed}: settled {summary['settled']} != reference"
        )
        assert not summary["commit_mismatches"]
        _check_invariants(summary, 4, 4)
        assert summary["final_regret_true_total"] < 1e-6
    print("   ok\n")


# ---------------------------------------------------------------- test 4

def test_gate_blocks_early_commit():
    print("== test 4: arm-readiness gate under worker/firm learning asymmetry ==")
    # Both workers love f0 (worker-side gaps ~0.7 => confident almost
    # immediately); firm f0's true margin between them is only 0.05, so its
    # empirical order flips for a while. Without the gate, the first GS after
    # worker confidence would flip a coin and lock the result forever.
    U = np.array([
        [0.90, 0.20],
        [0.85, 0.15],
    ])
    expected = {0: 0, 1: 1}
    blocked_runs = 0
    for seed in range(4):
        algo = _make(
            dict(Nw=2, Nf=2, d=2, horizon=4000, sigma_interview=0.12,
                 sigma_match=0.01, non_negative_features=True),
            dict(L=0.25, verbose=False),
            seed=seed,
        )
        summary = algo.run(x_override=U, y_override=np.eye(2))
        assert summary["all_settled"], f"seed {seed}: not settled"
        assert summary["settled"] == expected, (
            f"seed {seed}: settled {summary['settled']} != {expected}"
        )
        assert not summary["commit_mismatches"]
        _check_invariants(summary, 2, 2)
        gate_blocks = [
            d for d in summary["round_log"]
            if not d["gate_ok"] and d["n_confident"] == d["n_remaining"]
        ]
        if gate_blocks:
            blocked_runs += 1
        print(f"   seed {seed}: rounds={summary['rounds']}, gate blocked a "
              f"fully-confident checkpoint {len(gate_blocks)} time(s)")
    assert blocked_runs >= 1, (
        "the gate never engaged across 4 seeds — asymmetry construction "
        "is not stressing it"
    )
    print(f"   ok (gate engaged in {blocked_runs}/4 runs)\n")


if __name__ == "__main__":
    t0 = __import__("time").time()
    test_gs_equals_reference_under_perfect_knowledge()
    test_example1_structured_market()
    test_random_market_env_features()
    test_gate_blocks_early_commit()
    print(f"ALL TESTS PASSED in {__import__('time').time() - t0:.1f}s")
