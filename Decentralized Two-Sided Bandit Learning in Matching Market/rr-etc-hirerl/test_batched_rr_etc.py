"""Tests for the batched RR-ETC driver.

Run:  python test_batched_rr_etc.py

1. Reset-derivation pairing: per-seed (x, y) bit-identical to
   BatchedCAETCBaseline built with the same base_rng.
2. Structured Example-1 market broadcast to 4 seeds: every seed must settle
   on the known unique stable matching (deterministic driver check —
   independent noise per seed, same market).
3. Random 6-seed batch: driver invariants under mixed feasibility. Seeds
   whose residual gaps fit the horizon must settle exactly; infeasible seeds
   must keep exploring gracefully (no mismatches, no misses, never "done"
   with workers left over).
"""

from __future__ import annotations

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


def test_market_pairing_with_ca_etc():
    print("== batched test 1: market pairing with BatchedCAETCBaseline ==")
    from batched_ca_etc_baseline import BatchedCAETCBaseline

    env = HireRLEnv(HireRLConfig(
        Nw=4, Nf=4, d=3, horizon=60, sigma_interview=0.2,
        sigma_match=0.05, non_negative_features=True,
    ))
    rr = BatchedRoundRobinETC(
        env, num_seeds=5, base_rng=jax.random.PRNGKey(123),
        cfg=RRETCConfig(verbose=False, compute_friction=False),
    )
    ca = BatchedCAETCBaseline(
        env=env, num_seeds=5, base_rng=jax.random.PRNGKey(123),
        Nw=4, Nf=4, horizon=60, T0=4, gamma=0.4,
    )
    x_rr = np.stack([s._x for s in rr.seeds])
    y_rr = np.stack([s._y for s in rr.seeds])
    assert np.array_equal(x_rr.astype(np.float32), np.asarray(ca.state.x))
    assert np.array_equal(y_rr.astype(np.float32), np.asarray(ca.state.y))
    print("   ok: per-seed (x, y) bit-identical across baselines\n")


def test_structured_market_all_seeds():
    print("== batched test 2: Example-1 market broadcast to 4 seeds ==")
    U = np.array([
        [0.95, 0.85, 0.55],
        [0.20, 0.35, 0.45],
        [0.30, 0.75, 0.65],
    ])
    expected = {0: 0, 2: 1, 1: 2}
    env = HireRLEnv(HireRLConfig(
        Nw=3, Nf=3, d=3, horizon=3000, sigma_interview=0.08,
        sigma_match=0.01, non_negative_features=True,
    ))
    algo = BatchedRoundRobinETC(
        env, num_seeds=4, base_rng=jax.random.PRNGKey(0),
        cfg=RRETCConfig(L=0.5, verbose=False, compute_friction=False),
        x_override=U, y_override=np.eye(3),
    )
    hist = algo.run()
    print(f"   wall {hist['wall_seconds']:.1f}s "
          f"({hist['market_periods'] / hist['wall_seconds']:.0f} periods/s x "
          f"{hist['num_seeds']} seeds)")
    assert hist["total_sched_misses"] == 0
    for s in hist["seed_summaries"]:
        assert s["all_settled"], f"seed {s['sid']} did not settle"
        assert s["settled"] == expected, (
            f"seed {s['sid']}: {s['settled']} != {expected}"
        )
        assert not s["commit_mismatches"]
    final_pos = hist["regret_true_pos_total"][:, -1]
    assert np.all(final_pos < 1e-5), final_pos
    print(f"   ok: 4/4 seeds settled on {expected}\n")


def test_random_batch_mixed_feasibility():
    print("== batched test 3: 6-seed random 4x4, mixed feasibility ==")
    env = HireRLEnv(HireRLConfig(
        Nw=4, Nf=4, d=4, horizon=9000, sigma_interview=0.08,
        sigma_match=0.02, non_negative_features=True,
    ))
    algo = BatchedRoundRobinETC(
        env, num_seeds=6, base_rng=jax.random.PRNGKey(0),
        cfg=RRETCConfig(L=1.0, verbose=False, compute_friction=False),
    )
    hist = algo.run()
    print(f"   wall {hist['wall_seconds']:.1f}s "
          f"({hist['market_periods'] / hist['wall_seconds']:.0f} periods/s x "
          f"{hist['num_seeds']} seeds)")
    n_settled = 0
    for s in hist["seed_summaries"]:
        print(f"   seed {s['sid']}: settled={s['all_settled']} "
              f"correct={s['settled_matches_reference']} rounds={s['rounds']}")
        assert s["sched_misses"] == 0
        assert not s["commit_mismatches"], (
            f"seed {s['sid']} committed a wrong pair: {s['commit_mismatches']}"
        )
        if s["all_settled"]:
            n_settled += 1
            assert s["settled_matches_reference"]
    # Known split for base_rng=PRNGKey(0) at this noise level: seeds 0, 2, 4
    # are horizon-feasible (partly thanks to residual-market shrinkage);
    # seeds 1, 3, 5 have pairs needing 4k-27k samples vs a ~2.2k budget.
    assert n_settled >= 3, f"only {n_settled}/6 seeds settled"
    final_pos = hist["regret_true_pos_total"][:, -1]
    for s in hist["seed_summaries"]:
        if s["all_settled"]:
            assert final_pos[s["sid"]] < 1e-5, (s["sid"], final_pos[s["sid"]])
    print(f"   ok: {n_settled}/6 settled exactly; the rest explored "
          f"gracefully to the horizon\n")


if __name__ == "__main__":
    import time
    t0 = time.time()
    test_market_pairing_with_ca_etc()
    test_structured_market_all_seeds()
    test_random_batch_mixed_feasibility()
    print(f"ALL BATCHED TESTS PASSED in {time.time() - t0:.1f}s")
