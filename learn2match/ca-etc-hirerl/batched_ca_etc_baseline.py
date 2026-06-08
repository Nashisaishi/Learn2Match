"""Vmapped CA-ETC baseline that runs N seeds in parallel.

Drop-in companion to :mod:`ca_etc_baseline`. Same algorithm, same metrics
schema, but:

* The env step is vmapped over a leading ``num_seeds`` axis via
  :class:`jax_pbt.env.batched_env.BatchedEnv`.
* Per-seed CA-ETC bookkeeping (mean rewards, counts) lives on device as
  ``jnp.ndarray`` of shape ``(N, ...)``; updates are jit'd.
* Preference estimation (with optional LCB/UCB confidence check) is jit'd.
* Gale-Shapley is implemented by converting per-seed preference lists into
  dense utility matrices and vmapping the env's worker-proposing DA from
  ``hirerl.metrics.worker_proposing_da``.
* One full market period (5 phase steps + bookkeeping + metrics) is one
  jit'd call. Metrics accumulate on device and are pulled back to host
  exactly once per period (a single ``np.asarray`` per scalar field).

Why this is faster
------------------
The single-seed baseline pays ~5 host↔device dispatches per market period
just to drive the env, plus another full host pull for ``np.asarray`` on
the entire state inside the exploration reward computation. With N=64
seeds, vmap turns those into a single dispatch shared across all 64
seeds, and bookkeeping never leaves the device.

All N seeds share the same epoch sequence (T0·l^c, T0·l^b) and the same
deterministic round-robin schedule -- those depend only on
hyperparameters, not on per-seed state. Per-seed differences are entirely
in (a) initial x/y, (b) interview/match noise, (c) the resulting
empirical means and the GS matchings derived from them.

Public surface
--------------
* :class:`BatchedCAETCBaseline` -- analogue of ``CAETCBaseline``. ``run()``
  returns a numpy-friendly history dict where every metric is shape
  ``(N, T)`` or ``(N, T, ...)``.
"""

from __future__ import annotations

import functools
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

# Make jax_pbt and the local hirerl package importable when this file is run
# directly (e.g. python learn2match/ca-etc-hirerl/batched_ca_etc_baseline.py
# from the repo root).
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)              # learn2match/
_GRANDPARENT = os.path.dirname(_PARENT)       # repo root (jax_pbt lives here)
for _p in (_HERE, _PARENT, _GRANDPARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
import jax.numpy as jnp
import numpy as np

from jax_pbt.env.batched_env import BatchedEnv
from hirerl.metrics import (
    belief_induced_match,
    reference_match,
    worker_proposing_da,
)


# ---------------------------------------------------------------------------
# Per-seed metric kernel (single state -> dict; vmapped where needed)
# ---------------------------------------------------------------------------


def _effective_metrics_single(state, outside: float) -> Dict[str, jax.Array]:
    """Hirerl-gauge metrics for a SINGLE (un-batched) state.

    Mirrors ``ca_etc_baseline._compute_effective_metrics`` exactly, but
    without the leading seed dim, so it can be vmapped from outside.
    """
    effective = state.matched | state.tentative_matched
    eff_f = effective.astype(jnp.float32)

    U_w_belief = jnp.sum(state.x[:, None, :] * state.hat_y, axis=-1)
    U_f_belief = jnp.sum(state.hat_x * state.y[None, :, :], axis=-1)
    U_true = jnp.einsum("id,jd->ij", state.x, state.y)

    ref_matched = reference_match(state, outside)
    ref_f = ref_matched.astype(jnp.float32)

    reward_w_per = jnp.sum(eff_f * U_w_belief, axis=1)
    reward_f_per = jnp.sum(eff_f * U_f_belief, axis=0)

    ref_per_w = jnp.sum(ref_f * U_true, axis=1)
    ref_per_f = jnp.sum(ref_f * U_true, axis=0)
    regret_w_per = ref_per_w - reward_w_per
    regret_f_per = ref_per_f - reward_f_per

    diff_y_sq = jnp.sum((state.hat_y - state.y[None, :, :]) ** 2, axis=-1)
    diff_x_sq = jnp.sum((state.hat_x - state.x[:, None, :]) ** 2, axis=-1)
    info_loss_w_per = jnp.sum(eff_f * diff_y_sq, axis=1)
    info_loss_f_per = jnp.sum(eff_f * diff_x_sq, axis=0)

    bi_matched = belief_induced_match(state, outside)
    bi_f = bi_matched.astype(jnp.float32)
    friction = jnp.sum(ref_f * U_true) - jnp.sum(bi_f * U_true)

    social_welfare = jnp.sum(eff_f * (U_w_belief + U_f_belief))
    # Apples-to-apples gauge for compare-with-PPO: same formulas as above but
    # on ``state.matched`` only (post-RETENTION committed pairs), matching
    # ``hirerl/metrics.py``. CA-ETC's exploration always NOOPs RETENTION so
    # ``matched_only`` is 0 throughout exploration and only nonzero after the
    # commit transition; the committed-gauge metrics therefore read 0 during
    # exploration and "settle" once exploitation begins. Saved alongside the
    # effective-gauge fields so callers pick which gauge to plot at compare
    # time without re-running.
    matched_only = state.matched.astype(jnp.float32)
    social_welfare_committed = jnp.sum(matched_only * (U_w_belief + U_f_belief))
    reward_w_per_committed = jnp.sum(matched_only * U_w_belief, axis=1)
    reward_f_per_committed = jnp.sum(matched_only * U_f_belief, axis=0)
    regret_w_total_committed = jnp.sum(ref_per_w - reward_w_per_committed)
    regret_f_total_committed = jnp.sum(ref_per_f - reward_f_per_committed)
    # friction_loss is gauge-independent: it depends only on ref_f and the
    # belief-induced match bi_f, neither of which involves state.matched or
    # state.tentative_matched. Same value under either gauge -- no
    # *_committed twin needed.

    Nw, Nf = effective.shape
    norm = jnp.float32(max(min(Nw, Nf), 1))
    match_rate = jnp.sum(eff_f) / norm

    return {
        "matched_matrix": effective,
        "reward_w_per_agent": reward_w_per,
        "reward_f_per_agent": reward_f_per,
        "regret_w_per_agent": regret_w_per,
        "regret_f_per_agent": regret_f_per,
        "info_loss_w_per_agent": info_loss_w_per,
        "info_loss_f_per_agent": info_loss_f_per,
        "reward_w_total": jnp.sum(reward_w_per),
        "reward_f_total": jnp.sum(reward_f_per),
        "regret_w_total": jnp.sum(regret_w_per),
        "regret_f_total": jnp.sum(regret_f_per),
        "info_loss_w_total": jnp.sum(info_loss_w_per),
        "info_loss_f_total": jnp.sum(info_loss_f_per),
        "social_welfare": social_welfare,
        "social_welfare_committed": social_welfare_committed,
        "regret_w_total_committed": regret_w_total_committed,
        "regret_f_total_committed": regret_f_total_committed,
        "friction_loss": friction,
        "match_rate": match_rate,
        "x_snap": state.x,
        "y_snap": state.y,
    }


def _vmapped_metrics(state_batch, outside: float) -> Dict[str, jax.Array]:
    """``_effective_metrics_single`` over the leading seed dim."""
    return jax.vmap(_effective_metrics_single, in_axes=(0, None))(state_batch, outside)


# ---------------------------------------------------------------------------
# Preference estimation (vectorized)
# ---------------------------------------------------------------------------


def _check_naive_separated(
    means: jax.Array,
    counts: jax.Array,
    t_explore: jax.Array,
    radius_scale: float,
):
    """Per-(seed, side) check: is the mean-sorted permutation LCB/UCB-separated?

    Mirrors ``CAETCBaseline._separated_perm`` but only verifies the natural
    permutation (descending mean). The brute-force fallback over all K!
    permutations is intentionally dropped: when the natural order isn't
    separated, the baseline now falls back to the same mean-sorted
    permutation, so brute-forcing the rest doesn't help.

    ``radius_scale`` multiplies the textbook sqrt(2 log T / n) radius. Values
    below 1 tighten the intervals so adjacent ranks separate sooner.

    Returns ``(naive_perm, is_separated)`` of shapes ``(..., K)`` and ``(...)``.
    """
    naive = jnp.argsort(-means, axis=-1)
    n_safe = jnp.maximum(counts, 1)
    log_factor = jnp.maximum(jnp.log2(jnp.maximum(t_explore.astype(jnp.float32), 2.0)), 1e-9)
    radius = jnp.where(
        counts >= 1,
        radius_scale * jnp.sqrt(2.0 * log_factor / n_safe.astype(jnp.float32)),
        jnp.inf,
    )
    ucb = means + radius
    lcb = means - radius
    naive_lcb = jnp.take_along_axis(lcb, naive, axis=-1)
    naive_ucb = jnp.take_along_axis(ucb, naive, axis=-1)
    is_separated = jnp.all(naive_lcb[..., :-1] > naive_ucb[..., 1:], axis=-1)
    return naive, is_separated


def _estimate_preferences(
    worker_means: jax.Array,
    worker_counts: jax.Array,
    firm_means: jax.Array,
    firm_counts: jax.Array,
    t_explore: jax.Array,
    use_confidence_check: bool,
    radius_scale: float,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Returns ``(worker_prefs, firm_prefs, w_separated, f_separated)``.

    ``w_separated`` and ``f_separated`` are ``(N, Nw)`` / ``(N, Nf)`` bool
    arrays marking which (seed, agent) actually had LCB/UCB-separated
    confidence intervals at the current ``t_explore``. The driver prints a
    seed-mean count per epoch so the user can see how saturated learning is.

    Whether ``use_confidence_check`` is True or False, the returned
    permutation is always the mean-sorted (naive) one — the confidence
    check now only affects the diagnostic ``*_separated`` flags. There is
    no identity fallback: failing the separation check no longer degrades
    the ranking, only the confidence label.
    """
    w_naive, w_sep = _check_naive_separated(
        worker_means, worker_counts, t_explore, radius_scale
    )
    f_naive, f_sep = _check_naive_separated(
        firm_means, firm_counts, t_explore, radius_scale
    )
    return w_naive.astype(jnp.int32), f_naive.astype(jnp.int32), w_sep, f_sep


# ---------------------------------------------------------------------------
# Gale-Shapley (worker-proposing) via vmap of metrics.worker_proposing_da
# ---------------------------------------------------------------------------


def _gale_shapley(worker_prefs: jax.Array, firm_prefs: jax.Array) -> jax.Array:
    """Worker-proposing GS, vmapped over the leading seed dim.

    Args:
        worker_prefs: ``(N, Nw, Nf)`` -- worker_prefs[n, i] is i's preference
            order over firms (firm ids, descending utility).
        firm_prefs:   ``(N, Nf, Nw)`` -- analogous on the firm side.

    Returns:
        worker_to_firm: ``(N, Nw)`` int32; ``-1`` if worker is unmatched.
    """
    N, Nw, Nf = worker_prefs.shape

    # Convert preferences -> dense utility matrices.
    # U_w[n, i, worker_prefs[n, i, k]] = (Nf - k); top choice has highest util.
    # All utilities are >= 1, so the >= outside (=0) acceptability check in
    # worker_proposing_da is trivially satisfied for in-list firms.
    arange_n = jnp.arange(N)[:, None, None]
    arange_w = jnp.arange(Nw)[None, :, None]
    pos_f = jnp.arange(Nf, dtype=jnp.float32)[None, None, :]
    U_w = jnp.zeros((N, Nw, Nf), dtype=jnp.float32)
    U_w = U_w.at[arange_n, arange_w, worker_prefs].set(jnp.float32(Nf) - pos_f)

    arange_f = jnp.arange(Nf)[None, :, None]
    pos_w = jnp.arange(Nw, dtype=jnp.float32)[None, None, :]
    U_f_jw = jnp.zeros((N, Nf, Nw), dtype=jnp.float32)
    U_f_jw = U_f_jw.at[arange_n, arange_f, firm_prefs].set(jnp.float32(Nw) - pos_w)
    U_f = U_f_jw.transpose(0, 2, 1)  # (N, Nw, Nf)

    matched = jax.vmap(worker_proposing_da, in_axes=(0, 0, None))(U_w, U_f, 0.0)
    # matched: (N, Nw, Nf) bool
    any_match = jnp.any(matched, axis=-1)
    firm_idx = jnp.argmax(matched.astype(jnp.int32), axis=-1)
    worker_to_firm = jnp.where(any_match, firm_idx, -1).astype(jnp.int32)
    return worker_to_firm


# ---------------------------------------------------------------------------
# Round-robin schedule (deterministic, identical across seeds)
# ---------------------------------------------------------------------------


def _round_robin_schedule(round_id: int, Nw: int, Nf: int) -> np.ndarray:
    """Same logic as ``CAETCBaseline.round_robin_schedule``, but returns an
    ``(Nw,)`` numpy int array with ``-1`` for unscheduled workers.

    For ``Nw <= Nf``: every worker is scheduled to ``(i + round_id) % Nf``.
    For ``Nw >  Nf``: workers are batched; only ``Nf`` workers fire per round.
    """
    scheduled = np.full(Nw, -1, dtype=np.int32)
    if Nw == 0 or Nf == 0:
        return scheduled
    if Nw <= Nf:
        for i in range(Nw):
            scheduled[i] = (i + round_id) % Nf
        return scheduled
    firms_per_period = Nf
    num_batches = (Nw + firms_per_period - 1) // firms_per_period
    batch_id = round_id % num_batches
    firm_offset = round_id // num_batches
    for k in range(firms_per_period):
        i = batch_id * firms_per_period + k
        if i >= Nw:
            break
        scheduled[i] = (k + firm_offset) % Nf
    return scheduled


# ---------------------------------------------------------------------------
# Phase-kernel factories. Built per-instance to capture (bench, const, Nw, Nf,
# N, outside) as jit constants without paying re-tracing per call.
# ---------------------------------------------------------------------------


def _build_explore_period(bench: BatchedEnv, const_batch, Nw: int, Nf: int, N: int, outside: float):
    """Returns a jit'd kernel for one exploration market period.

    Inputs:
        rng: jax.random key. Split into 5 sub-keys inside.
        state: batched env state (leading dim N).
        w_means, w_counts: (N, Nw, Nf)
        f_means, f_counts: (N, Nf, Nw)
        schedule: (Nw,) int32 -- scheduled firm per worker, -1 for unscheduled.
            Same across seeds since the round-robin schedule is deterministic.

    Outputs:
        new (state, w_means, w_counts, f_means, f_counts, record_dict).
    """

    @jax.jit
    def kernel(rng, state, w_means, w_counts, f_means, f_counts, schedule):
        # ---- Build per-seed action arrays from the (Nw,) schedule. ----
        # worker_action[n, i] = schedule[i] + 1 if scheduled, else 0 (NOOP).
        scheduled_mask_w = schedule >= 0                          # (Nw,) bool
        worker_act_single = jnp.where(scheduled_mask_w, schedule + 1, 0).astype(jnp.int32)

        # firm_action[n, j] = i + 1 iff schedule[i] == j; 0 otherwise.
        # Round-robin guarantees at most one i per j, so this is well-defined.
        sched_2d = (schedule[:, None] == jnp.arange(Nf)[None, :])     # (Nw, Nf)
        worker_id_plus_one = jnp.arange(Nw, dtype=jnp.int32) + 1
        firm_act_single = jnp.sum(sched_2d.astype(jnp.int32) * worker_id_plus_one[:, None], axis=0)
        firm_act_single = firm_act_single.astype(jnp.int32)            # (Nf,)

        zeros_w = jnp.zeros((N, Nw), dtype=jnp.int32)
        zeros_f = jnp.zeros((N, Nf), dtype=jnp.int32)
        wa_b = jnp.broadcast_to(worker_act_single, (N, Nw))
        fa_b = jnp.broadcast_to(firm_act_single, (N, Nf))

        rng_p1, rng_p2, rng_p3, rng_p4, rng_p5 = jax.random.split(rng, 5)

        # Phase 1: INTERVIEW_PROPOSE -- workers propose, firms NOOP.
        state, _, _, _, _ = bench.step(rng_p1, const_batch, state,
                                       [{"choice": wa_b}, {"choice": zeros_f}])
        # Phase 2: INTERVIEW_RESPOND -- workers NOOP, firms accept.
        state, _, _, _, _ = bench.step(rng_p2, const_batch, state,
                                       [{"choice": zeros_w}, {"choice": fa_b}])
        # Phase 3: MATCH_PROPOSE -- workers propose match.
        state, _, _, _, _ = bench.step(rng_p3, const_batch, state,
                                       [{"choice": wa_b}, {"choice": zeros_f}])
        # Phase 4: MATCH_RESPOND -- firms accept match.
        state, _, _, _, _ = bench.step(rng_p4, const_batch, state,
                                       [{"choice": zeros_w}, {"choice": fa_b}])

        # ---- Update per-seed CA-ETC means + counts on scheduled pairs. ----
        # A pair (i, j) is "pulled" iff scheduled AND it landed in tentative_matched
        # (the env may have rejected the proposal; we don't bump counts in that case,
        # matching ca_etc_baseline.py:434-447).
        sched_2d_b = jnp.broadcast_to(sched_2d, (N, Nw, Nf))
        pulled = state.tentative_matched & sched_2d_b                  # (N, Nw, Nf) bool
        pulled_int = pulled.astype(jnp.int32)

        # Belief-induced reward per pair (mirrors the single-seed baseline).
        w_reward = jnp.sum(state.x[:, :, None, :] * state.hat_y, axis=-1)        # (N, Nw, Nf)
        f_reward = jnp.sum(state.hat_x * state.y[:, None, :, :], axis=-1)        # (N, Nw, Nf)

        new_w_counts = w_counts + pulled_int
        new_w_means = jnp.where(
            pulled,
            (w_means * w_counts.astype(jnp.float32) + w_reward)
                / jnp.maximum(new_w_counts, 1).astype(jnp.float32),
            w_means,
        )
        pulled_t = pulled.transpose(0, 2, 1)                             # (N, Nf, Nw)
        f_reward_t = f_reward.transpose(0, 2, 1)                         # (N, Nf, Nw)
        new_f_counts = f_counts + pulled_t.astype(jnp.int32)
        new_f_means = jnp.where(
            pulled_t,
            (f_means * f_counts.astype(jnp.float32) + f_reward_t)
                / jnp.maximum(new_f_counts, 1).astype(jnp.float32),
            f_means,
        )

        # ---- Per-seed metric record (post-MATCH_RESPOND view). ----
        record = _vmapped_metrics(state, outside)

        # Phase 5: RETENTION -- NOOP/release everything (frees agents for next round).
        state, _, _, _, _ = bench.step(rng_p5, const_batch, state,
                                       [{"choice": zeros_w}, {"choice": zeros_f}])

        return state, new_w_means, new_w_counts, new_f_means, new_f_counts, record

    return kernel


def _build_exploit_period(bench: BatchedEnv, const_batch, Nw: int, Nf: int, N: int, outside: float):
    """Returns a jit'd kernel for one exploitation market period."""

    @jax.jit
    def kernel(rng, state, worker_to_firm):
        # worker_to_firm: (N, Nw) int32, value in [-1, Nf).
        zeros_w = jnp.zeros((N, Nw), dtype=jnp.int32)
        zeros_f = jnp.zeros((N, Nf), dtype=jnp.int32)
        rng_p1, rng_p2, rng_p3, rng_p4, rng_p5 = jax.random.split(rng, 5)

        # Phase 1, 2: NOOP (no new interviews in exploitation).
        state, _, _, _, _ = bench.step(rng_p1, const_batch, state,
                                       [{"choice": zeros_w}, {"choice": zeros_f}])
        state, _, _, _, _ = bench.step(rng_p2, const_batch, state,
                                       [{"choice": zeros_w}, {"choice": zeros_f}])

        # ---- Phase 3: MATCH_PROPOSE -- workers propose to assigned firms. ----
        valid = (worker_to_firm >= 0)                                # (N, Nw)
        safe_j = jnp.where(valid, worker_to_firm, 0)                 # (N, Nw)

        # Skip workers already matched, or whose assigned firm is already taken
        # (mirrors ca_etc_baseline.py:619-630).
        matched = state.matched                                      # (N, Nw, Nf)
        worker_unmatched = ~jnp.any(matched, axis=-1)                # (N, Nw)
        firm_unmatched = ~jnp.any(matched, axis=-2)                  # (N, Nf)
        firm_j_unmatched = jnp.take_along_axis(firm_unmatched, safe_j, axis=-1)
        propose = valid & worker_unmatched & firm_j_unmatched         # (N, Nw)

        worker_act_propose = jnp.where(propose, worker_to_firm + 1, 0).astype(jnp.int32)
        state, _, _, _, _ = bench.step(rng_p3, const_batch, state,
                                       [{"choice": worker_act_propose}, {"choice": zeros_f}])

        # ---- Phase 4: MATCH_RESPOND -- firms accept assigned workers. ----
        # Build firm_inverse[n, j] = i such that worker_to_firm[n, i] == j AND
        # propose[n, i] is True; -1 otherwise. Round-robin/GS uniqueness ensures
        # at most one such i per (n, j).
        sentinel = Nf
        scatter_pos = jnp.where(propose, worker_to_firm, sentinel)   # (N, Nw)
        scatter_val = jnp.broadcast_to(jnp.arange(Nw, dtype=jnp.int32), (N, Nw))

        def _scatter_one_seed(pos, val):
            base = jnp.full((Nf + 1,), -1, dtype=jnp.int32)
            return base.at[pos].set(val)

        firm_inverse_padded = jax.vmap(_scatter_one_seed)(scatter_pos, scatter_val)
        firm_inverse = firm_inverse_padded[..., :Nf]                  # (N, Nf)

        firm_act_accept = jnp.where(firm_inverse >= 0, firm_inverse + 1, 0).astype(jnp.int32)
        state, _, _, _, _ = bench.step(rng_p4, const_batch, state,
                                       [{"choice": zeros_w}, {"choice": firm_act_accept}])

        # ---- Per-seed metric record (post-MATCH_RESPOND view). ----
        record = _vmapped_metrics(state, outside)

        # ---- Phase 5: RETENTION -- retain assigned pairs that are matched/tentative. ----
        matched_now = state.matched | state.tentative_matched         # (N, Nw, Nf)
        retain_w = jnp.take_along_axis(matched_now, safe_j[..., None], axis=-1).squeeze(-1)
        retain_w = retain_w & valid

        # Firm j retains iff its assigned worker (firm_inverse[n, j]) has a
        # tentative/matched edge to it.
        safe_inv = jnp.where(firm_inverse >= 0, firm_inverse, 0)
        matched_t = matched_now.transpose(0, 2, 1)                    # (N, Nf, Nw)
        retain_f_raw = jnp.take_along_axis(matched_t, safe_inv[..., None], axis=-1).squeeze(-1)
        retain_f = retain_f_raw & (firm_inverse >= 0)

        worker_act_retain = retain_w.astype(jnp.int32)
        firm_act_retain = retain_f.astype(jnp.int32)
        state, _, _, _, _ = bench.step(rng_p5, const_batch, state,
                                       [{"choice": worker_act_retain}, {"choice": firm_act_retain}])

        return state, record

    return kernel


# ---------------------------------------------------------------------------
# BatchedCAETCBaseline
# ---------------------------------------------------------------------------


class BatchedCAETCBaseline:
    """N-seed parallel CA-ETC over the HireRL phase-level interaction protocol.

    Public usage mirrors :class:`ca_etc_baseline.CAETCBaseline`::

        env = HireRLEnv(config)
        baseline = BatchedCAETCBaseline(
            env=env, num_seeds=64, base_rng=jax.random.PRNGKey(0),
            Nw=10, Nf=10, horizon=300, T0=10, gamma=0.4,
        )
        history = baseline.run()
        # history["regret_w_total"] has shape (num_seeds, num_settled_records)
    """

    def __init__(
        self,
        env,
        num_seeds: int,
        base_rng: jax.Array,
        Nw: int,
        Nf: int,
        horizon: int,
        T0: Optional[int] = None,
        gamma: float = 0.4,
        use_confidence_check: bool = True,
        radius_scale: float = 0.5,
        unified_seeds: bool = False,
    ) -> None:
        if num_seeds <= 0:
            raise ValueError(f"num_seeds must be positive, got {num_seeds}")
        if Nw <= 0 or Nf <= 0:
            raise ValueError(f"Nw, Nf must be positive (got Nw={Nw}, Nf={Nf})")

        self.env_single = env
        self.bench = BatchedEnv(env, num_envs=int(num_seeds))
        self.const_batch = self.bench.default_const

        self.N = int(num_seeds)
        self.Nw = int(Nw)
        self.Nf = int(Nf)
        self.horizon = int(horizon)
        self.gamma = float(gamma)
        self.use_confidence_check = bool(use_confidence_check)
        # Multiplicative scale on the LCB/UCB confidence radius. <1 makes
        # the separation check easier to satisfy. The check now only affects
        # the diagnostic ``*_separated`` flag; preferences always use the
        # mean-sorted permutation (no identity fallback).
        self.radius_scale = float(radius_scale)
        self.outside_option = float(env.config.outside_option)

        # Same paper-spec invariant as the single-seed baseline.
        num_batches = (self.Nw + self.Nf - 1) // self.Nf
        min_rounds = num_batches * self.Nf
        if T0 is None:
            T0 = min_rounds
        self.T0 = int(T0)
        if self.T0 < min_rounds:
            raise ValueError(
                f"T0={self.T0} is too small to cover one full round-robin over "
                f"all {self.Nw}*{self.Nf} pairs. Need T0 >= {min_rounds} "
                f"(= ceil(Nw/Nf) * Nf)."
            )
        if self.horizon < min_rounds:
            raise ValueError(
                f"horizon={self.horizon} cannot accommodate one full "
                f"round-robin ({min_rounds} market periods). Increase horizon "
                f"or shrink Nw/Nf."
            )

        # Build jit'd per-period kernels (closures over bench/const/Nw/Nf/N/outside).
        self._explore_period = _build_explore_period(
            self.bench, self.const_batch, self.Nw, self.Nf, self.N, self.outside_option
        )
        self._exploit_period = _build_exploit_period(
            self.bench, self.const_batch, self.Nw, self.Nf, self.N, self.outside_option
        )
        # Jit'd auxiliary kernels.
        self._estimate_prefs_jit = jax.jit(
            functools.partial(
                _estimate_preferences,
                use_confidence_check=self.use_confidence_check,
                radius_scale=self.radius_scale,
            ),
        )
        self._gale_shapley_jit = jax.jit(_gale_shapley)

        # Per-seed bookkeeping (kept on device throughout the run).
        self.worker_means = jnp.zeros((self.N, self.Nw, self.Nf), dtype=jnp.float32)
        self.firm_means = jnp.zeros((self.N, self.Nf, self.Nw), dtype=jnp.float32)
        self.worker_counts = jnp.zeros((self.N, self.Nw, self.Nf), dtype=jnp.int32)
        self.firm_counts = jnp.zeros((self.N, self.Nf, self.Nw), dtype=jnp.int32)

        self.total_explore_rounds = 0
        self.market_period = 0

        # Reset the batched env to get the initial batched state.
        if unified_seeds:
            # Match eval_and_plot.py's per-seed reset-key derivation so that,
            # given the same `--seed` and `num_seeds`, the i-th seed's (x, y)
            # are bit-equal across the PPO eval and CA-ETC scripts -- making
            # DA-ref / planner upper bounds line up exactly when overlaying
            # the two runs in compare_ppo_vs_caetc.py. PPO eval does:
            #     seed_rngs = split(PRNGKey(seed), N)
            #     # inside per-seed rollout_fn:
            #     rng, rng_reset = split(rng)
            #     env.reset(rng_reset, const)
            seed_rngs = jax.random.split(base_rng, self.N)
            rst_keys = jax.vmap(lambda k: jax.random.split(k)[1])(seed_rngs)
            self.state, _ = jax.vmap(self.env_single.reset)(rst_keys, self.const_batch)
            # PRNG used downstream by explore/exploit kernels; per-seed action
            # randomness need not match PPO's (it doesn't affect x, y or refs).
            self.rng = base_rng
        else:
            self.rng, rst_rng = jax.random.split(base_rng)
            self.state, _ = self.bench.reset(rst_rng, self.const_batch)

        # Settled-history records (list of per-period dicts of jnp arrays).
        # Pulled to host in a single transfer at end of run().
        self._records: List[Dict[str, jax.Array]] = []
        # Per-record absolute market period index (1-indexed, like the
        # single-seed baseline).
        self._record_t: List[int] = []
        # Epoch summary (for log_epoch parity).
        self._epoch_log: List[Dict[str, Any]] = []

    # ---------------- horizon ----------------

    def _horizon_reached(self) -> bool:
        return self.market_period >= self.horizon

    # ---------------- epoch sizing (identical to single-seed baseline) ----------------

    def _get_epoch_market_periods(self, epoch: int) -> Tuple[int, int]:
        c = 2
        b = 2 ** (1 / self.gamma)
        exploration_market_periods = int(self.T0 * (epoch ** c))
        total_market_periods = int(self.T0 * (epoch ** b))
        exploitation_market_periods = max(0, total_market_periods - exploration_market_periods)
        remaining = self.horizon - self.market_period
        exploration_market_periods = max(0, min(exploration_market_periods, remaining))
        remaining -= exploration_market_periods
        exploitation_market_periods = max(0, min(exploitation_market_periods, remaining))
        return exploration_market_periods, exploitation_market_periods

    # ---------------- exploration block ----------------

    def _exploration_block(self, num_periods: int) -> None:
        for _ in range(num_periods):
            if self._horizon_reached():
                break
            schedule_np = _round_robin_schedule(self.total_explore_rounds, self.Nw, self.Nf)
            schedule = jnp.asarray(schedule_np, dtype=jnp.int32)

            self.rng, sub = jax.random.split(self.rng)
            (self.state, self.worker_means, self.worker_counts,
             self.firm_means, self.firm_counts, record) = self._explore_period(
                sub,
                self.state,
                self.worker_means, self.worker_counts,
                self.firm_means, self.firm_counts,
                schedule,
            )
            self.total_explore_rounds += 1
            self.market_period += 1
            self._records.append(record)
            self._record_t.append(self.market_period)

    # ---------------- exploitation block ----------------

    def _exploitation_block(self, num_periods: int, worker_to_firm: jax.Array) -> None:
        for _ in range(num_periods):
            if self._horizon_reached():
                break
            self.rng, sub = jax.random.split(self.rng)
            self.state, record = self._exploit_period(sub, self.state, worker_to_firm)
            self.market_period += 1
            self._records.append(record)
            self._record_t.append(self.market_period)

    # ---------------- main run ----------------

    def run(self) -> Dict[str, np.ndarray]:
        """Drive the baseline to completion. Returns numpy-friendly history.

        Schema:
            t:                          (T,) int   absolute market period index
            reward_w_total, ...:        (N, T)
            reward_w_per_agent, ...:    (N, T, Nw)
            reward_f_per_agent, ...:    (N, T, Nf)
            matched_matrix:             (N, T, Nw, Nf) bool
        Where N = num_seeds, T = number of settled records (= self.market_period).
        """
        epoch = 1
        while not self._horizon_reached():
            exp_periods, exploit_periods = self._get_epoch_market_periods(epoch)
            if exp_periods == 0 and exploit_periods == 0:
                break

            self._exploration_block(exp_periods)

            t_explore = jnp.asarray(self.total_explore_rounds, dtype=jnp.int32)
            worker_prefs, firm_prefs, w_sep, f_sep = self._estimate_prefs_jit(
                self.worker_means, self.worker_counts,
                self.firm_means, self.firm_counts,
                t_explore,
            )
            worker_to_firm = self._gale_shapley_jit(worker_prefs, firm_prefs)

            # LCB/UCB separation summary (per-seed mean of separated counts).
            # The separation flag is now diagnostic only -- preferences always
            # use the mean-sorted permutation, so a low separated-count means
            # the ranking is held together by point estimates rather than
            # confidence-checked separation. Ranking errors here surface as
            # regret that decays slowly during exploitation in small/short runs.
            w_sep_per_seed = np.asarray(jnp.sum(w_sep.astype(jnp.int32), axis=-1))  # (N,)
            f_sep_per_seed = np.asarray(jnp.sum(f_sep.astype(jnp.int32), axis=-1))  # (N,)
            wc_min = int(jnp.min(self.worker_counts))
            fc_min = int(jnp.min(self.firm_counts))
            n_all_w = int(np.sum(w_sep_per_seed == self.Nw))
            n_all_f = int(np.sum(f_sep_per_seed == self.Nf))
            print(
                f"  [epoch {epoch}] t_explore={self.total_explore_rounds}  "
                f"min_count(worker)={wc_min}  min_count(firm)={fc_min}  "
                f"separated workers/seed={w_sep_per_seed.mean():.2f}/{self.Nw}  "
                f"firms/seed={f_sep_per_seed.mean():.2f}/{self.Nf}  "
                f"({n_all_w}/{self.N} seeds fully w-separated, "
                f"{n_all_f}/{self.N} seeds fully f-separated)"
            )

            self._exploitation_block(exploit_periods, worker_to_firm)

            self._epoch_log.append({
                "epoch": int(epoch),
                "market_period": int(self.market_period),
                "exp_periods": int(exp_periods),
                "exploit_periods": int(exploit_periods),
                "w_separated_mean": float(w_sep_per_seed.mean()),
                "f_separated_mean": float(f_sep_per_seed.mean()),
                "seeds_fully_w_separated": n_all_w,
                "seeds_fully_f_separated": n_all_f,
            })
            epoch += 1

        return self._materialize_history()

    def _materialize_history(self) -> Dict[str, np.ndarray]:
        """Stack records to (N, T, ...) and pull to host once."""
        if not self._records:
            return {"t": np.zeros((0,), dtype=np.int32)}
        # Stack along a new "time" axis (axis=1; axis=0 is already seed).
        stacked = jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs, axis=1), *self._records
        )
        history_np: Dict[str, np.ndarray] = {
            k: np.asarray(v) for k, v in stacked.items()
        }
        history_np["t"] = np.asarray(self._record_t, dtype=np.int32)
        history_np["epoch_log"] = self._epoch_log         # list of dicts
        history_np["num_seeds"] = self.N
        history_np["Nw"] = self.Nw
        history_np["Nf"] = self.Nf
        return history_np


# ---------------------------------------------------------------------------
# Self-check (runs when this module is imported with --selftest)
# ---------------------------------------------------------------------------


def _selftest_smoke(num_seeds: int = 4, Nw: int = 3, Nf: int = 3, horizon: int = 30) -> None:
    """Tiny smoke check that builds + runs a few periods. Useful before the
    real run because compilation errors surface here in <1s rather than
    waiting for the JIT cache to populate during a 5-min run."""
    from hirerl import HireRLConfig, HireRLEnv

    config = HireRLConfig(
        Nw=Nw, Nf=Nf, d=2, horizon=horizon + 5,
        sigma_interview=0.3, sigma_match=0.1, lambda_reveal=2.0,
    )
    env = HireRLEnv(config)
    baseline = BatchedCAETCBaseline(
        env=env, num_seeds=num_seeds, base_rng=jax.random.PRNGKey(0),
        Nw=Nw, Nf=Nf, horizon=horizon, T0=Nf, gamma=0.4,
    )
    history = baseline.run()
    assert history["regret_w_total"].shape[0] == num_seeds, history["regret_w_total"].shape
    assert history["regret_w_total"].shape[1] == baseline.market_period
    assert history["matched_matrix"].shape == (num_seeds, baseline.market_period, Nw, Nf)
    print(
        f"[OK] selftest: N={num_seeds}, Nw={Nw}, Nf={Nf}, "
        f"records={history['regret_w_total'].shape[1]}, "
        f"mean_reg_w[final] = {history['regret_w_total'][:, -1].mean():.3f} "
        f"(per-seed std {history['regret_w_total'][:, -1].std():.3f})"
    )


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest_smoke()
    else:
        print(
            "batched_ca_etc_baseline.py is a library. Run "
            "plot_batched_ca_etc_baseline.py for a full experiment, or "
            "this file with --selftest for a tiny smoke check."
        )
