"""HireRL metrics: deferred-acceptance reference, social welfare, regret, friction loss.

All operations are JAX-compatible (run under jit/vmap).

Gating
------
The per-market-step metrics defined here (regret, friction_loss, social_welfare,
*_rate) are only semantically meaningful at *post-RETENTION* snapshots, where
``matched``, ``hat_x``, ``hat_y``, tenure and the ``*_this_period`` counters all
reflect the just-completed market step ``t``. Mid-step snapshots (after
INTERVIEW_RESPOND / MATCH_PROPOSE / MATCH_RESPOND) describe transient state and
must not be accumulated.

Use :func:`is_settled_state` to gate post-hoc on a state, or
:func:`is_settled_transition` (on the *previous* phase) to gate at env-step
emission time so that the final market step is not lost when the env
auto-resets at the horizon.
"""

import jax
import jax.numpy as jnp

from .constants import INTERVIEW_PROPOSE, RETENTION
from .state import HireRLState


def worker_proposing_da(U_w: jax.Array, U_f: jax.Array, outside: float = 0.0) -> jax.Array:
    """Worker-proposing deferred-acceptance with deterministic id-based tie-breaks.

    Args:
        U_w: (Nw, Nf) worker-side utilities for each pair.
        U_f: (Nw, Nf) firm-side utilities for each pair.
        outside: outside option utility threshold (default 0.0).

    Returns:
        matched: (Nw, Nf) bool matrix of stable matched pairs.
    """
    Nw, Nf = U_w.shape

    # Worker preference list: stable argsort by descending U_w; ties resolved by lower firm id.
    firm_pref = jnp.argsort(-U_w, axis=1, stable=True)                       # (Nw, Nf)

    arange_w = jnp.arange(Nw, dtype=jnp.int32)
    arange_f = jnp.arange(Nf, dtype=jnp.int32)
    sentinel = jnp.array(Nw, dtype=jnp.int32)

    init = (
        jnp.zeros((Nw,), dtype=jnp.int32),               # proposal_rank
        jnp.full((Nf,), -1, dtype=jnp.int32),            # firm_holder (worker id or -1)
    )

    def body(_, carry):
        proposal_rank, firm_holder = carry

        # Reverse map firm_holder -> worker_match (Nw,) of firm id (or -1)
        valid_holder = (firm_holder != -1)
        scatter_idx = jnp.where(valid_holder, firm_holder, sentinel)
        worker_match_padded = jnp.full((Nw + 1,), -1, dtype=jnp.int32).at[scatter_idx].set(
            jnp.where(valid_holder, arange_f, -1)
        )
        worker_match = worker_match_padded[:Nw]

        worker_unmatched = (worker_match == -1)
        can_propose = worker_unmatched & (proposal_rank < Nf)

        safe_rank = jnp.clip(proposal_rank, 0, max(Nf - 1, 0))
        candidate_firm = firm_pref[arange_w, safe_rank]                      # (Nw,)
        candidate_util = U_w[arange_w, candidate_firm]
        worker_proposes = can_propose & (candidate_util >= outside)

        proposed_to = worker_proposes[:, None] & (candidate_firm[:, None] == arange_f[None, :])    # (Nw, Nf)
        is_holder = (firm_holder[None, :] == arange_w[:, None])                                     # (Nw, Nf)
        candidates = proposed_to | is_holder

        firm_acceptable = (U_f >= outside)
        # Tie-break for firms: lower worker id is preferred.
        score = U_f - arange_w[:, None].astype(U_f.dtype) * 1e-9
        masked_score = jnp.where(candidates & firm_acceptable, score, -jnp.inf)

        best_worker = jnp.argmax(masked_score, axis=0)                       # (Nf,)
        any_candidate = jnp.any(candidates & firm_acceptable, axis=0)
        new_firm_holder = jnp.where(any_candidate, best_worker, -1)

        valid_new = (new_firm_holder != -1)
        scatter_idx_new = jnp.where(valid_new, new_firm_holder, sentinel)
        new_worker_match_padded = jnp.full((Nw + 1,), -1, dtype=jnp.int32).at[scatter_idx_new].set(
            jnp.where(valid_new, arange_f, -1)
        )
        new_worker_match = new_worker_match_padded[:Nw]

        worker_unmatched_end = (new_worker_match == -1)
        new_proposal_rank = jnp.minimum(proposal_rank + worker_unmatched_end.astype(jnp.int32), Nf)
        return (new_proposal_rank, new_firm_holder)

    # Theoretical worst-case is Nw*Nf rounds, but synchronous worker-proposing
    # DA converges in O(max(Nw, Nf)) rounds for non-degenerate preferences.
    # Tight cap matters at large N: at Nw=Nf=100 the old Nw*Nf+1=10001 bound
    # was ~25x over actual convergence and dominated env step cost (this DA
    # runs 3x per env step via compute_all_metrics). Reference matching is
    # diagnostic-only -- not in the policy gradient path -- so a generous
    # but bounded cap is safe.
    max_iters = max(min(Nw * Nf + 1, 4 * max(Nw, Nf) + 10), 1)
    _, firm_holder = jax.lax.fori_loop(0, max_iters, body, init)

    matched = (firm_holder[None, :] == arange_w[:, None]) & (firm_holder[None, :] != -1)
    return matched


def true_utility(state: HireRLState) -> jax.Array:
    """U[i, j] = <x[i], y[j]>"""
    return jnp.einsum("id,jd->ij", state.x, state.y)


def belief_utilities(state: HireRLState) -> tuple[jax.Array, jax.Array]:
    """Belief-induced utilities used by RL agents.

    Returns:
        U_w_belief: (Nw, Nf) worker-side using <x[i], hat_y[i, j]>
        U_f_belief: (Nw, Nf) firm-side using <hat_x[i, j], y[j]>
    """
    U_w = jnp.sum(state.x[:, None, :] * state.hat_y, axis=-1)
    U_f = jnp.sum(state.hat_x * state.y[None, :, :], axis=-1)
    return U_w, U_f


def social_welfare(state: HireRLState) -> jax.Array:
    matched = state.matched.astype(jnp.float32)
    U_w, U_f = belief_utilities(state)
    return jnp.sum(matched * (U_w + U_f))


def reference_match(state: HireRLState, outside: float) -> jax.Array:
    U = true_utility(state)
    return worker_proposing_da(U, U, outside=outside)


def belief_induced_match(state: HireRLState, outside: float) -> jax.Array:
    U_w, U_f = belief_utilities(state)
    return worker_proposing_da(U_w, U_f, outside=outside)


def worker_regret(state: HireRLState, ref_matched: jax.Array) -> jax.Array:
    U = true_utility(state)
    U_w_belief, _ = belief_utilities(state)
    ref_value = jnp.sum(ref_matched.astype(jnp.float32) * U)
    cur_value = jnp.sum(state.matched.astype(jnp.float32) * U_w_belief)
    return ref_value - cur_value


def firm_regret(state: HireRLState, ref_matched: jax.Array) -> jax.Array:
    U = true_utility(state)
    _, U_f_belief = belief_utilities(state)
    ref_value = jnp.sum(ref_matched.astype(jnp.float32) * U)
    cur_value = jnp.sum(state.matched.astype(jnp.float32) * U_f_belief)
    return ref_value - cur_value


def per_worker_regret(state: HireRLState, ref_matched: jax.Array) -> jax.Array:
    """Per-worker regret: ref-match true value minus actual-match belief value. Shape (Nw,)."""
    U = true_utility(state)
    U_w_belief, _ = belief_utilities(state)
    ref_per_worker = jnp.sum(ref_matched.astype(jnp.float32) * U, axis=1)
    cur_per_worker = jnp.sum(state.matched.astype(jnp.float32) * U_w_belief, axis=1)
    return ref_per_worker - cur_per_worker


def per_firm_regret(state: HireRLState, ref_matched: jax.Array) -> jax.Array:
    """Per-firm regret. Shape (Nf,)."""
    U = true_utility(state)
    _, U_f_belief = belief_utilities(state)
    ref_per_firm = jnp.sum(ref_matched.astype(jnp.float32) * U, axis=0)
    cur_per_firm = jnp.sum(state.matched.astype(jnp.float32) * U_f_belief, axis=0)
    return ref_per_firm - cur_per_firm


def per_worker_information_loss(state: HireRLState) -> jax.Array:
    """Squared L2 distance between hat_y[i, j*] and y[j*] for each worker i (j* = i's match).
    Shape (Nw,). Zero if worker i is unmatched.
    """
    diff_sq = jnp.sum((state.hat_y - state.y[None, :, :]) ** 2, axis=-1)   # (Nw, Nf)
    return jnp.sum(state.matched.astype(jnp.float32) * diff_sq, axis=1)


def per_firm_information_loss(state: HireRLState) -> jax.Array:
    """Squared L2 distance between hat_x[i*, j] and x[i*] for each firm j (i* = j's match).
    Shape (Nf,). Zero if firm j is unmatched.
    """
    diff_sq = jnp.sum((state.hat_x - state.x[:, None, :]) ** 2, axis=-1)   # (Nw, Nf)
    return jnp.sum(state.matched.astype(jnp.float32) * diff_sq, axis=0)


@jax.jit
def compute_per_agent_metrics(state: HireRLState, outside: float) -> dict:
    """Jit-compiled bundle of the four per-agent metrics + the reference matching.

    Equivalent to calling ``reference_match``, ``per_worker_regret``,
    ``per_firm_regret``, ``per_worker_information_loss``,
    ``per_firm_information_loss`` individually, but the four computations
    share the ``ref_matched`` and ``true_utility`` traces and run inside one
    XLA program. Per-call wall time on the rollout hot path drops from
    ~30-100ms (5 separate jax dispatches + DA fori_loop overhead each) to
    ~1-5ms after the first call traces.

    Returns a dict with keys: ``ref_matched`` (Nw, Nf bool),
    ``per_worker_regret`` (Nw,), ``per_firm_regret`` (Nf,),
    ``per_worker_information_loss`` (Nw,), ``per_firm_information_loss`` (Nf,).
    Values are jax arrays; callers convert with ``np.asarray`` if they need
    host arrays. The first call traces and caches under jit; subsequent calls
    with the same state shape hit the cache.
    """
    ref_matched = reference_match(state, outside)
    return {
        "ref_matched": ref_matched,
        "per_worker_regret": per_worker_regret(state, ref_matched),
        "per_firm_regret": per_firm_regret(state, ref_matched),
        "per_worker_information_loss": per_worker_information_loss(state),
        "per_firm_information_loss": per_firm_information_loss(state),
    }


def friction_loss(state: HireRLState, outside: float) -> jax.Array:
    U = true_utility(state)
    ref_matched = reference_match(state, outside)
    bi_matched = belief_induced_match(state, outside)
    ref_true_value = jnp.sum(ref_matched.astype(jnp.float32) * U)
    bi_true_value = jnp.sum(bi_matched.astype(jnp.float32) * U)
    return ref_true_value - bi_true_value


def match_rate(state: HireRLState) -> jax.Array:
    Nw, Nf = state.matched.shape
    norm = jnp.float32(max(min(Nw, Nf), 1))
    return jnp.sum(state.matched).astype(jnp.float32) / norm


def interview_rate(state: HireRLState) -> jax.Array:
    Nw, Nf = state.matched.shape
    norm = jnp.float32(max(min(Nw, Nf), 1))
    return state.interviews_this_period.astype(jnp.float32) / norm


def dissolution_rate(state: HireRLState) -> jax.Array:
    candidate = (state.prev_matched | jnp.zeros_like(state.matched))  # cheap "candidate" proxy
    candidate_count = jnp.sum(candidate).astype(jnp.float32)
    return state.dissolutions_this_period.astype(jnp.float32) / jnp.maximum(candidate_count, 1.0)


def compute_all_metrics(state: HireRLState, outside: float) -> dict:
    """Compute all per-state metrics. Only meaningful at settled snapshots.

    Callers are expected to gate the result with :func:`is_settled_state` or
    :func:`is_settled_transition`. See module docstring.
    """
    ref_matched = reference_match(state, outside)
    U = true_utility(state)
    ref_f = ref_matched.astype(jnp.float32)
    return {
        "social_welfare": social_welfare(state),
        "social_welfare_da_ref": jnp.sum(ref_f * (U + U)),
        "worker_regret": worker_regret(state, ref_matched),
        "firm_regret": firm_regret(state, ref_matched),
        "friction_loss": friction_loss(state, outside),
        "match_rate": match_rate(state),
        "interview_rate": interview_rate(state),
        "dissolution_rate": dissolution_rate(state),
    }


def compute_all_metrics_zero() -> dict:
    """Zero-valued counterpart of :func:`compute_all_metrics`.

    Provides a same-shape, same-dtype pytree of zeros so callers can use
    ``jax.lax.cond(is_settled, compute_all_metrics, compute_all_metrics_zero)``
    to skip the expensive DA solves at non-settled env steps. Must stay in
    sync with the key set returned by :func:`compute_all_metrics`.
    """
    z = jnp.float32(0.0)
    return {
        "social_welfare":         z,
        "social_welfare_da_ref":  z,
        "worker_regret":          z,
        "firm_regret":            z,
        "friction_loss":          z,
        "match_rate":             z,
        "interview_rate":         z,
        "dissolution_rate":       z,
    }


def is_settled_state(state: HireRLState) -> jax.Array:
    """True iff ``state`` is a post-RETENTION snapshot.

    Equivalent to: phase has just been reset to INTERVIEW_PROPOSE *and* at
    least one market step has completed (``market_t > 0``). The first state of
    an episode (after env_reset, ``market_t == 0``) and the auto-reset state
    after a done step both fail this predicate.
    """
    return (state.phase == jnp.int32(INTERVIEW_PROPOSE)) & (state.market_t > 0)


def is_settled_transition(prev_phase: jax.Array) -> jax.Array:
    """True iff the just-applied transition concluded a market step.

    Equivalent to ``is_settled_state(new_state)`` evaluated *before* any
    end-of-episode auto-reset. Use this signal at env-step emission time so
    the final market step is captured even when the env auto-resets at horizon.
    """
    return prev_phase == jnp.int32(RETENTION)


def gate_metrics(metrics: dict, is_settled: jax.Array) -> dict:
    """Zero out each metric where ``is_settled`` is False.

    Sums of gated metrics over an episode equal the paper's
    :math:`\\sum_{t=1}^T (\\cdot)` quantities.
    """
    return jax.tree_util.tree_map(
        lambda v: jnp.where(is_settled, v, jnp.zeros_like(v)),
        metrics,
    )


def empty_episode_accumulator() -> dict:
    """Zero carry for accumulating gated per-step metrics across an episode."""
    z = jnp.float32(0.0)
    return {
        "social_welfare_sum": z,
        "worker_regret_sum": z,
        "firm_regret_sum": z,
        "friction_loss_sum": z,
        "match_rate_sum": z,
        "interview_rate_sum": z,
        "dissolution_rate_sum": z,
        "num_market_steps": z,
    }


def add_to_episode_accumulator(carry: dict, gated: dict, is_settled: jax.Array) -> dict:
    v = is_settled.astype(jnp.float32)
    return {
        "social_welfare_sum":   carry["social_welfare_sum"]   + gated["social_welfare"],
        "worker_regret_sum":    carry["worker_regret_sum"]    + gated["worker_regret"],
        "firm_regret_sum":      carry["firm_regret_sum"]      + gated["firm_regret"],
        "friction_loss_sum":    carry["friction_loss_sum"]    + gated["friction_loss"],
        "match_rate_sum":       carry["match_rate_sum"]       + gated["match_rate"],
        "interview_rate_sum":   carry["interview_rate_sum"]   + gated["interview_rate"],
        "dissolution_rate_sum": carry["dissolution_rate_sum"] + gated["dissolution_rate"],
        "num_market_steps":     carry["num_market_steps"]     + v,
    }
