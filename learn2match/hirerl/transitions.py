"""Per-phase transitions for HireRL."""

import jax
import jax.numpy as jnp

from .constants import (
    INTERVIEW_MODE_EXCLUSIVE,
    INTERVIEW_PROPOSE,
    INTERVIEW_RESPOND,
    MATCH_PROPOSE,
    MATCH_RESPOND,
    RECEIVER_FIRM_FIRST,
    RECEIVER_WORKER_FIRST,
    RETENTION,
)
from .state import HireRLConst, HireRLState


def _capacity_resolve_max1_firm_first(candidate: jax.Array) -> jax.Array:
    """Fast path for max_per_worker == max_per_firm == 1, RECEIVER_FIRM_FIRST.

    Equivalent to the edge-sorted greedy in :func:`_capacity_resolve_general`,
    but with O(Nw + Nf) scan instead of O((Nw*Nf)^2): process firms in id
    order; each firm gets the lowest-id un-used worker among its candidates.

    Under the firm-first priority (firm_id * Nw + worker_id), the first edge
    a firm sees is its lowest-id candidate worker, and once a worker is
    matched (max=1) it's used. So per-firm-argmax exactly reproduces the
    edge-sorted greedy.
    """
    Nw, Nf = candidate.shape

    def body(carry, j):
        worker_used, kept = carry
        eligible = candidate[:, j] & ~worker_used                     # (Nw,)
        any_eligible = jnp.any(eligible)
        i_star = jnp.argmax(eligible.astype(jnp.int32))               # first True idx
        new_worker_used = worker_used.at[i_star].set(worker_used[i_star] | any_eligible)
        new_kept = kept.at[i_star, j].set(kept[i_star, j] | any_eligible)
        return (new_worker_used, new_kept), None

    init = (
        jnp.zeros((Nw,), dtype=jnp.bool_),
        jnp.zeros((Nw, Nf), dtype=jnp.bool_),
    )
    (_, kept), _ = jax.lax.scan(body, init, jnp.arange(Nf, dtype=jnp.int32))
    return kept


def _capacity_resolve_max1_worker_first(candidate: jax.Array) -> jax.Array:
    """Fast path for max=1 with RECEIVER_WORKER_FIRST priority. Dual of the
    firm-first variant: scan over workers, each takes its lowest-id un-used
    candidate firm.
    """
    Nw, Nf = candidate.shape

    def body(carry, i):
        firm_used, kept = carry
        eligible = candidate[i, :] & ~firm_used                       # (Nf,)
        any_eligible = jnp.any(eligible)
        j_star = jnp.argmax(eligible.astype(jnp.int32))
        new_firm_used = firm_used.at[j_star].set(firm_used[j_star] | any_eligible)
        new_kept = kept.at[i, j_star].set(kept[i, j_star] | any_eligible)
        return (new_firm_used, new_kept), None

    init = (
        jnp.zeros((Nf,), dtype=jnp.bool_),
        jnp.zeros((Nw, Nf), dtype=jnp.bool_),
    )
    (_, kept), _ = jax.lax.scan(body, init, jnp.arange(Nw, dtype=jnp.int32))
    return kept


def _capacity_resolve_general(
    candidate: jax.Array,
    max_per_worker: int,
    max_per_firm: int,
    receiver_side_order: str,
) -> jax.Array:
    """Edge-sorted greedy. O(Nw*Nf) scan length, O(1) body via .at[].set/add.

    The body used to do ``kept = kept | (can_add & arange_w==i & arange_f==j)``
    which constructs a full (Nw, Nf) boolean broadcast just to flip one bit;
    XLA usually can't fuse this away under scan. Single-element scatter via
    ``.at[i, j].set(...)`` is what XLA actually wants here.
    """
    Nw, Nf = candidate.shape

    if receiver_side_order == RECEIVER_FIRM_FIRST:
        priority = jnp.arange(Nf, dtype=jnp.int32)[None, :] * Nw + jnp.arange(Nw, dtype=jnp.int32)[:, None]
    else:
        priority = jnp.arange(Nw, dtype=jnp.int32)[:, None] * Nf + jnp.arange(Nf, dtype=jnp.int32)[None, :]

    sentinel = jnp.array(Nw * Nf + 1, dtype=jnp.int32)
    flat_priority = jnp.where(candidate, priority, sentinel).reshape(-1)
    sort_idx = jnp.argsort(flat_priority)                  # (Nw*Nf,)

    init = (
        jnp.zeros((Nw, Nf), dtype=jnp.bool_),
        jnp.zeros((Nw,), dtype=jnp.int32),
        jnp.zeros((Nf,), dtype=jnp.int32),
    )

    def body(carry, edge_idx):
        kept, w_count, f_count = carry
        i = edge_idx // Nf
        j = edge_idx % Nf
        is_cand = candidate[i, j]
        can_add = is_cand & (w_count[i] < max_per_worker) & (f_count[j] < max_per_firm)
        kept = kept.at[i, j].set(kept[i, j] | can_add)
        w_count = w_count.at[i].add(can_add.astype(jnp.int32))
        f_count = f_count.at[j].add(can_add.astype(jnp.int32))
        return (kept, w_count, f_count), None

    final, _ = jax.lax.scan(body, init, sort_idx)
    return final[0]


def _capacity_resolve(
    candidate: jax.Array,
    max_per_worker: int,
    max_per_firm: int,
    receiver_side_order: str,
) -> jax.Array:
    """Greedy id-based deterministic resolution of candidate edges.

    Edges are processed in priority order; an edge is kept only if both ends
    are still under capacity. Priority is purely id-based (no latent utility).

    Dispatches at trace time (max_per_*, receiver_side_order are Python ints
    / strings, not traced values) to a max=1 fast path when both caps are 1;
    otherwise falls back to the general edge-sorted scan.
    """
    if max_per_worker == 1 and max_per_firm == 1:
        if receiver_side_order == RECEIVER_FIRM_FIRST:
            return _capacity_resolve_max1_firm_first(candidate)
        if receiver_side_order == RECEIVER_WORKER_FIRST:
            return _capacity_resolve_max1_worker_first(candidate)
    return _capacity_resolve_general(
        candidate, max_per_worker, max_per_firm, receiver_side_order,
    )


def _to_target_idx(action: jax.Array, max_idx: int) -> jax.Array:
    """choice 0 -> -1; choice k+1 -> k. Out-of-range collapsed to -1."""
    target = jnp.where(action >= 1, action - 1, -1)
    target = jnp.where((target >= 0) & (target < max_idx), target, -1)
    return target.astype(jnp.int32)


def _row_unmatched(matched: jax.Array) -> jax.Array:
    return ~jnp.any(matched, axis=1)


def _col_unmatched(matched: jax.Array) -> jax.Array:
    return ~jnp.any(matched, axis=0)


def phase_interview_propose(
    rng: jax.Array,
    const: HireRLConst,
    state: HireRLState,
    worker_act: jax.Array,
    firm_act: jax.Array,
    Nw: int,
    Nf: int,
    interview_mode: str,
    receiver_side_order: str,
    max_per_step: int,
    allow_on_the_job_search: bool = False,
    public_retention_signal: bool = False,  # accepted for dispatch symmetry; unused
):
    arange_w = jnp.arange(Nw)
    arange_f = jnp.arange(Nf)
    worker_unmatched = _row_unmatched(state.matched)
    firm_unmatched = _col_unmatched(state.matched)

    target_w = _to_target_idx(worker_act, Nf)
    safe_w = jnp.clip(target_w, 0, max(Nf - 1, 0))
    if allow_on_the_job_search:
        # Worker i may propose to firm j as long as (i, j) are not currently
        # matched to each other. Both sides may already be matched to others.
        valid_w = (target_w >= 0) & (~state.matched[arange_w, safe_w])
    else:
        valid_w = (target_w >= 0) & worker_unmatched & firm_unmatched[safe_w]
    new_prop_w = jnp.where(valid_w, target_w, -1)

    target_f = _to_target_idx(firm_act, Nw)
    safe_f = jnp.clip(target_f, 0, max(Nw - 1, 0))
    if allow_on_the_job_search:
        valid_f = (target_f >= 0) & (~state.matched[safe_f, arange_f])
    else:
        valid_f = (target_f >= 0) & firm_unmatched & worker_unmatched[safe_f]
    new_prop_f = jnp.where(valid_f, target_f, -1)

    new_state = state.replace(
        interview_proposals_w_to_f=new_prop_w,
        interview_proposals_f_to_w=new_prop_f,
        interview_accept_w=jnp.full((Nw,), -1, dtype=jnp.int32),
        interview_accept_f=jnp.full((Nf,), -1, dtype=jnp.int32),
        match_proposals_w_to_f=jnp.full((Nw,), -1, dtype=jnp.int32),
        match_accept_f=jnp.full((Nf,), -1, dtype=jnp.int32),
        interviews_this_period=jnp.int32(0),
        matches_formed_this_period=jnp.int32(0),
        dissolutions_this_period=jnp.int32(0),
        phase=jnp.int32(INTERVIEW_RESPOND),
        _step=state._step + 1,
    )
    rew_w = jnp.zeros((Nw,), dtype=jnp.float32)
    rew_f = jnp.zeros((Nf,), dtype=jnp.float32)
    return new_state, rew_w, rew_f


def phase_interview_respond(
    rng: jax.Array,
    const: HireRLConst,
    state: HireRLState,
    worker_act: jax.Array,
    firm_act: jax.Array,
    Nw: int,
    Nf: int,
    interview_mode: str,
    receiver_side_order: str,
    max_per_step: int,
    allow_on_the_job_search: bool = False,
    public_retention_signal: bool = False,
):
    arange_w = jnp.arange(Nw)
    arange_f = jnp.arange(Nf)
    worker_unmatched = _row_unmatched(state.matched)
    firm_unmatched = _col_unmatched(state.matched)

    # Worker accept choice
    accept_w = _to_target_idx(worker_act, Nf)
    safe_aw = jnp.clip(accept_w, 0, max(Nf - 1, 0))
    firm_did_propose_to_me = state.interview_proposals_f_to_w[safe_aw] == arange_w
    if allow_on_the_job_search:
        not_already_matched_pair_w = ~state.matched[arange_w, safe_aw]
        valid_aw = (accept_w >= 0) & not_already_matched_pair_w & firm_did_propose_to_me
    else:
        valid_aw = (accept_w >= 0) & worker_unmatched & firm_unmatched[safe_aw] & firm_did_propose_to_me
    if interview_mode == INTERVIEW_MODE_EXCLUSIVE:
        worker_did_propose = state.interview_proposals_w_to_f != -1
        valid_aw = valid_aw & (~worker_did_propose)
    new_accept_w = jnp.where(valid_aw, accept_w, -1)

    # Firm accept choice
    accept_f = _to_target_idx(firm_act, Nw)
    safe_af = jnp.clip(accept_f, 0, max(Nw - 1, 0))
    worker_did_propose_to_me = state.interview_proposals_w_to_f[safe_af] == arange_f
    if allow_on_the_job_search:
        not_already_matched_pair_f = ~state.matched[safe_af, arange_f]
        valid_af = (accept_f >= 0) & not_already_matched_pair_f & worker_did_propose_to_me
    else:
        valid_af = (accept_f >= 0) & firm_unmatched & worker_unmatched[safe_af] & worker_did_propose_to_me
    if interview_mode == INTERVIEW_MODE_EXCLUSIVE:
        firm_did_propose = state.interview_proposals_f_to_w != -1
        valid_af = valid_af & (~firm_did_propose)
    new_accept_f = jnp.where(valid_af, accept_f, -1)

    # Build candidate edges (Nw, Nf)
    worker_proposed = (state.interview_proposals_w_to_f[:, None] == arange_f[None, :])
    firm_accepted = (new_accept_f[None, :] == arange_w[:, None])
    firm_proposed = (state.interview_proposals_f_to_w[None, :] == arange_w[:, None])
    worker_accepted = (new_accept_w[:, None] == arange_f[None, :])

    candidate = (worker_proposed & firm_accepted) | (firm_proposed & worker_accepted)
    if allow_on_the_job_search:
        # Same relax: allow already-matched pairs to interview, except no
        # self-loops with one's own current match.
        candidate = candidate & (~state.matched)
    else:
        candidate = candidate & worker_unmatched[:, None] & firm_unmatched[None, :]

    final_edges = _capacity_resolve(
        candidate,
        max_per_worker=max_per_step,
        max_per_firm=max_per_step,
        receiver_side_order=receiver_side_order,
    )

    # Sample interview noise and update beliefs.
    #
    # Semantics: pairs that have ever been retained (cumulative_tenure > 0) keep
    # whatever belief retention left them with — re-interviewing such a pair
    # does NOT overwrite hat_x/hat_y with a fresh noisy observation. This
    # encodes the modeling intent that two agents who once worked together
    # already know each other; sitting through another interview shouldn't
    # erase that knowledge.
    d = state.x.shape[1]
    rng_w, rng_f = jax.random.split(rng)
    eps_w = jax.random.normal(rng_w, shape=(Nw, Nf, d)) * const.sigma_interview
    eps_f = jax.random.normal(rng_f, shape=(Nw, Nf, d)) * const.sigma_interview

    new_hat_x_pair = state.x[:, None, :] + eps_w
    new_hat_y_pair = state.y[None, :, :] + eps_f
    if public_retention_signal:
        # Once worker i has been retained anywhere, hat_x[i, :, :] holds the
        # public retention signal -- strictly more informative than a fresh
        # interview obs (sigmoid(lambda*tau)*x_i + small eps_match vs
        # x_i + larger eps_interview), so don't overwrite. Symmetric on firm
        # side. Note hat_x and hat_y use *different* masks under this flag,
        # because worker / firm "ever-public" status are independent.
        worker_ever_retained = (state.cumulative_tenure > 0).any(axis=1)   # (Nw,)
        firm_ever_retained   = (state.cumulative_tenure > 0).any(axis=0)   # (Nf,)
        mask_x = (final_edges & (~worker_ever_retained[:, None]))[:, :, None]
        mask_y = (final_edges & (~firm_ever_retained[None, :]))[:, :, None]
        new_hat_x = jnp.where(mask_x, new_hat_x_pair, state.hat_x)
        new_hat_y = jnp.where(mask_y, new_hat_y_pair, state.hat_y)
    else:
        ever_matched = state.cumulative_tenure > 0                   # (Nw, Nf)
        update_mask = (final_edges & (~ever_matched))[:, :, None]
        new_hat_x = jnp.where(update_mask, new_hat_x_pair, state.hat_x)
        new_hat_y = jnp.where(update_mask, new_hat_y_pair, state.hat_y)

    new_interviewed = state.interviewed | final_edges
    interviews_this = jnp.sum(final_edges).astype(jnp.int32)

    new_state = state.replace(
        interview_accept_w=new_accept_w,
        interview_accept_f=new_accept_f,
        hat_x=new_hat_x,
        hat_y=new_hat_y,
        interviewed=new_interviewed,
        interviews_this_period=interviews_this,
        total_interviews=state.total_interviews + interviews_this,
        phase=jnp.int32(MATCH_PROPOSE),
        _step=state._step + 1,
    )
    rew_w = jnp.zeros((Nw,), dtype=jnp.float32)
    rew_f = jnp.zeros((Nf,), dtype=jnp.float32)
    return new_state, rew_w, rew_f


def phase_match_propose(
    rng: jax.Array,
    const: HireRLConst,
    state: HireRLState,
    worker_act: jax.Array,
    firm_act: jax.Array,
    Nw: int,
    Nf: int,
    interview_mode: str,
    receiver_side_order: str,
    max_per_step: int,
    allow_on_the_job_search: bool = False,  # accepted for dispatch symmetry; unused
    public_retention_signal: bool = False,  # accepted for dispatch symmetry; unused
):
    arange_w = jnp.arange(Nw)
    worker_unmatched = _row_unmatched(state.matched)
    firm_unmatched = _col_unmatched(state.matched)

    target_w = _to_target_idx(worker_act, Nf)
    safe_w = jnp.clip(target_w, 0, max(Nf - 1, 0))
    interviewed_pair = state.interviewed[arange_w, safe_w]
    valid = (target_w >= 0) & worker_unmatched & firm_unmatched[safe_w] & interviewed_pair
    new_match_prop = jnp.where(valid, target_w, -1)

    new_state = state.replace(
        match_proposals_w_to_f=new_match_prop,
        phase=jnp.int32(MATCH_RESPOND),
        _step=state._step + 1,
    )
    rew_w = jnp.zeros((Nw,), dtype=jnp.float32)
    rew_f = jnp.zeros((Nf,), dtype=jnp.float32)
    return new_state, rew_w, rew_f


def phase_match_respond(
    rng: jax.Array,
    const: HireRLConst,
    state: HireRLState,
    worker_act: jax.Array,
    firm_act: jax.Array,
    Nw: int,
    Nf: int,
    interview_mode: str,
    receiver_side_order: str,
    max_per_step: int,
    allow_on_the_job_search: bool = False,  # accepted for dispatch symmetry; unused
    public_retention_signal: bool = False,  # accepted for dispatch symmetry; unused
):
    arange_w = jnp.arange(Nw)
    arange_f = jnp.arange(Nf)
    worker_unmatched = _row_unmatched(state.matched)
    firm_unmatched = _col_unmatched(state.matched)

    accept_f = _to_target_idx(firm_act, Nw)
    safe_af = jnp.clip(accept_f, 0, max(Nw - 1, 0))
    worker_did_propose_to_me = state.match_proposals_w_to_f[safe_af] == arange_f
    interviewed_pair = state.interviewed[safe_af, arange_f]
    valid = (accept_f >= 0) & firm_unmatched & worker_unmatched[safe_af] & worker_did_propose_to_me & interviewed_pair
    new_match_acc = jnp.where(valid, accept_f, -1)

    accepted_pair = (new_match_acc[None, :] == arange_w[:, None])         # (Nw, Nf)
    proposed_pair = (state.match_proposals_w_to_f[:, None] == arange_f[None, :])
    tentative = accepted_pair & proposed_pair
    tentative = tentative & worker_unmatched[:, None] & firm_unmatched[None, :]

    new_state = state.replace(
        match_accept_f=new_match_acc,
        tentative_matched=tentative,
        phase=jnp.int32(RETENTION),
        _step=state._step + 1,
    )
    rew_w = jnp.zeros((Nw,), dtype=jnp.float32)
    rew_f = jnp.zeros((Nf,), dtype=jnp.float32)
    return new_state, rew_w, rew_f


def phase_retention(
    rng: jax.Array,
    const: HireRLConst,
    state: HireRLState,
    worker_act: jax.Array,
    firm_act: jax.Array,
    Nw: int,
    Nf: int,
    interview_mode: str,
    receiver_side_order: str,
    max_per_step: int,
    allow_on_the_job_search: bool = False,  # accepted for dispatch symmetry; unused
    public_retention_signal: bool = False,
):
    d = state.x.shape[1]
    candidate = state.matched | state.tentative_matched

    worker_retain = (worker_act == 1)
    firm_retain = (firm_act == 1)
    retained = candidate & worker_retain[:, None] & firm_retain[None, :]
    dissolved = candidate & (~retained)

    cumulative_tenure_new = jnp.where(retained, state.cumulative_tenure + 1, state.cumulative_tenure)
    current_tenure_new = jnp.where(
        retained,
        state.current_tenure + 1,
        jnp.where(dissolved, jnp.zeros_like(state.current_tenure), state.current_tenure),
    )

    rng_w, rng_f = jax.random.split(rng)
    eps_w = jax.random.normal(rng_w, (Nw, Nf, d)) * const.sigma_match
    eps_f = jax.random.normal(rng_f, (Nw, Nf, d)) * const.sigma_match

    reveal_factor = jax.nn.sigmoid(
        const.lambda_reveal * cumulative_tenure_new.astype(jnp.float32)
    )                                                        # (Nw, Nf)
    new_hat_x_pair = reveal_factor[:, :, None] * state.x[:, None, :] + eps_w
    new_hat_y_pair = reveal_factor[:, :, None] * state.y[None, :, :] + eps_f

    if public_retention_signal:
        # Each worker / firm has at most one True in their row / column under
        # the matching capacity = 1 invariant. Sum-mask collapses the per-pair
        # signal to a per-worker (or per-firm) signal, then broadcast across
        # the other axis: every firm sees the same signal about a retained
        # worker (single-eps shared public observation).
        retained_f = retained.astype(jnp.float32)
        worker_signal_x = jnp.sum(new_hat_x_pair * retained_f[:, :, None], axis=1)   # (Nw, d)
        firm_signal_y = jnp.sum(new_hat_y_pair * retained_f[:, :, None], axis=0)     # (Nf, d)
        broadcast_hat_x = jnp.broadcast_to(worker_signal_x[:, None, :], state.hat_x.shape)
        broadcast_hat_y = jnp.broadcast_to(firm_signal_y[None, :, :],  state.hat_y.shape)
        worker_was_retained = retained.any(axis=1)            # (Nw,)
        firm_was_retained = retained.any(axis=0)              # (Nf,)
        new_hat_x = jnp.where(worker_was_retained[:, None, None], broadcast_hat_x, state.hat_x)
        new_hat_y = jnp.where(firm_was_retained[None, :, None],  broadcast_hat_y, state.hat_y)
    else:
        edge_mask = retained[:, :, None]
        new_hat_x = jnp.where(edge_mask, new_hat_x_pair, state.hat_x)
        new_hat_y = jnp.where(edge_mask, new_hat_y_pair, state.hat_y)

    reward_per_pair_w = jnp.sum(state.x[:, None, :] * new_hat_y, axis=-1)        # (Nw, Nf)
    reward_per_pair_f = jnp.sum(new_hat_x * state.y[None, :, :], axis=-1)        # (Nw, Nf)
    rew_w = jnp.sum(retained.astype(jnp.float32) * reward_per_pair_w, axis=1)    # (Nw,)
    rew_f = jnp.sum(retained.astype(jnp.float32) * reward_per_pair_f, axis=0)    # (Nf,)

    matches_formed = jnp.sum(retained & (~state.matched)).astype(jnp.int32)
    dissolutions = jnp.sum(dissolved).astype(jnp.int32)

    new_state = state.replace(
        prev_matched=state.matched,
        matched=retained,
        tentative_matched=jnp.zeros((Nw, Nf), dtype=jnp.bool_),
        cumulative_tenure=cumulative_tenure_new,
        current_tenure=current_tenure_new,
        hat_x=new_hat_x,
        hat_y=new_hat_y,
        last_reward_w=rew_w,
        last_reward_f=rew_f,
        market_t=state.market_t + 1,
        matches_formed_this_period=matches_formed,
        dissolutions_this_period=dissolutions,
        total_matches_formed=state.total_matches_formed + matches_formed,
        total_dissolutions=state.total_dissolutions + dissolutions,
        phase=jnp.int32(INTERVIEW_PROPOSE),
        _step=state._step + 1,
    )
    return new_state, rew_w, rew_f
