"""Per-phase action masks for HireRL workers and firms.

Masks are computed by stacking the per-phase masks and selecting via
``state.phase`` so the result is JAX-/vmap-friendly.
"""

import jax
import jax.numpy as jnp

from .constants import (
    INTERVIEW_MODE_EXCLUSIVE,
    NUM_PHASES,
)
from .state import HireRLState


def _row_unmatched(matched: jax.Array) -> jax.Array:
    return ~jnp.any(matched, axis=1)


def _col_unmatched(matched: jax.Array) -> jax.Array:
    return ~jnp.any(matched, axis=0)


def worker_action_masks_all_phases(
    state: HireRLState,
    Nw: int,
    Nf: int,
    interview_mode: str,
    num_choices: int,
    allow_on_the_job_search: bool = False,
) -> jax.Array:
    """Stack of (NUM_PHASES, Nw, num_choices) bool action masks for workers."""
    arange_w = jnp.arange(Nw)
    arange_f = jnp.arange(Nf)

    worker_unmatched = _row_unmatched(state.matched)
    firm_unmatched = _col_unmatched(state.matched)

    # match_valid_pair: both sides currently unmatched. Required for
    # MATCH_PROPOSE so that committing to a new match still requires prior
    # dissolution.
    match_valid_pair = worker_unmatched[:, None] & firm_unmatched[None, :]   # (Nw, Nf)
    # interview_valid_pair: same as above by default; under
    # allow_on_the_job_search, relaxed to "any pair except the worker's
    # current match", so already-matched agents can interview alternatives.
    if allow_on_the_job_search:
        interview_valid_pair = ~state.matched
    else:
        interview_valid_pair = match_valid_pair

    base_false = jnp.zeros((Nw, num_choices), dtype=jnp.bool_)
    noop_only = base_false.at[:, 0].set(True)

    def fill_pair_mask(valid_pair_mask: jax.Array) -> jax.Array:
        m = noop_only
        if Nf > 0:
            m = m.at[:, 1:Nf + 1].set(valid_pair_mask)
        return m

    # INTERVIEW_PROPOSE
    m_propose = fill_pair_mask(interview_valid_pair)

    # INTERVIEW_RESPOND
    firm_proposed = (state.interview_proposals_f_to_w[None, :] == arange_w[:, None])  # (Nw, Nf)
    valid_resp = firm_proposed & interview_valid_pair
    if interview_mode == INTERVIEW_MODE_EXCLUSIVE:
        worker_did_propose = state.interview_proposals_w_to_f != -1
        valid_resp = valid_resp & (~worker_did_propose)[:, None]
    m_respond = fill_pair_mask(valid_resp)

    # MATCH_PROPOSE -- always requires both unmatched (switching matches
    # still goes through dissolution + re-match, regardless of the flag).
    valid_match = match_valid_pair & state.interviewed
    m_match_propose = fill_pair_mask(valid_match)

    # MATCH_RESPOND -- workers can only NOOP
    m_match_respond = noop_only

    # RETENTION
    candidate = state.matched | state.tentative_matched
    has_candidate = jnp.any(candidate, axis=1)
    m_retention = noop_only
    if num_choices >= 2:
        m_retention = m_retention.at[:, 1].set(has_candidate)

    return jnp.stack([m_propose, m_respond, m_match_propose, m_match_respond, m_retention], axis=0)


def firm_action_masks_all_phases(
    state: HireRLState,
    Nw: int,
    Nf: int,
    interview_mode: str,
    num_choices: int,
    allow_on_the_job_search: bool = False,
) -> jax.Array:
    """Stack of (NUM_PHASES, Nf, num_choices) bool action masks for firms."""
    arange_w = jnp.arange(Nw)
    arange_f = jnp.arange(Nf)

    worker_unmatched = _row_unmatched(state.matched)
    firm_unmatched = _col_unmatched(state.matched)

    # See worker_action_masks_all_phases for the match/interview valid_pair
    # split rationale. Firm-side matrix is the transpose: shape (Nf, Nw).
    match_valid_pair = firm_unmatched[:, None] & worker_unmatched[None, :]   # (Nf, Nw)
    if allow_on_the_job_search:
        interview_valid_pair = ~state.matched.T
    else:
        interview_valid_pair = match_valid_pair

    base_false = jnp.zeros((Nf, num_choices), dtype=jnp.bool_)
    noop_only = base_false.at[:, 0].set(True)

    def fill_pair_mask(valid_pair_mask: jax.Array) -> jax.Array:
        m = noop_only
        if Nw > 0:
            m = m.at[:, 1:Nw + 1].set(valid_pair_mask)
        return m

    # INTERVIEW_PROPOSE
    m_propose = fill_pair_mask(interview_valid_pair)

    # INTERVIEW_RESPOND
    worker_proposed = (state.interview_proposals_w_to_f[None, :] == arange_f[:, None])  # (Nf, Nw)
    valid_resp = worker_proposed & interview_valid_pair
    if interview_mode == INTERVIEW_MODE_EXCLUSIVE:
        firm_did_propose = state.interview_proposals_f_to_w != -1
        valid_resp = valid_resp & (~firm_did_propose)[:, None]
    m_respond = fill_pair_mask(valid_resp)

    # MATCH_PROPOSE -- firms can only NOOP
    m_match_propose = noop_only

    # MATCH_RESPOND -- always requires both unmatched (regardless of flag).
    worker_match_proposed = (state.match_proposals_w_to_f[None, :] == arange_f[:, None])  # (Nf, Nw)
    valid_match = worker_match_proposed & match_valid_pair & state.interviewed.T
    m_match_respond = fill_pair_mask(valid_match)

    # RETENTION
    candidate = state.matched | state.tentative_matched   # (Nw, Nf)
    has_candidate = jnp.any(candidate, axis=0)            # (Nf,)
    m_retention = noop_only
    if num_choices >= 2:
        m_retention = m_retention.at[:, 1].set(has_candidate)

    return jnp.stack([m_propose, m_respond, m_match_propose, m_match_respond, m_retention], axis=0)


def select_phase_mask(stacked_masks: jax.Array, phase: jax.Array) -> jax.Array:
    """Select the per-phase action mask. Works under vmap."""
    return stacked_masks[phase]
