"""Per-role observations for HireRL."""

import jax
import jax.numpy as jnp

from jax_pbt.env.spaces import (
    ContinuousSpace,
    Observation,
    ObservationSpace,
)

from .action_masks import (
    firm_action_masks_all_phases,
    select_phase_mask,
    worker_action_masks_all_phases,
)
from .constants import (
    NUM_PHASES,
    VISIBILITY_MATCHED_STATUS,
    VISIBILITY_MATCHED_STATUS_AND_CHANGE,
    VISIBILITY_MATCHING_PAIRS,
)
from .state import HireRLState


def _own_match_one_hot(matched: jax.Array, side: str) -> jax.Array:
    """Return per-agent one-hot match vector. side ∈ {'worker','firm'}."""
    if side == "worker":
        # matched: (Nw, Nf) -> per-worker firm idx (or -1)
        any_match = jnp.any(matched, axis=1)
        firm_idx = jnp.argmax(matched.astype(jnp.int32), axis=1)
        idx = jnp.where(any_match, firm_idx + 1, 0)        # 0 = unmatched
        Nf = matched.shape[1]
        return jax.nn.one_hot(idx, Nf + 1, dtype=jnp.float32)
    else:
        any_match = jnp.any(matched, axis=0)
        worker_idx = jnp.argmax(matched.astype(jnp.int32), axis=0)
        idx = jnp.where(any_match, worker_idx + 1, 0)
        Nw = matched.shape[0]
        return jax.nn.one_hot(idx, Nw + 1, dtype=jnp.float32)


def _public_visibility_fields(state: HireRLState, visibility: str) -> dict:
    Nw, Nf = state.matched.shape
    norm = float(max(min(Nw, Nf), 1))
    fields = {}

    matched_count = jnp.array(jnp.sum(state.matched), dtype=jnp.float32) / norm
    fields["public_matched_count"] = matched_count[None]   # (1,)

    if visibility == VISIBILITY_MATCHED_STATUS:
        return fields

    if visibility == VISIBILITY_MATCHED_STATUS_AND_CHANGE:
        changed = jnp.any(state.matched != state.prev_matched).astype(jnp.float32)
        fields["public_matched_changed"] = changed[None]
        return fields

    if visibility == VISIBILITY_MATCHING_PAIRS:
        flat = state.matched.astype(jnp.float32).reshape(-1)  # (Nw*Nf,)
        fields["public_matching_pairs"] = flat
        return fields

    raise ValueError(f"Unknown public_matching_visibility: {visibility}")


def _public_visibility_space(visibility: str, Nw: int, Nf: int) -> dict:
    spaces = {
        "public_matched_count": ContinuousSpace((1,), low=0.0, high=1.0),
    }
    if visibility == VISIBILITY_MATCHED_STATUS_AND_CHANGE:
        spaces["public_matched_changed"] = ContinuousSpace((1,), low=0.0, high=1.0)
    elif visibility == VISIBILITY_MATCHING_PAIRS:
        spaces["public_matching_pairs"] = ContinuousSpace((Nw * Nf,), low=0.0, high=1.0)
    return spaces


def build_worker_observation_space(
    Nw: int,
    Nf: int,
    d: int,
    num_worker_choices: int,
    visibility: str,
) -> ObservationSpace:
    base = {
        "phase": ContinuousSpace((NUM_PHASES,), low=0.0, high=1.0),
        "own_match": ContinuousSpace((Nf + 1,), low=0.0, high=1.0),
        "incoming_interview_proposals": ContinuousSpace((Nf,), low=0.0, high=1.0),
        "outgoing_interview_proposals": ContinuousSpace((Nf,), low=0.0, high=1.0),
        "interviewed_firms": ContinuousSpace((Nf,), low=0.0, high=1.0),
        "hat_y": ContinuousSpace((Nf * d,), low=-10.0, high=10.0),
        "cumulative_tenure": ContinuousSpace((Nf,), low=0.0, high=1e6),
        "current_tenure": ContinuousSpace((Nf,), low=0.0, high=1e6),
        "last_reward": ContinuousSpace((1,), low=-1e6, high=1e6),
        "action_mask": ContinuousSpace((num_worker_choices,), low=0.0, high=1.0),
    }
    base.update(_public_visibility_space(visibility, Nw, Nf))
    return ObservationSpace(base)


def build_firm_observation_space(
    Nw: int,
    Nf: int,
    d: int,
    num_firm_choices: int,
    visibility: str,
) -> ObservationSpace:
    base = {
        "phase": ContinuousSpace((NUM_PHASES,), low=0.0, high=1.0),
        "own_match": ContinuousSpace((Nw + 1,), low=0.0, high=1.0),
        "incoming_interview_proposals": ContinuousSpace((Nw,), low=0.0, high=1.0),
        "outgoing_interview_proposals": ContinuousSpace((Nw,), low=0.0, high=1.0),
        "incoming_match_proposals": ContinuousSpace((Nw,), low=0.0, high=1.0),
        "interviewed_workers": ContinuousSpace((Nw,), low=0.0, high=1.0),
        "hat_x": ContinuousSpace((Nw * d,), low=-10.0, high=10.0),
        "cumulative_tenure": ContinuousSpace((Nw,), low=0.0, high=1e6),
        "current_tenure": ContinuousSpace((Nw,), low=0.0, high=1e6),
        "last_reward": ContinuousSpace((1,), low=-1e6, high=1e6),
        "action_mask": ContinuousSpace((num_firm_choices,), low=0.0, high=1.0),
    }
    base.update(_public_visibility_space(visibility, Nw, Nf))
    return ObservationSpace(base)


def build_worker_observation(
    state: HireRLState,
    Nw: int,
    Nf: int,
    d: int,
    interview_mode: str,
    num_worker_choices: int,
    visibility: str,
    allow_on_the_job_search: bool = False,
) -> Observation:
    arange_w = jnp.arange(Nw)
    arange_f = jnp.arange(Nf)

    phase_oh = jax.nn.one_hot(state.phase, NUM_PHASES, dtype=jnp.float32)            # (NUM_PHASES,)
    own_match = _own_match_one_hot(state.matched, side="worker")                     # (Nw, Nf+1)

    incoming = (state.interview_proposals_f_to_w[None, :] == arange_w[:, None]).astype(jnp.float32)  # (Nw, Nf)
    outgoing = (state.interview_proposals_w_to_f[:, None] == arange_f[None, :]).astype(jnp.float32)  # (Nw, Nf)
    interviewed = state.interviewed.astype(jnp.float32)
    hat_y_flat = state.hat_y.reshape(Nw, Nf * d)
    cum_ten = state.cumulative_tenure.astype(jnp.float32)
    cur_ten = state.current_tenure.astype(jnp.float32)
    last_r = state.last_reward_w[:, None]

    masks_all = worker_action_masks_all_phases(
        state, Nw, Nf, interview_mode, num_worker_choices,
        allow_on_the_job_search=allow_on_the_job_search,
    )
    mask = select_phase_mask(masks_all, state.phase)                                 # (Nw, num_choices)

    fields = {
        "phase": jnp.broadcast_to(phase_oh, (Nw, NUM_PHASES)),
        "own_match": own_match,
        "incoming_interview_proposals": incoming,
        "outgoing_interview_proposals": outgoing,
        "interviewed_firms": interviewed,
        "hat_y": hat_y_flat,
        "cumulative_tenure": cum_ten,
        "current_tenure": cur_ten,
        "last_reward": last_r,
        "action_mask": mask,
    }

    pub = _public_visibility_fields(state, visibility)
    for k, v in pub.items():
        # Broadcast public-feature row to all workers
        fields[k] = jnp.broadcast_to(v, (Nw, *v.shape))

    return Observation(fields)


def build_firm_observation(
    state: HireRLState,
    Nw: int,
    Nf: int,
    d: int,
    interview_mode: str,
    num_firm_choices: int,
    visibility: str,
    allow_on_the_job_search: bool = False,
) -> Observation:
    arange_w = jnp.arange(Nw)
    arange_f = jnp.arange(Nf)

    phase_oh = jax.nn.one_hot(state.phase, NUM_PHASES, dtype=jnp.float32)
    own_match = _own_match_one_hot(state.matched, side="firm")                       # (Nf, Nw+1)

    # incoming proposals to firm j: workers who proposed firm j
    incoming = (state.interview_proposals_w_to_f[None, :] == arange_f[:, None]).astype(jnp.float32)  # (Nf, Nw)
    outgoing = (state.interview_proposals_f_to_w[:, None] == arange_w[None, :]).astype(jnp.float32)  # (Nf, Nw)
    incoming_match = (state.match_proposals_w_to_f[None, :] == arange_f[:, None]).astype(jnp.float32)  # (Nf, Nw)
    interviewed_workers = state.interviewed.T.astype(jnp.float32)                    # (Nf, Nw)

    hat_x_flat = jnp.transpose(state.hat_x, (1, 0, 2)).reshape(Nf, Nw * d)           # (Nf, Nw*d)
    cum_ten = state.cumulative_tenure.T.astype(jnp.float32)                          # (Nf, Nw)
    cur_ten = state.current_tenure.T.astype(jnp.float32)                             # (Nf, Nw)
    last_r = state.last_reward_f[:, None]

    masks_all = firm_action_masks_all_phases(
        state, Nw, Nf, interview_mode, num_firm_choices,
        allow_on_the_job_search=allow_on_the_job_search,
    )
    mask = select_phase_mask(masks_all, state.phase)                                 # (Nf, num_choices)

    fields = {
        "phase": jnp.broadcast_to(phase_oh, (Nf, NUM_PHASES)),
        "own_match": own_match,
        "incoming_interview_proposals": incoming,
        "outgoing_interview_proposals": outgoing,
        "incoming_match_proposals": incoming_match,
        "interviewed_workers": interviewed_workers,
        "hat_x": hat_x_flat,
        "cumulative_tenure": cum_ten,
        "current_tenure": cur_ten,
        "last_reward": last_r,
        "action_mask": mask,
    }

    pub = _public_visibility_fields(state, visibility)
    for k, v in pub.items():
        fields[k] = jnp.broadcast_to(v, (Nf, *v.shape))

    return Observation(fields)
