"""HireRL state and per-step constants (PyTreeNode)."""

import jax

from jax_pbt.env.base_env import BaseEnvConst, BaseEnvState


class HireRLConst(BaseEnvConst):
    sigma_interview: jax.Array
    sigma_match: jax.Array
    lambda_reveal: jax.Array
    outside_option: jax.Array


class HireRLState(BaseEnvState):
    market_t: jax.Array
    phase: jax.Array

    x: jax.Array            # (Nw, d)
    y: jax.Array            # (Nf, d)
    hat_x: jax.Array        # (Nw, Nf, d) firm-side belief about worker
    hat_y: jax.Array        # (Nw, Nf, d) worker-side belief about firm

    interviewed: jax.Array          # (Nw, Nf) bool
    matched: jax.Array              # (Nw, Nf) bool
    tentative_matched: jax.Array    # (Nw, Nf) bool
    prev_matched: jax.Array         # (Nw, Nf) bool

    current_tenure: jax.Array       # (Nw, Nf) int32
    cumulative_tenure: jax.Array    # (Nw, Nf) int32

    last_reward_w: jax.Array        # (Nw,) float32
    last_reward_f: jax.Array        # (Nf,) float32

    interview_proposals_w_to_f: jax.Array   # (Nw,) int32, -1 = NOOP
    interview_proposals_f_to_w: jax.Array   # (Nf,) int32
    interview_accept_w: jax.Array           # (Nw,) int32
    interview_accept_f: jax.Array           # (Nf,) int32

    match_proposals_w_to_f: jax.Array       # (Nw,) int32
    match_accept_f: jax.Array               # (Nf,) int32

    interviews_this_period: jax.Array       # scalar int32
    matches_formed_this_period: jax.Array   # scalar int32
    dissolutions_this_period: jax.Array     # scalar int32

    total_interviews: jax.Array             # scalar int32
    total_matches_formed: jax.Array         # scalar int32
    total_dissolutions: jax.Array           # scalar int32
