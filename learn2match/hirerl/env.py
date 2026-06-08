"""HireRL multi-agent environment.

Two roles are exposed (worker = agent 0, firm = agent 1).
Inside one phase-level env step, both worker and firm actions are passed in
``[worker_action, firm_action]`` form (single ``"choice"`` discrete head each).
"""

from typing import Any, Sequence

import jax
import jax.numpy as jnp

from jax_pbt.env.base_env import BaseEnv
from jax_pbt.env.spaces import (
    Action,
    ActionSpace,
    Observation,
    ObservationSpace,
)

from .config import HireRLConfig
from .constants import (
    INTERVIEW_PROPOSE,
    INTERVIEW_RESPOND,
    MATCH_PROPOSE,
    MATCH_RESPOND,
    NUM_PHASES,
    RETENTION,
)
from .metrics import (
    compute_all_metrics,
    compute_all_metrics_zero,
    is_settled_transition,
)
from .observations import (
    build_firm_observation,
    build_firm_observation_space,
    build_worker_observation,
    build_worker_observation_space,
)
from .state import HireRLConst, HireRLState
from .transitions import (
    phase_interview_propose,
    phase_interview_respond,
    phase_match_propose,
    phase_match_respond,
    phase_retention,
)


class HireRLEnv(BaseEnv[HireRLConst, HireRLState]):
    """Phase-level HireRL environment exposing two RL roles."""

    def __init__(self, config: HireRLConfig | None = None) -> None:
        if config is None:
            config = HireRLConfig()
        self.config = config

        self.Nw = int(config.Nw)
        self.Nf = int(config.Nf)
        self.d = int(config.d)
        self.horizon = int(config.horizon)
        self.interview_mode = config.interview_mode
        self.receiver_side_order = config.receiver_side_order
        self.public_matching_visibility = config.public_matching_visibility
        self.max_interviews_per_step = int(config.max_interviews_per_agent_per_step)
        self.allow_on_the_job_search = bool(config.allow_on_the_job_search)
        self.public_retention_signal = bool(config.public_retention_signal)
        self._compute_settled_metrics = bool(config.compute_settled_metrics)

        self.num_worker_choices = max(self.Nf + 1, 2)
        self.num_firm_choices = max(self.Nw + 1, 2)

        self._sigma_interview = float(config.sigma_interview)
        self._sigma_match = float(config.sigma_match)
        self._lambda_reveal = float(config.lambda_reveal)
        self._outside_option = float(config.outside_option)

        self._worker_obs_space = build_worker_observation_space(
            Nw=self.Nw, Nf=self.Nf, d=self.d,
            num_worker_choices=self.num_worker_choices,
            visibility=self.public_matching_visibility,
        )
        self._firm_obs_space = build_firm_observation_space(
            Nw=self.Nw, Nf=self.Nf, d=self.d,
            num_firm_choices=self.num_firm_choices,
            visibility=self.public_matching_visibility,
        )

    # -- BaseEnv API ---------------------------------------------------------

    @property
    def default_const(self) -> HireRLConst:
        return HireRLConst(
            max_steps=NUM_PHASES * self.horizon,
            sigma_interview=jnp.float32(self._sigma_interview),
            sigma_match=jnp.float32(self._sigma_match),
            lambda_reveal=jnp.float32(self._lambda_reveal),
            outside_option=jnp.float32(self._outside_option),
        )

    @property
    def env_observation_space(self) -> tuple[ObservationSpace, dict[str, tuple[int]]]:
        # The two roles do not share an internal-agent ObservationSpace, so the
        # combined env_observation_space is meaningful only via get_observation_space.
        # Provide the worker space as a representative.
        return self._worker_obs_space, {k: (self.Nw,) for k in self._worker_obs_space.keys()}

    @property
    def env_action_space(self) -> tuple[ActionSpace, dict[str, tuple[int]]]:
        worker_action = ActionSpace({"choice": self.num_worker_choices})
        return worker_action, {"choice": (self.Nw,)}

    @property
    def num_agents(self) -> int:
        return 2

    def get_observation_space(self) -> list[ObservationSpace]:
        return [self._worker_obs_space, self._firm_obs_space]

    def get_action_space(self) -> list[ActionSpace]:
        return [
            ActionSpace({"choice": self.num_worker_choices}),
            ActionSpace({"choice": self.num_firm_choices}),
        ]

    def get_agent_batch_shape(self) -> list[tuple[int]]:
        return [(self.Nw,), (self.Nf,)]

    # -- internals -----------------------------------------------------------

    def _initial_state(self, rng: jax.Array) -> HireRLState:
        # Keep the original 2-way split so the (x, y) draw is bit-exact
        # identical to the pre-noisy_hat_init code path for the same seed.
        # When noisy_hat_init=True, derive the noise rngs via fold_in so the
        # default branch's rng chain is unaffected.
        rng_x, rng_y = jax.random.split(rng)
        x = jax.random.normal(rng_x, (self.Nw, self.d))
        y = jax.random.normal(rng_y, (self.Nf, self.d))
        if self.config.non_negative_features:
            # Half-normal: every component is non-negative, so x[i]·y[j] >= 0
            # for all pairs and outside_option=0 doesn't reject anyone.
            x = jnp.abs(x)
            y = jnp.abs(y)
        Nw, Nf = self.Nw, self.Nf
        if self.config.noisy_hat_init:
            rng_hx = jax.random.fold_in(rng, 1)
            rng_hy = jax.random.fold_in(rng, 2)
            sigma_init = jnp.float32(self.config.sigma_init)
            eps_hx = jax.random.normal(rng_hx, (Nw, Nf, self.d)) * sigma_init
            eps_hy = jax.random.normal(rng_hy, (Nw, Nf, self.d)) * sigma_init
            hat_x_init = x[:, None, :] + eps_hx                     # (Nw, Nf, d)
            hat_y_init = y[None, :, :] + eps_hy                     # broadcast to (Nw, Nf, d)
        else:
            init_v = jnp.float32(self.config.hat_init_value)
            hat_x_init = jnp.full((Nw, Nf, self.d), init_v, dtype=jnp.float32)
            hat_y_init = jnp.full((Nw, Nf, self.d), init_v, dtype=jnp.float32)
        return HireRLState(
            _step=jnp.int32(0),
            market_t=jnp.int32(0),
            phase=jnp.int32(INTERVIEW_PROPOSE),
            x=x.astype(jnp.float32),
            y=y.astype(jnp.float32),
            hat_x=hat_x_init.astype(jnp.float32),
            hat_y=hat_y_init.astype(jnp.float32),
            interviewed=jnp.zeros((Nw, Nf), dtype=jnp.bool_),
            matched=jnp.zeros((Nw, Nf), dtype=jnp.bool_),
            tentative_matched=jnp.zeros((Nw, Nf), dtype=jnp.bool_),
            prev_matched=jnp.zeros((Nw, Nf), dtype=jnp.bool_),
            current_tenure=jnp.zeros((Nw, Nf), dtype=jnp.int32),
            cumulative_tenure=jnp.zeros((Nw, Nf), dtype=jnp.int32),
            last_reward_w=jnp.zeros((Nw,), dtype=jnp.float32),
            last_reward_f=jnp.zeros((Nf,), dtype=jnp.float32),
            interview_proposals_w_to_f=jnp.full((Nw,), -1, dtype=jnp.int32),
            interview_proposals_f_to_w=jnp.full((Nf,), -1, dtype=jnp.int32),
            interview_accept_w=jnp.full((Nw,), -1, dtype=jnp.int32),
            interview_accept_f=jnp.full((Nf,), -1, dtype=jnp.int32),
            match_proposals_w_to_f=jnp.full((Nw,), -1, dtype=jnp.int32),
            match_accept_f=jnp.full((Nf,), -1, dtype=jnp.int32),
            interviews_this_period=jnp.int32(0),
            matches_formed_this_period=jnp.int32(0),
            dissolutions_this_period=jnp.int32(0),
            total_interviews=jnp.int32(0),
            total_matches_formed=jnp.int32(0),
            total_dissolutions=jnp.int32(0),
        )

    def _build_obs_lst(self, state: HireRLState) -> list[Observation]:
        worker_obs = build_worker_observation(
            state, self.Nw, self.Nf, self.d,
            interview_mode=self.interview_mode,
            num_worker_choices=self.num_worker_choices,
            visibility=self.public_matching_visibility,
            allow_on_the_job_search=self.allow_on_the_job_search,
        )
        firm_obs = build_firm_observation(
            state, self.Nw, self.Nf, self.d,
            interview_mode=self.interview_mode,
            num_firm_choices=self.num_firm_choices,
            visibility=self.public_matching_visibility,
            allow_on_the_job_search=self.allow_on_the_job_search,
        )
        return [worker_obs, firm_obs]

    def _phase_dispatch(
        self,
        rng: jax.Array,
        const: HireRLConst,
        state: HireRLState,
        worker_act: jax.Array,
        firm_act: jax.Array,
    ) -> tuple[HireRLState, jax.Array, jax.Array]:
        kwargs = dict(
            Nw=self.Nw, Nf=self.Nf,
            interview_mode=self.interview_mode,
            receiver_side_order=self.receiver_side_order,
            max_per_step=self.max_interviews_per_step,
            allow_on_the_job_search=self.allow_on_the_job_search,
            public_retention_signal=self.public_retention_signal,
        )
        # All five branches are computed; selection happens via a state pytree
        # select. This works under vmap (each env may be on a different phase).
        branches = []
        for fn in (
            phase_interview_propose,
            phase_interview_respond,
            phase_match_propose,
            phase_match_respond,
            phase_retention,
        ):
            branches.append(fn(rng, const, state, worker_act, firm_act, **kwargs))

        states = [b[0] for b in branches]
        rew_ws = jnp.stack([b[1] for b in branches], axis=0)        # (5, Nw)
        rew_fs = jnp.stack([b[2] for b in branches], axis=0)        # (5, Nf)

        def select_state_field(*fields):
            stacked = jnp.stack(fields, axis=0)
            return stacked[state.phase]

        new_state = jax.tree_util.tree_map(select_state_field, *states)
        rew_w = rew_ws[state.phase]
        rew_f = rew_fs[state.phase]
        return new_state, rew_w, rew_f

    # -- public reset/step ---------------------------------------------------

    def env_reset(self, rng: jax.Array, const: HireRLConst) -> tuple[HireRLState, list[Observation]]:
        state = self._initial_state(rng)
        return state, self._build_obs_lst(state)

    def env_step(
        self,
        rng: jax.Array,
        const: HireRLConst,
        state: HireRLState,
        action: Sequence[Action],
    ):
        return self._step_impl(rng, const, state, action)

    def reset(self, rng: jax.Array, const: HireRLConst) -> tuple[HireRLState, list[Observation]]:
        return self.env_reset(rng, const)

    def step(
        self,
        rng: jax.Array,
        const: HireRLConst,
        state: HireRLState,
        action: Sequence[Action],
    ) -> tuple[HireRLState, list[Observation], list[jax.Array], list[jax.Array], dict[Any, Any]]:
        return self._step_impl(rng, const, state, action)

    def _step_impl(
        self,
        rng: jax.Array,
        const: HireRLConst,
        state: HireRLState,
        action: Sequence[Action],
    ):
        worker_act = action[0]["choice"].astype(jnp.int32)         # (Nw,)
        firm_act = action[1]["choice"].astype(jnp.int32)            # (Nf,)

        rng, rng_phase = jax.random.split(rng)
        new_state, rew_w, rew_f = self._phase_dispatch(rng_phase, const, state, worker_act, firm_act)

        # Done only after RETENTION transition increments market_t.
        was_retention = is_settled_transition(state.phase)
        done = was_retention & (new_state.market_t >= self.horizon)

        # Per-market-step metrics. Computed on `new_state` (PRE auto-reset) so
        # the final market step's snapshot survives the horizon-boundary
        # auto-reset; zeroed at non-settled phases so summing across env-steps
        # yields the paper's per-episode totals.
        #
        # When config.compute_settled_metrics=False (training envs), skip the
        # entire compute_all_metrics call -- training never reads info, and
        # the 3 Nw*Nf-iter DA loops inside it dominate env-step cost at large
        # N. When True (eval envs), still emit valid metrics: gate via
        # ``jnp.where`` -- under vmap ``lax.cond`` degrades to select anyway
        # (both branches computed), so ``where`` is no worse and simpler.
        if self._compute_settled_metrics:
            raw_metrics = compute_all_metrics(new_state, self._outside_option)
            settled_metrics = jax.tree_util.tree_map(
                lambda v: jnp.where(was_retention, v, jnp.zeros_like(v)),
                raw_metrics,
            )
        else:
            settled_metrics = compute_all_metrics_zero()

        rng, rng_reset = jax.random.split(rng)
        reset_state = self._initial_state(rng_reset)
        final_state = jax.tree_util.tree_map(
            lambda new, reset: jax.lax.select(done, reset, new),
            new_state, reset_state,
        )

        obs_lst = self._build_obs_lst(final_state)
        done_w = jnp.broadcast_to(done, (self.Nw,))
        done_f = jnp.broadcast_to(done, (self.Nf,))
        info = {
            "phase": new_state.phase,
            "market_t": new_state.market_t,
            "interviews_this_period": new_state.interviews_this_period,
            "matches_formed_this_period": new_state.matches_formed_this_period,
            "dissolutions_this_period": new_state.dissolutions_this_period,
            "is_settled": was_retention,
            "settled_metrics": settled_metrics,
        }
        return final_state, obs_lst, [rew_w, rew_f], [done_w, done_f], info
