"""Centralized pair-grid critic for HireRL MAPPO.

One shared critic produces the values of ALL Nw + Nf agents of an env in a
single forward pass -- the JaxMARL-style "replicate the world state once per
agent and run the critic N times" pattern is never used, so nothing in the
rollout buffer or the critic input scales O(N^2).

Input representation
--------------------
HireRL's global state is naturally a bipartite pair grid, not a flat
concatenation of per-agent observations:

* per-pair features  ``P[i, j]``: worker-side belief row ``hat_y[i, j, :]``,
  firm-side belief ``hat_x[i, j, :]``, matched / interviewed / tenure /
  proposal indicator scalars                       -> ``(..., Nw, Nf, k)``
* per-agent context: ``last_reward``, own unmatched bit
* global context: phase one-hot, public matched count (+ optional public
  matched-changed bit, + privileged market-time fraction)

Everything above is reconstructed from the two role observations that the
rollout already stores (worker obs rows stack into the full ``hat_y`` tensor,
firm obs rows into ``hat_x``; the boolean matrices are visible from either
side), so the centralized critic adds ZERO new rollout storage in strict
mode. In privileged mode the env-state extras (true ``x_i . y_j`` and
``market_t / horizon``) are stored once per env-step -- O(Nw*Nf) scalars,
with no per-agent replication either.

Architecture (parameters independent of Nw, Nf)
-----------------------------------------------
One shared pair encoder embeds every (worker, firm) pair once:

    E = LayerNorm(MLP(P))                          # (..., Nw, Nf, H)

then two pooled readouts emit all values simultaneously:

    V_w[i] = MLP([mean_j E[i,:], max_j E[i,:], worker_ctx[i], g])   # (..., Nw)
    V_f[j] = MLP([mean_i E[:,j], max_i E[:,j], firm_ctx[j],  g])    # (..., Nf)

where ``g`` mixes the global context with a grid-wide mean pool, so every
agent's value sees the whole market (congestion by other agents included).
All weight shapes depend only on ``d`` and the per-pair scalar count -- the
same parameters run on any market size, mirroring PerCandidateActorCritic.
"""

import os
import sys
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))   # repo root (for jax_pbt)
sys.path.insert(0, os.path.join(_HERE, ".."))          # learn2match (for hirerl)

from jax_pbt.env.spaces import Observation, ObservationSpace
from jax_pbt.model.nn_blocks import MLP

from hirerl.constants import NUM_PHASES


@dataclass(frozen=True)
class CriticInputSpec:
    """Static description of the critic inputs, derived from the role
    observation spaces plus env constants. Knows how wide each input block is
    so the critic can be initialized without touching a real observation."""

    Nw: int
    Nf: int
    d: int
    horizon: int
    privileged: bool
    has_matched_changed: bool

    # Per-pair scalar features appended after the two belief rows, in order:
    # matched, interviewed, current_tenure, cumulative_tenure,
    # retention_candidate, outgoing itv proposal (w->f), incoming itv
    # proposal (f->w), match proposal (w->f).
    NUM_PAIR_SCALARS = 8

    @property
    def pair_dim(self) -> int:
        return 2 * self.d + self.NUM_PAIR_SCALARS + (1 if self.privileged else 0)

    @property
    def worker_ctx_dim(self) -> int:
        return 2  # last_reward, own unmatched bit

    @property
    def firm_ctx_dim(self) -> int:
        return 2

    @property
    def global_ctx_dim(self) -> int:
        return (
            NUM_PHASES
            + 1                                         # public_matched_count
            + (1 if self.has_matched_changed else 0)    # public_matched_changed
            + (1 if self.privileged else 0)             # market_t / horizon
        )

    @classmethod
    def from_spaces(
        cls,
        worker_obs_space: ObservationSpace,
        Nw: int,
        Nf: int,
        d: int,
        horizon: int,
        privileged: bool,
    ) -> "CriticInputSpec":
        return cls(
            Nw=Nw, Nf=Nf, d=d, horizon=horizon,
            privileged=privileged,
            has_matched_changed="public_matched_changed" in worker_obs_space.keys(),
        )

    def zero_inputs(self, batch_shape: tuple[int, ...] = (1,)) -> dict[str, jax.Array]:
        """Zero-filled inputs with the right shapes, for module init."""
        return {
            "pair": jnp.zeros((*batch_shape, self.Nw, self.Nf, self.pair_dim)),
            "worker_ctx": jnp.zeros((*batch_shape, self.Nw, self.worker_ctx_dim)),
            "firm_ctx": jnp.zeros((*batch_shape, self.Nf, self.firm_ctx_dim)),
            "global_ctx": jnp.zeros((*batch_shape, self.global_ctx_dim)),
        }


def privileged_extras_from_state(env_state, horizon: int) -> dict[str, jax.Array]:
    """Per-env privileged critic inputs from the (vmapped) HireRLState.

    ``xy[e, i, j] = x[e, i] . y[e, j]`` is the sufficient statistic of the
    true expected match reward for the pair; ``market_frac`` gives the critic
    the time-to-horizon information that role observations do not carry.
    Both are legal under CTDE: the critic runs only at training time.
    """
    xy = jnp.einsum("...id,...jd->...ij", env_state.x, env_state.y)
    market_frac = env_state.market_t.astype(jnp.float32) / float(horizon)
    return {"xy": xy, "market_frac": market_frac}


def build_critic_inputs(
    spec: CriticInputSpec,
    worker_obs: Observation,
    firm_obs: Observation,
    extras: dict[str, jax.Array] | None = None,
) -> dict[str, jax.Array]:
    """Reconstruct the pair-grid critic inputs from the two role observations.

    [Input]
        - worker_obs: leaves shaped (*B, Nw, field_width) -- the per-worker
          observations of one env grouped on the second-to-last axis. ``*B``
          is any leading batch shape ((num_envs,) at rollout time,
          (chunk_length, minibatch_rows) at update time).
        - firm_obs: leaves shaped (*B, Nf, field_width).
        - extras: privileged extras (from ``privileged_extras_from_state``)
          with leaves (*B, Nw, Nf) / (*B,), required iff ``spec.privileged``.

    [Output]
        - dict(pair=(*B, Nw, Nf, pair_dim), worker_ctx=(*B, Nw, 2),
          firm_ctx=(*B, Nf, 2), global_ctx=(*B, global_ctx_dim)).

    Field mapping (see hirerl/observations.py for the forward direction):
        - worker ``hat_y`` rows stack into the full (Nw, Nf, d) tensor;
        - firm ``hat_x`` is stored (Nf, Nw*d) row-major from
          ``transpose(state.hat_x, (1, 0, 2))``, so reshape + swap the agent
          axes recovers (Nw, Nf, d);
        - worker ``own_match[..., 1:]`` is exactly ``state.matched``;
        - firm ``incoming_match_proposals`` is the (Nf, Nw) view of the
          worker->firm match-proposal matrix, transposed back here.
    """
    Nw, Nf, d = spec.Nw, spec.Nf, spec.d
    if spec.privileged and (extras is None or "xy" not in extras):
        raise ValueError("spec.privileged=True requires extras with 'xy'/'market_frac'.")

    hat_y = worker_obs["hat_y"]
    hat_y = hat_y.reshape(*hat_y.shape[:-1], Nf, d)                 # (*B, Nw, Nf, d)
    hat_x = firm_obs["hat_x"]
    hat_x = hat_x.reshape(*hat_x.shape[:-1], Nw, d)                 # (*B, Nf, Nw, d)
    hat_x = jnp.swapaxes(hat_x, -3, -2)                             # (*B, Nw, Nf, d)

    inv_h = 1.0 / float(spec.horizon)
    pair_feats = [
        hat_y,
        hat_x,
        worker_obs["own_match"][..., 1:][..., None],
        worker_obs["interviewed_firms"][..., None],
        (worker_obs["current_tenure"] * inv_h)[..., None],
        (worker_obs["cumulative_tenure"] * inv_h)[..., None],
        worker_obs["retention_candidate"][..., None],
        worker_obs["outgoing_interview_proposals"][..., None],
        worker_obs["incoming_interview_proposals"][..., None],
        jnp.swapaxes(firm_obs["incoming_match_proposals"], -2, -1)[..., None],
    ]
    if spec.privileged:
        pair_feats.append(extras["xy"][..., None])
    pair = jnp.concatenate(pair_feats, axis=-1)                     # (*B, Nw, Nf, k)

    worker_ctx = jnp.concatenate(
        [worker_obs["last_reward"], worker_obs["own_match"][..., :1]], axis=-1
    )                                                               # (*B, Nw, 2)
    firm_ctx = jnp.concatenate(
        [firm_obs["last_reward"], firm_obs["own_match"][..., :1]], axis=-1
    )                                                               # (*B, Nf, 2)

    # Global fields are broadcast identically to every worker; agent slot 0 is
    # an arbitrary representative.
    global_parts = [
        worker_obs["phase"][..., 0, :],                             # (*B, NUM_PHASES)
        worker_obs["public_matched_count"][..., 0, :],              # (*B, 1)
    ]
    if spec.has_matched_changed:
        global_parts.append(worker_obs["public_matched_changed"][..., 0, :])
    if spec.privileged:
        global_parts.append(extras["market_frac"][..., None])
    global_ctx = jnp.concatenate(global_parts, axis=-1)

    return {
        "pair": pair,
        "worker_ctx": worker_ctx,
        "firm_ctx": firm_ctx,
        "global_ctx": global_ctx,
    }


class PairGridCritic(nn.Module):
    """Shared centralized critic: one pair-encoder trunk, two pooled readouts.

    ``__call__(pair, worker_ctx, firm_ctx, global_ctx) -> (V_w, V_f)`` with
    ``V_w: (*B, Nw)`` and ``V_f: (*B, Nf)`` from a single forward pass.
    Feedforward on purpose: ``hat_x`` / ``hat_y`` are belief summaries of the
    whole interaction history, so the pair grid is (close to) a sufficient
    statistic and no critic RNN state is needed.
    """

    hidden_size: int = 64

    def setup(self) -> None:
        hidden = self.hidden_size
        self.pair_mlp = MLP(hidden_size_lst=[hidden, hidden])
        self.pair_norm = nn.LayerNorm()
        self.global_mlp = MLP(hidden_size_lst=[hidden])
        self.worker_head = MLP(hidden_size_lst=[hidden])
        self.worker_out = nn.Dense(1)
        self.firm_head = MLP(hidden_size_lst=[hidden])
        self.firm_out = nn.Dense(1)

    def __call__(
        self,
        pair: jax.Array,        # (*B, Nw, Nf, k)
        worker_ctx: jax.Array,  # (*B, Nw, cw)
        firm_ctx: jax.Array,    # (*B, Nf, cf)
        global_ctx: jax.Array,  # (*B, g)
    ) -> tuple[jax.Array, jax.Array]:
        embed = self.pair_norm(self.pair_mlp(pair))                 # (*B, Nw, Nf, H)

        row_mean = embed.mean(axis=-2)                              # (*B, Nw, H)
        row_max = embed.max(axis=-2)
        col_mean = embed.mean(axis=-3)                              # (*B, Nf, H)
        col_max = embed.max(axis=-3)
        grid_pool = embed.mean(axis=(-3, -2))                       # (*B, H)

        g = self.global_mlp(jnp.concatenate([global_ctx, grid_pool], axis=-1))
        g_w = jnp.broadcast_to(
            jnp.expand_dims(g, -2), (*row_mean.shape[:-1], g.shape[-1])
        )
        g_f = jnp.broadcast_to(
            jnp.expand_dims(g, -2), (*col_mean.shape[:-1], g.shape[-1])
        )

        v_w = self.worker_out(
            self.worker_head(
                jnp.concatenate([row_mean, row_max, worker_ctx, g_w], axis=-1)
            )
        ).squeeze(-1)                                               # (*B, Nw)
        v_f = self.firm_out(
            self.firm_head(
                jnp.concatenate([col_mean, col_max, firm_ctx, g_f], axis=-1)
            )
        ).squeeze(-1)                                               # (*B, Nf)
        return v_w, v_f

    def init_params(self, rng: jax.Array, spec: CriticInputSpec):
        return self.init(rng, **spec.zero_inputs())
