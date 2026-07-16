"""Per-candidate scoring actor-critic for HireRL (pointer-network style).

Replaces the flat "concatenate everything -> Dense(N+1)" policy with an
architecture whose parameter count is independent of market size:

* Each candidate j on the other side gets a small feature vector
  ``f_j = [belief row (d,), per-pair scalars]``, assembled inside the model
  from the existing observation fields (``hat_y``/``hat_x`` reshaped from
  ``(N*d,)`` to ``(N, d)``, plus every ``(N,)``-shaped scalar field).
* One shared encoder embeds every candidate; one shared additive scorer
  turns each embedding into a scalar logit, conditioned on a per-agent
  context vector (phase, last reward, public stats, own match status,
  pooled candidate summary, and the GRU memory when ``use_rnn``).
* Action 0 (NOOP) is scored through the same pathway using a learnable
  "null candidate" input.
* The masked logits ``[s_0, s_1..s_N]`` feed one Categorical: a softmax
  over N+1 scores. Every weight shape depends only on ``d`` and the number
  of per-pair scalars -- never on N -- so the same parameters run on any
  market size and are permutation-equivariant across candidates.

RETENTION special case: the env decodes action 1 as "retain my current
partner", not "choose candidate 0" (transitions.py). When
``phase == RETENTION`` the model swaps candidate slot 0's input for the
features of the current retention partner, gathered with the
``retention_candidate`` one-hot from the observation (which includes a
just-formed tentative match), so the retain/release decision is scored on
the right counterparty.

RNN placement: with ``use_rnn`` a single agent-level GRU runs on the
(pooled candidates + context) pathway; its output conditions both the
scorer and the value head. The recurrent state stays ``(batch, hidden)`` --
independent of N -- so the jax_pbt agent/trainer plumbing is unchanged.
"""

import flax.linen as nn
import flax.struct as struct
import jax
import jax.numpy as jnp

from jax_pbt.env.spaces import (
    ActionSpace,
    Observation,
    ObservationSpace,
)
from jax_pbt.model.action_distribution import ActionDistribution
from jax_pbt.model.nn_blocks import MLP, RNN
from jax_pbt.model.nn_blocks.distribution_layers import Categorical
from jax_pbt.policy.actor_critic import SharedActorCriticModel

from .constants import RETENTION

MASK_KEY = "action_mask"
ACTION_KEY = "choice"
LARGE_NEG = -1e9

# Candidate-independent observation keys, concatenated into the per-agent
# context vector (only those present in the obs space are used).
CONTEXT_KEYS = (
    "phase",
    "last_reward",
    "public_matched_count",
    "public_matched_changed",
)
# Flattened belief matrices, shape (num_candidates * d,). Exactly one of
# these must be present; its name also tells the model which side it is on
# (workers score firms via hat_y; firms score workers via hat_x).
HAT_KEYS = ("hat_x", "hat_y")
OWN_MATCH_KEY = "own_match"
RETENTION_CAND_KEY = "retention_candidate"
# Optional (Nw*Nf,) public matching matrix under VISIBILITY_MATCHING_PAIRS.
# Projected to a per-candidate "is taken" bit; the full pairing pattern is
# not representable with N-independent parameters.
PAIRS_KEY = "public_matching_pairs"


class PerCandidateActorCritic(nn.Module):
    obs_space: ObservationSpace = struct.field(pytree_node=False)
    action_space: ActionSpace = struct.field(pytree_node=False)
    common_hidden_size: int = 64
    use_rnn: bool = False
    rnn_hidden_layers: int = 1

    def setup(self) -> None:
        n_choices = self.action_space[ACTION_KEY].n
        self.num_candidates = n_choices - 1

        hat_keys = [k for k in self.obs_space.keys() if k in HAT_KEYS]
        if len(hat_keys) != 1:
            raise ValueError(
                f"Expected exactly one of {HAT_KEYS} in the observation "
                f"space, found {hat_keys}."
            )
        self.hat_key = hat_keys[0]
        hat_size = self.obs_space[self.hat_key].shape[0]
        if hat_size % self.num_candidates != 0:
            raise ValueError(
                f"obs[{self.hat_key}] has size {hat_size}, not divisible by "
                f"num_candidates={self.num_candidates} (= action_space.n - 1)."
            )
        self.d = hat_size // self.num_candidates

        if OWN_MATCH_KEY not in self.obs_space.keys():
            raise ValueError(f"Observation space is missing '{OWN_MATCH_KEY}'.")
        if self.obs_space[OWN_MATCH_KEY].shape != (self.num_candidates + 1,):
            raise ValueError(
                f"obs[{OWN_MATCH_KEY}] has shape "
                f"{self.obs_space[OWN_MATCH_KEY].shape}, expected "
                f"({self.num_candidates + 1},)."
            )
        if "phase" not in self.obs_space.keys():
            raise ValueError("Observation space is missing 'phase'.")

        # Every remaining key must be a per-candidate (num_candidates,)
        # scalar; anything else is ambiguous and rejected loudly.
        scalar_keys = []
        for k in self.obs_space.keys():
            if k in (MASK_KEY, OWN_MATCH_KEY, PAIRS_KEY, self.hat_key):
                continue
            if k in CONTEXT_KEYS:
                continue
            if self.obs_space[k].shape == (self.num_candidates,):
                scalar_keys.append(k)
            else:
                raise ValueError(
                    f"Observation key '{k}' with shape "
                    f"{self.obs_space[k].shape} is neither a known context "
                    f"key nor a per-candidate ({self.num_candidates},) field; "
                    f"teach PerCandidateActorCritic how to route it."
                )
        if RETENTION_CAND_KEY not in scalar_keys:
            raise ValueError(
                f"Observation space is missing '{RETENTION_CAND_KEY}' "
                f"(required to score the RETENTION phase on the current "
                f"partner; see observations.py)."
            )
        self.cand_scalar_keys = tuple(sorted(scalar_keys))
        self.ctx_keys = tuple(k for k in self.obs_space.keys() if k in CONTEXT_KEYS)
        self.has_pairs = PAIRS_KEY in self.obs_space.keys()

        # Per-candidate input width: belief row + scalar fields + own-match
        # bit (+ optional public "taken" bit). Independent of N.
        self.cand_dim = (
            self.d + len(self.cand_scalar_keys) + 1 + (1 if self.has_pairs else 0)
        )

        hidden = self.common_hidden_size
        self.cand_mlp = MLP(hidden_size_lst=[hidden, hidden])
        self.cand_norm = nn.LayerNorm()
        self.ctx_mlp = MLP(hidden_size_lst=[hidden, hidden])
        self.ctx_norm = nn.LayerNorm()
        self.core_dense = nn.Dense(hidden)
        if self.use_rnn:
            self.rnn_model = RNN(
                rnn_layers=self.rnn_hidden_layers,
                hidden_size=hidden,
                use_layer_norm=True,
            )
        # Learnable NOOP input, scored through the same encoder/scorer as
        # real candidates. nn.Embed creates its parameter lazily on first
        # call, keeping the param-free init_rnn_state apply() path valid.
        self.null_candidate = nn.Embed(
            num_embeddings=1,
            features=self.cand_dim,
            embedding_init=nn.initializers.zeros,
        )
        # Additive scorer: s = w . tanh(W_e e_j + W_z z). The context
        # projection is computed once per agent, not once per candidate.
        self.score_cand = nn.Dense(hidden)
        self.score_ctx = nn.Dense(hidden)
        self.score_out = nn.Dense(1)
        self.value_head = nn.Dense(1)

    def _candidate_features(self, obs: Observation) -> jax.Array:
        """Stack per-candidate inputs to (..., num_candidates, cand_dim).

        Concatenation order is fixed: belief row, sorted scalar keys,
        own-match bit, optional public "taken" bit. The null-candidate
        embedding must match this layout (it does by construction: it is a
        free parameter of the same width).
        """
        Nc = self.num_candidates
        hat = obs[self.hat_key]
        feats = [hat.reshape(*hat.shape[:-1], Nc, self.d)]
        for k in self.cand_scalar_keys:
            feats.append(obs[k][..., None])
        feats.append(obs[OWN_MATCH_KEY][..., 1:, None])
        if self.has_pairs:
            pairs = obs[PAIRS_KEY]
            if self.hat_key == "hat_y":
                # Worker side: candidates are firms = columns of (Nw, Nf).
                m = pairs.reshape(*pairs.shape[:-1], -1, Nc)
                taken = m.max(axis=-2)
            else:
                # Firm side: candidates are workers = rows of (Nw, Nf).
                m = pairs.reshape(*pairs.shape[:-1], Nc, -1)
                taken = m.max(axis=-1)
            feats.append(taken[..., None])
        return jnp.concatenate(feats, axis=-1)

    def __call__(self, rnn_state: jax.Array, obs: Observation):
        F = self._candidate_features(obs)                       # (..., Nc, Dc)

        # RETENTION: action 1 means "retain my current partner", so slot 0
        # must be scored on the partner's features, whoever that is.
        r = obs[RETENTION_CAND_KEY]                             # (..., Nc)
        partner = jnp.sum(F * r[..., None], axis=-2)            # (..., Dc)
        F_retention = jnp.concatenate(
            [partner[..., None, :], F[..., 1:, :]], axis=-2
        )
        is_retention = obs["phase"][..., RETENTION]             # (...,)
        F = jnp.where(is_retention[..., None, None] > 0.5, F_retention, F)

        context = jnp.concatenate(
            [obs[k] for k in self.ctx_keys] + [obs[OWN_MATCH_KEY][..., :1]],
            axis=-1,
        )

        cand_embed = self.cand_norm(self.cand_mlp(F))           # (..., Nc, H)
        ctx_embed = self.ctx_norm(self.ctx_mlp(context))        # (..., H)
        pooled = cand_embed.mean(axis=-2)                       # (..., H)
        core = nn.relu(
            self.core_dense(jnp.concatenate([ctx_embed, pooled], axis=-1))
        )                                                       # (..., H)
        if self.use_rnn:
            rnn_state, z = self.rnn_model(rnn_state, core)
        else:
            z = core

        # flax's Embed fast path for num_embeddings=1 broadcasts the table to
        # inputs.shape + (features,), which rejects scalar inputs; index with
        # shape (1,) and drop the axis instead.
        null_feat = self.null_candidate(jnp.zeros((1,), dtype=jnp.int32))[0]
        null_embed = self.cand_norm(self.cand_mlp(null_feat))   # (H,)

        ctx_proj = self.score_ctx(z)                            # (..., H)
        s_cand = self.score_out(
            jnp.tanh(self.score_cand(cand_embed) + ctx_proj[..., None, :])
        ).squeeze(-1)                                           # (..., Nc)
        s_null = self.score_out(
            jnp.tanh(self.score_cand(null_embed) + ctx_proj)
        )                                                       # (..., 1)

        logits = jnp.concatenate([s_null, s_cand], axis=-1)     # (..., Nc+1)
        mask = obs[MASK_KEY].astype(jnp.bool_)
        masked_logits = jnp.where(mask, logits, LARGE_NEG)
        pi = ActionDistribution({ACTION_KEY: Categorical(logits=masked_logits)})
        val = self.value_head(z).squeeze(-1)
        return rnn_state, pi, val

    def init_rnn_state(self, dummy_batch: jax.Array) -> jax.Array:
        if self.use_rnn:
            return jnp.zeros(
                (*dummy_batch.shape, self.rnn_hidden_layers * self.common_hidden_size)
            )
        return jnp.zeros(dummy_batch.shape)


class PerCandidateActorCriticModel(SharedActorCriticModel):
    """Wrap PerCandidateActorCritic so it plugs into ActorCriticPPOAgent/Trainer."""

    @classmethod
    def build(
        cls,
        obs_space: ObservationSpace,
        action_space: ActionSpace,
        hidden_size: int = 64,
        use_rnn: bool = False,
    ) -> "PerCandidateActorCriticModel":
        actor_critic_fn = PerCandidateActorCritic(
            obs_space=obs_space,
            action_space=action_space,
            common_hidden_size=hidden_size,
            use_rnn=use_rnn,
        )
        return cls(actor_critic_fn)
