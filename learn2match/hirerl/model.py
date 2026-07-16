"""Mask-aware shared actor-critic for HireRL.

The action_mask key is removed from the feature-extractor input and used to
mask the policy logits before constructing the Categorical distribution.
"""

from typing import Sequence

import flax.linen as nn
import flax.struct as struct
import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode

from jax_pbt.env.spaces import (
    ActionSpace,
    Observation,
    ObservationSpace,
)
from jax_pbt.model.action_distribution import ActionDistribution
from jax_pbt.model.feature_extractors import build_feature_extractor_with_consistent_layers
from jax_pbt.model.nn_blocks import MLP
from jax_pbt.model.nn_blocks.distribution_layers import Categorical
from jax_pbt.policy.actor_critic import SharedActorCriticModel


MASK_KEY = "action_mask"
ACTION_KEY = "choice"
LARGE_NEG = -1e9


class MaskedSharedActorCritic(nn.Module):
    obs_space: ObservationSpace = struct.field(pytree_node=False)
    action_space: ActionSpace = struct.field(pytree_node=False)
    common_hidden_size: int = 64
    use_rnn: bool = False

    def setup(self) -> None:
        feature_obs_space = ObservationSpace({
            k: v for k, v in self.obs_space.items() if k != MASK_KEY
        })
        self.feature_extractor = build_feature_extractor_with_consistent_layers(
            feature_obs_space,
            hidden_size=self.common_hidden_size,
            use_rnn=self.use_rnn,
            rnn_hidden_size=self.common_hidden_size,
        )
        n_choices = self.action_space[ACTION_KEY].n
        # The "set" observation (hat_y for workers, hat_x for firms) is the only
        # 2-D obs field; each of its elements gets one score from a shared scorer
        # that sees (global_feature, element_embedding). The remaining choices
        # (e.g. NOOP) are produced by a flat head.
        set_keys = [
            k for k, v in self.obs_space.items()
            if k != MASK_KEY and len(v.shape) == 2
        ]
        self._set_key = set_keys[0]
        n_set = self.obs_space[self._set_key].shape[0]
        self.scorer = MLP(hidden_size_lst=[self.common_hidden_size, 1])
        self.extra_head = nn.Dense(n_choices - n_set)
        self.value_head = nn.Dense(1)

    def __call__(self, rnn_state: jax.Array, obs: Observation):
        feature_obs = Observation({k: v for k, v in obs.items() if k != MASK_KEY})
        rnn_state, feature, firm_emb = self.feature_extractor(rnn_state, feature_obs)
        h = feature
        # Score each set element with the shared scorer over (global feature,
        # element embedding). Broadcast h across the element axis while keeping
        # its own feature width (the flat feature and element embedding widths
        # need not match), then concatenate along the last axis.
        h_b = jnp.broadcast_to(h[..., None, :], firm_emb.shape[:-1] + h.shape[-1:])
        firm_scores = self.scorer(jnp.concatenate([h_b, firm_emb], axis=-1)).squeeze(-1)
        # Logit order MUST match the env/action-mask layout, which is
        # [NOOP, partner_0, ..., partner_{N-1}] (see hirerl/action_masks.py:
        # index 0 is the NOOP slot, indices 1..N are the per-partner choices).
        # So the flat extra head (NOOP) goes FIRST, then the per-element scores.
        # Concatenating the scores first would shift every logit by one relative
        # to the mask, decoding each partner score as the wrong partner and
        # mis-masking NOOP -- which cripples learning.
        logits = jnp.concatenate([self.extra_head(h), firm_scores], axis=-1)
        mask = obs[MASK_KEY].astype(jnp.bool_)
        masked_logits = jnp.where(mask, logits, LARGE_NEG)
        pi = ActionDistribution({ACTION_KEY: Categorical(logits=masked_logits)})
        # Pool the set embeddings so the critic sees a permutation-invariant
        # summary of the set alongside the global feature.
        pooled = jnp.mean(firm_emb, axis=-2)
        val = self.value_head(jnp.concatenate([h, pooled], axis=-1))
        return rnn_state, pi, val.squeeze(-1)

    def init_rnn_state(self, dummy_batch: jax.Array) -> jax.Array:
        fe = self.feature_extractor
        if fe.use_rnn:
            return jnp.zeros(
                (*dummy_batch.shape, fe.rnn_hidden_layers * fe.rnn_hidden_size)
            )
        return jnp.zeros(dummy_batch.shape)


class MaskedSharedActorCriticModel(SharedActorCriticModel):
    """Wrap MaskedSharedActorCritic so it plugs into ActorCriticPPOAgent/Trainer."""

    @classmethod
    def build(
        cls,
        obs_space: ObservationSpace,
        action_space: ActionSpace,
        hidden_size: int = 64,
        use_rnn: bool = False,
    ) -> "MaskedSharedActorCriticModel":
        actor_critic_fn = MaskedSharedActorCritic(
            obs_space=obs_space,
            action_space=action_space,
            common_hidden_size=hidden_size,
            use_rnn=use_rnn,
        )
        return cls(actor_critic_fn)
