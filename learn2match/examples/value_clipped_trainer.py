"""ActorCriticPPOTrainer with a corrected PPO value-clipping term.

jax_pbt's built-in ``value_clip`` is a no-op. It builds the clipped value as

    clipped_val = target_val + (val - target_val).clip(-c, c)

so ``clipped_val - target_val == clip(val - target_val, -c, c)``, whose square
is always <= ``(val - target_val)**2``. The subsequent
``jnp.maximum(unclipped_val_loss, clipped_val_loss)`` therefore always returns
the unclipped loss, no matter what ``c`` is.

Standard PPO clips around the value prediction *from the rollout* (``v_old``),
which is what actually bounds how far the critic may move in a single update:

    v_clipped = v_old + clip(val - v_old, -c, c)
    val_loss  = max((val - target)^2, (v_clipped - target)^2)

``v_old`` is available as ``buffer.val``. This subclass overrides ``ppo_loss``
with that formulation and leaves jax_pbt untouched. Everything else in the loss
is copied verbatim from ``PPOTrainer.ppo_loss``.

``val_clip_fraction`` is logged so the clip threshold is self-diagnosing: near
0 means the clip never binds (too loose), near 1 means it binds always (too
tight and the critic is frozen).
"""

import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode

from jax_pbt.policy.actor_critic import ActorCriticPPOTrainer
from jax_pbt.trainer.ppo import PPOTransition


class ValueClippedActorCriticPPOTrainer(ActorCriticPPOTrainer):
    def ppo_loss(
        self,
        model_state: PyTreeNode,
        buffer: PPOTransition,
        advantage: jax.Array,
        target_val: jax.Array,
        aux_data: dict,
    ):
        log_p, entropy, val, info = self.evaluate_action(
            model_state, buffer.obs, buffer.action, aux_data
        )

        ratio = jnp.exp(log_p - buffer.log_p)
        unclipped_pi_obj = ratio * advantage
        clipped_pi_obj = jnp.clip(
            ratio,
            1 - self.ppo_config["ratio_clip"],
            1 + self.ppo_config["ratio_clip"],
        ) * advantage

        pi_loss = -jnp.minimum(unclipped_pi_obj, clipped_pi_obj)
        if "mask" in aux_data:
            pi_loss = (pi_loss * aux_data["mask"]).mean()
            entropy = (entropy * aux_data["mask"]).mean()
            info.update({"effective_mask_fraction": aux_data["mask"].mean()})
        else:
            pi_loss = pi_loss.mean()
            entropy = entropy.mean()
        info.update({
            "pi_loss": pi_loss,
            "entropy": entropy,
            "log_p": log_p.mean(),
            "ratio_clip_fraction":
                (jnp.abs(ratio - 1) > self.ppo_config["ratio_clip"]).mean(),
        })

        unclipped_val_loss = 0.5 * jnp.square(val - target_val)
        clip = self.ppo_config["value_clip"]
        if clip is not None:
            v_old = buffer.val                                   # rollout-time prediction
            v_clipped = v_old + (val - v_old).clip(-clip, clip)
            clipped_val_loss = 0.5 * jnp.square(v_clipped - target_val)
            val_loss = jnp.maximum(unclipped_val_loss, clipped_val_loss)
            info["val_clip_fraction"] = (jnp.abs(val - v_old) > clip).mean()
            info["val_delta_abs"] = jnp.abs(val - v_old).mean()
        else:
            val_loss = unclipped_val_loss

        if "mask" in aux_data:
            val_loss = (val_loss * aux_data["mask"]).mean()
        else:
            val_loss = val_loss.mean()
        info["val_loss"] = val_loss

        loss = (
            pi_loss
            - self.ppo_config["entropy_coef"] * entropy
            + self.ppo_config["val_loss_coef"] * val_loss
        )
        info["loss"] = loss
        return loss, info
