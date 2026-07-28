"""MAPPO rollout / update logic for HireRL with a shared centralized critic.

Differences from the IPPO path (jax_pbt IPPOController + ActorCriticPPOTrainer):

* Values come from ONE PairGridCritic forward per env-step covering all
  Nw + Nf agents, not from each role's local value head. The actors' built-in
  value heads receive no gradient (their value output is ignored end to end).
* The PPO minibatch sampling unit is an (env, time-chunk) block, not a
  flattened agent row: the critic input is the whole env's pair grid, so all
  agents of an env must travel together. ``jax_pbt.utils.split_into_minibatch``
  is reused verbatim -- the batch axis fed to it is simply ``num_envs`` with
  the per-role agent axes folded into the attribute dims. The actor losses
  are agnostic to the grouping (their per-agent terms are flattened back
  inside the loss).
* Three optimizers: worker actor, firm actor, shared critic. The critic loss
  is the value MSE pooled over both roles' agents (with the CORRECTED value
  clipping around the rollout-time prediction -- see
  examples/value_clipped_trainer.py for why jax_pbt's built-in clip is a
  no-op).
* GAE is unchanged and per-agent (rewards are individual); it reuses
  ``PPOTransition.compute_advantage``, which is elementwise over any trailing
  batch shape, here (num_envs, N_role).

Data layout contract
--------------------
``rollout`` (the scan-stacked trajectory) is a dict:

    rollout['w'|'f'] = {
        'obs':    Observation, leaves (T, E, N, field),
        'action': Action, leaves (T, E, N),
        'reward' | 'done' | 'log_p' | 'val': (T, E, N),
        'h0':     actor rnn-state pytree, leaves (T, E, N, ...)   # pre-step
    }
    rollout['x'] = privileged extras with leaves (T, E, ...) or {} (strict).
"""

import os
import sys
from typing import Callable

import jax
import jax.numpy as jnp
from flax.struct import PyTreeNode

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))   # repo root (for jax_pbt)
sys.path.insert(0, os.path.join(_HERE, ".."))          # learn2match (for hirerl)

from jax_pbt.trainer.ppo import PPOTransition
from jax_pbt.utils import global_norm, rng_batch_split, split_into_minibatch

from central_critic import (
    CriticInputSpec,
    PairGridCritic,
    build_critic_inputs,
    privileged_extras_from_state,
)


def _group_leading(tree: PyTreeNode, num_envs: int, n_agents: int) -> PyTreeNode:
    """(E*N, ...) leaves -> (E, N, ...): the rollout-time (single step) view."""
    return jax.tree_util.tree_map(
        lambda x: x.reshape(num_envs, n_agents, *x.shape[1:]), tree
    )


def _flatten_agents(tree: PyTreeNode) -> PyTreeNode:
    """(L, R, N, ...) leaves -> (L, R*N, ...): per-agent view inside a loss."""
    return jax.tree_util.tree_map(
        lambda x: x.reshape(x.shape[0], -1, *x.shape[3:]), tree
    )


def make_compute_central_values(
    critic: PairGridCritic, spec: CriticInputSpec, num_envs: int
) -> Callable:
    """One critic forward per env covering every agent.

    Returns ``compute(critic_params, obs_w, obs_f, env_state) ->
    (val_w (E*Nw,), val_f (E*Nf,), extras)`` where obs_w / obs_f are the flat
    role observations exactly as the batched env emits them.
    """

    def compute(critic_params, obs_w, obs_f, env_state):
        obs_w_g = _group_leading(obs_w, num_envs, spec.Nw)
        obs_f_g = _group_leading(obs_f, num_envs, spec.Nf)
        extras = (
            privileged_extras_from_state(env_state, spec.horizon)
            if spec.privileged else {}
        )
        inputs = build_critic_inputs(
            spec, obs_w_g, obs_f_g, extras if spec.privileged else None
        )
        v_w, v_f = critic.apply(critic_params, **inputs)
        return v_w.reshape(-1), v_f.reshape(-1), extras

    return compute


def make_mappo_rollout_step(
    env_fn, worker_agent, firm_agent,
    critic: PairGridCritic, spec: CriticInputSpec, num_envs: int,
) -> Callable:
    """One env interaction step: both actors act on their own observations
    (unchanged from IPPO); the shared critic fills in every agent's value."""

    compute_values = make_compute_central_values(critic, spec, num_envs)

    def rollout_step(rng, w_ms, f_ms, critic_params, agent_state_lst,
                     env_const, env_state, obs_lst):
        h_w, h_f = agent_state_lst
        rng, rng_w, rng_f = jax.random.split(rng, 3)
        next_h_w, act_w, ex_w = worker_agent.step(rng_w, w_ms, h_w, obs_lst[0])
        next_h_f, act_f, ex_f = firm_agent.step(rng_f, f_ms, h_f, obs_lst[1])

        val_w, val_f, extras = compute_values(
            critic_params, obs_lst[0], obs_lst[1], env_state
        )

        rng, rng_env = jax.random.split(rng)
        next_env_state, next_obs_lst, reward_lst, done_lst, info = env_fn.step(
            rng_env, env_const, env_state, [act_w, act_f]
        )
        next_h_w = worker_agent.reset_agent_state(next_h_w, done_lst[0])
        next_h_f = firm_agent.reset_agent_state(next_h_f, done_lst[1])

        data = {
            "w": {
                "obs": obs_lst[0], "action": act_w, "log_p": ex_w["log_p"],
                "val": val_w, "h0": h_w,
                "reward": reward_lst[0], "done": done_lst[0],
            },
            "f": {
                "obs": obs_lst[1], "action": act_f, "log_p": ex_f["log_p"],
                "val": val_f, "h0": h_f,
                "reward": reward_lst[1], "done": done_lst[1],
            },
            "x": extras,
        }
        return rng, [next_h_w, next_h_f], next_env_state, next_obs_lst, data, info

    return rollout_step


def rollout_to_env_major(data_seq: dict, num_envs: int, Nw: int, Nf: int) -> dict:
    """Regroup the scan-stacked trajectory from flat (T, E*N, ...) buffers to
    env-major (T, E, N, ...), the layout the update consumes. Free reshapes."""
    def g(tree, n):
        return jax.tree_util.tree_map(
            lambda x: x.reshape(x.shape[0], num_envs, n, *x.shape[2:]), tree
        )
    return {
        "w": g(data_seq["w"], Nw),
        "f": g(data_seq["f"], Nf),
        "x": data_seq["x"],   # already (T, E, ...)
    }


def make_mappo_learn(
    worker_model, firm_model,
    critic: PairGridCritic, spec: CriticInputSpec,
    *,
    gamma: float,
    gae_lam: float,
    ppo_epochs: int,
    chunk_length: int,
    num_minibatches: int | None,
    minibatch_num_chunks: int | None,
    ratio_clip_w: float,
    ratio_clip_f: float,
    entropy_coef: float,
    value_clip: float | None,
    val_loss_coef: float,
    use_advantage_normalization: bool = True,
) -> Callable:
    """Build ``learn(rng, w_ts, f_ts, c_ts, rollout, last_val_w, last_val_f)``.

    Returns updated train states plus a log dict with 'worker' / 'firm' /
    'critic' sub-dicts (leaves shaped (ppo_epochs, num_minibatches)).
    """

    def actor_loss_fn(params, mb_role, model, ratio_clip):
        obs = _flatten_agents(mb_role["obs"])                     # (L, R*N, F)
        action = _flatten_agents(mb_role["action"])
        old_log_p = mb_role["log_p"].reshape(mb_role["log_p"].shape[0], -1)
        adv = mb_role["adv"].reshape(mb_role["adv"].shape[0], -1)
        # ``h0`` is already the agent_state pytree ({'rnn_state': ...}); take
        # the chunk's first timestep and fold the agent axis into the batch,
        # mirroring PPOTrainer.evaluate_action's ``agent_state['rnn_state'][0]``.
        h0 = jax.tree_util.tree_map(
            lambda x: x[0].reshape(-1, *x.shape[3:]), mb_role["h0"]
        )

        model_state = {"actor_critic_params": params}
        _, pi, _ = model.forward(jax.random.key(0), model_state, h0, obs)
        log_p = pi.log_prob(action)

        log_ratio = log_p - old_log_p
        ratio = jnp.exp(log_ratio)
        unclipped = ratio * adv
        clipped = jnp.clip(ratio, 1 - ratio_clip, 1 + ratio_clip) * adv
        pi_loss = -jnp.minimum(unclipped, clipped).mean()
        entropy = pi.entropy().mean()
        loss = pi_loss - entropy_coef * entropy
        info = {
            "pi_loss": pi_loss,
            "entropy": entropy,
            "log_p": log_p.mean(),
            "approx_kl": ((ratio - 1) - log_ratio).mean(),
            "ratio_clip_fraction": (jnp.abs(ratio - 1) > ratio_clip).mean(),
        }
        return loss, info

    def critic_loss_fn(critic_params, inputs, tgt_w, tgt_f, old_w, old_f):
        v_w, v_f = critic.apply(critic_params, **inputs)          # (L,R,Nw)/(L,R,Nf)

        def role_sq_err(v, tgt, old):
            unclipped = 0.5 * jnp.square(v - tgt)
            if value_clip is None:
                return unclipped, jnp.zeros(())
            v_clip = old + (v - old).clip(-value_clip, value_clip)
            clipped = 0.5 * jnp.square(v_clip - tgt)
            frac = (jnp.abs(v - old) > value_clip).mean()
            return jnp.maximum(unclipped, clipped), frac

        err_w, frac_w = role_sq_err(v_w, tgt_w, old_w)
        err_f, frac_f = role_sq_err(v_f, tgt_f, old_f)
        # Every agent weighs equally, whichever side it is on.
        val_loss = (err_w.sum() + err_f.sum()) / (err_w.size + err_f.size)
        loss = val_loss_coef * val_loss
        info = {"val_loss": val_loss}
        if value_clip is not None:
            info["val_clip_fraction"] = (
                frac_w * err_w.size + frac_f * err_f.size
            ) / (err_w.size + err_f.size)
        return loss, info

    def learn(rng, w_ts, f_ts, c_ts, rollout, last_val_w, last_val_f):
        # --- GAE (per agent; identical math to the IPPO path) --------------
        def gae_for(role, last_val):
            # Bootstrap values arrive flat (E*N,); match the env-major
            # (E, N) trailing batch of the buffers. Shapes are static.
            last_val = last_val.reshape(rollout[role]["reward"].shape[1:])
            buf = PPOTransition(
                obs=rollout[role]["obs"], action=rollout[role]["action"],
                reward=rollout[role]["reward"], done=rollout[role]["done"],
                log_p=rollout[role]["log_p"], val=rollout[role]["val"],
            )
            adv = buf.compute_advantage(last_val, gamma=gamma, gae_lam=gae_lam)
            tgt = adv + rollout[role]["val"]
            ev = 1.0 - jnp.var(tgt - rollout[role]["val"]) / (jnp.var(tgt) + 1e-8)
            if use_advantage_normalization:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            return adv, tgt, ev

        adv_w, tgt_w, ev_w = gae_for("w", last_val_w)
        adv_f, tgt_f, ev_f = gae_for("f", last_val_f)

        data = {
            "w": {**rollout["w"], "adv": adv_w, "tgt": tgt_w},
            "f": {**rollout["f"], "adv": adv_f, "tgt": tgt_f},
            "x": rollout["x"],
        }

        # --- epochs over env-block minibatches ------------------------------
        def epoch_step(train_states, rng_epoch):
            minibatches = split_into_minibatch(
                rng=rng_epoch,
                data=data,
                chunk_length=chunk_length,
                num_minibatches=num_minibatches,
                minibatch_num_chunks=minibatch_num_chunks,
            )

            def minibatch_step(train_states, mb):
                w_ts, f_ts, c_ts = train_states

                (w_loss, w_info), w_grads = jax.value_and_grad(
                    actor_loss_fn, has_aux=True
                )(w_ts.params, mb["w"], worker_model, ratio_clip_w)
                (f_loss, f_info), f_grads = jax.value_and_grad(
                    actor_loss_fn, has_aux=True
                )(f_ts.params, mb["f"], firm_model, ratio_clip_f)

                inputs = build_critic_inputs(
                    spec, mb["w"]["obs"], mb["f"]["obs"],
                    mb["x"] if spec.privileged else None,
                )
                (c_loss, c_info), c_grads = jax.value_and_grad(
                    critic_loss_fn, has_aux=True
                )(
                    c_ts.params, inputs,
                    mb["w"]["tgt"], mb["f"]["tgt"],
                    mb["w"]["val"], mb["f"]["val"],
                )

                w_info["grad_norm"] = global_norm(w_grads)
                f_info["grad_norm"] = global_norm(f_grads)
                c_info["grad_norm"] = global_norm(c_grads)
                w_info["loss"] = w_loss
                f_info["loss"] = f_loss
                c_info["loss"] = c_loss

                new_states = (
                    w_ts.apply_gradients(grads=w_grads),
                    f_ts.apply_gradients(grads=f_grads),
                    c_ts.apply_gradients(grads=c_grads),
                )
                return new_states, {"worker": w_info, "firm": f_info, "critic": c_info}

            return jax.lax.scan(minibatch_step, train_states, minibatches)

        rng, rng_epochs = rng_batch_split(rng, ppo_epochs)
        (w_ts, f_ts, c_ts), logs = jax.lax.scan(
            epoch_step, (w_ts, f_ts, c_ts), rng_epochs, ppo_epochs
        )
        logs["critic"]["explained_variance_w"] = jnp.broadcast_to(
            ev_w, logs["critic"]["val_loss"].shape
        )
        logs["critic"]["explained_variance_f"] = jnp.broadcast_to(
            ev_f, logs["critic"]["val_loss"].shape
        )
        return (w_ts, f_ts, c_ts), logs

    return learn
