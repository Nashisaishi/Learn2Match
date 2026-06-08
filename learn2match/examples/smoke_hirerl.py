"""HireRL smoke tests.

Runs the seven test cases described in the spec:

    1) reset shapes
    2) Nw != Nf step robustness
    3) action mask correctness across phases (incl. exclusive_role)
    4) interview state updates
    5) match-form + retain produces expected reward and tenure
    6) dissolution clears matched and current_tenure
    7) one PPO update pass

Run with: python smoke_hirerl.py
"""

import os
import sys
import traceback
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))   # repo root for jax_pbt
sys.path.insert(0, os.path.join(_HERE, ".."))           # learn2match for hirerl

from jax_pbt.env.batched_env import RoleAssignmentBatchedEnv
from jax_pbt.env.spaces import Action
from jax_pbt.policy.actor_critic import (
    ActorCriticPPOAgent as PPOAgent,
    ActorCriticPPOTrainer as PPOTrainer,
)
from jax_pbt.trainer.ppo import PPOTransition
from jax_pbt.utils import pytree_repeat_stack, rng_batch_split

from hirerl import HireRLConfig, HireRLEnv, MaskedSharedActorCriticModel
from hirerl.constants import (
    INTERVIEW_PROPOSE,
    INTERVIEW_RESPOND,
    MATCH_PROPOSE,
    MATCH_RESPOND,
    RETENTION,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_env(Nw=2, Nf=2, d=1, horizon=10, **kw) -> HireRLEnv:
    return HireRLEnv(HireRLConfig(Nw=Nw, Nf=Nf, d=d, horizon=horizon, **kw))


def step_env(env, state, w_choice, f_choice, rng, const):
    action = [
        Action({"choice": jnp.asarray(w_choice, dtype=jnp.int32)}),
        Action({"choice": jnp.asarray(f_choice, dtype=jnp.int32)}),
    ]
    return env.step(rng, const, state, action)


def assert_eq(name, got, expected):
    got_v = np.asarray(got)
    exp_v = np.asarray(expected)
    if got_v.shape != exp_v.shape or not np.allclose(got_v, exp_v):
        raise AssertionError(f"{name}: got {got_v}, expected {exp_v}")


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_reset_shapes():
    env = make_env(Nw=2, Nf=3, d=1)
    rng = jax.random.key(0)
    state, obs_lst = env.reset(rng, env.default_const)
    assert state.x.shape == (2, 1), state.x.shape
    assert state.y.shape == (3, 1), state.y.shape
    assert state.hat_x.shape == (2, 3, 1)
    assert state.hat_y.shape == (2, 3, 1)
    assert bool(jnp.all(~state.interviewed))
    assert bool(jnp.all(~state.matched))
    assert int(state.phase) == INTERVIEW_PROPOSE
    worker_obs, firm_obs = obs_lst
    # Each worker obs leaf has leading dim Nw=2; each firm obs leaf has Nf=3.
    assert worker_obs["phase"].shape[0] == 2
    assert firm_obs["phase"].shape[0] == 3
    print("[OK] reset shapes")


def test_nw_ne_nf_random_steps():
    env = make_env(Nw=2, Nf=3, d=1, horizon=10)
    const = env.default_const
    rng = jax.random.key(1)
    rng, rng_reset = jax.random.split(rng)
    state, obs_lst = env.reset(rng_reset, const)
    for k in range(20):
        w_mask = obs_lst[0]["action_mask"]   # (Nw, 4)
        f_mask = obs_lst[1]["action_mask"]   # (Nf, 3)
        rng, k1, k2, k3 = jax.random.split(rng, 4)
        # Sample uniformly from valid actions
        w_logits = jnp.where(w_mask, 0.0, -1e9)
        f_logits = jnp.where(f_mask, 0.0, -1e9)
        w_choice = jax.random.categorical(k1, w_logits, axis=-1)
        f_choice = jax.random.categorical(k2, f_logits, axis=-1)
        state, obs_lst, rew, done, info = step_env(env, state, w_choice, f_choice, k3, const)
    print("[OK] Nw != Nf random steps")


def test_action_masks():
    env = make_env(Nw=2, Nf=2, d=1, horizon=10)
    const = env.default_const
    rng = jax.random.key(2)
    state, obs_lst = env.reset(rng, const)

    # In INTERVIEW_PROPOSE, force worker 0 propose firm 0; firm 0 propose worker 1; others NOOP.
    state_p1, obs1, _, _, _ = step_env(
        env, state,
        w_choice=jnp.array([1, 0], dtype=jnp.int32),
        f_choice=jnp.array([2, 0], dtype=jnp.int32),
        rng=rng, const=const,
    )
    assert int(state_p1.phase) == INTERVIEW_RESPOND

    # exclusive_role: worker 0 (proposed) should be unable to accept; mask only allows NOOP for them.
    w_mask_resp = obs1[0]["action_mask"]   # (Nw, num_worker_choices)
    assert bool(w_mask_resp[0, 0]) and not bool(w_mask_resp[0, 1:].any()), w_mask_resp
    # worker 1 did not propose; firm 0 proposed to worker 1, so worker 1 may accept firm 0.
    assert bool(w_mask_resp[1, 1])

    # Firm 0 proposed; should also be NOOP-only in INTERVIEW_RESPOND.
    f_mask_resp = obs1[1]["action_mask"]
    assert bool(f_mask_resp[0, 0]) and not bool(f_mask_resp[0, 1:].any())
    # firm 1 did not propose and worker 0 proposed to firm 0 not firm 1, so firm 1 has no incoming proposal -> NOOP only.
    assert bool(f_mask_resp[1, 0]) and not bool(f_mask_resp[1, 1:].any())

    # Walk to MATCH_PROPOSE / MATCH_RESPOND with no interviews having occurred (everyone NOOPs).
    rng, sub = jax.random.split(rng)
    state_a, _ = env.reset(sub, const)
    state_b, obs_b, _, _, _ = step_env(
        env, state_a, w_choice=jnp.zeros(2, jnp.int32), f_choice=jnp.zeros(2, jnp.int32), rng=sub, const=const
    )  # INTERVIEW_RESPOND
    state_c, obs_c, _, _, _ = step_env(
        env, state_b, w_choice=jnp.zeros(2, jnp.int32), f_choice=jnp.zeros(2, jnp.int32), rng=sub, const=const
    )  # MATCH_PROPOSE
    assert int(state_c.phase) == MATCH_PROPOSE
    w_mask_mp = obs_c[0]["action_mask"]
    # No interviews -> only NOOP allowed for workers.
    assert bool((~w_mask_mp[:, 1:]).all())
    # Firms only NOOP in MATCH_PROPOSE.
    f_mask_mp = obs_c[1]["action_mask"]
    assert bool((~f_mask_mp[:, 1:]).all())

    # Step to MATCH_RESPOND -- workers have only NOOP.
    state_d, obs_d, _, _, _ = step_env(
        env, state_c, w_choice=jnp.zeros(2, jnp.int32), f_choice=jnp.zeros(2, jnp.int32), rng=sub, const=const
    )
    assert int(state_d.phase) == MATCH_RESPOND
    w_mask_mr = obs_d[0]["action_mask"]
    assert bool(w_mask_mr[:, 0].all()) and bool((~w_mask_mr[:, 1:]).all())

    # Step to RETENTION; nobody is matched/tentative -> only NOOP/dissolve.
    state_e, obs_e, _, _, _ = step_env(
        env, state_d, w_choice=jnp.zeros(2, jnp.int32), f_choice=jnp.zeros(2, jnp.int32), rng=sub, const=const
    )
    assert int(state_e.phase) == RETENTION
    w_mask_re = obs_e[0]["action_mask"]
    assert bool(w_mask_re[:, 0].all()) and bool((~w_mask_re[:, 1:]).all())
    print("[OK] action masks (incl. exclusive_role propose-then-respond restriction)")


def test_interview_updates_belief():
    env = make_env(Nw=2, Nf=2, d=2, sigma_interview=0.0)  # zero noise so updates are deterministic
    const = env.default_const
    rng = jax.random.key(3)
    state, _ = env.reset(rng, const)

    # Propose: worker 0 -> firm 0; firm 1 -> worker 1 (so worker 0 will not accept firm 1).
    w_choice = jnp.array([1, 0], dtype=jnp.int32)
    f_choice = jnp.array([0, 0], dtype=jnp.int32)   # only worker 0 proposes; firm 0 will accept.
    state_p1, _, _, _, _ = step_env(env, state, w_choice, f_choice, rng, const)

    # Respond: firm 0 accepts worker 0.
    w_choice2 = jnp.array([0, 0], dtype=jnp.int32)   # workers don't accept (worker 0 cannot in exclusive)
    f_choice2 = jnp.array([1, 0], dtype=jnp.int32)   # firm 0 accepts worker 0 (choice = i+1 = 1)
    state_p2, _, _, _, _ = step_env(env, state_p1, w_choice2, f_choice2, rng, const)
    assert bool(state_p2.interviewed[0, 0])
    # With zero noise, hat_x[0,0] == x[0] and hat_y[0,0] == y[0]
    assert jnp.allclose(state_p2.hat_x[0, 0], state.x[0])
    assert jnp.allclose(state_p2.hat_y[0, 0], state.y[0])
    print("[OK] interview belief update")


def test_match_retention_reward():
    env = make_env(Nw=2, Nf=2, d=2, sigma_interview=0.0, sigma_match=0.0)
    const = env.default_const
    rng = jax.random.key(4)
    state, _ = env.reset(rng, const)

    # PHASE 1: worker 0 proposes firm 0
    s1, _, _, _, _ = step_env(env, state,
        w_choice=jnp.array([1, 0], dtype=jnp.int32),
        f_choice=jnp.array([0, 0], dtype=jnp.int32),
        rng=rng, const=const)
    # PHASE 2: firm 0 accepts worker 0
    s2, _, _, _, _ = step_env(env, s1,
        w_choice=jnp.array([0, 0], dtype=jnp.int32),
        f_choice=jnp.array([1, 0], dtype=jnp.int32),
        rng=rng, const=const)
    # PHASE 3: worker 0 match-proposes firm 0
    s3, _, _, _, _ = step_env(env, s2,
        w_choice=jnp.array([1, 0], dtype=jnp.int32),
        f_choice=jnp.array([0, 0], dtype=jnp.int32),
        rng=rng, const=const)
    # PHASE 4: firm 0 match-accepts worker 0
    s4, _, _, _, _ = step_env(env, s3,
        w_choice=jnp.array([0, 0], dtype=jnp.int32),
        f_choice=jnp.array([1, 0], dtype=jnp.int32),
        rng=rng, const=const)
    # PHASE 5: both retain
    s5, obs5, rew_lst, _, _ = step_env(env, s4,
        w_choice=jnp.array([1, 0], dtype=jnp.int32),
        f_choice=jnp.array([1, 0], dtype=jnp.int32),
        rng=rng, const=const)
    assert bool(s5.matched[0, 0]), s5.matched
    assert int(s5.cumulative_tenure[0, 0]) == 1
    assert int(s5.current_tenure[0, 0]) == 1

    # Predicted reward: dot(x[0], hat_y[0,0]) where hat_y[0,0] = sigmoid(lambda*1) * y[0]
    sig = float(jax.nn.sigmoid(env.config.lambda_reveal * 1.0))
    expected_w = float(jnp.dot(state.x[0], sig * state.y[0]))
    expected_f = float(jnp.dot(sig * state.x[0], state.y[0]))
    rew_w, rew_f = rew_lst
    assert np.isclose(float(rew_w[0]), expected_w, atol=1e-5), (float(rew_w[0]), expected_w)
    assert np.isclose(float(rew_f[0]), expected_f, atol=1e-5), (float(rew_f[0]), expected_f)
    # Other phases produced zero reward in the trajectory (reward only on RETENTION phase).
    print(f"[OK] match + retention reward (rew_w={float(rew_w[0]):.3f}, rew_f={float(rew_f[0]):.3f})")


def test_dissolution_clears():
    env = make_env(Nw=2, Nf=2, d=2, sigma_interview=0.0, sigma_match=0.0)
    const = env.default_const
    rng = jax.random.key(5)
    state, _ = env.reset(rng, const)

    # Form a match (steps 1-5 with both retain to get matched[0,0]=True, cum_ten=1).
    s = state
    for w, f in [
        ([1, 0], [0, 0]),   # propose
        ([0, 0], [1, 0]),   # respond
        ([1, 0], [0, 0]),   # match propose
        ([0, 0], [1, 0]),   # match respond
        ([1, 0], [1, 0]),   # retention -> matched
    ]:
        s, _, _, _, _ = step_env(env, s,
            w_choice=jnp.array(w, dtype=jnp.int32),
            f_choice=jnp.array(f, dtype=jnp.int32),
            rng=rng, const=const)
    assert bool(s.matched[0, 0])
    cum_before = int(s.cumulative_tenure[0, 0])

    # Next market period: workers and firms NOOP through interview / match phases (matched cannot interview).
    for w, f in [
        ([0, 0], [0, 0]),   # interview propose
        ([0, 0], [0, 0]),   # interview respond
        ([0, 0], [0, 0]),   # match propose
        ([0, 0], [0, 0]),   # match respond
    ]:
        s, _, _, _, _ = step_env(env, s,
            w_choice=jnp.array(w, dtype=jnp.int32),
            f_choice=jnp.array(f, dtype=jnp.int32),
            rng=rng, const=const)

    # RETENTION: worker 0 chooses 0 (dissolve)
    s, _, rew_lst, _, _ = step_env(env, s,
        w_choice=jnp.array([0, 0], dtype=jnp.int32),
        f_choice=jnp.array([1, 0], dtype=jnp.int32),
        rng=rng, const=const)
    assert not bool(s.matched[0, 0])
    assert int(s.current_tenure[0, 0]) == 0
    assert int(s.cumulative_tenure[0, 0]) == cum_before
    rew_w, rew_f = rew_lst
    assert float(rew_w[0]) == 0.0
    assert float(rew_f[0]) == 0.0
    print("[OK] dissolution clears matched + current_tenure but preserves cumulative_tenure")


def test_ppo_one_update():
    env = HireRLEnv(HireRLConfig(Nw=2, Nf=3, d=1, horizon=4))
    num_envs = 4
    bench = RoleAssignmentBatchedEnv(env, num_envs=num_envs, assignments=[[0], [1]])
    worker_bs, firm_bs = bench.get_agent_batch_shape()
    assert worker_bs == (num_envs * env.Nw,), worker_bs
    assert firm_bs == (num_envs * env.Nf,), firm_bs

    worker_obs_space, firm_obs_space = bench.get_observation_space()
    worker_act_space, firm_act_space = bench.get_action_space()

    worker_model = MaskedSharedActorCriticModel.build(worker_obs_space, worker_act_space, hidden_size=32)
    firm_model = MaskedSharedActorCriticModel.build(firm_obs_space, firm_act_space, hidden_size=32)

    common = dict(
        feature_shared=True, lr=3e-4, grad_clip_norm=1.0,
        val_loss_coef=0.5, entropy_coef=0.01,
        gamma=0.99, gae_lam=0.95, ppo_epochs=1,
        ratio_clip=0.2, chunk_length=4, num_minibatches=1,
    )
    worker_trainer = PPOTrainer(actor_critic_fn=worker_model, **common)
    firm_trainer = PPOTrainer(actor_critic_fn=firm_model, **common)

    ppo_agent_lst = [PPOAgent(worker_model), PPOAgent(firm_model)]
    ppo_trainer_lst = [worker_trainer, firm_trainer]

    rng = jax.random.key(7)
    rng, rng_trainers = rng_batch_split(rng, len(ppo_trainer_lst))
    trainer_state_lst = [t.init_trainer_state(r) for r, t in zip(list(rng_trainers), ppo_trainer_lst)]
    agent_state_lst = [a.init_agent_state(bs) for a, bs in zip(ppo_agent_lst, bench.get_agent_batch_shape())]
    rng, rng_reset = jax.random.split(rng)
    env_const = pytree_repeat_stack(env.default_const, (num_envs,))
    env_state, obs_lst = bench.reset(rng_reset, env_const)

    rollout_len = 8
    from jax_pbt.controller.ippo_controller import IPPOController

    model_state_lst = [t.model_state_from_trainer_state(s) for t, s in zip(ppo_trainer_lst, trainer_state_lst)]
    data_lst_hist: list = []
    for _ in range(rollout_len):
        rng, agent_state_lst, env_state, obs_lst, data_lst, info = IPPOController.ppo_rollout_step(
            bench, ppo_agent_lst, rng, model_state_lst, agent_state_lst, env_const, env_state, obs_lst,
        )
        data_lst_hist.append(data_lst)

    # Stack across time using tree_map so Observation/Action PyTrees stack correctly.
    stacked_lst = [
        jax.tree_util.tree_map(
            lambda *xs: jnp.stack(xs, axis=0),
            *[d[i] for d in data_lst_hist],
        )
        for i in range(len(ppo_agent_lst))
    ]
    buffer_lst = [
        PPOTransition(d["obs"], d["action"], d["reward"], d["done"], d["log_p"], d["val"])
        for d in stacked_lst
    ]
    aux_data_lst = [{"agent_state": d["agent_state"]} for d in stacked_lst]

    last_val_lst = []
    rng, rng_val_lst = rng_batch_split(rng, len(ppo_agent_lst))
    for r, agent_fn, ms, ag, lo in zip(list(rng_val_lst), ppo_agent_lst, model_state_lst, agent_state_lst, obs_lst):
        _, _, extra_data = agent_fn.step(r, ms, ag, lo)
        last_val_lst.append(extra_data["val"])

    rng, rng_train = jax.random.split(rng)
    new_trainer_state_lst, optim_info_lst = IPPOController.ppo_update(
        ppo_trainer_lst, rng_train, trainer_state_lst, buffer_lst, last_val_lst, aux_data_lst,
    )
    assert "loss" in optim_info_lst[0]
    assert "loss" in optim_info_lst[1]
    print(
        f"[OK] PPO update (worker loss={float(optim_info_lst[0]['loss']):.3f}, "
        f"firm loss={float(optim_info_lst[1]['loss']):.3f})"
    )


# --------------------------------------------------------------------------- #
def main() -> None:
    tests: list[Callable[[], None]] = [
        test_reset_shapes,
        test_nw_ne_nf_random_steps,
        test_action_masks,
        test_interview_updates_belief,
        test_match_retention_reward,
        test_dissolution_clears,
        test_ppo_one_update,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
        except Exception:
            failed += 1
            print(f"[FAIL] {fn.__name__}")
            traceback.print_exc()
    if failed == 0:
        print(f"\n{len(tests)} smoke tests passed.")
    else:
        print(f"\n{failed}/{len(tests)} smoke tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
