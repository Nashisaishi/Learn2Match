"""Smoke + correctness tests for the HireRL MAPPO shared-critic path.

Run:  python learn2match/mappo-hirerl/test_mappo_smoke.py

Covers
  1. build_critic_inputs reconstructs the pair grid EXACTLY from the two role
     observations -- checked against the env state's ground truth (hat_x,
     hat_y, matched, interviewed, tenure, proposals). This is the load-bearing
     claim of the design: the centralized critic needs no extra rollout
     storage because its input is recoverable from what IPPO already stores.
  2. One critic forward emits all Nw + Nf values, and the values are
     permutation-equivariant (relabelling workers permutes V_w the same way)
     -- i.e. the pooled architecture really is size/order agnostic.
  3. The critic parameter count is independent of Nw / Nf.
  4. End-to-end training cycles run and actually move all three train states,
     for privileged/strict critics and with the actor RNN on and off.
"""

import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", "examples"))

from hirerl.config import HireRLConfig
from hirerl.env import HireRLEnv

from central_critic import (
    CriticInputSpec,
    PairGridCritic,
    build_critic_inputs,
    privileged_extras_from_state,
)
from train_hirerl_mappo import HireRLMAPPOController

PASS, FAIL = "  PASS", "  FAIL"
_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{PASS if cond else FAIL}  {name}{'' if cond else '  <- ' + detail}")
    if not cond:
        _failures.append(name)


def _make_env(Nw=3, Nf=4, d=2, horizon=8):
    cfg = HireRLConfig(Nw=Nw, Nf=Nf, d=d, horizon=horizon,
                       sigma_interview=0.3, sigma_match=0.1,
                       compute_settled_metrics=False)
    return HireRLEnv(cfg), cfg


def _rolled_state(env, cfg, rng, num_steps=7):
    """Step a single env a few times so beliefs / matches / proposals are
    non-trivial, then return (state, worker_obs, firm_obs) with a leading
    env axis of size 1."""
    const = env.default_const
    rng, k = jax.random.split(rng)
    state, obs_lst = env.reset(k, const)
    for _ in range(num_steps):
        rng, ka, ks = jax.random.split(rng, 3)
        actions = []
        for side, obs in enumerate(obs_lst):
            ka, ksub = jax.random.split(ka)
            mask = obs["action_mask"]
            logits = jnp.where(mask > 0, 0.0, -1e9)
            choice = jax.random.categorical(ksub, logits, axis=-1)
            from jax_pbt.env.spaces import Action
            actions.append(Action({"choice": choice.astype(jnp.int32)}))
        state, obs_lst, _, _, _ = env.step(ks, const, state, actions)
    add_axis = lambda t: jax.tree_util.tree_map(lambda x: x[None], t)
    return add_axis(state), add_axis(obs_lst[0]), add_axis(obs_lst[1])


def test_reconstruction() -> None:
    print("\n[1] pair-grid reconstruction from role observations")
    Nw, Nf, d, horizon = 3, 4, 2, 8
    env, cfg = _make_env(Nw, Nf, d, horizon)
    state, wobs, fobs = _rolled_state(env, cfg, jax.random.key(0))

    worker_obs_space, _ = env.get_observation_space()
    for privileged in (True, False):
        spec = CriticInputSpec.from_spaces(
            worker_obs_space, Nw=Nw, Nf=Nf, d=d, horizon=horizon,
            privileged=privileged,
        )
        extras = privileged_extras_from_state(state, horizon) if privileged else None
        inputs = build_critic_inputs(spec, wobs, fobs, extras)
        pair = inputs["pair"]
        tag = "privileged" if privileged else "strict"

        check(f"[{tag}] pair shape (1, Nw, Nf, pair_dim)",
              pair.shape == (1, Nw, Nf, spec.pair_dim),
              f"got {pair.shape}, expected {(1, Nw, Nf, spec.pair_dim)}")

        # Ground truth straight off the env state.
        check(f"[{tag}] hat_y block == state.hat_y",
              jnp.allclose(pair[..., 0:d], state.hat_y))
        check(f"[{tag}] hat_x block == state.hat_x",
              jnp.allclose(pair[..., d:2 * d], state.hat_x))
        check(f"[{tag}] matched block == state.matched",
              jnp.allclose(pair[..., 2 * d + 0], state.matched.astype(jnp.float32)))
        check(f"[{tag}] interviewed block == state.interviewed",
              jnp.allclose(pair[..., 2 * d + 1], state.interviewed.astype(jnp.float32)))
        check(f"[{tag}] current_tenure block == state.current_tenure / horizon",
              jnp.allclose(pair[..., 2 * d + 2],
                           state.current_tenure.astype(jnp.float32) / horizon))
        check(f"[{tag}] cumulative_tenure block == state.cumulative_tenure / horizon",
              jnp.allclose(pair[..., 2 * d + 3],
                           state.cumulative_tenure.astype(jnp.float32) / horizon))
        # Firm-side field transposed back into worker-major orientation.
        check(f"[{tag}] match-proposal block == worker->firm proposals",
              jnp.allclose(pair[..., 2 * d + 7],
                           jnp.swapaxes(fobs["incoming_match_proposals"], -2, -1)))
        if privileged:
            xy = jnp.einsum("...id,...jd->...ij", state.x, state.y)
            check("[privileged] xy block == x_i . y_j",
                  jnp.allclose(pair[..., -1], xy))
            check("[privileged] global market_frac present",
                  inputs["global_ctx"].shape[-1] == spec.global_ctx_dim)

        check(f"[{tag}] worker_ctx / firm_ctx shapes",
              inputs["worker_ctx"].shape == (1, Nw, spec.worker_ctx_dim)
              and inputs["firm_ctx"].shape == (1, Nf, spec.firm_ctx_dim))
        check(f"[{tag}] no NaNs in critic inputs",
              all(bool(jnp.all(jnp.isfinite(v))) for v in inputs.values()))


def test_one_forward_all_values() -> None:
    print("\n[2] one forward -> all Nw + Nf values, permutation-equivariant")
    Nw, Nf, d, horizon = 3, 4, 2, 8
    env, cfg = _make_env(Nw, Nf, d, horizon)
    state, wobs, fobs = _rolled_state(env, cfg, jax.random.key(1))
    worker_obs_space, _ = env.get_observation_space()
    spec = CriticInputSpec.from_spaces(worker_obs_space, Nw=Nw, Nf=Nf, d=d,
                                       horizon=horizon, privileged=True)
    critic = PairGridCritic(hidden_size=16)
    params = critic.init_params(jax.random.key(0), spec)

    extras = privileged_extras_from_state(state, horizon)
    inputs = build_critic_inputs(spec, wobs, fobs, extras)
    v_w, v_f = critic.apply(params, **inputs)
    check("single forward returns V_w with shape (1, Nw)", v_w.shape == (1, Nw),
          f"got {v_w.shape}")
    check("single forward returns V_f with shape (1, Nf)", v_f.shape == (1, Nf),
          f"got {v_f.shape}")
    check("values are finite",
          bool(jnp.all(jnp.isfinite(v_w))) and bool(jnp.all(jnp.isfinite(v_f))))

    # Permute workers: V_w must permute identically, V_f must be unchanged.
    perm = jnp.array([2, 0, 1])
    p_inputs = dict(inputs)
    p_inputs["pair"] = inputs["pair"][:, perm]
    p_inputs["worker_ctx"] = inputs["worker_ctx"][:, perm]
    pv_w, pv_f = critic.apply(params, **p_inputs)
    check("worker permutation permutes V_w equivariantly",
          bool(jnp.allclose(pv_w, v_w[:, perm], atol=1e-5)),
          f"max diff {float(jnp.max(jnp.abs(pv_w - v_w[:, perm]))):.2e}")
    check("worker permutation leaves V_f invariant",
          bool(jnp.allclose(pv_f, v_f, atol=1e-5)),
          f"max diff {float(jnp.max(jnp.abs(pv_f - v_f))):.2e}")

    # Same params must run on a different market size.
    Nw2, Nf2 = 6, 5
    env2, cfg2 = _make_env(Nw2, Nf2, d, horizon)
    st2, w2, f2 = _rolled_state(env2, cfg2, jax.random.key(2))
    wspace2, _ = env2.get_observation_space()
    spec2 = CriticInputSpec.from_spaces(wspace2, Nw=Nw2, Nf=Nf2, d=d,
                                        horizon=horizon, privileged=True)
    in2 = build_critic_inputs(spec2, w2, f2, privileged_extras_from_state(st2, horizon))
    v_w2, v_f2 = critic.apply(params, **in2)
    check("same params transfer to a different market size",
          v_w2.shape == (1, Nw2) and v_f2.shape == (1, Nf2),
          f"got {v_w2.shape}, {v_f2.shape}")

    n_small = sum(x.size for x in jax.tree_util.tree_leaves(params))
    spec_big = CriticInputSpec.from_spaces(wspace2, Nw=100, Nf=100, d=d,
                                           horizon=horizon, privileged=True)
    params_big = critic.init_params(jax.random.key(0), spec_big)
    n_big = sum(x.size for x in jax.tree_util.tree_leaves(params_big))
    check("critic param count independent of Nw / Nf", n_small == n_big,
          f"{n_small} vs {n_big}")


def _train_args(**over):
    a = argparse.Namespace(
        seed=0, run_name="smoke", Nw=3, Nf=3, d=2, horizon=8,
        sigma_interview=0.3, sigma_match=0.1, lambda_reveal=1.0,
        outside_option=0.0, non_negative_features=False, hat_init_value=0.0,
        allow_on_the_job_search=False, noisy_hat_init=False, sigma_init=3.0,
        public_retention_signal=False, interview_mode="exclusive_role",
        init_from_ckpt="",
        total_env_steps=0, num_envs=4, rollout_length=8, ppo_epochs=2,
        num_minibatches=2, chunk_length=4, minibatch_size=None,
        gamma=0.99, gae_lam=0.95, ratio_clip=0.2, scale_clip_eps=False,
        value_clip=None, entropy_coef=0.01, val_loss_coef=0.5,
        grad_clip_norm=1.0, lr=3e-4, hidden_size=16, use_rnn=False,
        critic_lr=None, critic_hidden_size=16, critic_mode="privileged",
        wandb_entity=None, wandb_project=None, wandb_group=None, wandb_job_type=None,
        metric_log_every_env_steps=0, num_eval_episodes=2, num_eval_periods=None,
        eval_seed=1, ckpt_dir="", save_best_ckpt=False, save_all_checkpoints=False,
        best_metric="social_welfare", plot_dir="", plot_true_matched_welfare=False,
    )
    for k, v in over.items():
        setattr(a, k, v)
    return a


def test_training_cycle() -> None:
    print("\n[3] end-to-end training cycles")
    for critic_mode in ("privileged", "strict"):
        for use_rnn in (False, True):
            tag = f"{critic_mode}, use_rnn={use_rnn}"
            ctrl = HireRLMAPPOController(
                _train_args(critic_mode=critic_mode, use_rnn=use_rnn)
            )
            run_state = ctrl.init_run_state(jax.random.key(0))
            const_seq = ctrl.get_train_env_const_seq()

            before = run_state[1]
            for _ in range(2):
                run_state, train_info, optim_info = ctrl.training_cycle(
                    run_state, const_seq
                )
            after = run_state[1]

            check(f"[{tag}] three trainer states carried", len(after) == 3,
                  f"got {len(after)}")
            finite = all(
                bool(jnp.all(jnp.isfinite(v)))
                for oi in optim_info for v in oi.values()
            )
            check(f"[{tag}] all optim stats finite", finite)
            for i, name in enumerate(("worker actor", "firm actor", "critic")):
                moved = not all(
                    bool(jnp.allclose(a, b))
                    for a, b in zip(jax.tree_util.tree_leaves(before[i]),
                                    jax.tree_util.tree_leaves(after[i]))
                )
                check(f"[{tag}] {name} params moved", moved)
            check(f"[{tag}] returns finite",
                  all(bool(jnp.isfinite(ti["avg_return"])) for ti in train_info))
            check(f"[{tag}] critic stats logged under worker/",
                  "critic_val_loss" in optim_info[0]
                  and "critic_explained_variance_w" in optim_info[0])


def test_minibatch_alignment() -> None:
    """The critic sees whole envs: worker and firm minibatch rows must come
    from the SAME (env, chunk) blocks, or the reconstructed pair grid mixes
    beliefs across envs. Verified by planting env-identifying values in the
    rollout and checking they agree after split_into_minibatch."""
    print("\n[4] minibatch keeps worker/firm rows env-aligned")
    from jax_pbt.utils import split_into_minibatch

    T, E, Nw, Nf = 8, 6, 3, 4
    env_id = jnp.broadcast_to(jnp.arange(E)[None, :, None], (T, E, Nw)).astype(float)
    env_id_f = jnp.broadcast_to(jnp.arange(E)[None, :, None], (T, E, Nf)).astype(float)
    data = {"w": {"id": env_id}, "f": {"id": env_id_f}}
    mbs = split_into_minibatch(jax.random.key(0), data, chunk_length=4,
                               num_minibatches=2, minibatch_num_chunks=None)
    w_ids = mbs["w"]["id"]                       # (n_mb, L, rows, Nw)
    f_ids = mbs["f"]["id"]
    check("worker rows are constant within an env block",
          bool(jnp.all(w_ids == w_ids[..., :1])))
    check("worker and firm minibatch rows carry the same env ids",
          bool(jnp.all(w_ids[..., 0] == f_ids[..., 0])),
          "shuffles diverged between roles")


def main() -> None:
    print("=" * 72)
    print("HireRL MAPPO shared-critic tests")
    print("=" * 72)
    test_reconstruction()
    test_one_forward_all_values()
    test_minibatch_alignment()
    test_training_cycle()
    print("\n" + "=" * 72)
    if _failures:
        print(f"FAILED ({len(_failures)}): " + ", ".join(_failures))
        sys.exit(1)
    print("All tests passed.")


if __name__ == "__main__":
    main()
