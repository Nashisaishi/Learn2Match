"""Eval rollout + W&B logging for HireRL using a trained PPO policy.

Loads a checkpoint produced by train_hirerl_ippo.py, vmaps a fixed-length
``lax.scan`` rollout across ``--num_seeds`` independent seeds in one shot,
collects per-agent and aggregate reward / regret / information loss at each
post-RETENTION snapshot, plots seed-aggregated curves (mean line + 95% CI band)
with matplotlib (including a seed-0 worker-side Gantt of who matched whom),
saves cumulative-metric CSVs, and uploads scalars + figures to W&B.

Per-agent regret / information-loss metrics come from
``hirerl.metrics.compute_per_agent_metrics`` (a jit-compiled bundle of
``per_{worker,firm}_regret`` and ``per_{worker,firm}_information_loss``).

Run example:
    python eval_and_plot.py --ppo_ckpt examples/checkpoints/baseline_10x10.pkl
    python eval_and_plot.py --ppo_ckpt path/to/ckpt.pkl --no_wandb
    python eval_and_plot.py --ppo_ckpt path/to/ckpt.pkl --num_seeds 64 \\
        --lambda_reveal 50 --sigma_match 0.0
"""

import argparse
import os
import sys
import time
from datetime import datetime

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))   # repo root for jax_pbt
sys.path.insert(0, os.path.join(_HERE, ".."))           # learn2match for hirerl

from jax_pbt.env.spaces import Action

from hirerl import HireRLConfig, HireRLEnv
from hirerl.metrics import compute_per_agent_metrics
from plot_helpers import make_batched_figures, save_batched_cumulative_csvs


# Number of phase steps per market period -- INTERVIEW_PROPOSE,
# INTERVIEW_RESPOND, MATCH_PROPOSE, MATCH_RESPOND, RETENTION (see hirerl.env).
# After RETENTION the env writes settled_metrics into info and increments
# market_t, so the post-step state at phase index 4 IS the settled snapshot.
STEPS_PER_PERIOD = 5


def compute_planner_welfare(x_arr: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
    """First-best two-sided welfare of a social planner who observes true x, y
    and solves the maximum-weight bipartite matching (linear assignment).

    Per seed: 2 * max_M sum_{(i,j) in M} <x_i, y_j>, where M ranges over all
    one-to-one matchings between workers and firms. Constant within an episode
    since x, y are fixed at reset.

    Args:
        x_arr: (num_seeds, Nw, d) worker types.
        y_arr: (num_seeds, Nf, d) firm types.

    Returns:
        (num_seeds,) per-seed planner welfare.
    """
    from scipy.optimize import linear_sum_assignment
    num_seeds = x_arr.shape[0]
    out = np.empty(num_seeds, dtype=np.float32)
    for s in range(num_seeds):
        U = x_arr[s] @ y_arr[s].T              # (Nw, Nf)
        rows, cols = linear_sum_assignment(-U)  # negate for max
        out[s] = 2.0 * float(U[rows, cols].sum())
    return out


def load_ppo_ckpt(env, ckpt_path):
    """Load a PPO checkpoint and return ``(ppo_agent_w, ppo_agent_f,
    w_model_state, f_model_state)``.

    The model_states are the inference parameters extracted from the trainer
    states stored in the checkpoint. Callers who want a model_state-injectable
    rollout (e.g. periodic eval during training) should call this directly
    plus :func:`build_stateful_rollout_fn`. Callers who want a one-shot
    closed-over rollout (this file's ``main()``) should call
    :func:`make_ppo_policy` instead.
    """
    import pickle
    from flax import serialization
    from jax_pbt.policy.actor_critic import (
        ActorCriticPPOAgent as PPOAgent,
        ActorCriticPPOTrainer as PPOTrainer,
    )
    from hirerl import MaskedSharedActorCriticModel

    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    env_args = ckpt["env_args"]
    if (env.Nw, env.Nf, env.d) != (env_args["Nw"], env_args["Nf"], env_args["d"]):
        raise ValueError(
            f"Checkpoint shapes (Nw={env_args['Nw']}, Nf={env_args['Nf']}, d={env_args['d']}) "
            f"differ from eval env (Nw={env.Nw}, Nf={env.Nf}, d={env.d}); model params won't fit."
        )
    hidden_size = ckpt["model_args"]["hidden_size"]
    use_rnn = ckpt["model_args"]["use_rnn"]

    obs_w, obs_f = env.get_observation_space()
    act_w, act_f = env.get_action_space()
    worker_model = MaskedSharedActorCriticModel.build(obs_w, act_w, hidden_size=hidden_size, use_rnn=use_rnn)
    firm_model   = MaskedSharedActorCriticModel.build(obs_f, act_f, hidden_size=hidden_size, use_rnn=use_rnn)

    # Trainer hyperparameters don't affect inference; defaults are fine for deserialization.
    ppo_kwargs = dict(
        feature_shared=True, lr=3e-4, grad_clip_norm=1.0,
        val_loss_coef=0.5, entropy_coef=0.01,
        gamma=0.99, gae_lam=0.95, ppo_epochs=1, ratio_clip=0.2,
        chunk_length=4, num_minibatches=1,
    )
    worker_trainer = PPOTrainer(actor_critic_fn=worker_model, **ppo_kwargs)
    firm_trainer   = PPOTrainer(actor_critic_fn=firm_model,   **ppo_kwargs)

    target_w = worker_trainer.init_trainer_state(jax.random.key(0))
    target_f = firm_trainer.init_trainer_state(jax.random.key(1))
    w_state = serialization.from_bytes(target_w, ckpt["worker_trainer_state"])
    f_state = serialization.from_bytes(target_f, ckpt["firm_trainer_state"])

    w_model_state = worker_trainer.model_state_from_trainer_state(w_state)
    f_model_state = firm_trainer.model_state_from_trainer_state(f_state)

    ppo_agent_w = PPOAgent(worker_model)
    ppo_agent_f = PPOAgent(firm_model)
    return ppo_agent_w, ppo_agent_f, w_model_state, f_model_state


def make_ppo_policy(env, ckpt_path):
    """Load a PPO checkpoint and return ``(policy_step, init_agent_state)``,
    both pure functions that compose with ``jax.lax.scan`` / ``jax.vmap``.

    * ``init_agent_state()`` -> agent_state pytree (worker_state, firm_state).
    * ``policy_step(rng, obs_lst, agent_state)`` ->
        ((w_choice, f_choice), next_agent_state, rng).

    Closes over the checkpoint's model_state. Use :func:`load_ppo_ckpt` +
    :func:`build_stateful_rollout_fn` instead if you need to swap model_state
    across calls (e.g. periodic eval during training).
    """
    ppo_agent_w, ppo_agent_f, w_model_state, f_model_state = load_ppo_ckpt(env, ckpt_path)
    Nw, Nf = env.Nw, env.Nf

    def init_agent_state():
        return (
            ppo_agent_w.init_agent_state((Nw,)),
            ppo_agent_f.init_agent_state((Nf,)),
        )

    def policy_step(rng, obs_lst, agent_state):
        agent_w_state, agent_f_state = agent_state
        rng, k_w, k_f = jax.random.split(rng, 3)
        next_w, w_action, _ = ppo_agent_w.step(k_w, w_model_state, agent_w_state, obs_lst[0])
        next_f, f_action, _ = ppo_agent_f.step(k_f, f_model_state, agent_f_state, obs_lst[1])
        return (w_action["choice"], f_action["choice"]), (next_w, next_f), rng

    return policy_step, init_agent_state


def build_rollout_fn(env, policy_step, init_agent_state, num_market_steps,
                     compute_true_matched_welfare: bool = False):
    """Build a pure ``rollout_fn(rng) -> history`` whose body is a
    ``lax.scan`` over ``num_market_steps`` market periods, each consisting of
    an inner ``lax.scan`` over ``STEPS_PER_PERIOD`` phase steps. The
    settled snapshot is the post-RETENTION state, i.e. the env state after
    the last phase step in the inner scan.

    The returned ``rollout_fn`` is intentionally not pre-jitted so the caller
    can ``jax.vmap`` then ``jax.jit`` it.

    Output history dict keys (each shape ``(num_market_steps, ...)``):
    * scalars: ``t``, ``reward_w_total``, ``reward_f_total``,
      ``regret_w_total``, ``regret_f_total``, ``info_loss_w_total``,
      ``info_loss_f_total``, ``social_welfare``, ``social_welfare_da_ref``, ``friction_loss``,
      ``match_rate``;
    * per-agent: ``reward_{w,f}_per_agent`` (T, N{w,f}),
      ``regret_{w,f}_per_agent``, ``info_loss_{w,f}_per_agent``;
    * ``matched_matrix`` (T, Nw, Nf) bool.

    When ``compute_true_matched_welfare`` is True, the record also contains
    ``true_matched_welfare = 2 * sum(matched * <x, y>)``, the welfare under
    the policy's actual matching evaluated on TRUE x, y. This is bounded
    above by ``social_welfare_planner`` per seed per period (since planner
    picks the max-weight matching), so it gives a clean apples-to-apples
    upper-bound comparison vs the belief-based ``social_welfare``.
    """
    const = env.default_const
    outside = float(env.config.outside_option)

    def step_phase(carry, _):
        rng, env_state, obs_lst, agent_state = carry
        rng, sub = jax.random.split(rng)
        (w_choice, f_choice), next_agent_state, rng = policy_step(rng, obs_lst, agent_state)
        action = [
            Action({"choice": jnp.asarray(w_choice, dtype=jnp.int32)}),
            Action({"choice": jnp.asarray(f_choice, dtype=jnp.int32)}),
        ]
        next_env_state, next_obs_lst, rew_lst, _, info = env.step(sub, const, env_state, action)
        return (rng, next_env_state, next_obs_lst, next_agent_state), (rew_lst, info)

    def step_market_period(carry, _):
        # Inner scan: one full market period (5 phase steps). The settled
        # snapshot is the post-RETENTION env state, i.e. carry after the inner
        # scan; settled_metrics live in info_seq at index -1.
        carry, (rew_seq, info_seq) = jax.lax.scan(
            step_phase, carry, xs=None, length=STEPS_PER_PERIOD,
        )
        _rng, env_state, _obs_lst, _agent_state = carry
        rew_w_settled = rew_seq[0][-1]   # (Nw,)
        rew_f_settled = rew_seq[1][-1]   # (Nf,)
        sm = jax.tree.map(lambda x: x[-1], info_seq["settled_metrics"])

        per = compute_per_agent_metrics(env_state, outside)
        record = {
            "t": env_state.market_t,
            "reward_w_total":    jnp.sum(rew_w_settled),
            "reward_f_total":    jnp.sum(rew_f_settled),
            "regret_w_total":    jnp.sum(per["per_worker_regret"]),
            "regret_f_total":    jnp.sum(per["per_firm_regret"]),
            "info_loss_w_total": jnp.sum(per["per_worker_information_loss"]),
            "info_loss_f_total": jnp.sum(per["per_firm_information_loss"]),
            "social_welfare":    sm["social_welfare"],
            "social_welfare_da_ref": sm["social_welfare_da_ref"],
            "friction_loss":     sm["friction_loss"],
            "match_rate":        sm["match_rate"],
            "reward_w_per_agent":    rew_w_settled,
            "reward_f_per_agent":    rew_f_settled,
            "regret_w_per_agent":    per["per_worker_regret"],
            "regret_f_per_agent":    per["per_firm_regret"],
            "info_loss_w_per_agent": per["per_worker_information_loss"],
            "info_loss_f_per_agent": per["per_firm_information_loss"],
            "matched_matrix":        env_state.matched,
            "interviewed_matrix":    env_state.interviewed,
            "cumulative_tenure":     env_state.cumulative_tenure,
        }
        if compute_true_matched_welfare:
            U_true = jnp.einsum("id,jd->ij", env_state.x, env_state.y)
            record["true_matched_welfare"] = 2.0 * jnp.sum(
                env_state.matched.astype(jnp.float32) * U_true
            )
        return carry, record

    def rollout_fn(rng):
        rng, rng_reset = jax.random.split(rng)
        env_state, obs_lst = env.reset(rng_reset, const)
        agent_state = init_agent_state()
        init_carry = (rng, env_state, obs_lst, agent_state)
        _, history = jax.lax.scan(
            step_market_period, init_carry, xs=None, length=num_market_steps,
        )
        return history, env_state.x, env_state.y

    return rollout_fn


def build_stateful_rollout_fn(env, ppo_agent_w, ppo_agent_f, num_market_steps,
                               compute_true_matched_welfare: bool = False):
    """Like :func:`build_rollout_fn` but takes ``(w_model_state, f_model_state)``
    as run-time arguments so the same compiled rollout can be reused across
    training as the policy parameters change.

    Returns ``rollout_fn(rng, w_model_state, f_model_state) -> history`` with
    the same history schema as :func:`build_rollout_fn`. ``ppo_agent_w`` and
    ``ppo_agent_f`` are static (closed over); the model_state pytrees are
    threaded through ``lax.scan`` carry so that ``jax.jit`` traces them as
    inputs.
    """
    const = env.default_const
    outside = float(env.config.outside_option)
    Nw, Nf = env.Nw, env.Nf

    def step_phase(carry, _):
        rng, env_state, obs_lst, agent_state, w_ms, f_ms = carry
        agent_w_state, agent_f_state = agent_state
        rng, k_w, k_f, sub = jax.random.split(rng, 4)
        next_w_ag, w_action, _ = ppo_agent_w.step(k_w, w_ms, agent_w_state, obs_lst[0])
        next_f_ag, f_action, _ = ppo_agent_f.step(k_f, f_ms, agent_f_state, obs_lst[1])
        action = [
            Action({"choice": jnp.asarray(w_action["choice"], dtype=jnp.int32)}),
            Action({"choice": jnp.asarray(f_action["choice"], dtype=jnp.int32)}),
        ]
        next_env_state, next_obs_lst, rew_lst, _, info = env.step(sub, const, env_state, action)
        return (
            (rng, next_env_state, next_obs_lst, (next_w_ag, next_f_ag), w_ms, f_ms),
            (rew_lst, info),
        )

    def step_market_period(carry, _):
        carry, (rew_seq, info_seq) = jax.lax.scan(
            step_phase, carry, xs=None, length=STEPS_PER_PERIOD,
        )
        _rng, env_state, _obs_lst, _agent_state, _w_ms, _f_ms = carry
        rew_w_settled = rew_seq[0][-1]
        rew_f_settled = rew_seq[1][-1]
        sm = jax.tree.map(lambda x: x[-1], info_seq["settled_metrics"])

        per = compute_per_agent_metrics(env_state, outside)
        record = {
            "t": env_state.market_t,
            "reward_w_total":    jnp.sum(rew_w_settled),
            "reward_f_total":    jnp.sum(rew_f_settled),
            "regret_w_total":    jnp.sum(per["per_worker_regret"]),
            "regret_f_total":    jnp.sum(per["per_firm_regret"]),
            "info_loss_w_total": jnp.sum(per["per_worker_information_loss"]),
            "info_loss_f_total": jnp.sum(per["per_firm_information_loss"]),
            "social_welfare":    sm["social_welfare"],
            "social_welfare_da_ref": sm["social_welfare_da_ref"],
            "friction_loss":     sm["friction_loss"],
            "match_rate":        sm["match_rate"],
        }
        if compute_true_matched_welfare:
            U_true = jnp.einsum("id,jd->ij", env_state.x, env_state.y)
            record["true_matched_welfare"] = 2.0 * jnp.sum(
                env_state.matched.astype(jnp.float32) * U_true
            )
        return carry, record

    def rollout_fn(rng, w_model_state, f_model_state):
        rng, rng_reset = jax.random.split(rng)
        env_state, obs_lst = env.reset(rng_reset, const)
        agent_state = (
            ppo_agent_w.init_agent_state((Nw,)),
            ppo_agent_f.init_agent_state((Nf,)),
        )
        init_carry = (rng, env_state, obs_lst, agent_state, w_model_state, f_model_state)
        _, history = jax.lax.scan(
            step_market_period, init_carry, xs=None, length=num_market_steps,
        )
        return history, env_state.x, env_state.y

    return rollout_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo_ckpt", type=str, required=True,
                        help="Path to a checkpoint produced by train_hirerl_ippo.py")
    parser.add_argument("--Nw", type=int, default=10)
    parser.add_argument("--Nf", type=int, default=10)
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--num_periods", type=int, default=100)
    parser.add_argument("--sigma_interview", type=float, default=0.3)
    parser.add_argument("--sigma_match", type=float, default=0.1)
    parser.add_argument("--lambda_reveal", type=float, default=1.0)
    parser.add_argument(
        "--outside_option", type=float, default=0.0,
        help="Reservation utility threshold; should match the value used at "
             "training time. Pass -1e9 to effectively disable (matches "
             "original CA-ETC paper).",
    )
    parser.add_argument(
        "--non_negative_features", action="store_true",
        help="Sample x, y as half-normal so x.y >= 0 always. Should match the "
             "value used at training time.",
    )
    parser.add_argument(
        "--hat_init_value", type=float, default=0.0,
        help="Optimistic init for hat_x and hat_y at episode reset. Default "
             "0.0 reproduces the historical pessimistic init. Should match "
             "the value used at training time unless intentionally probing "
             "OOD belief priors. See train_hirerl_ippo.py for value guidance.",
    )
    parser.add_argument(
        "--allow_on_the_job_search", action="store_true",
        help="Allow matched workers / firms to interview while still matched. "
             "Should match the value used at training time -- mismatched flags "
             "shift the action mask (and obs) distribution and degrade the "
             "policy's behaviour out-of-distribution. See train_hirerl_ippo.py.",
    )
    parser.add_argument(
        "--noisy_hat_init", action="store_true",
        help="Initialize hat_x and hat_y as per-pair noisy observations of "
             "the true latent (instead of the constant hat_init_value). "
             "Should match the value used at training time -- mismatched "
             "flags change the obs distribution at episode start and degrade "
             "the policy's behaviour OOD. See train_hirerl_ippo.py.",
    )
    parser.add_argument(
        "--sigma_init", type=float, default=3.0,
        help="Std-dev of the noisy hat init when --noisy_hat_init is set. "
             "Must be > --sigma_interview. Should match the training-time "
             "value. Ignored when --noisy_hat_init is not set.",
    )
    parser.add_argument(
        "--public_retention_signal", action="store_true",
        help="Make post-retention belief updates public (broadcast across all "
             "firms / workers, and don't let interview obs overwrite a worker "
             "/ firm that has been retained anywhere). Should match the value "
             "used at training time -- mismatched flags shift the obs "
             "distribution and degrade the policy OOD.",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed; per-seed rollouts use seed + i for i in 0..num_seeds-1.")
    parser.add_argument("--num_seeds", type=int, default=32,
                        help="Number of independent eval rollouts to run sequentially. "
                             "Aggregates (mean / 95% CI) are reported across these seeds.")
    parser.add_argument("--wandb_entity", default="haijingzong-university-of-washington")
    parser.add_argument("--wandb_project", default="hireRL")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--no_wandb", action="store_true",
                        help="Skip W&B upload; just save PNGs to ./eval_outputs")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory for PNGs / CSVs. Defaults to a "
                             "timestamped subdir under examples/eval_outputs/.")
    parser.add_argument(
        "--plot_true_matched_welfare", action="store_true",
        help="Also compute and plot true_matched_welfare = 2*sum(matched * <x, y>) "
             "on the social-welfare aux figure. This is the welfare under the "
             "policy's actual matching evaluated on TRUE x, y, so it is bounded "
             "above by the planner welfare per seed per period -- gives a clean "
             "apples-to-apples comparison without belief-noise inflation.",
    )
    args = parser.parse_args()

    config = HireRLConfig(
        Nw=args.Nw, Nf=args.Nf, d=args.d,
        # +10 buffer keeps env from auto-resetting at horizon mid-eval (env.py:260).
        horizon=args.num_periods + 10,
        sigma_interview=args.sigma_interview,
        sigma_match=args.sigma_match,
        lambda_reveal=args.lambda_reveal,
        outside_option=args.outside_option,
        non_negative_features=args.non_negative_features,
        hat_init_value=args.hat_init_value,
        allow_on_the_job_search=args.allow_on_the_job_search,
        noisy_hat_init=args.noisy_hat_init,
        sigma_init=args.sigma_init,
        public_retention_signal=args.public_retention_signal,
    )
    env = HireRLEnv(config)

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        run_name = args.run_name or f"eval-{datetime.now():%Y%m%d-%H%M%S}"
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            config={
                "Nw": args.Nw, "Nf": args.Nf, "d": args.d,
                "num_periods": args.num_periods,
                "num_seeds": args.num_seeds,
                "sigma_interview": args.sigma_interview,
                "sigma_match": args.sigma_match,
                "lambda_reveal": args.lambda_reveal,
                "outside_option": args.outside_option,
                "non_negative_features": args.non_negative_features,
                "hat_init_value": args.hat_init_value,
                "allow_on_the_job_search": args.allow_on_the_job_search,
                "noisy_hat_init": args.noisy_hat_init,
                "sigma_init": args.sigma_init,
                "public_retention_signal": args.public_retention_signal,
                "seed": args.seed,
                "ppo_ckpt": args.ppo_ckpt,
            },
        )

    policy_step, init_agent_state = make_ppo_policy(env, args.ppo_ckpt)
    rollout_fn = build_rollout_fn(
        env, policy_step, init_agent_state, args.num_periods,
        compute_true_matched_welfare=args.plot_true_matched_welfare,
    )
    rollout_batched = jax.jit(jax.vmap(rollout_fn))

    base_rng = jax.random.key(args.seed)
    seed_rngs = jax.random.split(base_rng, args.num_seeds)

    print(
        f"Rolling out {args.num_seeds} seeds x {args.num_periods} market periods "
        f"(Nw={args.Nw}, Nf={args.Nf}, d={args.d}) via lax.scan + vmap..."
    )
    t0 = time.perf_counter()
    batched_history, batched_x, batched_y = rollout_batched(seed_rngs)
    # Block on one leaf so wall-clock measurement reflects compute, not async dispatch.
    jax.block_until_ready(batched_history["regret_w_total"])
    elapsed = time.perf_counter() - t0
    T_out = int(batched_history["regret_w_total"].shape[1])
    print(
        f"  done in {elapsed:.1f}s "
        f"({args.num_seeds * T_out / max(elapsed, 1e-9):.0f} seed-periods/sec)"
    )

    # Convert to numpy + add scalar metadata fields expected by
    # plot_helpers.make_batched_figures / save_batched_cumulative_csvs.
    # ``t`` is identical across seeds (market_t advances deterministically once
    # per RETENTION), so we collapse the leading seed axis for it only.
    history = {
        "t": np.asarray(batched_history["t"][0]),
        "num_seeds": args.num_seeds,
        "Nw": args.Nw,
        "Nf": args.Nf,
    }
    for key, arr in batched_history.items():
        if key == "t":
            continue
        history[key] = np.asarray(arr)

    # First-best welfare under a social planner (Hungarian on true utilities).
    # Per-seed scalar; episode-constant since x, y are fixed at reset.
    history["social_welfare_planner"] = compute_planner_welfare(
        np.asarray(batched_x), np.asarray(batched_y)
    )

    if use_wandb:
        cum_reg_w = np.cumsum(history["regret_w_total"], axis=1)
        cum_reg_f = np.cumsum(history["regret_f_total"], axis=1)
        cum_fric  = np.cumsum(history["friction_loss"], axis=1)
        for idx, t in enumerate(history["t"]):
            log = {}
            for key in ("reward_w_total", "reward_f_total",
                        "regret_w_total", "regret_f_total",
                        "info_loss_w_total", "info_loss_f_total",
                        "social_welfare", "friction_loss", "match_rate"):
                arr = history[key][:, idx]
                log[f"agg/{key}_mean"] = float(arr.mean())
                log[f"agg/{key}_std"]  = float(arr.std())
            for key, arr in [("cum_regret_w", cum_reg_w),
                             ("cum_regret_f", cum_reg_f),
                             ("cum_friction_loss", cum_fric)]:
                slc = arr[:, idx]
                log[f"cum/{key}_mean"] = float(slc.mean())
                log[f"cum/{key}_std"]  = float(slc.std())
            wandb.log(log, step=int(t))

    figs = make_batched_figures(history, config)
    if args.out_dir is None:
        out_dir = os.path.join(
            _HERE, "eval_outputs",
            f"ppo_eval_{args.Nw}x{args.Nf}_h{args.num_periods}_"
            f"N{args.num_seeds}_seed{args.seed}_{datetime.now():%Y%m%d-%H%M%S}",
        )
    else:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    # Whitelist the figures that go to W&B. PNGs are saved locally for every
    # figure regardless. Update this set if you want to surface more in W&B.
    # Slash-keyed figures (e.g. "aux/social_welfare") nest under a panel
    # section in W&B (plot/aux/social_welfare) and a subdirectory locally.
    WANDB_FIG_KEYS = {
        "cumulative_worker_regret",
        "cumulative_firm_regret",
        "cumulative_friction_loss",
        "cumulative_per_worker",
        "cumulative_per_firm",
        "aux/social_welfare",
        "aux/friction_loss",
        "aux/match_rate",
        "coverage_heatmap",
    }
    for name, fig in figs.items():
        path = os.path.join(out_dir, f"{name}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"  saved {path}")
        if use_wandb and name in WANDB_FIG_KEYS:
            wandb.log({f"plot/{name}": wandb.Image(fig)})
        plt.close(fig)

    csv_paths = save_batched_cumulative_csvs(history, out_dir)
    for name, path in csv_paths.items():
        print(f"  saved {path}")

    print(f"All eval outputs written to {out_dir}")

    if use_wandb:
        cum_reg_w = np.cumsum(history["regret_w_total"], axis=1)
        cum_reg_f = np.cumsum(history["regret_f_total"], axis=1)
        cum_fric  = np.cumsum(history["friction_loss"], axis=1)
        wandb.run.summary["mean_final_cum_regret_w"]      = float(cum_reg_w[:, -1].mean())
        wandb.run.summary["std_final_cum_regret_w"]       = float(cum_reg_w[:, -1].std())
        wandb.run.summary["mean_final_cum_regret_f"]      = float(cum_reg_f[:, -1].mean())
        wandb.run.summary["std_final_cum_regret_f"]       = float(cum_reg_f[:, -1].std())
        wandb.run.summary["mean_final_cum_friction_loss"] = float(cum_fric[:, -1].mean())
        wandb.run.summary["std_final_cum_friction_loss"]  = float(cum_fric[:, -1].std())
        wandb.run.summary["num_seeds"]                    = int(args.num_seeds)
        wandb.run.summary["market_periods"]               = int(history["regret_w_total"].shape[1])
        wandb.finish()
        print("W&B run finished.")


if __name__ == "__main__":
    main()
