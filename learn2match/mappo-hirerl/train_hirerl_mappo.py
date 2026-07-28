"""Two-role MAPPO training for HireRL with a shared centralized critic.

Same experiment shell as ``examples/train_hirerl_ippo.py`` (identical env
flags, eval rollouts, learning-curve npz/pngs, checkpoint cadence -- the
controller below subclasses ``HireRLController`` and reuses its ``run()``),
with the learning path swapped for MAPPO:

* Worker / firm actors are unchanged ``PerCandidateActorCritic`` policies
  acting on their own observations (their built-in value heads are unused).
* All Nw + Nf values per env come from ONE ``PairGridCritic`` forward per
  env-step (see central_critic.py) -- the world state is never replicated
  per agent, so nothing scales O(N^2) in memory or FLOPs.
* PPO minibatches sample (env, time-chunk) blocks so the critic always sees
  whole envs; three optimizers (worker actor, firm actor, shared critic).

Checkpoints keep the IPPO key layout (``worker_trainer_state`` /
``firm_trainer_state`` bytes have identical pytree structure), so
``eval_and_plot.py --policy ppo`` can load MAPPO checkpoints for acting;
``critic_trainer_state`` rides alongside for warm-starting MAPPO runs.
"""

import argparse
import os
import sys
import time
from typing import Callable

import jax
import optax
from flax.training.train_state import TrainState

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))          # repo root (jax_pbt)
sys.path.insert(0, os.path.join(_HERE, ".."))                 # learn2match (hirerl)
sys.path.insert(0, os.path.join(_HERE, "..", "examples"))     # IPPO controller + eval helpers

from train_hirerl_ippo import HireRLController

from central_critic import CriticInputSpec, PairGridCritic
from mappo_trainer import (
    make_compute_central_values,
    make_mappo_learn,
    make_mappo_rollout_step,
    rollout_to_env_major,
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MAPPO (shared centralized critic) training for HireRL. "
                    "Env / eval / logging flags match train_hirerl_ippo.py."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_name", type=str, default="hirerl_mappo")

    # Env (semantics documented in examples/train_hirerl_ippo.py)
    parser.add_argument("--Nw", type=int, default=10)
    parser.add_argument("--Nf", type=int, default=10)
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--sigma_interview", type=float, default=0.3)
    parser.add_argument("--sigma_match", type=float, default=0.1)
    parser.add_argument("--lambda_reveal", type=float, default=1.0)
    parser.add_argument("--outside_option", type=float, default=0.0,
                        help="Reservation utility threshold (see IPPO script).")
    parser.add_argument("--non_negative_features", action="store_true",
                        help="Sample x, y as |N(0,1)| so all pair utilities are >= 0.")
    parser.add_argument("--hat_init_value", type=float, default=0.0,
                        help="Constant optimistic init for hat_x / hat_y at reset.")
    parser.add_argument("--allow_on_the_job_search", action="store_true",
                        help="Matched agents may still interview elsewhere.")
    parser.add_argument("--noisy_hat_init", action="store_true",
                        help="Init hat_x / hat_y as noisy per-pair observations of the truth.")
    parser.add_argument("--sigma_init", type=float, default=3.0,
                        help="Std of the noisy hat init (requires > sigma_interview).")
    parser.add_argument("--public_retention_signal", action="store_true",
                        help="Broadcast post-retention belief updates market-wide.")
    parser.add_argument("--interview_mode", type=str, default="exclusive_role",
                        choices=["exclusive_role", "capacity_limited"])
    parser.add_argument("--init_from_ckpt", type=str, default="",
                        help="Warm-start trainer states from this checkpoint. MAPPO "
                             "checkpoints restore all three states; an IPPO checkpoint "
                             "restores the two actors and leaves the critic fresh.")

    # Training
    parser.add_argument("--total_env_steps", type=int, default=int(1e6))
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--rollout_length", type=int, default=128)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=4)
    parser.add_argument("--chunk_length", type=int, default=16)
    parser.add_argument(
        "--minibatch_size", type=int, default=None,
        help="Absolute minibatch size in ENV-TIMESTEPS (each sample carries all "
             "Nw + Nf agents of that env). Overrides --num_minibatches via "
             "minibatch_num_chunks = minibatch_size // chunk_length. NOTE the "
             "unit differs from the IPPO script, where it counts per-agent "
             "samples: MAPPO must keep whole envs together for the critic.")
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--gae_lam", type=float, default=0.95)
    parser.add_argument("--ratio_clip", type=float, default=0.2)
    parser.add_argument(
        "--scale_clip_eps", action="store_true",
        help="MAPPO-paper trick: divide ratio_clip by the number of agents "
             "sharing the policy update (Nw for the worker policy, Nf for the "
             "firm policy). Worth trying at large N.")
    parser.add_argument(
        "--value_clip", type=float, default=None,
        help="PPO value clipping threshold for the CENTRAL critic, clipped "
             "around the rollout-time prediction (the corrected formulation "
             "from examples/value_clipped_trainer.py). None disables.")
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--val_loss_coef", type=float, default=0.5,
                        help="Scales the shared critic's value MSE.")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--use_rnn", action="store_true",
                        help="Recurrent ACTORS (unchanged from IPPO). The central "
                             "critic is always feedforward: the belief tensors it "
                             "reads already summarize history.")

    # MAPPO-specific
    parser.add_argument("--critic_lr", type=float, default=None,
                        help="Learning rate for the shared critic (default: --lr).")
    parser.add_argument("--critic_hidden_size", type=int, default=64)
    parser.add_argument(
        "--critic_mode", type=str, default="privileged",
        choices=["privileged", "strict"],
        help="'privileged' (default): the critic additionally sees the true "
             "pair utilities x_i.y_j and market_t/horizon (legal under CTDE; "
             "stores one extra (Nw, Nf) grid per env-step). 'strict': the "
             "critic input is exactly the union of the two roles' observations, "
             "reconstructed from the rollout buffers with ZERO extra storage.")

    # Logging
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_job_type", type=str, default=None)

    # Learning-curve eval (identical to the IPPO script)
    parser.add_argument("--metric_log_every_env_steps", type=int, default=50_000)
    parser.add_argument("--num_eval_episodes", type=int, default=16)
    parser.add_argument("--num_eval_periods", type=int, default=None)
    parser.add_argument("--eval_seed", type=int, default=12345)

    # Checkpointing / plotting
    parser.add_argument("--ckpt_dir", type=str, default=os.path.join(_HERE, "checkpoints"))
    parser.add_argument("--save_best_ckpt", action="store_true")
    parser.add_argument("--save_all_checkpoints", action="store_true")
    parser.add_argument("--best_metric", type=str, default="social_welfare")
    parser.add_argument("--plot_dir", type=str, default=os.path.join(_HERE, "plots"))
    parser.add_argument("--plot_true_matched_welfare", action="store_true")
    return parser.parse_args()


class HireRLMAPPOController(HireRLController):
    """HireRLController with the IPPO learning path replaced by MAPPO.

    Reused from the parent: env/model/agent construction, eval rollouts,
    learning-curve buffers and npz/png output, the whole ``run()`` loop.
    Overridden: ``init_train`` (rollout with central values + env-block PPO
    update over three train states) and ``_save_checkpoint`` (adds the critic).
    """

    def __init__(self, args: argparse.Namespace) -> None:
        # Warm start is handled inside our init_run_state so the critic state
        # is restored too; blank the flag so the parent run()'s two-state
        # loader (which would drop the critic) never fires.
        self._mappo_init_ckpt = args.init_from_ckpt
        args.init_from_ckpt = ""
        super().__init__(args)

    def init_train(self) -> None:
        args = self.args
        env_fn = self.batched_env
        num_envs, Nw, Nf = self.num_envs, args.Nw, args.Nf

        worker_obs_space, _ = env_fn.get_observation_space()
        self.critic_spec = CriticInputSpec.from_spaces(
            worker_obs_space, Nw=Nw, Nf=Nf, d=args.d, horizon=args.horizon,
            privileged=(args.critic_mode == "privileged"),
        )
        self.critic = PairGridCritic(hidden_size=args.critic_hidden_size)
        critic_lr = args.critic_lr if args.critic_lr is not None else args.lr
        critic_tx = optax.chain(
            optax.clip_by_global_norm(args.grad_clip_norm),
            optax.adam(critic_lr),
        )

        rollout_step = make_mappo_rollout_step(
            env_fn, self.worker_agent, self.firm_agent,
            self.critic, self.critic_spec, num_envs,
        )
        compute_values = make_compute_central_values(
            self.critic, self.critic_spec, num_envs
        )
        minibatch_num_chunks = (
            args.minibatch_size // args.chunk_length
            if args.minibatch_size is not None else None
        )
        learn = make_mappo_learn(
            self.worker_model, self.firm_model, self.critic, self.critic_spec,
            gamma=args.gamma,
            gae_lam=args.gae_lam,
            ppo_epochs=args.ppo_epochs,
            chunk_length=args.chunk_length,
            num_minibatches=args.num_minibatches,
            minibatch_num_chunks=minibatch_num_chunks,
            ratio_clip_w=args.ratio_clip / (Nw if args.scale_clip_eps else 1),
            ratio_clip_f=args.ratio_clip / (Nf if args.scale_clip_eps else 1),
            entropy_coef=args.entropy_coef,
            value_clip=args.value_clip,
            val_loss_coef=args.val_loss_coef,
        )

        def init_run_state(rng):
            rng, rng_w, rng_f, rng_c = jax.random.split(rng, 4)
            trainer_state_lst = [
                self.worker_trainer.init_trainer_state(rng_w),
                self.firm_trainer.init_trainer_state(rng_f),
                {"critic_train_state": TrainState.create(
                    apply_fn=self.critic.apply,
                    params=self.critic.init_params(rng_c, self.critic_spec),
                    tx=critic_tx,
                )},
            ]
            if self._mappo_init_ckpt:
                trainer_state_lst = self._load_warm_start(trainer_state_lst)
            agent_state_lst = [
                agent_fn.init_agent_state(batch_shape)
                for agent_fn, batch_shape in zip(
                    [self.worker_agent, self.firm_agent],
                    env_fn.get_agent_batch_shape(),
                )
            ]
            rng, rng_reset = jax.random.split(rng)
            env_state, obs_lst = env_fn.reset(rng_reset, self.get_train_env_const_step())
            return rng, trainer_state_lst, agent_state_lst, env_state, obs_lst
        self.init_run_state = init_run_state

        def mappo_training_cycle(run_state, env_const_seq):
            rng, trainer_state_lst, agent_state_lst, env_state, last_obs_lst = run_state
            w_state, f_state, c_state = trainer_state_lst
            w_ms = self.worker_trainer.model_state_from_trainer_state(w_state)
            f_ms = self.firm_trainer.model_state_from_trainer_state(f_state)
            c_params = c_state["critic_train_state"].params

            def step(carry, env_const):
                rng, agent_state_lst, env_state, obs_lst = carry
                rng, next_agents, next_env_state, next_obs_lst, data, info = rollout_step(
                    rng, w_ms, f_ms, c_params, agent_state_lst,
                    env_const, env_state, obs_lst,
                )
                return (rng, next_agents, next_env_state, next_obs_lst), (data, info)

            (rng, agent_state_lst, env_state, last_obs_lst), (data_seq, _info_seq) = (
                jax.lax.scan(
                    step, (rng, agent_state_lst, env_state, last_obs_lst),
                    env_const_seq, length=self.train_rollout_length,
                )
            )

            rollout = rollout_to_env_major(data_seq, num_envs, Nw, Nf)
            last_val_w, last_val_f, _ = compute_values(
                c_params, last_obs_lst[0], last_obs_lst[1], env_state
            )

            rng, rng_learn = jax.random.split(rng)
            (w_ts, f_ts, c_ts), logs = learn(
                rng_learn,
                w_state["actor_critic_train_state"],
                f_state["actor_critic_train_state"],
                c_state["critic_train_state"],
                rollout, last_val_w, last_val_f,
            )
            new_trainer_state_lst = [
                {"actor_critic_train_state": w_ts},
                {"actor_critic_train_state": f_ts},
                {"critic_train_state": c_ts},
            ]

            train_info_lst = [
                {"avg_return": rollout["w"]["reward"].mean() * self.train_rollout_length},
                {"avg_return": rollout["f"]["reward"].mean() * self.train_rollout_length},
            ]
            # run() logs optim_info_lst[0] under worker/ and [1] under firm/;
            # shared-critic stats ride on the worker dict with a critic_ prefix
            # (they describe the one critic both roles share).
            optim_info_lst = [
                {**logs["worker"],
                 **{f"critic_{k}": v for k, v in logs["critic"].items()}},
                dict(logs["firm"]),
            ]
            return (
                (rng, new_trainer_state_lst, agent_state_lst, env_state, last_obs_lst),
                train_info_lst, optim_info_lst,
            )

        print("JIT-compiling MAPPO training cycle...")
        t0 = time.perf_counter()
        example_run_state = init_run_state(jax.random.key(0))
        env_const_seq = self.get_train_env_const_seq()
        self.training_cycle: Callable = jax.jit(mappo_training_cycle).lower(
            example_run_state, env_const_seq,
        ).compile()
        print(f"Compiled in {time.perf_counter() - t0:.1f}s")

    def _load_warm_start(self, templates: list) -> list:
        import pickle
        from flax import serialization
        with open(self._mappo_init_ckpt, "rb") as f:
            ck = pickle.load(f)
        out = [
            serialization.from_bytes(templates[0], ck["worker_trainer_state"]),
            serialization.from_bytes(templates[1], ck["firm_trainer_state"]),
        ]
        if "critic_trainer_state" in ck:
            out.append(serialization.from_bytes(templates[2], ck["critic_trainer_state"]))
            print(f"Warm-started worker/firm/critic trainer states from {self._mappo_init_ckpt}")
        else:
            out.append(templates[2])
            print(f"Warm-started worker/firm actors from {self._mappo_init_ckpt} "
                  f"(no critic_trainer_state key -- IPPO checkpoint? critic starts fresh)")
        return out

    def _save_checkpoint(self, trainer_state_lst, suffix: str = "") -> None:
        import pickle
        from flax import serialization
        os.makedirs(self.args.ckpt_dir, exist_ok=True)
        path = os.path.join(self.args.ckpt_dir, f"{self.args.run_name}{suffix}.pkl")
        data = {
            "worker_trainer_state": serialization.to_bytes(trainer_state_lst[0]),
            "firm_trainer_state":   serialization.to_bytes(trainer_state_lst[1]),
            "critic_trainer_state": serialization.to_bytes(trainer_state_lst[2]),
            "model_args": {
                "hidden_size": self.args.hidden_size,
                "use_rnn": self.args.use_rnn,
                "critic_hidden_size": self.args.critic_hidden_size,
                "critic_mode": self.args.critic_mode,
            },
            "env_args": {
                "Nw": self.args.Nw, "Nf": self.args.Nf, "d": self.args.d,
                "horizon": self.args.horizon,
                "sigma_interview": self.args.sigma_interview,
                "sigma_match": self.args.sigma_match,
                "lambda_reveal": self.args.lambda_reveal,
                "outside_option": self.args.outside_option,
                "non_negative_features": self.args.non_negative_features,
                "hat_init_value": self.args.hat_init_value,
                "allow_on_the_job_search": self.args.allow_on_the_job_search,
                "noisy_hat_init": self.args.noisy_hat_init,
                "sigma_init": self.args.sigma_init,
                "public_retention_signal": self.args.public_retention_signal,
                "interview_mode": self.args.interview_mode,
            },
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved MAPPO checkpoint to {path}")


def _validate_minibatch(args: argparse.Namespace) -> None:
    """Fail fast, and legibly, on an IPPO-unit --minibatch_size.

    The sampling unit here is the env-timestep (all Nw + Nf agents of an env
    travel together, because the critic input is that env's whole pair grid).
    Passing the IPPO value -- which counts per-agent samples and is therefore
    ~N times larger -- otherwise dies with a ZeroDivisionError deep inside
    split_into_minibatch, after the JIT compile has already been paid for.
    """
    if args.minibatch_size is None:
        return
    chunks = (args.rollout_length // args.chunk_length) * args.num_envs
    want = args.minibatch_size // args.chunk_length
    if want < 1:
        raise SystemExit(
            f"--minibatch_size {args.minibatch_size} is smaller than "
            f"--chunk_length {args.chunk_length}; it must cover at least one chunk."
        )
    if want > chunks:
        per_role = max(args.Nw, args.Nf)
        raise SystemExit(
            f"--minibatch_size {args.minibatch_size} env-timesteps needs {want} "
            f"chunks, but this rollout only has {chunks} "
            f"(rollout_length {args.rollout_length} / chunk_length "
            f"{args.chunk_length} x num_envs {args.num_envs}).\n"
            f"NOTE MAPPO counts ENV-TIMESTEPS here, while train_hirerl_ippo.py "
            f"counts PER-AGENT samples -- each env-timestep already carries all "
            f"{args.Nw + args.Nf} agents. To match an IPPO run, divide its value "
            f"by the per-role agent count: {args.minibatch_size} / {per_role} = "
            f"{args.minibatch_size // per_role}."
        )


def main() -> None:
    args = get_args()
    _validate_minibatch(args)
    HireRLMAPPOController(args).run()


if __name__ == "__main__":
    main()
