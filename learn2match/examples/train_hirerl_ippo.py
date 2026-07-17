"""Two-role IPPO training for HireRL.

Worker policy controls all workers (PPO batch = num_envs * Nw).
Firm policy controls all firms   (PPO batch = num_envs * Nf).
"""

import argparse
import os
import sys
import time
from typing import Callable, Sequence, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import PyTreeNode
from tqdm import tqdm

# Make jax_pbt and the local hirerl package importable when run from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))   # repo root (for jax_pbt)
sys.path.insert(0, os.path.join(_HERE, ".."))           # learn2match (for hirerl)

from jax_pbt.controller.ippo_controller import IPPOController
from jax_pbt.env.batched_env import RoleAssignmentBatchedEnv
from jax_pbt.policy.actor_critic import (
    ActorCriticPPOAgent as PPOAgent,
    ActorCriticPPOTrainer as PPOTrainer,
)
from jax_pbt.trainer.ppo import PPOTransition
from jax_pbt.utils import pytree_repeat_stack, rng_batch_split

from hirerl import HireRLConfig, HireRLEnv, PerCandidateActorCriticModel
from hirerl.metrics import compute_all_metrics

from eval_and_plot import build_stateful_rollout_fn, compute_planner_welfare


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_name", type=str, default="hirerl_ippo")

    # Env
    parser.add_argument("--Nw", type=int, default=10)
    parser.add_argument("--Nf", type=int, default=10)
    parser.add_argument("--d", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--sigma_interview", type=float, default=0.3)
    parser.add_argument("--sigma_match", type=float, default=0.1)
    parser.add_argument("--lambda_reveal", type=float, default=1.0)
    parser.add_argument(
        "--outside_option", type=float, default=0.0,
        help="Reservation utility threshold. With default 0.0, pairs with x.y < 0 "
             "would be rejected by the reference matching while the policy may "
             "still match them (positive per-period regret in steady state). "
             "Pass a very negative value (e.g. -1e9) to disable, matching the "
             "original CA-ETC paper's no-outside-option setup.",
    )
    parser.add_argument(
        "--non_negative_features", action="store_true",
        help="Sample x and y as half-normal (|N(0, 1)|) so every pair has "
             "x.y >= 0. With this flag, outside_option=0 doesn't bite -- "
             "closest to the original CA-ETC paper where all mean rewards "
             "lie in [0, 0.99].",
    )
    parser.add_argument(
        "--hat_init_value", type=float, default=0.0,
        help="Optimistic initialization for hat_x and hat_y at episode reset "
             "(applied symmetrically to both belief tensors). Default 0.0 "
             "reproduces the historical pessimistic init under "
             "non_negative_features=True (unobserved pairs look like utility=0). "
             "Suggested values under |N(0,1)| features: ~0.8 (neutral prior, "
             "matches E[y_dim]); ~1.5 (mild optimism, ~75-90th percentile); "
             "~2.0 (UCB-style 95th percentile, strong exploration push); "
             ">=3.0 (extreme optimism, may suppress settling). Larger values "
             "induce more exploration but risk dropping match rate.",
    )
    parser.add_argument(
        "--allow_on_the_job_search", action="store_true",
        help="Allow matched workers / firms to propose and accept *interviews* "
             "with other agents while remaining matched. Match-switching still "
             "requires explicit dissolution in the retention phase first. "
             "Default off reproduces the original 'only the unemployed search' "
             "constraint. Turn this on to break the exploration deadlock where "
             "high match rates make INTERVIEW_PROPOSE action masks degenerate "
             "(only NOOP allowed for matched agents); enables hat_init_value "
             "and entropy_coef to actually drive exploration.",
    )
    parser.add_argument(
        "--noisy_hat_init", action="store_true",
        help="Initialize hat_x and hat_y as per-pair noisy observations of "
             "the true latent (hat_x[i,j] = x[i] + N(0, sigma_init), and "
             "likewise for hat_y) instead of the constant hat_init_value. "
             "Every pair gets a distinguishable but noisy prior, which lets "
             "policies form informed initial preferences and gives matched "
             "agents a basis for on-the-job search. When set, --hat_init_value "
             "is ignored. Requires --sigma_init > --sigma_interview.",
    )
    parser.add_argument(
        "--sigma_init", type=float, default=3.0,
        help="Std-dev of the noisy hat init when --noisy_hat_init is set. "
             "Must be strictly greater than --sigma_interview (recommended "
             ">= 3 * sigma_interview) so the prior is meaningfully noisier "
             "than any subsequent interview observation. Ignored when "
             "--noisy_hat_init is not set.",
    )
    parser.add_argument(
        "--public_retention_signal", action="store_true",
        help="Make post-retention belief updates public: the noisy signal "
             "sigmoid(lambda*tau)*x + eps about a retained worker is broadcast "
             "across all firms (symmetric on the firm side), and once a worker "
             "has been retained anywhere, subsequent interview observations no "
             "longer overwrite hat_x[worker, :, :]. Default off preserves the "
             "original per-pair private-belief semantics bit-exactly. Should "
             "match between training and eval -- mismatched flags shift the "
             "obs distribution and degrade the policy OOD.",
    )

    # Training
    parser.add_argument("--total_env_steps", type=int, default=int(1e6))
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--rollout_length", type=int, default=128)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=4)
    parser.add_argument("--chunk_length", type=int, default=16)
    parser.add_argument(
        "--minibatch_size", type=int, default=None,
        help="Absolute minibatch size in samples (across vmap-Nw-rollout dims). "
             "When set, overrides --num_minibatches by feeding "
             "minibatch_num_chunks = minibatch_size // chunk_length to the "
             "PPO trainer. Use this to decouple batch size from buffer size "
             "(num_envs * Nw * rollout_length), so memory stays predictable "
             "when scaling N or num_envs. Rounds DOWN to a multiple of "
             "chunk_length=16.")
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--gae_lam", type=float, default=0.95)
    parser.add_argument("--ratio_clip", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--val_loss_coef", type=float, default=0.5)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--use_rnn", action="store_true")

    # Logging
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_job_type", type=str, default=None)

    # Learning-curve eval (periodic eval rollouts averaged over episodes)
    parser.add_argument(
        "--metric_log_every_env_steps", type=int, default=50_000,
        help="How often (in env steps) to run an eval rollout, log aggregate "
             "metrics to W&B, and append to the learning-curve buffer. "
             "0 disables eval entirely (also skips the final "
             "learning_curve.png/.npz output).",
    )
    parser.add_argument("--num_eval_episodes", type=int, default=16,
                        help="Number of independent eval episodes (vmapped) "
                             "averaged at each eval point.")
    parser.add_argument("--num_eval_periods", type=int, default=None,
                        help="Market periods per eval episode. Defaults to --horizon.")
    parser.add_argument("--eval_seed", type=int, default=12345,
                        help="Base seed for eval RNGs. Held fixed across all "
                             "eval points so noise across the learning curve "
                             "comes from the policy, not from episode init.")

    # Checkpointing / plotting
    parser.add_argument("--ckpt_dir", type=str, default=os.path.join(_HERE, "checkpoints"),
                        help="Directory to save final checkpoint. Empty string disables saving.")
    parser.add_argument("--save_best_ckpt", action="store_true",
                        help="Also save a *_best.pkl whenever eval metric (default: "
                             "social_welfare mean) reaches a new high. Final *.pkl is "
                             "still saved at end-of-training as before.")
    parser.add_argument("--save_all_checkpoints", action="store_true",
                        help="Save a checkpoint at EVERY eval point, tagged "
                             "*_step{env_step}.pkl, in addition to the final "
                             "*.pkl. The step tags line up 1:1 with the "
                             "learning-curve npz snapshots, so any point on the "
                             "curve (e.g. the low-regret 'good point' before a "
                             "divergence) can be reloaded for post-hoc analysis. "
                             "Requires --metric_log_every_env_steps > 0 (no-op "
                             "otherwise). Each file is small (~1-2 MB at N=100); "
                             "a 1.5M-step run writes ~26 per seed.")
    parser.add_argument("--best_metric", type=str, default="social_welfare",
                        help="Which eval metric (key in eval_stats) to track for "
                             "--save_best_ckpt. Default: social_welfare.")
    parser.add_argument("--plot_dir", type=str, default=os.path.join(_HERE, "plots"),
                        help="Directory to save the final learning_curve.png/.npz "
                             "(kept separate from --ckpt_dir so plots don't mix with .pkl files).")
    parser.add_argument(
        "--plot_true_matched_welfare", action="store_true",
        help="Also compute and co-plot true_matched_welfare = 2*sum(matched * <x, y>) "
             "on the SW learning-curve panel. This is the welfare under the policy's "
             "actual matching evaluated on TRUE x, y, so it is bounded above by "
             "social_welfare_planner per episode -- gives a clean apples-to-apples "
             "comparison without belief-noise inflation.",
    )
    return parser.parse_args()


class HireRLController(IPPOController):
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        config = HireRLConfig(
            Nw=args.Nw, Nf=args.Nf, d=args.d, horizon=args.horizon,
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
            # Training cycle discards env info, so the 3 DA fori_loops inside
            # compute_all_metrics are pure waste here. The separate eval_env
            # built below keeps the default True for its info-consuming path.
            compute_settled_metrics=False,
        )
        self.config = config
        self.env_fn = HireRLEnv(config)
        self.num_envs = int(args.num_envs)
        self.batched_env = RoleAssignmentBatchedEnv(
            self.env_fn, num_envs=self.num_envs, assignments=[[0], [1]]
        )

        worker_obs_space, firm_obs_space = self.batched_env.get_observation_space()
        worker_act_space, firm_act_space = self.batched_env.get_action_space()

        self.worker_model = PerCandidateActorCriticModel.build(
            worker_obs_space, worker_act_space, hidden_size=args.hidden_size, use_rnn=args.use_rnn
        )
        self.firm_model = PerCandidateActorCriticModel.build(
            firm_obs_space, firm_act_space, hidden_size=args.hidden_size, use_rnn=args.use_rnn
        )

        # --minibatch_size (absolute, in samples) takes precedence over
        # --num_minibatches when set. PPOTrainer's minibatch_num_chunks
        # (priority over num_minibatches; see jax_pbt/utils.py:split_into_minibatch)
        # is in chunks, so divide by chunk_length to convert.
        minibatch_num_chunks = (
            args.minibatch_size // args.chunk_length
            if args.minibatch_size is not None else None
        )
        common_kwargs = dict(
            feature_shared=True,
            lr=args.lr,
            grad_clip_norm=args.grad_clip_norm,
            val_loss_coef=args.val_loss_coef,
            entropy_coef=args.entropy_coef,
            gamma=args.gamma,
            gae_lam=args.gae_lam,
            ppo_epochs=args.ppo_epochs,
            ratio_clip=args.ratio_clip,
            chunk_length=args.chunk_length,
            num_minibatches=args.num_minibatches,
            minibatch_num_chunks=minibatch_num_chunks,
        )
        self.worker_trainer = PPOTrainer(actor_critic_fn=self.worker_model, **common_kwargs)
        self.firm_trainer = PPOTrainer(actor_critic_fn=self.firm_model, **common_kwargs)

        # Stateless PPOAgent objects are reused for both train rollouts (built
        # inside init_train) and eval rollouts (build_stateful_rollout_fn).
        self.worker_agent = PPOAgent(self.worker_model)
        self.firm_agent = PPOAgent(self.firm_model)

        self.train_rollout_length = int(args.rollout_length)
        self.total_env_steps = int(args.total_env_steps)

        self.rng = jax.random.key(args.seed)
        self.init_train()
        self._init_eval()
        self.init_wandb(
            name=args.run_name, entity=args.wandb_entity, project=args.wandb_project,
            group=args.wandb_group, job_type=args.wandb_job_type, config=vars(args),
        )

        # Learning-curve buffer: one entry per eval log point.
        # Each entry = {"env_step", "metrics": {key: {mean, std, ci_lo, ci_hi}},
        #               "raw": {key: ndarray(num_episodes, num_periods)}}.
        self.metric_buffer: list[dict] = []

    def get_train_env_const_step(self) -> PyTreeNode:
        return pytree_repeat_stack(self.env_fn.default_const, (self.num_envs,))

    def get_train_env_const_seq(self) -> PyTreeNode:
        return pytree_repeat_stack(self.get_train_env_const_step(), (self.train_rollout_length,))

    def init_train(self) -> None:
        env_fn = self.batched_env
        ppo_agent_lst = [PPOAgent(self.worker_model), PPOAgent(self.firm_model)]
        ppo_trainer_lst = [self.worker_trainer, self.firm_trainer]

        RunState: TypeAlias = tuple
        def init_run_state(rng):
            rng, rng_trainers = rng_batch_split(rng, len(ppo_trainer_lst))
            trainer_state_lst = [
                trainer_fn.init_trainer_state(r)
                for r, trainer_fn in zip(list(rng_trainers), ppo_trainer_lst)
            ]
            agent_state_lst = [
                agent_fn.init_agent_state(batch_shape)
                for agent_fn, batch_shape in zip(ppo_agent_lst, env_fn.get_agent_batch_shape())
            ]
            rng, rng_reset = jax.random.split(rng)
            env_state, obs_lst = env_fn.reset(rng_reset, self.get_train_env_const_step())
            return rng, trainer_state_lst, agent_state_lst, env_state, obs_lst
        self.init_run_state = init_run_state

        def ppo_training_cycle(run_state, env_const_seq):
            rng, trainer_state_lst, agent_state_lst, env_state, last_obs_lst = run_state
            model_state_lst = [
                t.model_state_from_trainer_state(s)
                for t, s in zip(ppo_trainer_lst, trainer_state_lst)
            ]

            def step(carry, env_const):
                rng, agent_state_lst, env_state, obs_lst = carry
                rng, next_agent_state_lst, next_env_state, next_obs_lst, data_lst, info = self.ppo_rollout_step(
                    env_fn, ppo_agent_lst, rng, model_state_lst, agent_state_lst,
                    env_const, env_state, obs_lst,
                )
                return (rng, next_agent_state_lst, next_env_state, next_obs_lst), (data_lst, info)
            (rng, agent_state_lst, env_state, last_obs_lst), (data_lst, info) = jax.lax.scan(
                step, (rng, agent_state_lst, env_state, last_obs_lst), env_const_seq,
                length=self.train_rollout_length,
            )

            buffer_lst = [
                PPOTransition(d["obs"], d["action"], d["reward"], d["done"], d["log_p"], d["val"])
                for d in data_lst
            ]

            last_val_lst = []
            rng, rng_val_lst = rng_batch_split(rng, len(ppo_agent_lst))
            for r, agent_fn, ms, ag, lo in zip(
                list(rng_val_lst), ppo_agent_lst, model_state_lst, agent_state_lst, last_obs_lst
            ):
                _, _, extra_data = agent_fn.step(r, ms, ag, lo)
                last_val_lst.append(extra_data["val"])

            aux_data_lst = [{"agent_state": d["agent_state"]} for d in data_lst]
            rng, rng_trainer = jax.random.split(rng)
            new_trainer_state_lst, optim_info_lst = self.ppo_update(
                ppo_trainer_lst, rng_trainer, trainer_state_lst, buffer_lst, last_val_lst, aux_data_lst,
            )

            train_info_lst = [
                {"avg_return": d["reward"].mean() * self.train_rollout_length}
                for d in data_lst
            ]

            return (
                (rng, new_trainer_state_lst, agent_state_lst, env_state, last_obs_lst),
                train_info_lst, optim_info_lst,
            )

        print("JIT-compiling PPO training cycle...")
        t0 = time.perf_counter()
        rng, trainer_state_lst, agent_state_lst, env_state, obs_lst = init_run_state(jax.random.key(0))
        env_const_seq = self.get_train_env_const_seq()
        self.training_cycle: Callable = jax.jit(ppo_training_cycle).lower(
            (rng, trainer_state_lst, agent_state_lst, env_state, obs_lst), env_const_seq,
        ).compile()
        print(f"Compiled in {time.perf_counter() - t0:.1f}s")

    # Eval metrics surfaced in the learning curve. Names map 1-1 to the keys
    # build_stateful_rollout_fn writes into the per-period history (except
    # ``social_welfare_planner``, which _run_eval injects from a host-side
    # Hungarian solve over the per-episode (x, y)).
    # ``social_welfare_da_ref`` is the worker-proposing DA reference SW (best
    # stable matching under true utilities); ``social_welfare_planner`` is the
    # first-best max-weight matching welfare -- both co-plotted with
    # ``social_welfare`` so the SW gaps are visible.
    EVAL_METRIC_KEYS = (
        "reward_w_total",
        "reward_f_total",
        "regret_w_total",
        "regret_f_total",
        "social_welfare",
        "social_welfare_da_ref",
        "social_welfare_planner",
        "friction_loss",
    )

    def _init_eval(self) -> None:
        """Build a jit+vmap'd eval rollout that takes (rngs, w_model_state,
        f_model_state) and returns a per-episode, per-period history.

        Compiled once; called periodically inside ``run()`` to produce the
        eval-only learning curve. ``--metric_log_every_env_steps <= 0``
        skips this entirely.
        """
        if self.args.metric_log_every_env_steps <= 0:
            self.eval_rollout = None
            return

        num_eval_periods = self.args.num_eval_periods or self.args.horizon
        self.num_eval_periods = int(num_eval_periods)
        self.num_eval_episodes = int(self.args.num_eval_episodes)

        # Independent eval env: matches train env config except horizon gets a
        # +10 buffer so the rollout's num_eval_periods market steps don't
        # bump into auto-reset (cf. eval_and_plot.py).
        eval_config = HireRLConfig(
            Nw=self.args.Nw, Nf=self.args.Nf, d=self.args.d,
            horizon=self.num_eval_periods + 10,
            sigma_interview=self.args.sigma_interview,
            sigma_match=self.args.sigma_match,
            lambda_reveal=self.args.lambda_reveal,
            outside_option=self.args.outside_option,
            non_negative_features=self.args.non_negative_features,
            hat_init_value=self.args.hat_init_value,
            allow_on_the_job_search=self.args.allow_on_the_job_search,
            noisy_hat_init=self.args.noisy_hat_init,
            sigma_init=self.args.sigma_init,
            public_retention_signal=self.args.public_retention_signal,
        )
        eval_env = HireRLEnv(eval_config)
        rollout_single = build_stateful_rollout_fn(
            eval_env, self.worker_agent, self.firm_agent, self.num_eval_periods,
            compute_true_matched_welfare=bool(self.args.plot_true_matched_welfare),
        )
        # Runtime metric-key list: base EVAL_METRIC_KEYS plus the optional
        # true_matched_welfare when --plot_true_matched_welfare is set.
        self.eval_metric_keys = tuple(self.EVAL_METRIC_KEYS) + (
            ("true_matched_welfare",) if self.args.plot_true_matched_welfare else ()
        )
        # vmap on the rng axis only; both model_state pytrees are shared across episodes.
        self.eval_rollout = jax.jit(jax.vmap(rollout_single, in_axes=(0, None, None)))
        # Fixed eval rngs => same set of episode initializations at every eval point,
        # so curve noise reflects policy changes rather than env randomness.
        self.eval_rngs = jax.random.split(
            jax.random.key(self.args.eval_seed), self.num_eval_episodes,
        )

        print(
            f"Eval setup: {self.num_eval_episodes} episodes x "
            f"{self.num_eval_periods} market periods, every "
            f"{self.args.metric_log_every_env_steps} env steps."
        )

    def _run_eval(self, trainer_state_lst) -> tuple[dict, dict]:
        """Run one eval rollout. Returns ``(stats, raw)``.

        * ``stats``: ``{metric_name: {mean, std, ci_lo, ci_hi}}``. ``mean`` and
          ``std`` are over the full ``(num_eval_episodes, num_eval_periods)``
          sample; ``ci_lo``/``ci_hi`` are a 95% normal-approx CI on the
          cross-episode mean (per-episode means -> 1.96 * SEM over episodes).
        * ``raw``: ``{metric_name: ndarray(num_eval_episodes, num_eval_periods)}``
          for downstream replotting.
        """
        w_ms = self.worker_trainer.model_state_from_trainer_state(trainer_state_lst[0])
        f_ms = self.firm_trainer.model_state_from_trainer_state(trainer_state_lst[1])
        history, batched_x, batched_y = self.eval_rollout(self.eval_rngs, w_ms, f_ms)
        # Block on one leaf so wall-time below reflects compute, not async dispatch.
        jax.block_until_ready(history["reward_w_total"])

        # Planner first-best welfare: per-episode constant from (x, y); broadcast
        # across periods so it matches the (num_episodes, num_periods) shape
        # contract of every other EVAL_METRIC_KEYS entry.
        planner_per_ep = compute_planner_welfare(
            np.asarray(batched_x), np.asarray(batched_y)
        )                                                          # (num_episodes,)
        history = {
            **history,
            "social_welfare_planner": np.broadcast_to(
                planner_per_ep[:, None],
                (self.num_eval_episodes, self.num_eval_periods),
            ),
        }

        stats: dict = {}
        raw: dict = {}
        for key in self.eval_metric_keys:
            arr = np.asarray(history[key])  # (num_episodes, num_periods)
            raw[key] = arr
            mean = float(arr.mean())
            ep_means = arr.mean(axis=1)  # (num_episodes,)
            n = ep_means.size
            sem = float(ep_means.std(ddof=1)) / np.sqrt(n) if n > 1 else 0.0
            half = 1.96 * sem  # 95% normal-approx CI on the cross-episode mean
            stats[key] = {
                "mean":  mean,
                "std":   float(arr.std()),
                "ci_lo": mean - half,
                "ci_hi": mean + half,
            }
        return stats, raw

    def run(self) -> None:
        rng, trainer_state_lst, agent_state_lst, env_state, obs_lst = self.init_run_state(self.rng)

        if self.args.save_all_checkpoints and self.eval_rollout is None:
            print("[warn] --save_all_checkpoints has no effect: eval is off "
                  "(--metric_log_every_env_steps <= 0), so there are no eval "
                  "points to checkpoint at.")

        env_step = 0
        last_log = 0
        last_eval = 0
        best_metric_val = -float("inf")
        best_metric_step = -1
        with tqdm(total=self.total_env_steps, desc="Train", unit="frames") as pbar:
            while env_step < self.total_env_steps:
                env_const_seq = self.get_train_env_const_seq()
                (rng, trainer_state_lst, agent_state_lst, env_state, obs_lst), train_info_lst, optim_info_lst = (
                    self.training_cycle(
                        (rng, trainer_state_lst, agent_state_lst, env_state, obs_lst),
                        env_const_seq,
                    )
                )

                steps = self.train_rollout_length * self.num_envs
                env_step += steps
                pbar.update(steps)

                if env_step - last_log > 5_000:
                    last_log = env_step
                    self.log(env_step, train_info_lst[0], log_prefix="worker")
                    self.log(env_step, train_info_lst[1], log_prefix="firm")
                    self.log(env_step, optim_info_lst[0], log_prefix="worker")
                    self.log(env_step, optim_info_lst[1], log_prefix="firm")
                    metrics = compute_all_metrics(
                        jax.tree_util.tree_map(lambda x: x[0], env_state),
                        outside=self.config.outside_option,
                    )
                    self.log(env_step, {k: float(v) for k, v in metrics.items()}, log_prefix="env")
                    tqdm.write(
                        f"step={env_step} worker_R={train_info_lst[0]['avg_return']:.3f} "
                        f"firm_R={train_info_lst[1]['avg_return']:.3f}"
                    )

                if (self.eval_rollout is not None and
                        env_step - last_eval >= self.args.metric_log_every_env_steps):
                    last_eval = env_step
                    # Every-eval checkpoint (tagged by step), so the policy at
                    # any snapshot on the learning curve can be reloaded later.
                    # Step tags match the metric_buffer / npz snapshots 1:1.
                    if self.args.save_all_checkpoints and self.args.ckpt_dir:
                        self._save_checkpoint(trainer_state_lst, suffix=f"_step{env_step}")
                    t_eval = time.perf_counter()
                    eval_stats, eval_raw = self._run_eval(trainer_state_lst)
                    eval_wall = time.perf_counter() - t_eval

                    # Flat dict for wandb. Every key gets the eval/ prefix via log_prefix.
                    flat = {
                        f"{key}_{stat}": v
                        for key, stats in eval_stats.items()
                        for stat, v in stats.items()
                    }
                    self.log(env_step, flat, log_prefix="eval")
                    self.metric_buffer.append({
                        "env_step": env_step,
                        "stats": eval_stats,
                        "raw": eval_raw,
                    })
                    tqdm.write(
                        f"  eval@{env_step}: "
                        f"R_w={eval_stats['reward_w_total']['mean']:.3f} "
                        f"R_f={eval_stats['reward_f_total']['mean']:.3f} "
                        f"reg_w={eval_stats['regret_w_total']['mean']:.3f} "
                        f"reg_f={eval_stats['regret_f_total']['mean']:.3f} "
                        f"SW={eval_stats['social_welfare']['mean']:.3f} "
                        f"(DA {eval_stats['social_welfare_da_ref']['mean']:.3f} / "
                        f"planner {eval_stats['social_welfare_planner']['mean']:.3f}) "
                        f"({eval_wall:.1f}s)"
                    )

                    # Best-eval ckpt: save *_best.pkl whenever the tracked metric
                    # reaches a new high. Independent of the end-of-training save.
                    if (self.args.save_best_ckpt and self.args.ckpt_dir
                            and self.args.best_metric in eval_stats):
                        metric_now = float(eval_stats[self.args.best_metric]["mean"])
                        if metric_now > best_metric_val:
                            best_metric_val = metric_now
                            best_metric_step = env_step
                            self._save_checkpoint(trainer_state_lst, suffix="_best")
                            tqdm.write(
                                f"  * new best {self.args.best_metric}={metric_now:.3f} "
                                f"@ step={env_step} -> saved *_best.pkl"
                            )

        # Persist final policy so eval_and_plot.py --policy ppo can load it.
        if self.args.ckpt_dir:
            self._save_checkpoint(trainer_state_lst)
        if self.args.save_best_ckpt and best_metric_step > 0:
            print(f"Best {self.args.best_metric}={best_metric_val:.3f} @ step={best_metric_step} "
                  f"(saved as {self.args.run_name}_best.pkl)")
        if self.eval_rollout is not None and self.metric_buffer:
            self._save_learning_curve()

    def _save_learning_curve(self) -> None:
        """Plot eval learning curves (mean line + 95% CI band) and dump raw history
        to npz. One figure per metric: worker / firm reward & regret, social
        welfare (policy + worker-proposing DA + first-best planner co-plotted),
        and friction loss. Each is also logged to W&B as
        ``plot/learning_curve/<slug>``.
        """
        import matplotlib.pyplot as plt

        out_dir = self.args.plot_dir
        os.makedirs(out_dir, exist_ok=True)

        steps = np.array([m["env_step"] for m in self.metric_buffer], dtype=np.int64)

        # One figure per panel. Each entry: (slug, title, [(metric_key, label, color), ...]).
        # The SW panel co-plots policy vs. worker-proposing DA vs. first-best planner;
        # when --plot_true_matched_welfare is set, also overlays the true-x,y welfare
        # of the policy's actual matching (a guaranteed lower-bound on planner).
        sw_panel_lines: list[tuple[str, str, str]] = [
            ("social_welfare",         "policy (belief)", "C1"),
            ("social_welfare_da_ref",  "DA ref",          "C2"),
            ("social_welfare_planner", "planner",         "C3"),
        ]
        if self.args.plot_true_matched_welfare:
            sw_panel_lines.append(
                ("true_matched_welfare", "policy (true x,y)", "C4")
            )
        panels: list[tuple[str, str, list[tuple[str, str, str]]]] = [
            ("worker_reward",  "Worker reward (per period)",  [("reward_w_total", "eval", "C1")]),
            ("firm_reward",    "Firm reward (per period)",    [("reward_f_total", "eval", "C1")]),
            ("worker_regret",  "Worker regret (per period)",  [("regret_w_total", "eval", "C1")]),
            ("firm_regret",    "Firm regret (per period)",    [("regret_f_total", "eval", "C1")]),
            ("social_welfare", "Social welfare (per period)", sw_panel_lines),
            ("friction_loss",  "Friction loss (per period)",  [("friction_loss", "eval", "C1")]),
        ]

        suptitle = (
            f"PPO eval learning curve "
            f"({self.num_eval_episodes} episodes x {self.num_eval_periods} periods, "
            f"every {self.args.metric_log_every_env_steps} env steps)"
        )

        wandb_run = None
        try:
            import wandb
            wandb_run = wandb.run
        except ImportError:
            pass

        for slug, title, lines in panels:
            fig, ax = plt.subplots(figsize=(6.0, 4.2))
            for key, label, color in lines:
                mean = np.array([m["stats"][key]["mean"] for m in self.metric_buffer])
                ci_lo = np.array([m["stats"][key]["ci_lo"] for m in self.metric_buffer])
                ci_hi = np.array([m["stats"][key]["ci_hi"] for m in self.metric_buffer])
                ax.plot(steps, mean, color=color, label=f"{label} mean")
                ax.fill_between(steps, ci_lo, ci_hi, alpha=0.2, color=color,
                                label=f"{label} 95% CI")
            ax.set_xlabel("env steps")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)
            fig.suptitle(suptitle, fontsize=10)
            fig.tight_layout()

            png_path = os.path.join(
                out_dir, f"{self.args.run_name}_learning_curve_{slug}.png"
            )
            fig.savefig(png_path, dpi=120, bbox_inches="tight")
            print(f"Saved {slug} learning curve to {png_path}")

            if wandb_run is not None:
                try:
                    wandb_run.log({f"plot/learning_curve/{slug}": wandb.Image(fig)})
                except Exception as e:
                    print(f"(skipping wandb image log for {slug}: {e})")
            plt.close(fig)

        # Stack raw histories into one (num_log_points, num_episodes, num_periods)
        # array per metric so downstream replots can recompute any aggregate.
        raw_arrays = {
            key: np.stack([m["raw"][key] for m in self.metric_buffer], axis=0)
            for key in self.eval_metric_keys
        }
        npz_path = os.path.join(out_dir, f"{self.args.run_name}_learning_curve.npz")
        np.savez(npz_path, env_steps=steps, **raw_arrays)
        print(f"Saved learning curve raw data to {npz_path}")

    def _save_checkpoint(self, trainer_state_lst, suffix: str = "") -> None:
        import pickle
        from flax import serialization
        os.makedirs(self.args.ckpt_dir, exist_ok=True)
        path = os.path.join(self.args.ckpt_dir, f"{self.args.run_name}{suffix}.pkl")
        data = {
            "worker_trainer_state": serialization.to_bytes(trainer_state_lst[0]),
            "firm_trainer_state":   serialization.to_bytes(trainer_state_lst[1]),
            "model_args": {
                "hidden_size": self.args.hidden_size,
                "use_rnn": self.args.use_rnn,
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
            },
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved PPO checkpoint to {path}")


def main() -> None:
    args = get_args()
    HireRLController(args).run()


if __name__ == "__main__":
    main()
