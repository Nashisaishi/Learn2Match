"""Limit test under a trained PPO policy.

Math claim (independent of policy): when sigma_match=0 and lambda_reveal is
large enough that sigmoid(lambda * tenure) saturates to 1.0 in float32, every
retained pair (i, j) satisfies hat_x[i,j] == x[i] and hat_y[i,j] == y[j]
exactly, because each retention overwrites belief with `1.0 * x_i + 0`.

This script drives the env with a PPO policy loaded from a train_hirerl_ippo.py
checkpoint, runs ``--num_periods`` market periods under ``--num_seeds``
independent rollouts (vmap+scan), then checks the math claim on every pair
that PPO actually retained at least once across all seeds.

Two tests are run:
    1. sigma_match=0.0  -> belief recovery should be exact (atol=1e-5)
    2. sigma_match=0.1  -> negative control; residual noise must be > 0.03

``sigma_match`` is intentionally NOT a CLI flag because the math claims fix
those values; everything else (Nw, Nf, d, num_periods, sigma_interview,
lambda_reveal, outside_option, non_negative_features, seed, num_seeds) is
configurable.

Run:
    python test_retention_limit.py \\
        --ppo_ckpt examples/checkpoints/<run_name>.pkl \\
        --Nw 5 --Nf 5 --d 3 --num_periods 10000 \\
        --sigma_interview 0.0003 --lambda_reveal 40.0 \\
        --non_negative_features --outside_option=-1e9 \\
        --num_seeds 32 --seed 0
"""

import argparse
import os
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))   # repo root for jax_pbt
sys.path.insert(0, os.path.join(_HERE, ".."))           # learn2match for hirerl

from jax_pbt.env.spaces import Action

from hirerl import HireRLConfig, HireRLEnv

# Reuse the (refactored) checkpoint-loading and STEPS_PER_PERIOD constant.
from eval_and_plot import STEPS_PER_PERIOD, make_ppo_policy


def _build_env(args: argparse.Namespace, sigma_match: float) -> HireRLEnv:
    """Build a HireRLEnv with CLI-configured params plus the test-specific
    ``sigma_match`` override (the math claims fix sigma_match per test).

    Horizon is set to ``num_periods + 10`` so the env's auto-reset never fires
    mid-rollout (env.py:260 gates done on `market_t >= horizon`).
    """
    return HireRLEnv(HireRLConfig(
        Nw=args.Nw, Nf=args.Nf, d=args.d,
        horizon=args.num_periods + 10,
        sigma_interview=args.sigma_interview,
        sigma_match=sigma_match,
        lambda_reveal=args.lambda_reveal,
        outside_option=args.outside_option,
        non_negative_features=args.non_negative_features,
    ))


def _build_final_state_rollout(env, policy_step, init_agent_state, num_periods):
    """Build a pure ``rollout(rng) -> (final_state, x_init, y_init)``.

    Unlike eval_and_plot.build_rollout_fn, we don't collect per-period
    history; the math tests only need the post-rollout state. So the inner
    body is a single flat scan over ``num_periods * STEPS_PER_PERIOD`` phase
    steps that throws away rewards/info and returns just the carried env
    state (and the initial x, y so callers can compare against ground truth).
    """
    const = env.default_const

    def step_phase(carry, _):
        rng, env_state, obs_lst, agent_state = carry
        rng, sub = jax.random.split(rng)
        (w_choice, f_choice), next_agent_state, rng = policy_step(rng, obs_lst, agent_state)
        action = [
            Action({"choice": jnp.asarray(w_choice, dtype=jnp.int32)}),
            Action({"choice": jnp.asarray(f_choice, dtype=jnp.int32)}),
        ]
        next_env_state, next_obs_lst, _, _, _ = env.step(sub, const, env_state, action)
        return (rng, next_env_state, next_obs_lst, next_agent_state), None

    def rollout(rng):
        rng, rng_reset = jax.random.split(rng)
        env_state, obs_lst = env.reset(rng_reset, const)
        x_init = env_state.x
        y_init = env_state.y
        agent_state = init_agent_state()
        init_carry = (rng, env_state, obs_lst, agent_state)
        final_carry, _ = jax.lax.scan(
            step_phase, init_carry, xs=None, length=num_periods * STEPS_PER_PERIOD,
        )
        _, final_state, _, _ = final_carry
        return final_state, x_init, y_init

    return rollout


def _run_episodes_under_ppo(env, ckpt_path, num_periods, base_seed, num_seeds):
    """Drive the env with a PPO policy across ``num_seeds`` parallel rollouts
    via ``jax.vmap`` + ``jax.lax.scan``, return ``(final_state, x_init, y_init)``
    where every leaf has a leading seed axis.
    """
    policy_step, init_agent_state = make_ppo_policy(env, ckpt_path)
    rollout_fn = _build_final_state_rollout(env, policy_step, init_agent_state, num_periods)
    rollout_batched = jax.jit(jax.vmap(rollout_fn))

    seed_rngs = jax.random.split(jax.random.key(base_seed), num_seeds)
    t0 = time.perf_counter()
    final_state, x_init, y_init = rollout_batched(seed_rngs)
    jax.block_until_ready(final_state.cumulative_tenure)
    elapsed = time.perf_counter() - t0
    print(
        f"  rollout: {num_seeds} seeds x {num_periods} periods in {elapsed:.1f}s "
        f"({num_seeds * num_periods / max(elapsed, 1e-9):.0f} seed-periods/sec)"
    )
    return final_state, np.asarray(x_init), np.asarray(y_init)


def test_retention_recovers_truth_under_ppo(args: argparse.Namespace) -> None:
    print(f"[1/2] sigma_match=0.0 (exact recovery limit)")
    env = _build_env(args, sigma_match=0.0)
    final_state, x_init_b, y_init_b = _run_episodes_under_ppo(
        env, args.ppo_ckpt, args.num_periods, args.seed, args.num_seeds,
    )

    # Under env semantics (transitions.py:184-191) interview-after-retention
    # does NOT overwrite hat_x, so cumulative_tenure > 0 cleanly selects pairs
    # whose hat_x is a retention update — currently matched or now dissolved.
    cum_ten_b = np.asarray(final_state.cumulative_tenure)   # (N, Nw, Nf)
    hat_x_b   = np.asarray(final_state.hat_x)               # (N, Nw, Nf, d)
    hat_y_b   = np.asarray(final_state.hat_y)               # (N, Nw, Nf, d)

    total_pairs = 0
    max_err_x, max_err_y = 0.0, 0.0
    for s in range(args.num_seeds):
        pairs = np.argwhere(cum_ten_b[s] > 0)
        if len(pairs) == 0:
            raise AssertionError(
                f"PPO policy never retained any pair on seed offset {s} "
                f"(base_seed={args.seed}). Train PPO longer or pick a "
                "different checkpoint."
            )
        for i, j in pairs:
            assert np.allclose(hat_x_b[s, i, j], x_init_b[s, i], atol=1e-5), (
                f"seed_offset={s}: hat_x[{i},{j}]={hat_x_b[s, i, j]} "
                f"expected {x_init_b[s, i]} (tenure={cum_ten_b[s, i, j]})"
            )
            assert np.allclose(hat_y_b[s, i, j], y_init_b[s, j], atol=1e-5), (
                f"seed_offset={s}: hat_y[{i},{j}]={hat_y_b[s, i, j]} "
                f"expected {y_init_b[s, j]} (tenure={cum_ten_b[s, i, j]})"
            )
            max_err_x = max(max_err_x, float(np.max(np.abs(hat_x_b[s, i, j] - x_init_b[s, i]))))
            max_err_y = max(max_err_y, float(np.max(np.abs(hat_y_b[s, i, j] - y_init_b[s, j]))))
            total_pairs += 1

    print(
        f"[OK] limit holds across {total_pairs} ever-retained pairs "
        f"({args.num_seeds} seeds x {args.num_periods} periods); "
        f"max |hat_x - x| = {max_err_x:.2e}, max |hat_y - y| = {max_err_y:.2e}"
    )


def test_negative_control_match_noise_breaks_recovery_under_ppo(args: argparse.Namespace) -> None:
    print(f"[2/2] sigma_match=0.1 (negative control, residual noise expected)")
    env = _build_env(args, sigma_match=0.1)
    final_state, x_init_b, _ = _run_episodes_under_ppo(
        env, args.ppo_ckpt, args.num_periods, args.seed, args.num_seeds,
    )
    cum_ten_b = np.asarray(final_state.cumulative_tenure)
    hat_x_b   = np.asarray(final_state.hat_x)

    total_pairs = 0
    all_diffs = []
    for s in range(args.num_seeds):
        pairs = np.argwhere(cum_ten_b[s] > 0)
        if len(pairs) == 0:
            raise AssertionError(
                f"PPO policy never retained any pair on seed offset {s} "
                f"(base_seed={args.seed})."
            )
        diffs = np.array([
            np.linalg.norm(hat_x_b[s, i, j] - x_init_b[s, i]) for i, j in pairs
        ])
        # E[||eps||] for d=args.d, sigma=0.1 is ~0.1 * sqrt(d); threshold 0.03
        # is below the lower tail for d >= 1. (For d=4, mean ~0.18.)
        assert (diffs > 0.03).all(), (
            f"seed_offset={s}: sigma_match=0.1 should leave residual noise on "
            f"every retained pair; got diffs={diffs}"
        )
        all_diffs.extend(diffs.tolist())
        total_pairs += len(pairs)

    print(
        f"[OK] negative control: mean ||hat_x - x|| over {total_pairs} retained pairs "
        f"({args.num_seeds} seeds x {args.num_periods} periods) = {np.mean(all_diffs):.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo_ckpt", type=str, required=True,
                        help="Path to a checkpoint from train_hirerl_ippo.py "
                             "(must have matching Nw/Nf/d).")
    # Env structural params (must match the training ckpt).
    parser.add_argument("--Nw", type=int, default=10)
    parser.add_argument("--Nf", type=int, default=10)
    parser.add_argument("--d", type=int, default=4)
    # Rollout / env continuous params.
    parser.add_argument("--num_periods", type=int, default=100,
                        help="Market periods to roll out before checking the "
                             "post-state. More periods give more retained "
                             "pairs (stronger test).")
    parser.add_argument("--sigma_interview", type=float, default=0.5,
                        help="Does NOT affect the math claim. Default matches "
                             "the original test setup.")
    parser.add_argument("--lambda_reveal", type=float, default=50.0,
                        help="Must be large enough that sigmoid(lambda * tenure) "
                             "saturates to 1.0 in float32 (i.e. >= ~17). "
                             "lambda=40 is fine; lambda=1.0 will fail test 1.")
    parser.add_argument("--outside_option", type=float, default=0.0)
    parser.add_argument("--non_negative_features", action="store_true")
    # Multi-seed setup.
    parser.add_argument("--num_seeds", type=int, default=1,
                        help="Run each test on ``num_seeds`` parallel rollouts "
                             "via vmap. Test passes only if the math claim "
                             "holds on EVERY seed.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base PRNG seed; per-seed rollouts use distinct "
                             "sub-keys split from this.")
    args = parser.parse_args()

    if not os.path.exists(args.ppo_ckpt):
        raise SystemExit(f"checkpoint not found: {args.ppo_ckpt}")

    print(
        f"Running retention-limit tests with Nw={args.Nw}, Nf={args.Nf}, "
        f"d={args.d}, num_periods={args.num_periods}, num_seeds={args.num_seeds}, "
        f"lambda_reveal={args.lambda_reveal}, outside_option={args.outside_option}, "
        f"non_negative_features={args.non_negative_features}"
    )

    tests = [
        test_retention_recovers_truth_under_ppo,
        test_negative_control_match_noise_breaks_recovery_under_ppo,
    ]
    failed = 0
    for fn in tests:
        try:
            fn(args)
        except Exception:
            failed += 1
            print(f"[FAIL] {fn.__name__}")
            traceback.print_exc()
    if failed == 0:
        print(f"\n{len(tests)} tests passed.")
    else:
        print(f"\n{failed}/{len(tests)} tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
