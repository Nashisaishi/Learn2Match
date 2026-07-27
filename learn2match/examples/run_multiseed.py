"""Multi-seed train + eval orchestrator for paired training-variance CI.

Pipeline
--------
1. ``--num_train_seeds`` PPO trainings via ``train_hirerl_ippo.py``, each with a
   distinct ``--seed`` and a unique ``--run_name`` (``{base_run_name}_s{i}``)
   so checkpoints land in distinct ``.pkl`` files under ``--ckpt_dir``.
2. ``--num_train_seeds`` evals via ``eval_and_plot.py``, each loading the
   matching ckpt and writing into ``{eval_root}/seed{i}``. All evals share
   ``--seed = --eval_seed`` (paired design): every trained policy sees the
   exact same ``--num_eval_envs`` env realizations, so the across-train-seed
   CI in the downstream plot reflects pure training variance, not env
   sampling noise.

Concurrency
-----------
Round-robin across ``--gpus`` (default ``"0,1"``). At any moment at most
``len(gpus)`` subprocesses run, one per GPU; finished jobs free their slot
for the next pending seed.

Logs
----
Each subprocess's stdout/stderr is tee'd to a per-seed log file under
``{eval_root}/logs/{train|eval}_s{i}.log``.

Skip / resume
-------------
* Training stage skips a seed whose ckpt file already exists (unless
  ``--force_train`` is set).
* Eval stage skips a seed whose ``cumulative_regret_welfare.csv`` already
  exists in its out_dir (unless ``--force_eval`` is set).

Aggregation
-----------
This script does NOT aggregate across train seeds. Run
``plot_multiseed_ci.py`` afterward, pointing it at ``--eval_root``.

Example
-------
    python learn2match/examples/run_multiseed.py \\
        --num_train_seeds 10 --gpus 0,1 \\
        --base_run_name ppo_multiseed_may6 \\
        --eval_root /home/hisaishi/hireRL/eval_runs/ppo_multiseed_may6 \\
        --Nw 5 --Nf 5 --d 3 --num_periods 200 \\
        --sigma_interview 1 --sigma_match 0.6 --lambda_reveal 1 \\
        --non_negative_features --outside_option=-1e9 \\
        --use_rnn --entropy_coef 0.05 \\
        --total_env_steps 1000000 --num_envs 128
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_TRAIN_SCRIPT = _HERE / "train_hirerl_ippo.py"
_EVAL_SCRIPT = _HERE / "eval_and_plot.py"


def _build_train_cmd(args: argparse.Namespace, seed: int, run_name: str) -> List[str]:
    cmd = [
        sys.executable, str(_TRAIN_SCRIPT),
        "--Nw", str(args.Nw),
        "--Nf", str(args.Nf),
        "--d", str(args.d),
        "--horizon", str(args.num_periods),
        "--sigma_interview", str(args.sigma_interview),
        "--sigma_match", str(args.sigma_match),
        "--lambda_reveal", str(args.lambda_reveal),
        f"--outside_option={args.outside_option}",
        "--interview_mode", args.interview_mode,
        "--num_envs", str(args.num_envs),
        "--total_env_steps", str(args.total_env_steps),
        "--metric_log_every_env_steps", str(args.metric_log_every_env_steps),
        "--num_eval_episodes", str(args.num_eval_episodes),
        *( ["--num_eval_periods", str(args.num_eval_periods)]
           if args.num_eval_periods is not None else [] ),
        *( ["--num_minibatches", str(args.num_minibatches)]
           if args.num_minibatches is not None else [] ),
        *( ["--minibatch_size", str(args.minibatch_size)]
           if args.minibatch_size is not None else [] ),
        "--entropy_coef", str(args.entropy_coef),
        "--hidden_size", str(args.hidden_size),
        "--hat_init_value", str(args.hat_init_value),
        "--sigma_init", str(args.sigma_init),
        "--ckpt_dir", str(args.ckpt_dir),
        "--seed", str(seed),
        "--run_name", run_name,
    ]
    if args.non_negative_features:
        cmd.append("--non_negative_features")
    if args.use_rnn:
        cmd.append("--use_rnn")
    if args.allow_on_the_job_search:
        cmd.append("--allow_on_the_job_search")
    if args.noisy_hat_init:
        cmd.append("--noisy_hat_init")
    if args.public_retention_signal:
        cmd.append("--public_retention_signal")
    if not args.no_wandb_train:
        cmd += [
            "--wandb_entity", args.wandb_entity,
            "--wandb_project", args.wandb_project,
            "--wandb_group", args.wandb_group or args.base_run_name,
            "--wandb_job_type", "train",
        ]
    return cmd


def _build_eval_cmd(
    args: argparse.Namespace,
    ckpt_path: Path,
    out_dir: Path,
    run_name: str,
) -> List[str]:
    cmd = [
        sys.executable, str(_EVAL_SCRIPT),
        "--ppo_ckpt", str(ckpt_path),
        "--Nw", str(args.Nw),
        "--Nf", str(args.Nf),
        "--d", str(args.d),
        "--num_periods", str(args.num_periods),
        "--sigma_interview", str(args.sigma_interview),
        "--sigma_match", str(args.sigma_match),
        "--lambda_reveal", str(args.lambda_reveal),
        f"--outside_option={args.outside_option}",
        "--interview_mode", args.interview_mode,
        "--hat_init_value", str(args.hat_init_value),
        "--sigma_init", str(args.sigma_init),
        "--num_seeds", str(args.num_eval_envs),
        "--seed", str(args.eval_seed),
        "--run_name", run_name,
        "--out_dir", str(out_dir),
        "--no_wandb",
    ]
    if args.non_negative_features:
        cmd.append("--non_negative_features")
    if args.allow_on_the_job_search:
        cmd.append("--allow_on_the_job_search")
    if args.noisy_hat_init:
        cmd.append("--noisy_hat_init")
    if args.public_retention_signal:
        cmd.append("--public_retention_signal")
    return cmd


def _run_pool(
    jobs: List[Tuple[List[str], str, str]],
    gpus: List[str],
    log_dir: Path,
) -> List[Tuple[str, int]]:
    """Round-robin scheduler: at most ``len(gpus)`` subprocesses run concurrently.

    Each entry in ``jobs`` is ``(cmd_argv, log_basename, label)``. Returns a list
    of ``(label, returncode)`` in completion order.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    pending: List[Tuple[List[str], str, str]] = list(jobs)
    running: dict = {}
    results: List[Tuple[str, int]] = []

    while pending or running:
        for gpu in gpus:
            if gpu in running or not pending:
                continue
            cmd, log_name, label = pending.pop(0)
            log_path = log_dir / log_name
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env.setdefault("JAX_PLATFORMS", "cuda")
            log_fh = log_path.open("w", buffering=1)
            stamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{stamp}] [GPU {gpu}] launch  {label}  -> log={log_path}")
            print(f"           $ {' '.join(shlex.quote(c) for c in cmd)}")
            proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env)
            running[gpu] = (proc, label, log_fh, time.time())

        finished = []
        for gpu, (proc, label, log_fh, t0) in running.items():
            rc = proc.poll()
            if rc is not None:
                log_fh.close()
                dt = time.time() - t0
                stamp = datetime.now().strftime("%H:%M:%S")
                tag = "OK" if rc == 0 else f"FAIL(rc={rc})"
                print(f"[{stamp}] [GPU {gpu}] {tag:<10} {label}  ({dt/60:.1f} min)")
                results.append((label, rc))
                finished.append(gpu)
        for gpu in finished:
            del running[gpu]

        if running:
            time.sleep(2)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Orchestration
    parser.add_argument("--num_train_seeds", type=int, default=10,
                        help="Number of independent training runs (each one ckpt).")
    parser.add_argument("--gpus", type=str, default="0,1",
                        help="Comma-separated CUDA device IDs to round-robin across.")
    parser.add_argument("--base_run_name", type=str, required=True,
                        help="Prefix for run_name; checkpoints land at "
                             "{ckpt_dir}/{base_run_name}_s{i}.pkl.")
    parser.add_argument("--ckpt_dir", type=str,
                        default=str(_HERE / "checkpoints"),
                        help="Where train_hirerl_ippo.py writes ckpts.")
    parser.add_argument("--eval_root", type=str, required=True,
                        help="Root dir for eval outputs; per-seed subdirs are "
                             "{eval_root}/seed{i}.")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Per-subprocess log dir. Defaults to "
                             "{eval_root}/logs.")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip the training stage entirely.")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip the eval stage entirely.")
    parser.add_argument("--force_train", action="store_true",
                        help="Re-train even if ckpt already exists.")
    parser.add_argument("--force_eval", action="store_true",
                        help="Re-eval even if cumulative_regret_welfare.csv "
                             "already exists.")

    # Env config (shared between train and eval; eval is paired so no
    # per-seed env-randomness flag here -- eval seed is fixed via --eval_seed).
    parser.add_argument("--Nw", type=int, default=5)
    parser.add_argument("--Nf", type=int, default=5)
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--num_periods", type=int, default=200,
                        help="Drives BOTH training --horizon AND the final "
                             "eval_and_plot.py --num_periods. These should match so "
                             "post-training eval is distribution-faithful to training. "
                             "For the in-training periodic eval (called every "
                             "--metric_log_every_env_steps), use --num_eval_periods to "
                             "set a separate (typically shorter) horizon.")
    parser.add_argument("--sigma_interview", type=float, default=1.0)
    parser.add_argument("--sigma_match", type=float, default=0.6)
    parser.add_argument("--lambda_reveal", type=float, default=1.0)
    parser.add_argument("--outside_option", type=float, default=-1e9)
    parser.add_argument(
        "--interview_mode", type=str, default="exclusive_role",
        choices=["exclusive_role", "capacity_limited"],
        help="Interview matching protocol; forwarded to BOTH the training "
             "subprocesses and the final eval_and_plot.py so the eval env "
             "always matches the training env.")
    parser.add_argument("--non_negative_features", action="store_true")
    parser.add_argument("--allow_on_the_job_search", action="store_true")
    parser.add_argument("--noisy_hat_init", action="store_true")
    parser.add_argument("--public_retention_signal", action="store_true")
    parser.add_argument("--hat_init_value", type=float, default=0.0)
    parser.add_argument("--sigma_init", type=float, default=3.0)

    # Train-only knobs (defaults mirror running_commands' ppo_may6_1).
    parser.add_argument("--total_env_steps", type=int, default=1_000_000)
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--metric_log_every_env_steps", type=int, default=50_000)
    parser.add_argument("--num_eval_episodes", type=int, default=16)
    parser.add_argument(
        "--num_minibatches", type=int, default=None,
        help="PPO minibatches per epoch. Default None falls back to "
             "train_hirerl_ippo.py's default (4). Increase to shrink "
             "PPO update activation memory at the cost of more SGD steps "
             "per cycle. Note that batch_size = buffer / num_minibatches is "
             "buffer-relative -- if you want an *absolute* batch size that "
             "stays constant when scaling Nw/num_envs, use --minibatch_size "
             "instead (it takes precedence).")
    parser.add_argument(
        "--minibatch_size", type=int, default=None,
        help="Absolute PPO minibatch size in samples (overrides "
             "--num_minibatches when set). Decouples PPO update activation "
             "memory from buffer size, so the same value can be reused "
             "across different Nw/Nf/num_envs configs. Rounds DOWN to a "
             "multiple of chunk_length=16 internally. Typical safe values "
             "on a 24 GB GPU at Nw=Nf=200: 10000-50000 samples.")
    parser.add_argument(
        "--num_eval_periods", type=int, default=None,
        help="Episode length for the in-training periodic eval (logged every "
             "--metric_log_every_env_steps). Default None falls back to "
             "--num_periods (matching legacy behavior). Cost per in-training "
             "eval scales linearly in this value, so for long-horizon training "
             "runs (e.g. --num_periods 10000) setting --num_eval_periods 200 "
             "drops periodic-eval cost ~50x without meaningfully degrading the "
             "learning-curve trend signal. The final eval_and_plot.py run is "
             "unaffected -- it still uses --num_periods for distribution-"
             "faithful paper-table numbers.")
    parser.add_argument("--entropy_coef", type=float, default=0.05)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--use_rnn", action="store_true")

    # Eval-only knobs.
    parser.add_argument("--num_eval_envs", type=int, default=32,
                        help="--num_seeds passed to eval_and_plot.py "
                             "(parallel env count per eval).")
    parser.add_argument("--eval_seed", type=int, default=0,
                        help="Eval base seed shared by ALL train seeds (paired "
                             "design). Don't change unless you want unpaired "
                             "evals.")

    # W&B for training subprocesses (eval is always --no_wandb here).
    parser.add_argument("--wandb_entity", default="haijingzong-university-of-washington")
    parser.add_argument("--wandb_project", default="hireRL")
    parser.add_argument("--wandb_group", default=None,
                        help="Defaults to --base_run_name.")
    parser.add_argument("--no_wandb_train", action="store_true",
                        help="Disable W&B for the training subprocesses.")

    args = parser.parse_args()

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise SystemExit("--gpus must list at least one device")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_root = Path(args.eval_root)
    eval_root.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir) if args.log_dir else eval_root / "logs"

    print("=" * 72)
    print(f"Multi-seed orchestrator | base={args.base_run_name} | "
          f"N={args.num_train_seeds} | gpus={gpus}")
    print(f"  ckpt_dir  = {ckpt_dir}")
    print(f"  eval_root = {eval_root}")
    print(f"  log_dir   = {log_dir}")
    print("=" * 72)

    # ------------------------------------------------------------ train
    if args.skip_train:
        print("[skip] training stage skipped (--skip_train)")
    else:
        train_jobs: List[Tuple[List[str], str, str]] = []
        for i in range(args.num_train_seeds):
            run_name = f"{args.base_run_name}_s{i}"
            ckpt_path = ckpt_dir / f"{run_name}.pkl"
            if ckpt_path.exists() and not args.force_train:
                print(f"[skip] train s{i}: ckpt exists -> {ckpt_path}")
                continue
            cmd = _build_train_cmd(args, seed=i, run_name=run_name)
            train_jobs.append((cmd, f"train_s{i}.log", f"train s{i}"))
        if train_jobs:
            print(f"\n[train] launching {len(train_jobs)} job(s) "
                  f"across {len(gpus)} GPU(s)...")
            results = _run_pool(train_jobs, gpus, log_dir)
            failures = [lbl for lbl, rc in results if rc != 0]
            if failures:
                print(f"\n[train] {len(failures)} failure(s): {failures}")
                print("        (eval stage will skip seeds with no ckpt.)")
        else:
            print("[train] nothing to do (all ckpts exist).")

    # ------------------------------------------------------------ eval
    if args.skip_eval:
        print("[skip] eval stage skipped (--skip_eval)")
    else:
        eval_jobs: List[Tuple[List[str], str, str]] = []
        for i in range(args.num_train_seeds):
            run_name = f"{args.base_run_name}_s{i}"
            ckpt_path = ckpt_dir / f"{run_name}.pkl"
            out_dir = eval_root / f"seed{i}"
            if not ckpt_path.exists():
                print(f"[skip] eval s{i}: ckpt missing -> {ckpt_path}")
                continue
            done_marker = out_dir / "cumulative_regret_welfare.csv"
            if done_marker.exists() and not args.force_eval:
                print(f"[skip] eval s{i}: out_dir already populated -> {out_dir}")
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = _build_eval_cmd(
                args,
                ckpt_path=ckpt_path,
                out_dir=out_dir,
                run_name=f"{run_name}_eval",
            )
            eval_jobs.append((cmd, f"eval_s{i}.log", f"eval s{i}"))
        if eval_jobs:
            print(f"\n[eval] launching {len(eval_jobs)} job(s) "
                  f"across {len(gpus)} GPU(s)...")
            results = _run_pool(eval_jobs, gpus, log_dir)
            failures = [lbl for lbl, rc in results if rc != 0]
            if failures:
                print(f"\n[eval] {len(failures)} failure(s): {failures}")
        else:
            print("[eval] nothing to do (all out_dirs already populated).")

    print("\nDone. Next step:")
    print(
        f"    python learn2match/examples/plot_multiseed_ci.py \\\n"
        f"        --ppo_eval_root {eval_root} \\\n"
        f"        --num_seeds {args.num_train_seeds} \\\n"
        f"        --out_dir {eval_root}/aggregated"
    )


if __name__ == "__main__":
    main()
