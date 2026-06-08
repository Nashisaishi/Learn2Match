# learn2match

A JAX-based reinforcement-learning environment and training pipeline for **dynamic two-sided matching markets** (workers and firms repeatedly interview, match, and retain over a finite horizon), together with a CA-ETC baseline and the scripts that reproduce the paper figures.

## Layout

```
learn2match/
├── hirerl/                  # JAX env: phases (interview/match/retention), masks, metrics, model
├── examples/                # PPO training, evaluation, multi-seed orchestration, plotting
├── ca-etc-hirerl/           # CA-ETC baseline (Pagare & Ghosh, 2024) driven through the HireRL phase protocol
├── graph_noise/             # Paper figures: small market, with noise        (PDF + CSV + regen)
├── graph_no_noise/          # Paper figures: small market, no noise          (PDF + CSV + regen)
├── large_market/            # Paper figures: large market, with noise        (PDF + CSV + regen)
├── learning_curve/          # Paper figures: in-training learning curves     (PDF + CSV + regen)
└── paper graph command      # Shell commands that regenerate every paper figure
```

## Dependencies

- Python ≥ 3.10. Other deps are listed in [`requirements.txt`](../requirements.txt) at the repo root.
- **`jax_pbt/`** must live at the repo root, as a sibling of `learn2match/`. The environment imports `jax_pbt.env.*`, `jax_pbt.controller.ippo_controller`, etc. (Already included if you clone this repo.)

## Install

```bash
git clone https://github.com/Nashisaishi/Learn2Match.git
cd Learn2Match
pip install -r requirements.txt
# For GPU training, additionally install the matching JAX CUDA wheel — see the comment in requirements.txt.
```

## Quick start

All commands run from the repo root (`Learn2Match/`).

**1. Train a PPO policy.** Writes a checkpoint to `learn2match/examples/checkpoints/<run_name>.pkl`:
```bash
python learn2match/examples/train_hirerl_ippo.py \
    --Nw 5 --Nf 5 --d 3 --horizon 200 \
    --num_envs 128 --total_env_steps 1000000 \
    --seed 0 --run_name my_run
```

**2. Evaluate that checkpoint.** Requires the `.pkl` from step 1:
```bash
python learn2match/examples/eval_and_plot.py \
    --ppo_ckpt learn2match/examples/checkpoints/my_run.pkl \
    --Nw 5 --Nf 5 --d 3 --num_periods 200 \
    --num_seeds 32 --run_name my_run_eval
```

**Independent: run the CA-ETC baseline (vmapped over seeds):**
```bash
python learn2match/ca-etc-hirerl/plot_batched_ca_etc_baseline.py --help
```

**Reproduce the paper figures:** see [`paper graph command`](paper%20graph%20command) for the full three-group pipeline (small market with/without noise + large market with noise). Each group runs (1) CA-ETC baseline → (2) multi-seed PPO train+eval → (3) aggregated CI / learning-curve plots.

## Outputs

- `examples/checkpoints/` — PPO checkpoints (`<run_name>.pkl`)
- `examples/plots/` — per-run figures and CSVs written during training/eval
- `graph_*/`, `large_market/`, `learning_curve/` — finalized paper figures (PDFs + the CSVs they were rendered from + a regenerator script that turns the CSVs into PDFs)

## References

The CA-ETC baseline in `ca-etc-hirerl/` is a re-implementation of the algorithm from:

> Pagare, T. and Ghosh, A., 2024, July. Explore-then-commit algorithms for decentralized two-sided matching markets. In *2024 IEEE International Symposium on Information Theory (ISIT)* (pp. 2092–2097). IEEE.
