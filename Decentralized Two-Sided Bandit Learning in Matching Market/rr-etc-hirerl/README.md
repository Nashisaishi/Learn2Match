# Oracle-Assisted Round-Robin ETC on HireRL

A scripted baseline that implements the player-side algorithm of
**Zhang & Fang, "Decentralized Two-Sided Bandit Learning in Matching Market"
(Round-Robin ETC)** on the `learn2match/hirerl` benchmark. One bandit time
step = one full HireRL market period (5 phase-level `env.step` calls); every
pull, rejection and commitment is realized through legal env actions only.

## Algorithm (per round on the remaining market)

1. **Round-robin exploration** — index-staggered, conflict-free: every
   remaining worker interviews + match-proposes each available firm
   `ceil(L * log2(horizon))` times per round. Each successful (tentative)
   match yields one fresh sample per side (`x_i·hat_y[i,j]` / `hat_x[i,j]·y_j`);
   the RETENTION phase releases everything, so `cumulative_tenure` stays 0 and
   interview noise is fresh i.i.d. every visit.
2. **Confidence test** (worker-local): full-ranking recovery over available
   firms — in mean-sorted order every adjacent pair must satisfy
   `UCB_next − LCB_cur < eps_w`, with the calibrated radius
   `c_radius * ||x_i|| * sigma_interview * sqrt(2 ln(horizon) / n)`.
3. **Arm-readiness gate** (oracle): commits are blocked until every available
   firm's *empirical* ranking over the remaining workers agrees with its
   *true* ranking on all `eps_f`-significant pairs. Firm means are frozen
   between the gate and the GS that follows (no fresh samples outside
   exploration), so gate-pass ⇒ every firm decision inside that GS is correct.
4. **Oracle safe set**: influence edge `p → q` iff some available firm truly
   prefers p over q; `J = Reach(unconfident)`, `I = remaining \ J` may commit.
5. **Physical decentralized GS**: all remaining workers run worker-proposing
   deferred acceptance through real market periods (propose by own empirical
   ranking; firms accept the empirical-best proposer; rejection observed via
   `tentative_matched`). Terminates at the first rejection-free period
   (≤ N_r·K_r + 1 periods).
6. **Commit & shrink**: only `I` locks its GS firm (permanent retention);
   their firms leave the available set; the checkpoint loop repeats
   immediately on the shrunken market (cascade commits need no new samples),
   otherwise the next round explores again.

Settled pairs are maintained every period (establish once via match phases,
then retain); the env's `matched` guards automatically reject any proposal to
an occupied firm — the paper's "occupied arm" semantics for free.

## Deviations from the paper (all documented, deliberate)

| # | Change | Why |
|---|--------|-----|
| 1 | Oracle safe-set replaces deliberate-conflict COMM | Computes the *same* predicate the protocol certifies (Lemma 7), at zero period cost; removes the O(N²K) communication block that is infeasible beyond small markets |
| 2 | **Arm-readiness gate (addition)** | The paper's `D·Δᵃ ≥ Δ` assumption guarantees firms are learned by the time workers are confident; HireRL provides no such guarantee. Without the gate, a firm that misranks two workers at GS time locks a wrong match forever (ablation below: 3/10 runs) |
| 3 | Indices oracle-assigned; available set updated directly | Pure coordination shortcuts (no preference information); replaces INDEX-ASSIGNMENT and occupied-arm probing |
| 4 | Firms play empirical-leader | The paper's rational-condition example strategy |
| 5 | Calibrated CI radius `‖x_i‖·σ_interview` | Exact sub-gaussian constant for HireRL's Gaussian interview noise; the paper's generic constant assumes 1-subgaussian rewards |
| 6 | Exploration length knob `L` (default 1.0; paper-faithful `K²`) | The K² factor only sets checkpoint granularity — total exploration is governed by the confidence test (the paper's round bound carries a cancelling 1/K²) |
| 7 | `eps_w` / `eps_f` tolerances | Random `x·y` markets have no gap floor; ε > 0 bounds the per-comparison utility loss where exact ranking would need unbounded samples. ε = 0 (default) is the faithful exact rule |

Estimation discipline: workers update only from exploration samples; firms
likewise (only periods with fresh interview noise — GS-period matches carry
no new information in HireRL and would inject duplicate samples).

Note on preferences: HireRL utilities are shared (`U[i,j] = x_i·y_j` for both
sides), so the stable matching is unique (assortative). The algorithm does not
use this fact, but it makes `reference_match` an unambiguous ground truth.

## Files

- `rr_etc_baseline.py` — `RoundRobinETC` + `RRETCConfig` + the shared pure
  algorithm pieces (confidence / gate / safe set / commit / metrics kernel)
- `batched_rr_etc.py` — `BatchedRoundRobinETC`: S seeds in lockstep on one
  `BatchedEnv`; each seed runs the same logic as a per-seed FSM. Reset
  derivation matches `BatchedCAETCBaseline`, so the same `base_rng` gives
  bit-identical per-seed markets across the two baselines.
- `compare_rr_vs_ca.py` — paired-market comparison vs the repo's batched
  CA-ETC: cumulative player/arm regret CSVs + a two-panel figure
- `run_rr_etc.py` — single-run CLI → `metrics.csv`, `round_log.csv`,
  `summary.json`, `curves.png`
- `test_rr_etc.py`, `test_batched_rr_etc.py` — self-tests (see below)
- `demo_5x5_seed0/`, `compare_5x5_exact/` — example outputs

Note on jax versions: the repo venv is pinned to jax 0.6.x — the CA-ETC
batched kernels hit an opaque XLA "Unknown MLIR failure" on jax >= 0.7 on
macOS arm64 CPU. The RR-ETC code runs on both.

## Run

```bash
# from this folder, with the repo venv (needs jax, flax, distrax)
python run_rr_etc.py --Nw 5 --Nf 5 --d 5 --horizon 8000 \
    --sigma-interview 0.08 --seed 0 --outdir out_5x5_seed0
python test_rr_etc.py
```

## Test results (local, CPU jax)

1. **Physical GS ≡ `worker_proposing_da`** under perfect knowledge — 3/3
   random utility matrices.
2. **Structured 3×3** (paper Example-1 preference pattern): converges to the
   unique stable matching on 3/3 seeds; also reproduces the paper's
   "confident ≠ safe" behavior (the bottom-ranked worker is confident first
   but cannot exit early).
3. **Random 4×4** (env-sampled half-normal features): 3/3 seeds settle on the
   exact reference matching, final per-period regret 0. Seed 1 demonstrates
   the shrinkage effect: its tightest (0.043) utility pair involves a firm
   that stronger workers commit away, so the residual market never needs to
   resolve it — the *effective* gap is the residual-market gap, not the
   global minimum.
4. **Gate stress** (worker gaps ≈ 0.7, firm gap 0.05): gate holds
   fully-confident checkpoints until the firm's ranking is right; correct
   final matching 4/4 seeds. **Ablation with `--no-gate`: 3/10 seeds lock a
   permanently wrong (unstable) matching** — the failure mode the gate exists
   to prevent. (That ablation also shows why `regret_true_pos_total`
   exists: the wrong matching's aggregate welfare can coincide with the
   stable one; the positive-part per-worker regret cannot.)

## Performance

Single env, CPU jax (`env.step` jitted per call): **~400 market periods/s**
(≈ 2000 phase steps/s). A 5×5 / 8000-period episode ≈ 20 s.

`BatchedRoundRobinETC` steps all seeds per period in one vmapped dispatch and
keeps the algorithm logic per-seed in numpy: **~440 periods/s regardless of
S** (measured S = 4, 6, 10) → ~S-fold effective speedup; a 10-seed 5×5 /
8000-period sweep ≈ 23 s. `run()` returns a CA-ETC-style history
(`regret_w_total` etc. as (S, T) arrays) plus per-seed summaries.

## Comparison vs CA-ETC (`compare_rr_vs_ca.py`)

Both baselines run on bit-identical per-seed markets (verified at startup).
Outputs `cumulative_worker_regret.csv` / `cumulative_firm_regret.csv`
(`t, rr_mean, rr_std, ca_mean, ca_std`) and `compare_cumregret.png`.

Interpretation caveat for the default config (5×5, σ_int = 0.08): this is a
**low-noise regime that strongly favors CA-ETC's fixed schedule** — with
T0 = 5, γ = 0.4 it explores only ~150 of 8000 periods and mean-sorted
rankings are already correct after a handful of samples, so its cumulative
regret is ≈ 0. RR-ETC's certified full-ranking stopping rule (CI separation
of *every* adjacent pair, both sides gated) is far more conservative: it
spends thousands of exploration periods, and seeds whose residual gaps are
too small for the horizon never commit at all. The regimes where RR-ETC's
O(log T) guarantee pays off are high noise relative to gaps and long
horizons (the paper's Bernoulli / T = 5×10⁶ setting) — where CA-ETC's early
epochs lock wrong matchings and its periodic re-exploration keeps adding
stair-step regret. Report the operating regime alongside the curves.

## Scale guidance

- 5×5 … 10×10 with σ_interview ≲ 0.1: exact mode (`eps = 0`) settles well
  within 10⁴ periods.
- Random large markets (e.g. 100×100): the global min gap shrinks like 1/K²,
  so exact full-ranking confidence is unreachable — use `eps_w`/`eps_f`
  (ε-stability, disclosed) or generate preferences with a gap floor.
  Staged shrinkage softens this (binding gaps are residual-market gaps), but
  does not remove the tail.
