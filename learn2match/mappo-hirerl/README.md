# MAPPO for HireRL — shared centralized critic

MAPPO (centralized critic, decentralized actors) for the HireRL two-sided
matching market, written in JAX/Flax on the same `jax_pbt` stack as the IPPO
path. **One shared critic emits the values of all `Nw + Nf` agents in a single
forward pass per env-step** — the world state is never replicated per agent,
which is what makes MAPPO fit in memory at market sizes where the textbook
implementation cannot.

```
central_critic.py       pair-grid critic + input reconstruction
mappo_trainer.py        rollout step, per-role GAE, env-block PPO update
train_hirerl_mappo.py   CLI + controller (subclasses HireRLController)
test_mappo_smoke.py     correctness + end-to-end tests
```

## Why one forward instead of N

The standard JaxMARL MAPPO recipe (`baselines/MAPPO/mappo_rnn.py`) builds a
`world_state` per agent, stacks it into `(T, E, N, N·obs)`, and runs the critic
once per agent. That is `O(N²)` in both rollout storage and critic FLOPs and is
the sole reason MAPPO OOMs here where IPPO does not.

Measured on this env at `Nw = Nf = 100, d = 10`, per env-step per env:

| buffer | floats | vs IPPO baseline |
| --- | ---: | ---: |
| IPPO baseline (the two role obs buffers) | 371,800 | — |
| naive MAPPO `world_state` | 74,360,000 | **200×** |
| this implementation, `--critic_mode privileged` | 10,001 | +2.7% |
| this implementation, `--critic_mode strict` | 0 | **+0%** |

At the paper shape `T=128, E=128` (float32) that is 24.4 GB for the IPPO
baseline, **+0.7 GB** for this critic, versus **4873 GB** for naive MAPPO.

Two things get us there:

1. **No per-agent replication.** The critic consumes one pair grid per env and
   emits every agent's value from pooled readouts, so nothing carries an `N`
   replication factor.
2. **No separate global-state buffer.** HireRL's global state is exactly the
   union of the two roles' observations — worker obs rows stack into the full
   `hat_y (Nw, Nf, d)` tensor, firm obs rows transpose into `hat_x`, and the
   boolean/tenure matrices are visible from both sides. `build_critic_inputs`
   *reconstructs* the pair grid from the buffers the rollout already stores, at
   update time. Even one un-replicated copy of the global state would have
   roughly doubled the buffer (24.4 GB → 48.8 GB); storing none avoids that.

`test_mappo_smoke.py` checks the reconstruction block-by-block against the env
state's ground truth (`hat_x`, `hat_y`, `matched`, `interviewed`, both tenure
matrices, the match-proposal matrix), so this is verified, not assumed.

## Architecture

Actors are unchanged `PerCandidateActorCritic` policies acting on their own
observations; their built-in value heads receive no gradient and are ignored.

The critic (`PairGridCritic`) embeds each (worker, firm) pair once and pools:

```
E      = LayerNorm(MLP(P))                                  # (..., Nw, Nf, H)
g      = MLP([global_ctx, mean_ij E])                       # market context
V_w[i] = MLP([mean_j E[i,:], max_j E[i,:], worker_ctx[i], g])   # (..., Nw)
V_f[j] = MLP([mean_i E[:,j], max_i E[:,j], firm_ctx[j],  g])    # (..., Nf)
```

Per-pair features are `hat_y[i,j,:]`, `hat_x[i,j,:]`, matched, interviewed,
current/cumulative tenure, retention-candidate, and the three proposal
indicators; plus `x_i · y_j` in privileged mode.

Every weight shape depends only on `d` and the per-pair scalar count, so the
parameter count is independent of `Nw`/`Nf` (tested) and the same params run on
any market size — mirroring `PerCandidateActorCritic`. Values are permutation
-equivariant: relabelling workers permutes `V_w` identically and leaves `V_f`
unchanged (tested).

The critic is feedforward on purpose: `hat_x`/`hat_y` are belief summaries of
the entire interaction history, so the pair grid is close to a sufficient
statistic and a critic RNN would add state for little gain. `--use_rnn` still
controls the actors, exactly as in IPPO.

### Minibatching

Because the critic input is a whole env's pair grid, the PPO sampling unit is
an **(env, time-chunk) block** rather than a flattened per-agent row. All
`Nw + Nf` agents of an env travel together, and `split_into_minibatch` is reused
verbatim with `num_envs` as its batch axis. A test plants env-identifying values
in both roles' buffers and confirms the shuffle keeps worker and firm rows
aligned — misalignment would silently mix beliefs across envs.

**`--minibatch_size` therefore counts env-timesteps here, not per-agent samples
as in the IPPO script.** With `Nw=Nf=100` one env-timestep carries 200 agents.

### Losses

Three optimizers: worker actor, firm actor, shared critic. GAE is per-agent and
identical to the IPPO path (rewards are individual). The critic loss is the
value MSE pooled over both roles, weighting every agent equally. `--value_clip`
uses the *corrected* formulation that clips around the rollout-time prediction
(`jax_pbt`'s built-in clip is a no-op — see `examples/value_clipped_trainer.py`);
`critic_val_clip_fraction` is logged so the threshold is self-diagnosing.

## Usage

Env flags, eval, learning curves, checkpoint cadence and W&B logging are
identical to `examples/train_hirerl_ippo.py` — the controller subclasses
`HireRLController` and reuses its `run()`.

```bash
python learn2match/mappo-hirerl/train_hirerl_mappo.py \
  --Nw 100 --Nf 100 --d 10 --horizon 100 \
  --num_envs 128 --rollout_length 128 \
  --total_env_steps 20000000 \
  --chunk_length 16 --num_minibatches 8 \
  --critic_mode privileged --critic_hidden_size 128 --critic_lr 5e-4 \
  --run_name hirerl_mappo_100x100
```

MAPPO-specific flags:

| flag | meaning |
| --- | --- |
| `--critic_mode` | `privileged` (default) adds true `x_i·y_j` and `market_t/horizon`, legal under CTDE, costs one `(Nw,Nf)` grid per env-step. `strict` restricts the critic to the union of the roles' observations, at zero extra storage. |
| `--critic_hidden_size` | critic width (default 64; scale up with `d` and `N`). |
| `--critic_lr` | critic learning rate (default: `--lr`). |
| `--scale_clip_eps` | MAPPO-paper trick: divide `ratio_clip` by the number of agents sharing a policy update. Worth trying at large `N`. |

Start with `privileged`: it is the stronger baseline and the +2.7% storage is
negligible. Use `strict` when you want the critic's information set to be
exactly the agents' joint observation.

### Logging and checkpoints

Shared-critic stats appear in W&B and the npz under the `worker/` prefix with a
`critic_` prefix (`worker/critic_val_loss`, `worker/critic_grad_norm`,
`worker/critic_explained_variance_w`, `..._f`) — they describe the one critic
both roles share, so they are not duplicated under `firm/`.

Checkpoints keep the IPPO key layout, so `eval_and_plot.py --ppo_ckpt` loads
MAPPO checkpoints directly (verified end to end). `critic_trainer_state` rides
alongside for warm-starting; `--init_from_ckpt` restores all three states from a
MAPPO checkpoint, or the two actors from an IPPO checkpoint with a fresh critic.

## Tests

```bash
python learn2match/mappo-hirerl/test_mappo_smoke.py
```

Covers pair-grid reconstruction against env ground truth, one-forward-all-values
plus permutation equivariance and `N`-independent params, worker/firm minibatch
alignment, and end-to-end training cycles across `{privileged, strict} ×
{use_rnn on, off}` asserting finite stats and that all three train states move.
All pass; a `100×100` cycle and a full CLI run (train → eval → checkpoint →
learning-curve npz/pngs) were also exercised.

## Limitations / upgrade paths

- **Actors still see only their own observations.** That is MAPPO by design; the
  centralization is in the critic. If the actors are the bottleneck, that is an
  IPPO-side architecture question, not a MAPPO one.
- **The critic is feedforward.** If a run shows value error concentrated at
  phase boundaries, a small GRU over the pooled `g` vector is the cheapest
  upgrade — it would add `O(H)` state per env, not per agent.
- **Value targets stay per-agent.** Individual rewards make this the right
  default; a joint-return critic would need a different GAE.
- **The IPPO baseline buffer (~24 GB at 100×100, T=128, E=128) is untouched.**
  It is unrelated to MAPPO and is managed by `--num_envs` / `--rollout_length`.
  There is compressible redundancy in it (worker and firm obs both store the
  `interviewed`/`tenure` matrices, transposed) if that ever becomes the binding
  constraint.
