"""Batched (multi-seed) oracle-assisted Round-Robin ETC on HireRL.

Design: one BatchedEnv steps all seeds together every market period; each
seed's algorithm logic runs as an independent per-seed finite-state machine
(EXPLORE -> checkpoint -> GS -> commit -> ... -> DONE) in plain numpy at the
driver level. Control flow is per-seed (seeds commit at different periods),
but every seed consumes exactly one market period per driver iteration, so
the batch stays in lockstep on the time axis and the env work is a single
vmapped dispatch per phase.

All algorithm math (confidence, gate, safe set, commit bookkeeping, metrics)
is imported from ``rr_etc_baseline`` — the single-env driver remains the
correctness reference; this file only re-hosts the same logic behind a
per-seed FSM.

Market pairing with CA-ETC: the reset path replicates
``BatchedCAETCBaseline``'s default derivation exactly —

    rng, rst_rng = jax.random.split(base_rng)
    state, _ = BatchedEnv(env, S).reset(rst_rng, const_batch)

so constructing both baselines with the same ``base_rng`` and ``num_seeds``
gives bit-identical per-seed (x, y) markets.
"""

from __future__ import annotations

import math
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "learn2match"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
import jax.numpy as jnp

from jax_pbt.env.batched_env import BatchedEnv
from hirerl.constants import MATCH_PROPOSE, MATCH_RESPOND, RETENTION
from hirerl.metrics import worker_proposing_da

from rr_etc_baseline import (
    RRETCConfig,
    build_metrics_fn,
    commit_members,
    explore_reps,
    gate_check,
    influence_safe_set,
    worker_confident,
)


class _SeedAlgo:
    """Per-seed algorithm state machine (no env access).

    Modes: "explore" (inside a round's exploration block), "gs" (inside a
    physical GS block), "done" (all workers settled / nothing left to do).
    The driver calls ``build_actions`` before a period and ``consume`` after
    it; checkpoint math (confidence / gate / safe set / commit) runs inside
    ``consume`` at block boundaries and costs no periods, mirroring the
    single-env driver's checkpoint loop.
    """

    def __init__(self, sid: int, Nw: int, Nf: int, horizon: int,
                 cfg: RRETCConfig, x: np.ndarray, y: np.ndarray,
                 sigma_interview: float, outside: float, ref_np: np.ndarray):
        self.sid = sid
        self.Nw, self.Nf = Nw, Nf
        self.horizon = horizon
        self.cfg = cfg
        self._x = x.astype(np.float64)
        self._y = y.astype(np.float64)
        self.U = self._x @ self._y.T
        self.sigma_w = np.linalg.norm(self._x, axis=1) * sigma_interview
        self.ref_np = ref_np
        if not bool(np.all(ref_np.sum(axis=1) == 1)):
            unmatched = np.nonzero(ref_np.sum(axis=1) == 0)[0].tolist()
            print(f"[rr-etc][seed {sid}] WARNING: outside option {outside} is "
                  f"binding — reference leaves workers {unmatched} unmatched.")

        self.w_means = np.zeros((Nw, Nf))
        self.w_counts = np.zeros((Nw, Nf), dtype=np.int64)
        self.f_means = np.zeros((Nf, Nw))
        self.f_counts = np.zeros((Nf, Nw), dtype=np.int64)
        self.remaining: List[int] = list(range(Nw))
        self.available: List[int] = list(range(Nf))
        self.index: Dict[int, int] = {w: p for p, w in enumerate(self.remaining)}
        self.settled: Dict[int, int] = {}
        self.round = 1
        self.round_log: List[Dict] = []
        self.commit_mismatches: List[Tuple] = []
        self.sched_misses = 0

        self.mode = "explore"
        self.t_in_block = 0
        self.block_len = explore_reps(cfg.L, horizon) * len(self.available)
        self._gs_prefs: Dict[int, List[int]] = {}
        self._gs_pointer: Dict[int, int] = {}
        self._gs_steps = 0
        self._gs_cap = 0
        self._pending_I: Set[int] = set()
        self._current_pairs: Dict[int, int] = {}
        self._current_props: Dict[int, int] = {}

    # ----------------------------------------------------------- actions

    def build_actions(self, matched_row: np.ndarray):
        """Return (w5, f5): per-phase action rows for this seed's period."""
        w5 = [np.zeros(self.Nw, dtype=np.int32) for _ in range(5)]
        f5 = [np.zeros(self.Nf, dtype=np.int32) for _ in range(5)]
        for i, j in self.settled.items():
            if not matched_row[i, j]:
                w5[MATCH_PROPOSE][i] = j + 1
                f5[MATCH_RESPOND][j] = i + 1
            w5[RETENTION][i] = 1
            f5[RETENTION][j] = 1

        if self.mode == "explore":
            K_r = len(self.available)
            self._current_pairs = {
                i: self.available[(self.index[i] + self.t_in_block) % K_r]
                for i in self.remaining
            }
            for i, j in self._current_pairs.items():
                w5[0][i] = j + 1          # INTERVIEW_PROPOSE
                f5[1][j] = i + 1          # INTERVIEW_RESPOND
                w5[MATCH_PROPOSE][i] = j + 1
                f5[MATCH_RESPOND][j] = i + 1
        elif self.mode == "gs":
            self._current_props = {
                i: self._gs_prefs[i][self._gs_pointer[i]] for i in self.remaining
            }
            by_firm: Dict[int, List[int]] = {}
            for i, j in self._current_props.items():
                w5[MATCH_PROPOSE][i] = j + 1
                by_firm.setdefault(j, []).append(i)
            for j, cands in by_firm.items():
                best = max(cands, key=lambda i: (self.f_means[j, i], -i))
                f5[MATCH_RESPOND][j] = best + 1
        return w5, f5

    # ----------------------------------------------------------- feedback

    def consume(self, tent_row: np.ndarray, hat_x_row: np.ndarray,
                hat_y_row: np.ndarray, t_global: int):
        if self.mode == "explore":
            for i, j in self._current_pairs.items():
                if tent_row[i, j]:
                    w_sample = float(self._x[i] @ hat_y_row[i, j])
                    f_sample = float(hat_x_row[i, j] @ self._y[j])
                    n = self.w_counts[i, j]
                    self.w_means[i, j] = (self.w_means[i, j] * n + w_sample) / (n + 1)
                    self.w_counts[i, j] = n + 1
                    m = self.f_counts[j, i]
                    self.f_means[j, i] = (self.f_means[j, i] * m + f_sample) / (m + 1)
                    self.f_counts[j, i] = m + 1
                else:
                    self.sched_misses += 1
            self.t_in_block += 1
            if self.t_in_block >= self.block_len:
                self._checkpoint(t_global)
        elif self.mode == "gs":
            rejected = {i for i, j in self._current_props.items()
                        if not tent_row[i, j]}
            self._gs_steps += 1
            if rejected:
                for i in rejected:
                    self._gs_pointer[i] += 1
                    if self._gs_pointer[i] >= len(self.available):
                        raise RuntimeError(
                            f"[seed {self.sid}] GS pointer overflow for worker {i}"
                        )
                if self._gs_steps > self._gs_cap:
                    raise RuntimeError(
                        f"[seed {self.sid}] GS did not converge within its cap"
                    )
            else:
                assign = dict(self._current_props)
                commit_members(self, self._pending_I, assign)
                self._log_commit(t_global, assign)
                self._checkpoint(t_global)

    # -------------------------------------------------------- checkpoints

    def _checkpoint(self, t_global: int):
        """Evaluate confidence/gate/safe-set; enter GS, a new exploration
        round, or DONE. Mirrors RoundRobinETC._checkpoint_loop — cascaded
        commits re-enter GS immediately (GS itself costs periods either way)."""
        if not self.remaining:
            self.mode = "done"
            return
        confident = {
            i: worker_confident(
                self.w_means[i], self.w_counts[i], self.available,
                self.sigma_w[i], self.horizon, self.cfg.c_radius, self.cfg.eps_w,
            )
            for i in self.remaining
        }
        unconf = [i for i, c in confident.items() if not c]
        if self.cfg.gate_enabled:
            gate_ok, gate_reason = gate_check(
                self.U, self.f_means, self.f_counts,
                self.remaining, self.available, self.cfg.eps_f,
            )
        else:
            gate_ok, gate_reason = True, "gate disabled"
        diag = {
            "round": self.round, "t": t_global,
            "n_remaining": len(self.remaining),
            "n_confident": len(self.remaining) - len(unconf),
            "gate_ok": gate_ok, "gate_reason": gate_reason,
            "committed": [],
        }
        I: Set[int] = set()
        if gate_ok:
            I, _J = influence_safe_set(
                self.U, self.remaining, self.available, unconf)
        self.round_log.append(diag)
        if gate_ok and I:
            self._pending_I = I
            self._gs_prefs = {
                i: sorted(self.available, key=lambda j: (-self.w_means[i, j], j))
                for i in self.remaining
            }
            self._gs_pointer = {i: 0 for i in self.remaining}
            self._gs_steps = 0
            self._gs_cap = len(self.remaining) * len(self.available) + 2
            self.mode = "gs"
        else:
            self.round += 1
            self.t_in_block = 0
            self.block_len = explore_reps(self.cfg.L, self.horizon) * len(self.available)
            self.mode = "explore"

    def _log_commit(self, t_global: int, assign: Dict[int, int]):
        committed = sorted((i, assign[i]) for i in self._pending_I)
        self.round_log.append({
            "round": self.round, "t": t_global,
            "n_remaining": len(self.remaining),
            "n_confident": -1, "gate_ok": True, "gate_reason": "",
            "committed": committed,
        })
        self._pending_I = set()

    # ------------------------------------------------------------ summary

    def summary(self) -> Dict:
        all_settled = len(self.settled) == self.Nw
        correct = all(bool(self.ref_np[i, j]) for i, j in self.settled.items()) \
            if self.settled else False
        return {
            "sid": self.sid,
            "settled": dict(self.settled),
            "all_settled": all_settled,
            "settled_matches_reference": correct and all_settled,
            "rounds": self.round,
            "commit_mismatches": list(self.commit_mismatches),
            "sched_misses": self.sched_misses,
        }


class BatchedRoundRobinETC:
    """S-seed lockstep driver. Same period-level semantics as the single-env
    ``RoundRobinETC`` for every seed; ~S-fold faster wall-clock."""

    def __init__(self, env, num_seeds: int, base_rng: jax.Array,
                 cfg: Optional[RRETCConfig] = None,
                 x_override: Optional[np.ndarray] = None,
                 y_override: Optional[np.ndarray] = None):
        self.env_single = env
        self.cfg = cfg or RRETCConfig()
        self.S = int(num_seeds)
        self.Nw, self.Nf = int(env.Nw), int(env.Nf)
        if self.Nw > self.Nf:
            raise ValueError("Round-Robin ETC requires Nw <= Nf")
        self.horizon = int(env.horizon)
        self.outside = float(env.config.outside_option)

        self.bench = BatchedEnv(env, num_envs=self.S)
        self.const_batch = self.bench.default_const

        # Reset derivation identical to BatchedCAETCBaseline (default path)
        # => same base_rng -> bit-identical per-seed markets across baselines.
        self.rng, rst_rng = jax.random.split(base_rng)
        self.state, _ = self.bench.reset(rst_rng, self.const_batch)

        # Test-harness hook: replace latent features after reset. (Nw, d) /
        # (Nf, d) broadcasts to all seeds; (S, Nw, d) sets them per seed.
        if x_override is not None or y_override is not None:
            def _bcast(arr, n_rows):
                a = np.asarray(arr, dtype=np.float32)
                if a.ndim == 2:
                    a = np.broadcast_to(a, (self.S, *a.shape))
                return jnp.asarray(a)
            new_x = _bcast(x_override, self.Nw) if x_override is not None else self.state.x
            new_y = _bcast(y_override, self.Nf) if y_override is not None else self.state.y
            self.state = self.state.replace(x=new_x, y=new_y)

        x_batch = np.asarray(self.state.x)                     # (S, Nw, d)
        y_batch = np.asarray(self.state.y)                     # (S, Nf, d)
        sigma_interview = float(env.config.sigma_interview)

        U_batch = np.einsum("sid,sjd->sij", x_batch, y_batch)
        ref_batch = jax.vmap(
            lambda u: worker_proposing_da(u, u, outside=self.outside)
        )(jnp.asarray(U_batch, jnp.float32))
        self.ref_batch = ref_batch                              # (S, Nw, Nf)
        ref_np = np.asarray(ref_batch)

        self.seeds = [
            _SeedAlgo(s, self.Nw, self.Nf, self.horizon, self.cfg,
                      x_batch[s], y_batch[s], sigma_interview, self.outside,
                      ref_np[s])
            for s in range(self.S)
        ]

        bench = self.bench
        const_batch = self.const_batch

        def _step(rng, state, w_act, f_act):
            return bench.step(rng, const_batch, state,
                              [{"choice": w_act}, {"choice": f_act}])

        self._step_jit = jax.jit(_step)

        # Metrics kernel returns one stacked (n_keys, S) array so each period
        # costs a single device->host transfer; per-key (S, horizon) numpy
        # buffers are preallocated (a jnp.stack over `horizon` per-period
        # records would create one op with `horizon` operands at materialize
        # time — fine at 8k periods, pathological at 150k).
        metrics_fn = build_metrics_fn(self.cfg.compute_friction)
        probe = metrics_fn(
            jax.tree_util.tree_map(lambda a: a[0], self.state),
            jnp.zeros((self.Nw, self.Nf), bool), self.outside)
        self._rec_keys = sorted(probe.keys())
        rec_keys = self._rec_keys

        def _metrics_stacked(state_b, ref_b, outside):
            out = jax.vmap(metrics_fn, in_axes=(0, 0, None))(state_b, ref_b, outside)
            return jnp.stack([out[k] for k in rec_keys])       # (n_keys, S)

        self._metrics_jit = jax.jit(_metrics_stacked)
        self._rec_buffers = {
            k: np.zeros((self.S, self.horizon), dtype=np.float32)
            for k in self._rec_keys
        } if self.cfg.record_metrics else None

        self.market_period = 0

    def run(self) -> Dict:
        t_start = time.time()
        while self.market_period < self.horizon:
            matched_np = np.asarray(self.state.matched)         # (S, Nw, Nf)
            w_stack = [np.zeros((self.S, self.Nw), np.int32) for _ in range(5)]
            f_stack = [np.zeros((self.S, self.Nf), np.int32) for _ in range(5)]
            for seed in self.seeds:
                w5, f5 = seed.build_actions(matched_np[seed.sid])
                for ph in range(5):
                    w_stack[ph][seed.sid] = w5[ph]
                    f_stack[ph][seed.sid] = f5[ph]

            post_match_state = None
            for ph in range(5):
                self.rng, sub = jax.random.split(self.rng)
                self.state, _obs, _rew, _done, _info = self._step_jit(
                    sub, self.state,
                    jnp.asarray(w_stack[ph]), jnp.asarray(f_stack[ph]),
                )
                if ph == MATCH_RESPOND:
                    post_match_state = self.state
            self.market_period += 1

            if self.cfg.record_metrics:
                rec = np.asarray(self._metrics_jit(
                    post_match_state, self.ref_batch, self.outside))
                for ki, k in enumerate(self._rec_keys):
                    self._rec_buffers[k][:, self.market_period - 1] = rec[ki]

            tent = np.asarray(post_match_state.tentative_matched)
            hat_x = np.asarray(post_match_state.hat_x)
            hat_y = np.asarray(post_match_state.hat_y)
            for seed in self.seeds:
                seed.consume(tent[seed.sid], hat_x[seed.sid], hat_y[seed.sid],
                             self.market_period)

            if self.cfg.verbose and self.market_period % 1000 == 0:
                n_done = sum(len(s.settled) for s in self.seeds)
                print(f"  [t={self.market_period:>6}] settled workers "
                      f"{n_done}/{self.S * self.Nw} across {self.S} seeds")

        history = self._materialize(time.time() - t_start)
        return history

    def _materialize(self, wall: float) -> Dict:
        stacked = {}
        if self._rec_buffers is not None:
            T = self.market_period
            stacked = {k: v[:, :T] for k, v in self._rec_buffers.items()}
        summaries = [s.summary() for s in self.seeds]
        return {
            "t": np.arange(1, self.market_period + 1, dtype=np.int32),
            **stacked,
            "seed_summaries": summaries,
            "num_seeds": self.S,
            "Nw": self.Nw,
            "Nf": self.Nf,
            "market_periods": self.market_period,
            "wall_seconds": wall,
            "n_all_settled": sum(1 for s in summaries if s["all_settled"]),
            "n_correct": sum(1 for s in summaries
                             if s["settled_matches_reference"]),
            "total_sched_misses": sum(s["sched_misses"] for s in summaries),
        }
