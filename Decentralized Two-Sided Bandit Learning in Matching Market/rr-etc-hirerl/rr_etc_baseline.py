"""Oracle-assisted Round-Robin ETC on the HireRL environment.

Implements the player-side algorithm of

    Zhang & Fang, "Decentralized Two-Sided Bandit Learning in Matching
    Market" (Round-Robin ETC, Algorithm 1)

as a scripted baseline that drives a ``hirerl.env.HireRLEnv`` through legal
phase-level actions only. Every bandit "pull" is one full HireRL market
period (5 env.step calls).

Mapping (bandit -> HireRL)
--------------------------
* player j            <-> worker i (side W)
* arm k               <-> firm j (side F)
* pull / proposal     <-> INTERVIEW_PROPOSE + MATCH_PROPOSE to the firm
* arm accepts one     <-> firm accepts exactly one proposer at MATCH_RESPOND
* rejection C_j(t)    <-> worker's proposal absent from state.tentative_matched
* reward sample       <-> x_i . hat_y[i, j] (worker) / hat_x[i, j] . y_j (firm),
                          read post-MATCH_RESPOND; fresh interview noise each
                          period because exploration periods never retain, so
                          cumulative_tenure stays 0 and the interview belief
                          update is never masked out
* occupied arm        <-> a committed pair held in state.matched; the env's
                          worker_unmatched/firm_unmatched guards make such a
                          firm reject all new proposals automatically
* exploitation        <-> RETENTION retain (action 1) on the committed pair
                          every period until the horizon

Deviations from the paper (all deliberate, see README.md)
---------------------------------------------------------
1. Oracle safe-set instead of deliberate-conflict communication. The paper's
   COMM sub-phase lets a player certify "no unfinished player can displace
   me" through staged rejections. Here an oracle computes the identical
   predicate directly: J_r = Reach(F_r) in the influence graph built from
   TRUE firm preferences, I_r = remaining \\ J_r commits.
2. Arm-readiness gate (our addition, not in the paper). Commits are allowed
   only when every available firm's EMPIRICAL ranking over the remaining
   workers agrees with its true ranking (on eps_f-significant pairs). The
   paper closes this hole with the D * Delta^a >= Delta assumption; HireRL
   gives no such guarantee, and a wrong firm ranking at GS time would lock a
   wrong match forever. The gate only ever delays commits, it never changes
   outcomes.
3. Indices are oracle-coordinated (rank among remaining workers) instead of
   collision-assigned; the available-firm set is updated directly instead of
   discovered by probing. Both are coordination shortcuts, not preference
   information.
4. Firms play the empirical-leader rational strategy: among this period's
   proposers, accept the one with the highest firm-side empirical mean.
5. Calibrated confidence radius: sigma_eff,i = ||x_i|| * sigma_interview
   (exact for HireRL's Gaussian interview noise), radius
   c_radius * sigma_eff * sqrt(2 ln(horizon) / n).
6. Exploration length per round is a granularity knob (L), not the paper's
   K^2 factor: reps_per_round = ceil(L * log2(horizon)) visits of every
   (remaining worker, available firm) pair. Total exploration is governed by
   the confidence check, so L only sets how often checkpoints happen.
7. eps_w / eps_f tolerances for gap-free random markets: worker pairs whose
   order cannot matter by more than eps_w may stay unresolved; firm pairs
   closer than eps_f are not required to be ranked correctly. eps = 0
   recovers the faithful (exact) rule.

Estimation discipline
---------------------
Workers update empirical means ONLY from exploration periods. Firms update
ONLY from exploration periods as well (the only periods with fresh interview
noise; matches formed during GS carry no new information in HireRL and
updating on them would inject duplicate samples). Firm means are frozen
between the gate check and the GS that follows it, so "gate passed" implies
every firm decision inside that GS is correct.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "learn2match")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
import jax.numpy as jnp

from hirerl.constants import (
    INTERVIEW_PROPOSE,
    INTERVIEW_RESPOND,
    MATCH_PROPOSE,
    MATCH_RESPOND,
    RETENTION,
)
from hirerl.metrics import worker_proposing_da


@dataclass
class RRETCConfig:
    """Knobs for the oracle-assisted Round-Robin ETC baseline."""

    # Exploration granularity: every round visits each (remaining worker,
    # available firm) pair ceil(L * log2(horizon)) times. Paper-faithful value
    # is L = K^2; the confidence check governs total exploration either way.
    L: float = 1.0
    # Multiplier on the calibrated confidence radius (1.0 = Hoeffding at
    # failure rate ~1/horizon per comparison). Commits are permanent, so keep
    # honest; values < 1 risk locking wrong matches.
    c_radius: float = 1.0
    # Worker-side indifference tolerance: adjacent arms are "resolved" when
    # UCB_next - LCB_cur < eps_w. 0.0 = exact full-ranking confidence.
    eps_w: float = 0.0
    # Firm-side gate significance: only true-utility pairs differing by at
    # least eps_f must be empirically ranked correctly. 0.0 = all pairs.
    eps_f: float = 0.0
    # Arm-readiness gate on commits (deviation #2). Disable only to reproduce
    # the failure mode it exists to prevent.
    gate_enabled: bool = True
    # Raise on a commit that mismatches the true stable matching (only
    # enforced when eps_w == eps_f == 0, where exactness is guaranteed w.h.p.)
    strict_checks: bool = True
    # Record a per-period metrics snapshot (post-MATCH_RESPOND effective
    # gauge). Adds one device sync per period.
    record_metrics: bool = True
    # Include friction_loss in the per-period record (runs one extra DA per
    # period inside the jitted kernel; cheap at small N).
    compute_friction: bool = True
    verbose: bool = True


# ---------------------------------------------------------------------------
# Pure algorithm pieces, shared by the single-env and batched drivers.
# ---------------------------------------------------------------------------


def explore_reps(L: float, horizon: int) -> int:
    """Visits of every (remaining worker, available firm) pair per round."""
    return max(1, math.ceil(L * math.log2(max(horizon, 4))))


def build_metrics_fn(compute_friction: bool):
    """Effective-gauge per-period metrics for ONE (un-batched) state.

    jit it directly for the single-env driver, or jit(vmap(...)) for the
    batched driver.
    """

    def _metrics(state, ref_matched, outside):
        eff = state.matched | state.tentative_matched
        eff_f = eff.astype(jnp.float32)
        U = jnp.einsum("id,jd->ij", state.x, state.y)
        U_wb = jnp.sum(state.x[:, None, :] * state.hat_y, axis=-1)
        U_fb = jnp.sum(state.hat_x * state.y[None, :, :], axis=-1)
        ref_f = ref_matched.astype(jnp.float32)
        ref_true = jnp.sum(ref_f * U)
        eff_true = jnp.sum(eff_f * U)
        # Positive-part per-worker regret: sensitive to unstable matchings
        # whose aggregate value coincides with the stable one.
        per_w_ref = jnp.sum(ref_f * U, axis=1)
        per_w_eff = jnp.sum(eff_f * U, axis=1)
        out = {
            "regret_true_total": ref_true - eff_true,
            "regret_true_pos_total": jnp.sum(
                jnp.maximum(per_w_ref - per_w_eff, 0.0)
            ),
            "regret_w_total": ref_true - jnp.sum(eff_f * U_wb),
            "regret_f_total": ref_true - jnp.sum(eff_f * U_fb),
            "social_welfare": jnp.sum(eff_f * (U_wb + U_fb)),
            "social_welfare_true": 2.0 * eff_true,
            "match_rate": jnp.sum(eff_f) / jnp.float32(max(min(eff.shape), 1)),
        }
        if compute_friction:
            bi = worker_proposing_da(U_wb, U_fb, outside)
            out["friction_loss"] = ref_true - jnp.sum(bi.astype(jnp.float32) * U)
        return out

    return _metrics


def worker_confident(
    means_row: np.ndarray,
    counts_row: np.ndarray,
    available: List[int],
    sigma_i: float,
    horizon: int,
    c_radius: float,
    eps_w: float,
) -> bool:
    """Full-ranking confidence over the available firms: in mean-sorted order
    every adjacent pair satisfies UCB_next - LCB_cur < eps_w. Adjacent
    separation is equivalent to all-pairs separation and O(K log K)."""
    idx = np.asarray(available)
    counts = counts_row[idx]
    if np.any(counts == 0):
        return False
    means = means_row[idx]
    rad = c_radius * sigma_i * np.sqrt(2.0 * math.log(max(horizon, 2)) / counts)
    order = np.argsort(-means)
    mo, ro = means[order], rad[order]
    if len(mo) <= 1:
        return True
    return bool(np.all((mo[1:] + ro[1:]) - (mo[:-1] - ro[:-1]) < eps_w))


def gate_check(
    U: np.ndarray,
    f_means: np.ndarray,
    f_counts: np.ndarray,
    remaining: List[int],
    available: List[int],
    eps_f: float,
) -> Tuple[bool, str]:
    """Arm-readiness gate: every available firm must rank every
    eps_f-significant pair of remaining workers correctly (empirically)."""
    R = np.asarray(remaining)
    if len(R) <= 1:
        return True, ""
    for j in available:
        tu = U[R, j]
        du = tu[:, None] - tu[None, :]
        sig = np.abs(du) >= max(eps_f, 1e-12)
        if not np.any(sig):
            continue
        cn = f_counts[j, R]
        if np.any(cn == 0):
            return False, f"firm {j} has unsampled remaining workers"
        de = f_means[j, R][:, None] - f_means[j, R][None, :]
        if np.any(sig & (np.sign(de) != np.sign(du))):
            return False, f"firm {j} misranks a significant pair"
    return True, ""


def influence_safe_set(
    U: np.ndarray,
    remaining: List[int],
    available: List[int],
    unconfident: List[int],
) -> Tuple[Set[int], Set[int]]:
    """Oracle influence closure. Edge p -> q iff some available firm truly
    prefers p over q; J = Reach(unconfident), I = remaining \\ J."""
    R = remaining
    sub = U[np.ix_(np.asarray(R), np.asarray(available))]
    adj = np.any(sub[:, None, :] > sub[None, :, :], axis=2)
    np.fill_diagonal(adj, False)
    pos = {w: p for p, w in enumerate(R)}
    reach = np.zeros(len(R), dtype=bool)
    stack = [pos[w] for w in unconfident]
    for p in stack:
        reach[p] = True
    while stack:
        p = stack.pop()
        for q in np.nonzero(adj[p] & ~reach)[0]:
            reach[q] = True
            stack.append(int(q))
    J = {R[p] for p in np.nonzero(reach)[0]}
    return set(R) - J, J


def commit_members(algo, I: Set[int], assign: Dict[int, int]):
    """Lock each worker in I to its GS firm; shrink the market; reindex.

    ``algo`` is any holder with the shared attribute set (cfg, ref_np, round,
    settled, remaining, available, index, commit_mismatches) — used by both
    the single-env and batched drivers.
    """
    exact = (algo.cfg.eps_w == 0.0 and algo.cfg.eps_f == 0.0)
    for i in sorted(I):
        j = assign[i]
        if not algo.ref_np[i, j]:
            true_j = int(np.argmax(algo.ref_np[i])) if algo.ref_np[i].any() else -1
            algo.commit_mismatches.append((algo.round, i, j, true_j))
            msg = (
                f"[rr-etc] commit mismatch: worker {i} locked firm {j}, "
                f"true stable partner is {true_j} (round {algo.round})"
            )
            if algo.cfg.strict_checks and exact:
                raise AssertionError(msg)
            print(msg)
        algo.settled[i] = j
        algo.remaining.remove(i)
        algo.available.remove(j)
    algo.index = {w: p for p, w in enumerate(sorted(algo.remaining))}


class RoundRobinETC:
    """Scripted oracle-assisted Round-Robin ETC driver for one HireRLEnv."""

    def __init__(self, env, const, rng: jax.Array, cfg: Optional[RRETCConfig] = None):
        self.env = env
        self.const = const
        self.rng = rng
        self.cfg = cfg or RRETCConfig()

        self.Nw = int(env.Nw)
        self.Nf = int(env.Nf)
        if self.Nw > self.Nf:
            raise ValueError(
                f"Round-Robin ETC requires Nw <= Nf (paper assumption N <= K); "
                f"got Nw={self.Nw}, Nf={self.Nf}."
            )
        self.horizon = int(env.horizon)
        self.outside = float(env.config.outside_option)
        self.sigma_interview = float(env.config.sigma_interview)

        self._build_jit_kernels()

    # ------------------------------------------------------------------ jit

    def _build_jit_kernels(self):
        env = self.env

        def _step(rng, const, state, w_act, f_act):
            return env.step(rng, const, state, [{"choice": w_act}, {"choice": f_act}])

        self._step_jit = jax.jit(_step)
        self._metrics_jit = jax.jit(build_metrics_fn(self.cfg.compute_friction))

    # -------------------------------------------------------------- episode

    def _init_episode(
        self,
        x_override: Optional[np.ndarray] = None,
        y_override: Optional[np.ndarray] = None,
    ):
        """Reset the env and initialize all oracle/algorithm state."""
        self.rng, r_reset = jax.random.split(self.rng)
        state, _ = self.env.reset(r_reset, self.const)
        if x_override is not None or y_override is not None:
            x_new = jnp.asarray(x_override, jnp.float32) if x_override is not None else state.x
            y_new = jnp.asarray(y_override, jnp.float32) if y_override is not None else state.y
            state = state.replace(x=x_new, y=y_new)
        self.state = state

        # Oracle-side constants for this episode.
        self._x = np.asarray(state.x, dtype=np.float64)
        self._y = np.asarray(state.y, dtype=np.float64)
        self.U = self._x @ self._y.T                                  # (Nw, Nf) true
        self.sigma_w = np.linalg.norm(self._x, axis=1) * self.sigma_interview
        self.sigma_f = np.linalg.norm(self._y, axis=1) * self.sigma_interview
        self.ref_matched = worker_proposing_da(
            jnp.asarray(self.U, jnp.float32), jnp.asarray(self.U, jnp.float32),
            outside=self.outside,
        )
        self.ref_np = np.asarray(self.ref_matched)
        if not bool(np.all(self.ref_np.sum(axis=1) == 1)):
            unmatched = np.nonzero(self.ref_np.sum(axis=1) == 0)[0].tolist()
            print(
                f"[rr-etc] WARNING: outside option {self.outside} is binding — "
                f"reference leaves workers {unmatched} unmatched. RR-ETC ignores "
                f"outside options; regret/fail checks will be distorted."
            )

        # Algorithm state.
        self.w_means = np.zeros((self.Nw, self.Nf))
        self.w_counts = np.zeros((self.Nw, self.Nf), dtype=np.int64)
        self.f_means = np.zeros((self.Nf, self.Nw))
        self.f_counts = np.zeros((self.Nf, self.Nw), dtype=np.int64)
        self.remaining: List[int] = list(range(self.Nw))
        self.available: List[int] = list(range(self.Nf))
        self.index: Dict[int, int] = {w: p for p, w in enumerate(self.remaining)}
        self.settled: Dict[int, int] = {}
        self.market_period = 0
        self.round = 0
        self.records: List[Dict] = []
        self.round_log: List[Dict] = []
        self.commit_mismatches: List[Tuple] = []
        self._sched_misses = 0

    def run(
        self,
        x_override: Optional[np.ndarray] = None,
        y_override: Optional[np.ndarray] = None,
    ) -> Dict:
        """Run one full episode (horizon market periods). Returns a summary.

        ``x_override`` / ``y_override`` replace the env-sampled latent features
        right after reset (test harness hook for constructed preference
        matrices); beliefs, matches and counters start from the usual reset
        state either way.
        """
        t_start = time.time()
        self._init_episode(x_override, y_override)

        # ----------------------------------------------------- main loop
        while self.remaining and self.market_period < self.horizon:
            self.round += 1
            if not self._exploration_block():
                break
            self._checkpoint_loop()

        # Exploitation tail: keep retaining committed pairs to the horizon.
        while self.market_period < self.horizon:
            w5, f5 = self._base_actions()
            self._run_period(w5, f5, "idle")

        return self._summary(time.time() - t_start)

    # ------------------------------------------------------------- periods

    def _base_actions(self):
        """NOOP actions for all 5 phases + committed-pair maintenance.

        A freshly committed pair is not yet in state.matched: it is
        established with one MATCH_PROPOSE / MATCH_RESPOND / retain sequence;
        thereafter retain (action 1) alone keeps it alive.
        """
        w5 = [np.zeros(self.Nw, dtype=np.int32) for _ in range(5)]
        f5 = [np.zeros(self.Nf, dtype=np.int32) for _ in range(5)]
        if self.settled:
            matched = np.asarray(self.state.matched)
            for i, j in self.settled.items():
                if not matched[i, j]:
                    w5[MATCH_PROPOSE][i] = j + 1
                    f5[MATCH_RESPOND][j] = i + 1
                w5[RETENTION][i] = 1
                f5[RETENTION][j] = 1
        return w5, f5

    def _run_period(self, w5, f5, label: str):
        """Advance the env through one full market period (5 phase steps)."""
        for ph in range(5):
            self.rng, sub = jax.random.split(self.rng)
            state, _obs, _rew, _done, _info = self._step_jit(
                sub, self.const, self.state, jnp.asarray(w5[ph]), jnp.asarray(f5[ph])
            )
            if ph == MATCH_RESPOND:
                self._post_match_state = state
            self.state = state
        self.market_period += 1
        if self.cfg.record_metrics:
            vals = self._metrics_jit(self._post_match_state, self.ref_matched, self.outside)
            rec = {k: float(v) for k, v in vals.items()}
            rec["t"] = self.market_period
            rec["phase_label"] = label
            rec["n_settled"] = len(self.settled)
            self.records.append(rec)

    def _period_explore(self, pairs: Dict[int, int]):
        """One conflict-free exploration period: interview + match the
        scheduled pairs, record one fresh sample per side per pair."""
        w5, f5 = self._base_actions()
        for i, j in pairs.items():
            w5[INTERVIEW_PROPOSE][i] = j + 1
            f5[INTERVIEW_RESPOND][j] = i + 1
            w5[MATCH_PROPOSE][i] = j + 1
            f5[MATCH_RESPOND][j] = i + 1
        self._run_period(w5, f5, "explore")

        ps = self._post_match_state
        tent = np.asarray(ps.tentative_matched)
        hat_x = np.asarray(ps.hat_x)
        hat_y = np.asarray(ps.hat_y)
        for i, j in pairs.items():
            if tent[i, j]:
                w_sample = float(self._x[i] @ hat_y[i, j])
                f_sample = float(hat_x[i, j] @ self._y[j])
                n = self.w_counts[i, j]
                self.w_means[i, j] = (self.w_means[i, j] * n + w_sample) / (n + 1)
                self.w_counts[i, j] = n + 1
                m = self.f_counts[j, i]
                self.f_means[j, i] = (self.f_means[j, i] * m + f_sample) / (m + 1)
                self.f_counts[j, i] = m + 1
            else:
                # The schedule is conflict-free by construction; a miss means
                # an action was invalidated by the env — always a bug.
                self._sched_misses += 1

    def _period_gs(self, proposals: Dict[int, int]) -> Set[int]:
        """One decentralized-GS period. Firms accept the empirical-best
        proposer. Returns the set of rejected workers (env-verified)."""
        w5, f5 = self._base_actions()
        by_firm: Dict[int, List[int]] = {}
        for i, j in proposals.items():
            w5[MATCH_PROPOSE][i] = j + 1
            by_firm.setdefault(j, []).append(i)
        for j, cands in by_firm.items():
            best = max(cands, key=lambda i: (self.f_means[j, i], -i))
            f5[MATCH_RESPOND][j] = best + 1
        self._run_period(w5, f5, "gs")
        tent = np.asarray(self._post_match_state.tentative_matched)
        return {i for i, j in proposals.items() if not tent[i, j]}

    # ---------------------------------------------------------- sub-blocks

    def _exploration_block(self) -> bool:
        """Index-staggered round-robin over available firms. Every remaining
        worker samples every available firm ``reps`` times. Returns False if
        the horizon was exhausted mid-block."""
        K_r = len(self.available)
        reps = explore_reps(self.cfg.L, self.horizon)
        for rep in range(reps):
            for s in range(K_r):
                if self.market_period >= self.horizon:
                    return False
                t = rep * K_r + s
                pairs = {
                    i: self.available[(self.index[i] + t) % K_r]
                    for i in self.remaining
                }
                self._period_explore(pairs)
        return True

    def _confident(self, i: int) -> bool:
        return worker_confident(
            self.w_means[i], self.w_counts[i], self.available,
            self.sigma_w[i], self.horizon, self.cfg.c_radius, self.cfg.eps_w,
        )

    def _gate_ok(self) -> Tuple[bool, str]:
        return gate_check(
            self.U, self.f_means, self.f_counts,
            self.remaining, self.available, self.cfg.eps_f,
        )

    def _safe_set(self, unconfident: List[int]) -> Tuple[Set[int], Set[int]]:
        return influence_safe_set(
            self.U, self.remaining, self.available, unconfident,
        )

    def _gs_block(self) -> Optional[Dict[int, int]]:
        """Physical worker-proposing deferred acceptance among all remaining
        workers on available firms, run through real market periods.
        Terminates at quiescence (a period with no rejections); every
        non-terminal period causes >= 1 permanent rejection so at most
        N_r * K_r + 1 periods are needed. Returns None on horizon exhaustion."""
        R = list(self.remaining)
        A = list(self.available)
        prefs = {i: sorted(A, key=lambda j: (-self.w_means[i, j], j)) for i in R}
        pointer = {i: 0 for i in R}
        for _ in range(len(R) * len(A) + 2):
            if self.market_period >= self.horizon:
                return None
            proposals = {}
            for i in R:
                if pointer[i] >= len(A):
                    raise RuntimeError(
                        f"GS pointer overflow for worker {i} (should be "
                        f"impossible with Nw <= Nf)"
                    )
                proposals[i] = prefs[i][pointer[i]]
            rejected = self._period_gs(proposals)
            if not rejected:
                return dict(proposals)
            for i in rejected:
                pointer[i] += 1
        raise RuntimeError("GS did not converge within its period cap")

    def _checkpoint_loop(self):
        """After an exploration block: repeatedly (test confidence -> gate ->
        safe set -> GS -> commit) until no further commits are possible
        without more samples. Shrinking the market never invalidates existing
        confidence (fewer firms => a subset of the separated pairs), so
        cascading commits inside one checkpoint are sound."""
        while self.remaining and self.market_period < self.horizon:
            confident = {i: self._confident(i) for i in self.remaining}
            unconf = [i for i, c in confident.items() if not c]
            if self.cfg.gate_enabled:
                gate_ok, gate_reason = self._gate_ok()
            else:
                gate_ok, gate_reason = True, "gate disabled"
            diag = {
                "round": self.round,
                "t": self.market_period,
                "n_remaining": len(self.remaining),
                "n_confident": len(self.remaining) - len(unconf),
                "gate_ok": gate_ok,
                "gate_reason": gate_reason,
                "committed": [],
            }
            if not gate_ok:
                self.round_log.append(diag)
                self._log(diag)
                return
            I, _J = self._safe_set(unconf)
            if not I:
                self.round_log.append(diag)
                self._log(diag)
                return
            assign = self._gs_block()
            if assign is None:
                self.round_log.append(diag)
                return
            self._commit(I, assign)
            diag["committed"] = sorted((i, self.settled[i]) for i in I)
            diag["n_remaining"] = len(self.remaining)
            self.round_log.append(diag)
            self._log(diag)

    def _commit(self, I: Set[int], assign: Dict[int, int]):
        commit_members(self, I, assign)

    # ------------------------------------------------------------- output

    def _log(self, diag: Dict):
        if not self.cfg.verbose:
            return
        msg = (
            f"  [round {diag['round']:>3} | t={diag['t']:>6}] "
            f"remaining={diag['n_remaining']} confident={diag['n_confident']} "
            f"gate_ok={diag['gate_ok']}"
        )
        if not diag["gate_ok"]:
            msg += f" ({diag['gate_reason']})"
        if diag["committed"]:
            msg += f" committed={diag['committed']}"
        print(msg)

    def _summary(self, wall_seconds: float) -> Dict:
        all_settled = len(self.settled) == self.Nw
        correct = all(
            bool(self.ref_np[i, j]) for i, j in self.settled.items()
        ) if self.settled else False
        final_regret = self.records[-1]["regret_true_total"] if self.records else float("nan")
        final_pos_regret = (
            self.records[-1]["regret_true_pos_total"] if self.records else float("nan")
        )
        return {
            "settled": dict(self.settled),
            "all_settled": all_settled,
            "settled_matches_reference": correct and all_settled,
            "rounds": self.round,
            "market_periods": self.market_period,
            "horizon": self.horizon,
            "records": self.records,
            "round_log": self.round_log,
            "commit_mismatches": self.commit_mismatches,
            "sched_misses": self._sched_misses,
            "final_regret_true_total": final_regret,
            "final_regret_true_pos_total": final_pos_regret,
            "reference_match": self.ref_np,
            "U_true": self.U,
            "wall_seconds": wall_seconds,
        }
