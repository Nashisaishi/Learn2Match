"""CA-ETC baseline driven through the HireRL phase-level interaction protocol.

Mapping
-------
* CA-ETC players  <-> HireRL workers (side W)
* CA-ETC arms     <-> HireRL firms   (side F)
* 1 CA-ETC round  <-> 1 HireRL market period = 5 phase-level env.step calls:
    INTERVIEW_PROPOSE, INTERVIEW_RESPOND, MATCH_PROPOSE, MATCH_RESPOND, RETENTION

The standalone Market simulator from baselines/author_code_bandit_learning/ca-etc.py
is used only as an algorithmic reference. Every CA-ETC exploration pull (i, j)
is materialized through legal HireRL actions; no environment matching state is
mutated directly.

Action encoding (HireRL):
* choice 0 = NOOP
* worker i picks firm j  -> worker_action[i] = j + 1
* firm j picks worker i  -> firm_action[j] = i + 1
* During RETENTION, choice 1 means "retain"; choice 0 means "release".
"""

from __future__ import annotations

from itertools import permutations
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from hirerl.constants import MATCH_RESPOND
from hirerl.metrics import belief_induced_match, reference_match


@jax.jit
def _compute_effective_metrics(state, outside: float) -> Dict[str, jax.Array]:
    """Hirerl-gauge metrics computed under ``effective_matched``.

    ``effective_matched = state.matched | state.tentative_matched`` —
    the union of pairs already committed by previous RETENTION steps and
    pairs just formed by the current MATCH_RESPOND. The two are disjoint
    by env construction (``transitions.py`` guards new tentatives against
    already-matched workers/firms), so the union is well-defined.

    Designed to be called **after every MATCH_RESPOND** so that:

    * exploration periods (where ``state.matched`` is empty because the
      previous RETENTION sent NOOP) still produce a meaningful regret /
      reward / welfare reading from ``state.tentative_matched``;
    * exploitation periods (where ``state.tentative_matched`` is empty
      after the first commit because no new pairs form) still produce a
      meaningful reading from the persisted ``state.matched``.

    All formulas mirror ``hirerl.metrics`` but substitute ``state.matched``
    with ``effective``; ``friction_loss`` is matched-independent and is
    computed identically.
    """
    effective = state.matched | state.tentative_matched
    eff_f = effective.astype(jnp.float32)

    U_w_belief = jnp.sum(state.x[:, None, :] * state.hat_y, axis=-1)
    U_f_belief = jnp.sum(state.hat_x * state.y[None, :, :], axis=-1)
    U_true     = jnp.einsum("id,jd->ij", state.x, state.y)

    ref_matched = reference_match(state, outside)
    ref_f = ref_matched.astype(jnp.float32)

    reward_w_per = jnp.sum(eff_f * U_w_belief, axis=1)
    reward_f_per = jnp.sum(eff_f * U_f_belief, axis=0)

    ref_per_w = jnp.sum(ref_f * U_true, axis=1)
    ref_per_f = jnp.sum(ref_f * U_true, axis=0)
    regret_w_per = ref_per_w - reward_w_per
    regret_f_per = ref_per_f - reward_f_per

    diff_y_sq = jnp.sum((state.hat_y - state.y[None, :, :]) ** 2, axis=-1)
    diff_x_sq = jnp.sum((state.hat_x - state.x[:, None, :]) ** 2, axis=-1)
    info_loss_w_per = jnp.sum(eff_f * diff_y_sq, axis=1)
    info_loss_f_per = jnp.sum(eff_f * diff_x_sq, axis=0)

    bi_matched = belief_induced_match(state, outside)
    bi_f = bi_matched.astype(jnp.float32)
    friction = jnp.sum(ref_f * U_true) - jnp.sum(bi_f * U_true)

    social_welfare = jnp.sum(eff_f * (U_w_belief + U_f_belief))

    Nw, Nf = effective.shape
    norm = jnp.float32(max(min(Nw, Nf), 1))
    match_rate = jnp.sum(eff_f) / norm

    return {
        "effective_matched": effective,
        "reward_w_per_agent":   reward_w_per,
        "reward_f_per_agent":   reward_f_per,
        "regret_w_per_agent":   regret_w_per,
        "regret_f_per_agent":   regret_f_per,
        "info_loss_w_per_agent": info_loss_w_per,
        "info_loss_f_per_agent": info_loss_f_per,
        "reward_w_total":      jnp.sum(reward_w_per),
        "reward_f_total":      jnp.sum(reward_f_per),
        "regret_w_total":      jnp.sum(regret_w_per),
        "regret_f_total":      jnp.sum(regret_f_per),
        "info_loss_w_total":   jnp.sum(info_loss_w_per),
        "info_loss_f_total":   jnp.sum(info_loss_f_per),
        "social_welfare":      social_welfare,
        "friction_loss":       friction,
        "match_rate":          match_rate,
    }


class CAETCBaseline:
    """Collision-Avoidance ETC baseline that drives a HireRLEnv.

    Sample-collection invariants (see paper-faithful spec):

    * A CA-ETC sample for pair (i, j) is recorded immediately after
      MATCH_RESPOND if the pair is in ``state.tentative_matched``. Whether the
      pair is later dissolved during RETENTION is irrelevant to preference
      estimation -- the sample has already landed.
    * Empirical means / counts are NEVER updated during exploitation.
    * No direct mutation of ``state.matched``, ``state.tentative_matched``,
      ``state.interviewed`` or any other env-managed field.
    """

    def __init__(
        self,
        env,
        const,
        rng: jax.Array,
        Nw: int,
        Nf: int,
        horizon: int,
        T0: int = 1,
        gamma: float = 0.4,
        use_confidence_check: bool = True,
        radius_scale: float = 0.5,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.const = const
        self.rng = rng
        self.Nw = int(Nw)
        self.Nf = int(Nf)
        self.horizon = int(horizon)
        self.T0 = int(T0)
        self.gamma = float(gamma)
        self.use_confidence_check = bool(use_confidence_check)
        # Multiplicative scale on the LCB/UCB confidence radius. The textbook
        # CA-ETC uses sqrt(2 * log(T) / n) (radius_scale=1.0); smaller values
        # tighten the intervals so adjacent ranks separate sooner — i.e. the
        # algorithm commits to a ranking with less data. Set <1 to relax the
        # strictness of the separation check; the cost is a higher chance of
        # locking in the wrong order when sample noise is still large.
        self.radius_scale = float(radius_scale)
        self.seed = int(seed)

        # Paper-spec invariant: every epoch's exploration block must cover at
        # least one full round-robin so that all (i, j) pairs are interviewed
        # before exploitation runs. The smallest exploration block has length
        # T0 * 1^c = T0 market periods, and the collision-avoidance schedule
        # needs ceil(Nw/Nf) * Nf rounds to visit every pair. We enforce the
        # constraint at construction time so users cannot silently violate it.
        if self.Nw <= 0 or self.Nf <= 0:
            raise ValueError(
                f"Nw and Nf must be positive (got Nw={self.Nw}, Nf={self.Nf})."
            )
        num_batches = (self.Nw + self.Nf - 1) // self.Nf
        min_rounds = num_batches * self.Nf
        if self.T0 < min_rounds:
            raise ValueError(
                f"T0={self.T0} is too small to cover one full round-robin over "
                f"all {self.Nw}*{self.Nf}={self.Nw * self.Nf} (worker, firm) "
                f"pairs. Need T0 >= {min_rounds} (= ceil(Nw/Nf) * Nf). With "
                f"this T0, epoch 1's exploration block (T0 * 1^c = {self.T0} "
                f"market periods) would not finish even one round-robin, "
                f"leaving some pairs not interviewed when GS-driven "
                f"exploitation begins."
            )
        if self.horizon < min_rounds:
            raise ValueError(
                f"horizon={self.horizon} cannot accommodate one full "
                f"round-robin ({min_rounds} market periods). Increase horizon "
                f"or shrink Nw/Nf."
            )
        self._min_round_robin_rounds = min_rounds

        self.worker_mean_rewards = np.zeros((self.Nw, self.Nf), dtype=float)
        self.firm_mean_rewards = np.zeros((self.Nf, self.Nw), dtype=float)
        self.worker_counts = np.zeros((self.Nw, self.Nf), dtype=int)
        self.firm_counts = np.zeros((self.Nf, self.Nw), dtype=int)

        self.timestep = 0           # raw phase-level env.step count
        self.market_period = 0      # full HireRL market period count
        self.total_explore_rounds = 0

        self.last_obs = None
        self.last_reward = None
        self.last_done = None
        self.last_info = None

        self.metrics: Dict[str, List[Any]] = {
            "worker_counts": [],
            "firm_counts": [],
            "worker_mean_rewards": [],
            "firm_mean_rewards": [],
            "worker_preferences": [],
            "firm_preferences": [],
            "matchings": [],
            "epoch": [],
            "market_period": [],
            # Per-period snapshots under the *effective_matched* gauge:
            # ``effective_matched = state.matched | state.tentative_matched``.
            # Populated by step_env immediately after every MATCH_RESPOND
            # (so reads include the just-formed pairs from this period as
            # well as any pairs persisted by previous RETENTION decisions).
            # This differs from hirerl's canonical post-RETENTION view used
            # by examples/eval_and_plot.py: in CA-ETC's exploration block
            # RETENTION sends NOOP and clears everything, so the canonical
            # view is uniformly empty there and yields constant per-agent
            # regret = ref_per_worker (a flat horizontal line). The
            # effective_matched gauge avoids this by reading at the
            # MATCH_RESPOND chokepoint where the just-tested pair is still
            # present in tentative_matched. Schema is otherwise identical to
            # the canonical view so learn2match/examples/plot_helpers.py works
            # without changes; ``matched_matrix`` here is the effective
            # union, not state.matched alone.
            "settled_history": [],
        }

        # outside-option threshold required by the hirerl reference / regret /
        # friction-loss computations. Read once from the env config.
        self.outside_option = float(env.config.outside_option)

    # ---------------- empirical-mean updates ----------------

    def update_worker_mean(self, i: int, j: int, reward: float) -> None:
        n = self.worker_counts[i, j]
        self.worker_mean_rewards[i, j] = (
            self.worker_mean_rewards[i, j] * n + reward
        ) / (n + 1)
        self.worker_counts[i, j] += 1

    def update_firm_mean(self, i: int, j: int, reward: float) -> None:
        # firm_mean_rewards has shape (Nf, Nw), so index by [j, i]
        n = self.firm_counts[j, i]
        self.firm_mean_rewards[j, i] = (
            self.firm_mean_rewards[j, i] * n + reward
        ) / (n + 1)
        self.firm_counts[j, i] += 1

    # ---------------- env.step wrapper ----------------

    def step_env(self, state, worker_action, firm_action):
        worker_action = np.asarray(worker_action, dtype=int)
        firm_action = np.asarray(firm_action, dtype=int)

        assert worker_action.shape == (self.Nw,), (
            f"worker_action shape {worker_action.shape} != ({self.Nw},)"
        )
        assert firm_action.shape == (self.Nf,), (
            f"firm_action shape {firm_action.shape} != ({self.Nf},)"
        )
        assert np.all(worker_action >= 0)
        assert np.all(worker_action <= self.Nf)
        assert np.all(firm_action >= 0)
        assert np.all(firm_action <= self.Nw)

        actions = [
            {"choice": jnp.asarray(worker_action, dtype=jnp.int32)},
            {"choice": jnp.asarray(firm_action, dtype=jnp.int32)},
        ]

        # Snapshot at MATCH_RESPOND so that exploration periods (where
        # state.matched is empty after RETENTION-NOOP) and exploitation
        # steady-state periods (where state.tentative_matched is empty
        # because no new pairs form) BOTH yield meaningful per-agent
        # reward / regret / welfare. See _compute_effective_metrics.
        prev_phase = int(state.phase)

        self.rng, step_rng = jax.random.split(self.rng)
        new_state, obs, reward, done, info = self.env.step(
            step_rng,
            self.const,
            state,
            actions,
        )

        self.timestep += 1
        self.last_obs = obs
        self.last_reward = reward
        self.last_done = done
        self.last_info = info

        if prev_phase == MATCH_RESPOND:
            self._record_effective_snapshot(new_state)
        return new_state

    def _record_effective_snapshot(self, state) -> None:
        """Append one per-period record under the effective_matched gauge.

        Called at post-MATCH_RESPOND (before RETENTION). Uses
        ``effective_matched = state.matched | state.tentative_matched`` so the
        recorded metrics reflect "everything currently matched" rather than
        the post-RETENTION view (which is empty during exploration). All
        heavy computation happens inside the jit-compiled
        ``_compute_effective_metrics`` kernel; the per-call cost on the hot
        path is one jax dispatch + a small set of np.asarray copies.

        ``market_t`` is incremented by env.RETENTION transition only
        (``transitions.py:347``); at post-MATCH_RESPOND it still reads as the
        previous period. We log ``state.market_t + 1`` to label this record
        with the period it actually belongs to (1-indexed), which keeps the
        x-axis aligned with the canonical post-RETENTION view used by
        ``examples/eval_and_plot.py``.
        """
        per = _compute_effective_metrics(state, self.outside_option)
        record = {
            "t": int(state.market_t) + 1,
            "reward_w_total":      float(per["reward_w_total"]),
            "reward_f_total":      float(per["reward_f_total"]),
            "regret_w_total":      float(per["regret_w_total"]),
            "regret_f_total":      float(per["regret_f_total"]),
            "info_loss_w_total":   float(per["info_loss_w_total"]),
            "info_loss_f_total":   float(per["info_loss_f_total"]),
            "social_welfare":      float(per["social_welfare"]),
            "friction_loss":       float(per["friction_loss"]),
            "match_rate":          float(per["match_rate"]),
            "reward_w_per_agent":  np.asarray(per["reward_w_per_agent"]),
            "reward_f_per_agent":  np.asarray(per["reward_f_per_agent"]),
            "regret_w_per_agent":  np.asarray(per["regret_w_per_agent"]),
            "regret_f_per_agent":  np.asarray(per["regret_f_per_agent"]),
            "info_loss_w_per_agent": np.asarray(per["info_loss_w_per_agent"]),
            "info_loss_f_per_agent": np.asarray(per["info_loss_f_per_agent"]),
            # plot_helpers expects this key; it's the bipartite mask we color
            # the worker-side Gantt with. Use effective_matched (the union)
            # so exploration periods show the just-tested pair and
            # exploitation periods show the persisted assignment.
            "matched_matrix":      np.asarray(per["effective_matched"]),
        }
        self.metrics["settled_history"].append(record)

    # ---------------- round-robin schedule ----------------

    def round_robin_schedule(self, round_id: int) -> List[Optional[int]]:
        """Build a collision-free schedule for one exploration market period.

        Returns a list of length ``Nw``: ``scheduled[i] = j`` means worker i
        is scheduled to explore firm j. ``scheduled[i] is None`` means worker
        i is not scheduled this round.

        For ``Nw <= Nf``: every worker is scheduled, mapping
        ``i -> (i + round_id) % Nf``. After ``Nf`` rounds the modular cycle
        guarantees every (i, j) pair is visited at least once.

        For ``Nw > Nf``: at most ``Nf`` workers can be scheduled per round (one
        per firm). Workers are batched and the firm-offset rotates so that
        every (i, j) pair is eventually reached.
        """
        scheduled: List[Optional[int]] = [None] * self.Nw
        if self.Nw == 0 or self.Nf == 0:
            return scheduled

        if self.Nw <= self.Nf:
            for i in range(self.Nw):
                scheduled[i] = (i + round_id) % self.Nf
            return scheduled

        # Nw > Nf branch.
        firms_per_period = self.Nf
        num_batches = (self.Nw + firms_per_period - 1) // firms_per_period
        batch_id = round_id % num_batches
        firm_offset = round_id // num_batches
        for k in range(firms_per_period):
            i = batch_id * firms_per_period + k
            if i >= self.Nw:
                break
            scheduled[i] = (k + firm_offset) % self.Nf
        return scheduled

    # ---------------- one exploration market period ----------------

    def run_one_exploration_market_period(self, state):
        """Run one CA-ETC exploration round through one full HireRL market period.

        1. build a collision-free round-robin schedule;
        2. INTERVIEW_PROPOSE   (workers propose to scheduled firms);
        3. INTERVIEW_RESPOND   (firms accept scheduled workers);
        4. MATCH_PROPOSE       (workers propose match to same firms);
        5. MATCH_RESPOND       (firms accept match);
        6. record CA-ETC samples for scheduled pairs that are tentatively
           matched -- this happens BEFORE retention so that any subsequent
           dissolution does not affect preference estimation;
        7. RETENTION with NOOP/release so agents are freed for the next round.
        """
        scheduled_firm = self.round_robin_schedule(self.total_explore_rounds)

        # --- Phase 1: INTERVIEW_PROPOSE
        worker_action = np.zeros(self.Nw, dtype=int)
        firm_action = np.zeros(self.Nf, dtype=int)
        for i in range(self.Nw):
            j = scheduled_firm[i]
            if j is not None:
                worker_action[i] = j + 1
        state = self.step_env(state, worker_action, firm_action)

        # --- Phase 2: INTERVIEW_RESPOND
        worker_action = np.zeros(self.Nw, dtype=int)
        firm_action = np.zeros(self.Nf, dtype=int)
        for i in range(self.Nw):
            j = scheduled_firm[i]
            if j is not None:
                firm_action[j] = i + 1
        state = self.step_env(state, worker_action, firm_action)

        # --- Phase 3: MATCH_PROPOSE
        worker_action = np.zeros(self.Nw, dtype=int)
        firm_action = np.zeros(self.Nf, dtype=int)
        for i in range(self.Nw):
            j = scheduled_firm[i]
            if j is not None:
                worker_action[i] = j + 1
        state = self.step_env(state, worker_action, firm_action)

        # --- Phase 4: MATCH_RESPOND
        worker_action = np.zeros(self.Nw, dtype=int)
        firm_action = np.zeros(self.Nf, dtype=int)
        for i in range(self.Nw):
            j = scheduled_firm[i]
            if j is not None:
                firm_action[j] = i + 1
        state = self.step_env(state, worker_action, firm_action)

        # --- Record CA-ETC samples (post MATCH_RESPOND, pre RETENTION).
        # Reading state.tentative_matched here is observation only -- we do
        # not modify any env state.
        tentative = np.asarray(state.tentative_matched)
        x_arr = np.asarray(state.x)
        y_arr = np.asarray(state.y)
        hat_x_arr = np.asarray(state.hat_x)
        hat_y_arr = np.asarray(state.hat_y)

        for i in range(self.Nw):
            j = scheduled_firm[i]
            if j is None:
                continue
            if bool(tentative[i, j]):
                worker_reward = float(np.dot(x_arr[i], hat_y_arr[i, j]))
                firm_reward = float(np.dot(hat_x_arr[i, j], y_arr[j]))
                self.update_worker_mean(i, j, worker_reward)
                self.update_firm_mean(i, j, firm_reward)
            # If the scheduled pair is not in tentative_matched (e.g. the
            # env rejected the proposal due to capacity / mode constraints),
            # we deliberately do NOT increase counts -- the pair was not
            # actually pulled.

        # --- Phase 5: RETENTION (release everything).
        # NOOP for both sides dissolves any tentative match so the next
        # exploration round can re-pair workers and firms freely. This does
        # NOT affect CA-ETC sample collection: the reward sample for (i, j)
        # was recorded immediately after MATCH_RESPOND above.
        worker_action = np.zeros(self.Nw, dtype=int)
        firm_action = np.zeros(self.Nf, dtype=int)
        state = self.step_env(state, worker_action, firm_action)

        self.total_explore_rounds += 1
        self.market_period += 1
        return state

    # ---------------- exploration block ----------------

    def exploration_block(
        self,
        state,
        epoch_id: int,
        exploration_market_periods: int,
    ):
        for _ in range(exploration_market_periods):
            if self.done_or_horizon_reached(state):
                break
            state = self.run_one_exploration_market_period(state)
        return state

    # ---------------- preference estimation ----------------

    def estimate_preferences(self) -> Tuple[np.ndarray, np.ndarray]:
        worker_preferences = np.zeros((self.Nw, self.Nf), dtype=int)
        firm_preferences = np.zeros((self.Nf, self.Nw), dtype=int)

        # Track who actually has LCB-UCB-separated confidence intervals at the
        # current exploration count. ``log_epoch`` reads these to print a
        # per-epoch summary (``X/Nw workers separated, Y/Nf firms separated``).
        # When the separation check fails we fall back to the mean-sorted
        # permutation (``naive``) rather than the identity. Identity is
        # essentially random and historically caused regret to keep climbing
        # in small markets; the naive sort is the maximum-likelihood ranking
        # under zero-mean noise and is strictly more informative.
        n_w_separated = 0
        n_f_separated = 0

        for i in range(self.Nw):
            mu = self.worker_mean_rewards[i]
            counts = self.worker_counts[i]
            naive = np.argsort(-mu)
            if self.use_confidence_check and self.total_explore_rounds > 1:
                sep = self._separated_perm(mu, counts)
                if sep is not None:
                    worker_preferences[i] = sep
                    n_w_separated += 1
                else:
                    worker_preferences[i] = naive
            else:
                worker_preferences[i] = naive

        for j in range(self.Nf):
            eta = self.firm_mean_rewards[j]
            counts = self.firm_counts[j]
            naive = np.argsort(-eta)
            if self.use_confidence_check and self.total_explore_rounds > 1:
                sep = self._separated_perm(eta, counts)
                if sep is not None:
                    firm_preferences[j] = sep
                    n_f_separated += 1
                else:
                    firm_preferences[j] = naive
            else:
                firm_preferences[j] = naive

        self._last_n_w_separated = n_w_separated
        self._last_n_f_separated = n_f_separated
        return worker_preferences, firm_preferences

    def _separated_perm(
        self,
        means: np.ndarray,
        counts: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Return a permutation s.t. for adjacent ranks LCB[top] > UCB[next].

        Uses the CA-ETC bonus radius sqrt(2 * log(total_explore_rounds) / n)
        with n clamped to >= 1 to avoid divide-by-zero on un-pulled arms.
        Returns ``None`` when no separated ranking exists.
        """
        n = np.maximum(counts, 1)
        log_factor = max(np.log2(max(self.total_explore_rounds, 2)), 1e-9)
        radius = self.radius_scale * np.sqrt(2.0 * log_factor / n)
        # Un-pulled arms cannot be separated: keep them as ambiguous.
        radius = np.where(counts >= 1, radius, np.inf)
        ucb = means + radius
        lcb = means - radius

        # Try the natural mean-sorted permutation first; fall back to brute
        # force across all permutations of the K items.
        naive = tuple(int(x) for x in np.argsort(-means))
        if self._is_separated(naive, lcb, ucb):
            return np.asarray(naive, dtype=int)
        K = len(means)
        for perm in permutations(range(K)):
            if self._is_separated(perm, lcb, ucb):
                return np.asarray(perm, dtype=int)
        return None

    @staticmethod
    def _is_separated(perm, lcb, ucb) -> bool:
        for k in range(len(perm) - 1):
            if not (lcb[perm[k]] > ucb[perm[k + 1]]):
                return False
        return True

    # ---------------- worker-proposing Gale-Shapley ----------------

    def gale_shapley(
        self,
        worker_preferences: np.ndarray,
        firm_preferences: np.ndarray,
    ) -> Dict[int, int]:
        """Worker-proposing GS. Returns ``{worker_id: firm_id}`` for matched workers.

        Pure computation -- does not touch the environment.
        """
        Nw, Nf = self.Nw, self.Nf

        # firm_rank[j, w] = position of worker w in firm j's preference list
        # (lower rank = more preferred).
        firm_rank = np.full((Nf, Nw), Nw, dtype=int)
        for j in range(Nf):
            for rank, w in enumerate(firm_preferences[j]):
                firm_rank[j, int(w)] = rank

        next_idx = np.zeros(Nw, dtype=int)
        worker_to_firm: Dict[int, int] = {}
        firm_to_worker: Dict[int, int] = {}

        free = list(range(Nw))
        while free:
            i = free.pop(0)
            while next_idx[i] < Nf:
                j = int(worker_preferences[i, next_idx[i]])
                next_idx[i] += 1
                if j not in firm_to_worker:
                    firm_to_worker[j] = i
                    worker_to_firm[i] = j
                    break
                cur = firm_to_worker[j]
                if firm_rank[j, i] < firm_rank[j, cur]:
                    firm_to_worker[j] = i
                    worker_to_firm[i] = j
                    if cur in worker_to_firm:
                        del worker_to_firm[cur]
                    free.append(cur)
                    break
                # else firm rejects; loop continues
            # if next_idx[i] == Nf and worker has no firm, leave unmatched

        return worker_to_firm

    # ---------------- one exploitation market period ----------------

    def run_one_exploitation_market_period(self, state, worker_to_firm: Dict[int, int]):
        """Drive one HireRL market period that establishes / retains the GS matching.

        Does NOT update worker_mean_rewards / firm_mean_rewards / counts.
        Does NOT mutate env state directly.

        Per the paper spec, the exploitation block does not perform interviews.
        The T0 >= ceil(Nw/Nf) * Nf invariant (enforced in __init__) guarantees
        every (i, j) pair has been interviewed during epoch 1's exploration
        block, so INTERVIEW_PROPOSE / INTERVIEW_RESPOND always send NOOP.

        Logic per phase:
          * INTERVIEW_PROPOSE  - NOOP (no new interviews in exploitation).
          * INTERVIEW_RESPOND  - NOOP.
          * MATCH_PROPOSE      - workers propose to assigned firms when not
            already matched.
          * MATCH_RESPOND      - firms accept assigned workers.
          * RETENTION          - both sides retain (action = 1) on assigned
            pairs that are matched or tentatively matched.
        """
        zeros_w = np.zeros(self.Nw, dtype=int)
        zeros_f = np.zeros(self.Nf, dtype=int)

        # Phase 1: INTERVIEW_PROPOSE -- NOOP.
        state = self.step_env(state, zeros_w, zeros_f)

        # Phase 2: INTERVIEW_RESPOND -- NOOP.
        state = self.step_env(state, zeros_w, zeros_f)

        # --- Phase 3: MATCH_PROPOSE
        matched = np.asarray(state.matched)
        worker_action = np.zeros(self.Nw, dtype=int)
        firm_action = np.zeros(self.Nf, dtype=int)
        for i, j in worker_to_firm.items():
            if bool(matched[i, j]):
                continue
            if bool(matched[i].any()):
                continue
            if bool(matched[:, j].any()):
                continue
            worker_action[i] = j + 1
        state = self.step_env(state, worker_action, firm_action)

        # --- Phase 4: MATCH_RESPOND
        matched = np.asarray(state.matched)
        worker_action = np.zeros(self.Nw, dtype=int)
        firm_action = np.zeros(self.Nf, dtype=int)
        for i, j in worker_to_firm.items():
            if bool(matched[i, j]):
                continue
            if bool(matched[i].any()):
                continue
            if bool(matched[:, j].any()):
                continue
            firm_action[j] = i + 1
        state = self.step_env(state, worker_action, firm_action)

        # --- Phase 5: RETENTION
        # Retain assigned pairs that are matched (already retained) or
        # tentative_matched (just formed this period).
        matched = np.asarray(state.matched)
        tentative = np.asarray(state.tentative_matched)
        worker_action = np.zeros(self.Nw, dtype=int)
        firm_action = np.zeros(self.Nf, dtype=int)
        for i, j in worker_to_firm.items():
            if bool(matched[i, j]) or bool(tentative[i, j]):
                worker_action[i] = 1
                firm_action[j] = 1
        state = self.step_env(state, worker_action, firm_action)

        self.market_period += 1
        return state

    def exploitation_block(
        self,
        state,
        worker_to_firm: Dict[int, int],
        exploitation_market_periods: int,
    ):
        for _ in range(exploitation_market_periods):
            if self.done_or_horizon_reached(state):
                break
            state = self.run_one_exploitation_market_period(state, worker_to_firm)
        return state

    # ---------------- horizon ----------------

    def done_or_horizon_reached(self, state) -> bool:
        if self.market_period >= self.horizon:
            return True
        if self.last_done is not None:
            done_w, done_f = self.last_done
            if bool(jnp.any(done_w)) or bool(jnp.any(done_f)):
                return True
        return False

    # ---------------- epoch sizing ----------------

    def get_epoch_market_periods(self, epoch: int, state) -> Tuple[int, int]:
        c = 2
        b = 2 ** (1 / self.gamma)

        exploration_market_periods = int(self.T0 * (epoch ** c))
        total_market_periods = int(self.T0 * (epoch ** b))
        exploitation_market_periods = max(
            0, total_market_periods - exploration_market_periods
        )

        # Track horizon via baseline counter. state.market_t works too, but
        # gets reset to 0 by the env's auto-reset at the horizon boundary.
        remaining = self.horizon - self.market_period
        exploration_market_periods = max(0, min(exploration_market_periods, remaining))
        remaining -= exploration_market_periods
        exploitation_market_periods = max(0, min(exploitation_market_periods, remaining))

        return exploration_market_periods, exploitation_market_periods

    # ---------------- main run ----------------

    def run(self, initial_state):
        state = initial_state
        epoch = 1
        while not self.done_or_horizon_reached(state):
            exp_periods, exploit_periods = self.get_epoch_market_periods(epoch, state)
            if exp_periods == 0 and exploit_periods == 0:
                break

            state = self.exploration_block(state, epoch, exp_periods)

            worker_prefs, firm_prefs = self.estimate_preferences()
            worker_to_firm = self.gale_shapley(worker_prefs, firm_prefs)

            state = self.exploitation_block(state, worker_to_firm, exploit_periods)

            self.log_epoch(epoch, worker_prefs, firm_prefs, worker_to_firm)
            epoch += 1

        return state, self.metrics

    def log_epoch(
        self,
        epoch: int,
        worker_prefs: np.ndarray,
        firm_prefs: np.ndarray,
        worker_to_firm: Dict[int, int],
    ) -> None:
        self.metrics["epoch"].append(int(epoch))
        self.metrics["market_period"].append(int(self.market_period))
        self.metrics["worker_counts"].append(self.worker_counts.copy())
        self.metrics["firm_counts"].append(self.firm_counts.copy())
        self.metrics["worker_mean_rewards"].append(self.worker_mean_rewards.copy())
        self.metrics["firm_mean_rewards"].append(self.firm_mean_rewards.copy())
        self.metrics["worker_preferences"].append(worker_prefs.copy())
        self.metrics["firm_preferences"].append(firm_prefs.copy())
        self.metrics["matchings"].append(dict(worker_to_firm))

        # LCB/UCB separation summary. When the count is < Nw (or < Nf) the
        # remaining agents fell back to the mean-sorted permutation, which
        # is the maximum-likelihood ranking under zero-mean noise — far better
        # than the old identity fallback, but still subject to ranking errors
        # when sample noise is comparable to mean gaps.
        nw_sep = getattr(self, "_last_n_w_separated", -1)
        nf_sep = getattr(self, "_last_n_f_separated", -1)
        # Min counts give a quick read on whether more exploration would help.
        wc_min = int(self.worker_counts.min())
        fc_min = int(self.firm_counts.min())
        print(
            f"  [epoch {epoch}] separated {nw_sep}/{self.Nw} workers, "
            f"{nf_sep}/{self.Nf} firms  "
            f"(t_explore={self.total_explore_rounds}, "
            f"min worker_count={wc_min}, min firm_count={fc_min})"
        )
