"""HireRL environment configuration."""

from dataclasses import dataclass

from .constants import (
    INTERVIEW_MODE_EXCLUSIVE,
    RECEIVER_FIRM_FIRST,
    VISIBILITY_MATCHED_STATUS,
)


@dataclass
class HireRLConfig:
    Nw: int = 2
    Nf: int = 2
    d: int = 1
    horizon: int = 10

    sigma_interview: float = 0.3
    sigma_match: float = 0.1
    lambda_reveal: float = 1.0

    interview_mode: str = INTERVIEW_MODE_EXCLUSIVE
    max_interview_proposals_per_agent: int = 1
    max_interview_acceptances_per_agent: int = 1
    max_interviews_per_agent_per_step: int = 1
    interview_tie_break: str = "deterministic_id"

    receiver_side_order: str = RECEIVER_FIRM_FIRST

    public_matching_visibility: str = VISIBILITY_MATCHED_STATUS

    outside_option: float = 0.0
    use_worker_proposing_DA_for_reference: bool = True

    # When True, x and y are sampled as |N(0, 1)| (half-normal) per
    # component, so every pair satisfies x[i] · y[j] >= 0 by construction
    # and the default outside_option=0 doesn't reject any pair. Use this
    # for paper-faithful CA-ETC runs (the original algorithm assumes all
    # mean rewards are non-negative). Default False keeps the standard
    # full-Gaussian features.
    non_negative_features: bool = False

    # Optimistic initialization for hat_x and hat_y at episode reset. Each
    # entry of both belief tensors is set to this constant per dimension, so
    # an unobserved pair (i, j) has perceived utility hat_init_value * sum(x[i])
    # (or sum(y[j]) on the firm side). Default 0.0 reproduces the historical
    # zero-init (pessimistic under non_negative_features=True). Useful values
    # under |N(0,1)| features: ~0.8 (neutral prior), ~2.0 (UCB-style 95th
    # percentile), >=3.0 (extreme optimism, may suppress settling). Applied
    # symmetrically to hat_x / hat_y so both sides have matching exploration
    # incentives.
    hat_init_value: float = 0.0

    # When True, matched workers / firms are allowed to propose and accept
    # *interviews* with other agents while remaining matched ("on-the-job
    # search"). Match-switching still requires explicit dissolution in the
    # retention phase first; this flag only relaxes the interview-side
    # action masks and transition validators. Useful for breaking the
    # exploration deadlock where high match rates keep INTERVIEW_PROPOSE
    # action masks degenerate (only NOOP allowed for matched agents).
    # Default False reproduces the original "only the unemployed search"
    # constraint.
    allow_on_the_job_search: bool = False

    # When True, hat_x and hat_y are initialized as per-pair noisy observations
    # of the true latent at episode reset:
    #   hat_x[i, j, :] = x[i] + N(0, sigma_init)
    #   hat_y[i, j, :] = y[j] + N(0, sigma_init)
    # with independent noise per (i, j). Every pair therefore enters the market
    # with a *distinguishable* (but noisy) prior, instead of the historical
    # constant-init in which all un-interviewed firms / workers look identical.
    # Motivation: relaxes the "first interview lock-in" / "no on-the-job search
    # signal" pathology of the constant init -- matched agents can now compare
    # current match against alternatives, and initial proposals can be informed.
    # Belief refinement stays monotone (sigma_init > sigma_interview > sigma_match),
    # so the existing ``~ever_matched`` interview-update mask is unchanged: the
    # first interview overwrites the noisy prior with a less-noisy observation,
    # and retention further refines it. Default False reproduces the historical
    # constant ``hat_init_value`` init (which then gates on hat_init_value).
    noisy_hat_init: bool = False

    # Standard deviation of the noisy hat init when noisy_hat_init=True.
    # Must be strictly greater than sigma_interview (enforced in __post_init__)
    # so the first interview is a strict information gain. Recommended
    # sigma_init >= 3 * sigma_interview to ensure the prior is meaningfully
    # noisier than any subsequent observation. Ignored when noisy_hat_init=False.
    sigma_init: float = 3.0

    # When True, the post-retention belief signal becomes a *public* signal.
    # Concretely: when pair (i, j) is retained, the single noisy realization
    #   signal_x_i = sigmoid(lambda_reveal * tenure_new[i, j]) * x[i] + eps_w[i, j]
    # is broadcast across all firms (hat_x[i, k, :] = signal_x_i for every k);
    # symmetrically signal_y_j is broadcast across all workers. Once worker i
    # has been retained anywhere, subsequent interviews no longer overwrite any
    # of hat_x[i, :, :] -- the public signal is strictly more informative than
    # an interview observation (sigma_match << sigma_interview, plus the
    # sigmoid factor approaches 1 with tenure). Symmetric on the firm side.
    # Default False preserves the original per-pair private-belief semantics
    # bit-exactly, including the per-pair ever_matched mask in the interview
    # phase. Should match between training and eval; mismatched flags shift
    # the obs distribution and degrade the policy OOD.
    public_retention_signal: bool = False

    # When False, ``HireRLEnv._step_impl`` skips ``compute_all_metrics`` entirely
    # and emits zero-valued ``info["settled_metrics"]``. compute_all_metrics
    # runs three Nw*Nf-iteration DA fori_loops per env step (regret, friction
    # loss, reference matching) and at Nw=Nf=100 it dominates env-step cost --
    # but ``train_hirerl_ippo.py`` does not read info anywhere, so during PPO
    # training the work is pure waste. Eval scripts (eval_and_plot.py,
    # train_hirerl_planner.py, friction_loss_demo.py) DO read settled_metrics,
    # so they need this True. Default True for backwards compatibility; flip
    # to False for training envs only.
    compute_settled_metrics: bool = True

    def __post_init__(self):
        # sigma_interview > sigma_match is required so that retention can be
        # asymptotically more accurate than interview as tenure accumulates.
        # If equal or reversed, retention always carries the sigmoid-bias
        # penalty without a compensating noise reduction.
        if self.sigma_interview < 0 or self.sigma_match < 0:
            raise ValueError(
                f"sigma_interview and sigma_match must be non-negative; "
                f"got sigma_interview={self.sigma_interview}, "
                f"sigma_match={self.sigma_match}."
            )
        if self.sigma_interview ** 2 <= self.sigma_match ** 2:
            raise ValueError(
                f"Require sigma_interview**2 > sigma_match**2 so retention "
                f"observations are eventually more accurate than interview "
                f"observations; got sigma_interview={self.sigma_interview}, "
                f"sigma_match={self.sigma_match}."
            )
        if self.noisy_hat_init:
            if self.sigma_init <= self.sigma_interview:
                raise ValueError(
                    f"When noisy_hat_init=True, sigma_init must be strictly "
                    f"greater than sigma_interview so the first interview is "
                    f"a strict information gain (recommended sigma_init >= "
                    f"3 * sigma_interview); got sigma_init={self.sigma_init}, "
                    f"sigma_interview={self.sigma_interview}."
                )


def smoke_config() -> HireRLConfig:
    return HireRLConfig(
        Nw=2, Nf=2, d=1, horizon=10,
        sigma_interview=0.3, sigma_match=0.1, lambda_reveal=1.0,
    )


def default_experiment_config() -> HireRLConfig:
    return HireRLConfig(
        Nw=10, Nf=10, d=3, horizon=100,
        sigma_interview=0.7, sigma_match=0.4, lambda_reveal=1.0,
    )
