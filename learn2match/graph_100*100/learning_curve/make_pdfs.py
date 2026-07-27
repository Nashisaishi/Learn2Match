"""Render 4 learning-curve PDFs from the multi-seed CSVs sitting next to this script.

Reads:
    worker_regret.csv
    firm_regret.csv
    social_welfare.csv
    friction_loss.csv

Each CSV has columns ``env_step, ppo_mean, ppo_std, ppo_ci_lo, ppo_ci_hi``.
The CI is across 10 train seeds (training-variance) at every env-step snapshot.

Writes a ``.pdf`` next to each ``.csv`` with the same plot style as the W&B
in-training learning curves: Learn2Match mean line + 95% CI band.

Usage:
    cd learn2match/learning_curve
    python3 make_pdfs.py            # writes 4 PDFs into this dir
    python3 make_pdfs.py --out_dir /tmp/pdfs   # or elsewhere

Dependencies: pandas, matplotlib.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

_HERE = Path(__file__).resolve().parent

# Times New Roman everywhere with serif fallbacks if the font is unavailable.
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size":         30,
    "axes.titlesize":    25,
    "axes.labelsize":    12,
    "xtick.labelsize":   25,
    "ytick.labelsize":   25,
    "legend.fontsize":   20,
})

# Number of market periods per eval episode for this run (= --num_periods).
# Used to convert per-period quantities to cumulative-over-T quantities so the
# regret panels match the \sum_{t=1}^T(...) form in the CA-ETC-style bound.
# If you re-run with a different --num_periods, update this constant.
T_PERIODS = 800

# (csv_filename, pdf_filename, ylabel, title, scale) -- kept in plot order.
# ``scale`` multiplies all four CSV columns (ppo_mean / ppo_std / ppo_ci_lo /
# ppo_ci_hi) before plotting; the cross-seed CI scales linearly with it.
SPECS = [
    (
        "worker_regret.csv",
        "worker_regret.pdf",
        f"cumulative worker regret over {T_PERIODS} periods",
        "Worker Regret",
        T_PERIODS,
    ),
    (
        "firm_regret.csv",
        "firm_regret.pdf",
        f"cumulative firm regret over {T_PERIODS} periods",
        "Firm Regret",
        T_PERIODS,
    ),
    (
        "social_welfare.csv",
        "social_welfare.pdf",
        f"cumulative social welfare over {T_PERIODS} periods",
        "Social Welfare",
        T_PERIODS,
    ),
    (
        "friction_loss.csv",
        "friction_loss.pdf",
        f"cumulative friction loss over {T_PERIODS} periods",
        "Friction Loss",
        T_PERIODS,
    ),
]


def _plot_one(
    df: pd.DataFrame,
    title: str,
    ylabel: str,
    ppo_label: str = "PPO",
) -> plt.Figure:
    t = df["env_step"].to_numpy()
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    line, = ax.plot(t, df["ppo_mean"].to_numpy(),
                    label=ppo_label, color="C1", linewidth=1.8)
    ax.fill_between(
        t,
        df["ppo_ci_lo"].to_numpy(),
        df["ppo_ci_hi"].to_numpy(),
        color=line.get_color(), alpha=0.18,
    )

    ax.set_title(title)
    ax.set_xlabel("env steps")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv_dir", type=Path, default=_HERE,
                        help="Directory containing the 4 CSVs (default: this script's dir).")
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="Where to write the PDFs (default: same as --csv_dir).")
    args = parser.parse_args()

    csv_dir = args.csv_dir
    out_dir = args.out_dir or csv_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for csv_name, pdf_name, ylabel, title, scale in SPECS:
        csv_path = csv_dir / csv_name
        if not csv_path.is_file():
            print(f"  [skip] missing {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        if scale != 1:
            for col in ("ppo_mean", "ppo_std", "ppo_ci_lo", "ppo_ci_hi"):
                df[col] = df[col] * scale
        fig = _plot_one(df, title=title, ylabel=ylabel)
        pdf_path = out_dir / pdf_name
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {pdf_path}  (scale={scale})")


if __name__ == "__main__":
    main()
