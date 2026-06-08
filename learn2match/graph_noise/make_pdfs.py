"""Render 4 PDFs from the multi-seed CSVs sitting next to this script.

Reads:
    cumulative_worker_regret.csv
    cumulative_firm_regret.csv
    cumulative_friction_loss.csv
    cumulative_social_welfare.csv

Each CSV has columns ``t, ppo_mean, ppo_std, ppo_ci_lo, ppo_ci_hi, caetc_mean``
(the SW one additionally has ``da_ref_mean`` and ``planner_first-best_mean``).

Writes a ``.pdf`` next to each ``.csv`` with the same plot style as the W&B
figures: PPO mean + 95% CI band, CA-ETC mean as a dashed line, plus DA ref
and planner as dotted reference lines on the SW plot.

Usage:
    cd learn2match/graph_noise
    python3 make_pdfs.py            # writes 4 PDFs into this dir
    python3 make_pdfs.py --out_dir /tmp/pdfs   # or elsewhere

Dependencies: pandas, numpy, matplotlib.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent

# Times New Roman everywhere with serif fallbacks if the font is unavailable on
# the system. mathtext.fontset = 'stix' keeps math glyphs (the ± in the title)
# visually consistent with Times.
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size":         30,
    "axes.titlesize":    25,
    "axes.labelsize":    25,
    "xtick.labelsize":   25,
    "ytick.labelsize":   25,
    "legend.fontsize":   16,
})

# (csv_filename, pdf_filename, ylabel, title) -- kept in plot order.
SPECS = [
    (
        "cumulative_worker_regret.csv",
        "cumulative_worker_regret.pdf",
        "cumulative worker regret",
        "Cumulative Worker Regret",
    ),
    (
        "cumulative_firm_regret.csv",
        "cumulative_firm_regret.pdf",
        "cumulative firm regret",
        "Cumulative Firm Regret",
    ),
    (
        "cumulative_friction_loss.csv",
        "cumulative_friction_loss.pdf",
        "cumulative friction loss",
        "Cumulative Friction Loss",
    ),
    (
        "cumulative_social_welfare.csv",
        "cumulative_social_welfare.pdf",
        "cumulative social welfare",
        "Cumulative Social Welfare",
    ),
]


def _plot_one(
    df: pd.DataFrame,
    title: str,
    ylabel: str,
    refs: Optional[List[Tuple[str, str, str, str]]] = None,
    ppo_label: str = "PPO",
    caetc_label: str = "CA-ETC",
) -> plt.Figure:
    """``refs``: list of ``(column_name, label, color, linestyle)`` for
    reference lines (no CI band). Used only on the SW plot.
    """
    t = df["t"].to_numpy()
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    line, = ax.plot(t, df["ppo_mean"].to_numpy(), label=ppo_label,
                    color="C1", linewidth=1.8)
    ax.fill_between(
        t,
        df["ppo_ci_lo"].to_numpy(),
        df["ppo_ci_hi"].to_numpy(),
        color=line.get_color(), alpha=0.18,
        label=f"_{ppo_label} 95% CI",
    )

    if "caetc_mean" in df.columns:
        ax.plot(t, df["caetc_mean"].to_numpy(),
                label=caetc_label, color="C0",
                linewidth=1.8, linestyle="--")

    for col, label, color, linestyle in (refs or []):
        if col in df.columns:
            ax.plot(t, df[col].to_numpy(),
                    label=label, color=color,
                    linewidth=1.8, linestyle=linestyle)

    ax.set_title(f"{title}")
    ax.set_xlabel("market period")
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

    sw_refs = [
        ("da_ref_mean", "Worker-Optimal Stable Matching", "green", "-"),
    ]

    for csv_name, pdf_name, ylabel, title in SPECS:
        csv_path = csv_dir / csv_name
        if not csv_path.is_file():
            print(f"  [skip] missing {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        refs = sw_refs if csv_name == "cumulative_social_welfare.csv" else None
        fig = _plot_one(df, title=title, ylabel=ylabel, refs=refs)
        pdf_path = out_dir / pdf_name
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {pdf_path}")


if __name__ == "__main__":
    main()
