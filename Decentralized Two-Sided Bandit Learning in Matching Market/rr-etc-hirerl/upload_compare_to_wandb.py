"""Upload an existing compare_rr_vs_ca.py output folder to W&B — no re-run.

The comparison script saves full-resolution results to --outdir regardless of
W&B; this tool replays exactly what the live integration would have uploaded
(same metric keys, downsampling, figure, files, summary), sourcing everything
from disk. Useful for runs finished before the W&B integration existed, or
runs done with --no-wandb / offline.

Usage:
    python upload_compare_to_wandb.py <outdir> \
        [--run-name NAME] [--wandb-entity E] [--wandb-project P]
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import wandb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("outdir", help="folder written by compare_rr_vs_ca.py")
    ap.add_argument("--wandb-entity", default="haijingzong-university-of-washington")
    ap.add_argument("--wandb-project", default="hireRL")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    frames = {}
    for side in ("worker", "firm"):
        p = outdir / f"cumulative_{side}_regret.csv"
        if p.exists():
            frames[side] = pd.read_csv(p)
    if not frames:
        raise SystemExit(f"[upload] no cumulative_*_regret.csv found in {outdir}")

    summary_path = outdir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    cfg = summary.get("args", {})

    run_name = args.run_name or (
        f"rr-vs-ca-upload-{outdir.name}-{datetime.now():%Y%m%d-%H%M%S}"
    )
    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=run_name,
        config={"algorithm": "rr_etc_vs_ca_etc_paired",
                "uploaded_from": str(outdir.resolve()), **cfg},
    )

    # Same downsampling and metric keys as compare_rr_vs_ca._log_wandb, so
    # replayed runs overlay cleanly with live ones in the W&B UI.
    T = min(len(df) for df in frames.values())
    stride = max(1, T // 5000)
    idxs = list(range(0, T, stride))
    if idxs[-1] != T - 1:
        idxs.append(T - 1)
    t_col = next(iter(frames.values()))["t"].to_numpy()
    for i in idxs:
        log = {}
        for side, df in frames.items():
            row = df.iloc[i]
            log[f"rr/cum_regret_{side}_mean"] = float(row["rr_mean"])
            log[f"rr/cum_regret_{side}_std"] = float(row["rr_std"])
            log[f"ca/cum_regret_{side}_mean"] = float(row["ca_mean"])
            log[f"ca/cum_regret_{side}_std"] = float(row["ca_std"])
        wandb.log(log, step=int(t_col[i]))

    png = outdir / "compare_cumregret.png"
    if png.exists():
        wandb.log({"compare_cumregret": wandb.Image(str(png))})
    for fname in ("cumulative_worker_regret.csv",
                  "cumulative_firm_regret.csv", "summary.json"):
        p = outdir / fname
        if p.exists():
            wandb.save(str(p), base_path=str(outdir), policy="now")

    final = {}
    for side, stats in summary.get("final_cumulative_regret", {}).items():
        for k, v in stats.items():
            algo, _, tail = k.partition("_final_")
            final[f"final/{side}_{algo}_{tail}"] = v
    rr_info = summary.get("rr", {})
    if "n_all_settled" in rr_info:
        final["rr_n_all_settled"] = rr_info["n_all_settled"]
    if "n_correct" in rr_info:
        final["rr_n_correct"] = rr_info["n_correct"]
    run.summary.update(final)
    wandb.finish()
    print(f"[upload] W&B run uploaded: {run_name} "
          f"({len(idxs)} logged steps from {T} periods)")


if __name__ == "__main__":
    main()
