#!/usr/bin/env python3
"""Regenerate table1_full.csv, k_sensitivity.csv (copy), negcontrol.csv from results/*.pt."""

import csv
import glob
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load_pt(path):
    return torch.load(path, map_location="cpu", weights_only=False)["metrics"]


def write_table1():
    rows = []
    for ds in ("texas", "cornell", "wisconsin"):
        for seed in range(5):
            p = RESULTS / f"{ds}_gcn_lam0.0_beta0.0_knn0_300_seed{seed}.pt"
            if p.exists():
                m = load_pt(p)
                rows.append([ds, "dgc_struct", 0, seed, m["NMI"], m.get("ARI", ""), m.get("ACC", "")])
        for knn in (5, 10):
            for seed in range(5):
                p = RESULTS / f"{ds}_gcn_lam0.0_beta0.0_knn{knn}_300_seed{seed}.pt"
                if p.exists():
                    m = load_pt(p)
                    rows.append([ds, f"dgc_knn{knn}", knn, seed, m["NMI"], m.get("ARI", ""), m.get("ACC", "")])
        for seed in range(5):
            for knn, tag in ((0, "dmon_struct"), (5, "dmon_knn5")):
                p = RESULTS / f"dmon_{ds}_knn{knn}_300_seed{seed}.pt"
                if p.exists():
                    m = load_pt(p)
                    rows.append([ds, tag, knn, seed, m["NMI"], m.get("ARI", ""), m.get("ACC", "")])
    out = ROOT / "table1_full.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "knn", "seed", "nmi", "ari", "acc"])
        w.writerows(rows)
    print(f"Wrote {out} ({len(rows)} rows)")


def write_negcontrol():
    rows = []
    for ds in ("chameleon", "actor", "cora"):
        for knn in (0, 5):
            for p in sorted(RESULTS.glob(f"{ds}_gcn_lam0.0_beta0.0_knn{knn}_300_seed*.pt")):
                seed = int(p.stem.split("seed")[-1])
                m = load_pt(p)
                tag = "dgc_struct" if knn == 0 else "dgc_knn5"
                rows.append([ds, tag, knn, seed, m["NMI"]])
    out = ROOT / "negcontrol.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "knn", "seed", "nmi"])
        w.writerows(rows)
    print(f"Wrote {out} ({len(rows)} rows)")


def write_k_sensitivity():
    src = ROOT.parent / "results" / "k_sensitivity.csv"
    dst = ROOT / "k_sensitivity.csv"
    if src.exists():
        dst.write_text(src.read_text())
        print(f"Copied {src.name} → {dst}")
    else:
        print("Skip k_sensitivity: run k-sensitivity experiments first")


if __name__ == "__main__":
    write_table1()
    write_negcontrol()
    write_k_sensitivity()
