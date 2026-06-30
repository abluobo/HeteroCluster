#!/usr/bin/env python3
"""Print label-based Δ = h(kNN) - h(struct) before training (Δ criterion)."""

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F

# Monorepo layout: data/ lives next to release/; standalone clone uses ./data/
_here = Path(__file__).resolve().parent
if (_here.parent / "data").is_dir() and not (_here / "data").is_dir():
    os.chdir(_here.parent)

from main_hetero import load_dataset


def edge_homophily(labels, edge_index):
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    return float((labels[src] == labels[dst]).mean())


def knn_homophily(data, k, labels):
    x = F.normalize(data.x.float(), dim=1)
    sim = x @ x.t()
    sim.fill_diagonal_(-1.0)
    _, idx = sim.topk(k, dim=1)
    same = 0
    total = 0
    for i in range(data.num_nodes):
        for j in idx[i].tolist():
            same += int(labels[i] == labels[j])
            total += 1
    return same / max(total, 1)


def main():
    p = argparse.ArgumentParser(description="Compute Δ criterion (label-based homophily gap)")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--k", type=int, default=10, help="k for feature kNN graph")
    args = p.parse_args()

    data = load_dataset(args.dataset)[0]
    labels = data.y.flatten().cpu().numpy()
    h_struct = edge_homophily(labels, data.edge_index)
    h_knn = knn_homophily(data, args.k, labels)
    delta = h_knn - h_struct

    print(f"dataset={args.dataset}  k={args.k}")
    print(f"h_struct = {h_struct:.3f}")
    print(f"h_knn    = {h_knn:.3f}")
    print(f"Δ        = {delta:+.3f}")
    if delta > 0.3:
        print("→ apply kNN substitution")
    elif delta < 0.1:
        print("→ keep structural graph")
    else:
        print("→ marginal / dataset-dependent")


if __name__ == "__main__":
    main()
