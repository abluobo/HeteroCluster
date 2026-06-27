# Feature-Driven Graph Reconstruction for Modularity-Based Clustering on Heterophilic Graphs

This repository contains the code for the paper:

> **Feature-Driven Graph Reconstruction for Modularity-Based Clustering on Heterophilic Graphs**  
> *Under review*

## Overview

Modularity-based graph clustering methods (e.g., DGCluster, DMoN) fail on heterophilic graphs because the modularity matrix **B = A − dd^T/(2|E|)** degenerates to near-zero when structural homophily is low.

We propose replacing the structural adjacency with a **feature k-NN graph**, which restores the modularity objective on heterophilic graphs. The fix requires no change to the clustering algorithm itself and generalizes to any modularity-based method.

We further introduce the **Δ criterion** (Δ = h(kNN) − h(struct)) to predict when this replacement will be effective.

## Requirements

Install PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/), then:

```bash
pip install -r requirements.txt
```

## Usage

**DGCluster + feature-kNN graph:**

```bash
# Baseline (structural graph)
python main_hetero.py --dataset texas --knn 0

# Ours (feature kNN graph, k=5)
python main_hetero.py --dataset texas --knn 5

# Supported datasets: texas, cornell, wisconsin, chameleon, actor, cora
```

**DMoN + feature-kNN graph:**

```bash
python dmon_hetero.py --dataset texas --knn 0   # baseline
python dmon_hetero.py --dataset texas --knn 5   # ours
```

**Reproduce all results:**

```bash
bash run.sh
```

Data will be downloaded automatically to `data/` on first run.

## Results

Main results (NMI, 5 seeds):

| Dataset  | DGCluster | DMoN  | Raw KMeans | **Ours (DGCluster+kNN)** | **Ours (DMoN+kNN)** |
|----------|-----------|-------|------------|--------------------------|----------------------|
| Texas    | 0.084     | 0.182 | 0.343      | 0.279                    | 0.287                |
| Cornell  | 0.058     | 0.164 | 0.337      | **0.431**                | 0.326                |
| Wisconsin| 0.068     | 0.229 | 0.429      | 0.395                    | 0.382                |

Full results (NMI/ARI/ACC) are in `table1_full.csv`. k-sensitivity results are in `k_sensitivity.csv`.

## Citation

```bibtex
@article{hetero_cluster_2026,
  title   = {Feature-Driven Graph Reconstruction for Modularity-Based Clustering on Heterophilic Graphs},
  year    = {2026},
}
```
