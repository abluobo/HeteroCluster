# Feature-Driven Graph Reconstruction for Modularity-Based Clustering on Heterophilic Graphs

Official implementation for:

> **Feature-Driven Graph Reconstruction for Modularity-Based Clustering on Heterophilic Graphs**  
> *Submitted to Neural Computing & Applications (NCA)* — Manuscript **NCAA-D-26-03736** (2026-06-30)

Paper: modularity methods (DGCluster, DMoN) fail on heterophilic graphs because **B = A − dd^T/(2|E|)** collapses when structural homophily is low. We replace the structural adjacency with a **feature k-NN graph** (drop-in, no architecture change) and use the **Δ criterion** (Δ = h_kNN − h_struct) to predict when substitution helps.

## Requirements

Install [PyTorch](https://pytorch.org/get-started/locally/) for your platform, then:

```bash
pip install -r requirements.txt
```

## Quick start

**DGCluster + feature kNN (main method):**

```bash
# Structural baseline
python main_hetero.py --dataset texas --knn 0 --cluster kmeans --seed 0

# Ours — paper default k=10 for DGCluster+kNN
python main_hetero.py --dataset texas --knn 10 --cluster kmeans --seed 0
```

**DMoN + feature kNN:**

```bash
python dmon_hetero.py --dataset texas --knn 0 --seed 0   # baseline
python dmon_hetero.py --dataset texas --knn 5 --seed 0   # ours (k=5 in paper)
```

**Δ criterion (seconds, before training):**

```bash
python compute_delta.py --dataset texas --k 10
python compute_delta.py --dataset chameleon --k 5
```

Supported datasets in `main_hetero.py`: `texas`, `cornell`, `wisconsin`, `chameleon`, `actor`, `cora` (and others). DMoN script supports WebKB + Cora/Citeseer.

## Reproduce paper results

```bash
bash run.sh
```

Checkpoints are written to `results/`. Pre-computed summaries:

| File | Contents |
|------|----------|
| `table1_full.csv` | WebKB main results (struct / kNN-5 / kNN-10 / DMoN), all seeds |
| `k_sensitivity.csv` | k ∈ {3,5,10,15,20} on WebKB |
| `negcontrol.csv` | Chameleon, Actor, Cora (struct vs kNN-5) |

Data download automatically to `data/` on first run (PyTorch Geometric).

## Main results (Table 1, k=10 for DGC+kNN, 5 seeds, mean NMI)

| Dataset | DGCluster | DMoN | Raw KMeans* | **DGC+kNN (k=10)** | **DMoN+kNN (k=5)** |
|---------|-----------|------|-------------|--------------------|--------------------|
| Texas | 0.084 | 0.182 | 0.343 | 0.302 | 0.287 |
| Cornell | 0.058 | 0.164 | 0.337 | **0.470** | 0.326 |
| Wisconsin | 0.068 | 0.229 | 0.429 | 0.406 | 0.382 |

\*Raw KMeans: L2-normalized features, same K as num_classes (not included in scripts; values from paper).

Full NMI/ARI/ACC per seed: `table1_full.csv`.

## Citation

```bibtex
@article{hetero_cluster_2026,
  title   = {Feature-Driven Graph Reconstruction for Modularity-Based Clustering on Heterophilic Graphs},
  author  = {Yin, Banghui},
  year    = {2026},
  note    = {Under review, Neural Computing \& Applications}
}
```

## License

MIT (see repository license file if present).
