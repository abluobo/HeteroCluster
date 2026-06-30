#!/bin/bash
# Reproduce paper experiments (Neural Computing & Applications submission).
# Default: k=10 for DGCluster+kNN (Table 1); k=5 for DMoN+kNN; 5 seeds each.

set -e

echo "=== WebKB: DGCluster structural baseline (knn=0) ==="
for dataset in texas cornell wisconsin; do
  for seed in 0 1 2 3 4; do
    python main_hetero.py --dataset "$dataset" --knn 0 --cluster kmeans --seed "$seed"
  done
done

echo "=== WebKB: DGCluster + feature kNN (k=10, main table) ==="
for dataset in texas cornell wisconsin; do
  for seed in 0 1 2 3 4; do
    python main_hetero.py --dataset "$dataset" --knn 10 --cluster kmeans --seed "$seed"
  done
done

echo "=== WebKB: DMoN baseline and +kNN (k=5) ==="
for dataset in texas cornell wisconsin; do
  for seed in 0 1 2 3 4; do
    python dmon_hetero.py --dataset "$dataset" --knn 0 --seed "$seed"
    python dmon_hetero.py --dataset "$dataset" --knn 5 --seed "$seed"
  done
done

echo "=== Negative controls: Chameleon, Actor, Cora (k=5) ==="
for dataset in chameleon actor cora; do
  for seed in 0 1 2 3 4; do
    python main_hetero.py --dataset "$dataset" --knn 0 --cluster kmeans --seed "$seed"
    python main_hetero.py --dataset "$dataset" --knn 5 --cluster kmeans --seed "$seed"
  done
done

echo "=== k-sensitivity: k in {3,5,10,15,20} on WebKB ==="
for k in 3 5 10 15 20; do
  for dataset in texas cornell wisconsin; do
    for seed in 0 1 2 3 4; do
      python main_hetero.py --dataset "$dataset" --knn "$k" --cluster kmeans --seed "$seed"
    done
  done
done

echo "Done. See table1_full.csv, k_sensitivity.csv, negcontrol.csv for logged summaries."
