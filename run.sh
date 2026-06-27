#!/bin/bash
# Reproduce main results (Table 1): DGCluster + kNN-5 on WebKB datasets
for dataset in texas cornell wisconsin; do
    python main_hetero.py --dataset $dataset --knn 5 --num_seeds 5
done

# DMoN + kNN-5
for dataset in texas cornell wisconsin; do
    python dmon_hetero.py --dataset $dataset --knn 5 --num_seeds 5
done

# k-sensitivity (k = 3, 5, 10, 15, 20)
for k in 3 5 10 15 20; do
    for dataset in texas cornell wisconsin; do
        python main_hetero.py --dataset $dataset --knn $k --num_seeds 5
    done
done
