#!/bin/bash
# Reproduce main results (Table 1): DGCluster + kNN-5 on WebKB datasets
for dataset in texas cornell wisconsin; do
    for seed in 0 1 2 3 4; do
        python main_hetero.py --dataset $dataset --knn 5 --cluster kmeans --seed $seed
    done
done

# DMoN + kNN-5
for dataset in texas cornell wisconsin; do
    for seed in 0 1 2 3 4; do
        python dmon_hetero.py --dataset $dataset --knn 5 --seed $seed
    done
done

# k-sensitivity (k = 3, 5, 10, 15, 20)
for k in 3 5 10 15 20; do
    for dataset in texas cornell wisconsin; do
        for seed in 0 1 2 3 4; do
            python main_hetero.py --dataset $dataset --knn $k --cluster kmeans --seed $seed
        done
    done
done
