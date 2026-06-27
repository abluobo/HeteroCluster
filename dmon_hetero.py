"""
DMoN baseline vs Feature-kNN DMoN on heterophilic graphs.

DMoN (Deep Modularity Networks, ICML 2021) uses a differentiable modularity
loss, similar to DGCluster.  We show the same kNN replacement also improves
DMoN, demonstrating that the fix generalizes beyond DGCluster.

Usage:
    python dmon_hetero.py --dataset texas --knn 0   # DMoN + structural adj
    python dmon_hetero.py --dataset texas --knn 5   # DMoN + feature-kNN adj
"""

import numpy as np
import random
import scipy as sp

import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from torch_geometric.datasets import Planetoid, WebKB
from torch_geometric.nn import DMoNPooling

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

import argparse
import os


# ── data ──────────────────────────────────────────────────────────────────────

def load_dataset(name):
    name = name.lower()
    if name == 'cora':
        return Planetoid(root='data', name='Cora')
    elif name == 'citeseer':
        return Planetoid(root='data', name='Citeseer')
    elif name in ('texas', 'wisconsin', 'cornell'):
        return WebKB(root='data', name=name.capitalize())
    else:
        raise NotImplementedError(name)


def build_feature_knn_adj_dense(data, k):
    """Return dense [N,N] adjacency of feature k-NN graph (symmetrized, binary)."""
    n = data.x.shape[0]
    x_norm = F.normalize(data.x.float(), dim=1)
    sim = x_norm @ x_norm.t()
    sim.fill_diagonal_(-1.0)
    _, idx = sim.topk(k, dim=1)                   # [N, k]

    adj = torch.zeros(n, n)
    rows = torch.arange(n).unsqueeze(1).expand_as(idx)
    adj[rows, idx] = 1.0
    adj = ((adj + adj.t()) > 0).float()           # symmetrize
    return adj


def build_structural_adj_dense(data):
    """Return dense [N,N] adjacency from edge_index."""
    n = data.x.shape[0]
    adj = torch.zeros(n, n)
    ei = data.edge_index
    adj[ei[0], ei[1]] = 1.0
    return adj


# ── model ─────────────────────────────────────────────────────────────────────

def make_model(in_dim, k):
    """DMoNPooling with a 3-layer MLP: in → 256 → 128 → k."""
    return DMoNPooling(channels=[in_dim, 256, 128], k=k)


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(clusters, true_labels):
    nmi = normalized_mutual_info_score(true_labels, clusters)
    ari = adjusted_rand_score(true_labels, clusters)
    cm  = confusion_matrix(true_labels, clusters)
    if cm.shape[0] != cm.shape[1]:
        # pad to square
        size = max(cm.shape)
        pad  = np.zeros((size, size), dtype=cm.dtype)
        pad[:cm.shape[0], :cm.shape[1]] = cm
        cm = pad
    row, col = linear_sum_assignment(cm, maximize=True)
    acc = cm[row, col].sum() / cm.sum()
    return {'NMI': nmi, 'ARI': ari, 'ACC': acc}


# ── train ─────────────────────────────────────────────────────────────────────

def train(model, optimizer, x, adj, epochs):
    sched = lr_scheduler.LinearLR(optimizer, 1.0, 0.1, epochs)
    model.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        # DMoNPooling expects batched input [B,N,F] and [B,N,N]
        s, _, _, spectral, ortho, cluster = model(x.unsqueeze(0), adj.unsqueeze(0))
        loss = spectral + ortho + cluster
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        sched.step()
        if ep % 50 == 0 or ep == epochs - 1:
            print(f'  ep {ep:3d}  loss={loss.item():.4f}  '
                  f'spectral={spectral.item():.4f}  ortho={ortho.item():.4f}')


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='texas')
    p.add_argument('--knn',     type=int, default=0,
                   help='0=structural adj, >0=feature kNN adj')
    p.add_argument('--epochs',  type=int, default=300)
    p.add_argument('--seed',    type=int, default=0)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f'\n=== DMoN  dataset={args.dataset}  knn={args.knn}  seed={args.seed} ===')

    dataset = load_dataset(args.dataset)
    data    = dataset[0]

    labels = data.y.flatten().cpu().numpy()
    K      = int(labels.max()) + 1

    # adjacency for loss
    if args.knn > 0:
        adj = build_feature_knn_adj_dense(data, args.knn)
        print(f'Feature kNN-{args.knn}: {int(adj.sum().item())} edges, '
              f'homophily={((labels[adj.nonzero(as_tuple=True)[0].numpy()] == labels[adj.nonzero(as_tuple=True)[1].numpy()])).mean():.3f}')
    else:
        adj = build_structural_adj_dense(data)
        print(f'Structural: {int(adj.sum().item())} edges, '
              f'homophily={(labels[data.edge_index[0].numpy()] == labels[data.edge_index[1].numpy()]).mean():.3f}')

    x     = data.x.float()
    model = make_model(x.shape[1], K)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train(model, opt, x, adj, args.epochs)

    # get cluster assignments
    model.eval()
    with torch.no_grad():
        s, _, _, _, _, _ = model(x.unsqueeze(0), adj.unsqueeze(0))
    clusters = s.squeeze(0).argmax(dim=-1).cpu().numpy()

    metrics = compute_metrics(clusters, labels)
    print(f'NMI={metrics["NMI"]:.4f}  ARI={metrics["ARI"]:.4f}  ACC={metrics["ACC"]:.4f}')
    print(f'Predicted K={len(np.unique(clusters))}  (true K={K})')

    os.makedirs('results', exist_ok=True)
    tag = f'dmon_{args.dataset}_knn{args.knn}_{args.epochs}_seed{args.seed}'
    torch.save({'args': vars(args), 'metrics': metrics}, f'results/{tag}.pt')
    print(f'Saved: results/{tag}.pt')
