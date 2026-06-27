"""
Feature-Similarity-Weighted Modularity for Heterophilic Graph Clustering.

Key idea: Standard modularity assumes homophily (connected nodes → same cluster).
On heterophilic graphs this fails. We weight the adjacency by cosine similarity:
    w_ij = tanh(beta * cos_sim(x_i, x_j))
so dissimilar-feature edges get negative weight (anti-modularity: repel from same cluster).

Usage:
    python main_hetero.py --dataset texas --lam 0.0 --beta 0.0  # baseline (standard DGCluster)
    python main_hetero.py --dataset texas --lam 0.0 --beta 2.0  # feature-weighted (ours)
"""

import numpy as np
import random
import scipy as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch.optim.lr_scheduler as lr_scheduler

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WebKB, Actor, WikipediaNetwork

from torch_geometric.nn import GCNConv

import scipy.sparse
from sklearn.cluster import Birch, KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

import networkx as nx
import argparse
import os


# ── datasets ──────────────────────────────────────────────────────────────────

def load_dataset(name):
    name = name.lower()
    if name == 'cora':
        return Planetoid(root='data', name='Cora')
    elif name == 'citeseer':
        return Planetoid(root='data', name='Citeseer')
    elif name == 'pubmed':
        return Planetoid(root='data', name='PubMed')
    elif name == 'computers':
        return Amazon(root='data', name='Computers')
    elif name == 'photo':
        return Amazon(root='data', name='Photo')
    elif name == 'coauthorcs':
        return Coauthor(root='data/Coauthor', name='CS')
    elif name in ('texas', 'wisconsin', 'cornell'):
        cap = name.capitalize()
        return WebKB(root='data', name=cap)
    elif name == 'actor':
        return Actor(root='data/actor')
    elif name == 'chameleon':
        return WikipediaNetwork(root='data', name='chameleon')
    elif name == 'squirrel':
        return WikipediaNetwork(root='data', name='squirrel')
    else:
        raise NotImplementedError(f'Unknown dataset: {name}')


# ── model ─────────────────────────────────────────────────────────────────────

class GNN(nn.Module):
    """GCN encoder — standard DGCluster architecture (for homophilic graphs)."""
    def __init__(self, in_dim, out_dim=64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, 256)
        self.conv2 = GCNConv(256, 128)
        self.conv3 = GCNConv(128, out_dim)

    def forward(self, data):
        x, ei = data.x, data.edge_index
        x = F.selu(self.conv1(x, ei))
        x = F.dropout(x, training=self.training)
        x = F.selu(self.conv2(x, ei))
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, ei)
        x = x / (x.norm() + 1e-8)
        x = F.tanh(x) ** 2
        x = F.normalize(x, dim=1)
        return x


class MLP(nn.Module):
    """MLP encoder — no graph aggregation, suitable for heterophilic graphs."""
    def __init__(self, in_dim, out_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, out_dim)

    def forward(self, data):
        x = data.x
        x = F.selu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.selu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        x = x / (x.norm() + 1e-8)
        x = F.tanh(x) ** 2
        x = F.normalize(x, dim=1)
        return x


# ── loss ──────────────────────────────────────────────────────────────────────

def build_weighted_adj(data, beta, device):
    """Precompute feature-similarity-weighted sparse adjacency.

    W_ij = tanh(beta * cosine_sim(x_i, x_j))
    For beta=0: plain adjacency (ones on structural edges).
    For beta>0: tanh-weighted structural edges.
    """
    ei = data.edge_index          # [2, E]
    x_norm = F.normalize(data.x.float(), dim=1)  # [N, d]
    sim = (x_norm[ei[0]] * x_norm[ei[1]]).sum(dim=1)

    if beta == 0.0:
        weights = torch.ones(ei.shape[1])
    else:
        weights = torch.tanh(beta * sim)

    n = data.x.shape[0]
    sp_adj = sp.sparse.csr_matrix(
        (weights.cpu().numpy(), ei.cpu().numpy()),
        shape=(n, n)
    )
    return sp_adj


def build_feature_knn_adj(data, k, device):
    """Build feature k-NN graph (sparse, symmetrized).

    For small graphs (N<=3000): exact cosine kNN via full pairwise matrix.
    For large graphs (N>3000): approximate kNN via sklearn NearestNeighbors.
    """
    n = data.x.shape[0]
    x_norm = F.normalize(data.x.float(), dim=1).cpu().numpy()

    if n <= 3000:
        # exact: pairwise cosine similarity
        sim_mat = torch.tensor(x_norm) @ torch.tensor(x_norm).t()
        sim_mat.fill_diagonal_(-1.0)
        _, idx = sim_mat.topk(k, dim=1)
        rows = torch.arange(n).unsqueeze(1).expand_as(idx).flatten().numpy()
        cols = idx.flatten().numpy()
    else:
        # approximate: sklearn ball-tree on L2-normalized vectors (cosine = L2 on unit sphere)
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree', metric='euclidean', n_jobs=-1)
        nbrs.fit(x_norm)
        indices = nbrs.kneighbors(x_norm, return_distance=False)[:, 1:]  # exclude self
        rows = np.repeat(np.arange(n), k)
        cols = indices.flatten()

    # symmetrize
    src = np.concatenate([rows, cols])
    dst = np.concatenate([cols, rows])
    ones = np.ones(src.shape[0])

    sp_adj = sp.sparse.csr_matrix((ones, (src, dst)), shape=(n, n))
    sp_adj.data[:] = 1.0
    return sp_adj


def build_signed_knn_adj(data, k, beta, device):
    """Signed Feature-Guided Graph (SFG).

    Combines structural edges (signed by feature similarity) with kNN edges:
    - Structural edge (i,j): weight = tanh(beta * sim(x_i, x_j))
        similar features → positive → attract to same cluster
        dissimilar features → negative → repel from same cluster
    - kNN edge (i,j) not in structural: weight = +1 (add positive connection)

    Subsumes pure kNN (beta=0) and adds signed correction of structural edges.
    """
    n = data.x.shape[0]
    x_norm = F.normalize(data.x.float(), dim=1)

    # -- structural edges with tanh(beta*sim) weights --
    ei = data.edge_index
    sim_struct = (x_norm[ei[0]] * x_norm[ei[1]]).sum(dim=1)
    w_struct = torch.tanh(beta * sim_struct).cpu().numpy()

    # -- kNN edges with weight +1 --
    x_np = x_norm.cpu().numpy()
    if n <= 3000:
        sim_mat = torch.tensor(x_np) @ torch.tensor(x_np).t()
        sim_mat.fill_diagonal_(-1.0)
        _, idx = sim_mat.topk(k, dim=1)
        rows_knn = torch.arange(n).unsqueeze(1).expand_as(idx).flatten().numpy()
        cols_knn = idx.flatten().numpy()
    else:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree',
                                metric='euclidean', n_jobs=-1).fit(x_np)
        indices = nbrs.kneighbors(x_np, return_distance=False)[:, 1:]
        rows_knn = np.repeat(np.arange(n), k)
        cols_knn = indices.flatten()

    # mark existing structural edges in a set for fast lookup
    struct_set = set(zip(ei[0].numpy(), ei[1].numpy()))

    # keep only kNN edges not already in structural graph
    mask_new = np.array([(r, c) not in struct_set for r, c in zip(rows_knn, cols_knn)])
    rows_new = rows_knn[mask_new]
    cols_new = cols_knn[mask_new]

    # combine: structural (signed) + new kNN (positive +1)
    all_rows = np.concatenate([ei[0].numpy(), ei[1].numpy(), rows_new, cols_new])
    all_cols = np.concatenate([ei[1].numpy(), ei[0].numpy(), cols_new, rows_new])
    all_w    = np.concatenate([w_struct, w_struct,
                               np.ones(len(rows_new)), np.ones(len(cols_new))])

    sp_adj = sp.sparse.csr_matrix((all_w, (all_rows, all_cols)), shape=(n, n))
    return sp_adj


def convert_scipy_torch_sp(sp_adj):
    coo = sp_adj.tocoo()
    idx = torch.LongTensor(np.vstack((coo.row, coo.col)))
    val = torch.FloatTensor(coo.data)
    return torch.sparse_coo_tensor(idx, val, torch.Size(coo.shape))


def aux_objective(output, s, oh_labels):
    out = output[s].float()
    C   = oh_labels[s].float()
    t1  = torch.trace(C.t().mm(C).mm(C.t().mm(C)))
    t2  = torch.trace(out.t().mm(out).mm(out.t().mm(out)))
    t3  = torch.trace(out.t().mm(C).mm(C.t().mm(out)))
    return (t1 + t2 - 2 * t3) / (len(s) ** 2)


def loss_fn(output, lam, sp_w_adj, degree, num_nodes, num_edges,
            oh_labels, device, alp=0.0):
    # sample a subset for efficiency on large graphs (WebKB are small → use all)
    max_sample = 2000
    if num_nodes > max_sample:
        s = random.sample(range(num_nodes), max_sample)
    else:
        s = list(range(num_nodes))

    s_out  = output[s]
    s_adj  = sp_w_adj[s, :][:, s]
    s_adj  = convert_scipy_torch_sp(s_adj).double().to(device)
    s_deg  = degree[s]

    # sparse matmul: Tr(Cᵀ·A·C) = Tr(Cᵀ · (A·C)) where A is sparse
    ac = torch.sparse.mm(s_adj, s_out.double())  # [N, K]
    x  = torch.trace(s_out.t().double().mm(ac))  # [K, N] @ [N, K] = [K, K]
    y  = ((s_out.t().double().mv(s_deg.double())) ** 2).sum() / (2 * num_edges)

    scaling = (num_nodes / len(s)) ** 2
    m_loss  = -((x - y) / (2 * num_edges)) * scaling

    aux_loss = lam * aux_objective(output, s, oh_labels) if lam > 0 else torch.tensor(0.)

    return m_loss + aux_loss


# ── training ──────────────────────────────────────────────────────────────────

def train(model, optimizer, data, epochs, lam, sp_w_adj,
          degree, num_nodes, num_edges, oh_labels, device):
    sched = lr_scheduler.LinearLR(optimizer, 1.0, 0.1, epochs)
    model.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        out  = model(data)
        loss = loss_fn(out, lam, sp_w_adj, degree, num_nodes,
                       num_edges, oh_labels, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        sched.step()
        if ep % 50 == 0 or ep == epochs - 1:
            print(f'  ep {ep:3d}  loss={loss.item():.4f}')


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(clusters, true_labels, adj_dense, num_edges):
    """Return dict with NMI, ARI, ACC, Q."""
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix

    nmi = normalized_mutual_info_score(true_labels, clusters)
    ari = adjusted_rand_score(true_labels, clusters)

    # clustering accuracy via Hungarian
    cm  = confusion_matrix(true_labels, clusters)
    row, col = linear_sum_assignment(cm, maximize=True)
    acc = cm[row, col].sum() / cm.sum()

    # modularity
    degrees = np.array(adj_dense.sum(axis=1)).flatten()
    twice_e = degrees.sum()
    Q = 0.0
    for cid in np.unique(clusters):
        idx = np.where(clusters == cid)[0]
        sub = adj_dense[np.ix_(idx, idx)]
        Q  += sub.sum() - (degrees[idx].sum() ** 2) / twice_e
    Q /= twice_e

    return {'NMI': nmi, 'ARI': ari, 'ACC': acc, 'Q': Q}


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='texas')
    p.add_argument('--lam',     type=float, default=0.0,
                   help='aux label loss weight (>0 uses GT labels)')
    p.add_argument('--beta',    type=float, default=0.0,
                   help='feature-sim weighting sharpness for structural adj; 0=plain')
    p.add_argument('--knn',     type=int,   default=0,
                   help='if >0, replace structural adj with feature-kNN graph in loss')
    p.add_argument('--cluster', type=str,   default='birch',
                   choices=['birch', 'kmeans'],
                   help='final clustering algorithm')
    p.add_argument('--struct_encoder', action='store_true',
                   help='when knn>0, keep structural graph for GCN encoder (ablation)')
    p.add_argument('--sfg', action='store_true',
                   help='Signed Feature-Guided graph: tanh-signed struct + kNN positive edges')
    p.add_argument('--encoder', type=str,   default='gcn',
                   choices=['gcn', 'mlp'],
                   help='gcn=DGCluster standard; mlp=no graph aggregation')
    p.add_argument('--epochs',  type=int,   default=300)
    p.add_argument('--seed',    type=int,   default=0)
    p.add_argument('--device',  type=str,   default='cpu')
    return p.parse_args()


if __name__ == '__main__':
    args   = parse_args()
    device = torch.device('cpu')

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f'\n=== dataset={args.dataset}  lam={args.lam}  beta={args.beta}  seed={args.seed} ===')

    dataset = load_dataset(args.dataset)
    data    = dataset[0].to(device)

    num_nodes = data.x.shape[0]
    num_edges_raw = data.edge_index.shape[1]

    labels    = data.y.flatten().cpu().numpy()
    oh_labels = F.one_hot(data.y.flatten(), num_classes=int(labels.max()) + 1).to(device)

    # structural adjacency (for metrics and baseline)
    sp_adj_plain = sp.sparse.csr_matrix(
        (np.ones(num_edges_raw), data.edge_index.cpu().numpy()),
        shape=(num_nodes, num_nodes)
    )
    adj_dense = np.array(sp_adj_plain.todense())

    # build loss adjacency
    if args.sfg:
        sp_w_adj = build_signed_knn_adj(data, args.knn if args.knn > 0 else 5,
                                        args.beta if args.beta > 0 else 2.0, device)
    elif args.knn > 0:
        sp_w_adj = build_feature_knn_adj(data, args.knn, device)
    else:
        sp_w_adj = build_weighted_adj(data, args.beta, device)

    # degree and edge count MUST come from the same graph used in the loss
    degree    = torch.tensor(sp_w_adj.sum(axis=1)).squeeze().float().to(device)
    num_edges = max(1, int(sp_w_adj.nnz / 2))

    # for GCN encoder: optionally replace edge_index with kNN edges
    train_data = data
    if args.encoder == 'gcn' and args.knn > 0 and not args.struct_encoder:
        coo = sp_w_adj.tocoo()
        knn_ei = torch.LongTensor(np.vstack((coo.row, coo.col))).to(device)
        from torch_geometric.data import Data
        train_data = Data(x=data.x, edge_index=knn_ei, y=data.y).to(device)

    in_dim = data.x.shape[1]
    model  = (MLP(in_dim) if args.encoder == 'mlp' else GNN(in_dim)).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3,
                              betas=(0.9, 0.999), weight_decay=0.001, amsgrad=True)

    train(model, opt, train_data, args.epochs, args.lam, sp_w_adj,
          degree, num_nodes, num_edges, oh_labels, device)

    model.eval()
    with torch.no_grad():
        emb = model(train_data).cpu().numpy()

    true_k = int(labels.max()) + 1
    if args.cluster == 'kmeans':
        clusters = KMeans(n_clusters=true_k, random_state=args.seed, n_init=20).fit_predict(emb)
        print(f'KMeans K={true_k}')
    else:
        clusters = Birch(n_clusters=None, threshold=0.5).fit_predict(emb)
        print(f'BIRCH clusters: {len(np.unique(clusters))}  (true K={true_k})')

    metrics = compute_metrics(clusters, labels, adj_dense, num_edges)
    print(f'NMI={metrics["NMI"]:.4f}  ARI={metrics["ARI"]:.4f}  '
          f'ACC={metrics["ACC"]:.4f}  Q={metrics["Q"]:.4f}')

    # save
    os.makedirs('results', exist_ok=True)
    tag = f'{args.dataset}_{args.encoder}_lam{args.lam}_beta{args.beta}_knn{args.knn}_{args.epochs}_seed{args.seed}'
    torch.save({'args': vars(args), 'metrics': metrics,
                'num_clusters': len(np.unique(clusters))},
               f'results/{tag}.pt')
    print(f'Saved: results/{tag}.pt')
