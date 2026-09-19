"""
Unified Tripartite Expert GNN Pretrainer for Graph2Map (pretrain_gcn.py).
Pretrains three fundamentally orthogonal mathematical paradigms simultaneously:
  1. Expert 1 (Chemical GCN): Isotropic spatial diffusion (local degree & chemical valency propagation).
  2. Expert 2 (Random Walk): Multi-scale return probabilities (RWPE) & transition diffusion (cycle & ring detection).
  3. Expert 3 (Spectral Laplacian): Sign-invariant harmonic eigenvectors (global manifold geometry & spectral gap).

All three experts are trained via contrastive link reconstruction on the unit hypersphere (tau=0.2).
Outputs three continuous 64-dimensional latent node representations for every graph.
"""

import os
import sys
import time
import argparse
import random
from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops, negative_sampling, degree

from exp_util import load_dataset


# =============================================================================
# 1. COMMON FEATURE & DEGREE NODE ENCODER
# =============================================================================

class GCNNodeEncoder(nn.Module):
    """
    Encodes raw input features and degree states into a continuous 64-dim representation:
      - Raw node attributes (x): Linear -> LayerNorm -> GELU
      - Discrete topological degree: Embedding(max_degree + 1, D)
      - Representation: h0 = feat_proj(x) + deg_embed(deg)
    """
    def __init__(self, in_features: int = 0, max_degree: int = 512, embedding_dim: int = 64):
        super().__init__()
        self.in_features = in_features
        self.max_degree = max_degree
        self.embedding_dim = embedding_dim

        if in_features > 0:
            self.feat_proj = nn.Sequential(
                nn.Linear(in_features, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU()
            )
        else:
            self.feat_proj = None

        self.deg_embed = nn.Embedding(max_degree + 1, embedding_dim)
        nn.init.xavier_uniform_(self.deg_embed.weight)

    def forward(self, x: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        clamped_deg = torch.clamp(deg, 0, self.max_degree)
        h = self.deg_embed(clamped_deg)
        if self.feat_proj is not None and x is not None and x.numel() > 0:
            if x.dim() == 1:
                x = x.unsqueeze(1)
            h = h + self.feat_proj(x.float())
        return h


# =============================================================================
# 2. EXPERT 1: CHEMICAL ISOTROPIC GCN
# =============================================================================

class GCNEncoder(nn.Module):
    """
    Expert 1: Multi-Scale Graph Convolutional Network (Isotropic Diffusion):
      - Normalized symmetric diffusion: A_norm = D^{-1/2} (A + I) D^{-1/2}
      - Combines multi-hop representations: mean([H0, H1, H2, H3])
      - Normalizes representations to unit hypersphere (L2-norm = 1.0)
    """
    def __init__(self, in_features: int = 0, max_degree: int = 512, embedding_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.input_encoder = GCNNodeEncoder(in_features=in_features, max_degree=max_degree, embedding_dim=embedding_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int, deg: torch.Tensor) -> torch.Tensor:
        h0 = self.input_encoder(x, deg)

        if edge_index.shape[1] == 0 or num_nodes <= 1:
            return F.normalize(h0, p=2, dim=-1)

        edge_index_loop, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index_loop[0], edge_index_loop[1]
        d = torch.bincount(row, minlength=num_nodes).float()
        d_inv = torch.pow(d, -0.5)
        d_inv[torch.isinf(d_inv)] = 0.0
        val = d_inv[row] * d_inv[col]
        adj_norm = torch.sparse_coo_tensor(edge_index_loop, val, (num_nodes, num_nodes)).coalesce()

        layers = [h0]
        curr = h0
        for _ in range(self.num_layers):
            curr = torch.sparse.mm(adj_norm, curr)
            layers.append(curr)

        out = torch.stack(layers, dim=0).mean(dim=0)
        return F.normalize(out, p=2, dim=-1)


# =============================================================================
# 3. EXPERT 2: RANDOM WALK RETURN PROBABILITIES (RWPE) & TRANSITION DIFFUSION
# =============================================================================

def compute_batch_rwpe(edge_index: torch.Tensor, num_nodes: int, ptr: torch.Tensor = None, steps: int = 16) -> torch.Tensor:
    """
    Computes k-step Random Walk Return Probabilities (RWPE) across 16 steps.
    Diag(P^k) measures closed walk count of length k starting and ending at node v.
    Essential for 3-member, 5-member, and 6-member ring detection.
    """
    if num_nodes == 0:
        return torch.zeros((0, steps), dtype=torch.float32)

    device = edge_index.device
    # Build transition matrix P = D^{-1} A
    row, col = edge_index[0], edge_index[1]
    deg = torch.bincount(row, minlength=num_nodes).float()
    deg_inv = torch.pow(deg, -1.0)
    deg_inv[torch.isinf(deg_inv)] = 0.0
    val = deg_inv[row]

    # Sparse transition matrix
    P_sparse = torch.sparse_coo_tensor(edge_index, val, (num_nodes, num_nodes)).coalesce()

    # If num_nodes is reasonably small (e.g. batch <= 3000), compute powers efficiently
    # For return probabilities, P^k_{vv} can be computed via sparse power iterations
    rwpe = []
    # k=1 return probability is diag(P) (non-zero if self loops exist)
    diag_P = torch.zeros(num_nodes, device=device)
    mask_self = (row == col)
    if mask_self.any():
        diag_P.scatter_add_(0, row[mask_self], val[mask_self])
    rwpe.append(diag_P)

    # Multi-step power iteration for diag(P^k)
    # Using probing vector method or block-dense calculation:
    # Since molecular batches have isolated components, we can compute P_dense per block
    # or iterate via sparse mm:
    if num_nodes <= 4096:
        P_dense = P_sparse.to_dense()
        curr_P = P_dense
        for k in range(2, steps + 1):
            curr_P = torch.mm(curr_P, P_dense)
            rwpe.append(torch.diag(curr_P))
    else:
        # Fallback for large batches: approximate with degree-normalized powers
        curr_vec = deg / max(deg.sum().item(), 1e-5)
        for k in range(2, steps + 1):
            curr_vec = torch.sparse.mm(P_sparse, curr_vec.unsqueeze(1)).squeeze(1)
            rwpe.append(curr_vec)

    return torch.stack(rwpe, dim=1)  # (num_nodes, steps)


class RandomWalkEncoder(nn.Module):
    """
    Expert 2: Random Walk Multi-Scale Cycle & Return Probability Encoder:
      - Takes k-step return probabilities (RWPE) capturing rings and cycles.
      - Projects RWPE via 2-layer MLP to continuous embedding space.
      - Propagates representations via asymmetric transition matrix P = D^{-1} (A + I).
      - Multi-hop hop averaging: mean([H0, H1, H2, H3])
      - Normalizes representations to unit hypersphere (L2-norm = 1.0)
    """
    def __init__(self, in_features: int = 0, max_degree: int = 512, embedding_dim: int = 64, num_layers: int = 3, rw_steps: int = 16):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.rw_steps = rw_steps

        self.input_encoder = GCNNodeEncoder(in_features=in_features, max_degree=max_degree, embedding_dim=embedding_dim)
        self.rwpe_proj = nn.Sequential(
            nn.Linear(rw_steps, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, num_nodes: int, deg: torch.Tensor, rwpe: torch.Tensor = None) -> torch.Tensor:
        h0 = self.input_encoder(x, deg)

        if rwpe is not None and rwpe.shape[0] == num_nodes:
            h0 = h0 + self.rwpe_proj(rwpe.to(h0.device))

        if edge_index.shape[1] == 0 or num_nodes <= 1:
            return F.normalize(h0, p=2, dim=-1)

        # Row-stochastic transition matrix with self-loops: P = D^{-1} (A + I)
        edge_index_loop, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index_loop[0], edge_index_loop[1]
        d = torch.bincount(row, minlength=num_nodes).float()
        d_inv = torch.pow(d, -1.0)
        d_inv[torch.isinf(d_inv)] = 0.0
        val = d_inv[row]
        P_norm = torch.sparse_coo_tensor(edge_index_loop, val, (num_nodes, num_nodes)).coalesce()

        layers = [h0]
        curr = h0
        for _ in range(self.num_layers):
            curr = torch.sparse.mm(P_norm, curr)
            layers.append(curr)

        out = torch.stack(layers, dim=0).mean(dim=0)
        return F.normalize(out, p=2, dim=-1)


# =============================================================================
# 4. EXPERT 3: SPECTRAL LAPLACIAN GEOMETRY (GLOBAL HARMONIC MANIFOLD)
# =============================================================================

def compute_graph_spectral_pe(edge_index: torch.Tensor, num_nodes: int, k_eigs: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes lowest non-trivial eigenvectors and eigenvalues of the normalized Laplacian:
      L = I - D^{-1/2} A D^{-1/2} = U Lambda U^T
    Returns:
      - evecs: (N, k_eigs)
      - evals: (k_eigs,)
    """
    if num_nodes <= 1:
        return torch.zeros((num_nodes, k_eigs), dtype=torch.float32), torch.zeros(k_eigs, dtype=torch.float32)

    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = 1.0
    deg = adj.sum(dim=1)
    d_inv_sqrt = torch.pow(torch.clamp(deg, min=1e-5), -0.5)
    d_inv_sqrt[deg == 0] = 0.0

    L_norm = torch.eye(num_nodes) - (d_inv_sqrt.unsqueeze(1) * adj * d_inv_sqrt.unsqueeze(0))

    try:
        evals, evecs = torch.linalg.eigh(L_norm)
    except Exception:
        evals = torch.zeros(num_nodes)
        evecs = torch.zeros((num_nodes, num_nodes))

    # Skip trivial 0th constant eigenvector if num_nodes > 1
    start = 1 if num_nodes > 1 else 0
    e_vals = evals[start:start + k_eigs]
    e_vecs = evecs[:, start:start + k_eigs]

    pad_k = k_eigs - e_vals.shape[0]
    if pad_k > 0:
        e_vals = F.pad(e_vals, (0, pad_k))
        e_vecs = F.pad(e_vecs, (0, pad_k))

    return e_vecs.float(), e_vals.float()


class SpectralEncoder(nn.Module):
    """
    Expert 3: Global Spectral Laplacian Harmonic Manifold:
      - Encodes lowest non-trivial Laplacian eigenvectors via sign-invariant network:
        phi(u_k) = MLP(u_k) + MLP(-u_k) (guarantees sign invariance u_k <-> -u_k)
      - Attenuates higher frequencies via heat diffusion kernel: exp(-lambda_k)
      - Combines global harmonic manifold coordinates with input atom features.
      - Propagates via spectral Chebyshev / Laplacian smoothing: H^{(l+1)} = (I - 0.5 * L) H^{(l)}
      - Normalizes representations to unit hypersphere (L2-norm = 1.0)
    """
    def __init__(self, in_features: int = 0, max_degree: int = 512, embedding_dim: int = 64, num_layers: int = 3, k_eigs: int = 8):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.k_eigs = k_eigs

        self.input_encoder = GCNNodeEncoder(in_features=in_features, max_degree=max_degree, embedding_dim=embedding_dim)

        # Sign-invariant network: phi(u) = mlp(u) + mlp(-u)
        head_dim = embedding_dim // k_eigs
        self.sign_mlp = nn.Sequential(
            nn.Linear(1, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, head_dim)
        )
        self.spectral_out = nn.Sequential(
            nn.Linear(head_dim * k_eigs, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        deg: torch.Tensor,
        evecs: torch.Tensor = None,
        evals: torch.Tensor = None
    ) -> torch.Tensor:
        h0 = self.input_encoder(x, deg)

        if evecs is not None and evecs.shape[0] == num_nodes:
            device = h0.device
            evecs = evecs.to(device)
            evals = evals.to(device) if evals is not None else torch.zeros(self.k_eigs, device=device)

            # Sign-invariant projection for each of the k eigenvectors
            k_feats = []
            for k in range(self.k_eigs):
                u_k = evecs[:, k:k+1]
                phi_k = self.sign_mlp(u_k) + self.sign_mlp(-u_k)
                # Attenuate by eigenvalue heat diffusion
                weight = torch.exp(-evals[k])
                k_feats.append(phi_k * weight)

            cat_spec = torch.cat(k_feats, dim=-1)
            h0 = h0 + self.spectral_out(cat_spec)

        if edge_index.shape[1] == 0 or num_nodes <= 1:
            return F.normalize(h0, p=2, dim=-1)

        # Spectral smoothing operator: S = (I + A_norm) / 2
        edge_index_loop, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index_loop[0], edge_index_loop[1]
        d = torch.bincount(row, minlength=num_nodes).float()
        d_inv = torch.pow(d, -0.5)
        d_inv[torch.isinf(d_inv)] = 0.0
        val = 0.5 * (d_inv[row] * d_inv[col])
        S_norm = torch.sparse_coo_tensor(edge_index_loop, val, (num_nodes, num_nodes)).coalesce()

        layers = [h0]
        curr = h0
        for _ in range(self.num_layers):
            curr = torch.sparse.mm(S_norm, curr)
            layers.append(curr)

        out = torch.stack(layers, dim=0).mean(dim=0)
        return F.normalize(out, p=2, dim=-1)


# =============================================================================
# 5. UNIFIED TRIPARTITE EXPERT MODULE
# =============================================================================

class TripartiteGNNPretrainer(nn.Module):
    """
    Unifies:
      1. Chemical Isotropic GCN (Local Message Passing)
      2. Random Walk Return Probabilities (Cycle & Subgraph Flow)
      3. Spectral Laplacian Eigenmaps (Global Harmonic Manifold)
    """
    def __init__(self, in_features: int = 0, max_degree: int = 512, embedding_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.gcn = GCNEncoder(in_features=in_features, max_degree=max_degree, embedding_dim=embedding_dim, num_layers=num_layers)
        self.walk = RandomWalkEncoder(in_features=in_features, max_degree=max_degree, embedding_dim=embedding_dim, num_layers=num_layers)
        self.spectral = SpectralEncoder(in_features=in_features, max_degree=max_degree, embedding_dim=embedding_dim, num_layers=num_layers)

    def forward(self, x, edge_index, num_nodes, deg, rwpe=None, evecs=None, evals=None):
        eg = self.gcn(x, edge_index, num_nodes, deg)
        ew = self.walk(x, edge_index, num_nodes, deg, rwpe)
        es = self.spectral(x, edge_index, num_nodes, deg, evecs, evals)
        return eg, ew, es


# =============================================================================
# 6. CONTRASTIVE LINK RECONSTRUCTION LOSS
# =============================================================================

def contrastive_link_loss(
    embeddings: torch.Tensor,
    pos_edges: torch.Tensor,
    neg_edges: torch.Tensor,
    tau: float = 0.2
):
    """
    Contrastive margin loss on the unit hypersphere:
      scores = (u * v) / tau
    """
    pos_u = embeddings[pos_edges[0]]
    pos_v = embeddings[pos_edges[1]]
    pos_scores = (pos_u * pos_v).sum(dim=-1) / tau

    neg_u = embeddings[neg_edges[0]]
    neg_v = embeddings[neg_edges[1]]
    neg_scores = (neg_u * neg_v).sum(dim=-1) / tau

    pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
    neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
    loss = 0.5 * (pos_loss + neg_loss)

    with torch.no_grad():
        auc = (pos_scores > neg_scores).float().mean().item()

    return loss, auc


# =============================================================================
# 7. TRIPARTITE PRETRAINING EXECUTION HARNESS
# =============================================================================

def pretrain_gcn(
    dataset_name: str,
    epochs: int = 200,
    embedding_dim: int = 64,
    batch_size: int = None,
    lr: float = 0.01,
    weight_decay: float = 1e-5,
    seed: int = 123,
    cache_dir: str = "./cache",
    force_retrain: bool = False
):
    """
    Pretrains Chemical GCN, Random Walk, and Spectral Laplacian experts in one unified pass.
    Saves:
      - cache/{dataset_name}_gcn_dim64.pt
      - cache/{dataset_name}_walk_dim64.pt
      - cache/{dataset_name}_spectral_dim64.pt
      - cache/{dataset_name}_tripartite_model.pt
    """
    os.makedirs(cache_dir, exist_ok=True)
    out_gcn_path = os.path.join(cache_dir, f"{dataset_name}_gcn_dim{embedding_dim}.pt")
    out_walk_path = os.path.join(cache_dir, f"{dataset_name}_walk_dim{embedding_dim}.pt")
    out_spec_path = os.path.join(cache_dir, f"{dataset_name}_spectral_dim{embedding_dim}.pt")
    out_model_path = os.path.join(cache_dir, f"{dataset_name}_tripartite_model.pt")

    if (os.path.exists(out_gcn_path) and os.path.exists(out_walk_path) and 
        os.path.exists(out_spec_path) and not force_retrain):
        print(f"[*] Tripartite embeddings for {dataset_name} already exist at:")
        print(f"    - GCN:      {out_gcn_path}")
        print(f"    - Walk:     {out_walk_path}")
        print(f"    - Spectral: {out_spec_path}")
        gcn_embs = torch.load(out_gcn_path, weights_only=False)
        walk_embs = torch.load(out_walk_path, weights_only=False)
        spec_embs = torch.load(out_spec_path, weights_only=False)
        return gcn_embs, walk_embs, spec_embs

    print("=" * 80)
    print(f"SELF-SUPERVISED TRIPARTITE PRETRAINING ON: {dataset_name}")
    print(f"Experts: Chemical GCN + Random Walk (RWPE) + Spectral Laplacian | Dim: {embedding_dim} | Epochs: {epochs}")
    print("=" * 80)

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")

    # Load dataset
    dataset = load_dataset(dataset_name, seed=seed)
    num_graphs = len(dataset)
    sample_data = dataset[0]

    in_feat = sample_data.x.shape[-1] if hasattr(sample_data, 'x') and sample_data.x is not None else 0

    max_deg_seen = 0
    for data in dataset:
        deg = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
        if deg.numel() > 0:
            m = int(deg.max().item())
            if m > max_deg_seen:
                max_deg_seen = m
    max_degree = max(max_deg_seen + 16, 256)

    print(f"[*] Total Graphs: {num_graphs} | Node Features: {in_feat} | Max Degree: {max_deg_seen}")

    # Pre-compute graph-level RWPE and Spectral PE for lightning fast batching
    print(f"[*] Pre-computing Random Walk Return Probabilities (16 steps) & Spectral LapPE (8 eigs)...")
    rwpe_cache = []
    spec_cache = []
    for data in dataset:
        rw = compute_batch_rwpe(data.edge_index, data.num_nodes, steps=16)
        rwpe_cache.append(rw)
        evecs, evals = compute_graph_spectral_pe(data.edge_index, data.num_nodes, k_eigs=8)
        spec_cache.append((evecs, evals))

    if batch_size is None:
        batch_size = 200 if dataset_name == "COLLAB" else (64 if num_graphs > 1000 else num_graphs)

    # Attach to dataset objects for DataLoader collate
    for idx, data in enumerate(dataset):
        data.rwpe = rwpe_cache[idx]
        data.evecs = spec_cache[idx][0]
        data.evals = spec_cache[idx][1]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_batches = len(loader)
    print(f"[*] Mini-Batches: {num_batches} per epoch (Batch size: {batch_size})")

    model = TripartiteGNNPretrainer(in_features=in_feat, max_degree=max_degree, embedding_dim=embedding_dim, num_layers=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    start_time = time.time()
    print(f"\n[*] Training Tripartite Experts for {epochs} epochs via Contrastive Link Reconstruction...")

    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        ep_loss_g = 0.0
        ep_auc_g = 0.0
        ep_loss_w = 0.0
        ep_auc_w = 0.0
        ep_loss_s = 0.0
        ep_auc_s = 0.0

        for batch_data in loader:
            batch = batch_data.to(device)
            b_nodes = batch.num_nodes
            b_edges = batch.edge_index

            if b_edges.shape[1] == 0 or b_nodes <= 1:
                continue

            num_edges = b_edges.shape[1]
            max_samples = 50000
            if num_edges > max_samples:
                perm = torch.randperm(num_edges, device=device)[:max_samples]
                sub_pos_edges = b_edges[:, perm]
                num_neg = max_samples
            else:
                sub_pos_edges = b_edges
                num_neg = num_edges

            neg_edge_index = negative_sampling(
                edge_index=b_edges,
                num_nodes=b_nodes,
                num_neg_samples=num_neg
            )

            b_degrees = degree(b_edges[0], b_nodes, dtype=torch.long)
            b_rwpe = getattr(batch, 'rwpe', None)
            b_evecs = getattr(batch, 'evecs', None)
            b_evals = getattr(batch, 'evals', None)

            emb_g, emb_w, emb_s = model(batch.x, b_edges, b_nodes, b_degrees, b_rwpe, b_evecs, b_evals)

            loss_g, auc_g = contrastive_link_loss(emb_g, sub_pos_edges, neg_edge_index, tau=0.2)
            loss_w, auc_w = contrastive_link_loss(emb_w, sub_pos_edges, neg_edge_index, tau=0.2)
            loss_s, auc_s = contrastive_link_loss(emb_s, sub_pos_edges, neg_edge_index, tau=0.2)

            loss = (loss_g + loss_w + loss_s) / 3.0
            (loss / num_batches).backward()

            ep_loss_g += loss_g.item()
            ep_auc_g += auc_g
            ep_loss_w += loss_w.item()
            ep_auc_w += auc_w
            ep_loss_s += loss_s.item()
            ep_auc_s += auc_s

        optimizer.step()
        scheduler.step()

        print_freq = 1 if epochs <= 20 else (10 if epochs <= 100 else 20)
        if ep % print_freq == 0 or ep == 1 or ep == epochs:
            elapsed = time.time() - start_time
            avg_auc_g = (ep_auc_g / max(num_batches, 1)) * 100.0
            avg_auc_w = (ep_auc_w / max(num_batches, 1)) * 100.0
            avg_auc_s = (ep_auc_s / max(num_batches, 1)) * 100.0
            avg_loss = (ep_loss_g + ep_loss_w + ep_loss_s) / (3.0 * max(num_batches, 1))
            print(f"  [Ep {ep:03d}/{epochs:03d}] Total Loss: {avg_loss:.4f} | AUC (GCN/Walk/Spectral): {avg_auc_g:4.1f}% / {avg_auc_w:4.1f}% / {avg_auc_s:4.1f}% | Elapsed: {elapsed:5.1f}s")

    train_time = time.time() - start_time
    print(f"\n[+] Tripartite pretraining completed in {train_time:.2f}s!")

    # Extract embeddings for all graphs individually
    model.eval()
    gcn_embeddings = []
    walk_embeddings = []
    spectral_embeddings = []
    print(f"[*] Extracting continuous representations across all {num_graphs} graphs...")
    with torch.no_grad():
        for idx, data in enumerate(dataset):
            d_dev = data.to(device)
            nn_nodes = d_dev.num_nodes
            e_idx = d_dev.edge_index
            deg = degree(e_idx[0], nn_nodes, dtype=torch.long)
            rw = rwpe_cache[idx].to(device)
            evecs, evals = spec_cache[idx][0].to(device), spec_cache[idx][1].to(device)

            eg, ew, es = model(d_dev.x, e_idx, nn_nodes, deg, rw, evecs, evals)
            gcn_embeddings.append(eg.cpu())
            walk_embeddings.append(ew.cpu())
            spectral_embeddings.append(es.cpu())

    torch.save(gcn_embeddings, out_gcn_path)
    torch.save(walk_embeddings, out_walk_path)
    torch.save(spectral_embeddings, out_spec_path)
    torch.save(model.state_dict(), out_model_path)

    print(f"[+] Saved {len(gcn_embeddings)} GCN embedding matrices to:      {out_gcn_path}")
    print(f"[+] Saved {len(walk_embeddings)} Walk embedding matrices to:     {out_walk_path}")
    print(f"[+] Saved {len(spectral_embeddings)} Spectral embedding matrices to: {out_spec_path}")
    print(f"[+] Saved Tripartite model checkpoint to:                   {out_model_path}")
    print("=" * 80)

    return gcn_embeddings, walk_embeddings, spectral_embeddings


# =============================================================================
# 8. CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tripartite GNN Pretrainer (GCN + Random Walk + Spectral)")
    parser.add_argument("-d", "--dataset", type=str, default="MUTAG",
                        choices=['MUTAG', 'PROTEINS', 'PTC_MR', 'NCI1', 'IMDB-BINARY', 'IMDB-MULTI', 'BZR', 'COLLAB', 'all'])
    parser.add_argument("-e", "--epochs", type=int, default=200, help="pretraining epochs (default: 200)")
    parser.add_argument("--dim", type=int, default=64, help="latent dimension (default: 64)")
    parser.add_argument("-b", "--batch", type=int, default=None, help="mini-batch size in graphs")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default: 0.01)")
    parser.add_argument("--wd", type=float, default=1e-5, help="weight decay (default: 1e-5)")
    parser.add_argument("--seed", type=int, default=123, help="random seed (default: 123)")
    parser.add_argument("--force", action="store_true", help="force retrain even if cached")
    args = parser.parse_args()

    target_datasets = (
        ['MUTAG', 'BZR', 'PTC_MR', 'PROTEINS', 'NCI1']
        if args.dataset == 'all'
        else [args.dataset]
    )

    for d_name in target_datasets:
        pretrain_gcn(
            dataset_name=d_name,
            epochs=args.epochs,
            embedding_dim=args.dim,
            batch_size=args.batch,
            lr=args.lr,
            weight_decay=args.wd,
            seed=args.seed,
            force_retrain=args.force
        )
