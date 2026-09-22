"""
Self-Supervised GraphSAGE (GS) Pretrainer for Graph2Map (pretrain_gs.py).
Pretrains an inductive GraphSAGE network via contrastive link reconstruction.
Captures ego-neighborhood structural roles (self vs. neighbor representation).
Outputs continuous 64-dimensional latent node representations for any graph dataset.
"""

import os
import sys
import time
import argparse
import random
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops, negative_sampling, degree

from exp_util import load_dataset


# =============================================================================
# 1. GRAPHSAGE NODE & LAYER ARCHITECTURE
# =============================================================================

class GSNodeEncoder(nn.Module):
    """
    Encodes raw input features and degree states into a continuous 64-dim representation:
      - If continuous/discrete features exist (x): Linear -> LayerNorm -> GELU
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

    def forward(self, x: Optional[torch.Tensor], deg: torch.Tensor) -> torch.Tensor:
        clamped_deg = torch.clamp(deg, 0, self.max_degree)
        h = self.deg_embed(clamped_deg)
        if self.feat_proj is not None and x is not None and x.numel() > 0:
            if x.dim() == 1:
                x = x.unsqueeze(1)
            h = h + self.feat_proj(x.float())
        return h


class SAGELayer(nn.Module):
    """
    Mean-Aggregator GraphSAGE Layer implemented with sparse matrix multiplication:
      h_neigh = D^{-1} * A * h
      h_out = LayerNorm(GELU(W_self * h + W_neigh * h_neigh))
      h_res = h + h_out
    """
    def __init__(self, dim: int = 64):
        super().__init__()
        self.w_self = nn.Linear(dim, dim, bias=False)
        self.w_neigh = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, h: torch.Tensor, adj_row_norm: torch.Tensor) -> torch.Tensor:
        h_neigh = torch.sparse.mm(adj_row_norm, h)
        h_trans = self.w_self(h) + self.w_neigh(h_neigh)
        h_out = self.norm(self.act(h_trans))
        return h + h_out


class GSEncoder(nn.Module):
    """
    Multi-Scale GraphSAGE Encoder:
      - Projects raw features and degree states to 64 dimensions.
      - Propagates representations through K GraphSAGE mean-aggregation layers.
      - Combines multi-hop representations: mean([H0, H1, H2, H3]).
      - Normalizes representations to unit hypersphere (L2-norm = 1.0).
    """
    def __init__(self, in_features: int = 0, max_degree: int = 512, embedding_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.input_encoder = GSNodeEncoder(in_features=in_features, max_degree=max_degree, embedding_dim=embedding_dim)
        self.layers = nn.ModuleList([SAGELayer(dim=embedding_dim) for _ in range(num_layers)])

    def forward(self, x: Optional[torch.Tensor], edge_index: torch.Tensor, num_nodes: int, deg: torch.Tensor) -> torch.Tensor:
        h0 = self.input_encoder(x, deg)

        if edge_index.shape[1] == 0 or num_nodes <= 1:
            return F.normalize(h0, p=2, dim=-1)

        # Build Row-Normalized Adjacency with self-loops: D^{-1} * (A + I)
        edge_index_loop, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row = edge_index_loop[0]
        d = torch.bincount(row, minlength=num_nodes).float()
        d_inv = torch.pow(d, -1.0)
        d_inv[torch.isinf(d_inv)] = 0.0
        val = d_inv[row]
        adj_row_norm = torch.sparse_coo_tensor(edge_index_loop, val, (num_nodes, num_nodes)).coalesce()

        hop_representations = [h0]
        curr = h0
        for layer in self.layers:
            curr = layer(curr, adj_row_norm)
            hop_representations.append(curr)

        out = torch.stack(hop_representations, dim=0).mean(dim=0)
        return F.normalize(out, p=2, dim=-1)


# =============================================================================
# 2. CONTRASTIVE LINK RECONSTRUCTION LOSS
# =============================================================================

def contrastive_link_loss(
    embeddings: torch.Tensor,
    pos_edges: torch.Tensor,
    neg_edges: torch.Tensor,
    tau: float = 0.2
) -> Tuple[torch.Tensor, float, float, float]:
    """
    Contrastive BCE margin loss on the unit hypersphere:
      - Pos scores = (u * v) / tau
      - Neg scores = (u * v_neg) / tau
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
        pos_recall = (torch.sigmoid(pos_scores) > 0.5).float().mean().item()
        neg_rejection = (torch.sigmoid(neg_scores) < 0.5).float().mean().item()
        auc = (pos_scores > neg_scores).float().mean().item()

    return loss, pos_recall, neg_rejection, auc


# =============================================================================
# 3. PRETRAINING EXECUTION HARNESS
# =============================================================================

def pretrain_gs(
    dataset_name: str,
    epochs: int = 200,
    embedding_dim: int = 64,
    batch_size: Optional[int] = None,
    lr: float = 0.01,
    weight_decay: float = 1e-5,
    seed: int = 123,
    cache_dir: str = "./cache",
    force_retrain: bool = False
) -> List[torch.Tensor]:
    os.makedirs(cache_dir, exist_ok=True)
    out_emb_path = os.path.join(cache_dir, f"{dataset_name}_gs_dim{embedding_dim}.pt")
    out_model_path = os.path.join(cache_dir, f"{dataset_name}_gs_dim{embedding_dim}_model.pt")

    if os.path.exists(out_emb_path) and not force_retrain:
        print(f"[*] Pretrained GraphSAGE (GS) embeddings for {dataset_name} already exist at: {out_emb_path}")
        return torch.load(out_emb_path, weights_only=False)

    print("=" * 80)
    print(f"SELF-SUPERVISED GRAPHSAGE (GS) PRETRAINING ON: {dataset_name}")
    print(f"Embedding Dim: {embedding_dim} | Epochs: {epochs} | LR: {lr} | Weight Decay: {weight_decay} | Seed: {seed}")
    print("=" * 80)

    # Set determinism stack
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)

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

    if batch_size is None:
        batch_size = 200 if dataset_name == "COLLAB" else (64 if num_graphs > 1000 else num_graphs)

    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)
    num_batches = len(loader)
    print(f"[*] Mini-Batches: {num_batches} per epoch (Batch size: {batch_size})")
    print(f"[*] Gradient Stitching: Enabled across all {num_batches} mini-batches")

    model = GSEncoder(in_features=in_feat, max_degree=max_degree, embedding_dim=embedding_dim, num_layers=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    start_time = time.time()
    print(f"\n[*] Training GraphSAGE for {epochs} epochs via Contrastive Link Reconstruction...")

    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        ep_loss = 0.0
        ep_auc = 0.0

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
            embeddings = model(batch.x, b_edges, b_nodes, b_degrees)

            loss, pos_rec, neg_rej, auc = contrastive_link_loss(embeddings, sub_pos_edges, neg_edge_index, tau=0.2)
            (loss / num_batches).backward()

            ep_loss += loss.item()
            ep_auc += auc

        optimizer.step()
        scheduler.step()

        print_freq = 1 if epochs <= 20 else (10 if epochs <= 100 else 20)
        if ep % print_freq == 0 or ep == 1 or ep == epochs:
            elapsed = time.time() - start_time
            avg_loss = ep_loss / max(num_batches, 1)
            avg_auc = (ep_auc / max(num_batches, 1)) * 100.0
            print(f"  [GS] [Ep {ep:03d}/{epochs:03d}] BCE Loss: {avg_loss:.4f} | Link AUC: {avg_auc:5.1f}% | Elapsed: {elapsed:5.1f}s")

    train_time = time.time() - start_time
    print(f"\n[+] GraphSAGE pretraining completed in {train_time:.2f}s!")

    # Extract continuous 64-dim representations for all graphs individually
    model.eval()
    embeddings_list = []
    print(f"[*] Extracting 64-dimensional GraphSAGE representations across all {num_graphs} graphs...")
    with torch.no_grad():
        for data in dataset:
            d_dev = data.to(device)
            nn_nodes = d_dev.num_nodes
            e_idx = d_dev.edge_index
            deg = degree(e_idx[0], nn_nodes, dtype=torch.long)
            emb = model(d_dev.x, e_idx, nn_nodes, deg).cpu()
            embeddings_list.append(emb)

    torch.save(embeddings_list, out_emb_path)
    torch.save(model.state_dict(), out_model_path)

    print(f"[+] Saved {len(embeddings_list)} graph embedding matrices to: {out_emb_path}")
    print(f"[+] Saved GraphSAGE model checkpoint to: {out_model_path}")
    print("=" * 80)

    return embeddings_list


# =============================================================================
# 4. CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-Supervised GraphSAGE Pretrainer")
    parser.add_argument("-d", "--dataset", type=str, default="MUTAG",
                        choices=['MUTAG', 'PROTEINS', 'PTC_MR', 'NCI1', 'IMDB-BINARY', 'IMDB-MULTI', 'BZR', 'COLLAB', 'all'])
    parser.add_argument("-e", "--epochs", type=int, default=200, help="pretraining epochs (default: 200)")
    parser.add_argument("--dim", type=int, default=64, help="latent dimension (default: 64)")
    parser.add_argument("-b", "--batch", type=int, default=None, help="mini-batch size in graphs")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default: 0.01)")
    parser.add_argument("--wd", type=float, default=1e-5, help="weight decay (default: 1e-5)")
    parser.add_argument("-s", "--seed", type=int, default=123, help="random seed (default: 123)")
    parser.add_argument("--force", action="store_true", help="force retrain even if cached")
    args = parser.parse_args()

    target_datasets = ['MUTAG', 'PROTEINS', 'PTC_MR', 'BZR', 'NCI1'] if args.dataset == "all" else [args.dataset]
    for ds in target_datasets:
        pretrain_gs(
            dataset_name=ds,
            epochs=args.epochs,
            embedding_dim=args.dim,
            batch_size=args.batch,
            lr=args.lr,
            weight_decay=args.wd,
            seed=args.seed,
            force_retrain=args.force
        )
