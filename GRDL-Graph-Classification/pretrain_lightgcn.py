"""
Self-Supervised LightGCN Pretrainer for Unattributed Social Graphs.
Extracts continuous 64-dimensional structural latent embeddings without human-crafted heuristics.
Trains via topological link reconstruction (BPR loss) across multi-hop diffusion.
"""

import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops, negative_sampling, degree
from exp_util import load_dataset


class LightGCNEncoder(nn.Module):
    """
    LightGCN Topological Feature Extractor:
      - Maps node degree states to a 64-dimensional continuous latent space.
      - Propagates representations through K linear normalized diffusion layers.
      - Discards non-linear activations and heavy parameter weights to prevent over-smoothing.
    """
    def __init__(self, max_degree: int = 512, embedding_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.max_degree = max_degree
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.deg_embedding = nn.Embedding(max_degree + 1, embedding_dim)
        nn.init.xavier_uniform_(self.deg_embedding.weight)

    def forward(self, num_nodes: int, edge_index: torch.Tensor, degrees: torch.Tensor) -> torch.Tensor:
        clamped_deg = torch.clamp(degrees, 0, self.max_degree)
        e0 = self.deg_embedding(clamped_deg)  # (N, D)

        if edge_index.shape[1] == 0 or num_nodes <= 1:
            return e0

        # Normalized Symmetric Adjacency: D^{-1/2} (A + I) D^{-1/2}
        edge_index_loop, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index_loop[0], edge_index_loop[1]
        deg = torch.bincount(row, minlength=num_nodes).float()
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        val = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        adj_norm = torch.sparse_coo_tensor(edge_index_loop, val, (num_nodes, num_nodes)).coalesce()

        layers = [e0]
        curr = e0
        for _ in range(self.num_layers):
            curr = torch.sparse.mm(adj_norm, curr)
            layers.append(curr)

        # Multi-scale layer combination
        out = torch.stack(layers, dim=0).mean(dim=0)  # (N, D)
        return out


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """Bayesian Personalized Ranking (BPR) margin loss for topological link reconstruction."""
    return -torch.mean(F.logsigmoid(pos_scores - neg_scores))


def pretrain_lightgcn(
    dataset_name: str = "IMDB-BINARY",
    epochs: int = 500,
    embedding_dim: int = 64,
    num_layers: int = 3,
    batch_size: int = None,
    lr: float = 0.01,
    weight_decay: float = 1e-5,
    seed: int = 123,
    cache_dir: str = "./cache"
):
    print("=" * 80)
    print(f"Self-Supervised LightGCN Pretraining on {dataset_name}")
    print(f"Embedding Dim: {embedding_dim} | Layers: {num_layers} | Epochs: {epochs} | LR: {lr}")
    print("=" * 80)

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Compute Device: {device}")

    # Load dataset
    dataset = load_dataset(dataset_name, seed=seed)
    num_graphs = len(dataset)

    # Calculate degrees for all graphs
    all_degrees = []
    max_deg_seen = 0
    for data in dataset:
        deg = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
        data.degrees = deg
        if deg.numel() > 0:
            m = int(deg.max().item())
            if m > max_deg_seen:
                max_deg_seen = m

    max_degree = max(max_deg_seen + 10, 256)
    print(f"[*] Total Graphs: {num_graphs} | Max Degree Found: {max_deg_seen} (Embedding table size: {max_degree + 1})")

    # Adaptive mini-batching to prevent GPU VRAM exhaustion on huge datasets (e.g. COLLAB)
    if batch_size is None:
        batch_size = 200 if dataset_name == "COLLAB" else num_graphs

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_batches = len(loader)
    print(f"[*] Mini-Batch Partitioning: {num_graphs} graphs -> {num_batches} mini-batches (Batch size: {batch_size})")
    print(f"[*] Gradient Stitching: Enabled (accumulating gradients across all {num_batches} mini-batches per epoch)")

    # Instantiate model
    model = LightGCNEncoder(max_degree=max_degree, embedding_dim=embedding_dim, num_layers=num_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    start_time = time.time()
    model.train()

    print(f"\n[*] Training LightGCN for {epochs} epochs with Gradient Stitching...")
    for ep in range(1, epochs + 1):
        optimizer.zero_grad()
        epoch_loss = 0.0
        epoch_pos_acc = 0.0
        epoch_neg_acc = 0.0
        epoch_auc = 0.0

        for batch_data in loader:
            batch = batch_data.to(device)
            b_nodes = batch.num_nodes
            b_edges = batch.edge_index
            b_degrees = degree(b_edges[0], b_nodes, dtype=torch.long)

            # Forward pass on mini-batch nodes
            embeddings = model(b_nodes, b_edges, b_degrees)

            # Subsample edges for fast topological link reconstruction (max 50,000 edges per mini-batch)
            num_edges = b_edges.shape[1]
            max_samples = 50000
            if num_edges > max_samples:
                perm = torch.randperm(num_edges, device=device)[:max_samples]
                sub_pos_edges = b_edges[:, perm]
                num_neg = max_samples
            else:
                sub_pos_edges = b_edges
                num_neg = num_edges

            # Fast negative sampling capped to 50k pairs
            neg_edge_index = negative_sampling(
                edge_index=b_edges,
                num_nodes=b_nodes,
                num_neg_samples=num_neg
            )

            # L2-normalize embeddings to unit hypersphere to prevent magnitude drift
            emb_norm = F.normalize(embeddings, p=2, dim=-1)
            tau = 0.2  # contrastive temperature

            # Positive link cosine similarities
            pos_u = emb_norm[sub_pos_edges[0]]
            pos_v = emb_norm[sub_pos_edges[1]]
            pos_scores = (pos_u * pos_v).sum(dim=-1) / tau

            # Negative link cosine similarities
            neg_u = emb_norm[neg_edge_index[0]]
            neg_v = emb_norm[neg_edge_index[1]]
            neg_scores = (neg_u * neg_v).sum(dim=-1) / tau

            # Contrastive BCE Loss: forces positive pairs -> 1 and negative pairs -> 0
            pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
            neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
            batch_loss = 0.5 * (pos_loss + neg_loss)

            # Gradient Stitching: accumulate normalized gradient across mini-batches
            scaled_loss = batch_loss / num_batches
            scaled_loss.backward()

            epoch_loss += batch_loss.item()
            with torch.no_grad():
                epoch_pos_acc += (torch.sigmoid(pos_scores) > 0.5).float().mean().item()
                epoch_neg_acc += (torch.sigmoid(neg_scores) < 0.5).float().mean().item()
                epoch_auc += (pos_scores > neg_scores).float().mean().item()

        # Step optimizer once per epoch after accumulating all mini-batch gradients
        optimizer.step()
        scheduler.step()

        print_freq = 1 if epochs <= 100 else 5
        if ep % print_freq == 0 or ep == 1 or ep == epochs:
            avg_loss = epoch_loss / num_batches
            pos_acc = (epoch_pos_acc / num_batches) * 100.0
            neg_acc = (epoch_neg_acc / num_batches) * 100.0
            auc = (epoch_auc / num_batches) * 100.0
            elapsed = time.time() - start_time
            print(f"  [Ep {ep:03d}/{epochs:03d}] BCE Loss: {avg_loss:.4f} | Pos Recall: {pos_acc:5.1f}% | Neg Rejection: {neg_acc:5.1f}% | Link AUC: {auc:5.1f}% | Time: {elapsed:5.1f}s | LR: {scheduler.get_last_lr()[0]:.5f}")



    train_time = time.time() - start_time
    print(f"[+] LightGCN pretraining completed in {train_time:.2f}s!")

    # Evaluation mode: extract and cache embeddings per graph
    model.eval()
    os.makedirs(cache_dir, exist_ok=True)
    embeddings_list = []

    print(f"[*] Extracting 64-dimensional structural embeddings for all {num_graphs} graphs...")
    with torch.no_grad():
        for data in dataset:
            n = data.num_nodes
            edges = data.edge_index.to(device)
            degs = degree(edges[0], n, dtype=torch.long)
            emb = model(n, edges, degs).cpu()  # (n, 64)
            embeddings_list.append(emb)

    # Save embeddings and model weights
    emb_cache_path = os.path.join(cache_dir, f"lightgcn_{dataset_name}_dim{embedding_dim}_emb.pt")
    model_cache_path = os.path.join(cache_dir, f"lightgcn_{dataset_name}_dim{embedding_dim}_model.pt")

    torch.save(embeddings_list, emb_cache_path)
    torch.save(model.state_dict(), model_cache_path)
    print(f"[+] Cached {len(embeddings_list)} graph embedding matrices to: {emb_cache_path}")
    print(f"[+] Saved model checkpoint to: {model_cache_path}")
    print("=" * 80)

    return emb_cache_path, model_cache_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain LightGCN on Unattributed Social Graphs")
    parser.add_argument("-d", "--dataset", type=str, default="IMDB-BINARY", choices=["IMDB-BINARY", "IMDB-MULTI", "COLLAB"])
    parser.add_argument("-e", "--epochs", type=int, default=500, help="epochs for self-supervised link reconstruction (default: 500)")
    parser.add_argument("--dim", type=int, default=64, help="structural latent dimension (default: 64)")
    parser.add_argument("--layers", type=int, default=3, help="number of LightGCN diffusion layers (default: 3)")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default: 0.01)")
    parser.add_argument("-b", "--batch_size", type=int, default=None, help="mini-batch size in graphs (default: 200 for COLLAB, full for others)")
    parser.add_argument("--seed", type=int, default=123, help="random seed (default: 123)")
    args = parser.parse_args()

    pretrain_lightgcn(
        dataset_name=args.dataset,
        epochs=args.epochs,
        embedding_dim=args.dim,
        num_layers=args.layers,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed
    )
