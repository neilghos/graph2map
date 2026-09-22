"""
Expert Rasterizer for GCN Latent Maps (expert_rasterizer.py).
Transforms pretrained GCN continuous node representations into multi-channel 2D convolutional heatmaps.

Channels:
  - Channel 0: GCN Latent Manifold (Y: 64 continuous latent dimensions, X: nodes ordered along 1D Fiedler sequence)
  - Channel 1: GCN Laplacian Gradient (High-frequency boundary field showing where latent features change across cuts)
  - Channel 2: Continuous Spatial Layout Flux (Euclidean edge flux weighted by bond orders when available)
"""

import os
import sys
import time
from typing import Optional
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
from tqdm import tqdm


# =============================================================================
# 1. CANONICAL TOPOLOGICAL NODE ORDERING (FIEDLER EIGENVECTOR)
# =============================================================================

def get_canonical_node_order(num_nodes: int, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Computes a canonical 1D ordering of graph nodes along the principal manifold.
    Uses the Fiedler vector (2nd eigenvector of the graph Laplacian), which provides
    the optimal 1D continuous embedding minimizing edge stretch.
    Falls back gracefully to degree ordering for disconnected or degenerate graphs.
    """
    if num_nodes <= 2:
        return torch.arange(num_nodes)

    try:
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        adj[edge_index[0], edge_index[1]] = 1.0
        deg = adj.sum(dim=1)
        L = torch.diag(deg) - adj
        evals, evecs = torch.linalg.eigh(L)
        fiedler = evecs[:, 1]
        return torch.argsort(fiedler, stable=True)
    except Exception:
        deg = torch.bincount(edge_index[0], minlength=num_nodes).float()
        return torch.argsort(-deg, stable=True)


# =============================================================================
# 2. CONTINUOUS 2D EUCLIDEAN LAYOUT
# =============================================================================

def get_2d_layout(
    num_nodes: int,
    edge_index: torch.Tensor,
    method: str = "spring",
    seed: int = 42
) -> np.ndarray:
    """
    Computes smooth 2D continuous node coordinates normalized to [-1, 1]^2.
    """
    if num_nodes == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if num_nodes == 1:
        return np.zeros((1, 2), dtype=np.float32)

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.t().cpu().numpy()
    G.add_edges_from(edges)

    try:
        if method == "spring":
            pos = nx.spring_layout(G, seed=seed, iterations=50)
        elif method == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif method == "spectral":
            pos = nx.spectral_layout(G)
        else:
            pos = nx.spring_layout(G, seed=seed)
    except Exception:
        pos = nx.random_layout(G, seed=seed)

    coords = np.array([pos[i] for i in range(num_nodes)], dtype=np.float32)

    # Normalize to [-1, 1] per axis
    c_min = coords.min(axis=0, keepdims=True)
    c_max = coords.max(axis=0, keepdims=True)
    c_range = np.maximum(c_max - c_min, 1e-5)
    coords = 2.0 * (coords - c_min) / c_range - 1.0
    return coords.astype(np.float32)


# =============================================================================
# 3. GCN MAP RASTERIZATION (3 CHANNELS)
# =============================================================================

def compute_gcn_map(
    data: Data,
    gcn_emb: torch.Tensor,
    resolution: int = 64,
    layout_method: str = "spring",
    sigma_edge: float = 0.04,
    seed: int = 42
) -> torch.Tensor:
    """
    Renders a single graph into a 3-channel 2D convolutional heatmap:
      - Channel 0: GCN Latent Manifold (Y: 64 latent dimensions, X: 1D Fiedler node sequence)
      - Channel 1: GCN Laplacian Gradient (High-frequency boundary field: |H - A_norm H|)
      - Channel 2: Continuous 2D Spatial Layout Edge Flux (weighted by bond orders when available)
    """
    num_nodes = data.num_nodes
    edge_index = data.edge_index

    if num_nodes == 0:
        return torch.zeros((3, resolution, resolution), dtype=torch.float32)

    H = gcn_emb.float().to(edge_index.device)  # (N, D)

    # 1. Canonical 1D ordering along principal Laplacian manifold
    order = get_canonical_node_order(num_nodes, edge_index).to(H.device)
    H_ordered = H[order, :]  # (N, D)
    raw_mat = H_ordered.t()  # (D, N)

    # 2. Normalized Laplacian Gradient on GCN Latent Features
    if edge_index.shape[1] > 0 and num_nodes > 1:
        edge_index_loop, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index_loop[0], edge_index_loop[1]
        deg = torch.bincount(row, minlength=num_nodes).float()
        deg_inv = torch.pow(deg, -0.5)
        deg_inv[torch.isinf(deg_inv)] = 0.0
        val = deg_inv[row] * deg_inv[col]
        adj_norm = torch.sparse_coo_tensor(edge_index_loop, val, (num_nodes, num_nodes)).coalesce()

        H_smooth = torch.sparse.mm(adj_norm, H)
        lap_H = torch.abs(H - H_smooth)
    else:
        lap_H = torch.zeros_like(H)

    lap_ordered = lap_H[order, :].t()  # (D, N)

    # Continuous Bilinear Interpolation Helper
    def to_channel_map(mat):
        mat_4d = mat.unsqueeze(0).unsqueeze(0)  # (1, 1, D, N)
        resized = F.interpolate(mat_4d, size=(resolution, resolution), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
        m_min = resized.min()
        m_max = resized.max()
        if m_max - m_min > 1e-6:
            normed = (resized - m_min) / (m_max - m_min)
        elif m_max > 0:
            normed = torch.ones_like(resized)
        else:
            normed = torch.zeros_like(resized)
        return normed

    ch0_manifold = to_channel_map(raw_mat)
    ch1_laplacian = to_channel_map(lap_ordered)

    # 3. Continuous 2D Spatial Layout Edge Flux
    coords = get_2d_layout(num_nodes, edge_index, method=layout_method, seed=seed)
    lin = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(lin, lin, indexing="ij")
    grid = np.stack([grid_x, grid_y], axis=-1)

    num_edges = edge_index.shape[1]
    if num_edges > 0:
        edges = edge_index.t().cpu().numpy()
        p0 = coords[edges[:, 0]]
        p1 = coords[edges[:, 1]]
        v = p1 - p0
        len_sq = np.maximum(np.sum(v ** 2, axis=-1), 1e-8)
        p_minus_p0 = grid[:, :, None, :] - p0[None, None, :, :]
        dot = np.sum(p_minus_p0 * v[None, None, :, :], axis=-1)
        t = np.clip(dot / len_sq[None, None, :], 0.0, 1.0)
        closest = p0[None, None, :, :] + t[:, :, :, None] * v[None, None, :, :]
        dist_sq_edge = np.sum((grid[:, :, None, :] - closest) ** 2, axis=-1)

        # Bond-order weighting when chemical bond orders are present (PTC-MR, MUTAG, etc.)
        if hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.shape[-1] >= 4:
            ea = data.edge_attr.float().cpu()
            bond_scale = (1.0 * ea[:, 0] + 2.0 * ea[:, 1] + 3.0 * ea[:, 2] + 1.5 * ea[:, 3]).numpy()
        else:
            bond_scale = np.ones((num_edges,), dtype=np.float32)

        edge_splats = np.exp(-dist_sq_edge / (2.0 * sigma_edge ** 2)) * bond_scale[None, None, :]
        edge_field = np.sum(edge_splats, axis=-1)
    else:
        edge_field = np.zeros((resolution, resolution), dtype=np.float32)

    e_min, e_max = edge_field.min(), edge_field.max()
    if e_max - e_min > 1e-6:
        edge_norm = (edge_field - e_min) / (e_max - e_min)
    else:
        edge_norm = edge_field
    ch2_spatial = torch.from_numpy(edge_norm.astype(np.float32))

    return torch.stack([ch0_manifold, ch1_laplacian, ch2_spatial], dim=0)


# =============================================================================
# 4. DATASET-LEVEL GCN ATTACHMENT & CACHING
# =============================================================================

def attach_gcn_embeddings(
    dataset,
    dataset_name: str,
    embedding_dim: int = 64,
    epochs: int = 200,
    cache_dir: str = "./cache"
):
    """
    Attaches continuous 64-dimensional pretrained GCN node representations to dataset graphs.
    If not found in cache, runs self-supervised GCN pretrainer automatically.
    """
    emb_cache_path = os.path.join(cache_dir, f"{dataset_name}_gcn_dim{embedding_dim}.pt")
    if not os.path.exists(emb_cache_path):
        print(f"[*] Pretrained GCN representations for {dataset_name} not found at {emb_cache_path}. Running pretrain_gcn now...")
        from pretrain_gcn import pretrain_gcn
        pretrain_gcn(dataset_name=dataset_name, epochs=epochs, embedding_dim=embedding_dim, cache_dir=cache_dir)

    print(f"[*] Loading pretrained 64-dim GCN representations from: {emb_cache_path}")
    embeddings_list = torch.load(emb_cache_path, weights_only=False)
    for idx, data in enumerate(dataset):
        data.gcn_emb = embeddings_list[idx].float()
    return dataset


compute_expert_map = compute_gcn_map


def get_or_create_expert_maps(
    dataset,
    dataset_name: str,
    expert: str = "gcn",
    resolution: int = 64,
    layout: str = "spring",
    cache_dir: str = "./cache"
):
    """
    Loads or pre-rasterizes all graphs in the dataset into 3-channel heatmaps for the specified expert:
      - 'gcn': Graph Convolutional Network (isotropic smoothing)
      - 'gs': GraphSAGE (ego vs. neighborhood mean aggregation)
      - 'lightgcn': Pure structural propagation
    Saves cache to disk: cache/{dataset_name}_{expert}_mote_res{resolution}_{layout}.pt
    """
    expert = expert.lower()
    if expert == "sage":
        expert = "gs"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{dataset_name}_{expert}_mote_res{resolution}_{layout}.pt")

    if os.path.exists(cache_path):
        print(f"[*] Loading pre-rasterized {expert.upper()} maps for {dataset_name} from: {cache_path}")
        cache_data = torch.load(cache_path, weights_only=False)
        return cache_data['X'], cache_data['Y']

    # Load pretrained representations
    emb_cache_path = os.path.join(cache_dir, f"{dataset_name}_{expert}_dim64.pt")
    if not os.path.exists(emb_cache_path):
        print(f"[*] Pretrained {expert.upper()} representations for {dataset_name} not found at {emb_cache_path}. Running pretrainer...")
        if expert == "gcn":
            from pretrain_gcn import pretrain_gcn
            pretrain_gcn(dataset_name=dataset_name, epochs=200, embedding_dim=64, cache_dir=cache_dir)
        elif expert == "gs":
            from pretrain_gs import pretrain_gs
            pretrain_gs(dataset_name=dataset_name, epochs=200, embedding_dim=64, cache_dir=cache_dir)
        elif expert == "lightgcn":
            from pretrain_lightgcn import pretrain_lightgcn
            pretrain_lightgcn(dataset_name=dataset_name, epochs=200, embedding_dim=64, cache_dir=cache_dir)
        else:
            raise ValueError(f"Unsupported expert: '{expert}'. Supported: 'gcn', 'gs', 'lightgcn'")

    print(f"[*] Loading pretrained 64-dim {expert.upper()} representations from: {emb_cache_path}")
    embeddings_list = torch.load(emb_cache_path, weights_only=False)

    print(f"[*] Pre-rasterizing {len(dataset)} graphs for {dataset_name} ({expert.upper()} Maps, Resolution: {resolution}x{resolution}, Layout: {layout})...")
    start_t = time.time()
    maps_list = []
    labels_list = []

    for idx, data in enumerate(tqdm(dataset, desc=f"Rasterizing {expert.upper()} Maps: {dataset_name}", unit="graph")):
        emb = embeddings_list[idx].float()
        m = compute_expert_map(data, emb, resolution=resolution, layout_method=layout, seed=42 + idx)
        maps_list.append(m)
        y_val = data.y.item() if hasattr(data.y, 'item') else int(data.y)
        labels_list.append(y_val)

    X = torch.stack(maps_list, dim=0)  # Shape: (N, 3, H, W)
    Y = torch.tensor(labels_list, dtype=torch.long)

    # Normalize labels to 0..C-1
    unique_labels = torch.unique(Y).sort().values
    label_map = {old.item(): new for new, old in enumerate(unique_labels)}
    Y_norm = torch.tensor([label_map[y.item()] for y in Y], dtype=torch.long)

    print(f"[+] Rasterization completed in {time.time() - start_t:.2f}s! Tensor shape: {X.shape}, Classes: {len(unique_labels)}")
    torch.save({'X': X, 'Y': Y_norm}, cache_path)
    print(f"[+] Saved rasterized maps to: {cache_path}")
    return X, Y_norm


def get_or_create_gcn_maps(
    dataset,
    dataset_name: str,
    resolution: int = 64,
    layout: str = "spring",
    cache_dir: str = "./cache"
):
    return get_or_create_expert_maps(
        dataset=dataset,
        dataset_name=dataset_name,
        expert="gcn",
        resolution=resolution,
        layout=layout,
        cache_dir=cache_dir
    )


def get_or_create_gs_maps(
    dataset,
    dataset_name: str,
    resolution: int = 64,
    layout: str = "spring",
    cache_dir: str = "./cache"
):
    return get_or_create_expert_maps(
        dataset=dataset,
        dataset_name=dataset_name,
        expert="gs",
        resolution=resolution,
        layout=layout,
        cache_dir=cache_dir
    )

