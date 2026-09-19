"""
Expert Rasterizer for Tripartite Latent Maps (expert_rasterizer.py).
Transforms pretrained GCN, Random Walk, and Spectral node representations into 3-channel 2D convolutional heatmaps.

3 Continuous Topological Channels per Expert:
  - Channel 0: Expert Latent Manifold (Y: 64 continuous latent dimensions, X: 1D Fiedler node sequence)
  - Channel 1: Expert Laplacian Gradient (|H - A_norm H| - exact 1-to-1 co-registered boundary transitions)
  - Channel 2: Continuous 2D Spatial Layout Edge Flux (Euclidean bond geometry weighted by bond orders)
"""

import os
import sys
import time
from typing import Optional, Tuple, Dict
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
        return torch.argsort(fiedler)
    except Exception:
        deg = torch.bincount(edge_index[0], minlength=num_nodes).float()
        return torch.argsort(-deg)


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
# 3. CLEAN 3-CHANNEL MAP RASTERIZATION
# =============================================================================

def to_channel_map(mat: torch.Tensor, resolution: int = 64) -> torch.Tensor:
    """Bilinear interpolation helper to resample (D, N) -> (resolution, resolution) normalized to [0, 1]."""
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


def compute_spatial_channel(
    data: Data,
    resolution: int = 64,
    layout_method: str = "spring",
    sigma_edge: float = 0.04,
    seed: int = 42
) -> torch.Tensor:
    """Renders the continuous 2D Euclidean edge flux channel."""
    num_nodes = data.num_nodes
    edge_index = data.edge_index

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
    return torch.from_numpy(edge_norm.astype(np.float32))


def compute_expert_map(
    data: Data,
    emb: torch.Tensor,
    resolution: int = 64,
    layout_method: str = "spring",
    sigma_edge: float = 0.04,
    seed: int = 42,
    cached_spatial: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Renders a single graph and an arbitrary node representation (GCN, Walk, or Spectral)
    into a pristine 3-channel 2D convolutional heatmap:
      - Channel 0: Latent Manifold (Y: 64 continuous latent dimensions, X: 1D Fiedler sequence)
      - Channel 1: Normalized Laplacian Gradient (|H - A_norm H| - 1:1 co-registered boundary field)
      - Channel 2: Continuous 2D Spatial Layout Edge Flux
    """
    num_nodes = data.num_nodes
    edge_index = data.edge_index

    if num_nodes == 0:
        return torch.zeros((3, resolution, resolution), dtype=torch.float32)

    H = emb.float().to(edge_index.device)  # (N, D)

    # 1. Canonical 1D ordering along principal Laplacian manifold
    order = get_canonical_node_order(num_nodes, edge_index).to(H.device)
    H_ordered = H[order, :]  # (N, D)
    raw_mat = H_ordered.t()  # (D, N)

    # 2. Normalized Laplacian Gradient Operator
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

    ch0 = to_channel_map(raw_mat, resolution=resolution)
    ch1 = to_channel_map(lap_ordered, resolution=resolution)

    if cached_spatial is not None:
        ch2 = cached_spatial
    else:
        ch2 = compute_spatial_channel(data, resolution=resolution, layout_method=layout_method, sigma_edge=sigma_edge, seed=seed)

    return torch.stack([ch0, ch1, ch2], dim=0)


compute_gcn_map = compute_expert_map


# =============================================================================
# 4. DATASET-LEVEL TRIPARTITE ATTACHMENT & CACHING
# =============================================================================

def get_or_create_tripartite_maps(
    dataset,
    dataset_name: str,
    resolution: int = 64,
    layout: str = "spring",
    cache_dir: str = "./cache",
    force: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loads or pre-rasterizes all graphs into three separate 3-channel heatmap streams:
      - X_gcn: (N, 3, resolution, resolution)
      - X_walk: (N, 3, resolution, resolution)
      - X_spec: (N, 3, resolution, resolution)
      - Y: (N,) labels
    Saves cache to: cache/{dataset_name}_tripartite_res{resolution}_{layout}.pt
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{dataset_name}_tripartite_res{resolution}_{layout}.pt")

    if os.path.exists(cache_path) and not force:
        print(f"[*] Loading pre-rasterized Tripartite maps for {dataset_name} from: {cache_path}")
        cache_data = torch.load(cache_path, weights_only=False)
        return cache_data['X_gcn'], cache_data['X_walk'], cache_data['X_spec'], cache_data['Y']

    # 1. Ensure GCN, Walk, and Spectral representations exist
    gcn_path = os.path.join(cache_dir, f"{dataset_name}_gcn_dim64.pt")
    walk_path = os.path.join(cache_dir, f"{dataset_name}_walk_dim64.pt")
    spec_path = os.path.join(cache_dir, f"{dataset_name}_spectral_dim64.pt")

    if not (os.path.exists(gcn_path) and os.path.exists(walk_path) and os.path.exists(spec_path)) or force:
        print(f"[*] Pretrained embeddings missing or force requested. Running pretrain_gcn for {dataset_name}...")
        from pretrain_gcn import pretrain_gcn
        pretrain_gcn(dataset_name=dataset_name, epochs=200, embedding_dim=64, cache_dir=cache_dir, force_retrain=force)

    print(f"[*] Loading pretrained representations for {dataset_name}:")
    print(f"    - GCN:      {gcn_path}")
    print(f"    - Walk:     {walk_path}")
    print(f"    - Spectral: {spec_path}")
    gcn_embs = torch.load(gcn_path, weights_only=False)
    walk_embs = torch.load(walk_path, weights_only=False)
    spec_embs = torch.load(spec_path, weights_only=False)

    print(f"[*] Pre-rasterizing {len(dataset)} graphs into 3-channel Tripartite maps for {dataset_name}...")
    start_t = time.time()
    gcn_list, walk_list, spec_list, labels_list = [], [], [], []

    for idx, data in enumerate(tqdm(dataset, desc=f"Rasterizing Tripartite Maps: {dataset_name}", unit="graph")):
        eg = gcn_embs[idx].float()
        ew = walk_embs[idx].float()
        es = spec_embs[idx].float()

        # Compute spatial channel once and reuse across all 3 experts
        spatial_ch = compute_spatial_channel(data, resolution=resolution, layout_method=layout, seed=42 + idx)

        m_g = compute_expert_map(data, eg, resolution=resolution, layout_method=layout, cached_spatial=spatial_ch)
        m_w = compute_expert_map(data, ew, resolution=resolution, layout_method=layout, cached_spatial=spatial_ch)
        m_s = compute_expert_map(data, es, resolution=resolution, layout_method=layout, cached_spatial=spatial_ch)

        gcn_list.append(m_g)
        walk_list.append(m_w)
        spec_list.append(m_s)

        y_val = data.y.item() if hasattr(data.y, 'item') else int(data.y)
        labels_list.append(y_val)

    X_gcn = torch.stack(gcn_list, dim=0)    # (N, 3, H, W)
    X_walk = torch.stack(walk_list, dim=0)  # (N, 3, H, W)
    X_spec = torch.stack(spec_list, dim=0)  # (N, 3, H, W)
    Y = torch.tensor(labels_list, dtype=torch.long)

    # Normalize labels to 0..C-1
    unique_labels = torch.unique(Y).sort().values
    label_map = {old.item(): new for new, old in enumerate(unique_labels)}
    Y_norm = torch.tensor([label_map[y.item()] for y in Y], dtype=torch.long)

    print(f"[+] Rasterization completed in {time.time() - start_t:.2f}s! Tensors: {X_gcn.shape}, Classes: {len(unique_labels)}")
    torch.save({
        'X_gcn': X_gcn,
        'X_walk': X_walk,
        'X_spec': X_spec,
        'Y': Y_norm
    }, cache_path)
    print(f"[+] Saved rasterized Tripartite maps to: {cache_path}")
    return X_gcn, X_walk, X_spec, Y_norm


def get_or_create_gcn_maps(
    dataset,
    dataset_name: str,
    resolution: int = 64,
    layout: str = "spring",
    cache_dir: str = "./cache",
    force: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible helper returning GCN maps."""
    X_gcn, _, _, Y = get_or_create_tripartite_maps(
        dataset=dataset,
        dataset_name=dataset_name,
        resolution=resolution,
        layout=layout,
        cache_dir=cache_dir,
        force=force
    )
    return X_gcn, Y


get_or_create_expert_maps = get_or_create_tripartite_maps
