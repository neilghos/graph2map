"""
Graph-to-Map Rasterizer (Graph2Map) for Graph Classification.
Converts graph structures into continuous multi-channel 2D topological heatmaps.
Runs entirely on PyTorch, NumPy, NetworkX, and PIL without external GUI dependencies.
"""

import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
from PIL import Image
from typing import Tuple, Optional, Union
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops


def get_2d_layout(
    num_nodes: int, 
    edge_index: torch.Tensor, 
    method: str = "spring",
    seed: int = 42
) -> np.ndarray:
    """
    Compute 2D continuous coordinates (x, y) in [-0.8, 0.8]^2 for graph nodes.
    Methods:
      - 'spring': Force-directed Fruchterman-Reingold layout
      - 'spectral': 2nd and 3rd eigenvectors of the Normalized Laplacian (Laplacian Eigenmaps)
      - 'kamada_kawai': Path-distance energy minimization
    """
    if num_nodes <= 1:
        return np.zeros((num_nodes, 2), dtype=np.float32)

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.t().cpu().numpy()
    for u, v in edges:
        if u != v:
            G.add_edge(int(u), int(v))

    pos_dict = None
    if method == "spectral":
        try:
            pos_dict = nx.spectral_layout(G, scale=1.0)
        except Exception:
            pos_dict = None

    if pos_dict is None and method == "kamada_kawai":
        try:
            pos_dict = nx.kamada_kawai_layout(G, scale=1.0)
        except Exception:
            pos_dict = None

    if pos_dict is None:
        pos_dict = nx.spring_layout(G, seed=seed, scale=1.0, iterations=50)

    coords = np.array([pos_dict[i] for i in range(num_nodes)], dtype=np.float32)

    if coords.shape[0] > 1:
        coords -= coords.mean(axis=0, keepdims=True)
        max_val = np.max(np.abs(coords))
        if max_val > 1e-6:
            coords = (coords / max_val) * 0.75

    return coords


def compute_multihop_echo(num_nodes: int, edge_index: torch.Tensor) -> np.ndarray:
    """
    Compute random walk return probabilities (signal bounce-back / echo):
    - 2-hop return P^2_{ii} (degree proportional)
    - 3-hop return P^3_{ii} (triangle participation)
    - 4-hop return P^4_{ii} (4-cycles and trapped echo basins)
    """
    if num_nodes <= 1:
        return np.ones((num_nodes,), dtype=np.float32)

    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    A[edge_index[0], edge_index[1]] = 1.0

    deg = A.sum(dim=1)
    inv_deg = torch.zeros_like(deg)
    mask = deg > 0
    inv_deg[mask] = 1.0 / deg[mask]

    P = torch.diag(inv_deg) @ A

    P2 = P @ P
    P3 = P2 @ P
    P4 = P3 @ P

    echo = (
        0.5 * torch.diag(P2) +
        2.0 * torch.diag(P3) +
        1.0 * torch.diag(P4)
    ).cpu().numpy()

    if echo.max() - echo.min() > 1e-6:
        echo = (echo - echo.min()) / (echo.max() - echo.min())
    else:
        echo = np.ones((num_nodes,), dtype=np.float32)

    return echo.astype(np.float32)


def compute_feature_gradients(
    num_nodes: int, 
    edge_index: torch.Tensor, 
    x: Optional[torch.Tensor]
) -> np.ndarray:
    """
    Compute feature dissonance / disparity across each edge.
    """
    num_edges = edge_index.shape[1]
    if num_edges == 0:
        return np.ones((0,), dtype=np.float32)

    u = edge_index[0]
    v = edge_index[1]

    if x is not None and x.numel() > 0 and x.shape[-1] > 0:
        x_u = x[u].float()
        x_v = x[v].float()
        diff = torch.norm(x_u - x_v, p=2, dim=1).cpu().numpy()
    else:
        deg = torch.bincount(edge_index[0], minlength=num_nodes).float()
        log_deg = torch.log1p(deg)
        diff = torch.abs(log_deg[u] - log_deg[v]).cpu().numpy()

    if diff.max() - diff.min() > 1e-6:
        diff = (diff - diff.min()) / (diff.max() - diff.min())
    else:
        diff = np.ones((num_edges,), dtype=np.float32)
    return diff.astype(np.float32)


def get_canonical_node_order(num_nodes: int, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Computes a canonical 1D ordering of graph nodes along the principal manifold.
    Uses the Fiedler vector (2nd eigenvector of the graph Laplacian), which provides
    the optimal 1D continuous embedding minimizing edge stretch.
    Falls back gracefully to degree/centrality ordering for disconnected or degenerate graphs.
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


def compute_graph_level_spectrogram(
    data: Data,
    num_hops: int = 16,
    resolution: int = 64,
    ppr_alpha: float = 0.20,
    gamma_spec: float = 20.0
) -> torch.Tensor:
    """
    Computes the 3-Channel Graph-Level Spectrogram Map:
      - Ch 0: Spectral Low-Pass Community Consensus (A^k * X)
      - Ch 1: Spectral High-Pass Boundary Wavelet Gradient (Delta A^k * X)
      - Ch 2: Structural PageRank / Echo Resonance (PPR dynamics)

    Uses canonical Fiedler 1D node ordering on the horizontal axis and multi-hop
    diffusion scale on the vertical axis, with Log-Mel dynamic range compression.
    """
    num_nodes = data.num_nodes
    edge_index = data.edge_index

    if num_nodes == 0:
        return torch.zeros((3, resolution, resolution), dtype=torch.float32)

    # Signal setup
    if data.x is not None and data.x.numel() > 0:
        x = data.x.float()
    else:
        deg = torch.bincount(edge_index[0], minlength=num_nodes).float().unsqueeze(1)
        x = torch.log1p(deg)

    # Symmetric normalized adjacency: D^{-1/2} (A + I) D^{-1/2}
    edge_index_loop, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index_loop[0], edge_index_loop[1]
    deg = torch.bincount(row, minlength=num_nodes).float()
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    val = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    adj_norm = torch.sparse_coo_tensor(edge_index_loop, val, (num_nodes, num_nodes)).coalesce()

    # Canonical 1D node ordering via Fiedler eigenvector
    order = get_canonical_node_order(num_nodes, edge_index)

    s0_hops, s1_hops, s2_hops = [], [], []
    x_k = x
    p_k = x

    s0_hops.append(x)
    s1_hops.append(torch.zeros_like(x))
    s2_hops.append(x)

    for k in range(1, num_hops):
        x_next = torch.sparse.mm(adj_norm, x_k)
        s0_hops.append(x_next)
        s1_hops.append(x_k - x_next)
        x_k = x_next

        p_next = ppr_alpha * x + (1.0 - ppr_alpha) * torch.sparse.mm(adj_norm, p_k)
        s2_hops.append(p_next)
        p_k = p_next

    # Energy aggregation per node at each hop
    def to_energy_map(hops_list):
        stacked = torch.stack(hops_list, dim=0)  # (K, N, D)
        if stacked.shape[-1] > 1:
            energy = torch.norm(stacked, p=2, dim=-1)  # (K, N)
        else:
            energy = stacked.squeeze(-1)  # (K, N)
        # Apply canonical node ordering along horizontal axis
        energy = energy[:, order]
        return energy

    c0 = to_energy_map(s0_hops)
    c1 = to_energy_map(s1_hops)
    c2 = to_energy_map(s2_hops)

    # 2D continuous bilinear interpolation to (resolution, resolution) + Log-Mel compression
    log_denom = float(np.log(1.0 + gamma_spec))

    def resize_and_compress(mat):
        m_4d = mat.unsqueeze(0).unsqueeze(0)  # (1, 1, K, N)
        resized = F.interpolate(m_4d, size=(resolution, resolution), mode='bilinear', align_corners=False).squeeze()
        m_min = resized.min()
        m_scaled = (resized - m_min) / (resized.max() - m_min + 1e-6)
        m_norm = torch.log1p(gamma_spec * m_scaled) / log_denom
        return m_norm

    spec0 = resize_and_compress(c0)
    spec1 = resize_and_compress(c1)
    spec2 = resize_and_compress(c2)

    return torch.stack([spec0, spec1, spec2], dim=0)


def compute_feature_manifold_map(
    data: Data,
    resolution: int = 64
) -> torch.Tensor:
    """
    Computes 2-Channel Feature-Manifold Map:
      - Ch 0: Normalized Raw Feature Distribution (Features x Canonical Nodes)
      - Ch 1: Normalized Graph Laplacian Feature Gradient (Boundary / Transition Curvature)

    X-axis (Horizontal): Canonical Nodes ordered along 1D Fiedler Laplacian manifold
    Y-axis (Vertical): Feature dimensions (atom types, functional attributes)
    """
    num_nodes = data.num_nodes
    edge_index = data.edge_index

    if num_nodes == 0:
        return torch.zeros((2, resolution, resolution), dtype=torch.float32)

    # 1. Feature matrix setup
    if data.x is not None and data.x.numel() > 0:
        x = data.x.float()
        if x.dim() == 1:
            x = x.unsqueeze(1)
    else:
        deg = torch.bincount(edge_index[0], minlength=num_nodes).float().unsqueeze(1)
        x = torch.log1p(deg)

    # 2. Canonical 1D node ordering via Fiedler eigenvector
    order = get_canonical_node_order(num_nodes, edge_index)

    # 3. Channel 0: Raw feature matrix (Features x Nodes)
    x_ordered = x[order, :]  # (N, D)
    raw_mat = x_ordered.t()  # (D, N)

    # 4. Channel 1: Normalized Graph Laplacian Feature Gradient
    if edge_index.shape[1] > 0 and num_nodes > 1:
        edge_index_loop, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index_loop[0], edge_index_loop[1]
        deg = torch.bincount(row, minlength=num_nodes).float()
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        val = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        adj_norm = torch.sparse_coo_tensor(edge_index_loop, val, (num_nodes, num_nodes)).coalesce()

        x_smooth = torch.sparse.mm(adj_norm, x)
        laplacian_x = torch.abs(x - x_smooth)
    else:
        laplacian_x = torch.zeros_like(x)

    laplacian_ordered = laplacian_x[order, :]
    lap_mat = laplacian_ordered.t()  # (D, N)

    # 5. Continuous bilinear interpolation to (resolution, resolution)
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

    ch0 = to_channel_map(raw_mat)
    ch1 = to_channel_map(lap_mat)

    return torch.stack([ch0, ch1], dim=0)


def graph_to_map(
    data: Data,
    resolution: int = 64,
    sigma_node: float = 0.08,
    sigma_edge: float = 0.04,
    layout_method: str = "spring",
    include_feature_grad: bool = True,
    channel_mode: str = "7ch",
    include_spectrogram: Optional[bool] = None,
    seed: int = 42
) -> torch.Tensor:
    """
    Rasterize a PyG Data graph into a continuous multi-channel 2D heatmap tensor (C, H, W).
    
    Modes:
      - '7ch' (Master Atlas): 2 Feature-Manifold Channels (Raw X + Laplacian LX) + 5 Spatial Cartography Channels
      - '8ch' (Legacy Atlas): 3 Hop-Diffusion Spectrogram Channels + 5 Spatial Cartography Channels
      - '5ch' (Spatial Only): 5 Spatial Cartography Channels
      - '2ch' (Feature Only): 2 Feature-Manifold Channels (Raw X + Laplacian LX)
    """
    num_nodes = data.num_nodes
    edge_index = data.edge_index

    # 1. 2D Coordinate Layout (N, 2) in [-1, 1]
    coords = get_2d_layout(num_nodes, edge_index, method=layout_method, seed=seed)

    # 2. 2D coordinate grid (H, W, 2)
    lin = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(lin, lin, indexing="ij")
    grid = np.stack([grid_x, grid_y], axis=-1)

    # --- Channel 0: Node Density Field ---
    if num_nodes > 0:
        diff_nodes = grid[:, :, None, :] - coords[None, None, :, :]
        dist_sq_nodes = np.sum(diff_nodes ** 2, axis=-1)  # (H, W, N)
        node_field = np.sum(np.exp(-dist_sq_nodes / (2.0 * sigma_node ** 2)), axis=-1)
    else:
        node_field = np.zeros((resolution, resolution), dtype=np.float32)

    # --- Channel 1: Edge Density Field ---
    num_edges = edge_index.shape[1]
    if num_edges > 0:
        edges = edge_index.t().cpu().numpy()
        p0 = coords[edges[:, 0]]
        p1 = coords[edges[:, 1]]
        v = p1 - p0
        len_sq = np.sum(v ** 2, axis=-1)
        len_sq = np.maximum(len_sq, 1e-8)

        p_minus_p0 = grid[:, :, None, :] - p0[None, None, :, :]
        dot = np.sum(p_minus_p0 * v[None, None, :, :], axis=-1)
        t = np.clip(dot / len_sq[None, None, :], 0.0, 1.0)
        closest = p0[None, None, :, :] + t[:, :, :, None] * v[None, None, :, :]
        dist_sq_edge = np.sum((grid[:, :, None, :] - closest) ** 2, axis=-1)

        edge_field = np.sum(np.exp(-dist_sq_edge / (2.0 * sigma_edge ** 2)), axis=-1)
    else:
        edge_field = np.zeros((resolution, resolution), dtype=np.float32)

    # --- Channel 2: Multi-Hop Echo Basin ---
    if num_nodes > 0:
        echo_weights = compute_multihop_echo(num_nodes, edge_index)
        echo_splat = np.exp(-dist_sq_nodes / (2.0 * sigma_node ** 2)) * echo_weights[None, None, :]
        echo_field = np.sum(echo_splat, axis=-1)
    else:
        echo_field = np.zeros((resolution, resolution), dtype=np.float32)

    # --- Channel 3: Feature Boundary Field ---
    if include_feature_grad and num_edges > 0 and data.x is not None:
        feat_weights = compute_feature_gradients(num_nodes, edge_index, data.x)
        feat_splat = np.exp(-dist_sq_edge / (2.0 * sigma_edge ** 2)) * feat_weights[None, None, :]
        feat_field = np.sum(feat_splat, axis=-1)
    else:
        feat_field = np.zeros((resolution, resolution), dtype=np.float32)

    # --- Channel 4: Node Semantic / Feature Field ---
    if num_nodes > 0 and data.x is not None and data.x.numel() > 0:
        x_float = data.x.float()
        if x_float.shape[1] > 1:
            node_intensity = torch.norm(x_float, p=2, dim=1).cpu().numpy()
        else:
            node_intensity = x_float[:, 0].cpu().numpy()
        if node_intensity.max() - node_intensity.min() > 1e-6:
            node_intensity = (node_intensity - node_intensity.min()) / (node_intensity.max() - node_intensity.min())
        else:
            node_intensity = np.ones((num_nodes,), dtype=np.float32)
        node_feat_splat = np.exp(-dist_sq_nodes / (2.0 * sigma_node ** 2)) * node_intensity[None, None, :]
        node_feat_field = np.sum(node_feat_splat, axis=-1)
    else:
        node_feat_field = np.zeros((resolution, resolution), dtype=np.float32)

    # Normalize spatial channels to [0, 1]
    channels = [node_field, edge_field, echo_field, feat_field, node_feat_field]
    normalized_channels = []
    for ch in channels:
        ch_min = ch.min()
        ch_max = ch.max()
        if ch_max - ch_min > 1e-6:
            ch_norm = (ch - ch_min) / (ch_max - ch_min)
        else:
            ch_norm = ch
        normalized_channels.append(ch_norm.astype(np.float32))

    spatial_tensor = torch.from_numpy(np.stack(normalized_channels, axis=0))

    if include_spectrogram is not None:
        if not include_spectrogram:
            channel_mode = "5ch"
        elif channel_mode == "5ch":
            channel_mode = "8ch"

    if channel_mode == "7ch":
        feat_manifold_2ch = compute_feature_manifold_map(data, resolution=resolution)
        return torch.cat([feat_manifold_2ch, spatial_tensor], dim=0)  # Shape: (7, H, W)
    elif channel_mode == "8ch":
        spectrogram_3ch = compute_graph_level_spectrogram(data, num_hops=16, resolution=resolution)
        return torch.cat([spectrogram_3ch, spatial_tensor], dim=0)  # Shape: (8, H, W)
    elif channel_mode == "2ch":
        return compute_feature_manifold_map(data, resolution=resolution)  # Shape: (2, H, W)
    else:  # "5ch"
        return spatial_tensor  # Shape: (5, H, W)

