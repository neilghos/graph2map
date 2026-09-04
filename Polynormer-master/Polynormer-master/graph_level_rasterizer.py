"""
Graph-to-Map Rasterizer (Graph2Map).
Converts graph structures into continuous multi-channel 2D topological heatmaps.
Runs entirely on PyTorch, NumPy, NetworkX, and PIL without external GUI dependencies.
"""

import numpy as np
import torch
import networkx as nx
from PIL import Image
from typing import Tuple, Optional, Union
from torch_geometric.data import Data


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

    # Build NetworkX graph
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
            pos_dict = None  # Fallback to spring if spectral fails (e.g. disconnected)

    if pos_dict is None and method == "kamada_kawai":
        try:
            pos_dict = nx.kamada_kawai_layout(G, scale=1.0)
        except Exception:
            pos_dict = None

    if pos_dict is None:
        # Default robust fallback: spring layout
        pos_dict = nx.spring_layout(G, seed=seed, scale=1.0, iterations=50)

    coords = np.array([pos_dict[i] for i in range(num_nodes)], dtype=np.float32)

    # Center and normalize coordinates to [-0.75, 0.75] to avoid clipping at canvas edges
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

    # Build adjacency matrix A
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    A[edge_index[0], edge_index[1]] = 1.0

    # Degree matrix D
    deg = A.sum(dim=1)
    inv_deg = torch.zeros_like(deg)
    mask = deg > 0
    inv_deg[mask] = 1.0 / deg[mask]

    # Random walk transition matrix P = D^{-1} A
    P = torch.diag(inv_deg) @ A

    P2 = P @ P
    P3 = P2 @ P
    P4 = P3 @ P

    # Weighted echo score: 3-hop (triangles) and 4-hop (cycles/basins)
    echo = (
        0.5 * torch.diag(P2) +
        2.0 * torch.diag(P3) +
        1.0 * torch.diag(P4)
    ).cpu().numpy()

    # Normalize to [0, 1]
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
    Compute feature dissonance / disparity across each edge:
    - If node features exist: ||x_u - x_v||_2 (semantic boundaries)
    - If no node features (e.g. social networks): |log(1+deg_u) - log(1+deg_v)| (structural degree gradient)
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
        # Fallback to degree gradient across edges
        deg = torch.bincount(edge_index[0], minlength=num_nodes).float()
        log_deg = torch.log1p(deg)
        diff = torch.abs(log_deg[u] - log_deg[v]).cpu().numpy()

    if diff.max() - diff.min() > 1e-6:
        diff = (diff - diff.min()) / (diff.max() - diff.min())
    else:
        diff = np.ones((num_edges,), dtype=np.float32)
    return diff.astype(np.float32)


def graph_to_map(
    data: Data,
    resolution: int = 64,
    sigma_node: float = 0.08,
    sigma_edge: float = 0.04,
    layout_method: str = "spring",
    include_feature_grad: bool = True,
    seed: int = 42
) -> torch.Tensor:
    """
    Rasterize a PyG Data graph into a continuous multi-channel 2D heatmap tensor (C, H, W).
    
    Channels:
      - Ch 0: Node Density Field (Gaussian KDE splatting)
      - Ch 1: Edge Flux / Line Density Field (Continuous line segment splatting)
      - Ch 2: Multi-Hop "Echo" Basin (Random walk return probability splatting)
      - Ch 3: Feature Gradient / Boundary Field (Edge disparity splatting)
    """
    num_nodes = data.num_nodes
    edge_index = data.edge_index

    # 1. 2D Coordinate Layout (N, 2) in [-1, 1]
    coords = get_2d_layout(num_nodes, edge_index, method=layout_method, seed=seed)

    # 2. Create 2D coordinate grid (H, W, 2)
    lin = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(lin, lin, indexing="ij")
    # Shape: (H, W, 2)
    grid = np.stack([grid_x, grid_y], axis=-1)

    # --- Channel 0: Node Density Field ---
    # grid: (H, W, 1, 2), coords: (1, 1, N, 2)
    if num_nodes > 0:
        diff_nodes = grid[:, :, None, :] - coords[None, None, :, :]
        dist_sq_nodes = np.sum(diff_nodes ** 2, axis=-1)  # (H, W, N)
        node_field = np.sum(np.exp(-dist_sq_nodes / (2.0 * sigma_node ** 2)), axis=-1)
    else:
        node_field = np.zeros((resolution, resolution), dtype=np.float32)

    # --- Channel 1: Edge Density Field (Gaussian line segment splatting) ---
    num_edges = edge_index.shape[1]
    if num_edges > 0:
        edges = edge_index.t().cpu().numpy()
        p0 = coords[edges[:, 0]]  # (E, 2)
        p1 = coords[edges[:, 1]]  # (E, 2)
        v = p1 - p0               # (E, 2)
        len_sq = np.sum(v ** 2, axis=-1)  # (E,)
        len_sq = np.maximum(len_sq, 1e-8)

        # Vectorized segment distance calculation:
        # grid: (H, W, 1, 2), p0: (1, 1, E, 2), v: (1, 1, E, 2)
        p_minus_p0 = grid[:, :, None, :] - p0[None, None, :, :]  # (H, W, E, 2)
        # Dot product with v
        dot = np.sum(p_minus_p0 * v[None, None, :, :], axis=-1)  # (H, W, E)
        t = np.clip(dot / len_sq[None, None, :], 0.0, 1.0)        # (H, W, E)
        # Closest point on segment
        closest = p0[None, None, :, :] + t[:, :, :, None] * v[None, None, :, :]
        dist_sq_edge = np.sum((grid[:, :, None, :] - closest) ** 2, axis=-1)  # (H, W, E)

        edge_field = np.sum(np.exp(-dist_sq_edge / (2.0 * sigma_edge ** 2)), axis=-1)
    else:
        edge_field = np.zeros((resolution, resolution), dtype=np.float32)

    # --- Channel 2: Multi-Hop "Echo" Basin ---
    if num_nodes > 0:
        echo_weights = compute_multihop_echo(num_nodes, edge_index)  # (N,)
        echo_splat = np.exp(-dist_sq_nodes / (2.0 * sigma_node ** 2)) * echo_weights[None, None, :]
        echo_field = np.sum(echo_splat, axis=-1)
    else:
        echo_field = np.zeros((resolution, resolution), dtype=np.float32)

    # --- Channel 3: Feature Boundary Field ---
    if include_feature_grad and num_edges > 0 and data.x is not None:
        feat_weights = compute_feature_gradients(num_nodes, edge_index, data.x)  # (E,)
        feat_splat = np.exp(-dist_sq_edge / (2.0 * sigma_edge ** 2)) * feat_weights[None, None, :]
        feat_field = np.sum(feat_splat, axis=-1)
    else:
        feat_field = np.zeros((resolution, resolution), dtype=np.float32)

    # Normalize each channel to [0, 1] for stable neural network inputs
    channels = [node_field, edge_field, echo_field, feat_field]
    normalized_channels = []
    for ch in channels:
        ch_min = ch.min()
        ch_max = ch.max()
        if ch_max - ch_min > 1e-6:
            ch_norm = (ch - ch_min) / (ch_max - ch_min)
        else:
            ch_norm = ch
        normalized_channels.append(ch_norm.astype(np.float32))

    # Output shape: (C, H, W)
    tensor = torch.from_numpy(np.stack(normalized_channels, axis=0))
    return tensor


def colormap_inferno(val: np.ndarray) -> np.ndarray:
    """
    Fast perceptually uniform Inferno-like colormap implemented via smooth polynomials.
    Takes normalized 2D array in [0, 1], returns RGB (H, W, 3) in uint8 [0, 255].
    """
    x = np.clip(val, 0.0, 1.0)
    # Polynomial approximations of Inferno colormap
    r = np.clip(2.5 * x**2 - 1.2 * x**3 + 0.1 * x, 0.0, 1.0)
    g = np.clip(1.8 * x**3 - 0.2 * x**2 + 0.05 * x, 0.0, 1.0)
    b = np.clip(1.5 * x - 2.0 * x**2 + 1.2 * x**3, 0.0, 1.0)
    # Highlight highest values with warm yellow/white
    highlight = np.clip((x - 0.7) * 3.33, 0.0, 1.0)
    r = np.clip(r + 0.3 * highlight, 0.0, 1.0)
    g = np.clip(g + 0.7 * highlight, 0.0, 1.0)
    b = np.clip(b + 0.5 * highlight, 0.0, 1.0)

    rgb = np.stack([r, g, b], axis=-1) * 255.0
    return rgb.astype(np.uint8)


def colormap_plasma(val: np.ndarray) -> np.ndarray:
    """
    Plasma-like colormap (Purple -> Pink -> Orange -> Yellow).
    """
    x = np.clip(val, 0.0, 1.0)
    r = np.clip(0.05 + 1.5 * x - 0.6 * x**2, 0.0, 1.0)
    g = np.clip(0.02 + 0.1 * x + 0.9 * x**3, 0.0, 1.0)
    b = np.clip(0.5 + 0.8 * x - 1.5 * x**2 + 0.3 * x**3, 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1) * 255.0
    return rgb.astype(np.uint8)


def save_heatmap_panel(
    tensor: torch.Tensor, 
    filepath: str, 
    channel_names: Optional[list] = None
):
    """
    Save all channels of a graph map side-by-side as a PNG image using PIL.
    """
    if channel_names is None:
        channel_names = ["Node Density", "Edge Flux", "Echo Basin", "Feature Gradient"]

    C, H, W = tensor.shape
    colormaps = [colormap_inferno, colormap_plasma, colormap_inferno, colormap_plasma]

    images = []
    for c in range(C):
        ch_arr = tensor[c].cpu().numpy()
        cmap = colormaps[c % len(colormaps)]
        rgb = cmap(ch_arr)
        img = Image.fromarray(rgb)
        images.append(img)

    # Stitch side-by-side with padding
    spacing = 8
    total_w = C * W + (C - 1) * spacing
    total_h = H

    combined = Image.new("RGB", (total_w, total_h), color=(20, 20, 25))
    for i, img in enumerate(images):
        combined.paste(img, (i * (W + spacing), 0))

    combined.save(filepath)
    print(f"Saved heatmap panel to {filepath}")
