"""
Node-Level Ego-Map Rasterizer (Graph2Map for Node Classification).
Converts the k-hop local computational neighborhood of a target node into a
multi-channel continuous 2D spatial heatmap with the target node pinned at (0, 0).

Channels:
  - Ch 0: Ego-Density Field (Target at (0, 0), neighbors weighted by inverse hop distance)
  - Ch 1: Ego-Edge Interconnection Field (Continuous line splatting of intra-neighborhood edges)
  - Ch 2: Personalized Diffusion Field (Personalized PageRank / RWR starting from target node v)
  - Ch 3: Relative Feature Homophily Field (Cosine similarity between neighbor features and target v)
"""

import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
from PIL import Image, ImageDraw, ImageFilter
from typing import Optional, Tuple
from torch_geometric.utils import k_hop_subgraph


def compute_ego_layout(
    num_sub_nodes: int,
    sub_edge_index: torch.Tensor,
    target_idx: int,
    method: str = "concentric",
    seed: int = 42
) -> np.ndarray:
    """
    Compute 2D coordinates for the ego-network with the target node strictly pinned at (0, 0).
    Coordinates are normalized into [-0.75, 0.75]^2.
    """
    if num_sub_nodes <= 1:
        return np.zeros((num_sub_nodes, 2), dtype=np.float32)

    # Ultra-fast concentric polar rings supporting arbitrary hop counts (sub-millisecond BFS)
    if method == "concentric":
        coords = np.zeros((num_sub_nodes, 2), dtype=np.float32)
        coords[target_idx] = [0.0, 0.0]

        # BFS hop distances from target_idx
        adj_list = [[] for _ in range(num_sub_nodes)]
        if sub_edge_index.numel() > 0:
            for u, v in sub_edge_index.t().cpu().numpy():
                adj_list[u].append(int(v))

        hop_dist = [-1] * num_sub_nodes
        hop_dist[target_idx] = 0
        queue = [target_idx]
        for curr in queue:
            d = hop_dist[curr]
            for nbr in adj_list[curr]:
                if hop_dist[nbr] == -1:
                    hop_dist[nbr] = d + 1
                    queue.append(nbr)

        max_hop = max([d for d in hop_dist if d > 0] + [1])
        for k in range(1, max_hop + 1):
            nodes_at_hop = [n for n in range(num_sub_nodes) if hop_dist[n] == k]
            n_k = len(nodes_at_hop)
            if n_k > 0:
                r_k = (k / float(max_hop)) * 0.75
                for i, node in enumerate(sorted(nodes_at_hop)):
                    angle = (2.0 * np.pi * i) / n_k
                    coords[node] = [r_k * np.cos(angle), r_k * np.sin(angle)]

        return coords.astype(np.float32)

    # Pinned spring layout (slower, force-directed)
    G = nx.Graph()
    G.add_nodes_from(range(num_sub_nodes))
    edges = sub_edge_index.t().cpu().numpy()
    for u, v in edges:
        if u != v:
            G.add_edge(int(u), int(v))

    pos_dict = None
    if method == "pinned_spring":
        try:
            initial_pos = {target_idx: np.array([0.0, 0.0], dtype=np.float32)}
            for i in range(num_sub_nodes):
                if i != target_idx:
                    angle = (2.0 * np.pi * i) / max(num_sub_nodes - 1, 1)
                    initial_pos[i] = np.array([0.3 * np.cos(angle), 0.3 * np.sin(angle)], dtype=np.float32)

            pos_dict = nx.spring_layout(
                G,
                pos=initial_pos,
                fixed=[target_idx],
                iterations=15,
                seed=seed,
                scale=0.8
            )
        except Exception:
            pos_dict = None

    if pos_dict is None:
        return compute_ego_layout(num_sub_nodes, sub_edge_index, target_idx, method="concentric")

    coords = np.array([pos_dict[i] for i in range(num_sub_nodes)], dtype=np.float32)
    target_pos = coords[target_idx].copy()
    coords -= target_pos

    max_val = np.max(np.abs(coords))
    if max_val > 1e-6:
        coords = (coords / max_val) * 0.75
    coords[target_idx] = [0.0, 0.0]

    return coords.astype(np.float32)


def compute_personalized_diffusion(
    num_sub_nodes: int,
    sub_edge_index: torch.Tensor,
    target_idx: int,
    alpha: float = 0.15,
    max_iter: int = 20
) -> np.ndarray:
    """
    Compute Personalized PageRank (PPR) / Random Walk with Restart starting at target node.
    pi = alpha * (I - (1 - alpha) * D^{-1} A)^{-1} e_target
    """
    if num_sub_nodes <= 1:
        return np.ones((num_sub_nodes,), dtype=np.float32)

    # Adjacency matrix
    A = torch.zeros((num_sub_nodes, num_sub_nodes), dtype=torch.float32)
    A[sub_edge_index[0], sub_edge_index[1]] = 1.0

    deg = A.sum(dim=1)
    inv_deg = torch.zeros_like(deg)
    mask = deg > 0
    inv_deg[mask] = 1.0 / deg[mask]
    P = torch.diag(inv_deg) @ A

    # Power iteration for PPR
    e_target = torch.zeros((num_sub_nodes,), dtype=torch.float32)
    e_target[target_idx] = 1.0
    pi = e_target.clone()

    for _ in range(max_iter):
        pi = (1.0 - alpha) * (pi @ P) + alpha * e_target

    res = pi.cpu().numpy()
    if res.max() - res.min() > 1e-6:
        res = (res - res.min()) / (res.max() - res.min())
    else:
        res = np.ones((num_sub_nodes,), dtype=np.float32)

    return res.astype(np.float32)


def compute_relative_homophily(
    sub_x: Optional[torch.Tensor],
    target_idx: int
) -> np.ndarray:
    """
    Compute cosine similarity between neighbor features and the target node feature.
    Values are normalized into [0, 1].
    """
    if sub_x is None or sub_x.numel() == 0:
        return np.ones((sub_x.shape[0] if sub_x is not None else 1,), dtype=np.float32)

    x_float = sub_x.float()
    target_feat = x_float[target_idx].unsqueeze(0)

    # Cosine similarity
    sim = F.cosine_similarity(x_float, target_feat, dim=-1).cpu().numpy()
    # Shift from [-1, 1] to [0, 1]
    sim = (sim + 1.0) / 2.0
    return sim.astype(np.float32)


def prune_ego_subgraph(
    target_idx: int,
    sub_edge_index: torch.Tensor,
    sub_x: Optional[torch.Tensor],
    num_sub_nodes: int,
    max_nodes: int = 128
) -> Tuple[torch.Tensor, Optional[torch.Tensor], int, int]:
    """
    If the ego-subgraph exceeds max_nodes, prune it while keeping target_idx
    and its most topologically significant neighbors (1-hop first, then highest degree/PPR).
    """
    if num_sub_nodes <= max_nodes:
        return sub_edge_index, sub_x, target_idx, num_sub_nodes

    # Find 1-hop neighbors of target
    row, col = sub_edge_index[0], sub_edge_index[1]
    one_hop = col[row == target_idx].unique().tolist()
    one_hop_set = set(one_hop)
    one_hop_set.discard(target_idx)

    # Degrees of all nodes
    degrees = torch.bincount(row, minlength=num_sub_nodes)

    selected = [target_idx]
    # Add 1-hop neighbors first (sorted by degree if too many)
    if len(one_hop_set) > (max_nodes - 1):
        sorted_1hop = sorted(list(one_hop_set), key=lambda n: degrees[n].item(), reverse=True)
        selected.extend(sorted_1hop[:max_nodes - 1])
    else:
        selected.extend(list(one_hop_set))
        # Fill remaining slots with 2-hop nodes having highest connection degree
        remaining_slots = max_nodes - len(selected)
        selected_set = set(selected)
        other_nodes = [n for n in range(num_sub_nodes) if n not in selected_set]
        sorted_others = sorted(other_nodes, key=lambda n: degrees[n].item(), reverse=True)
        selected.extend(sorted_others[:remaining_slots])

    selected_tensor = torch.tensor(selected, dtype=torch.long)
    new_target_idx = 0  # target_idx was placed first at index 0

    # Node mapping
    node_map = torch.full((num_sub_nodes,), -1, dtype=torch.long)
    node_map[selected_tensor] = torch.arange(len(selected))

    # Filter edges
    mask = (node_map[row] >= 0) & (node_map[col] >= 0)
    new_sub_edge_index = torch.stack([node_map[row[mask]], node_map[col[mask]]], dim=0)
    new_sub_x = sub_x[selected_tensor] if sub_x is not None else None

    return new_sub_edge_index, new_sub_x, new_target_idx, len(selected)


def compute_canonical_matrices(
    num_sub_nodes: int,
    sub_edge_index: torch.Tensor,
    sub_x: Optional[torch.Tensor],
    target_idx: int,
    resolution: int = 128,
    white_bg: bool = True
) -> np.ndarray:
    """
    Approach B: Canonical Matrix Reordering (Channels 4-7).
    Constructs deterministic, canonical relational matrices of the ego-subgraph:
      - Ch 4: Canonical Subgraph Adjacency (A_can)
      - Ch 5: Canonical 2-Hop Co-neighbor / Path Matrix (A^2_can)
      - Ch 6: Canonical Pairwise Feature Similarity (S_can)
      - Ch 7: Canonical Transition / Diffusion Matrix (P_can)
    Rescaled via nearest-neighbor block interpolation to (4, resolution, resolution).
    """
    if num_sub_nodes <= 0:
        val = 1.0 if white_bg else 0.0
        return np.full((4, resolution, resolution), val, dtype=np.float32)

    # 1. Deterministic canonical node ordering
    # Target node is always placed strictly at index 0
    deg = np.zeros(num_sub_nodes, dtype=np.float32)
    if sub_edge_index.numel() > 0:
        row, col = sub_edge_index[0].cpu().numpy(), sub_edge_index[1].cpu().numpy()
        np.add.at(deg, row, 1.0)

        # 1-hop neighbors of target
        one_hop = np.unique(col[row == target_idx])
        one_hop = one_hop[one_hop != target_idx]
        one_hop_set = set(one_hop.tolist())

        # 2-hop neighbors
        two_hop = np.array([n for n in range(num_sub_nodes) if n != target_idx and n not in one_hop_set], dtype=np.int64)
    else:
        one_hop = np.array([], dtype=np.int64)
        two_hop = np.array([n for n in range(num_sub_nodes) if n != target_idx], dtype=np.int64)

    # Sort 1-hop and 2-hop by descending degree
    if len(one_hop) > 0:
        one_hop_sorted = one_hop[np.argsort(-deg[one_hop])]
    else:
        one_hop_sorted = np.array([], dtype=np.int64)

    if len(two_hop) > 0:
        two_hop_sorted = two_hop[np.argsort(-deg[two_hop])]
    else:
        two_hop_sorted = np.array([], dtype=np.int64)

    canonical_order = np.concatenate([[target_idx], one_hop_sorted, two_hop_sorted])

    # 2. Canonical Subgraph Adjacency Matrix (A_can)
    A = np.zeros((num_sub_nodes, num_sub_nodes), dtype=np.float32)
    if sub_edge_index.numel() > 0:
        A[row, col] = 1.0
    np.fill_diagonal(A, 1.0)
    A_can = A[np.ix_(canonical_order, canonical_order)]

    # 3. Canonical 2-Hop Co-neighbor Matrix (A^2_can)
    A2_can = A_can @ A_can
    max_a2 = A2_can.max()
    if max_a2 > 1e-6:
        A2_can = A2_can / max_a2

    # 4. Canonical Pairwise Feature Similarity Matrix (S_can)
    if sub_x is not None and sub_x.numel() > 0:
        x_can = sub_x[canonical_order].float().cpu().numpy()
        norms = np.linalg.norm(x_can, axis=-1, keepdims=True) + 1e-6
        x_norm = x_can / norms
        S_can = np.clip((x_norm @ x_norm.T + 1.0) * 0.5, 0.0, 1.0)
    else:
        S_can = np.eye(num_sub_nodes, dtype=np.float32)

    # 5. Canonical Transition / Diffusion Matrix (P_can)
    row_sum = A_can.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    P_can = A_can / row_sum
    if P_can.max() > 1e-6:
        P_can = P_can / P_can.max()

    # 6. Stack and resize via nearest-neighbor to (4, resolution, resolution)
    mats = np.stack([A_can, A2_can, S_can, P_can], axis=0)  # (4, N, N)
    mats_t = torch.from_numpy(mats).unsqueeze(0).float()   # (1, 4, N, N)
    mats_resized = F.interpolate(mats_t, size=(resolution, resolution), mode='nearest').squeeze(0).numpy()

    # White-anchored background: 1.0 = white canvas, dark cells = connections
    if white_bg:
        mats_resized = 1.0 - mats_resized

    return mats_resized.astype(np.float32)


def compute_hillshade(Z: np.ndarray, altitude_deg: float = 45.0, azimuth_deg: float = 315.0) -> np.ndarray:
    """
    Compute 3D shaded relief (hillshade) from a 2D continuous elevation heightmap Z.
    Simulates directional lighting from azimuth_deg (default 315 = top-left) at altitude_deg.
    """
    H, W = Z.shape
    dz_dx = np.zeros((H, W), dtype=np.float32)
    dz_dy = np.zeros((H, W), dtype=np.float32)

    # Central finite differences
    dz_dx[:, 1:-1] = (Z[:, 2:] - Z[:, :-2]) * 0.5 * 10.0
    dz_dy[1:-1, :] = (Z[2:, :] - Z[:-2, :]) * 0.5 * 10.0

    # Slope and aspect angles
    slope = np.pi * 0.5 - np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    aspect = np.arctan2(dz_dy, -dz_dx)

    alt_rad = altitude_deg * np.pi / 180.0
    az_rad = azimuth_deg * np.pi / 180.0

    # Lambertian reflectance model
    shaded = np.sin(alt_rad) * np.sin(slope) + np.cos(alt_rad) * np.cos(slope) * np.cos(az_rad - aspect)
    shaded = np.clip(shaded, 0.0, 1.0)
    return shaded.astype(np.float32)


def compute_contour_isolines(Z: np.ndarray, num_levels: int = 10) -> np.ndarray:
    """
    Extract sharp topographic contour isoline rings from continuous elevation Z in [0, 1].
    Dense rings represent steep community boundaries; wide gaps represent flat homophilous clusters.
    """
    phase = num_levels * Z
    dist_to_contour = np.abs(phase - np.round(phase))
    contour = np.exp(-(dist_to_contour / 0.09) ** 2)

    # Major index contours (every 2nd level is bolder)
    major_phase = (num_levels / 2.0) * Z
    major_dist = np.abs(major_phase - np.round(major_phase))
    major_contour = np.exp(-(major_dist / 0.13) ** 2)

    isolines = np.maximum(contour * 0.65, major_contour * 1.0)
    return np.clip(isolines, 0.0, 1.0).astype(np.float32)


def node_to_ego_map(
    target_node: int,
    edge_index: torch.Tensor,
    x: Optional[torch.Tensor],
    num_hops: int = 2,
    resolution: int = 128,
    sigma_node: float = 0.06,
    sigma_edge: float = 0.02,
    max_nodes: int = 128,
    layout_method: str = "concentric",
    include_canonical_matrices: bool = False,
    white_bg: bool = True,
    seed: int = 42
) -> torch.Tensor:
    """
    Topographic Graph Cartography:
    Converts the ego-neighborhood into a 4-channel continuous Topographic Traffic Map:
      - Ch 0: Topological Elevation & 3D Hillshade (Gravitational potential + directional lighting relief)
      - Ch 1: Topographic Contour Isolines (Concentric elevation rings tracing slope steepness)
      - Ch 2: GNN Message-Passing Traffic Flux (Arterial highway network weighted by message volume)
      - Ch 3: Semantic Fault Lines (Spatial gradient magnitude of feature dissonance)
    If include_canonical_matrices=True, appends 4 canonical relational matrices (8 channels).
    """
    # 1. Extract k-hop ego subgraph around target node
    subset, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_idx=int(target_node),
        num_hops=num_hops,
        edge_index=edge_index,
        relabel_nodes=True
    )
    num_sub_nodes = subset.size(0)
    target_idx = mapping.item()
    sub_x = x[subset] if x is not None else None

    # Prune if ego-network is excessively dense/large to prevent CPU/memory bottleneck
    if num_sub_nodes > max_nodes:
        sub_edge_index, sub_x, target_idx, num_sub_nodes = prune_ego_subgraph(
            target_idx=target_idx,
            sub_edge_index=sub_edge_index,
            sub_x=sub_x,
            num_sub_nodes=num_sub_nodes,
            max_nodes=max_nodes
        )

    # 2. Compute 2D coordinate layout with target node pinned at (0, 0)
    coords = compute_ego_layout(num_sub_nodes, sub_edge_index, target_idx, method=layout_method, seed=seed)

    # Pixel coordinates for vector drawing
    px = (coords[:, 0] + 1.0) * 0.5 * (resolution - 1)
    py = (coords[:, 1] + 1.0) * 0.5 * (resolution - 1)

    # 3. Precompute 2D coordinate grid & Gaussian basis
    lin = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(lin, lin, indexing="ij")
    grid = np.stack([grid_x, grid_y], axis=-1)  # (H, W, 2)

    diff_nodes = grid[:, :, None, :] - coords[None, None, :, :]
    dist_sq_nodes = np.sum(diff_nodes ** 2, axis=-1)  # (H, W, N)
    gaussian_nodes = np.exp(-dist_sq_nodes / (2.0 * sigma_node ** 2))

    # Compute node degrees and PPR
    degrees = np.zeros(num_sub_nodes, dtype=np.float32)
    if sub_edge_index.numel() > 0:
        row, col = sub_edge_index[0].cpu().numpy(), sub_edge_index[1].cpu().numpy()
        np.add.at(degrees, row, 1.0)
    else:
        row, col = np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    ppr_weights = compute_personalized_diffusion(num_sub_nodes, sub_edge_index, target_idx)

    # ------------------------------------------------------------------------
    # CHANNEL 0: Topological Elevation & 3D Hillshade
    # ------------------------------------------------------------------------
    # Elevation: Degree * PPR * Inverse Hop distance
    elev_weights = np.sqrt(degrees + 1.0) * (ppr_weights + 0.2)
    elev_weights[target_idx] = elev_weights.max() * 1.5  # Central summit

    Z = np.sum(elev_weights[None, None, :] * gaussian_nodes, axis=-1)
    Z_norm = Z / (Z.max() + 1e-6)

    # Add sharp summit glyph at center (64, 64)
    summit_img = Image.new("L", (resolution, resolution), 0)
    summit_draw = ImageDraw.Draw(summit_img)
    cx, cy = float(px[target_idx]), float(py[target_idx])
    summit_draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=255)
    summit_draw.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], outline=255, width=1)
    summit_arr = np.array(summit_img, dtype=np.float32) / 255.0
    Z_norm = np.maximum(Z_norm, summit_arr * 0.8)

    # 3D Hillshade illumination
    hillshade = compute_hillshade(Z_norm, altitude_deg=45.0, azimuth_deg=315.0)
    ch0_elevation = 0.55 * Z_norm + 0.45 * hillshade

    # ------------------------------------------------------------------------
    # CHANNEL 1: Topographic Contour Isolines
    # ------------------------------------------------------------------------
    ch1_contours = compute_contour_isolines(Z_norm, num_levels=10)

    # ------------------------------------------------------------------------
    # CHANNEL 2: GNN Message-Passing Traffic Flux (Road Network)
    # ------------------------------------------------------------------------
    traffic_img = Image.new("L", (resolution, resolution), 0)
    traffic_draw = ImageDraw.Draw(traffic_img)

    num_sub_edges = sub_edge_index.shape[1]
    if num_sub_edges > 0 and sub_x is not None:
        edges_np = sub_edge_index.t().cpu().numpy()
        x_float = sub_x.float()

        # Compute pairwise feature compatibility for traffic attention
        src_x = x_float[edges_np[:, 0]]
        dst_x = x_float[edges_np[:, 1]]
        cos_sim = F.cosine_similarity(src_x, dst_x, dim=-1).cpu().numpy()
        attention = (cos_sim + 1.0) * 0.5  # [0, 1]

        # Gravity flux: sqrt(deg_u * deg_v) * attention * (ppr_u + ppr_v)
        deg_u = degrees[edges_np[:, 0]]
        deg_v = degrees[edges_np[:, 1]]
        ppr_sum = ppr_weights[edges_np[:, 0]] + ppr_weights[edges_np[:, 1]]
        flux = np.sqrt((deg_u + 1.0) * (deg_v + 1.0)) * (attention + 0.3) * (ppr_sum + 0.1)
        flux_norm = flux / (flux.max() + 1e-6)

        # Sort edges by flux so major highways draw on top
        sorted_edge_indices = np.argsort(flux_norm)
        for e_idx in sorted_edge_indices:
            u, v = edges_np[e_idx]
            fl = flux_norm[e_idx]

            # Arterial highways get width 3, medium width 2, local width 1
            if fl > 0.65:
                w_px = 3
                intensity = int(200 + 55 * fl)
            elif fl > 0.35:
                w_px = 2
                intensity = int(140 + 60 * fl)
            else:
                w_px = 1
                intensity = int(80 + 60 * fl)

            traffic_draw.line(
                [(float(px[u]), float(py[u])), (float(px[v]), float(py[v]))],
                fill=intensity,
                width=w_px
            )

        # Draw traffic junction circles at nodes
        for n_i in range(num_sub_nodes):
            r_junc = 4.5 if n_i == target_idx else 2.5
            junc_intensity = 255 if n_i == target_idx else 180
            traffic_draw.ellipse(
                [px[n_i] - r_junc, py[n_i] - r_junc, px[n_i] + r_junc, py[n_i] + r_junc],
                fill=junc_intensity
            )

        # Anti-aliasing
        traffic_blurred = traffic_img.filter(ImageFilter.GaussianBlur(radius=0.7))
        ch2_traffic = np.array(traffic_blurred, dtype=np.float32) / 255.0
    else:
        ch2_traffic = np.zeros((resolution, resolution), dtype=np.float32)

    # ------------------------------------------------------------------------
    # CHANNEL 3: Semantic Fault Lines (Feature Dissonance Gradient Magnitude)
    # ------------------------------------------------------------------------
    if sub_x is not None and sub_x.numel() > 0:
        x_float = sub_x.float()
        target_feat = x_float[target_idx].unsqueeze(0)
        cos_to_target = F.cosine_similarity(x_float, target_feat, dim=-1).cpu().numpy()
        # Semantic dissonance: 0 where features match target, 1 where they disagree
        dissonance = 1.0 - (cos_to_target + 1.0) * 0.5  # [0, 1]

        # Splat continuous dissonance field
        D = np.sum(dissonance[None, None, :] * gaussian_nodes, axis=-1)
        D_norm = D / (D.max() + 1e-6)

        # Compute spatial gradient magnitude ||grad D|| (Tectonic fault lines)
        gx = np.zeros((resolution, resolution), dtype=np.float32)
        gy = np.zeros((resolution, resolution), dtype=np.float32)
        gx[:, 1:-1] = (D_norm[:, 2:] - D_norm[:, :-2]) * 0.5 * 10.0
        gy[1:-1, :] = (D_norm[2:, :] - D_norm[:-2, :]) * 0.5 * 10.0
        fault_lines = np.sqrt(gx**2 + gy**2)
        fault_lines = fault_lines / (fault_lines.max() + 1e-6)
        ch3_fault = np.power(fault_lines, 0.5)  # Contrast stretch
    else:
        ch3_fault = np.zeros((resolution, resolution), dtype=np.float32)

    # ------------------------------------------------------------------------
    # White-Anchored Background Normalization
    # ------------------------------------------------------------------------
    raw_channels = [ch0_elevation, ch1_contours, ch2_traffic, ch3_fault]
    processed_channels = []
    for ch in raw_channels:
        ch_c = np.clip(ch, 0.0, 1.0)
        if white_bg:
            # 1.0 = Pure white canvas, contrasting dark features pop
            ch_c = 1.0 - ch_c
        processed_channels.append(ch_c.astype(np.float32))

    if include_canonical_matrices:
        can_mats = compute_canonical_matrices(
            num_sub_nodes=num_sub_nodes,
            sub_edge_index=sub_edge_index,
            sub_x=sub_x,
            target_idx=target_idx,
            resolution=resolution,
            white_bg=white_bg
        )
        all_channels = processed_channels + [can_mats[i] for i in range(4)]
    else:
        all_channels = processed_channels

    return torch.from_numpy(np.stack(all_channels, axis=0))


# ============================================================================
# Visualizer & Image Renderer
# ============================================================================

def colormap_white_anchored(val: np.ndarray, palette: str = "navy") -> np.ndarray:
    """
    Renders crisp, high-contrast features on a clean white-anchored background.
    val: in [0, 1] where 1.0 is white background, and < 1.0 represents edges/nodes/signals.
    """
    v = np.clip(1.0 - val, 0.0, 1.0)  # Signal intensity from 0 (empty white) to 1 (peak signal)

    if palette == "navy":
        # Pure White -> Vibrant Sky Blue -> Deep Royal Navy
        r = 1.0 - 0.92 * (v ** 0.75)
        g = 1.0 - 0.80 * (v ** 0.75) + 0.15 * (v ** 2)
        b = 1.0 - 0.35 * (v ** 0.75)
    elif palette == "crimson":
        # Pure White -> Coral Rose -> Deep Wine Crimson
        r = 1.0 - 0.25 * (v ** 0.75)
        g = 1.0 - 0.88 * (v ** 0.75)
        b = 1.0 - 0.80 * (v ** 0.75) + 0.20 * (v ** 2)
    elif palette == "emerald":
        # Pure White -> Mint -> Deep Forest Emerald
        r = 1.0 - 0.88 * (v ** 0.75)
        g = 1.0 - 0.30 * (v ** 0.75)
        b = 1.0 - 0.70 * (v ** 0.75)
    elif palette == "amber":
        # Pure White -> Gold -> Deep Bronze Amber
        r = 1.0 - 0.20 * (v ** 0.75)
        g = 1.0 - 0.60 * (v ** 0.75)
        b = 1.0 - 0.95 * (v ** 0.75)
    else:
        # High-contrast Grayscale on White
        r = g = b = 1.0 - v

    rgb = np.stack([np.clip(r, 0.0, 1.0), np.clip(g, 0.0, 1.0), np.clip(b, 0.0, 1.0)], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def save_ego_map_panel(
    tensor: torch.Tensor,
    filepath: str,
    target_node_id: int,
    label: Optional[int] = None,
    split_name: Optional[str] = None
):
    """
    Save multi-channel Ego-Map panel side-by-side as a publication-grade PNG with white-anchored background.
    Supports 4-channel continuous panels and 8-channel full hybrid (Continuous + Canonical Matrix) panels.
    """
    C, H, W = tensor.shape

    if C == 4:
        channel_titles = [
            "Ch 0: Topo Elevation & 3D Hillshade",
            "Ch 1: Topo Contour Isolines",
            "Ch 2: GNN Message Traffic Flux",
            "Ch 3: Semantic Fault Lines"
        ]
        palettes = ["amber", "navy", "crimson", "emerald"]
    else:
        channel_titles = [
            "Ch 0: Topo Elevation & 3D Hillshade",
            "Ch 1: Topo Contour Isolines",
            "Ch 2: GNN Message Traffic Flux",
            "Ch 3: Semantic Fault Lines",
            "Ch 4: Canonical Adjacency",
            "Ch 5: 2-Hop Co-Neighbors",
            "Ch 6: Feat Correlation Matrix",
            "Ch 7: Transition Diffusion"
        ]
        palettes = ["amber", "navy", "crimson", "emerald", "navy", "crimson", "emerald", "amber"]

    images = []
    for c in range(C):
        ch_arr = tensor[c].cpu().numpy()
        rgb = colormap_white_anchored(ch_arr, palette=palettes[c % len(palettes)])
        images.append(Image.fromarray(rgb))

    header_h = 38
    spacing = 8

    if C == 8:
        # 2 rows of 4 columns
        cols = 4
        rows = 2
        total_w = cols * W + (cols - 1) * spacing + 20
        total_h = header_h + rows * H + (rows - 1) * spacing + 48
    else:
        cols = C
        rows = 1
        total_w = cols * W + (cols - 1) * spacing + 20
        total_h = header_h + H + 30

    combined = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(combined)

    # Top banner title
    split_str = f" | Split: {split_name.upper()}" if split_name else ""
    meta_text = f"Target Node #{target_node_id} (Pinned at center 0,0){split_str} | Class: {label} | White-Anchored Multi-Channel Ego-Map"
    draw.text((10, 10), meta_text, fill=(18, 24, 38))

    for idx, img in enumerate(images):
        r_idx = idx // cols
        c_idx = idx % cols

        x_offset = 10 + c_idx * (W + spacing)
        y_offset = header_h + 18 + r_idx * (H + spacing + 18)

        # Title above channel
        draw.text((x_offset, y_offset - 14), channel_titles[idx], fill=(60, 75, 100))
        combined.paste(img, (x_offset, y_offset))

    combined.save(filepath)
    print(f"Saved publication-grade Ego-Map panel for Node #{target_node_id} (label={label}) to {filepath}")
