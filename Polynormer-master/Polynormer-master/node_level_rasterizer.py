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
from PIL import Image
from typing import Optional, Tuple
from torch_geometric.utils import k_hop_subgraph


def compute_ego_layout(
    num_sub_nodes: int,
    sub_edge_index: torch.Tensor,
    target_idx: int,
    method: str = "pinned_spring",
    seed: int = 42
) -> np.ndarray:
    """
    Compute 2D coordinates for the ego-network with the target node strictly pinned at (0, 0).
    Coordinates are normalized into [-0.75, 0.75]^2.
    """
    if num_sub_nodes <= 1:
        return np.zeros((num_sub_nodes, 2), dtype=np.float32)

    G = nx.Graph()
    G.add_nodes_from(range(num_sub_nodes))
    edges = sub_edge_index.t().cpu().numpy()
    for u, v in edges:
        if u != v:
            G.add_edge(int(u), int(v))

    pos_dict = None
    if method == "pinned_spring":
        try:
            # Target node is fixed at (0, 0)
            initial_pos = {target_idx: np.array([0.0, 0.0], dtype=np.float32)}
            for i in range(num_sub_nodes):
                if i != target_idx:
                    # Initialize in a small ring around origin
                    angle = (2.0 * np.pi * i) / max(num_sub_nodes - 1, 1)
                    initial_pos[i] = np.array([0.3 * np.cos(angle), 0.3 * np.sin(angle)], dtype=np.float32)

            pos_dict = nx.spring_layout(
                G,
                pos=initial_pos,
                fixed=[target_idx],
                iterations=35,
                seed=seed,
                scale=0.8
            )
        except Exception:
            pos_dict = None

    if pos_dict is None:
        # Fallback: concentric rings based on shortest path distance
        lengths = nx.single_source_shortest_path_length(G, target_idx)
        coords = np.zeros((num_sub_nodes, 2), dtype=np.float32)
        by_hop = {}
        for node, hop in lengths.items():
            by_hop.setdefault(hop, []).append(node)

        for hop, nodes in by_hop.items():
            if hop == 0:
                coords[target_idx] = [0.0, 0.0]
            else:
                radius = min(0.35 * hop, 0.75)
                n_count = len(nodes)
                for idx_in_hop, node in enumerate(nodes):
                    angle = (2.0 * np.pi * idx_in_hop) / n_count
                    coords[node] = [radius * np.cos(angle), radius * np.sin(angle)]
        return coords

    coords = np.array([pos_dict[i] for i in range(num_sub_nodes)], dtype=np.float32)
    # Ensure target node is strictly at (0, 0)
    target_pos = coords[target_idx].copy()
    coords -= target_pos

    # Scale so outer neighbors fit within [-0.75, 0.75]
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


def node_to_ego_map(
    target_node: int,
    edge_index: torch.Tensor,
    x: Optional[torch.Tensor],
    num_hops: int = 2,
    resolution: int = 128,
    sigma_node: float = 0.08,
    sigma_edge: float = 0.035,
    max_nodes: int = 128,
    layout_method: str = "concentric",
    seed: int = 42
) -> torch.Tensor:
    """
    Rasterize the k-hop ego-neighborhood of target_node into a 4-channel continuous 2D tensor (4, H, W).
    Target node is pinned at canvas center (0, 0).
    Bounded to max_nodes to ensure constant fast runtime and low memory footprint.
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

    # 3. Create 2D coordinate grid (H, W, 2)
    lin = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(lin, lin, indexing="ij")
    grid = np.stack([grid_x, grid_y], axis=-1)  # (H, W, 2)

    # --- Channel 0: Ego-Density & Distance Field ---
    diff_nodes = grid[:, :, None, :] - coords[None, None, :, :]
    dist_sq_nodes = np.sum(diff_nodes ** 2, axis=-1)  # (H, W, N)

    # Compute hop distances from target
    G_sub = nx.Graph()
    G_sub.add_nodes_from(range(num_sub_nodes))
    edges_np = sub_edge_index.t().cpu().numpy()
    for u, v in edges_np:
        G_sub.add_edge(int(u), int(v))

    lengths = nx.single_source_shortest_path_length(G_sub, target_idx)
    weights = np.array([1.0 / max(lengths.get(i, num_hops), 0.5) for i in range(num_sub_nodes)], dtype=np.float32)
    weights[target_idx] = 2.5  # Distinct center anchor glow

    ego_density = np.sum(weights[None, None, :] * np.exp(-dist_sq_nodes / (2.0 * sigma_node ** 2)), axis=-1)

    # --- Channel 1: Ego-Edge Interconnections Field (Chunked for memory safety) ---
    num_sub_edges = sub_edge_index.shape[1]
    ego_edges = np.zeros((resolution, resolution), dtype=np.float32)

    if num_sub_edges > 0:
        p0_all = coords[edges_np[:, 0]]
        p1_all = coords[edges_np[:, 1]]
        v_all = p1_all - p0_all
        len_sq_all = np.maximum(np.sum(v_all ** 2, axis=-1), 1e-8)

        # Chunk edges to keep peak RAM under a few megabytes
        chunk_size = 64
        for chunk_start in range(0, num_sub_edges, chunk_size):
            p0 = p0_all[chunk_start:chunk_start + chunk_size]
            v = v_all[chunk_start:chunk_start + chunk_size]
            len_sq = len_sq_all[chunk_start:chunk_start + chunk_size]

            p_minus_p0 = grid[:, :, None, :] - p0[None, None, :, :]
            dot = np.sum(p_minus_p0 * v[None, None, :, :], axis=-1)
            t = np.clip(dot / len_sq[None, None, :], 0.0, 1.0)
            closest = p0[None, None, :, :] + t[:, :, :, None] * v[None, None, :, :]
            dist_sq_edge = np.sum((grid[:, :, None, :] - closest) ** 2, axis=-1)

            ego_edges += np.sum(np.exp(-dist_sq_edge / (2.0 * sigma_edge ** 2)), axis=-1)

    # --- Channel 2: Personalized Diffusion Field (PPR from target node) ---
    ppr_weights = compute_personalized_diffusion(num_sub_nodes, sub_edge_index, target_idx)
    ego_diffusion = np.sum(ppr_weights[None, None, :] * np.exp(-dist_sq_nodes / (2.0 * sigma_node ** 2)), axis=-1)

    # --- Channel 3: Relative Feature Homophily Field ---
    homophily_weights = compute_relative_homophily(sub_x, target_idx)
    ego_homophily = np.sum(homophily_weights[None, None, :] * np.exp(-dist_sq_nodes / (2.0 * sigma_node ** 2)), axis=-1)

    # Normalize each channel into [0, 1]
    channels = [ego_density, ego_edges, ego_diffusion, ego_homophily]
    norm_channels = []
    for ch in channels:
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max - ch_min > 1e-6:
            norm_channels.append(((ch - ch_min) / (ch_max - ch_min)).astype(np.float32))
        else:
            norm_channels.append(ch.astype(np.float32))

    # Return shape: (4, H, W)
    return torch.from_numpy(np.stack(norm_channels, axis=0))


# ============================================================================
# Visualizer & Image Renderer
# ============================================================================

def colormap_inferno(val: np.ndarray) -> np.ndarray:
    x = np.clip(val, 0.0, 1.0)
    r = np.clip(2.5 * x**2 - 1.2 * x**3 + 0.1 * x, 0.0, 1.0)
    g = np.clip(1.8 * x**3 - 0.2 * x**2 + 0.05 * x, 0.0, 1.0)
    b = np.clip(1.5 * x - 2.0 * x**2 + 1.2 * x**3, 0.0, 1.0)
    highlight = np.clip((x - 0.7) * 3.33, 0.0, 1.0)
    r = np.clip(r + 0.3 * highlight, 0.0, 1.0)
    g = np.clip(g + 0.7 * highlight, 0.0, 1.0)
    b = np.clip(b + 0.5 * highlight, 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def colormap_plasma(val: np.ndarray) -> np.ndarray:
    x = np.clip(val, 0.0, 1.0)
    r = np.clip(0.05 + 1.5 * x - 0.6 * x**2, 0.0, 1.0)
    g = np.clip(0.02 + 0.1 * x + 0.9 * x**3, 0.0, 1.0)
    b = np.clip(0.5 + 0.8 * x - 1.5 * x**2 + 0.3 * x**3, 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def save_ego_map_panel(
    tensor: torch.Tensor,
    filepath: str,
    target_node_id: int,
    label: Optional[int] = None,
    split_name: Optional[str] = None
):
    """
    Save 4-channel Ego-Map panel side-by-side as a publication-ready PNG with clear annotations.
    """
    from PIL import ImageDraw

    C, H, W = tensor.shape
    colormaps = [colormap_inferno, colormap_plasma, colormap_inferno, colormap_plasma]
    channel_titles = [
        "Ch 0: Ego Density",
        "Ch 1: Subgraph Edges",
        "Ch 2: PPR Diffusion",
        "Ch 3: Feat Homophily"
    ]

    images = []
    for c in range(C):
        ch_arr = tensor[c].cpu().numpy()
        rgb = colormaps[c % len(colormaps)](ch_arr)
        images.append(Image.fromarray(rgb))

    header_h = 36
    spacing = 8
    total_w = C * W + (C - 1) * spacing + 16
    total_h = H + header_h + 16

    combined = Image.new("RGB", (total_w, total_h), color=(12, 14, 22))
    draw = ImageDraw.Draw(combined)

    # Top banner title
    split_str = f" | Split: {split_name.upper()}" if split_name else ""
    meta_text = f"Target Node #{target_node_id} (Pinned at center 0,0){split_str} | Class: {label}"
    draw.text((8, 6), meta_text, fill=(240, 240, 250))

    # Paste channel images and labels
    for i, img in enumerate(images):
        x_offset = 8 + i * (W + spacing)
        y_offset = header_h + 8
        combined.paste(img, (x_offset, y_offset))

        # Channel label above image
        draw.text((x_offset + 2, header_h - 14), channel_titles[i], fill=(180, 195, 220))

    combined.save(filepath)
    print(f"Saved Ego-Map panel for Node #{target_node_id} (label={label}) to {filepath}")
