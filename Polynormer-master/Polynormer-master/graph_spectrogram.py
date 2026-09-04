"""
Graph Spectrogram Rasterizer: The Literal Audio-Mel Analogy for Graph Learning.
Pure PyTorch + PIL implementation (Zero Matplotlib dependency).

Module structure mirrors node_level_rasterizer.py to provide a unified interface
for graph representation caching and visualization.

Channels:
  - Ch 0: Low-Pass Diffusion Field (A_norm^k * X: multi-hop community consensus and homophily)
  - Ch 1: High-Pass Boundary Field (Delta A_norm^k * X: boundary gradient and local heterophily)
  - Ch 2: Structural Resonance Field (Personalized PageRank / RWR return basins and trapping)
"""

import os
import time
import argparse
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops


# =============================================================================
# 1. COLORMAP & VISUALIZATION UTILITIES
# =============================================================================

def apply_spectrogram_colormap(matrix: np.ndarray, colormap: str = "magma") -> np.ndarray:
    """
    Map a 2D float matrix in [0, 1] to an (H, W, 3) RGB uint8 image using authentic spectrogram colormaps.
    """
    v = np.clip(matrix, 0.0, 1.0)
    if colormap == "magma":
        r = np.clip(1.4 * (v ** 0.65) - 0.3 * (v ** 3), 0.0, 1.0)
        g = np.clip(1.25 * (v ** 1.8) + 0.1 * (v ** 4), 0.0, 1.0)
        b = np.clip(0.4 * (v ** 0.5) + 0.6 * (v ** 2.5), 0.0, 1.0)
    elif colormap == "inferno":
        r = np.clip(1.5 * (v ** 0.8) - 0.2 * (v ** 3), 0.0, 1.0)
        g = np.clip(1.1 * (v ** 1.6), 0.0, 1.0)
        b = np.clip(0.3 * np.sin(np.pi * v) + 0.9 * (v ** 3), 0.0, 1.0)
    elif colormap == "viridis":
        r = np.clip(0.1 + 0.9 * (v ** 1.5), 0.0, 1.0)
        g = np.clip(0.2 + 0.8 * (v ** 0.8), 0.0, 1.0)
        b = np.clip(0.5 * (1.0 - v) + 0.2 * (v ** 2), 0.0, 1.0)
    else:
        r = g = b = v

    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def save_spectrogram_panel(
    spec_tensor: torch.Tensor,
    output_path: str,
    target_node_id: int = 0,
    label: int = None,
    split_name: str = "sample",
    dataset_name: str = "graph"
):
    """
    Renders and saves a publication-grade 3-panel comparison image for a single node's spectrogram.
    spec_tensor: 3D Tensor of shape (C, num_hops, num_bands) in [0.0, 1.0].
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    spec_np = spec_tensor.cpu().float().numpy()
    num_ch = spec_np.shape[0]

    c0 = spec_np[0]
    c1 = spec_np[1] if num_ch > 1 else np.zeros_like(c0)
    c2 = spec_np[2] if num_ch > 2 else np.zeros_like(c0)

    rgb0 = apply_spectrogram_colormap(c0, "magma")
    rgb1 = apply_spectrogram_colormap(c1, "inferno")
    rgb2 = apply_spectrogram_colormap(c2, "viridis")

    target_h, target_w = 256, 384
    im0 = Image.fromarray(rgb0).resize((target_w, target_h), Image.NEAREST)
    im1 = Image.fromarray(rgb1).resize((target_w, target_h), Image.NEAREST)
    im2 = Image.fromarray(rgb2).resize((target_w, target_h), Image.NEAREST)

    panel_w = 3 * target_w + 80
    panel_h = target_h + 100
    canvas = Image.new("RGB", (panel_w, panel_h), (13, 17, 23))
    draw = ImageDraw.Draw(canvas)

    header = f"GRAPH SPECTROGRAM: NODE #{target_node_id} (CLASS {label}) ON [{dataset_name.upper()}] - {split_name.upper()}"
    draw.text((20, 15), header, fill=(255, 255, 255))
    draw.text((20, 35), "Time-Frequency Channels: [Ch0: Consensus (Magma) | Ch1: Boundary Difference (Inferno) | Ch2: Resonance (Viridis)]", fill=(160, 175, 195))

    canvas.paste(im0, (20, 60))
    canvas.paste(im1, (target_w + 40, 60))
    canvas.paste(im2, (2 * target_w + 60, 60))

    draw.text((20, 65 + target_h), "Ch 0: Low-Pass Diffusion (A^k X)", fill=(255, 165, 0))
    draw.text((target_w + 40, 65 + target_h), "Ch 1: High-Pass Boundary Gradient", fill=(255, 200, 50))
    draw.text((2 * target_w + 60, 65 + target_h), "Ch 2: Structural PageRank Resonance", fill=(80, 220, 140))

    canvas.save(output_path, "PNG")
    print(f"-> Saved crisp spectrogram panel to: {output_path}")


# =============================================================================
# 2. SPECTROGRAM COMPUTATION ENGINES
# =============================================================================

def compute_graph_spectrograms(
    edge_index: torch.Tensor,
    x: torch.Tensor,
    num_nodes: int,
    num_hops: int = 8,
    num_bands: int = 128,
    ppr_alpha: float = 0.20,
    device: torch.device = torch.device('cpu')
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Computes 3-channel Graph Spectrograms for ALL nodes in the graph simultaneously
    using fast sparse matrix operations on device.

    Returns:
        spectrograms: Tensor of shape (num_nodes, 3, num_hops, num_bands) in [0.0, 1.0]
        band_indices: Array of selected feature indices
    """
    t0 = time.time()
    x = x.float()
    num_raw_feats = x.shape[1]

    # 1. Feature Band Selection (The "Mel Filterbank" of Graphs)
    if num_bands > 0 and num_bands < num_raw_feats:
        feat_var = torch.var(x, dim=0).cpu().numpy()
        band_indices = np.argsort(-feat_var)[:num_bands]
        x_sub = x[:, band_indices].to(device)
    else:
        num_bands = num_raw_feats
        band_indices = np.arange(num_raw_feats)
        x_sub = x.to(device)

    # 2. Symmetric Normalized Adjacency Matrix: D^{-1/2} (A + I) D^{-1/2}
    edge_index_loop, _ = add_self_loops(edge_index.to(device), num_nodes=num_nodes)
    row, col = edge_index_loop[0], edge_index_loop[1]
    deg = torch.bincount(row, minlength=num_nodes).float()
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    val = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    adj_norm = torch.sparse_coo_tensor(edge_index_loop, val, (num_nodes, num_nodes)).to(device)

    # 3. Vectorized Multi-Horizon Propagation across all channels
    s0_hops = []
    s1_hops = []
    s2_hops = []

    x_k = x_sub
    p_k = x_sub

    # Hop 0
    s0_hops.append(x_sub)
    s1_hops.append(torch.zeros_like(x_sub))
    s2_hops.append(x_sub)

    for k in range(1, num_hops):
        # Low-pass step: A^k X
        x_next = torch.sparse.mm(adj_norm, x_k)
        s0_hops.append(x_next)

        # High-pass boundary gradient: A^{k-1} X - A^k X
        s1_hops.append(x_k - x_next)
        x_k = x_next

        # Structural resonance: Personalized PageRank
        p_next = ppr_alpha * x_sub + (1.0 - ppr_alpha) * torch.sparse.mm(adj_norm, p_k)
        s2_hops.append(p_next)
        p_k = p_next

    # Stack along hops: (num_hops, num_nodes, num_bands) -> (num_nodes, num_hops, num_bands)
    c0 = torch.stack(s0_hops, dim=0).permute(1, 0, 2)
    c1 = torch.stack(s1_hops, dim=0).permute(1, 0, 2)
    c2 = torch.stack(s2_hops, dim=0).permute(1, 0, 2)

    # Normalize each channel into dynamic range [0, 1]
    def min_max_norm(tensor):
        t_min = tensor.amin(dim=(1, 2), keepdim=True)
        t_max = tensor.amax(dim=(1, 2), keepdim=True)
        return (tensor - t_min) / (t_max - t_min + 1e-6)

    c0 = min_max_norm(c0)
    c1 = min_max_norm(c1)
    c2 = min_max_norm(c2)

    spectrograms = torch.stack([c0, c1, c2], dim=1).cpu()

    elapsed = time.time() - t0
    size_mb = (spectrograms.element_size() * spectrograms.nelement()) / (1024 * 1024)
    print(f"-> Generated {num_nodes} Graph Spectrograms ({num_hops} hops x {num_bands} bands, 3 channels) in {elapsed:.3f}s ({size_mb:.1f} MB)")

    return spectrograms, band_indices


def node_to_spectrogram(
    target_node: int,
    all_spectrograms: torch.Tensor
) -> torch.Tensor:
    """
    Extracts the (C, num_hops, num_bands) spectrogram tensor for a single target node.
    """
    return all_spectrograms[target_node]


# =============================================================================
# 3. CACHING & PIPELINE INTEGRATION
# =============================================================================

def get_or_create_spectrogram_cache(dataset, edge_index_cpu, x_cpu, args) -> torch.Tensor:
    """
    Check for pre-computed Graph Spectrogram cache on disk.
    If missing, computes all node spectrograms in parallel via sparse matrix operations and caches to disk.
    Returns tensor of shape (N, channels, num_hops, num_bands) in uint8 (0-255).
    """
    n = dataset.graph['num_nodes']
    cache_dir = os.path.join(args.data_dir, "spectrogram_cache")
    os.makedirs(cache_dir, exist_ok=True)

    num_hops = getattr(args, 'num_hops', 8)
    num_bands_arg = getattr(args, 'num_bands', 128)
    channels = getattr(args, 'channels', 3)
    raw_feats = x_cpu.shape[1] if x_cpu is not None else 128
    effective_bands = raw_feats if (num_bands_arg <= 0 or num_bands_arg > raw_feats) else num_bands_arg

    cache_file = os.path.join(
        cache_dir,
        f"{args.dataset}_spectrogram_hops{num_hops}_bands{effective_bands}_ch{channels}.pt"
    )

    if os.path.exists(cache_file):
        print(f"Loading pre-computed Graph Spectrogram cache from: {cache_file} ...")
        t0 = time.time()
        cached_specs = torch.load(cache_file)
        print(f"Loaded {cached_specs.shape[0]} Spectrograms in {time.time() - t0:.2f}s | Tensor: {list(cached_specs.shape)} ({cached_specs.dtype})")
        return cached_specs

    print(f"\n{'='*75}")
    print(f"Computing {n} Graph Spectrograms ({channels}ch, {num_hops} hops x {effective_bands} bands) for '{args.dataset}'")
    print(f"Cache will be saved to: {cache_file}")
    print(f"{'='*75}")

    device = torch.device(f"cuda:{args.device}" if (torch.cuda.is_available() and not getattr(args, 'cpu', False)) else "cpu")
    specs_float, _ = compute_graph_spectrograms(
        edge_index=edge_index_cpu,
        x=x_cpu,
        num_nodes=n,
        num_hops=num_hops,
        num_bands=effective_bands,
        device=device
    )

    if channels == 1:
        specs_float = specs_float[:, :1, :, :]

    # Convert to uint8 (0-255) for compact storage
    cached_specs = (specs_float * 255.0).clamp(0, 255).to(torch.uint8)

    torch.save(cached_specs, cache_file)
    size_mb = (cached_specs.element_size() * cached_specs.nelement()) / (1024 * 1024)
    print(f"Spectrogram caching complete! Saved {size_mb:.1f} MB to: {cache_file}\n")

    # Save a publication-grade sample visualization panel of Node 0
    try:
        results_dir = "./results"
        os.makedirs(results_dir, exist_ok=True)
        sample_path = os.path.join(results_dir, f"{args.dataset}_spectrogram_sample.png")
        sample_tensor = (cached_specs[0].float()) / 255.0
        label_val = int(dataset.label[0].item()) if hasattr(dataset, 'label') else None
        save_spectrogram_panel(
            spec_tensor=sample_tensor,
            output_path=sample_path,
            target_node_id=0,
            label=label_val,
            split_name="sample",
            dataset_name=args.dataset
        )
    except Exception as e:
        print(f"Note: Could not save sample panel: {e}")

    return cached_specs


# =============================================================================
# 4. STANDALONE VERIFICATION / TEST RUNNER
# =============================================================================

if __name__ == '__main__':
    from dataset import load_dataset

    print("Running Graph Spectrogram Rasterizer verification...")
    data_dir = "./data/"
    dataset_name = "amazon-photo"
    dataset = load_dataset(data_dir, dataset_name)
    edge_index = to_undirected(dataset.graph['edge_index'])
    edge_index, _ = remove_self_loops(edge_index)
    x = dataset.graph['node_feat']
    num_nodes = dataset.graph['num_nodes']

    specs, _ = compute_graph_spectrograms(
        edge_index=edge_index,
        x=x,
        num_nodes=num_nodes,
        num_hops=8,
        num_bands=128
    )
    print(f"Computed Spectrograms Shape: {specs.shape}")
    sample_out = "./results/verification_spectrogram_node0.png"
    save_spectrogram_panel(specs[0], sample_out, target_node_id=0, label=int(dataset.label[0].item()), dataset_name=dataset_name)
    print("Verification complete!")
