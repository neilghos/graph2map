"""
Graph Spectrogram Generator: The Literal Audio-Mel Analogy for Graphs.
Pure PyTorch + PIL implementation (Zero Matplotlib dependency).

Generates 3 literal Graph Spectrograms:
  1. Multi-Hop Diffusion Spectrogram (Hop depth k = 0 to 7 vs Feature Bands)
  2. Continuous Heat Kernel Spectrogram (Diffusion Time tau in [0, 3.0] vs Feature Bands)
  3. Local Graph Fourier Transform (Laplacian Eigenmodes lambda vs Feature Bands)
"""

import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from torch_geometric.utils import k_hop_subgraph, to_undirected, remove_self_loops, add_self_loops
from dataset import load_dataset


def apply_spectrogram_colormap(matrix: np.ndarray, colormap: str = "magma") -> np.ndarray:
    """
    Map a 2D float matrix in [0, 1] to an (H, W, 3) RGB uint8 image using authentic spectrogram colormaps.
    """
    v = np.clip(matrix, 0.0, 1.0)

    if colormap == "magma":
        # Deep Black/Purple -> Vibrant Crimson/Orange -> Bright Yellow/White
        r = np.clip(1.4 * (v ** 0.65) - 0.3 * (v ** 3), 0.0, 1.0)
        g = np.clip(1.25 * (v ** 1.8) + 0.1 * (v ** 4), 0.0, 1.0)
        b = np.clip(0.4 * (v ** 0.5) + 0.6 * (v ** 2.5), 0.0, 1.0)
    elif colormap == "inferno":
        # Deep Black -> Glowing Violet -> Warm Gold -> Bright White
        r = np.clip(1.5 * (v ** 0.8) - 0.2 * (v ** 3), 0.0, 1.0)
        g = np.clip(1.1 * (v ** 1.6), 0.0, 1.0)
        b = np.clip(0.3 * np.sin(np.pi * v) + 0.9 * (v ** 3), 0.0, 1.0)
    elif colormap == "viridis":
        # Dark Teal -> Emerald Green -> Vibrant Yellow
        r = np.clip(0.1 + 0.9 * (v ** 1.5), 0.0, 1.0)
        g = np.clip(0.2 + 0.8 * (v ** 0.8), 0.0, 1.0)
        b = np.clip(0.5 * (1.0 - v) + 0.2 * (v ** 2), 0.0, 1.0)
    else:
        r = g = b = v

    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def generate_graph_spectrogram(
    node_id: int = 0,
    dataset_name: str = "amazon-photo",
    data_dir: str = "./data/",
    num_hops: int = 7,
    num_feat_bins: int = 128,
    output_path: str = "./results/graph_spectrogram_node0.png"
):
    print(f"\n{'='*75}")
    print(f"Generating Literal Graph Spectrogram for Node #{node_id} on '{dataset_name}' (Pure PIL)")
    print(f"{'='*75}")

    dataset = load_dataset(data_dir, dataset_name)
    edge_index = to_undirected(dataset.graph['edge_index'])
    edge_index, _ = remove_self_loops(edge_index)
    x = dataset.graph['node_feat']
    num_nodes = dataset.graph['num_nodes']
    num_features = x.shape[1]

    label = int(dataset.label[node_id].item()) if hasattr(dataset, 'label') else None
    print(f"Target Node #{node_id} | Total Nodes: {num_nodes} | Total Feats: {num_features} | Class: {label}")

    # ------------------------------------------------------------------------
    # 1. Multi-Hop Polynomial Diffusion Spectrogram
    # Y-axis: Hop depth k = 0, 1, 2, ..., num_hops
    # X-axis: Top num_feat_bins feature channels
    # ------------------------------------------------------------------------
    print("Computing Multi-Hop Adjacency Diffusion Spectrogram ...")
    edge_index_loop, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index_loop[0], edge_index_loop[1]
    deg = torch.bincount(row, minlength=num_nodes).float()
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    val = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    adj_norm = torch.sparse_coo_tensor(edge_index_loop, val, (num_nodes, num_nodes))

    # Pick top informative feature channels based on overall variance
    feat_var = torch.var(x, dim=0).cpu().numpy()
    top_feat_indices = np.argsort(-feat_var)[:num_feat_bins]

    hop_frames = []
    x_curr = x.clone()
    for k in range(num_hops + 1):
        node_feat_k = x_curr[node_id, top_feat_indices].cpu().numpy()
        hop_frames.append(node_feat_k)
        x_curr = torch.sparse.mm(adj_norm, x_curr)

    hop_matrix = np.array(hop_frames)  # Shape: (num_hops + 1, num_feat_bins)
    hop_norm = (hop_matrix - hop_matrix.min()) / (hop_matrix.max() - hop_matrix.min() + 1e-6)

    # ------------------------------------------------------------------------
    # 2. Continuous Heat Kernel Spectrogram: exp(-tau * L) * X
    # Continuous time tau in [0, 3.0] across 32 time frames
    # ------------------------------------------------------------------------
    print("Computing Continuous Heat Kernel Spectrogram ...")
    sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
        node_idx=node_id,
        num_hops=3,
        edge_index=edge_index,
        relabel_nodes=True
    )
    sub_n = sub_nodes.size(0)
    target_sub_idx = mapping.item()
    sub_x = x[sub_nodes][:, top_feat_indices].cpu().numpy()

    A_sub = np.zeros((sub_n, sub_n), dtype=np.float32)
    if sub_edge_index.numel() > 0:
        s_row, s_col = sub_edge_index[0].cpu().numpy(), sub_edge_index[1].cpu().numpy()
        A_sub[s_row, s_col] = 1.0
        A_sub[s_col, s_row] = 1.0

    deg_sub = A_sub.sum(axis=1)
    d_inv_sqrt = np.power(deg_sub, -0.5, where=deg_sub > 0)
    d_inv_sqrt[deg_sub == 0] = 0.0
    L_norm = np.eye(sub_n) - (d_inv_sqrt[:, None] * A_sub * d_inv_sqrt[None, :])

    eigenvals, eigenvecs = np.linalg.eigh(L_norm)

    tau_steps = 32
    taus = np.linspace(0.0, 3.0, tau_steps)
    heat_frames = []
    x_fourier = eigenvecs.T @ sub_x

    for tau in taus:
        h_tau = np.exp(-tau * eigenvals)[:, None] * x_fourier
        x_diffused = eigenvecs @ h_tau
        heat_frames.append(x_diffused[target_sub_idx])

    heat_matrix = np.array(heat_frames)  # Shape: (tau_steps, num_feat_bins)
    heat_norm = (heat_matrix - heat_matrix.min()) / (heat_matrix.max() - heat_matrix.min() + 1e-6)

    # ------------------------------------------------------------------------
    # 3. Local Graph Fourier Transform (GFT) Spectrogram
    # Y-axis: Graph Frequencies lambda_j (Laplacian Eigenmodes)
    # X-axis: Feature Channels
    # ------------------------------------------------------------------------
    print("Computing Local Graph Fourier Transform (GFT) ...")
    gft_matrix = np.abs(x_fourier)
    gft_norm = gft_matrix / (gft_matrix.max() + 1e-6)

    # ------------------------------------------------------------------------
    # 4. Render Crisp Publication Spectrogram Panel using PIL
    # ------------------------------------------------------------------------
    print(f"Rendering publication-grade spectrograms to: {output_path} ...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Target display dimensions per spectrogram plot
    plot_w = 420
    plot_h = 320

    def matrix_to_pil(mat: np.ndarray, cmap_name: str) -> Image.Image:
        rgb = apply_spectrogram_colormap(mat, colormap=cmap_name)
        img = Image.fromarray(rgb)
        # Flip vertically so origin is at bottom (standard spectrogram convention)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        # Resize cleanly with nearest neighbor to keep pixels crisp
        return img.resize((plot_w, plot_h), resample=Image.NEAREST)

    img_hop = matrix_to_pil(hop_norm, "magma")
    img_heat = matrix_to_pil(heat_norm, "inferno")
    img_gft = matrix_to_pil(gft_norm, "viridis")

    # Composite canvas setup
    margin_left = 60
    margin_right = 30
    margin_top = 80
    margin_bottom = 60
    spacing = 50

    total_w = margin_left + 3 * plot_w + 2 * spacing + margin_right
    total_h = margin_top + plot_h + margin_bottom

    canvas = Image.new("RGB", (total_w, total_h), color=(20, 22, 28))
    draw = ImageDraw.Draw(canvas)

    # Main Header
    title_text = f"THE GRAPH SPECTROGRAM: NODE #{node_id} (CLASS {label}) ON {dataset_name.upper()}"
    subtitle_text = "The Literal Audio-Mel Spectrogram for Graphs | Multi-Hop Diffusion, Heat Kernel & Graph Fourier Harmonics"
    draw.text((margin_left, 16), title_text, fill=(255, 255, 255))
    draw.text((margin_left, 38), subtitle_text, fill=(160, 175, 200))

    plots_data = [
        (img_hop, "A. Multi-Hop Diffusion Spectrogram", "Hop Horizon (k=0 to 7)", "Semantic Feature Bands", (255, 140, 80)),
        (img_heat, "B. Continuous Heat Diffusion Spectrogram", r"Diffusion Time \tau (0.0 to 3.0)", "Semantic Feature Bands", (255, 200, 80)),
        (img_gft, "C. Graph Fourier Spectral Harmonics", r"Graph Frequency \lambda (Low to High)", "Semantic Feature Bands", (80, 220, 180)),
    ]

    for idx, (img, plot_title, y_label, x_label, title_color) in enumerate(plots_data):
        x_pos = margin_left + idx * (plot_w + spacing)
        y_pos = margin_top

        # Paste spectrogram
        canvas.paste(img, (x_pos, y_pos))

        # Border around spectrogram
        draw.rectangle([x_pos - 1, y_pos - 1, x_pos + plot_w, y_pos + plot_h], outline=(80, 90, 110), width=1)

        # Title
        draw.text((x_pos, y_pos - 26), plot_title, fill=title_color)

        # X-axis label
        draw.text((x_pos + plot_w // 2 - 60, y_pos + plot_h + 12), x_label, fill=(170, 180, 200))

        # Y-axis annotations
        if idx == 0:
            for k in range(num_hops + 1):
                y_coord = y_pos + plot_h - int((k + 0.5) * (plot_h / (num_hops + 1))) - 4
                draw.text((x_pos - 48, y_coord), f"Hop {k}", fill=(180, 190, 210))
        elif idx == 1:
            for t_val in [0.0, 1.0, 2.0, 3.0]:
                frac = t_val / 3.0
                y_coord = y_pos + plot_h - int(frac * plot_h) - 6
                draw.text((x_pos - 40, y_coord), f"{t_val:.1f}s", fill=(180, 190, 210))
        elif idx == 2:
            draw.text((x_pos - 52, y_pos + plot_h - 14), r"\lambda_0", fill=(180, 190, 210))
            draw.text((x_pos - 52, y_pos + 6), r"\lambda_max", fill=(180, 190, 210))

    canvas.save(output_path)
    print(f"Successfully generated crisp Graph Spectrogram to: {output_path}!\n")


if __name__ == "__main__":
    generate_graph_spectrogram(
        node_id=0,
        dataset_name="amazon-photo",
        output_path="./results/graph_spectrogram_node0.png"
    )
