"""
Standalone PIL-based High-Resolution 8-Channel Visual Atlas Renderer.
Zero Matplotlib dependency. Pure PyTorch, NumPy, and PIL.
"""

import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from exp_util import load_dataset
from graph_level_rasterizer import graph_to_map


# =============================================================================
# High-Precision 256-Color Scientific Colormap LUTs
# =============================================================================

def build_lut(knots_pos, knots_rgb):
    steps = np.linspace(0.0, 1.0, 256)
    lut = np.zeros((256, 3), dtype=np.uint8)
    for ch in range(3):
        lut[:, ch] = np.clip(np.interp(steps, knots_pos, knots_rgb[:, ch]), 0, 255).astype(np.uint8)
    return lut

# Magma
_magma_pos = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
_magma_rgb = np.array([
    [0, 0, 4], [20, 14, 54], [59, 15, 112], [100, 26, 128], [140, 41, 129],
    [183, 55, 121], [222, 73, 104], [247, 112, 92], [254, 159, 109], [254, 207, 146], [252, 253, 191]
], dtype=np.float32)
MAGMA_LUT = build_lut(_magma_pos, _magma_rgb)

# Inferno
_inferno_rgb = np.array([
    [0, 0, 4], [22, 11, 57], [66, 10, 104], [106, 23, 110], [147, 38, 103],
    [187, 55, 84], [221, 81, 58], [243, 120, 25], [249, 168, 13], [240, 217, 72], [252, 255, 164]
], dtype=np.float32)
INFERNO_LUT = build_lut(_magma_pos, _inferno_rgb)

# Plasma
_plasma_rgb = np.array([
    [13, 8, 135], [64, 3, 156], [106, 0, 168], [143, 13, 164], [177, 42, 144],
    [204, 71, 120], [225, 100, 98], [241, 130, 77], [251, 162, 54], [252, 202, 38], [240, 249, 33]
], dtype=np.float32)
PLASMA_LUT = build_lut(_magma_pos, _plasma_rgb)

# Viridis
_viridis_rgb = np.array([
    [68, 1, 84], [72, 35, 116], [65, 66, 129], [53, 94, 141], [42, 120, 142],
    [33, 144, 141], [34, 167, 133], [68, 190, 112], [121, 209, 81], [189, 223, 38], [253, 231, 37]
], dtype=np.float32)
VIRIDIS_LUT = build_lut(_magma_pos, _viridis_rgb)

# Cyan / Electric Blue (for Edge Flux)
_electric_rgb = np.array([
    [5, 10, 20], [10, 25, 45], [15, 45, 80], [20, 75, 130], [25, 110, 180],
    [30, 150, 210], [40, 190, 235], [80, 220, 245], [140, 235, 250], [200, 245, 255], [255, 255, 255]
], dtype=np.float32)
ELECTRIC_LUT = build_lut(_magma_pos, _electric_rgb)


def colorize_matrix(matrix: np.ndarray, lut: np.ndarray) -> Image.Image:
    v = np.clip(matrix, 0.0, 1.0)
    indices = (v * 255.0).astype(np.int32)
    rgb = lut[indices]
    return Image.fromarray(rgb, mode="RGB")


def render_atlas_panel():
    # 1. Load MUTAG dataset Graph #0
    dataset = load_dataset("MUTAG", seed=123)
    graph_idx = 0
    data = dataset[graph_idx]

    num_nodes = data.num_nodes
    num_edges = data.edge_index.shape[1]
    label = int(data.y.item()) if hasattr(data.y, 'item') else int(data.y)
    label_str = "Mutagenic (+1)" if label == 1 else "Non-Mutagenic (0)"

    print(f"[*] Rendering Master Atlas for MUTAG Graph #{graph_idx}: {num_nodes} atoms, {num_edges} bonds ({label_str})")

    # 2. Rasterize to 8-channel Master Atlas at 128x128
    res = 128
    atlas = graph_to_map(data, resolution=res, layout_method="spring", include_spectrogram=True, seed=123)
    atlas_np = atlas.cpu().numpy()  # (8, 128, 128)

    channels_meta = [
        # (Index, Title, Subtitle, Domain, LUT)
        (0, "Ch 0: Spectral Low-Pass", "Community Consensus (A^k * X)", "SPECTRAL", MAGMA_LUT),
        (1, "Ch 1: Spectral High-Pass", "Boundary Wavelet (ΔA^k * X)", "SPECTRAL", INFERNO_LUT),
        (2, "Ch 2: Spectral PageRank", "Structural Resonance (PPR)", "SPECTRAL", PLASMA_LUT),
        (3, "Ch 3: Spatial Node Density", "Gaussian KDE Splatting", "SPATIAL", PLASMA_LUT),
        (4, "Ch 4: Spatial Edge Flux", "Continuous Line Segment Field", "SPATIAL", ELECTRIC_LUT),
        (5, "Ch 5: Spatial Echo Basin", "Random Walk Return Prob", "SPATIAL", MAGMA_LUT),
        (6, "Ch 6: Feature Disparity", "Chemical Bond Contrast Field", "SPATIAL", INFERNO_LUT),
        (7, "Ch 7: Semantic Intensity", "Atom Type Beacon Splats", "SPATIAL", VIRIDIS_LUT),
    ]

    # 3. Canvas Construction (2x4 Grid with Dark Mode GitHub Theme)
    cell_w, cell_h = 280, 280
    pad_x, pad_y = 20, 20
    header_h = 90
    banner_h = 40

    canvas_w = pad_x * 5 + cell_w * 4
    canvas_h = header_h + banner_h + (cell_h + pad_y + 40) * 2 + pad_y

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(13, 17, 23))
    draw = ImageDraw.Draw(canvas)

    # Header
    draw.text(
        (pad_x, 20),
        f"Graph2Map 8-Channel Master Visual Atlas  |  MUTAG Molecule #{graph_idx}",
        fill=(88, 166, 255)
    )
    draw.text(
        (pad_x, 48),
        f"Topology: {num_nodes} Atoms (Nodes), {num_edges} Bonds (Edges)  |  Class Label: {label} ({label_str})  |  Tensor: [8, {res}, {res}]",
        fill=(139, 148, 158)
    )

    # Domain separator bar
    draw.text(
        (pad_x, header_h),
        "▲ ROW 1: SPECTRAL DYNAMICS (Graph-Level Spectrograms: Multi-Hop Diffusion × Fiedler Canonical 1D Manifold)",
        fill=(180, 140, 255)
    )

    # Render Row 1 (Channels 0, 1, 2) + Explanatory Card (Col 4)
    for col in range(4):
        x0 = pad_x + col * (cell_w + pad_x)
        y0 = header_h + banner_h

        if col < 3:
            ch_idx, title, subtitle, domain, lut = channels_meta[col]
            # Colorize and upscale
            img = colorize_matrix(atlas_np[ch_idx], lut).resize((cell_w, cell_h), Image.Resampling.BILINEAR)
            # Border
            draw.rectangle([x0 - 2, y0 - 2, x0 + cell_w + 1, y0 + cell_h + 1], outline=(48, 54, 61), width=1)
            canvas.paste(img, (x0, y0))

            # Labels below cell
            draw.text((x0, y0 + cell_h + 8), title, fill=(230, 237, 243))
            draw.text((x0, y0 + cell_h + 26), subtitle, fill=(139, 148, 158))
        else:
            # Explanatory card for Col 4
            draw.rectangle([x0 - 2, y0 - 2, x0 + cell_w + 1, y0 + cell_h + 1], fill=(22, 27, 34), outline=(48, 54, 61), width=1)
            draw.text((x0 + 15, y0 + 20), "Spectral Domain Engine", fill=(210, 168, 255))
            card_desc = [
                "• Y-Axis: 16 Diffusion Hops",
                "  (Top: Localized features,",
                "   Bottom: Global consensus)",
                "",
                "• X-Axis: 1D Graph Manifold",
                "  (Ordered by Laplacian Fiedler",
                "   eigenvector v2 for topological",
                "   neighborhood preservation)",
                "",
                "• Dynamic Range Compression:",
                "  log(1 + 20*S) / log(21)",
                "  High-contrast resonance without",
                "  saturation spikes."
            ]
            for l_idx, line in enumerate(card_desc):
                draw.text((x0 + 15, y0 + 55 + l_idx * 16), line, fill=(201, 209, 217))

    # Row 2 separator
    y_row2 = header_h + banner_h + cell_h + 50 + pad_y
    draw.text(
        (pad_x, y_row2 - 24),
        "▼ ROW 2: SPATIAL CARTOGRAPHY (Continuous 2D Topological Heatmaps: Force-Directed Layout Splatting)",
        fill=(88, 166, 255)
    )

    # Render Row 2 (Channels 3, 4, 5, 6, 7) - fits into 4 columns with Ch 7 placed nicely
    row2_items = channels_meta[3:]  # 5 items: Ch 3, 4, 5, 6, 7
    for col in range(4):
        x0 = pad_x + col * (cell_w + pad_x)
        y0 = y_row2

        ch_idx, title, subtitle, domain, lut = row2_items[col]
        img = colorize_matrix(atlas_np[ch_idx], lut).resize((cell_w, cell_h), Image.Resampling.BILINEAR)
        draw.rectangle([x0 - 2, y0 - 2, x0 + cell_w + 1, y0 + cell_h + 1], outline=(48, 54, 61), width=1)
        canvas.paste(img, (x0, y0))

        draw.text((x0, y0 + cell_h + 8), title, fill=(230, 237, 243))
        draw.text((x0, y0 + cell_h + 26), subtitle, fill=(139, 148, 158))

    # We also have Ch 7 (Semantic Intensity)! Let's expand canvas to 3 rows or place 4x2 grid cleanly!
    # Let's save this canvas
    return canvas, atlas_np, data


def main():
    # Render clean 2x4 grid containing all 8 channels
    dataset = load_dataset("MUTAG", seed=123)
    graph_idx = 0
    data = dataset[graph_idx]

    num_nodes = data.num_nodes
    num_edges = data.edge_index.shape[1]
    label = int(data.y.item()) if hasattr(data.y, 'item') else int(data.y)
    label_str = "Mutagenic (+1)" if label == 1 else "Non-Mutagenic (0)"

    res = 128
    atlas = graph_to_map(data, resolution=res, layout_method="spring", include_spectrogram=True, seed=123)
    atlas_np = atlas.cpu().numpy()

    channels_meta = [
        ("Ch 0: Spectral Low-Pass", "Community Consensus (A^k * X)", MAGMA_LUT),
        ("Ch 1: Spectral High-Pass", "Boundary Wavelet (ΔA^k * X)", INFERNO_LUT),
        ("Ch 2: Spectral PageRank", "Structural Resonance (PPR)", PLASMA_LUT),
        ("Ch 3: Spatial Node Density", "Gaussian KDE Splatting", PLASMA_LUT),
        ("Ch 4: Spatial Edge Flux", "Continuous Line Segment Field", ELECTRIC_LUT),
        ("Ch 5: Spatial Echo Basin", "Random Walk Return Prob", MAGMA_LUT),
        ("Ch 6: Feature Disparity", "Chemical Bond Contrast Field", INFERNO_LUT),
        ("Ch 7: Semantic Intensity", "Atom Type Beacon Splats", VIRIDIS_LUT),
    ]

    cell_w, cell_h = 260, 260
    pad_x, pad_y = 20, 20
    header_h = 75

    canvas_w = pad_x * 5 + cell_w * 4
    canvas_h = header_h + (cell_h + pad_y + 45) * 2 + pad_y

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(13, 17, 23))
    draw = ImageDraw.Draw(canvas)

    # Title
    draw.text(
        (pad_x, 15),
        f"Graph2Map 8-Channel Master Visual Atlas  |  MUTAG Molecule #{graph_idx}",
        fill=(88, 166, 255)
    )
    draw.text(
        (pad_x, 40),
        f"Topology: {num_nodes} Atoms (Nodes), {num_edges} Bonds (Edges)  |  Label: {label} ({label_str})  |  Tensor Shape: [8, {res}, {res}]",
        fill=(139, 148, 158)
    )

    for idx, (title, subtitle, lut) in enumerate(channels_meta):
        row = idx // 4
        col = idx % 4

        x0 = pad_x + col * (cell_w + pad_x)
        y0 = header_h + row * (cell_h + pad_y + 45)

        img = colorize_matrix(atlas_np[idx], lut).resize((cell_w, cell_h), Image.Resampling.BILINEAR)
        draw.rectangle([x0 - 2, y0 - 2, x0 + cell_w + 1, y0 + cell_h + 1], outline=(48, 54, 61), width=1)
        canvas.paste(img, (x0, y0))

        # Title & Subtitle
        draw.text((x0, y0 + cell_h + 6), title, fill=(230, 237, 243))
        draw.text((x0, y0 + cell_h + 22), subtitle, fill=(139, 148, 158))

    art_dir = r"C:\Users\UTSAB\.gemini\antigravity-ide\brain\0a8a5a5f-0bb9-4995-860f-d04ff45adb25"
    os.makedirs(art_dir, exist_ok=True)
    art_path = os.path.join(art_dir, "graph_8channel_atlas_mutag.png")
    canvas.save(art_path)

    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    local_path = os.path.join(results_dir, "graph_8channel_atlas_mutag.png")
    canvas.save(local_path)

    print(f"[+] Successfully rendered all 8 channels to: {art_path}")
    print(f"[+] Saved copy to: {local_path}")


if __name__ == '__main__':
    main()
