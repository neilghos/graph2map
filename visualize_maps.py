"""
Visualize Graph2Map Topological Heatmaps for Sample Graphs.
Runs without modifying environment, saves high-res panels using PIL.
"""

import os
import time
import torch
from pyg_loader import get_graph_dataset
from rasterizer import graph_to_map, save_heatmap_panel


def main():
    output_dir = "./maps_output"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading MUTAG dataset...")
    dataset = get_graph_dataset("MUTAG")

    # Select a few distinct graphs
    sample_indices = [0, 1, 5, 10]

    print(f"\nRasterizing {len(sample_indices)} sample graphs into 2D multi-channel maps...")
    print(f"Canvas resolution: 128x128 | Channels: 4")
    print(f"{'-'*60}")

    for idx in sample_indices:
        data = dataset[idx]
        t0 = time.time()

        # Generate (4, 128, 128) heatmap tensor
        tensor = graph_to_map(
            data,
            resolution=128,
            sigma_node=0.07,
            sigma_edge=0.035,
            layout_method="spring",
            seed=42 + idx
        )
        elapsed_ms = (time.time() - t0) * 1000.0

        label = data.y.item() if data.y.numel() == 1 else data.y.tolist()
        out_filename = os.path.join(output_dir, f"graph_{idx}_label_{label}.png")

        save_heatmap_panel(tensor, out_filename)

        print(
            f"Graph #{idx:02d} | Label: {label} | Nodes: {data.num_nodes:2d} | "
            f"Edges: {data.num_edges:2d} | Map Shape: {list(tensor.shape)} | "
            f"Time: {elapsed_ms:.1f}ms"
        )

    print(f"{'-'*60}")
    print(f"All sample maps successfully saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
