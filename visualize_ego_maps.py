"""
Inspect and Visualize Node-Level Ego-Maps using Polynormer's Dataset Loader and Benchmark Splits.

This script:
1. Loads the 'amazon-photo' dataset using Polynormer's official NCDataset loader.
2. Loads the official benchmark split from:
   './Polynormer-master/Polynormer-master/data/amazon-photo_split.npz'
3. Inspects and prints split statistics (train / valid / test distribution, class counts).
4. Extracts and rasterizes 4-channel Ego-Maps for sample nodes across splits.
5. Computes channel statistics (density, edge coverage, PPR diffusion, feature homophily).
6. Saves publication-grade visual panels to './ego_maps_output/'.
"""

import os
import sys
import time
import torch
import numpy as np

# Add Polynormer repository to path
POLYNORMER_ROOT = os.path.abspath(r"./Polynormer-master/Polynormer-master")
if POLYNORMER_ROOT not in sys.path:
    sys.path.insert(0, POLYNORMER_ROOT)

from dataset import load_dataset
from data_utils import load_fixed_splits
from node_level_rasterizer import node_to_ego_map, save_ego_map_panel


def inspect_and_visualize():
    output_dir = "./ego_maps_output"
    os.makedirs(output_dir, exist_ok=True)

    polynormer_data_dir = os.path.join(POLYNORMER_ROOT, "data")

    print("=" * 80)
    print("STEP 1: Loading 'amazon-photo' with Polynormer's official loader")
    print(f"Data directory: {os.path.abspath(polynormer_data_dir)}")
    print("=" * 80)

    # 1. Load Dataset directly from Polynormer's data directory
    ds = load_dataset(polynormer_data_dir, "amazon-photo")
    graph, label = ds[0]

    edge_index = graph["edge_index"]
    x = graph["node_feat"]
    num_nodes = graph["num_nodes"]
    num_edges = edge_index.size(1)
    num_features = x.size(1)
    num_classes = int(label.max().item()) + 1

    print(f"  • Total Nodes:      {num_nodes:,}")
    print(f"  • Total Edges:      {num_edges:,}")
    print(f"  • Feature Dim:      {num_features}")
    print(f"  • Num Classes:      {num_classes}")

    print("\n" + "=" * 80)
    print("STEP 2: Loading official benchmark splits from Polynormer data directory")
    print("=" * 80)

    split_file = os.path.join(polynormer_data_dir, "amazon-photo_split.npz")
    print(f"  • Split File Path:  {os.path.abspath(split_file)}")

    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Cannot find split file at {split_file}")

    splits_lst = load_fixed_splits(polynormer_data_dir, ds, "amazon-photo")
    split = splits_lst[0]  # First fold / standard benchmark split

    train_idx = split["train"]
    valid_idx = split["valid"]
    test_idx = split["test"]

    print(f"  • Train Nodes:      {len(train_idx):,} ({len(train_idx) / num_nodes * 100:.1f}%)")
    print(f"  • Validation Nodes: {len(valid_idx):,} ({len(valid_idx) / num_nodes * 100:.1f}%)")
    print(f"  • Test Nodes:       {len(test_idx):,} ({len(test_idx) / num_nodes * 100:.1f}%)")

    # Pick representative nodes across splits and classes
    # 2 from train, 1 from valid, 1 from test (with different labels)
    samples_to_plot = []

    # Train samples
    train_labels = label[train_idx]
    unique_train_classes = torch.unique(train_labels)
    for c in unique_train_classes[:2]:
        match = train_idx[train_labels == c]
        if len(match) > 0:
            samples_to_plot.append(("train", int(match[0].item()), int(c.item())))

    # Valid sample
    val_labels = label[valid_idx]
    for c in torch.unique(val_labels):
        if c not in [s[2] for s in samples_to_plot]:
            match = valid_idx[val_labels == c]
            if len(match) > 0:
                samples_to_plot.append(("valid", int(match[0].item()), int(c.item())))
                break

    # Test sample
    test_labels = label[test_idx]
    for c in torch.unique(test_labels):
        if c not in [s[2] for s in samples_to_plot]:
            match = test_idx[test_labels == c]
            if len(match) > 0:
                samples_to_plot.append(("test", int(match[0].item()), int(c.item())))
                break

    print("\n" + "=" * 80)
    print(f"STEP 3: Rasterizing and Inspecting {len(samples_to_plot)} Target Node Ego-Maps")
    print(f"Canvas Resolution: 128x128 | 4 Channels | Bounded to Top-128 Local Topology")
    print("=" * 80)

    for split_name, node_id, node_label in samples_to_plot:
        t0 = time.time()

        # Generate 4-channel 128x128 continuous Ego-Map
        tensor_map = node_to_ego_map(
            target_node=node_id,
            edge_index=edge_index,
            x=x,
            num_hops=2,
            resolution=128,
            sigma_node=0.08,
            sigma_edge=0.035,
            max_nodes=128,
            seed=42 + node_id
        )
        elapsed_ms = (time.time() - t0) * 1000.0

        # Save annotated image panel
        out_filename = f"split_{split_name}_node_{node_id}_class_{node_label}.png"
        out_path = os.path.join(output_dir, out_filename)
        save_ego_map_panel(
            tensor=tensor_map,
            filepath=out_path,
            target_node_id=node_id,
            label=node_label,
            split_name=split_name
        )

        # Inspection statistics per channel
        ch_names = ["Ch 0 (Ego-Density)", "Ch 1 (Edges)", "Ch 2 (PPR Diffusion)", "Ch 3 (Homophily)"]
        ch_stats = []
        for c in range(4):
            ch_data = tensor_map[c].numpy()
            mean_val = float(ch_data.mean())
            max_val = float(ch_data.max())
            sparsity = float((ch_data < 0.05).mean() * 100.0)
            ch_stats.append(f"{ch_names[c]}: mean={mean_val:.3f}, max={max_val:.2f}, bg_sparsity={sparsity:.1f}%")

        print(f"\n[Target Node #{node_id}] | Split: {split_name.upper():5s} | Class: {node_label} | Time: {elapsed_ms:.1f}ms")
        print(f"  Saved Image: {out_filename}")
        for stat in ch_stats:
            print(f"    - {stat}")

    print("\n" + "=" * 80)
    print(f"SUCCESS: Inspection complete. Visual panels saved to:\n  {os.path.abspath(output_dir)}")
    print("=" * 80)


if __name__ == "__main__":
    inspect_and_visualize()
