"""
Graph-Level Task Data Loader using PyTorch Geometric (PyG).
Supports standard graph classification benchmarks (MUTAG, PROTEINS, ENZYMES, etc.).
"""

import os
import argparse
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader


def get_graph_dataset(name: str = "MUTAG", root: str = "./data"):
    """
    Load a graph classification dataset from TUDataset.
    
    Common options:
      - MUTAG: 188 mutagenic aromatic/heteroaromatic compounds (binary classification)
      - PROTEINS: 1,113 protein structures (binary classification)
      - ENZYMES: 600 enzymes across 6 EC top-level classes
    """
    os.makedirs(root, exist_ok=True)
    dataset = TUDataset(root=root, name=name)
    return dataset


def create_dataloaders(
    dataset, 
    batch_size: int = 32, 
    train_ratio: float = 0.8, 
    val_ratio: float = 0.1, 
    seed: int = 42,
    shuffle: bool = True
):
    """
    Split a PyG dataset into train, validation, and test sets and wrap in PyG DataLoaders.
    """
    generator = torch.Generator().manual_seed(seed)
    num_graphs = len(dataset)
    num_train = int(train_ratio * num_graphs)
    num_val = int(val_ratio * num_graphs)
    num_test = num_graphs - num_train - num_val

    # Random permutation split
    perm = torch.randperm(num_graphs, generator=generator).tolist()
    train_indices = perm[:num_train]
    val_indices = perm[num_train:num_train + num_val]
    test_indices = perm[num_train + num_val:]

    train_set = dataset[train_indices]
    val_set = dataset[val_indices]
    test_set = dataset[test_indices]

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, (train_set, val_set, test_set)


def print_dataset_stats(dataset):
    """
    Print structural statistics and metadata of the graph dataset.
    """
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset.name}")
    print(f"{'='*50}")
    print(f"Number of graphs:       {len(dataset)}")
    print(f"Number of classes:      {dataset.num_classes}")
    print(f"Node feature dim:       {dataset.num_node_features}")
    print(f"Edge feature dim:       {dataset.num_edge_features}")

    # Compute average nodes and edges
    total_nodes = sum(data.num_nodes for data in dataset)
    total_edges = sum(data.num_edges for data in dataset)
    avg_nodes = total_nodes / len(dataset)
    avg_edges = total_edges / len(dataset)
    print(f"Average nodes / graph:  {avg_nodes:.2f}")
    print(f"Average edges / graph:  {avg_edges:.2f}")

    # Inspect the first graph sample
    sample = dataset[0]
    print(f"\n--- First Graph Sample ---")
    print(f"Nodes: {sample.num_nodes}, Edges: {sample.num_edges}")
    print(f"Label (y): {sample.y.item() if sample.y.numel() == 1 else sample.y.tolist()}")
    if sample.x is not None:
        print(f"Node feature shape (x): {sample.x.shape}")
    if sample.edge_index is not None:
        print(f"Edge index shape:       {sample.edge_index.shape}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PyG Graph Dataset Loader")
    parser.add_argument("--dataset", type=str, default="MUTAG", help="Dataset name (MUTAG, PROTEINS, ENZYMES)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to store datasets")
    args = parser.parse_args()

    print(f"Loading dataset '{args.dataset}' from '{args.data_dir}'...")
    ds = get_graph_dataset(name=args.dataset, root=args.data_dir)
    print_dataset_stats(ds)

    train_loader, val_loader, test_loader, splits = create_dataloaders(ds, batch_size=args.batch_size)
    print(f"Train batches: {len(train_loader)} ({len(splits[0])} graphs)")
    print(f"Val batches:   {len(val_loader)} ({len(splits[1])} graphs)")
    print(f"Test batches:  {len(test_loader)} ({len(splits[2])} graphs)")

    # Test iterating one batch
    for batch in train_loader:
        print(f"\nSample Batch:")
        print(f"Batch object: {batch}")
        print(f"Total nodes in batch: {batch.num_nodes}")
        print(f"Total edges in batch: {batch.num_edges}")
        print(f"Batch graph labels:   {batch.y.tolist()}")
        break
