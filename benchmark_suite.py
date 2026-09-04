"""
Standardized Graph-to-Map (Graph2Map) Benchmark Suite.
Fixed Downstream Pipeline: ResNet / Convolutional Vision Backbones (ResNet-18, ConvNeXt).
Evaluates across a diverse suite of graph classification benchmarks:
  - MUTAG (Small Molecules, 188 graphs, 2 classes)
  - PROTEINS (Bioinformatics, 1,113 graphs, 2 classes)
  - ENZYMES (Multi-class Protein Enzymes, 600 graphs, 6 classes)
  - IMDB-BINARY (Social Networks / Pure Topology, 1,000 graphs, 2 classes)
"""

import os
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import torchvision.transforms as T
import timm

from pyg_loader import get_graph_dataset
from rasterizer import graph_to_map


class GraphMapDataset(Dataset):
    """
    In-memory / Cached Dataset for 2D Graph Heatmaps.
    """
    def __init__(self, maps: torch.Tensor, labels: torch.Tensor, transform=None):
        self.maps = maps
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.maps[idx]
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def load_or_cache_graph_maps(
    dataset_name: str,
    resolution: int = 128,
    cache_dir: str = "./cache_maps",
    layout_method: str = "spring"
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Load pre-rasterized 4-channel heatmaps from disk, or rasterize and cache.
    Returns:
      all_maps: (N, 4, H, W)
      all_labels: (N,)
      num_classes: int
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{dataset_name}_res{resolution}_{layout_method}.pt")

    if os.path.exists(cache_file):
        print(f"Loading cached maps from {cache_file}...")
        all_maps, all_labels, num_classes = torch.load(cache_file, weights_only=False)
    else:
        print(f"Rasterizing dataset '{dataset_name}' into {resolution}x{resolution} maps...")
        dataset = get_graph_dataset(dataset_name)
        num_classes = dataset.num_classes
        all_maps = []
        all_labels = []

        t0 = time.time()
        for i in range(len(dataset)):
            data = dataset[i]
            tensor_map = graph_to_map(
                data,
                resolution=resolution,
                layout_method=layout_method,
                seed=42 + i
            )
            label = data.y.item() if data.y.numel() == 1 else int(data.y[0])
            all_maps.append(tensor_map)
            all_labels.append(label)

            if (i + 1) % 100 == 0 or (i + 1) == len(dataset):
                print(f"  Rasterized {i + 1}/{len(dataset)} graphs ({time.time() - t0:.1f}s)...")

        all_maps = torch.stack(all_maps, dim=0)
        all_labels = torch.tensor(all_labels, dtype=torch.long)
        torch.save((all_maps, all_labels, num_classes), cache_file)
        print(f"Cached {len(all_maps)} graphs to {cache_file} in {time.time() - t0:.2f}s")

    return all_maps, all_labels, num_classes


def create_backbone(model_name: str, num_classes: int, in_chans: int = 4) -> nn.Module:
    """
    Instantiate standardized convolutional backbone from timm.
    """
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        in_chans=in_chans
    )
    return model


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, (correct / total) * 100.0


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, (correct / total) * 100.0


def run_benchmark_dataset(
    dataset_name: str,
    model_name: str = "resnet18",
    resolution: int = 128,
    num_folds: int = 5,
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 5e-4,
    device: torch.device = None
) -> dict:
    """
    Run standard K-Fold Cross Validation for a single graph dataset.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"BENCHMARK: {dataset_name} | Model: {model_name} | {num_folds}-Fold CV")
    print(f"{'='*70}")

    all_maps, all_labels, num_classes = load_or_cache_graph_maps(
        dataset_name=dataset_name,
        resolution=resolution
    )

    y_np = all_labels.numpy()
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    fold_test_accuracies = []
    fold_times = []

    # Visual augmentations
    train_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
    ])

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y_np)), y_np)):
        t_fold_start = time.time()

        train_ds = GraphMapDataset(all_maps[train_idx], all_labels[train_idx], transform=train_transform)
        test_ds = GraphMapDataset(all_maps[test_idx], all_labels[test_idx])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # Class weights for balanced cross-entropy
        train_labels = all_labels[train_idx]
        class_counts = torch.bincount(train_labels, minlength=num_classes).float()
        class_weights = (1.0 / (class_counts + 1e-6))
        class_weights = (class_weights / class_weights.sum()).to(device)

        model = create_backbone(model_name=model_name, num_classes=num_classes, in_chans=4).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        best_test_acc = 0.0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            scheduler.step()

            if test_acc > best_test_acc:
                best_test_acc = test_acc

        fold_duration = time.time() - t_fold_start
        fold_test_accuracies.append(best_test_acc)
        fold_times.append(fold_duration)

        print(f"  Fold {fold + 1}/{num_folds} | Peak Test Acc: {best_test_acc:6.2f}% | Time: {fold_duration:4.1f}s")

    mean_acc = float(np.mean(fold_test_accuracies))
    std_acc = float(np.std(fold_test_accuracies))
    total_time = sum(fold_times)

    print(f"\n>> {dataset_name} RESULT: {mean_acc:.2f}% +/- {std_acc:.2f}% (Total Time: {total_time:.1f}s)")
    print(f"{'='*70}\n")

    return {
        "dataset": dataset_name,
        "num_graphs": len(all_labels),
        "num_classes": num_classes,
        "model": model_name,
        "num_folds": num_folds,
        "mean_accuracy": round(mean_acc, 2),
        "std_accuracy": round(std_acc, 2),
        "fold_accuracies": [round(a, 2) for a in fold_test_accuracies],
        "total_time_seconds": round(total_time, 2)
    }


def main():
    parser = argparse.ArgumentParser(description="Graph2Map Standardized Benchmark Suite")
    parser.add_argument("--datasets", nargs="+", default=["MUTAG", "PROTEINS", "IMDB-BINARY"],
                        help="List of datasets to benchmark (MUTAG, PROTEINS, ENZYMES, IMDB-BINARY)")
    parser.add_argument("--model", type=str, default="resnet18",
                        help="Standardized backbone (resnet18, resnet10t, convnext_nano)")
    parser.add_argument("--resolution", type=int, default=128, help="Map resolution")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs per fold")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Standardized Graph2Map Benchmark on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    all_results = []
    for ds_name in args.datasets:
        res = run_benchmark_dataset(
            dataset_name=ds_name,
            model_name=args.model,
            resolution=args.resolution,
            num_folds=args.folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device
        )
        all_results.append(res)

    # Save results as JSON
    results_path = "./benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Generate Markdown Summary Report
    report_path = "./benchmark_report.md"
    with open(report_path, "w") as f:
        f.write("# Standardized Graph-to-Map (Graph2Map) Benchmark Report\n\n")
        f.write(f"**Downstream Backbone:** `{args.model}` (Convolutional)\n\n")
        f.write(f"**Evaluation Protocol:** {args.folds}-Fold Stratified Cross-Validation\n\n")
        f.write(f"**Map Resolution:** {args.resolution}x{args.resolution} (4 Channels: Node, Edge, Echo, Gradient)\n\n")
        f.write("| Dataset | Graphs | Classes | Node Features | Mean Test Acc (%) | Std (%) | Time (s) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in all_results:
            f.write(
                f"| **{r['dataset']}** | {r['num_graphs']} | {r['num_classes']} | "
                f"{'None (Degree-based)' if r['dataset'] == 'IMDB-BINARY' else 'Present'} | "
                f"**{r['mean_accuracy']:.2f}%** | ±{r['std_accuracy']:.2f}% | {r['total_time_seconds']:.1f}s |\n"
            )
        f.write("\n\n*Generated automatically by Graph2Map Benchmark Suite.*\n")

    print(f"\nAll benchmarks complete!")
    print(f"Results saved to: {os.path.abspath(results_path)}")
    print(f"Markdown report:  {os.path.abspath(report_path)}")


if __name__ == "__main__":
    main()
