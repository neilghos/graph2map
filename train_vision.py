"""
Train Modern Fast Vision Backbones (ConvNeXt, ResNet) on Graph2Map Heatmaps.
Supports timm architectures: convnext_nano, convnext_tiny, resnet18, resnet10t.
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

from pyg_loader import get_graph_dataset
from rasterizer import graph_to_map


class GraphMapDataset(Dataset):
    """
    Dataset that maps PyG graph objects into 2D multi-channel continuous heatmaps.
    Supports pre-rasterization and caching for high-speed GPU training.
    """
    def __init__(
        self,
        pyg_dataset,
        indices: list,
        resolution: int = 128,
        cache_dir: str = "./cache_maps",
        dataset_name: str = "MUTAG",
        transform = None
    ):
        self.indices = indices
        self.transform = transform

        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{dataset_name}_res{resolution}.pt")

        if os.path.exists(cache_file):
            print(f"Loading pre-rasterized maps from cache: {cache_file}")
            all_maps, all_labels = torch.load(cache_file, weights_only=False)
        else:
            print(f"Rasterizing {len(pyg_dataset)} graphs into {resolution}x{resolution} maps (one-time)...")
            all_maps = []
            all_labels = []
            t0 = time.time()
            for idx in range(len(pyg_dataset)):
                data = pyg_dataset[idx]
                tensor_map = graph_to_map(data, resolution=resolution, seed=42 + idx)
                label = data.y.item() if data.y.numel() == 1 else int(data.y[0])
                all_maps.append(tensor_map)
                all_labels.append(label)
                if (idx + 1) % 50 == 0 or (idx + 1) == len(pyg_dataset):
                    print(f"  Rasterized {idx + 1}/{len(pyg_dataset)} graphs...")
            
            all_maps = torch.stack(all_maps, dim=0)  # (N, C, H, W)
            all_labels = torch.tensor(all_labels, dtype=torch.long)
            torch.save((all_maps, all_labels), cache_file)
            print(f"Cached all maps to {cache_file} in {time.time() - t0:.2f}s")

        self.maps = all_maps[self.indices]
        self.labels = all_labels[self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img = self.maps[idx]
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_vision_model(
    model_name: str = "convnext_nano", 
    num_classes: int = 2, 
    in_chans: int = 4
):
    """
    Instantiate ConvNeXt, ResNet, or other vision backbone from timm.
    """
    print(f"Initializing Backbone: '{model_name}' (in_chans={in_chans}, num_classes={num_classes})")
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        in_chans=in_chans
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    return model


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, (correct / total) * 100.0


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, (correct / total) * 100.0


def run_experiment(
    dataset_name: str = "MUTAG",
    model_name: str = "convnext_nano",
    resolution: int = 128,
    batch_size: int = 16,
    epochs: int = 45,
    lr: float = 5e-4,
    weight_decay: float = 1e-3,
    seed: int = 42
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    raw_dataset = get_graph_dataset(dataset_name)
    num_graphs = len(raw_dataset)
    num_classes = raw_dataset.num_classes

    # Train/Val/Test Split (80% / 10% / 10%)
    perm = torch.randperm(num_graphs, generator=torch.Generator().manual_seed(seed)).tolist()
    num_train = int(0.80 * num_graphs)
    num_val = int(0.10 * num_graphs)

    train_idx = perm[:num_train]
    val_idx = perm[num_train:num_train + num_val]
    test_idx = perm[num_train + num_val:]

    # Spatial augmentations: reflection & slight rotations
    train_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
    ])

    print("\nPreparing GraphMap Datasets...")
    train_ds = GraphMapDataset(raw_dataset, train_idx, resolution=resolution, dataset_name=dataset_name, transform=train_transform)
    val_ds = GraphMapDataset(raw_dataset, val_idx, resolution=resolution, dataset_name=dataset_name)
    test_ds = GraphMapDataset(raw_dataset, test_idx, resolution=resolution, dataset_name=dataset_name)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Split sizes: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    # Class weight calculation for balanced training
    labels_train = train_ds.labels
    class_counts = torch.bincount(labels_train, minlength=num_classes).float()
    class_weights = (1.0 / (class_counts + 1e-6))
    class_weights = (class_weights / class_weights.sum()).to(device)

    # Instantiate Model
    model = get_vision_model(model_name=model_name, num_classes=num_classes, in_chans=4).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_test_acc = 0.0

    print(f"\nStarting {model_name} Training for {epochs} Epochs...")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>10} | {'Val Loss':>9} | {'Val Acc':>8} | {'Test Acc':>8} | {'LR':>9}")
    print("-" * 75)

    for epoch in range(1, epochs + 1):
        current_lr = scheduler.get_last_lr()[0]
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"{epoch:6d} | {train_loss:10.4f} | {train_acc:9.2f}% | "
                f"{val_loss:9.4f} | {val_acc:7.2f}% | {test_acc:7.2f}% | {current_lr:9.2e}"
            )

    print("-" * 75)
    print(f"Training Complete!")
    print(f"Best Val Accuracy:  {best_val_acc:.2f}%")
    print(f"Corresponding Test: {best_test_acc:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vision Backbone on Graph2Map Heatmaps")
    parser.add_argument("--dataset", type=str, default="MUTAG", help="Graph dataset name")
    parser.add_argument("--model", type=str, default="convnext_nano", help="timm backbone: convnext_nano, resnet18, resnet10t")
    parser.add_argument("--resolution", type=int, default=128, help="Map resolution")
    parser.add_argument("--epochs", type=int, default=45, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    args = parser.parse_args()

    run_experiment(
        dataset_name=args.dataset,
        model_name=args.model,
        resolution=args.resolution,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
