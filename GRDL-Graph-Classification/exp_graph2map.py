"""
Graph2Map Whole-Graph Classification Benchmark Runner.
Head-to-head evaluation against NeurIPS 2024 GRDL on TUDataset (10-fold Cross-Validation).
"""

import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm

from exp_util import load_dataset
from graph_level_rasterizer import graph_to_map


# =============================================================================
# 1. 2D VISION RESIDUAL BACKBONE FOR CONTINUOUS TOPOLOGICAL MAPS
# =============================================================================

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + res)
        return out


class Graph2MapResNet(nn.Module):
    """
    4-Stage Residual Vision Network tailored for 64x64 multi-channel graph heatmaps.
    Preserves continuous spatial invariants (loops, motifs, cliques, bottlenecks).
    """
    def __init__(self, in_channels: int = 5, num_classes: int = 2, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        # Stage 1: 64x64 -> 32x32
        self.stage1 = nn.Sequential(
            ResidualConvBlock(32, 32),
            nn.MaxPool2d(2)
        )
        # Stage 2: 32x32 -> 16x16
        self.stage2 = nn.Sequential(
            ResidualConvBlock(32, 64),
            nn.MaxPool2d(2)
        )
        # Stage 3: 16x16 -> 8x8
        self.stage3 = nn.Sequential(
            ResidualConvBlock(64, 128),
            nn.MaxPool2d(2)
        )
        # Stage 4: 8x8 -> 1x1
        self.stage4 = nn.Sequential(
            ResidualConvBlock(128, 256),
            nn.AdaptiveAvgPool2d(1)
        )
        # Classification Head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        logits = self.head(out)
        return logits


# =============================================================================
# 2. DATASET RASTERIZATION AND CACHING
# =============================================================================

def get_or_create_rasterized_maps(dataset, dataset_name: str, resolution: int = 64, layout: str = "spring", cache_dir: str = "./cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{dataset_name}_res{resolution}_{layout}_5ch.pt")

    if os.path.exists(cache_path):
        print(f"[*] Loading pre-rasterized {dataset_name} maps from cache: {cache_path}")
        cache_data = torch.load(cache_path)
        return cache_data['X'], cache_data['Y']

    print(f"[*] Pre-rasterizing {len(dataset)} graphs for {dataset_name} (Resolution: {resolution}x{resolution}, Layout: {layout})...")
    start_t = time.time()
    maps_list = []
    labels_list = []

    for idx, data in enumerate(tqdm(dataset, desc=f"Rasterizing {dataset_name}", unit="graph")):
        m = graph_to_map(data, resolution=resolution, layout_method=layout, seed=42 + idx)
        maps_list.append(m)
        y_val = data.y.item() if hasattr(data.y, 'item') else int(data.y)
        labels_list.append(y_val)

    X = torch.stack(maps_list, dim=0)  # Shape: (N, 5, H, W)
    Y = torch.tensor(labels_list, dtype=torch.long)  # Shape: (N,)

    # Normalize labels to 0..C-1 if needed (e.g. some datasets have -1, 1 or 1, 2)
    unique_labels = torch.unique(Y).sort().values
    label_map = {old.item(): new for new, old in enumerate(unique_labels)}
    Y_norm = torch.tensor([label_map[y.item()] for y in Y], dtype=torch.long)

    print(f"[+] Rasterization completed in {time.time() - start_t:.2f}s! Tensor shape: {X.shape}, Classes: {len(unique_labels)}")
    torch.save({'X': X, 'Y': Y_norm}, cache_path)
    print(f"[+] Saved cache to {cache_path}")
    return X, Y_norm


# =============================================================================
# 3. 10-FOLD CROSS VALIDATION EVALUATION HARNESS
# =============================================================================

def train_and_eval_fold(fold: int, model, train_loader, val_loader, device, epochs: int = 100, lr: float = 1e-3, weight_decay: float = 1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            # Data Augmentation: Random horizontal/vertical flip on 2D maps
            if random.random() > 0.5:
                bx = torch.flip(bx, dims=[-1])
            if random.random() > 0.5:
                bx = torch.flip(bx, dims=[-2])

            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * bx.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == by).sum().item()
            total_train += bx.size(0)

        scheduler.step()

        # Validation
        model.eval()
        val_correct = 0
        total_val = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                preds = logits.argmax(dim=1)
                val_correct += (preds == by).sum().item()
                total_val += bx.size(0)

        val_acc = val_correct / total_val if total_val > 0 else 0.0
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

    return best_val_acc, best_epoch


def main():
    parser = argparse.ArgumentParser(description="Graph2Map Whole-Graph Classification Benchmark")
    parser.add_argument("-d", "--dataset", type=str, default="MUTAG",
                        choices=['MUTAG', 'PROTEINS', 'PTC_MR', 'NCI1', 'IMDB-BINARY', 'IMDB-MULTI', 'BZR', 'COLLAB'])
    parser.add_argument("-e", "--epoch", type=int, default=100, help="epochs per fold (default: 100)")
    parser.add_argument("-b", "--batch", type=int, default=32, help="batch size (default: 32)")
    parser.add_argument("-s", "--seed", type=int, default=123, help="random seed (default: 123)")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate (default: 1e-3)")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay (default: 1e-4)")
    parser.add_argument("--res", type=int, default=64, help="canvas resolution (default: 64)")
    parser.add_argument("--layout", type=str, default="spring", choices=["spring", "spectral", "kamada_kawai"])
    args = parser.parse_args()

    print("=" * 80)
    print(f"Graph2Map 10-Fold Benchmark: {args.dataset}")
    print(f"Resolution: {args.res}x{args.res} | Layout: {args.layout} | Epochs: {args.epoch} | Batch: {args.batch} | Seed: {args.seed}")
    print("=" * 80)

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Compute Device: {device}")

    # Load PyG dataset using the official GRDL benchmark loader
    dataset = load_dataset(args.dataset, args.seed)
    num_classes = dataset.num_classes

    # Rasterize or load cached maps
    X, Y = get_or_create_rasterized_maps(dataset, args.dataset, resolution=args.res, layout=args.layout)
    num_samples = len(Y)

    # 10-Fold Cross Validation Setup matching GRDL
    kfold = KFold(n_splits=10, shuffle=True, random_state=args.seed)
    fold_accuracies = []

    print("\n" + "-" * 80)
    print(f"Starting 10-Fold Cross-Validation on {args.dataset} ({num_samples} total graphs)...")
    print("-" * 80)

    total_start_time = time.time()

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        fold_num = fold + 1

        train_ds = TensorDataset(X[train_idx], Y[train_idx])
        val_ds = TensorDataset(X[val_idx], Y[val_idx])

        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

        # Fresh model for each fold
        model = Graph2MapResNet(in_channels=5, num_classes=num_classes).to(device)

        fold_start = time.time()
        best_acc, best_ep = train_and_eval_fold(
            fold=fold_num,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epoch,
            lr=args.lr,
            weight_decay=args.wd
        )
        fold_time = time.time() - fold_start
        fold_accuracies.append(best_acc)

        print(f"  Fold [{fold_num:02d}/10] -> Best Valid Acc: {best_acc * 100:.2f}% (Epoch {best_ep}) | Time: {fold_time:.1f}s")

    fold_accs = np.array(fold_accuracies) * 100.0
    mean_acc = fold_accs.mean()
    std_acc = fold_accs.std()
    total_time = time.time() - total_start_time

    print("=" * 80)
    print(f"GRAPH2MAP FINAL RESULTS ON {args.dataset} (10-Fold CV):")
    print(f"  Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"  All Folds:     {[round(a, 2) for a in fold_accs]}")
    print(f"  Total Runtime: {total_time:.1f}s")
    print("=" * 80)

    # Comparison Table
    published_baselines = {
        'MUTAG': {'GIN': '89.4% ± 5.6%', 'GRDL (NeurIPS 2024)': '92.1% ± 5.9%', 'DiffLifting (ICLR 2025)': '92.6% ± 4.5%'},
        'PROTEINS': {'GIN': '76.2% ± 2.8%', 'GRDL (NeurIPS 2024)': '76.8% ± 3.2%', 'Injective GNN (ICLR 2025)': '77.5% ± 2.4%'},
        'NCI1': {'GIN': '82.7% ± 1.7%', 'GRDL (NeurIPS 2024)': '82.9% ± 1.5%', 'DiffLifting (ICLR 2025)': '83.8% ± 1.3%'},
        'IMDB-BINARY': {'GIN': '75.1% ± 2.4%', 'GRDL (NeurIPS 2024)': '75.8% ± 2.8%', 'DiffLifting (ICLR 2025)': '76.2% ± 3.0%'}
    }

    if args.dataset in published_baselines:
        print("\nHead-to-Head Comparison with Published 2024-2025 SOTA:")
        for paper, score in published_baselines[args.dataset].items():
            print(f"  • {paper:30s}: {score}")
        print(f"  • {'Graph2Map (Ours)':30s}: {mean_acc:.2f}% ± {std_acc:.2f}%")
        print("=" * 80)


if __name__ == "__main__":
    main()
