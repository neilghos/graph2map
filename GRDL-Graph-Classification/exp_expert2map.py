"""
Graph2Map GCN-to-Map Benchmark Runner (exp_expert2map.py).
Evaluates the single-model GCN baseline via 10-fold cross validation on 3-channel GCN heatmaps.
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
from datetime import datetime

from exp_util import load_dataset
from expert_rasterizer import get_or_create_gcn_maps


# =============================================================================
# 1. 2D VISION RESIDUAL BACKBONE FOR GCN HEATMAPS
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
        return self.act(out + res)


class GCNMapResNet(nn.Module):
    """
    4-Stage Residual Vision Network tailored for 64x64 GCN Heatmaps:
      - Stage 1: 64x64 -> 32x32 (32 channels)
      - Stage 2: 32x32 -> 16x16 (64 channels)
      - Stage 3: 16x16 -> 8x8 (128 channels)
      - Stage 4: 8x8 -> 1x1 (256 channels)
      - Classification Head with Dropout
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        self.stage1 = nn.Sequential(
            ResidualConvBlock(32, 32),
            nn.MaxPool2d(2)
        )
        self.stage2 = nn.Sequential(
            ResidualConvBlock(32, 64),
            nn.MaxPool2d(2)
        )
        self.stage3 = nn.Sequential(
            ResidualConvBlock(64, 128),
            nn.MaxPool2d(2)
        )
        self.stage4 = nn.Sequential(
            ResidualConvBlock(128, 256),
            nn.AdaptiveAvgPool2d(1)
        )
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
# 2. TOPOLOGICALLY INVARIANT MAP AUGMENTATION
# =============================================================================

def augment_gcn_maps(bx: torch.Tensor, max_shift: int = 3, cutout_prob: float = 0.3, cutout_size: int = 8) -> torch.Tensor:
    """
    Applies domain-aware topological augmentations to 3-channel GCN maps:
      - Channels 0 & 1 (GCN Manifold & Laplacian): Horizontal flip only (reverses 1D Fiedler sequence).
        Rotations and vertical flips are disabled to prevent scrambling the 64 latent feature rows.
      - Channel 2 (2D Spatial Layout): Full 2D rotations (rot90), vertical flip, and spatial jitter.
    """
    bx = bx.clone()

    # 1. Horizontal flip across all channels (preserves 1D graph manifold traversal)
    if random.random() > 0.5:
        bx = torch.flip(bx, dims=[-1])

    # 2. 2D Euclidean operations on Spatial Channel 2 only
    k = random.randint(0, 3)
    if k > 0:
        bx[:, 2:] = torch.rot90(bx[:, 2:], k=k, dims=[-2, -1])

    if random.random() > 0.5:
        bx[:, 2:] = torch.flip(bx[:, 2:], dims=[-2])

    if max_shift > 0 and random.random() > 0.5:
        dy = random.randint(-max_shift, max_shift)
        dx = random.randint(-max_shift, max_shift)
        if dy != 0 or dx != 0:
            bx[:, 2:] = torch.roll(bx[:, 2:], shifts=(dy, dx), dims=(-2, -1))
            if dy > 0:
                bx[:, 2:, :dy, :] = 0.0
            elif dy < 0:
                bx[:, 2:, dy:, :] = 0.0
            if dx > 0:
                bx[:, 2:, :, :dx] = 0.0
            elif dx < 0:
                bx[:, 2:, :, dx:] = 0.0

    if cutout_prob > 0 and random.random() < cutout_prob:
        h, w = bx.shape[-2], bx.shape[-1]
        top = random.randint(0, max(0, h - cutout_size))
        left = random.randint(0, max(0, w - cutout_size))
        bx[:, 2:, top:top + cutout_size, left:left + cutout_size] = 0.0

    return bx


# =============================================================================
# 3. 10-FOLD CV TRAINING HARNESS WITH EARLY STOPPING
# =============================================================================

def train_and_eval_fold(
    fold: int,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 80,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    patience: int = 20
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0
    best_train_acc = 0.0
    patience_counter = 0

    print_step = max(1, epochs // 5)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            bx = augment_gcn_maps(bx)

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
        val_loss = 0.0
        total_val = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss += loss.item() * bx.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == by).sum().item()
                total_val += bx.size(0)

        ep_train_acc = train_correct / total_train if total_train > 0 else 0.0
        ep_train_loss = train_loss / total_train if total_train > 0 else 0.0
        ep_val_acc = val_correct / total_val if total_val > 0 else 0.0
        ep_val_loss = val_loss / total_val if total_val > 0 else 0.0

        if ep_val_acc > best_val_acc:
            best_val_acc = ep_val_acc
            best_epoch = epoch
            best_train_acc = ep_train_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % print_step == 0 or epoch == epochs or epoch == 1:
            print(f"    [Ep {epoch:03d}/{epochs:03d}] Train Loss: {ep_train_loss:.4f}, Train Acc: {ep_train_acc * 100:6.2f}% | Val Loss: {ep_val_loss:.4f}, Val Acc: {ep_val_acc * 100:6.2f}% (Best Val: {best_val_acc * 100:.2f}% @ Ep {best_epoch:02d})")

        if patience > 0 and patience_counter >= patience and epoch >= 30:
            print(f"    [*] Early stopping triggered at Epoch {epoch:03d} (Best Val: {best_val_acc * 100:.2f}% @ Ep {best_epoch:02d})")
            break

    return best_val_acc, best_epoch, best_train_acc, ep_train_acc, ep_train_loss


# =============================================================================
# 4. MAIN BENCHMARK FUNCTION
# =============================================================================

def run_expert2map_benchmark(
    dataset_name: str = "PROTEINS",
    epochs: int = 80,
    patience: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    dropout: float = 0.3,
    resolution: int = 64,
    layout: str = "spring",
    seed: int = 123,
    cache_dir: str = "./cache"
):
    print("=" * 80)
    print(f"Graph2Map Single-Model GCN Baseline: {dataset_name}")
    print(f"Resolution: {resolution}x{resolution} | Layout: {layout} | Epochs: {epochs} | Batch: {batch_size} | WD: {weight_decay} | Seed: {seed}")
    print("=" * 80)

    # Set seeds
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Compute Device: {device}")

    # Load dataset
    raw_dataset = load_dataset(dataset_name, seed=seed)

    # Rasterize or load pre-rasterized GCN maps
    X, Y = get_or_create_gcn_maps(
        dataset=raw_dataset,
        dataset_name=dataset_name,
        resolution=resolution,
        layout=layout,
        cache_dir=cache_dir
    )

    num_samples, in_channels, H, W = X.shape
    num_classes = len(torch.unique(Y))

    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    fold_val_accs = []
    fold_train_accs = []
    fold_final_train_accs = []

    print("\n" + "-" * 80)
    print(f"Starting 10-Fold Cross-Validation on {dataset_name} ({num_samples} total graphs, 3 channels)...")
    print("-" * 80)

    total_start = time.time()

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        fold_num = fold + 1
        # Explicitly seed per-fold so RNG state doesn't drift based on previous fold's training steps
        fold_seed = seed + fold
        random.seed(fold_seed)
        np.random.seed(fold_seed)
        torch.manual_seed(fold_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(fold_seed)
            torch.cuda.manual_seed_all(fold_seed)

        train_ds = TensorDataset(X[train_idx], Y[train_idx])
        val_ds = TensorDataset(X[val_idx], Y[val_idx])

        g = torch.Generator()
        g.manual_seed(fold_seed)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = GCNMapResNet(in_channels=in_channels, num_classes=num_classes, dropout=dropout).to(device)

        fold_start = time.time()
        best_acc, best_ep, best_train_acc, final_train_acc, final_train_loss = train_and_eval_fold(
            fold=fold_num,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience
        )
        fold_time = time.time() - fold_start
        fold_val_accs.append(best_acc)
        fold_train_accs.append(best_train_acc)
        fold_final_train_accs.append(final_train_acc)

        print(f"  Fold [{fold_num:02d}/10] -> Best Valid Acc: {best_acc * 100:6.2f}% (Epoch {best_ep:02d}) | Final Train Acc: {final_train_acc * 100:6.2f}% | Time: {fold_time:.1f}s")

    fold_val_arr = np.array(fold_val_accs) * 100.0
    mean_val = fold_val_arr.mean()
    std_val = fold_val_arr.std()
    total_time = time.time() - total_start

    print("=" * 80)
    print(f"GRAPH2MAP GCN BASELINE FINAL RESULTS ON {dataset_name} (10-Fold CV):")
    print(f"  Validation Accuracy:   {mean_val:.2f}% ± {std_val:.2f}%")
    print(f"  All Valid Folds:       {[round(float(a), 2) for a in fold_val_arr]}")
    print(f"  Total Runtime:         {total_time:.1f}s")
    print("=" * 80)

    # Published SOTA Baselines
    published_baselines = {
        'MUTAG': {'GIN': '89.4% ± 5.6%', 'Graphormer': '89.6% ± 6.2%', 'SAT': '92.6% ± 4.3%', 'GRDL': '92.1% ± 5.9%'},
        'PROTEINS': {'GIN': '76.2% ± 2.8%', 'Graphormer': '76.3% ± 2.7%', 'SAT': '77.7% ± 3.2%', 'WiTTopoPool': '80.0% ± 3.2%', 'GRDL': '82.6% ± 1.2%'},
        'BZR': {'GIN': '82.6% ± 3.5%', 'Graphormer': '85.3% ± 2.3%', 'SAT': '91.7% ± 2.1%', 'GRDL': '92.0% ± 1.1%'},
        'PTC_MR': {'GIN': '64.6% ± 7.0%', 'GRDL': '68.3% ± 5.4%', 'Graphormer': '71.4% ± 5.2%'},
        'NCI1': {'GIN': '82.7% ± 1.7%', 'SAT': '79.2% ± 1.4%', 'GRDL': '82.5% ± 1.4%'}
    }

    if dataset_name in published_baselines:
        print("\nHead-to-Head Comparison with Published Baselines:")
        for name, score in published_baselines[dataset_name].items():
            print(f"  • {name:<28}: {score}")
        print(f"  • Graph2Map GCN (Ours)       : {mean_val:.2f}% ± {std_val:.2f}%")
        print("=" * 80 + "\n")

    # Log to files
    os.makedirs("./results", exist_ok=True)
    log_path = "./results/expert2map_log.txt"
    with open(log_path, "a") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {dataset_name} (GCN-Map, Res {resolution}, WD {weight_decay}): {mean_val:.2f}% ± {std_val:.2f}% (Folds: {[round(float(a), 2) for a in fold_val_arr]})\n")

    csv_path = "./results/expert2map_results.csv"
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a") as f:
        if not file_exists:
            f.write("dataset,mean_acc,std_acc,resolution,layout,epochs,batch_size,lr,time_sec\n")
        f.write(f"{dataset_name},{mean_val:.2f},{std_val:.2f},{resolution},{layout},{epochs},{batch_size},{lr},{total_time:.1f}\n")

    print(f"[+] Logged results to {log_path} and {csv_path}")
    return mean_val, std_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph2Map Single-Model GCN Baseline")
    parser.add_argument("-d", "--dataset", type=str, default="PROTEINS",
                        choices=['MUTAG', 'PROTEINS', 'PTC_MR', 'NCI1', 'IMDB-BINARY', 'IMDB-MULTI', 'BZR', 'COLLAB'])
    parser.add_argument("-e", "--epoch", type=int, default=200, help="epochs per fold (default: 80)")
    parser.add_argument("--patience", type=int, default=0, help="early stopping patience (default: 20)")
    parser.add_argument("-b", "--batch", type=int, default=32, help="batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate (default: 1e-3)")
    parser.add_argument("--wd", type=float, default=1e-3, help="weight decay (default: 1e-3)")
    parser.add_argument("--dropout", type=float, default=0.3, help="dropout probability (default: 0.3)")
    parser.add_argument("--res", type=int, default=64, help="canvas resolution (default: 64)")
    parser.add_argument("--layout", type=str, default="spring", choices=["spring", "spectral", "kamada_kawai"])
    parser.add_argument("-s", "--seed", type=int, default=123, help="random seed (default: 123)")
    args = parser.parse_args()

    run_expert2map_benchmark(
        dataset_name=args.dataset,
        epochs=args.epoch,
        patience=args.patience,
        batch_size=args.batch,
        lr=args.lr,
        weight_decay=args.wd,
        dropout=args.dropout,
        resolution=args.res,
        layout=args.layout,
        seed=args.seed
    )
