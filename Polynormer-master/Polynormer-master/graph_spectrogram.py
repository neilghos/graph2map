"""
Graph Spectrogram Representation: The Literal Audio-Mel Analogy for Graph Learning.
Pure PyTorch + PIL implementation (Zero Matplotlib dependency).

Core Concept:
  In audio, 1D acoustic signals are converted into 2D Time-Frequency representations (Log-Mel Spectrograms).
  In GraphSpectrogram, irregular non-Euclidean graphs are converted into 2D (or 3D multi-channel)
  Time-Frequency Spectrograms:
    - Vertical Axis (Time / Topological Horizon): Multi-hop propagation horizons k = 0, 1, ..., K-1
    - Horizontal Axis (Frequency / Semantic Channels): Informative feature frequency bands
    - Channel 0 (Low-Pass Consensus): A_norm^k * X (homophilic community consensus)
    - Channel 1 (High-Pass Difference): Delta A_norm^k * X (local boundary contrast / heterophily)
    - Channel 2 (Structural Resonance): Personalized PageRank / Heat diffusion (community trapping)

Processed natively by Time-Frequency 2D ConvNets (AST/PANNs style) with SpecAugment.
"""

import os
import time
import argparse
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from tqdm import tqdm

from dataset import load_dataset
from data_utils import load_fixed_splits, eval_acc, eval_rocauc
from logger import Logger


# =============================================================================
# 1. GRAPH SPECTROGRAM GENERATOR (Vectorized, Blazing Fast)
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
    using fast sparse matrix operations.

    Returns:
        spectrograms: Tensor of shape (num_nodes, 3, num_hops, num_bands) in [0.0, 1.0]
        band_indices: Array of selected feature indices
    """
    t0 = time.time()
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
    edge_index_loop, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    row, col = edge_index_loop[0], edge_index_loop[1]
    deg = torch.bincount(row, minlength=num_nodes).float()
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    val = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    adj_norm = torch.sparse_coo_tensor(edge_index_loop, val, (num_nodes, num_nodes)).to(device)

    # 3. Vectorized Multi-Horizon Propagation across all channels
    # Channel 0: Low-Pass Diffusion S_0[k, :] = A^k X
    # Channel 1: High-Pass Boundary Difference S_1[k, :] = A^{k-1} X - A^k X
    # Channel 2: Personalized PageRank S_2[k, :] = alpha * X + (1 - alpha) * A * P_{k-1}
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
        # Low-pass step
        x_next = torch.sparse.mm(adj_norm, x_k)
        s0_hops.append(x_next)
        # High-pass boundary gradient
        s1_hops.append(x_k - x_next)
        x_k = x_next

        # Structural resonance (PPR)
        p_next = ppr_alpha * x_sub + (1.0 - ppr_alpha) * torch.sparse.mm(adj_norm, p_k)
        s2_hops.append(p_next)
        p_k = p_next

    # Stack along hops: (num_hops, num_nodes, num_bands)
    c0 = torch.stack(s0_hops, dim=0).permute(1, 0, 2)  # (num_nodes, num_hops, num_bands)
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

    # Stack into 3-channel Spectrogram Tensor: (num_nodes, 3, num_hops, num_bands)
    spectrograms = torch.stack([c0, c1, c2], dim=1).cpu()

    elapsed = time.time() - t0
    size_mb = (spectrograms.element_size() * spectrograms.nelement()) / (1024 * 1024)
    print(f"-> Generated {num_nodes} Graph Spectrograms ({num_hops} hops x {num_bands} bands, 3 channels) in {elapsed:.3f}s ({size_mb:.1f} MB)")

    return spectrograms, band_indices


# =============================================================================
# 2. PIL HIGH-CONTRAST SPECTROGRAM VISUALIZER
# =============================================================================

def apply_spectrogram_colormap(matrix: np.ndarray, colormap: str = "magma") -> np.ndarray:
    """Map 2D float matrix [0, 1] to RGB uint8 image using authentic spectrogram colors."""
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


def save_spectrogram_sample_panel(
    spec_node: torch.Tensor,
    output_path: str,
    target_node_id: int = 0,
    label: int = None,
    dataset_name: str = "graph"
):
    """
    Renders a publication-grade 3-panel comparison image for a single node's spectrogram.
    spec_node: Tensor of shape (3, num_hops, num_bands)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    c0 = spec_node[0].numpy()  # Low-pass
    c1 = spec_node[1].numpy()  # High-pass
    c2 = spec_node[2].numpy()  # Resonance

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

    header = f"GRAPH SPECTROGRAM: NODE #{target_node_id} (CLASS {label}) ON [{dataset_name.upper()}]"
    draw.text((20, 15), header, fill=(255, 255, 255))
    draw.text((20, 35), "Channels: [Ch0: Multi-Hop Consensus (Magma) | Ch1: Boundary Gradient (Inferno) | Ch2: Resonance (Viridis)]", fill=(160, 175, 195))

    canvas.paste(im0, (20, 60))
    canvas.paste(im1, (target_w + 40, 60))
    canvas.paste(im2, (2 * target_w + 60, 60))

    draw.text((20, 65 + target_h), "Ch 0: Low-Pass Diffusion (A^k X)", fill=(255, 165, 0))
    draw.text((target_w + 40, 65 + target_h), "Ch 1: High-Pass Boundary Difference", fill=(255, 200, 50))
    draw.text((2 * target_w + 60, 65 + target_h), "Ch 2: Structural PageRank Resonance", fill=(80, 220, 140))

    canvas.save(output_path, "PNG")
    print(f"-> Saved crisp spectrogram visualization to: {output_path}")


# =============================================================================
# 3. TIME-FREQUENCY SPECTROGRAM VISION BACKBONE (AST / PANNs Style)
# =============================================================================

class SpecAugment(nn.Module):
    """SpecAugment adapted for Graph Spectrograms: random hop masking & frequency band masking."""
    def __init__(self, hop_mask_max: int = 2, freq_mask_max: int = 16):
        super().__init__()
        self.hop_mask_max = hop_mask_max
        self.freq_mask_max = freq_mask_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        b, c, h, w = x.shape
        # Hop masking
        if self.hop_mask_max > 0:
            h_len = torch.randint(1, self.hop_mask_max + 1, (1,)).item()
            h_start = torch.randint(0, max(1, h - h_len), (1,)).item()
            x = x.clone()
            x[:, :, h_start:h_start + h_len, :] = 0.0
        # Frequency band masking
        if self.freq_mask_max > 0 and w > self.freq_mask_max:
            f_len = torch.randint(1, self.freq_mask_max + 1, (1,)).item()
            f_start = torch.randint(0, max(1, w - f_len), (1,)).item()
            x = x.clone()
            x[:, :, :, f_start:f_start + f_len] = 0.0
        return x


class GraphSpectrogramNet(nn.Module):
    """
    Time-Frequency 2D ConvNet specifically optimized for Graph Spectrogram inputs.
    Uses rectangular kernels (capturing multi-hop temporal evolution across frequency bands)
    with residual connections and adaptive global spectral pooling.
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 8,
        node_feat_dim: int = 0,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        use_node_features: bool = True
    ):
        super().__init__()
        self.use_node_features = use_node_features and (node_feat_dim > 0)
        self.spec_augment = SpecAugment(hop_mask_max=2, freq_mask_max=16)

        # Stage 1: Local Time-Frequency Feature Extractor
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(3, 5), padding=(1, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Stage 2: Multi-Hop Co-occurrence Block
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 5), padding=(1, 2), bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))  # Downsample frequency bands, preserve hops

        # Stage 3: Deep ResNet-style Spectrogram Block
        self.conv3 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, hidden_dim, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(hidden_dim)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)

        # Raw node feature projection (optional fusion)
        if self.use_node_features:
            self.feat_mlp = nn.Sequential(
                nn.Linear(node_feat_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            classifier_in = hidden_dim * 2
        else:
            classifier_in = hidden_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, spec: torch.Tensor, raw_x: torch.Tensor = None) -> torch.Tensor:
        # spec: (B, C, H, W)
        x = self.spec_augment(spec)
        x = F.gelu(self.bn1(self.conv1(x)))

        res1 = x
        x = F.gelu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.gelu(self.bn3(self.conv3(x)))
        x = F.gelu(self.bn4(self.conv4(x)))

        v_emb = self.pool(x).flatten(1)  # (B, hidden_dim)
        v_emb = self.dropout(v_emb)

        if self.use_node_features and raw_x is not None:
            f_emb = self.feat_mlp(raw_x)
            fused = torch.cat([v_emb, f_emb], dim=1)
        else:
            fused = v_emb

        logits = self.classifier(fused)
        return logits


# =============================================================================
# 4. TRAINING & EVALUATION PIPELINE
# =============================================================================

@torch.no_grad()
def evaluate_spectrogram_model(model, spectrograms, node_feat, labels, split_idx, eval_func, args, device):
    model.eval()
    all_out = []
    n_nodes = spectrograms.size(0)
    batch_size = args.eval_batch_size

    for b_start in range(0, n_nodes, batch_size):
        b_idx = torch.arange(b_start, min(b_start + batch_size, n_nodes))
        b_spec = spectrograms[b_idx].to(device).float()
        b_feat = node_feat[b_idx].to(device) if (node_feat is not None and not args.no_node_features) else None
        logits = model(b_spec, b_feat)
        all_out.append(logits.cpu())

    out = torch.cat(all_out, dim=0).to(device)
    train_acc = eval_func(labels[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(labels[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(labels[split_idx['test']], out[split_idx['test']])
    return train_acc, valid_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description="Graph Spectrogram Learning Pipeline")
    parser.add_argument('--dataset', type=str, default='amazon-photo')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')

    # Spectrogram parameters
    parser.add_argument('--num_hops', type=int, default=8, help='diffusion horizon (spectrogram height)')
    parser.add_argument('--num_bands', type=int, default=128, help='number of informative feature bands (spectrogram width)')
    parser.add_argument('--channels', type=int, default=3, choices=[1, 3], help='spectrogram channels: 3 (Consensus + Diff + PPR) or 1 (Consensus)')
    parser.add_argument('--no_node_features', action='store_true', help='pure spectrogram mode (no MLP bypass)')

    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--cosine_lr', action='store_true')
    parser.add_argument('--save_sample', action='store_true', default=True, help='export sample PNG visualization')

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.device}" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    print(f"Executing on device: {device}")

    # Load dataset
    dataset = load_dataset(args.data_dir, args.dataset)
    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)
    labels = dataset.label.to(device)

    split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)
    split_idx = split_idx_lst[0]

    n = dataset.graph['num_nodes']
    e = dataset.graph['edge_index'].shape[1]
    c = max(labels.max().item() + 1, labels.shape[1])
    d = dataset.graph['node_feat'].shape[1] if dataset.graph['node_feat'] is not None else 0
    print(f"\nDataset: {args.dataset} | Nodes: {n} | Edges: {e} | Features: {d} | Classes: {c}")

    # Ensure undirected graph
    edge_index = to_undirected(dataset.graph['edge_index'])
    edge_index, _ = remove_self_loops(edge_index)

    # 1. Compute Full Graph Spectrograms
    print(f"\nComputing Graph Spectrograms for all {n} nodes...")
    spectrograms, band_indices = compute_graph_spectrograms(
        edge_index=edge_index,
        x=dataset.graph['node_feat'],
        num_nodes=n,
        num_hops=args.num_hops,
        num_bands=args.num_bands,
        device=device
    )

    if args.channels == 1:
        spectrograms = spectrograms[:, :1, :, :]

    # Save visual sample panel of Node 0
    if args.save_sample:
        out_sample = f"./results/graph_spectrogram_{args.dataset}_node0.png"
        save_spectrogram_sample_panel(
            spec_node=spectrograms[0],
            output_path=out_sample,
            target_node_id=0,
            label=int(labels[0].item()),
            dataset_name=args.dataset
        )

    # 2. Instantiate Spectrogram Vision Backbone
    model = GraphSpectrogramNet(
        in_channels=args.channels,
        num_classes=c,
        node_feat_dim=d,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_node_features=not args.no_node_features
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: GraphSpectrogramNet | Input: ({args.channels}, {args.num_hops}, {args.num_bands}) | Params: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    criterion = nn.CrossEntropyLoss()
    train_idx = split_idx['train']

    best_val = 0.0
    best_test = 0.0
    best_epoch = 0

    print(f"\n{'='*75}")
    print(f"Starting Graph Spectrogram Training on '{args.dataset}' for {args.epochs} epochs")
    print(f"{'='*75}")

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(train_idx))
        shuffled_train = train_idx[perm]
        total_loss = 0.0

        for b_start in range(0, len(shuffled_train), args.batch_size):
            b_idx = shuffled_train[b_start:b_start + args.batch_size]
            b_spec = spectrograms[b_idx].to(device).float()
            b_feat = dataset.graph['node_feat'][b_idx].to(device).float() if not args.no_node_features else None
            b_target = labels.squeeze(1)[b_idx].long()

            optimizer.zero_grad()
            logits = model(b_spec, b_feat)
            loss = criterion(logits, b_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(b_idx)

        if scheduler is not None:
            scheduler.step()

        # Evaluate every epoch
        train_acc, valid_acc, test_acc = evaluate_spectrogram_model(
            model=model,
            spectrograms=spectrograms,
            node_feat=dataset.graph['node_feat'],
            labels=labels,
            split_idx=split_idx,
            eval_func=eval_acc,
            args=args,
            device=device
        )

        if valid_acc > best_val:
            best_val = valid_acc
            best_test = test_acc
            best_epoch = epoch

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            lr_curr = optimizer.param_groups[0]['lr']
            print(f"Epoch: {epoch:03d} | LR: {lr_curr:.6f} | Loss: {total_loss / len(train_idx):.4f} | "
                  f"Train: {train_acc*100:.2f}% | Valid: {valid_acc*100:.2f}% | Test: {test_acc*100:.2f}% | "
                  f"Best Valid: {best_val*100:.2f}% | Best Test: {best_test*100:.2f}% (Ep {best_epoch})")

    total_time = time.time() - t_start
    print(f"\n{'='*75}")
    print(f"GRAPH SPECTROGRAM RESULTS ON [{args.dataset.upper()}]:")
    print(f"-> Best Validation Accuracy: {best_val*100:.2f}% (Epoch {best_epoch})")
    print(f"-> Test Accuracy at Best Val: {best_test*100:.2f}%")
    print(f"-> Training Duration: {total_time:.2f}s ({total_time/args.epochs:.3f}s/epoch)")
    print(f"{'='*75}\n")


if __name__ == '__main__':
    main()
