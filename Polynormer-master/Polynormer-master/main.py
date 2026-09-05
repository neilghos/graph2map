import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops

from logger import *
from dataset import load_dataset
from data_utils import eval_acc, eval_rocauc, load_fixed_splits
from eval import *
from parse import parse_method, parser_add_main_args
from node_level_rasterizer import node_to_ego_map, save_ego_map_panel
from extractor import get_or_create_spectrogram_cache, get_or_create_atlas_cache, get_or_create_10ch_atlas_cache


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_or_create_raster_cache(dataset, edge_index_cpu, x_cpu, args):
    """
    Check for pre-rasterized multi-channel Ego-Maps cache on disk.
    If missing, rasterizes all nodes using bounded ego-subgraphs and caches as uint8.
    """
    n = dataset.graph['num_nodes']
    cache_dir = os.path.join(args.data_dir, "ego_cache")
    os.makedirs(cache_dir, exist_ok=True)
    ch_tag = f"ch{args.channels}_topo"
    bg_tag = "darkbg" if getattr(args, 'dark_bg', False) else "whitebg"
    cache_file = os.path.join(
        cache_dir,
        f"{args.dataset}_res{args.resolution}_hops{args.num_hops}_max{args.max_nodes}_{args.layout_method}_{ch_tag}_{bg_tag}.pt"
    )

    if os.path.exists(cache_file):
        print(f"Loading pre-computed Ego-Map cache from: {cache_file} ...")
        t0 = time.time()
        cached_maps = torch.load(cache_file)
        print(f"Loaded {cached_maps.shape[0]} Ego-Maps in {time.time() - t0:.2f}s | Tensor: {list(cached_maps.shape)} ({cached_maps.dtype})")
        return cached_maps

    print(f"\n{'='*75}")
    print(f"Pre-rasterizing {n} multi-channel ({args.channels}ch) Ego-Maps for '{args.dataset}'")
    print(f"Channels: {args.channels} | White-Anchored: {not getattr(args, 'dark_bg', False)} | Resolution: {args.resolution}x{args.resolution} | Hops: {args.num_hops}")
    print(f"Cache will be saved to: {cache_file}")
    print(f"{'='*75}")

    cached_maps = torch.empty((n, args.channels, args.resolution, args.resolution), dtype=torch.uint8)
    t0 = time.time()

    include_canonical = (args.channels == 8)
    white_bg = not getattr(args, 'dark_bg', False)

    pbar = tqdm(range(n), desc=f"Rasterizing [{args.dataset}]", unit="node", dynamic_ncols=True)
    for i in pbar:
        map_tensor = node_to_ego_map(
            target_node=i,
            edge_index=edge_index_cpu,
            x=x_cpu,
            num_hops=args.num_hops,
            resolution=args.resolution,
            sigma_node=0.05,
            sigma_edge=0.02,
            max_nodes=args.max_nodes,
            layout_method=args.layout_method,
            include_canonical_matrices=include_canonical,
            white_bg=white_bg,
            seed=args.seed + i
        )
        # Store as uint8 (0-255) to reduce RAM / disk size by 4x
        cached_maps[i] = (map_tensor * 255.0).clamp(0, 255).to(torch.uint8)

    # Save to disk
    torch.save(cached_maps, cache_file)
    size_mb = (cached_maps.element_size() * cached_maps.nelement()) / (1024 * 1024)
    print(f"\nRasterization caching complete in {time.time() - t0:.2f}s! Saved {size_mb:.1f} MB to: {cache_file}\n")

    # Save a publication-grade sample visualization panel of Node 0
    try:
        results_dir = "./results"
        os.makedirs(results_dir, exist_ok=True)
        sample_path = os.path.join(results_dir, f"{args.dataset}_ego_map_sample.png")
        sample_tensor = (cached_maps[0].float()) / 255.0
        label_val = int(dataset.label[0].item()) if hasattr(dataset, 'label') else None
        save_ego_map_panel(sample_tensor, sample_path, target_node_id=0, label=label_val, split_name="sample")
    except Exception as e:
        print(f"Note: Could not save sample panel: {e}")

    return cached_maps


@torch.no_grad()
def predict_all_nodes(model, cached_maps, node_feat, args, device):
    """
    Compute node-level predictions in mini-batches of nodes (e.g. 256 nodes per batch)
    to get class logits for every node in the dataset.
    """
    model.eval()
    all_out = []
    n_nodes = cached_maps.size(0)
    batch_size = args.eval_batch_size

    for b_start in range(0, n_nodes, batch_size):
        b_idx = torch.arange(b_start, min(b_start + batch_size, n_nodes))
        b_maps = (cached_maps[b_idx].to(device).float()) / 255.0
        b_feats = node_feat[b_idx] if (node_feat is not None and not args.no_node_features) else None
        logits = model(b_maps, b_feats)
        all_out.append(logits.cpu())

    return torch.cat(all_out, dim=0).to(device)


def main():
    ### Parse args ###
    parser = argparse.ArgumentParser(description='Graph2Map Training Pipeline for Node Classification')
    parser_add_main_args(parser)
    args = parser.parse_args()
    print(args)

    fix_seed(args.seed)

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### Load dataset via Polynormer dataset loader ###
    dataset = load_dataset(args.data_dir, args.dataset)

    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)
    dataset.label = dataset.label.to(device)

    split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)

    ### Dataset information ###
    n = dataset.graph['num_nodes']
    e = dataset.graph['edge_index'].shape[1]
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1] if dataset.graph['node_feat'] is not None else 0

    print(f"Dataset: {args.dataset} | Nodes: {n} | Directed Edges: {e} | Undirected: {e//2} | Feats: {d} | Classes: {c}")

    # Ensure undirected and standard graph representation on CPU for rasterization
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
    dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])

    edge_index_cpu = dataset.graph['edge_index'].cpu()
    x_cpu = dataset.graph['node_feat'].cpu() if dataset.graph['node_feat'] is not None else None

    # Pre-rasterize / load representation cache based on --representation flag
    if getattr(args, 'representation', 'spectrogram') == 'spectrogram':
        cached_maps = get_or_create_spectrogram_cache(dataset, edge_index_cpu, x_cpu, args)
    elif getattr(args, 'representation', 'spectrogram') in ('atlas', 'atlas_10ch', 'atlas_7ch'):
        cached_maps = get_or_create_10ch_atlas_cache(dataset, edge_index_cpu, x_cpu, args)
    else:
        cached_maps = get_or_create_raster_cache(dataset, edge_index_cpu, x_cpu, args)

    # Move node features to device for training and apply polynomial graph smoothing if enabled
    if dataset.graph['node_feat'] is not None:
        if not getattr(args, 'no_smooth_features', False) and args.beta > 0 and getattr(args, 'representation', 'spectrogram') == 'ego_map':
            print(f"Applying Polynomial Graph Feature Smoothing (beta={args.beta}) ...")
            edge_index = dataset.graph['edge_index']
            num_nodes = dataset.graph['num_nodes']
            x = dataset.graph['node_feat']

            # Symmetric normalized adjacency: D^{-1/2} (A + I) D^{-1/2}
            edge_index_loop, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            row, col = edge_index_loop[0], edge_index_loop[1]
            deg = torch.bincount(row, minlength=num_nodes).float()
            deg_inv_sqrt = torch.pow(deg, -0.5)
            val = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            adj_norm = torch.sparse_coo_tensor(edge_index_loop, val, (num_nodes, num_nodes))

            # Multi-step Polynomial Diffusion across local_layers (matching Polynormer's local receptive field)
            num_steps = getattr(args, 'local_layers', 7)
            print(f"Applying {num_steps}-step Polynomial Graph Feature Diffusion (beta={args.beta}) ...")
            x_curr = x.clone()
            for step in range(num_steps):
                x_curr = (1.0 - args.beta) * x + args.beta * torch.sparse.mm(adj_norm, x_curr)
            dataset.graph['node_feat'] = x_curr
            print(f"-> Node features regularized with {num_steps}-step polynomial neighborhood consensus!")

        dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)

    ### Instantiate Graph2Map Vision Classifier ###
    model = parse_method(args, n, c, d, device)

    ### Loss function (Single-class, Multi-class) ###
    if args.dataset in ('questions'):
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.NLLLoss()

    ### Performance metric (Acc, AUC) ###
    if args.metric == 'rocauc':
        eval_func = eval_rocauc
    else:
        eval_func = eval_acc

    logger = Logger(args.runs, args)
    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in model.backbone.parameters()) if hasattr(model, 'backbone') else total_params
    print('\nMODEL ARCHITECTURE:', model)
    print(f"-> Model Parameters: Total = {total_params:,} | Backbone ({args.backbone}) = {backbone_params:,}\n")

    total_epochs = args.epochs if args.epochs > 0 else (args.local_epochs + args.global_epochs)

    ### Training loop ###
    for run in range(args.runs):
        if args.dataset in ('coauthor-cs', 'coauthor-physics', 'amazon-computer', 'amazon-photo'):
            split_idx = split_idx_lst[0]
        else:
            split_idx = split_idx_lst[run]

        train_idx = split_idx['train']
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
        scheduler = None
        if args.cosine_lr:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_epochs, eta_min=args.lr * 0.01
            )
        best_val = float('-inf')
        best_test = float('-inf')

        epoch_pbar = tqdm(range(total_epochs), desc=f"Run {run + 1}/{args.runs}", unit="epoch", dynamic_ncols=True)
        for epoch in epoch_pbar:
            model.train()
            perm = torch.randperm(len(train_idx))
            shuffled_train = train_idx[perm]
            total_loss = 0.0

            # Mini-batch training over target nodes
            batch_pbar = tqdm(
                range(0, len(shuffled_train), args.batch_size),
                desc=f"  Epoch {epoch:02d}",
                leave=False,
                dynamic_ncols=True
            )
            for b_start in batch_pbar:
                b_idx = shuffled_train[b_start:b_start + args.batch_size]
                b_maps = (cached_maps[b_idx].to(device).float()) / 255.0

                # Data Augmentation (D4 dihedral group rotation & flips around center node for Ego-Maps)
                if args.augment and getattr(args, 'representation', 'spectrogram') == 'ego_map':
                    k = torch.randint(0, 4, (1,)).item()
                    if k > 0:
                        b_maps = torch.rot90(b_maps, k=k, dims=(-2, -1))
                    if torch.rand(1).item() > 0.5:
                        b_maps = torch.flip(b_maps, dims=[-1])
                    if torch.rand(1).item() > 0.5:
                        b_maps = torch.flip(b_maps, dims=[-2])

                b_feats = dataset.graph['node_feat'][b_idx] if not args.no_node_features else None
                b_targets = dataset.label.squeeze(1)[b_idx.to(device)]

                optimizer.zero_grad()
                out_batch = model(b_maps, b_feats)

                if args.dataset in ('questions'):
                    if dataset.label.shape[1] == 1:
                        true_label = F.one_hot(b_targets, dataset.label.max() + 1).squeeze(1)
                    else:
                        true_label = b_targets
                    loss = criterion(out_batch, true_label.to(torch.float))
                else:
                    log_probs = F.log_softmax(out_batch, dim=1)
                    loss = criterion(log_probs, b_targets)

                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(b_idx)
                batch_pbar.set_postfix(batch_loss=f"{loss.item():.4f}")

            if scheduler is not None:
                scheduler.step()

            epoch_loss = total_loss / len(train_idx)

            # Evaluate node predictions across train, valid, test splits using eval.py
            out_full = predict_all_nodes(model, cached_maps, dataset.graph['node_feat'], args, device)
            result = evaluate(model, dataset, split_idx, eval_func, criterion, args, result=out_full)

            logger.add_result(run, result[:-1])

            if result[1] > best_val:
                best_val = result[1]
                best_test = result[2]
                if args.save_model:
                    save_model(args, model, optimizer, run)

            # Update epoch progress bar status
            epoch_pbar.set_postfix({
                'loss': f"{epoch_loss:.4f}",
                'val_acc': f"{100 * result[1]:.2f}%",
                'test_acc': f"{100 * result[2]:.2f}%",
                'best_test': f"{100 * best_test:.2f}%"
            })

            if epoch % args.display_step == 0:
                tqdm.write(
                    f'Epoch: {epoch:02d} | '
                    f'Loss: {epoch_loss:.4f} | '
                    f'Train: {100 * result[0]:.2f}% | '
                    f'Valid: {100 * result[1]:.2f}% | '
                    f'Test: {100 * result[2]:.2f}% | '
                    f'Best Valid: {100 * best_val:.2f}% | '
                    f'Best Test: {100 * best_test:.2f}%'
                )

        logger.print_statistics(run)

    results = logger.print_statistics()
    ### Save results ###
    save_result(args, results)


if __name__ == "__main__":
    main()
