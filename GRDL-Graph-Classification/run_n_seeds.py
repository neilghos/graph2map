"""
Multi-Seed (N-Seeds) 10-Fold Cross-Validation Benchmark Runner for Graph2Map.
Evaluates stability, statistical variance, and grand mean across multiple random seeds.
Matches NeurIPS 2024 / ICLR 2025 multi-run evaluation standards on TUDatasets.
"""

import os
import sys
import time
import argparse
import random
import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader

from exp_util import load_dataset
from graph_level_rasterizer import graph_to_map
from exp_graph2map import Graph2MapResNet, train_and_eval_fold, get_or_create_rasterized_maps

# Standard published Table 1 baselines from NeurIPS 2024 GRDL paper
PUBLISHED_SOTA = {
    'MUTAG': {'GIN': '89.4% ± 5.6%', 'Graphormer': '89.6% ± 6.2%', 'SAT': '92.6% ± 4.3%', 'PATCHY-SAN': '92.6% ± 4.2%', 'GRDL': '92.1% ± 5.9%'},
    'PROTEINS': {'GIN': '76.2% ± 2.8%', 'Graphormer': '76.3% ± 2.7%', 'SAT': '77.7% ± 3.2%', 'WiTTopoPool': '80.0% ± 3.2%', 'GRDL': '82.6% ± 1.2%'},
    'NCI1': {'GRDL': '80.4% ± 0.8%', 'GIN': '82.2% ± 0.8%', 'SAT': '82.5% ± 0.8%', 'OT-GNN': '82.9% ± 2.1%', 'WWL': '85.7% ± 0.8%'},
    'IMDB-BINARY': {'GIN': '64.3% ± 3.1%', 'Graphormer': '70.3% ± 0.9%', 'WWL': '71.6% ± 3.8%', 'SEP': '74.1% ± 0.6%', 'GRDL': '74.8% ± 2.0%'},
    'IMDB-MULTI': {'Graphormer': '48.9% ± 2.0%', 'GIN': '50.9% ± 1.7%', 'WWL': '52.6% ± 3.0%', 'WiTTopoPool': '52.9% ± 0.8%', 'GRDL': '52.9% ± 1.8%', 'GRDL-W': '53.1% ± 0.9%'},
    'PTC_MR': {'GIN': '64.6% ± 7.0%', 'OT-GNN': '68.0% ± 7.5%', 'GRDL': '68.3% ± 5.4%', 'GMT': '70.2% ± 6.2%', 'Graphormer': '71.4% ± 5.2%'},
    'BZR': {'GIN': '82.6% ± 3.5%', 'Graphormer': '85.3% ± 2.3%', 'WWL': '87.6% ± 0.6%', 'SAT': '91.7% ± 2.1%', 'GRDL': '92.0% ± 1.1%'},
    'COLLAB': {'GIN': '79.3% ± 1.7%', 'GRDL': '79.8% ± 0.9%', 'Graphormer': '80.3% ± 1.3%', 'SAT': '80.6% ± 0.6%', 'MinCutPool': '80.9% ± 0.3%', 'SEP': '81.3% ± 0.2%', 'WWL': '81.4% ± 2.1%'}
}

DEFAULT_SEEDS = [123, 456, 789, 42, 2024, 1007, 2025, 999, 1337, 777]


def run_benchmark_for_dataset(dataset_name: str, seeds: list, epochs: int = 100, batch_size: int = 32, lr: float = 1e-3, layout: str = "spring", resolution: int = 64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"[*] BENCHMARKING: {dataset_name} across {len(seeds)} seeds (Seeds: {seeds})")
    print(f"[*] Configuration: Res {resolution}x{resolution} | Layout: {layout} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print("=" * 80)

    # Load dataset using standard loader
    dataset = load_dataset(dataset_name, seed=seeds[0])
    X, Y = get_or_create_rasterized_maps(dataset, dataset_name, resolution=resolution, layout=layout)
    num_classes = len(torch.unique(Y))
    num_samples = len(Y)

    all_seed_results = []
    seed_means = []

    grand_start_t = time.time()

    for s_idx, current_seed in enumerate(seeds):
        seed_start_t = time.time()
        print(f"\n>>> [SEED {s_idx + 1}/{len(seeds)}] Seed Value: {current_seed}")
        random.seed(current_seed)
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(current_seed)

        kfold = KFold(n_splits=10, shuffle=True, random_state=current_seed)
        fold_accs = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            fold_num = fold + 1
            train_ds = TensorDataset(X[train_idx], Y[train_idx])
            val_ds = TensorDataset(X[val_idx], Y[val_idx])

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

            model = Graph2MapResNet(in_channels=5, num_classes=num_classes).to(device)
            best_acc, best_ep = train_and_eval_fold(
                fold=fold_num,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=epochs,
                lr=lr,
                weight_decay=1e-4
            )
            fold_accs.append(best_acc * 100.0)

        seed_arr = np.array(fold_accs)
        s_mean = seed_arr.mean()
        s_std = seed_arr.std()
        seed_means.append(s_mean)
        all_seed_results.extend(fold_accs)

        elapsed = time.time() - seed_start_t
        print(f"    Seed {current_seed} Result -> Mean: {s_mean:.2f}% ± {s_std:.2f}% | Time: {elapsed:.1f}s")

    all_folds_arr = np.array(all_seed_results)
    grand_mean = all_folds_arr.mean()
    grand_std = all_folds_arr.std()
    seed_means_arr = np.array(seed_means)
    total_time = time.time() - grand_start_t

    print("\n" + "=" * 80)
    print(f"GRAPH2MAP MULTI-SEED FINAL RESULTS: {dataset_name}")
    print(f"  Grand Mean (Across all {len(all_folds_arr)} folds):  {grand_mean:.2f}% ± {grand_std:.2f}%")
    print(f"  Mean of Seed Averages (Across {len(seeds)} seeds):   {seed_means_arr.mean():.2f}% ± {seed_means_arr.std():.2f}%")
    print(f"  Per-Seed Averages: {[round(m, 2) for m in seed_means]}")
    print(f"  Total Benchmark Time: {total_time:.1f}s")
    print("=" * 80)

    if dataset_name in PUBLISHED_SOTA:
        print("\nHead-to-Head Comparison with Table 1 SOTA:")
        for baseline, score in PUBLISHED_SOTA[dataset_name].items():
            print(f"  • {baseline:30s}: {score}")
        print(f"  • {'Graph2Map (Ours - Multi-Seed)':30s}: {grand_mean:.2f}% ± {grand_std:.2f}%")
        print("=" * 80)

    # Save to disk
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "n_seeds_results.csv")
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", encoding="utf-8") as f:
        if write_header:
            f.write("dataset,num_seeds,grand_mean,grand_std,seed_means_std,seeds,epochs,batch_size,lr,time_sec\n")
        f.write(f"{dataset_name},{len(seeds)},{grand_mean:.2f},{grand_std:.2f},{seed_means_arr.std():.2f},\"{seeds}\",{epochs},{batch_size},{lr},{total_time:.1f}\n")
    print(f"[+] Saved multi-seed results to {out_csv}\n")

    return grand_mean, grand_std


def main():
    parser = argparse.ArgumentParser(description="Multi-Seed 10-Fold Benchmark Runner for Graph2Map")
    parser.add_argument("-d", "--dataset", type=str, default="MUTAG",
                        choices=['MUTAG', 'PROTEINS', 'PTC_MR', 'NCI1', 'IMDB-BINARY', 'IMDB-MULTI', 'BZR', 'COLLAB', 'all'])
    parser.add_argument("-n", "--num_seeds", type=int, default=5,
                        help="number of random seeds to evaluate (default: 5)")
    parser.add_argument("-e", "--epoch", type=int, default=100, help="epochs per fold (default: 100)")
    parser.add_argument("-b", "--batch", type=int, default=32, help="batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate (default: 1e-3)")
    parser.add_argument("--res", type=int, default=64, help="canvas resolution (default: 64)")
    parser.add_argument("--layout", type=str, default="spring", choices=["spring", "spectral", "kamada_kawai"])
    args = parser.parse_args()

    selected_seeds = DEFAULT_SEEDS[:args.num_seeds]

    # Dataset-specific optimal batch sizing
    batch_map = {'NCI1': 64, 'COLLAB': 64}

    if args.dataset == 'all':
        suite = ['MUTAG', 'PROTEINS', 'IMDB-BINARY', 'BZR', 'PTC_MR', 'IMDB-MULTI', 'NCI1', 'COLLAB']
        for ds in suite:
            bs = batch_map.get(ds, args.batch)
            run_benchmark_for_dataset(
                dataset_name=ds,
                seeds=selected_seeds,
                epochs=args.epoch,
                batch_size=bs,
                lr=args.lr,
                layout=args.layout,
                resolution=args.res
            )
    else:
        bs = batch_map.get(args.dataset, args.batch)
        run_benchmark_for_dataset(
            dataset_name=args.dataset,
            seeds=selected_seeds,
            epochs=args.epoch,
            batch_size=bs,
            lr=args.lr,
            layout=args.layout,
            resolution=args.res
        )


if __name__ == "__main__":
    main()
