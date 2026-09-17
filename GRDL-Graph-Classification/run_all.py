"""
Automated Multi-Dataset Benchmark Suite for Graph2Map.
Executes 10-Fold Cross-Validation across standard TUDataset benchmarks
and prints a clean comparison table against published NeurIPS 2024 / ICLR 2025 SOTA.
"""

import sys
import subprocess
import argparse
import time

PYTHON_EXE = sys.executable

BENCHMARK_TARGETS = {
    'MUTAG': {
        'epochs': 10,
        'batch': 32,
        'lr': 1e-3,
        'sota': '92.6% ± 4.2% (PATCHY-SAN) / 92.6% ± 4.3% (SAT) / 92.1% ± 5.9% (GRDL NeurIPS 2024)'
    },
    'PROTEINS': {
        'epochs': 10,
        'batch': 32,
        'lr': 1e-3,
        'sota': '82.6% ± 1.2% (GRDL NeurIPS 2024) / 80.0% ± 3.2% (WiTTopoPool) / 77.7% (SAT)'
    },
    'NCI1': {
        'epochs': 10,
        'batch': 64,
        'lr': 1e-3,
        'sota': '85.7% ± 0.8% (WWL) / 82.9% ± 2.1% (OT-GNN) / 82.5% (SAT) / 80.4% (GRDL)'
    },
    'IMDB-BINARY': {
        'epochs': 10,
        'batch': 32,
        'lr': 1e-3,
        'sota': '74.8% ± 2.0% (GRDL NeurIPS 2024) / 74.1% ± 0.6% (SEP) / 70.3% (Graphormer)'
    },
    'IMDB-MULTI': {
        'epochs': 10,
        'batch': 32,
        'lr': 1e-3,
        'sota': '53.1% ± 0.9% (GRDL-W) / 52.9% ± 1.8% (GRDL) / 52.6% (WWL)'
    },
    'PTC_MR': {
        'epochs': 10,
        'batch': 32,
        'lr': 1e-3,
        'sota': '71.4% ± 5.2% (Graphormer) / 70.2% ± 6.2% (GMT) / 68.3% (GRDL)'
    },
    'BZR': {
        'epochs': 10,
        'batch': 32,
        'lr': 1e-3,
        'sota': '92.0% ± 1.1% (GRDL NeurIPS 2024) / 91.7% ± 2.1% (SAT) / 87.6% (WWL)'
    },
    'COLLAB': {
        'epochs': 10,
        'batch': 64,
        'lr': 1e-3,
        'sota': '81.4% ± 2.1% (WWL) / 81.3% ± 0.2% (SEP) / 80.6% (SAT) / 79.8% (GRDL)'
    }
}


def run_dataset(dataset_name: str, epochs: int = None, batch: int = None, lr: float = None, layout: str = "spring", channels: int = None):
    cfg = BENCHMARK_TARGETS.get(dataset_name, {'epochs': 10, 'batch': 32, 'lr': 1e-3, 'sota': 'N/A'})
    ep = epochs if epochs is not None else cfg['epochs']
    bs = batch if batch is not None else cfg['batch']
    learning_rate = lr if lr is not None else cfg['lr']

    cmd = [
        PYTHON_EXE, "exp_graph2map.py",
        "-d", dataset_name,
        "-e", str(ep),
        "-b", str(bs),
        "--lr", str(learning_rate),
        "--layout", layout
    ]
    if channels is not None:
        cmd.extend(["-c", str(channels)])

    print("\n" + "=" * 80)
    print(f"[*] LAUNCHING: {' '.join(cmd)}")
    print(f"[*] Published Target to Beat: {cfg['sota']}")
    print("=" * 80)

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start
    print(f"[+] Completed {dataset_name} in {elapsed:.1f}s (Exit code: {result.returncode})\n")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Graph2Map Benchmark Master Runner")
    parser.add_argument("-d", "--dataset", type=str, default=None,
                        choices=list(BENCHMARK_TARGETS.keys()),
                        help="specific dataset to run (default: run all key benchmarks)")
    parser.add_argument("--all", action="store_true", help="run all 8 Table 1 benchmarks sequentially")
    parser.add_argument("--remaining", action="store_true", help="run the un-benchmarked datasets: BZR, PTC_MR, IMDB-MULTI, COLLAB")
    parser.add_argument("--featured", action="store_true", help="run all 5 featured benchmarks: MUTAG, BZR, PTC_MR, PROTEINS, NCI1")
    parser.add_argument("-e", "--epoch", type=int, default=None, help="override epochs (e.g. 200)")
    parser.add_argument("-b", "--batch", type=int, default=None, help="override batch size")
    parser.add_argument("-c", "--channels", type=int, default=7, choices=[2, 5, 7, 8], help="override channels (default: 7 for Feature-Manifold Atlas)")
    parser.add_argument("--layout", type=str, default="spring", choices=["spring", "spectral", "kamada_kawai"])
    args = parser.parse_args()

    ALL_SUITE = ['MUTAG', 'PROTEINS', 'IMDB-BINARY', 'BZR', 'PTC_MR', 'IMDB-MULTI', 'NCI1', 'COLLAB']
    FEATURED_SUITE = ['MUTAG', 'BZR', 'PTC_MR', 'PROTEINS', 'NCI1']
    REMAINING_SUITE = ['BZR', 'PTC_MR', 'IMDB-MULTI', 'COLLAB']

    if args.dataset:
        run_dataset(args.dataset, epochs=args.epoch, batch=args.batch, layout=args.layout, channels=args.channels)
    elif args.featured:
        print(f"[*] Running all 5 featured benchmarks: {FEATURED_SUITE} (Channels: {args.channels})")
        for ds in FEATURED_SUITE:
            rc = run_dataset(ds, epochs=args.epoch, batch=args.batch, layout=args.layout, channels=args.channels)
            if rc != 0:
                print(f"[!] Error occurred on {ds}, stopping suite.")
                break
    elif args.remaining:
        print(f"[*] Running remaining benchmarks: {REMAINING_SUITE}")
        for ds in REMAINING_SUITE:
            rc = run_dataset(ds, epochs=args.epoch, batch=args.batch, layout=args.layout, channels=args.channels)
            if rc != 0:
                print(f"[!] Error occurred on {ds}, stopping suite.")
                break
    elif args.all:
        print(f"[*] Running full Table 1 suite: {ALL_SUITE}")
        for ds in ALL_SUITE:
            rc = run_dataset(ds, epochs=args.epoch, batch=args.batch, layout=args.layout, channels=args.channels)
            if rc != 0:
                print(f"[!] Error occurred on {ds}, stopping suite.")
                break
    else:
        # Default quick duo
        print(f"[*] Running quick duo: ['MUTAG', 'PROTEINS']")
        for ds in ['MUTAG', 'PROTEINS']:
            run_dataset(ds, epochs=args.epoch, batch=args.batch, layout=args.layout, channels=args.channels)


if __name__ == "__main__":
    main()
