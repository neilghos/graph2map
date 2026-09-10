import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="10-Seed Benchmark Runner for Graph2Map")
    parser.add_argument('--dataset', type=str, default='amazon-photo')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=250)
    args, unknown = parser.parse_known_args()

    save_filename = f"{args.dataset}_{args.runs}seeds.csv"

    cmd = [
        sys.executable, "main.py",
        "--dataset", args.dataset,
        "--runs", str(args.runs),
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--augment",
        "--save_filename", save_filename,
        "--save_result"
    ] + [arg for arg in unknown if arg != "--augment"]

    print(f"Launching {args.runs}-seed run. Results will be saved to results/{args.dataset}/{save_filename}")
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
