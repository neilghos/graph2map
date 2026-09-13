import warnings
import os
import torch

warnings.filterwarnings("ignore")


class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 4
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None, mode='max_acc', burn_in_ratio=0.0, last_epochs=10):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            if last_epochs > 0 and len(result) >= last_epochs:
                start_idx = len(result) - last_epochs
                tail_str = f' (Selected from final {last_epochs} annealed epochs: {start_idx}..{len(result)-1})'
            else:
                start_idx = int(len(result) * burn_in_ratio)
                tail_str = f' (Burn-in: >= epoch {start_idx})' if start_idx > 0 else ''

            if mode == 'max_acc':
                ind = start_idx + result[start_idx:, 1].argmax().item()
            else:
                ind = start_idx + result[start_idx:, 3].argmin().item()

            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}%')
            print(f'Highest Valid: {result[:, 1].max():.2f}% (Annealed Tail Best: {result[start_idx:, 1].max():.2f}%)')
            print(f'Highest Test:  {result[:, 2].max():.2f}%')
            print(f'Chosen epoch:  {ind}{tail_str}')
            print(f'Final Train:   {result[ind, 0]:.2f}%')
            print(f'Final Test:    {result[ind, 2]:.2f}%')
            self.test = result[ind, 2]
        else:
            best_results = []
            for r in self.results:
                r = 100 * torch.tensor(r)
                if last_epochs > 0 and len(r) >= last_epochs:
                    start_idx = len(r) - last_epochs
                else:
                    start_idx = int(len(r) * burn_in_ratio) if burn_in_ratio > 0 else 0

                train1 = r[:, 0].max().item()
                valid = r[start_idx:, 1].max().item()
                test1 = r[:, 2].max().item()
                if mode == 'max_acc':
                    best_idx = start_idx + r[start_idx:, 1].argmax().item()
                    train2 = r[best_idx, 0].item()
                    test2 = r[best_idx, 2].item()
                else:
                    best_idx = start_idx + r[start_idx:, 3].argmin().item()
                    train2 = r[best_idx, 0].item()
                    test2 = r[best_idx, 2].item()
                best_results.append((train1, test1, valid, train2, test2))

            best_result = torch.tensor(best_results)

            print(f'\nSummary Across All Runs:')
            def fmt(col):
                r = best_result[:, col]
                if r.numel() > 1:
                    return f"{r.mean():.2f}% ± {r.std():.2f}%"
                return f"{r.mean():.2f}%"

            print(f'  Highest Train: {fmt(0)}')
            print(f'  Highest Test:  {fmt(1)}')
            print(f'  Highest Valid: {fmt(2)}')
            print(f'  Final Train:   {fmt(3)}')
            print(f'  Final Test:    {fmt(4)}')

            self.test = best_result[:, 4].mean()
            return best_result[:, 4]

    def output(self, out_path, info):
        with open(out_path, 'a') as f:
            f.write(info)
            f.write(f'test acc:{self.test}\n')


def save_model(args, model, optimizer, run):
    if not os.path.exists(f'models/{args.dataset}'):
        os.makedirs(f'models/{args.dataset}')
    model_path = f'models/{args.dataset}/{args.method}_{run}.pt'
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, model_path)


def load_model(args, model, optimizer, run):
    model_path = f'models/{args.dataset}/{args.method}_{run}.pt'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer


def save_result(args, results):
    if not os.path.exists(f'results/{args.dataset}'):
        os.makedirs(f'results/{args.dataset}')

    if getattr(args, 'save_filename', None):
        filename = f'results/{args.dataset}/{args.save_filename}'
    elif getattr(args, 'runs', 1) > 1:
        filename = f'results/{args.dataset}/{args.dataset}_{args.runs}seeds.csv'
    else:
        filename = f'results/{args.dataset}/{args.method}.csv'

    print(f"\nSaved benchmark results to {filename}")
    std_str = f"± {results.std():.2f}" if results.numel() > 1 else "± 0.00"
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(
            f"{args.method} {args.dropout} {args.lr} "
            f"{results.mean():.2f} {std_str}\n")
