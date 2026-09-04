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

    def print_statistics(self, run=None, mode='max_acc'):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            argmin = result[:, 3].argmin().item()
            if mode == 'max_acc':
                ind = argmax
            else:
                ind = argmin
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}%')
            print(f'Highest Valid: {result[:, 1].max():.2f}%')
            print(f'Highest Test:  {result[:, 2].max():.2f}%')
            print(f'Chosen epoch:  {ind}')
            print(f'Final Train:   {result[ind, 0]:.2f}%')
            print(f'Final Test:    {result[ind, 2]:.2f}%')
            self.test = result[ind, 2]
        else:
            best_results = []
            for r in self.results:
                r = 100 * torch.tensor(r)
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                test1 = r[:, 2].max().item()
                if mode == 'max_acc':
                    train2 = r[r[:, 1].argmax(), 0].item()
                    test2 = r[r[:, 1].argmax(), 2].item()
                else:
                    train2 = r[r[:, 3].argmin(), 0].item()
                    test2 = r[r[:, 3].argmin(), 2].item()
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
    filename = f'results/{args.dataset}/{args.method}.csv'
    print(f"\nSaved benchmark results to {filename}")
    std_str = f"± {results.std():.2f}" if results.numel() > 1 else "± 0.00"
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(
            f"{args.method} {args.dropout} {args.lr} "
            f"{results.mean():.2f} {std_str}\n")
