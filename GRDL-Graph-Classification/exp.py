import argparse
import torch
import numpy as np
import random
from exp_util import *
from model import *
from torch.optim.lr_scheduler import ExponentialLR

parser = argparse.ArgumentParser(
    prog="MMD_GNN",
    description="Experiments of MMD_GNN"
)
parser.add_argument("-d", "--dataset",
                    choices=['MUTAG', 'PROTEINS', 'PTC_MR', 'NCI1', 'IMDB-BINARY', 'IMDB-MULTI', 'BZR', 'COLLAB'],
                    default="MUTAG")
parser.add_argument("-b", "--batch",
                    default=32,
                    type=int)
parser.add_argument("-e", "--epoch",
                    default=100,
                    type=int)
parser.add_argument("-s", "--seed",
                    default=123,
                    type=int)
parser.add_argument("-ns", "--num_supp",
                    default='G3')
parser.add_argument("-ls", "--lr_schedule",
                    default=0.95,
                    type=float)
parser.add_argument("-nl", "--num_layer",
                    default=5,
                    type=int)
parser.add_argument("-nl_mlp", "--num_layer_mlp",
                    default=2,
                    type=int)
parser.add_argument("-hc", "--hidden_channels",
                    default=32,
                    type=int)
parser.add_argument("-lr1", "--lr1",
                    default=1e-3,
                    type=float)
parser.add_argument("-lr2", "--lr2",
                    default=1e-1,
                    type=float)
parser.add_argument("-lam", "--lam",
                    default=1,
                    type=float)
args = parser.parse_args()
print("#"*100)
print(args)

# Set random seed
np.random.seed(seed=args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

class MMD_Trainer(Classifier_Trainer):
    def __init__(self, dataset, k_fold=10, batch_size=32, num_epoch=100, hyper_param_model=..., hyper_param_optimizer=..., root_directory=None) -> None:
        super().__init__(dataset, k_fold, batch_size, num_epoch, hyper_param_model, hyper_param_optimizer, root_directory)

    def config_optimizer(self, hyper_param):
        optim = torch.optim.Adam([
            {'params': self.model.extractor.parameters()},
            {'params': self.model.mmd.atoms},
            {'params': self.model.mmd.gamma, 'lr': args.lr2}
        ], lr=hyper_param['lr1'])
        scheduler = ExponentialLR(optim, gamma=args.lr_schedule, verbose=False)
        return optim, scheduler
    
    def config_model(self, hyper_param):
        extractor_param = hyper_param['extractor']
        mmd_param = hyper_param['mmd']
        extractor = GINExtractor(extractor_param['input_channels'],
                                 extractor_param['hidden_channels'],
                                 extractor_param['num_layer_mlp'],
                                 extractor_param['num_layer_gin'],
                                 extractor_param['jump_mode'])
        # extractor.apply(init_weights)
        if extractor_param['jump_mode'] == 'cat':
            mmd_param['input_channels'] = extractor_param['num_layer_gin'] * extractor_param['hidden_channels']
        model = GRDL(extractor,
                       mmd_param['input_channels'],
                       mmd_param['num_atoms'],
                       mmd_param['num_atom_supp'],
                       mmd_param['gamma'])
        model.init_atoms(self.train_dataset)
        return model
    
    def feed_forward(self, data):
        return self.model(data)
    
    def criterion(self, out, data, weight = None):
        # weight = torch.tensor([1568/27509, 25941/27509], device=device)
        if weight is None:
            loss1 = torch.sum(F.one_hot(data.y, dataset.num_classes) * (-torch.log(out)), dim=1).mean()
        else:
            loss1 = torch.sum(F.one_hot(data.y, dataset.num_classes) * (-torch.log(out)*weight), dim=1).mean()
        loss2 = self.model.mmd.discriminate_loss()
        loss = loss1 + args.lam * loss2
        return loss

# Get the dataset
dataset = load_dataset(args.dataset, args.seed)
dataset_info(dataset)
k_fold = 10
split = int(len(dataset)*((k_fold-1)/k_fold))
if args.dataset in ['MUTAG', 'PTC_MR']:
    train_dataset = dataset
else:
    train_dataset = dataset[:split]
    test_dataset = dataset[split:]
    print(f'Number of testing graphs: {len(test_dataset)}')
print(f'Number of training graphs: {len(train_dataset)}')
num_nodes = [graph.num_nodes for graph in dataset]
supp_min = int(np.min(num_nodes))
supp_med = int(np.median(num_nodes))
supp_max = int(np.max(num_nodes))
if args.num_supp == "G1":
    num_supp = int(supp_min)
elif args.num_supp == "G2":
    num_supp = int((supp_min + supp_med)/2)
elif args.num_supp == "G3":
    num_supp = int(supp_med)
elif args.num_supp == "G4":
    num_supp = int((supp_max + supp_med)/2)
elif args.num_supp == "G5":
    num_supp = int(supp_max)
else:
    num_supp = int(args.num_supp)

gin_param = {'input_channels': dataset.num_node_features,
             'hidden_channels': args.hidden_channels,
             'num_layer_mlp': args.num_layer_mlp,
             'num_layer_gin': args.num_layer,
             'jump_mode': "max"}
mmd_param = {'input_channels': args.hidden_channels,
             'num_atoms': dataset.num_classes,
             'num_atom_supp': num_supp,
             'gamma': 500.0}
model_param = {'extractor': gin_param,
               'mmd': mmd_param}
optim_param = {'lr1':args.lr1}

trainer = MMD_Trainer(train_dataset, k_fold, args.batch, args.epoch, model_param, optim_param, root_directory='check_point')
trainer.train()
acc = np.array(trainer.valid_acc).max(axis=1)
print(f"Validation Classification Accuracy: MEAN = {acc.mean():.4f} STD = {acc.std():.4f}")

# Record the results
result_path = "./Log_Info/result.out"
if args.dataset in ['MUTAG', 'PTC_MR']:
    with open(result_path, "a") as f:
        f.write(f"{args}\n")
        f.write(f"Classification Accuracy: MEAN = {acc.mean():.4f} STD = {acc.std():.4f}\n")
        f.close()
else:
    save_directory = f"check_point/{dataset.name}/"
    acc = []
    for i in range(10):
        model_path = f"model_{i+1}.pt"
        extractor = GINExtractor(gin_param['input_channels'],
                                gin_param['hidden_channels'],
                                gin_param['num_layer_mlp'],
                                gin_param['num_layer_gin'],
                                gin_param['jump_mode'])
        model = GRDL(extractor,
                        mmd_param['input_channels'],
                        mmd_param['num_atoms'],
                        mmd_param['num_atom_supp'],
                        mmd_param['gamma'])
        model.load_state_dict(torch.load(os.path.join(save_directory, model_path), map_location='cpu'))
        acc.append(evaluate(model, test_dataset))
        print(model_path, acc[-1])
    acc = np.array(acc)
    print(f"Classification Accuracy: MEAN = {acc.mean():.4f} STD = {acc.std():.4f}")

    with open(result_path, "a") as f:
        f.write(f"{args}\n")
        f.write(f"Classification Accuracy: MEAN = {acc.mean():.4f} STD = {acc.std():.4f}\n")
        f.close()