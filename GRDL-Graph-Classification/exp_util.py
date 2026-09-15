from torch_geometric.loader import DataLoader
import torch
import numpy as np
from sklearn.model_selection import KFold
import os
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import torch.nn.functional as F
import torch_geometric.transforms as T
import random
from sklearn.metrics import roc_auc_score

device = "cuda" if torch.cuda.is_available() else "cpu"
manual_seed = 1234

class Classifier_Trainer:
    def __init__(self,
                 dataset,
                 k_fold=10,
                 batch_size=32,
                 num_epoch=100,
                 hyper_param_model={'hidden_channel':64},
                 hyper_param_optimizer={'lr':1e-3},
                 root_directory = None) -> None:
        self.dataset = dataset
        self.k_fold = k_fold
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.model_param = hyper_param_model
        self.optim_param = hyper_param_optimizer
        self.root_dir = root_directory
    
    def config_optimizer(self, hyper_param) -> torch.optim.Optimizer:
        raise NotImplementedError
    
    def config_model(self, hyper_param) -> torch.nn.Module:
        raise NotImplementedError
    
    def feed_forward(self, data):
        """Determine the forward propagate"""
        raise NotImplementedError
    
    def criterion(self, out, data):
        """Calculate the loss based on the output of network and data"""
        raise NotImplemented
    
    def train(self):
        if self.k_fold > 1:
            self.train_loss = []
            self.train_acc = []
            self.valid_loss = []
            self.valid_acc = []
            self.retained_valid_acc = []
            self.retained_train_acc = []
            kfold = KFold(n_splits=self.k_fold)
            splits = kfold.split(self.dataset)
            # the directory to save the checkpoints
            if self.root_dir is None:
                save_directory = f"./check_point/{self.dataset.name}/"
            else:
                save_directory = f"./{self.root_dir}/{self.dataset.name}/"
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            for fold, (train_idx, valid_idx) in enumerate(splits):
                # path to save the checkpoint
                save_path = os.path.join(save_directory, f"model_{fold+1}.pt")
                # Config optimizer and model
                print(f'[FOLD {fold+1} / {self.k_fold}]')
                print('--------------------------------')
                self.train_dataset = self.dataset[train_idx]
                self.valid_dataset = self.dataset[valid_idx]
                torch.manual_seed(manual_seed)
                train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
                valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)
                self.model = self.config_model(self.model_param).to(device)
                self.optim, self.scheduler = self.config_optimizer(self.optim_param)
                # self.scheduler = ReduceLROnPlateau(self.optim, threshold=1e-3, patience=5, verbose=True, min_lr=1e-5)
                # self.scheduler = StepLR(self.optim, step_size=50, gamma=0.5)
                fold_train_loss = []
                fold_train_acc = []
                fold_valid_loss = []
                fold_valid_acc = []
                for epoch in range(self.num_epoch):
                    # Initial States
                    if epoch == 0:
                        train_acc, train_loss = self.valid_one_epoch(train_loader)
                        valid_acc, valid_loss = self.valid_one_epoch(valid_loader)
                        valid_acc_retained = valid_acc
                        train_acc_retained = train_acc
                        fold_train_loss.append(train_loss)
                        fold_train_acc.append(train_acc)
                        fold_valid_loss.append(valid_loss)
                        fold_valid_acc.append(valid_acc)
                        print(f'Epoch: {epoch+1:03d}\tTrain Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}, Train Loss: {train_loss: .4f}, Valid Loss: {valid_loss: .4f}')   
                    _ = self.train_one_epoch(train_loader)
                    # Check the valid loss and accuracy every 5 epochs
                    if (epoch + 1) % 5 == 0: 
                        train_acc, train_loss = self.valid_one_epoch(train_loader)
                        valid_acc, valid_loss = self.valid_one_epoch(valid_loader)
                        # Retain the model with highest valid accuracy
                        if valid_acc_retained <= valid_acc:
                            valid_acc_retained = valid_acc
                            train_acc_retained = train_acc
                            torch.save(self.model.state_dict(), save_path)
                        fold_train_loss.append(train_loss)
                        fold_train_acc.append(train_acc)
                        fold_valid_loss.append(valid_loss)
                        fold_valid_acc.append(valid_acc)
                        print(f'Epoch: {epoch+1:03d}\tTrain Acc: {train_acc:.4f}, Valid Acc: {valid_acc:.4f}, Train Loss: {train_loss: .4f}, Valid Loss: {valid_loss: .4f}')                 
                    self.scheduler.step()          
                self.train_loss.append(fold_train_loss)
                self.train_acc.append(fold_train_acc)
                self.valid_loss.append(fold_valid_loss)
                self.valid_acc.append(fold_valid_acc)
                self.retained_valid_acc.append(valid_acc_retained)
                self.retained_train_acc.append(train_acc_retained)
            ##TODO: Find the validation accuracy and test the 
                
                
        else:
            # self.model = self.config_model(self.model_param).to(device)
            # self.optim = self.config_optimizer(self.optim_param)
            # self.scheduler = ExponentialLR(self.optim, gamma=0.90, verbose=False)
            # loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
            # loss = []
            # acc = []
            # self.model.reset_parameters()
            # for epoch in range(self.num_epoch):
            #     train_acc, train_loss = self.valid_one_epoch(loader)
            #     loss.append(train_loss)
            #     acc.append(train_acc)
            #     _ = self.train_one_epoch(loader)
            #     print(f'Epoch: {epoch+1:03d}\tTrain Acc: {train_acc:.4f}, Train Loss: {train_loss: .4f}')
            raise ValueError("The number of folds should be larger than 1")

    def select_candidate_models(self):
        pass

    def train_one_epoch(self, train_loader):
        self.model.train()
        # losses = []
        for data in train_loader:
            data = data.to(device)
            out = self.feed_forward(data).to(device)
            loss = self.criterion(out, data)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        #     losses.append(loss.cpu().item())
        # return np.mean(losses)

    def valid_one_epoch(self, loader):
        self.model.eval()
        correct = 0
        losses = []
        # output = []
        # y = []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                out = self.feed_forward(data)
                losses.append(self.criterion(out, data).item())
                pred = out.argmax(dim=1)
                correct += int((pred == data.y).sum())
                # output.append(out.cpu())
                # y.append(data.y.cpu())
        acc = correct / len(loader.dataset)
        # output = torch.vstack(output)[:, 1]
        # y = torch.cat(y)
        # print(f"AUC score: {roc_auc_score(y, output)}")
        return acc, np.mean(losses)

def load_dataset(name, seed):
    np.random.seed(seed=seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Get the dataset and perform train-test split
    if name == "IMDB-BINARY":
        dataset = TUDataset(root='datasets/', name=name, pre_transform=Transform_imdbb())
    elif name == "IMDB-MULTI":
        dataset = TUDataset(root='datasets/', name=name, pre_transform=Transform_imdbm())
    elif name == "ENZYMES":
        dataset = TUDataset(root='datasets/', name=name, use_node_attr=True, pre_transform=T.Compose([Transform_enzyme(), T.NormalizeFeatures()]))
    elif name == "PROTEINS":
        dataset = TUDataset(root='datasets/', name='PROTEINS_full', use_node_attr=True, pre_transform=T.Compose([Transform_protein(), T.NormalizeFeatures()])).shuffle()
    elif name == "BZR":
        dataset = TUDataset(root='./datasets/', name=name, use_node_attr=True, pre_transform=T.Compose([Transform_bzr(), T.NormalizeFeatures()])).shuffle()
    elif name == "COLLAB":
        dataset = TUDataset(root='./datasets/', name=name, use_node_attr=True, pre_transform=T.Compose([Transform_collab()]))
    else:
        dataset = TUDataset(root='datasets/', name=name)
    dataset = dataset.shuffle()
    return dataset
        
class Transform_imdbb(object):
    def __call__(self, data):
        data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)
        data.x = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
        data.x = F.one_hot(data.x, num_classes=136).to(torch.float)
        return data
    
class Transform_imdbm(object):
    def __call__(self, data):
        data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)
        data.x = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
        data.x = F.one_hot(data.x, num_classes=89).to(torch.float)
        return data

class Transform_protein():
    def __call__(self, data):
        data.x = data.x[:, :29]
        return data

class Transform_enzyme():
    def __call__(self, data):
        data.x = data.x[:, :18]
        return data
    
class Transform_bzr():
    def __call__(self, data):
        data.x = data.x[:, :3]
        return data
    
class Transform_collab():
    r"""Adds the globally normalized node degree to the node features.

    Args:
        cat (bool, optional): If set to :obj:`False`, all existing node
            features will be replaced. (default: :obj:`True`)
    """

    # def __init__(self, norm=True, max_value=None, cat=True):
    #     self.norm = norm
    #     self.max = max_value
    #     self.cat = cat

    # def __call__(self, data):
    #     col, x = data.edge_index[1], data.x
    #     deg = degree(col, data.num_nodes)

    #     if self.norm:
    #         deg = deg / (deg.max() if self.max is None else self.max)

    #     deg = deg.view(-1, 1)

    #     if x is not None and self.cat:
    #         x = x.view(-1, 1) if x.dim() == 1 else x
    #         data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
    #     else:
    #         data.x = deg

    #     return data

    # def __repr__(self):
    #     return '{}(norm={}, max_value={})'.format(self.__class__.__name__, self.norm, self.max)

    def __call__(self, data):
        data.x = torch.zeros((data.num_nodes, 1), dtype=torch.float)
        data.x = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
        data.x = F.one_hot(data.x, num_classes=492).to(torch.float)
        return data
    
def dataset_info(dataset):
    print()
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]
    print()
    print(data)
    print('=============================================================')
    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

    print()
    for i in range(dataset.num_classes):
        print(f"Class {i}:, number of observations: {(dataset.y == i).sum().item()}")

def evaluate(model, dataset):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in DataLoader(dataset, 64):
            # data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
        acc = correct / len(dataset)
    return acc