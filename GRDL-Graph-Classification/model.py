import torch
from torch_geometric.loader import DataLoader
import numpy as np
from torch_geometric.nn import GINConv, JumpingKnowledge
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
from torch.nn import Linear, Parameter, ModuleList, BatchNorm1d
from sklearn.cluster import KMeans
import math
device = "cuda" if torch.cuda.is_available() else "cpu"

class ReferenceLayer(torch.nn.Module):
    def __init__(self, input_channels, output_channels, num_supp, gamma, init_atoms=None):
        """
        Args
        -----
        input_channels: int
            The dimension of the layer input.
        output_channels: int
            The dimension of the layer output.
        num_supp: int
            The number of support of the atoms.
        gamma: float
            The hyper-parameter used in gaussian kernel.
        random_init: bool
            Flag used to determine whether to initialize the atoms randomly from a gaussian distribution.
        init_atoms: list or torch.Tensor
            The initialization of the atoms if choose not to use random initialization. 
            Can be a 3-dimension tensor with shape (`output_channels`, `num_supp`, `input_channels`) OR
            a list with length equal to `output_channels`, and tensor with shape (arbitray, `input_channels`)
        """
        super().__init__()
        # self.input_channels = input_channels
        self.output_channels = output_channels
        # self.num_supp = num_supp
        self.init_gamma = gamma
        self.init_atoms = init_atoms
        self.atoms = Parameter(torch.empty(size=(output_channels, num_supp, input_channels)))
        self.gamma = Parameter(torch.empty(size=(1,)))
        self.reset_parameters()

    def init_atoms(self, num_supp, random_init, init_atoms):
        if random_init:
            atoms = Parameter(torch.randn(size=(self.output_channels, num_supp, self.input_channels)))
        else:
            if isinstance(init_atoms, list):
                atoms = [Parameter(atom) for atom in init_atoms]
            else:
                atoms = Parameter(init_atoms)
        return atoms
    
    def forward(self, x, data):
        mmd_distance = cal_mmd(self.atoms, x, data.batch, self.gamma)
        S = -mmd_distance
        y_hat = torch.exp(S)
        result = y_hat / y_hat.sum(dim=1, keepdim=True)
        return result
    
    def discriminate_loss(self):
        num_atoms = self.output_channels
        loss = 0
        for i in range(num_atoms-1):
            x = self.atoms[i]
            y = self.atoms[i+1:]
            d_xy = ((x[np.newaxis, :, np.newaxis, :] - y[:, np.newaxis, :, :])**2).sum(dim=-1)
            k_xy = neg_exp(d_xy, self.gamma)
            d_xx = ((x[:, np.newaxis, :] - x[np.newaxis, :, :])**2).sum(dim=-1)
            k_xx = neg_exp(d_xx, self.gamma)
            d_yy = ((y[:, :, np.newaxis, :] - y[:, np.newaxis, :, :])**2).sum(dim=-1)
            k_yy = neg_exp(d_yy, self.gamma)
            mmd_distance = k_xx.mean() + k_yy.mean(dim=-1).mean(dim=-1) - 2*k_xy.mean(dim=-1).mean(dim=-1)
            loss += -(mmd_distance**0.5).sum()        
        return loss
    
    def reset_parameters(self):
        if self.init_atoms is not None:
            self.atoms = self.init_atoms
        else:
            self.atoms.data.normal_(0, 1)
        self.gamma.data.fill_(self.init_gamma)
    
class GRDL(torch.nn.Module):
    def __init__(self, 
                 extractor,
                 mmd_in_channels, 
                 num_atoms,
                 num_atom_supp, 
                 gamma,):
        super().__init__()
        self.num_atoms = num_atoms
        self.num_atom_supp = num_atom_supp
        self.extractor = extractor
        self.mmd = ReferenceLayer(mmd_in_channels, num_atoms, num_atom_supp, gamma)

    def init_atoms(self, dataset):
        results = []
        print("Initializing the atoms...")
        shapes = torch.tensor([data.num_nodes for data in dataset])
        labels = dataset.y
        unique_labels = torch.unique(labels)
        # Get transformed x and batch information
        with torch.no_grad():
            loader = DataLoader(dataset, len(dataset))
            for data in loader:
                x = self.extractor(data)
                batch = data.batch
        start_idx = torch.concat((torch.tensor([0]), shapes)).cumsum(dim=0)
        for label in unique_labels:
            idx_of_label = torch.where(labels == label)[0]
            idx = idx_of_label[torch.where(shapes[idx_of_label] == self.num_atom_supp)[0]]
            if len(idx) == 0:
                data_of_label = []
                print(f"No samples for shape {self.num_atom_supp} -- computing kmeans on features within the label")
                for i in idx_of_label:
                    data_of_label.append(x[start_idx[i]:start_idx[i+1]])
                data_of_label = torch.cat(data_of_label)
                km = KMeans(n_clusters=self.num_atom_supp, init='k-means++', n_init=10, random_state=0)
                km.fit(data_of_label)
                X = torch.tensor(km.cluster_centers_, dtype=torch.float)
                results.append(X)
            else:
                pos = idx[0].item()
                results.append(x[start_idx[pos]: start_idx[pos+1]])
        self.mmd.atoms = Parameter(torch.stack(results))

    def forward(self, data):
        x = self.extractor(data)
        x = self.mmd(x, data)
        return x
    
    def reset_parameters(self):
        self.extractor.reset_parameters()
        self.mmd.reset_parameters()

class GIN(torch.nn.Module):
    def __init__(self, 
                 input_channels, 
                 hidden_channels, 
                 num_layer_mlp, 
                 num_layer_gin,
                 num_layer_pred, 
                 num_classes, 
                 readout='add',
                 jump_mode=None):
        super().__init__()
        torch.manual_seed(1234)
        self.extractor = GINExtractor(input_channels, 
                                      hidden_channels, 
                                      num_layer_mlp,
                                      num_layer_gin,
                                      jump_mode)

        if jump_mode == 'cat':
            self.mlp = MLP(hidden_channels*num_layer_gin, hidden_channels, num_classes, num_layer_pred)
        else:
            self.mlp = MLP(hidden_channels, hidden_channels, num_classes, num_layer_pred)
        self.readout = readout

    def reset_parameters(self):
        for l in list(self.children()):
            l.reset_parameters()
    
    def forward(self, data):
        x = data.x
        batch = data.batch
        # 1. Obtain node embeddings 
        x = self.extractor(data)
        # 2. Readout layer
        if self.readout == 'add':
            x = global_add_pool(x, batch)
        elif self.readout == 'mean':
            x = global_mean_pool(x, batch)
        elif self.readout == 'max':
            x = global_max_pool(x, batch)
        else:
            raise ValueError
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.mlp(x)
        return x
    
    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.extractor(x, edge_index)
        # 2. Readout layer
        if self.readout == 'add':
            x = global_add_pool(x, batch)
        elif self.readout == 'mean':
            x = global_mean_pool(x, batch)
        elif self.readout == 'max':
            x = global_max_pool(x, batch)
        else:
            raise ValueError
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.mlp(x)
        return x
    
class MLP(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.linears = ModuleList()
        self.bns = ModuleList()

        self.linears.append(Linear(input_channels, hidden_channels))
        self.bns.append(BatchNorm1d(hidden_channels))
        for layer in range(num_layers - 2):
            self.linears.append(Linear(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm1d(hidden_channels))
        self.linears.append(Linear(hidden_channels, output_channels))

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.linears[i](x)
            x = self.bns[i](x)
            x = F.leaky_relu(x, negative_slope=0.1)
            # x = F.relu(x)
        x = self.linears[-1](x)
        return x
    
    def reset_parameters(self):
        for layer in self.linears:
            layer.reset_parameters()
        for layer in self.bns:
            layer.reset_parameters()

class GINExtractor(torch.nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_channels,
                 num_layer_mlp,
                 num_layer_gin,
                 jump_mode = None):
        super().__init__()
        torch.manual_seed(1234)
        if jump_mode is not None:
            self.jump_layer = JumpingKnowledge(jump_mode)
        else:
            self.jump_layer = None
        self.GIN_layers = ModuleList()
        self._num_layer_gin = num_layer_gin
        for layer in range(num_layer_gin):
            if layer == 0:
                local_input_channels = input_channels
            else:
                local_input_channels = hidden_channels
            self.GIN_layers.append(GINConv(MLP(local_input_channels, hidden_channels, hidden_channels, num_layer_mlp), train_eps=False))
    
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        xs = []
        curr_x = x
        for i in range(self._num_layer_gin):
            curr_x = self.GIN_layers[i](x=curr_x, edge_index=edge_index)
            xs.append(curr_x)
        if self.jump_layer is None:
            return xs[-1]
        else:
            return self.jump_layer(xs)
    
    def reset_parameters(self):
        for layer in self.GIN_layers:
            layer.reset_parameters()

def cal_mmd(atoms, x, batch, gamma):
    d_xy = ((atoms[:, :, np.newaxis, :] - x[np.newaxis, np.newaxis, :, :])**2).sum(axis=-1)
    k_xy = neg_exp(d_xy, gamma)
    d_yy = ((atoms[:, :, np.newaxis, :] - atoms[:, np.newaxis, :, :])**2).sum(axis=-1)
    k_yy = neg_exp(d_yy, gamma)
    d_xx = ((x[:, np.newaxis, :] - x)**2).sum(axis=-1)
    k_xx = neg_exp(d_xx, gamma)
    # Construct matrix
    unique_batch, num_element = torch.unique(batch,return_counts=True)
    num_element = num_element.cpu()
    index = torch.hstack((torch.tensor(0), torch.cumsum(num_element, 0)))
    A = torch.zeros(batch.shape[0], unique_batch.shape[0])
    for i in range(unique_batch.shape[0]):
        A[index[i]: index[i+1], i] = torch.ones(num_element[i])/ num_element[i]
    A = A.to(x.device)
    xx = torch.diag(A.T @ k_xx @ A)
    xy = torch.mean(k_xy @ A, dim=1)
    yy = k_yy.mean(dim=1).mean(dim=1)
    mmd_distance = (yy.reshape(-1, 1) + xx - 2 * xy).T 
    mmd_distance = mmd_distance ** 0.5
    return mmd_distance

def neg_exp(dist, gamma):
    return torch.exp(-dist/gamma)

def init_weights(m):
    if isinstance(m, Linear):
        m.weight.data = torch.nn.init.kaiming_uniform_(
            m.weight.data, a=0.1
        )
        if m.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(m.bias, -bound, bound)
    elif (isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d)):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
