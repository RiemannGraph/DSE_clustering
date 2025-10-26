import torch
import networkx as nx
from torch_geometric.data import Dataset, Data
from torch_geometric.data.data import BaseData
from torch_geometric.datasets import (Amazon, KarateClub, Planetoid, WebKB)
from torch_geometric.utils import from_networkx
import urllib.request
import io
import zipfile
import numpy as np
from modules.layers import IsoTransform
from utils.model_utils import normalize_adj, adjacency2index


def load_data(configs):
    dataset = None
    if configs.dataset in ["computers", "Photo"]:
        dataset = Amazon(configs.root_path, name=configs.dataset)
    elif configs.dataset in ['Cora', 'Citeseer', 'PubMed']:
        dataset = Planetoid(configs.root_path, name=configs.dataset)
    elif configs.dataset == 'KarateClub':
        dataset = KarateClub()
    elif configs.dataset == 'FootBall':
        dataset = Football()
    elif configs.dataset in ['eat', 'bat', 'uat']:
        dataset = ATsDataset(root=configs.root_path, name=configs.dataset)
    elif configs.dataset in ['Cornell', 'Texas', 'Wisconsin']:
        dataset = WebKB(root=configs.root_path, name=configs.dataset)
    data = dataset[0].clone()
    N = data.x.shape[0]
    data.adj = torch.sparse_coo_tensor(indices=data.edge_index,
                                       values=torch.ones(data.edge_index.shape[1]),
                                       size=(N, N))
    data.adj = normalize_adj(data.adj, sparse=True)
    data.adj_aug = IsoTransform(configs.ax_i, configs.ax_j, configs.L, configs.top_k_sim, configs.top_k_aug, configs.omega, configs.alpha)(data.x, data.adj)
    data.num_classes = data.y.max().item()
    return data


class Football(Dataset):
    """
    Refer to https://networkx.org/documentation/stable/auto_examples/graph/plot_football.html
    """
    def __init__(self):
        super().__init__()
        url = "http://www-personal.umich.edu/~mejn/netdata/football.zip"

        sock = urllib.request.urlopen(url)  # open URL
        s = io.BytesIO(sock.read())  # read into BytesIO "file"
        sock.close()

        zf = zipfile.ZipFile(s)  # zipfile object
        txt = zf.read("football.txt").decode()  # read info file
        gml = zf.read("football.gml").decode()  # read gml data
        # throw away bogus first line with # from mejn files
        gml = gml.split("\n")[1:]
        graph = nx.parse_gml(gml)  # parse gml data

        data = from_networkx(graph)
        data.x = torch.eye(data.num_nodes)
        data.y = torch.tensor(data.value.tolist()).long()
        self.data = data

    def len(self) -> int:
        return 1

    def get(self, idx: int) -> BaseData:
        return self.data

    @property
    def num_node_features(self) -> int:
        return self.data.num_nodes

    @property
    def num_features(self) -> int:
        return self.data.num_nodes

    @property
    def num_classes(self) -> int:
        return len(np.unique(self.data.y))


class ATsDataset(Dataset):
    def __init__(self, root, name='eat'):
        super().__init__(root)
        adj = np.load(f'{root}/{name}/{name}_adj.npy')
        feat = np.load(f'{root}/{name}/{name}_feat.npy')
        label = np.load(f'{root}/{name}/{name}_label.npy')

        self.num_nodes = feat.shape[0]
        x = torch.tensor(feat).float()
        y = list(label)
        edge_index = adjacency2index(torch.tensor(adj))
        data = Data(x=x, edge_index=edge_index, y=y)
        self.data = data

    def len(self) -> int:
        return 1

    def get(self, idx: int) -> BaseData:
        return self.data

    @property
    def num_node_features(self) -> int:
        return self.data.x.shape[1]

    @property
    def num_features(self) -> int:
        return self.data.x.shape[1]

    @property
    def num_classes(self) -> int:
        return len(np.unique(self.data.y))
