import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.layers import LSENetLayer
from models.encoders import GraphEncoder
from utils.utils import select_activation
from models.encoders import GraphEncoder


class LSENet(nn.Module):
    def __init__(self, manifold, in_features, hidden_dim_enc, hidden_features, num_nodes, height=3, temperature=0.1,
                 embed_dim=64, dropout=0.5, nonlin='relu', decay_rate=None, max_nums=None):
        super(LSENet, self).__init__()
        if max_nums is not None:
            assert len(max_nums) == height - 1, "length of max_nums must equal height-1."
        self.manifold = manifold
        self.nonlin = select_activation(nonlin) if nonlin is not None else None
        self.temperature = temperature
        self.num_nodes = num_nodes
        self.height = height
        self.scale = nn.Parameter(torch.tensor([0.999]), requires_grad=True)
        self.embed_layer = GraphEncoder(self.manifold, 2, in_features + 1, hidden_dim_enc, embed_dim + 1, use_att=False,
                                                     use_bias=True, dropout=dropout, nonlin=self.nonlin)
        self.layers = nn.ModuleList([])
        if max_nums is None:
            decay_rate = int(np.exp(np.log(num_nodes) / height)) if decay_rate is None else decay_rate
            max_nums = [int(num_nodes / (decay_rate ** i)) for i in range(1, height)]
        for i in range(height - 1):
            self.layers.append(LSENetLayer(self.manifold, embed_dim + 1, hidden_features, max_nums[i],
                                           bias=True, use_att=False, dropout=dropout,
                                           nonlin=self.nonlin, temperature=self.temperature))

    def forward(self, x, adj):
        """mapping x into Lorentz model"""
        o = torch.zeros_like(x).to(x.device)
        x = torch.cat([o[:, 0:1], x], dim=1)
        x = self.manifold.expmap0(x)
        z = self.embed_layer(x, adj)
        z = self.normalize(z)

        tree_node_coords = {self.height: z}
        assignments = {}
        adj_set = {self.height: adj}

        edge = adj.clone()
        ass = None
        for i, layer in enumerate(self.layers):
            z, edge, ass = layer(z, edge)
            tree_node_coords[self.height - i - 1] = z
            assignments[self.height - i] = ass
            adj_set[self.height - i - 1] = edge


        tree_node_coords[0] = self.manifold.Frechet_mean(z)
        assignments[1] = torch.ones(ass.shape[-1], 1).to(x.device)

        return tree_node_coords, assignments, adj_set

    def normalize(self, x):
        x = self.manifold.to_poincare(x)
        x = F.normalize(x, p=2, dim=-1) * self.scale.clamp(1e-2, 0.999)
        x = self.manifold.from_poincare(x)
        return x


