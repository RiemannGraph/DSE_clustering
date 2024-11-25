import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt.manifolds import PoincareBall
from geoopt.manifolds.stereographic.math import project, dist0, dist
import numpy as np
from utils.utils import gumbel_softmax, graph_top_K
from torch_scatter import scatter_sum
from utils.decode import construct_tree
from models.layers import LorentzGraphConvolution, LorentzLinear
from manifold.lorentz import Lorentz
from models.encoders import GraphEncoder
import math
from models.l_se_net import LSENet
from torch_geometric.utils import negative_sampling

MIN_NORM = 1e-15
EPS = 1e-6


class HyperSE(nn.Module):
    def __init__(self, in_features, hidden_dim_enc, hidden_features, num_nodes, height=3, temperature=0.2,
                 embed_dim=2, dropout=0.5, nonlin='relu', decay_rate=None, max_nums=None):
        super(HyperSE, self).__init__()
        self.num_nodes = num_nodes
        self.height = height
        self.tau = temperature
        self.manifold = Lorentz()
        self.iso_trans = IsoTransform(in_features, 32, 2, 30)
        self.encoder = LSENet(self.manifold, in_features, hidden_dim_enc, hidden_features,
                              num_nodes, height, temperature, embed_dim, dropout,
                              nonlin, decay_rate, max_nums)

    def forward(self, data, device=torch.device('cuda:0')):
        features = data['feature'].to(device)
        adj = data['adj'].to(device)
        embeddings, clu_mat = self.encoder(features, adj)
        self.embeddings = {}
        for height, x in embeddings.items():
            self.embeddings[height] = x.detach()
        ass_mat = {self.height: torch.eye(self.num_nodes).to(device)}
        for k in range(self.height - 1, 0, -1):
            ass_mat[k] = ass_mat[k + 1] @ clu_mat[k + 1]
        for k, v in ass_mat.items():
            idx = v.max(1)[1]
            t = torch.zeros_like(v)
            t[torch.arange(t.shape[0]), idx] = 1.
            ass_mat[k] = t
        self.ass_mat = ass_mat
        return self.embeddings[self.height]

    def loss(self, data, edge_index, device=torch.device('cuda:0')):
        """_summary_

        Args:
            data: dict
            edge_index: torch.LongTensor
            device: torch.Device
        """
        weight = data['weight']
        adj = data['adj'].to(device)
        degrees = data['degrees']
        features = data['feature']
        adj = self.iso_trans(features, adj)
        embeddings, clu_mat = self.encoder(features, adj)

        se_loss = 0
        vol_G = weight.sum()
        ass_mat = {self.height: torch.eye(self.num_nodes).to(device)}
        vol_dict = {self.height: degrees, 0: vol_G.unsqueeze(0)}
        for k in range(self.height - 1, 0, -1):
            ass_mat[k] = ass_mat[k + 1] @ clu_mat[k + 1]
            vol_dict[k] = torch.einsum('ij, i->j', ass_mat[k], degrees)

        for k in range(1, self.height + 1):
            vol_parent = torch.einsum('ij, j->i', clu_mat[k], vol_dict[k - 1])  # (N_k, )
            log_vol_ratio_k = torch.log2((vol_dict[k] + EPS) / (vol_parent + EPS))  # (N_k, )
            ass_i = ass_mat[k][edge_index[0]]  # (E, N_k)
            ass_j = ass_mat[k][edge_index[1]]
            weight_sum = torch.einsum('en, e->n', ass_i * ass_j, weight)  # (N_k, )
            delta_vol = vol_dict[k] - weight_sum  # (N_k, )
            se_loss += torch.sum(delta_vol * log_vol_ratio_k)
        se_loss = -1 / vol_G * se_loss
        return se_loss + self.manifold.dist0(embeddings[0])


class IsoTransform(nn.Module):
    def __init__(self, in_dim, out_dim, ax_i, ax_j, n_layers=2, k=20, omega=0.1, mode='sim'):
        super(IsoTransform, self).__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.theta = nn.Parameter(torch.tensor([np.pi / 4]), requires_grad=True)
        self.ax_i = ax_i
        self.ax_j = ax_j
        self.dim = out_dim
        self.k = k
        self.l = n_layers
        self.omega = omega

    def givens_rot_mat(self):
        i, j = self.ax_i, self.ax_j
        G = torch.eye(self.dim).to(self.theta.device)
        c = torch.cos(self.theta)
        s = torch.sin(self.theta)
        G[i, i] = c
        G[j, j] = c
        G[i, j] = -s
        G[j, i] = s
        return G

    def forward(self, x, raw_adj):
        rot = self.givens_rot_mat()
        x = raw_adj @ self.lin(x)
        x_ex = torch.linalg.inv(x @ x.t() + self.omega * torch.eye(x.shape[0]).to(x.device))
        sim_adj = torch.cosine_similarity(x[None], x[:, None], dim=-1)
        sim_adj = graph_top_K(sim_adj, k=self.k)
        adj_l = torch.matrix_power(raw_adj, self.l)
        sim_l = torch.matrix_power(sim_adj, self.l)
        B = (adj_l + sim_l) @ x @ rot.t()
        xb = x @ B.t() + B @ x.t()
        M = 0.5 * (x_ex @ xb + xb @ x_ex)
        v, P = torch.linalg.eig(M)
        v = v.real
        P = P.real
        A = P @ torch.diag(v.real.clamp(min=1e-6) ** (1. / self.l)) @ P.t()
        A = graph_top_K(A, k=self.k)
        return A
