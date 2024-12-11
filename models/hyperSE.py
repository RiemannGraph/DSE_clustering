import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import gumbel_softmax, graph_top_K
from torch_scatter import scatter_sum
from models.layers import IsoTransform, LorentzTransformation
from manifold.lorentz import Lorentz
from models.encoders import GraphEncoder
import math
from models.l_se_net import LSENet

MIN_NORM = 1e-15
EPS = 1e-6


class HyperSE(nn.Module):
    def __init__(self, in_features, hidden_dim_enc, hidden_features, num_nodes, height=3, temperature=0.2,
                 embed_dim=32, cl_dim=32, dropout=0.5, nonlin='relu', decay_rate=None, max_nums=None, tau=2.):
        super(HyperSE, self).__init__()
        self.num_nodes = num_nodes
        self.height = height
        self.tau = tau
        self.manifold = Lorentz()
        self.encoder = LSENet(self.manifold, in_features, hidden_dim_enc, hidden_features,
                              num_nodes, height, temperature, embed_dim, dropout,
                              nonlin, decay_rate, max_nums)
        self.proj = LorentzTransformation(embed_dim + 1, cl_dim)

    def forward(self, data):
        features = data.x
        adj = data.adj.clone()
        embeddings, ass_mat, _ = self.encoder(features, adj)
        self.embeddings = {}
        for height, x in embeddings.items():
            self.embeddings[height] = x.detach()
        clu_mat = {self.height: torch.eye(self.num_nodes).to(data.x.device)}
        for k in range(self.height - 1, 0, -1):
            clu_mat[k] = clu_mat[k + 1] @ ass_mat[k + 1]
        for k, v in clu_mat.items():
            idx = v.max(1)[1]
            t = torch.zeros_like(v)
            t[torch.arange(t.shape[0]), idx] = 1.
            clu_mat[k] = t
        self.clu_mat = clu_mat
        return self.embeddings[self.height]

    def loss(self, data, scale=0.1, gamma=0.6):
        adj = data.adj.clone()
        aug_adj = data.aug_adj.clone()
        x = data.x.clone()
        z, ass_mats, adj_set = self.encoder(x, adj)
        z_aug, ass_mats_aug, adj_aug_set = self.encoder(x, aug_adj)

        se_loss = self.se_loss(self.height, adj_set, ass_mats) + self.manifold.dist0(z[0])
        se_loss_aug = self.se_loss(self.height, adj_aug_set, ass_mats_aug) + self.manifold.dist0(z_aug[0])

        z = self.proj(z[self.height])
        z_aug = self.proj(z_aug[self.height])

        tree_cl_loss = self.tree_cl_loss(self.manifold, z, z_aug, self.tau)

        return (1 - gamma) * (se_loss + se_loss_aug) + scale * tree_cl_loss * gamma

    @staticmethod
    def tree_cl_loss(manifold, z1, z2, tau=2.):
        sim = 2. + 2 * manifold.cinner(z1, z2)
        sim = -torch.log_softmax(sim / tau, dim=-1).diag()
        loss = torch.mean(sim)
        return loss

    @staticmethod
    def se_loss(height, adj_set: dict, ass_mats: dict, eps: float = 1e-6):
        se_loss = 0
        vol_G = adj_set[height].sum()

        for k in range(height, 0, -1):
            adj_dense = adj_set[k].to_dense()
            degree = adj_dense.sum(dim=1)
            diag = adj_dense.diag()
            if k == 1:
                vol_parent = vol_G
            else:
                vol_parent = adj_set[k - 1].to_dense().sum(dim=-1)
                vol_parent = torch.einsum('ij, j->i', ass_mats[k], vol_parent)
            delta_vol = degree - diag
            log_vol_ratio_k = torch.log2((degree + eps) / (vol_parent + eps))
            se_loss += torch.sum(delta_vol * log_vol_ratio_k)
        se_loss = -1 / vol_G * se_loss
        return se_loss