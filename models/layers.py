import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from geoopt.manifolds.stereographic.math import mobius_matvec, project, expmap0, mobius_add, logmap0
from torch_scatter import scatter_sum, scatter_softmax
from torch_geometric.utils import add_self_loops
import math
from utils.utils import gumbel_softmax, adjacency2index, index2adjacency, normalize_adj, graph_top_K


class LorentzGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, use_bias, dropout, use_att, nonlin=None):
        super(LorentzGraphConvolution, self).__init__()
        self.linear = LorentzLinear(manifold, in_features, out_features, use_bias, dropout, nonlin=nonlin)
        self.agg = LorentzAgg(manifold, out_features, dropout, use_att)

    def forward(self, x, adj):
        h = self.linear(x)
        h = self.agg(h, adj)
        return h


class LorentzLinear(nn.Module):
    def __init__(self,
                 manifold,
                 in_features,
                 out_features,
                 bias=True,
                 dropout=0.1,
                 scale=10,
                 fixscale=False,
                 nonlin=None):
        super().__init__()
        self.manifold = manifold
        self.nonlin = nonlin
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = nn.Linear(
            self.in_features, self.out_features, bias=bias)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(()) * math.log(scale), requires_grad=not fixscale)

    def forward(self, x):
        if self.nonlin is not None:
            x = self.nonlin(x)
        x = self.weight(self.dropout(x))
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1.1
        scale = (time * time - 1) / \
            (x_narrow * x_narrow).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        x = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)
        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        step = self.in_features
        nn.init.uniform_(self.weight.weight, -stdv, stdv)
        with torch.no_grad():
            for idx in range(0, self.in_features, step):
                self.weight.weight[:, idx] = 0
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)


class LorentzAgg(nn.Module):
    """
    Lorentz aggregation layer.
    """

    def __init__(self, manifold, in_features, dropout, use_att):
        super(LorentzAgg, self).__init__()
        self.manifold = manifold

        self.in_features = in_features
        self.dropout = dropout
        self.use_att = use_att
        if self.use_att:
            self.key_linear = LorentzLinear(manifold, in_features, in_features)
            self.query_linear = LorentzLinear(manifold, in_features, in_features)
            self.bias = nn.Parameter(torch.zeros(()) + 20)
            self.scale = nn.Parameter(torch.zeros(()) + math.sqrt(in_features))

    def forward(self, x, adj):
        if self.use_att:
            query = self.query_linear(x)
            key = self.key_linear(x)
            att_adj = 2 + 2 * self.manifold.cinner(query, key)
            att_adj = att_adj / self.scale + self.bias
            att_adj = torch.sigmoid(att_adj)
            att_adj = torch.mul(adj.to_dense(), att_adj)
            support_t = torch.matmul(att_adj, x)
        else:
            support_t = torch.matmul(adj, x)

        denorm = (-self.manifold.inner(None, support_t, keepdim=True))
        denorm = denorm.abs().clamp_min(1e-8).sqrt()
        output = support_t / denorm
        return output


class LorentzAssignment(nn.Module):
    def __init__(self, manifold, in_features, hidden_features, num_assign, dropout,
                 bias=False, use_att=False, nonlin=None, temperature=0.2):
        super(LorentzAssignment, self).__init__()
        self.manifold = manifold
        self.num_assign = num_assign
        self.proj = LorentzGraphConvolution(manifold, in_features, hidden_features,
                                            use_bias=bias, dropout=dropout, nonlin=None, use_att=False)
        self.assign_linear = nn.Sequential(
            nn.Linear(hidden_features, num_assign, bias=False),
                                           )
        self.temperature = temperature
        self.key_linear = LorentzLinear(manifold, in_features, hidden_features, bias=False)
        self.query_linear = LorentzLinear(manifold, in_features, hidden_features, bias=False)
        self.scalar_map = nn.Sequential(
            nn.Linear(2 * hidden_features, 1, bias=False),
            nn.LeakyReLU()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        ass = self.assign_linear(self.manifold.logmap0(self.proj(x, adj))).softmax(-1)
        q = self.query_linear(x)
        k = self.key_linear(x)
        edge_index = adj.coalesce().indices()
        src, dst = edge_index[0], edge_index[1]
        qk = torch.cat([q[src], k[dst]], dim=-1)
        score = self.scalar_map(qk).squeeze(-1)
        # score = 2 + 2 * self.manifold.inner(q[src], k[dst])
        score = scatter_softmax(score, src, dim=-1)
        att = torch.sparse_coo_tensor(edge_index, score, size=(x.shape[0], x.shape[0])).to(x.device)
        ass = torch.matmul(att, ass)   # (N_k, N_{k-1})
        ass = gumbel_softmax(torch.log(ass + 1e-6), temperature=self.temperature)
        return ass


class LSENetLayer(nn.Module):
    def __init__(self, manifold, in_features, hidden_features, num_assign, dropout,
                 bias=False, use_att=False, nonlin=None, temperature=0.2):
        super(LSENetLayer, self).__init__()
        self.manifold = manifold
        self.assignor = LorentzAssignment(manifold, in_features, hidden_features, num_assign,
                                          use_att=use_att, bias=bias,
                                          dropout=dropout, nonlin=nonlin, temperature=temperature)
        self.temperature = temperature

    def forward(self, x, adj):
        ass = self.assignor(x, adj)
        support_t = ass.t() @ x
        denorm = (-self.manifold.inner(None, support_t, keepdim=True))
        denorm = denorm.abs().clamp_min(1e-8).sqrt()
        x_assigned = support_t / denorm

        adj = ass.t() @ adj @ ass
        idx = adj.nonzero().t()
        adj = torch.sparse_coo_tensor(idx, adj[idx[0], idx[1]], size=adj.shape)
        return x_assigned, adj, ass


class IsoTransform(nn.Module):
    def __init__(self, in_dim, out_dim, ax_i, ax_j, n_layers=2, k=20, omega=0.1, mode='sim'):
        super(IsoTransform, self).__init__()
        # self.lin = nn.Linear(in_dim, out_dim)
        self.theta = nn.Parameter(torch.tensor([np.pi / 4]), requires_grad=False)
        self.ax_i = ax_i
        self.ax_j = ax_j
        self.dim = in_dim
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
        # x = raw_adj @ self.lin(x)
        xxt = x @ x.t()
        x_ex = torch.linalg.inv(xxt + self.omega * torch.eye(x.shape[0]).to(x.device))
        sim_adj = graph_top_K(xxt, k=self.k)
        adj_l = torch.matrix_power(raw_adj.to_dense(), self.l)
        sim_l = torch.matrix_power(sim_adj.to_dense(), self.l)
        B = (adj_l + sim_l) @ x @ rot.t()
        xb = x @ B.t() + B @ x.t()
        M = 0.5 * (x_ex @ xb + xb @ x_ex)
        v, P = torch.linalg.eig(M)
        v = v.real
        P = P.real
        A = P @ torch.diag(v.real.clamp(min=1e-6) ** (1. / self.l)) @ P.t()
        A = graph_top_K(A, k=self.k)
        return A