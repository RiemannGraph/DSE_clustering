import torch
import torch.nn as nn
from torch_scatter import scatter_softmax
import math
from utils.model_utils import gumbel_softmax, graph_top_K, normalize_adj, givens_rot_mat


class LorentzGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_dim, out_dim, use_bias, dropout, use_att, nonlin=None):
        super(LorentzGraphConvolution, self).__init__()
        self.linear = LorentzLinear(manifold, in_dim, out_dim, use_bias, dropout, nonlin=nonlin)
        self.agg = LorentzAgg(manifold, out_dim, dropout, use_att)

    def forward(self, x, adj):
        h = self.linear(x)
        h = self.agg(h, adj)
        return h


class LorentzLinear(nn.Module):
    def __init__(self,
                 manifold,
                 in_dim,
                 out_dim,
                 bias=True,
                 dropout=0.1,
                 scale=10,
                 fixscale=False,
                 nonlin=None):
        super().__init__()
        self.manifold = manifold
        self.nonlin = nonlin
        self.in_features = in_dim
        self.out_features = out_dim
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

    def __init__(self, manifold, in_dim, dropout, use_att):
        super(LorentzAgg, self).__init__()
        self.manifold = manifold

        self.in_features = in_dim
        self.dropout = dropout
        self.use_att = use_att
        if self.use_att:
            self.key_linear = LorentzLinear(manifold, in_dim, in_dim)
            self.query_linear = LorentzLinear(manifold, in_dim, in_dim)
            self.bias = nn.Parameter(torch.zeros(()) + 20)
            self.scale = nn.Parameter(torch.zeros(()) + math.sqrt(in_dim))

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
    def __init__(self, manifold, in_dim, hid_dim, num_assign, dropout,
                 bias=False, temperature=0.2):
        super(LorentzAssignment, self).__init__()
        self.manifold = manifold
        self.num_assign = num_assign
        self.assign_linear = nn.Linear(in_dim, num_assign, bias=bias)
        nn.init.xavier_normal_(self.assign_linear.weight)
        self.temperature = temperature
        self.key_linear = LorentzLinear(manifold, in_dim, hid_dim, bias=False)
        self.query_linear = LorentzLinear(manifold, in_dim, hid_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        ass = self.assign_linear(self.manifold.logmap0(x)).softmax(-1)
        q = self.query_linear(x)
        k = self.key_linear(x)
        edge_index = adj.coalesce().indices()
        src, dst = edge_index[0], edge_index[1]
        score = self.manifold.dist(q[src], k[dst])
        score = scatter_softmax(-score, src, dim=-1)
        att = torch.sparse_coo_tensor(edge_index, score, size=(x.shape[0], x.shape[0])).to(x.device)
        ass = torch.matmul(att, ass)   # (N_k, N_{k-1})
        ass = gumbel_softmax(torch.log(ass + 1e-6), temperature=self.temperature)
        return ass


class LSENetLayer(nn.Module):
    def __init__(self, manifold, in_dim, hid_dim, num_assign, dropout,
                 bias=False, use_att=False, nonlin=None, temperature=0.2):
        super(LSENetLayer, self).__init__()
        self.manifold = manifold
        # self.embeder = LorentzGraphConvolution(manifold, in_dim, hid_dim,
        #                                        True, dropout, use_att, nonlin)
        self.assigner = LorentzAssignment(manifold, hid_dim,
                                          hid_dim, num_assign,
                                          dropout, bias, temperature)

    def forward(self, x, adj):
        # x = self.embeder(x, adj)
        ass = self.assigner(x, adj)
        support_t = ass.t() @ x
        denorm = (-self.manifold.inner(None, support_t, keepdim=True))
        denorm = denorm.abs().clamp_min(1e-8).sqrt()
        x_par = support_t / denorm

        adj_par = ass.t() @ adj @ ass
        idx = adj_par.nonzero().t()
        adj_par = torch.sparse_coo_tensor(idx, adj_par[idx[0], idx[1]], size=adj_par.shape)
        return x_par, adj_par, ass, x


class LorentzTransformation(nn.Module):
    """
    Input size: [N, in_dim]
    v : [1, in_dim]
    W : [in_dim - 1, in_dim]
    Output size: [N, in_dim]
    """
    def __init__(self, in_dim):
        super(LorentzTransformation, self).__init__()
        self.v = nn.Parameter(torch.randn(1, in_dim), requires_grad=True)
        diag = torch.ones(in_dim)
        diag.narrow(-1, 0, 1).mul_(-1)
        self.metric = nn.Parameter(torch.diag(diag), requires_grad=False)
        self.in_dim = in_dim
        self.out_dim = in_dim - 1

    def forward(self, x):
        vvT = self.v.t() @ self.v + self.metric
        eig, U = torch.linalg.eigh(vvT.detach())
        U = U.narrow(1, 1, self.in_dim - 1)
        L = torch.diag(eig.narrow(0, 1, self.in_dim - 1).clamp(min=0.).sqrt())
        W = L @ U.t()
        x = torch.concat([x @ self.v.t(), x @ W.t()], dim=-1)
        return x