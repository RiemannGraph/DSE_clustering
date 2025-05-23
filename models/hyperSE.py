import torch
import torch.nn as nn
from utils.utils import gumbel_softmax
from models.layers import LorentzTransformation
from manifold.lorentz import Lorentz
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
        self.temperature = temperature

    def forward(self, data):
        features = data.x
        adj = data.adj.clone()
        coords, ass_mat, _ = self.encoder(features, adj)
        embeddings = {}
        for height, x in coords.items():
            embeddings[height] = x.detach()
        clu_mat = {self.height: torch.eye(self.num_nodes).to(data.x.device)}
        for k in range(self.height - 1, 0, -1):
            clu_mat[k] = clu_mat[k + 1] @ ass_mat[k + 1]
        for k, v in clu_mat.items():
            idx = v.max(1)[1]
            t = torch.zeros_like(v)
            t[torch.arange(t.shape[0]), idx] = 1.
            clu_mat[k] = t
        return embeddings, clu_mat
    
    def fix_cluster_results(self, clu_res_mat, embeds, epsInt: int = 7):
        clu_nums = clu_res_mat.sum(0)
        clu_res = clu_res_mat.argmax(1)
        corr_idx = clu_nums > epsInt
        if torch.all(corr_idx):
            return clu_res
        idx = torch.arange(clu_res_mat.shape[1]).to(clu_res.device)
        idx = idx[corr_idx]
        err_idx = torch.where(clu_res_mat[:, clu_nums <= epsInt] == 1.)[0]
        node = embeds[self.height]
        parent = embeds[1]
        error_node = node[err_idx]
        fixed_parent = parent[corr_idx]
        score = torch.log_softmax(2 + 2 * self.manifold.cinner(error_node, fixed_parent), dim=-1)
        fixed_res = gumbel_softmax(score, self.temperature)
        fixed_res = idx[fixed_res.argmax(1)]
        clu_res[err_idx] = fixed_res
        return clu_res

    def loss(self, data, scale=0.1, gamma=0.8):
        adj = data.adj.clone()
        aug_adj = data.aug_adj.clone()
        edge_index = data.edge_index
        x = data.x.clone()
        z, ass_mats, adj_set = self.encoder(x, adj)
        z_aug, ass_mats_aug, adj_aug_set = self.encoder(x, aug_adj)

        se_loss = self.se_loss(self.height, adj_set, ass_mats) + self.manifold.dist0(z[0])
        se_loss_aug = self.se_loss(self.height, adj_aug_set, ass_mats_aug) + self.manifold.dist0(z_aug[0])

        z = self.proj(z[self.height])
        z_aug = self.proj(z_aug[self.height])

        tree_cl_loss = self.tree_cl_loss(self.manifold, z, z_aug, edge_index, self.tau, neg_ratio=0.3) * scale

        return (1 - gamma) * (se_loss + se_loss_aug) + tree_cl_loss * gamma

    @staticmethod
    def tree_cl_loss(manifold, z1, z2, edge_index, tau=2.0, neg_ratio=0.1):
        device = z1.device
    
        pos_sim = torch.clamp(2.0 + 2.0 * manifold.cinner(z1, z2), min=-1.0, max=1.0)
        pos_exp = torch.exp(pos_sim / tau)
    
        row, col = edge_index
        num_edges = row.size(0)
        neg_sample_size = max(1, int(num_edges * neg_ratio))
    
        neg_idx = torch.randint(0, col.size(0), (neg_sample_size,), device=device)
        neg_row, neg_col = row[neg_idx], col[neg_idx]
    
        neg_sim = torch.clamp(2.0 + 2.0 * manifold.cinner(z1[neg_row], z2[neg_col]), min=-1.0, max=1.0)
        neg_exp = torch.exp(neg_sim / tau).sum()
    
        loss = -torch.log(pos_exp / (pos_exp + neg_exp))
        return loss.mean()


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