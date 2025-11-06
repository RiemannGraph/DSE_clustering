import torch
import torch.nn as nn
from utils.model_utils import gumbel_softmax, graph_top_K
from manifold.lorentz import Lorentz
from modules.layers import LorentzTransformation
from modules.model import LSENet

MIN_NORM = 1e-15
EPS = 1e-6


class DSI(nn.Module):
    def __init__(self, in_dim, hid_dim, num_nodes, max_nums, temperature=0.2,
                 dropout=0.5, nonlin_str='relu', tau=1.0):
        super(DSI, self).__init__()
        self.num_nodes = num_nodes
        self.height = len(max_nums) + 1
        self.manifold = Lorentz()
        self.encoder = LSENet(self.manifold, in_dim, hid_dim, max_nums, temperature, dropout, nonlin_str)
        self.lorentz_proj = LorentzTransformation(hid_dim + 1)
        self.temperature = temperature
        self.tau = tau

    def forward(self, data):
        features = data.x
        adj = data.adj.clone()
        tree_coord_dict, ass_dict, adj_dict = self.encoder(features, adj)
        return tree_coord_dict, ass_dict, adj_dict

    def get_cluster_results(self, data):
        features = data.x
        adj = data.adj.clone()
        coord_dict, ass_dict, _ = self.encoder(features, adj)
        embed_dict = {}
        for height, x in coord_dict.items():
            embed_dict[height] = x.detach()
        clu_mat_dict = {self.height: torch.eye(self.num_nodes).to(data.x.device)}
        for k in range(self.height - 1, 0, -1):
            clu_mat_dict[k] = clu_mat_dict[k + 1] @ ass_dict[k + 1]
        for k, v in clu_mat_dict.items():
            idx = v.max(1)[1]
            t = torch.zeros_like(v)
            t[torch.arange(t.shape[0]), idx] = 1.
            clu_mat_dict[k] = t
        return embed_dict, clu_mat_dict

    def fix_cluster_results(self, clu_res_mat, embed_dict, epsInt: int = 7):
        clu_nums = clu_res_mat.sum(0)
        clu_res = clu_res_mat.argmax(1)
        corr_idx = clu_nums > epsInt
        if torch.all(corr_idx):
            return clu_res
        idx = torch.arange(clu_res_mat.shape[1]).to(clu_res.device)
        idx = idx[corr_idx]
        err_idx = torch.where(clu_res_mat[:, clu_nums <= epsInt] == 1.)[0]
        node = embed_dict[self.height]
        parent = embed_dict[1]
        error_node = node[err_idx]
        fixed_parent = parent[corr_idx]
        score = torch.log_softmax(2 + 2 * self.manifold.cinner(error_node, fixed_parent), dim=-1)
        fixed_res = gumbel_softmax(score, self.temperature)
        fixed_res = idx[fixed_res.argmax(1)]
        clu_res[err_idx] = fixed_res
        return clu_res

    def se_loss(self, data, eps=1e-6):
        z_leaf = self.encoder.embed_leaf(data.x, data.adj)
        # z_leaf = self.lorentz_proj(z_leaf)
        neg_dist2 = 2 + 2 * self.manifold.cinner(z_leaf, z_leaf)
        adj_aug = graph_top_K(torch.softmax(neg_dist2/ self.tau, dim=-1), k=8)
        tree_coord_aug_dict, ass_aug_dict, adj_aug_dict = self.encoder(data.x, 0.01 * adj_aug + data.adj)
        loss = self._si_loss(ass_aug_dict, adj_aug_dict, eps)
        return loss

    def _si_loss(self, ass_dict: dict, adj_dict: dict, eps: float = 1e-6):
        se_loss = 0
        vol_G = adj_dict[self.height].sum()

        for k in range(self.height, 0, -1):
            adj_dense = adj_dict[k].to_dense()
            degree = adj_dense.sum(dim=1)
            diag = adj_dense.diag()
            if k == 1:
                vol_parent = vol_G
            else:
                vol_parent = adj_dict[k - 1].to_dense().sum(dim=-1)
                vol_parent = torch.einsum('ij, j->i', ass_dict[k], vol_parent)
            delta_vol = degree - diag
            log_vol_ratio_k = torch.log2((degree + eps) / (vol_parent + eps))
            se_loss += torch.sum(delta_vol * log_vol_ratio_k)
        se_loss = -1 / vol_G * se_loss
        return se_loss