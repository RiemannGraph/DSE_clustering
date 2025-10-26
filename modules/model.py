import torch
import torch.nn as nn
from modules.layers import LSENetLayer
from utils.model_utils import select_activation


class LSENet(nn.Module):
    def __init__(self, manifold, in_dim, hid_dim, max_nums,
                 temperature=0.2, dropout=0.5, nonlin_str='relu'):
        super(LSENet, self).__init__()
        assert max_nums is not None
        self.manifold = manifold
        self.max_nums = max_nums  # [N_{H-1}, ..., N_1]
        self.height = len(max_nums) + 1

        # Project input to Lorentz space (d+1)
        self.input_proj = nn.Linear(in_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

        # Build layers bottom-up: layer[0] = leaf → level H-1; layer[-1] = level 1 → root
        self.layers = nn.ModuleList()
        curr_dim = hid_dim + 1  # +1 for time-like coordinate
        for i in range(self.height - 1):
            self.layers.append(LSENetLayer(
                manifold, curr_dim, hid_dim + 1, max_nums[i],
                dropout=dropout, temperature=temperature,
                nonlin=select_activation(nonlin_str)
            ))
            curr_dim = hid_dim + 1  # parent embedding dim

    def embed_leaf(self, x):
        # Map raw features to Lorentz leaf embedding
        x = self.dropout(self.input_proj(x))          # (N, d)
        o = torch.zeros_like(x[:, :1])                # (N, 1)
        x = torch.cat([o, x], dim=1)           # (N, d+1)
        x = self.manifold.expmap0(x)                  # project to Lorentz
        return x

    def forward(self, x, adj, freeze_levels=None):
        """
        Args:
            x: raw node features (N, D)
            adj: sparse adjacency
            freeze_levels: list of level indices to freeze (e.g., [0,1] for leader levels)
                - Level H = leaf (not a layer)
                - Layer i corresponds to assignment from level (H - i) → (H - i - 1)
                - So layer 0 → level H → H-1; layer -1 → level 1 → 0 (root)
        """
        z = self.embed_leaf(x)
        tree_coord_dict = {self.height: z}
        ass_dict = {}
        adj_dict = {self.height: adj}

        current_z = z
        current_adj = adj

        for i, layer in enumerate(self.layers):
            # Optional: skip gradient if this layer is frozen
            if freeze_levels is not None and (self.height - i - 1) in freeze_levels:
                with torch.no_grad():
                    z_par, adj_par, ass, z_curr = layer(current_z, current_adj)
            else:
                z_par, adj_par, ass, z_curr = layer(current_z, current_adj)

            level_curr = self.height - i
            level_par = self.height - i - 1
            tree_coord_dict[level_par] = z_par
            ass_dict[level_curr] = ass
            adj_dict[level_par] = adj_par

            current_z = z_par
            current_adj = adj_par

        # Root (level 0) is Frechet mean of level 1
        root = self.manifold.frechet_mean(current_z)
        tree_coord_dict[0] = root
        ass_dict[1] = torch.ones(current_z.size(0), 1, device=x.device)

        return tree_coord_dict, ass_dict, adj_dict