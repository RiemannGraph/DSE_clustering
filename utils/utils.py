import torch
import torch.nn.functional as F


def grad_round(x):
    return torch.round(x) - x.detach() + x


def select_activation(activation):
    if activation == 'elu':
        return F.elu
    elif activation == 'relu':
        return F.relu
    elif activation == 'sigmoid':
        return F.sigmoid
    elif activation == 'tanh':
        return F.tanh
    elif activation == 'leaky_relu':
        return F.leaky_relu
    elif activation is None:
        return None
    else:
        raise NotImplementedError('the non_linear_function is not implemented')


def Frechet_mean_poincare(manifold, embeddings, weights=None, keepdim=False):
    z = manifold.from_poincare(embeddings)
    if weights is None:
        z = torch.sum(z, dim=0, keepdim=True)
    else:
        z = torch.sum(z * weights, dim=0, keepdim=keepdim)
    denorm = manifold.inner(None, z, keepdim=keepdim)
    denorm = denorm.abs().clamp_min(1e-8).sqrt()
    z = z / denorm
    z = manifold.to_poincare(z).to(embeddings.device)
    return z


"""
The codes of gumbel-softmax refers to
https://github.com/YongfeiYan/Gumbel_Softmax_VAE/blob/master/gumbel_softmax_vae.py
"""

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size()).to(logits.device)
    return torch.nn.functional.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=0.2, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


def graph_top_K(dense_adj, k):
    assert k < dense_adj.shape[-1]
    _, indices = dense_adj.topk(k=k, dim=-1)
    mask = torch.zeros(dense_adj.shape).bool().to(dense_adj.device)
    mask[torch.arange(dense_adj.shape[0])[:, None], indices] = True
    sparse_adj = torch.masked_fill(dense_adj, ~mask, value=0.).to_sparse_coo()
    return sparse_adj


def adjacency2index(adjacency, weight=False, topk=False, k=10):
    """_summary_

    Args:
        adjacency (torch.tensor): [N, N] matrix
    return:
        edge_index: [2, E]
        edge_weight: optional
    """
    if topk and k:
        adj = graph_top_K(adjacency, k)
    else:
        adj = adjacency
    edge_index = torch.nonzero(adj).t().contiguous()
    if weight:
        weight = adjacency[edge_index[0], edge_index[1]].reshape(-1)
        return edge_index, weight

    else:
        return edge_index


def index2adjacency(N, edge_index, weight=None, is_sparse=True):
    adjacency = torch.zeros(N, N).to(edge_index.device)
    m = edge_index.shape[1]
    if weight is None:
        adjacency[edge_index[0], edge_index[1]] = 1
    else:
        adjacency[edge_index[0], edge_index[1]] = weight.reshape(-1)
    adjacency = normalize_adj(adjacency)
    if is_sparse:
        weight = adjacency[edge_index[0], edge_index[1]]
        adjacency = torch.sparse_coo_tensor(indices=edge_index, values=weight, size=(N, N))
    return adjacency


def normalize_adj(adj, sparse=False):
    if sparse:
        adj = adj.coalesce()
        degree = adj.sum(1)
        d_idx = degree.indices().reshape(-1)
        inv_sqrt_degree = torch.zeros(degree.shape).to(adj.device)
        inv_sqrt_degree[d_idx] = 1. / (torch.sqrt(degree.values()))
        a_idx = adj.indices()
        div = inv_sqrt_degree[a_idx[0]] * inv_sqrt_degree[a_idx[1]]
        weight = adj.values() * div
        return torch.sparse_coo_tensor(adj.indices(), weight, adj.size())
    else:
        degree_matrix = 1. / (torch.sqrt(adj.sum(-1)) + 1e-10)
        return torch.diag(degree_matrix) @ adj @ torch.diag(degree_matrix)


def givens_rot_mat(i, j, theta: torch.Tensor, n):
    assert 0 <= i <= n - 1 and 0 <= j <= n - 1, "Invalid rotation axis"
    if isinstance(theta, float):
        theta = torch.tensor([theta])
    G = torch.eye(n).to(theta.device)
    c = torch.cos(theta)
    s = torch.sin(theta)
    G[i, i] = c
    G[j, j] = c
    G[i, j] = -s
    G[j, i] = s
    return G