import torch
import numpy as np
import os
import random
import argparse
from exp import Exp
from logger import create_logger
import json
from utils.train_utils import DotDict


seed = 3047
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser(description='Lorentz Structural Entropy')

# Experiment settings
parser.add_argument('--dataset', type=str, default='Photo')
parser.add_argument('--task', type=str, default='Clustering',
                    choices=['Clustering'])
parser.add_argument('--root_path', type=str, default='datasets')
parser.add_argument('--eval_freq', type=int, default=2)
parser.add_argument('--exp_iters', type=int, default=5)
parser.add_argument('--version', type=str, default="run")
parser.add_argument('--log_path', type=str, default="./results/FootBall.log")

parser.add_argument('--epochs', type=int, default=1500)
parser.add_argument('--height', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--w_decay', type=float, default=1e-2)
parser.add_argument('--decay_rate', type=float, default=None)
parser.add_argument('--max_nums', type=int, nargs='+', default=[10], help="such as [50, 10]")
parser.add_argument('--embed_dim', type=int, default=512)
parser.add_argument('--hidden_dim_enc', type=int, default=512)
parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--nonlin', type=str, default="leaky_relu")
parser.add_argument('--temperature', type=float, default=0.9)
parser.add_argument('--n_cluster_trials', type=int, default=3)
parser.add_argument('--t', type=float, default=1., help='for Fermi-Dirac decoder')
parser.add_argument('--r', type=float, default=2., help='Fermi-Dirac decoder')

parser.add_argument('--L', type=int, default=2, help="Power of approximation Adjacency")
parser.add_argument('--omega', type=float, default=1e-2, help="omega of ridge regression penalty")
parser.add_argument('--alpha', type=float, default=0.1, help="rate of augmentation and origin")
parser.add_argument('--top_k_sim', type=int, default=20, help="Top k number of sparsify")
parser.add_argument('--top_k_aug', type=int, default=12, help="Top k number of sparsify")
parser.add_argument('--gamma', type=float, default=0.8, help="rate of tree contrastive loss")
parser.add_argument('--scale', type=float, default=0.15, help="scale of tree contrastive loss")
parser.add_argument('--cl_dim', type=int, default=513, help="dimension of contrastive views")
parser.add_argument('--tau', type=float, default=2.0, help="temperature for CL")
parser.add_argument("--ax_i", type=int, default=1)
parser.add_argument("--ax_j", type=int, default=2)
parser.add_argument("--epsInt", type=int, default=8)

parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
parser.add_argument('--save_path', type=str, default='model.pt')

# GPU
parser.add_argument('--use_gpu', action='store_false', help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--devices', type=str, default='0,1',
                    help='device ids of multiple gpus')


configs = parser.parse_args()
# with open(f'./configs/{configs.dataset}.json', 'wt') as f:
#     json.dump(vars(configs), f, indent=4)

configs_dict = vars(configs)
with open(f'./configs/{configs.dataset}.json', 'rt') as f:
    configs_dict.update(json.load(f))
configs = DotDict(configs_dict)
f.close()

log_path = f"./results/{configs.version}/{configs.dataset}.log"
configs.log_path = log_path
if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')
if not os.path.exists(f"./results"):
    os.mkdir("./results")
if not os.path.exists(f"./results/{configs.dataset}"):
    os.mkdir(f"./results/{configs.dataset}")
if not os.path.exists(f"./results/{configs.version}"):
    os.mkdir(f"./results/{configs.version}")
print(f"Log path: {configs.log_path}")
logger = create_logger(configs.log_path)
logger.info(configs)

exp = Exp(configs)
exp.train()
torch.cuda.empty_cache()
