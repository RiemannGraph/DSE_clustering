import torch
import numpy as np
from modules.dsi import DSI
from geoopt.optim import RiemannianAdam
from utils.eval_utils import cluster_metrics
from data import load_data
from logger import create_logger
from torch.optim.lr_scheduler import LambdaLR
import math


def get_lr_scheduler(optimizer, total_epochs=2000, warmup_epochs=200, base_lr=0.0005):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup from 0 to base_lr
            return float(epoch) / float(max(1, warmup_epochs))
        else:
            # Cosine decay to 0
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


class Exp:
    def __init__(self, configs):
        self.configs = configs
        if self.configs.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

    def train(self):
        logger = create_logger(self.configs.log_path)
        device = self.device
        data = load_data(self.configs).to(device)

        total_nmi = []
        total_ari = []
        for exp_iter in range(self.configs.exp_iters):
            logger.info(f"\ntrain iters {exp_iter}")
            model = DSI(in_dim=data.x.shape[1],
                        hid_dim=self.configs.hid_dim,
                        num_nodes=data.x.shape[0],
                        temperature=self.configs.temperature,
                        dropout=self.configs.dropout,
                        nonlin_str=self.configs.nonlin,
                        max_nums=self.configs.max_nums).to(device)
            optimizer = RiemannianAdam(model.parameters(), lr=self.configs.lr, weight_decay=self.configs.w_decay)
            if self.configs.task == 'Clustering':
                nmi, ari = self.train_clu(data, model, optimizer, logger)
                total_nmi.append(nmi)
                total_ari.append(ari)

        if self.configs.task == 'Clustering':
            logger.info(f"NMI: {np.mean(total_nmi)}+-{np.std(total_nmi)}, "
                        f"ARI: {np.mean(total_ari)}+-{np.std(total_ari)}")

    def train_clu(self, data, model, optimizer, logger):
        best_cluster_result = {}
        best_cluster = {'nmi': 0, 'ari': 0}

        logger.info("--------------------------Training Start-------------------------")
        n_cluster_trials = self.configs.n_cluster_trials
        epoch_acc = []
        epoch_nmi = []
        epoch_ari = []

        # scheduler = get_lr_scheduler(optimizer, total_epochs=4000, warmup_epochs=400)
        for epoch in range(1, self.configs.epochs + 1):
            model.train()

            loss = model.hybird_loss(data, None, self.configs.gamma, self.configs.scale)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.info(f"Epoch {epoch}: loss={loss.item()}")

            leader_levels = [0]
            follower_levels = list(range(2, model.height + 1))

            # ===== 阶段1: 更新 Follower =====
            _, ass_dict, adj_dict = model(data, freeze_levels=leader_levels)
            loss_follower = model.se_loss(ass_dict, adj_dict)
            loss_follower.backward()
            optimizer.step()

            # ===== 阶段2: 更新 Leader =====
            optimizer.zero_grad()
            _, ass_dict, adj_dict = model(data, freeze_levels=follower_levels)
            loss_leader = model.se_loss(ass_dict, adj_dict)
            loss_leader.backward()
            optimizer.step()

            logger.info(f"Epoch {epoch}: leader_loss={loss_leader.item()}, follower_loss={loss_follower.item()}")

            _, ass_dict, adj_dict = model(data)
            loss = model.se_loss(ass_dict, adj_dict)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.info(f"Epoch {epoch}: loss={loss.item()}")

            # scheduler.step()

            if epoch % self.configs.eval_freq == 0:
                logger.info("-----------------------Evaluation Start---------------------")
                model.eval()
                embed_dict, clu_mat_dict = model.get_cluster_results(data)
                predicts = model.fix_cluster_results(clu_mat_dict[1], embed_dict, self.configs.epsInt).cpu().numpy()
                trues = data.y.cpu().numpy()
                acc, nmi, ari = [], [], []
                for step in range(n_cluster_trials):
                    metrics = cluster_metrics(trues, predicts)
                    acc_, nmi_, ari_ = metrics.evaluateFromLabel(use_acc=True)
                    acc.append(acc_)
                    nmi.append(nmi_)
                    ari.append(ari_)
                acc, nmi, ari = np.mean(acc), np.mean(nmi), np.mean(ari)

                epoch_acc.append(acc)
                epoch_nmi.append(nmi)
                epoch_ari.append(ari)

                if nmi > best_cluster['nmi']:
                    best_cluster['nmi'] = nmi
                    best_cluster_result['nmi'] = [nmi, ari]
                    logger.info('------------------Saving best model-------------------')
                    torch.save(model.state_dict(), f"./checkpoints/{self.configs.save_path}")
                logger.info(
                    f"Epoch {epoch}: ACC: {acc * 100: .2f}, NMI: {nmi * 100: .2f}, ARI: {ari * 100: .2f}")
                logger.info(
                    "-------------------------------------------------------------------------")

        for k, result in best_cluster_result.items():
            nmi, ari = result
            logger.info(
                f"Best Results according to {k}: ACC: {acc * 100: .2f}, NMI: {nmi * 100: .2f}, ARI: {ari * 100: .2f} \n")
        return best_cluster['nmi'], best_cluster["ari"]