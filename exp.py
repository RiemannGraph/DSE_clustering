import torch
import numpy as np
from models.hyperSE import HyperSE
from geoopt.optim import RiemannianAdam
from utils.eval_utils import cluster_metrics
from dataset import load_data
from logger import create_logger


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
            model = HyperSE(in_features=data.x.shape[1],
                            hidden_features=self.configs.hidden_dim,
                            hidden_dim_enc=self.configs.hidden_dim_enc,
                            num_nodes=data.x.shape[0],
                            height=self.configs.height,
                            temperature=self.configs.temperature,
                            embed_dim=self.configs.embed_dim,
                            cl_dim=self.configs.cl_dim,
                            dropout=self.configs.dropout,
                            nonlin=self.configs.nonlin,
                            decay_rate=self.configs.decay_rate,
                            max_nums=self.configs.max_nums,
                            tau=self.configs.tau).to(device)
            optimizer = RiemannianAdam(model.parameters(), lr=self.configs.lr, weight_decay=self.configs.w_decay)
            if self.configs.task == 'Clustering':
                nmi, ari = self.train_clu(data, model, optimizer, logger, device, exp_iter)
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
        epoch_nmi = []
        epoch_ari = []

        for epoch in range(1, self.configs.epochs + 1):
            model.train()
            loss = model.loss(data, self.configs.scale, self.configs.gamma)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.info(f"Epoch {epoch}: train_loss={loss.item()}")
            if epoch % self.configs.eval_freq == 0:
                logger.info("-----------------------Evaluation Start---------------------")
                model.eval()
                embeddings, clu_mat = model(data)
                predicts = model.fix_cluster_results(clu_mat[1], embeddings, self.configs.epsInt).cpu().numpy()
                trues = data.y.cpu().numpy()
                nmi, ari = [], []
                for step in range(n_cluster_trials):
                    metrics = cluster_metrics(trues, predicts)
                    nmi_, ari_ = metrics.evaluateFromLabel(use_acc=False)
                    nmi.append(nmi_)
                    ari.append(ari_)
                nmi, ari = np.mean(nmi), np.mean(ari)

                epoch_nmi.append(nmi)
                epoch_ari.append(ari)

                if nmi > best_cluster['nmi']:
                    best_cluster['nmi'] = nmi
                    best_cluster_result['nmi'] = [nmi, ari]
                    logger.info('------------------Saving best model-------------------')
                    torch.save(model.state_dict(), f"./checkpoints/{self.configs.save_path}")
                if ari > best_cluster['ari']:
                    best_cluster['ari'] = ari
                    best_cluster_result['ari'] = [nmi, ari]
                logger.info(
                    f"Epoch {epoch}: NMI: {nmi}, ARI: {ari}")
                logger.info(
                    "-------------------------------------------------------------------------")

        for k, result in best_cluster_result.items():
            nmi, ari = result
            logger.info(
                f"Best Results according to {k}: NMI: {nmi}, ARI: {ari} \n")
        return best_cluster['nmi'], best_cluster["ari"]