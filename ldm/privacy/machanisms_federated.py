import os
import sys
import argparse
import torch
from torch.utils.data import Subset, DataLoader

from omegaconf import OmegaConf

from ldm.privacy.schedules import linear_beta_schedule
# from src.utils import load_dataset_from_config

from scipy import optimize
from scipy.stats import norm
from math import sqrt
import numpy as np

from ldm.util import instantiate_from_config
from pytorch_lightning import LightningDataModule


# Dual between mu-GDP and (epsilon,delta)-DP
def delta_eps_mu(eps, mu):
    return norm.cdf(-eps / mu +
                    mu / 2) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2)


# inverse Dual
def eps_from_mu(mu, delta):

    def f(x):
        return delta_eps_mu(x, mu) - delta

    return optimize.root_scalar(f, bracket=[0, 500], method='brentq').root


def gdp_mech(sample_rate1, sample_rate2, niter1, niter2, sigma, sigma_s, alpha_cumprod_S, d, delta):

    mu_1 = sample_rate1 * sqrt(niter1 * (np.exp(4 * d / (sigma_s ** 2)) - 1)) 
    mu_2 = sample_rate2 * sqrt(niter2 * (np.exp(1 / (sigma ** 2)) - 1))

    mu = sqrt(mu_1 ** 2 + mu_2 ** 2)
    epsilon = eps_from_mu(mu, delta)
    return epsilon


# Federated DataModule-dirichlet split non-IID
class FederatedDataModule(LightningDataModule):
    def __init__(self, base_datamodule, client_id, num_clients, batch_size=None, alpha=0.5):
        super().__init__()
        self.base_dm = base_datamodule
        self.client_id = client_id
        self.num_clients = num_clients
        self.batch_size = batch_size or base_datamodule.batch_size
        self.alpha = alpha  # Dirichlet

    def prepare_data(self):
        self.base_dm.prepare_data()

    def setup(self, stage=None):
        self.base_dm.setup(stage)
        full_train_dataset = self.base_dm.train_dataloader().dataset

        client_indices_list = self.dirichlet_split_noniid(
            dataset=full_train_dataset,
            num_clients=self.num_clients,
            alpha=self.alpha,
            seed=42,
        )
        self.train_dataset = Subset(full_train_dataset, client_indices_list[self.client_id])

        full_val_dataset = self.base_dm.val_dataloader().dataset
        val_len = len(full_val_dataset)
        val_indices = np.arange(val_len)
        np.random.shuffle(val_indices)
        val_client_size = val_len // self.num_clients
        val_start = self.client_id * val_client_size
        val_end = (self.client_id + 1) * val_client_size if self.client_id != self.num_clients - 1 else val_len
        self.val_dataset = Subset(full_val_dataset, val_indices[val_start:val_end])

    def dirichlet_split_noniid(self, dataset, num_clients, alpha=0.5, seed=42):
        np.random.seed(seed)
        labels = np.array(dataset.targets)
        num_classes = len(np.unique(labels))

        class_indices = [np.where(labels == y)[0] for y in range(num_classes)]
        for c in class_indices:
            np.random.shuffle(c)

        total_samples = sum(len(c) for c in class_indices)
        samples_per_client = total_samples // num_clients

        client_indices = [[] for _ in range(num_clients)]
        client_class_counts = [0 for _ in range(num_clients)]

        for class_id, idxs in enumerate(class_indices):
            proportions = np.random.dirichlet(alpha * np.ones(num_clients))
            
            proportions = np.clip(proportions, 1e-6, 1)
            proportions = proportions / proportions.sum()

            available_space = [samples_per_client - len(client_indices[i]) for i in range(num_clients)]
            class_allocation = (proportions * len(idxs)).astype(int)

            while class_allocation.sum() > len(idxs):
                class_allocation[np.argmax(class_allocation)] -= 1
            while class_allocation.sum() < len(idxs):
                class_allocation[np.argmin(class_allocation)] += 1

            idx_ptr = 0
            for client_id in range(num_clients):
                need = available_space[client_id]
                take = min(need, class_allocation[client_id])
                client_indices[client_id].extend(idxs[idx_ptr: idx_ptr + take])
                idx_ptr += take

        min_size = min(len(idxs) for idxs in client_indices)
        client_indices = [np.random.choice(idxs, samples_per_client, replace=False).tolist()
                          if len(idxs) > samples_per_client
                          else idxs + np.random.choice(idxs, samples_per_client - len(idxs)).tolist()
                          for idxs in client_indices]

        return client_indices

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


def eps_from_config(config):

    d = config.model.params.image_size * config.model.params.image_size * config.model.params.channels

    # Define number of clients
    num_clients = 6

    clients_data1 = []
    clients_data2 = []

    # Instantiate Non-Private Training data
    data1 = instantiate_from_config(config.data)
    data1.batch_size = config.data.batch_size1
    data1.prepare_data()
    data1.setup()
    # Instantiate Private Training data
    data2 = instantiate_from_config(config.data)
    data2.batch_size = config.data.batch_size2
    data2.prepare_data()
    data2.setup()

    # Distribute dataset for each client
    for i in range(num_clients):
        fed_data1 = FederatedDataModule(data1, client_id=i, num_clients=num_clients, batch_size=config.data.batch_size1, alpha=0.5)
        fed_data1.prepare_data()
        fed_data1.setup()
        clients_data1.append(fed_data1)

        fed_dm2 = FederatedDataModule(data2, client_id=i, num_clients=num_clients, batch_size=config.data.batch_size2, alpha=0.5)
        fed_dm2.prepare_data()
        fed_dm2.setup()
        clients_data2.append(fed_dm2)

    dataloader1 = clients_data1[0].train_dataloader()
    dataloader2 = clients_data2[0].train_dataloader()

    # dataloader1 = DataLoader(
    #     dataset,
    #     batch_size=config.train.batch_size1,
    # )
    #
    # dataloader2 = DataLoader(
    #     dataset,
    #     batch_size=config.train.batch_size2,
    # )

    prob1 = 1 / len(dataloader1)
    prob2 = 1 / len(dataloader2)
    niter1 = config.lightning.trainer.epochs1 * len(dataloader1)
    niter2 = config.lightning.trainer.epochs2 * len(dataloader2)

    betas = linear_beta_schedule(
        config.model.params.timesteps,
        config.model.params.linear_start,
        config.model.params.linear_end,
    )

    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alpha_cumprod_S = alphas_cumprod[config.model.params.s - 1].numpy()

    epsilon = gdp_mech(
        sample_rate1=prob1,
        sample_rate2=prob2,
        niter1=niter1,
        niter2=niter2,
        sigma=config.model.params.dp_config.noise_scale,
        sigma_s=config.model.params.dp_config.noise_scale_s,
        alpha_cumprod_S=alpha_cumprod_S,
        d=d,
        delta=config.model.params.dp_config.delta,
    )

    return epsilon


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )
    opt, _ = parser.parse_known_args()
    config = OmegaConf.load(opt.config)

    delta = config.model.params.dp_config.delta
    eps = eps_from_config(config)
    print(f"(epsilon, delta) = ({eps}, {delta})")

    sigma1 = config.model.params.dp_config.noise_scale_s
    sigma2 = config.model.params.dp_config.noise_scale

    print(f"(noise1, noise2) = ({sigma1}, {sigma2})")
