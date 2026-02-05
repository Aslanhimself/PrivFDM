import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

from omegaconf import OmegaConf

from ldm.privacy.schedules import linear_beta_schedule
# from src.utils import load_dataset_from_config

from scipy import optimize
from scipy.stats import norm
from math import sqrt
import numpy as np

from ldm.util import instantiate_from_config

# Dual between mu-GDP and (epsilon,delta)-DP
def delta_eps_mu(eps, mu):
    return norm.cdf(-eps / mu +
                    mu / 2) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2)


# inverse Dual
def eps_from_mu(mu, delta):

    def f(x):
        return delta_eps_mu(x, mu) - delta

    return optimize.root_scalar(f, bracket=[0, 500], method='brentq').root


def gdp_mech(sample_rate1, sample_rate2, niter1, niter2, sigma,
             alpha_cumprod_S, d, delta):
    mu_1 = sample_rate1 * sqrt(niter1 * (np.exp(4 * d * alpha_cumprod_S / (1 - alpha_cumprod_S)) - 1))
    mu_2 = sample_rate2 * sqrt(niter2 * (np.exp(1 / (sigma ** 2)) - 1))

    mu = sqrt(mu_1 ** 2 + mu_2 ** 2)
    epsilon = eps_from_mu(mu, delta)
    return epsilon


def eps_from_config(config):
    # dataset = load_dataset_from_config(config)

    d = config.model.params.image_size * config.model.params.image_size * config.model.params.channels

    # Instantiate Non-Private Training data
    data1 = instantiate_from_config(config.data)
    data1.batch_size = config.data.batch_size1
    data1.prepare_data()
    data1.setup()
    dataloader1 = data1.train_dataloader()

    # Instantiate Private Training data
    data2 = instantiate_from_config(config.data)
    data2.batch_size = config.data.batch_size2
    data2.prepare_data()
    data2.setup()
    dataloader2 = data2.train_dataloader()

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
        sigma=7.1,
        # sigma=config.dp.sigma,
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
