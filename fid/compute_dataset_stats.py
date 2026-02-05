"""
This script computes the mean and covariance of the InceptionV3 activations on a
given dataset. A separate script should be used for computing the same
statistics on generated samples. To compute the FID, these stats can be combined
using a third script.
"""
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from omegaconf import OmegaConf
from pytorch_fid.inception import InceptionV3
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fid.cifar10_fid_stats_pytorch_fid import stats_from_dataloader, set_seeds
from ldm.util import instantiate_from_config


class DatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset.__getitem__(i)
        item = torch.as_tensor(item["image"])

        item = torch.clamp((item + 1.0) / 2.0, min=0.0, max=1.0)
        if item.dim() == 2:
            item = item.unsqueeze(-1)  # 变成 [H, W, 1]
            item = item.repeat(1, 1, 3)  # [H, W, 1] → [H, W, 3]

        item = item.permute(2, 0, 1)
        return item


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_config = OmegaConf.create({"target": args.dataset})
    for arg in args.args:
        key, value = arg.split(':')
        try:
            value = int(value)
        except ValueError:
            pass
        OmegaConf.update(dataset_config, "params." + key, value)
    dataset = instantiate_from_config(dataset_config)
    dataset = DatasetWrapper(dataset)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    inception_model = InceptionV3(normalize_input=False).to(device)
    mu, sigma = stats_from_dataloader(dataloader, inception_model, device)

    if args.output:
        np.savez(args.output, mu=mu, sigma=sigma)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('')
    parser.add_argument('--batch_size', type=int, default=500, help='Number of samples per batch')
    parser.add_argument('--dataset', type=str, help='Path to dataset class')
    parser.add_argument('--args', nargs='*', default=[], help='Additional dataset constructor arguments (param:value)')
    parser.add_argument('--output', type=str, help='Path to output FID stats (.npz)')
    parser.add_argument('--num_workers', type=int, default=0, help="Number of dataloader workers")
    args = parser.parse_args()

    if not args.output:
        print("[WARN]: --output not provided, generated stats will not be saved")

    set_seeds(0, 0)

    main(args)
