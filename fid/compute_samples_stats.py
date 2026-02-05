"""
This script computes the mean and covariance of the InceptionV3 activations on
generated samples. A separate script should be used for computing the same
statistics on the real dataset. To compute the FID, these stats can be combined
using a third script.
"""
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pytorch_fid.inception import InceptionV3
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fid.cifar10_fid_stats_pytorch_fid import stats_from_dataloader, set_seeds


class DatasetWrapper(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, i):
        image = self.images[i]
        assert image.shape[0] == 3 and image.shape[1] == image.shape[2], \
               f"Samples not in CxHxW format, instead got {image.shape}"
        # image = image.clamp(min=0, max=1)
        return image


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    images = torch.load(args.samples)["image"]

    if images.shape[2] == 3:
        images = images.view(-1, 3, 32, 32)

    elif images.shape[2] == 1:

        h, w = images.shape[-2], images.shape[-1]  
        images = images.view(-1, 1, h, w)
        images = images.repeat(1, 3, 1, 1)
        # # images = images.view(-1, 1, 32, 32)
        # images = images.view(-1, 1, 28, 28)
        # images = images.repeat(1, 3, 1, 1)

    dataset = DatasetWrapper(images)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size)

    inception_model = InceptionV3(normalize_input=False).to(device)
    mu, sigma = stats_from_dataloader(dataloader, inception_model, device)

    if args.output:
        np.savez(args.output, mu=mu, sigma=sigma)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('')
    parser.add_argument('--batch_size', type=int, default=500, help='Number of samples per batch')
    parser.add_argument('--samples', type=str, help='Path to samples class')
    parser.add_argument('--output', type=str, help='Path to output fid stats (.npz)')
    args = parser.parse_args()

    if not args.output:
        print("[WARN]: --output not provided, generated stats will not be saved")

    set_seeds(0, 0)

    main(args)
