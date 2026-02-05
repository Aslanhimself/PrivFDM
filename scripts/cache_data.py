import argparse
import ast

from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
import tqdm

from ldm.util import instantiate_from_config


def make_buffer(x, N):
    if isinstance(x, torch.Tensor):
        return torch.empty(N, *x.shape, dtype=x.dtype)
    elif isinstance(x, int):
        return torch.empty(N, dtype=torch.int32)
    else:
        return [None] * N


def main(args):
    dataset_config = OmegaConf.create({"target": args.dataset})
    dataset_config.params = ast.literal_eval(args.params)
    dataset = instantiate_from_config(dataset_config)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    result = {k: make_buffer(v, len(dataset)) for k, v in dataset[0].items()}
    
    i = 0
    for batch in tqdm.tqdm(dataloader):
        batch_size = batch["image"].size(0)
        for k, v in batch.items():
            result[k][i : i + batch_size] = v
        i += batch_size

    torch.save(result, args.output)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dotted path to the dataset class")
    parser.add_argument("--params", default="{}", help="A python dict with param:value entries")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--output", required=True, help="Path to output file (.pt)")
    args = parser.parse_args()
    main(args)