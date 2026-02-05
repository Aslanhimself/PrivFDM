"""
This script samples from the given LatentDiffusion model with the DDIM sampler.
The samples are saved to a file with `torch.save` for use in other scripts, such
as FID computation.
"""
import argparse
import random

import numpy as np
from omegaconf import OmegaConf
import torch
from torchvision.utils import save_image

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentDiffusion


def main(args):
    # Set a random seed for reproducability
    print("[INFO]: Setting random seed to", args.seed)
    set_seeds(args.seed)

    # Load the model
    config = OmegaConf.load(args.yaml)
    model: LatentDiffusion = load_model_from_config(config, args.ckpt)
    shape = (model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size)

    # Get conditioning information for conditional models
    if args.classes is not None:
        samples_per_class = args.num_samples // len(args.classes)
        num_samples = samples_per_class * len(args.classes)
        if num_samples < args.num_samples:
            print(f"[WARN]: Changing --num_samples from {args.num_samples} to {num_samples}")
            args.num_samples = num_samples
        classes = torch.tensor(args.classes, device="cuda")
        cond = []
        for batch_classes in classes.split(args.batch_size):
            cond.append(model.get_learned_conditioning({model.cond_stage_key: batch_classes}))
        cond = torch.vstack(cond)
    elif args.prompts is not None:
        samples_per_class = 1
        prompts = [line.strip() for line in args.prompts.readlines()][:args.num_samples]
        args.prompts.close()
        if len(prompts) < len(args.num_samples):
            print(f"[WARN]: Changing --num_samples from {args.num_samples} to {len(prompts)}")
            args.num_samples = len(prompts)
        cond = []
        for i in range(len(prompts) + 1):
            cond.append(model.get_learned_conditioning(prompts[i*args.batch_size: (i+1)*args.batch_size]))
        cond = torch.vstack(cond)
    else:
        cond = None
    cond_list = (
        torch.arange(cond.size(0), device="cuda")
            .repeat_interleave(args.num_samples // cond.size(0))
            if cond is not None else None
    )

    # Sampling loop
    sampler = DDIMSampler(model)
    samples_list = list()
    i = 0
    while i < args.num_samples:
        batch_size = min(args.batch_size, args.num_samples - i)
        conditioning = cond[cond_list[i : i + batch_size]] if cond is not None else None
        samples, _ = sampler.sample(
            S=args.ddim_steps,
            batch_size=batch_size,
            shape=shape,
            conditioning=conditioning,
            eta=args.ddim_eta,
            verbose=False
        )
        samples = model.decode_first_stage(samples)
        samples_list.append(samples.cpu())
        i += batch_size

    all_samples = torch.vstack(samples_list)
    if args.output.endswith(".png"):
        save_image(all_samples[:100] * 0.5 + 0.5, args.output)
    else:
        dic = {'image': all_samples}
        if args.classes is not None:
            dic["class_label"] = classes[cond_list].cpu()
        elif args.prompts is not None:
            print("[WARN]: This script is not set up to save prompts in the output.")
        torch.save(dic, args.output)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model_from_config(config, ckpt):
    print(f"[INFO]: Loading model from {ckpt}")
    config.model.params.ckpt_path = ckpt
    model = instantiate_from_config(config.model)
    return model.cuda().eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    io_args = parser.add_argument_group("Input/output")
    io_args.add_argument("--yaml", required=True, help="Path to the config file for the model")
    io_args.add_argument("--ckpt", required=True, help="Path to the model checkpoint")
    io_args.add_argument("-o", "--output", required=True, help="Path to the output file (.pt|.png)")
    sampling_args = parser.add_argument_group("Sampling options")
    sampling_args.add_argument("--num_samples", type=int, default=5000, help="Total number of samples to generate")
    sampling_args.add_argument("--batch_size", type=int, default=500, help="Number of samples to generate per batch")
    sampling_args.add_argument("--seed", type=int, default=21, help="Number of DDIM steps")
    sampling_args.add_argument("--ddim_steps", type=int, default=200, help="Number of DDIM steps")
    sampling_args.add_argument("--ddim_eta", type=float, default=1.0, help="eta for DDIM sampling (0.0 yields deterministic sampling)")
    cond_args = parser.add_mutually_exclusive_group()
    cond_args.add_argument("--classes", type=eval, help="Python list of classes to generate images for")
    cond_args.add_argument("--prompts", type=open, help="Path to file containing prompts")
    args = parser.parse_args()
    main(args)
