import argparse
import random

from ema_pytorch import EMA
import numpy as np
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip
import tqdm

from accuracy.cnn import CNN
from accuracy.mlp import MLP
from accuracy.wideresnet import Wide_ResNet
from ldm.data.in_memory_dataset import InMemoryDataset
from ldm.util import instantiate_from_config


class LdmDatasetWrapper(Dataset):
    def __init__(self, dataset, augmentations=[], augmult=1, permute=True, mean=None):
        self.dataset = dataset
        self.permute = permute
        self.augmult = augmult
        self.augment_fn = Compose(augmentations)
        if mean is not None:
            self.mean = mean
        else:
            self.mean = torch.zeros_like(self.dataset[0]["image"])
            if self.permute: self.mean = self.mean.permute(2, 0, 1)
            for data in self.dataset:
                image = data["image"] * 0.5 + 0.5
                if self.permute: image = image.permute(2, 0, 1)
                self.mean += (data["image"] * 0.5 + 0.5) / len(self.dataset)
    def __len__(self):
        return self.dataset.__len__() * self.augmult
    def __getitem__(self, index):
        data = self.dataset.__getitem__(index // self.augmult)
        image = data["image"] * 0.5 + 0.5
        if self.permute: image = image.permute(2, 0, 1)
        image = image - self.mean
        image = self.augment_fn(image)
        label = data["class_label"]
        return image, label


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)


    # Load data
    train_dset = LdmDatasetWrapper(
        InMemoryDataset(args.train_data),
        augmentations=[
            RandomHorizontalFlip(),
            RandomCrop(32, padding=4, padding_mode="reflect")
        ],
        augmult=2,
        permute=False
    )
    val_dset = LdmDatasetWrapper(
        instantiate_from_config({"target": args.val_data, "params": args.val_data_args}),
        mean=train_dset.mean
    )
    train_loader = DataLoader(
        train_dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=torch.cuda.is_available()
    )

    # Create model
    model = load_model(args.model).to(device)
    
    # Create EMA model
    ema = EMA(model, beta=args.ema_beta, update_after_step=100, update_every=10)

    # Create optimizer, learning rate scheduler, and loss
    optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.lr_anneal_period, T_mult=args.lr_anneal_mult)
    loss_fn = CrossEntropyLoss()

    # Training loop
    train_losses = torch.empty(args.num_epochs, len(train_loader))
    val_epochs = list(range(args.num_epochs-1, -1, -args.val_period)[::-1])
    val_accuracies = []
    ema_val_accuracies = []
    epoch_pbar = tqdm.trange(args.num_epochs, disable=args.quiet, desc="Epoch", leave=True)
    for epoch in epoch_pbar:
        train_pbar = tqdm.tqdm(train_loader, disable=args.quiet, desc="Train", leave=False)
        train_losses[epoch] = train(model, train_pbar, optimizer, lr_scheduler, loss_fn, epoch, device)
        optimizer.zero_grad(set_to_none=True)
        ema.update()
        if epoch in val_epochs:
            val_pbar = tqdm.tqdm(val_loader, disable=args.quiet, desc="Valid", leave=False)
            val_accuracies.append(validate(model, val_pbar, device))
            ema_val_accuracies.append(validate(ema, val_pbar, device))

        if args.output:
            torch.save({"train_losses": train_losses,
                        "val_accuracies": val_accuracies,
                        "ema_val_accuracies": ema_val_accuracies,
                        "val_epochs": val_epochs},
                       args.output)
    
    # Print the best validation accuracies
    best_epoch = torch.Tensor(val_accuracies).argmax().item()
    best_ema_epoch = torch.Tensor(ema_val_accuracies).argmax().item()
    print(f"Model: epoch {best_epoch:03}: {val_accuracies[best_epoch]:0.3f}")
    print(f"EMA  : epoch {best_ema_epoch:03}: {ema_val_accuracies[best_ema_epoch]:0.3f}")


def load_model(model_type):
    if model_type == "cnn":
        return CNN(in_ch=args.image_channels, num_classes=args.num_classes)
    elif model_type == "mlp":
        return MLP((args.image_channels, 32, 32), num_classes=args.num_classes)
    elif model_type == "wrn_40_4":
        return Wide_ResNet(40, 4, 0.1, args.num_classes)
    else:
        raise NotImplementedError("Unsupported model type " + model_type)


def train(model, batch_pbar, optimizer, lr_scheduler, loss_fn, epoch, device):
    model.train()
    epoch_losses = torch.empty(len(batch_pbar))
    for i, (x, y) in enumerate(batch_pbar):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        lr_scheduler.step(epoch + i / len(batch_pbar))
        batch_pbar.set_postfix({"loss": loss.item()})
        epoch_losses[i] = loss.item()
    return epoch_losses


@torch.no_grad()
def validate(model, batch_pbar, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    for x, y in batch_pbar:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        pred = outputs.argmax(dim=1)
        num_correct = (pred == y).sum().item()
        total_correct += num_correct
        total_samples += y.size(0)
        batch_pbar.set_postfix({"acc": num_correct / y.size(0)})
    return total_correct / total_samples
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    data_args = parser.add_argument_group("Training and Validation Data")
    data_args.add_argument("--train_data", required=True, help="Path to synthetic data file (.pt)")
    data_args.add_argument("--val_data", required=True, help="Classpath to validation data")
    data_args.add_argument("--val_data_args", type=eval, default="{}", help="Dict of kwargs passed to the val dataset")
    model_args = parser.add_argument_group("Classification Model")
    model_args.add_argument("--model", required=True, choices=["cnn", "mlp", "wrn_40_4"], help="The classifier model")
    model_args.add_argument("--num_classes", type=int, default=10, help="Number of classes")
    model_args.add_argument("--image_channels", type=int, default=3, help="Number of channels")
    train_args = parser.add_argument_group("Training options and Hyperparameters")
    train_args.add_argument("--num_epochs", type=int, default=200, help="Number of epochs to train for")
    train_args.add_argument("--batch_size", "--bs", type=int, default=128, help="Train batch size")
    train_args.add_argument("--val_batch_size", "--val_bs", type=int, default=None, help="Validation batch size")
    train_args.add_argument("--learning_rate", "--lr", type=float, default=0.1, help="Initial learning rate")
    train_args.add_argument("--lr_anneal_period", "--T_0", type=int, default=10, help="SGDR T_0 parameter")
    train_args.add_argument("--lr_anneal_mult", "--T_mult", type=int, default=2, help="SGDR T_mult parameter")
    train_args.add_argument("--ema_beta", type=float, default=0.9999, help="EMA decay parameter")
    other_args = parser.add_argument_group("Other Options")
    other_args.add_argument("--quiet", action="store_true", help="Disables most print statements")
    other_args.add_argument("--val_period", type=int, default=1, help="Number of training epochs per val epoch")
    other_args.add_argument("--output", help="Path to output training information (.pt)")
    other_args.add_argument("--seed", type=int, default=21, help="Sets the seed of as many PRNGS as we can")
    args = parser.parse_args()
    if args.val_batch_size is None: args.val_batch_size = args.batch_size
    main(args)
