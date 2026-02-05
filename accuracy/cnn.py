import torch.nn as nn


def CNN(in_ch=3, num_classes=10):
    return nn.Sequential(
        nn.Conv2d(in_ch, 32, 3, 1),
        nn.MaxPool2d(2, 2),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Conv2d(32, 64, 3, 1),
        nn.MaxPool2d(2, 2),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Conv2d(64, 128, 3, 1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(2048, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes),
    )
