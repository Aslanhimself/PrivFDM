import math

import torch.nn as nn
from torch.nn.functional import softmax, log_softmax


def MLP(image_shape, num_classes=10):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(math.prod(image_shape), 100),
        nn.ReLU(),
        nn.Linear(100, num_classes),
    )
