import os
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


class CIFAR10Base(CIFAR10):
    def __init__(self, **kwargs):
        cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        super().__init__(
            root=os.path.join(cachedir, "CIFAR10"),
            transform=ToTensor(),
            **kwargs
        )

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        image = image * 2 - 1
        image = image.permute(1, 2, 0).contiguous()
        return {
            "image": image,
            "class_label": label
        }


class CIFAR10Train(CIFAR10Base):
    def __init__(self, **kwargs):
        super().__init__(train=True, download=True, **kwargs)


class CIFAR10Val(CIFAR10Base):
    def __init__(self, **kwargs):
        super().__init__(train=False, download=True, **kwargs)
