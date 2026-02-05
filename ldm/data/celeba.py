import os
from torchvision.datasets import CelebA
from torchvision.transforms import Resize, CenterCrop, ToTensor, Compose


class CelebABase(CelebA):
    def __init__(self, datadir, config, **kwargs):
        cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        super().__init__(
            root=os.path.join(cachedir, datadir),
            # target_type=[],
            target_type='attr',
            transform=Compose([Resize(config.size), CenterCrop(config.size), ToTensor()]),
            download=False,
            **kwargs
        )

        # 找到 Male 属性的索引
        self.male_idx = self.attr_names.index("Male")

        self.targets = [int(attr[self.male_idx]) for attr in self.attr]

    def __getitem__(self, index):
        # image, _ = super().__getitem__(index)
        image, attr = super().__getitem__(index)
        # Rescale from [0, 1] to [-1, 1]
        image = (image * 2) - 1
        # Reshape from C x W x H to H x W x C
        image = image.permute(1, 2, 0).contiguous()

        # 取出 Male 属性 (0 or 1)
        label = attr[self.male_idx].long()

        # return {"image": image}
        return {
            "image": image,
            "class_label": label
        }


class CelebATrain(CelebABase):
    def __init__(self, **kwargs):
        super().__init__(datadir="CelebA", split="train", **kwargs)


class CelebAValidation(CelebABase):
    def __init__(self, **kwargs):
        super().__init__(datadir="CelebA", split="valid", **kwargs)
