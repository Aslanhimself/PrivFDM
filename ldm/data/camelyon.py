import os

from torch.utils.data import Dataset
from torchvision.transforms.functional import center_crop, to_tensor
from wilds import get_dataset


class CamelyonBase(Dataset):
    def __init__(self,  split="train", **kwargs):
        cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        root_dir = os.path.join(cachedir, "camelyon17")
        dataset = get_dataset(dataset="camelyon17", download=True, root_dir=root_dir)
        self.data = dataset.get_subset(split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, class_label, _ = self.data[index]
        image = to_tensor(image)
        image = center_crop(image, 32)
        image = image * 2 - 1
        return {
            "image": image.permute(1, 2, 0),
            "class_label": class_label.item()
        }


class CamelyonTrain(CamelyonBase):
    def __init__(self, **kwargs):
         super().__init__(split="train",  **kwargs)


class CamelyonVal(CamelyonBase):
    def __init__(self, **kwargs):
         super().__init__(split="test", **kwargs)
