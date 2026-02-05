import torch
from torch.utils.data import Dataset


def maybe_item(x):
    if isinstance(x, torch.Tensor) and x.nelement() == 1:
        return x.item()
    return x


class InMemoryDataset(Dataset):
    """Dataset which loads all examples into CPU memory.

    IMPORTANT: This function assumes that the data given to it is a dictionary
               containing at least the key `image` mapping to a (N, ...) tensor
               of images. Any additional keys (e.g. `class_label`) must also map
               to tensors of shape (N, ...).

    Args:
        path: Path to the dataset file (.pt)
        transform_data: A function that takes in data and returns a transformed
                        version, applied to all data when initially loaded
        transform_item: A function that takes in data and returns a transformed
                        version, applied to each item on retrieval
    """

    def __init__(self, path, *, transform=lambda x: x):
        self._data = torch.load(path, map_location="cpu")
        self.transform = transform

    def __len__(self):
        return self._data["image"].size(0)

    def __getitem__(self, index):
        result = {}
        if "metadata" in self._data:
            metadata = self._data["metadata"][index]
            offset = metadata[0].item()
            nelem = metadata[1].item()
            image_shape = metadata[2:]
            result["image"] = self._data["image"][offset: offset + nelem].reshape(image_shape)
        else:
            result["image"] = self._data["image"][index]

        if "class_label" in self._data:
            result["class_label"] = self._data["class_label"][index].item()
        return result
