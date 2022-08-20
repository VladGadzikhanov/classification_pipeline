import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose, RandomAffine, RandomHorizontalFlip, RandomRotation, RandomVerticalFlip


class CollectionAttention(Dataset):
    def __init__(self, df, phase):
        self.df = df
        self.transforms = self._transforms(phase)
        self.objs = df.groupby("obj_id")["target"].max().reset_index()

    def __getitem__(self, idx):
        obj = self.objs.iloc[idx]
        img_paths = self.df.loc[self.df["obj_id"] == obj["obj_id"], "img_path"]
        # sort values by camera name
        # imgs = imgs.sort_values()
        imgs = [self.transforms(Image.open(img)) for img in img_paths]
        imgs = torch.stack(imgs)
        obj_id = obj["obj_id"]
        return obj_id, "no_path", imgs, torch.tensor(obj["target"])  # 0 in the end is used to have output of size 4

    def __len__(self) -> int:
        return len(self.objs)

    @staticmethod
    def _transforms(phase):
        list_transforms = []

        if phase == "train":
            list_transforms.extend(
                [
                    RandomHorizontalFlip(),
                    RandomVerticalFlip(),
                    RandomAffine(0, translate=(0.1, 0.1)),
                    RandomRotation(90),
                ]
            )

        list_transforms.extend(
            [
                transforms.CenterCrop(300),
                transforms.Resize(224),
                # transforms.Grayscale(),
                transforms.ToTensor(),
            ]
        )
        return Compose(list_transforms)
