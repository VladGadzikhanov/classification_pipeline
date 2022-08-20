import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose, RandomAffine, RandomHorizontalFlip, RandomRotation, RandomVerticalFlip


class CollectionSeparate(Dataset):
    def __init__(self, df, phase: str):
        self.df = df
        self.transforms = self._transforms(phase)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self.transforms(Image.open(row["img_path"]).convert("RGB"))
        return row["obj_id"], row["img_path"], img, torch.tensor(self.df.iloc[idx]["target"])

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _transforms(phase: str):
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
                transforms.Resize((400,400)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        return Compose(list_transforms)
