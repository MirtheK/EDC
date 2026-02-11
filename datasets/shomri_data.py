from pathlib import Path
from typing import Any
import pandas as pd

import numpy as np
import torch

from monai.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    Resized,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    RandAffined,
    ToTensord,
    ResizeWithPadOrCropd,
    EnsureTyped
)


class SHOMRIDataset(Dataset):
    """3D NIfTI dataset with MONAI for anomalib."""

    def __init__(
        self,
        root: str | Path,
        split: str | Split,
        augmentations: Compose | None = None,
        image_size: tuple[int, int, int] = (128, 128, 128),
    ):
        super().__init__(augmentations=augmentations)

        self.root = validate_path(root)
        self.split = Split(split)
        self.image_size = image_size

        self.samples = make_shomri_dataset(self.root, self.split)

        self.transforms = self._build_transforms()

        self.monai_dataset = MonaiDataset(
            data=self.samples.to_dict(orient="records"),
            transform=self.transforms,
        )

    def _build_transforms(self):
        keys = ["image"]
        if self.samples.attrs["task"] == "segmentation":
            keys.append("mask")

        return Compose(
            [
                LoadImaged(keys=keys),
                EnsureChannelFirstd(keys=keys),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=0,
                    a_max=600,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                ResizeWithPadOrCropd(
                    keys=keys,
                    spatial_size=self.image_size,
                ),
                EnsureTyped(keys=keys),
            ]
        )

    def __len__(self):
        return len(self.monai_dataset)

    def __getitem__(self, index):
        sample = self.monai_dataset[index]

        out = {
            "image": sample["image"],
            "label": sample["label"],
            "image_path": sample["image_path"],
        }

        if "mask" in sample:
            out["mask"] = sample["mask"]
            out["mask_path"] = sample["mask_path"]

        return out


def make_shomri_dataset(root: Path, split: Split) -> pd.DataFrame:
    records = []

    root = Path(root)

    if split == Split.TRAIN:
        image_dir = root / "train" / "NORMAL"
        for img in image_dir.glob("*.nii.gz"):
            records.append(
                {
                    "image_path": str(img),
                    "mask_path": None,
                    "label": "NORMAL",
                    "label_index": LabelName.NORMAL,
                    "split": "train",
                }
            )

    else:  # TEST
        for label_name in ["NORMAL", "ABNORMAL"]:
            image_dir = root / "test" / label_name
            for img in image_dir.glob("*.nii.gz"):
                record = {
                    "image_path": str(img),
                    "label": label_name,
                    "label_index": (
                        LabelName.NORMAL if label_name == "NORMAL" else LabelName.ABNORMAL
                     ),
                    "split": "test",
                    "mask_path": None,
                }

                if label_name == "ABNORMAL":
                    mask_path = root / "segmentations" / "ABNORMAL" / img.name
                    if not mask_path.exists():
                        raise MisMatchError(
                            f"Missing mask for abnormal image: {img.name}"
                        )
                    record["mask_path"] = str(mask_path)

                records.append(record)

    samples = pd.DataFrame(records)
    samples["label_index"] = samples["label_index"].astype(int)

    samples.attrs["task"] = "segmentation"
    return samples.reset_index(drop=True)
