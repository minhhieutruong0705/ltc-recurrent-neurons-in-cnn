import os
import csv
import numpy as np
import cv2
import torch

from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from torch.utils.data import Dataset


def read_list(path):
    with open(path, 'r') as f:
        lines = []
        reader = csv.DictReader(f)
        for line in reader:
            lines.append({
                "image_name": f"{line['image']}.jpeg",
                "label": float(line["level"])
            })
    return lines


class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, image_dir, data_list, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.transform = transform

        # get images with their labels
        self.data_instances = read_list(data_list)

        # print data size
        print("[INFO] data size:", self.__len__())
        # print instance count of each class
        print("[INFO] data stats:")
        instance_count = self.__count_instances__()
        print(f"[INFO] {len(instance_count)} classes in total")
        for class_id in sorted(instance_count.keys()):
            print(f"\t{int(class_id)}: {instance_count[class_id]}")

    def __len__(self):
        return len(self.data_instances)

    def __count_instances__(self):
        instance_count = {}
        for data_instance in self.data_instances:
            if data_instance["label"] not in instance_count.keys():
                instance_count.update({data_instance["label"]: 0})
            instance_count[data_instance["label"]] += 1
        return instance_count

    def get_reverse_class_weight(self):
        instance_count = self.__count_instances__()
        rev_weight = [instance_count[class_id] for class_id in sorted(instance_count.keys())]
        return max(rev_weight) / np.array(rev_weight)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.data_instances[index]["image_name"])
        image = np.array(Image.open(image_path).convert("RGB"))
        label = self.data_instances[index]["label"]

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image, label


if __name__ == '__main__':
    # test dataset implementation
    img_dim = 256
    img_crop_dim = 224
    train_transformer = A.Compose([
        A.Resize(height=img_dim, width=img_dim, interpolation=cv2.INTER_AREA),
        A.RandomResizedCrop(height=img_crop_dim, width=img_crop_dim, scale=(0.7, 1.0), interpolation=cv2.INTER_AREA),
        A.HorizontalFlip(),
        A.Rotate(limit=15),
        A.GaussNoise(),
        A.GaussianBlur(blur_limit=(3, 7)),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ), ToTensorV2()
    ])

    retino_train_dataset = DiabeticRetinopathyDataset(
        image_dir="../../datasets/Dataset_DiabeticRetinopathy/train",
        data_list="../../datasets/Dataset_DiabeticRetinopathy/trainLabels.csv",
        transform=train_transformer
    )

    image, label = retino_train_dataset.__getitem__(6)
    transforms.ToPILImage()(image).show()
    print(label)

    rev_class_weight = retino_train_dataset.get_reverse_class_weight()
    print(rev_class_weight)
