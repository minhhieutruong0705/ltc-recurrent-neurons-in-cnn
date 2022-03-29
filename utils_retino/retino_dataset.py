import os
import csv
import numpy as np
import cv2

from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from torch.utils.data import Dataset


def read_list(path):
    lines = []
    with open(path, 'r') as f:
        csv_reader = csv.DictReader(f)
        for line in csv_reader:
            lines.append({
                "image_name": f"{line['image']}.jpeg",
                "label": int(line["level"])
            })
    return lines


class DiabeticRetinopathyDataset(Dataset):
    def __init__(self, image_dir, data_list, data_balance=False, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.transform = transform

        # get images with their labels
        self.data_instances = read_list(data_list)
        self.sorted_instances = self.__sort_instances__()  # sort instances

        print("[INFO] Dataset stats")
        self.__print_dataset_stats()

        # sub-sample class No-DR
        self.instance_count = self.__count_instances__()
        if data_balance:
            self.data_instances = self.__balance_dataset__()
            self.__print_dataset_stats()

    def __len__(self):
        return len(self.data_instances)

    def __sort_instances__(self):
        sorted_instances = {}
        for data_instance in self.data_instances:
            label_id = data_instance["label"]
            if label_id not in sorted_instances.keys():
                sorted_instances.update({label_id: []})
            sorted_instances[label_id].append(data_instance)
        return sorted_instances

    def __count_instances__(self):
        labels = [data_instance["label"] for data_instance in self.data_instances]
        return np.bincount(labels)

    def __print_dataset_stats(self):
        # print data size
        print("[INFO] Data size:", self.__len__())
        # print instance count of each class
        print("[INFO] Data stats:")
        subsample_instance_count = self.__count_instances__()
        print(f"[INFO] {len(subsample_instance_count)} classes in total")
        for class_id in range(len(subsample_instance_count)):
            print(f"\t{int(class_id)}: {subsample_instance_count[class_id]}")

    def __balance_dataset__(self):
        print("[INFO] Balancing dataset")
        # sub_sample_size = self.instance_count.max()  # largest class
        sub_sample_size = self.instance_count[1:].sum()  # No-DR is too large
        subsample_data_instances = []
        for i in range(len(self.instance_count)):
            class_i_instances = self.sorted_instances[i]
            np.random.shuffle(class_i_instances)  # shuffle
            class_i_balance = class_i_instances.copy() * int(sub_sample_size / len(class_i_instances))  # replicate
            class_i_balance.extend(class_i_instances[:(sub_sample_size - len(class_i_balance))])  # fill up
            subsample_data_instances.extend(class_i_balance)
        np.random.shuffle(subsample_data_instances)
        return subsample_data_instances

    def get_class_weight(self):
        instance_count = self.__count_instances__()
        class_weight = instance_count.sum() / (len(instance_count) * instance_count)
        return class_weight

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

    retino_train_dataset = DiabeticRetinopathyDataset(image_dir="../../datasets/Dataset_DiabeticRetinopathy/train",
                                                      data_list="../../datasets/Dataset_DiabeticRetinopathy/trainLabels.csv",
                                                      data_balance=True, transform=train_transformer)

    image, label = retino_train_dataset.__getitem__(6)
    transforms.ToPILImage()(image).show()
    print(label)

    class_weight = retino_train_dataset.get_class_weight()
    print(class_weight)
