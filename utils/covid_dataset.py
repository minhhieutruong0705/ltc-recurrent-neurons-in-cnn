import os
import numpy as np
import cv2

from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from torch.utils.data import Dataset


def read_list(path):
    with open(path, 'r') as f:
        lines = f.read()
    return lines[:-1].split('\n') # last element is blank


class CovidDataset(Dataset):
    def __init__(self, covid_dir, non_covid_dir, list_covid, list_non_covid, transform=None):
        super().__init__()
        self.covid_dir = covid_dir
        self.non_covid_dir = non_covid_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # read image names
        images_covid = read_list(list_covid)
        images_non_covid = read_list(list_non_covid)
        print("[INFO] covid size", len(images_covid))
        print("[INFO] non-covid size", len(images_non_covid))

        # generate full paths for the images and their labels
        for img_name in images_covid:
            self.image_paths.append(os.path.join(covid_dir, "Covid", img_name))
            self.labels.append(1)  # positive to COVID-19 is labeled as 1; otherwise 0
        for img_name in images_non_covid:
            self.image_paths.append(os.path.join(non_covid_dir, "Normal", img_name))
            self.labels.append(0)

        # validate the dataset
        if len(self.image_paths) == len(self.labels):
            print("[INFO] Dataset Size:", self.__len__())
            print("[INFO] Dataset OK!")
        else:
            print("[ERROR] Image size does not match label size!")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = np.array(Image.open(self.image_paths[index]).convert("RGB"))
        label = self.labels[index]

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
        A.GaussianBlur(blur_limit=5),
        A.RandomContrast(limit=0.2),
        A.RandomBrightness(limit=0.2),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ), ToTensorV2()
    ])

    covid_train_dataset = CovidDataset(
        covid_dir="../../datasets/Dataset_PNG/COVID",
        non_covid_dir="../../datasets/Dataset_PNG/NONCOVID",
        list_covid="../../datasets/Dataset_PNG/covid_train.txt",
        list_non_covid="../../datasets/Dataset_PNG/normal_train.txt",
        transform=train_transformer
    )

    image, label = covid_train_dataset.__getitem__(0)
    transforms.ToPILImage()(image).show()
    print(label)
