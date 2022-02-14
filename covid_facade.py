from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from utils.covid_dataset import CovidDataset


# loaders put data into batches and make the batches iterable
def get_data_loaders(
        covid_dir, non_covid_dir,
        list_train_covid, list_train_non_covid, train_transformer,
        list_val_covid, list_val_non_covid, val_transformer,
        list_test_covid, list_test_non_covid,
        batch_size, lung_mask_incor
):
    print("[INFO] Loading train dataset ...")
    train_dataset = CovidDataset(
        covid_dir=covid_dir,
        non_covid_dir=non_covid_dir,
        list_covid=list_train_covid,
        list_non_covid=list_train_non_covid,
        transform=train_transformer,
        lung_mask_incor=lung_mask_incor
    )

    print("[INFO] Loading validation dataset ...")
    val_dataset = CovidDataset(
        covid_dir=covid_dir,
        non_covid_dir=non_covid_dir,
        list_covid=list_val_covid,
        list_non_covid=list_val_non_covid,
        transform=val_transformer,
        lung_mask_incor=lung_mask_incor
    )

    print("[INFO] Loading test dataset ...")
    test_dataset = CovidDataset(
        covid_dir=covid_dir,
        non_covid_dir=non_covid_dir,
        list_covid=list_test_covid,
        list_non_covid=list_test_non_covid,
        transform=val_transformer,
        lung_mask_incor=lung_mask_incor
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, val_loader, test_loader


# transformer is for the augmentation of data
def get_transformers(
        img_dim,
        img_crop_dim,
        random_crop_scale,
        rotation_limit,
        blur_kernel_range,
        contrast_factor,
        brightness_factor,
        mean_norm,
        std_norm,
        max_pixel_value,
):
    train_transformer = A.Compose([
        A.Resize(height=img_dim, width=img_dim, interpolation=cv2.INTER_AREA),
        A.RandomResizedCrop(height=img_crop_dim, width=img_crop_dim, scale=(random_crop_scale, 1.0),
                            interpolation=cv2.INTER_AREA),
        A.HorizontalFlip(),
        A.Rotate(limit=rotation_limit),
        A.GaussNoise(),
        A.GaussianBlur(blur_limit=blur_kernel_range),
        A.RandomBrightnessContrast(brightness_limit=brightness_factor, contrast_limit=contrast_factor),
        A.Normalize(
            mean=mean_norm,
            std=std_norm,
            max_pixel_value=max_pixel_value,
        ), ToTensorV2()
    ])

    val_transformer = A.Compose([
        A.Resize(height=img_crop_dim, width=img_crop_dim, interpolation=cv2.INTER_AREA),
        A.CenterCrop(height=img_crop_dim, width=img_crop_dim),
        A.Normalize(
            mean=mean_norm,
            std=std_norm,
            max_pixel_value=max_pixel_value,
        ), ToTensorV2()
    ])

    return train_transformer, val_transformer
