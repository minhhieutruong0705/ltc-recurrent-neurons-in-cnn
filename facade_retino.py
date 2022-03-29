from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from utils_retino.retino_dataset import DiabeticRetinopathyDataset


# loaders put data into batches and make the batches iterable
def get_data_loaders(
        train_dir, test_dir,
        list_train, list_val, list_test,
        train_transformer, val_transformer,
        batch_size, data_load_workers
):
    print("[INFO] Loading train dataset ...")
    train_dataset = DiabeticRetinopathyDataset(image_dir=train_dir, data_list=list_train, data_balance=True,
                                               transform=train_transformer)
    class_weight = train_dataset.get_class_weight()

    print("[INFO] Loading validation dataset ...")
    val_dataset = DiabeticRetinopathyDataset(image_dir=train_dir, data_list=list_val, data_balance=False,
                                             transform=val_transformer)

    print("[INFO] Loading test dataset ...")
    test_dataset = DiabeticRetinopathyDataset(image_dir=test_dir, data_list=list_test, data_balance=False,
                                              transform=val_transformer)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_load_workers
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_load_workers
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_load_workers
    )
    return train_loader, val_loader, test_loader, class_weight


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
        A.Resize(height=img_dim, width=img_dim, interpolation=cv2.INTER_CUBIC),
        A.RandomResizedCrop(height=img_crop_dim, width=img_crop_dim, scale=(random_crop_scale, 1.0),
                            interpolation=cv2.INTER_CUBIC),
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
