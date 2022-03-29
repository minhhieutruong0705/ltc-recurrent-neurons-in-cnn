import cv2
import os
from tqdm import tqdm
from image_utils import crop_by_ratio

"""
Images of Diabetic Retinopathy are varying in spatial size. They also have the dark area surrounding the retina,
especially on the left and right of an image. To reduce the uninformative area, the images are first cropped. Then,
they are resized to a smaller scale of 600x600
"""


def downsize_image(img, crop_ratio_h, crop_ratio_w, resize_dim_h, resize_dim_w):
    # crop image
    img = crop_by_ratio(
        img=img,
        h_ratio=crop_ratio_h,
        w_ratio=crop_ratio_w
    )
    # resize image
    return cv2.resize(img, (resize_dim_w, resize_dim_h))


if __name__ == '__main__':
    train_img_dir = "../../datasets/Dataset_DiabeticRetinopathy/train"
    test_img_dir = "../../datasets/Dataset_DiabeticRetinopathy/test"

    # create save folders
    save_dir_train = "../../datasets/Dataset_DiabeticRetinopathy/train_small"
    save_dir_test = "../../datasets/Dataset_DiabeticRetinopathy/test_small"
    os.makedirs(save_dir_train, exist_ok=True)
    os.makedirs(save_dir_test, exist_ok=True)

    # params
    crop_ratio_h = 1.0
    crop_ratio_w = 0.9
    resize_dim_h = resize_dim_w = 600

    # process train images
    for img_name in tqdm(os.listdir(train_img_dir)):
        img_path = os.path.join(train_img_dir, img_name)
        # load image
        img = cv2.imread(img_path, -1)
        # process image
        img = downsize_image(
            img=img,
            crop_ratio_h=crop_ratio_h,
            crop_ratio_w=crop_ratio_w,
            resize_dim_h=resize_dim_h,
            resize_dim_w=resize_dim_w
        )
        # save images
        cv2.imwrite(os.path.join(save_dir_train, img_name), img)

    # process test images
    for img_name in tqdm(os.listdir(test_img_dir)):
        img_path = os.path.join(test_img_dir, img_name)
        # load image
        img = cv2.imread(img_path, -1)
        # process image
        img = downsize_image(
            img=img,
            crop_ratio_h=crop_ratio_h,
            crop_ratio_w=crop_ratio_w,
            resize_dim_h=resize_dim_h,
            resize_dim_w=resize_dim_w
        )
        # save images
        cv2.imwrite(os.path.join(save_dir_test, img_name), img)
