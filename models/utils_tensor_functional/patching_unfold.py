import cv2
import torch
from torchvision import transforms

"""
spatial_unfold provides overlapping patching of a tensor image using a sliding window

:argument
tensor_img: B, C, H, W
spatial_kernel_size, stride, padding
horizontal_seq=True indicates that the sequence of patches will follow rows and then columns
horizontal_seq=False indicates that the sequence of patches will follow columns and then rows

:return
tensor_patches = B, P, C, H_p, W_p
"""


def spatial_unfold(tensor_img, spatial_kernel_size, stride, horizontal_seq=True):
    # patching the tensor
    patches = tensor_img.unfold(
        dimension=2, size=spatial_kernel_size, step=stride  # unfold y-axis
    ).unfold(
        dimension=3, size=spatial_kernel_size, step=stride  # unfold x-axis
    )  # shape: B, C, P_h, P_w, H_p, W_p

    # rearrange the tensor
    redundant_axis = 2 if horizontal_seq else 3
    patches = torch.split(patches, split_size_or_sections=1, dim=redundant_axis)
    patches = [patch.squeeze(dim=redundant_axis) for patch in patches]
    patches = torch.cat(patches, dim=2)
    return patches.permute(0, 2, 1, 3, 4)


def __load_img_as_tensor__(img_path):
    img = cv2.imread(img_path, -1)
    return transforms.ToTensor()(img)


def __show_tensor_img__(tensor_img):
    transforms.ToPILImage()(tensor_img).show()


if __name__ == '__main__':
    img = __load_img_as_tensor__("./astronaut.jpeg")  # 512x512 image
    img = img.unsqueeze(dim=0)  # put img in a batch of 1
    patches = spatial_unfold(img, spatial_kernel_size=256, stride=256, horizontal_seq=True)
    assert patches.size() == (1, 4, 3, 256, 256)
    print("[ASSERT] spatial_unfold OK!")

    # display patches
    patches = patches.squeeze(dim=0)  # get rid of batch
    # check the sequence of patches
    __show_tensor_img__(patches[0])
    __show_tensor_img__(patches[1])
    # for patch in chunks:
    #     __show_tensor_img__(patch)
