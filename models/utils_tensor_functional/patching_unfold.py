import cv2
import torch
import torch.nn as nn
from torchvision import transforms

"""
Un-folder provides overlapping patching of a tensor image using a sliding window

:argument
x (tensor_img): (B, C, H, W)
spatial_kernel_size, stride, padding
horizontal_seq=True indicates that the sequence of patches will follow rows and then columns
horizontal_seq=False indicates that the sequence of patches will follow columns and then rows
zigzag=True indicate the sequence of patches will be in zigzag order 
zigzag=False indicate the sequence of patches will be a queue of of patch rows (columns)

:return
tensor_patches = (B, P, C, H_p, W_p)
"""


class Unfolder(nn.Module):
    def __init__(self, spatial_kernel_size, stride, horizontal_seq=True, zigzag=False):
        super().__init__()
        self.spatial_kernel_size = spatial_kernel_size
        self.stride = stride
        self.horizontal_seq = horizontal_seq
        self.zigzag = zigzag

    def extra_repr(self):
        return f"spatial_kernel_size={self.spatial_kernel_size}, " \
               f"stride={self.stride}, " \
               f"horizontal_seq={self.horizontal_seq}"

    def forward(self, x):
        # patching the tensor
        patches = x.unfold(
            dimension=2, size=self.spatial_kernel_size, step=self.stride  # unfold y-axis
        ).unfold(
            dimension=3, size=self.spatial_kernel_size, step=self.stride  # unfold x-axis
        )  # shape: (B, C, P_h, P_w, H_p, W_p)

        # rearrange the tensor
        redundant_axis = 2 if self.horizontal_seq else 3
        patches = torch.split(patches, split_size_or_sections=1, dim=redundant_axis)
        patches = [patch.squeeze(dim=redundant_axis) for patch in patches]
        if self.zigzag:
            patches = [patch.flip(dims=[2]) if i % 2 == 1 else patch for i, patch in enumerate(patches)]
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
    model = Unfolder(spatial_kernel_size=256, stride=256, horizontal_seq=True, zigzag=True)
    patches = model(img)
    assert patches.size() == (1, 4, 3, 256, 256)
    print("[ASSERT] Unfolder OK!")
    print(model)

    # display patches
    patches = patches.squeeze(dim=0)  # get rid of batch
    # check the sequence of patches
    # __show_tensor_img__(patches[0])
    # __show_tensor_img__(patches[1])
    __show_tensor_img__(patches[2])
    __show_tensor_img__(patches[3])
    # for patch in chunks:
    #     __show_tensor_img__(patch)
