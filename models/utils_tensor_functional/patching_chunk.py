import torch
import cv2
from torchvision import transforms

"""
spatial_chunk provides non-overlapping patching of a tensor image

:argument
tensor_img: B, C, H, W
tensor_img must be a square image and is divisible by chunks_per_side
horizontal_seq=True indicates that the sequence of patches will follow rows and then columns
horizontal_seq=False indicates that the sequence of patches will follow columns and then rows

:return
tensor_patches = B, P, C, H_p, W_p
"""


def spatial_chunk(tensor_img, chunks_per_side, horizontal_seq=True):
    img_h, img_w = tensor_img.shape[2: 4]
    h_index = 3
    w_index = 4
    assert img_h == img_w  # check square image
    assert img_h % chunks_per_side == 0  # check divisible

    # patching the tensor
    tensor_img = tensor_img.unsqueeze(dim=1)  # create a dimension for patches:B, C, H, W -> B, P, C, H, W
    chunks_h = torch.chunk(tensor_img, chunks=chunks_per_side, dim=w_index if horizontal_seq else h_index)  # x-axis
    chunks_h = torch.cat(chunks_h, dim=1)  # merge a list of chunks into a tensor
    chunks = torch.chunk(chunks_h, chunks=chunks_per_side, dim=h_index if horizontal_seq else w_index)  # y-axis
    return torch.cat(chunks, dim=1)  # merge a list of chunks into a tensor


def __load_img_as_tensor__(img_path):
    img = cv2.imread(img_path, -1)
    return transforms.ToTensor()(img)


def __show_tensor_img__(tensor_img):
    transforms.ToPILImage()(tensor_img).show()


if __name__ == '__main__':
    img = __load_img_as_tensor__("./astronaut.jpeg")  # 512x512 image
    img = img.unsqueeze(dim=0)  # put img in a batch of 1
    chunks = spatial_chunk(img, chunks_per_side=2, horizontal_seq=True)
    assert chunks.size() == (1, 4, 3, 256, 256)
    print("[ASSERT] spatial_chunk OK!")

    # display patches
    chunks = chunks.squeeze(dim=0)  # get rid of batch
    # check the sequence of patches
    __show_tensor_img__(chunks[0])
    __show_tensor_img__(chunks[1])
    # for patch in chunks:
    #     __show_tensor_img__(patch)
