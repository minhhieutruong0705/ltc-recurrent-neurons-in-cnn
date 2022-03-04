import torch
import torch.nn as nn
import cv2
from torchvision import transforms

"""
Chunker provides non-overlapping patching of a tensor image

:argument
x (tensor_img): (B, C, H, W)
tensor_img must be a square image and is divisible by chunks_per_side
horizontal_seq=True indicates that the sequence of patches will follow rows and then columns
horizontal_seq=False indicates that the sequence of patches will follow columns and then rows
zigzag=True indicate the sequence of patches will be in zigzag order 
zigzag=False indicate the sequence of patches will be a queue of of patch rows (columns)

:return
tensor of patches = (B, P, C, H_p, W_p)
"""


class Chunker(nn.Module):
    def __init__(self, chunks_per_side, horizontal_seq=True, zigzag=False):
        super().__init__()
        self.chunks_per_side = chunks_per_side
        self.horizontal_seq = horizontal_seq
        self.zigzag = zigzag

    def extra_repr(self):
        return f"chunks_per_side={self.chunks_per_side}, horizontal_seq={self.horizontal_seq}, zigzag={self.zigzag}"

    def forward(self, x):
        img_h, img_w = x.shape[2: 4]
        h_index = 3
        w_index = 4
        assert img_h == img_w  # check square image
        assert img_h % self.chunks_per_side == 0  # check divisible

        # patching the tensor
        tensor_img = x.unsqueeze(dim=1)  # create a dimension for patches:(B, C, H, W) -> (B, P, C, H, W)
        chunks_1side = torch.chunk(tensor_img, chunks=self.chunks_per_side,
                                   dim=w_index if self.horizontal_seq else h_index)  # x-axis
        chunks_1side = torch.cat(chunks_1side, dim=1)  # merge a list of chunks on 1 side into a tensor
        chunks = torch.chunk(chunks_1side, chunks=self.chunks_per_side,
                             dim=h_index if self.horizontal_seq else w_index)  # y-axis
        if self.zigzag:
            chunks = [chunk.flip(dims=[1]) if i % 2 == 1 else chunk for i, chunk in enumerate(chunks)]
        return torch.cat(chunks, dim=1)  # merge a list of chunks into a tensor


def __load_img_as_tensor__(img_path):
    img = cv2.imread(img_path, -1)
    return transforms.ToTensor()(img)


def __show_tensor_img__(tensor_img):
    transforms.ToPILImage()(tensor_img).show()


if __name__ == '__main__':
    img = __load_img_as_tensor__("./astronaut.jpeg")  # 512x512 image
    img = img.unsqueeze(dim=0)  # put img in a batch of 1
    model = Chunker(chunks_per_side=2, horizontal_seq=True, zigzag=True)
    chunks = model(img)
    assert chunks.size() == (1, 4, 3, 256, 256)
    print("[ASSERT] Chunker OK!")
    print(model)

    # display patches
    chunks = chunks.squeeze(dim=0)  # get rid of batch
    # check the sequence of patches
    # __show_tensor_img__(chunks[0])
    # __show_tensor_img__(chunks[1])
    __show_tensor_img__(chunks[2])
    __show_tensor_img__(chunks[3])
    # for patch in chunks:
    #     __show_tensor_img__(patch)
