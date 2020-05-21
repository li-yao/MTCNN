from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

trans = transforms.Compose([
    transforms.ToTensor(),
    # Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.  Converts a PIL Image or numpy.ndarray (H x W x C)...
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


class FaceDataset(Dataset):

    def __init__(self, path):
        self.path = path
        self.dataset = []
        """ Extend list by appending elements from the iterable. """
        self.dataset.extend(open(os.path.join(path, "positive.txt")).readlines())
        # print(self.dataset)
        self.dataset.extend(open(os.path.join(path, "negative.txt")).readlines())
        self.dataset.extend(open(os.path.join(path, "part.txt")).readlines())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        strs = self.dataset[index].strip().split(" ")
        print(strs)
        img_path = os.path.join(self.path, strs[0])  # 图片路径 str[0]=negative/0.jpg
        cls = torch.tensor([int(strs[1])], dtype=torch.float32)  # 正负部分样本标签  tensor的运算都是要是float
        # 坐标点偏移率offset
        offset = torch.tensor([float(strs[2]), float(strs[3]), float(strs[4]), float(strs[5])], dtype=torch.float32)
        # img_data = torch.tensor(np.array(Image.open(img_path)) / 255. - 0.5, dtype=torch.float32)  # ?
        img_data = trans(Image.open(img_path))  # ToTensor就把PIL转为CHW了
        # print(img_data.shape)  # H W C
        # img_data = img_data.permute(2, 0, 1)  # C H W

        return img_data, cls, offset


if __name__ == '__main__':
    dataset = FaceDataset(r"C:\Users\admin\Desktop\CelebA\My_MTCNN_dataset\12")  # r是说明不需要转义
    print(dataset[0][0].shape)
    # print(dataset[0][1].shape)
    # print(dataset[0][2].shape)
    # dataloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=4)  # num_workers-->进程数
    # for i, (img, cls, offset) in enumerate(dataloader):
    #     print(img.shape)
    #     print(cls.shape)
    #     print(cls)
    #     print(offset.shape)
