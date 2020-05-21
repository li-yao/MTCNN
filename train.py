import os
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.optim as optim
from MTCNN.MTCNN_PyTorch_teach.sampling import FaceDataset


class Trainer:
    def __init__(self, net, save_path, dataset_path):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.net = net.to(self.device)
        self.save_path = save_path
        self.dataset_path = dataset_path
        self.cls_loss_fn = nn.BCELoss()
        self.offset_loss_fn = nn.MSELoss()

        self.optimizer = optim.Adam(self.net.parameters())

        if os.path.exists(self.save_path):  # 方便在训练断了之后，继续下次训练这个网络
            net.load_state_dict(torch.load(self.save_path))
        else:
            print("NO Param")

    def train(self, stop_value):
        faceDataset = FaceDataset(self.dataset_path)
        dataloader = DataLoader(faceDataset, batch_size=512, shuffle=True, num_workers=4)
        loss = 0
        while True:
            for i, (img_data_, category_, offset_) in enumerate(dataloader):
                img_data_ = img_data_.to(self.device)
                category_ = category_.to(self.device)
                offset_ = offset_.to(self.device)

                _output_category, _output_offset = self.net(img_data_)
                output_category = _output_category.view(-1, 1)  # reshape
                output_offset = _output_offset.view(-1, 4)
                # output_landmark = _output_landmark.view(-1, 10)
                # 计算分类的损失
                # eq:等于，lt:小于，gt:大于，le:小于等于，ge:大于等于
                # 计算置信度loss 正样本+负样本计算
                category_mask = torch.lt(category_, 2)  # 得到分类标签小于2的布尔值，a<2,[0,1,2]-->[1,1,0]
                category = torch.masked_select(category_, category_mask)  # 通过掩码，得到符合条件的置信度标签值
                output_category = torch.masked_select(output_category, category_mask)  # 得到符合条件的置信度输出值
                cls_loss = self.cls_loss_fn(output_category, category)  # 置信度损失

                # 计算bound的损失 正样本+部分样本计算
                offset_mask = torch.gt(category_, 0)  # 得到分类标签大于0的布尔值,a>0,[0,1,2]-->[0,1,1]
                offset = torch.masked_select(offset_, offset_mask)  # 通过掩码，得到符合条件的偏移量标签值
                output_offset = torch.masked_select(output_offset, offset_mask)  # 得到符合条件的偏移量输出值
                offset_loss = self.offset_loss_fn(output_offset, offset)  # 偏移量损失

                loss = cls_loss + offset_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                cls_loss = cls_loss.cpu().item()
                offset_loss = offset_loss.cpu().item()
                loss = loss.cpu().item()
                print(" loss:", loss, " cls_loss:", cls_loss, " offset_loss",offset_loss)

            torch.save(self.net.state_dict(), self.save_path)
            print("save success")

            if loss < stop_value:
                break

