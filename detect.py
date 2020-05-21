import torch
from PIL import Image
from PIL import ImageDraw
import numpy as np
from MTCNN.MTCNN_PyTorch_teach.tool import utils
from MTCNN.MTCNN_PyTorch_teach import nets
from torchvision import transforms
import time


class Detector:

    def __init__(self, pnet_param="./param/p_net.pth", rnet_param="./param/r_net.pth", onet_param="./param/o_net.pth",
                 isCuda=False):

        self.isCuda = isCuda

        self.pnet = nets.PNet()
        self.rnet = nets.RNet()
        self.onet = nets.ONet()

        if self.isCuda:
            self.pnet.cuda()
            self.rnet.cuda()
            self.onet.cuda()

        self.pnet.load_state_dict(torch.load(pnet_param, map_location='cpu'))  # map_location ?
        self.rnet.load_state_dict(torch.load(rnet_param, map_location='cpu'))
        self.onet.load_state_dict(torch.load(onet_param, map_location='cpu'))

        self.pnet.eval()  # 关掉dropout
        self.rnet.eval()
        self.onet.eval()

        self.__image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def detect(self, image):

        start_time = time.time()
        pnet_boxes = self.__pnet_detect(image)

        if pnet_boxes.shape[0] == 0:
            return np.array([])

        end_time = time.time()
        t_pnet = end_time - start_time

        start_time = time.time()
        rnet_boxes = self.__rnet_detect(image, pnet_boxes)

        if rnet_boxes.shape[0] == 0:
            return np.array([])
        end_time = time.time()
        t_rnet = end_time - start_time

        start_time = time.time()
        onet_boxes = self.__onet_detect(image, rnet_boxes)

        if onet_boxes.shape[0] == 0:
            return np.array([])

        end_time = time.time()
        t_onet = end_time - start_time

        t_sum = t_pnet + t_rnet + t_onet

        print("total:{0} pnet:{1} rnet:{2} onet:{3}".format(t_sum, t_pnet, t_rnet, t_onet))

        return onet_boxes

    def __pnet_detect(self, image):

        boxes = []
        img = image
        w, h = img.size
        min_side_len = min(w, h)

        scale = 1

        while min_side_len > 12:
            img_data = self.__image_transform(img)
            if self.isCuda:
                img_data = img_data.cuda()
            img_data.unsqueeze_(0)  # c h w --> n c h w

            _cls, _offest = self.pnet(img_data)  # cls: n1hw  offset:n4hw

            cls, offest = _cls[0][0].cpu().data, _offest[0].cpu().data  # 如果tensor有梯度的话，要用data取出数据  此处没有

            idxs = torch.nonzero(torch.gt(cls, 0.6))  # 返回的索引的值

            for idx in idxs:
                boxes.append(self.__box(idx, offest, cls[idx[0], idx[1]], scale))

            scale *= 0.6  # 图像金字塔
            _w = int(w * scale)
            _h = int(h * scale)

            img = img.resize((_w, _h))
            min_side_len = np.minimum(_w, _h)  # 如果两个都是标量，则返回的也是标量
        return utils.nms(np.array(boxes), thresh=0.3)

    def __box(self, start_index, offset, cls, scale, stride=2, side_len=12):  # 原图坐标反算

        _x1 = int(start_index[1] * stride) / scale  # 宽，W，x  表示第start_index[1]个stride，即为_x1 在宽度上的第index[1]个12所对应的
        _y1 = int(start_index[0] * stride) / scale  # 高，H, y
        _x2 = int(start_index[1] * stride + side_len) / scale  # 那么_x1加上side_len即为x2
        _y2 = int(start_index[0] * stride + side_len) / scale

        ow = _x2 - _x1  # 12
        oh = _y2 - _y1  # 12

        _offset = offset[:, start_index[0], start_index[1]]

        x1 = _x1 + ow * _offset[0]
        y1 = _y1 + oh * _offset[1]
        x2 = _x2 + ow * _offset[2]
        y2 = _y2 + oh * _offset[3]

        return [x1, y1, x2, y2, cls]

    def __rnet_detect(self, image, pnet_boxes):

        _img_dataset = []
        _pnet_boxes = utils.convert_to_square(pnet_boxes)
        for _box in _pnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))  # 反算回原图截取size=24*24
            img = img.resize((24, 24))
            img_data = self.__image_transform(img)
            _img_dataset.append(img_data)

        img_dataset =torch.stack(_img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset = self.rnet(img_dataset)  # 重新计算

        _cls = _cls.cpu().data.numpy()  # 置信度(342, 1)

        offset = _offset.cpu().data.numpy()  # 偏移率(342, 4)

        boxes = []

        idxs, _ = np.where(_cls > 0.6)  # 取出置信度>0.6的索引

        for idx in idxs:  # 按照索引取

            _box = _pnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]
            cls = _cls[idx][0]

            boxes.append([x1, y1, x2, y2, cls])

        return utils.nms(np.array(boxes), 0.3)

    def __onet_detect(self, image, rnet_boxes):

        _img_dataset = []
        _rnet_boxes = utils.convert_to_square(rnet_boxes)
        for _box in _rnet_boxes:
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            img = image.crop((_x1, _y1, _x2, _y2))
            img = img.resize((48, 48))
            img_data = self.__image_transform(img)
            _img_dataset.append(img_data)

        img_dataset = torch.stack(_img_dataset)
        if self.isCuda:
            img_dataset = img_dataset.cuda()

        _cls, _offset = self.onet(img_dataset)

        _cls = _cls.cpu().data.numpy()
        offset = _offset.cpu().data.numpy()

        boxes = []
        idxs, _ = np.where(_cls > 0.97)
        for idx in idxs:
            _box = _rnet_boxes[idx]
            _x1 = int(_box[0])
            _y1 = int(_box[1])
            _x2 = int(_box[2])
            _y2 = int(_box[3])

            ow = _x2 - _x1
            oh = _y2 - _y1

            x1 = _x1 + ow * offset[idx][0]
            y1 = _y1 + oh * offset[idx][1]
            x2 = _x2 + ow * offset[idx][2]
            y2 = _y2 + oh * offset[idx][3]
            cls = _cls[idx][0]

            boxes.append([x1, y1, x2, y2, cls])

        return utils.nms(np.array(boxes), 0.3, isMin=True)


if __name__ == '__main__':
    x = time.time()
    with torch.no_grad() as grad:
        image_file = r"timg.jpg"
        detector = Detector()
        #
        with Image.open(image_file) as im:
            boxes = detector.detect(im)
            imDraw = ImageDraw.Draw(im)
            for box in boxes:
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])

                imDraw.rectangle((x1, y1, x2, y2), outline='red')
            y = time.time()
            print(y - x)
            im.show()
