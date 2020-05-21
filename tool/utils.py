import numpy as np

"iou函数为了方便nms时候调用，所以boxes设计成了二维的，调用的时候需要注意"
def iou(box, boxes, isMin = False):  # boxs:除置信度最大的框box以外，其他框的坐标
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    w = np.maximum(0, xx2 - xx1)  # 有可能没有交集，所以存在一个和0的比对选择
    h = np.maximum(0, yy2 - yy1)

    inter = w * h
    if isMin:  # np.true_divide返回两值进行除法后的真值，而不是floor division这样的向下取整
        # np.true_divide的return： ndarray or scalar
        # but：This is a scalar if both `x1` and `x2` are scalars
        ovr = np.true_divide(inter, np.minimum(box_area, area))  # 最小值iou  面积最小的框 意义：这个是用来做同心框的iou的
    else:
        ovr = np.true_divide(inter, (box_area + area - inter))

    return ovr


def nms(boxes, thresh=0.3, isMin = False):  # 可以改进为softnms

    if boxes.shape[0] == 0:
        return np.array([])

    _boxes = boxes[(-boxes[:, 4]).argsort()]  # 送过来的是一堆box，所以用[:, 4]   负号表示从大到小排序 argsort返回的是索引值
    r_boxes = []

    while _boxes.shape[0] > 1:
        a_box = _boxes[0]  # 最大置信度框
        b_boxes = _boxes[1:]  # 其他框

        r_boxes.append(a_box)

        # print(iou(a_box, b_boxes))

        index = np.where(iou(a_box, b_boxes, isMin) < thresh)  # 取b_boxes中符合条件的索引
        _boxes = b_boxes[index]  # 把符合条件的找出来，生成新的_boxes

    if _boxes.shape[0] > 0:
        r_boxes.append(_boxes[0])

    return np.stack(r_boxes)  # r_boxes堆叠成新的array，如果没有说明，axis=0


def convert_to_square(bbox):
    square_bbox = bbox.copy()
    if bbox.shape[0] == 0:
        return np.array([])
    h = bbox[:, 3] - bbox[:, 1]
    w = bbox[:, 2] - bbox[:, 0]
    max_side = np.maximum(h, w)

    # 无论w长或者h长，都是这样计算
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - max_side * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - max_side * 0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side
    square_bbox[:, 3] = square_bbox[:, 1] + max_side

    return square_bbox  # [x1 y1 x2 y2]左上角右下角坐标


if __name__ == '__main__':
    a = np.array([1, 1, 11, 11])
    # bs = np.array([[1, 1, 10, 10], [14, 15, 20, 20]])
    bs = np.array([[1, 1, 10, 10]])
    print(iou(a, bs))
    print(iou(a, bs) > 0)

    # bs = np.array([[1, 1, 10, 10, 0.98], [1, 1, 9, 9, 0.8], [9, 8, 13, 20, 0.7], [6, 11, 18, 17, 0.85]])
    # print((-bs[:, 4]).argsort())
    # print(nms(bs))
