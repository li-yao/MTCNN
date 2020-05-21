import os
from PIL import Image
import numpy as np
from MTCNN.MTCNN_PyTorch_teach.tool import utils
import traceback

anno_src = r"C:\Users\admin\Desktop\CelebA\Anno\list_bbox_celeba.txt"  # 样本标签
img_dir = r"C:\Users\admin\Desktop\CelebA\Img\img_celeba.7z\img_celeba"  # 样本图片

save_path = r"C:\Users\admin\Desktop\CelebA\My_MTCNN_dataset"

float_num = [0.1, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.9]  # 正:部分:负= 1:1:3


def gen_sample(face_size, stop_value):
    print("gen size:{} image" .format(face_size))
    # 样本图片存储路径
    positive_image_dir = os.path.join(save_path, str(face_size), "positive")
    negative_image_dir = os.path.join(save_path, str(face_size), "negative")
    part_image_dir = os.path.join(save_path, str(face_size), "part")

    # 造出三种路径下的9个文件夹，// 12,24,48
    for dir_path in [positive_image_dir, negative_image_dir, part_image_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 样本标签存储路径
    positive_anno_filename = os.path.join(save_path, str(face_size), "positive.txt")
    negative_anno_filename = os.path.join(save_path, str(face_size), "negative.txt")
    part_anno_filename = os.path.join(save_path, str(face_size), "part.txt")

    positive_count = 0
    negative_count = 0
    part_count = 0

    "什么时候需要添加异常？-->在我们不希望发生异常，但又无法预防用户输入错误的数据，这个时候就需要用到try..except..语句，来对异常进行处理"
    try:
        # 创建标签txt存放文件
        positive_anno_file = open(positive_anno_filename, "w")
        negative_anno_file = open(negative_anno_filename, "w")
        part_anno_file = open(part_anno_filename, "w")

        for i, line in enumerate(open(anno_src)):
            if i < 2:  # 前两行不是想要的，所以前两行不执行，从第三行开始执行
                continue
            try:
                # 切割非空值元素---默认所有的空字符，包括空格、换行(\n)、制表符(\t)等
                strs = line.split()  # eg: ['008843.jpg', '67', '107', '341', '472']

                image_filename = strs[0].strip()  # 008843.jpg strip去掉空格
                print(image_filename)
                image_file = os.path.join(img_dir, image_filename)  # 数据集单张图片路径

                with Image.open(image_file) as img:  # with as语句操作上下文管理器，用于自动分配并且释放资源
                    img_w, img_h = img.size  # PIL中是WHC

                    # 获取左上角坐标和w h
                    x1 = float(strs[1].strip())
                    y1 = float(strs[2].strip())
                    w = float(strs[3].strip())
                    h = float(strs[4].strip())

                    # 计算右下角坐标
                    x2 = float(x1 + w)
                    y2 = float(y1 + h)

                    # 下面这些是人脸特征点的位置坐标 因为没用到 所以全置0
                    px1 = 0  # float(strs[5].strip())
                    py1 = 0  # float(strs[6].strip())
                    px2 = 0  # float(strs[7].strip())
                    py2 = 0  # float(strs[8].strip())
                    px3 = 0  # float(strs[9].strip())
                    py3 = 0  # float(strs[10].strip())
                    px4 = 0  # float(strs[11].strip())
                    py4 = 0  # float(strs[12].strip())
                    px5 = 0  # float(strs[13].strip())
                    py5 = 0  # float(strs[14].strip())

                    if x1 < 0 or y1 < 0 or w < 0 or h < 0:
                        continue

                    boxes = [[x1, y1, x2, y2]]  # 实际框 因为iou函数boxes设计是二维的，所以这里这样写

                    # 计算出人脸中心点位置
                    cx = x1 + w / 2
                    cy = y1 + h / 2
                    side_len = max(w, h)  # 取最大的边，（正方形）
                    seed = float_num[np.random.randint(0, len(float_num))]
                    count = 0
                    for _ in range(5):  # 一张图片上偏移5次，生成5个偏移框
                        _side_len = side_len + np.random.randint(int(-side_len * seed), int(side_len * seed))
                        # 偏移最大边长，最大的边长再加上或减去一个随机数
                        _cx = cx + np.random.randint(int(-cx * seed), int(cx * seed))  # 偏移中心点X
                        _cy = cy + np.random.randint(int(-cy * seed), int(cy * seed))  # 偏移中心点Y

                        _x1 = _cx - _side_len / 2  # 偏移后的中心点换算回偏移后起始点X,Y（左上角）
                        _y1 = _cy - _side_len / 2
                        _x2 = _x1 + _side_len  # 获得偏移后的X2,Y2
                        _y2 = _y1 + _side_len
                        # 偏移后的的坐标点对应的是正方形

                        if _x1 < 0 or _y1 < 0 or _x2 > img_w or _y2 > img_h:  # 判断偏移超出整张图片的就跳过，不截图
                            continue

                        offset_x1 = (x1 - _x1) / _side_len  # 获得换算后的偏移率
                        offset_y1 = (y1 - _y1) / _side_len
                        offset_x2 = (x2 - _x2) / _side_len
                        offset_y2 = (y2 - _y2) / _side_len

                        offset_px1 = 0  # (px1 - x1_) / side_len
                        offset_py1 = 0  # (py1 - y1_) / side_len
                        offset_px2 = 0  # (px2 - x1_) / side_len
                        offset_py2 = 0  # (py2 - y1_) / side_len
                        offset_px3 = 0  # (px3 - x1_) / side_len
                        offset_py3 = 0  # (py3 - y1_) / side_len
                        offset_px4 = 0  # (px4 - x1_) / side_len
                        offset_py4 = 0  # (py4 - y1_) / side_len
                        offset_px5 = 0  # (px5 - x1_) / side_len
                        offset_py5 = 0  # (py5 - y1_) / side_len

                        # 剪切下图片，并进行大小缩放
                        crop_box = [_x1, _y1, _x2, _y2]  # 获得需要截取图片样本的坐标
                        face_crop = img.crop(crop_box)  # 截图
                        face_resize = face_crop.resize((face_size, face_size))  # 如果截图比resize的图像大小要小呢，那不就模糊了--强制resize

                        iou = utils.iou(crop_box, np.array(boxes))[0]  # format：[x]
                        if iou > 0.65:  # 正样本// >0.65
                            positive_anno_file.write(
                                "positive/{0}.jpg {1} {2} {3} {4} {5}\n".format(
                                    positive_count, 1, offset_x1, offset_y1, offset_x2, offset_y2))
                            positive_anno_file.flush()  # flush() 方法是用来刷新缓冲区的，即将缓冲区中的数据立刻写入文件，同时清空缓冲区
                            face_resize.save(os.path.join(positive_image_dir, "{0}.jpg".format(positive_count)))
                            positive_count += 1
                        elif 0.6 > iou > 0.4:  # 部分样本// >0.4
                            part_anno_file.write(
                                "part/{0}.jpg {1} {2} {3} {4} {5}\n".format(
                                    part_count, 2, offset_x1, offset_y1, offset_x2, offset_y2))
                            part_anno_file.flush()
                            face_resize.save(os.path.join(part_image_dir, "{0}.jpg".format(part_count)))
                            part_count += 1
                        elif iou < 0.3:  # 负样本// <0.3
                            negative_anno_file.write(
                                "negative/{0}.jpg {1} 0 0 0 0\n".format(negative_count, 0))
                            negative_anno_file.flush()
                            face_resize.save(os.path.join(negative_image_dir, "{0}.jpg".format(negative_count)))
                            negative_count += 1

                        count = positive_count+part_count+negative_count
                        print(count)
                    if count >= stop_value:
                        break

            except:
                traceback.print_exc()  # Python使用traceback.print_exc()来代替print e 来输出详细的异常信息
    except:
        traceback.print_exc()

    # finally:
    #     positive_anno_file.close()
    #     negative_anno_file.close()
    #     part_anno_file.close()


gen_sample(12, 50000)
gen_sample(24, 50000)
gen_sample(48, 50000)