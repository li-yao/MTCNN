from MTCNN.MTCNN_PyTorch_teach import nets
from MTCNN.MTCNN_PyTorch_teach import train
import os

if __name__ == '__main__':
    net = nets.ONet()
    if not os.path.exists("./param"):
        os.makedirs("./param")
    trainer = train.Trainer(net, './param/o_net.pth', r"C:\Users\Administrator\Desktop\临时文件夹\CelebA\datasets\48")
    # trainer.train(0.0005, 0.4)
    trainer.train(0.0005)