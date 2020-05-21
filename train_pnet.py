from MTCNN.MTCNN_PyTorch_teach import nets
from MTCNN.MTCNN_PyTorch_teach import train
import os

if __name__ == '__main__':
    net = nets.PNet()
    if not os.path.exists("./param"):
        os.makedirs("./param")
    trainer = train.Trainer(net, './param/p_net.pth', r"C:\Users\Administrator\Desktop\临时文件夹\CelebA\datasets\12")
    trainer.train(0.01)
