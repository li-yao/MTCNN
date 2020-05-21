from PIL import Image
from PIL import ImageDraw
import os

IMG_DIR = r"E:\datasets\img_celeba_ALL"
ANO_DIR = r"E:\datasets\list_bbox_celeba_ALL.txt"

img = Image.open(os.path.join(IMG_DIR, "000018.jpg"))
imgDraw = ImageDraw.Draw(img)
imgDraw.rectangle((140, 84, 140 + 195, 84 + 270),outline="red")
img.show()
