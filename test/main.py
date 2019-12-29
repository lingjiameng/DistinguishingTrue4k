# -*-coding:utf-8 -*-
# 测试网络
'''Test model with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os

import argparse
from models import *

from PIL import Image
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import cv2

SMALL_IMG_HEIGHT = 224
SMALL_IMG_WIDTH = 224


parser = argparse.ArgumentParser(description='True 4k Classifier')
parser.add_argument('--device', default="0", type=str, help='device number of GPU')
parser.add_argument("--image","-i",default="x",type=str,help="image file path like ./4k.jpg")
parser.add_argument("--image-dir","-d",default="x",type=str,help="image folder path like ./img/")
parser.add_argument("--model","-m",default="Resnet18",type=str,help="choose which model to use(Resnet18,DCT-CNN or CNN. Default: Resnet18)")

args = parser.parse_args()

# 设置可见gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
device = "cuda"  if torch.cuda.is_available() else 'cpu'


## 加载网络及模型
ALL_MODELS = ["Resnet18","DCT-CNN","CNN"]
assert args.model in ALL_MODELS, "Invalid model name"
assert os.path.exists("./models/{}.pth".format(args.model)),"No trained model file"

print("==> Loading model {} from ./models/{}.pth..".format(args.model,args.model))
net = []
checkpoint = []
USE_DCT = False
if args.model=="Resnet18":
    net = ResNet18()
    checkpoint = torch.load("./models/Resnet18.pth")
elif args.model== "DCT-CNN":
    net = CNN()
    checkpoint = torch.load("./models/DCT-CNN.pth")
    USE_DCT = True
elif args.model =="CNN":
    net = CNN()
    checkpoint = torch.load("./models/CNN.pth")
# print(net)

net = net.to(device)  
if device == 'cuda':
	net = torch.nn.DataParallel(net)
	cudnn.benchmark = True

# load model to net
net.load_state_dict(checkpoint['net'])


## 加载图片文件或者图片目录
assert os.path.exists(args.image) or os.path.exists(args.image_dir),"Invalid image path"

IS_DIR = False
image_path = []
if args.image_dir != "x":
    IS_DIR = True

if IS_DIR:
    print("==> Load image from folder",args.image_dir)
    files = os.listdir(args.image_dir)
    try:
        files.sort(key=lambda x: int(x.split(".")[0]))
    except:
        pass
    for fn in files:
        if fn.split(".")[-1] in ["bmp","png","jpg", "jpeg"]:
            image_path.append(os.path.join(args.image_dir,fn))
else:
    print("==> Load image from file", args.image)
    image_path.append(args.image)

# 图片转换为tensor
loader = transforms.Compose([
	transforms.ToTensor()])  

def dct(src):
    """
    Parameters:
        src : image set in RGB*H*W format and float32 dtpye 
    Returns:
        dst: image set returnd in 3*H*W DCT format and float32 dtype
    """
    result = []
    result.append(cv2.dct(src[0, :]))
    result.append(cv2.dct(src[1, :]))
    result.append(cv2.dct(src[2, :]))
    dst = np.stack(result)
    return dst


def is_4k(image):
    """
    Inputs:
        image : 4k resolution image RGB*H*W in numpy format
    Returns:
        Boolean: whether is true 4k
    """
    ## cut image to 9 * 224*224 pic
    image_tiles = []
    delta_height = int(image.shape[1] / 4)
    delta_width = int(image.shape[2] / 4)

    for i in [1, 2, 3]:
        for j in [1, 2, 3]:
            sx = delta_height * i - int(SMALL_IMG_HEIGHT / 2)
            sy = delta_width * j - int(SMALL_IMG_WIDTH / 2)
            small_img = image[:, sx:sx+SMALL_IMG_HEIGHT, sy:sy+SMALL_IMG_WIDTH]
            if USE_DCT:  # whether dct or not
                small_img = dct(small_img)
            image_tiles.append(small_img)

    # judge it
    net.eval()
    with torch.no_grad():
        pos, neg = 0, 0
        for image_tile in image_tiles:
            inputs = torch.from_numpy(image_tile[np.newaxis,:])
            inputs = inputs.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            if int(predicted) == 0:
                neg += 1
            else:
                pos += 1
        # print("pos and neg:", pos, neg)
        return pos > neg


for fn in image_path:
    ## read image
    image = loader(Image.open(fn).convert('RGB')).numpy()
    print("Image:",fn,"is true 4k: ",is_4k(image))
