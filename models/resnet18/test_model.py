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
from utils import progress_bar

from PIL import Image
from torchvision import transforms
import torch.utils.data as Data
import numpy as np



# 重要的全局变量
POS_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../", "dataset", "positive"))
NEG_DIR = os.path.abspath(os.path.join(POS_DIR, "..", "negative"))

# 保存小图片的文件夹
POS_224_DIR = os.path.abspath(os.path.join(POS_DIR, "..", "positive224"))
NEG_224_DIR = os.path.abspath(os.path.join(POS_DIR, "..", "negative224"))
# 
SMALL_IMG_HEIGHT = 224
SMALL_IMG_WIDTH = 224
# 读取tensor
TENSOR_DIR = os.path.abspath(os.path.join(POS_DIR, "..", "tensor"))

parser = argparse.ArgumentParser(description='PyTorch 4k Training')
parser.add_argument('--device', default="2,3", type=str, help='device og GPU')
parser.add_argument("--negdataset",default="1080p",type=str,help="negative dataset(1080p or 720p)")
parser.add_argument("--checkpoint",default=".",type=str,help="model path")
args = parser.parse_args()

# NEG_DATA = "neg.npy" # 1080p
# if args.negdataset == "720p":
#     NEG_DATA="neg720.npy" #720p

# 设置可见gpu
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
device = "cuda"  if torch.cuda.is_available() else 'cpu'


# 加载网络及模型
print('==> Building model..')
net = ResNet18()

net = net.to(device)  
if device == 'cuda':
	net = torch.nn.DataParallel(net)
	cudnn.benchmark = True
# load model
checkpoint = torch.load(args.checkpoint)
net.load_state_dict(checkpoint['net'])

# 图片转换为tensor
loader = transforms.Compose([
	transforms.ToTensor()])  

def get_tensor(image_name):
	image = Image.open(image_name).convert('RGB')
	image_tensor = loader(image)
	return image_tensor

posfiles = os.listdir(POS_224_DIR)
negfiles = os.listdir(NEG_224_DIR)

poslist = [[] for i in range(201)]
neglist = [[] for i in range(201)]

for name in posfiles:
	if "pos" not in name:
		continue
	poslist[int(name.split('.')[0])].append(name)

for name in negfiles:
	if args.negdataset not in name:
		continue
	neglist[int(name.split('.')[0])].append(name)

TP, FN = 0, 0
FP, TN = 0, 0

_TP, _FN = 0, 0
_FP, _TN = 0, 0


net.eval()
with torch.no_grad():
		
	print("正在计算正样本...")
	for i in range(1,201):
		pos, neg = 0, 0
		for image_name in poslist[i]:
			inputs = get_tensor(os.path.join(POS_224_DIR, image_name)).unsqueeze(0)
			inputs = inputs.to(device)
			outputs = net(inputs)
			_, predicted = outputs.max(1)
			if int(predicted) == 0:
				neg += 1
				_FN += 1
			else:
				pos += 1
				_TP += 1
		if pos > neg:
			TP += 1
		else:
			FN += 1

	print("正在计算负样本...")
	for i in range(1,201):
		pos, neg = 0, 0
		for image_name in neglist[i]:
			inputs = get_tensor(os.path.join(NEG_224_DIR, image_name)).unsqueeze(0)
			inputs = inputs.to(device)
			# print(inputs.shape)
			outputs = net(inputs)
			# print(outputs.cpu())
			_, predicted = outputs.max(1)
			if int(predicted) == 0:
				neg += 1
				_TN += 1
			else:
				pos += 1
				_FP += 1
		if pos > neg:
			FP += 1
		else:
			TN += 1
	print("_TP, _FN, _FP, _TN:", _TP, _FN, _FP, _TN)
	print("TP, FN, FP, TN:", TP, FN, FP, TN)

	def compute_performance(TP, FN, FP, TN):
		F1 = float(2 * TP) / float(2 * TP + FP + FN)
		accuracy = float(TP + TN) / float(TP + FP + TN + FN)
		recall = float(TP) / float(TP + FN)
		precision = float(TP) / float(TP + FP)

		return F1, accuracy, recall, precision

	_F1, _accuracy, _recall, _precision = compute_performance(_TP, _FN, _FP, _TN)
	print("_F1:", _F1)
	print("_accuracy:", _accuracy)
	print("_recall:", _recall)
	print("_precision:", _precision)


	F1, accuracy, recall, precision = compute_performance(TP, FN, FP, TN)
	print("F1:", F1)
	print("accuracy:", accuracy)
	print("recall:", recall)
	print("precision:", precision)

