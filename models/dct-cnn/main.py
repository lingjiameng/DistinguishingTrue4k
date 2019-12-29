# -*-coding:utf-8 -*-
# 训练4k分辨网络 
'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import os

import argparse

from PIL import Image
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import cv2

from simple_net import *
# from utils import progress_bar

# 记录训练过程
writer = SummaryWriter('run_data')

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

BATCH_SIZE = 64

parser = argparse.ArgumentParser(description='PyTorch 4k Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--device', default="2,3", type=str, help='device og GPU')
parser.add_argument('--epoch', default=200, type=int, help='epoch for trainning')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--batch-size','-b',default=64,type=int,help="batch size")
parser.add_argument("--negdataset",default="1080p",type=str,help="negative dataset(1080p or 720p)")

args = parser.parse_args()

## 设定 batch_size
BATCH_SIZE = args.batch_size

NEG_DATA = "neg.npy" # 1080p
if args.negdataset == "720p":
    NEG_DATA="neg720.npy" #720p

# 设置可见gpu
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
device = "cuda"  if torch.cuda.is_available() else 'cpu'

print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.current_device())

# device = args.device
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print("加载并DCT处理训练数据...")
def dct(src):
    """
    Parameters:
        src : image set in n*RGB*H*W format and float32 dtpye 
    Returns:
        dst: image set returnd in n*H*W DCT format and float32 dtype
    """
    # tmp = np.matmul(src.transpose((0,2,3,1)),np.array([0.299,0.287,0.114],dtype=np.float32))
    result = []
    for i in range(src.shape[0]):
        tmp = []
        tmp.append(cv2.dct(src[i,0,:]))
        tmp.append(cv2.dct(src[i,1,:]))
        tmp.append(cv2.dct(src[i,2,:]))
        result.append(np.stack(tmp))
    dst = np.stack(result)
    return dst

posx = torch.from_numpy(dct(np.load(os.path.join(TENSOR_DIR, "pos.npy"))))
posy = torch.ones(posx.shape[0])
negx = torch.from_numpy(dct(np.load(os.path.join(TENSOR_DIR, NEG_DATA))))
negy = torch.zeros(negx.shape[0])
x = torch.cat([posx, negx], 0)
y = torch.cat([posy, negy], 0)
print(x.shape, y.shape)
torch_dataset = Data.TensorDataset(x,y)
# 分割数据集
train_size = int(0.8 * len(torch_dataset))
test_size = len(torch_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(torch_dataset, [train_size, test_size])
# 加载成loader
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

# Model
print('==> Building model..')
net = CNN()

net = net.to(device)  
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    # print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    num_batch = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # print(batch_idx)
        num_batch += 1
        inputs, targets = inputs.to(device), targets.long().to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += float(loss.item())
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return float(train_loss) / num_batch, correct, total
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    # print("test", epoch)
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    num_batch = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            num_batch += 1
            # print( "inputs:", inputs.shape)
            # print("device", device)
            inputs, targets = inputs.to(device), targets.long().to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
    return float(test_loss) / num_batch, correct, total

print("start trainning...")
for epoch in range(start_epoch, start_epoch+args.epoch):
    
    train_loss, correct, total = train(epoch)
    train_accu = float(correct) / total
    print("epoch:{} train_loss: {:.5f}  train_accu:{:.3f}".format(epoch, train_loss, train_accu), end = " ")

    test_loss, correct, total = test(epoch)
    test_accu = float(correct) / total
    print("test_loss: {:.5f}  test_accu:{:.3f}".format(test_loss, test_accu))
    writer.add_scalars('DCT-DNN 4k vs'+args.negdataset+" batch-size "+str(BATCH_SIZE), {'train_loss': train_loss, 
                                            'train_accu':train_accu,
                                            'test_loss': test_loss,
                                            'test_accu':test_accu}, epoch)
