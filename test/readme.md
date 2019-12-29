# 真4k分类器

## 环境

- python 3.6
  - jupyter==1.0.0
  - matplotlib==3.1.0
  - numpy==1.16.4
  - opencv3==3.1.0
  - Pillow==6.0.0
  - tensorboard==2.0.2
  - tensorboardX==1.7
  - torch==1.1.0
  - torchvision==0.3.0

## 使用方法
```bash
usage: main.py [-h] [--device DEVICE] [--image IMAGE] [--image-dir IMAGE_DIR]
               [--model MODEL]

True 4k Classifier

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE       device number of GPU
  --image IMAGE, -i IMAGE
                        image file path like ./4k.jpg
  --image-dir IMAGE_DIR, -d IMAGE_DIR
                        image folder path like ./img/
  --model MODEL, -m MODEL
                        choose which model to use(Resnet18,DCT-CNN or CNN.Default: Resnet18)
```

示例
```shell
python main.py -d ./ -m DCT-CNN
```
## 可用模型

- Resnet18
- DCT-CNN **我们特别提供的模型**
- CNN
