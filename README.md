## DIP Project Program Readme

**测试模型请直接查看test文件夹内的README.md文档**

文件夹功能说明：

- **./negative-samples-generation** 生成假数据
  - generate-negative-samples.ipynb 根据4k图片生成加图片
  - generate_224.ipynb 在大图片上采样，生成224*224小图片
  - image_tensor.ipynb 把图片转换为numpy数据，保存在npy文件中

- **./models** 训练模型
  - dct-cnn, pytorch-cnn, resnet18 共三种模型
  - 各文件夹内main.py用于训练，模型自动保存为checkpoint文件夹内的ckpt文件
  - 各文件夹内test_model.py用于测试模型在数据集上的表现，计算accuracy等
- **./test** 测试模型请查看该文件夹内的README.md文档

