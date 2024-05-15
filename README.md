# zynq7020_mnist
A simple neural network that identifies mnist is deployed in the PL of zynq7020 to implement forward prediction.

## 项目流程

### 一、生成预先计划的图片数据文件
使用python脚本读取csv文件：read_image_v01.py

### 二、搭建模型，训练神经网络，得到权重参数
使用python脚本实现：neural_v02.py

### 三、HLS + Vivado + Vitis 设计
在xilinx文件夹下存放已经编译过的工程文件


## 文件结构

项目主要脚本存放在 `main/` 文件夹下，模型所需数据存放于 `data/` 文件夹，输出文件存放在 `out/` 文件夹下。
`xilinx/`文件夹下放置zynq开发的软件工程。

## 运行指南

本项目基于 Python 编程语言，程序运行使用的 Python 版本不限，建议使用 [Anaconda](https://www.anaconda.com) 配置 Python 环境。以下配置过程已在 windows10 系统测试通过。


### python配置环境

```
conda create -n ZyMn python=3.10
conda activate ZyMn
cd ZyMn

```

### zynq开发环境

使用xilinx开发三件套版本为2020.2



## 参考
本项目借助贴文 [如何从零开始将神经网络移植到FPGA(ZYNQ7020)加速]([https://github.com/RVC-Boss/GPT-SoVITS](https://blog.csdn.net/u012116328/article/details/117246023?spm=1001.2014.3001.5502)) 进行工程实现。

参考项目 [mnist-nnet-hls-zynq7020-fpga]([https://github.com/RVC-Boss/GPT-SoVITS](https://github.com/doveyour/mnist-nnet-hls-zynq7020-fpga)) 的开源代码进行加工生成程序。



