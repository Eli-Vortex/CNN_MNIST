# Day 1｜环境与项目初始化（CNN + MNIST）

## 一、任务简介

Day 1 的目标是**完成深度学习实验的基础环境搭建**，并跑通一个最小可用的 **MNIST + CNN 训练流程**，为后续 Dataset、数据增强和模型训练实验打下稳定基础。

本日重点关注：

* 环境是否可复现
* 依赖是否完整
* 训练流程是否能正确跑通

---

## 二、环境准备

### 1. 创建并激活虚拟环境（示例）

```bash
conda create -n cnn python=3.9 -y
conda activate cnn
```

### 2. 安装依赖

```bash
pip install torch torchvision tensorboard pyyaml
```

### 3. 验证依赖是否安装成功

```bash
python - <<EOF
import torch
import torchvision
import tensorboard
import yaml
print("All OK")
EOF
```

### 4. 验证计算设备

```bash
python - <<EOF
import torch
print("CUDA available:", torch.cuda.is_available())
EOF
```

### 5. 生成依赖清单

```bash
pip freeze > requirements.txt
```

---

## 三、最小训练流程

### 1. 创建项目目录结构

```bash
mkdir -p cnn/data cnn/models
cd cnn
```

### 2. 创建训练脚本

```bash
touch train.py
```

### 3. 编写并运行训练代码

在 `train.py` 中完成以下内容：

* 使用 `torchvision.datasets.MNIST` 加载数据集（`download=True`）
* 定义一个最小 CNN 网络（卷积 + 池化 + 全连接）
* 完成 1 个 epoch 的训练

运行：

```bash
python train.py
```

---

## 四、验收标准

当满足以下条件时，判定 **Day 1 完成**：

1. 虚拟环境创建成功，依赖安装无报错
2. 成功生成 `requirements.txt`
3. MNIST 数据可正常下载
4. CNN 训练脚本可运行并完成 **至少 1 个 epoch**
