# Day 2｜Dataset 与数据增强（MNIST）

## 一、任务简介

本实验为 **Day 2：Dataset 与数据增强**，目标是：

* 使用 PyTorch 自定义 `MNISTDataset`
* 设计基础的数据增强流程（旋转、裁剪、仿射、归一化等）
* 验证数据增强是否能够正确作用于训练数据

本实验重点在于 **数据输入流水线的规范化**，而非模型结构设计。

---

## 二、目录结构（已给定）

```text
day2/
├── cnn/
│   ├── datasets/        # Dataset 与 transforms
│   ├── utils/           # 可视化工具
│   ├── data/            # MNIST 数据（自动生成）
│   ├── train.py         # 运行入口
│   └── requirements.txt
```

> `data/` 目录在首次运行时会自动下载 MNIST 数据。

---

## 三、代码运行方式

### 1. 环境准备（示例）

```bash
conda activate eiv
pip install -r requirements.txt
```

### 2. 运行程序

在 `day2/cnn` 目录下执行：

```bash
python train.py
```

程序将：

* 加载 MNIST 数据集
* 应用训练集数据增强
* 对数据进行简单遍历与可视化验证

---

## 四、实验任务要求

完成以下内容即可视为 **Day 2 合格完成**：

1. `MNISTDataset` 正确实现 `__len__` 与 `__getitem__`
2. 数据增强至少包含：

   * 随机旋转（Rotation）
   * 随机裁剪或仿射变换（Crop / Affine）
   * 归一化（Normalize）
3. 程序可正常运行，无报错

---

## 五、说明

* 本实验不要求训练模型或输出准确率
* 数据增强是否生效，可通过可视化或调试方式自行验证
* 本代码将在 Day 3 中继续用于 CNN 模型训练
