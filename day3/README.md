# Day 3｜CNN 模型训练与评估（MNIST）

## 一、任务简介

Day 3 的目标是完成 **CNN 分类模型的完整训练与评估流程**，在 MNIST 手写数字数据集上实现一个可运行、结构清晰的训练闭环，并达到 **≥90% 的测试准确率**。

本实验重点关注：

* CNN 网络结构设计
* Train / Eval 标准流程
* 训练与评估模式的正确区分

---

## 二、前置条件

在开始 Day 3 之前，请确保：

* 已完成 **Day 1：环境与项目初始化**
* 已完成 **Day 2：Dataset 与数据增强**
* MNIST 数据加载与增强流程运行正常

---

## 三、目录结构

```text
day3/
└── cnn/
    ├── datasets/      # 数据集与增强（复用 Day2）
    ├── models/        # CNN 模型定义
    │   └── cnn.py
    ├── utils/         # 辅助工具（如可视化）
    ├── train.py       # 训练与评估入口
    └── eval.py        # （可选）独立评估脚本
```

---

## 四、主要任务

### 1. CNN 网络结构

* 在 `models/cnn.py` 中定义 CNN 网络
* 至少包含：

  * 卷积层（Conv）
  * 池化层（Pooling）
  * 全连接层（FC）
* 使用随机输入验证 forward 输出形状为：

```text
(batch_size, 10)
```

---

### 2. 训练与评估流程

在 `train.py` 中完成：

* `train_one_epoch`：单 epoch 训练逻辑
* `evaluate`：测试集评估逻辑
* 正确使用：

  * `model.train()`
  * `model.eval()`
  * `torch.no_grad()`

训练 3–5 个 epoch，并在测试集上计算分类准确率。

---

## 五、运行方式

在服务器或本地环境中执行：

```bash
conda activate eiv
python train.py
```

程序将自动完成：

* 模型训练
* 测试集评估
* 输出准确率结果

---

## 六、验收标准

完成以下要求即可判定 **Day 3 合格完成**：

1. CNN forward 验证无报错
2. 训练与评估逻辑结构清晰
3. 评估阶段使用 `torch.no_grad()`
4. **3–5 个 epoch 内测试准确率 ≥ 90%**

---

## 七、说明

* 本实验不要求复杂模型调参
* 重点在于流程完整性与代码规范性
* 该代码可作为后续实验或课程项目的基础模板
