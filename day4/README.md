# Day 4｜Config 管理（CNN 训练工程化）

## 一、任务简介

Day 4 的目标是**引入配置文件（Config）机制**，将训练过程中使用的超参数与代码逻辑解耦，实现：

> **“只改配置文件，不改代码即可复现实验”**

通过本实验，你将把 Day3 中已经跑通的 CNN 训练流程，升级为**工程化、可复现的实验系统**。

---

## 二、前置条件

在开始 Day 4 之前，请确保：

* 已完成 **Day 1：环境与项目初始化**
* 已完成 **Day 2：Dataset 与数据增强**
* 已完成 **Day 3：CNN 模型训练与评估**（准确率 ≥ 90%）
* 当前 CNN 训练代码可以正常运行

---

## 三、目录结构规范

Day 4 要求在 Day3 的基础上，引入 `configs/` 目录：

```text
day4/
└── cnn/
    ├── data/                 # MNIST 数据（自动生成，不提交）
    ├── datasets/             # Dataset 与 transforms
    ├── models/               # CNN 模型
    │   └── cnn.py
    ├── utils/
    ├── configs/              # 配置文件目录
    │   └── config.yaml
    └── train.py              # 完全由配置文件驱动
```

---

## 四、核心任务说明

### 1. 设计配置文件（config.yaml）

在 `configs/config.yaml` 中集中管理以下参数：

* **数据参数**：

  * `batch_size`
  * `data_root`

* **模型参数**：

  * `num_classes`

* **训练参数**：

  * `epochs`
  * `learning_rate`

* **运行参数**：

  * `device`（如：`cpu` / `cuda` / `auto`）

---

### 2. 训练脚本配置化（train.py）

在 `train.py` 中完成以下修改：

* 使用 `pyyaml` 读取 `config.yaml`
* 用配置文件参数替换所有硬编码超参数
* 根据 `device` 自动选择运行设备

要求：

* `train.py` 中**不允许出现写死的训练超参数**
* 修改 `config.yaml` 后，无需改代码即可改变训练行为

---

## 五、运行方式

在 `day4/cnn` 目录下执行：

```bash
conda activate eiv
python train.py
```

程序将：

* 从 `configs/config.yaml` 读取配置
* 自动完成模型训练与评估

---

## 六、验收标准

完成以下要求即可判定 **Day 4 合格完成**：

1. 所有训练相关参数均来自 `config.yaml`
2. `train.py` 中无硬编码训练参数
3. 修改配置文件即可复现实验结果
4. 训练流程可正常运行

---

## 七、说明

* 本阶段不涉及日志系统、TensorBoard 或模型保存
* 重点在于 **配置驱动训练流程** 的工程思想
* 后续实验将在此基础上继续扩展
