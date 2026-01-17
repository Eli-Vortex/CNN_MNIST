# Day 6｜训练过程可视化与调参分析（TensorBoard）

## 一、任务简介

Day 6 的目标是**引入 TensorBoard 对训练过程进行可视化**，将训练中的 Loss、Accuracy 等指标以曲线形式展示，并通过多组超参数配置进行对比分析，理解不同参数对模型收敛行为的影响。

本实验重点关注：

* TensorBoard 的接入与使用
* 多组实验结果的可视化对比
* 基于曲线进行调参分析

---

## 二、前置条件

在开始 Day 6 之前，请确保已经完成：

* **Day 1**：环境与项目初始化
* **Day 2**：Dataset 与数据增强
* **Day 3**：CNN 模型训练与评估
* **Day 4**：Config 管理
* **Day 5**：日志系统与 Checkpoint

当前训练代码已支持通过配置文件控制训练参数。

---

## 三、目录结构规范

Day 6 在 Day5 的基础上，新增 TensorBoard 日志与多组配置文件：

```text
day6/
└── cnn/
    ├── configs/              # 不同超参数配置
    │   ├── lr_1e-3.yaml
    │   └── lr_5e-4.yaml
    ├── runs/                 # TensorBoard 日志目录
    │   ├── lr_1e-3/
    │   └── lr_5e-4/
    └── train.py
```

> `runs/` 目录为运行时生成，不建议提交到 GitHub。

---

## 四、上午任务：TensorBoard 接入

### 1. 确认 TensorBoard 是否已安装

```bash
tensorboard --version
```

### 2. 运行训练并生成日志

```bash
python train.py --config configs/lr_5e-4.yaml
```

训练过程中将自动在 `runs/` 目录下生成 TensorBoard 事件文件。

### 3. 启动 TensorBoard

```bash
tensorboard --logdir runs
```

在浏览器中访问：

```text
http://localhost:6006
```

---

## 五、下午任务：调参与可视化对比

### 1. 使用不同配置文件运行实验

```bash
python train.py --config configs/lr_1e-3.yaml
python train.py --config configs/lr_5e-4.yaml
```

### 2. 对比分析

在 TensorBoard 页面中：

* 对比不同学习率下的 Loss 曲线
* 对比 Accuracy 的收敛速度与稳定性
* 分析是否出现震荡、过拟合或收敛过慢现象

---

## 六、验收标准

完成以下要求即可判定 **Day 6 合格完成**：

1. TensorBoard 页面可正常打开
2. `runs/` 目录下存在事件文件
3. TensorBoard 中显示至少两组实验曲线
4. 能给出明确的调参结论
