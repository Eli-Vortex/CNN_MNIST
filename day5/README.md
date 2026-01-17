\# Day 5｜日志系统与模型保存 / 恢复（Checkpoint）



\## 一、任务简介



Day 5 的目标是为 CNN 训练流程引入\*\*日志系统（Logging）\*\*与\*\*模型保存 / 恢复机制（Checkpoint）\*\*，使训练过程具备：



\* 可追溯（有日志）

\* 可中断（训练可随时停止）

\* 可恢复（从中断处继续训练）



这是从“实验代码”走向“工程化训练代码”的关键一步。



---



\## 二、前置条件



在开始 Day 5 之前，请确保已经完成：



\* \*\*Day 1\*\*：环境与项目初始化

\* \*\*Day 2\*\*：Dataset 与数据增强

\* \*\*Day 3\*\*：CNN 模型训练与评估

\* \*\*Day 4\*\*：Config 管理（配置驱动训练）



当前 `train.py` 已可通过 `config.yaml` 正常训练模型。



---



\## 三、目录结构规范



在 Day4 的基础上，新增日志与 checkpoint 相关目录：



```text

day5/

└── cnn/

&nbsp;   ├── datasets/

&nbsp;   ├── models/

&nbsp;   ├── utils/

&nbsp;   │   └── logger.py        # 日志工具

&nbsp;   ├── configs/

&nbsp;   │   └── config.yaml

&nbsp;   ├── logs/                # 训练日志

&nbsp;   │   └── train.log

&nbsp;   ├── checkpoints/         # 模型保存目录

&nbsp;   │   └── best.pth

&nbsp;   └── train.py

```



> `logs/` 与 `checkpoints/` 目录仅用于运行时生成，不建议提交到 GitHub。



---



\## 四、上午任务：日志系统



\### 1. 创建日志目录



```bash

mkdir -p logs

```



\### 2. 创建日志工具文件



```bash

touch utils/logger.py

```



\### 3. 日志实现要求



\* 使用 Python 标准库 `logging`

\* 日志同时输出到：



&nbsp; \* 终端

&nbsp; \* `logs/train.log`

\* 在 `train.py` 中使用 `logger.info()` 替代 `print()`



---



\## 五、下午任务：模型保存与恢复（Checkpoint）



\### 1. 创建 checkpoint 目录



```bash

mkdir -p checkpoints

```



\### 2. 保存内容要求



保存以下信息：



\* 模型参数（`model.state\_dict()`）

\* 优化器状态（`optimizer.state\_dict()`）

\* 当前 epoch

\* 当前或最佳准确率



\### 3. 保存策略



\* 当验证集准确率提升时，保存 `best.pth`



\### 4. 恢复训练



\* 支持从 checkpoint 恢复训练

\* 不需要新建脚本，通过参数或配置控制是否 resume



示例：



```bash

python train.py --resume checkpoints/best.pth

```



---



\## 六、运行方式



\### 1. 正常训练



```bash

conda activate eiv

python train.py

```



\### 2. 恢复训练



```bash

python train.py --resume checkpoints/best.pth

```



---



\## 七、验收标准



完成以下要求即可判定 \*\*Day 5 合格完成\*\*：



1\. `logs/train.log` 成功生成并持续记录训练信息

2\. `checkpoints/best.pth` 成功生成

3\. 训练中断后可从 checkpoint 恢复

4\. 恢复后训练指标连续、不突变



---



\## 八、说明



\* 本阶段不涉及 TensorBoard（将在 Day 6 实现）

\* 重点在于训练过程的工程化与可复现性

\* Day5 是后续调参、对比实验的基础



