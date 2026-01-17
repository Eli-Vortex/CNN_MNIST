# CNN 手写数字识别（MNIST） 
 
 本项目基于 **PyTorch** 实现 CNN 手写数字识别任务（MNIST），  
 按照 **7 天 SOP（Standard Operating Procedure）** 的方式逐步完成： 
 
 > 环境 → 数据 → 模型 → 训练 → 工程化 → 可视化 → 交付 
 
 适用于： 
 - 深度学习课程实验 
 - CNN 入门教学 
 - 工程化训练流程示例 
 - 可复现实验模板 
 
 --- 
 
 ## 一、项目目标 
 
 - 使用 CNN 完成 MNIST 手写数字分类任务 
 - 构建 **可复现、可中断、可对比** 的训练流程 
 - 形成一套 **教学级 SOP 实验范式** 
 
 最终效果： 
 - MNIST 测试集准确率 ≥ **95%** 
 - 支持 TensorBoard 可视化 
 - 支持模型保存与恢复（Checkpoint） 
 
 --- 
 
 ## 二、环境要求 
 
 - Python ≥ 3.8 
 - PyTorch 
 - torchvision 
 - tensorboard 
 - pyyaml 
 
 推荐使用 **conda 虚拟环境**。 
 
 --- 
 
 ## 三、环境安装 
 
 ### 1. 创建并激活环境（示例） 
 
 ```bash 
 conda create -n eiv python=3.10 -y 
 conda activate eiv 
 ```
 
 ### 2. 安装依赖 
 
 ```bash 
 pip install torch torchvision tensorboard pyyaml 
 ```
 
 或使用： 
 
 ```bash 
 pip install -r requirements.txt 
 ```
 
 ## 四、项目目录结构 
 
 ```powershell 
 cnn/ 
 ├── configs/                  # 配置文件（Day 4 / Day 6） 
 │   ├── config.yaml 
 │   ├── lr_1e-3.yaml 
 │   └── lr_5e-4.yaml 
 ├── data/                     # 数据集（MNIST 自动下载） 
 ├── data_modules/             # 数据模块（Day 2） 
 │   ├── __init__.py 
 │   ├── mnist_dataset.py 
 │   └── transforms.py 
 ├── models/                   # 模型定义（Day 3） 
 │   ├── __init__.py 
 │   └── cnn.py 
 ├── utils/                    # 工具模块（Day 5） 
 │   └── logger.py 
 ├── logs/                     # 训练日志（Day 5） 
 ├── checkpoints/              # 模型权重（Day 5） 
 │   └── best.pth 
 ├── runs/                     # TensorBoard 日志（Day 6） 
 ├── train.py                  # 主训练脚本 
 ├── requirements.txt 
 └── README.md 
 ```
 
 ## 五、快速开始（最小复现） 
 
 1. 进入项目目录 
 
 ```bash 
 cd cnn 
 ```
 
 2. 运行训练（默认配置） 
 
 ```bash 
 python train.py 
 ```
 
 训练过程中会： 
 - 自动下载 MNIST 
 - 训练 CNN 
 - 保存最优模型到 checkpoints/best.pth 
 - 记录日志到 logs/ 
 - 写入 TensorBoard 数据到 runs/ 
 
 ## 六、使用配置文件运行实验（Day 4） 
 
 示例：使用不同学习率配置 
 
 ```bash 
 python train.py --config configs/lr_1e-3.yaml 
 python train.py --config configs/lr_5e-4.yaml 
 ```
 
 无需修改任何代码，只需切换配置文件。 
 
 ## 七、恢复训练（Checkpoint，Day 5） 
 
 如果训练中断或希望从最优模型继续训练： 
 
 ```bash 
 python train.py --resume checkpoints/best.pth 
 ```
 
 恢复内容包括： 
 - 模型参数 
 - 优化器状态 
 - 上一次 epoch 
 - 最佳准确率 
 
 ## 八、TensorBoard 可视化（Day 6） 
 
 1. 启动 TensorBoard（本地） 
 
 ```bash 
 tensorboard --logdir runs 
 ```
 
 浏览器访问： 
 `http://localhost:6006` 
 
 2. 查看内容 
 - Loss 曲线（训练过程） 
 - Accuracy 曲线（测试集） 
 - 多配置实验对比（不同 learning rate） 
 
 ## 九、数据增强说明（Day 2） 
 
 训练集使用的数据增强包括： 
 - 随机旋转（RandomRotation） 
 - 随机裁剪（RandomCrop + padding） 
 - 随机仿射变换（RandomAffine） 
 - 平移（translate） 
 - 缩放（scale） 
 - 剪切（shear） 
 - 归一化（Normalize） 
 
 测试集仅使用： 
 - ToTensor 
 - Normalize 
 
 ## 十、训练流程说明（Day 3） 
 
 训练流程遵循标准深度学习范式： 
 
 ```css 
 for epoch: 
     train_one_epoch() 
     evaluate() 
 ```
 
 - `model.train()`：训练模式（Dropout / BN 生效） 
 - `model.eval()`：评估模式（关闭 Dropout，使用 BN 统计量） 
 - 使用 CrossEntropyLoss 
 - 使用 Adam 优化器
