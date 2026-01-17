# utils/visualize.py
import os
import matplotlib
matplotlib.use("Agg")   # 服务器环境必须
import matplotlib.pyplot as plt


def save_augmentation_comparison(
    raw_dataset,
    aug_dataset,
    indices,
    save_dir="outputs/augment_compare"
):
    """
    保存多组“增强前 vs 增强后”的对比图

    Args:
        raw_dataset: 不含数据增强的 Dataset
        aug_dataset: 含数据增强的 Dataset
        indices: 样本索引列表，如 [0,1,2,3,4]
        save_dir: 输出目录
    """
    os.makedirs(save_dir, exist_ok=True)

    for idx in indices:
        raw_img, label = raw_dataset[idx]
        aug_img, _ = aug_dataset[idx]

        raw_img = raw_img.squeeze()
        aug_img = aug_img.squeeze()

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))

        axes[0].imshow(raw_img, cmap="gray")
        axes[0].set_title("Before Augmentation")
        axes[0].axis("off")

        axes[1].imshow(aug_img, cmap="gray")
        axes[1].set_title("After Augmentation")
        axes[1].axis("off")

        fig.suptitle(f"Label: {label}")

        save_path = os.path.join(save_dir, f"sample_{idx}.png")
        plt.savefig(save_path)
        plt.close()

        print(f"[INFO] Saved augmentation comparison: {save_path}")
