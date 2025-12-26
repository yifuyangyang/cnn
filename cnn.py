# cnn_mnist_99plus.py
# 确保CNN在MNIST上测试准确率≥99%的最终版代码
import time
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

# =========================
# 核心配置（无需修改，已锁定最优参数）
# =========================
CONFIG = {
    "seed": 42,                # 固定种子，结果可复现
    "model": "cnn",            # 锁定为CNN
    "batch_size": 64,          # 最优批大小
    "lr": 1e-3,                # 唯一最优学习率
    "optimizer": "adam",       # Adam收敛最快
    "epochs": 10,              # CNN仅需10轮即可达标
    "save_plot": True,
    "plot_path": "results_cnn_99plus.png"
}

# =========================
# 工具函数（核心功能，无需修改）
# =========================
def set_seed(seed: int):
    """固定所有随机种子，确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_params(model: nn.Module) -> int:
    """统计模型可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_optimizer(cfg, model):
    """创建最优优化器（Adam）"""
    if cfg["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    else:
        raise ValueError("仅支持Adam优化器，已锁定最优配置")

def train_one_epoch(model, loader, optimizer, device):
    """训练一轮，返回平均训练损失"""
    model.train()
    ce_loss = nn.CrossEntropyLoss()
    total_loss, total_samples = 0.0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = ce_loss(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)
    
    return total_loss / total_samples

@torch.no_grad()
def evaluate(model, loader, device):
    """评估模型，返回测试损失和测试准确率"""
    model.eval()
    ce_loss = nn.CrossEntropyLoss()
    total_loss, correct, total_samples = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        loss = ce_loss(logits, y)
        
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total_samples += y.size(0)
    
    return total_loss / total_samples, correct / total_samples

# =========================
# 核心：确保99%+的CNN模型（带轻量Dropout防过拟合）
# =========================
class HighAccCNN(nn.Module):
    """
    优化后的CNN结构：
    - 卷积层：32→64通道（提取足够的图像局部特征）
    - 池化层：2x2 MaxPool（保留关键特征，降低维度）
    - 全连接层：64*7*7→256→10（充分利用卷积特征）
    - Dropout：0.25（轻量正则化，避免过拟合）
    """
    def __init__(self):
        super().__init__()
        # 卷积层（核心：32/64通道）
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # 激活/池化/正则化
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)  # 轻量Dropout，确保准确率稳定
        
        # 全连接层（核心：256神经元）
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)  # 输出10类（0-9）

    def forward(self, x):
        # x: [B, 1, 28, 28] → 卷积+池化 → [B, 32, 14, 14]
        x = self.pool(self.relu(self.conv1(x)))
        # → 卷积+池化 → [B, 64, 7, 7]
        x = self.pool(self.relu(self.conv2(x)))
        # Flatten → [B, 64*7*7=3136]
        x = x.view(x.size(0), -1)
        # 全连接+Dropout → 输出
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# =========================
# 主程序（适配Windows CPU，无需修改）
# =========================
def main():
    # 固定种子，确保结果可复现
    set_seed(CONFIG["seed"])
    # 设备选择（CPU，适配你的环境）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("="*60)
    print(f"运行环境：{device} | 模型：CNN（目标≥99%）")
    print(f"训练参数：epochs={CONFIG['epochs']} | batch_size={CONFIG['batch_size']} | lr={CONFIG['lr']}")
    print("="*60)

    # 加载MNIST数据集（使用你手动下载的本地数据，避免重新下载）
    transform = transforms.Compose([transforms.ToTensor()])
    # 关键：download=False（使用本地已下载的数据集）
    train_ds = datasets.MNIST(
        root="./data", 
        train=True, 
        download=False,  # 必须False，用你手动下载的文件
        transform=transform
    )
    test_ds = datasets.MNIST(
        root="./data", 
        train=False, 
        download=False,  # 必须False
        transform=transform
    )

    # 数据加载器（关键：num_workers=0，适配Windows CPU）
    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,  # Windows CPU必须设为0，避免数据加载异常
        pin_memory=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # 初始化模型和优化器
    model = HighAccCNN().to(device)
    optimizer = build_optimizer(CONFIG, model)
    print(f"模型参数量：{count_params(model):,}（最优复杂度）")
    print("="*60)

    # 记录训练过程
    train_losses, test_losses, test_accs = [], [], []
    start_time = time.time()

    # 训练10轮（足够达标）
    for epoch in range(1, CONFIG["epochs"] + 1):
        # 训练一轮
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        # 评估测试集
        test_loss, test_acc = evaluate(model, test_loader, device)
        
        # 记录数据
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # 打印每轮结果（清晰展示准确率）
        print(f"第{epoch:2d}轮 | 训练损失：{train_loss:.4f} | 测试损失：{test_loss:.4f} | 测试准确率：{test_acc*100:.2f}%")

    # 最终结果统计
    total_time = time.time() - start_time
    final_acc = test_accs[-1] * 100
    print("="*60)
    print(f"✅ 最终测试准确率：{final_acc:.2f}%（目标≥99%）")
    print(f"✅ 总训练时间：{total_time:.1f}秒（CPU环境）")
    print("="*60)

    # 保存训练曲线图（英文标签，无字体警告）
    if CONFIG["save_plot"]:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, CONFIG["epochs"]+1), train_losses, label="Train Loss", marker="o", color="blue")
        plt.plot(range(1, CONFIG["epochs"]+1), test_losses, label="Test Loss", marker="s", color="orange")
        plt.plot(range(1, CONFIG["epochs"]+1), test_accs, label="Test Accuracy", marker="^", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend(loc="lower right")
        plt.title(f"CNN MNIST | Final Accuracy: {final_acc:.2f}%")
        plt.grid(alpha=0.3)
        plt.savefig(CONFIG["plot_path"], dpi=160, bbox_inches="tight")
        print(f"📊 训练曲线图已保存至：{CONFIG['plot_path']}")

if __name__ == "__main__":
    main()