# bp_vs_cnn_mnist.py
# ============================================================
# 实验：BP(MLP) vs CNN 在 MNIST 上的对比
# - MLP 需要 Flatten，会丢失空间结构
# - CNN 利用卷积/共享/池化，更适合图像
#
# 运行方式：
#   python bp_vs_cnn_mnist.py
#
# 必须改的地方：
#   1) MLP 的隐藏层规模/层数（在 MLP.__init__ 的 ★★★★★ 区域）
#   2) CNN 的卷积通道数/全连接层（在 SimpleCNN.__init__ 的 ★★★★★ 区域）
# 可选改的地方：
#   学习率、epoch、batch_size（在 CONFIG 区域）
# ============================================================

import time
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


# =========================
# CONFIG（学生可改：训练参数）
# =========================
CONFIG = {
    # 固定随机种子：方便对比（同参数下结果更稳定）
    "seed": 42,

    # 选择要训练的模型： "mlp" 或 "cnn"
    "model": "cnn",  # 手动改为 mlp 或 cnn

    # 训练相关参数（可以改，用于观察收敛与精度变化）
    "epochs": 15,      # 增加训练轮数以获得更好性能
    "batch_size": 128,  # 增大批大小以稳定训练
    "lr": 0.001,       # 学习率
    "optimizer": "adam",    # "adam" 或 "sgd"

    # 输出
    "save_plot": True,
    "plot_path": "results_cnn.png",
}


# =========================
# 工具函数
# =========================
def set_seed(seed: int):
    """固定随机种子：让结果更可复现（便于公平对比）"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_params(model: nn.Module) -> int:
    """统计可训练参数量（衡量模型复杂度）"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_optimizer(cfg, model):
    """根据配置创建优化器"""
    lr = cfg["lr"]
    opt = cfg["optimizer"].lower()

    if opt == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif opt == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError("CONFIG['optimizer'] must be 'adam' or 'sgd'")


def train_one_epoch(model, loader, optimizer, device):
    """训练一个 epoch，返回平均 train loss"""
    model.train()
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total += x.size(0)

    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, device):
    """在测试集评估，返回 test loss 和 test accuracy"""
    model.eval()
    ce = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = ce(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


# =========================
# 模型定义：BP(MLP) - 优化后的版本
# =========================
class MLP(nn.Module):
    """
    BP 神经网络（多层感知机，MLP）
    - 输入：MNIST 图像 [B, 1, 28, 28]
    - 先 Flatten 成向量 [B, 784]
    - 再走全连接层做分类
    """

    def __init__(self):
        super().__init__()

        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★★ 修改区（MLP 网络结构参数）★★★★★
        # 使用3个隐藏层以达到98%+准确率
        # 结构：784 -> 512 -> 256 -> 128 -> 10
        # 添加Dropout和BatchNorm以防止过拟合
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

        self.fc1 = nn.Linear(28 * 28, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.out = nn.Linear(128, 10)

        # 激活函数和Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: [B, 1, 28, 28]
        x = x.view(x.size(0), -1)  # Flatten

        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.out(x)
        return x


# =========================
# 模型定义：CNN - 优化后的版本
# =========================
class SimpleCNN(nn.Module):
    """
    卷积神经网络（CNN）
    - 输入保持图像结构：[B, 1, 28, 28]
    - 通过卷积提取局部特征（边缘、拐角、笔画组合）
    """

    def __init__(self):
        super().__init__()

        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★★ 修改区（CNN 网络结构参数）★★★★★
        # 使用更大的卷积核和更多的通道数以达到99%+准确率
        # 添加BatchNorm和Dropout以提升性能
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

        c1_out = 32   # 第一层卷积输出通道
        c2_out = 64   # 第二层卷积输出通道
        c3_out = 128  # 第三层卷积输出通道

        # 卷积层1
        self.conv1 = nn.Conv2d(1, c1_out, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(c1_out)
        
        # 卷积层2
        self.conv2 = nn.Conv2d(c1_out, c2_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c2_out)
        
        # 卷积层3
        self.conv3 = nn.Conv2d(c2_out, c3_out, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(c3_out)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)  # 2x2 池化，尺寸减半
        self.dropout = nn.Dropout(0.25)

        # 全连接层
        self.fc1 = nn.Linear(c3_out * 3 * 3, 256)  # 经过3次池化：28->14->7->3
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 10)

    def forward(self, x):
        # x: [B, 1, 28, 28]
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # -> [B, 32, 14, 14]
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # -> [B, 64, 7, 7]
        x = self.pool(self.relu(self.bn3(self.conv3(x))))  # -> [B, 128, 3, 3]
        x = x.view(x.size(0), -1)                         # -> [B, 128*3*3]
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.out(x)
        return x


def build_model(model_name: str) -> nn.Module:
    if model_name == "mlp":
        return MLP()
    elif model_name == "cnn":
        return SimpleCNN()
    else:
        raise ValueError("CONFIG['model'] must be 'mlp' or 'cnn'")


# =========================
# 主程序
# =========================
def main():
    set_seed(CONFIG["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # 数据获取（自动下载 MNIST）
    # -------------------------
    # 添加数据增强以提升性能
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
    ])

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # -------------------------
    # 建模与优化器
    # -------------------------
    model = build_model(CONFIG["model"]).to(device)
    optimizer = build_optimizer(CONFIG, model)
    
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model:  {CONFIG['model']}")
    print(f"Params: {count_params(model):,}")
    print(f"Epochs: {CONFIG['epochs']} | Batch: {CONFIG['batch_size']} | LR: {CONFIG['lr']} | Opt: {CONFIG['optimizer']}")
    print("=" * 60)

    # 记录曲线
    train_losses, test_losses, test_accs = [], [], []

    start = time.time()

    best_acc = 0.0
    patience = 5
    patience_counter = 0

    for epoch in range(1, CONFIG["epochs"] + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, device)
        
        scheduler.step()

        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        test_accs.append(te_acc)

        print(f"Epoch {epoch:02d}/{CONFIG['epochs']} | "
              f"train_loss={tr_loss:.4f} | test_loss={te_loss:.4f} | test_acc={te_acc*100:.2f}%")
        
        # 早停机制
        if te_acc > best_acc:
            best_acc = te_acc
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), f"best_{CONFIG['model']}.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch > 10:
                print(f"Early stopping at epoch {epoch}")
                break

    elapsed = time.time() - start

    print("=" * 60)
    print(f"Best Test Accuracy: {best_acc*100:.2f}%")
    print(f"Final Test Accuracy: {test_accs[-1]*100:.2f}%")
    print(f"Training Time: {elapsed:.1f}s")
    print("=" * 60)

    # -------------------------
    # 保存曲线图：loss + acc
    # -------------------------
    if CONFIG["save_plot"]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax1.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", linewidth=2)
        ax1.plot(range(1, len(test_losses) + 1), test_losses, label="Test Loss", linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Test Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2.plot(range(1, len(test_accs) + 1), test_accs, label="Test Accuracy", 
                color='green', linewidth=2)
        
        # 添加目标线
        target_acc = 0.98 if CONFIG['model'] == 'mlp' else 0.99
        ax2.axhline(y=target_acc, color='r', linestyle='--', 
                   label=f"Target: {target_acc*100}%", alpha=0.5)
        
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title(f"{CONFIG['model'].upper()} Test Accuracy")
        ax2.set_ylim([0.9, 1.0])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f"{CONFIG['model'].upper()} | LR={CONFIG['lr']} | Batch={CONFIG['batch_size']}", fontsize=14)
        plt.tight_layout()
        plt.savefig(CONFIG["plot_path"], dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {CONFIG['plot_path']}")

if __name__ == "__main__":
    main()