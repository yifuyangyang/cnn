
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
# CONFIG（优化后的训练参数）
# =========================
CONFIG = {
    # 固定随机种子：方便对比（同参数下结果更稳定）
    "seed": 42,

    # 选择要训练的模型： "mlp" 或 "cnn"
    "model": "mlp",  # 切换为"cnn"测试CNN效果

    # 训练相关参数（优化后）
    "epochs": 20,    # 增加训练轮数，让模型充分收敛
    "batch_size": 128,  # 增大batch_size，提升训练稳定性
    "lr": 1e-3,             # MLP/CNN均使用1e-3（adam优化器效果更好）
    "optimizer": "adam",    # adam优化器收敛更快，优于sgd

    # 输出
    "save_plot": True,
    "plot_path": "results.png",
}


# =========================
# 工具函数（无修改）
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
# 模型定义：BP(MLP)（优化后）
# =========================
class MLP(nn.Module):
    """
    BP 神经网络（多层感知机，MLP）
    - 输入：MNIST 图像 [B, 1, 28, 28]
    - 先 Flatten 成向量 [B, 784]
    - 再走全连接层做分类
    优化点：
    1. 增加隐藏层神经元数量+层数
    2. 加入Dropout抑制过拟合
    3. 加入BatchNorm1d加速收敛并稳定训练
    """

    def __init__(self):
        super().__init__()

        # ★★★★★ 优化后的MLP结构 ★★★★★
        # 784 -> 512 -> 256 -> 128 -> 10（3层隐藏层，提升容量）
        self.fc1 = nn.Linear(28 * 28, 512)
        self.bn1 = nn.BatchNorm1d(512)  # 批归一化
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.out = nn.Linear(128, 10)

        # 激活函数+正则化
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # Dropout概率0.2（抑制过拟合）

    def forward(self, x):
        # x: [B, 1, 28, 28] -> [B, 784]
        x = x.view(x.size(0), -1)

        # 全连接+批归一化+激活+Dropout（顺序：Linear -> BN -> ReLU -> Dropout）
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout(self.relu(self.bn3(self.fc3(x))))
        x = self.out(x)

        return x


# =========================
# 模型定义：CNN（优化后）
# =========================
class SimpleCNN(nn.Module):
    """
    卷积神经网络（CNN）
    优化点：
    1. 增加卷积通道数（提升特征提取能力）
    2. 加入BatchNorm2d加速收敛并稳定训练
    3. 增加全连接层神经元数量
    4. 加入Dropout抑制过拟合
    """

    def __init__(self):
        super().__init__()

        # ★★★★★ 优化后的CNN通道数 ★★★★★
        c1_out = 32   # 从16→32（提升卷积特征提取能力）
        c2_out = 64   # 从32→64
        c3_out = 128  # 新增第三层卷积（进一步提升特征提取）

        # 卷积层（加入BatchNorm2d）
        self.conv1 = nn.Conv2d(1, c1_out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(c1_out)
        self.conv2 = nn.Conv2d(c1_out, c2_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c2_out)
        self.conv3 = nn.Conv2d(c2_out, c3_out, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(c3_out)

        # 池化+激活+正则化
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)  # 2x2池化，尺寸减半
        self.dropout = nn.Dropout(0.2)  # Dropout抑制过拟合

        # 全连接层（计算特征图尺寸：28→14→7→3（三次池化后为3x3））
        # 注意：三次卷积+池化后，尺寸从28→14→7→3（最后一次池化是3x3向下取整）
        self.fc1 = nn.Linear(c3_out * 3 * 3, 256)  # 增加全连接层神经元数量
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # x: [B, 1, 28, 28]
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # -> [B, 32, 14, 14]
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # -> [B, 64, 7, 7]
        x = self.pool(self.relu(self.bn3(self.conv3(x))))  # -> [B, 128, 3, 3]
        x = x.view(x.size(0), -1)                          # -> [B, 128*3*3=1152]
        x = self.dropout(self.relu(self.bn4(self.fc1(x)))) # 全连接+批归一化+激活+Dropout
        x = self.fc2(x)

        return x


def build_model(model_name: str) -> nn.Module:
    if model_name == "mlp":
        return MLP()
    elif model_name == "cnn":
        return SimpleCNN()
    else:
        raise ValueError("CONFIG['model'] must be 'mlp' or 'cnn'")


# =========================
# 主程序（优化数据加载+数据增强）
# =========================
def main():
    set_seed(CONFIG["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # 数据获取（加入数据增强）
    # -------------------------
    # 训练集：加入随机平移、旋转（数据增强，提升泛化能力）
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),  # 随机旋转5度+平移10%
        transforms.ToTensor(),
    ])
    # 测试集：仅归一化（不增强，保持测试真实性）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # download=True：若本地没有MNIST，会自动联网下载并解压到 ./data/
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=test_transform)

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

    print("=================================================")
    print(f"Device: {device}")
    print(f"Model:  {CONFIG['model']}")
    print(f"Params: {count_params(model):,}")
    print(f"Epochs: {CONFIG['epochs']} | Batch: {CONFIG['batch_size']} | LR: {CONFIG['lr']} | Opt: {CONFIG['optimizer']}")
    print("=================================================")

    # 记录曲线
    train_losses, test_losses, test_accs = [], [], []

    start = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, device)

        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        test_accs.append(te_acc)

        print(f"Epoch {epoch:02d}/{CONFIG['epochs']} | "
              f"train_loss={tr_loss:.4f} | test_loss={te_loss:.4f} | test_acc={te_acc*100:.2f}%")

    elapsed = time.time() - start

    print("=================================================")
    print(f"Final Test Accuracy: {test_accs[-1]*100:.2f}%")
    print(f"Training Time: {elapsed:.1f}s")
    print("=================================================")

    # -------------------------
    # 保存曲线图：loss + acc
    # -------------------------
    if CONFIG["save_plot"]:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, CONFIG["epochs"] + 1), train_losses, label="train_loss", marker="o")
        plt.plot(range(1, CONFIG["epochs"] + 1), test_losses, label="test_loss", marker="s")
        plt.plot(range(1, CONFIG["epochs"] + 1), test_accs, label="test_acc", marker="^")
        plt.xlabel("epoch")
        plt.ylabel("value")
        plt.legend()
        plt.title(f"{CONFIG['model']} | lr={CONFIG['lr']} | batch_size={CONFIG['batch_size']}")
        plt.grid(True, alpha=0.3)
        plt.savefig(CONFIG["plot_path"], dpi=160, bbox_inches="tight")
        print(f"Saved plot to: {CONFIG['plot_path']}")

if __name__ == "__main__":
    main()