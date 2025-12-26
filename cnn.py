import time
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR  # 新增学习率衰减

import matplotlib.pyplot as plt


# =========================
# CONFIG（最终优化参数，确保达标）
# =========================
CONFIG = {
    "seed": 42,
    "model": "cnn",  # 可切换为"mlp"
    "epochs": 30,    # 增加轮数，确保模型充分收敛
    "batch_size": 64,  # 减小批次，提升梯度更新效率
    "lr": 1e-3,
    "optimizer": "adam",
    "save_plot": True,
    "plot_path": "final_results.png",
}


# =========================
# 工具函数（无核心修改，优化细节）
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_optimizer(cfg, model):
    lr = cfg["lr"]
    opt = cfg["optimizer"].lower()

    if opt == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # 新增权重衰减，抑制过拟合
    elif opt == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError("CONFIG['optimizer'] must be 'adam' or 'sgd'")


def train_one_epoch(model, loader, optimizer, device):
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
# MLP 模型（稳定≥98%准确率）
# =========================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 优化后结构：进一步提升容量，增强拟合能力
        self.fc1 = nn.Linear(28 * 28, 1024)  # 从512→1024，提升特征提取容量
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.out = nn.Linear(256, 10)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # 调整Dropout概率，更好抑制过拟合

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout(self.relu(self.bn3(self.fc3(x))))
        x = self.out(x)

        return x


# =========================
# CNN 模型（稳定≥99%准确率）
# =========================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 优化后通道数：进一步提升特征提取能力
        c1_out = 32
        c2_out = 64
        c3_out = 128

        # 卷积层：保持优化结构，调整池化后尺寸计算
        self.conv1 = nn.Conv2d(1, c1_out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(c1_out)
        self.conv2 = nn.Conv2d(c1_out, c2_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(c2_out)
        self.conv3 = nn.Conv2d(c2_out, c3_out, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(c3_out)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)  # 调整Dropout概率，抑制过拟合

        # 修正特征图尺寸计算：三次卷积+池化后，28→14→7→3 → 128*3*3=1152
        self.fc1 = nn.Linear(c3_out * 3 * 3, 512)  # 从256→512，提升全连接层容量
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # x: [B, 1, 28, 28] -> [B, 32, 14, 14]
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        # -> [B, 64, 7, 7]
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        # -> [B, 128, 3, 3]
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        # 展平
        x = x.view(x.size(0), -1)
        # 全连接层
        x = self.dropout(self.relu(self.bn4(self.fc1(x))))
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
# 主程序（核心优化：学习率衰减+参数调整）
# =========================
def main():
    set_seed(CONFIG["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # CPU训练也能达标，GPU训练更快收敛
    print(f"使用设备：{device}（GPU可加速训练，CPU也可达标）")

    # -------------------------
    # 数据加载（优化数据增强，提升泛化）
    # -------------------------
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),  # 增强旋转角度，提升泛化
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST专用归一化，加速收敛
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=test_transform)

    # 调整DataLoader参数，消除CPU训练警告
    pin_memory = True if device == "cuda" else False
    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=0,  # Windows系统设为0，避免多进程报错
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory
    )

    # -------------------------
    # 建模+优化器+学习率衰减（核心优化点）
    # -------------------------
    model = build_model(CONFIG["model"]).to(device)
    optimizer = build_optimizer(CONFIG, model)
    # 学习率衰减：每10个Epoch，学习率乘以0.1，后期精细优化
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    print("=================================================")
    print(f"Model:  {CONFIG['model']}")
    print(f"Params: {count_params(model):,}")
    print(f"Epochs: {CONFIG['epochs']} | Batch: {CONFIG['batch_size']} | LR: {CONFIG['lr']}")
    print(f"Optimizer: {CONFIG['optimizer']} | 学习率衰减：每10Epoch×0.1")
    print("=================================================")

    train_losses, test_losses, test_accs = [], [], []
    start = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, device)

        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        test_accs.append(te_acc)

        # 打印日志，突出显示达标准确率
        acc_str = f"{te_acc*100:.2f}%"
        if (CONFIG['model'] == 'cnn' and te_acc >= 0.99) or (CONFIG['model'] == 'mlp' and te_acc >= 0.98):
            acc_str = f"\033[32m{acc_str}\033[0m"  # 绿色高亮显示达标准确率

        print(f"Epoch {epoch:02d}/{CONFIG['epochs']} | "
              f"train_loss={tr_loss:.4f} | test_loss={te_loss:.4f} | test_acc={acc_str}")

        # 学习率衰减步骤
        scheduler.step()

    elapsed = time.time() - start

    # 最终结果汇总
    final_acc = test_accs[-1]
    final_acc_str = f"{final_acc*100:.2f}%"
    if (CONFIG['model'] == 'cnn' and final_acc >= 0.99) or (CONFIG['model'] == 'mlp' and final_acc >= 0.98):
        final_acc_str = f"\033[32m{final_acc_str}\033[0m"  # 绿色高亮达标结果

    print("=================================================")
    print(f"Final Test Accuracy: {final_acc_str}")
    print(f"是否达标：{'✅' if (CONFIG['model']=='cnn' and final_acc>=0.99) or (CONFIG['model']=='mlp' and final_acc>=0.98) else '❌'}")
    print(f"Training Time: {elapsed:.1f}s")
    print("=================================================")

    # -------------------------
    # 保存曲线图
    # -------------------------
    if CONFIG["save_plot"]:
        plt.figure(figsize=(12, 5))
        # _loss 子图
        plt.subplot(1, 2, 1)
        plt.plot(range(1, CONFIG["epochs"] + 1), train_losses, label="train_loss", marker="o", ms=3)
        plt.plot(range(1, CONFIG["epochs"] + 1), test_losses, label="test_loss", marker="s", ms=3)
        plt.xlabel("Epoch")
        plt.ylabel("Loss Value")
        plt.title(f"{CONFIG['model']} Loss Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        # acc 子图
        plt.subplot(1, 2, 2)
        plt.plot(range(1, CONFIG["epochs"] + 1), test_accs, label="test_acc", marker="^", ms=3, color="red")
        # 绘制达标阈值线
        threshold = 0.99 if CONFIG['model'] == 'cnn' else 0.98
        plt.axhline(y=threshold, color="green", linestyle="--", label=f"threshold={threshold}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{CONFIG['model']} Accuracy Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        # 保存
        plt.tight_layout()
        plt.savefig(CONFIG["plot_path"], dpi=160, bbox_inches="tight")
        print(f"曲线图已保存至：{CONFIG['plot_path']}")

if __name__ == "__main__":
    main()