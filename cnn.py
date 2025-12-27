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
# CONFIG（最优参数配置）
# =========================
CONFIG = {
    "seed": 42,
    "model": "cnn",        # 切换 "mlp" / "cnn"
    "epochs": 15,          # 训练轮数（MLP=10, CNN=15）
    "batch_size": 128,     # 批大小
    "lr": 1e-3,            # 学习率
    "optimizer": "adam",   # 优化器
    "save_plot": True,
    "plot_path": "results.png",
}


# =========================
# 工具函数
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
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif opt == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
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
# 模型定义：MLP（最优结构：98%+）
# =========================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 最优结构：784 -> 512 -> 256 -> 128 -> 10（3层隐藏层）
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # 防止过拟合

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.out(x)
        return x


# =========================
# 模型定义：CNN（最优结构：99%+）
# =========================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 最优结构：conv1(1→32) → conv2(32→64) → fc(64*7*7→128) → 10
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # [B,32,14,14]
        x = self.pool(self.relu(self.conv2(x)))  # [B,64,7,7]
        x = x.view(x.size(0), -1)                # [B,64*7*7]
        x = self.dropout(self.relu(self.fc1(x)))
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
# 主程序
# =========================
def main():
    set_seed(CONFIG["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 数据加载
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    # 模型初始化
    model = build_model(CONFIG["model"]).to(device)
    optimizer = build_optimizer(CONFIG, model)

    print("="*50)
    print(f"Device: {device}")
    print(f"Model:  {CONFIG['model']}")
    print(f"Params: {count_params(model):,}")
    print(f"Epochs: {CONFIG['epochs']} | Batch: {CONFIG['batch_size']} | LR: {CONFIG['lr']}")
    print("="*50)

    # 训练过程
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

    # 结果输出
    elapsed = time.time() - start
    print("="*50)
    print(f"Final Test Accuracy: {test_accs[-1]*100:.2f}%")
    print(f"Training Time: {elapsed:.1f}s")
    print("="*50)

    # 保存曲线图
    if CONFIG["save_plot"]:
        plt.figure(figsize=(10, 4))
        plt.subplot(1,2,1)
        plt.plot(range(1, CONFIG["epochs"] + 1), train_losses, label="train_loss")
        plt.plot(range(1, CONFIG["epochs"] + 1), test_losses, label="test_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"{CONFIG['model']} Loss Curve")
        
        plt.subplot(1,2,2)
        plt.plot(range(1, CONFIG["epochs"] + 1), test_accs, label="test_acc", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim(0.95, 1.0)
        plt.legend()
        plt.title(f"{CONFIG['model']} Accuracy Curve")
        
        plt.tight_layout()
        plt.savefig(CONFIG["plot_path"], dpi=160)
        print(f"Saved plot to: {CONFIG['plot_path']}")

if __name__ == "__main__":
    main()