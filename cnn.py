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
    "model": "mlp",

    # 训练相关参数（可以改，用于观察收敛与精度变化）
    "epochs": 25,
    "batch_size": 128,
    "lr": 1e-3,             # 建议对比：1e-2 / 1e-3 / 1e-4
    "optimizer": "adam",    # "adam" 或 "sgd"

    # 输出
    "save_plot": True,
    "plot_path": "mlp.png",
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
# 模型定义：BP(MLP)
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
        # 目标：通过修改隐藏层的“层数/每层神经元数”，观察准确率变化
        #
        # 推荐你至少测试 3 组：
        # ① 1 个隐藏层（简单）：
        #    784 -> 128 -> 10
        #
        # ② 2 个隐藏层（推荐起步）：
        #    784 -> 256 -> 128 -> 10
        #
        # ③ 3 个隐藏层（更大、更慢）：
        #    784 -> 512 -> 256 -> 128 -> 10
        #
        # 提示：
        # - 隐藏层越大：表达能力更强，但训练更慢、参数更多
        # - 层数太深：可能提升有限，且更容易过拟合（MNIST相对简单）
        #
        # 你只需要改下面这些 Linear 的输入/输出维度即可。
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

        self.fc1 = nn.Linear(28 * 28, 512)   # 改这里：例如 128 / 256 / 512
        self.fc2 = nn.Linear(512, 256)       # 改这里：例如 64 / 128 / 256
        # 如需增加第三个隐藏层，可新增 fc3，并把最后输出层改名
        self.fc3=nn.Linear(256,128)
        self.out = nn.Linear(128, 10)        # 最后一层输出固定 10 类（0~9）
        # 激活函数（通常用 ReLU）
        self.drop=nn.Dropout(0.3)
        self.relu = nn.ReLU()


    def forward(self, x):
        # x: [B, 1, 28, 28]
        # MLP 必须 Flatten： [B, 784]
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.drop(x)  # 若启用 Dropout
        x = self.relu(self.fc2(x))
        x = self.drop(x)
        x = self.relu(self.fc3(x))
        x = self.drop(x)
        x = self.out(x)
        return x


# =========================
# 模型定义：CNN
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
        # 目标：通过修改卷积通道数/全连接层大小，观察准确率变化
        #
        # 推荐你至少测试 3 组通道数：
        # ① 小模型（更快，精度略低）：
        #    conv1: 1 -> 8
        #    conv2: 8 -> 16
        #
        # ② 中等模型（推荐起步）：
        #    conv1: 1 -> 16
        #    conv2: 16 -> 32
        #
        # ③ 大模型（更慢，精度更高）：
        #    conv1: 1 -> 32
        #    conv2: 32 -> 64
        #
        # 提示：
        # - 通道数越大：特征表达能力越强（通常精度↑），但参数/耗时↑
        #
        # 注意：本网络有两次 2x2 MaxPool：
        # - 图片 28x28 -> 14x14 -> 7x7
        # 所以第二层卷积输出的特征图尺寸为 7x7
        # 全连接层输入维度要写成： (conv2_out_channels * 7 * 7)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

        c1_out = 32   # 改这里：8 / 16 / 32
        c2_out = 64   # 改这里：16 / 32 / 64

        self.conv1 = nn.Conv2d(1, c1_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1_out, c2_out, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)  # 2x2 池化，尺寸减半

        # 全连接层：输入是 c2_out * 7 * 7
        self.fc1 = nn.Linear(c2_out * 7 * 7, 256)  # 可以改 128 -> 256 试试
        self.fc2 = nn.Linear(256, 128)
        self.fc3=nn.Linear(128,10)

    def forward(self, x):
        # x: [B, 1, 28, 28]  (CNN 不需要 Flatten 输入)
        x = self.pool(self.relu(self.conv1(x)))  # -> [B, c1_out, 14, 14]
        x = self.pool(self.relu(self.conv2(x)))  # -> [B, c2_out, 7, 7]
        x = x.view(x.size(0), -1)                # -> [B, c2_out*7*7]
        x = self.relu(self.fc1(x))
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

    # -------------------------
    # 数据获取（自动下载 MNIST）
    # -------------------------
    # download=True：若本地没有 MNIST，会自动联网下载并解压到 ./data/
    transform = transforms.Compose([transforms.ToTensor()])

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
        plt.figure()
        plt.plot(range(1, CONFIG["epochs"] + 1), train_losses, label="train_loss")
        plt.plot(range(1, CONFIG["epochs"] + 1), test_losses, label="test_loss")
        plt.plot(range(1, CONFIG["epochs"] + 1), test_accs, label="test_acc")
        plt.xlabel("epoch")
        plt.legend()
        plt.title(f"{CONFIG['model']} | lr={CONFIG['lr']}")
        plt.savefig(CONFIG["plot_path"], dpi=160, bbox_inches="tight")
        print(f"Saved plot to: {CONFIG['plot_path']}")

if __name__ == "__main__":
    main()
