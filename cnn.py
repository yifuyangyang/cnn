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
    "seed": 42,
    "model": "cnn",  # 选择CNN
    "epochs": 15,    # 训练轮数15
    "batch_size": 128,  # 批大小128
    "lr": 1e-3,         # 学习率1e-3
    "optimizer": "adam",
    "save_plot": True,
    "plot_path": "cnn_results.png",  # 单独保存CNN曲线图
}

# =========================
# 工具函数（无修改）
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
# 模型定义：BP(MLP)（保留，不影响CNN运行）
# =========================
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.out(x)
        return x

# =========================
# 模型定义：CNN
# =========================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 修改后：卷积通道32→64，全连接层256神经元
        c1_out = 32   # 第一层卷积32通道
        c2_out = 64   # 第二层卷积64通道
        self.conv1 = nn.Conv2d(1, c1_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1_out, c2_out, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(c2_out * 7 * 7, 256)  # 全连接层256神经元
        self.fc2 = nn.Linear(256, 10)              # 输出层10类

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 28→14
        x = self.pool(self.relu(self.conv2(x)))  # 14→7
        x = x.view(x.size(0), -1)                # Flatten为64*7*7=3136
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
    # 数据加载
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)
    # 建模
    model = build_model(CONFIG["model"]).to(device)
    optimizer = build_optimizer(CONFIG, model)
    print("=================================================")
    print(f"Device: {device}")
    print(f"Model:  {CONFIG['model']}")
    print(f"Params: {count_params(model):,}")
    print(f"Epochs: {CONFIG['epochs']} | Batch: {CONFIG['batch_size']} | LR: {CONFIG['lr']} | Opt: {CONFIG['optimizer']}")
    print("=================================================")
    # 训练记录
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
    # 保存曲线图
    if CONFIG["save_plot"]:
        plt.figure()
        plt.plot(range(1, CONFIG["epochs"] + 1), train_losses, label="train_loss")
        plt.plot(range(1, CONFIG["epochs"] + 1), test_losses, label="test_loss")
        plt.plot(range(1, CONFIG["epochs"] + 1), test_accs, label="test_acc")
        plt.xlabel("epoch")
        plt.legend()
        plt.title(f"CNN | lr={CONFIG['lr']} | batch={CONFIG['batch_size']}")
        plt.savefig(CONFIG["plot_path"], dpi=160, bbox_inches="tight")
        print(f"Saved plot to: {CONFIG['plot_path']}")

if __name__ == "__main__":
    main()