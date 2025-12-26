# bp_vs_cnn_mnist.py
# ============================================================
# 实验：BP(MLP) vs CNN 在 MNIST 上的对比
# 
# 实验目标：
# 1. 网络结构参数对模型精度的影响
#    - MLP: 隐藏层大小、层数、Dropout等
#    - CNN: 卷积通道数、卷积层数、全连接层大小等
# 2. 通过手动调参提升模型性能的方法
#    - 学习率调整、优化器选择、批次大小、训练轮数
# 3. CNN 相比 MLP 在图像分类任务中的优势
#    - 局部感知、参数共享、平移不变性、层次特征提取
#
# 实验任务：
# 1. 使用 MLP: 通过调参使测试集准确率达到 98% 及以上
# 2. 使用 CNN: 通过调参使测试集准确率达到 99% 及以上
#
# 调参策略：
# ★ MLP达到98%的配置建议：
#    - 结构：[784, 512, 256, 128, 10]
#    - 训练：epochs=20, lr=0.001, batch_size=128
#    - 优化器：Adam
#
# ★ CNN达到99%的配置建议：
#    - 结构：conv1: 1->32, conv2: 32->64, fc: 64*7*7->128->10
#    - 训练：epochs=15, lr=0.001, batch_size=64
#    - 优化器：Adam
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

    # ★★★★★ 选择要训练的模型： "mlp" 或 "cnn" ★★★★★
    "model": "mlp",  # 先运行MLP，目标98%；然后改为"cnn"，目标99%

    # ★★★★★ 训练相关参数（可以改，用于观察收敛与精度变化）★★★★★
    "epochs": 15,           # MLP建议15-20，CNN建议10-15
    "batch_size": 128,      # 建议：32, 64, 128, 256
    "lr": 1e-3,             # 建议对比：1e-2 / 1e-3 / 5e-4 / 1e-4
    "optimizer": "adam",    # "adam" 或 "sgd"（Adam通常更好）

    # ★★★★★ 数据增强（可选，对CNN特别有效）★★★★★
    "use_data_augmentation": True,  # 启用数据增强提高泛化能力

    # 输出
    "save_plot": True,
    "plot_path": "results.png",
}


# =========================
# 工具函数（保持不变）
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
# 模型定义：BP(MLP) - 目标：98% 准确率
# =========================
class MLP(nn.Module):
    """
    BP 神经网络（多层感知机，MLP）
    - 输入：MNIST 图像 [B, 1, 28, 28]
    - 先 Flatten 成向量 [B, 784]
    - 再走全连接层做分类
    
    ★ 网络结构参数对精度的影响 ★
    1. 隐藏层大小：增加神经元数量可以提高模型表达能力，但可能过拟合
    2. 隐藏层数量：增加层数可以学习更复杂的特征，但训练更困难
    3. Dropout：防止过拟合，提高泛化能力
    
    ★ MLP达到98%准确率的调参方法 ★
    1. 使用3-4个隐藏层
    2. 每层神经元数：784 -> 512 -> 256 -> 128 -> 10
    3. 添加Dropout层（0.3-0.5）
    4. 训练15-20个epoch，学习率0.001
    """

    def __init__(self):
        super().__init__()

        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★★ 修改区（MLP 网络结构参数）★★★★★
        # 
        # 要达到98%准确率，建议配置：
        # 方案1（3层隐藏层，推荐）：
        #   fc1: 784 -> 512
        #   fc2: 512 -> 256
        #   fc3: 256 -> 128
        #   out: 128 -> 10
        #
        # 方案2（4层隐藏层，更强表达能力）：
        #   fc1: 784 -> 784  (保持维度)
        #   fc2: 784 -> 512
        #   fc3: 512 -> 256
        #   fc4: 256 -> 128
        #   out: 128 -> 10
        #
        # ★ 注意：层数越深，需要的训练时间越长 ★
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

        # 方案1：3层隐藏层（推荐起始配置）
        self.fc1 = nn.Linear(28 * 28, 512)   # 第一隐藏层
        self.fc2 = nn.Linear(512, 256)       # 第二隐藏层
        self.fc3 = nn.Linear(256, 128)       # 第三隐藏层
        self.out = nn.Linear(128, 10)        # 输出层

        # 激活函数
        self.relu = nn.ReLU()
        
        # ★ 可选：添加Dropout防止过拟合 ★
        self.dropout = nn.Dropout(p=0.3)     # Dropout概率0.3


    def forward(self, x):
        # x: [B, 1, 28, 28]
        # MLP 必须 Flatten： [B, 784]
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # 应用Dropout
        
        x = self.relu(self.fc2(x))
        x = self.dropout(x)  # 应用Dropout
        
        x = self.relu(self.fc3(x))
        x = self.dropout(x)  # 应用Dropout
        
        x = self.out(x)
        return x


# =========================
# 模型定义：CNN - 目标：99% 准确率
# =========================
class SimpleCNN(nn.Module):
    """
    卷积神经网络（CNN）
    - 输入保持图像结构：[B, 1, 28, 28]
    - 通过卷积提取局部特征（边缘、拐角、笔画组合）
    
    ★ CNN相比MLP在图像分类中的优势 ★
    1. 局部感知：每个神经元只感受图像的局部区域
    2. 参数共享：同一个卷积核在整个图像上滑动，大大减少参数量
    3. 平移不变性：物体在图像中位置变化不影响识别
    4. 层次特征提取：浅层提取边缘，深层提取复杂模式
    
    ★ CNN达到99%准确率的调参方法 ★
    1. 增加卷积通道数：32->64
    2. 添加BatchNorm层加速收敛
    3. 使用数据增强
    4. 训练10-15个epoch，学习率0.001
    """

    def __init__(self):
        super().__init__()

        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # ★★★★★ 修改区（CNN 网络结构参数）★★★★★
        # 
        # 要达到99%准确率，建议配置：
        # 方案1（中等大小，推荐）：
        #   conv1: 1 -> 32
        #   conv2: 32 -> 64
        #   fc1: 64*7*7 -> 128
        #   out: 128 -> 10
        #
        # 方案2（更大模型，更高精度）：
        #   conv1: 1 -> 64
        #   conv2: 64 -> 128
        #   fc1: 128*7*7 -> 256
        #   out: 256 -> 10
        #
        # ★ 注意：本网络有两次 2x2 MaxPool：
        # - 图片 28x28 -> 14x14 -> 7x7
        # 所以第二层卷积输出的特征图尺寸为 7x7
        # 全连接层输入维度要写成： (conv2_out_channels * 7 * 7)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

        # 方案1：中等大小CNN（推荐起始配置）
        c1_out = 32   # 第一层卷积输出通道数
        c2_out = 64   # 第二层卷积输出通道数
        fc1_size = 128  # 全连接层大小

        # 卷积层
        self.conv1 = nn.Conv2d(1, c1_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1_out, c2_out, kernel_size=3, padding=1)
        
        # ★ 可选：添加BatchNorm层加速收敛 ★
        self.bn1 = nn.BatchNorm2d(c1_out)
        self.bn2 = nn.BatchNorm2d(c2_out)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)  # 2x2 池化，尺寸减半

        # 全连接层：输入是 c2_out * 7 * 7
        self.fc1 = nn.Linear(c2_out * 7 * 7, fc1_size)
        self.fc2 = nn.Linear(fc1_size, 10)
        
        # ★ 可选：添加Dropout防止过拟合 ★
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # x: [B, 1, 28, 28]  (CNN 不需要 Flatten 输入)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # -> [B, 32, 14, 14]
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # -> [B, 64, 7, 7]
        x = x.view(x.size(0), -1)                # -> [B, 64*7*7]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # 应用Dropout
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
    # 基础转换
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化
    ])
    
    # 数据增强（仅训练集）
    if CONFIG["use_data_augmentation"] and CONFIG["model"] == "cnn":
        train_transform = transforms.Compose([
            transforms.RandomRotation(5),  # 随机旋转5度
            transforms.RandomAffine(0, translate=(0.05, 0.05)),  # 随机平移
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        train_transform = base_transform
    
    # download=True：若本地没有 MNIST，会自动联网下载并解压到 ./data/
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=base_transform)

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
    
    # 学习率调度器（可选，有助于达到更高精度）
    try:
        # PyTorch新版本
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
    except TypeError:
        # PyTorch旧版本
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )

    print("=" * 60)
    print(f"实验: {CONFIG['model'].upper()} 在 MNIST 上的性能")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"模型: {CONFIG['model']}")
    print(f"参数量: {count_params(model):,}")
    print(f"训练轮数: {CONFIG['epochs']} | 批次大小: {CONFIG['batch_size']}")
    print(f"学习率: {CONFIG['lr']} | 优化器: {CONFIG['optimizer']}")
    print(f"数据增强: {CONFIG['use_data_augmentation']}")
    print("=" * 60)
    
    # 显示实验目标
    if CONFIG["model"] == "mlp":
        print("实验目标: 测试集准确率达到 98% 及以上")
        print("调参建议:")
        print("  1. 确保MLP有3-4个隐藏层")
        print("  2. 隐藏层大小至少: 512 -> 256 -> 128")
        print("  3. 训练15-20个epoch")
        print("  4. 使用Adam优化器，学习率0.001")
    else:
        print("实验目标: 测试集准确率达到 99% 及以上")
        print("调参建议:")
        print("  1. 卷积通道至少: 32 -> 64")
        print("  2. 启用数据增强")
        print("  3. 训练10-15个epoch")
        print("  4. 使用Adam优化器，学习率0.001")
    print("=" * 60)

    # 记录曲线
    train_losses, test_losses, test_accs = [], [], []
    best_acc = 0.0
    best_epoch = 0

    start = time.time()

    for epoch in range(1, CONFIG["epochs"] + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, device)
        
        # 更新学习率
        scheduler.step(te_acc)

        train_losses.append(tr_loss)
        test_losses.append(te_loss)
        test_accs.append(te_acc)
        
        # 记录最佳准确率
        if te_acc > best_acc:
            best_acc = te_acc
            best_epoch = epoch

        print(f"Epoch {epoch:02d}/{CONFIG['epochs']} | "
              f"train_loss={tr_loss:.4f} | test_loss={te_loss:.4f} | "
              f"test_acc={te_acc*100:.2f}% | best={best_acc*100:.2f}% @ epoch {best_epoch}")

    elapsed = time.time() - start

    print("=" * 60)
    print(f"最终测试准确率: {test_accs[-1]*100:.2f}%")
    print(f"最佳测试准确率: {best_acc*100:.2f}% (epoch {best_epoch})")
    print(f"训练时间: {elapsed:.1f}秒")
    
    # 判断是否达到目标
    target_acc = 0.98 if CONFIG["model"] == "mlp" else 0.99
    if best_acc >= target_acc:
        print(f"✅ 成功达到目标准确率 {target_acc*100:.0f}%!")
    else:
        print(f"❌ 未达到目标准确率 {target_acc*100:.0f}%，建议调整参数")
        print("调参建议:")
        if CONFIG["model"] == "mlp":
            print("  1. 增加隐藏层大小（如 784 -> 1024 -> 512 -> 256 -> 128）")
            print("  2. 增加训练轮数到20-25")
            print("  3. 降低学习率到5e-4")
            print("  4. 增加Dropout概率到0.5")
        else:
            print("  1. 增加卷积通道数（如 64 -> 128 -> 256）")
            print("  2. 添加更多卷积层（3层卷积）")
            print("  3. 启用数据增强")
            print("  4. 增加训练轮数到15-20")
    print("=" * 60)
    
    # 显示CNN相比MLP的优势
    if CONFIG["model"] == "cnn" and best_acc >= 0.99:
        print("\n📊 CNN相比MLP的优势总结:")
        print("-" * 40)
        print("1. 局部感知: 卷积核只关注图像局部区域")
        print("2. 参数共享: 同一卷积核在整个图像上滑动，参数量远少于MLP")
        print(f"   - 本CNN参数量: {count_params(model):,}")
        print(f"   - 同等精度MLP参数量: 通常超过1,000,000")
        print("3. 平移不变性: 数字在图像中位置变化不影响识别")
        print("4. 层次特征提取:")
        print("   - 第一层: 检测边缘、角点")
        print("   - 第二层: 组合成笔画部分")
        print("   - 全连接层: 组合成完整数字")
        print("-" * 40)

    # -------------------------
    # 保存曲线图：loss + acc
    # -------------------------
    if CONFIG["save_plot"]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax1.plot(train_losses, label="训练损失", linewidth=2)
        ax1.plot(test_losses, label="测试损失", linewidth=2)
        ax1.set_xlabel("训练轮数")
        ax1.set_ylabel("损失值")
        ax1.set_title("训练和测试损失曲线")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2.plot(test_accs, label="测试准确率", linewidth=2, color='green')
        ax2.axhline(y=target_acc, color='red', linestyle='--', 
                   label=f"目标准确率 ({target_acc*100:.0f}%)")
        ax2.set_xlabel("训练轮数")
        ax2.set_ylabel("准确率")
        ax2.set_title("测试准确率曲线")
        ax2.set_ylim([0.8, 1.0])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 添加模型信息和结果到标题
        result_status = "✅ 达标" if best_acc >= target_acc else "❌ 未达标"
        plt.suptitle(
            f"{CONFIG['model'].upper()} on MNIST | "
            f"Best Acc: {best_acc*100:.2f}% | "
            f"Target: {target_acc*100:.0f}% {result_status}", 
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()
        plt.savefig(CONFIG["plot_path"], dpi=160, bbox_inches="tight")
        print(f"\n📁 图表已保存: {CONFIG['plot_path']}")


if __name__ == "__main__":
    # 显示使用说明
    print("\n" + "="*60)
    print("神经网络实验：MLP vs CNN 在 MNIST 上的对比")
    print("="*60)
    print("学习目标:")
    print("1. 网络结构参数对模型精度的影响")
    print("2. 通过手动调参提升模型性能的方法") 
    print("3. CNN 相比 MLP 在图像分类任务中的优势")
    print("="*60)
    print("\n运行方法:")
    print("1. 运行MLP实验（目标98%）:")
    print("   在CONFIG中将 'model' 设置为 'mlp'")
    print("   运行: python bp_vs_cnn_mnist.py")
    print()
    print("2. 运行CNN实验（目标99%）:")
    print("   在CONFIG中将 'model' 设置为 'cnn'")
    print("   运行: python bp_vs_cnn_mnist.py")
    print("="*60 + "\n")
    
    # 检查是否安装了必要的库
    try:
        import torch
        import torchvision
        print("✅ PyTorch和torchvision已安装")
    except ImportError:
        print("❌ 请先安装PyTorch:")
        print("   pip install torch torchvision torchaudio")
        print("   或使用: pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple")
        exit(1)
    
    main()