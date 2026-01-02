    "seed": 42,

    # 选择要训练的模型： "mlp" 或 "cnn"
    "model": "mlp",
    "model": "cnn",

    # 训练相关参数（可以改，用于观察收敛与精度变化）
    "epochs": 10,
@@ -44,7 +44,7 @@

    # 输出
    "save_plot": True,
    "plot_path": "results.png",
    "plot_path": "cnn_results.png",
}


@@ -161,8 +161,8 @@ def __init__(self):
        # 你只需要改下面这些 Linear 的输入/输出维度即可。
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

        self.fc1 = nn.Linear(28 * 28, 256)   # 改这里：例如 128 / 256 / 512
        self.fc2 = nn.Linear(256, 128)       # 改这里：例如 64 / 128 / 256
        self.fc1 = nn.Linear(28 * 28, 500)   # 改这里：例如 128 / 256 / 512
        self.fc2 = nn.Linear(500, 128)       # 改这里：例如 64 / 128 / 256
        # 如需增加第三个隐藏层，可新增 fc3，并把最后输出层改名
        self.out = nn.Linear(128, 10)        # 最后一层输出固定 10 类（0~9）

@@ -222,25 +222,31 @@ def __init__(self):
        # 全连接层输入维度要写成： (conv2_out_channels * 7 * 7)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

        c1_out = 16   # 改这里：8 / 16 / 32
        c2_out = 32   # 改这里：16 / 32 / 64
        c1_out = 32   # 改这里：8 / 16 / 32
        c2_out = 64   # 改这里：16 / 32 / 64

        self.conv1 = nn.Conv2d(1, c1_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1_out, c2_out, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)  # 2x2 池化，尺寸减半
        self.pool = nn.MaxPool2d(2) 

        self.bn1=nn.BatchNorm2d(c1_out)
        self.bn2=nn.BatchNorm2d(c2_out)
         # 2x2 池化，尺寸减半
        self.dropout=nn.Dropout(0.25)
        # 全连接层：输入是 c2_out * 7 * 7
        self.fc1 = nn.Linear(c2_out * 7 * 7, 128)  # 可以改 128 -> 256 试试
        self.fc2 = nn.Linear(128, 10)
        self.fc1 = nn.Linear(c2_out * 7 * 7, 256)  # 可以改 128 -> 256 试试
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # x: [B, 1, 28, 28]  (CNN 不需要 Flatten 输入)
        x = self.pool(self.relu(self.conv1(x)))  # -> [B, c1_out, 14, 14]
        x = self.pool(self.relu(self.conv2(x)))  # -> [B, c2_out, 7, 7]
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # -> [B, c1_out, 14, 14]
        x = self.pool(self.relu(self.bn2(self.conv2(x)))) # -> [B, c2_out, 7, 7]
        x = x.view(x.size(0), -1)                # -> [B, c2_out*7*7]
        x = self.dropout(x)
        x = self.relu(self.fc1(x))

        x = self.fc2(x)
        return x

