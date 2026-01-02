"seed": 42,

    # 选择要训练的模型： "mlp" 或 "cnn"
    "model": "cnn",

    # 训练相关参数（可以改，用于观察收敛与精度变化）
    "epochs": 20,
    "batch_size": 128,
    "lr": 1e-3,             # 建议对比：1e-2 / 1e-3 / 1e-4
    "optimizer": "adam",    # "adam" 或 "sgd"

	@@ -136,50 +136,30 @@ class MLP(nn.Module):
    - 先 Flatten 成向量 [B, 784]
    - 再走全连接层做分类
    """
    def __init__(self):
        super().__init__()
        
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
        # 修改为 3 层隐藏层，增加模型容量
        self.fc1 = nn.Linear(28 * 28, 512)   # 第一层：784 -> 512
        self.fc2 = nn.Linear(512, 256)       # 第二层：512 -> 256
        self.fc3 = nn.Linear(256, 128)       # 第三层：256 -> 128
        self.out = nn.Linear(128, 10)        # 输出层：128 -> 10

        # 添加 Dropout 防止过拟合
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # 第一层后使用 Dropout
        x = self.relu(self.fc2(x))
        x = self.dropout(x)  # 第二层后使用 Dropout
        x = self.relu(self.fc3(x))
        x = self.out(x)      # 最后一层不使用 Dropout

        return x


	@@ -222,26 +202,49 @@ def __init__(self):
        # 全连接层输入维度要写成： (conv2_out_channels * 7 * 7)
        # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

        c1_out = 32   # 第一层卷积：1 -> 32
        c2_out = 64   # 第二层卷积：32 -> 64

        # 添加第三个卷积层
        c3_out = 128  # 第三层卷积：64 -> 128

        self.conv1 = nn.Conv2d(1, c1_out, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1_out, c2_out, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(c2_out, c3_out, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

        # 添加 BatchNorm 层，帮助训练稳定
        self.bn1 = nn.BatchNorm2d(c1_out)
        self.bn2 = nn.BatchNorm2d(c2_out)
        self.bn3 = nn.BatchNorm2d(c3_out)

        # 注意：由于有3次池化，特征图尺寸：28x28 -> 14x14 -> 7x7 -> 3x3
        # 所以全连接层输入维度 = c3_out * 3 * 3
        self.fc1 = nn.Linear(c3_out * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: [B, 1, 28, 28]
        x = self.relu(self.bn1(self.conv1(x)))  # -> [B, 32, 28, 28]
        x = self.pool(x)                        # -> [B, 32, 14, 14]

        x = self.relu(self.bn2(self.conv2(x)))  # -> [B, 64, 14, 14]
        x = self.pool(x)                        # -> [B, 64, 7, 7]

        x = self.relu(self.bn3(self.conv3(x)))  # -> [B, 128, 7, 7]
        x = self.pool(x)                        # -> [B, 128, 3, 3]

        x = x.view(x.size(0), -1)               # -> [B, 128*3*3=1152]
        x = self.dropout(x)

        x = self.relu(self.fc1(x))              # -> [B, 256]
        x = self.dropout(x)
        x = self.fc2(x)                         # -> [B, 10]

        return x