import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import time
import torch.nn.functional as F
from scipy.io import loadmat

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 创建保存结果的目录
os.makedirs("noise_locator_results", exist_ok=True)


# 1. 数据加载与处理 - 添加分段功能
def load_and_process_data(clean_path, noisy_path, noise_threshold=0.1, segment_length=1024):
    """
    加载干净信号和含噪信号，生成噪声掩码，并将长信号分段
    :param clean_path: 干净信号.mat文件路径
    :param noisy_path: 含噪信号.mat文件路径
    :param noise_threshold: 噪声阈值，用于生成二值掩码
    :param segment_length: 分段长度
    :return: 含噪信号张量，噪声掩码张量
    """
    # 从.mat文件加载数据
    clean_data = loadmat(clean_path)['MNorigin1']
    noisy_data = loadmat(noisy_path)['MNnoisy1']

    # 确保数据形状相同
    assert clean_data.shape == noisy_data.shape, "Clean and noisy data must have the same shape"

    # 计算噪声信号
    noise = noisy_data - clean_data

    # 生成噪声掩码 - 噪声强度超过阈值的位置标记为1
    mask = (np.abs(noise) > noise_threshold).astype(np.float32)

    # 将长信号分割成多个片段
    def segment_signal(signal, seg_length):
        num_segments = signal.shape[1] // seg_length
        segments = []
        for i in range(num_segments):
            start = i * seg_length
            end = start + seg_length
            segments.append(signal[:, start:end])
        return np.array(segments)

    # 分割信号和掩码
    noisy_segments = segment_signal(noisy_data, segment_length)
    mask_segments = segment_signal(mask, segment_length)

    print(f"Original signal length: {noisy_data.shape[1]}")
    print(f"Segmented into {noisy_segments.shape[0]} segments of length {segment_length}")

    # 转换为PyTorch张量并增加通道维度 [样本数, 1, 信号长度]
    noisy_tensor = torch.tensor(noisy_segments, dtype=torch.float32)
    mask_tensor = torch.tensor(mask_segments, dtype=torch.float32)

    return noisy_tensor, mask_tensor


# 2. UNet噪声定位器模型（保持不变）
class UNet1D(nn.Module):
    """一维UNet噪声定位器 - 识别信号中的噪声区域"""

    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super().__init__()
        features = init_features

        # 编码器路径 (下采样)
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 瓶颈层
        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")

        # 解码器路径 (上采样)
        self.upconv4 = nn.ConvTranspose1d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose1d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose1d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose1d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = self._block(features * 2, features, name="dec1")

        # 输出层
        self.conv = nn.Conv1d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 编码器路径
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # 瓶颈层
        bottleneck = self.bottleneck(self.pool4(enc4))

        # 解码器路径
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        # 输出层
        out = self.conv(dec1)
        return out

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=features),
            nn.ReLU(inplace=True),
        )


# 3. 训练噪声定位器（修复BCELoss问题）
def train_noise_locator(model, train_loader, val_loader, epochs=50, lr=0.001):
    """
    训练噪声定位器模型
    :param model: 噪声定位器模型
    :param train_loader: 训练数据加载器
    :param val_loader: 验证数据加载器
    :param epochs: 训练轮数
    :param lr: 学习率
    :return: 训练好的模型
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 修复：使用BCEWithLogitsLoss替代BCELoss
    criterion = nn.BCEWithLogitsLoss()  # 修改这里

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # 记录训练过程
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_path = "noise_locator_results/best_model.pth"

    print("Starting noise locator training...")
    print(f"Training on {len(train_loader.dataset)} samples, validating on {len(val_loader.dataset)} samples")

    start_time = time.time()

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0.0
        for noisy, mask in train_loader:
            noisy = noisy.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            # 混合精度前向传播
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                pred_mask = model(noisy)
                loss = criterion(pred_mask, mask)  # 使用BCEWithLogitsLoss

            # 混合精度反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss += loss.item() * noisy.size(0)

        # 验证阶段
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for noisy, mask in val_loader:
                noisy = noisy.to(device)
                mask = mask.to(device)

                pred_mask = model(noisy)
                loss = criterion(pred_mask, mask)  # 使用BCEWithLogitsLoss

                epoch_val_loss += loss.item() * noisy.size(0)

        # 计算平均损失
        avg_train_loss = epoch_train_loss / len(train_loader.dataset)
        avg_val_loss = epoch_val_loss / len(val_loader.dataset)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # 更新学习率
        scheduler.step(avg_val_loss)

        # 打印训练进度
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{epochs}] | Time: {epoch_time:.1f}s | "
              f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        start_time = time.time()

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with val loss: {avg_val_loss:.6f}")

    # 保存最终模型
    final_model_path = "noise_locator_results/final_model.pth"
    torch.save(model.state_dict(), final_model_path)

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Noise Locator Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('noise_locator_results/loss_curves.png')
    plt.close()

    print(f"Noise locator training completed! Best model saved to {best_model_path}")
    return model


# 4. 可视化噪声定位结果（修复样本数量问题）
def visualize_noise_locator(model, full_noisy, full_mask, segment_length, num_samples=3,
                            save_dir="noise_locator_results"):
    """
    可视化噪声定位结果（包括完整信号）
    :param model: 训练好的噪声定位器模型
    :param full_noisy: 完整含噪信号
    :param full_mask: 完整真实掩码
    :param segment_length: 分段长度
    :param num_samples: 可视化样本数量
    :param save_dir: 保存目录
    """
    model.eval()

    # 处理完整信号
    def process_full_signal(signal, model, seg_length):
        num_segments = signal.shape[1] // seg_length
        full_pred = np.zeros_like(signal)

        for i in range(num_segments):
            start = i * seg_length
            end = start + seg_length
            segment = signal[:, start:end]

            # 转换为模型输入格式
            segment_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(1).to(device)

            with torch.no_grad():
                # 添加sigmoid激活
                pred_segment = torch.sigmoid(model(segment_tensor)).cpu().numpy()

            full_pred[:, start:end] = pred_segment[0, 0]

        return full_pred

    # 确保不超过可用样本数量
    n_samples = min(full_noisy.shape[0], num_samples)
    if n_samples == 0:
        print("No samples available for visualization!")
        return

    # 创建样本索引
    if full_noisy.shape[0] > n_samples:
        sample_indices = np.random.choice(full_noisy.shape[0], n_samples, replace=False)
    else:
        # 如果样本数量不足，使用所有样本
        sample_indices = np.arange(full_noisy.shape[0])

    plt.figure(figsize=(15, 5 * n_samples))

    for i, idx in enumerate(sample_indices):
        # 获取当前样本的完整信号
        noisy_signal = full_noisy[idx]
        true_mask = full_mask[idx]

        # 预测完整掩码
        pred_mask = process_full_signal(noisy_signal[np.newaxis, :], model, segment_length)[0]

        # 计算噪声信号
        noise_signal = noisy_signal * true_mask

        # 绘制原始含噪信号
        plt.subplot(n_samples, 3, i * 3 + 1)
        plt.plot(noisy_signal, 'b-', alpha=0.7, label='Noisy Signal')
        plt.title(f'Sample {i + 1}: Noisy Signal')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()

        # 绘制真实噪声区域
        plt.subplot(n_samples, 3, i * 3 + 2)
        plt.plot(noisy_signal, 'b-', alpha=0.3, label='Signal')
        plt.plot(noise_signal, 'r-', alpha=0.7, label='True Noise')
        plt.fill_between(range(len(noisy_signal)), noisy_signal,
                         where=true_mask > 0.5, color='red', alpha=0.3)
        plt.title('True Noise Regions')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()

        # 绘制预测噪声区域
        plt.subplot(n_samples, 3, i * 3 + 3)
        plt.plot(noisy_signal, 'b-', alpha=0.3, label='Signal')
        plt.plot(pred_mask, 'g-', alpha=0.5, label='Predicted Mask')
        plt.fill_between(range(len(noisy_signal)), noisy_signal,
                         where=pred_mask > 0.5, color='green', alpha=0.3)
        plt.title('Predicted Noise Regions')
        plt.xlabel('Time')
        plt.ylabel('Mask Probability')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    plt.savefig(f'{save_dir}/noise_locator_visualization.png')
    plt.show()
    print(f"Noise locator visualization saved to {save_dir}/noise_locator_visualization.png")

# 5. 主函数 - 添加分段处理
def main():
    # 设置分段长度
    SEGMENT_LENGTH = 256  # 根据GPU内存调整（1024, 2048等）

    # 加载数据并生成噪声掩码
    print("Loading and processing data...")
    noisy_data, mask_data = load_and_process_data(
        clean_path="C://Users//zh//Desktop//模拟数据训练//MNorigin1.mat",
        noisy_path="C://Users//zh//Desktop//模拟数据训练//MNnoisy1.mat",
        noise_threshold=0.4,  # 根据实际噪声水平调整
        segment_length=SEGMENT_LENGTH
    )

    # 保存完整信号用于可视化
    full_data = loadmat("C://Users//zh//Desktop//模拟数据训练//MNnoisy1.mat")['MNnoisy1']
    full_mask = loadmat("C://Users//zh//Desktop/模拟数据训练//MNorigin1.mat")['MNorigin1']
    noise = full_data - full_mask
    full_mask = (np.abs(noise) > 0.4).astype(np.float32)

    # 划分训练集和验证集 (80%训练, 20%验证)
    X_train, X_val, M_train, M_val = train_test_split(
        noisy_data, mask_data, test_size=0.2, random_state=42
    )

    # 创建数据加载器
    batch_size = 32  # 根据GPU内存调整
    train_dataset = TensorDataset(X_train, M_train)
    val_dataset = TensorDataset(X_val, M_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化UNet噪声定位器
    print("Initializing UNet noise locator...")
    model = UNet1D(in_channels=1, out_channels=1, init_features=32)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {total_params:,} parameters")

    # 训练噪声定位器
    trained_model = train_noise_locator(
        model,
        train_loader,
        val_loader,
        epochs=50,
        lr=0.001
    )

    # 可视化结果（使用完整信号）
    visualize_noise_locator(
        trained_model,
        full_data,
        full_mask,
        SEGMENT_LENGTH,
        num_samples=5
    )

    print("Noise locator training and evaluation completed!")


if __name__ == "__main__":
    main()