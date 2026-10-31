import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from noiselocator import UNet1D
import librosa
import matplotlib as mpl

# ==============================
# 2. Mask-Conditioned WaveNet 去噪模型
# ==============================

class ResidualBlock(nn.Module):
    """带有门控机制的WaveNet残差块"""

    def __init__(self, in_channels, res_channels, skip_channels, dilation, kernel_size=3):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size

        # 膨胀卷积层
        self.filter_conv = nn.Conv1d(
            in_channels,
            res_channels,
            kernel_size=kernel_size,
            padding=(dilation * (kernel_size - 1)) // 2,
            dilation=dilation
        )
        self.gate_conv = nn.Conv1d(
            in_channels,
            res_channels,
            kernel_size=kernel_size,
            padding=(dilation * (kernel_size - 1)) // 2,
            dilation=dilation
        )

        # 1x1卷积用于残差连接和跳跃连接
        self.res_conv = nn.Conv1d(res_channels, in_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)

    def forward(self, x):
        # 门控机制
        filtered = self.filter_conv(x)
        gated = self.gate_conv(x)
        activated = torch.tanh(filtered) * torch.sigmoid(gated)

        # 残差连接
        res = self.res_conv(activated)
        res += x  # 残差连接

        # 跳跃连接
        skip = self.skip_conv(activated)

        return res, skip


class MaskConditionedWaveNet(nn.Module):
    """Mask-Conditioned WaveNet去噪模型"""

    def __init__(self,
                 in_channels=2,
                 res_channels=64,
                 skip_channels=64,
                 out_channels=1,
                 num_blocks=8,
                 dilation_cycle=4,
                 kernel_size=3):
        super().__init__()

        # 输入预处理层
        self.pre_conv = nn.Conv1d(in_channels, res_channels, kernel_size=1)

        # 创建残差块
        self.res_blocks = nn.ModuleList()
        dilations = [2 ** i for i in range(dilation_cycle)] * (num_blocks // dilation_cycle)
        for d in dilations:
            self.res_blocks.append(ResidualBlock(
                res_channels,
                res_channels,
                skip_channels,
                dilation=d,
                kernel_size=kernel_size
            ))

        # 跳跃连接处理
        self.post_conv1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.post_conv2 = nn.Conv1d(skip_channels, out_channels, kernel_size=1)

        # 输出层不使用激活函数，因为处理的是原始音频信号
        self.dilations = dilations  # 保存膨胀系数用于可视化

    def forward(self, x, mask):
        # 拼接输入信号和噪声掩码
        inputs = torch.cat([x, mask], dim=1)

        # 初始卷积
        x = self.pre_conv(inputs)

        # 残差块处理
        skip_connections = []
        for block in self.res_blocks:
            x, skip = block(x)
            skip_connections.append(skip)

        # 聚合所有跳跃连接
        skips_sum = sum(skip_connections)

        # 后处理
        x = F.relu(skips_sum)
        x = F.relu(self.post_conv1(x))
        x = self.post_conv2(x)  # 输出层不使用激活函数

        return x

    def visualize_architecture(self, writer, input_shape=(1, 1, 1024)):
        """可视化模型架构"""
        # 生成虚拟输入
        dummy_x = torch.randn(2, input_shape[1], input_shape[2])
        dummy_mask = torch.rand(2, 1, input_shape[2])

        # 写入TensorBoard
        writer.add_graph(self, (dummy_x, dummy_mask))

        # 打印模型结构
        print("Mask-Conditioned WaveNet Architecture:")
        print(self)

        # 打印膨胀系数配置
        print("\nDilation configuration:")
        for i, d in enumerate(self.dilations):
            print(f"Block {i + 1}: dilation={d}")

        # 参数统计
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")


# ==============================
# 3. 端到端去噪系统
# ==============================

class DenoisingPipeline:
    """端到端去噪流程（定位+去噪）"""

    def __init__(self, locator, denoiser, device='cpu', threshold=0.5):
        self.locator = locator.to(device).eval()
        self.denoiser = denoiser.to(device).eval()
        self.device = device
        self.threshold = threshold

    def denoise(self, noisy_signal):
        """执行完整去噪流程"""
        with torch.no_grad():
            # 噪声定位
            noise_mask = self.locator(noisy_signal.unsqueeze(0).to(self.device))
            binary_mask = (noise_mask > self.threshold).float()

            # 去噪处理
            denoised = self.denoiser(
                noisy_signal.unsqueeze(0).to(self.device),
                binary_mask
            )

        return denoised.squeeze(0).cpu(), noise_mask.squeeze(0).cpu(), binary_mask.squeeze(0).cpu()

    def visualize_results(self, noisy, denoised, clean, noise_mask, binary_mask,
                          sample_rate=150, filename='denoising_results'):
        """
        可视化去噪结果，每个图表单独保存为SVG矢量图格式
        参数:
            filename: 不带扩展名的文件名基础部分
        """
        # 确保denoised是numpy数组
        if hasattr(denoised, 'numpy'):
            denoised = denoised.numpy()

        # 创建时间轴
        time = np.arange(len(noisy)) / sample_rate

        # 设置中文字体和全局字号 - 学术论文格式
        plt.rcParams['font.family'] = 'SimHei'
        plt.rcParams['font.size'] = 20
        plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
        plt.rcParams['svg.fonttype'] = 'path'  # 确保字体正确嵌入SVG

        # 1. 原始含噪信号
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax.plot(time, noisy, 'b', linewidth=1.5)
        ax.set_title('含噪信号', fontsize=24)
        ax.set_xlabel('时间 (s)', fontsize=22)
        ax.set_ylabel('幅度', fontsize=22)
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{filename}_noisy_signal.svg", format='svg', bbox_inches='tight')
        plt.close()

        # 2. 噪声掩码
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax.plot(time, noise_mask, 'g', linewidth=1.5, label='噪声概率')
        ax.plot(time, binary_mask, 'r', alpha=0.8, linewidth=1.5, label='二值掩码')
        ax.set_title('噪声掩码', fontsize=24)
        ax.set_xlabel('时间 (s)', fontsize=22)
        ax.set_ylabel('概率', fontsize=22)
        ax.legend(fontsize=18, loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(-0.1, 1.1)
        plt.tight_layout()
        plt.savefig(f"{filename}_noise_mask.svg", format='svg', bbox_inches='tight')
        plt.close()

        # 3. 去噪结果对比
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax.plot(time, noisy, 'b', alpha=0.5, linewidth=1.2, label='含噪信号')
        ax.plot(time, denoised, 'r', linewidth=1.8, label='去噪结果')
        ax.set_title('去噪结果对比', fontsize=24)
        ax.set_xlabel('时间 (s)', fontsize=22)
        ax.set_ylabel('幅度', fontsize=22)
        ax.legend(fontsize=18, loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{filename}_denoising_comparison.svg", format='svg', bbox_inches='tight')
        plt.close()

        # 4. 去噪效果对比
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        ax.plot(time, clean, 'g', linewidth=1.8, label='纯净信号')
        ax.plot(time, denoised, 'r', alpha=0.8, linewidth=1.8, label='去噪结果')
        ax.set_title('去噪效果对比', fontsize=24)
        ax.set_xlabel('时间 (s)', fontsize=22)
        ax.set_ylabel('幅度', fontsize=22)
        ax.legend(fontsize=18, loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{filename}_clean_denoised_comparison.svg", format='svg', bbox_inches='tight')
        plt.close()

        # 5. 频谱对比
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)

        n_fft = 1024
        hop_length = n_fft // 4

        # 含噪信号频谱
        S_noisy = librosa.amplitude_to_db(
            np.abs(librosa.stft(noisy, n_fft=n_fft, hop_length=hop_length)),
            ref=np.max
        )

        # 去噪信号频谱
        S_denoised = librosa.amplitude_to_db(
            np.abs(librosa.stft(denoised, n_fft=n_fft, hop_length=hop_length)),
            ref=np.max
        )

        # 频谱差异
        diff = S_noisy - S_denoised
        im = ax.imshow(diff, aspect='auto', origin='lower',
                       extent=[0, time[-1], 0, sample_rate // 2],
                       cmap='coolwarm')

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('幅度差 (dB)', fontsize=20)
        cbar.ax.tick_params(labelsize=18)

        ax.set_title('频谱差异 (含噪 - 去噪)', fontsize=24)
        ax.set_xlabel('时间 (s)', fontsize=22)
        ax.set_ylabel('频率 (Hz)', fontsize=22)

        plt.tight_layout()
        plt.savefig(f"{filename}_spectrogram_diff.svg", format='svg', bbox_inches='tight')
        plt.close()

        # 恢复默认设置
        plt.rcParams.update(mpl.rcParamsDefault)
        return

    def calculate_metrics(self, clean, denoised, noisy):
        """计算去噪性能指标"""
        clean = clean.numpy()
        denoised = denoised.numpy()
        noisy = noisy.numpy()

        # 信噪比改进
        snr_before = 10 * np.log10(np.mean(clean ** 2) / np.mean((clean - noisy) ** 2))
        snr_after = 10 * np.log10(np.mean(clean ** 2) / np.mean((clean - denoised) ** 2))
        snr_improvement = snr_after - snr_before

        # 分段信噪比
        def segmental_snr(x, y, segment_size=256):
            segments = []
            for i in range(0, len(x) - segment_size, segment_size // 2):
                segment_x = x[i:i + segment_size]
                segment_y = y[i:i + segment_size]
                seg_snr = 10 * np.log10(np.mean(segment_x ** 2) / np.mean((segment_x - segment_y) ** 2))
                segments.append(seg_snr)
            return np.mean(segments)

        seg_snr = segmental_snr(clean, denoised)

        # 波形相似度
        waveform_sim = np.dot(clean, denoised) / (np.linalg.norm(clean) * np.linalg.norm(denoised))

        return {
            'SNR_before': snr_before,
            'SNR_after': snr_after,
            'SNR_improvement': snr_improvement,
            'Segmental_SNR': seg_snr,
            'Waveform_Cosine_Similarity': waveform_sim
        }


# ==============================
# 4. 数据处理与训练
# ==============================

def load_and_preprocess_data(noisy_path, clean_path, segment_length=256):
    # 加载含噪信号
    noisy_data = loadmat(noisy_path)
    noisy_signals = noisy_data['MNnoisy1'].flatten()  # 展平为一维数组

    # 加载干净信号
    clean_data = loadmat(clean_path)
    clean_signals = clean_data['MNorigin1'].flatten()  # 展平为一维数组

    # 确保两个信号长度一致
    min_length = min(len(noisy_signals), len(clean_signals))
    noisy_signals = noisy_signals[:min_length]
    clean_signals = clean_signals[:min_length]

    # 计算可以分割的段数
    num_segments = min_length // segment_length
    # 裁剪数据使其能被segment_length整除
    noisy_signals = noisy_signals[:num_segments * segment_length]
    clean_signals = clean_signals[:num_segments * segment_length]

    # 分割成256长度的片段
    noisy_segments = noisy_signals.reshape(num_segments, segment_length)
    clean_segments = clean_signals.reshape(num_segments, segment_length)

    # 划分训练集和验证集 (80%训练, 20%验证)
    X_train, X_val, Y_train, Y_val = train_test_split(
        noisy_segments, clean_segments, test_size=0.2, random_state=42
    )

    # 转换为PyTorch张量并增加通道维度 [样本数, 1, 256]
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
    Y_train = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(1)
    Y_val = torch.tensor(Y_val, dtype=torch.float32).unsqueeze(1)

    return X_train, X_val, Y_train, Y_val


def train_denoiser(denoiser, locator, train_loader, val_loader, epochs=100, lr=0.001, device='cuda'):
    """训练去噪模型"""
    denoiser = denoiser.to(device)
    locator = locator.to(device).eval()  # 定位器设为评估模式

    optimizer = torch.optim.Adam(denoiser.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    criterion = nn.MSELoss()

    writer = SummaryWriter()
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # 训练阶段
        denoiser.train()
        train_loss = 0.0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)

            # 使用定位器生成噪声掩码
            with torch.no_grad():
                noise_mask = locator(noisy)
                binary_mask = (noise_mask > 0.5).float()

            optimizer.zero_grad()
            outputs = denoiser(noisy, binary_mask)
            loss = criterion(outputs, clean)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * noisy.size(0)

        train_loss /= len(train_loader.dataset)

        # 验证阶段
        denoiser.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)

                # 使用定位器生成噪声掩码
                noise_mask = locator(noisy)
                binary_mask = (noise_mask > 0.5).float()

                outputs = denoiser(noisy, binary_mask)
                loss = criterion(outputs, clean)
                val_loss += loss.item() * noisy.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        # 记录到TensorBoard
        writer.add_scalars('Loss', {
            'train': train_loss,
            'val': val_loss
        }, epoch)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'denoiser_state_dict': denoiser.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'epoch': epoch
            }, 'best_denoiser.pth')

        # 打印进度
        print(f'Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')

    writer.close()
    print(f'Training complete. Best validation loss: {best_val_loss:.6f}')
    return denoiser


# ==============================
# 5. 实验流程
# ==============================

def main():
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载预训练的UNet噪声定位器
    locator = UNet1D()
    # 假设定位器已经预训练好，加载权重
    locator.load_state_dict(torch.load("noise_locator_results/final_model.pth", map_location=device))
    locator = locator.to(device).eval()
    print("Loaded pretrained UNet noise locator")

    # 2. 初始化Mask-Conditioned WaveNet
    denoiser = MaskConditionedWaveNet(
        in_channels=2,  # 输入：信号 + 噪声掩码
        res_channels=64,  # 残差通道数
        skip_channels=64,  # 跳跃连接通道数
        out_channels=1,  # 输出：去噪信号
        num_blocks=10,  # 残差块数量
        dilation_cycle=4,  # 膨胀系数循环周期
        kernel_size=3  # 卷积核大小
    )

    # 3. 可视化模型架构
    writer = SummaryWriter()
    denoiser.visualize_architecture(writer, input_shape=(1, 1, 100000))
    print("Model architecture visualized in TensorBoard")

    # 4. 准备数据集
    X_train, X_val, Y_train, Y_val = load_and_preprocess_data(
        "C://Users//zh//Desktop//模拟数据训练//MNnoisy1.mat",
        "C://Users//zh//Desktop//模拟数据训练//MNorigin1.mat"
    )

    # 创建数据加载器
    batch_size = 64
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    print(f"Dataset prepared: {len(train_dataset)} training, {len(val_dataset)} validation samples")

    # 5. 训练去噪模型
    print("\nStarting training for Mask-Conditioned WaveNet...")
    trained_denoiser = train_denoiser(
        denoiser,
        locator,
        train_loader,
        val_loader,
        epochs=100,
        lr=0.001,
        device=device
    )

    # 6. 创建完整去噪流程
    pipeline = DenoisingPipeline(locator, trained_denoiser, device=device)

    # 7. 测试与评估
    # 加载测试音频
    test_audio = loadmat("C://Users//zh//Desktop//模拟数据训练//模拟数据验证//MNnoisy1.mat")
    test_audio = test_audio['MNnoisy1'].flatten()
    test_clean = loadmat("C://Users//zh//Desktop//模拟数据训练//模拟数据验证//MNorigin1.mat")
    test_clean = test_clean['MNorigin1'].flatten()


    # 转换为张量
    test_audio_tensor = torch.FloatTensor(test_audio).unsqueeze(0)

    # 执行去噪
    denoised, noise_mask, binary_mask = pipeline.denoise(test_audio_tensor)

    # 可视化结果
    fig = pipeline.visualize_results(
        test_audio,
        denoised.squeeze(0),
        test_clean,
        noise_mask.squeeze(0),
        binary_mask.squeeze(0),
        sample_rate=150,
        filename='denoising_results.png'
    )

    # 计算评估指标
    metrics = pipeline.calculate_metrics(
        torch.FloatTensor(test_clean),
        denoised.squeeze(0),
        torch.FloatTensor(test_audio)
    )

    print("\nDenoising Performance Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 保存
    savemat("denoised_signal.mat", {'denoised_signal': denoised})
    print("Denoising complete. Results saved to denoising_results.png and denoised.mat")


if __name__ == "__main__":
    main()