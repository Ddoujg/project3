# 基于 UNet-WaveNet 的音频大地电磁信号去噪

本仓库提供论文《基于 UNet-WaveNet 级联模型的音频大地电磁噪声识别与抑制》的核心 PyTorch 代码。模型首先利用一维 UNet 识别噪声区域并生成掩码，再通过掩码引导 WaveNet 进行信号重建，实现噪声定位与抑制。

## 文件说明

- `noiselocator.py`：训练一维 UNet 噪声定位模型，生成噪声掩码。
- `unet-wavenet.py`：实现 UNet 与掩码条件 WaveNet 的级联去噪流程。
- `unet-dncnn.py`：实现 UNet 与 DnCNN 的级联去噪，用于对比实验。
- `wavenet.py`：实现直接使用 WaveNet 去噪的对比模型。

## 环境依赖

```bash
pip install torch numpy scipy matplotlib scikit-learn librosa tensorboard
```

## 使用方法

1. 准备配对的纯净信号与含噪信号，采用 MATLAB `.mat` 格式，变量名分别为 `MNorigin1` 和 `MNnoisy1`，并修改脚本中的训练及测试数据路径。
2. 运行噪声定位程序，生成后续级联模型所需的权重：

   ```bash
   python noiselocator.py
   ```

3. 运行级联去噪模型：

   ```bash
   python unet-wavenet.py
   ```

对比实验可分别运行 `unet-dncnn.py` 和 `wavenet.py`。程序提供模型训练、结果可视化及信噪比等指标评估功能；级联模型默认加载 `noise_locator_results/final_model.pth`。

## 相关论文

张浩，张良，徐军，等．基于 UNet-WaveNet 级联模型的音频大地电磁噪声识别与抑制．中南大学学报（自然科学版）。

如本代码对您的研究有所帮助，欢迎引用相关论文。
