"""
MONAI GenerativeModels 使用示例
基于 https://github.com/Project-MONAI/GenerativeModels
"""

from monai.generative.networks.nets import DiffusionModelUNet, ControlNet
from monai.generative.schedulers import DDPMScheduler
import torch
import torch.nn.functional as F


def create_diffusion_model():
    """创建3D扩散模型"""
    
    model = DiffusionModelUNet(
        spatial_dims=3,           # 3D医学影像
        in_channels=1,            # 输入通道
        out_channels=1,           # 输出通道
        num_res_blocks=2,         # 残差块数
        num_channels=(32, 64, 128, 256),  # 通道数
        attention_levels=(False, False, True, True),  # 注意力层
    )
    
    return model


def train_step(model, scheduler, images):
    """
    单步训练
    
    Args:
        model: DiffusionModelUNet
        scheduler: DDPMScheduler
        images: 训练图像 [B, C, H, W, D]
    """
    batch_size = images.shape[0]
    device = images.device
    
    # 随机时间步
    timesteps = torch.randint(
        0, scheduler.num_train_timesteps, 
        (batch_size,), device=device
    )
    
    # 添加噪声
    noise = torch.randn_like(images)
    noisy_images = scheduler.add_noise(images, noise, timesteps)
    
    # 预测噪声
    noise_pred = model(noisy_images, timesteps)
    
    # MSE损失
    loss = F.mse_loss(noise_pred, noise)
    
    return loss


def create_controlnet_model():
    """创建ControlNet用于条件控制"""
    
    controlnet = ControlNet(
        spatial_dims=3,
        in_channels=1,
        num_res_blocks=2,
        num_channels=(32, 64, 128, 256),
    )
    
    return controlnet


def sample(model, scheduler, shape, num_steps=50):
    """
    采样生成
    
    Args:
        model: 训练好的模型
        scheduler: 噪声调度器
        shape: 输出形状
        num_steps: 采样步数
    """
    model.eval()
    device = next(model.parameters()).device
    
    # 从纯噪声开始
    x = torch.randn(shape, device=device)
    
    # 逐步去噪
    scheduler.set_timesteps(num_steps)
    
    with torch.no_grad():
        for t in scheduler.timesteps:
            # 预测噪声
            noise_pred = model(x, t)
            
            # 计算去噪后的x
            x = scheduler.step(noise_pred, t, x).prev_sample
    
    return x


# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = create_diffusion_model().cuda()
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # 模拟训练
    images = torch.randn(2, 1, 64, 64, 64).cuda()
    loss = train_step(model, scheduler, images)
    print(f"Training loss: {loss.item()}")
    
    # 采样 (需要训练后)
    # generated = sample(model, scheduler, (1, 1, 64, 64, 64))
