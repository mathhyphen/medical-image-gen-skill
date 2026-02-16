"""
RectifiedFlow 使用示例
基于 https://github.com/gnobitab/RectifiedFlow
"""

import sys
sys.path.append('RectifiedFlow/ImageGeneration')

import torch
import torch.nn as nn


class SimpleUNet(nn.Module):
    """简化的3D U-Net作为示例"""
    
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        # 实际使用RectifiedFlow的UNet实现
        self.encoder = nn.Sequential(
            nn.Conv3d(in_ch, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, out_ch, 3, padding=1),
        )
    
    def forward(self, x, t):
        # t是时间参数
        return self.decoder(self.encoder(x))


def rectified_flow_loss(model, x0, x1):
    """
    Rectified Flow训练损失
    
    目标：学习从x0(噪声)到x1(数据)的直线路径
    """
    batch_size = x0.shape[0]
    
    # 随机采样时间t
    t = torch.rand(batch_size, device=x0.device)
    
    # 线性插值: x_t = t*x_1 + (1-t)*x_0
    t = t.view(batch_size, 1, 1, 1, 1)
    x_t = t * x1 + (1 - t) * x0
    
    # 预测velocity
    v_pred = model(x_t, t.squeeze())
    
    # 目标velocity: x_1 - x_0
    v_target = x1 - x0
    
    return nn.functional.mse_loss(v_pred, v_target)


@torch.no_grad()
def sample_single_step(model, shape, device='cuda'):
    """
    单步生成 - RectifiedFlow的核心优势！
    
    从噪声直接生成目标图像，只需一步
    """
    model.eval()
    
    # 从噪声开始
    x = torch.randn(shape, device=device)
    
    # 单步推理
    t = torch.zeros(x.shape[0], device=device)
    v = model(x, t)
    
    # 直接到终点！
    x_generated = x + v
    
    return x_generated


# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = SimpleUNet(in_ch=1, out_ch=1).cuda()
    
    # 模拟数据
    x0 = torch.randn(1, 1, 64, 64, 64).cuda()  # 噪声
    x1 = torch.randn(1, 1, 64, 64, 64).cuda()  # 目标
    
    # 训练
    loss = rectified_flow_loss(model, x0, x1)
    print(f"Loss: {loss.item()}")
    
    # 单步生成
    generated = sample_single_step(model, (1, 1, 64, 64, 64))
    print(f"Generated shape: {generated.shape}")
