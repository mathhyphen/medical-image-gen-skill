---
name: medical-image-gen
description: |
  医学影像生成模型推荐与代码生成。当用户需要：
  1. 选择医学影像生成模型（MRI/CT跨模态合成、去噪、超分辨率等）
  2. 获取基于真实仓库的代码示例
  3. 了解RectifiedFlow/MONAI等生成模型的使用方法
  使用此Skill获取推荐和代码模板。
---

# Medical Image Generation

帮助用户选择合适的医学影像生成模型，并提供可直接使用的代码示例。

## 模型选择

根据场景推荐：

**实时推理（单步）**
- 推荐: @gnobitab/RectifiedFlow
- 场景: 临床实时生成、低延迟需求
- 代码: 见 `repositories/rectified-flow-example.py`

**高质量生成**
- 推荐: MONAI GenerativeModels (DDPM/ControlNet)
- 场景: 最高质量、条件控制
- 代码: 见 `repositories/monai-example.py`

**超分辨率**
- 推荐: @openai/consistency_models 或 RectifiedFlow
- 场景: 实时超分

## 快速使用

```python
# 获取模型推荐
from scripts.recommend_model import recommend

result = recommend(
    task="mri_to_ct",  # 或 denoising, super_resolution
    speed="real_time"  # 或 high_quality
)
# 返回: 推荐模型 + 安装命令 + 代码示例
```

## 外部依赖

使用此Skill需要用户自行安装：

```bash
# RectifiedFlow (ICLR 2023)
git clone https://github.com/gnobitab/RectifiedFlow.git

# MONAI GenerativeModels
pip install monai-generative

# Consistency Models (optional)
git clone https://github.com/openai/consistency_models.git
```

## 代码示例

### RectifiedFlow - 单步生成

```python
# 基于 @gnobitab/RectifiedFlow
import sys
sys.path.append('RectifiedFlow/ImageGeneration')

from model import UNet
import torch

model = UNet(in_channels=1, out_channels=1)

# 训练
loss = rectified_flow_loss(model, x0, x1)

# 单步推理 (核心优势!)
x1 = model.sample_straight(x0, num_steps=1)
```

### MONAI - 可控生成

```python
# 基于 MONAI GenerativeModels
from monai.generative.networks.nets import DiffusionModelUNet

model = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1
)
```

## 参考文献

详细仓库信息和代码模板：
- `references/external-repos.yaml` - 外部仓库完整信息
- `repositories/` - 代码示例目录

## 使用场景

**使用此Skill当：**
- 不知道医学影像生成该用什么模型
- 需要基于真实仓库的代码模板
- 了解 RectifiedFlow vs Diffusion 的区别

**不使用当：**
- 已有明确的模型选择
- 只需要通用的PyTorch代码
