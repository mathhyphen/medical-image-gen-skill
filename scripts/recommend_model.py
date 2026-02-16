"""
Model recommendation for medical image generation
"""

RECOMMENDATIONS = {
    "cross_modality": {
        "real_time": {
            "model": "RectifiedFlow",
            "repo": "gnobitab/RectifiedFlow",
            "install": "git clone https://github.com/gnobitab/RectifiedFlow.git",
            "steps": 1,
            "note": "单步推理，适合MRI↔CT实时转换"
        },
        "high_quality": {
            "model": "MONAI DDPM",
            "repo": "Project-MONAI/GenerativeModels", 
            "install": "pip install monai-generative",
            "steps": 50,
            "note": "多步采样，质量最高"
        }
    },
    "denoising": {
        "real_time": {
            "model": "RectifiedFlow", 
            "repo": "gnobitab/RectifiedFlow",
            "install": "git clone https://github.com/gnobitab/RectifiedFlow.git",
            "note": "低剂量CT去噪，实时处理"
        }
    },
    "super_resolution": {
        "real_time": {
            "model": "Consistency Models",
            "repo": "openai/consistency_models",
            "install": "git clone https://github.com/openai/consistency_models.git",
            "note": "单步超分辨率"
        }
    }
}


def recommend(task: str = "cross_modality", speed: str = "real_time") -> dict:
    """
    推荐医学影像生成模型
    
    Args:
        task: cross_modality / denoising / super_resolution
        speed: real_time / high_quality
    
    Returns:
        推荐结果字典
    """
    result = RECOMMENDATIONS.get(task, {}).get(speed)
    
    if not result:
        return {
            "error": f"Unknown combination: {task} + {speed}",
            "suggestion": "Try: task='cross_modality', speed='real_time'"
        }
    
    return {
        "task": task,
        "speed": speed,
        **result,
        "example_code": get_example_code(result["model"])
    }


def get_example_code(model: str) -> str:
    """获取代码示例"""
    
    if "RectifiedFlow" in model:
        return '''
# RectifiedFlow 单步生成示例
import sys
sys.path.append('RectifiedFlow/ImageGeneration')
import torch

# 加载模型
model = load_model('rectified_flow.pt')

# 单步推理
with torch.no_grad():
    output = model.sample_straight(input_image, num_steps=1)
'''
    
    elif "MONAI" in model:
        return '''
# MONAI DDPM 示例
from monai.generative.networks.nets import DiffusionModelUNet

model = DiffusionModelUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1
)

# 训练/推理
# 见 MONAI 官方文档
'''
    
    elif "Consistency" in model:
        return '''
# Consistency Models 单步生成
from consistency_models import ConsistencyModel

model = ConsistencyModel.load_from_checkpoint("path")
output = model.sample(input_shape)  # 单步!
'''
    
    return "See documentation for example code."


def list_options():
    """列出所有可用选项"""
    print("Available tasks:")
    for task in RECOMMENDATIONS:
        print(f"  - {task}")
        for speed in RECOMMENDATIONS[task]:
            model = RECOMMENDATIONS[task][speed]["model"]
            print(f"    {speed}: {model}")


if __name__ == "__main__":
    # 测试
    import json
    result = recommend("cross_modality", "real_time")
    print(json.dumps(result, indent=2, ensure_ascii=False))
