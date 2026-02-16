# Medical Image Generation Skill

医学影像生成模型推荐与代码生成工具 | Medical Image Generation Model Recommendation Tool

---

## 🎯 用途 | Purpose

帮助用户快速选择合适的医学影像生成模型，并提供可直接使用的代码示例。

Help users quickly select appropriate medical image generation models and provide ready-to-use code examples.

## 📦 安装 | Installation

```bash
pip install git+https://github.com/mathhyphen/medical-image-gen-skill.git
```

## 🚀 快速使用 | Quick Usage

```python
from scripts.recommend_model import recommend

# 获取模型推荐 | Get model recommendation
result = recommend(
    task="cross_modality",  # 任务: cross_modality/denoising/super_resolution
    speed="real_time"       # 速度: real_time/high_quality
)

print(result["model"])      # 推荐模型
print(result["install"])    # 安装命令
print(result["example_code"])  # 代码示例
```

## 📋 支持的任务 | Supported Tasks

| 任务 | Task | 实时推荐 | 高质量推荐 |
|:---|:---|:---|:---|
| 跨模态合成 | Cross-Modality | RectifiedFlow | MONAI DDPM |
| 去噪 | Denoising | RectifiedFlow | - |
| 超分辨率 | Super-Resolution | Consistency Models | - |

## 🔗 外部依赖 | External Dependencies

本Skill推荐以下仓库（需自行安装）：

- [@gnobitab/RectifiedFlow](https://github.com/gnobitab/RectifiedFlow) - ICLR 2023
- [MONAI GenerativeModels](https://github.com/Project-MONAI/GenerativeModels)
- [@openai/consistency_models](https://github.com/openai/consistency_models)

## 📂 文件结构 | File Structure

```
medical-image-gen-skill/
├── SKILL.md                      # Skill主文档
├── scripts/
│   └── recommend_model.py        # 模型推荐脚本
├── references/
│   └── external-repos.yaml       # 外部仓库信息
└── repositories/
    ├── rectified-flow-example.py # RectifiedFlow示例
    └── monai-example.py          # MONAI示例
```

## 📝 示例输出 | Example Output

```python
{
    "task": "cross_modality",
    "speed": "real_time",
    "model": "RectifiedFlow",
    "repo": "gnobitab/RectifiedFlow",
    "install": "git clone https://github.com/gnobitab/RectifiedFlow.git",
    "steps": 1,
    "note": "单步推理，适合MRI↔CT实时转换",
    "example_code": "..."
}
```

## 📄 许可证 | License

MIT License

## 👤 作者 | Author

John Yphen - Xi'an Jiaotong University
