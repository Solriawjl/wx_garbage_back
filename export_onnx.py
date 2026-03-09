import torch
import torch.nn as nn
import torchvision.models as tv_models
import os

print("🚀 开始转换模型为端侧格式 (ONNX)...")

# 🚀 必须和最新的 train.py 里的结构保持百分之百一致！
def build_inference_model():
    model = tv_models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),      # 🚀 从 1024 改成 512
        nn.BatchNorm1d(512),              # 🚀 这里也改成 512
        nn.Hardswish(),
        nn.Dropout(0.5),                  # 🚀 对应改成 0.5
        nn.Linear(512, 4)                 # 🚀 从 512 输出到 4
    )
    return model

# 2. 实例化模型并加载你的 .pth 权重
model = build_inference_model()

# ... 后面的代码完全保持不变 ...
weights_path = "weights/best_mobilenetv3.pth"
if not os.path.exists(weights_path):
    print(f"❌ 找不到权重文件: {weights_path}，请检查路径！")
    exit()

model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
os.makedirs("weights", exist_ok=True)
onnx_file_path = "weights/mobilenetv3_edge.onnx"

torch.onnx.export(
    model, dummy_input, onnx_file_path,
    export_params=True, opset_version=11, do_constant_folding=True,
    input_names=['input'], output_names=['output'],
)

print(f"✅ 转换大功告成！端侧模型已保存至：{onnx_file_path}")