import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 初始化模型，使用新的weights参数
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # 四分类问题

# 设置设备 - 先确定设备类型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载最佳模型
model_path = 'models/best_model.pth'
if os.path.exists(model_path):
    # 关键修改：添加 map_location=device 参数
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print(f"成功加载模型: {model_path}")
else:
    raise FileNotFoundError(f"未找到模型文件 {model_path}")

# 将模型移到设备上
model = model.to(device)
model.eval()
print("模型设置为评估模式")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = model.to(device)
# model.eval()

# 类名映射
class_to_idx = {'week1': 0, 'week2': 1, 'week3': 2, 'week4': 3}
idx_to_class = {v: k for k, v in class_to_idx.items()}

# 读取并预处理图片
def predict_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        # image_tensor = transform(image).unsqueeze(0)  # 添加批次维度

        # 确保输入数据也在正确的设备上
        # image_tensor = image_tensor.to(device)
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
            predicted_class = idx_to_class[preds.item()]

            # 获取置信度
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence = probabilities[0, preds.item()].item()

            return predicted_class, confidence
    except Exception as e:
        print(f"预测时出现错误: {e}")
        return None, 0.0

# 示例：指定图片路径进行预测
image_path = r'D:\python\类器官\DAY1_HUVEC+RSC(29).png'  # 替换为实际的图片路径
predicted_class, confidence = predict_image(image_path)
if predicted_class:
    print(f"该细胞图像属于 {predicted_class} 阶段，置信度: {confidence:.2%}")