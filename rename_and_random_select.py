import os
import shutil
import random
import re


def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split('(\d+)', s)]


base_dir = r"E:\CellAnalysis"
train_dir = os.path.join(base_dir, "train")
valid_dir = os.path.join(base_dir, "valid")

# 创建目标文件夹
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

# 处理每个week文件夹
for week_num in range(1, 5):
    src_folder = os.path.join(base_dir, f"week{week_num}_processed")

    # 获取所有PNG文件并按自然顺序排序
    png_files = [f for f in os.listdir(src_folder) if f.lower().endswith('.png')]
    png_files.sort(key=natural_sort_key)

    # 复制并重命名文件到train目录
    for idx, filename in enumerate(png_files, 1):
        src_path = os.path.join(src_folder, filename)
        new_name = f"week{week_num}_{idx}.png"
        dest_path = os.path.join(train_dir, new_name)
        shutil.copy2(src_path, dest_path)
        print(f"Copied: {src_path} -> {dest_path}")

# 按周划分验证集
for week_num in range(1, 5):
    # 获取当前周的所有图片
    week_files = [f for f in os.listdir(train_dir)
                  if f.startswith(f"week{week_num}_") and f.endswith(".png")]

    # 计算需要移动的数量（向上取整）
    num_total = len(week_files)
    num_to_move = (num_total + 4) // 5  # 相当于math.ceil(num_total/5)

    # 随机选择并移动文件
    files_to_move = random.sample(week_files, num_to_move)
    for filename in files_to_move:
        src = os.path.join(train_dir, filename)
        dest = os.path.join(valid_dir, filename)
        shutil.move(src, dest)
        print(f"Moved to validation: {src} -> {dest}")

print("=" * 50)
print(f"操作完成！共处理 {sum(1 for _ in range(1, 5))} 周数据")
print(f"训练集位置: {train_dir}")
print(f"验证集位置: {valid_dir}")