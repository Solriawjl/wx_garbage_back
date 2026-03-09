import os
import random
from PIL import Image, ImageEnhance, ImageOps

# 设置源路径和目标路径
source_base_dir = r"E:\pycharm\test1\Waste"
target_train_dir = r"E:\pycharm\test1\Waste\train"
target_val_dir = r"E:\pycharm\test1\Waste\val"

# 定义类别映射配置
# 格式为 -> 源文件夹名: (目标文件夹名, 中文类名, train基础抽取数量, train放大倍数, val抽取数量)
config = {
    "FoodWaste": ("0", "厨余垃圾", 1000, 5, 200),
    "RecyclableWaste": ("1", "可回收物", 1000, 5, 200),
    "HarmfulWaste": ("2", "有害垃圾", 1000, 5, 200),
    "OtherWaste": ("3", "其他垃圾", 1000, 7, 200)
}


def random_augment(image):
    """对图像进行随机数据增强 (仅用于训练集)"""
    if random.random() > 0.5:
        image = ImageOps.mirror(image)

    angle = random.uniform(-15, 15)
    image = image.rotate(angle, fillcolor=(255, 255, 255))

    brightness_enhancer = ImageEnhance.Brightness(image)
    image = brightness_enhancer.enhance(random.uniform(0.8, 1.2))

    contrast_enhancer = ImageEnhance.Contrast(image)
    image = contrast_enhancer.enhance(random.uniform(0.8, 1.2))

    return image


def process_split(file_list, src_path, tgt_path, cn_name, copy_times, is_train):
    """
    处理划分好的文件列表（执行增强/复制和重命名）
    """
    global_counter = 1
    for file_name in file_list:
        src_file_path = os.path.join(src_path, file_name)
        try:
            with Image.open(src_file_path) as img:
                img = img.convert('RGB')

                for i in range(copy_times):
                    new_file_name = f"{cn_name}_{global_counter}.jpg"
                    tgt_file_path = os.path.join(tgt_path, new_file_name)

                    # 只有训练集且不是第一张底图时，才进行数据增强
                    if is_train and i > 0:
                        out_img = random_augment(img)
                    else:
                        out_img = img

                    out_img.save(tgt_file_path, quality=95)
                    global_counter += 1

        except Exception as e:
            print(f"❌ 读取或处理图片 {file_name} 时出错: {e}")


def build_datasets():
    os.makedirs(target_train_dir, exist_ok=True)
    os.makedirs(target_val_dir, exist_ok=True)

    for src_folder, (tgt_folder, cn_name, train_size, train_copy_times, val_size) in config.items():
        src_path = os.path.join(source_base_dir, src_folder)
        tgt_train_path = os.path.join(target_train_dir, tgt_folder)
        tgt_val_path = os.path.join(target_val_dir, tgt_folder)

        os.makedirs(tgt_train_path, exist_ok=True)
        os.makedirs(tgt_val_path, exist_ok=True)

        if not os.path.exists(src_path):
            print(f"⚠️ 警告：找不到源文件夹 {src_path}，已跳过。")
            continue

        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        all_files = [f for f in os.listdir(src_path) if f.lower().endswith(valid_extensions)]

        # 1. 随机打乱所有文件列表（非常重要，保证每次抽取的随机性）
        random.shuffle(all_files)

        # 2. 划分训练集
        actual_train_size = min(train_size, len(all_files))
        train_files = all_files[:actual_train_size]

        # 3. 划分验证集 (从训练集挑剩下的里面选，杜绝数据泄露)
        remaining_files = all_files[actual_train_size:]
        actual_val_size = min(val_size, len(remaining_files))
        val_files = remaining_files[:actual_val_size]

        print(f"\n📁 正在处理: {src_folder}")
        print(
            f"   -> 训练集(train\\{tgt_folder}): 抽取 {actual_train_size} 张，扩充 {train_copy_times} 倍，共生成 {actual_train_size * train_copy_times} 张")
        print(f"   -> 验证集(val\\{tgt_folder}): 抽取 {actual_val_size} 张，保持原样，共生成 {actual_val_size} 张")
        if len(remaining_files) < val_size:
            print(f"   💡 提示：剩余图片不足 {val_size} 张，已将剩余的 {actual_val_size} 张全部分配给验证集。")

        # 处理并保存训练集图片
        process_split(train_files, src_path, tgt_train_path, cn_name, copy_times=train_copy_times, is_train=True)

        # 处理并保存验证集图片 (注意 copy_times=1, is_train=False)
        process_split(val_files, src_path, tgt_val_path, cn_name, copy_times=1, is_train=False)

    print("\n✅ Train 和 Val 数据集构建完毕！")


if __name__ == '__main__':
    # 设定随机种子，保证每次运行划分的数据是一样的（方便复现实验结果）
    random.seed(42)
    build_datasets()