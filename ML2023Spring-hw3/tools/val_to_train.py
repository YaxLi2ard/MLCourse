import os
import random
import shutil

def move_random_images(src_dir, dst_dir, num_images):
    """
    从 src_dir 中随机选择 num_images 张图像，移动到 dst_dir。
    """
    src_count = len([f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    dst_count = len([f for f in os.listdir(dst_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    print(f"当前 {src_dir} 中图像数: {src_count}")
    print(f"当前 {dst_dir} 中图像数: {dst_count}")
    
    # 获取所有图像文件
    images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    if len(images) < num_images:
        raise ValueError(f"源文件夹中只有 {len(images)} 张图像，少于请求的 {num_images} 张。")
    # 随机选取
    selected_images = random.sample(images, num_images)
    # 移动文件
    for img in selected_images:
        src_path = os.path.join(src_dir, img)
        dst_path = os.path.join(dst_dir, img)
        # 如果目标路径有重名文件，则重命名（添加 v）
        if os.path.exists(dst_path):
            name, ext = os.path.splitext(img)
            new_name = f"{name}v{ext}"
            new_dst_path = os.path.join(dst_dir, new_name)
            dst_path = new_dst_path  # 使用重命名后的路径

        shutil.move(src_path, dst_path)
    print(f"成功移动了 {num_images} 张图像到 {dst_dir}")
    
    src_count = len([f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    dst_count = len([f for f in os.listdir(dst_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    print(f"当前 {src_dir} 中图像数: {src_count}")
    print(f"当前 {dst_dir} 中图像数: {dst_count}")


if __name__ == "__main__":
    random.seed(999)
    src_dir = '/root/autodl-tmp/MLCourseDataset/ml2023spring-hw3/valid'
    dst_dir = '/root/autodl-tmp/MLCourseDataset/ml2023spring-hw3/train'
    num_images = 2643
    
    move_random_images(src_dir, dst_dir, num_images)