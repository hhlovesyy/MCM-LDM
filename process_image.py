import os
from PIL import Image
import torch # 尽管这里不需要 PyTorch，但考虑到您的环境可能有，保险起见

def stitch_png_images(input_dir, output_filename="stitched_image.png", images_per_row=3):
    """
    读取指定目录下的所有 PNG 文件，按顺序将它们拼接成一张大图。

    Args:
        input_dir (str): 包含 PNG 文件的目录路径。
        output_filename (str): 拼接后大图的保存文件名。
        images_per_row (int): 每行放置的图片数量。
    """
    
    # 1. 获取所有 PNG 文件路径并排序
    all_files = [os.path.join(input_dir, f) 
                 for f in os.listdir(input_dir) 
                 if f.endswith('.png') and os.path.isfile(os.path.join(input_dir, f))]
    
    # 按文件名排序，确保顺序正确 (例如: 000.png, 001.png, ...)
    all_files.sort()

    if not all_files:
        print(f"错误：在目录 '{input_dir}' 中未找到任何 .png 文件。")
        return

    print(f"找到 {len(all_files)} 张图片进行拼接。")

    # 2. 打开所有图片并确定尺寸
    images = [Image.open(f) for f in all_files]
    
    # 假设所有图片尺寸相同，使用第一张图的尺寸作为标准
    img_width, img_height = images[0].size
    
    # 3. 计算大图的尺寸
    num_images = len(images)
    num_rows = (num_images + images_per_row - 1) // images_per_row
    
    stitched_width = img_width * images_per_row
    stitched_height = img_height * num_rows
    
    # 创建新的大图画布 (使用RGB模式，避免透明度问题)
    stitched_image = Image.new('RGB', (stitched_width, stitched_height))

    # 4. 逐个粘贴图片到大图上
    for index, img in enumerate(images):
        row = index // images_per_row
        col = index % images_per_row
        
        x_offset = col * img_width
        y_offset = row * img_height
        
        # 确保尺寸匹配
        if img.size != (img_width, img_height):
            print(f"警告：图片 {all_files[index]} 尺寸不匹配，已跳过或调整。")
            # 简单处理：将尺寸调整到标准尺寸
            img = img.resize((img_width, img_height))
            
        stitched_image.paste(img, (x_offset, y_offset))

    # 5. 保存结果
    output_path = os.path.join(input_dir, output_filename)
    stitched_image.save(output_path)
    print(f"\n成功保存拼接图到：{output_path}")

# --- 配置和运行 ---

# ⭐ 请确认这个路径是正确的，这是您的输入目录
IMAGE_DIR = "/root/autodl-tmp/MyRepository/MCM-LDM/vis_debug" 

if __name__ == "__main__":
    stitch_png_images(
        input_dir=IMAGE_DIR,
        output_filename="stitched_debug_image.png",
        images_per_row=4 # 您可以根据需要调整每行显示的图片数量
    )