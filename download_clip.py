import os
# 强制使用镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import CLIPTokenizer, CLIPTextModel

# 模型名称
model_name = "openai/clip-vit-base-patch32"
# 指定保存到本地的路径 (建议放在项目里的一个文件夹)
local_save_path = "./local_clip_model"

print(f"开始从 {os.environ['HF_ENDPOINT']} 下载模型...")

# 下载 Tokenizer
print("正在下载 Tokenizer...")
tokenizer = CLIPTokenizer.from_pretrained(model_name, force_download=True)  # 强制进行全新下载)
tokenizer.save_pretrained(local_save_path)

# 下载 Model
print("正在下载 Model (约400MB)...")
model = CLIPTextModel.from_pretrained(model_name, force_download=True)
model.save_pretrained(local_save_path)

print(f"✅ 下载完成！模型已保存在: {os.path.abspath(local_save_path)}")