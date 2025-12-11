import argparse
import json
import os
# MoviePy v2.x 导入
from moviepy import VideoFileClip, ImageClip, clips_array, TextClip, CompositeVideoClip, ColorClip

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_json', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.task_json, 'r') as f:
        config = json.load(f)
        
    files = config['files']
    cols = config['grid_cols']
    padding = int(config['padding'])
    
    clips = []
    
    # --- 1. 加载素材 ---
    for fname in files:
        if fname.endswith('.mp4'):
            clip = VideoFileClip(fname)
        else:
            clip = ImageClip(fname).with_duration(5) 
            
        # 调整大小
        try:
            target_width = 512
            if hasattr(clip, 'resized'):
                clip = clip.resized(width=target_width)
            else:
                clip = clip.resize(width=target_width)
        except Exception as e:
            print(f"Warning: Resize failed for {fname}: {e}")
            
        # 添加边距 (Margin)
        margin_size = int(padding / 2)
        try:
            # 确保 margin 也是整数
            if hasattr(clip, 'with_margin'):
                clip = clip.with_margin(margin_size, color=(255, 255, 255))
            else:
                clip = clip.margin(margin_size, color=(255, 255, 255))
        except Exception as e:
            print(f"Warning: Margin failed: {e}")
            
        clips.append(clip)

    if not clips:
        print("No clips loaded.")
        return

    # --- 2. 补齐网格 (Fix NoneType Error) ---
    rows = (len(clips) + cols - 1) // cols
    grid = []
    
    # 获取参考尺寸和时长，用于创建空白占位符
    ref_w, ref_h = clips[0].size
    ref_duration = clips[0].duration
    
    for r in range(rows):
        row_clips = clips[r*cols : (r+1)*cols]
        
        # 填充空白
        while len(row_clips) < cols:
            # 创建一个白色的空白片段代替 None
            try:
                # MoviePy v2写法: ColorClip(size=..., color=..., duration=...)
                blank = ColorClip(size=(ref_w, ref_h), color=(255, 255, 255), duration=ref_duration)
            except:
                # 备用写法
                blank = ColorClip(size=(ref_w, ref_h), color=(255, 255, 255)).with_duration(ref_duration)
                
            row_clips.append(blank)
            
        grid.append(row_clips)
        
    # --- 3. 拼接 ---
    try:
        final_clip = clips_array(grid, bg_color=(255, 255, 255))
    except TypeError:
        final_clip = clips_array(grid)
    
    # --- 4. 输出 ---
    # 使用 libx264 编码，fps 24
    output_path = config['output_path']
    final_clip.write_videofile(output_path, fps=24, codec='libx264')
    print(f"Montage saved to {output_path}")

if __name__ == "__main__":
    main()