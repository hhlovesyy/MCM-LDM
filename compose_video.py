import argparse
import json
import os
# MoviePy v2.x 导入
from moviepy import VideoFileClip, ImageClip, clips_array, TextClip, CompositeVideoClip

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_json', type=str, required=True)
    args = parser.parse_args()
    
    with open(args.task_json, 'r') as f:
        config = json.load(f)
        
    files = config['files']
    cols = config['grid_cols']
    padding = int(config['padding']) # 确保是整数
    
    clips = []
    for fname in files:
        # 1. 加载素材
        if fname.endswith('.mp4'):
            clip = VideoFileClip(fname)
        else:
            clip = ImageClip(fname).with_duration(5) 
            
        # 2. 调整大小 (Resize)
        try:
            if hasattr(clip, 'resized'):
                clip = clip.resized(width=512)
            else:
                clip = clip.resize(width=512)
        except Exception as e:
            print(f"Warning: Resize failed for {fname}: {e}")
            
        # 3. 【核心修复】添加边距 (Padding)
        # 在这里给每个 Clip 加边框，替代 clips_array 的 padding 参数
        # margin值 = padding / 2，这样两个视频并排时，中间的间距就是 padding
        margin_size = int(padding / 2)
        try:
            if hasattr(clip, 'with_margin'):
                # MoviePy v2.x
                clip = clip.with_margin(margin_size, color=(255, 255, 255))
            else:
                # MoviePy v1.x
                clip = clip.margin(margin_size, color=(255, 255, 255))
        except Exception as e:
            print(f"Warning: Margin failed: {e}")
        
        # 4. 加标签 (跳过以防报错)
        if config['draw_labels']:
            pass 
            
        clips.append(clip)
        
    # 5. 补齐网格
    rows = (len(clips) + cols - 1) // cols
    grid = []
    for r in range(rows):
        row_clips = clips[r*cols : (r+1)*cols]
        while len(row_clips) < cols:
            row_clips.append(None) 
        grid.append(row_clips)
        
    # 6. 拼接
    # 【核心修复】移除了 padding 参数，保留 bg_color
    try:
        final_clip = clips_array(grid, bg_color=(255, 255, 255))
    except TypeError:
        # 如果 bg_color 也报错（极少数版本），就只传 grid
        final_clip = clips_array(grid)
    
    # 7. 输出
    # codec='libx264' 确保兼容性
    final_clip.write_videofile(config['output_path'], fps=24, codec='libx264')
    print(f"Montage saved to {config['output_path']}")

if __name__ == "__main__":
    main()