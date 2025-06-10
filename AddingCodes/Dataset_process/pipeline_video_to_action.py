# pipeline_video_to_action.py

import os
import argparse
import subprocess
import datetime
import shutil
import yaml
import torch
import sys
import numpy as np # For step2 config default values

# 获取当前脚本文件所在的目录 (.../Dataset_process/)
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"[DEBUG] Current script directory: {current_dir}") # 调试打印

# 构造 VIBE 文件夹的绝对路径 (.../Dataset_process/VIBE/)
vibe_dir = os.path.join(current_dir, 'VIBE')
print(f"[DEBUG] Calculated VIBE directory: {vibe_dir}") # 调试打印

# 检查 VIBE 目录是否存在
if not os.path.isdir(vibe_dir):
    print(f"[ERROR] The calculated VIBE directory does not exist: {vibe_dir}") # 错误检查

# 将 VIBE 目录添加到 Python 的模块搜索路径列表的最前面
if vibe_dir not in sys.path:
    sys.path.insert(0, vibe_dir)
    print(f"[DEBUG] Successfully added to sys.path. New sys.path[0]: {sys.path[0]}") # 调试打印
else:
    print(f"[DEBUG] VIBE directory already in sys.path.") # 调试打印

# Ensure the current directory is in PATH to find the step scripts
# (Usually not needed if running from the same directory, but good for robustness)
# current_dir = os.path.dirname(os.path.abspath(__file__))
# if current_dir not in sys.path:
#    sys.path.append(current_dir)

# --- Import functions from your scripts ---
# Make sure step2_pkl_to_npz.py and step3_render_npz.py are in the same directory
# or accessible via PYTHONPATH
try:
    # 将当前文件的上上上级目录（即项目根目录 MCM-LDM）添加到 sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # 在 macOS/Linux 上，可以简化为下面的写法，效果一样
    # project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from step2_pkl_to_npz import process_single_vibe_file
    from step3_render_npz import render_amass_to_mp4
except ImportError as e:
    print(f"Error importing helper scripts: {e}")
    print("Make sure step2_pkl_to_npz.py and step3_render_npz.py are in the same directory as this pipeline script.")
    sys.exit(1)

def run_vibe(video_path, base_output_folder, render_vibe_video, vibe_script_path="AddingCodes/Dataset_process/step1_vibe.py"):
    """
    Runs the VIBE script (step1) for a single video.
    """
    video_file_name = os.path.basename(video_path)
    video_base_name = os.path.splitext(video_file_name)[0]
    
    # VIBE creates its own subfolder named after the video inside this output_folder
    vibe_step_output_dir = os.path.join(base_output_folder, "1_vibe_outputs")
    os.makedirs(vibe_step_output_dir, exist_ok=True)

    # The actual pkl file will be in: vibe_step_output_dir/video_base_name/vibe_output.pkl
    expected_pkl_path = os.path.join(vibe_step_output_dir, video_base_name, "vibe_output.pkl")
    # Optional VIBE rendered video path
    expected_vibe_render_path = os.path.join(vibe_step_output_dir, video_base_name, f"{video_base_name}_vibe_result.mp4")

    print(f"  [Step 1] Running VIBE for {video_file_name}...")
    cmd = [
        'python', vibe_script_path,
        '--vid_file', video_path,
        '--output_folder', vibe_step_output_dir, # VIBE will create video_base_name inside this
        # Add any other VIBE arguments you commonly use.
        # For example, if you always use specific batch sizes or tracking:
        # '--tracker_batch_size', '12',
        # '--vibe_batch_size', '450',
        # '--tracking_method', 'bbox', # or 'pose' if STAF is set up
    ]
    if not render_vibe_video:
        cmd.append('--no_render')

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"  [Step 1] VIBE Error for {video_file_name}:")
            print("    STDOUT:", stdout)
            print("    STDERR:", stderr)
            return None, None
        else:
            print(f"  [Step 1] VIBE finished successfully for {video_file_name}.")
            print("    STDOUT:", stdout) # Print VIBE output for user
            if stderr: print("    STDERR:", stderr)


        if not os.path.exists(expected_pkl_path):
            print(f"  [Step 1] Error: VIBE PKL file not found at {expected_pkl_path} after running VIBE.")
            return None, None
        
        vibe_render_output = expected_vibe_render_path if render_vibe_video and os.path.exists(expected_vibe_render_path) else None
        return expected_pkl_path, vibe_render_output

    except FileNotFoundError:
        print(f"  [Step 1] Error: '{vibe_script_path}' not found. Make sure it's in the correct path and Python is callable.")
        return None, None
    except Exception as e:
        print(f"  [Step 1] An unexpected error occurred while running VIBE: {e}")
        return None, None


def convert_pkl_to_npz(vibe_pkl_path, base_output_folder, smpl_models_dir_config):
    """
    Runs the PKL to NPZ conversion (step2) for a single VIBE output.
    """
    if vibe_pkl_path is None:
        return None

    video_base_name = os.path.splitext(os.path.basename(os.path.dirname(vibe_pkl_path)))[0] # Get name from pkl's parent dir
    
    amass_npz_output_dir = os.path.join(base_output_folder, "2_amass_npz")
    os.makedirs(amass_npz_output_dir, exist_ok=True)
    output_npz_path = os.path.join(amass_npz_output_dir, f"{video_base_name}_amass.npz")

    print(f"  [Step 2] Converting {os.path.basename(vibe_pkl_path)} to NPZ format...")

    # Construct the configuration for step2_pkl_to_npz.py
    # This should mirror the structure of its default_config_content or your processing_config.yaml
    config_step2 = {
        'mode': "single", # This will be handled by calling the function directly
        'input_vibe_pkl_path': vibe_pkl_path, # Populated dynamically
        'input_vibe_folder': None, # Not used in single mode
        'output_amass_folder': amass_npz_output_dir, # Populated dynamically (used for plots if any)
        'smpl_models_dir': smpl_models_dir_config, # From pipeline args
        'vibe_person_id_key': 1, # Default for VIBE output (usually int 1, or str '1')
        'output_mocap_framerate': 30.0,
        'output_gender': "neutral",
        'amass_output_pose_params': 72, # Standard AMASS format
        'amass_output_beta_params': 16, # Standard AMASS format (SMPL uses 10, will be padded/truncated)
        'temporal_smoothing': {
            'enable': True,
            'method': "ema", 
            'smooth_body_pose': True,
            'savgol_filter': {
                'global_orient_window': 15, 'global_orient_polyorder': 3,
                'camera_transl_window': 15, 'camera_transl_polyorder': 3,
                'body_pose_window': 9, 'body_pose_polyorder': 2
            },
            'ema_filter': {
                'global_orient_smooth_factor': 0.6, 
                'camera_transl_smooth_factor': 0.6, 
                'body_pose_core_smooth_factor': 0.7,
                'body_pose_limbs_smooth_factor': 0.3,
                'body_pose_ends_smooth_factor': 0.1
            },
            'median_filter': {
                'global_orient_kernel_size': 5,
                'camera_transl_kernel_size': 5,
                'body_pose_kernel_size': 7
            }
        },
        'coordinate_correction': {
            'enable': True,
            'fixed_rotation_euler_xyz_deg': [-90, 0, 0],
            'fixed_rotation_euler_order': 'xyz'
        },
        'camera_to_world': {
            'enable': True, 
            'initial_y_offset_adjustment': 0.0,
            'ground_plane_y': 0.0,
            'default_z_for_pred_cam': 2.5, # Used if 'transl' is missing and 'pred_cam' is used
            'orientation_align_mode': "target_vector", 
            'target_world_forward_vector': [0.0, 1.0, 0.0], # Example: character faces +Y in world
            'path_refinement': {
                'enable': False
            }
        },
        'foot_grounding_ik': {
            'enable': False
        },
        'current_file_id': video_base_name # For logging within process_single_vibe_file
    }

    device_step2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        success = process_single_vibe_file(vibe_pkl_path, output_npz_path, config_step2, device_step2)
        if success and os.path.exists(output_npz_path):
            print(f"  [Step 2] Successfully converted to {output_npz_path}")
            return output_npz_path
        else:
            print(f"  [Step 2] PKL to NPZ conversion failed for {vibe_pkl_path}.")
            return None
    except Exception as e:
        print(f"  [Step 2] An error occurred during PKL to NPZ conversion: {e}")
        # import traceback
        # traceback.print_exc()
        return None


def render_smpl_from_npz(amass_npz_path, base_output_folder, smpl_models_dir_render, video_fps=30):
    """
    Renders an SMPL video from an AMASS NPZ file (step3).
    """
    if amass_npz_path is None:
        return None

    video_base_name = os.path.splitext(os.path.basename(amass_npz_path))[0].replace("_amass", "")
    
    smpl_render_output_dir = os.path.join(base_output_folder, "3_smpl_renders")
    os.makedirs(smpl_render_output_dir, exist_ok=True)
    output_smpl_video_path = os.path.join(smpl_render_output_dir, f"{video_base_name}_smpl_final.mp4")
    
    # Temporary folder for frames, make it unique per video or manage cleanup carefully
    temp_frames_dir = os.path.join(base_output_folder, "temp_smpl_frames", video_base_name)
    # os.makedirs(temp_frames_dir, exist_ok=True) # render_amass_to_mp4 handles this

    print(f"  [Step 3] Rendering SMPL video for {os.path.basename(amass_npz_path)}...")
    
    try:
        render_amass_to_mp4(
            amass_file_path=amass_npz_path,
            smpl_models_dir=smpl_models_dir_render,
            output_video_path=output_smpl_video_path,
            temp_image_folder=temp_frames_dir, # Crucial to be unique or cleaned
            video_fps=video_fps, # Or read from NPZ if desired and step3 supports it
            cleanup_temp_files=True # Very important!
        )
        if os.path.exists(output_smpl_video_path):
            print(f"  [Step 3] Successfully rendered SMPL video to {output_smpl_video_path}")
            return output_smpl_video_path
        else:
            print(f"  [Step 3] SMPL video rendering failed, output file not found.")
            return None
    except Exception as e:
        print(f"  [Step 3] An error occurred during SMPL rendering: {e}")
        # import traceback
        # traceback.print_exc()
        return None
    finally:
        # Ensure temp folder is cleaned up if cleanup_temp_files=False or if an error occurred before cleanup
        if os.path.exists(temp_frames_dir) and os.path.isdir(temp_frames_dir):
             try:
                 shutil.rmtree(temp_frames_dir)
                 print(f"  [Step 3] Cleaned up temporary render folder: {temp_frames_dir}")
             except Exception as e_clean:
                 print(f"  [Step 3] Error cleaning up temp folder {temp_frames_dir}: {e_clean}")


def main():
    parser = argparse.ArgumentParser(description="Pipeline to convert videos to action NPZ files and optionally render.")
    parser.add_argument('--input_folder', type=str, required=True,
                        help="Folder containing input video files (.mp4, .avi, etc.).")
    parser.add_argument('--output_root_folder', type=str, default=".",
                        help="Root directory where the pipeline's main output folder (with timestamp) will be created.")
    parser.add_argument('--smpl_models_dir', type=str, required=True,
                        help="Path to the directory containing SMPL model files (e.g., SMPL_NEUTRAL.pkl).")
    parser.add_argument('--render_vibe', action='store_true',
                        help="Render the VIBE output video (Step 1 visualization).")
    parser.add_argument('--render_smpl', action='store_true',
                        help="Render the final SMPL animation video (Step 3 visualization).")
    parser.add_argument('--vibe_script', type=str, default="/root/autodl-tmp/MyRepository/MCM-LDM/AddingCodes/Dataset_process/step1_vibe.py",
                        help="Path to the VIBE script (default: step1_vibe.py in current dir).")
    parser.add_argument('--target_fps', type=int, default=30,
                        help="Target FPS for the final rendered SMPL video.")

    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' not found.")
        sys.exit(1)
    if not os.path.isdir(args.smpl_models_dir):
        print(f"Error: SMPL models directory '{args.smpl_models_dir}' not found.")
        sys.exit(1)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_output_foldername = f"VideoToAction_Output_{timestamp}"
    main_output_folder = os.path.join(os.path.abspath(args.output_root_folder), pipeline_output_foldername)
    
    try:
        os.makedirs(main_output_folder, exist_ok=True)
        print(f"Pipeline started. Main output directory: {main_output_folder}")
    except OSError as e:
        print(f"Error creating main output directory {main_output_folder}: {e}")
        sys.exit(1)

    video_extensions = ('.mp4', '.avi', '.mov', '.mkv') # Add more if needed
    video_files = [f for f in os.listdir(args.input_folder) if f.lower().endswith(video_extensions)]

    if not video_files:
        print(f"No video files found in '{args.input_folder}'.")
        sys.exit(0)

    print(f"Found {len(video_files)} videos to process: {video_files}")

    for video_file in video_files:
        print(f"\nProcessing video: {video_file} {'-'*30}")
        full_video_path = os.path.join(args.input_folder, video_file)

        # --- Step 1: VIBE ---
        vibe_pkl_output, vibe_render_output_path = run_vibe(
            full_video_path, 
            main_output_folder, 
            args.render_vibe,
            args.vibe_script
        )
        if vibe_pkl_output:
            print(f"  [Step 1] VIBE PKL generated at: {vibe_pkl_output}")
            if vibe_render_output_path:
                print(f"  [Step 1] VIBE Rendered video at: {vibe_render_output_path}")
        else:
            print(f"  [Step 1] VIBE processing failed for {video_file}. Skipping subsequent steps for this video.")
            continue # Skip to the next video

        # --- Step 2: PKL to NPZ ---
        amass_npz_output = convert_pkl_to_npz(vibe_pkl_output, main_output_folder, args.smpl_models_dir)
        if amass_npz_output:
            print(f"  [Step 2] AMASS NPZ generated at: {amass_npz_output}")
        else:
            print(f"  [Step 2] PKL to NPZ conversion failed for {video_file}. Skipping SMPL rendering.")
            continue # Skip to the next video

        # --- Step 3: Render SMPL from NPZ (Optional) ---
        if args.render_smpl:
            smpl_render_video_output = render_smpl_from_npz(
                amass_npz_output, 
                main_output_folder, 
                args.smpl_models_dir,
                video_fps=args.target_fps
            )
            if smpl_render_video_output:
                print(f"  [Step 3] Final SMPL video rendered to: {smpl_render_video_output}")
        
        print(f"Finished processing for: {video_file}\n{'='*50}")

    print(f"\nPipeline finished. All outputs are in: {main_output_folder}")

if __name__ == '__main__':
    main()