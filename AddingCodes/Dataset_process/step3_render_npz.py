import numpy as np
import trimesh
from smplx import SMPL
import torch
import os
import shutil
import subprocess
from PIL import Image

# --- Try to set PyOpenGL platform ---
# Order of preference: egl, osmesa, then default (hoping for an X server or other auto-detected backend)
PREFERRED_PYOPENGL_PLATFORMS = ['egl', 'osmesa', None] # None means let PyOpenGL decide

pyrender_imported_successfully = False
for platform_name in PREFERRED_PYOPENGL_PLATFORMS:
    if platform_name:
        print(f"Attempting to set PYOPENGL_PLATFORM='{platform_name}'...")
        os.environ['PYOPENGL_PLATFORM'] = platform_name
    else:
        print("Attempting to import pyrender with default PYOPENGL_PLATFORM...")
        if 'PYOPENGL_PLATFORM' in os.environ: # Clear previous attempt if any
            del os.environ['PYOPENGL_PLATFORM']
    
    try:
        import pyrender # Try importing here, after setting env var
        from OpenGL import GL # Also try importing GL to see if basic OpenGL context is possible
        print(f"Successfully imported pyrender (and OpenGL.GL) with PYOPENGL_PLATFORM='{os.environ.get('PYOPENGL_PLATFORM', 'Default')}'")
        pyrender_imported_successfully = True
        break # Success, stop trying other platforms
    except Exception as e:
        print(f"Failed to import pyrender or OpenGL.GL with PYOPENGL_PLATFORM='{os.environ.get('PYOPENGL_PLATFORM', 'Default')}': {e}")
        if platform_name == PREFERRED_PYOPENGL_PLATFORMS[-1]: # If this was the last attempt
            print("--------------------------------------------------------------------")
            print("FATAL: Could not initialize PyOpenGL/pyrender with any preferred platform.")
            print("Please ensure you have a working OpenGL environment and the necessary backends:")
            print("  - For EGL: NVIDIA drivers (or MESA EGL) and libegl1-mesa-dev (or equivalent).")
            print("  - For OSMesa: libosmesa6-dev (or equivalent).")
            print("  - For X11: A running X server (e.g., Xvfb for headless).")
            print("--------------------------------------------------------------------")
            exit()

if not pyrender_imported_successfully:
    # This should not be reached if exit() above works, but as a safeguard:
    print("Critical error: pyrender could not be imported. Exiting.")
    exit()


# ... (rest of your render_amass_to_mp4 function, starting from its definition) ...
# Make sure the render_amass_to_mp4 function is defined after the import block.

def render_amass_to_mp4(
    amass_file_path,
    smpl_models_dir,
    output_video_path="output_render.mp4",
    temp_image_folder="temp_render_frames",
    img_width=1280,
    img_height=720,
    video_fps=None, # If None, use mocap_framerate from AMASS file
    cleanup_temp_files=True,
    camera_distance=3.0,
    camera_height=1.5,
    camera_fov_y=np.pi / 3.0,
    light_intensity=2.0,
    bg_color=(0.1, 0.1, 0.3, 1.0), 
    ground_plane=True
):
    # --- 1. Load AMASS Data ---
    # (Same as before)
    if not os.path.exists(amass_file_path):
        print(f"Error: AMASS file not found at {amass_file_path}")
        return
    
    try:
        data = np.load(amass_file_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading AMASS file {amass_file_path}: {e}")
        return

    poses = data['poses']
    betas = data['betas'] # This is where the warning "using a SMPL model, with only 10 shape coefficients" comes from. It's fine.
    trans = data.get('trans', None)
    gender = str(data.get('gender', 'neutral')).lower()
    mocap_framerate = data.get('mocap_framerate', 30.0)

    if video_fps is None:
        video_fps = mocap_framerate

    num_frames = poses.shape[0]

    print(f"Loaded AMASS data: {os.path.basename(amass_file_path)}")
    print(f"  Frames: {num_frames}, Gender: {gender}, Mocap FPS: {mocap_framerate}, Output Video FPS: {video_fps}")

    # --- 2. Prepare SMPL Model and Data for PyTorch ---
    # (Same as before)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    global_orient = torch.tensor(poses[:, :3], dtype=torch.float32).to(device)
    body_pose = torch.tensor(poses[:, 3:72], dtype=torch.float32).to(device)
    # Ensure we only use 10 betas for standard SMPL, even if more are provided in AMASS
    betas_torch = torch.tensor(betas[:10], dtype=torch.float32).unsqueeze(0).to(device) 
    
    # Inside render_amass_to_mp4, when preparing trans_torch:
    if trans is not None:
        trans_torch = torch.tensor(trans, dtype=torch.float32).to(device)
        
        y_offset_adjustment = 2.5 # Try values like 0.5, 1.0, or based on T-pose height
        trans_torch[:, 1] += y_offset_adjustment 
        print(f"DEBUG: Manually added {y_offset_adjustment} to Y translations.")
        # After this adjustment, re-print the first frame's vertex Y range if you want to verify.
    else:
        trans_torch = torch.zeros(num_frames, 3, dtype=torch.float32).to(device)
        # If trans was None, you'd still need to lift the T-pose if it's also low
        # trans_torch[:, 1] += 1.0 # Example if character at origin T-pose is centered around Y=-1
        print("Warning: 'trans' data not found. Visualizing motion in place.")

    if gender == 'male':
        smpl_model_path = os.path.join(smpl_models_dir, 'smpl/SMPL_MALE.pkl')
    elif gender == 'female':
        smpl_model_path = os.path.join(smpl_models_dir, 'smpl/SMPL_FEMALE.pkl')
    else:
        smpl_model_path = os.path.join(smpl_models_dir, 'smpl/SMPL_NEUTRAL.pkl')
        if gender != 'neutral': print(f"Warning: Gender '{gender}' not recognized. Using SMPL_NEUTRAL.pkl.")
    
    if not os.path.exists(smpl_model_path):
        print(f"Error: SMPL model file not found at {smpl_model_path}")
        return

    try:
        # Explicitly pass num_betas=10 to SMPL constructor to match betas_torch
        smpl = SMPL(model_path=smpl_model_path, num_betas=10, batch_size=1).to(device)
    except Exception as e:
        print(f"Error initializing SMPL model from {smpl_model_path}: {e}")
        return

    # --- 3. Setup Pyrender Scene and Offscreen Renderer ---
    # (Scene setup is mostly the same)
    scene = pyrender.Scene(bg_color=bg_color, ambient_light=[0.3, 0.3, 0.3, 1.0])
    
    smpl_output_initial = smpl(global_orient=global_orient[0:1], body_pose=body_pose[0:1], betas=betas_torch, transl=trans_torch[0:1])
    initial_vertices = smpl_output_initial.vertices.detach().cpu().numpy().squeeze()
    smpl_faces = smpl.faces

    human_mesh_trimesh = trimesh.Trimesh(vertices=initial_vertices, faces=smpl_faces, process=False)
    mesh_node_render = pyrender.Mesh.from_trimesh(human_mesh_trimesh, smooth=True)
    scene_human_node = scene.add(mesh_node_render)

    # Camera
    camera = pyrender.PerspectiveCamera(yfov=camera_fov_y, aspectRatio=float(img_width)/img_height)
    

    # New desired camera parameters (tweak these)
    # 修改相机参数部分
    new_camera_y_position = 1.5  # 相机高度大约在角色眼睛位置
    new_camera_z_distance = 2.5  # 相机与角色之间的距离
    look_at_y_offset = 0.6       # 视线稍微向上看一点，看向角色头部

    character_center_target = np.array([
        trans_torch[0,0].item(), 
        trans_torch[0,1].item() + look_at_y_offset, 
        trans_torch[0,2].item()  
    ])

    # 角色的大致高度（假设角色站立时高度≈1.8）
    character_height = 1.8  

    # 相机位置：放在角色正前方（Z轴负方向），高度≈1.5（平视）
    camera_distance = 2  # 相机与角色的距离
    camera_height = 3    # 相机高度（平视）

    # 计算相机位置
    camera_world_position = np.array([
        trans_torch[0, 0].item(),          # X: 与角色对齐
        trans_torch[0, 1].item() - camera_height,  # Y: 相机高度
        trans_torch[0, 2].item() + camera_distance  # Z: 相机在角色前方
    ])

    # 计算目标点（看向角色的头部）
    look_at_target = np.array([
        trans_torch[0, 0].item(),          # X: 与角色对齐
        trans_torch[0, 1].item() + 0.9,    # Y: 看向头部（眼睛位置）
        trans_torch[0, 2].item()           # Z: 角色位置
    ])

    # 计算相机朝向（从相机位置看向目标点）
    look_direction = look_at_target - camera_world_position
    look_direction = look_direction / np.linalg.norm(look_direction)  # 归一化

    # 计算相机的坐标系
    world_up = np.array([0.0, 1.0, 0.0])  # 世界坐标系的上方向（Y轴）
    Z_cam_world = -look_direction
    X_cam_world = np.cross(world_up, Z_cam_world)
    X_cam_world = X_cam_world / np.linalg.norm(X_cam_world)
    Y_cam_world = np.cross(Z_cam_world, X_cam_world)

    # 构造相机姿态矩阵
    camera_pose = np.eye(4)
    camera_pose[:3, 0] = X_cam_world  # 相机的右方向
    camera_pose[:3, 1] = Y_cam_world  # 相机的上方向
    camera_pose[:3, 2] = Z_cam_world  # 相机的朝向
    camera_pose[:3, 3] = camera_world_position  # 相机位置
    
    print(f"\n--- REVISED CAMERA DEBUG INFO ---")
    print(f"Character target (approx): {character_center_target}")
    print(f"Camera world position: {camera_world_position}")
    print(f"Camera Look Direction (world): {look_direction}") # -Z_cam_world
    print(f"Camera X_axis (world): {X_cam_world}")
    print(f"Camera Y_axis (world): {Y_cam_world}")
    print(f"Camera Z_axis (world): {Z_cam_world}")
    print("--- END REVISED CAMERA DEBUG INFO ---\n")

    scene.add(camera, pose=camera_pose)

    # Light setup (should also be adjusted if camera moves significantly)
    # For simplicity, let's place one light slightly offset from the camera
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=light_intensity)
    
    # Create a pose for the light, e.g., slightly above and to the right of camera, pointing similarly
    light_camera_offset = np.array([0.5, 0.5, -0.5, 0.0]) # Offset in camera's local space (X,Y,Z,W)
    light_world_offset = camera_pose @ light_camera_offset
    
    light_pose = np.eye(4)
    light_pose[:3, :3] = camera_pose[:3, :3] # Align light orientation with camera
    light_pose[:3, 3] = camera_world_position + light_world_offset[:3] # Position light relative to camera
    scene.add(light, pose=light_pose)
    
    point_light = pyrender.PointLight(color=[0.8,0.8,0.8], intensity=100.0)
    point_light_pose = np.eye(4)
    point_light_pose[:3,3] = np.array([0, character_center_target[1] + 0.5, character_center_target[2] + 1.0]) # Position relative to character
    scene.add(point_light, pose=point_light_pose)
    
    if ground_plane:
        ground_trimesh = trimesh.creation.box(extents=(10.0, 0.02, 10.0))
        ground_trimesh.visual.vertex_colors = [150, 150, 150, 255]
        ground_mesh_render = pyrender.Mesh.from_trimesh(ground_trimesh)
        ground_pose = np.eye(4)
        ground_pose[1, 3] = -0.01
        # note: 移除地面，查看效果
        # scene.add(ground_mesh_render, pose=ground_pose)

    # Offscreen Renderer - this is where the error occurs
    renderer = None # Initialize to None
    try:
        print(f"Attempting to create OffscreenRenderer with viewport {img_width}x{img_height} using platform: {os.environ.get('PYOPENGL_PLATFORM', 'Default')}")
        renderer = pyrender.OffscreenRenderer(viewport_width=img_width, viewport_height=img_height)
        print("OffscreenRenderer created successfully.")
    except Exception as e:
        print(f"---------------- FAILED TO CREATE OFFSCREEN RENDERER ----------------")
        print(f"Error: {e}")
        print(f"Current PYOPENGL_PLATFORM: {os.environ.get('PYOPENGL_PLATFORM', 'Not set or Default')}")
        print("This often means pyrender could not initialize an OpenGL context.")
        print("Troubleshooting steps:")
        print("  1. If on a headless server, ensure OSMesa (libosmesa6-dev) or EGL (NVIDIA drivers + EGL libs) is correctly installed.")
        print("  2. Try running 'glxinfo' or 'nvidia-smi' to check GPU/driver status (if using GPU/EGL).")
        print("  3. Ensure PyOpenGL can find the backend libraries.")
        print("  4. If using Xvfb, make sure it's running and DISPLAY environment variable is set correctly before running the script.")
        print("--------------------------------------------------------------------")
        return # Exit the function if renderer creation fails

    # --- 4. Render Frames to Temporary Folder ---
    # (Same as before)
    if os.path.exists(temp_image_folder):
        shutil.rmtree(temp_image_folder)
    os.makedirs(temp_image_folder, exist_ok=True)
    print(f"Rendering {num_frames} frames to {temp_image_folder}...")

    # --- Debug: Print initial SMPL output and trans ---
    smpl_output_debug = smpl(
        global_orient=global_orient[0:1], 
        body_pose=body_pose[0:1], 
        betas=betas_torch, 
        transl=trans_torch[0:1] # Use the first frame's translation
    )
    initial_vertices_debug = smpl_output_debug.vertices.detach().cpu().numpy().squeeze()
    initial_joint_locs_debug = smpl_output_debug.joints.detach().cpu().numpy().squeeze() # Get joint locations too

    print("\n--- DEBUG INFO FOR FIRST FRAME ---")
    print(f"First frame trans_torch: {trans_torch[0].cpu().numpy()}")
    print(f"Initial SMPL Root Joint (approx): {initial_joint_locs_debug[0]}") # SMPL root joint
    print(f"Initial Vertices min values (x,y,z): {np.min(initial_vertices_debug, axis=0)}")
    print(f"Initial Vertices max values (x,y,z): {np.max(initial_vertices_debug, axis=0)}")
    print(f"Initial Vertices mean values (x,y,z): {np.mean(initial_vertices_debug, axis=0)}")

    camera_pos_world = camera_pose[:3, 3]
    print(f"Camera position in world: {camera_pos_world}")
    # You can also calculate the direction the camera is looking if needed.
    # Camera Z-axis in world frame (points from object to camera if using standard view matrix)
    # Or from camera towards object if using standard camera matrix (like OpenGL's modelview)
    # Pyrender camera pose is T_world_camera. So Z axis of camera in world is camera_pose[:3, 2]
    # The camera looks along its -Z axis.
    camera_look_direction_world = -camera_pose[:3, 2] 
    print(f"Camera look direction in world: {camera_look_direction_world}")
    print("--- END DEBUG INFO ---\n")


    for i in range(num_frames):
        smpl_output = smpl(
            global_orient=global_orient[i:i+1],
            body_pose=body_pose[i:i+1],
            betas=betas_torch,
            transl=trans_torch[i:i+1]
        )
        new_vertices = smpl_output.vertices.detach().cpu().numpy().squeeze()
        
        human_mesh_trimesh.vertices = new_vertices
        scene_human_node.mesh = pyrender.Mesh.from_trimesh(human_mesh_trimesh, smooth=True)

        color_img, depth_img = renderer.render(scene)
        img_path = os.path.join(temp_image_folder, f"frame_{i:05d}.png")
        Image.fromarray(color_img, 'RGB').save(img_path)

        if (i + 1) % 50 == 0 or i == num_frames - 1:
            print(f"  Rendered frame {i+1}/{num_frames}")
    
    print("Finished rendering frames.")

    # --- 5. Compile Images to MP4 using ffmpeg ---
    # (Same as before)
    print(f"Compiling video to {output_video_path} at {video_fps} FPS...")
    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
    ffmpeg_cmd = [
        'ffmpeg', '-r', str(video_fps), 
        '-i', os.path.join(temp_image_folder, 'frame_%05d.png'),
        '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-y', output_video_path
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Video successfully saved to {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg execution: {e.stderr.decode() if e.stderr else 'N/A'}")
        return
    except FileNotFoundError:
        print("Error: ffmpeg command not found.")
        return

    # --- 6. Cleanup Temporary Files (Optional) ---
    # (Same as before)
    if cleanup_temp_files:
        print(f"Cleaning up temporary image folder: {temp_image_folder}")
        shutil.rmtree(temp_image_folder)
        print("Cleanup complete.")
    
    if renderer: # Check if renderer was successfully created before trying to delete
        renderer.delete()
        print("Renderer resources released.")


if __name__ == '__main__':
    # --- Configuration ---
    # amass_data_file = '/root/autodl-tmp/HumanML3D/HumanML3D/HumanML3D/amass_data/BMLmovi/BMLmovi/Subject_1_F_MoSh/Subject_1_F_1_poses.npz'
    # amass_data_file = '/root/autodl-tmp/MCM-LDM/dummy_vibe_to_amass.npz'
    # amass_data_file = '/root/autodl-tmp/MCM-LDM/amass_converted_output/testVibeOutput_amass_format.npz'
    # amass_data_file = '/root/autodl-tmp/MCM-LDM/amass_converted_output/vibe_output1630_amass_format.npz'
    amass_data_file = '/root/autodl-tmp/MCM-LDM/processed_amass_data/testhot_amass.npz'
    smpl_model_base_dir = '/root/autodl-tmp/MCM-LDM/deps/smpl_models' 
    output_video_file = 'genshinHot.mp4'
    temp_frames_dir = 'temp_video_frames_soma_clap'

    if not os.path.exists(amass_data_file) and "autodl-tmp" in amass_data_file:
        print(f"Default AMASS file {amass_data_file} not found.")
    else:
        render_amass_to_mp4(
            amass_file_path=amass_data_file,
            smpl_models_dir=smpl_model_base_dir,
            output_video_path=output_video_file,
            temp_image_folder=temp_frames_dir,
            img_width=1280,
            img_height=720,
            video_fps=30,
            cleanup_temp_files=True
        )