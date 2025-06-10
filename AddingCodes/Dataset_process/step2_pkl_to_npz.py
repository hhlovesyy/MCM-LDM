import yaml
import os
# import pickle
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R_scipy
from scipy.signal import savgol_filter, medfilt
from smplx import SMPL
from glob import glob
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Constants ---
SMPL_JOINT_MAPPER = { # Standard SMPL joint indices (0-23, pelvis is 0)
    'pelvis': 0, 'left_hip': 1, 'right_hip': 2, 'spine1': 3, 'left_knee': 4, 'right_knee': 5,
    'spine2': 6, 'left_ankle': 7, 'right_ankle': 8, 'spine3': 9, 'left_foot': 10, 'right_foot': 11,
    'neck': 12, 'left_collar': 13, 'right_collar': 14, 'head': 15, 'left_shoulder': 16,
    'right_shoulder': 17, 'left_elbow': 18, 'right_elbow': 19, 'left_wrist': 20, 'right_wrist': 21,
    # SMPL model has 22 high-DOF joints + root + pelvis = 24. body_pose is for 23 joints (excluding root).
}

# SMPL body_pose parameter indices (0-68 for 23 joints)
# This mapping assumes standard SMPL joint order for the 23 pose parameters.
# (global_orient is separate, body_pose starts with joint 1 from SMPL_JOINT_MAPPER)
BODY_POSE_JOINT_NAMES = [ # Order for the 23 joints in SMPL's body_pose
    'left_hip', 'right_hip', 'spine1', 
    'left_knee', 'right_knee', 'spine2', 
    'left_ankle', 'right_ankle', 'spine3', 
    'left_foot', 'right_foot', 
    'neck', 
    'left_collar', 'right_collar', 'head', 
    'left_shoulder', 'right_shoulder', 
    'left_elbow', 'right_elbow', 
    'left_wrist', 'right_wrist',
    'left_hand', 'right_hand' # These are often zero for basic SMPL from VIBE
]
# Validate that BODY_POSE_JOINT_NAMES has 23 joints
if len(BODY_POSE_JOINT_NAMES) != 23:
    raise ValueError("BODY_POSE_JOINT_NAMES must contain 23 joint names for SMPL body_pose.")

def get_body_pose_param_indices(joint_names_list):
    indices = []
    for name in joint_names_list:
        try:
            # Find the index of the joint in the body_pose sequence
            pose_idx = BODY_POSE_JOINT_NAMES.index(name)
            indices.extend(range(pose_idx * 3, pose_idx * 3 + 3))
        except ValueError:
            # This can happen if a joint name in your list is not in BODY_POSE_JOINT_NAMES
            # Or if your SMPL_JOINT_MAPPER is used incorrectly here.
            # For body_pose, we directly use its own joint order.
            print(f"[Warning] Joint name '{name}' not found in BODY_POSE_JOINT_NAMES for param index lookup.")
            pass # Or raise an error if this should not happen
    return sorted(list(set(indices))) # Sorted unique indices

# --- Helper Functions (rodrigues, quat conversions, log, etc.) ---
# (Keep your rodrigues_torch, rotation_matrix_to_axis_angle_torch,
# axis_angle_to_quaternion_scipy, quaternion_to_axis_angle_scipy, normalize_quaternions_torch, log_message)
# ... (These functions are assumed to be correct as provided before) ...
def rodrigues_torch(rvecs):
    if rvecs.ndim == 1: rvecs = rvecs.unsqueeze(0)
    if rvecs.shape[0] == 0: return torch.empty((0,3,3), device=rvecs.device, dtype=rvecs.dtype)
    angle = torch.norm(rvecs + 1e-8, dim=1, keepdim=True)
    axis = rvecs / angle
    angle = angle * 0.5
    v_sin, v_cos = torch.sin(angle), torch.cos(angle)
    qw, qx, qy, qz = v_cos, v_sin * axis[:, 0:1], v_sin * axis[:, 1:2], v_sin * axis[:, 2:3]
    q = torch.cat([qw, qx, qy, qz], dim=1)
    q0, q1, q2, q3 = q[:,0], q[:,1], q[:,2], q[:,3]
    r00 = 2*(q0*q0+q1*q1)-1; r01 = 2*(q1*q2-q0*q3); r02 = 2*(q1*q3+q0*q2)
    r10 = 2*(q1*q2+q0*q3); r11 = 2*(q0*q0+q2*q2)-1; r12 = 2*(q2*q3-q0*q1)
    r20 = 2*(q1*q3-q0*q2); r21 = 2*(q2*q3+q0*q1); r22 = 2*(q0*q0+q3*q3)-1
    return torch.stack((r00,r01,r02,r10,r11,r12,r20,r21,r22),dim=1).view(-1,3,3)

def rotation_matrix_to_axis_angle_torch(rot_mats):
    if rot_mats.ndim == 2: rot_mats = rot_mats.unsqueeze(0)
    if rot_mats.shape[0] == 0: return torch.empty((0,3), device=rot_mats.device, dtype=rot_mats.dtype)
    rot_mats_np = rot_mats.detach().cpu().numpy()
    try:
        axis_angle_np = R_scipy.from_matrix(rot_mats_np).as_rotvec()
    except ValueError as e:
        print(f"Scipy R_scipy.from_matrix error: {e}. Input matrix might not be a valid rotation matrix.")
        print("Input matrices (first few if many):", rot_mats_np[:min(3, rot_mats_np.shape[0])])
        print("Returning zero vector as fallback for axis-angle.")
        return torch.zeros((rot_mats.shape[0], 3), dtype=rot_mats.dtype, device=rot_mats.device)
    return torch.tensor(axis_angle_np, dtype=rot_mats.dtype, device=rot_mats.device)

def log_message(message, config, level="INFO"):
    file_id = config.get('current_file_id', 'Log')
    print(f"[{level}][{file_id}] {message}")

def axis_angle_to_quaternion_scipy(aa_tensor):
    if aa_tensor.shape[0] == 0: return torch.empty((0, 4), device=aa_tensor.device, dtype=aa_tensor.dtype)
    aa_np = aa_tensor.detach().cpu().numpy()
    quats_np = R_scipy.from_rotvec(aa_np).as_quat() 
    return torch.tensor(quats_np, device=aa_tensor.device, dtype=aa_tensor.dtype)

def quaternion_to_axis_angle_scipy(quat_tensor):
    if quat_tensor.shape[0] == 0: return torch.empty((0, 3), device=quat_tensor.device, dtype=quat_tensor.dtype)
    quats_np = quat_tensor.detach().cpu().numpy()
    aa_np = R_scipy.from_quat(quats_np).as_rotvec()
    return torch.tensor(aa_np, device=quat_tensor.device, dtype=quat_tensor.dtype)

def normalize_quaternions_torch(quats_tensor):
    if quats_tensor.shape[0] == 0: return quats_tensor
    return quats_tensor / (torch.norm(quats_tensor, p=2, dim=1, keepdim=True) + 1e-8) # Add epsilon for stability


def process_single_vibe_file(vibe_pkl_path, output_npz_path, config, device):
    # ... (Initial loading and person_id key logic - keep as is from your working version) ...
    # (Fuller data extraction logic from your last complete version)
    log_message(f"Starting processing for: {vibe_pkl_path}", config)
    if not os.path.exists(vibe_pkl_path):
        log_message(f"VIBE input file not found: {vibe_pkl_path}", config, "ERROR"); return False
    try:
        vibe_output_all_persons = joblib.load(vibe_pkl_path)
    except Exception as e:
        log_message(f"Error loading PKL: {e}", config, "ERROR"); return False
    person_id_key_config = config['vibe_person_id_key'] 
    person_id_key_to_use = None
    available_keys = list(vibe_output_all_persons.keys())
    if person_id_key_config in available_keys: person_id_key_to_use = person_id_key_config
    elif isinstance(person_id_key_config, int) and str(person_id_key_config) in available_keys: person_id_key_to_use = str(person_id_key_config)
    elif isinstance(person_id_key_config, str) and person_id_key_config.isdigit() and int(person_id_key_config) in available_keys: person_id_key_to_use = int(person_id_key_config)
    if person_id_key_to_use is None:
        log_message(f"Person ID key '{person_id_key_config}' not found. Avail: {available_keys}", config, "ERROR"); return False
    vibe_data_person = vibe_output_all_persons[person_id_key_to_use]
    log_message(f"Using person ID '{person_id_key_to_use}'. Keys: {list(vibe_data_person.keys())}", config)

    # ----- 1. Initial Data Extraction from VIBE -----
    global_orient_aa_cam, body_pose_aa_cam, transl_cam_orig, betas_orig = None, None, None, None
    param_keys_found = False

    if 'pred_smpl_params' in vibe_data_person:
        log_message("Found 'pred_smpl_params' key. Attempting to extract SMPL parameters from it.", config)
        smpl_params = vibe_data_person['pred_smpl_params']
        # log_message(f"  Keys within 'pred_smpl_params': {list(smpl_params.keys())}", config) # Already in your log
        req_keys = ['global_orient', 'body_pose', 'transl', 'betas']
        missing_sub_keys = [k for k in req_keys if k not in smpl_params]

        if not missing_sub_keys:
            try:
                global_orient_aa_cam = torch.tensor(smpl_params['global_orient'], dtype=torch.float32).to(device)
                body_pose_aa_cam = torch.tensor(smpl_params['body_pose'], dtype=torch.float32).to(device)
                transl_cam_orig = torch.tensor(smpl_params['transl'], dtype=torch.float32).to(device)
                betas_orig = torch.tensor(smpl_params['betas'], dtype=torch.float32).to(device)
                param_keys_found = True
                log_message("  Successfully extracted global_orient, body_pose, transl, betas from 'pred_smpl_params'.", config)
                # log_message(f"    Shapes: global_orient={global_orient_aa_cam.shape}, body_pose={body_pose_aa_cam.shape}, transl={transl_cam_orig.shape}, betas={betas_orig.shape}", config)
            except Exception as e:
                log_message(f"  Error converting data from 'pred_smpl_params' to tensors: {e}", config, "ERROR")
        else:
             log_message(f"  'pred_smpl_params' is missing sub-keys: {missing_sub_keys}. Required: {req_keys}", config, "ERROR")
    
    if not param_keys_found and all(k in vibe_data_person for k in ['pose', 'betas', 'transl']): # Check for 'transl' explicitly
        log_message("Did not find/use 'pred_smpl_params'. Found 'pose', 'betas', 'transl' keys. Attempting to use them.", config)
        try:
            pose_aa_vibe = torch.tensor(vibe_data_person['pose'], dtype=torch.float32).to(device)
            # log_message(f"  'pose' shape: {pose_aa_vibe.shape}", config)
            if pose_aa_vibe.shape[1] == 72:
                global_orient_aa_cam = pose_aa_vibe[:, :3]
                body_pose_aa_cam = pose_aa_vibe[:, 3:72] 
                transl_cam_orig = torch.tensor(vibe_data_person['transl'], dtype=torch.float32).to(device)
                betas_orig = torch.tensor(vibe_data_person['betas'], dtype=torch.float32).to(device)
                param_keys_found = True
                log_message("  Successfully extracted from 'pose', 'betas', 'transl'.", config)
            else:
                log_message(f"  'pose' field has unexpected shape {pose_aa_vibe.shape}, expected (N, 72)", config, "ERROR")
        except Exception as e:
            log_message(f"  Error converting data from 'pose', 'betas', 'transl' to tensors: {e}", config, "ERROR")

    # THIS BLOCK IS LIKELY WHAT YOU NEED if 'transl' is missing but 'pred_cam' is present
    if not param_keys_found and all(k in vibe_data_person for k in ['pose', 'betas', 'pred_cam']):
        log_message("Did not find/use previous structures. Found 'pose', 'betas', 'pred_cam' keys. Attempting to use them.", config)
        try:
            pose_aa_vibe = torch.tensor(vibe_data_person['pose'], dtype=torch.float32).to(device)
            # log_message(f"  'pose' shape: {pose_aa_vibe.shape}", config)
            if pose_aa_vibe.shape[1] == 72:
                global_orient_aa_cam = pose_aa_vibe[:, :3]
                body_pose_aa_cam = pose_aa_vibe[:, 3:72]
                pred_cam_vibe = torch.tensor(vibe_data_person['pred_cam'], dtype=torch.float32).to(device)
                
                if 'transl' in vibe_data_person: # Check again just in case
                     transl_cam_orig = torch.tensor(vibe_data_person['transl'], dtype=torch.float32).to(device)
                     log_message("  Used 'transl' from VIBE data alongside 'pred_cam'.", config)
                else:
                    log_message("  'transl' not in VIBE data, using 'pred_cam' (s, tx, ty) for XY and default Z.", config, "INFO") # Changed to INFO
                    transl_cam_orig = torch.zeros(pose_aa_vibe.shape[0], 3, device=device, dtype=torch.float32)
                    transl_cam_orig[:, 0] = pred_cam_vibe[:, 1] # tx
                    transl_cam_orig[:, 1] = pred_cam_vibe[:, 2] # ty
                    default_z = config.get('camera_to_world', {}).get('default_z_for_pred_cam', 2.5)
                    transl_cam_orig[:, 2] = default_z 
                    log_message(f"    Estimated transl XY from pred_cam, Z set to default: {default_z}", config)
                
                betas_orig = torch.tensor(vibe_data_person['betas'], dtype=torch.float32).to(device)
                param_keys_found = True
                log_message("  Successfully extracted from 'pose', 'betas', 'pred_cam' (and possibly 'transl').", config)
            else:
                log_message(f"  'pose' field in pkl has unexpected shape {pose_aa_vibe.shape}, expected (N, 72)", config, "ERROR")
        except Exception as e:
            log_message(f"  Error converting data from 'pose', 'betas', 'pred_cam' to tensors: {e}", config, "ERROR")
    
    # Fallback: if 'transl' is not found by any means, but 'joints3d' exists (like in 1.0)
    # This is less ideal as joints3d are already in a specific pose, but better than nothing if transl is truly missing.
    if not param_keys_found and global_orient_aa_cam is not None and body_pose_aa_cam is not None and betas_orig is not None and transl_cam_orig is None and 'joints3d' in vibe_data_person:
        log_message("  'transl' parameter not found directly or via 'pred_cam'. Attempting to use root joint from 'joints3d' as fallback for translation.", config, "WARNING")
        try:
            vibe_joints3d = torch.tensor(vibe_data_person['joints3d'], dtype=torch.float32).to(device) # N, 49, 3 or N, 24, 3
            # Determine root joint index from 'joints3d'. SMPL pelvis is often used.
            # VIBE's 'joints3d' might be OpenPose format (49 joints) or SMPL (24 joints)
            # If it's from SPIN-like output (49 joints), 'OP MidHip' (index 8) is a good root.
            # If it's direct SMPL joints3d (24 joints), 'pelvis' (index 0) is the root.
            root_idx_joints3d = 0 # Default to pelvis if it's SMPL joints3d
            if vibe_joints3d.shape[1] == 49: # Likely SPIN/OpenPose format
                root_idx_joints3d = 8 # OP MidHip
                log_message("    Using 'OP MidHip' (idx 8) from 49 joints3d as root for translation.", config)
            elif vibe_joints3d.shape[1] == 24: # Likely SMPL joints3d
                root_idx_joints3d = 0 # Pelvis
                log_message("    Using 'pelvis' (idx 0) from 24 joints3d as root for translation.", config)
            else:
                log_message(f"    'joints3d' has unexpected shape {vibe_joints3d.shape}. Cannot determine root for translation. Using zeros.", config, "ERROR")
                transl_cam_orig = torch.zeros(global_orient_aa_cam.shape[0], 3, device=device, dtype=torch.float32)
            
            if vibe_joints3d.shape[1] in [24, 49]:
                 transl_cam_orig = vibe_joints3d[:, root_idx_joints3d, :]
                 param_keys_found = True # We have all params now
                 log_message("    Successfully used root from 'joints3d' for 'transl_cam_orig'.", config)

        except Exception as e:
            log_message(f"  Error processing 'joints3d' as fallback for translation: {e}", config, "ERROR")


    if not param_keys_found:
        log_message(f"Could not find required SMPL parameters in any of the expected structures. Check VIBE PKL content.", config, "ERROR")
        return False
    # ... rest of the function (num_frames check, betas_static processing etc.)
    num_frames = global_orient_aa_cam.shape[0]
    if num_frames == 0: log_message("No frames.", config, "ERROR"); return False
    log_message(f"Extracted {num_frames} frames.", config)
    if betas_orig.ndim == 2 and betas_orig.shape[0] == num_frames: betas_static = betas_orig.mean(dim=0)
    elif betas_orig.ndim == 2 and betas_orig.shape[0] == 1: betas_static = betas_orig.squeeze(0)
    elif betas_orig.ndim == 1 and betas_orig.shape[0] >= 10: betas_static = betas_orig[:10] # Ensure 10
    else: betas_static = torch.zeros(10, device=device, dtype=betas_orig.dtype) # Simplified
    if betas_static.shape[0] != 10: log_message("Betas shape err.", config, "ERROR"); return False

    global_orient_aa_cam_smoothed = global_orient_aa_cam.clone()
    transl_cam_orig_smoothed = transl_cam_orig.clone()
    body_pose_aa_cam_smoothed = body_pose_aa_cam.clone()

    cfg_smoothing = config.get('temporal_smoothing', {})
    if cfg_smoothing.get('enable', False) and num_frames > 1:
        log_message("Applying temporal smoothing...", config)
        method = cfg_smoothing.get('method', 'ema')
        
        def plot_smoothing_comparison(original_data_np, smoothed_data_np, title_prefix, param_name, config_params_str, output_folder, file_id_for_log, config_for_log):
            # (Keep your plot_smoothing_comparison function)
            num_components = original_data_np.shape[1]
            if num_components == 0: return # Skip plotting if no components (e.g. empty indices_to_plot)
            fig, axes = plt.subplots(num_components, 1, sharex=True, figsize=(15, 3 * num_components))
            if num_components == 1: axes = [axes]
            for i in range(num_components):
                axes[i].plot(original_data_np[:, i], label=f'Original Comp {i}', alpha=0.7)
                axes[i].plot(smoothed_data_np[:, i], label=f'Smoothed Comp {i}', linewidth=1.5)
                axes[i].legend(); axes[i].set_title(f'{title_prefix} - Component {i}'); axes[i].grid(True, linestyle='--', alpha=0.6)
            plt.xlabel("Frame")
            fig.suptitle(f'{title_prefix} Smoothing\n{param_name} | Config: {config_params_str}\nFile: {file_id_for_log}', fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plot_filename_base = f"{file_id_for_log}_{param_name.lower().replace(' ', '_').replace('(','').replace(')','').replace(':','_').replace('/','_')}_smoothing_debug.png"
            plot_filename = os.path.join(output_folder, plot_filename_base)
            try: plt.savefig(plot_filename); log_message(f"  DEBUG: Saved plot to {plot_filename}", config_for_log)
            except Exception as e: log_message(f"  DEBUG: Error saving plot {plot_filename}: {e}", config_for_log, "ERROR")
            plt.close(fig)

        if method == 'ema':
            cfg_ema = cfg_smoothing.get('ema_filter', {})
            def _apply_ema_single_channel(data_channel, alpha_val):
                if data_channel.shape[0] <= 1: return data_channel
                smoothed_channel = torch.zeros_like(data_channel); smoothed_channel[0] = data_channel[0]
                for i in range(1, data_channel.shape[0]):
                    smoothed_channel[i] = alpha_val * data_channel[i] + (1.0 - alpha_val) * smoothed_channel[i-1]
                return smoothed_channel

            log_message("  Applying EMA to Global Orient (via Quaternions)...", config)
            original_go_aa_torch = global_orient_aa_cam.clone()
            global_orient_quat_cam_raw = axis_angle_to_quaternion_scipy(global_orient_aa_cam) # Raw, potentially discontinuous signs

            #smoothed_global_orient_quat_cam = global_orient_quat_cam.clone() # OLD
            #alpha_go = np.clip(cfg_ema.get('global_orient_smooth_factor'), 0.01, 0.99) # OLD
            #for i in range(4): # OLD
            #    smoothed_global_orient_quat_cam[:, i] = _apply_ema_single_channel(global_orient_quat_cam[:, i], alpha_go) # OLD
            #smoothed_global_orient_quat_cam = normalize_quaternions_torch(smoothed_global_orient_quat_cam) # OLD
            #global_orient_aa_cam_smoothed = quaternion_to_axis_angle_scipy(smoothed_global_orient_quat_cam) # OLD
            
            # --- NEW Quaternion EMA with sign handling ---
            if num_frames > 0:
                smoothed_global_orient_quat_cam = torch.empty_like(global_orient_quat_cam_raw)
                # Initialize with the first raw quaternion, normalized
                smoothed_global_orient_quat_cam[0] = normalize_quaternions_torch(global_orient_quat_cam_raw[0:1]).squeeze(0)

                alpha_go = np.clip(cfg_ema.get('global_orient_smooth_factor'), 0.01, 0.99)

                for t in range(1, num_frames):
                    q_prev_smooth = smoothed_global_orient_quat_cam[t-1] # Already normalized from previous step
                    q_curr_raw = global_orient_quat_cam_raw[t].clone() # Use a clone to modify if needed

                    # Ensure shortest path for interpolation by checking dot product
                    # If dot product is negative, q_curr_raw is on the "wrong side" of the hypersphere
                    if torch.dot(q_prev_smooth, q_curr_raw) < 0.0:
                        q_curr_raw = -q_curr_raw # Flip the sign of q_curr_raw

                    # Apply EMA (linear interpolation for quaternions)
                    interpolated_q = (1.0 - alpha_go) * q_prev_smooth + alpha_go * q_curr_raw
                    
                    # Normalize the result to ensure it's a valid unit quaternion
                    smoothed_global_orient_quat_cam[t] = normalize_quaternions_torch(interpolated_q.unsqueeze(0)).squeeze(0)
                
                global_orient_aa_cam_smoothed = quaternion_to_axis_angle_scipy(smoothed_global_orient_quat_cam)
            
            elif num_frames == 0: # Handle empty case if VIBE output was empty
                global_orient_aa_cam_smoothed = torch.empty((0,3), device=device, dtype=global_orient_aa_cam.dtype)
                smoothed_global_orient_quat_cam = torch.empty((0,4), device=device, dtype=global_orient_quat_cam_raw.dtype) # For plotting
            else: # num_frames == 1
                global_orient_aa_cam_smoothed = global_orient_aa_cam.clone() # No smoothing needed
                smoothed_global_orient_quat_cam = global_orient_quat_cam_raw.clone() # For plotting

            # --- END NEW Quaternion EMA ---

            # Update plotting to show original AA and smoothed AA (derived from corrected quat EMA)
            # And optionally, plot the quaternions themselves if debugging deeper
            if num_frames > 0: # Only plot if there is data
                plot_smoothing_comparison(
                    global_orient_quat_cam_raw.cpu().numpy(), # Original quaternions from VIBE
                    smoothed_global_orient_quat_cam.cpu().numpy(), # Smoothed quaternions
                    "Global Orient Quat (EMA Corrected)", "GO Quat",
                    f"alpha={alpha_go:.3f}",
                    config['output_amass_folder'], config.get('current_file_id'), config
                )
            plot_smoothing_comparison(original_go_aa_torch.cpu().numpy(), 
                                      global_orient_aa_cam_smoothed.cpu().numpy(), 
                                      "Global Orient AA (from EMA Quat Corrected)", "GO AA", 
                                      f"alpha={alpha_go:.3f}", 
                                      config['output_amass_folder'], config.get('current_file_id'), config)

            if cfg_smoothing.get('smooth_body_pose', False):
                log_message("  Applying Differentiated EMA to Body Pose...", config)
                original_bp_torch = body_pose_aa_cam.clone()
                # Make sure body_pose_aa_cam_smoothed is used here if it's the target for modification
                # body_pose_aa_cam_smoothed = body_pose_aa_cam.clone() # Already cloned above

                alpha_core = np.clip(cfg_ema.get('body_pose_core_smooth_factor'), 0.01, 0.99)
                alpha_limbs = np.clip(cfg_ema.get('body_pose_limbs_smooth_factor'), 0.01, 0.99)
                alpha_ends = np.clip(cfg_ema.get('body_pose_ends_smooth_factor'), 0.01, 0.99)
                log_message(f"    Body Pose Alphas: Core={alpha_core:.2f}, Limbs={alpha_limbs:.2f}, Ends={alpha_ends:.2f}", config)

                # Define joint groups for differentiated smoothing based on BODY_POSE_JOINT_NAMES
                core_joint_names = ['left_hip', 'right_hip', 'spine1', 'spine2', 'spine3', 'neck', 'head', 'left_collar', 'right_collar', 'left_shoulder', 'right_shoulder']
                limbs_joint_names = ['left_knee', 'right_knee', 'left_elbow', 'right_elbow']
                ends_joint_names = ['left_ankle', 'right_ankle', 'left_foot', 'right_foot', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand'] # Include hands if they are part of the 69 params

                core_param_indices = get_body_pose_param_indices(core_joint_names)
                limbs_param_indices = get_body_pose_param_indices(limbs_joint_names)
                ends_param_indices = get_body_pose_param_indices(ends_joint_names)
                
                # Sanity check: Ensure no overlaps and all params are covered (optional)
                # all_defined_indices = set(core_param_indices + limbs_param_indices + ends_param_indices)
                # if len(all_defined_indices) != 69 or len(all_defined_indices) != len(core_param_indices) + len(limbs_param_indices) + len(ends_param_indices) :
                #     log_message("Warning: Overlap or incomplete coverage in body_pose param indices for smoothing.", config, "WARNING")


                for i in range(body_pose_aa_cam.shape[1]):
                    current_alpha = alpha_core # Default
                    if i in ends_param_indices: current_alpha = alpha_ends
                    elif i in limbs_param_indices: current_alpha = alpha_limbs
                    body_pose_aa_cam_smoothed[:, i] = _apply_ema_single_channel(body_pose_aa_cam_smoothed[:, i], current_alpha) # Apply to the copy
                
                # Visualization for body pose (as before, but use the new indices for clarity)
                indices_to_plot_detailed = {
                    "Core (spine1)": get_body_pose_param_indices(['spine1']),
                    "Limb (l_elbow)": get_body_pose_param_indices(['left_elbow']),
                    "End (l_wrist)": get_body_pose_param_indices(['left_wrist'])
                }
                for group_name, indices in indices_to_plot_detailed.items():
                    if indices and max(indices) < original_bp_torch.shape[1]:
                         plot_smoothing_comparison(original_bp_torch.cpu().numpy()[:, indices], 
                                                   body_pose_aa_cam_smoothed.cpu().numpy()[:, indices], 
                                                   f"Body Pose (EMA - {group_name})", group_name, 
                                                   f"alphas:C={alpha_core:.2f},L={alpha_limbs:.2f},E={alpha_ends:.2f}", 
                                                   config['output_amass_folder'], config.get('current_file_id'), config)
        # Savgol and Median implementations (keep them as they were if you want to switch)
        elif method == 'savgol': pass # Placeholder for your savgol logic
        elif method == 'median': pass # Placeholder for your median logic
        else: log_message(f"Unknown smoothing method: {method}. No smoothing applied.", config, "WARNING")
    else:
        log_message("Temporal smoothing disabled or not enough frames.", config)
        # global_orient_aa_cam_smoothed, transl_cam_orig_smoothed, body_pose_aa_cam_smoothed
        # are already clones of original, so no explicit assignment needed here.

    # ----- 3. Coordinate System Correction -----
    cfg_coord_correction = config.get('coordinate_correction', {})
    apply_coord_correction = cfg_coord_correction.get('enable', True)
    global_orient_aa_world_candidate = global_orient_aa_cam_smoothed.clone()
    transl_world_candidate = transl_cam_orig_smoothed.clone()

    if apply_coord_correction:
        log_message("Applying fixed coordinate system correction...", config)
        correction_euler_angles_deg = cfg_coord_correction.get('fixed_rotation_euler_xyz_deg', [-90, 0, 0])
        correction_euler_order = cfg_coord_correction.get('fixed_rotation_euler_order', 'xyz')
        log_message(f"  Using fixed Euler: {correction_euler_angles_deg} deg, order '{correction_euler_order}'", config)
        try:
            r_coord_correction_obj = R_scipy.from_euler(correction_euler_order, correction_euler_angles_deg, degrees=True)
            R_coord_correction_matrix_torch = torch.tensor(r_coord_correction_obj.as_matrix(), dtype=torch.float32, device=device)
            R_global_orient_cam_smoothed = rodrigues_torch(global_orient_aa_cam_smoothed) # Use smoothed
            R_global_orient_corrected_world = torch.matmul(R_coord_correction_matrix_torch.unsqueeze(0), R_global_orient_cam_smoothed)
            global_orient_aa_world_candidate = rotation_matrix_to_axis_angle_torch(R_global_orient_corrected_world)
            transl_world_candidate = torch.matmul(R_coord_correction_matrix_torch, transl_cam_orig_smoothed.T).T # Use smoothed
        except Exception as e:
            log_message(f"  Error during fixed coordinate correction: {e}. Using uncorrected.", config, "ERROR")
    else:
        log_message("Fixed coordinate system correction disabled.", config)

    # ----- 4. Camera to World Transformation & Path Refinement (Optional) -----
    cfg_cam_to_world = config.get('camera_to_world', {})
    final_global_orient_aa_world = global_orient_aa_world_candidate.clone()
    final_transl_world = transl_world_candidate.clone()
    final_body_pose_aa = body_pose_aa_cam_smoothed.clone() # Use the smoothed body pose

    if cfg_cam_to_world.get('enable', False):
        log_message("Applying further C2W transformations (Y-align, orient, center)...", config)
        # ... (Your full C2W logic here, operating on final_... variables)
        # This part is complex and depends on SMPL model. Assuming it's mostly correct from your version.
        # Ensure it uses final_global_orient_aa_world, final_transl_world, final_body_pose_aa
        # and updates final_global_orient_aa_world, final_transl_world.
        # Simplified placeholder for the C2W block from your code
        smpl_for_align = None # Placeholder for get_smpl_model
        # ... if smpl_for_align: ... Y-align, Orient-align, Center ...

    # ----- 5. Final AMASS Data Assembly -----
    # (Keep your AMASS data assembly logic)
    final_poses_np = np.concatenate([final_global_orient_aa_world.cpu().numpy(), final_body_pose_aa.cpu().numpy()], axis=1)
    # ... (rest of your AMASS assembly) ...
    final_trans_np = final_transl_world.cpu().numpy(); final_betas_np = betas_static.cpu().numpy()
    num_amass_pose_params = config.get('amass_output_pose_params', 72)
    if final_poses_np.shape[1] < num_amass_pose_params: final_poses_np = np.concatenate([final_poses_np, np.zeros((num_frames, num_amass_pose_params - final_poses_np.shape[1]), dtype=final_poses_np.dtype)], axis=1) # Use dtype of original
    elif final_poses_np.shape[1] > num_amass_pose_params: final_poses_np = final_poses_np[:, :num_amass_pose_params]
    amass_data_to_save = {'poses': final_poses_np.astype(np.float32), 'trans': final_trans_np.astype(np.float32),
                          'betas': final_betas_np.astype(np.float32), 'gender': np.array([config.get('output_gender', 'neutral')], dtype='<U32'),
                          'mocap_framerate': float(config.get('output_mocap_framerate', 30.0)), 'dmpls': np.zeros((num_frames, 8), dtype=np.float32)}
    num_amass_beta_params = config.get('amass_output_beta_params', 16)
    if amass_data_to_save['betas'].shape[0] != num_amass_beta_params:
        padded_betas = np.zeros(num_amass_beta_params, dtype=np.float32)
        copy_len = min(amass_data_to_save['betas'].shape[0], num_amass_beta_params)
        padded_betas[:copy_len] = amass_data_to_save['betas'][:copy_len]; amass_data_to_save['betas'] = padded_betas
    try:
        os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
        np.savez(output_npz_path, **amass_data_to_save)
        log_message(f"Successfully saved to: {output_npz_path}", config); return True
    except Exception as e:
        log_message(f"Error saving .npz: {e}", config, "ERROR"); return False

# --- Main Execution Logic ---
# (Keep your __main__ block, ensure default_config_content matches the one I provided in the previous message,
#  especially the 'coordinate_correction' block and the differentiated EMA parameters)
if __name__ == '__main__':
    config_path = 'processing_config.yaml'
    # ... (Your main logic for loading config, creating dummy data if needed, and calling process_single_vibe_file)
    # Ensure the default_config_content in __main__ is updated with the new 'coordinate_correction' block.
    if not os.path.exists(config_path):
        print(f"ERROR: Config '{config_path}' not found. Creating template.")
        default_config_content = """
mode: "single"
input_vibe_pkl_path: "/root/autodl-tmp/VIBE/VIBE/output/400 Wetland Walk/vibe_output1630.pkl" # EDIT THIS
input_vibe_folder: "./vibe_inputs/"
output_amass_folder: "/root/autodl-tmp/MCM-LDM/processed_amass_data/" # EDIT THIS
smpl_models_dir: '/root/autodl-tmp/MCM-LDM/deps/smpl_models' # EDIT THIS
vibe_person_id_key: 1
output_mocap_framerate: 30.0
output_gender: "neutral"
amass_output_pose_params: 72
amass_output_beta_params: 16

temporal_smoothing:
  enable: true
  method: "ema" 
  smooth_body_pose: true

  savgol_filter:
    global_orient_window: 15; global_orient_polyorder: 3
    camera_transl_window: 15; camera_transl_polyorder: 3
    body_pose_window: 9; body_pose_polyorder: 2
    
  ema_filter:
    global_orient_smooth_factor: 0.6 # Alpha for pivot
    camera_transl_smooth_factor: 0.6 # Alpha for pivot
    body_pose_core_smooth_factor: 0.7  # Alpha for core body
    body_pose_limbs_smooth_factor: 0.3 # Alpha for limbs
    body_pose_ends_smooth_factor: 0.1  # Alpha for ends (hands, feet)

  median_filter:
    global_orient_kernel_size: 5
    camera_transl_kernel_size: 5
    body_pose_kernel_size: 7

coordinate_correction:
  enable: true
  fixed_rotation_euler_xyz_deg: [-90, 0, 0] # START WITH THIS. If upside down: [180,0,0]. Then adjust Y if needed.
  fixed_rotation_euler_order: 'xyz'

camera_to_world:
  enable: true 
  initial_y_offset_adjustment: 0.0
  ground_plane_y: 0.0
  default_z_for_pred_cam: 2.5
  orientation_align_mode: "target_vector" 
  target_world_forward_vector: [0.0, 1.0, 0.0]
  path_refinement:
    enable: false

foot_grounding_ik:
  enable: false
"""
        with open(config_path, 'w') as f_cfg: f_cfg.write(default_config_content)
        print(f"Template '{config_path}' created. EDIT paths & fixed_rotation, then RERUN."); exit()
    
    try:
        with open(config_path, 'r') as f: config = yaml.safe_load(f)
    except Exception as e: print(f"ERROR loading config: {e}"); exit()
    if 'current_file_id' not in config: config['current_file_id'] = "MAIN"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(f"Using device: {device}", config)
    os.makedirs(config['output_amass_folder'], exist_ok=True)

    if config['mode'] == 'single':
        input_path = config['input_vibe_pkl_path']
        if not os.path.exists(input_path) and "dummy" in input_path: 
            import pickle 
            dummy_num_frames = 50; person_key = config.get('vibe_person_id_key', 1)
            dummy_data = {person_key: {'pred_smpl_params': {
                'global_orient': np.random.randn(dummy_num_frames, 3).astype(np.float32) * 0.2,
                'body_pose': np.random.randn(dummy_num_frames, 69).astype(np.float32) * 0.1,
                'betas': (np.random.randn(1, 10).astype(np.float32) * 0.05),
                'transl': (np.random.randn(dummy_num_frames, 3).astype(np.float32) * 0.05 + np.array([0,0,2.5])).astype(np.float32)}}}
            with open(input_path, 'wb') as f: pickle.dump(dummy_data, f)
            log_message(f"Created dummy PKL: {input_path}", config)
        elif not os.path.exists(input_path): log_message(f"Input '{input_path}' not found.", config, "ERROR"); exit()
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(config['output_amass_folder'], f"{base_name}_amass.npz")
        config['current_file_id'] = base_name
        process_single_vibe_file(input_path, output_path, config, device)
    elif config['mode'] == 'folder': 
        # (Your folder processing logic here)
        pass 
    else: log_message(f"Invalid mode: {config['mode']}", config, "ERROR")
    log_message("All processing finished.", config)