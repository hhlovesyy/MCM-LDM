import numpy as np
import torch
from torch.utils import data
import random
from os.path import join as pjoin
import logging
import json
import collections

# Initialize a logger for this module
logger = logging.getLogger(__name__)

class PhysiMoS100StyleDataset(data.Dataset):
    """
    A Dataset class for the PhysiMoS project's technical probe.
    - Loads motions from the 100Style dataset.
    - Generates "pseudo" physical parameters and scene categories based on a mapping file.
    - Follows the self-reconstruction proxy: motion_before = motion_after.
    """
    def __init__(
        self,
        mean,
        std,
        split_file,
        motion_dir,
        max_motion_length,
        min_motion_length,
        unit_length,
        style_dict_path,
        scene_mapping_path, # Path to the new scenes.json file
        style_subset=None,    # Optional: A list of style names for a mini-dataset probe
        **kwargs,             # Gracefully accept and ignore other dataset params
    ):
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length
        self.mean = mean
        self.std = std

        # --- 1. Load scene and physics mappings from the JSON file ---
        with open(scene_mapping_path, 'r') as f:
            scene_data = json.load(f)
        self.style_to_phys = scene_data["style_mapping"]
        self.scene_categories = scene_data["scene_categories"]
        self.scene_to_id = {name: i for i, name in enumerate(self.scene_categories)}
        self.num_scenes = len(self.scene_categories)
        self.phys_params_dim = len(scene_data["physical_parameters_desc"])
        logger.info(f"PhysiMoS: Loaded scene mapping for {len(self.style_to_phys)} styles across {self.num_scenes} scenes.")
        logger.info(f"PhysiMoS: Physical parameter dimension is {self.phys_params_dim}.")

        # --- 2. Load file ID to style name mapping ---
        self.id_to_style = {}
        with open(style_dict_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 2:
                    file_id = parts[0]
                    style_name_key = parts[1].split('_')[0].lower().replace(" ", "")
                    self.id_to_style[file_id] = style_name_key
        
        split_name = split_file.split('/')[-1].split('.')[0].upper() # e.g., 'TRAIN' or 'TEST'
        logger.info(f"--- [DIAGNOSTIC REPORT FOR {split_name} SPLIT] ---")

        # --- 3. Load IDs from the dataset split file ---
        id_list = []
        with open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        logger.info(f"1. Total file IDs found in '{split_name}.txt': {len(id_list)}")

        # --- 4. Define known broken styles to exclude ---
        styles_to_exclude = { "whirlarms", "widelegs", "wigglehips", "wildarms", "wildlegs", "zombie" }

        # --- 5. [Core Logic] Load and filter data ---
        self.data_dict = {}
        self.name_list = []
        
        # Determine which styles to load
        valid_styles_from_json = set(self.style_to_phys.keys())
        if style_subset:
            # If a subset is provided for the probe, use the intersection of both sets
            selected_styles = set(style_subset).intersection(valid_styles_from_json)
            logger.info(f"[PROBE MODE] Loading a mini-dataset with {len(selected_styles)} styles: {selected_styles}")
        else:
            selected_styles = valid_styles_from_json
        
        rejection_reasons = collections.defaultdict(int)
        for name in id_list:
            style_name = self.id_to_style.get(name)
            
            # Filtering criteria:
            # 1. Style name must exist
            # 2. Must not be in the exclusion list
            # 3. Must be one of the selected styles for loading
            if not style_name or style_name in styles_to_exclude or style_name not in selected_styles:
                rejection_reasons['unselected_style'] += 1
                continue

            try:
                motion_path = pjoin(motion_dir, name + ".npy")
                motion = np.load(motion_path)

                # 4. Filter by motion length
                if not (self.min_motion_length <= len(motion) < self.max_motion_length):
                    rejection_reasons['invalid_length'] += 1
                    continue
                
                # If all checks pass, add to our dictionary
                self.data_dict[name] = {
                    "motion": motion,
                    "length": len(motion),
                    "style_name": style_name,
                }
                self.name_list.append(name)
            except Exception:
                rejection_reasons['file_not_found_or_corrupt'] += 1
                continue
        
        # 3. Print the report
        total_rejected = sum(rejection_reasons.values())
        total_processed = len(id_list)
        pass_rate = (len(self.name_list) / total_processed) * 100 if total_processed > 0 else 0

        logger.info(f"2. Total samples processed: {total_processed}")
        logger.info(f"   - Samples REJECTED: {total_rejected}")
        logger.info(f"     - Reason 'Unselected Style': {rejection_reasons['unselected_style']}")
        logger.info(f"     - Reason 'Invalid Length (not in [{self.min_motion_length}, {self.max_motion_length}))': {rejection_reasons['invalid_length']}")
        logger.info(f"     - Reason 'File Not Found/Corrupt': {rejection_reasons['file_not_found_or_corrupt']}")
        logger.info(f"   - Samples ACCEPTED: {len(self.name_list)}")
        logger.info(f"3. Pass Rate for this split: {pass_rate:.2f}%")
        logger.info(f"--- [END OF DIAGNOSTIC REPORT FOR {split_name} SPLIT] ---")

        if not self.name_list:
            raise ValueError(f"PhysiMoS ({split_name}): No valid samples were loaded.")
        
        logger.info(f"PhysiMoS: Successfully loaded {len(self.name_list)} motion samples.")
        self.nfeats = self.data_dict[self.name_list[0]]['motion'].shape[1]

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]
        
        motion, m_length, style_name = data["motion"], data["length"], data["style_name"]

        # --- Step 1: Get the physics and scene conditions for this style ---
        phys_data = self.style_to_phys[style_name]
        scene_name = phys_data["scene_category"]
        phys_params = phys_data["physical_parameters"]

        # Convert scene name to a one-hot vector
        scene_id = self.scene_to_id[scene_name]
        scene_cat_one_hot = np.zeros(self.num_scenes, dtype=np.float32)
        scene_cat_one_hot[scene_id] = 1.0

        # --- Step 2: Crop the motion randomly and standardize ---
        m_length_cropped = (m_length // self.unit_length) * self.unit_length
        
        # Robustness: ensure cropped length is at least one unit
        if m_length_cropped < self.unit_length:
            m_length_cropped = self.unit_length
        
        idx = random.randint(0, m_length - m_length_cropped)
        motion_cropped = motion[idx:idx + m_length_cropped]
        
        motion_normalized = (motion_cropped - self.mean) / self.std

        # In case of rare NaN values after normalization, resample.
        if np.any(np.isnan(motion_normalized)):
            logger.warning(f"NaN detected in motion sample {name}. Resampling...")
            return self.__getitem__(np.random.randint(0, len(self.name_list)))

        # --- Step 3: Create the self-reconstruction pair ---
        # As discussed, `motion_before` is simply a copy of `motion_after` for our probe.
        motion_after = motion_normalized
        motion_before = motion_after.copy()

        # --- Step 4: Assemble and return the data dictionary ---
        return {
            "motion_after": torch.from_numpy(motion_after).float(),
            "motion_before": torch.from_numpy(motion_before).float(),
            "length": m_length_cropped,
            "phys_params": torch.from_numpy(np.array(phys_params, dtype=np.float32)),
            "scene_cat": torch.from_numpy(scene_cat_one_hot),
            "caption": f"Style: {style_name}, Scene: {scene_name}",  # Included for easier debugging
        }