# -*- coding: utf-8 -*-
"""
Created on Mon May 12 09:02:32 2025 (Original by raxephion)
Modified to check LoRA Strength on [Your Modification Date] <--- UPDATE THIS

@author: raxephion (Original), [Your Name/Handle for modification] <--- UPDATE THIS

LoRA Strength Checker App

This script analyzes images generated with a chosen LoRA applied at different
strength levels. It compares these images against control images (base model
output without LoRA) and assesses their perceptual quality.

It calculates:
- BRISQUE score for each LoRA-strength image (lower is better).
- SSIM score between each LoRA-strength image and its corresponding control image.

This helps in finding an optimal strength for applying a given LoRA.

Activate and Deactivate venv (example):
# conda create -n lora_analyzer python=3.9
# conda activate lora_analyzer
# pip install -r requirements.txt
# conda deactivate
"""

import os
from pathlib import Path
from PIL import Image
# Corrected import for scikit-image
try:
    from skimage.measure import compare_ssim as ssim_func # Use a generic name
    # Check if compare_ssim is the intended one, skimage.metrics.structural_similarity is preferred for newer versions
    # Forcing specific args for compare_ssim later if this is what's imported.
    _using_compare_ssim = True
except ImportError:
    try:
        from skimage.metrics import structural_similarity as ssim_func
        print("Using structural_similarity from skimage.metrics as SSIM function.")
        _using_compare_ssim = False
    except ImportError:
        raise ImportError("Could not import SSIM function. Please ensure scikit-image is installed correctly.")

import numpy as np
import imquality.brisque as brisque

# --- Configuration ---
# !!! IMPORTANT !!!
# Adjust these paths to your actual folder locations.
LORA_STRENGTH_IMAGES_DIR = Path(r"C:\...\LoRA_Training\lora_strength_images") # Placeholder - UPDATE THIS
CONTROL_IMAGES_DIR = Path(r"C:\...\LoRA_Training\control_images") # Placeholder - UPDATE THIS

# LoRA Strength settings
MIN_LORA_STRENGTH = 0.1
MAX_LORA_STRENGTH = 1.0  # Adjust as needed (e.g., 1.0, 1.2, etc.)
LORA_STRENGTH_INCREMENT = 0.1
# Number of decimal places for strength value formatting (e.g., 1 for 0.1, 2 for 0.01)
STRENGTH_DECIMAL_PLACES = 1 # Ensure this matches your LORA_STRENGTH_INCREMENT's precision

# Naming convention assumptions (modify if yours is different)
# Assumes LoRA strength images are like "strength_0.1.png", "strength_0.2.png", ..., "strength_1.0.png"
LORA_IMAGE_PREFIX = "strength_"
# Assumes control images are like "control_01.png", "control_02.png", ...
# where control_01 corresponds to the first strength tested (MIN_LORA_STRENGTH),
# control_02 to the second strength, and so on.
# If you use a SINGLE control image for all strengths, modify get_image_paths.
CONTROL_IMAGE_PREFIX = "control_"
IMAGE_EXTENSION = ".png" # Common image extension, change if different

# --- Optional: Configuration for a single control image ---
# If you want to use the *same* control image for comparison against *all* LoRA strengths:
# 1. Set USE_SINGLE_CONTROL_IMAGE to True.
# 2. Specify SINGLE_CONTROL_IMAGE_NAME.
# 3. Ensure this single control image exists in CONTROL_IMAGES_DIR.
USE_SINGLE_CONTROL_IMAGE = False # Set to True to use one control image for all strengths
SINGLE_CONTROL_IMAGE_NAME = "control_base.png" # e.g., "control_base.png" - UPDATE if USE_SINGLE_CONTROL_IMAGE is True

# --- Helper Functions ---
def get_image_paths(strength_val, control_num):
    """
    Constructs paths for LoRA strength and control images.
    """
    # LoRA strength image: e.g., strength_0.1.png, strength_1.0.png
    strength_str = f"{strength_val:.{STRENGTH_DECIMAL_PLACES}f}"
    lora_img_name = f"{LORA_IMAGE_PREFIX}{strength_str}{IMAGE_EXTENSION}"
    lora_img_path = LORA_STRENGTH_IMAGES_DIR / lora_img_name

    if USE_SINGLE_CONTROL_IMAGE:
        control_img_name = SINGLE_CONTROL_IMAGE_NAME
        control_img_path = CONTROL_IMAGES_DIR / control_img_name
    else:
        # Control image: e.g., control_01.png (linked to the iteration count)
        control_str = f"{control_num:02d}" # e.g., 1 -> "01", 10 -> "10"
        control_img_name = f"{CONTROL_IMAGE_PREFIX}{control_str}{IMAGE_EXTENSION}"
        control_img_path = CONTROL_IMAGES_DIR / control_img_name

    return lora_img_path, control_img_path

def calculate_metrics(lora_img_path, control_img_path):
    """
    Calculates BRISQUE for the LoRA image and SSIM between LoRA and control image.
    """
    try:
        lora_img_pil = Image.open(lora_img_path).convert('RGB') # Ensure RGB for consistency
        control_img_pil = Image.open(control_img_path).convert('RGB') # Ensure RGB for consistency

        lora_img_np = np.array(lora_img_pil)
        control_img_np = np.array(control_img_pil)

        brisque_score = brisque.score(lora_img_pil)

        if lora_img_np.shape != control_img_np.shape:
             print(f"Warning: Image shapes differ for SSIM. LoRA: {lora_img_np.shape}, Control: {control_img_np.shape}. Resizing control to LoRA image size for SSIM.")
             control_img_pil_resized = control_img_pil.resize(lora_img_pil.size)
             # control_img_np = np.array(control_img_pil_resized) # Not needed if we convert to gray from PIL
        else:
             control_img_pil_resized = control_img_pil

        lora_img_gray_np = np.array(lora_img_pil.convert('L'))
        control_img_gray_np = np.array(control_img_pil_resized.convert('L'))

        # Call SSIM function
        # The ssim_func could be compare_ssim or structural_similarity
        if _using_compare_ssim:
            # skimage.measure.compare_ssim specific call
            ssim_index = ssim_func(lora_img_gray_np, control_img_gray_np,
                                   data_range=lora_img_gray_np.max() - lora_img_gray_np.min(),
                                   full=False) # compare_ssim returns score directly with full=False
        else:
            # skimage.metrics.structural_similarity call
            # It may or may not need channel_axis depending on exact version and if it handles 2D arrays well.
            # Most recent versions infer data_range for common dtypes.
            try:
                 ssim_index = ssim_func(lora_img_gray_np, control_img_gray_np,
                                       data_range=lora_img_gray_np.max() - lora_img_gray_np.min())
            except TypeError: # Older structural_similarity might need multichannel or channel_axis
                 ssim_index = ssim_func(lora_img_gray_np, control_img_gray_np,
                                       data_range=lora_img_gray_np.max() - lora_img_gray_np.min(),
                                       channel_axis=None, # For grayscale
                                       multichannel=False) # If it's an older skimage.metrics version

        if isinstance(ssim_index, tuple): # If it returned (score, grad_image)
            ssim_index = ssim_index[0]

        return brisque_score, ssim_index

    except FileNotFoundError:
        print(f"Error: Could not find images for comparison: {lora_img_path} or {control_img_path}")
        return None, None
    except Exception as e:
        print(f"Error processing images {lora_img_path} / {control_img_path}: {e}")
        return None, None

# --- Main Analysis Logic ---
def analyze_lora_strengths():
    print("Starting LoRA Strength Analysis...")
    print(f"LoRA Strength Images Dir: {LORA_STRENGTH_IMAGES_DIR}")
    print(f"Control Images Dir: {CONTROL_IMAGES_DIR}")
    if USE_SINGLE_CONTROL_IMAGE:
        print(f"Using single control image: {CONTROL_IMAGES_DIR / SINGLE_CONTROL_IMAGE_NAME}")
    print(f"Analyzing strengths from {MIN_LORA_STRENGTH} to {MAX_LORA_STRENGTH} in steps of {LORA_STRENGTH_INCREMENT}\n")

    if not LORA_STRENGTH_IMAGES_DIR.is_dir():
        print(f"Error: LoRA strength images directory not found: {LORA_STRENGTH_IMAGES_DIR}")
        print("Please update LORA_STRENGTH_IMAGES_DIR in the script.")
        return
    if not CONTROL_IMAGES_DIR.is_dir():
        print(f"Error: Control images directory not found: {CONTROL_IMAGES_DIR}")
        print("Please update CONTROL_IMAGES_DIR in the script.")
        return

    if not any(LORA_STRENGTH_IMAGES_DIR.iterdir()):
        print(f"Error: LoRA strength images directory is empty: {LORA_STRENGTH_IMAGES_DIR}")
        return
    if not USE_SINGLE_CONTROL_IMAGE and not any(CONTROL_IMAGES_DIR.iterdir()):
         print(f"Error: Control images directory is empty: {CONTROL_IMAGES_DIR} (and not using single control image mode)")
         return
    if USE_SINGLE_CONTROL_IMAGE and not (CONTROL_IMAGES_DIR / SINGLE_CONTROL_IMAGE_NAME).exists():
        print(f"Error: Single control image not found: {CONTROL_IMAGES_DIR / SINGLE_CONTROL_IMAGE_NAME}")
        return

    results = []
    current_strengths_raw = np.arange(MIN_LORA_STRENGTH, MAX_LORA_STRENGTH + LORA_STRENGTH_INCREMENT / 2, LORA_STRENGTH_INCREMENT)

    for idx, strength_val_unrounded in enumerate(current_strengths_raw):
        strength_val = round(strength_val_unrounded, STRENGTH_DECIMAL_PLACES + 2) # Round with a bit more precision initially
        strength_val = round(strength_val, STRENGTH_DECIMAL_PLACES) # Then round to target for display/filename

        control_num = idx + 1 # 1-indexed for multiple control images
        lora_img_path, control_img_path = get_image_paths(strength_val, control_num)
        strength_display_format = f"{strength_val:.{STRENGTH_DECIMAL_PLACES}f}"

        if not lora_img_path.exists():
            print(f"Skipping Strength {strength_display_format}: LoRA image not found at {lora_img_path}")
            results.append({'strength': strength_val, 'brisque': float('inf'), 'ssim': float('-inf'), 'error': 'LoRA image missing'})
            continue
        if not control_img_path.exists():
            control_id_str = SINGLE_CONTROL_IMAGE_NAME if USE_SINGLE_CONTROL_IMAGE else f"set {control_num:02d}"
            print(f"Skipping Strength {strength_display_format} (Control {control_id_str}): Control image not found at {control_img_path}")
            results.append({'strength': strength_val, 'brisque': float('inf'), 'ssim': float('-inf'), 'error': f'Control image {control_id_str} missing'})
            continue

        control_id_print = SINGLE_CONTROL_IMAGE_NAME if USE_SINGLE_CONTROL_IMAGE else f"set {control_num:02d}"
        print(f"Processing Strength {strength_display_format} (Control: {control_id_print})...")
        brisque_val, ssim_val = calculate_metrics(lora_img_path, control_img_path)

        if brisque_val is not None and ssim_val is not None:
            results.append({'strength': strength_val, 'brisque': brisque_val, 'ssim': ssim_val})
            print(f"  Strength {strength_display_format}: BRISQUE = {brisque_val:.2f}, SSIM (to control {control_id_print}) = {ssim_val:.4f}")
        else:
            results.append({'strength': strength_val, 'brisque': float('inf'), 'ssim': float('-inf'), 'error': 'Processing error'})

    print("\n--- Strength Analysis Summary ---")
    strength_col_width = max(8, STRENGTH_DECIMAL_PLACES + 4)
    print(f"{'Strength':<{strength_col_width}} | BRISQUE (Lower is better) | SSIM (to Control, 1.0=identical)")
    print("-" * (strength_col_width + 1) + "------------------------------------------------------------------")
    for res in results:
        strength_display = f"{res['strength']:.{STRENGTH_DECIMAL_PLACES}f}"
        if 'error' not in res:
            print(f"{strength_display:<{strength_col_width}} | {res['brisque']:<25.2f} | {res['ssim']:.4f}")
        else:
            print(f"{strength_display:<{strength_col_width}} | {'N/A':<25} | {'N/A'} ({res['error']})")

    valid_results = [r for r in results if 'error' not in r and r['brisque'] != float('inf')]
    if not valid_results:
        print("\nNo valid results with computable metrics to suggest a best strength.")
        return

    min_brisque_val = min(r['brisque'] for r in valid_results)
    tolerance = 0.01
    best_strength_candidates = [r for r in valid_results if abs(r['brisque'] - min_brisque_val) < tolerance]
    best_strength_info = min(best_strength_candidates, key=lambda x: x['strength'])

    print(f"\n--- Suggested Best Strength (based on lowest BRISQUE, preferring lower strength on tie) ---")
    best_strength_display = f"{best_strength_info['strength']:.{STRENGTH_DECIMAL_PLACES}f}"
    print(f"Strength: {best_strength_display}")
    print(f"  BRISQUE: {best_strength_info['brisque']:.2f}")
    print(f"  SSIM to Control: {best_strength_info['ssim']:.4f}")

    print("\nConsiderations for choosing the best strength:")
    print("1. Low BRISQUE score: Better perceptual image quality (fewer artifacts).")
    print("2. SSIM score: Similarity to the control image (without LoRA).")
    print("   - Subtle LoRA: Aim for higher SSIM with good BRISQUE.")
    print("   - Significant LoRA changes: SSIM will be lower; focus on desired effect and BRISQUE.")
    print("3. Balance: Good quality (low BRISQUE) and desired LoRA effect (informed by SSIM and visual inspection).")
    print("4. Excessive Strength: Watch for BRISQUE increasing or SSIM dropping too drastically/unnaturally.")

if __name__ == "__main__":
    if "C:\\...\\" in str(LORA_STRENGTH_IMAGES_DIR) or "C:\\...\\" in str(CONTROL_IMAGES_DIR):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! WARNING: Default placeholder paths are still in use.         !!!")
        print("!!! Please edit LORA_STRENGTH_IMAGES_DIR and CONTROL_IMAGES_DIR  !!!")
        print("!!! in lora_strength_analyzer.py before running.                 !!!")
        if USE_SINGLE_CONTROL_IMAGE and SINGLE_CONTROL_IMAGE_NAME == "control_base.png":
            print("!!! Also, verify SINGLE_CONTROL_IMAGE_NAME if using single control mode. !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    analyze_lora_strengths()
