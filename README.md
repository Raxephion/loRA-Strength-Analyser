# LoRA Strength Analyzer
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python script to analyze images generated using a LoRA (Low-Rank Adaptation) model applied at various strength levels. This tool helps determine an optimal strength for a given LoRA by evaluating image quality and similarity to control images.

*(Note: Gradio webUI coming soon)*

### 🧠 1. Structural Similarity Index (SSIM)

SSIM measures the similarity between two images in terms of:

*   Luminance
*   Contrast
*   Structural features

It’s computed as:

$$
\text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

Where:

*   $\mu_x, \mu_y$: mean pixel intensities
*   $\sigma_x, \sigma_y$: standard deviations
*   $\sigma_{xy}$: cross-covariance
*   $C_1, C_2$: constants to stabilize division

**Interpretation:**

*   SSIM ≈ 1.0 → Very similar (minimal LoRA effect)
*   SSIM ≪ 1.0 → Significant difference (strong LoRA effect)

### 🔍 2. BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)

BRISQUE estimates the perceptual quality of an image without needing a reference. It uses machine learning and natural scene statistics to assess artifacts and distortions.

**How it works:**

*   Extracts statistical features from image patches
*   Feeds them into a pretrained model (typically SVM)
*   Outputs a quality score

**Interpretation:**

*   Lower score → Better image quality
*   Higher score → More visible artifacts, noise, or degradation

## 🎯 Goal

By analyzing both:

*   **SSIM** (similarity to original)
*   **BRISQUE** (perceptual quality)

The tool helps you:

*   Detect the best LoRA strength for subtle or strong stylistic changes
*   Avoid over-strengthening that introduces artifacts
*   Maintain good image quality while applying desired effects


## Features

-   Calculates **BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)** score for each LoRA-strength image. Lower BRISQUE scores generally indicate better perceptual quality.
-   Calculates **SSIM (Structural Similarity Index Measure)** between each LoRA-strength image and its corresponding control image (or a single control image). An SSIM score of 1.0 means identical.
-   Provides a summary table of scores for all tested strengths.
-   Suggests a "best" strength based on the lowest BRISQUE score (preferring lower strength in case of a tie).
-   Supports using either individual control images for each strength or a single control image for all strengths.

## Prerequisites

-   Python 3.7+
-   Git (for cloning)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Raxephion/loRA-Strength-Analyser.git
    cd lora-strength-analyzer
    ```

2.  **Create and activate a virtual environment:**

    *   Using `venv`:
        ```bash
        python -m venv venv
        # On Windows
        .\venv\Scripts\activate
        # On macOS/Linux
        source venv/bin/activate
        ```
    *   Using `conda`:
        ```bash
        conda create -n lora_analyzer python=3.9 # Or your preferred Python 3.x version
        conda activate lora_analyzer
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Script:**
    Open `lora_strength_analyzer.py` in a text editor.

    *   **Update Author Information (Optional but good practice):**
        ```python
        # Modified to check LoRA Strength on [Your Modification Date] <--- UPDATE THIS
        # @author: raxephion (Original), [Your Name/Handle for modification] <--- UPDATE THIS
        ```

    *   **MUST Update Paths:**
        Adjust the following placeholder paths to your actual directory locations:
        ```python
        LORA_STRENGTH_IMAGES_DIR = Path(r"C:\...\LoRA_Training\lora_strength_images") # UPDATE THIS
        CONTROL_IMAGES_DIR = Path(r"C:\...\LoRA_Training\control_images") # UPDATE THIS
        ```

    *   **Adjust LoRA Strength Parameters:**
        Configure the range and step for LoRA strengths:
        ```python
        MIN_LORA_STRENGTH = 0.1
        MAX_LORA_STRENGTH = 1.0
        LORA_STRENGTH_INCREMENT = 0.1
        STRENGTH_DECIMAL_PLACES = 1 # For formatting e.g., 0.1, 1.0
        ```

    *   **Verify Image Naming Convention:**
        The script assumes image names like:
        -   LoRA strength images: `strength_0.1.png`, `strength_0.2.png`, ...
        Modify `LORA_IMAGE_PREFIX` and `IMAGE_EXTENSION` if yours differ.

    *   **Control Image Configuration:**
        You have two options for control images:
        1.  **Multiple Control Images:** (Default) One control image per LoRA strength tested.
            -   Control images named: `control_01.png`, `control_02.png`, ... where `control_01.png` corresponds to `MIN_LORA_STRENGTH`, `control_02.png` to the next strength, and so on.
            -   Modify `CONTROL_IMAGE_PREFIX` if needed.
            -   Keep `USE_SINGLE_CONTROL_IMAGE = False`.
        2.  **Single Control Image:** Use the *same* control image for all LoRA strengths.
            -   Set `USE_SINGLE_CONTROL_IMAGE = True`.
            -   Specify `SINGLE_CONTROL_IMAGE_NAME = "your_control_image.png"` (update this name).
            -   Ensure this image exists in your `CONTROL_IMAGES_DIR`.

## Usage

Once configured, run the script from your terminal:

```bash
python lora_strength_analyzer.py
