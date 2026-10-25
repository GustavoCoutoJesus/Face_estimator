# Step-by-Step Instructions: Running the Social Judgment Face Estimator

This guide explains how to set up and run the **Perceived Social Judgment Estimator** on your own naturalistic photographs of faces, producing the exact same 34-trait CSV and two sets of diagnostic figures (Social/Todorov 2D Profiles and Demographic/Physical Profiles).

---

## 1. Folder Structure

When you unzip this folder or clone the GitHub repository, it has the following structure:

```text
Face_estimator/
├── input_faces/                 <-- Put the face photos (.jpg, .png, etc.) you want to analyze here
├── results/                     <-- All outputs are automatically saved here:
│   ├── social_judgments.csv     <-- Spreadsheet with all 34 OMI traits + Todorov 2D scores
│   ├── aligned_faces/           <-- 512x512 FFHQ landmark-aligned face crops
│   ├── figures/                 <-- Figure Set 1: 18 Social Traits + Todorov 2D Valence × Dominance Space
│   └── figures_demographics/    <-- Figure Set 2: 9 Physical/Demographic Traits + 7 Ethnic Attributions
├── estimate_faces.py            <-- Main CLI & batch runner script
├── ffhq_align.py                <-- Automatic MediaPipe + NVIDIA FFHQ facial landmark alignment
├── generate_demographic_figures.py <-- Complementary demographic/physical figure generator
├── requirements.txt             <-- Pinned Python package versions
├── run_estimator.bat            <-- One-click launcher for Windows
├── run_estimator.sh             <-- One-click launcher for macOS / Linux
└── INSTRUCTIONS.md              <-- This guide
```

---

## 2. Prerequisites

1. **Python 3.10 or 3.11** (recommended: Python `3.11.x` from [python.org](https://www.python.org/downloads/)).
   - *Windows tip*: During Python installation, check **"Add python.exe to PATH"**.
2. **Internet connection on the first run** (to download the Python packages and the ~1.28 GB pre-trained DINOv2 OMI model checkpoint from Hugging Face `kiante/face-trait-transformer`).
   - Once downloaded once, the model is cached locally in `~/.cache/huggingface/hub/` and `~/.cache/torch/hub/`, so subsequent runs work completely offline and take only **~3–5 seconds per photo on CPU**.

---

## 3. How to Run (Option A: One-Click Launcher)

1. Copy your face photographs (`.jpg`, `.jpeg`, `.png`, `.webp`) into the **`input_faces/`** folder.
   - Photos can be uncropped, naturalistic portraits — the pipeline automatically detects facial landmarks (eyes and mouth corners) and aligns them to the FFHQ standard before scoring.
2. Run the launcher for your operating system:
   - **Windows**: Double-click **`run_estimator.bat`** (or run `.\run_estimator.bat` in PowerShell / Command Prompt).
   - **macOS / Linux**: Run `bash run_estimator.sh` in Terminal.
3. On the first run, the script automatically installs the dependencies from `requirements.txt` and downloads the model weights.
4. Open the **`results/`** folder to view your `social_judgments.csv` and both sets of figures!

---

## 4. How to Run (Option B: Manual Terminal / Command Line)

Open a terminal (PowerShell, Command Prompt, or macOS/Linux Terminal) inside the `Face_estimator` directory:

```bash
# Step 1: Install required packages (only needed once)
python -m pip install -r requirements.txt

# Step 2: Run the estimator on all images inside input_faces/
python estimate_faces.py input_faces
```

### Useful Command-Line Options
- **Score a specific photo or custom folder**:
  ```bash
  python estimate_faces.py path/to/my_photo.jpg --output-dir my_results
  ```
- **Re-generate only the demographic/physical figures from an existing CSV**:
  ```bash
  python generate_demographic_figures.py
  ```
- **Skip FFHQ alignment** (only if your images are already 512×512 FFHQ-aligned crops):
  ```bash
  python estimate_faces.py input_faces --no-align
  ```
- **Optional Full ViT-G/14 + ViT-L/14 Ensemble (`--ensemble`)**:
  By default, `estimate_faces.py` runs the end-to-end fine-tuned **DINOv2 ViT-L/14** model (`finetune_vitl_lb1.pt`, 1.28 GB), which runs fast on CPU (~3–5s/face). If you have a dedicated GPU and want to also run the 4.23 GB DINOv2 ViT-G/14 (518px) branch, pass `--ensemble`.

---

## 5. Offline Weight Sharing (Optional)

If you want to send the pre-downloaded model weights directly to a collaborator (e.g., via USB drive or institutional cloud storage) so their computer does not need to download the 1.28 GB Hugging Face bundle:

1. Copy the contents of your local Hugging Face snapshot folder:
   `C:\Users\<username>\.cache\huggingface\hub\models--kiante--face-trait-transformer\snapshots\<hash>\`
2. Place those files (`manifest.json`, `finetune/`, `head_vitg_518_vw/`) into a subfolder named **`model_bundle/`** inside `Face_estimator/`.
3. `estimate_faces.py` automatically checks for `./model_bundle/manifest.json` first and loads the weights directly from disk without connecting to Hugging Face.

---

## 6. Understanding the Outputs (`0–100` Scale)

All 34 attributes are scored on a `0–100` scale (`50` = neutral midpoint), trained on **1.3 million human ratings** from the **One Million Impressions (OMI)** dataset (Peterson, Uddenberg, Griffiths, Todorov, & Suchow, 2022, *PNAS*):

- **Todorov 2D Structural Composites** (Todorov & Oh, 2021):
  - `todorov_valence_index`: Perceived approachability / warmth (`trustworthy`, `happy`, `outgoing`, `cute`, and inverse `smug`). Highly sensitive to facial expression.
  - `todorov_dominance_index`: Perceived capability / power (`dominant`, `alert`, `smart`, `electable`). Primarily driven by static bone structure / facial morphology.
- **Figure Set 1 (`results/figures/<name>_profile.png`)**:
  - Displays the original photo, FFHQ-aligned crop, the 18 psychological/social judgment traits (`trustworthy`, `dominant`, `smart`, `attractive`, `happy`, `outgoing`, `familiar`, `memorable`, `well-groomed`, `alert`, `cute`, `electable`, `privileged`, `liberal`, `godly`, `outdoors`, `smug`, `dorky`), and where the face lands in **Todorov's 2D Valence × Dominance space**.
- **Figure Set 2 (`results/figures_demographics/<name>_demographics_profile.png`)**:
  - Displays the remaining 16 OMI attributes split into **Physical & Demographic Impressions** (`age`, `gender`, `weight`, `typical`, `long-haired`, `hair-color`, `skin-color`, `looks-like-you`, `gay`) and **Perceived Ethnic / Racial Attributions** (`white`, `hispanic`, `middle-eastern`, `islander`, `native`, `asian`, `black`).

> **Scientific Disclaimer**: As emphasized by Todorov et al., these models measure **superficial human perceptual stereotypes** (how an average observer judges a face at first glance), **not** a person's real personality, intelligence, or moral character.
