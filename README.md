# Local Estimator of Social Judgments from Faces (Todorov & Peterson OMI)

A self-contained Python pipeline to estimate **34 perceived social, physical, and demographic dimensions** (e.g., *trustworthy, dominant, smart, attractive, happy, outgoing, electable*, etc.) plus **Todorov & Oh's (2021) 2D Valence & Dominance composite scores** from naturalistic photographs of faces.

> **Note on Scope**: This project is a **simplified, turnkey wrapper designed for easy local application and replication** on experimental or personal face photos. It streamlines the underlying models into a lightweight, one-click workflow (automatic landmark alignment, fast CPU inference, batch processing, and dual diagnostic reporting).

👉 **For full setup and replication instructions, see [INSTRUCTIONS.md](./INSTRUCTIONS.md).**

---

## Attribution & Acknowledgments

This pipeline builds directly upon the ideas, architectures, and open datasets developed in the following repositories and papers:

1. **Model Backbone & Inference Engine**:
   - Repository: [**`kiante-fernandez/face-trait-transformer`**](https://github.com/kiante-fernandez/face-trait-transformer) (by Kianté Fernandez)
   - Model on Hugging Face: [`kiante/face-trait-transformer`](https://huggingface.co/kiante/face-trait-transformer)
   - *Contribution*: Provided the DINOv2 vision transformer architecture fine-tuned on human perceptual judgments, along with the core `TraitPredictor` inference API.

2. **One Million Impressions (OMI) Dataset**:
   - Repository: [**`jcpeterson/omi`**](https://github.com/jcpeterson/omi)
   - Paper: Peterson, J. C., Uddenberg, S., Griffiths, T. L., Todorov, A., & Suchow, J. W. (2022). *Deep models of superficial face judgments*. Proceedings of the National Academy of Sciences (PNAS), 119(17), e2115228119.
   - *Contribution*: Collected the ~1.3M human perceptual ratings across 34 attributes on synthetic face stimuli that form the ground truth for these models.

3. **Face Alignment Geometry**:
   - Repository: [**`NVlabs/ffhq-dataset`**](https://github.com/NVlabs/ffhq-dataset) (NVIDIA)
   - *Contribution*: The facial landmark alignment and reflection-padded cropping logic in `ffhq_align.py` implements NVIDIA's canonical FFHQ geometry (via MediaPipe FaceMesh) so that wild photos match the framing of the training stimuli.

4. **Theoretical Framework for Social Face Perception**:
   - Paper: Todorov, A., & Oh, D. (2021). *The structure and perceptual basis of social judgments from faces*. Advances in Experimental Social Psychology, 63, 189–245.
   - Paper: Oosterhof, N. N., & Todorov, A. (2008). *The functional basis of face evaluation*. PNAS, 105(32), 11087–11092.
   - *Contribution*: Defined the foundational 2D structural space of face evaluation (**Valence / Trustworthiness** and **Power / Dominance**).

---

## Quick Start

1. Place your face photographs (`.jpg`, `.png`, `.webp`) inside the **`input_faces/`** folder.
2. Run:
   - **Windows**: Double-click **`run_estimator.bat`** (or run `python -m pip install -r requirements.txt` followed by `python estimate_faces.py input_faces`).
   - **macOS / Linux**: Run `bash run_estimator.sh`.
3. Check the **`results/`** folder for:
   - `results/social_judgments.csv` — Full table of all 34 OMI dimensions + Todorov 2D Valence & Dominance scores.
   - `results/figures/<image>_profile.png` — **Figure Set 1**: Original vs. FFHQ-Aligned Face, 18 Social Judgment Traits, and Todorov 2D Social Face Space.
   - `results/figures_demographics/<image>_demographics_profile.png` — **Figure Set 2**: 9 Physical/Demographic Impressions + 7 Perceived Ethnic/Racial Attributions.
   - `results/aligned_faces/<image>_aligned.png` — 512×512 FFHQ landmark-aligned crops.

---

## Scientific Background & References

1. **Todorov, A., & Oh, D. (2021)**. *The structure and perceptual basis of social judgments from faces*. Advances in Experimental Social Psychology, 63, 189–245.
2. **Peterson, J. C., Uddenberg, S., Griffiths, T. L., Todorov, A., & Suchow, J. W. (2022)**. *Deep models of superficial face judgments*. Proceedings of the National Academy of Sciences (PNAS), 119(17), e2115228119. ([`jcpeterson/omi`](https://github.com/jcpeterson/omi))
3. **Model Backend**: Uses [`kiante/face-trait-transformer`](https://huggingface.co/kiante/face-trait-transformer) (DINOv2 vision backbone fine-tuned on the 1,004 OMI stimuli and ~1.3M human ratings) paired with automatic MediaPipe + NVIDIA FFHQ facial landmark alignment ([`ffhq_align.py`](./ffhq_align.py)).

> **Disclaimer**: These models predict *average human perceptual biases* (first-impression stereotypes from facial appearance), **not** an individual's actual character or competence.
