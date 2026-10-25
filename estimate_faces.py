"""
Local Estimator of Perceived Social Judgments from Naturalistic Face Photos.

Based on:
- Todorov, A., & Oh, D. (2021). The structure and perceptual basis of social
  judgments from faces. Advances in Experimental Social Psychology, 63, 189-245.
- Peterson, J. C., Uddenberg, S., Griffiths, T. L., Todorov, A., & Suchow, J. W.
  (2022). Deep models of superficial face judgments. PNAS, 119(17), e2115228119.
  (One Million Impressions [OMI] dataset, 34 attributes)
- Model backend: `kiante/face-trait-transformer` (DINOv2 ensemble trained on OMI).

NOTE: As emphasized by Todorov et al., these scores reflect average human
perceptual biases (first impressions / superficial stereotypes from facial
appearance), NOT a person's actual character or personality.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from ffhq_align import align_face_ffhq

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Core psychological/social judgment traits in OMI (highlighted in plots & console)
SOCIAL_TRAITS = [
    "trustworthy",
    "dominant",
    "smart",
    "attractive",
    "happy",
    "outgoing",
    "familiar",
    "memorable",
    "well-groomed",
    "alert",
    "cute",
    "electable",
    "privileged",
    "liberal",
    "godly",
    "outdoors",
    "smug",
    "dorky",
]


def collect_image_paths(inputs: Iterable[str]) -> List[Path]:
    """Resolve input files and folders into a sorted list of image paths."""
    paths: List[Path] = []
    for item in inputs:
        p = Path(item).expanduser().resolve()
        if p.is_dir():
            for child in sorted(p.iterdir()):
                if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
                    paths.append(child)
        elif p.is_file():
            paths.append(p)
        else:
            print(f"[WARNING] Path not found, skipping: {p}", file=sys.stderr)
    return paths


def compute_todorov_composites(series: pd.Series) -> dict:
    """
    Compute Todorov & Oh (2021) 2D structural summary indices (0-100 scale)
    from the 34 OMI trait predictions:
      1. Valence / Trustworthiness dimension
      2. Power / Dominance dimension
    """
    valence_pos = [t for t in ["trustworthy", "happy", "outgoing", "cute"] if t in series]
    valence_neg = [t for t in ["smug"] if t in series]

    valence_vals = [float(series[t]) for t in valence_pos] + [
        100.0 - float(series[t]) for t in valence_neg
    ]
    valence_index = float(np.mean(valence_vals)) if valence_vals else float("nan")

    dom_traits = [t for t in ["dominant", "alert", "smart", "electable"] if t in series]
    dominance_index = (
        float(np.mean([float(series[t]) for t in dom_traits]))
        if dom_traits
        else float("nan")
    )

    return {
        "todorov_valence_index": round(valence_index, 2),
        "todorov_dominance_index": round(dominance_index, 2),
    }


def save_diagnostic_plot(
    orig_img: Image.Image,
    aligned_img: Image.Image,
    face_detected: bool,
    scores: pd.Series,
    composites: dict,
    image_name: str,
    out_path: Path,
) -> None:
    """Create a 3-panel diagnostic report (Alignment, 34-Trait Profile, Todorov 2D Space)."""
    fig = plt.figure(figsize=(16, 6.5), dpi=150)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.45, 1.15], wspace=0.32, hspace=0.28)

    # Panel 1A: Original image
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(orig_img)
    ax_orig.set_title(f"Original Input\n({image_name})", fontsize=10, fontweight="bold")
    ax_orig.axis("off")

    # Panel 1B: FFHQ Aligned crop
    ax_align = fig.add_subplot(gs[1, 0])
    ax_align.imshow(aligned_img)
    align_status = "FFHQ Landmark Aligned" if face_detected else "Fallback Center Crop"
    ax_align.set_title(f"Model Input ({align_status})", fontsize=10, fontweight="bold")
    ax_align.axis("off")

    # Panel 2: Horizontal bar plot of social traits (0-100 scale)
    ax_bars = fig.add_subplot(gs[:, 1])
    traits_present = [t for t in SOCIAL_TRAITS if t in scores.index]
    # Sort by score ascending for horizontal bar chart
    traits_sorted = sorted(traits_present, key=lambda k: float(scores[k]))
    vals = [float(scores[k]) for k in traits_sorted]

    colors = [
        "#1f77b4" if k not in ("trustworthy", "dominant") else "#d62728"
        for k in traits_sorted
    ]
    bars = ax_bars.barh(traits_sorted, vals, color=colors, height=0.68, alpha=0.88)
    ax_bars.axvline(50.0, color="#666666", linestyle="--", linewidth=1.0, label="Neutral (50)")
    ax_bars.set_xlim(0, 105)
    ax_bars.set_xlabel("Estimated Perceived Score (0–100 OMI Scale)", fontsize=10)
    ax_bars.set_title(
        "Estimated Social Judgments (Peterson & Todorov OMI)",
        fontsize=11,
        fontweight="bold",
    )
    ax_bars.grid(axis="x", linestyle=":", alpha=0.5)
    for bar, v in zip(bars, vals):
        ax_bars.text(
            min(v + 1.5, 99.0),
            bar.get_y() + bar.get_height() / 2.0,
            f"{v:.1f}",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
        )

    # Panel 3: Todorov 2D Space (Valence / Trustworthiness vs Dominance / Power)
    ax_2d = fig.add_subplot(gs[:, 2])
    trust_val = float(scores.get("trustworthy", composites["todorov_valence_index"]))
    dom_val = float(scores.get("dominant", composites["todorov_dominance_index"]))
    comp_val = composites["todorov_valence_index"]
    comp_dom = composites["todorov_dominance_index"]

    ax_2d.axhline(50, color="#888888", linestyle="--", linewidth=1.0)
    ax_2d.axvline(50, color="#888888", linestyle="--", linewidth=1.0)
    ax_2d.set_xlim(0, 100)
    ax_2d.set_ylim(0, 100)
    ax_2d.set_aspect("equal", adjustable="box")

    # Quadrant labels
    quad_style = dict(fontsize=8, color="#666666", alpha=0.85, ha="center", va="center")
    ax_2d.text(75, 75, "High Valence\nHigh Dominance", **quad_style)
    ax_2d.text(25, 75, "Low Valence\nHigh Dominance", **quad_style)
    ax_2d.text(75, 25, "High Valence\nLow Dominance", **quad_style)
    ax_2d.text(25, 25, "Low Valence\nLow Dominance", **quad_style)

    # Plot primary trait point (Trustworthy vs Dominant) and composite index
    ax_2d.scatter(
        [trust_val],
        [dom_val],
        s=150,
        color="#d62728",
        edgecolor="black",
        zorder=5,
        label=f"Trustworthy ({trust_val:.1f}) × Dominant ({dom_val:.1f})",
    )
    ax_2d.scatter(
        [comp_val],
        [comp_dom],
        s=110,
        marker="D",
        color="#1f77b4",
        edgecolor="black",
        zorder=4,
        label=f"2D Composite ({comp_val:.1f}, {comp_dom:.1f})",
    )

    ax_2d.set_xlabel("Perceived Valence / Trustworthiness (0–100)", fontsize=10)
    ax_2d.set_ylabel("Perceived Power / Dominance (0–100)", fontsize=10)
    ax_2d.set_title(
        "Todorov & Oh (2021) 2D Face Space",
        fontsize=11,
        fontweight="bold",
    )
    ax_2d.grid(True, linestyle=":", alpha=0.4)
    ax_2d.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), fontsize=8.5, frameon=True)

    fig.suptitle(
        f"Perceived Social Judgment Profile — {image_name}  "
        "(Models Human Perceptual Stereotypes, Not Character)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate 34 OMI social judgments & Todorov 2D scores on naturalistic face images."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["input_faces"],
        help="One or more image file paths or directories containing face images (default: ./input_faces).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="results",
        help="Directory to save CSV scores, aligned faces, and diagnostic figures (default: ./results).",
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Disable automatic FFHQ facial landmark alignment (use only if images are already FFHQ-aligned).",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip generating per-image diagnostic PNG figures.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to use (e.g. 'cpu' or 'cuda'). Auto-detected by default.",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Also run the 4.23 GB DINOv2 ViT-G/14 518px backbone in addition to the fine-tuned ViT-L/14 model.",
    )
    args = parser.parse_args()

    image_paths = collect_image_paths(args.inputs)
    if not image_paths:
        print("[ERROR] No valid image files found in the provided inputs.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir).expanduser().resolve()
    aligned_dir = out_dir / "aligned_faces"
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)

    import os
    import warnings
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    warnings.filterwarnings("ignore", message=".*xFormers is not available.*")

    print(f"Found {len(image_paths)} image(s). Loading `kiante/face-trait-transformer`...")
    import torch
    from face_trait_transformer import TraitPredictor

    # Patch torch.hub.load so fine-tuned checkpoints (which already carry full backbone weights)
    # do not redundantly download a separate 1.2 GB base checkpoint from dl.fbaipublicfiles.com.
    _orig_hub_load = torch.hub.load

    def _patched_hub_load(repo_or_dir, model, *h_args, **h_kwargs):
        if "dinov2" in str(repo_or_dir) and model == "dinov2_vitl14":
            h_kwargs.setdefault("pretrained", False)
        return _orig_hub_load(repo_or_dir, model, *h_args, **h_kwargs)

    torch.hub.load = _patched_hub_load

    kwargs = {"device": args.device} if args.device else {}
    local_bundle = Path(__file__).resolve().parent / "model_bundle"
    if (local_bundle / "manifest.json").exists():
        print(f"Using local model bundle from: {local_bundle}")
        predictor = TraitPredictor.from_bundle(local_bundle, **kwargs)
    else:
        predictor = TraitPredictor.from_pretrained("kiante/face-trait-transformer", **kwargs)

    if not args.ensemble:
        # Use the end-to-end fine-tuned DINOv2 ViT-L/14 model (already in the 1.28 GB bundle)
        # without requiring the extra 4.23 GB ViT-G/14 download on CPU.
        predictor._backbone_groups = {}

    rows = []
    for idx, img_path in enumerate(image_paths, start=1):
        print(f"\n[{idx}/{len(image_paths)}] Processing: {img_path.name}")
        orig_img = Image.open(img_path).convert("RGB")

        if args.no_align:
            aligned_img = orig_img
            face_detected = True
        else:
            aligned_img, face_detected = align_face_ffhq(orig_img, output_size=512)
            if not face_detected:
                print(
                    f"  [!] No face landmarks detected in {img_path.name}; used fallback center-crop."
                )

        aligned_save_path = aligned_dir / f"{img_path.stem}_aligned.png"
        aligned_img.save(aligned_save_path)

        # Score the aligned image with face-trait-transformer
        scores: pd.Series = predictor.predict(str(aligned_save_path))
        composites = compute_todorov_composites(scores)

        # Console summary of key dimensions
        print(
            f"  -> Trustworthy: {scores.get('trustworthy', float('nan')):.1f} | "
            f"Dominant: {scores.get('dominant', float('nan')):.1f} | "
            f"Smart: {scores.get('smart', float('nan')):.1f} | "
            f"Attractive: {scores.get('attractive', float('nan')):.1f} | "
            f"Happy: {scores.get('happy', float('nan')):.1f} | "
            f"Outgoing: {scores.get('outgoing', float('nan')):.1f}"
        )
        print(
            f"  -> Todorov 2D Composite - Valence: {composites['todorov_valence_index']:.1f}, "
            f"Dominance: {composites['todorov_dominance_index']:.1f}"
        )

        if not args.no_figures:
            fig_social_path = figures_dir / f"{img_path.stem}_profile.png"
            save_diagnostic_plot(
                orig_img=orig_img,
                aligned_img=aligned_img,
                face_detected=face_detected,
                scores=scores,
                composites=composites,
                image_name=img_path.name,
                out_path=fig_social_path,
            )
            from generate_demographic_figures import render_demographic_figure
            fig_demo_dir = out_dir / "figures_demographics"
            fig_demo_dir.mkdir(parents=True, exist_ok=True)
            fig_demo_path = fig_demo_dir / f"{img_path.stem}_demographics_profile.png"
            row_dict_temp = {
                "image_file": img_path.name,
                "image_path": str(img_path),
                "face_aligned": face_detected,
                **{k: float(v) for k, v in scores.items() if k != "filename"},
            }
            render_demographic_figure(
                orig_img_path=img_path,
                aligned_img_path=aligned_save_path,
                row=pd.Series(row_dict_temp),
                out_path=fig_demo_path,
            )
            print(f"  -> Saved social profile: {fig_social_path}")
            print(f"  -> Saved demographic profile: {fig_demo_path}")

        row_dict = {
            "image_file": img_path.name,
            "image_path": str(img_path),
            "face_aligned": face_detected,
            **composites,
            **{
                k: round(float(v), 3)
                for k, v in scores.items()
                if k != "filename"
            },
        }
        rows.append(row_dict)

    df_out = pd.DataFrame(rows)
    csv_path = out_dir / "social_judgments.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\nDone! Saved all 34 OMI dimensions + Todorov 2D scores to:\n  {csv_path}")


if __name__ == "__main__":
    main()
