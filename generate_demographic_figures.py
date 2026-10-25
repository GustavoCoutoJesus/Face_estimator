"""
Generates the complementary diagnostic figures for the remaining 16 OMI traits
(Physical / Demographic impressions and Perceived Racial / Ethnic categories)
using the results already saved in results/social_judgments.csv.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PHYSICAL_TRAITS = [
    "age",
    "gender",
    "weight",
    "typical",
    "long-haired",
    "hair-color",
    "skin-color",
    "looks-like-you",
    "gay",
]

ETHNICITY_TRAITS = [
    "white",
    "hispanic",
    "middle-eastern",
    "islander",
    "native",
    "asian",
    "black",
]


def render_demographic_figure(
    orig_img_path: Path,
    aligned_img_path: Path,
    row: pd.Series,
    out_path: Path,
) -> None:
    orig_img = Image.open(orig_img_path).convert("RGB")
    aligned_img = (
        Image.open(aligned_img_path).convert("RGB")
        if aligned_img_path.exists()
        else orig_img
    )

    fig = plt.figure(figsize=(16, 6.5), dpi=150)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.35, 1.25], wspace=0.32, hspace=0.28)

    # Panel 1A: Original Image
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(orig_img)
    ax_orig.set_title(f"Original Input\n({orig_img_path.name})", fontsize=10, fontweight="bold")
    ax_orig.axis("off")

    # Panel 1B: Aligned Crop
    ax_align = fig.add_subplot(gs[1, 0])
    ax_align.imshow(aligned_img)
    align_status = "FFHQ Landmark Aligned" if row.get("face_aligned", True) else "Center Crop"
    ax_align.set_title(f"Model Input ({align_status})", fontsize=10, fontweight="bold")
    ax_align.axis("off")

    # Panel 2: Physical & Demographic Impressions
    ax_phys = fig.add_subplot(gs[:, 1])
    phys_sorted = sorted(PHYSICAL_TRAITS, key=lambda k: float(row.get(k, 0.0)))
    phys_vals = [float(row.get(k, 0.0)) for k in phys_sorted]

    bars = ax_phys.barh(phys_sorted, phys_vals, color="#2b5c8f", height=0.62, alpha=0.88)
    ax_phys.axvline(50.0, color="#666666", linestyle="--", linewidth=1.0, label="Neutral (50)")
    ax_phys.set_xlim(0, 105)
    ax_phys.set_xlabel("Estimated Perceived Score (0–100 Scale)", fontsize=10)
    ax_phys.set_title(
        "Physical & Demographic Impressions",
        fontsize=11,
        fontweight="bold",
    )
    ax_phys.grid(axis="x", linestyle=":", alpha=0.5)

    for bar, v in zip(bars, phys_vals):
        ax_phys.text(
            min(v + 1.5, 98.0),
            bar.get_y() + bar.get_height() / 2.0,
            f"{v:.1f}",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
        )

    # Panel 3: Perceived Ethnic / Racial Categories
    ax_eth = fig.add_subplot(gs[:, 2])
    eth_sorted = sorted(ETHNICITY_TRAITS, key=lambda k: float(row.get(k, 0.0)))
    eth_vals = [float(row.get(k, 0.0)) for k in eth_sorted]

    colors = [
        "#4c72b0" if v < 40 else "#c44e52"
        for v in eth_vals
    ]
    bars_eth = ax_eth.barh(eth_sorted, eth_vals, color=colors, height=0.62, alpha=0.88)
    ax_eth.set_xlim(0, 105)
    ax_eth.set_xlabel("Observer Attribution Score (0–100)", fontsize=10)
    ax_eth.set_title(
        "Perceived Ethnic / Racial Attribution",
        fontsize=11,
        fontweight="bold",
    )
    ax_eth.grid(axis="x", linestyle=":", alpha=0.5)

    for bar, v in zip(bars_eth, eth_vals):
        ax_eth.text(
            min(v + 1.5, 98.0),
            bar.get_y() + bar.get_height() / 2.0,
            f"{v:.1f}",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
        )

    fig.suptitle(
        f"Demographic & Physical Profile — {orig_img_path.name}  "
        "(Observer Perceptual Biases, Peterson & Todorov OMI)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> None:
    csv_path = Path("results/social_judgments.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    out_dir = Path("results/figures_demographics")
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        img_file = row["image_file"]
        orig_path = Path(row["image_path"])
        if not orig_path.exists():
            orig_path = Path("input_faces") / img_file

        stem = Path(img_file).stem
        aligned_path = Path(f"results/aligned_faces/{stem}_aligned.png")
        out_fig_path = out_dir / f"{stem}_demographics_profile.png"

        print(f"Generating demographic figure for: {img_file} -> {out_fig_path.name}")
        render_demographic_figure(orig_path, aligned_path, row, out_fig_path)

    print(f"\nAll demographic figures generated successfully in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
