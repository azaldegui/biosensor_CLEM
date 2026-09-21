# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 10:03:18 2025

@author: azaldec

Description:
  Correlative Light and Electron Microscopy (CLEM) Overlay Visualization.

  This display script renders high-resolution multi-panel figures overlaying
  registered fluorescence microscopy (FM) maps (e.g., biosensor ratios or
  labeled proteins) directly onto their corresponding electron microscopy (EM)
  search maps or tomographic overviews.

  Note:
    Spatial registration/warping is performed prior to this script (e.g., via
    MATLAB DL_CLEM_analysis.m or Python registration routines). This script
    focuses on visual rendering, threshold masking, alpha-blending, and publication export.

Workflow:
  1. Loads the registered/warped fluorescence image and the base EM image.
  2. Masks out low-intensity background/extracellular fluorescence pixels (<= 0.05).
  3. Renders a 3-panel figure:
     - Panel 1: Pseudocolored fluorescence intensity map (Turbo colormap).
     - Panel 2: Grayscale ultrastructural EM image.
     - Panel 3: Composite CLEM overlay (FM blended onto EM at 50% opacity).
  4. Formats axes with 1:1 physical aspect ratios and saves vector SVG and high-res PNG.

Usage:
  python fancy_overlay.py "<path_to_directory_containing_images>/"
"""

import os
import sys
import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap


# ==============================================================================
# Image Loading & Configuration
# ==============================================================================

# Parse target directory from command-line argument
if len(sys.argv) < 2:
    print("Usage: python fancy_overlay.py <path_to_directory>/")
    sys.exit(1)

path = sys.argv[1]
print(f"Working directory: {path}")

# Default filenames for registered fluorescence and base EM images
image_1_fn = 'SUM_grid3_W_1r_635nm_warped_to_MAX_grid3_W_1r_WL_warped_to_W1r_163413_searchmap.tif'
image_2_fn = 'W1r_163413_searchmap.tif'

# Load registered fluorescence microscopy (FM) image
fm_path = os.path.join(path, image_1_fn)
FM_image = tif.imread(fm_path)
print(f"Loaded Fluorescence Image: {fm_path} (shape: {FM_image.shape})")

# Load corresponding structural Electron Microscopy (EM) image
em_path = os.path.join(path, image_2_fn)
EM_image = tif.imread(em_path)
print(f"Loaded EM Search Map:     {em_path} (shape: {EM_image.shape})")


# ==============================================================================
# Colormap & Background Masking Setup
# ==============================================================================

# Mask out extracellular background pixels (intensity <= 0.05) to render them completely transparent
mask = np.ma.masked_where(FM_image <= 0.05, FM_image)

# Initialize 'turbo' colormap and set masked (bad) values to full transparency
turbo_cmap = get_cmap('turbo')
turbo_cmap.set_bad(color='white', alpha=0.0)


# ==============================================================================
# Multi-Panel CLEM Figure Rendering
# ==============================================================================

# Create a 1-row by 3-column figure layout with shared coordinate systems
fig, axes = plt.subplots(ncols=3, figsize=(5.5, 2.0), dpi=300, sharex=True, sharey=True)
ax = axes.ravel()

# Panel 1: Isolated Fluorescence Map
FM = ax[0].imshow(mask, cmap=turbo_cmap, norm=Normalize(vmin=0.0, vmax=1.0))
# Optional colorbar code:
# FM_intensity = plt.colorbar(FM, ax=ax[0])
# FM_intensity.set_label("YFP/CFP")

# Panel 2: Isolated Electron Microscopy (EM) Search Map
EM = ax[1].imshow(EM_image, vmin=0.1, vmax=0.7, cmap='gray')

# Panel 3: Combined CLEM Overlay (FM blended over EM background with 50% opacity)
EM_OL = ax[2].imshow(EM_image, vmin=0.1, vmax=0.7, cmap='gray')
FM_OL = ax[2].imshow(mask, alpha=0.5, cmap=turbo_cmap, norm=Normalize(vmin=0.0, vmax=1.0))

# Configure 1:1 aspect ratio so pixels and structures maintain correct geometric proportions
for a in ax:
    a.set_aspect('equal', adjustable='box')

# Remove axis lines, ticks, and labels for clean publication-style presentation
for ii in range(3):
    ax[ii].set_axis_off()

fig.tight_layout()


# ==============================================================================
# File Export
# ==============================================================================

# Output file paths
out_svg = os.path.join(path, image_1_fn[:-4] + '_ratio_overlay_plot.svg')
out_png = os.path.join(path, image_1_fn[:-4] + '_ratio_overlay_plot.png')

plt.savefig(out_svg, dpi=300)
plt.savefig(out_png, dpi=300)
print(f"Saved CLEM overlays:\n  {out_svg}\n  {out_png}")

plt.show()
