# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 12:11:48 2025

@author: azaldec

Description:
  Generates calibrated ratiometric macromolecular crowding images and single-cell
  statistics from dual-channel fluorescence microscopy data using the Sed1 crowding biosensor
  (calibrated against PEG4000 % w/v).

Mathematical Model & Inversion:
  Forward sigmoidal crowding calibration curve:
      y = L / (1 + exp(-k * (x - x0))) + b

  Analytical inversion for macromolecular crowding (x, in % w/v PEG4000):
      x = x0 + (1 / k) * ln( (L / (y - b)) - 1 )

  Where:
      y  = Measured fluorescence emission ratio (Channel 2 / Channel 1)
      x  = Inferred macromolecular crowding level (PEG4000 % w/v)
      L  = Dynamic ratio span (4.921)
      x0 = Midpoint / inflection point (9.113 % w/v)
      k  = Logistic slope factor (0.321)
      b  = Baseline ratio offset (2.433)

Processing Pipeline:
  1. Identifies paired multi-channel TIFF images (*channels.tif) and cell masks (*masks.tif).
  2. Extracts Channel 1 (index 1) and Channel 2 (index 0).
  3. Preprocesses cell segmentation masks (filters objects < 50 pixels).
  4. Estimates background thresholds via Gaussian fitting on the lowest 15% of pixels.
  5. Computes the pixel-by-pixel fluorescence ratio (Channel 2 / Channel 1) within masked cells.
  6. Quantifies single-cell morphometrics and average ratios (cell area in µm², mean ratio).
  7. Converts the ratiometric map to macromolecular crowding (% w/v PEG4000) via the inverted 4PL model.
  8. Exports 32-bit TIFFs, publication figures (SVG/PNG), and single-cell CSV tables.

Usage:
  python generate_ratiometric_image_sed1_channels.py "<path_to_data_folder>/*"
"""

import sys
import glob
import numpy as np
import pandas as pd
import tifffile as tif
import scipy.stats as ss
import matplotlib.pyplot as plt
from skimage import restoration, measure, morphology


# ==============================================================================
# Visualization & Image Plotting Helper
# ==============================================================================

def plot_images(img_list, titles=None, cmap="gray"):
    """
    Displays up to 5 images in a single row for visual inspection.

    Parameters:
        img_list (list of ndarray): 2D image arrays to display.
        titles (list of str, optional): Titles for each panel. Length must match img_list.
        cmap (str): Matplotlib colormap name (default: "gray").
    """
    n = len(img_list)
    if n == 0:
        raise ValueError("Image list is empty.")
    if n > 5:
        raise ValueError("Can only plot up to 5 images.")
    
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 2.5),
                             sharex=True, sharey=True, dpi=200)
    
    # Handle single image case
    if n == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.imshow(img_list[i], cmap=cmap)
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=10)
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()


# ==============================================================================
# Background Estimation via Gaussian Fitting
# ==============================================================================

def threshold(image, cutoff_pct=90):
    """
    Estimates a background intensity threshold by fitting a Gaussian distribution
    to the lowest percentile of pixel intensities.

    Parameters:
        image (ndarray): 2D input fluorescence intensity array.
        cutoff_pct (float): Percentile threshold used to isolate background pixels.

    Returns:
        float: Estimated background threshold defined as mean + 2 * std (µ + 2σ).
    """
    pixels = image.ravel()
    cutoff = np.percentile(pixels, cutoff_pct)
    bg_pixels = pixels[pixels <= cutoff]
    
    # Fit normal distribution to background pixel population
    mu, sigma = ss.norm.fit(bg_pixels)
    
    # Plot histogram and Gaussian fit
    counts, bins, _ = plt.hist(pixels, bins=1000, density=True, alpha=0.6, color='gray')
    x = np.linspace(bins.min(), bins.max(), 10000)
    pdf = ss.norm.pdf(x, mu, sigma)
    
    print(f"Background Gaussian Fit -> Mean (μ): {mu:.4f}, Std dev (σ): {sigma:.4f}")

    plt.plot(x, pdf, 'r-', linewidth=2, label=f'Fit: μ={mu:.3f}, σ={sigma:.3f}')
    plt.title('Histogram of Pixel Intensities with Gaussian Fit')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
    return float(mu + 2.0 * sigma)


# ==============================================================================
# Model Inversion: Ratio to Macromolecular Crowding (PEG4000 % w/v)
# ==============================================================================

def y_to_x_image(y_image, background_value=0.001, out_of_range_value=1.0):
    """
    Converts a ratiometric image (y) into a calibrated macromolecular crowding image (x)
    using the inverted 4-parameter logistic model for the Sed1 biosensor:
        x = x0 + (1 / k) * ln((L / (y - b)) - 1)

    Parameters:
        y_image (ndarray): 2D array of fluorescence ratio values.
        background_value (float): Threshold below which pixels are treated as background.
        out_of_range_value (float): Fallback value assigned to unphysical ratios.

    Returns:
        ndarray: Calibrated macromolecular crowding image (PEG4000 % w/v).
    """
    # Sed1 calibration coefficients (calibrated against PEG4000 standards)
    L = 4.921   # Dynamic ratio span
    k = 0.321   # Slope factor
    x0 = 9.113  # Inflection point (% w/v PEG4000)
    b = 2.433   # Baseline ratio offset

    y = y_image.astype(float)

    # Mask background and unphysical values to prevent math domain errors in ln()
    mask_invalid = (y <= background_value) | (y <= b) | (y >= L + b)
    
    # Clamp within open interval (b, L + b)
    y_clipped = np.clip(y, b + 1e-6, L + b - 1e-6)

    # Invert the logistic function for macromolecular crowding
    inner = (L / (y_clipped - b)) - 1.0
    inner = np.maximum(inner, 1e-9)
    x_image = x0 + (1.0 / k) * np.log(inner)
    
    # Assign fallback value to invalid or background pixels
    x_image[mask_invalid] = out_of_range_value

    return x_image


# ==============================================================================
# Single-Cell Morphometric & Ratio Extraction
# ==============================================================================

def calc_avg_cell_ratio(mask, ratio_img):
    """
    Extracts morphological properties and mean fluorescence ratio for each
    segmented cell.

    Parameters:
        mask (ndarray): Binary or labeled segmentation mask.
        ratio_img (ndarray): 2D ratiometric image.

    Returns:
        ndarray: 2D array with columns [cell_label, cell_area_um2, mean_ratio].
    """
    labels, n_labels = measure.label(mask, background=0, return_num=True)
    print(f"Number of segmented cells found: {n_labels}")
    
    cells = measure.regionprops(labels, intensity_image=ratio_img)
    ratio_ints = [cell.mean_intensity for cell in cells]
    areas = [cell.area for cell in cells]
    
    pixel_size = 0.167  # Physical pixel size in microns (µm/pixel)
    
    cell_ratios = np.array(ratio_ints)
    cell_areas = np.array(areas) * (pixel_size ** 2)  # Convert area to µm²
    cell_labels = np.array([cell.label for cell in cells])
    
    return np.column_stack((cell_labels, cell_areas, cell_ratios))


# ==============================================================================
# Main Batch Processing Script
# ==============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_ratiometric_image_sed1_channels.py '<path_pattern>'")
        sys.exit(1)

    directory = sys.argv[1]
    files = [file for file in glob.glob(directory)]

    fluor_imgs = sorted([file for file in files if "channels.tif" in file])
    mask_imgs = sorted([file for file in files if 'masks.tif' in file])

    print(f"Matched {len(fluor_imgs)} fluorescence images with {len(mask_imgs)} mask files.")

    for fluor, mask in zip(fluor_imgs, mask_imgs):
        print(f"\nProcessing:\n  Fluorescence: {fluor}\n  Mask:         {mask}")
        
        # Load multi-channel TIFF: shape (2, height, width)
        # Channel 1: index 1 (reference channel)
        # Channel 2: index 0 (crowding-sensitive channel)
        chan1_arr = tif.imread(fluor)[1].astype(np.float32)
        chan2_arr = tif.imread(fluor)[0].astype(np.float32)
        
        # Load cell mask and filter out small debris (< 50 pixels)
        mask_arr = tif.imread(mask)
        mask_arr = morphology.remove_small_objects(mask_arr, min_size=50)
        cell_mask = mask_arr > 0 

        # Display raw channels and cell mask
        plot_images([chan1_arr, chan2_arr, mask_arr], titles=["Chan 1", "Chan 2", "Cell Mask"])
        
        # Estimate background thresholds via Gaussian fitting on the lowest 15% of pixels
        thresh1 = threshold(chan1_arr, cutoff_pct=15)
        thresh2 = threshold(chan2_arr, cutoff_pct=15)
        
        # Background subtraction with lower-bound clipping
        chan1_corr = np.clip(chan1_arr - thresh1, 0.1, None)
        chan2_corr = np.clip(chan2_arr - thresh2, 0.0001, None)
        
        # Compute ratiometric image (Chan 2 / Chan 1) inside cell mask
        ratio_image = np.full(chan1_corr.shape, 0.001, dtype=np.float32)
        ratio_image[cell_mask] = chan2_corr[cell_mask] / chan1_corr[cell_mask]
        
        # Single-cell extraction and CSV export
        single_cell_data = calc_avg_cell_ratio(mask_arr, ratio_image)
        data_df = pd.DataFrame(single_cell_data)
        data_df.columns = ['Cell_label', 'Cell_area', 'Cell_avg_ratio']
        name = 'ctrl_H2d1'
        data_df['Sample'] = 'ctrl'
        data_df['File'] = name
        print("\nSingle-Cell Metrics Table:")
        print(data_df)
        
        out_prefix = directory[:-5]
        csv_out = out_prefix + name + '_singlecell_analysis.csv'
        data_df.to_csv(csv_out, index=False)
        print(f"Saved single-cell analysis to: {csv_out}")
        
        # Display and save ratiometric TIFF
        plot_images([chan1_corr, chan2_corr, ratio_image], titles=["Chan 1 Corr", "Chan 2 Corr", "Ratio Map"])
        tif.imwrite(out_prefix + "ratiometric_image.tif", ratio_image)
        
        # Convert fluorescence ratio map to calibrated macromolecular crowding (% w/v PEG4000)
        mk_image = np.zeros_like(ratio_image)
        mk_image[cell_mask] = y_to_x_image(ratio_image[cell_mask])
        
        # Render publication dual-panel figure: Ratio Image vs. Calibrated Crowding Map
        fig, axes = plt.subplots(1, 2, figsize=(6, 2.5), dpi=300, sharex=True, sharey=True)
        
        # Panel 1: Ratiometric Image
        rat1 = axes[0].imshow(ratio_image, vmin=0.2, vmax=2.0, cmap='turbo')
        bar1 = plt.colorbar(rat1, ax=axes[0], fraction=0.046, pad=0.04) 
        bar1.set_label('488nm / 561nm Ratio', fontsize=9) 
        axes[0].set_title("Fluorescence Ratio", fontsize=10)
        axes[0].axis("off")
        
        # Panel 2: Calibrated Macromolecular Crowding Map
        rat2 = axes[1].imshow(mk_image, vmin=0.0, vmax=20.0, cmap='turbo')
        bar2 = plt.colorbar(rat2, ax=axes[1], fraction=0.046, pad=0.04) 
        bar2.set_label('PEG4000% (w/v)', fontsize=9) 
        axes[1].set_title("Macromolecular Crowding", fontsize=10)
        axes[1].axis("off")
        
        fig.tight_layout()
        plot_out = out_prefix + 'ratiometric_image_plot.svg'
        plt.savefig(plot_out, dpi=300)
        print(f"Saved publication plot to: {plot_out}")
        plt.show()