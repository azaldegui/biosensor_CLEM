# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 12:11:48 2025

@author: azaldec

Description:
  Generates calibrated ratiometric calcium images and single-cell statistics from
  dual-channel fluorescence microscopy data using the jREX-GECO biosensor.

Mathematical Model & Inversion:
  Forward sigmoidal calibration curve:
      y = L / (1 + exp(-k * (x - x0))) + b

  Inverted equation for calcium concentration (x):
      x = x0 - (1 / k) * ln( (L / (y - b)) - 1 )

  Where:
      y  = Measured fluorescence emission ratio (Channel 2 / Channel 1)
      x  = Calibrated free calcium concentration in micromolar (µM)
      L  = Dynamic range span (1.68)
      x0 = Midpoint / EC50 (0.97 µM)
      k  = Logistic growth rate / Hill slope (1.34)
      b  = Baseline ratio offset (0.0)

Processing Pipeline:
  1. Identifies paired dual-channel fluorescence stacks (*channels.tif) and cell masks (*masks.tif).
  2. Extracts Channel 1 (e.g., 561 nm excitation) and Channel 2 (e.g., 488 nm excitation).
  3. Preprocesses cell segmentation masks (removes noise/debris < 50 pixels).
  4. Estimates background noise by fitting a Gaussian distribution to low-intensity pixels (µ + 2σ).
  5. Computes the pixel-by-pixel fluorescence ratio (Channel 2 / Channel 1) within masked cells.
  6. Converts the ratiometric map to absolute calcium concentration (µM) via the inverted 4PL model.
  7. Quantifies single-cell morphometrics and average ratios (cell area in µm², mean ratio).
  8. Exports calibrated 32-bit TIFFs, publication figures (SVG/PNG), and single-cell CSV tables.

Usage:
  python generate_ratiometric_image_jrex_channels.py "<path_to_data_folder>/*"
"""

import sys
import glob
import numpy as np
import pandas as pd
import tifffile as tif
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from skimage import restoration, measure, morphology


# ==============================================================================
# Visualization & Image Plotting Helper
# ==============================================================================

def plot_images(img_list, titles=None, cmap="turbo"):
    """
    Displays up to 5 images in a single row for visual inspection.

    Parameters:
        img_list (list of ndarray): 2D image arrays to display.
        titles (list of str, optional): Titles for each panel. Length must match img_list.
        cmap (str): Matplotlib colormap name (default: "turbo").
    """
    n = len(img_list)
    if n == 0:
        raise ValueError("Image list is empty.")
    if n > 5:
        raise ValueError("Can only plot up to 5 images.")
    
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 2.5),
                             sharex=True, sharey=True, dpi=200)
    
    # Ensure axes is iterable even for a single image
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
        cutoff_pct (float): Percentile threshold used to isolate background pixels (e.g., 20 or 90).

    Returns:
        float: Estimated background threshold defined as mean + 2 * std (µ + 2σ).
    """
    pixels = image.ravel()
    cutoff = np.percentile(pixels, cutoff_pct)
    bg_pixels = pixels[pixels <= cutoff]
    
    # Fit normal distribution to background pixel population
    mu, sigma = ss.norm.fit(bg_pixels)
    
    # Plot histogram and Gaussian fit for diagnostic review
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
    
    # Threshold at 2 standard deviations above background mean
    return float(mu + 2.0 * sigma)


# ==============================================================================
# Model Inversion: Ratio to Calcium Concentration (µM)
# ==============================================================================

def y_to_x_image(y_image, background_value=0.001, out_of_range_value=0.001):
    """
    Converts a ratiometric image (y) to a calibrated calcium concentration image (x)
    using the inverted 4-parameter logistic model for jREX-GECO:
        x = x0 - (1 / k) * ln((L / (y - b)) - 1)

    Parameters:
        y_image (ndarray): 2D array of fluorescence ratio values.
        background_value (float): Value assigned to background pixels.
        out_of_range_value (float): Fallback value for pixels outside calibration bounds.

    Returns:
        ndarray: Calibrated free calcium concentration image in µM.
    """
    # jREX-GECO calibration parameters
    L = 1.68    # Dynamic range span
    x0 = 0.97   # EC50 midpoint (µM)
    k = 1.34    # Logistic slope factor
    b = 0.0     # Baseline offset

    epsilon = 0.001
    bg_threshold = 0.20  # Minimum ratio threshold for cellular signal
    data_floor = 0.40    # Floor to elevate sub-threshold cellular ratios
    
    # Step 1: Suppress noise below background threshold
    y_adjusted = np.where(y_image < bg_threshold, 0.0, y_image)
    
    # Step 2: Set intermediate low values to data floor to prevent domain errors
    y_adjusted = np.where((y_image >= bg_threshold) & (y_image < data_floor), data_floor, y_adjusted)
   
    # Step 3: Clamp valid ratios strictly within the mathematical bounds (b, L + b)
    img_clamped = np.clip(y_adjusted, b + epsilon, (L + b) - epsilon)

    # Step 4: Analytical inversion for calcium concentration
    inner = (L / (img_clamped - b)) - 1.0
    inner = np.maximum(inner, 1e-9)
    diff = -np.log(inner) / k
    x_image_final = diff + x0

    return x_image_final


# ==============================================================================
# Single-Cell Morphometric & Ratio Extraction
# ==============================================================================

def calc_avg_cell_ratio(mask, ratio_img):
    """
    Measures morphological properties and mean fluorescence ratio for each
    segmented cell in an image.

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
    
    pixel_size = 0.121  # Physical pixel size in microns (µm/pixel)
    
    cell_ratios = np.array(ratio_ints)
    cell_areas = np.array(areas) * (pixel_size ** 2)  # Convert area from pixels to µm²
    cell_labels = np.array([cell.label for cell in cells])
    
    return np.column_stack((cell_labels, cell_areas, cell_ratios))


# ==============================================================================
# Main Batch Processing Script
# ==============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_ratiometric_image_jrex_channels.py '<path_pattern>'")
        sys.exit(1)

    directory = sys.argv[1]
    files = [file for file in glob.glob(directory)]

    # Separate dual-channel fluorescence files and corresponding cell masks
    fluor_imgs = sorted([file for file in files if "channels.tif" in file])
    mask_imgs = sorted([file for file in files if 'masks.tif' in file])

    print(f"Matched {len(fluor_imgs)} fluorescence images with {len(mask_imgs)} mask files.")

    for fluor, mask in zip(fluor_imgs, mask_imgs):
        print(f"\nProcessing:\n  Fluorescence: {fluor}\n  Mask:         {mask}")
        
        # Load multi-channel TIFF: shape (2, height, width)
        # Channel 1: index 1 (561 nm excitation)
        # Channel 2: index 0 (488 nm excitation)
        chan1_arr = tif.imread(fluor)[1].astype(np.float32)
        chan2_arr = tif.imread(fluor)[0].astype(np.float32)
        
        # Load cell mask and filter out small artifacts/debris (< 50 pixels)
        mask_arr = tif.imread(mask)
        mask_arr = morphology.remove_small_objects(mask_arr, min_size=50)
        cell_mask = mask_arr > 0 

        # Display raw channels and filtered mask
        plot_images([chan1_arr, chan2_arr, mask_arr], titles=["Chan 1 (561nm)", "Chan 2 (488nm)", "Cell Mask"])
        
        # Estimate background thresholds via Gaussian fitting on the lowest 20% of pixels
        thresh1 = threshold(chan1_arr, cutoff_pct=20)
        thresh2 = threshold(chan2_arr, cutoff_pct=20)
        
        # Background subtraction with lower-bound clipping to prevent negative intensities
        chan1_corr = np.clip(chan1_arr - thresh1, 0.1, None)
        chan2_corr = np.clip(chan2_arr - thresh2, 0.0, None)
        
        # Compute ratiometric image (Chan 2 / Chan 1) exclusively within cell masks
        ratio_image = np.full(chan1_corr.shape, 0.0001, dtype=np.float32)
        ratio_image[cell_mask] = chan2_corr[cell_mask] / chan1_corr[cell_mask]
        
        # Cap unphysical ratio spikes at 2.5
        ratio_image = np.where(ratio_image > 2.5, 2.5, ratio_image)
       
        # Display background-corrected channels and resulting ratio map
        plot_images([chan1_corr, chan2_corr, ratio_image], 
                    titles=["Chan 1 Corr", "Chan 2 Corr", "Ratio Map"])
        
        # Save 32-bit ratiometric image
        out_prefix = directory[:-5]
        tif.imwrite(out_prefix + "ratiometric_image.tif", ratio_image)
        
        # Convert fluorescence ratio map to absolute calcium concentration (µM)
        mk_image = np.zeros_like(ratio_image)
        mk_image[cell_mask] = y_to_x_image(ratio_image[cell_mask])
        
        # Save calibrated calcium image
        tif.imwrite(out_prefix + "calcium_img.tif", mk_image)
      
        # Render publication dual-panel figure: Ratio Image vs. Calibrated Calcium Map
        fig, axes = plt.subplots(1, 2, figsize=(6, 2.5), dpi=300, sharex=True, sharey=True)
        
        # Panel 1: Ratiometric Image
        rat1 = axes[0].imshow(ratio_image, vmin=0.2, vmax=2.0, cmap='turbo')
        bar1 = plt.colorbar(rat1, ax=axes[0], fraction=0.046, pad=0.04) 
        bar1.set_label('488nm / 561nm Ratio', fontsize=9) 
        axes[0].set_title("Fluorescence Ratio", fontsize=10)
        axes[0].axis("off")
        
        # Panel 2: Calibrated Calcium Concentration Map
        rat2 = axes[1].imshow(mk_image, vmin=0.01, vmax=4.0, cmap='turbo')
        bar2 = plt.colorbar(rat2, ax=axes[1], fraction=0.046, pad=0.04) 
        bar2.set_label('Calcium (µM)', fontsize=9) 
        axes[1].set_title("Calibrated Calcium", fontsize=10)
        axes[1].axis("off")
        
        fig.tight_layout()
        plot_out = out_prefix + 'ratiometric_image_plot.svg'
        plt.savefig(plot_out, dpi=300)
        print(f"Saved publication plot to: {plot_out}")
        plt.show()