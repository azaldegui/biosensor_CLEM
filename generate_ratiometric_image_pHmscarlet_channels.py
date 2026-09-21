# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 12:11:48 2025

@author: azaldec

Description:
  Generates calibrated ratiometric pH images and single-cell statistics from
  dual-channel fluorescence microscopy data using the pHmScarlet biosensor.

Mathematical Model & Inversion:
  Forward sigmoidal pH calibration curve:
      y = L / (1 + exp(-k * (x - x0))) + b

  Analytical inversion for apparent intracellular pH (x):
      x = x0 + (1 / k) * ln( (L / (y - b)) - 1 )

  Where:
      y  = Measured fluorescence emission ratio (Channel 2 / Channel 1)
      x  = Calibrated apparent pH
      L  = Dynamic ratio span (5.72)
      x0 = Apparent pKa / inflection point pH (6.54)
      k  = Logistic slope factor (2.93)
      b  = Lower asymptote / baseline ratio offset (0.04)

Processing Pipeline:
  1. Loads paired multi-channel TIFF images (*channels.tif) and cell masks (*masks.tif).
  2. Slices Channel 1 (index 1) and Channel 2 (index 0).
  3. Preprocesses cell segmentation masks (filters objects < 10 pixels).
  4. Performs two-stage background subtraction:
     - Rolling-ball background filtering (radius = 20 pixels) to correct non-uniform illumination.
     - Gaussian fitting on low-intensity pixels (cutoff_pct = 5) to remove residual camera offset.
  5. Computes pixel-wise fluorescence ratio (Channel 2 / Channel 1) within masked cells.
  6. Performs single-cell morphological and ratio analysis (pixel size = 0.108 µm).
  7. Converts the ratiometric map to apparent pH via the inverted 4PL model.
  8. Exports 32-bit TIFFs (ratiometric and pH maps), publication plots (SVG/PNG), and single-cell CSV tables.

Usage:
  python generate_ratiometric_image_pHmscarlet_channels.py "<path_to_data_folder>/*"
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
# Visualization & Image Display Helper
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
    Estimates a residual background threshold by fitting a Gaussian distribution
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
# Model Inversion: Ratio to Apparent pH
# ==============================================================================

def y_to_x_image(y_image, background_value=0.001, out_of_range_value=1.0):
    """
    Converts a ratiometric image (y) into a calibrated apparent pH image (x)
    using the inverted 4-parameter logistic model for pHmScarlet:
        pH = x0 + (1 / k) * ln((L / (y - b)) - 1)

    Parameters:
        y_image (ndarray): 2D array of fluorescence ratio values.
        background_value (float): Threshold below which pixels are treated as background.
        out_of_range_value (float): Fallback value assigned to unphysical ratios.

    Returns:
        ndarray: Calibrated apparent pH image.
    """
    # pHmScarlet calibration coefficients
    L = 5.72    # Dynamic ratio span
    x0 = 6.54   # Apparent pKa / inflection point pH
    k = 2.93    # Logistic slope factor
    b = 0.04    # Baseline ratio offset

    y = y_image.astype(float)

    # Mask background and unphysical values to prevent math domain errors in ln()
    mask_invalid = (y <= background_value) | (y <= b) | (y >= L + b)
    
    # Clamp within open interval (b, L + b)
    y_clipped = np.clip(y, b + 1e-6, L + b - 1e-6)

    # Invert the logistic function for pH
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
    
    pixel_size = 0.108  # Physical pixel size in microns (µm/pixel)
    
    cell_ratios = np.array(ratio_ints)
    cell_areas = np.array(areas) * (pixel_size ** 2)  # Convert area to µm²
    cell_labels = np.array([cell.label for cell in cells])
    
    return np.column_stack((cell_labels, cell_areas, cell_ratios))


# ==============================================================================
# Main Batch Processing Script
# ==============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_ratiometric_image_pHmscarlet_channels.py '<path_pattern>'")
        sys.exit(1)

    directory = sys.argv[1]
    files = [file for file in glob.glob(directory)]

    fluor_imgs = sorted([file for file in files if "channels.tif" in file])
    mask_imgs = sorted([file for file in files if 'masks.tif' in file])

    print(f"Matched {len(fluor_imgs)} fluorescence images with {len(mask_imgs)} mask files.")

    for fluor, mask in zip(fluor_imgs, mask_imgs):
        print(f"\nProcessing:\n  Fluorescence: {fluor}\n  Mask:         {mask}")
        
        # Load multi-channel TIFF: shape (2, height, width)
        # Channel 1: index 1 (emission 1 / reference)
        # Channel 2: index 0 (emission 2 / pH-sensitive)
        chan1_arr = tif.imread(fluor)[1].astype(np.float32)
        chan2_arr = tif.imread(fluor)[0].astype(np.float32)
        
        # Load cell mask and filter out small debris (< 10 pixels)
        mask_arr = tif.imread(mask)
        mask_arr = morphology.remove_small_objects(mask_arr, min_size=10)
        cell_mask = mask_arr > 0 

        # Display raw channels and cell mask
        plot_images([chan1_arr, chan2_arr, mask_arr], titles=["Chan 1", "Chan 2", "Cell Mask"])
        
        # Stage 1: Rolling-ball background subtraction to flatten non-uniform illumination
        # (radius=20 pixels should be larger than typical cell radius)
        chan1_bg = restoration.rolling_ball(chan1_arr, radius=20)
        chan1_corr = chan1_arr - chan1_bg
        chan2_bg = restoration.rolling_ball(chan2_arr, radius=20)
        chan2_corr = chan2_arr - chan2_bg
        print("Completed rolling-ball background correction.")
        
        # Display background-subtracted channels
        plot_images([chan1_arr, chan1_bg, chan1_corr], titles=["Chan 1 Raw", "Chan 1 BG", "Chan 1 Subtracted"])
        plot_images([chan2_arr, chan2_bg, chan2_corr], titles=["Chan 2 Raw", "Chan 2 BG", "Chan 2 Subtracted"])
        
        # Stage 2: Gaussian fitting on the lowest 5% to subtract residual dark noise
        thresh1 = threshold(chan1_corr, cutoff_pct=5)
        thresh2 = threshold(chan2_corr, cutoff_pct=5)
        chan1_corr = np.clip(chan1_corr - thresh1, 1.0, None)
        chan2_corr = np.clip(chan2_corr - thresh2, 0.0001, None)
        
        # Compute ratiometric image (Chan 2 / Chan 1) inside cell mask
        ratio_image = np.full(chan1_corr.shape, 0.001, dtype=np.float32)
        ratio_image[cell_mask] = chan2_corr[cell_mask] / chan1_corr[cell_mask]
        
        # Single-cell extraction and CSV export
        single_cell_data = calc_avg_cell_ratio(mask_arr, ratio_image)
        data_df = pd.DataFrame(single_cell_data)
        data_df.columns = ['Cell_label', 'Cell_area', 'Cell_avg_ratio']
        name = 'Hn_jLeft'
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
        
        # Convert fluorescence ratio map to calibrated apparent pH
        mk_image = np.zeros_like(ratio_image)
        mk_image[cell_mask] = y_to_x_image(ratio_image[cell_mask])
        
        # Save calibrated pH TIFF
        tif.imwrite(out_prefix + "pH_img.tif", mk_image)
      
        # Render publication dual-panel figure: Ratio Image vs. Calibrated pH Map
        fig, axes = plt.subplots(1, 2, figsize=(6, 2.5), dpi=300, sharex=True, sharey=True)
        
        # Panel 1: Ratiometric Image
        rat1 = axes[0].imshow(ratio_image, vmin=0.05, vmax=1.0, cmap='turbo')
        bar1 = plt.colorbar(rat1, ax=axes[0], fraction=0.046, pad=0.04) 
        bar1.set_label('488nm / 561nm Ratio', fontsize=9) 
        axes[0].set_title("Fluorescence Ratio", fontsize=10)
        axes[0].axis("off")
        
        # Panel 2: Calibrated pH Map
        rat2 = axes[1].imshow(mk_image, vmin=5.5, vmax=8.5, cmap='turbo')
        bar2 = plt.colorbar(rat2, ax=axes[1], fraction=0.046, pad=0.04) 
        bar2.set_label('Apparent pH', fontsize=9) 
        axes[1].set_title("Calibrated pH Map", fontsize=10)
        axes[1].axis("off")
        
        fig.tight_layout()
        plot_out = out_prefix + 'ratiometric_image_plot.svg'
        plt.savefig(plot_out, dpi=300)
        print(f"Saved publication plot to: {plot_out}")
        plt.show()