# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:28:04 2023

@author: azaldec

Updates:
  - 2025/04/09: Added option to perform analysis without cropping (full-frame mode).

Description:
  Automated ratiometric calibration analysis for biosensor imaging (e.g., jREX-GECO).
  
  This script analyzes time-series TIFF movies recorded under alternating dual-wavelength
  excitation (e.g., 488 nm and 561 nm lasers). It:
    1. Loads each time-lapse TIFF stack matching the specified glob pattern.
    2. (Optional) Crops a centered region of interest (ROI) to isolate the sample area.
    3. Calculates the spatially averaged fluorescence intensity for each frame.
    4. Automatically identifies alternating excitation cycles:
       - Peaks (maxima) corresponding to excitation channel 1 (e.g., 488 nm).
       - Valleys (minima) corresponding to excitation channel 2 (e.g., 561 nm).
    5. Averages the peak and valley intensities across each acquisition.
    6. (Optional) Renders a diagnostic plot showing the intensity trace with marked peaks and valleys.
    7. Compiles and exports the results for all files into a summary CSV spreadsheet.

Usage:
  python ratiometric_calibration_analysis.py "<path_to_tiff_directory>/*.tif"
"""

import tifffile as tif
import numpy as np
import sys
import glob
import matplotlib.pyplot as plt
import scipy.signal as sig
import pandas as pd


# ==============================================================================
# Configuration & Analysis Parameters
# ==============================================================================

# Dimensions of the square Region of Interest (ROI) in pixels when cropping is enabled
mov_l = 100  # width/height of ROI in pixels

# Enable or disable centered spatial cropping
# - True:  Crops a centered (mov_l x mov_l) sub-region to avoid edge artifacts or background.
# - False: Analyzes the entire field of view across all frames.
crop = False

# Input directory or file pattern passed as the first command-line argument
# Example: sys.argv[1] = "Data/calibration_series/*.tif"
directory = sys.argv[1]

# Collect all matching TIFF files from the provided path pattern
files = [file for file in glob.glob(directory)]


# ==============================================================================
# Plotting Utility Function
# ==============================================================================

def plot_intensity(x, y, filename, maxima=None, minima=None):
    """
    Plots the mean intensity time-course across frames, highlighting detected
    peaks (channel 1, e.g., 488 nm) and valleys (channel 2, e.g., 561 nm).

    Parameters:
        x (list or array-like): Frame indices (x-axis).
        y (array-like): Spatially averaged pixel intensity per frame (y-axis).
        filename (str): Name or path of the image stack (used as the plot title).
        maxima (array-like, optional): Frame indices corresponding to peak intensities.
        minima (array-like, optional): Frame indices corresponding to valley intensities.
    """
    # Configure publication-quality plot parameters
    plt.rcParams.update({'font.size': 14})
    plt.rcParams['font.family'] = 'Calibri'
    plt.rcParams['svg.fonttype'] = 'none'  # Preserve text as text when exporting to SVG
    
    fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=300)
    plt.plot(x, y)
    
    # Overlay markers at detected peak ('x') and valley ('o') positions
    if maxima is not None and minima is not None:
        for peak, valley in zip(maxima, minima):
            plt.plot(x[peak], y[peak], "x")
            plt.plot(x[valley], y[valley], 'o')
            print(f"Peak: {y[peak]:.2f}, Valley: {y[valley]:.2f}")
            
    plt.xlabel('Frame', fontsize=10)
    plt.ylabel('Intensity (a.u.)', fontsize=10)
    plt.ylim(0, 8000)
    plt.title(filename, fontsize=8)
    fig.tight_layout()
    plt.show()


# ==============================================================================
# Main Processing Loop: File-by-File Analysis
# ==============================================================================

# List to accumulate extracted intensity metrics for each processed file
store_data = []

for file in files[:]:
    # Load 3D TIFF time-series movie: shape is (num_frames, height, width)
    mov = tif.imread(file)
    mov_r, mov_c = mov.shape[1], mov.shape[2]  # Image dimensions: rows (height) and columns (width)
    
    # Apply optional centered spatial cropping
    if crop == True:
        # Crop centered square of size (mov_l x mov_l)
        crop_mov = mov[:, 
                       int(mov_r/2 - mov_l/2): int(mov_r/2 + mov_l/2), 
                       int(mov_c/2 - mov_l/2): int(mov_c/2 + mov_l/2)]
    elif crop == False:
        # Use full field of view without cropping
        crop_mov = mov
    
    # Calculate the spatial mean intensity for each time frame
    avg_frames = []
    for frame in crop_mov:
        avg_int = np.average(frame)
        avg_frames.append(avg_int)
        
    avg_frames = np.asarray(avg_frames)
    
    # Detect peaks and valleys corresponding to alternating laser illumination:
    # - peaks:   Local maxima in avg_frames (channel 1, e.g., 488 nm excitation)
    # - valleys: Local maxima in inverted signal (-avg_frames) (channel 2, e.g., 561 nm excitation)
    # distance=15: Minimum frame spacing between consecutive peaks
    # width=5:    Minimum peak width to filter out single-frame noise spikes
    peaks, _ = sig.find_peaks(avg_frames, distance=15, width=5)
    valleys, _ = sig.find_peaks(-avg_frames, distance=15, width=5)
    
    print(f"Processing: {file}")
    print(f"  Peaks ({len(peaks)} found): {avg_frames[peaks]}")
    print(f"  Valleys ({len(valleys)} found): {avg_frames[valleys]}")
    
    # Append summary record: (file path, mean peak intensity, mean valley intensity)
    store_data.append((file, np.average(avg_frames[peaks]), np.average(avg_frames[valleys])))
    
    # Display diagnostic intensity trace with peak and valley markers
    plot_intensity(list(range(len(crop_mov))), avg_frames, file, peaks, valleys)
    
    print()


# ==============================================================================
# Summary Table Generation & CSV Export
# ==============================================================================

# Compile collected metrics into a structured pandas DataFrame
data_df = pd.DataFrame(store_data)
data_df.columns = ['FILE', '488nm', '561nm']

# Display summary results in the console
print(data_df)

# Save the DataFrame to CSV in the target directory
# (directory[:-6] trims the trailing file pattern, e.g., '*.tif', to name the output file)
out_csv = directory[:-6] + 'jREX-GECO_S7_intensities.csv'
data_df.to_csv(out_csv, index=False)
print(f"Saved intensity summary to: {out_csv}")