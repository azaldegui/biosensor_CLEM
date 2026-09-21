# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 10:36:06 2025

@author: azaldec

Description:
  Demultiplexes and separates interleaved multi-channel time-series microscopy movies
  into distinct excitation/emission channels based on cyclical acquisition timing.

Imaging Scheme & Channel Classification:
  Microscopy time-series data acquired with alternating laser lines often records
  frames sequentially into a single continuous TIFF stack with a fixed repetition period.
  
  In this protocol (repetition period = 10 frames):
    - Channel 1 (e.g., 488 nm excitation): Frames [i : i+4] (frames 0 to 3 of each 10-frame block)
    - Channel 2 (e.g., 561 nm excitation): Frames [i+5 : i+9] (frames 5 to 8 of each 10-frame block)
    (Frames 4 and 9 serve as transition/shutter intervals between alternating laser states).

Workflow:
  1. Identifies all matching raw time-series TIFF stacks containing 'trim' in their filename.
  2. Computes the spatial mean pixel intensity for every frame in the stack.
  3. Displays an intensity-vs-time diagnostic trace to verify alternating laser switching.
  4. Slices and pools all frames corresponding to Channel 1 and Channel 2 across all cycles.
  5. Computes time-averaged intensity projections for both channels.
  6. Exports the resulting projections as separate TIFF files:
       - <filename>_channel_1.tif
       - <filename>_channel_2.tif

Usage:
  python extract_channels.py "<path_to_input_directory>/*trim*.tif"
"""

import sys
import glob
import numpy as np
import pandas as pd
import tifffile as tif
import matplotlib.pyplot as plt
import scipy.signal as sig


# ==============================================================================
# Plotting Utility Function
# ==============================================================================

def plot_intensity(x, y, filename):
    """
    Plots the spatial mean intensity across frames to verify the alternating
    periodicity of multi-channel laser illumination.

    Parameters:
        x (list or array-like): Frame indices or time points.
        y (array-like): Mean intensity counts per frame.
        filename (str): Name of the source file (displayed as plot title).
    """
    fig, ax = plt.subplots(figsize=(5, 3), dpi=200)
    plt.plot(x, y, color='#2c3e50', linewidth=1.5)
    
    plt.xlabel('Frame Number', fontsize=11)
    plt.ylabel('Mean Intensity Counts', fontsize=11)
    plt.title(filename, fontsize=9, pad=8)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    fig.tight_layout()
    plt.show()


# ==============================================================================
# Main Demultiplexing Loop
# ==============================================================================

# Parse input directory or glob pattern from command-line arguments
directory = sys.argv[1]

# Find all matching files containing the substring 'trim'
files = [file for file in glob.glob(directory) if 'trim' in file]
print(f"Found {len(files)} trimmed multi-channel TIFF movies to process.")

store_data = []

for file in files[:]:
    # Load 3D time-series image stack: shape is (n_frames, height, width)
    mov = tif.imread(file)
    n_frames = mov.shape[0]
    print(f"\nProcessing: {file}")
    print(f"  Stack Dimensions: {mov.shape} (Frames: {n_frames}, Height: {mov.shape[1]}, Width: {mov.shape[2]})")

    # Calculate spatial mean intensity for each time frame to observe the periodic pattern
    avg_frames = []
    for frame in mov:     
        avg_int = np.average(frame)
        avg_frames.append(avg_int)
   
    avg_frames = np.asarray(avg_frames) 
    
    # Plot intensity-vs-time trace to visually inspect periodicity
    plot_intensity([x for x in range(len(mov))], avg_frames, file)
    
    # Interleaving period parameters
    period = 10  # Full repeat cycle length in frames
    
    channel_1_frames = []
    channel_2_frames = []
    
    # Iterate through the time series in blocks of 'period' frames
    for i in range(0, n_frames, period):
        # Extract Channel 1 frames: first 4 frames of the cycle (indices i to i+3)
        if i + 3 < n_frames:
            channel_1_frames.append(mov[i:i+4])
            
        # Extract Channel 2 frames: subsequent 4 frames of the cycle (indices i+5 to i+8)
        if i + 8 < n_frames:
            channel_2_frames.append(mov[i+5:i+9])
            
    out_filename = file[:-4]  # Strip '.tif' extension for naming output files
            
    # Concatenate all collected frames for Channel 1 and compute the time-averaged projection
    channel_1 = np.concatenate(channel_1_frames, axis=0)
    channel_1_avg = np.mean(channel_1, axis=0).astype(np.float32)
    ch1_path = out_filename + "_channel_1.tif"
    tif.imwrite(ch1_path, channel_1_avg)
    print(f"  Saved Channel 1 projection ({channel_1.shape[0]} frames averaged) -> {ch1_path}")
    
    # Concatenate all collected frames for Channel 2 and compute the time-averaged projection
    channel_2 = np.concatenate(channel_2_frames, axis=0)
    channel_2_avg = np.mean(channel_2, axis=0).astype(np.float32)
    ch2_path = out_filename + "_channel_2.tif"
    tif.imwrite(ch2_path, channel_2_avg)
    print(f"  Saved Channel 2 projection ({channel_2.shape[0]} frames averaged) -> {ch2_path}")