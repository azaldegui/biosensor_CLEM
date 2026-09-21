# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 13:38:57 2026

@author: azaldec

Description:
  Conversion of single-cell fluorescence intensity ratios to apparent intracellular pH
  (pH_app) using an inverted 4-parameter logistic (4PL / sigmoidal) calibration curve.

Mathematical Model:
  Forward sigmoidal pH calibration model:
      y = L / (1 + exp(-k * (x - x0))) + b

  Analytical inversion for apparent pH (x):
      x = x0 + (1 / k) * ln( (L / (y - b)) - 1 )

  Where:
      y  = Measured single-cell average fluorescence ratio (Cell_avg_ratio)
      x  = Inferred apparent intracellular pH (Converted_pH_app)
      L  = Maximum dynamic ratio span (5.72)
      x0 = Apparent pKa / inflection point pH (6.54)
      k  = Slope factor (2.93)
      b  = Lower asymptote / baseline ratio offset (0.04)

Workflow:
  1. Reads a CSV spreadsheet containing single-cell segmentation and ratio measurements.
  2. Applies the inverted 4PL calibration function to convert each cell's ratio to pH.
  3. Validates and bounds inputs to avoid mathematical domain errors (log of negative numbers).
  4. Calculates population-level mean and standard deviation of apparent pH.
  5. Generates publication-ready distribution plots (histogram + KDE), saving both SVG and PNG.

Usage:
  python convert_singlecell_F2pH.py "<path_to_singlecell_csv>"
"""

import sys
import math
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit


# ==============================================================================
# Model Inversion & Calibration Function
# ==============================================================================

def convert_ratio_to_x(y_val, background_value=0.001, out_of_range_value=1.0):
    """
    Converts a single-cell average fluorescence ratio (y) to an apparent pH value (x)
    based on the inverted 4-parameter logistic calibration curve.

    Parameters:
        y_val (float): Measured average fluorescence ratio for an individual cell.
        background_value (float): Threshold below which a signal is treated as non-cellular noise.
        out_of_range_value (float): Fallback value assigned to unphysical or out-of-range ratios.

    Returns:
        float: Calibrated apparent intracellular pH (x_val), or out_of_range_value if invalid.
    """
    # 4-parameter logistic model coefficients calibrated for the pH biosensor
    L = 5.72    # Dynamic range span between upper and lower plateaus
    x0 = 6.54   # Apparent pKa / midpoint pH
    k = 2.93    # Logistic slope factor
    b = 0.04    # Baseline ratio offset (lower asymptote)

    # 1. Handle Background and Out-of-Range values:
    # Ensures ratio is strictly within the valid range (b, L + b) to prevent
    # Math Domain Errors when taking the natural logarithm of zero or negative numbers.
    if y_val <= background_value or y_val <= b or y_val >= (L + b):
        return out_of_range_value

    # 2. Analytical Inversion Calculation:
    # Formula: x = x0 + (1 / k) * ln( (L / (y - b)) - 1 )
    try:
        inner_val = (L / (y_val - b)) - 1.0
        if inner_val <= 0:
            return out_of_range_value
        x_val = x0 + (1.0 / k) * math.log(inner_val)
        return x_val
    except (ValueError, ZeroDivisionError):
        # Fallback safeguard in case of unexpected floating-point edge cases
        return out_of_range_value


# ==============================================================================
# Visualization: Population Distribution Plot
# ==============================================================================

def plot_converted_distribution(data_series, title="Distribution of single-cell pH", out_prefix=""):
    """
    Generates and saves a publication-quality histogram and summary overlay
    for the converted single-cell apparent pH population.

    Parameters:
        data_series (pd.Series): Series of converted pH values.
        title (str): Title displayed above the plot.
        out_prefix (str): File prefix used when saving output image files.
    """
    # Configure clean publication aesthetic
    sns.set_theme(style="ticks")
    plt.rcParams['svg.fonttype'] = 'none'  # Keep text editable in vector SVG
    
    fig = plt.figure(figsize=(5, 5), dpi=100)
    
    # Plot probability density histogram across cells
    sns.histplot(data_series, stat='probability', color="gray", bins=10, edgecolor='black', alpha=0.7)
    
    # Formatting, titles, and axis labels
    plt.title(title, fontsize=15, fontweight='bold', pad=10)
    plt.xlabel("Apparent pH", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    
    # Overlay vertical line indicating the population mean
    pop_mean = data_series.mean()
    plt.axvline(pop_mean, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {pop_mean:.2f}')
    plt.legend(frameon=True, fontsize=11)
    
    fig.tight_layout()
    
    # Save vector SVG and high-res PNG plots
    if out_prefix:
        svg_out = out_prefix + '_pH_plot.svg'
        png_out = out_prefix + '_pH_plot.png'
        plt.savefig(svg_out, dpi=300)
        plt.savefig(png_out, dpi=300)
        print(f"Saved distribution plots to:\n  {svg_out}\n  {png_out}")
    
    plt.show()


# ==============================================================================
# Execution Workflow
# ==============================================================================

if __name__ == "__main__":
    # Ensure input CSV file is provided
    if len(sys.argv) < 2:
        print("Usage: python convert_singlecell_F2pH.py <path_to_singlecell_analysis.csv>")
        sys.exit(1)

    file = sys.argv[1]
    print(f"Loading single-cell data from: {file}")
    
    # Load single-cell metrics table
    df = pd.read_csv(file, header=0)

    # Convert single-cell average ratios to calibrated apparent pH
    df['Converted_pH_app'] = df['Cell_avg_ratio'].apply(convert_ratio_to_x)
    
    # Compute population statistics
    mean_val = df['Converted_pH_app'].mean()
    std_val = df['Converted_pH_app'].std()

    print("\n--- Single-Cell pH Summary ---")
    print(f"Total Cells Analyzed: {len(df)}")
    print(f"Mean Apparent pH:     {mean_val:.4f}")
    print(f"Standard Deviation:   {std_val:.4f}")

    # Generate and save publication distribution figure
    plot_prefix = file[:-4]
    plot_converted_distribution(df['Converted_pH_app'], out_prefix=plot_prefix)
