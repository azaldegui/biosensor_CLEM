# Biosensor Quantitative Image Analysis Toolkit

A Python suite for automated quantitative analysis of genetically encoded fluorescent biosensors, including **intracellular calcium** (jREX-GECO), **intracellular pH** (pHmScarlet), and **macromolecular crowding** (Sed1).

---

## Biosensor Calibration Parameters

All sensor conversions utilize analytical inversions of the empirical 4-parameter logistic (4PL / sigmoidal) calibration equation:

$$\text{Forward: } y = \frac{L}{1 + e^{-k(x - x_0)}} + b$$

$$\text{Inverse: } x = x_0 \pm \frac{1}{k}\ln\left( \frac{L}{y - b} - 1 \right)$$

where $y$ is the measured fluorescence emission ratio ($F_{\text{chan2}} / F_{\text{chan1}}$) and $x$ is the calibrated physiological variable.

| Biosensor | Target Physiological Metric | Sign in Inversion | Asymptotic Span ($L$) | Inflection Midpoint ($x_0$) | Hill / Slope Factor ($k$) | Baseline Offset ($b$) | Valid Range |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **jREX-GECO** | Calcium Concentration ($\mu\text{M}$) | $-$ | `1.68` | `0.97 µM` | `1.34` | `0.00` | $0.01 - 4.0\ \mu\text{M}$ |
| **pHmScarlet** | Apparent Intracellular pH ($pH_{\text{app}}$) | $+$ | `5.72` | `6.54 pH` | `2.93` | `0.04` | $5.5 - 8.5\text{ pH}$ |
| **Sed1** | Macromolecular Crowding (% w/v PEG4000) | $+$ | `4.921` | `9.113 % w/v` | `0.321` | `2.433` | $0.0 - 20.0\%\text{ (w/v)}$ |

---

## Script Catalog & Usage

### 1. Data Extraction & Demultiplexing

#### `extract_channels.py`
Separates interleaved multi-channel time-lapse movies into separate excitation/emission channel projections.
- **Acquisition Timing**: Processes 10-frame repeating cycles where Channel 1 is active on frames $[i : i+4]$ (indices $0\text{--}3$) and Channel 2 on frames $[i+5 : i+9]$ (indices $5\text{--}8$).
- **Outputs**: `<filename>_channel_1.tif` and `<filename>_channel_2.tif` (temporal average projections).
- **Usage**:
  ```bash
  python extract_channels.py "path/to/raw_movies/*trim*.tif"
  ```

#### `ratiometric_calibration_analysis.py`
Quantifies peak (Channel 1) and valley (Channel 2) signals from calibration series acquired with alternating laser lines.
- **Features**: Optional centered spatial cropping (`crop = True`), per-frame spatial averaging, automatic peak/valley detection using `scipy.signal.find_peaks`, and diagnostic trace plotting.
- **Outputs**: Multi-file summary spreadsheet `jREX-GECO_S7_intensities.csv` containing columns `['FILE', '488nm', '561nm']`.
- **Usage**:
  ```bash
  python ratiometric_calibration_analysis.py "path/to/calibration_tifs/*.tif"
  ```

---

### 2. Ratiometric Image Generation & Single-Cell Analysis

Each of the following scripts processes dual-channel fluorescence images (`*channels.tif`) along with corresponding segmentation masks (`*masks.tif`).

#### `generate_ratiometric_image_jrex_channels.py`
Quantifies free intracellular calcium using the **jREX-GECO** biosensor.
- **Processing**:
  1. Removes small segmentation artifacts ($< 50$ pixels).
  2. Estimates background noise by Gaussian fitting on lowest 20% pixel intensities ($\mu + 2\sigma$).
  3. Computes cell-masked fluorescence ratio ($488\,\text{nm} / 561\,\text{nm}$).
  4. Inverts ratio map into absolute calcium concentration ($\mu\text{M}$).
  5. Measures single-cell morphology (area in $\mu\text{m}^2$ using pixel size $0.121\,\mu\text{m}$) and mean cell ratio.
- **Outputs**: `ratiometric_image.tif`, `calcium_img.tif`, and publication-ready vector graphics (`ratiometric_image_plot.svg`).
- **Usage**:
  ```bash
  python generate_ratiometric_image_jrex_channels.py "path/to/data/*"
  ```

#### `generate_ratiometric_image_pHmscarlet_channels.py`
Quantifies apparent intracellular pH using the **pHmScarlet** biosensor.
- **Processing**:
  1. Two-stage background correction: rolling-ball background filtering ($\text{radius} = 20\,\text{px}$) followed by Gaussian fitting ($\text{cutoff} = 5\%$).
  2. Single-cell extraction using physical pixel size $0.108\,\mu\text{m}$.
  3. Inverts ratio map into calibrated apparent pH ($5.5 - 8.5$).
- **Outputs**: `ratiometric_image.tif`, `pH_img.tif`, `<Sample>_singlecell_analysis.csv`, and SVG/PNG figures.
- **Usage**:
  ```bash
  python generate_ratiometric_image_pHmscarlet_channels.py "path/to/data/*"
  ```

#### `generate_ratiometric_image_sed1_channels.py`
Quantifies intracellular macromolecular crowding using the **Sed1** biosensor (calibrated against PEG4000 standards).
- **Processing**:
  1. Gaussian-fit background subtraction ($\text{cutoff} = 15\%$).
  2. Single-cell extraction using physical pixel size $0.167\,\mu\text{m}$.
  3. Inverts ratio map into % (w/v) PEG4000 equivalent ($0 - 20\%$).
- **Outputs**: `ratiometric_image.tif`, crowding map TIFF, `<Sample>_singlecell_analysis.csv`, and SVG/PNG figures.
- **Usage**:
  ```bash
  python generate_ratiometric_image_sed1_channels.py "path/to/data/*"
  ```

---

### 3. Calibration Curve Inversion


#### `convert_singlecell_F2pH.py`
Converts single-cell average ratios from CSV tables into calibrated apparent pH and plots population-level distributions.
- **Inputs**: CSV containing single-cell segmentation records with column `Cell_avg_ratio`.
- **Outputs**: Population summary statistics (mean, standard deviation) and distribution figures (`*_pH_plot.svg`, `*_pH_plot.png`).
- **Usage**:
  ```bash
  python convert_singlecell_F2pH.py "path/to/singlecell_analysis.csv"
  ```

---

### 4. Correlative Visualization (CLEM)

#### `fancy_overlay.py`
Renders composite Correlative Light and Electron Microscopy (CLEM) figures by blending registered fluorescence maps over high-resolution electron micrographs (e.g. TEM search maps or cryo-FIB/SEM overviews).
- **Features**:
  - Transparent colormap masking for background pixels ($\le 0.05$).
  - Alpha-blending of fluorescence signal ($50\%$ opacity) onto structural EM context.
  - Locked 1:1 physical aspect ratios (`adjustable='box'`).
  - Axis tick removal for publication display.
- **Outputs**: Multi-panel composite figures (`*_ratio_overlay_plot.svg` and `*_ratio_overlay_plot.png`).
- **Usage**:
  ```bash
  python fancy_overlay.py "path/to/clem_directory/"
  ```

---

## Prerequisites & Installation

### Required Python Libraries
- Python $\ge$ 3.8
- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-image`
- `tifffile`
- `openpyxl`

### Installation
Install all required dependencies using conda or pip:

```bash
# Using conda
conda install numpy scipy pandas matplotlib seaborn scikit-image tifffile openpyxl

# Or using pip
pip install numpy scipy pandas matplotlib seaborn scikit-image tifffile openpyxl
```
