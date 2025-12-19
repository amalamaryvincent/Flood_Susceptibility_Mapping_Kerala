# Flood Susceptibility Mapping (Kerala) — AutoML + 3D CNN with Evolutionary HPO

This repository **accompanies and reproduces the results** from the paper:

**Vincent, A.M., Parthasarathy K.S.S., Jidesh P. (2023)**  
*Flood susceptibility mapping using AutoML and a deep learning framework with evolutionary algorithms for hyperparameter optimization*  
**Applied Soft Computing 148 (2023) 110846** — https://doi.org/10.1016/j.asoc.2023.110846

---

## Overview

Flood susceptibility mapping (FSM) identifies areas that are more likely to flood during extreme rainfall.  
In this work, we build FSM for **Kerala, India** using:

- **Ensemble ML**: Random Forest, Gradient Boost, XGBoost, AdaBoost  
- **AutoML**: AutoKeras, TPOT  
- **Deep learning**: **1D / 2D / 3D CNN**, with **hyperparameter optimization (HPO)**

We optimize key CNN hyperparameters using **Bayesian Optimization (BO)** and BO combined with evolutionary algorithms:
- **BO-DE** (Differential Evolution)
- **BO-CMAES** (CMA-ES / Evolutionary Strategies)

---

## Study area (Kerala)

![Study area](docs/figures/fig1_study_area.png)

---

## Data

### Flood inventory (labels)
Flood inventory is built from **Sentinel-1 SAR** flood events for **2018–2021**, sampling:
- 4000 flooded points
- 4000 non-flooded points  
(80% train / 20% test)

Expected format in this repo:
- a point file (Shapefile or GeoPackage)
- a label column: `Flood` (1=flood, 0=non-flood)

### Flood conditioning factors (predictors)
We use 12 flood-triggering factors (thematic rasters):
1. Elevation
2. Slope angle
3. Curvature
4. Aspect
5. TWI
6. Rainfall
7. Distance to river
8. SPI
9. Soil type
10. Lithology/Geology
11. NDVI
12. Land use / LULC

> **Important:** All rasters must be aligned (same CRS, resolution, extent, grid).

---

## CNN input representations (1D / 2D / 3D)

### 1D + 2D representations
![1D and 2D representation](docs/figures/fig4_1d_2d_representation.png)

- **1D-CNN** uses a 12×1 input vector (one value per factor).
- **2D-CNN** uses a 17×17 encoding derived from 1D (17 = max category count among factors).

### 3D representation (patch-based)
![3D representation](docs/figures/fig5_3d_representation.png)

- **3D-CNN** uses a **32×32×12** stack (spatial neighborhood + all 12 factors).

---

## Methodology workflow

![Workflow](docs/figures/fig9_method_flowchart.png)

---

## Flood susceptibility maps (examples from models)

![FSM maps a–e](docs/figures/fig11_models_a-e.png)

![FSM maps f–j](docs/figures/fig11_models_f-j.png)

---

## structure 

```text
configs/                 # YAML configs
scripts/                 # data prep / training scripts
notebooks/               # notebooks (e.g., raster_workbook.ipynb)
src/                     # reusable python modules
docs/figures/            # images used in this README
data/
  raw/
    rasters/             # 12 aligned GeoTIFFs (usually NOT committed)
    inventory/           # flood inventory points (usually NOT committed)
  processed/             # generated samples / patches



```
data/raw/rasters/
  Elevation.tif
  Aspect.tif
  Curvature.tif
  dtr.tif
  ndvi.tif
  rainfall.tif
  Slope.tif
  spi_extract.tif
  TWI.tif
  Geology.tif
  lulc.tif
  Soil.tif

data/raw/inventory/
  fandn.shp  (+ .shx/.dbf/.prj)
```

Inventory shapefile fields expected (same as the notebook):
- `X` (x coordinate in the raster CRS)
- `Y` (y coordinate in the raster CRS)
- `Flood` (0 = non-flood, 1 = flood)

If `X`/`Y` are missing, the builder will derive them from the point geometry.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# If you want imports to work without fiddling with PYTHONPATH:
pip install -e .
```

## 1) Check raster alignment

All 12 rasters must share **CRS, transform, width/height**.

```bash
python scripts/check_alignment.py --glob "data/raw/rasters/*.tif"
```

## 2) Build notebook-style patches

This reproduces the workbook’s key pattern:
- stack rasters to `x` shaped (bands, rows, cols)
- convert each point (X,Y) -> (row,col) using the reference raster
- slice `x[:12, r-half:r+half+1, c-half:c+half+1]`

```bash
python scripts/build_notebook_dataset.py   --config configs/notebook_compatible.yaml   --out data/processed/notebook_patches_3d.npz
```

## 3) Train a quick baseline CNN

```bash
python scripts/train_simple_3dcnn.py   --data data/processed/notebook_patches_3d.npz   --epochs 10 --batch 16 --lr 1e-3


**Citation**
If you use this repository, please cite:

@article{vincent2023fsm,
  title   = {Flood susceptibility mapping using AutoML and a deep learning framework with evolutionary algorithms for hyperparameter optimization},
  author  = {Vincent, Amala Mary and Parthasarathy, K.S.S. and Jidesh, P.},
  journal = {Applied Soft Computing},
  volume  = {148},
  pages   = {110846},
  year    = {2023},
  doi     = {10.1016/j.asoc.2023.110846}
}

```

