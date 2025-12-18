# Flood Susceptibility Mapping

Flood susceptibility mapping (FSM) using **12 GeoTIFF conditioning factors**
and a **point inventory** (flood/non-flood) 

## Data you need


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
```

