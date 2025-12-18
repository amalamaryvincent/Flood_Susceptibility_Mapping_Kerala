#!/usr/bin/env python
from __future__ import annotations
import argparse
import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

from fsm_autml.io.inventory import InventorySpec, read_inventory_points
from fsm_autml.io.raster_stack import RasterStack
from fsm_autml.datasets.build import PatchSpec, build_patches_from_inventory
from fsm_autml.utils.repro import set_seed

def main():
    ap = argparse.ArgumentParser(description="Notebook-compatible dataset builder (12 GeoTIFFs + point shapefile).")
    ap.add_argument("--config", default="configs/notebook_compatible.yaml")
    ap.add_argument("--out", default="data/processed/notebook_patches_3d.npz")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(int(cfg.get("seed", 42)))

    raster_paths = [os.path.join(cfg["rasters"]["root"], fn) for fn in cfg["rasters"]["factors"]]
    stack = RasterStack(raster_paths, nodata=cfg["rasters"].get("nodata"))
    if cfg.get("check_alignment", True):
        stack.check_alignment()
    x = stack.read(bands_first=True, dtype=np.float32)  # (B,H,W)

    inv = InventorySpec(
        path=cfg["inventory"]["path"],
        label_field=cfg["inventory"].get("label_field", "Flood"),
        x_field=cfg["inventory"].get("x_field", "X"),
        y_field=cfg["inventory"].get("y_field", "Y"),
    )
    gdf = read_inventory_points(inv)

    ps = PatchSpec(
        patch_size=int(cfg["patches"].get("patch_size", 227)),
        bands=int(cfg["patches"].get("bands", 12)),
        pad_value=float(cfg["patches"].get("pad_value", 0.0)),
        skip_out_of_bounds=bool(cfg["patches"].get("skip_out_of_bounds", True)),
    )

    ref_raster = raster_paths[0]  # notebook used Elevation.tif as reference for index()
    X, y = build_patches_from_inventory(
        x_bhw=x,
        ref_raster_path=ref_raster,
        xs=gdf["__x"].to_numpy(),
        ys=gdf["__y"].to_numpy(),
        labels=gdf["__label"].to_numpy(),
        spec=ps,
        max_points=cfg.get("max_points"),
    )

    # Train/test split (stratified)
    test_size = float(cfg.get("test_size", 0.2))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=cfg.get("seed", 42), stratify=y)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte, config=cfg)
    print(f"Wrote: {args.out}")
    print(f"Train: X={Xtr.shape} y={ytr.shape} | Test: X={Xte.shape} y={yte.shape}")

if __name__ == "__main__":
    main()
