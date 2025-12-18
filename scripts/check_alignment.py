#!/usr/bin/env python
from __future__ import annotations
import argparse, glob, os
from fsm_autml.io.raster_stack import RasterStack

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/raw/rasters/*.tif", help="Glob pattern for input rasters")
    args = ap.parse_args()
    paths = sorted(glob.glob(args.glob))
    if not paths:
        raise SystemExit(f"No rasters found for glob: {args.glob}")
    stack = RasterStack(paths)
    stack.check_alignment()
    print(f"OK: {len(paths)} rasters are aligned (CRS/transform/shape match).")

if __name__ == "__main__":
    main()
