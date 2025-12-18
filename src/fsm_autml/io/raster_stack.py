from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import rasterio

@dataclass(frozen=True)
class RasterStack:
    paths: List[str]
    nodata: Optional[float] = None

    def open_ref(self):
        return rasterio.open(self.paths[0])

    def check_alignment(self) -> None:
        with rasterio.open(self.paths[0]) as ref:
            ref_crs = ref.crs
            ref_transform = ref.transform
            ref_shape = (ref.height, ref.width)
        for p in self.paths[1:]:
            with rasterio.open(p) as ds:
                if ds.crs != ref_crs or ds.transform != ref_transform or (ds.height, ds.width) != ref_shape:
                    raise ValueError(
                        "Rasters are not aligned.\n"
                        f"Reference: {self.paths[0]} crs={ref_crs} transform={ref_transform} shape={ref_shape}\n"
                        f"Mismatch : {p} crs={ds.crs} transform={ds.transform} shape={(ds.height, ds.width)}"
                    )

    def read(self, bands_first: bool = True, dtype=np.float32) -> np.ndarray:
        """Read all rasters into a single array.

        Returns:
            If bands_first=True: (B, H, W)
            else: (H, W, B)
        """
        arrays = []
        nd = self.nodata
        for p in self.paths:
            with rasterio.open(p) as ds:
                a = ds.read(1).astype(dtype, copy=False)
                if nd is None:
                    nd = ds.nodata
                arrays.append(a)
        x = np.stack(arrays, axis=0)  # (B,H,W)
        if not bands_first:
            x = np.moveaxis(x, 0, -1)
        return x

def world_to_rc(ref_raster_path: str, x: float, y: float) -> Tuple[int, int]:
    """Notebook-compatible conversion (uses rasterio index)."""
    with rasterio.open(ref_raster_path) as ds:
        r, c = ds.index(x, y)
    return int(r), int(c)
