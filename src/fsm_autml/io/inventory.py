from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import geopandas as gpd

@dataclass(frozen=True)
class InventorySpec:
    path: str
    label_field: str = "Flood"
    x_field: Optional[str] = "X"
    y_field: Optional[str] = "Y"

def read_inventory_points(spec: InventorySpec) -> gpd.GeoDataFrame:
    """Read inventory points.

    Notebook-compatible behavior:
    - If X/Y fields exist, uses them.
    - Otherwise derives X/Y from geometry coordinates.
    - Label is taken from spec.label_field.
    """
    gdf = gpd.read_file(spec.path)
    if spec.label_field not in gdf.columns:
        raise ValueError(f"Label field '{spec.label_field}' not found in {spec.path}. "
                         f"Available columns: {list(gdf.columns)}")
    if spec.x_field and spec.x_field in gdf.columns and spec.y_field and spec.y_field in gdf.columns:
        gdf["__x"] = gdf[spec.x_field].astype(float)
        gdf["__y"] = gdf[spec.y_field].astype(float)
    else:
        if gdf.geometry is None:
            raise ValueError("No geometry column found and X/Y fields missing.")
        gdf["__x"] = gdf.geometry.x.astype(float)
        gdf["__y"] = gdf.geometry.y.astype(float)
    gdf["__label"] = gdf[spec.label_field].astype(int)
    return gdf
