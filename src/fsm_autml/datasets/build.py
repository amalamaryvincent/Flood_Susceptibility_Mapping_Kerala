from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from tqdm import tqdm

from fsm_autml.io.raster_stack import world_to_rc

@dataclass(frozen=True)
class PatchSpec:
    patch_size: int = 227  # notebook used 113/114 offsets -> 227
    bands: int = 12
    pad_value: float = 0.0
    skip_out_of_bounds: bool = True

def extract_patch_bchw(
    x_bhw: np.ndarray,
    row: int,
    col: int,
    patch_size: int,
    pad_value: float = 0.0,
    skip_out_of_bounds: bool = True,
) -> Optional[np.ndarray]:
    """Extract (B, patch, patch) from (B,H,W)."""
    assert x_bhw.ndim == 3
    B, H, W = x_bhw.shape
    half = patch_size // 2
    r1 = row - half
    r2 = row + half + 1
    c1 = col - half
    c2 = col + half + 1

    if skip_out_of_bounds:
        if r1 < 0 or c1 < 0 or r2 > H or c2 > W:
            return None
        return x_bhw[:, r1:r2, c1:c2]

    # pad mode
    out = np.full((B, patch_size, patch_size), pad_value, dtype=x_bhw.dtype)
    rr1 = max(r1, 0); rr2 = min(r2, H)
    cc1 = max(c1, 0); cc2 = min(c2, W)
    out_r1 = rr1 - r1
    out_c1 = cc1 - c1
    out[:, out_r1:out_r1+(rr2-rr1), out_c1:out_c1+(cc2-cc1)] = x_bhw[:, rr1:rr2, cc1:cc2]
    return out

def build_patches_from_inventory(
    x_bhw: np.ndarray,
    ref_raster_path: str,
    xs: np.ndarray,
    ys: np.ndarray,
    labels: np.ndarray,
    spec: PatchSpec,
    max_points: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build patches exactly like the notebook logic (convert X/Y -> row/col -> slice band stack)."""
    n = len(labels) if max_points is None else min(len(labels), max_points)
    patches = []
    out_labels = []
    for i in tqdm(range(n), desc="Extracting patches"):
        r, c = world_to_rc(ref_raster_path, float(xs[i]), float(ys[i]))
        patch = extract_patch_bchw(
            x_bhw[:spec.bands, :, :],
            row=r,
            col=c,
            patch_size=spec.patch_size,
            pad_value=spec.pad_value,
            skip_out_of_bounds=spec.skip_out_of_bounds,
        )
        if patch is None:
            continue
        patches.append(patch)
        out_labels.append(int(labels[i]))
    if len(patches) == 0:
        raise RuntimeError("No patches extracted. Check CRS alignment between inventory points and rasters, "
                           "and/or reduce patch_size or disable skip_out_of_bounds.")
    X = np.stack(patches, axis=0)  # (N,B,P,P)
    y = np.asarray(out_labels, dtype=np.int64)
    return X, y
