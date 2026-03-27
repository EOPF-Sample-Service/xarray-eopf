# The MIT License (MIT)
# Copyright (c) 2025 by the xcube development team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from collections.abc import Iterable
from dataclasses import dataclass

import dask.array as da
import numba as nb
import numpy as np

import pyproj
import xarray as xr

from xcube_resampling.affine import resample_dataset
from xcube_resampling.constants import (
    LOG,
    SCALE_LIMIT,
    FillValues,
    FloatInt,
    PreventNaNPropagations,
    SpatialAggMethods,
    SpatialInterpMethods,
    SpatialInterpMethod,
    SpatialInterpMethodStr,
)
from xcube_resampling.gridmapping import GridMapping
from xcube_resampling.utils import (
    _get_fill_value,
    _get_spatial_interp_method_str,
    _is_equal_crs,
    _prep_spatial_interp_methods_downscale,
    _select_variables,
    bbox_overlap,
    clip_dataset_by_bbox,
    normalize_grid_mapping,
)

FX_MIN = FY_MIN = 1e-3
FXFY_MAX = 1.0 + 2 * 1e-3


@dataclass(frozen=True)
class SourceTileIndexing:
    ij_bboxes: np.ndarray  # (4, ny_tiles, nx_tiles)
    pad_width: tuple[tuple[int, int], tuple[int, int]]
    output_size: tuple[int, int]
    tile_size: tuple[int, int]


def rectify_dataset(
    source_ds: xr.Dataset,
    *,
    target_gm: GridMapping | None = None,
    source_gm: GridMapping | None = None,
    variables: str | Iterable[str] | None = None,
    interp_methods: SpatialInterpMethods | None = None,
    agg_methods: SpatialAggMethods | None = None,
    prevent_nan_propagations: PreventNaNPropagations = False,
    fill_values: FillValues | None = None,
    tile_size: int | tuple[int, int] | None = None,
    output_indices_names: tuple[str, str] | None = None,
) -> xr.Dataset:
    """
    Rectify a dataset with non-regular grid to a regular grid defined by a target
    grid mapping.

    This function transforms spatial coordinates to a regular grid while preserving
    data values. It optionally downsamples high-resolution inputs prior to rectifying.

    Args:
        source_ds: The source dataset with 2D spatial coordinate variables.
        target_gm: Optional target grid mapping defining the output regular grid.
            If not provided, one is derived from the source grid mapping.
        source_gm: Optional grid mapping of the source dataset. If not given, it is
            inferred from the dataset.
        variables: Optional variable(s) to rectify. If None, all eligible variables
            are processed.
        interp_methods: Optional interpolation method to be used for upsampling spatial
            data variables. Can be a single interpolation method for all variables or a
            dictionary mapping variable names or dtypes to interpolation method.
            Supported methods include:

            - `0` (nearest neighbor)
            - `1` (linear / bilinear)
            - `"nearest"`
            - `"triangular"`
            - `"bilinear"`

            The default is `0` for integer arrays, else `1`.
        agg_methods: Optional aggregation methods for downsampling spatial variables.
            Can be a single method for all variables or a dictionary mapping variable
            names or dtypes to methods. Supported methods include:
                "center", "count", "first", "last", "max", "mean", "median",
                "mode", "min", "prod", "std", "sum", and "var".
            Defaults to "center" for integer arrays, else "mean".
        prevent_nan_propagations: Optional boolean or mapping to prevent NaN
            propagation during upsampling (only applies when interpolation method
            is not nearest). Can be a single boolean or a dictionary mapping
            variable names or dtypes to booleans. Defaults to False.
        fill_values: Optional fill value(s) for areas outside input coverage.
            Can be a single value or dictionary by variable or type. If not provided,
            defaults based on data type are used:

            - float: NaN
            - uint8: 255
            - uint16: 65535
            - other ints: -1

        tile_size: Optional tile size for inferring a regular grid, if `target_gm` is
            not provided.
        output_indices_names: Optional names for two variables that store the source
            pixel indices for the last and second-last dimension, respectively.

    Returns:
        A new dataset with spatial variables rectified to a regular grid.
        Variables not having 2D spatial dimensions are copied as-is. 1D spatial
        coordinate variables are ignored in the output.
    """
    source_ds = _select_variables(source_ds, variables)

    source_gm = source_gm or GridMapping.from_dataset(source_ds)
    source_ds = normalize_grid_mapping(source_ds, source_gm)

    target_gm = target_gm or source_gm.to_regular(tile_size=tile_size)

    source_ds, source_gm = _align_source_crs_to_target(source_ds, source_gm, target_gm)
    source_ds, source_gm = _clip_if_needed(source_ds, source_gm, target_gm, fill_values)
    source_ds, source_gm = _downscale_source_dataset(
        source_ds,
        source_gm,
        target_gm,
        interp_methods,
        agg_methods,
        prevent_nan_propagations,
    )

    indexing = compute_source_tile_indexing(source_gm, target_gm)

    src_x, src_y = (
        _reorganize_tiled_array(
            source_ds[v].data, indexing, _get_fill_value(fill_values, v, source_ds[v])
        )
        for v in source_gm.xy_var_names
    )

    pixel_target_ij = _get_pixel_target_ij(src_x, src_y, target_gm)

    return _assemble_rectified_dataset(
        source_ds,
        source_gm,
        target_gm,
        indexing,
        pixel_target_ij,
        interp_methods,
        fill_values,
        output_indices_names,
    )


def _align_source_crs_to_target(
    source_ds: xr.Dataset,
    source_gm: GridMapping,
    target_gm: GridMapping,
) -> tuple[xr.Dataset, GridMapping]:
    """
    Ensure source dataset coordinates are in the same CRS as the target grid mapping.
    """
    if _is_equal_crs(source_gm, target_gm):
        return source_ds, source_gm

    source_ds = _transform_coords(source_ds, source_gm, target_gm)
    source_gm = GridMapping.from_dataset(source_ds)
    return source_ds, source_gm


def _clip_if_needed(
    source_ds: xr.Dataset,
    source_gm: GridMapping,
    target_gm: GridMapping,
    fill_values: FillValues | None,
) -> tuple[xr.Dataset, GridMapping]:
    """
    Clip the source dataset to the target bounding box if overlap is limited.
    """

    overlap = bbox_overlap(source_gm.xy_bbox, target_gm.xy_bbox)

    if overlap < 1e-5:
        LOG.info(
            "Target grid mapping does not overlap with the source grid mapping. "
            "Returning empty target dataset."
        )
        empty_ds = _create_empty_dataset(source_ds, source_gm, target_gm, fill_values)
        return empty_ds, target_gm

    if overlap >= 0.8:
        return source_ds, source_gm

    # Expand target bbox slightly to ensure coverage
    bbox = [
        target_gm.xy_bbox[0] - 3 * source_gm.x_res,
        target_gm.xy_bbox[1] - 3 * source_gm.y_res,
        target_gm.xy_bbox[2] + 3 * source_gm.x_res,
        target_gm.xy_bbox[3] + 3 * source_gm.y_res,
    ]

    clipped = clip_dataset_by_bbox(source_ds, bbox)

    if any(clipped.sizes[source_gm.xy_dim_names[i]] < 2 for i in range(2)):
        LOG.warning(
            "Clipped dataset contains a spatial dimension with size < 2. "
            "Returning empty target dataset."
        )
        empty_ds = _create_empty_dataset(source_ds, source_gm, target_gm, fill_values)
        return empty_ds, target_gm

    return clipped, GridMapping.from_dataset(clipped)


def _assemble_rectified_dataset(
    source_ds: xr.Dataset,
    source_gm: GridMapping,
    target_gm: GridMapping,
    indexing: SourceTileIndexing,
    pixel_target_ij: da.Array,
    interp_methods: SpatialInterpMethods | None,
    fill_values: FillValues | None,
    output_indices_names: tuple[str, str] | None,
) -> xr.Dataset:
    """
    Assemble the final rectified dataset on the target grid.
    """
    src_x_name, src_y_name = source_gm.xy_var_names
    tgt_x_name, tgt_y_name = target_gm.xy_var_names

    coords = source_ds.coords.to_dataset()
    coords = coords.drop_vars((src_x_name, src_y_name), errors="ignore")

    target_coords = target_gm.to_coords()
    coords[tgt_x_name] = target_coords[tgt_x_name]
    coords[tgt_y_name] = target_coords[tgt_y_name]
    coords["spatial_ref"] = xr.DataArray(0, attrs=target_gm.crs.to_cf())
    target_ds = xr.Dataset(coords=coords, attrs=source_ds.attrs)

    yx_dims = (source_gm.xy_dim_names[1], source_gm.xy_dim_names[0])
    for var_name, da_src in source_ds.data_vars.items():
        if da_src.dims[-2:] == yx_dims:
            assert len(da_src.dims) in (
                2,
                3,
            ), f"Data variable {var_name} has {len(da_src.dims)} dimensions."
            fill_value = _get_fill_value(fill_values, var_name, da_src)
            interp_method = _get_spatial_interp_method_str(
                interp_methods, var_name, da_src
            )

            rectified = _rectify_data_array(
                da_src.data,
                indexing,
                target_gm,
                pixel_target_ij,
                interp_method,
                fill_value,
            )
            dims = da_src.dims[:-2] + (
                target_gm.xy_dim_names[1],
                target_gm.xy_dim_names[0],
            )
            target_ds[var_name] = xr.DataArray(rectified, dims=dims, attrs=da_src.attrs)

        elif yx_dims[0] not in da_src.dims and yx_dims[1] not in da_src.dims:
            # Non-spatial variable → copy as-is
            target_ds[var_name] = da_src

    # optional pixel index outputs
    if output_indices_names:
        target_ds[output_indices_names[0]] = (
            (tgt_y_name, tgt_x_name),
            pixel_target_ij[0],
        )
        target_ds[output_indices_names[1]] = (
            (tgt_y_name, tgt_x_name),
            pixel_target_ij[1],
        )

    return target_ds


def _get_pixel_target_ij(
    src_x_coords: da.Array,
    src_y_coords: da.Array,
    target_gm: GridMapping,
) -> da.Array:
    """
    Compute fractional source pixel indices for each target grid pixel.

    For each target pixel, this function determines the fractional (i, j)
    location in the source grid using forward mapping from source grid
    quadrilaterals to target grid pixels.

    The computation is performed blockwise using Dask. Each block receives:
    - a 1D slice of target x coordinates
    - a 1D slice of target y coordinates
    - a 2D tile of source x/y coordinates

    Args:
        src_x_coords: Source grid x-coordinates (2D, tiled).
        src_y_coords: Source grid y-coordinates (2D, tiled).
        target_gm: Target grid mapping.

    Returns:
        Dask array of shape (2, target_height, target_width) containing
        fractional source pixel indices (ix, iy) per target pixel.
        Missing values are NaN.
    """

    def _pixel_index_block(
        tgt_x_block: np.ndarray,
        tgt_y_block: np.ndarray,
        src_x_block: np.ndarray,
        src_y_block: np.ndarray,
    ) -> np.ndarray:
        """
        Compute fractional source indices for a single target tile.

        Args:
            tgt_x_block: Target x coordinates for this block (shape (1, nx)).
            tgt_y_block: Target y coordinates for this block (shape (ny, 1)).
            src_x_block: Source x coordinates (2D tile).
            src_y_block: Source y coordinates (2D tile).

        Returns:
            Array of shape (2, ny, nx) with fractional (ix, iy) indices.
        """
        target_x = tgt_x_block.squeeze().astype(np.float32)
        target_y = tgt_y_block.squeeze().astype(np.float32)

        nx_tgt = target_x.size
        ny_tgt = target_y.size

        idx_frac = np.full((2, ny_tgt, nx_tgt), np.nan, dtype=np.float32)

        ny_src, nx_src = src_x_block.shape

        # Quad origin indices (top-left corner per quad)
        ix0 = np.repeat(np.arange(nx_src - 1), ny_src - 1)
        iy0 = np.tile(np.arange(ny_src - 1), nx_src - 1)
        quad_ij0 = np.stack((ix0, iy0)).astype(np.int32)

        # Quad corner coordinates
        src_coords = np.stack((src_x_block, src_y_block)).astype(np.float32)
        quad_corners = np.stack(
            (
                src_coords[:, quad_ij0[1], quad_ij0[0]],
                src_coords[:, quad_ij0[1], quad_ij0[0] + 1],
                src_coords[:, quad_ij0[1] + 1, quad_ij0[0]],
                src_coords[:, quad_ij0[1] + 1, quad_ij0[0] + 1],
            ),
            axis=1,
        )

        valid = ~np.any(np.isnan(quad_corners[0]), axis=0)
        quad_corners = quad_corners[:, :, valid]
        quad_ij0 = quad_ij0[:, valid]

        if quad_corners.size == 0:
            return idx_frac

        # Target grid affine parameters (local to block)
        offset = np.array([target_x[0], target_y[0]], dtype=np.float32)
        scale = np.array(
            [target_x[1] - target_x[0], target_y[1] - target_y[0]],
            dtype=np.float32,
        )

        offset = offset[:, None, None]
        scale = scale[:, None, None]

        # Bounding target pixel indices per quad
        tgt_bbox = np.floor((quad_corners - offset) / scale).astype(np.int32)
        tgt_bbox = np.vstack((tgt_bbox.min(axis=1), tgt_bbox.max(axis=1)))

        valid = (
            (tgt_bbox[2] >= 0)
            & (tgt_bbox[3] >= 0)
            & (tgt_bbox[0] < nx_tgt)
            & (tgt_bbox[1] < ny_tgt)
        )

        tgt_bbox = tgt_bbox[:, valid]
        quad_corners = quad_corners[:, :, valid]
        quad_ij0 = quad_ij0[:, valid]

        if tgt_bbox.size == 0:
            return idx_frac

        tgt_bbox[0:2] = np.maximum(tgt_bbox[0:2], 0)
        tgt_bbox[2] = np.minimum(tgt_bbox[2], nx_tgt - 1)
        tgt_bbox[3] = np.minimum(tgt_bbox[3], ny_tgt - 1)

        _fill_pixel_indices_from_quads(
            quad_corners,
            tgt_bbox,
            idx_frac,
            target_x,
            target_y,
            quad_ij0,
        )

        return idx_frac

    # Build block-aligned target coordinate views
    tgt_x = target_gm.x_coords.data
    tgt_y = target_gm.y_coords.data

    tgt_x = da.stack([tgt_x] * len(tgt_y.chunks[0]), axis=0)
    tgt_y = da.stack([tgt_y] * len(tgt_x.chunks[1]), axis=1)

    return da.map_blocks(
        _pixel_index_block,
        tgt_x,
        tgt_y,
        src_x_coords,
        src_y_coords,
        dtype=np.float32,
        chunks=(2, tgt_y.chunks[0], tgt_x.chunks[1]),
    )


@nb.njit(nogil=True, cache=True)
def _fill_pixel_indices_from_quads(
    quad_corners: np.ndarray,
    target_bbox: np.ndarray,
    out_idx_frac: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    quad_ij0: np.ndarray,
) -> None:
    """
    Fill fractional source pixel indices for target pixels covered by source quads.

    For each source quadrilateral, this function iterates over all overlapping
    target pixels and computes barycentric coordinates using two triangle
    decompositions.

    Args:
        quad_corners: Array of quad corner coordinates (shape (2, 4, n_quads)).
        target_bbox: Target pixel bounding boxes per quad
            (shape (4, n_quads): x0, y0, x1, y1).
        out_idx_frac: Output array of fractional indices (2, ny, nx).
        target_x: Target x coordinate vector for this block.
        target_y: Target y coordinate vector for this block.
        quad_ij0: Source pixel top-left indices per quad (2, n_quads).
    """
    n_quads = quad_corners.shape[2]

    for q in range(n_quads):
        x0, y0, x1, y1 = target_bbox[:, q]

        c0 = quad_corners[:, 0, q]
        c1 = quad_corners[:, 1, q]
        c2 = quad_corners[:, 2, q]
        c3 = quad_corners[:, 3, q]

        # Triangle A
        dx1 = c1[0] - c0[0]
        dx2 = c2[0] - c0[0]
        dy1 = c1[1] - c0[1]
        dy2 = c2[1] - c0[1]
        det_a = dx1 * dy2 - dx2 * dy1

        # Triangle B
        dx1b = c3[0] - c2[0]
        dx2b = c1[0] - c2[0]
        dy1b = c3[1] - c2[1]
        dy2b = c1[1] - c2[1]
        det_b = dx1b * dy2b - dx2b * dy1b

        for i in range(x0, x1 + 1):
            tx = target_x[i]
            for j in range(y0, y1 + 1):
                if not np.isnan(out_idx_frac[0, j, i]):
                    continue

                ty = target_y[j]

                # Triangle A
                if det_a != 0.0:
                    fx = ((tx - c0[0]) * dy2 - (ty - c0[1]) * dx2) / det_a
                    fy = (dx1 * (ty - c0[1]) - dy1 * (tx - c0[0])) / det_a
                else:
                    fx = fy = np.nan

                if not np.isnan(fx) and not (
                    fx < -FX_MIN and fy < -FY_MIN and fx + fy > FXFY_MAX
                ):
                    out_idx_frac[0, j, i] = quad_ij0[0, q] + fx
                    out_idx_frac[1, j, i] = quad_ij0[1, q] + fy
                    continue

                # Triangle B
                if det_b != 0.0:
                    fx = ((tx - c3[0]) * dy2b - (ty - c3[1]) * dx2b) / det_b
                    fy = (dx1b * (ty - c3[1]) - dy1b * (tx - c3[0])) / det_b
                else:
                    fx = fy = np.nan

                if not np.isnan(fx) and not (
                    fx < -FX_MIN and fy < -FY_MIN and fx + fy > FXFY_MAX
                ):
                    out_idx_frac[0, j, i] = quad_ij0[0, q] + 1.0 - fx
                    out_idx_frac[1, j, i] = quad_ij0[1, q] + 1.0 - fy


def _reorganize_tiled_array(
    array: da.Array,
    indexing: SourceTileIndexing,
    fill_value: FloatInt,
) -> da.Array:
    """
    Reorganize a 2D or 3D array into uniformly sized tiles based on source index bboxes.
    """

    is_3d = array.ndim == 3
    if not is_3d:
        array = array[None, ...]

    pad_width = ((0, 0),) + indexing.pad_width
    padded = da.pad(array, pad_width, mode="constant", constant_values=fill_value)

    out_shape = (
        array.shape[0],
        indexing.output_size[0],
        indexing.output_size[1],
    )
    chunks = (
        array.chunks[0],
        indexing.tile_size[0],
        indexing.tile_size[1],
    )

    out = da.zeros(out_shape, chunks=chunks, dtype=array.dtype)

    ny, nx = indexing.ij_bboxes.shape[1:]
    th, tw = indexing.tile_size

    for j in range(ny):
        for i in range(nx):
            bbox = indexing.ij_bboxes[:, j, i]

            y0, y1 = j * th, (j + 1) * th
            x0, x1 = i * tw, (i + 1) * tw

            if bbox[0] == -1:
                out[:, y0:y1, x0:x1] = da.full(
                    (array.shape[0], th, tw),
                    fill_value,
                    chunks=chunks,
                    dtype=array.dtype,
                )
            else:
                out[:, y0:y1, x0:x1] = padded[
                    :,
                    bbox[1] : bbox[3],
                    bbox[0] : bbox[2],
                ]

    return out if is_3d else out[0]


def overlapping_bboxes(bbox, target_bboxes):
    block_i = []
    block_j = []
    for i in range(target_bboxes.shape[1]):
        for j in range(target_bboxes.shape[2]):
            if bbox_overlap(bbox, target_bboxes[:, i, j]) > 0:
                block_i.append(i)
                block_j.append(j)
    return block_i, block_j


def _xy_bbox_block(x_coords: np.ndarray, y_coords: np.ndarray):
    x_edges = np.concatenate([x_coords[:, 0], x_coords[:, -1]])
    y_edges = np.concatenate([y_coords[0, :], y_coords[-1, :]])
    bbox = np.array(
        [
            x_edges.min(),
            y_edges.min(),
            x_edges.max(),
            y_edges.max(),
        ],
        dtype=np.float32,
    )
    return bbox[:, None, None]


def _get_xy_bboxes(gm_2d: GridMapping):
    return da.map_blocks(
        _xy_bbox_block,
        gm_2d.x_coords.data,
        gm_2d.y_coords.data,
        dtype=np.float32,
        chunks=(4, 1, 1),
    )


def compute_source_tile_indexing(
    source_gm: GridMapping, target_gm: GridMapping
) -> SourceTileIndexing:
    target_xy_bboxes = target_gm.xy_bboxes
    source_xy_bboxes = _get_xy_bboxes(source_gm).compute()
    scr_ij_bboxes = np.full_like(target_xy_bboxes, np.nan)

    tasks = []
    meta = []
    for tile_idx, bbox in enumerate(target_xy_bboxes):
        block_i, block_j = overlapping_bboxes(bbox, source_xy_bboxes)
        if not block_i:
            continue
        i_min = source_gm.tile_height * np.min(block_i)
        i_max = source_gm.tile_height * (np.max(block_i) + 1)
        j_min = source_gm.tile_width * np.min(block_j)
        j_max = source_gm.tile_width * (np.max(block_j) + 1)

        y_coords_sub = source_gm.y_coords[i_min:i_max, j_min:j_max].data
        x_coords_sub = source_gm.x_coords[i_min:i_max, j_min:j_max].data

        mask = (
            (x_coords_sub >= bbox[0])
            & (x_coords_sub <= bbox[2])
            & (y_coords_sub >= bbox[1])
            & (y_coords_sub <= bbox[3])
        )
        rows = da.any(mask, axis=1)
        cols = da.any(mask, axis=0)
        row_idxs = da.arange(rows.shape[0], chunks=rows.chunks)
        col_idxs = da.arange(cols.shape[0], chunks=cols.chunks)
        valid_rows = da.where(rows, row_idxs, np.nan)
        valid_cols = da.where(cols, col_idxs, np.nan)

        tasks.append(
            (
                da.nanmin(valid_rows),
                da.nanmax(valid_rows),
                da.nanmin(valid_cols),
                da.nanmax(valid_cols),
            )
        )
        meta.append((tile_idx, i_min, j_min))

    results = da.compute(*tasks)
    for (rmin, rmax, cmin, cmax), (tile_idx, i_min, j_min) in zip(results, meta):
        if np.isnan(rmin):
            continue
        scr_ij_bboxes[tile_idx, 1] = rmin + i_min - 1
        scr_ij_bboxes[tile_idx, 3] = rmax + i_min + 1
        scr_ij_bboxes[tile_idx, 0] = cmin + j_min - 1
        scr_ij_bboxes[tile_idx, 2] = cmax + j_min + 1
    target_block_j = int(np.ceil(target_gm.height / target_gm.tile_height))
    target_block_i = int(np.ceil(target_gm.width / target_gm.tile_width))
    scr_ij_bboxes = scr_ij_bboxes.reshape(
        (target_block_j, target_block_i, 4)
    ).transpose((2, 0, 1))

    # Extend bounding box indices to match the largest bounding box.
    # This ensures uniform chunk sizes, which are required for da.map_blocks.
    i_diff = scr_ij_bboxes[2] - scr_ij_bboxes[0]
    j_diff = scr_ij_bboxes[3] - scr_ij_bboxes[1]
    i_diff_max = np.nanmax(i_diff) + 1
    j_diff_max = np.nanmax(j_diff) + 1
    i_half = (i_diff_max - i_diff) // 2
    j_half = (j_diff_max - j_diff) // 2
    scr_ij_bboxes[0] -= i_half
    scr_ij_bboxes[2] = scr_ij_bboxes[0] + i_diff_max
    scr_ij_bboxes[1] -= j_half
    scr_ij_bboxes[3] = scr_ij_bboxes[1] + j_diff_max

    # assign padding if needed
    i_min = np.nanmin(scr_ij_bboxes[0])
    i_max = np.nanmax(scr_ij_bboxes[2])
    j_min = np.nanmin(scr_ij_bboxes[[1, 3]])
    j_max = np.nanmax(scr_ij_bboxes[[1, 3]])
    pad_width = (
        (-min(0, int(j_min)), max(0, int(j_max - source_gm.height))),
        (-min(0, int(i_min)), max(0, int(i_max - source_gm.width))),
    )
    scr_ij_bboxes[[1, 3]] += pad_width[0][0]
    scr_ij_bboxes[[0, 2]] += pad_width[1][0]

    scr_ij_bboxes = np.where(np.isnan(scr_ij_bboxes), -1, scr_ij_bboxes).astype(
        np.int32
    )
    scr_ij_bboxes = scr_ij_bboxes
    tile_size = (int(j_diff_max), int(i_diff_max))
    size = (
        int(j_diff_max * scr_ij_bboxes.shape[1]),
        int(i_diff_max * scr_ij_bboxes.shape[2]),
    )

    return SourceTileIndexing(
        ij_bboxes=scr_ij_bboxes,
        pad_width=pad_width,
        output_size=size,
        tile_size=tile_size,
    )


def _create_empty_dataset(
    source_ds: xr.Dataset,
    source_gm: GridMapping,
    target_gm: GridMapping,
    fill_values: FillValues | None = None,
) -> xr.Dataset:
    x_name, y_name = source_gm.xy_var_names
    coords = source_ds.coords.to_dataset()
    coords = coords.drop_vars((x_name, y_name), errors="ignore")
    x_name, y_name = target_gm.xy_var_names
    coords[x_name] = target_gm.x_coords
    coords[y_name] = target_gm.y_coords
    coords["spatial_ref"] = xr.DataArray(0, attrs=target_gm.crs.to_cf())
    target_ds = xr.Dataset(coords=coords, attrs=source_ds.attrs)
    for key, data in source_ds.data_vars.items():
        shape = list(source_ds[key].shape)
        shape[-1] = target_gm.width
        shape[-2] = target_gm.height
        dims = list(source_ds[key].dims)
        dims[-1] = target_gm.xy_var_names[0]
        dims[-2] = target_gm.xy_var_names[1]
        if source_ds[key].ndim == 3:
            chunks = (
                source_ds[key].chunks[0][0],
                target_gm.height,
                target_gm.width,
            )
        else:
            chunks = (target_gm.height, target_gm.width)
        target_ds[key] = xr.DataArray(
            da.full(
                shape,
                fill_value=_get_fill_value(fill_values, key, data),
                chunks=chunks,
            ),
            dims=dims,
            attrs=source_ds[key].attrs,
        )
    return target_ds


def _transform_coords(
    source_ds: xr.Dataset,
    source_gm: GridMapping,
    target_gm: GridMapping,
) -> xr.Dataset:
    source_xx = source_gm.x_coords.data
    source_yy = source_gm.y_coords.data
    if isinstance(source_xx, np.ndarray):
        is_numpy_array = True
        source_xx = da.asarray(source_xx)
        source_yy = da.asarray(source_yy)
    else:
        is_numpy_array = False

    transformer_forward = pyproj.Transformer.from_crs(
        source_gm.crs, target_gm.crs, always_xy=True
    )

    # get transformed coordinates
    # noinspection PyShadowingNames
    def transform_block(source_xx: np.ndarray, source_yy: np.ndarray):
        target_xx, target_yy = transformer_forward.transform(source_xx, source_yy)
        return np.stack([target_xx, target_yy])

    target_xx_yy = da.map_blocks(
        transform_block,
        source_xx,
        source_yy,
        dtype=np.float32,
        chunks=(2, source_yy.chunks[0][0], source_yy.chunks[1][0]),
    )
    target_xx_yy = target_xx_yy[:, : source_gm.height, : source_gm.width]
    source_ds = source_ds.drop_vars(source_gm.xy_var_names)
    yx_dims = (source_gm.xy_dim_names[1], source_gm.xy_dim_names[0])
    yx_var_names = (
        ("lon", "lat")
        if target_gm.crs.is_geographic
        else ("transformed_x", "transformed_y")
    )
    if is_numpy_array:
        target_xx_yy = target_xx_yy.compute()
    source_ds = source_ds.assign_coords(
        {
            "spatial_ref": xr.DataArray(0, attrs=target_gm.crs.to_cf()),
            yx_var_names[0]: (yx_dims, target_xx_yy[0]),
            yx_var_names[1]: (yx_dims, target_xx_yy[1]),
        }
    )

    return source_ds


def _downscale_source_dataset(
    source_ds: xr.Dataset,
    source_gm: GridMapping,
    target_gm: GridMapping,
    interp_methods: SpatialInterpMethods | None,
    agg_methods: SpatialAggMethods | None,
    prevent_nan_propagations: PreventNaNPropagations,
) -> (xr.Dataset, GridMapping):
    if interp_methods in [0, "nearest"]:
        return source_ds, source_gm

    x_scale = source_gm.x_res / target_gm.x_res
    y_scale = source_gm.y_res / target_gm.y_res
    if x_scale >= SCALE_LIMIT and y_scale >= SCALE_LIMIT:
        return source_ds, source_gm

    w, h = np.floor(x_scale * source_gm.width), np.floor(y_scale * source_gm.height)
    downscaled_size = (w if w >= 2 else 2, h if h >= 2 else 2)

    source_ds = resample_dataset(
        source_ds,
        ((1 / x_scale, 0, 0), (0, 1 / y_scale, 0)),
        (source_gm.xy_dim_names[1], source_gm.xy_dim_names[0]),
        downscaled_size,
        source_gm.tile_size,
        _prep_spatial_interp_methods_downscale(interp_methods),
        agg_methods,
        prevent_nan_propagations,
    )
    source_gm = GridMapping.from_dataset(source_ds)

    return source_ds, source_gm


def _rectify_data_array(
    data: da.Array,
    indexing: SourceTileIndexing,
    target_gm: GridMapping,
    pixel_target_ij: da.Array,
    interp_method: SpatialInterpMethod,
    fill_value: FloatInt,
) -> da.Array:

    expanded = data.ndim == 2
    if expanded:
        data = data[None, ...]

    if isinstance(data, np.ndarray):
        data = da.asarray(data)

    tiled = _reorganize_tiled_array(data, indexing, fill_value)

    rectified_chunks = []
    offset = 0

    for chunk in data.chunks[0]:
        rectified_chunks.append(
            da.map_blocks(
                _rectify_block,
                pixel_target_ij,
                tiled[offset : offset + chunk],
                interp_method=interp_method,
                fill_value=fill_value,
                dtype=data.dtype,
                chunks=(chunk, *pixel_target_ij.chunks[1:]),
            )
        )
        offset += chunk

    result = da.concatenate(rectified_chunks, axis=0)

    if expanded:
        result = result[0]

    if isinstance(data, np.ndarray) and not target_gm.is_tiled:
        result = result.compute()

    return result


def _rectify_block(
    pixel_target_ij: np.ndarray,
    data_array: np.ndarray,
    interp_method: SpatialInterpMethodStr,
    fill_value: FloatInt,
) -> np.ndarray:
    ix = pixel_target_ij[0]
    iy = pixel_target_ij[1]

    out_shape = (data_array.shape[0],) + iy.shape
    data_rectified = np.full(out_shape, fill_value, dtype=data_array.dtype)
    valid = ~np.isnan(ix)
    iy_valid = iy[valid]
    ix_valid = ix[valid]
    if ix_valid.size == 0:
        return data_rectified

    if interp_method == "nearest":
        ix_valid = np.rint(ix_valid).astype(np.intp)
        iy_valid = np.rint(iy_valid).astype(np.intp)
        data_rectified[:, valid] = data_array[:, iy_valid, ix_valid]
    elif interp_method == "bilinear":
        ix_ceil = np.ceil(ix_valid).astype(np.intp)
        ix_floor = np.floor(ix_valid).astype(np.intp)
        iy_ceil = np.ceil(iy_valid).astype(np.intp)
        iy_floor = np.floor(iy_valid).astype(np.intp)
        diff_ix = ix_valid - ix_floor
        diff_iy = iy_valid - iy_floor
        value_00 = data_array[:, iy_floor, ix_floor]
        value_01 = data_array[:, iy_floor, ix_ceil]
        value_10 = data_array[:, iy_ceil, ix_floor]
        value_11 = data_array[:, iy_ceil, ix_ceil]
        value_u0 = value_00 + diff_ix * (value_01 - value_00)
        value_u1 = value_10 + diff_ix * (value_11 - value_10)
        data_rectified[:, valid] = value_u0 + diff_iy * (value_u1 - value_u0)
    else:
        raise NotImplementedError(
            f"interp_methods must be one of 0, 1, 'nearest', 'bilinear', "
            f"was '{interp_method}'."
        )
    return data_rectified
