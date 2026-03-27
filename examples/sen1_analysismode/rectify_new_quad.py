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

import dask.array as da
import numpy as np

from scipy.spatial import KDTree
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

import matplotlib.pyplot as plt


UV_EPS = 0.1


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

    if source_gm is None:
        source_gm = GridMapping.from_dataset(source_ds)
    source_ds = normalize_grid_mapping(source_ds, source_gm)

    if target_gm is None:
        target_gm = source_gm.to_regular(tile_size=tile_size)

    # ToDo add log if tile size of target gm is not between 1024 and 2048

    # transform 2d spatial coordinate of source dataset to target CRS
    if not _is_equal_crs(source_gm, target_gm):
        source_ds = _transform_coords(source_ds, source_gm, target_gm)
        source_gm = GridMapping.from_dataset(source_ds)

    # if the bbox of the target grid mapping overlaps less than 80% with the
    # bounding box of the source grid mapping, clip the source dataset.
    overlap = bbox_overlap(source_gm.xy_bbox, target_gm.xy_bbox)
    if overlap < 1e-5:
        LOG.info(
            "Target grid mapping does not overlap with the source grid mapping. "
            "Target dataset filled with the respective fill value is returned."
        )
        return _create_empty_dataset(source_ds, source_gm, target_gm, fill_values)
    if overlap < 0.8:
        bbox = [
            target_gm.xy_bbox[0] - 3 * source_gm.x_res,
            target_gm.xy_bbox[1] - 3 * source_gm.y_res,
            target_gm.xy_bbox[2] + 3 * source_gm.x_res,
            target_gm.xy_bbox[3] + 3 * source_gm.x_res,
        ]
        source_ds = clip_dataset_by_bbox(source_ds, bbox)
        if any(source_ds.sizes[source_gm.xy_dim_names[i]] < 2 for i in range(2)):
            LOG.warning(
                "Clipped dataset contains at least dimension with size < 2. "
                "Target dataset filled with the respective fill value is returned."
            )
            return _create_empty_dataset(source_ds, source_gm, target_gm, fill_values)
        source_gm = GridMapping.from_dataset(source_ds)

    # If source has higher resolution than target, downscale first, then rectify
    source_ds, source_gm = _downscale_source_dataset(
        source_ds,
        source_gm,
        target_gm,
        interp_methods,
        agg_methods,
        prevent_nan_propagations,
    )

    # calculate ij bboxes in source grid-mapping
    scr_ij_bboxes, pad_width, size, tile_size = _get_scr_bboxes_indices(
        source_gm, target_gm
    )

    # reorganize coordinates
    reorg_coords = []
    for var_name in source_gm.xy_var_names:
        reorg_coords.append(
            _reorganize_coords(
                source_ds[var_name].data,
                scr_ij_bboxes,
                pad_width,
                size,
                tile_size,
                _get_fill_value(fill_values, var_name, source_ds[var_name]),
            )
        )

    # calculate source pixel in target grid pixel location
    pixel_target_ij = _get_pixel_target_ij(
        reorg_coords[0], reorg_coords[1], source_gm, target_gm
    )

    # rectify dataset
    x_name, y_name = source_gm.xy_var_names
    coords = source_ds.coords.to_dataset()
    coords = coords.drop_vars((x_name, y_name), errors="ignore")
    x_name, y_name = target_gm.xy_var_names
    target_coords = target_gm.to_coords()
    coords[x_name] = target_coords[x_name]
    coords[y_name] = target_coords[y_name]
    coords["spatial_ref"] = xr.DataArray(0, attrs=target_gm.crs.to_cf())
    target_ds = xr.Dataset(coords=coords, attrs=source_ds.attrs)

    yx_dims = (source_gm.xy_dim_names[1], source_gm.xy_dim_names[0])
    for var_name, data_array in source_ds.data_vars.items():
        if data_array.dims[-2:] == yx_dims:
            assert len(data_array.dims) in (
                2,
                3,
            ), f"Data variable {var_name} has {len(data_array.dims)} dimensions."
            fill_value = _get_fill_value(fill_values, var_name, data_array)
            interp_method = _get_spatial_interp_method_str(
                interp_methods, var_name, data_array
            )
            rect_data_array = _rectify_data_array(
                data_array.data,
                reorg_coords[0],
                reorg_coords[1],
                scr_ij_bboxes,
                pad_width,
                size,
                tile_size,
                target_gm,
                pixel_target_ij,
                interp_method,
                fill_value,
            )
            dims = data_array.dims[:-2] + (
                target_gm.xy_dim_names[1],
                target_gm.xy_dim_names[0],
            )
            target_ds[var_name] = xr.DataArray(
                data=rect_data_array, dims=dims, attrs=data_array.attrs
            )

        elif yx_dims[0] not in data_array.dims and yx_dims[1] not in data_array.dims:
            target_ds[var_name] = data_array

    if output_indices_names:
        target_ds[output_indices_names[0]] = ((y_name, x_name), (pixel_target_ij[0]))
        target_ds[output_indices_names[1]] = ((y_name, x_name), (pixel_target_ij[1]))

    return target_ds


def _get_pixel_target_ij(
    x_coords: da.Array,
    y_coords: da.Array,
    source_gm: GridMapping,
    target_gm: GridMapping,
) -> da.Array:
    def _get_pixel_target_ij_block(
        target_x_coords: np.ndarray,
        target_y_coords: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        distance_upper_bound: float,
    ) -> np.ndarray:
        # ToDo save pixel value as int with fill value -1
        valid_mask = ~np.isnan(x_coords)
        points = np.column_stack((x_coords[valid_mask], y_coords[valid_mask]))
        if points.size == 0:
            return np.full((2, *target_x_coords.shape), np.nan, dtype=np.float32)

        # Remember their original ij indices
        valid_ij = np.column_stack(np.nonzero(valid_mask))

        tree = KDTree(points)
        dist, idx = tree.query(
            np.column_stack((target_x_coords.ravel(), target_y_coords.ravel())),
            k=1,
            distance_upper_bound=distance_upper_bound,
        )
        valid_dist = ~np.isinf(dist)

        # map back to original grid indices
        nearest_ij = np.full((target_x_coords.size, 2), np.nan, dtype=np.float32)
        nearest_ij[valid_dist] = valid_ij[idx[valid_dist]]
        return np.stack(
            [
                nearest_ij[:, 0].reshape(target_x_coords.shape),
                nearest_ij[:, 1].reshape(target_x_coords.shape),
            ]
        ).astype(np.float32)

    target_x_coords, target_y_coords = da.meshgrid(
        target_gm.x_coords.data.rechunk(target_gm.tile_size[0]),
        target_gm.y_coords.data.rechunk(target_gm.tile_size[1]),
    )
    pixel_target_ij = da.map_blocks(
        _get_pixel_target_ij_block,
        target_x_coords,
        target_y_coords,
        x_coords,
        y_coords,
        dtype=np.float32,
        chunks=(2, target_x_coords.chunks[0], target_x_coords.chunks[1]),
        distance_upper_bound=np.sqrt(source_gm.x_res**2 + source_gm.y_res**2),
    )

    return pixel_target_ij[:, : target_gm.size[1], : target_gm.size[0]]


def _reorganize_coords(
    array: da.Array,
    scr_ij_bboxes: np.ndarray,
    pad_width: tuple[tuple[int]],
    size: tuple[int, int],
    tile_size: tuple[int, int],
    fill_value: FloatInt,
) -> da.Array:
    data_out = da.zeros(size, chunks=tile_size, dtype=array.dtype)
    data_in = da.pad(array, pad_width, mode="constant", constant_values=fill_value)
    for i in range(scr_ij_bboxes.shape[2]):
        for j in range(scr_ij_bboxes.shape[1]):
            scr_ij_bbox = scr_ij_bboxes[:, j, i]
            if scr_ij_bbox[0] == -1:
                data_out[
                    j * tile_size[0] : (j + 1) * tile_size[0],
                    i * tile_size[1] : (i + 1) * tile_size[1],
                ] = da.full(tile_size, fill_value, chunks=tile_size, dtype=array.dtype)
            else:
                data_out[
                    j * tile_size[0] : (j + 1) * tile_size[0],
                    i * tile_size[1] : (i + 1) * tile_size[1],
                ] = data_in[
                    scr_ij_bbox[1] : scr_ij_bbox[3],
                    scr_ij_bbox[0] : scr_ij_bbox[2],
                ]
    return data_out


def _reorganize_3d_data_array(
    array: da.array,
    scr_ij_bboxes: np.ndarray,
    pad_width: tuple[tuple[int]],
    size: tuple[int, int],
    tile_size: tuple[int, int],
    fill_value: FloatInt,
) -> da.Array:
    size = (array.shape[0], size[0], size[1])
    tile_size = (array.chunksize[0], tile_size[0], tile_size[1])
    pad_width = ((0, 0), pad_width[0], pad_width[1])
    data_out = da.zeros(size, chunks=tile_size, dtype=array.dtype)
    data_in = da.pad(array, pad_width, mode="constant", constant_values=fill_value)
    for i in range(scr_ij_bboxes.shape[2]):
        for j in range(scr_ij_bboxes.shape[1]):
            scr_ij_bbox = scr_ij_bboxes[:, j, i]
            if scr_ij_bbox[0] == -1:
                data_out[
                    :,
                    j * tile_size[1] : (j + 1) * tile_size[1],
                    i * tile_size[2] : (i + 1) * tile_size[2],
                ] = da.full(tile_size, fill_value, chunks=tile_size, dtype=array.dtype)
            else:
                data_out[
                    :,
                    j * tile_size[1] : (j + 1) * tile_size[1],
                    i * tile_size[2] : (i + 1) * tile_size[2],
                ] = data_in[
                    :,
                    scr_ij_bbox[1] : scr_ij_bbox[3],
                    scr_ij_bbox[0] : scr_ij_bbox[2],
                ]
    return data_out


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


def _get_scr_bboxes_indices(
    source_gm: GridMapping, target_gm: GridMapping
) -> (np.ndarray, tuple[tuple[int]]):
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
        scr_ij_bboxes[tile_idx, 1] = rmin + i_min - 4
        scr_ij_bboxes[tile_idx, 3] = rmax + i_min + 4
        scr_ij_bboxes[tile_idx, 0] = cmin + j_min - 4
        scr_ij_bboxes[tile_idx, 2] = cmax + j_min + 4
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

    return scr_ij_bboxes, pad_width, size, tile_size


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
    if x_scale < SCALE_LIMIT or y_scale < SCALE_LIMIT:
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
    data_array: da.Array,
    x_coords: da.Array,
    y_coords: da.Array,
    scr_ij_bboxes: np.ndarray,
    path_width: tuple[tuple[int]],
    size: [int, int],
    tile_size: [int, int],
    target_gm: GridMapping,
    pixel_target_ij: da.Array,
    interp_method: SpatialInterpMethod,
    fill_value: FloatInt,
) -> da.Array:
    data_array_expanded = False
    if data_array.ndim == 2:
        data_array = data_array[None, :, :]
        data_array_expanded = True

    if isinstance(data_array, np.ndarray):
        is_numpy_array = True
        data_array = da.asarray(data_array)
    else:
        is_numpy_array = False

    reorg_data_array = _reorganize_3d_data_array(
        data_array,
        scr_ij_bboxes,
        path_width,
        size,
        tile_size,
        fill_value,
    )

    target_x_coords, target_y_coords = da.meshgrid(
        target_gm.x_coords.data.rechunk(target_gm.tile_size[0]),
        target_gm.y_coords.data.rechunk(target_gm.tile_size[1]),
    )

    # calculate rectification of each chunk along the 1st (non-spatial) dimension.
    slices_rectified = []
    dim0_end = 0
    for chunk_size in data_array.chunks[0]:
        dim0_start = dim0_end
        dim0_end = dim0_start + chunk_size

        data_rectified = da.map_blocks(
            _rectify_block,
            pixel_target_ij,
            reorg_data_array[dim0_start:dim0_end],
            x_coords,
            y_coords,
            target_x_coords,
            target_y_coords,
            dtype=data_array.dtype,
            interp_method=interp_method,
            fill_value=fill_value,
        )
        slices_rectified.append(data_rectified)
    array_rectified = da.concatenate(slices_rectified, axis=0)
    if is_numpy_array and not target_gm.is_tiled:
        array_rectified = array_rectified.compute()
    if data_array_expanded:
        array_rectified = array_rectified[0, :, :]

    return array_rectified


def _rectify_block(
    pixel_target_ij: np.ndarray,
    data_array: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    target_x_coords: np.ndarray,
    target_y_coords: np.ndarray,
    interp_method: SpatialInterpMethodStr,
    fill_value: FloatInt,
) -> np.ndarray:
    iy = pixel_target_ij[0]
    ix = pixel_target_ij[1]

    out_shape = (data_array.shape[0],) + iy.shape
    data_rectified = np.full(out_shape, fill_value, dtype=data_array.dtype)
    valid = ~np.isnan(ix)
    iy_valid = iy[valid].astype(np.intp)
    ix_valid = ix[valid].astype(np.intp)
    if ix_valid.size == 0:
        return data_rectified

    if interp_method == "nearest":
        data_rectified[:, valid] = data_array[:, iy_valid, ix_valid]
    elif interp_method == "bilinear":
        target_x_valid = target_x_coords[valid]
        target_y_valid = target_y_coords[valid]
        fx, fy, valid_fracs, ix_corners, iy_corners = bilinear_fractions(
            x_coords, y_coords, ix_valid, iy_valid, target_x_valid, target_y_valid
        )

        value_00 = data_array[:, iy_corners[:, 0], ix_corners[:, 0]]
        value_01 = data_array[:, iy_corners[:, 1], ix_corners[:, 1]]
        value_10 = data_array[:, iy_corners[:, 2], ix_corners[:, 2]]
        value_11 = data_array[:, iy_corners[:, 3], ix_corners[:, 3]]
        value_u0 = value_00 + fx * (value_01 - value_00)
        value_u1 = value_10 + fx * (value_11 - value_10)
        valid0, valid1 = np.where(valid)
        valid0 = valid0[valid_fracs]
        valid1 = valid1[valid_fracs]
        data_rectified[:, valid0, valid1] = value_u0 + fy * (value_u1 - value_u0)
    else:
        raise NotImplementedError(
            f"interp_methods must be one of 0, 1, 'nearest', 'bilinear', "
            f"was '{interp_method}'."
        )
    return data_rectified


def bilinear_fractions(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
    x_target: np.ndarray,
    y_target: np.ndarray,
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """Compute bilinear fractions (fx, fy) for targets inside quads."""

    run_idx = np.arange(y_target.size)

    fxs, fys, valid_fracs, ix_corners, iy_corners = [], [], [], [], []
    shifts = (
        # center 4 quarter
        (0, 0),
        (0, -1),
        (-1, 0),
        (-1, -1),
        # 2. ring
        (0, -2),
        (0, 1),
        (-1, -2),
        (-1, 1),
        (-2, 0),
        (1, 0),
        (-2, -1),
        (1, -1),
        (-2, -2),
        (-2, 1),
        (1, 1),
        (1, -2),
    )
    for y_shift, x_shift in shifts:
        # print(y_shift, x_shift)
        if run_idx.size == 0:
            break
        ix_temp = ix[run_idx]
        iy_temp = iy[run_idx]
        ix_corners_temp = np.column_stack(
            (
                ix_temp + x_shift,
                ix_temp + x_shift + 1,
                ix_temp + x_shift,
                ix_temp + x_shift + 1,
            )
        )
        iy_corners_temp = np.column_stack(
            (
                iy_temp + y_shift,
                iy_temp + y_shift,
                iy_temp + y_shift + 1,
                iy_temp + y_shift + 1,
            )
        )
        x_corners = np.column_stack(
            (
                x_coords[iy_corners_temp[:, 0], ix_corners_temp[:, 0]],
                x_coords[iy_corners_temp[:, 1], ix_corners_temp[:, 1]],
                x_coords[iy_corners_temp[:, 2], ix_corners_temp[:, 2]],
                x_coords[iy_corners_temp[:, 3], ix_corners_temp[:, 3]],
            )
        )
        y_corners = np.column_stack(
            (
                y_coords[iy_corners_temp[:, 0], ix_corners_temp[:, 0]],
                y_coords[iy_corners_temp[:, 1], ix_corners_temp[:, 1]],
                y_coords[iy_corners_temp[:, 2], ix_corners_temp[:, 2]],
                y_coords[iy_corners_temp[:, 3], ix_corners_temp[:, 3]],
            )
        )
        idx = np.where(x_corners[:, 0] == x_corners[:, 1])[0]
        ix_corners_temp[idx, 1] += 1
        idx = np.where(x_corners[:, 2] == x_corners[:, 3])[0]
        ix_corners_temp[idx, 3] += 1
        idx = np.where(y_corners[:, 0] == y_corners[:, 2])[0]
        iy_corners_temp[idx, 2] += 1
        idx = np.where(y_corners[:, 1] == y_corners[:, 3])[0]
        iy_corners_temp[idx, 3] += 1
        x_corners = np.column_stack(
            (
                x_coords[iy_corners_temp[:, 0], ix_corners_temp[:, 0]],
                x_coords[iy_corners_temp[:, 1], ix_corners_temp[:, 1]],
                x_coords[iy_corners_temp[:, 2], ix_corners_temp[:, 2]],
                x_coords[iy_corners_temp[:, 3], ix_corners_temp[:, 3]],
            )
        )
        y_corners = np.column_stack(
            (
                y_coords[iy_corners_temp[:, 0], ix_corners_temp[:, 0]],
                y_coords[iy_corners_temp[:, 1], ix_corners_temp[:, 1]],
                y_coords[iy_corners_temp[:, 2], ix_corners_temp[:, 2]],
                y_coords[iy_corners_temp[:, 3], ix_corners_temp[:, 3]],
            )
        )

        fx, fy, valid_frac, select_idx = _calc_fx_fy(
            x_corners,
            y_corners,
            x_target[run_idx],
            y_target[run_idx],
            run_idx,
        )
        if fx is None:
            break
        # plt.scatter(x_corners[:, 0], y_corners[:, 0], color="red", marker="D")
        # plt.scatter(x_corners[:, 1], y_corners[:, 1], color="orange", marker="D")
        # plt.scatter(x_corners[:, 2], y_corners[:, 2], color="pink", marker="D")
        # plt.scatter(x_corners[:, 3], y_corners[:, 3], color="yellow", marker="D")
        # plt.scatter(
        #     x_target[run_idx].flatten(),
        #     y_target[run_idx].flatten(),
        #     color="blue",
        #     marker="o",
        # )
        # plt.scatter(
        #     x_target[valid_frac].flatten(),
        #     y_target[valid_frac].flatten(),
        #     color="green",
        #     marker="x",
        # )
        # plt.show()

        fxs.append(fx)
        fys.append(fy)
        ix_corners.append(ix_corners_temp[select_idx, :])
        iy_corners.append(iy_corners_temp[select_idx, :])

        valid_fracs.append(valid_frac)
        run_idx = run_idx[~np.isin(run_idx, valid_frac)]

    valid_frac = np.concatenate(valid_fracs)
    fx = np.concatenate(fxs)
    fy = np.concatenate(fys)
    ix_corner = np.concatenate(ix_corners)
    iy_corner = np.concatenate(iy_corners)

    return fx, fy, valid_frac, ix_corner, iy_corner


def _calc_fx_fy(x_corners, y_corners, x_target, y_target, run_idx):

    select_idx = np.arange(x_target.size)

    valid = np.all(~np.isnan(x_corners), axis=1)
    x_corners = x_corners[valid]
    y_corners = y_corners[valid]
    x_target = x_target[valid]
    y_target = y_target[valid]
    run_idx = run_idx[valid]
    select_idx = select_idx[valid]

    if x_corners.size == 0:
        return None, None, None, None

    u, v = invert_bilinear_quad(x_corners, y_corners, x_target, y_target)

    valid_corners = (
        (u >= -UV_EPS) & (u <= 1 + UV_EPS) & (v >= -UV_EPS) & (v <= 1 + UV_EPS)
    )

    return (
        u[valid_corners],
        v[valid_corners],
        run_idx[valid_corners],
        select_idx[valid_corners],
    )


def invert_bilinear_quad(xc, yc, xt, yt, max_iter=8, tol=1e-6):
    """
    Invert bilinear quad mapping for many points at once.

    xc, yc : (N,4) corner coordinates in order
             [P0, P1, P2, P3] =
             [(0,0), (1,0), (0,1), (1,1)] in logical cell space

    xt, yt : (N,) target coordinates

    Returns
    -------
    u, v : bilinear coordinates (same shape as xt)
    """

    select_idx = np.arange(xt.size)

    x0, x1, x2, x3 = xc.T
    y0, y1, y2, y3 = yc.T

    # Bilinear coefficients
    ax = x0
    bx = x1 - x0
    cx = x2 - x0
    dx = x3 - x2 - x1 + x0

    ay = y0
    by = y1 - y0
    cy = y2 - y0
    dy = y3 - y2 - y1 + y0

    # ---- Initial guess (affine approx, better than 0.5) ----
    det_aff = bx * cy - cx * by
    det_aff[det_aff == 0] = np.nan

    u = ((xt - ax) * cy - (yt - ay) * cx) / det_aff
    v = (bx * (yt - ay) - by * (xt - ax)) / det_aff

    # Fallback where affine fails
    bad = np.isnan(u) | np.isnan(v)
    u[bad] = 0.5
    v[bad] = 0.5

    # ---- Newton iterations ----
    for _ in range(max_iter):
        xu = ax + bx * u + cx * v + dx * u * v
        yu = ay + by * u + cy * v + dy * u * v

        Fx = xu - xt
        Fy = yu - yt

        J11 = bx + dx * v  # dx/du
        J12 = cx + dx * u  # dx/dv
        J21 = by + dy * v  # dy/du
        J22 = cy + dy * u  # dy/dv

        detJ = J11 * J22 - J12 * J21
        small = np.abs(detJ) < 1e-12
        detJ[small] = np.nan

        du = (Fx * J22 - Fy * J12) / detJ
        dv = (-Fx * J21 + Fy * J11) / detJ

        u -= du
        v -= dv

        if np.nanmax(np.abs(du) + np.abs(dv)) < tol:
            break

    return u, v
