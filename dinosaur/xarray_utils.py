# Copyright 2023 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions for converting trajectories to xarray datasets."""

import dataclasses
import functools
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from dinosaur import coordinate_systems
from dinosaur import layer_coordinates
from dinosaur import primitive_equations
from dinosaur import scales
from dinosaur import shallow_water
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic
from dinosaur import typing
from dinosaur import vertical_interpolation
from dinosaur import weatherbench_utils
import fsspec
import jax
import numpy as np
import pandas as pd
from sklearn import neighbors
import xarray
import xarray_tensorstore


# pylint: disable=g-bare-generic
# pylint: disable=line-too-long

# Axes and coordinate names
XR_SAMPLE_NAME = 'sample'
XR_TIME_NAME = 'time'
XR_INIT_TIME_NAME = 'initial_time'
XR_TIMEDELTA_NAME = 'prediction_timedelta'
XR_LEVEL_NAME = 'level'
XR_SURFACE_NAME = 'surface'
XR_LON_NAME = 'lon'
XR_LAT_NAME = 'lat'
XR_LON_MODE_NAME = 'longitudinal_mode'
XR_LAT_MODE_NAME = 'total_wavenumber'
XR_AUX_FEATURES_LIST_KEY = 'aux_features_key'
XR_REALIZATION_NAME = 'realization'

# Key names for auxiliary features
# LAND_SEA_MASK defined in static single-level variables section
OROGRAPHY = 'orography'  # key for referring to orography.
GEOPOTENTIAL_KEY = 'geopotential'  # key for referring to geopotential.
GEOPOTENTIAL_AT_SURFACE_KEY = 'geopotential_at_surface'
REF_TEMP_KEY = 'ref_temperatures'  # key for referring to reference temperature.
REF_POTENTIAL_KEY = 'reference_potential'  # key for referring to ref potential.
REFERENCE_DATETIME_KEY = 'reference_datetime'  # key for referencing 0-time.
XARRAY_DS_KEY = 'xarray_dataset'

# Key names for static single-level variables
# comment line has {long_name, short_name, units} from CDS.
GEOPOTENTIAL_AT_SURFACE = 'geopotential_at_surface'  # Geopotential, z, m**2 s**-2
HIGH_VEGETATION_COVER = 'high_vegetation_cover'  # High vegetation cover, cvh, (0 - 1)
LAKE_COVER = 'lake_cover'  # Lake cover, cl, (0 - 1)
LAKE_DEPTH = 'lake_depth'  # Lake total depth, dl, m
LAND_SEA_MASK = 'land_sea_mask'  # Land-sea mask, lsm, (0 - 1)
LOW_VEGETATION_COVER = 'low_vegetation_cover'  # Low vegetation cover, cvl, (0 - 1)
SOIL_TYPE = 'soil_type'  # Soil type, slt, ~
TYPE_OF_HIGH_VEGETATION = 'type_of_high_vegetation'  # Type of high vegetation, tvh, ~
TYPE_OF_LOW_VEGETATION = 'type_of_low_vegetation'  # Type of low vegetation, tvl, ~
single_level_static_vars = [
    GEOPOTENTIAL_AT_SURFACE,
    HIGH_VEGETATION_COVER,
    LAKE_COVER,
    LAKE_DEPTH,
    LAND_SEA_MASK,
    LOW_VEGETATION_COVER,
    SOIL_TYPE,
    TYPE_OF_HIGH_VEGETATION,
    TYPE_OF_LOW_VEGETATION,
]

# Key names for dynamic single-level variables
# comment line has {long_name, short_name, units} from CDS.
ICE_TEMPERATURE_LAYER_4 = 'ice_temperature_layer_4'  # Ice temperature layer 4, istl4, K
LAKE_ICE_DEPTH = 'lake_ice_depth'  # Lake ice total depth, licd, m
LAKE_ICE_TEMPERATURE = 'lake_ice_temperature'  # Lake ice surface temperature, lict, K
SEA_ICE_COVER = 'sea_ice_cover'  # Sea ice area fraction, siconc, (0 - 1)
SEA_SURFACE_TEMPERATURE = 'sea_surface_temperature'  # Sea surface temperature, sst, K
SNOW_DEPTH = 'snow_depth'  # Snow depth, sd, m of water equivalent
SOIL_TEMPERATURE_LEVEL_4 = 'soil_temperature_level_4'  # Soil temperature level 4, stl4, K
VOLUMETRIC_SOIL_WATER_LAYER_4 = 'volumetric_soil_water_layer_4'  # Volumetric soil water layer 4, swvl4, m**3 m**-3
single_level_dynamic_vars = [
    ICE_TEMPERATURE_LAYER_4,
    LAKE_ICE_DEPTH,
    LAKE_ICE_TEMPERATURE,
    SEA_ICE_COVER,
    SEA_SURFACE_TEMPERATURE,
    SNOW_DEPTH,
    SOIL_TEMPERATURE_LEVEL_4,
    VOLUMETRIC_SOIL_WATER_LAYER_4,
]

# Axes for `Dataset`s in the nodal/spatial harmonic basis.
NODAL_AXES_NAMES = (
    XR_LON_NAME,
    XR_LAT_NAME,
)

MODAL_AXES_NAMES = (
    XR_LON_MODE_NAME,
    XR_LAT_MODE_NAME,
)

GRID_REGISTRY = {
    'SigmaCoordinates': sigma_coordinates.SigmaCoordinates,
    'LayerCoordinates': layer_coordinates.LayerCoordinates,
    'PressureCoordinates': vertical_interpolation.PressureCoordinates,
    'Grid': spherical_harmonic.Grid,
    'RealSphericalHarmonics': spherical_harmonic.RealSphericalHarmonics,
    'RealSphericalHarmonicsWithZeroImag':
        spherical_harmonic.RealSphericalHarmonicsWithZeroImag,
    'ComplexSphericalHarmonics': spherical_harmonic.ComplexSphericalHarmonics,
}


# Types of horizontal grids supported by Dinosaur codebase.
LINEAR = 'LINEAR'
CUBIC = 'CUBIC'


Grid = spherical_harmonic.Grid
CUBIC_SHAPE_TO_GRID_DICT = {
    Grid.T21().nodal_shape: Grid.T21,
    Grid.T31().nodal_shape: Grid.T31,
    Grid.T42().nodal_shape: Grid.T42,
    Grid.T85().nodal_shape: Grid.T85,
    Grid.T106().nodal_shape: Grid.T106,
    Grid.T119().nodal_shape: Grid.T119,
    Grid.T170().nodal_shape: Grid.T170,
    Grid.T213().nodal_shape: Grid.T213,
    Grid.T340().nodal_shape: Grid.T340,
    Grid.T425().nodal_shape: Grid.T425,
}
LINEAR_SHAPE_TO_GRID_DICT = {
    Grid.TL31().nodal_shape: Grid.TL31,
    Grid.TL47().nodal_shape: Grid.TL47,
    Grid.TL63().nodal_shape: Grid.TL63,
    Grid.TL95().nodal_shape: Grid.TL95,
    Grid.TL127().nodal_shape: Grid.TL127,
    Grid.TL159().nodal_shape: Grid.TL159,
    Grid.TL179().nodal_shape: Grid.TL179,
    Grid.TL255().nodal_shape: Grid.TL255,
    Grid.TL639().nodal_shape: Grid.TL639,
    Grid.TL1279().nodal_shape: Grid.TL1279,
}


def is_dir(path: str) -> bool:
  protocol, path = fsspec.core.split_protocol(path)
  fs = fsspec.filesystem(protocol=protocol)
  return fs.isdir(path)


def open_dataset(
    path: str,
    **kwargs,
) -> xarray.Dataset:  # pylint: disable=redefined-builtin
  """Load a dataset from either Zarr or NetCDF."""
  if is_dir(path):
    return xarray_tensorstore.open_zarr(path, **kwargs)
  else:
    return open_netcdf(path, **kwargs)


def open_netcdf(
    path: str,
    max_parallel_reads: int | None = None, **kwargs
) -> xarray.Dataset:
  """Load a dataset stored in NetCDF format."""
  del max_parallel_reads  # unused.
  with fsspec.open(path, 'rb') as f:
    return xarray.load_dataset(f.read(), **kwargs)


def save_netcdf(dataset: xarray.Dataset, path: str):
  """Save a dataset in the NetCDF file format."""
  with fsspec.open(path, 'wb') as f:
    f.write(dataset.to_netcdf())


def _maybe_update_shape_and_dim_with_realization_time_sample(
    shape: tuple[int, ...],
    dims: tuple[str, ...],
    times: typing.Array,
    sample_ids: typing.Array,
    include_realization: bool,
) -> tuple[Sequence[int], Sequence[str]]:
  """Shape and dims with prepended realization/sample/time values if provided.

  Note that we assume that `realization`, `sample`, and `time` dimensions appear
  in the order `[realization, sample, time, ...]`.

  Args:
    shape: array shape excluding potential `time` and `sample` axes.
    dims: names of the axes corresponding to `shape`.
    times: expected time values. If `None` time shape/dim is not added.
    sample_ids: expected sample values. If `None` sample shape/dim is not added.
    include_realization: Whether to prepend a `realization` dim to non-scalars.

  Returns:
    shape: `shape` with prepended `sample` and `time` shapes if provided.
    dims: dimension names with prepended `sample` and `time` dims if provided.
  """
  not_scalar = bool(shape)

  if times is not None:
    shape = times.shape + shape
    dims = (XR_TIME_NAME,) + dims
  if sample_ids is not None:
    shape = sample_ids.shape + shape
    dims = (XR_SAMPLE_NAME,) + dims
  if not_scalar and include_realization:
    shape = (1,) + shape
    dims = (XR_REALIZATION_NAME,) + dims
  return shape, dims


def _infer_dims_shape_and_coords(
    coords: coordinate_systems.CoordinateSystem,
    times: typing.Array | None,
    sample_ids: typing.Array,
    additional_coords: Mapping[str, typing.Array],
) -> tuple[dict[str, typing.Array], dict[tuple[int, ...], tuple[int, ...]]]:
  """Returns full coordinates for given grids and default shape to dims mapping.

  Args:
    coords: horizontal and vertical descritization.
    times: expected time values. If `None` time shape/dim is not added.
    sample_ids: expected sample values. If `None` sample shape/dim is not added.
    additional_coords: additional coordinates to include.

  Returns:
    all_coords: mapping that represents all supported coordinates.
    shape_to_dims: mapping from array shape to dimensions. `sample` is assumed
      to come prior to `time`.
  """
  lon_k, lat_k = coords.horizontal.modal_axes  # k stands for wavenumbers
  lon, sin_lat = coords.horizontal.nodal_axes
  all_xr_coords = {
      XR_LON_NAME: lon * 180 / np.pi,
      XR_LAT_NAME: np.arcsin(sin_lat) * 180 / np.pi,
      XR_LON_MODE_NAME: lon_k,
      XR_LAT_MODE_NAME: lat_k,
      XR_LEVEL_NAME: coords.vertical.centers,
      **additional_coords
  }
  if times is not None:
    all_xr_coords[XR_TIME_NAME] = times
  if sample_ids is not None:
    all_xr_coords[XR_SAMPLE_NAME] = sample_ids
  basic_shape_to_dims = {}
  basic_shape_to_dims[tuple()] = tuple()  # scalar variables
  modal_shape = coords.horizontal.modal_shape
  nodal_shape = coords.horizontal.nodal_shape
  basic_shape_to_dims[(coords.vertical.layers,) + modal_shape] = (
      (XR_LEVEL_NAME,) + MODAL_AXES_NAMES)
  basic_shape_to_dims[(coords.vertical.layers,) + nodal_shape] = (
      (XR_LEVEL_NAME,) + NODAL_AXES_NAMES)
  basic_shape_to_dims[nodal_shape] = NODAL_AXES_NAMES
  basic_shape_to_dims[modal_shape] = MODAL_AXES_NAMES
  # Add unconventional shape for nodal covariate surface data, which have dim=2
  # (lon, lat) in xarray. The singleton dimension for level is added when
  # converting to covariate data.
  basic_shape_to_dims[coords.surface_nodal_shape] = NODAL_AXES_NAMES
  for dim, value in additional_coords.items():
    if dim == XR_REALIZATION_NAME:
      continue  # Handled in _maybe_update_shape_and_dim_with_time_sample
    if value.ndim != 1:
      raise ValueError(
          '`additional_coords` must be 1d vectors, but got: '
          f'{value.shape=} for {dim=}'
      )
    if value.shape == (coords.vertical.layers,):
      raise ValueError(
          f'`additional_coords` {dim=} has shape={value.shape} that collides '
          f'with {XR_LEVEL_NAME=}. Since matching of axes is done using shape, '
          'consider renaming after the fact.'
      )
    basic_shape_to_dims[value.shape + modal_shape] = (dim,) + MODAL_AXES_NAMES
    basic_shape_to_dims[value.shape + nodal_shape] = (dim,) + NODAL_AXES_NAMES
    basic_shape_to_dims[value.shape] = (dim,)

  update_shape_dims_fn = functools.partial(
      _maybe_update_shape_and_dim_with_realization_time_sample,
      times=times,
      sample_ids=sample_ids,
      include_realization=XR_REALIZATION_NAME in additional_coords,
  )
  shape_to_dims = {}
  for shape, dims in basic_shape_to_dims.items():
    full_shape, full_dims = update_shape_dims_fn(shape, dims)
    shape_to_dims[full_shape] = full_dims
  return all_xr_coords, shape_to_dims  # pytype: disable=bad-return-type


def nodal_orography_from_ds(ds: xarray.Dataset) -> typing.Array:
  """Returns orography in nodal representation from `ds`."""
  orography_key = OROGRAPHY
  if orography_key not in ds:
    ds[orography_key] = (ds[GEOPOTENTIAL_AT_SURFACE_KEY]
                         / scales.GRAVITY_ACCELERATION.magnitude)
  lon_lat_order = (XR_LON_NAME, XR_LAT_NAME)
  return ds[orography_key].transpose(*lon_lat_order).values


def nodal_land_sea_mask_from_ds(ds: xarray.Dataset) -> typing.Array:
  """Returns land_sea_mask in nodal representation from `ds`."""
  land_sea_mask_key = LAND_SEA_MASK
  lon_lat_order = ('longitude', 'latitude')
  return ds[land_sea_mask_key].transpose(*lon_lat_order).values


def coordinate_system_from_attrs(
    attrs: ...,
) -> coordinate_systems.CoordinateSystem:
  """Creates a `CoordinateSystem` object based on `attrs`."""
  horizontal_coordinate_cls = GRID_REGISTRY[
      attrs[coordinate_systems.HORIZONTAL_COORD_TYPE_KEY]]
  horizontal_attrs = {
      f.name: attrs[f.name]
      for f in dataclasses.fields(horizontal_coordinate_cls)}
  horizontal_attrs.pop(spherical_harmonic.SPHERICAL_HARMONICS_IMPL_KEY, None)
  horizontal_attrs.pop(spherical_harmonic.SPMD_MESH_KEY, None)
  horizontal = horizontal_coordinate_cls(**horizontal_attrs)
  if coordinate_systems.VERTICAL_COORD_TYPE_KEY in attrs:
    vertical_coordinate_cls = GRID_REGISTRY[
        attrs[coordinate_systems.VERTICAL_COORD_TYPE_KEY]]
    vertical_attrs = {
        f.name: attrs[f.name]
        for f in dataclasses.fields(vertical_coordinate_cls)}
    vertical = vertical_coordinate_cls(**vertical_attrs)
  else:
    vertical = None  # no vertical coordinate has been specified.
  return coordinate_systems.CoordinateSystem(horizontal, vertical)


def data_to_xarray(
    data: dict,
    *,
    coords: coordinate_systems.CoordinateSystem,
    times: typing.Array | None,
    sample_ids: typing.Array | None = None,
    additional_coords: MutableMapping[str, typing.Array] | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> xarray.Dataset:
  """Returns a sample/time referenced xarray.Dataset of primitive equation data.

  Args:
    data: dictionary representation of the primitive equation states.
    coords: horizontal and vertical descritization.
    times: xarray coordinates to use for `time` axis.
    sample_ids: xarray coordinates to use for `sample` axis.
    additional_coords: additional coordinates to include.
    attrs: additional attributes to include in the xarray.Dataset metadata.

  Returns:
    xarray.Dataset with containing `data`.
  """
  # check that prognostic and tracer names do not collide;
  prognostic_keys = set(data.keys()) - {'tracers'} - {'diagnostics'}
  tracer_keys = data['tracers'].keys() if 'tracers' in data else set()
  diagnostic_keys = (data['diagnostics'].keys() if 'diagnostics' in data
                     else set())
  if not prognostic_keys.isdisjoint(tracer_keys):
    raise ValueError('Tracer names collide with prognostic variables',
                     f'Tracers: {tracer_keys}; prognostics: {prognostic_keys}')
  if not prognostic_keys.isdisjoint(diagnostic_keys):
    raise ValueError('Diagnostic names collide with prognostic variables',
                     f'Diagnostic: {diagnostic_keys}; ',
                     f'prognostics: {prognostic_keys}')

  if additional_coords is None:
    additional_coords = {}
  # if XR_SURFACE_NAME is not specified manually, set by default.
  if (isinstance(coords.vertical, sigma_coordinates.SigmaCoordinates) and
      XR_SURFACE_NAME not in additional_coords):
    additional_coords[XR_SURFACE_NAME] = np.ones(1)

  all_coords, shape_to_dims = _infer_dims_shape_and_coords(
      coords, times, sample_ids, additional_coords)

  dims_in_state = set()  # keep track which coordinates should be included.
  data_vars = {}
  for key in prognostic_keys:
    value = data[key]
    if value.shape not in shape_to_dims:
      raise ValueError(
          f'Value of shape {value.shape} is not in {shape_to_dims=}')
    else:
      dims = shape_to_dims[value.shape]
      data_vars[key] = (dims, value)
      dims_in_state.update(set(dims))

  for key in tracer_keys:
    value = data['tracers'][key]
    if value.shape not in shape_to_dims:
      raise ValueError(f'Value of shape {value.shape} is not recognized.')
    else:
      dims = shape_to_dims[value.shape]
      data_vars[key] = (dims, value)
      dims_in_state.update(set(dims))

  for key in diagnostic_keys:
    value = data['diagnostics'][key]
    if value.shape not in shape_to_dims:
      raise ValueError(f'Value of shape {value.shape} is not recognized.')
    else:
      dims = shape_to_dims[value.shape]
      data_vars[key] = (dims, value)
      dims_in_state.update(set(dims))

  dataset_attrs = coords.asdict()
  if attrs is not None:
    for key in dataset_attrs.keys():
      if key in attrs:
        raise ValueError(f'Key {key} is not allowed in `attrs`.')
    dataset_attrs.update(attrs)
  # only include coordinates for dimensions that are present in the dataset.
  coords = {k: v for k, v in all_coords.items() if k in dims_in_state}
  return xarray.Dataset(
      data_vars,
      coords,
      attrs=dataset_attrs
  )


def dynamic_covariate_data_to_xarray(
    data: dict,
    *,
    coords: coordinate_systems.CoordinateSystem,
    times: typing.Array | None,
    sample_ids: typing.Array | None = None,
    additional_coords: MutableMapping[str, typing.Array] | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> xarray.Dataset:
  """Returns an xarray.Dataset of dynamic covariate data.

  Args:
    data: dictionary representation of the dynamic_covariates.
    coords: horizontal and vertical descritization.
    times: xarray coordinates to use for `time` axis.
    sample_ids: xarray coordinates to use for `sample` axis.
    additional_coords: additional coordinates to include.
    attrs: additional attributes to include in the xarray.Dataset metadata.

  Returns:
    xarray.Dataset containing `data`.
  """
  if additional_coords is None:
    additional_coords = {}

  all_coords, shape_to_dims = _infer_dims_shape_and_coords(
      coords, times, sample_ids, additional_coords)

  dims_in_state = set()  # keep track which coordinates should be included.
  data_vars = {}
  for key in data.keys():
    value = data[key]
    if value.shape not in shape_to_dims:
      raise ValueError(f'Value of shape {value.shape} is not recognized.')
    else:
      dims = shape_to_dims[value.shape]
      data_vars[key] = (dims, np.squeeze(value))  # remove singleton dims
      dims_in_state.update(set(dims))

  dataset_attrs = coords.asdict()
  if attrs is not None:
    for key in dataset_attrs.keys():
      if key in attrs:
        raise ValueError(f'Key {key} is not allowed in `attrs`.')
    dataset_attrs.update(attrs)

  # only include coordinates for dimensions that are present in the dataset.
  xr_coords = {k: v for k, v in all_coords.items() if k in dims_in_state}
  return xarray.Dataset(
      data_vars,
      xr_coords,
      attrs=dataset_attrs
  )


def xarray_to_shallow_water_eq_data(
    dataset: xarray.Dataset,
    *,
    values: str = 'values',
) -> dict[str, Any]:
  """Returns `values` attribute of shallow water equation data from `dataset`.

  Args:
    dataset: dataset from which to extract a dictionary representation of the
      `shallow_water.State`.
    values: attribute to extract. Typically is `values` or `dtype`.

  Returns:
    Dictionary that contains atmosphere state variables in a format compatible
    with `shallow_water.State`.
  """
  return shallow_water.State(
      vorticity=getattr(dataset['vorticity'], values),
      divergence=getattr(dataset['divergence'], values),
      potential=getattr(dataset['potential'], values),).asdict()


def xarray_to_primitive_eq_data(
    dataset: xarray.Dataset,
    *,
    values: str = 'values',
    tracers_to_include: Sequence[str] = tuple(),
) -> dict:
  """Returns `values` attribute of primitive equation data from `dataset`.

  Args:
    dataset: dataset from which to extract a dictionary representation of the
      `primitive_equations.State`.
    values: attribute to extract. Typically is `values` or `dtype`.
    tracers_to_include: names of tracers present in the `dataset` to include in
      the `State`.

  Returns:
    Dictionary that contains atmosphere state variables in a format compatible
    with `primitive_equations.State`.
  """
  return primitive_equations.State(
      vorticity=getattr(dataset['vorticity'], values),
      divergence=getattr(dataset['divergence'], values),
      temperature_variation=getattr(dataset['temperature_variation'], values),
      log_surface_pressure=getattr(dataset['log_surface_pressure'], values),
      tracers={k: getattr(dataset[k], values) for k in tracers_to_include}
      ).asdict()


def xarray_to_primitive_equations_with_time_data(
    dataset: xarray.Dataset,
    *,
    values: str = 'values',
    tracers_to_include: Sequence[str] = tuple(),
) -> dict:
  """Returns `values` of primitive equation data with time from `dataset`.

  Args:
    dataset: dataset from which to extract `values` attributes of the
      `primitive_equations.StateWithTime` in dict representation.
    values: attribute to extract. Typically is `values`, `dtype` or `shape`.
    tracers_to_include: names of tracers present in the `dataset` to include in
      the `StateWithTime`.

  Returns:
    Dictionary that contains atmosphere state variables in a format compatible
    with `primitive_equations.State`.
  """
  return primitive_equations.StateWithTime(
      vorticity=getattr(dataset['vorticity'], values),
      divergence=getattr(dataset['divergence'], values),
      temperature_variation=getattr(dataset['temperature_variation'], values),
      log_surface_pressure=getattr(dataset['log_surface_pressure'], values),
      sim_time=getattr(dataset['sim_time'], values),
      tracers={k: getattr(dataset[k], values) for k in tracers_to_include}
  ).asdict()


def xarray_to_weatherbench_data(
    dataset: xarray.Dataset,
    *,
    values: str = 'values',
    tracers_to_include: Sequence[str] = tuple(),
    diagnostics_to_include: Sequence[str] = tuple(),
) -> dict:
  """Returns `values` of weatherbench data with time from `dataset`.

  Args:
    dataset: dataset from which to extract `values` attributes of the
      `weatherbench.State` in dict representation.
    values: attribute to extract. Typically is `values`, `dtype` or `shape`.
    tracers_to_include: names of tracers present in the `dataset` to include in
      the `weatherbench.State`.
    diagnostics_to_include: names of diagnostics present in the `dataset` to
      include in the `weatherbench.State`.

  Returns:
    Dictionary that contains atmosphere state variables in a format compatible
    with `weatherbench.State`.
  """
  return weatherbench_utils.State(
      u=getattr(dataset['u'], values),
      v=getattr(dataset['v'], values),
      t=getattr(dataset['t'], values),
      z=getattr(dataset['z'], values),
      sim_time=getattr(dataset['sim_time'], values),
      tracers={k: getattr(dataset[k], values) for k in tracers_to_include},
      diagnostics={
          k: getattr(dataset[k], values) for k in diagnostics_to_include},
  ).asdict()


def xarray_to_dynamic_covariate_data(
    dataset: xarray.Dataset,
    *,
    values: str = 'values',
    covariates_to_include: Sequence[str] = tuple(),
) -> dict:
  """Returns `values` of dynamic covariate data with time from `dataset`.

  Args:
    dataset: dataset from which to extract `values` attributes of the
      dynamic covariates in dict representation.
    values: attribute to extract. Typically is `values`, `dtype` or `shape`.
    covariates_to_include: names of covariates present in the `dataset` to
      include in the `dynamic_covariate_data`.

  Returns:
    Dictionary that contains time-varying covariate variables, and also the
    array `sim_time` that specifies the nondimensionalized time_axis.
  """
  data = {}
  for k in covariates_to_include:
    v = getattr(dataset[k], values)
    dims = dataset[k].dims
    if 'level' not in dims and dims[-3:] == ('time', 'lon', 'lat'):
      # surface quantity
      v = np.expand_dims(v, axis=-3)  # singleton dim for level
    data[k] = v
  data['sim_time'] = getattr(dataset['sim_time'], values)
  return data


def xarray_to_state_and_dynamic_covariate_data(
    dataset: xarray.Dataset,
    *,
    values: str = 'values',
    xarray_to_state_data_fn: Callable[..., dict],
    xarray_to_dynamic_covariate_data_fn: Callable[..., dict] | None = None,
) -> tuple[dict, dict]:
  """Returns `values` of state and dynamic covariate data from `dataset`.

  Args:
    dataset: dataset from which to extract `values` attributes of the
      `weatherbench.State` in dict representation.
    values: attribute to extract. Typically is `values`, `dtype` or `shape`.
    xarray_to_state_data_fn: function converting xarray.Dataset to state dict.
    xarray_to_dynamic_covariate_data_fn: function converting xarray.Dataset to
      dynamic covariates dictionary. If None, function returns an empty
      dictionary for the dynamic covariate data.

  Returns:
    Tuple (state, dynamic_covariates), where state is a dict that contains
    atmosphere state variables, and dynamic_covariates is a dict that contains
    time-varying covariate variables.
  """
  state_data = xarray_to_state_data_fn(dataset, values=values)
  if xarray_to_dynamic_covariate_data_fn is None:
    covariate_data = {}
  else:
    covariate_data = xarray_to_dynamic_covariate_data_fn(dataset, values=values)
  return (state_data, covariate_data)


def xarray_to_data_with_renaming(
    dataset: xarray.Dataset,
    *,
    values: str = 'values',
    xarray_to_data_fn: Callable[..., dict],
    renaming_dict: dict[str, str],
) -> dict:
  """Adapts naming convention before calling `xarray_to_data_fn`."""
  return xarray_to_data_fn(dataset.rename(renaming_dict), values=values)


def data_to_xarray_with_renaming(
    data: dict,
    *,
    to_xarray_fn: Callable[..., xarray.Dataset],
    renaming_dict: dict[str, str],
    coords: coordinate_systems.CoordinateSystem,
    times: typing.Array | None,
    sample_ids: typing.Array | None = None,
    additional_coords: MutableMapping[str, typing.Array] | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> xarray.Dataset:
  """Adapts naming convention after calling `to_xarray_fn`.

  Args:
    data: dictionary representation of the data states.
    to_xarray_fn: wrapped function converting `data` to xarray.Dataset.
    renaming_dict: dictionary that maps desired variable names to our defaults.
    coords: horizontal and vertical descritization.
    times: xarray coordinates to use for `time` axis.
    sample_ids: xarray coordinates to use for `sample` axis.
    additional_coords: additional coordinates to include.
    attrs: additional attributes to include in the xarray.Dataset metadata.

  Returns:
    xarray.Dataset that stores `data` using external naming convention specified
    in `renaming_dict`.
  """
  inverse_ranaming_dict = {v: k for k, v in renaming_dict.items()}
  ds = to_xarray_fn(
      data, coords=coords, times=times, sample_ids=sample_ids,
      additional_coords=additional_coords, attrs=attrs)
  return ds.rename(inverse_ranaming_dict)


def aux_features_from_xarray(ds: xarray.Dataset) -> typing.AuxFeatures:
  """Reads `aux_features` from an Xarray dataset."""
  # we split string on `,` because some file formats (e.g. netcdf) do not
  # support lists of strings in the attrs.
  aux_keys = ds.attrs[XR_AUX_FEATURES_LIST_KEY].split(',')
  return {k: ds[k].values for k in aux_keys}


def aux_features_to_xarray(
    aux_features: typing.AuxFeatures,
    xr_coords: Mapping[str, np.ndarray] | None = None,
) -> xarray.Dataset:
  """Creates an Xarray dataset containing aux_features.

  Args:
    aux_features: dictionary holding auxiliary features.
    xr_coords: coordinates for the dataset.

  Returns:
    xarray.Dataset that holds `aux_features`.
  """
  if xr_coords is None:
    xr_coords = {}
  data_vars = {}
  for k, v in aux_features.items():
    if k == OROGRAPHY:
      data_vars[k] = ((XR_LON_NAME, XR_LAT_NAME), v)
    elif k == LAND_SEA_MASK:
      data_vars[k] = ((XR_LON_NAME, XR_LAT_NAME), v)
    elif k == REF_TEMP_KEY:
      data_vars[k] = ((XR_LEVEL_NAME,), v)
    elif k == REF_POTENTIAL_KEY:
      data_vars[k] = ((XR_LEVEL_NAME,), v)
    elif k == REFERENCE_DATETIME_KEY:
      data_vars[k] = (tuple(), v)
    else:
      raise ValueError(f'Got unrecognized aux_feature {k}')
  attrs = {XR_AUX_FEATURES_LIST_KEY: ','.join(list(aux_features.keys()))}
  return xarray.Dataset(data_vars=data_vars, coords=xr_coords, attrs=attrs)


def attach_data_array_units(array: xarray.DataArray) -> xarray.DataArray:
  attrs = dict(array.attrs)
  units = attrs.pop('units', None)
  if units is not None:
    data = scales.parse_units(units) * array.data
  else:
    data = scales.units.dimensionless * array.data
  return xarray.DataArray(data, array.coords, array.dims, attrs=attrs)


def attach_xarray_units(ds: xarray.Dataset) -> xarray.Dataset:
  return ds.map(attach_data_array_units)


def xarray_nondimensionalize(
    ds: xarray.Dataset,
    physics_specs: Any,
) -> xarray.Dataset:
  return xarray.apply_ufunc(physics_specs.nondimensionalize, ds)


def verify_grid_consistency(
    longitude: np.ndarray | xarray.DataArray,
    latitude: np.ndarray | xarray.DataArray,
    grid: spherical_harmonic.Grid,
):
  """Verifies that longitude and latitude axes are compatible with `grid`."""
  np.testing.assert_allclose(
      180 / np.pi * grid.longitudes, longitude, atol=1e-3
  )
  np.testing.assert_allclose(
      180 / np.pi * grid.latitudes, latitude, atol=1e-3
  )


def xarray_selective_shift(
    dataset: xarray.Dataset,
    variables: Sequence[str] = tuple(),
    time_shift: str|np.timedelta64|pd.Timedelta = '0 hour',
    time_name: str = 'time',
) -> xarray.Dataset:
  """Shifts specified variables in time and truncates associated time values.

  As with xarray.shift(), positive values of the shift move values "to the
  right", negative values "to the left" relative to the original dataset time
  coordinates. This implies that specifying a positive `time_shift` will produce
  a dataset where for each time the values of `variables` are from an earlier
  time in the original dataset. See unit tests for examples.

  Args:
    dataset: Input dataset.
    variables: Variables to which shift is applied.
    time_shift: Timedelta by which to shift `variables.`
    time_name: Name of the time coordinate.

  Returns:
    Dataset where every DataArray in `variables` have been shifted on the
    `time_name` coordinate by `time_shift`.  The head or tail times associated
    with the shifted indices are truncated.
  """
  time_shift = pd.Timedelta(time_shift)
  time_spacing = dataset[time_name][1] - dataset[time_name][0]

  shift, remainder = divmod(time_shift, time_spacing)
  shift = int(shift)  # convert from xarray value
  if shift == 0 or not variables:
    return dataset
  if remainder:
    raise ValueError(f'Does not divide evenly, got {remainder=}')

  ds = dataset.copy()
  if shift > 0:
    ds = ds.isel({time_name: slice(shift, None)})
    for var in variables:
      ds[var] = dataset.variables[var].isel({time_name: slice(None, -shift)})
  else:
    ds = ds.isel({time_name: slice(None, shift)})
    for var in variables:
      ds[var] = dataset.variables[var].isel({time_name: slice(-shift, None)})
  return ds


def datetime64_to_nondim_time(
    time: np.ndarray,
    physics_specs: Any,
    reference_datetime: np.datetime64,
) -> np.ndarray:
  """Converts `time` in datetime64 format to nondimensional sim_time."""
  return physics_specs.nondimensionalize(
      ((time - reference_datetime) / np.timedelta64(1, 'h'))
      * scales.units.hour
  )


def nondim_time_to_datetime64(
    time: np.ndarray,
    physics_specs: Any,
    reference_datetime: np.datetime64,
) -> np.ndarray:
  """Converts `time` in datetime64 format to nondimensional sim_time."""
  minutes = physics_specs.dimensionalize(time, scales.units.minute).magnitude
  return reference_datetime + np.timedelta64(np.round(minutes).astype(int), 'm')


def ds_from_path_or_aux(
    path: str,
    aux_features: typing.AuxFeatures,
) -> xarray.Dataset:
  """Loads dataset from CNS `path` or returns aux_features[XARRAY_DS_KEY]."""
  # If more flexibility is needed, consider adaptors http://shortn/_whtT6Nk74p.
  aux_xarray_ds = aux_features.get(XARRAY_DS_KEY, None)
  if path is not None:
    if aux_xarray_ds is not None:
      raise ValueError(f'Specifying both {path=} and {type(aux_xarray_ds)=} is '
                       'error prone and not supported')
    return open_dataset(path)
  elif aux_xarray_ds is not None:
    return aux_xarray_ds
  else:
    keys = aux_features.keys()
    raise ValueError(f'{path} can be `None` only if {XARRAY_DS_KEY} in {keys=}')


def nondim_time_delta_from_time_axis(
    time: np.ndarray,
    physics_specs: Any,
) -> float:
  """Infers time delta along `time` axis in nondimensional units."""
  time_delta = time[1] - time[0]
  if not np.issubdtype(time.dtype, np.floating):
    time_delta = np.timedelta64(time_delta, 's') / np.timedelta64(1, 's')
    return physics_specs.nondimensionalize(time_delta * scales.units.second)
  return float(time_delta)


def ds_with_sim_time(
    ds: xarray.Dataset,
    physics_specs: Any,
    reference_datetime: np.datetime64,
) -> xarray.Dataset:
  """Returns `ds` with nondimensional time added as `sim_time` if absent."""
  if 'sim_time' in ds:
    return ds
  if np.issubdtype(ds.time.dtype, np.floating):
    nondim_time = ds.time.data
  else:
    nondim_time = datetime64_to_nondim_time(
        ds.time.data, physics_specs, reference_datetime)
  # if dataset contains `sample` axis, sim_time should have it as well.
  if XR_SAMPLE_NAME in ds.coords:
    nondim_time = nondim_time[np.newaxis, ...]
    nondim_time = np.repeat(nondim_time, ds.sizes[XR_SAMPLE_NAME], 0)
    ds['sim_time'] = ((XR_SAMPLE_NAME, XR_TIME_NAME,), nondim_time)
  else:
    ds['sim_time'] = ((XR_TIME_NAME,), nondim_time)
  return ds


def infer_longitude_offset(lon: np.ndarray | xarray.DataArray) -> float:
  """Infers the longitude offset in radians given the longitude values in degrees."""
  if isinstance(lon, xarray.DataArray):
    lon = lon.data
  if lon.max() < 2 * np.pi:
    raise ValueError(f'Expected longitude values in degrees, got {lon=}')
  return lon[0] * np.pi / 180


def infer_latitude_spacing(
    lat: np.ndarray | xarray.DataArray
) -> str:
  """Infers the type of latitude spacing given the latitude values."""
  if np.allclose(np.diff(lat), lat[1] - lat[0]):
    if np.isclose(max(lat), 90.):
      spacing = 'equiangular_with_poles'
    else:
      spacing = 'equiangular'
  else:
    spacing = 'gauss'
  return spacing


def coordinate_system_from_dataset_shape(
    ds: xarray.Dataset,
    truncation: str = CUBIC,
) -> coordinate_systems.CoordinateSystem:
  """Creates a `CoordinateSystem` object based on `dataset`.

  Args:
    ds: dataset with data axes that specify a compatible coordinate system.
    truncation: enum indicating the type of spectral grid to construct.

  Returns:
    coordinate system infered from the shape of the data axes of the dataset.
  """
  if truncation == CUBIC:
    shape_to_grid_dict = CUBIC_SHAPE_TO_GRID_DICT
  elif truncation == LINEAR:
    shape_to_grid_dict = LINEAR_SHAPE_TO_GRID_DICT
  else:
    raise ValueError(f'{truncation=} is not supported.')
  if XR_LON_NAME in ds and XR_LAT_NAME in ds:
    lon, lat = ds[XR_LON_NAME], ds[XR_LAT_NAME]
  elif 'longitude' in ds and 'latitude' in ds:
    lon, lat = ds.longitude, ds.latitude
  else:
    raise ValueError('Dataset must provide lon/lat or longitude/latitude axes.')
  grid_cls = shape_to_grid_dict[lon.shape + lat.shape]
  horizontal = grid_cls(latitude_spacing=infer_latitude_spacing(lat))
  verify_grid_consistency(lon, lat, horizontal)
  if XR_LEVEL_NAME in ds:
    # we assume the default pressure coordinates for vertical discretization.
    vertical_centers = ds.level.values
    vertical = vertical_interpolation.PressureCoordinates(vertical_centers)
  else:
    vertical = None  # no vertical discretization provided.
  return coordinate_systems.CoordinateSystem(horizontal, vertical)


def coordinate_system_from_dataset(
    ds: xarray.Dataset,
    truncation: str = CUBIC,
    spherical_harmonics_impl: Callable[
        ..., spherical_harmonic.SphericalHarmonics
    ]
    | None = None,
    spmd_mesh: jax.sharding.Mesh | None = None,
) -> coordinate_systems.CoordinateSystem:
  """Creates a `CoordinateSystem` object based on `dataset`.

  Tries to extract coordinate system metadata from attrs first, falling back to
  a shape-based method.

  Args:
    ds: dataset with data axes that specify a compatible coordinate system.
    truncation: enum indicating the type of spectral grid to construct.
    spherical_harmonics_impl: non-default implementation of spherical harmonics.
    spmd_mesh: optional SPMD mesh to set.

  Returns:
    coordinate system infered from the shape of the data axes of the dataset.
  """
  try:
    coords = coordinate_system_from_attrs(ds.attrs)
  except KeyError:
    coords = coordinate_system_from_dataset_shape(ds, truncation=truncation)
  if spherical_harmonics_impl is not None:
    coords = dataclasses.replace(
        coords,
        horizontal=dataclasses.replace(
            coords.horizontal, spherical_harmonics_impl=spherical_harmonics_impl
        ),
    )
  coords = dataclasses.replace(coords, spmd_mesh=spmd_mesh)
  return coords


def temperature_variation_to_absolute(
    temperature_variation: np.ndarray,
    ref_temperature: np.ndarray,
) -> np.ndarray:
  """Computes absolute temperature from nodal `temperature variation`."""
  ndim = temperature_variation.ndim
  if ndim == 3 or ndim == 4:
    return temperature_variation + ref_temperature[:, np.newaxis, np.newaxis]
  else:
    raise ValueError(f'{temperature_variation.ndim=}, while expecting 3|4.')


def fill_nan_with_nearest(dataset: xarray.Dataset) -> xarray.Dataset:
  """Replaces nan values in `dataset` with nearest horizontal value."""

  def fill_nan_for_array(array: xarray.DataArray) -> xarray.DataArray:
    if array.chunks:
      raise ValueError(
          f'Expected data to be loaded in memory, got chunks = {array.chunks}'
      )

    extra_dims = list(set(array.dims) - {'latitude', 'longitude'})
    isnan_mask = array.isnull().any(extra_dims)
    anynan_mask = array.isnull().all(extra_dims)
    if not isnan_mask.equals(anynan_mask):
      raise ValueError('NaN mask is not fixed')

    lat, lon = xarray.broadcast(array.latitude, array.longitude)
    # Shape lat, lon to match order of dims in data var
    lat = lat.transpose(*isnan_mask.dims)
    lon = lon.transpose(*isnan_mask.dims)

    # index_coords store have non-nan values, query_coords have nan values
    index_coords = np.deg2rad(
        np.stack(
            [lat.data[~isnan_mask.data], lon.data[~isnan_mask.data]], axis=-1
        )
    )
    query_coords = np.deg2rad(
        np.stack(
            [lat.data[isnan_mask.data], lon.data[isnan_mask.data]], axis=-1
        )
    )

    # construct a BallTree to find nearest neighbor on the surface of a sphere
    tree = neighbors.BallTree(index_coords, metric='haversine')
    indices = tree.query(query_coords, return_distance=False).squeeze(axis=-1)

    # Replace nan values (target) with nearest non-nan value (source)
    source_lats = xarray.DataArray(
        lat.data[~isnan_mask.data][indices], dims=['query']
    )
    source_lons = xarray.DataArray(
        lon.data[~isnan_mask.data][indices], dims=['query']
    )
    target_lats = xarray.DataArray(lat.data[isnan_mask.data], dims=['query'])
    target_lons = xarray.DataArray(lon.data[isnan_mask.data], dims=['query'])

    array = array.copy(deep=True)
    array.loc[{'latitude': target_lats, 'longitude': target_lons}] = array.loc[
        {'latitude': source_lats, 'longitude': source_lons}
    ]
    return array

  filled_arrays = {}
  for k, v in dataset.data_vars.items():
    if 'time' not in v.dims:
      continue
    if v.isel(time=0).isnull().all():
      raise ValueError(f'DataArray {k} has all nan values for isel(time=0)')
    if v.isnull().any():
      array = v.copy(deep=True)
      filled_arrays[k] = fill_nan_for_array(array.compute())
  return dataset.assign(filled_arrays)
