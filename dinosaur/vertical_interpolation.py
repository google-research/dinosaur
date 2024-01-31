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

"""Routines for regridding between sigma and pressure levels."""
import dataclasses
import functools
from typing import Any, Callable, Dict, Sequence, Union

from dinosaur import pytree_utils
from dinosaur import sigma_coordinates
from dinosaur import typing

import jax
import jax.numpy as jnp
import numpy as np


Array = typing.Array
InterpolateFn = Callable[[Array, Array, Array], Array]


def vectorize_vertical_interpolation(
    interpolate_fn: InterpolateFn,
) -> InterpolateFn:
  """Vectorizes vertical `interpolate_fn` function to work on 3+d fields."""
  # inerpolate_fn operates on `x, xp, fp` (target, source loc, source vals).
  # vmap over spatial axes for targets and source values.
  interpolate_fn = jax.vmap(interpolate_fn, (-1, None, -1), out_axes=-1)
  interpolate_fn = jax.vmap(interpolate_fn, (-1, None, -1), out_axes=-1)
  # vmap over multiple vertical targets.
  interpolate_fn = jax.vmap(interpolate_fn, (0, None, None), out_axes=0)
  # vectorize an leading dimensions.
  return jnp.vectorize(interpolate_fn, signature='(a,x,y),(b),(b,x,y)->(a,x,y)')


def _dot_interp(x, xp, fp):
  # pylint: disable=g-doc-return-or-yield,g-doc-args
  """Interpolate with a dot product instead of indexing.

  This is often much faster than the default jnp.interp on TPUs.
  """
  # TODO(shoyer): upstream this into jnp.interp
  n = len(xp)
  i = jnp.arange(n)
  dx = xp[1:] - xp[:-1]
  delta = x - xp[:-1]
  w = delta / dx
  w_left = jnp.pad(1 - w, [(0, 1)])
  w_right = jnp.pad(w, [(1, 0)])
  u = jnp.searchsorted(xp, x, side='right', method='compare_all')
  u = jnp.clip(u, 1, n - 1)
  weights = w_left * (i == (u - 1)) + w_right * (i == u)
  weights = jnp.where(x < xp[0], i == 0, weights)
  weights = jnp.where(x > xp[-1], i == (n - 1), weights)
  return jnp.dot(weights, fp, precision='highest')


@jax.jit
def interp(
    x: typing.Numeric, xp: typing.Array, fp: typing.Array
) -> jnp.ndarray:
  """Optimized version of jnp.interp."""
  on_tpu = all(device.platform == 'tpu' for device in jax.local_devices())
  if on_tpu:
    return _dot_interp(x, xp, fp)
  else:
    return jnp.interp(x, xp, fp)


def _extrapolate_left(y):
  delta = y[1] - y[0]
  return jnp.concatenate([jnp.array([y[0] - delta]), y])


def _extrapolate_right(y):
  delta = y[-1] - y[-2]
  return jnp.concatenate([y, jnp.array([y[-1] + delta])])


def _extrapolate_both(y):
  return _extrapolate_left(_extrapolate_right(y))


def _linear_interp_with_safe_extrap(x, xp, fp, n=1):
  """Linear interpolation with extrapolation for n grid cells at each end."""
  for _ in range(n):
    xp = _extrapolate_both(xp)
    fp = _extrapolate_both(fp)
  return jnp.interp(x, xp, fp, left=np.nan, right=np.nan)


@jax.jit
def linear_interp_with_linear_extrap(
    x: typing.Numeric, xp: typing.Array, fp: typing.Array
) -> jnp.ndarray:
  """Linear interpolation with unlimited linear extrapolation at each end."""
  n = len(xp)
  i = jnp.arange(n)
  dx = xp[1:] - xp[:-1]
  delta = x - xp[:-1]
  w = delta / dx
  w_left = jnp.pad(1 - w, [(0, 1)])
  w_right = jnp.pad(w, [(1, 0)])
  u = jnp.searchsorted(xp, x, side='right', method='compare_all')
  u = jnp.clip(u, 1, n - 1)
  weights = w_left * (i == (u - 1)) + w_right * (i == u)
  return jnp.dot(weights, fp, precision='highest')


# TODO(shoyer): add higher order interpolation schemes, e.g., with splines.
# These aren't supported in JAX yet, but can be found in JAX-Cosmo:
# https://github.com/DifferentiableUniverseInitiative/jax_cosmo/issues/29
# https://github.com/DifferentiableUniverseInitiative/jax_cosmo/blob/master/jax_cosmo/scipy/interpolate.py


@dataclasses.dataclass(frozen=True)
class PressureCoordinates:
  """Specifies the vertical coordinate with pressure levels.

  Attributes:
    centers: center of each pressure level, starting at the level closest to the
      top of the atmosphere. Must be monotonic increasing.
    layers: number of vertical layers.
  """

  centers: np.ndarray

  def __init__(self, centers: Union[Sequence[float], np.ndarray]):
    object.__setattr__(self, 'centers', np.asarray(centers))
    if not all(np.diff(self.centers) > 0):
      raise ValueError(
          'Expected `centers` to be monotonically increasing, '
          f'got centers = {self.centers}'
      )

  @property
  def layers(self) -> int:
    return len(self.centers)

  def asdict(self) -> Dict[str, Any]:
    return {k: v.tolist() for k, v in dataclasses.asdict(self).items()}

  def __hash__(self):
    return hash(tuple(self.centers.tolist()))

  def __eq__(self, other):
    return isinstance(other, PressureCoordinates) and np.array_equal(
        self.centers, other.centers
    )


@dataclasses.dataclass(frozen=True)
class HybridCoordinates:
  """Specifies the vertical coordinate with hybrid sigma-pressure levels.

  This allows for matching the vertical coordinate system used by ECMWF and most
  other operational forecasting systems.

  The pressure corresponding to a level is given by the formula `a + b * sp`
  where `sp` is surface pressure.

  Attributes:
    a_centers: offset coefficient for the center of each level, starting at the
      level closest to the top of the atmosphere.
    b_centers: slope coefficient for the center of each level, starting at the
      level closest to the top of the atmosphere.
    layers: number of vertical layers.
  """

  a_centers: np.ndarray
  b_centers: np.ndarray

  @property
  def layers(self) -> int:
    return len(self.a_centers)

  def __hash__(self):
    return hash(
        (tuple(self.a_centers.tolist()), tuple(self.b_centers.tolist()))
    )

  def __eq__(self, other):
    return (
        isinstance(other, HybridCoordinates)
        and np.array_equal(self.a_centers, other.a_centers)
        and np.array_equal(self.b_centers, other.b_centers)
    )


@functools.partial(jax.jit, static_argnums=0)
def get_surface_pressure(
    pressure_levels: PressureCoordinates,
    geopotential: typing.Array,
    orography: typing.Array,
    gravity_acceleration: float,
) -> typing.Array:
  """Calculate surface pressure from geopotential on pressure levels.

  Args:
    pressure_levels: 1D array with pressure levels.
    geopotential: array with dimensions [..., level, x, y]
    orography: array with dimensions [1, x, y]
    gravity_acceleration: acceleration due to gravity.

  Returns:
    Array with dimensions [..., x, y].
  """
  # note: relative height must be an increasing function along the level axis,
  # which is why we subtract geopotential (which decreases as you get closer to
  # the surface of the Earth)
  relative_height = orography * gravity_acceleration - geopotential

  @functools.partial(jnp.vectorize, signature='(z,x,y),(z)->(1,x,y)')
  @functools.partial(jax.vmap, in_axes=(-1, None), out_axes=-1)
  @functools.partial(jax.vmap, in_axes=(-1, None), out_axes=-1)
  def find_intercept(rh, levels):
    return linear_interp_with_linear_extrap(0.0, rh, levels)[np.newaxis]

  return find_intercept(relative_height, pressure_levels.centers)


def vertical_interpolation(
    x: typing.Array,
    xp: typing.Array,
    fp: typing.Array,
) -> typing.Array:
  """Computes linear interpolation `f(x)` with constant extrapolation.

  Args:
    x: array of vertical coordinates for which interpolation is computed.
    xp: array of vertical coordinates on which values are known.
    fp: array of values of on the grid `xp`.

  Returns:
    Interpolated values `f(x)`.
  """
  return interp(x, jnp.asarray(xp), fp)


@functools.partial(jax.jit, static_argnums=(1, 2, 4))
def interp_pressure_to_sigma(
    fields: typing.Pytree,
    pressure_coords: PressureCoordinates,
    sigma_coords: sigma_coordinates.SigmaCoordinates,
    surface_pressure: typing.Array,
    interpolate_fn: InterpolateFn = (
        vectorize_vertical_interpolation(_linear_interp_with_safe_extrap)),
) -> typing.Pytree:
  """Interpolate 3D fields from pressure to sigma levels."""
  desired = sigma_coords.centers[:, np.newaxis, np.newaxis] * surface_pressure
  regrid = lambda x: interpolate_fn(desired, pressure_coords.centers, x)
  return pytree_utils.tree_map_over_nonscalars(regrid, fields)


@functools.partial(jax.jit, static_argnums=(1, 2, 4))
def interp_sigma_to_pressure(
    fields: typing.Pytree,
    pressure_coords: PressureCoordinates,
    sigma_coords: sigma_coordinates.SigmaCoordinates,
    surface_pressure: typing.Array,
    interpolate_fn: InterpolateFn = (
        vectorize_vertical_interpolation(_linear_interp_with_safe_extrap)),
) -> typing.Pytree:
  """Interpolate 3D fields from sigma to pressure levels."""
  desired = (
      pressure_coords.centers[:, np.newaxis, np.newaxis] / surface_pressure
  )
  regrid = lambda x: interpolate_fn(desired, sigma_coords.centers, x)
  return pytree_utils.tree_map_over_nonscalars(regrid, fields)


@functools.partial(jnp.vectorize, signature='(a,x,y),(b,x,y),(b,x,y)->(a,x,y)')
@functools.partial(jax.vmap, in_axes=(0, None, None), out_axes=0)
@functools.partial(jax.vmap, in_axes=(-1, -1, -1), out_axes=-1)
@functools.partial(jax.vmap, in_axes=(-1, -1, -1), out_axes=-1)
def _vertical_interp_3d(x, xp, fp):
  return _linear_interp_with_safe_extrap(x, xp, fp)


@functools.partial(jax.jit, static_argnums=(1, 2))
def interp_hybrid_to_sigma(
    fields: typing.Pytree,
    hybrid_coords: HybridCoordinates,
    sigma_coords: sigma_coordinates.SigmaCoordinates,
    surface_pressure: typing.Array,
) -> typing.Pytree:
  """Interpolate 3D fields from hybrid to sigma levels."""
  desired_pressure = (
      sigma_coords.centers[:, np.newaxis, np.newaxis] * surface_pressure
  )
  source_pressure = (
      hybrid_coords.a_centers[:, np.newaxis, np.newaxis]
      + hybrid_coords.b_centers[:, np.newaxis, np.newaxis] * surface_pressure
  )
  regrid = lambda x: _vertical_interp_3d(desired_pressure, source_pressure, x)
  return pytree_utils.tree_map_over_nonscalars(regrid, fields)
