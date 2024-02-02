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

"""Resting and perturbed initial states for primitive equations atmosphere."""

from typing import Callable

from dinosaur import coordinate_systems
from dinosaur import filtering
from dinosaur import primitive_equations
from dinosaur import scales
from dinosaur import typing
from dinosaur import xarray_utils

import jax
import jax.numpy as jnp
import numpy as np

units = scales.units

Array = typing.Array
Quantity = typing.Quantity
QuantityOrStr = Quantity | str


def isothermal_rest_atmosphere(
    coords: coordinate_systems.CoordinateSystem,
    physics_specs: primitive_equations.PrimitiveEquationsSpecs,
    tref: QuantityOrStr = 288. * units.degK,
    p0: QuantityOrStr = 1e5 * units.pascal,
    p1: QuantityOrStr = 0. * units.pascal,
    surface_height: Quantity | None = None,
) -> tuple[Callable[..., primitive_equations.State], typing.AuxFeatures]:
  """Returns a function that generates random states and static features.

  This function implements initial states for Held-Suarez test [1].
  The returned features includes reference temperatures and flat orography.

  The choice of specifying the isothermal atmosphere with uniform T_ref is to
  avoid the Simmons-Hoskins-Burridge instability [2], also see page 601 of [3].

  References:

    [1]: Held, I. M., & Suarez, M. J. (1994). A proposal for the intercomparison
         of the dynamical cores of atmospheric general circulation models.

    [2]: Simmons, A. J., Hoskins, B. J., Burridge, D. M. (1978). Stability of
         the semi-implicit method of time integration.

    [3]: Satoh, M. (2014). Atmospheric Circulation and General Circulation
         Models, 2nd ed.

  Args:
    coords: horizontal and vertical descritization.
    physics_specs: object holding physical constants and definition of custom
      units to use for initialization of the state.
    tref: horizontal reference temperature at all altitudes.
    p0: reference surface pressure.
    p1: magnitude of surface pressure perturbation.
    surface_height: altitude above sea level. Must be an Array with units
      defined on the nodal locations form coords.horizontal.nodal_mesh.
      Expected to have Pint units assigned so that it can be nondimenionalized
      by physics_specs.

  Returns:
    state: a steady atmosphere state for the primitive equations.
    orography: an orography data in modal representation.
    reference_temperatures: a reference temperatures for all vertical levels.
  """
  lon, sin_lat = coords.horizontal.nodal_mesh
  lat = np.arcsin(sin_lat)
  tref = physics_specs.nondimensionalize(units.Quantity(tref))
  p0 = physics_specs.nondimensionalize(units.Quantity(p0))
  p1 = physics_specs.nondimensionalize(units.Quantity(p1))
  if surface_height is None:
    orography = np.zeros_like(lat)  # flat planet
  else:
    orography = np.asarray(physics_specs.nondimensionalize(surface_height))
    if orography.shape != lat.shape:
      raise ValueError(f'Expected surface_height to have shape {lat.shape}, '
                       f'got {surface_height.shape}')

  def _get_vorticity(sigma, lon, lat):
    """Computes vorticity at sigma-level as a function of lon, lat."""
    del sigma, lon  # unused.
    return jnp.zeros_like(lat)

  def _get_surface_pressure(lon, lat, rng_key):
    """Computes surface pressure as a function of lon, lat."""
    # --Get the surface pressure due to orography--
    def relative_pressure(altitude_m):
      # https://en.wikipedia.org/wiki/Atmospheric_pressure
      # pylint: disable=invalid-name
      g = 9.80665  # m/s2
      cp = 1004.68506  # J/(kg·K)
      T0 = 288.16  # K
      M = 0.02896968  # kg/mol
      R0 = 8.314462618  # J/(mol·K)
      return (1 - g * altitude_m / (cp * T0))**(cp * M / R0)
    # pylint: enable=invalid-name

    # Get nodal altitude in meters from modal orography
    # Note that this will likely be different from surface height since the
    # orography is represented with a finite number of spectral basis functions
    altitude_m = physics_specs.dimensionalize(orography, units.meter).magnitude
    surface_pressure = (p0 * np.ones(coords.surface_nodal_shape)
                        * relative_pressure(altitude_m))

    # --Add a surface pressure perturbation--
    keys = jax.random.split(rng_key, 2)
    lon0 = jax.random.uniform(keys[1], minval=np.pi / 2, maxval=3 * np.pi / 2)
    lat0 = jax.random.uniform(keys[0], minval=-np.pi / 4, maxval=np.pi / 4)
    stddev = np.pi / 20  # std deviation in lon, lat
    k = 4  # wavenumber in lon
    perturbation = (jnp.exp(-(lon - lon0)**2 / (2 * stddev**2)) *
                    jnp.exp(-(lat - lat0)**2 / (2 * stddev**2)) *
                    jnp.sin(k * (lon - lon0)))
    return surface_pressure + p1 * perturbation

  def random_state_fn(rng_key: jnp.ndarray) -> primitive_equations.State:
    nodal_vorticity = jnp.stack(
        [_get_vorticity(sigma, lon, lat) for sigma in coords.vertical.centers])
    modal_vorticity = coords.horizontal.to_modal(nodal_vorticity)
    nodal_surface_pressure = _get_surface_pressure(lon, lat, rng_key)
    return primitive_equations.State(
        vorticity=modal_vorticity,
        divergence=jnp.zeros_like(modal_vorticity),
        temperature_variation=jnp.zeros_like(modal_vorticity),
        log_surface_pressure=(
            coords.horizontal.to_modal(jnp.log(nodal_surface_pressure))),
        )

  aux_features = {
      xarray_utils.OROGRAPHY: orography,
      xarray_utils.REF_TEMP_KEY: np.full((coords.vertical.layers,), tref)
  }
  return random_state_fn, aux_features


def isothermal_rest_atmosphere_with_orography_path(
    coords: coordinate_systems.CoordinateSystem,
    physics_specs: primitive_equations.PrimitiveEquationsSpecs,
    path_to_orography_data: str,
    tref: QuantityOrStr = 288. * units.degK,
    p0: QuantityOrStr = 1e5 * units.pascal,
    p1: QuantityOrStr = 0. * units.pascal,
) -> tuple[Callable[..., primitive_equations.State], typing.AuxFeatures]:
  """Wrapper around `isothermal_rest_atmosphere` that loads orography data."""
  ds = xarray_utils.open_dataset(path_to_orography_data)
  input_coords = xarray_utils.coordinate_system_from_dataset(
      ds, spmd_mesh=coords.spmd_mesh)
  filter_fns = [filtering.exponential_filter(coords.horizontal)]
  orography_key = xarray_utils.OROGRAPHY
  nodal_orography = ds[orography_key].transpose('lon', 'lat').values
  orography = primitive_equations.filtered_modal_orography(
      nodal_orography, coords, input_coords, filter_fns)
  nodal_orography_filtered = coords.horizontal.to_nodal(orography)
  nodal_orography_filtered = np.asarray(nodal_orography_filtered)
  return isothermal_rest_atmosphere(
      coords=coords, physics_specs=physics_specs, tref=tref, p0=p0, p1=p1,
      surface_height=(nodal_orography_filtered * units.meters))


def steady_state_jw(
    coords: coordinate_systems.CoordinateSystem,
    physics_specs: primitive_equations.PrimitiveEquationsSpecs,
    u0: Quantity = 35. * units.m / units.s,
    p0: Quantity = 1e5 * units.pascal,
    t0: Quantity = 288. * units.degK,
    delta_t: Quantity = 4.8e5 * units.degK,
    gamma: Quantity = 0.005 * units.degK / units.m,
    sigma_tropo: float = 0.2,
    sigma0: float = 0.252,
) -> tuple[Callable[..., primitive_equations.State], typing.AuxFeatures]:
  """Returns a function that generates steady state and static features.

  This steady state is based on the baroclinic wave test case paper by
  Jablonowski and Williamson [1]. It is provided in a closed form and should
  remain stationary for up-to 30 simulation days. An additional perturbation
  `baroclinic_perturbation_jw` can be introduced on-top of the steady state to
  trigger a baroclinic instability. Note that to achieve stability for 30 days
  we currently need to enable 64bit precision.

  References:

    [1]: Jablonowski, C., & Williamson, D. L. (2006). A baroclinic instability
         test case for atmospheric model dynamical cores.

  Args:
    coords: horizontal and vertical descritization.
    physics_specs: object holding physical constants and definition of custom
      units to use for initialization of the state.
    u0: velocity scale of the zonal jets in the steady state.
    p0: reference pressure of the steady state.
    t0: horizontal mean temperature at the surface.
    delta_t: empirical temperature difference.
    gamma: temperature lapse rate.
    sigma_tropo: tropopause level in sigma coordinates.
    sigma0: a sigma constant used to define sigma_nu in specification of the
      initial state.

  Returns:
    state: a steady atmosphere state for the primitive equations.
    orography: an orography data in modal representation.
    reference_temperatures: a reference temperatures for all vertical levels.
  """
  u0 = physics_specs.nondimensionalize(u0)
  t0 = physics_specs.nondimensionalize(t0)
  delta_t = physics_specs.nondimensionalize(delta_t)
  p0 = physics_specs.nondimensionalize(p0)
  gamma = physics_specs.nondimensionalize(gamma)
  a = physics_specs.radius
  g = physics_specs.g
  r_gas = physics_specs.R
  omega = physics_specs.angular_velocity

  def _get_reference_temperature(sigma):
    """Computes reference temperature for a given sigma level."""
    top_mean_t = t0 * sigma ** (r_gas * gamma / g)
    if sigma < sigma_tropo:
      return top_mean_t + delta_t * (sigma_tropo - sigma) ** 5
    else:
      return top_mean_t

  def _get_reference_geopotential(sigma):
    """Computes reference geopotential for a given sigma level."""
    top_mean_potential = (t0 * g / gamma) * (1 - sigma ** (r_gas * gamma / g))
    if sigma < sigma_tropo:
      return top_mean_potential - r_gas * delta_t * (
          (np.log(sigma / sigma_tropo) + 137 / 60) * sigma_tropo ** 5 -
          5 * sigma * sigma_tropo ** 4 + 5 * (sigma ** 2) * (sigma_tropo ** 3) -
          (10 / 3) * (sigma_tropo ** 2) * sigma ** 3 +
          (5 / 4) * sigma_tropo * sigma ** 4 - (sigma ** 5) / 5
      )
    else:
      return top_mean_potential

  def _get_geopotential(lat, lon, sigma):
    """Computes geopotential at sigma-level as a function of lat."""
    del lon  # unused.
    sigma_nu = (sigma - sigma0) * np.pi / 2
    return _get_reference_geopotential(sigma) + u0 * np.cos(sigma_nu) ** 1.5 * (
        ((-2 * np.sin(lat) ** 6 * (np.cos(lat) ** 2 + 1 / 3) + 10 / 63) *
         u0 * np.cos(sigma_nu) ** (3 / 2)) +
        ((1.6 * (np.cos(lat) ** 3) * (np.sin(lat) ** 2 + 2 / 3) - np.pi / 4) *
         a * omega)
    )

  def _get_temperature_variation(lat, lon, sigma):
    """Computes temperature variation at sigma-level as a function of lat."""
    del lon  # unused.
    sigma_nu = (sigma - sigma0) * np.pi / 2
    cos_𝜎ν = np.cos(sigma_nu)  # pylint: disable=invalid-name
    sin_𝜎ν = np.sin(sigma_nu)  # pylint: disable=invalid-name
    return 0.75 * (sigma * np.pi * u0 / r_gas) * sin_𝜎ν * np.sqrt(cos_𝜎ν) * (
        ((-2 * (np.cos(lat) ** 2 + 1 / 3) * np.sin(lat) ** 6 + 10 / 63) *
         2 * u0 * cos_𝜎ν ** (3 / 2)) +
        ((1.6 * (np.cos(lat) ** 3) * (np.sin(lat) ** 2 + 2 / 3) - np.pi / 4) *
         a * omega)
    )

  def _get_vorticity(lat, lon, sigma):
    """Computes vorticity at sigma-level as a function of lat."""
    del lon  # unused.
    sigma_nu = (sigma - sigma0) * np.pi / 2
    return ((-4 * u0 / a) * (np.cos(sigma_nu) ** (3 / 2)) *
            np.sin(lat) * np.cos(lat) * (2 - 5 * np.sin(lat) ** 2))

  def _get_surface_pressure(lat, lon,):
    """Computes surface pressure as a function of lat."""
    del lon  # unused.
    return p0 * np.ones(lat.shape)[np.newaxis, ...]

  lon, sin_lat = coords.horizontal.nodal_mesh
  lat = np.arcsin(sin_lat)
  def initial_state_fn(
      rng_key: jnp.ndarray | None = None
  ) -> primitive_equations.State:
    del rng_key  # unused.
    nodal_vorticity = np.stack(
        [_get_vorticity(lat, lon, sigma) for sigma in coords.vertical.centers])
    modal_vorticity = coords.horizontal.to_modal(nodal_vorticity)
    nodal_temperature_variation = np.stack(
        [_get_temperature_variation(lat, lon, sigma)
         for sigma in coords.vertical.centers])
    # shift in log_surf_pressure due to custom units doesn't affect dynamics.
    log_nodal_surface_pressure = np.log(_get_surface_pressure(lat, lon))
    state = primitive_equations.State(
        vorticity=modal_vorticity,
        divergence=np.zeros_like(modal_vorticity),
        temperature_variation=coords.horizontal.to_modal(
            nodal_temperature_variation),
        log_surface_pressure=coords.horizontal.to_modal(
            log_nodal_surface_pressure))
    return state

  orography = _get_geopotential(lat, lon, 1.) / g
  geopotential = np.stack(
      [_get_geopotential(lat, lon, sigma) for sigma in coords.vertical.centers])
  reference_temperatures = np.stack(
      [_get_reference_temperature(sigma) for sigma in coords.vertical.centers])
  aux_features = {
      xarray_utils.GEOPOTENTIAL_KEY: geopotential,
      xarray_utils.OROGRAPHY: orography,
      xarray_utils.REF_TEMP_KEY: reference_temperatures,
  }
  return initial_state_fn, aux_features


def baroclinic_perturbation_jw(
    coords: coordinate_systems.CoordinateSystem,
    physics_specs: primitive_equations.PrimitiveEquationsSpecs,
    u_perturb: Quantity = 1. * units.m / units.s,
    lon_location: Quantity = np.pi / 9,
    lat_location: Quantity = 2 * np.pi / 9,
    perturbation_radius: Quantity = 0.1,
) -> primitive_equations.State:
  """Returns perturbation that triggers baroclinic instability.

  The perturbation to the state returned by this function triggers baroclinic
  instability in the rest state produced by `steady_state_jw` and is taken from
  the baroclinic wave test case paper by Jablonowski and Williamson [1].

  References:

    [1]: Jablonowski, C., & Williamson, D. L. (2006). A baroclinic instability
         test case for atmospheric model dynamical cores.

  Args:
    coords: horizontal and vertical descritization.
    physics_specs: object holding physical constants and definition of custom
      units to use for initialization of the state.
    u_perturb: velocity scale of the perturbation.
    lon_location: location of the perturbation along longitude.
    lat_location: location of the perturbation along latitude.
    perturbation_radius: ratio of the spatial scale of the perturbation to the
      radius of the system.

  Returns:
    a perturbation to the atmosphere state that triggers baroclinic instability.
  """
  u_p = physics_specs.nondimensionalize(u_perturb)
  a = physics_specs.radius

  def _get_vorticity_perturbation(lat, lon, sigma):
    del sigma  # unused.
    x = (np.sin(lat_location) * np.sin(lat) +
         np.cos(lat_location) * np.cos(lat) * np.cos(lon - lon_location))
    r = a * np.arccos(x)
    R = a * perturbation_radius  # pylint: disable=invalid-name
    return (u_p / a) * np.exp(-(r / R) ** 2) * (
        np.tan(lat) - (2 * ((a / R) ** 2) * np.arccos(x)) *
        (np.sin(lat_location) * np.cos(lat) -
         np.cos(lat_location) * np.sin(lat) * np.cos(lon - lon_location)) /
        (np.sqrt(1 - x ** 2)))

  def _get_divergence_perturbation(lat, lon, sigma):
    del sigma  # unused.
    x = (np.sin(lat_location) * np.sin(lat) +
         np.cos(lat_location) * np.cos(lat) * np.cos(lon - lon_location))
    r = a * np.arccos(x)
    R = a * perturbation_radius  # pylint: disable=invalid-name
    return (-2 * u_p * a / (R ** 2)) * np.exp(-(r / R) ** 2) * np.arccos(x) * (
        (np.cos(lat_location) * np.sin(lon - lon_location)) /
        (np.sqrt(1 - x ** 2)))

  lon, sin_lat = coords.horizontal.nodal_mesh
  lat = np.arcsin(sin_lat)
  nodal_vorticity = np.stack(
      [_get_vorticity_perturbation(lat, lon, sigma)
       for sigma in coords.vertical.centers])
  nodal_divergence = np.stack(
      [_get_divergence_perturbation(lat, lon, sigma)
       for sigma in coords.vertical.centers])
  modal_vorticity = coords.horizontal.to_modal(nodal_vorticity)
  modal_divergence = coords.horizontal.to_modal(nodal_divergence)
  state = primitive_equations.State(
      vorticity=modal_vorticity,
      divergence=modal_divergence,
      temperature_variation=np.zeros_like(modal_vorticity),
      log_surface_pressure=np.zeros_like(modal_vorticity[:1, ...]),)
  return state


def gaussian_scalar(
    coords: coordinate_systems.CoordinateSystem,
    physics_specs: primitive_equations.PrimitiveEquationsSpecs,
    lon_location: float = np.pi / 9,
    lat_location: float = 2 * np.pi / 9,
    perturbation_radius: float = 0.2,
    amplitude: float = 1.
) -> Array:
  """Returns a scalar gaussian field in modal representation.

  Args:
    coords: horizontal and vertical descritization.
    physics_specs: object holding physical constants and definition of custom
      units to use for initialization of the state.
    lon_location: location of the field along longitude.
    lat_location: location of the field along latitude.
    perturbation_radius: ratio of the spatial scale of the field to the radius.
    amplitude: maximum amplitude of the field.

  Returns:
    Array holding modal values of the field.
  """
  a = physics_specs.radius

  def _get_field_values(lat, lon, sigma):
    del sigma  # unused.
    x = (np.sin(lat_location) * np.sin(lat) +
         np.cos(lat_location) * np.cos(lat) * np.cos(lon - lon_location))
    r = a * np.arccos(x)
    R = a * perturbation_radius  # pylint: disable=invalid-name
    return amplitude * np.exp(-(r / R) ** 2)

  lon, sin_lat = coords.horizontal.nodal_mesh
  lat = np.arcsin(sin_lat)
  return coords.horizontal.to_modal(
      np.stack([_get_field_values(lat, lon, sigma)
                for sigma in coords.vertical.centers]))
