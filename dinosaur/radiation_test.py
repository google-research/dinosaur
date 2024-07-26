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

"""Tests for radiation."""

import datetime

from absl.testing import absltest
from absl.testing import parameterized

from dinosaur import coordinate_systems
from dinosaur import primitive_equations
from dinosaur import radiation
from dinosaur import scales
from dinosaur import sigma_coordinates
from dinosaur import spherical_harmonic

import jax
from jax import config
import jax.numpy as jnp
import numpy as np

units = scales.units
config.parse_flags_with_absl()

TWOPI = 2 * jnp.pi


class RadiationTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(when=np.datetime64('1980-01-01T00:00:00.000000000'),
           expected=datetime.datetime(1980, 1, 1, 0, 0)),
      dict(when=np.datetime64('2001-02-03T04:05'),
           expected=datetime.datetime(2001, 2, 3, 4, 5)),
      dict(when=np.datetime64('1982-05-04'),
           expected=datetime.datetime(1982, 5, 4, 0, 0)),
  )
  def test_datetime64_to_datetime(self, when, expected):
    actual = radiation.datetime64_to_datetime(when)
    self.assertEqual(actual, expected)

  @parameterized.parameters(
      dict(when=datetime.datetime(1984, 1, 1), expected_days=366),
      dict(when=datetime.datetime(1985, 1, 1), expected_days=365),
      dict(when=datetime.datetime(2020, 1, 1), expected_days=366),
      dict(when=datetime.datetime(2001, 1, 1), expected_days=365),
  )
  def test_days_in_year(self, when, expected_days):
    actual_days = radiation.days_in_year(when)
    self.assertEqual(actual_days, expected_days)

  @parameterized.parameters(
      # 1980 is a leap year
      dict(when=datetime.datetime(1980, 1, 1, 0, 0),
           exp_orbital_phase=0,
           exp_synodic_phase=0),
      dict(when=datetime.datetime(1980, 1, 1, 12, 0),
           exp_orbital_phase=0.5 * TWOPI / 366,
           exp_synodic_phase=jnp.pi),
      dict(when=datetime.datetime(1980, 1, 1, 23, 59),
           exp_orbital_phase=(1439 / 1440) * TWOPI / 366,
           exp_synodic_phase=1439 * TWOPI / 1440),
      dict(when=datetime.datetime(1980, 1, 1, 23, 59),
           exp_orbital_phase=(1439 / 1440) * TWOPI / 366,
           exp_synodic_phase=1439 * TWOPI / 1440),
      dict(when=datetime.datetime(1980, 5, 1, 0, 0),
           exp_orbital_phase=(31 + 29 + 31 + 30) * TWOPI / 366,
           exp_synodic_phase=0),
      dict(when=datetime.datetime(1980, 12, 31, 23, 59),
           exp_orbital_phase=TWOPI - (1 / 1440) * TWOPI / 366,
           exp_synodic_phase=1439 * TWOPI / 1440),
      # 1981 is not a leap year
      dict(when=datetime.datetime(1981, 1, 1, 0, 0),
           exp_orbital_phase=0,
           exp_synodic_phase=0),
      dict(when=datetime.datetime(1981, 1, 1, 12, 0),
           exp_orbital_phase=0.5 * TWOPI / 365,
           exp_synodic_phase=jnp.pi),
      dict(when=datetime.datetime(1981, 1, 1, 23, 59),
           exp_orbital_phase=(1439 / 1440) * TWOPI / 365,
           exp_synodic_phase=1439 * TWOPI / 1440),
      dict(when=datetime.datetime(1981, 1, 1, 23, 59),
           exp_orbital_phase=(1439 / 1440) * TWOPI / 365,
           exp_synodic_phase=1439 * TWOPI / 1440),
      dict(when=datetime.datetime(1981, 5, 1, 0, 0),
           exp_orbital_phase=(31 + 28 + 31 + 30) * TWOPI / 365,
           exp_synodic_phase=0),
      dict(when=datetime.datetime(1981, 12, 31, 23, 59),
           exp_orbital_phase=TWOPI - (1 / 1440) * TWOPI / 365,
           exp_synodic_phase=1439 * TWOPI / 1440),
      # The distant future
      dict(when=datetime.datetime(2022, 5, 1, 0, 0),
           exp_orbital_phase=(31 + 28 + 31 + 30) * TWOPI / 365,
           exp_synodic_phase=0),
      dict(when=datetime.datetime(2045, 12, 31, 23, 59),
           exp_orbital_phase=TWOPI - (1 / 1440) * TWOPI / 365,
           exp_synodic_phase=1439 * TWOPI / 1440),
  )
  def test_datetime_to_orbital_time(
      self, when, exp_orbital_phase, exp_synodic_phase):
    actual = radiation.datetime_to_orbital_time(when)
    self.assertAlmostEqual(actual.orbital_phase, exp_orbital_phase)
    self.assertAlmostEqual(actual.synodic_phase, exp_synodic_phase)

  def test_get_direct_solar_irradiance_no_units(self):
    flux = radiation.get_direct_solar_irradiance(
        orbital_phase=jnp.linspace(0, TWOPI, 4, endpoint=False),
        mean_irradiance=1.2,
        variation=0.3,
        perihelion=0)
    np.testing.assert_allclose(flux, [1.5, 1.2, 0.9, 1.2])

  def test_direct_solar_irradiance_pint_units(self):
    flux = radiation.get_direct_solar_irradiance(
        orbital_phase=jnp.linspace(0, TWOPI, 4, endpoint=False),
        mean_irradiance=1.2 * units.W / units.meter**2,
        variation=0.3 * units.W / units.meter**2,
        perihelion=0)
    np.testing.assert_allclose(
        flux, jnp.array([1.5, 1.2, 0.9, 1.2]) * units.W / units.meter**2)

  # Days of the year when equation of time is nearly zero
  @parameterized.parameters(
      (datetime.datetime(2000, 4, 16),),
      (datetime.datetime(2000, 6, 14),),
      (datetime.datetime(2000, 8, 31),),
      (datetime.datetime(2000, 12, 25),),
  )
  def test_equation_of_time(self, when):
    ot = radiation.datetime_to_orbital_time(when)
    delta_phase = radiation.equation_of_time(ot.orbital_phase)
    np.testing.assert_allclose(delta_phase, 0, atol=0.005)


def _get_expected_value_modulo_2pi(expected, actual):
  """Returns expected values for an approximate comparison modulo 2pi."""
    # Use np.fmod to avoid loss of precision with jnp.fmod.
  expected = np.fmod(expected, TWOPI)
  expected = jnp.where(expected < actual - jnp.pi, expected + TWOPI, expected)
  expected = jnp.where(expected > actual + jnp.pi, expected - TWOPI, expected)
  return expected


class SolarRadiationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.coords = coordinate_systems.CoordinateSystem(
        horizontal=spherical_harmonic.Grid.T85(),
        vertical=sigma_coordinates.SigmaCoordinates.equidistant(layers=8))
    self.physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
    self.reference_datetime = radiation.WB_REFERENCE_DATETIME

  def test_radiation_shapes(self):
    solar_radiation = radiation.SolarRadiation(
        self.coords, self.physics_specs, self.reference_datetime)
    with self.subTest('radiation_flux'):
      radiation_flux_fn = jax.jit(solar_radiation.radiation_flux)
      tisr = radiation_flux_fn(time=0)
      self.assertEqual(tisr.shape, self.coords.horizontal.nodal_shape)
    with self.subTest('solar_hour_angle'):
      solar_hour_angle_fn = jax.jit(solar_radiation.solar_hour_angle)
      solor_hour_angle = solar_hour_angle_fn(time=0)
      self.assertEqual(solor_hour_angle.shape,
                       self.coords.horizontal.nodal_shape)

  def test_radiation_flux_nondimensionalization(self):
    solar_radiation = radiation.SolarRadiation(
        self.coords, self.physics_specs, self.reference_datetime)
    perihelion_datetime = datetime.datetime(1979, 1, 3)
    time = solar_radiation.datetime_to_time(perihelion_datetime)
    radiation_flux_fn = jax.jit(solar_radiation.radiation_flux)
    actual_max = np.max(radiation_flux_fn(time))
    # At perihelion, irradiance is at its highest
    expected_max = self.physics_specs.nondimensionalize(
        radiation.TOTAL_SOLAR_IRRADIANCE + radiation.SOLAR_IRRADIANCE_VARIATION)
    np.testing.assert_allclose(actual_max, expected_max, rtol=5e-5)

  @parameterized.parameters(
      dict(reference_datetime=radiation.WB_REFERENCE_DATETIME,
           days_until_aphelion=3 + radiation.DAYS_PER_YEAR / 2),
      dict(reference_datetime=radiation.WB_REFERENCE_DATETIME,
           days_until_aphelion=3 - radiation.DAYS_PER_YEAR / 2),
      dict(reference_datetime=datetime.datetime(1979, 7, 4),
           days_until_aphelion=0),
      dict(reference_datetime=datetime.datetime(1980, 7, 3),
           days_until_aphelion=0),
      dict(reference_datetime=datetime.datetime(2022, 5, 4),
           days_until_aphelion=61),
  )
  def test_reference_datetime(self, reference_datetime, days_until_aphelion):
    # Check that combinations of reference_datetime and time work correctly
    solar_radiation = radiation.SolarRadiation(
        self.coords, self.physics_specs, reference_datetime)
    time = self.physics_specs.nondimensionalize(days_until_aphelion * units.day)
    radiation_flux_fn = jax.jit(solar_radiation.radiation_flux)
    actual_max = np.max(radiation_flux_fn(time))
    # At aphelion, irradiance is at its lowest
    expected_max = self.physics_specs.nondimensionalize(
        radiation.TOTAL_SOLAR_IRRADIANCE - radiation.SOLAR_IRRADIANCE_VARIATION)
    np.testing.assert_allclose(actual_max, expected_max, rtol=5e-5)

  def test_normalized_radiation_flux(self):
    solar_radiation = radiation.SolarRadiation.normalized(
        self.coords, self.physics_specs, self.reference_datetime)
    perihelion_datetime = datetime.datetime(1979, 1, 3)
    time = solar_radiation.datetime_to_time(perihelion_datetime)
    radiation_flux_fn = jax.jit(solar_radiation.radiation_flux)
    tisr = radiation_flux_fn(time=time)
    np.testing.assert_allclose(np.max(tisr), 1.0, rtol=5e-5)

  @parameterized.parameters(
      dict(when=datetime.datetime(1979, 1, 1, 0, 0),
           expected_time_days=0.),
      dict(when=datetime.datetime(1979, 1, 1, 0, 1),
           expected_time_days=1 / (60 * 24)),
      dict(when=datetime.datetime(1979, 1, 1, 12, 0),
           expected_time_days=0.5),
      dict(when=datetime.datetime(1979, 1, 3, 0, 0),
           expected_time_days=2.),
      dict(when=datetime.datetime(1980, 1, 1, 0, 0),
           expected_time_days=365.),
      dict(when=datetime.datetime(1981, 1, 1, 0, 0),
           expected_time_days=731.),  # 365 + 366, 1980 is a leap year
      dict(when=datetime.datetime(2020, 1, 1, 0, 0),
           expected_time_days=14975.),  # 31 * 365 + 10 * 366
      dict(when=datetime.datetime(2022, 5, 4, 4, 20),
           expected_time_days=15829.180555555555),
  )
  def test_datetime_to_time(self, when, expected_time_days):
    solar_radiation = radiation.SolarRadiation(
        self.coords, self.physics_specs, self.reference_datetime)
    actual_time = solar_radiation.datetime_to_time(when)
    expected_time = solar_radiation.physics_specs.nondimensionalize(
        expected_time_days * units.day)
    self.assertAlmostEqual(actual_time, expected_time)

  @parameterized.parameters(
      # Note, WB_REFERENCE_DATETIME = (1979, 1, 1, 0, 0)
      # Nondim time based on radiation.DAYS_PER_YEAR = 365.25
      dict(when=radiation.WB_REFERENCE_DATETIME,
           expected_orbital_phase=0,
           expected_synodic_phase=0),
      dict(when=(radiation.WB_REFERENCE_DATETIME
                 + datetime.timedelta(days=365.25)),
           expected_orbital_phase=TWOPI,
           expected_synodic_phase=365.25 * TWOPI),
      dict(when=(radiation.WB_REFERENCE_DATETIME
                 + datetime.timedelta(days=-365.25)),
           expected_orbital_phase=-TWOPI,
           expected_synodic_phase=-365.25 * TWOPI),
      dict(when=datetime.datetime(2019, 1, 1, 0, 0),
           expected_orbital_phase=40 * TWOPI,
           expected_synodic_phase=40 * 365.25 * TWOPI),
  )
  def test_time_to_orbital_time(
      self, when, expected_orbital_phase, expected_synodic_phase):
    solar_radiation = radiation.SolarRadiation(
        self.coords, self.physics_specs, self.reference_datetime)
    time = solar_radiation.datetime_to_time(when)
    actual = solar_radiation.time_to_orbital_time(time)

    expected_orbital_phase = _get_expected_value_modulo_2pi(
        expected_orbital_phase, actual.orbital_phase
    )
    self.assertAlmostEqual(actual.orbital_phase, expected_orbital_phase)

    expected_synodic_phase = _get_expected_value_modulo_2pi(
        expected_synodic_phase, actual.synodic_phase
    )
    self.assertAlmostEqual(actual.synodic_phase, expected_synodic_phase)

  def test_solar_hour_angle(self):
    solar_radiation = radiation.SolarRadiation(
        self.coords, self.physics_specs, self.reference_datetime)
    time = solar_radiation.datetime_to_time(
        datetime.datetime(1979, 4, 15, 0, 0))  # Equation of time is approx 0
    actual = solar_radiation.solar_hour_angle(time)
    lon, _ = self.coords.horizontal.nodal_mesh
    # At UTC=0, when equation of time is 0, expect hour_angle=`longitude - pi`
    expected = lon - jnp.pi
    expected = _get_expected_value_modulo_2pi(expected, actual)
    np.testing.assert_allclose(actual, expected, atol=1e-4)

  def test_nondim_minutes_day_year_constants(self):
    physics_specs = primitive_equations.PrimitiveEquationsSpecs.from_si()
    days_per_year = physics_specs.nondimensionalize(units.year / units.day)
    np.testing.assert_almost_equal(days_per_year, radiation.DAYS_PER_YEAR)
    minutes_per_day = physics_specs.nondimensionalize(units.day / units.minute)
    np.testing.assert_almost_equal(minutes_per_day, radiation.MINUTES_PER_DAY)


if __name__ == '__main__':
  absltest.main()
