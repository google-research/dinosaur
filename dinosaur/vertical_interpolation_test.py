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

"""Tests for vertical_interpolation."""

from absl.testing import absltest
from absl.testing import parameterized

from dinosaur import sigma_coordinates
from dinosaur import vertical_interpolation

import jax.numpy as jnp
import numpy as np


class InterpTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(x=0.5, expected=10.0),
      dict(x=1.1, expected=11.0),
      dict(x=2.5, expected=22.5),
      dict(x=10.0, expected=30.0),
  )
  def test_interp(self, x, expected):
    xp = jnp.array([1, 2, 4])
    fp = jnp.array([10, 20, 30])
    actual = vertical_interpolation.interp(x, xp, fp)
    self.assertAlmostEqual(actual, expected)

  @parameterized.parameters(
      dict(x=0.5, expected=5.0),
      dict(x=1.1, expected=11.0),
      dict(x=2.5, expected=22.5),
      dict(x=6.0, expected=40.0),
  )
  def test_linear_interp_with_linear_extrap(self, x, expected):
    xp = jnp.array([1, 2, 4])
    fp = jnp.array([10, 20, 30])
    actual = vertical_interpolation.linear_interp_with_linear_extrap(x, xp, fp)
    self.assertAlmostEqual(actual, expected)


class PressureLevelsTest(parameterized.TestCase):

  def test_get_surface_pressure(self):
    levels = np.array([100, 200, 300, 400, 500])
    orography = np.array([[0, 5, 10, 15]])
    geopotential = np.moveaxis([[
        [400, 250, 150, 50, -50],
        [1000, 900, 140, 40, 20],
        [500, 400, 300, 200, 100],
        [600, 500, 400, 300, 200],
    ]], -1, 0)  # reorder from [x, y, level] to [level, x, y]
    expected = np.array([[[450, 390, 500, 550]]])
    actual = vertical_interpolation.get_surface_pressure(
        vertical_interpolation.PressureCoordinates(levels),
        geopotential,
        orography,
        gravity_acceleration=10,
    )
    np.testing.assert_allclose(actual, expected)

  def test_sigma_to_pressure_roundtrip(self):
    sigma_coords = sigma_coordinates.SigmaCoordinates.equidistant(10)
    pressure_coords = vertical_interpolation.PressureCoordinates(
        np.array([100, 200, 300, 400]))
    surface_pressure = np.array([[[250, 350, 450]]])
    original = np.moveaxis([[[1, 2, 3, 4]] * 3], -1, 0)
    on_sigma_levels = vertical_interpolation.interp_pressure_to_sigma(
        original, pressure_coords, sigma_coords, surface_pressure,
    )
    self.assertEqual(on_sigma_levels.shape, (10, 1, 3))
    roundtripped = vertical_interpolation.interp_sigma_to_pressure(
        on_sigma_levels, pressure_coords, sigma_coords, surface_pressure
    )
    # mask points below the surface
    expected = original.astype(float)
    expected[2:, :, 0] = np.nan
    expected[3:, :, 1] = np.nan
    np.testing.assert_allclose(roundtripped, expected, atol=1e-6)

  def test_hybrid_to_sigma(self):
    sigma_coords = sigma_coordinates.SigmaCoordinates.equidistant(5)
    hybrid_coords = vertical_interpolation.HybridCoordinates(
        a_centers=np.array([100, 100, 0]), b_centers=np.array([0, 0.5, 0.9]),
    )
    surface_pressure = np.array([[[1000]]])
    # at pressures [100, 600, 900]
    original = np.array([1.0, 2.0, 3.0])[:, np.newaxis, np.newaxis]
    # at pressures [100, 300, 500, 700, 900]
    expected = np.array([1.0, 1.4, 1.8, 2 + 1/3, 3.0])
    actual = vertical_interpolation.interp_hybrid_to_sigma(
        original, hybrid_coords, sigma_coords, surface_pressure
    ).ravel()
    np.testing.assert_allclose(actual, expected, atol=1e-6)


if __name__ == '__main__':
  absltest.main()
