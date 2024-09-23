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

"""Tests for shallow_water_states."""

import operator

from absl.testing import absltest
from absl.testing import parameterized

from dinosaur import scales
from dinosaur import shallow_water_states

import jax
import numpy as np


DEFAULT_SCALE = scales.DEFAULT_SCALE


def assert_array_less_equal(x, y, err_msg='', verbose=True):
  return np.testing.assert_array_compare(
      operator.__le__, x, y, err_msg=err_msg,
      verbose=verbose,
      header='x is not less than or equal to y.',
      equal_inf=False)


class BarotropicInstabilityTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(a=2, b=1, expected=1),
      dict(a=1, b=-1, expected=2),
      dict(a=.1, b=2 * np.pi - .1, expected=.2),
      dict(a=2 * np.pi - .1, b=.1, expected=-.2),
      dict(a=np.pi + 1, b=np.pi - 1, expected=2),
      dict(a=np.pi - 1, b=np.pi + 1, expected=-2),
      dict(a=np.pi, b=np.pi, expected=0)
  )
  def testLongitudeSubtraction(self, a, b, expected):
    np.testing.assert_allclose(
        shallow_water_states.subtract_longitudes(a, b), expected, atol=1e-5)

  @parameterized.parameters(
      dict(seed=0),
      dict(seed=1),
      dict(seed=2)
  )
  def testZonalVelocity(self, seed):
    parameters = shallow_water_states.get_random_parameters(
        jax.random.PRNGKey(seed), shallow_water_states.get_default_parameters())
    parameters = jax.tree.map(DEFAULT_SCALE.nondimensionalize, parameters)

    latitude = np.linspace(-np.pi / 2, np.pi / 2, 101)
    zonal_velocity = shallow_water_states.get_zonal_velocity(
        latitude, parameters)
    atol = np.finfo(np.float32).eps

    with self.subTest('IsZeroOutsideJet'):
      outside_jet = ((latitude > parameters.jet_northern_lat)
                     + (latitude < parameters.jet_southern_lat))
      np.testing.assert_allclose(outside_jet * zonal_velocity, 0)

    with self.subTest('IsBoundedByMaxVelocity'):
      assert_array_less_equal(
          np.abs(zonal_velocity), parameters.jet_max_velocity + atol
      )

    with self.subTest('AttainsMaxVelocityAtJetCenter'):
      center = (parameters.jet_northern_lat + parameters.jet_southern_lat) / 2
      center_velocity = shallow_water_states.get_zonal_velocity(
          center, parameters)
      np.testing.assert_allclose(
          center_velocity, parameters.jet_max_velocity, atol=atol
      )

  @parameterized.parameters(
      dict(seed=0),
      dict(seed=1),
      dict(seed=2)
  )
  def testHeight(self, seed):
    parameters = shallow_water_states.get_random_parameters(
        jax.random.PRNGKey(seed), shallow_water_states.get_default_parameters())
    parameters = jax.tree.map(DEFAULT_SCALE.nondimensionalize, parameters)
    longitude = np.linspace(0, 2 * np.pi, 101)
    latitude = np.linspace(-np.pi / 2, np.pi / 2, 101)
    longitude, latitude = np.meshgrid(longitude, latitude, indexing='ij')
    height = shallow_water_states.get_height(longitude, latitude, parameters)

    with self.subTest('BumpInsideJet'):
      self.assertGreaterEqual(parameters.bump_lat_location,
                              parameters.jet_southern_lat)
      self.assertLessEqual(parameters.bump_lat_location,
                           parameters.jet_northern_lat)

    with self.subTest('IsNonNegative'):
      assert_array_less_equal(0, height)

    with self.subTest('IsBoundedByScale'):
      assert_array_less_equal(height, parameters.bump_height_scale)

    with self.subTest('AttainsMaxHeightAtCenter'):
      max_height = (
          np.cos(parameters.bump_lat_location) * parameters.bump_height_scale)
      center_height = shallow_water_states.get_height(
          parameters.bump_lon_location,
          parameters.bump_lat_location,
          parameters)
      np.testing.assert_allclose(center_height, max_height)

if __name__ == '__main__':
  absltest.main()
