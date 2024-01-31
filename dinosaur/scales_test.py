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

"""Tests for scales."""

from absl.testing import absltest
from absl.testing import parameterized

from dinosaur import scales

import jax.numpy as jnp
import numpy as np

units = scales.units


class ScalesTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(get_x=lambda: 24 * units.m / units.s,
           scalez=(10 * units.m,
                   1 * units.hour)),
      dict(get_x=lambda: 9.8 * units.m / units.s ** 2,
           scalez=(123 * units.mile,
                   1 * units.year,
                   32 * units.degK)),
      dict(get_x=lambda: np.arange(20) * units.J,
           scalez=(6 * units.mile,
                   np.pi * units.week,
                   32 * units.kilogram)),
      dict(get_x=lambda: jnp.arange(77) * units.J / units.m ** 2,
           scalez=(11 * units.angstrom,
                   np.pi * units.fortnight,
                   1 * units.UK_ton)),
  )
  def testRoundTrip(self, get_x, scalez):
    """Tests that `dimensionalize(nondimensionalize(...)) is the identity."""
    x = get_x()
    scale = scales.Scale(*scalez)
    y = scale.nondimensionalize(x)
    # `y` should be a numertical value, not a `pint.Quantity`.
    self.assertNotIsInstance(y, scales.Quantity)
    z = scale.dimensionalize(y, x.units)
    np.testing.assert_allclose(x.magnitude, z.magnitude)
    self.assertEqual(x.units, z.units)

  @parameterized.parameters(
      dict(get_x=lambda: 24 * units.m / units.s,
           scalez=(1 * units.hour,)),
      dict(get_x=lambda: 9.8 * units.m / units.s ** 2,
           scalez=(123 * units.mile,
                   32 * units.degK)),
      dict(get_x=lambda: np.arange(20) * units.J,
           scalez=(6 * units.mile,
                   32 * units.kilogram)),
      dict(get_x=lambda: jnp.arange(77) * units.J / units.m ** 2,
           scalez=(1 * units.UK_ton,)),
  )
  def testUnspecifiedScale(self, get_x, scalez):
    """Tests that an exception is raised if a required scale isn't specified."""
    x = get_x()
    scale = scales.Scale(*scalez)
    with self.assertRaisesRegex(ValueError, 'No scale has been set'):
      _ = scale.nondimensionalize(x)

  @parameterized.parameters(
      dict(scalez=(1 * units.J,)),
      dict(scalez=(123 * units.mile,
                   1 / units.year,
                   32 * units.degK)),
      dict(scalez=(6 * units.mile,
                   np.pi * units.week * units.newton,
                   32 * units.kilogram)),
      dict(scalez=(11 * units.KPH,
                   np.pi * units.fortnight,
                   1 * units.UK_ton)),
  )
  def testIllegalCompoundDimension(self, scalez):
    """Tests that an exception is raised for 'compound' dimensions."""
    with self.assertRaisesRegex(
        ValueError, 'All scales must describe a single dimension'):
      _ = scales.Scale(*scalez)

  @parameterized.parameters(
      dict(scalez=(1 * units.m,
                   10 * units.parsec)),
      dict(scalez=(123 * units.mile,
                   1 * units.year,
                   32 * units.year)),
      dict(scalez=(6 * units.mile,
                   np.pi * units.lb,
                   32 * units.kilogram)),
      dict(scalez=(11 * units.hour,
                   np.pi * units.fortnight,
                   1 * units.UK_ton)),
  )
  def testDuplicateScale(self, scalez):
    """Tests that an exception is raised if a dimension is repeated."""
    with self.assertRaisesRegex(
        ValueError, 'Got duplicate scales for dimension'):
      _ = scales.Scale(*scalez)

  @parameterized.parameters(
      dict(scale=scales.Scale(1 * units.m),
           quantity=1 * units.m),
      dict(scale=scales.Scale(17 * units.m),
           quantity=11 * units.mm),
      dict(scale=scales.Scale(33 * units.m, 11 * units.year),
           quantity=np.pi * units.m**2 / units.s **2),
      dict(scale=scales.Scale(123 * units.kg, 345 * units.m, 456 * units.year),
           quantity=5 * units.pascal),
  )
  def testRoundTripNonStandard(self, scale, quantity):
    """Tests that units are converted back and forth correctly."""
    nondimensionalized = scale.nondimensionalize(quantity)
    reconstructed = scale.dimensionalize(nondimensionalized, quantity.units)
    self.assertEqual(quantity.units, reconstructed.units)
    np.testing.assert_allclose(quantity.m, reconstructed.m)

if __name__ == '__main__':
  absltest.main()
