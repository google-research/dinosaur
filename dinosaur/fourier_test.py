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
import itertools

from absl.testing import absltest
from absl.testing import parameterized

from dinosaur import fourier
import numpy as np


class RealFourierTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(wavenumbers=4, nodes=7),
      dict(wavenumbers=11, nodes=11),
      dict(wavenumbers=32, nodes=63),
  )
  def testBasis(self, wavenumbers, nodes):
    f = fourier.real_basis(wavenumbers, nodes)
    nodes = np.linspace(0, 2 * np.pi, nodes, endpoint=False)
    np.testing.assert_allclose(f[:, 0], 1 / np.sqrt(2 * np.pi))
    for j in range(1, wavenumbers):
      np.testing.assert_allclose(
          f[:, 2 * j - 1], np.cos(j * nodes) / np.sqrt(np.pi), atol=1e-12
      )
      np.testing.assert_allclose(
          f[:, 2 * j], np.sin(j * nodes) / np.sqrt(np.pi), atol=1e-12
      )

  @parameterized.parameters(
      dict(wavenumbers=4, seed=0),
      dict(wavenumbers=11, seed=0),
      dict(wavenumbers=32, seed=0),
  )
  def testDerivatives(self, wavenumbers, seed):
    f = np.random.RandomState(seed).normal(size=[2 * wavenumbers - 1])
    f_x = fourier.real_basis_derivative(f)
    np.testing.assert_allclose(f_x[0], 0)
    for j in range(1, wavenumbers):
      np.testing.assert_allclose(f_x[2 * j - 1], j * f[2 * j], atol=1e-12)
      np.testing.assert_allclose(f_x[2 * j], -j * f[2 * j - 1], atol=1e-12)

  @parameterized.parameters(
      dict(wavenumbers=4),
      dict(wavenumbers=16),
      dict(wavenumbers=256),
  )
  def testNormalized(self, wavenumbers):
    """Tests that the basis functions are normalized on [0, 2π]."""
    nodes = 2 * wavenumbers - 1
    f = fourier.real_basis(wavenumbers, nodes)
    _, w = fourier.quadrature_nodes(nodes)
    eye = np.eye(2 * wavenumbers - 1)
    np.testing.assert_allclose((f.T * w).dot(f), eye, atol=1e-12)


if __name__ == '__main__':
  absltest.main()
