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

"""Tests for associated_legendre."""

from absl.testing import absltest
from absl.testing import parameterized

from dinosaur import associated_legendre

import numpy as np
import scipy.special as sps


class AssociatedLegendreTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(n=4),
      dict(n=11),
      dict(n=32),
  )
  def testOrthonormality(self, n):
    """Tests that the computed associated Legendre functions are orthonormal."""
    z, w = associated_legendre.gauss_legendre_nodes(n)
    p = associated_legendre.evaluate(n, n, z)
    i = np.eye(n, dtype=np.float64)
    inner_products = np.einsum('mil,mik,i->mlk', p, p, w)
    for m in range(n):
      np.testing.assert_allclose(i, inner_products[m], atol=1e-8)
      i[m, m] = 0

  @parameterized.parameters(
      dict(m=4, l=4),
      dict(m=3, l=8),
      dict(m=12, l=20)
  )
  def testAgainstScipy(self, m, l):
    x, _ = associated_legendre.gauss_legendre_nodes(l + 1)
    p_lm = associated_legendre.evaluate(m + 1, l + 1, x)[-1, :, -1]
    q_lm = sps.lpmv(m, l, x)

    # We assert that `q_lm = ratio * p_lm` for some scalar `ratio`.
    ratio = q_lm[0] / p_lm[0]
    for p_j, q_j in zip(p_lm, q_lm):
      np.testing.assert_almost_equal(q_j / ratio, p_j)


if __name__ == '__main__':
  absltest.main()
