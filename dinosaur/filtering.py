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

"""Filters for GCM models."""
import functools
from typing import Callable

from dinosaur import spherical_harmonic
from dinosaur import typing

import jax
import jax.numpy as jnp
import numpy as np


def _preserves_shape(target, scaling):
  target_shape = np.shape(target)
  return target_shape == np.broadcast_shapes(target_shape, scaling.shape)


def _make_filter_fn(scaling, name=None):
  rescale = lambda x: scaling * x if _preserves_shape(x, scaling) else x
  return functools.partial(
      jax.tree_util.tree_map, jax.named_call(rescale, name=name))


def exponential_filter(
    grid: spherical_harmonic.Grid,
    attenuation: float | typing.Array = 16,
    order: int | typing.Array = 18,
    cutoff: float = 0,
) -> Callable[[typing.PyTreeState], typing.PyTreeState]:
  """Returns a filter that attenuates modes with high total wavenumber.

  Components with `k > cutoff` are damped by a factor of:

    exp(-attenuation * ((k - cutoff) / (1 - cutoff)) ** (2 * order))

  where `k = total_wavenumber / maximum_total_wavenumber`.

  For more on the theory of filtering for resolving Gibbs Phenomenon, see Ref
  [1]. Default parameters here are chosen here based on Ref [2], with the
  attenuation factor adjusted for float32 precision.

  Args:
    grid: the `spherical_harmonic.Grid` to use for the computation.
    attenuation: controls the steepness of the attenuation above the cutoff
      frequency. Typically attentuation is chosen as -log(epsilon), so the max
      frequency components are multiplied by floating point esilon. Using
      `Array` specifications enable applying variable filter parameters for
      different levels/times.
    order: controls the polynomial order of the exponential filter. A higher
      order filter is smoother, and starts attenuating at a higher frequency.
    cutoff: a hard threshold with which to start attenuation, expressed as a
      proportion of maximum total wavenumber, e.g., as used in Reference [3].

  Returns:
    A function that accepts a state and returns a filtered state.

  References:
  [1] Gottlieb, D. & Shu, C.-W. On the Gibbs Phenomenon and Its Resolution. SIAM
      Rev. 39, 644-668 (1997)
  [2] Hou, T. Y. & Li, R. Computing nearly singular solutions using
      pseudo-spectral methods. J. Comput. Phys. 226, 379-397 (2007)
  [3] Arbic, B. K. & Flierl, G. R. Coherent vortices and kinetic energy ribbons
      in asymptotic, quasi two-dimensional f-plane turbulence. Phys. Fluids 15,
      2177-2189 (2003)
  """

  _, total_wavenumber = grid.modal_axes

  k = total_wavenumber / total_wavenumber.max()
  a = attenuation
  c = cutoff
  p = order

  scaling = jnp.exp((k > c) * (-a * (((k - c) / (1 - c)) ** (2 * p))))
  return _make_filter_fn(scaling, "exponential_filter")


def horizontal_diffusion_filter(
    grid: spherical_harmonic.Grid,
    scale: float | typing.Array,
    order: int = 1,
) -> Callable[[typing.PyTreeState], typing.PyTreeState]:
  """Returns a filter that applies a horizontal diffusion step."""
  eigenvalues = grid.laplacian_eigenvalues
  scaling = jnp.exp(-scale * (-eigenvalues) ** order)
  return _make_filter_fn(scaling, "horizontal_diffusion_filter")
