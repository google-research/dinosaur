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

"""A vertical coordinate system for a system with layers of fixed density."""

from __future__ import annotations

import dataclasses
import numpy as np


@dataclasses.dataclass(frozen=True)
class LayerCoordinates:
  """Vertical coordinate system for a multi-layer shallow water equations.

  A description of a discrete, constant density vertical layer. Layers are
  indexed from the "top"  of the atmosphere to the surface of the earth.

  Attributes:
    layers: the number of layers.
    centers: indices of layers.
  """
  layers: int

  @property
  def centers(self) -> np.ndarray:
    return np.arange(self.layers)

  def asdict(self):
    return dataclasses.asdict(self)

