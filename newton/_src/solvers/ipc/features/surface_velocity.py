# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The surface velocity of a collision mesh (``Mesh(enable_surface_velocity=True)``), a conveyor belt.

Newton's contacts sample the velocity of the mesh's surface; SolverIPC finds its own contacts, which have no surface
velocity, so it refuses such a mesh. The hooks are those of ``features`` (its docstring).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from newton import GeoType, Mesh

from ..scene import _colliding_shapes

if TYPE_CHECKING:
    from newton import Model

    from ..joints import _JointDofs
    from . import FeatureOptions


def check(model: Model, options: FeatureOptions, at: str) -> None:
    """``NotImplementedError`` for the first colliding mesh shape with a surface velocity (``"shapes"``)."""
    if at != "shapes" or model.shape_count == 0:
        return
    meshes = np.flatnonzero(_colliding_shapes(model) & (model.shape_type.numpy() == int(GeoType.MESH)))
    for s in meshes:
        source = model.shape_source[s]
        if isinstance(source, Mesh) and source.enable_surface_velocity:
            raise NotImplementedError(
                f"SolverIPC: shape {int(s)} is a mesh with enable_surface_velocity=True; turn the surface velocity "
                "off. SolverIPC finds its own contacts, which do not apply surface velocities."
            )


def joint_refusal(category: str, dofs: _JointDofs) -> tuple[int, str] | None:
    """Nothing of the joints."""
    return None
