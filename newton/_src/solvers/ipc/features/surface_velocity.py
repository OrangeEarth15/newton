# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The surface velocity of a collision mesh (``Mesh(enable_surface_velocity=True)``): a conveyor belt or a tank tread,
whose material moves along its surface.

Newton gives such a mesh a velocity per vertex, in the mesh's own coordinates: the ``velocities`` of its finalized Warp
mesh, which a caller may write on the device at any time. Newton's contacts turn the velocity at a contact point into
the world by the shape's pose at the step's start, ``body_q * shape_transform``, after scaling it by the shape's
scale, and its solvers take the part along the contact's tangent plane. SolverIPC gives every vertex of the shape's
triangle mesh that velocity, through libuipc-warp's ``SurfaceVelocity`` extension: friction then measures a contact's
slip from where the material was a step ago, ``x_prev - dt v``, and keeps the part along the contact's tangent plane.
A vertex the triangulation welded from several of the mesh's vertices takes the velocity of the one the mesh's first
triangle at its place uses.

The hooks are those of ``features`` (its docstring): ``build`` finds the meshes and refuses AL-IPC, ``engine_inputs``
gives every engine the extension, and ``prepare`` writes the velocities of the step from the meshes' velocities and the
shapes' transforms and scales, which it reads at every step, so a notification needs nothing of this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from newton import GeoType, Mesh

from ..scene import _colliding_shapes, _shape_mesh

if TYPE_CHECKING:
    from newton import Control, Model, State

    from ..solver_ipc import SolverIPC


@wp.kernel(enable_backward=False)
def _write_surface_velocity(
    rows: wp.array[wp.int32],
    shapes: wp.array[wp.int32],
    corners: wp.array[wp.int32],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    velocity: wp.array[wp.vec3d],
):
    """The world velocity [m/s] of the material at engine vertex ``rows[k]``: the velocity of the vertex at corner
    ``corners[k]`` of the triangles of shape ``shapes[k]``'s Warp mesh (``wp.mesh_get_velocity`` takes an index into the
    mesh's index buffer), scaled by the shape's scale and turned by the rotation of the shape's pose,
    ``body_q * shape_transform`` (``shape_transform`` for a static shape), as Newton's ``_mesh_surface_velocity``."""
    k = wp.tid()
    s = shapes[k]
    X = shape_transform[s]
    b = shape_body[s]
    if b >= 0:
        X = wp.transform_multiply(body_q[b], X)
    local = wp.mesh_get_velocity(shape_source_ptr[s], corners[k])
    velocity[rows[k]] = wp.vec3d(wp.transform_vector(X, wp.cw_mul(local, shape_scale[s])))


@dataclass
class _Belts:
    """The meshes with a surface velocity of a solve, and the extension that carries their velocities to its engines."""

    shapes: np.ndarray  # the colliding mesh shapes with the property
    extension: object = None  # libuipc-warp's SurfaceVelocity, made with the first engine
    rows: wp.array = None  # per vertex of their triangulations: the engine row,
    shape: wp.array = None  # the shape,
    corner: wp.array = None  # and the corner of the mesh's triangles whose vertex's velocity it takes


def _belt_shapes(model: Model) -> np.ndarray:
    """The colliding mesh shapes with ``Mesh.enable_surface_velocity``, whose surface velocity Newton's contacts
    sample."""
    if model.shape_count == 0:
        return np.zeros(0, np.int64)
    meshes = np.flatnonzero(_colliding_shapes(model) & (model.shape_type.numpy() == int(GeoType.MESH)))
    flagged = [
        s for s in meshes if isinstance(model.shape_source[s], Mesh) and model.shape_source[s].enable_surface_velocity
    ]
    return np.array(flagged, np.int64)


def _first_corners(model: Model, s: int, welded: np.ndarray) -> np.ndarray:
    """For every vertex of mesh shape ``s``'s triangulation ``welded`` (``scene._shape_mesh``), the first corner of the
    mesh's triangles at its place: an index into the mesh's index buffer, the index ``wp.mesh_get_velocity`` takes.
    ``_shape_mesh`` welds the scaled mesh vertices that round to the same point at a millionth of the mesh's extent
    (``scene._weld``), keeping a copy of the first; this repeats that rounding, and raises ``RuntimeError`` where it
    does not find the triangulation's vertices."""
    mesh = model.shape_source[s]
    scaled = np.asarray(mesh.vertices, np.float64).reshape(-1, 3)
    scaled = scaled * model.shape_scale.numpy()[s].astype(np.float64)
    tol = 1.0e-6 * max(float(np.ptp(scaled, axis=0).max()), 1.0e-12)  # scene._weld's
    keys = [tuple(k) for k in np.round(scaled / tol).astype(np.int64)]
    place = {tuple(k): w for w, k in enumerate(np.round(np.asarray(welded, np.float64) / tol).astype(np.int64))}
    corners = np.full(len(welded), -1, np.int64)
    for c, i in enumerate(np.asarray(mesh.indices).reshape(-1)):
        w = place.get(keys[i])
        if w is not None and corners[w] < 0:
            corners[w] = c
    if len(place) != len(welded) or np.any(corners < 0):
        raise RuntimeError(
            f"SolverIPC: the triangulation of the mesh of shape {s} is not the weld of its vertices that the surface "
            "velocity assumes (scene._weld); the two must change together."
        )
    return corners.astype(np.int32)


def build(solver: SolverIPC, scene) -> None:
    """Find the colliding meshes with a surface velocity (``solver._features.surface_velocity``, None without one).
    ``NotImplementedError`` for one under ``contact_constitution="al-ipc"``, whose friction in libuipc-warp takes no
    surface velocity, and ``ImportError`` when the installed libuipc-warp has no ``uipc_warp.ext.surface_velocity``."""
    shapes = _belt_shapes(solver.model)
    solver._features.surface_velocity = _Belts(shapes) if len(shapes) else None
    if not len(shapes):
        return
    if solver._config.contact_constitution == "al-ipc":
        raise NotImplementedError(
            f"SolverIPC: shape {int(shapes[0])} is a mesh with enable_surface_velocity=True, which "
            "contact_constitution='al-ipc' does not take: libuipc-warp's AL-IPC friction has no surface velocity. "
            "Use contact_constitution='ipc', or turn the surface velocity off."
        )
    try:
        import uipc_warp.ext.surface_velocity  # noqa: F401, PLC0415
    except ImportError as e:
        from ..solver_ipc import _UIPC_WARP_INSTALL  # noqa: PLC0415

        raise ImportError(
            f"SolverIPC: shape {int(shapes[0])} is a mesh with enable_surface_velocity=True, which needs "
            "libuipc-warp's uipc_warp.ext.surface_velocity, and the installed libuipc-warp has none. Install a newer "
            f"one: {_UIPC_WARP_INSTALL}"
        ) from e


def engine_inputs(solver: SolverIPC, inputs: dict) -> None:
    """Give the engine the solve's ``SurfaceVelocity``: made with the solve's first engine, for its vertex count, and
    given again to every engine with longer lists that an eager step or ``grow()`` builds."""
    belts = solver._features.surface_velocity
    if belts is None:
        return
    if belts.extension is None:
        import uipc_warp.ext.surface_velocity as moving  # noqa: PLC0415

        model, device = solver.model, solver.model.device
        rows, shape, corner = [], [], []
        for s in belts.shapes:
            mine = np.flatnonzero(solver._vertex_shape == s)  # the shape's rows, in the order of its triangulation
            welded, _ = _shape_mesh(model, int(s), solver._shape_segments)
            rows.append(mine)
            shape.append(np.full(len(mine), s))
            corner.append(_first_corners(model, int(s), welded))
        belts.rows, belts.shape, belts.corner = (
            wp.array(np.concatenate(a).astype(np.int32), dtype=wp.int32, device=device) for a in (rows, shape, corner)
        )
        belts.extension = moving.SurfaceVelocity(len(inputs["x"]), device)
    inputs["extensions"].append(belts.extension)


def prepare(solver: SolverIPC, state_in: State, control: Control, dt: float) -> None:
    """Write the world velocity of every vertex of the meshes with a surface velocity, from their Warp meshes'
    velocities and the shapes' poses in ``state_in`` (device work only)."""
    belts = solver._features.surface_velocity
    if belts is None:
        return
    model = solver.model
    wp.launch(
        _write_surface_velocity,
        belts.rows.shape[0],
        inputs=[
            belts.rows, belts.shape, belts.corner, model.shape_body, state_in.body_q, model.shape_transform,
            model.shape_scale, model.shape_source_ptr,
        ],
        outputs=[belts.extension.velocity],
        device=model.device,
    )  # fmt: skip
