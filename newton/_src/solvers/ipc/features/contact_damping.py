# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact damping: a viscous force against the motion of surfaces in contact along their normal (the constructor's
``contact_damping``), through libuipc-warp's ``ContactDamping`` extension (see :ref:`ipc-contact-damping`).

libuipc's contact has no damping term: its barrier and friction are its only forces. ``contact_damping=None``, the
default, adds none. A number gives every pair in contact that coefficient [N s/m]. ``"material"`` takes Newton's
contact damping coefficients as Newton's VBD solver combines them: the mean of the two shapes'
:attr:`~newton.Model.shape_material_kd` for a pair of shapes, the mean of :attr:`~newton.Model.soft_contact_kd` and the
shape's for a particle and a shape, and ``soft_contact_kd`` for two particles (the vertices of cloth and tetrahedra
are particles). The extension's table has one row per distinct coefficient, every engine vertex and half plane the row
of its own, so that a pair's coefficient, the mean of the table's entries over the rows of its two sides, is that
mean.

The hooks are those of ``features`` (its docstring): ``check`` refuses a value it does not take, ``build`` refuses
AL-IPC, ``engine_inputs`` gives every engine of the solve the extension, and ``notify`` builds the solve again when the
coefficients ``"material"`` read have changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from newton import ModelFlags

if TYPE_CHECKING:
    from newton import Model

    from ..solver_ipc import SolverIPC
    from . import FeatureOptions


@dataclass
class _Damping:
    """The contact damping of a solve, and the extension that gives it to the solve's engines."""

    value: float | str  # the constructor's contact_damping: a number, or "material"
    materials: tuple | None  # with "material", what the table was made from: shape_material_kd and soft_contact_kd
    extension: object = None  # libuipc-warp's ContactDamping, made with the solve's first engine


def _materials(model: Model) -> tuple:
    """What ``"material"`` reads of ``model``: every shape's contact damping and that of soft contacts."""
    kd = model.shape_material_kd.numpy().astype(np.float64) if model.shape_count else np.zeros(0)
    return kd, float(model.soft_contact_kd)


def check(model: Model, options: FeatureOptions, at: str) -> None:
    """``ValueError`` for a ``contact_damping`` other than None, ``"material"`` or a finite number not below 0
    (``"options"``)."""
    value = options.contact_damping
    if at != "options" or value is None or (isinstance(value, str) and value == "material"):
        return
    number = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))
    if not number or not (np.isfinite(value) and value >= 0.0):
        raise ValueError(
            f"SolverIPC: contact_damping must be None (no damping), 'material' (Newton's contact damping "
            f"coefficients) or a finite coefficient not below 0 [N s/m], got {value!r}."
        )


def build(solver: SolverIPC, scene) -> None:
    """The solve's contact damping (``solver._features.contact_damping``, None without one). ``NotImplementedError``
    under ``contact_constitution="al-ipc"``, whose contact terms are an active set of its own."""
    value = solver._feature_options.contact_damping
    solver._features.contact_damping = None
    if value is None:
        return
    if solver._config.contact_constitution == "al-ipc":
        raise NotImplementedError(
            f"SolverIPC: contact_damping={value!r} is not offered with contact_constitution='al-ipc': AL-IPC's contact "
            "terms are an active set with multipliers, which libuipc-warp's contact damping does not act on. Use "
            "contact_constitution='ipc', or leave contact_damping at None."
        )
    materials = _materials(solver.model) if value == "material" else None
    solver._features.contact_damping = _Damping(value, materials)


def _material_rows(solver: SolverIPC, n: int, n_planes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(table, the row of every engine vertex, the row of every half plane) of ``"material"`` (module docstring): one
    row per coefficient in use, entry ``[a, b]`` the mean of rows ``a`` and ``b``. The vertices of a shape take its
    ``shape_material_kd``, particles ``soft_contact_kd``, half planes their shape's; the rows on no contact primitive
    (the bodies' degrees of freedom) take row 0, which no pair reads for them."""
    kd, soft = solver._features.contact_damping.materials
    vertex = np.full(n, np.nan)
    vertex[: solver.model.particle_count] = soft
    on_shape = solver._vertex_shape >= 0
    vertex[on_shape] = kd[solver._vertex_shape[on_shape]]
    plane = kd[np.asarray(solver._planes[2], np.int64)] if n_planes else np.zeros(0)
    primitive = ~np.isnan(vertex)
    values = np.unique(np.concatenate([vertex[primitive], plane, [soft] if not primitive.any() else []]))
    rows = np.zeros(n, np.int64)
    rows[primitive] = np.searchsorted(values, vertex[primitive])
    table = 0.5 * (values[:, None] + values[None, :])
    return table, rows, np.searchsorted(values, plane)


def engine_inputs(solver: SolverIPC, inputs: dict) -> None:
    """Give the engine the solve's ``ContactDamping``: made with the solve's first engine, and given again to every
    engine with longer lists that an eager step or ``grow()`` builds."""
    damping = solver._features.contact_damping
    if damping is None:
        return
    if damping.extension is None:
        import uipc_warp.ext.contact_damping as contact_damping  # noqa: PLC0415

        device = solver.model.device
        if damping.value == "material":
            table, rows, plane_rows = _material_rows(solver, len(inputs["x"]), len(inputs["plane_P"]))
            damping.extension = contact_damping.ContactDamping(
                table, elements=rows, plane_elements=plane_rows, device=device
            )
        else:
            damping.extension = contact_damping.ContactDamping(float(damping.value), device=device)
    inputs["extensions"].append(damping.extension)


def notify(solver: SolverIPC, flags: ModelFlags | int) -> bool:
    """With ``"material"`` and ``SHAPE_PROPERTIES`` or ``MODEL_PROPERTIES``, whether the shapes' or the soft contacts'
    damping coefficients have changed since the solve was built: the solve is then built again with the new ones, as
    after a change of a friction coefficient."""
    damping = solver._features.contact_damping
    properties = ModelFlags.SHAPE_PROPERTIES | ModelFlags.MODEL_PROPERTIES
    if damping is None or damping.materials is None or not flags & properties:
        return False
    kd, soft = _materials(solver.model)
    return not (np.array_equal(kd, damping.materials[0]) and soft == damping.materials[1])
