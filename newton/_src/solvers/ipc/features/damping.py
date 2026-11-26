# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The damping of springs, particle-body attachments, triangles, tetrahedra and bending edges, and drag and lift.

libuipc has no damping term; libuipc-warp's extension ``uipc_warp.ext.damping`` has one per rate that Newton's VBD
solver damps, with VBD's coefficient (see :ref:`ipc-damping`):

- ``spring_damping``: the rate of a spring's length (``SpringDamping``); for a spring of rest length 0, whose stitch
  pulls in every direction, the rate of the vector between its particles (``VectorDamping``);
- ``attachment_particle_body_damping``: the rate of the vector between the particle and its point on the body
  (``VectorDamping`` on the attachment's stitch), none for a disabled attachment;
- ``tet_materials[:, 2]``: the rate of a tetrahedron's ``F^T F`` (``StrainRateDamping``), where ``k_mu`` or
  ``k_lambda`` is positive;
- ``tri_materials[:, 2]``: the rate of a cloth triangle's ``F^T F``, with its ``tri_poses`` and ``tri_areas``
  (``StrainRateDamping``), where ``tri_ke`` or ``tri_ka`` is positive;
- ``edge_bending_properties[:, 1]``: the rate of a cloth hinge's dihedral angle, with its ``edge_rest_length``
  (``HingeDamping``), where the hinge's stiffness and rest length are positive;
- ``tri_materials[:, 3]`` and ``[:, 4]``: drag and lift, which SolverIPC computes as SemiImplicit does, explicitly from
  ``state_in``, and adds to the free particles' external accelerations.

When the solve is built a kind of damping takes every term where its condition holds, if one of them has a coefficient
above 0, and none otherwise. The coefficients are read on the device at every step. The hooks are those of
``features`` (its docstring).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..scene import _own_elements

if TYPE_CHECKING:
    from newton import Control, Model, ModelFlags, State

    from ..solver_ipc import SolverIPC
    from . import FeatureOptions


# ---------------------------------------------------------------- kernels
@wp.kernel
def _coefficient(source: wp.array[wp.int32], values: wp.array[wp.float32], out: wp.array[wp.float64]):
    """``out[i] = values[source[i]]``: a term's coefficient from the model's array of its kind."""
    i = wp.tid()
    out[i] = wp.float64(values[source[i]])


@wp.kernel
def _material(source: wp.array[wp.int32], materials: wp.array2d[wp.float32], column: int, out: wp.array[wp.float64]):
    """``out[i] = materials[source[i], column]``: an element's damping from its row of the model's materials."""
    i = wp.tid()
    out[i] = wp.float64(materials[source[i], column])


@wp.kernel
def _stitch_damping(
    source: wp.array[wp.int32],
    damping: wp.array[wp.float32],
    enabled: wp.array[wp.bool],
    spring_damping: wp.array[wp.float32],
    out: wp.array[wp.float64],
):
    """The damping [N s/m] of every stitch: ``source`` holds its attachment, or -1 less its spring of rest length 0
    (``SolverIPC._stitch_source``). A disabled attachment takes none."""
    r = wp.tid()
    s = source[r]
    c = wp.float64(0.0)
    if s >= 0:
        if enabled[s]:
            c = wp.float64(damping[s])
    else:
        c = wp.float64(spring_damping[-1 - s])
    out[r] = c


@wp.kernel
def _drag_and_lift(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    tris: wp.array[wp.vec3i],
    source: wp.array[wp.int32],
    materials: wp.array2d[wp.float32],
    force: wp.array[wp.vec3],
):
    """``f_drag + f_lift`` of every triangle at the state ``x``, ``v``, as SemiImplicit's ``eval_triangles`` computes
    them: each of the triangle's vertices takes the negative."""
    t = wp.tid()
    tri = tris[t]
    x0 = x[tri[0]]
    n = wp.cross(x[tri[1]] - x0, x[tri[2]] - x0)
    area = wp.length(n) * 0.5
    n = wp.normalize(n)
    vmid = (v[tri[0]] + v[tri[1]] + v[tri[2]]) * 0.3333
    vdir = wp.normalize(vmid)
    k_drag = materials[source[t], 3]
    k_lift = materials[source[t], 4]
    f_drag = vmid * (k_drag * area * wp.abs(wp.dot(n, vmid)))
    f_lift = n * (k_lift * area * (wp.HALF_PI - wp.acos(wp.dot(n, vdir)))) * wp.dot(vmid, vmid)
    force[t] = f_drag + f_lift


@wp.kernel
def _add_drag_and_lift(
    first: wp.array[wp.int32],
    tris: wp.array[wp.int32],
    force: wp.array[wp.vec3],
    particles: wp.array[wp.int32],
    mass: wp.array[wp.float32],
    acc: wp.array[wp.vec3d],
):
    """The acceleration of every free particle (row ``r`` of the external accelerations, particle ``particles[r]``) by
    the drag and lift of its triangles ``tris[first[r] : first[r + 1]]``, summed in that order: ``-sum f / m``."""
    r = wp.tid()
    f = wp.vec3d()
    for k in range(first[r], first[r + 1]):
        f = f + wp.vec3d(force[tris[k]])
    acc[r] = acc[r] - f / wp.float64(mass[particles[r]])


# ---------------------------------------------------------------- the terms of a solve
@dataclass
class _Terms:
    """The terms of one kind of damping in a solve: the model's row of every term (``rows``, and on the device
    ``source``), what their group is made from (host arrays, engine vertices), and the group of the engine."""

    rows: np.ndarray
    source: wp.array
    inputs: tuple
    group: object = None


@dataclass
class _Plan:
    """The damping of a solve (``solver._features.damping``): the terms of every kind (None for a kind without), and
    the drag and lift."""

    springs: _Terms | None = None  # SpringDamping, inputs (edges,)
    stitches: _Terms | None = None  # VectorDamping, inputs (pairs,); rows: SolverIPC's stitch sources
    tets: _Terms | None = None  # StrainRateDamping, inputs (tets, Dm_inv, volume)
    tris: _Terms | None = None  # StrainRateDamping, inputs (tris, Dm_inv, area)
    hinges: _Terms | None = None  # HingeDamping, inputs (hinges, rest_length)
    drag: dict | None = None  # _plan_drag


def _terms(rows: np.ndarray, device, *inputs) -> _Terms:
    return _Terms(rows, wp.array(np.asarray(rows, np.int32), dtype=wp.int32, device=device), inputs)


def _cloth(model: Model) -> tuple[np.ndarray, np.ndarray]:
    """The cloth of the solve as ``scene._add_cloth`` lays it out: the rows of ``tri_indices`` among the solver's own
    that are not on one of its tetrahedra, and the rows of ``edge_indices`` that are hinges of that cloth (both opposite
    particles present, the edge an edge of a cloth triangle)."""
    if model.tri_count == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    own_tris, own_tets, _ = _own_elements(model)
    tris = model.tri_indices.numpy().reshape(-1, 3).astype(np.int64)
    on_tet = np.zeros(model.particle_count, bool)
    if model.tet_count:
        on_tet[model.tet_indices.numpy().reshape(-1, 4)[own_tets].ravel()] = True
    cloth = np.flatnonzero(~on_tet[tris].all(axis=1) & own_tris)
    if not (model.edge_count and len(cloth)):
        return cloth, np.zeros(0, np.int64)
    e = model.edge_indices.numpy().reshape(-1, 4).astype(np.int64)
    n = model.particle_count
    t = tris[cloth]
    keys = np.unique(np.sort(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1) @ [n, 1])
    hinges = np.flatnonzero((e[:, 0] >= 0) & (e[:, 1] >= 0) & np.isin(np.sort(e[:, 2:4], axis=1) @ [n, 1], keys))
    return cloth, hinges


def _damped(rows: np.ndarray, eligible: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """The terms of a kind: those of ``rows`` that are ``eligible`` (a mask over the model's rows), if one of them has
    a coefficient above 0, and none otherwise."""
    rows = rows[eligible[rows]]
    return rows if np.any(coefficients[rows] > 0.0) else rows[:0]


# ---------------------------------------------------------------- hooks
def check(model: Model, options: FeatureOptions, at: str) -> None:
    """``NotImplementedError`` for a damping, drag or lift coefficient that is not finite, and for a damping or drag
    coefficient that is negative, which would make a step add energy: the springs' at ``"springs"`` and the particle-body
    attachments' at ``"attachments"``, before their other checks, and the elements' with the last checks of the model
    (``"shapes"``), where a damped hinge also needs a positive ``edge_rest_length``."""
    if at == "springs" and model.spring_count:
        _check_values(model.spring_damping.numpy(), _own_elements(model)[2], "spring {k} has spring_damping")
    elif at == "attachments" and model.attachment_particle_body_count:
        values = model.attachment_particle_body_damping.numpy()
        _check_values(values, np.ones(len(values), bool), "particle-body attachment {k} has damping")
    elif at == "shapes":
        own_tris, own_tets, _ = _own_elements(model)
        if model.tet_count:
            _check_values(model.tet_materials.numpy()[:, 2], own_tets, "tetrahedron {k} has the damping k_damp")
        if model.tri_count:
            materials = model.tri_materials.numpy()
            _check_values(materials[:, 2], own_tris, "triangle {k} has the damping tri_kd")
            _check_values(materials[:, 3], own_tris, "triangle {k} has the drag tri_drag")
            _check_values(materials[:, 4], own_tris, "triangle {k} has the lift tri_lift", signed=True)
        if model.edge_count:
            properties = model.edge_bending_properties.numpy()
            hinge = np.zeros(model.edge_count, bool)
            hinge[_cloth(model)[1]] = True
            _check_values(properties[:, 1], hinge, "bending edge {k} has the damping edge_kd")
            length = model.edge_rest_length.numpy()
            bad = np.flatnonzero(hinge & (properties[:, 0] > 0.0) & (properties[:, 1] > 0.0) & ~(length > 0.0))
            if len(bad):
                k = int(bad[0])
                raise NotImplementedError(
                    f"SolverIPC: bending edge {k} has the damping edge_kd {float(properties[k, 1])!r} and the "
                    f"edge_rest_length {float(length[k])!r}; make the rest length positive, which the damping of a "
                    "hinge scales with."
                )


def _check_values(values: np.ndarray, mine: np.ndarray, what: str, signed: bool = False) -> None:
    """``NotImplementedError`` for the first of ``values`` among ``mine`` that is not finite, or negative unless
    ``signed``, named by ``what`` (a format of ``k``, its index)."""
    bad = np.flatnonzero(mine & ~(np.isfinite(values) & (signed | (values >= 0.0))))
    if len(bad):
        k = int(bad[0])
        rule = "finite" if signed else "0 or positive"
        raise NotImplementedError(f"SolverIPC: {what.format(k=k)} {float(values[k])!r}; make it {rule}.")


def build(solver: SolverIPC, scene) -> None:
    """The damped terms of the solve (``solver._features.damping``), from what the solver laid out in ``scene``: its
    springs, stitches, tetrahedra and cloth, whose groups ``engine_inputs`` makes for every engine."""
    model, device = solver.model, solver.model.device
    plan = _Plan()
    own_tris, own_tets, own_springs = _own_elements(model)
    if model.spring_count:
        rest = model.spring_rest_length.numpy()
        rows = _damped(np.arange(model.spring_count), own_springs & (rest > 0.0), model.spring_damping.numpy())
        if len(rows):
            plan.springs = _terms(rows, device, model.spring_indices.numpy().reshape(-1, 2).astype(np.int64)[rows])
    source = solver._stitch_source
    if len(source):
        attached = source >= 0
        c = np.zeros(len(source))
        if attached.any():
            enabled = model.attachment_particle_body_enabled.numpy()[source[attached]]
            c[attached] = np.where(enabled, model.attachment_particle_body_damping.numpy()[source[attached]], 0.0)
        if (~attached).any():
            c[~attached] = model.spring_damping.numpy()[-1 - source[~attached]]
        if np.any(c > 0.0):
            plan.stitches = _terms(source, device, _stitch_pairs(solver, scene))
    if model.tet_count:
        materials = model.tet_materials.numpy()
        eligible = own_tets & ((materials[:, 0] > 0.0) | (materials[:, 1] > 0.0))
        rows = _damped(np.arange(model.tet_count), eligible, materials[:, 2])
        if len(rows):
            Dm_inv = model.tet_poses.numpy().astype(np.float64)[rows]
            tets = model.tet_indices.numpy().reshape(-1, 4).astype(np.int64)[rows]
            plan.tets = _terms(rows, device, tets, Dm_inv, 1.0 / (6.0 * np.abs(np.linalg.det(Dm_inv))))
    cloth, hinges = _cloth(model)
    if len(cloth):
        materials = model.tri_materials.numpy()
        eligible = np.zeros(model.tri_count, bool)
        eligible[cloth] = (materials[cloth, 0] > 0.0) | (materials[cloth, 1] > 0.0)
        rows = _damped(cloth, eligible, materials[:, 2])
        if len(rows):
            tris = model.tri_indices.numpy().reshape(-1, 3).astype(np.int64)[rows]
            Dm_inv = model.tri_poses.numpy().astype(np.float64)[rows]
            plan.tris = _terms(rows, device, tris, Dm_inv, model.tri_areas.numpy().astype(np.float64)[rows])
    if len(hinges):
        properties = model.edge_bending_properties.numpy()
        eligible = np.zeros(model.edge_count, bool)
        eligible[hinges] = (properties[hinges, 0] > 0.0) & (model.edge_rest_length.numpy()[hinges] > 0.0)
        rows = _damped(hinges, eligible, properties[:, 1])
        if len(rows):
            # Newton's bending edge (o0, o1, v1, v2) is libuipc's hinge (o0, v1, v2, o1)
            stencils = model.edge_indices.numpy().reshape(-1, 4).astype(np.int64)[rows][:, [0, 2, 3, 1]]
            plan.hinges = _terms(rows, device, stencils, model.edge_rest_length.numpy().astype(np.float64)[rows])
    plan.drag = _plan_drag(solver, own_tris)
    solver._features.damping = plan


def _stitch_pairs(solver: SolverIPC, scene) -> np.ndarray:
    """The engine vertices of the stitches of the solver's attachments and springs of rest length 0, in the order
    ``scene._add_stitches`` adds them (that of ``solver._stitch_source``): the particle and its body's anchor of an
    attachment, the two particles of a spring."""
    model, pairs = solver.model, []
    if model.attachment_particle_body_count:
        body, particle, _ = solver._attachment_inputs
        pairs.append(np.stack([particle, scene.layout().body_anchor_offsets[body] + solver._attachment_anchor], axis=1))
    if len(solver._zero_springs):
        pairs.append(model.spring_indices.numpy().reshape(-1, 2).astype(np.int64)[solver._zero_springs])
    return np.concatenate(pairs).astype(np.int64)


def _plan_drag(solver: SolverIPC, own_tris: np.ndarray) -> dict | None:
    """The triangles with drag or lift among the solver's own, and the free particles' triangles among them, for the
    rows of the external accelerations (one per free particle, in order): None where there are none."""
    model = solver.model
    if model.tri_count == 0:
        return None
    materials = model.tri_materials.numpy()
    rows = np.flatnonzero(own_tris & ((materials[:, 3] != 0.0) | (materials[:, 4] != 0.0)))
    free = np.flatnonzero(~solver._fixed_particle)  # the rows of SolverIPC's external accelerations
    if not (len(rows) and len(free)):
        return None
    tris = model.tri_indices.numpy().reshape(-1, 3).astype(np.int64)[rows]
    row_of = np.full(model.particle_count, -1, np.int64)
    row_of[free] = np.arange(len(free))
    corner = row_of[tris.ravel()]
    owner, tri = corner[corner >= 0], np.repeat(np.arange(len(rows)), 3)[corner >= 0]
    order = np.lexsort((tri, owner))
    device = model.device
    return {
        "tris": wp.array(tris.astype(np.int32), dtype=wp.vec3i, device=device),
        "source": wp.array(rows.astype(np.int32), dtype=wp.int32, device=device),
        "force": wp.zeros(len(rows), dtype=wp.vec3, device=device),
        "first": wp.array(
            np.searchsorted(owner[order], np.arange(len(free) + 1)).astype(np.int32), dtype=wp.int32, device=device
        ),
        "of_particle": wp.array(tri[order].astype(np.int32), dtype=wp.int32, device=device),
    }


def engine_inputs(solver: SolverIPC, inputs: dict) -> None:
    """The damping groups of an engine made from the scene, with the model's coefficients now."""
    plan = solver._features.damping
    if plan is None:
        return
    from uipc_warp.ext import damping as ext  # noqa: PLC0415

    device = solver.model.device
    if plan.springs is not None:
        plan.springs.group = ext.SpringDamping(*plan.springs.inputs, 0.0, device)
        inputs["elements"].append(plan.springs.group)
    if plan.stitches is not None:
        plan.stitches.group = ext.VectorDamping(*plan.stitches.inputs, 0.0, inputs["affine_bodies"], device)
        inputs["elements"] += plan.stitches.group.groups
    for terms in (plan.tets, plan.tris):
        if terms is not None:
            terms.group = ext.StrainRateDamping(*terms.inputs, 0.0, device)
            inputs["elements"].append(terms.group)
    if plan.hinges is not None:
        plan.hinges.group = ext.HingeDamping(*plan.hinges.inputs, 0.0, device)
        inputs["elements"].append(plan.hinges.group)
    _write_coefficients(solver)


def _write_coefficients(solver: SolverIPC) -> None:
    """The coefficients of every damping group from the model's arrays, on the device."""
    model, plan, device = solver.model, solver._features.damping, solver.model.device
    if plan.springs is not None:
        t = plan.springs
        wp.launch(_coefficient, len(t.rows), [t.source, model.spring_damping], [t.group.c], device=device)
    if plan.stitches is not None:
        t = plan.stitches
        attached = model.attachment_particle_body_count > 0
        inputs = [
            t.source,
            model.attachment_particle_body_damping if attached else None,
            model.attachment_particle_body_enabled if attached else None,
            model.spring_damping if model.spring_count else None,
        ]
        wp.launch(_stitch_damping, len(t.rows), inputs, [t.group.c], device=device)
    for t, materials, column in (
        (plan.tets, model.tet_materials, 2),
        (plan.tris, model.tri_materials, 2),
        (plan.hinges, model.edge_bending_properties, 1),
    ):
        if t is not None:
            wp.launch(_material, len(t.rows), [t.source, materials, column], [t.group.k], device=device)


def prepare(solver: SolverIPC, state_in: State, control: Control, dt: float) -> None:
    """The step's damping coefficients from the model, and the drag and lift of ``state_in`` added to the free
    particles' external accelerations, which the solver wrote from ``particle_f`` before."""
    plan = solver._features.damping
    if plan is None:
        return
    _write_coefficients(solver)
    drag = plan.drag
    if drag is not None:
        model, device = solver.model, solver.model.device
        inputs = [state_in.particle_q, state_in.particle_qd, drag["tris"], drag["source"], model.tri_materials]
        wp.launch(_drag_and_lift, drag["force"].shape[0], inputs, [drag["force"]], device=device)
        inputs = [drag["first"], drag["of_particle"], drag["force"], solver._free_particles, model.particle_mass]
        wp.launch(_add_drag_and_lift, solver._n_free, inputs, [solver._forces.a], device=device)


def notify(solver: SolverIPC, flags: ModelFlags | int) -> bool:
    """Whether an attachment's damping appeared that the solve has no terms for (``CONSTRAINT_PROPERTIES``, which
    builds the solve again only for new bodies, particles or points). A change of the springs' or the elements' damping,
    drag or lift already builds the solve again: ``notify_model_changed`` compares their arrays."""
    from newton import ModelFlags  # noqa: PLC0415

    model = solver.model
    plan = solver._features.damping
    if plan is None or plan.stitches is not None or not (flags & ModelFlags.CONSTRAINT_PROPERTIES):
        return False  # with stitch terms every attachment has one, its coefficient read at every step
    if not model.attachment_particle_body_count:
        return False
    damping = model.attachment_particle_body_damping.numpy()
    return bool(np.any((damping > 0.0) & model.attachment_particle_body_enabled.numpy()))
