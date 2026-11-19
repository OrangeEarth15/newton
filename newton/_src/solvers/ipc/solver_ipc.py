# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Incremental Potential Contact (IPC) solver backed by libuipc-warp."""

from __future__ import annotations

import gc
import importlib.metadata
import operator
import re
import warnings
from dataclasses import dataclass
from enum import IntFlag
from typing import ClassVar

import numpy as np
import warp as wp

from ...core.types import Axis, override
from ...exceptions import NewtonWarning
from ...geometry import GeoType, Mesh, ParticleFlags, ShapeFlags
from ...sim import BodyFlags, Contacts, Control, JointTargetMode, JointType, Model, ModelFlags, State, StateFlags
from ..solver import SolverBase

# libuipc-warp releases the solver is written for: from the first to before the second
_UIPC_WARP_RELEASES = ((0, 4), (0, 5))

# grow-and-redo attempts of an eager step whose contact lists keep overflowing, as libuipc-warp's FrameRunner
_MAX_REGROWS = 8

# the most edge-triangle pairs the start check tests: libuipc-warp's SurfaceChecker tests every pair, about 1e10
# pairs per second on a recent GPU, in one launch
_START_CHECK_PAIRS = 2**31 - 1

# libuipc's barrier activation distance (contact/d_hat) [m]. The default d_hat stays below the shares of the smallest
# colliding shape's half-size and of the median edge of the cloth and tetrahedral meshes at which the barrier spans the
# feature: a contact list overflows from about half the radius of a small sphere, a mesh stretches from one edge length
_D_HAT = 0.01
_D_HAT_SHAPE_SHARE = 0.25
_D_HAT_MESH_SHARE = 0.5


@wp.func
def _to64(a: wp.vec3) -> wp.vec3d:
    return wp.vec3d(wp.float64(a[0]), wp.float64(a[1]), wp.float64(a[2]))


@wp.func
def _to32(a: wp.vec3d) -> wp.vec3:
    return wp.vec3(float(a[0]), float(a[1]), float(a[2]))


@wp.func
def _differs(a: wp.vec3, b: wp.vec3) -> bool:
    return a[0] != b[0] or a[1] != b[1] or a[2] != b[2]


@wp.func
def _rotation64(q: wp.quat) -> wp.mat33d:
    qd = wp.quatd(wp.float64(q[0]), wp.float64(q[1]), wp.float64(q[2]), wp.float64(q[3]))
    return wp.quat_to_matrix(wp.normalize(qd))


@wp.func
def _nearest_rotation(A: wp.mat33d) -> wp.mat33d:
    """Rotation factor R of the polar decomposition A = R S (S symmetric positive semi-definite)."""
    U = wp.mat33d()
    s = wp.vec3d()
    V = wp.mat33d()
    wp.svd3(A, U, s, V)
    R = U * wp.transpose(V)
    if wp.determinant(R) < wp.float64(0.0):
        # a reflection: flip the direction of the smallest singular value
        k = int(2)
        if s[0] <= s[1] and s[0] <= s[2]:
            k = 0
        elif s[1] <= s[2]:
            k = 1
        D = wp.identity(n=3, dtype=wp.float64)
        D[k, k] = wp.float64(-1.0)
        R = U * D * wp.transpose(V)
    return R


@wp.kernel
def _load_particles(
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    fixed: wp.array[wp.int32],
    x: wp.array[wp.vec3d],
    v: wp.array[wp.vec3d],
    changed: wp.array[wp.int32],
):
    """Keep the float64 state where ``state_in`` holds its float32 rounding, take ``state_in`` elsewhere.

    ``changed[0]`` is set where the state of a free particle was taken.
    """
    i = wp.tid()
    taken = int(0)
    if _differs(_to32(x[i]), q[i]):
        x[i] = _to64(q[i])
        taken = 1
    if _differs(_to32(v[i]), qd[i]):
        v[i] = _to64(qd[i])
        taken = 1
    if taken != 0 and fixed[i] == 0:
        changed[0] = 1


@wp.kernel
def _load_bodies(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    written_q: wp.array[wp.transform],
    written_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    base: wp.array[wp.int32],
    kinematic: wp.array[wp.int32],
    x: wp.array[wp.vec3d],
    v: wp.array[wp.vec3d],
    changed: wp.array[wp.int32],
):
    """Keep the affine state of a body whose ``state_in`` pose and twist are what the solver wrote last; give
    another the rigid state of ``state_in``: q = (p, rows of R) and its rate from Newton's COM twist.

    ``changed[0]`` is set where a dynamic body took ``state_in``.
    """
    b = wp.tid()
    tf = body_q[b]
    tw = body_qd[b]
    seen = written_q[b]
    seen_tw = written_qd[b]
    same = int(1)
    for k in range(7):
        if tf[k] != seen[k]:
            same = 0
    for k in range(6):
        if tw[k] != seen_tw[k]:
            same = 0
    if same != 0:
        return
    R = _rotation64(wp.transform_get_rotation(tf))
    o = base[b]
    x[o] = _to64(wp.transform_get_translation(tf))
    for i in range(3):
        x[o + 1 + i] = wp.vec3d(R[i, 0], R[i, 1], R[i, 2])
    w = _to64(wp.spatial_bottom(tw))
    # Newton's linear velocity is the COM's; p is the body origin's position
    v[o] = _to64(wp.spatial_top(tw)) - wp.cross(w, R * _to64(body_com[b]))
    A_dot = wp.skew(w) * R
    for i in range(3):
        v[o + 1 + i] = wp.vec3d(A_dot[i, 0], A_dot[i, 1], A_dot[i, 2])
    if kinematic[b] == 0:
        changed[0] = 1


@wp.kernel
def _store_particles(
    x: wp.array[wp.vec3d],
    v: wp.array[wp.vec3d],
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
):
    i = wp.tid()
    q[i] = _to32(x[i])
    qd[i] = _to32(v[i])


@wp.kernel
def _store_bodies(
    x: wp.array[wp.vec3d],
    v: wp.array[wp.vec3d],
    base: wp.array[wp.int32],
    body_com: wp.array[wp.vec3],
    kinematic: wp.array[wp.int32],
    body_q_in: wp.array[wp.transform],
    body_qd_in: wp.array[wp.spatial_vector],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    written_q: wp.array[wp.transform],
    written_qd: wp.array[wp.spatial_vector],
):
    """Newton's body pose and COM twist from the affine state, A projected to its nearest rotation; also kept in
    ``written_q`` / ``written_qd`` for the next load."""
    b = wp.tid()
    if kinematic[b] != 0:
        tf = body_q_in[b]
        tw = body_qd_in[b]
        body_q[b] = tf
        body_qd[b] = tw
        written_q[b] = tf
        written_qd[b] = tw
        return
    o = base[b]
    A = wp.matrix_from_rows(x[o + 1], x[o + 2], x[o + 3])
    A_dot = wp.matrix_from_rows(v[o + 1], v[o + 2], v[o + 3])
    R = _nearest_rotation(A)
    c = _to64(body_com[b])
    # keep the centre of mass where the affine map puts it
    x_com = A * c + x[o]
    W = A_dot * wp.inverse(A)
    half = wp.float64(0.5)
    w = wp.vec3d(half * (W[2, 1] - W[1, 2]), half * (W[0, 2] - W[2, 0]), half * (W[1, 0] - W[0, 1]))
    r = wp.quat_from_matrix(R)
    tf = wp.transform(_to32(x_com - R * c), wp.quat(float(r[0]), float(r[1]), float(r[2]), float(r[3])))
    tw = wp.spatial_vector(_to32(v[o] + A_dot * c), _to32(w))
    body_q[b] = tf
    body_qd[b] = tw
    written_q[b] = tf
    written_qd[b] = tw


@wp.kernel
def _particle_accelerations(
    particle_f: wp.array[wp.vec3],
    has_force: int,
    particle_mass: wp.array[wp.float32],
    free: wp.array[wp.int32],
    acc: wp.array[wp.vec3d],
):
    """``f / m`` of the free particles, the first rows of the external accelerations."""
    k = wp.tid()
    a = wp.vec3d()
    if has_force != 0:
        i = free[k]
        a = _to64(particle_f[i]) / wp.float64(particle_mass[i])
    acc[k] = a


@wp.kernel
def _body_accelerations(
    body_f: wp.array[wp.spatial_vector],
    has_body_f: int,
    joint_f: wp.array[wp.float32],
    has_joint_f: int,
    joint_f_start: wp.array[wp.int32],
    dynamic_body: wp.array[wp.int32],
    body_com: wp.array[wp.vec3],
    mass_inv: wp.array[wp.mat44d],
    base: wp.array[wp.int32],
    x: wp.array[wp.vec3d],
    row0: int,
    acc: wp.array[wp.vec3d],
):
    """The accelerations of a dynamic body's q = (p, rows of A) under the wrench ``body_f`` plus the ``joint_f``
    of its free joint: M^-1 Q of its generalized force Q, four rows from ``row0 + 4 j``."""
    j = wp.tid()
    b = dynamic_body[j]
    f = wp.vec3d()
    t = wp.vec3d()
    if has_body_f != 0:
        wrench = body_f[b]
        f = _to64(wp.spatial_top(wrench))
        t = _to64(wp.spatial_bottom(wrench))
    s = joint_f_start[b]
    if has_joint_f != 0 and s >= 0:
        f = f + wp.vec3d(wp.float64(joint_f[s]), wp.float64(joint_f[s + 1]), wp.float64(joint_f[s + 2]))
        t = t + wp.vec3d(wp.float64(joint_f[s + 3]), wp.float64(joint_f[s + 4]), wp.float64(joint_f[s + 5]))
    o = base[b]
    A = wp.matrix_from_rows(x[o + 1], x[o + 2], x[o + 3])
    # The force f at the centre of mass c does the work f . (dp + dA c). The torque t does t . dtheta on a
    # rotation dA = [dtheta]x A and nothing on a stretch dA = S A (S symmetric).
    Q = wp.outer(f, _to64(body_com[b])) + wp.float64(0.5) * wp.skew(t) * wp.transpose(wp.inverse(A))
    # the dyadic mass couples p_k with row k of A only, through one 4 x 4 block for every k
    M_inv = mass_inv[j]
    r = row0 + 4 * j
    p_acc = wp.vec3d()
    for k in range(3):
        a = M_inv * wp.vec4d(f[k], Q[k, 0], Q[k, 1], Q[k, 2])
        p_acc[k] = a[0]
        acc[r + 1 + k] = wp.vec3d(a[1], a[2], a[3])
    acc[r] = p_acc


@wp.kernel
def _any_selected(world_mask: wp.array[wp.bool], entries: wp.array[wp.int32], selected: wp.array[wp.int32]):
    """``selected[0]`` = 1 when ``world_mask`` selects a world the solver simulates."""
    s = int(0)
    for k in range(entries.shape[0]):
        if world_mask[entries[k]]:
            s = 1
    selected[0] = s


def _release(version: str) -> tuple[int, ...]:
    """The release numbers at the start of a version string (``"0.4.0.dev0"`` gives ``(0, 4, 0)``)."""
    match = re.match(r"\d+(?:\.\d+)*", version)
    return tuple(int(part) for part in match.group(0).split(".")) if match else ()


_VERSION_OPERATORS = {
    ">=": operator.ge,
    ">": operator.gt,
    "<=": operator.le,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}


def _satisfies(version: str, specifier: str) -> bool:
    """Whether ``version`` meets every clause of a requirement specifier such as ``>=1.19,<1.20``, comparing
    release numbers only (a pre-release counts as its release)."""
    have = _release(version)
    for clause in specifier.split(","):
        match = re.fullmatch(r"\s*(~=|>=|<=|==|!=|>|<)\s*([0-9][0-9.]*)\S*\s*", clause)
        if match is None:
            continue
        op, bound = match.group(1), _release(match.group(2))
        width = max(len(have), len(bound))
        a, b = have + (0,) * (width - len(have)), bound + (0,) * (width - len(bound))
        ok = a >= b and have[: len(bound) - 1] == bound[:-1] if op == "~=" else _VERSION_OPERATORS[op](a, b)
        if not ok:
            return False
    return True


def _version_mismatches(uipc_warp_version: str, warp_version: str, warp_requirement: str | None) -> list[str]:
    """The installed versions outside the range this solver is written for, and Warp outside libuipc-warp's."""
    low, high = _UIPC_WARP_RELEASES
    out = []
    if not low <= _release(uipc_warp_version) < high:
        out.append(
            f"libuipc-warp {uipc_warp_version} (SolverIPC is written for libuipc-warp >={'.'.join(map(str, low))},"
            f"<{'.'.join(map(str, high))})"
        )
    if warp_requirement and not _satisfies(warp_version, warp_requirement):
        out.append(f"warp-lang {warp_version} (libuipc-warp requires warp-lang{warp_requirement})")
    return out


def _warp_requirement() -> str | None:
    """libuipc-warp's requirement on warp-lang, from its installed metadata (None without one)."""
    try:
        requirements = importlib.metadata.requires("libuipc-warp") or []
    except importlib.metadata.PackageNotFoundError:
        return None
    for requirement in requirements:
        match = re.match(r"warp-lang\s*\(?([^;)]*)\)?", requirement)
        if match:
            return match.group(1).strip() or None
    return None


def _colliding_shapes(model: Model) -> np.ndarray:
    """Which shapes the solver collides: a shape that collides with shapes, and, in a model with particles, a shape
    that collides with particles. In a model without particles a shape that collides with particles only collides
    with nothing."""
    flags = model.shape_flags.numpy()
    mask = int(ShapeFlags.COLLIDE_SHAPES) | (int(ShapeFlags.COLLIDE_PARTICLES) if model.particle_count else 0)
    return (flags & mask) != 0


def _default_d_hat(model: Model) -> float:
    """libuipc's d_hat, or less when the model has smaller colliding features [m]: a quarter of the smallest half-size of
    a colliding shape other than a plane (a sphere's or capsule's radius, the smaller of a cylinder's or cone's radius
    and half height, a box's or ellipsoid's smallest half extent, half of a mesh's thinnest extent that is not zero),
    and half the median rest length of the edges of the cloth triangles and tetrahedra."""
    size = np.inf
    if model.shape_count:
        types, scale = model.shape_type.numpy(), model.shape_scale.numpy().astype(np.float64)
        for s in np.flatnonzero(_colliding_shapes(model)):
            t, sc = int(types[s]), scale[s]
            if t in (GeoType.BOX, GeoType.ELLIPSOID):
                half = sc.min()
            elif t in (GeoType.SPHERE, GeoType.CAPSULE):
                half = sc[0]
            elif t in (GeoType.CYLINDER, GeoType.CONE):
                half = min(sc[0], sc[1])
            elif t in (GeoType.MESH, GeoType.CONVEX_MESH) and model.shape_source[s] is not None:
                extent = np.ptp(np.asarray(model.shape_source[s].vertices, np.float64).reshape(-1, 3) * sc, axis=0)
                extent = extent[extent > 0.0]
                half = 0.5 * extent.min() if len(extent) else np.inf
            else:
                continue
            if half > 0.0:
                size = min(size, float(half))
    d_hat = min(_D_HAT, _D_HAT_SHAPE_SHARE * size)
    edges = []
    if model.tri_count:
        t = model.tri_indices.numpy().reshape(-1, 3)
        edges.append(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]))
    if model.tet_count:
        t = model.tet_indices.numpy().reshape(-1, 4)
        edges.append(np.concatenate([t[:, [a, b]] for a in range(3) for b in range(a + 1, 4)]))
    if edges:
        e = np.unique(np.sort(np.concatenate(edges), axis=1), axis=0)
        x = model.particle_q.numpy().astype(np.float64)
        length = np.linalg.norm(x[e[:, 0]] - x[e[:, 1]], axis=1)
        length = length[length > 0.0]
        if len(length):
            d_hat = min(d_hat, _D_HAT_MESH_SHARE * float(np.median(length)))
    return d_hat


def _check_supported(model: Model) -> None:
    """Reject model content the solver would otherwise accept and ignore."""
    if model.requires_grad:
        raise NotImplementedError("SolverIPC does not support gradients (model.requires_grad=True).")
    for name, count in (
        ("springs", model.spring_count),
        ("particle-body attachments", model.attachment_particle_body_count),
        ("tendons", model.tendon_count + model.custom_frequency_counts.get("mujoco:tendon", 0)),
        ("muscles", model.muscle_count),
    ):
        if count:
            raise NotImplementedError(f"SolverIPC does not simulate {name}; the model has {count}.")
    mimic = int(model.constraint_mimic_count)
    if model.joint_count and model.joint_mimic_joint is not None:
        mimic += int(np.count_nonzero(model.joint_mimic_joint.numpy() >= 0))
    if mimic:
        raise NotImplementedError(f"SolverIPC does not enforce mimic constraints; the model has {mimic}.")
    mujoco = getattr(model, "mujoco", None)
    rows = int(getattr(mujoco, "equality_constraint_count", 0) or 0) if mujoco is not None else 0
    if rows:
        enabled = getattr(mujoco, "equality_constraint_enabled", None)
        rows = int(np.count_nonzero(enabled.numpy())) if enabled is not None else rows
    if rows:
        raise NotImplementedError(
            f"SolverIPC does not enforce equality constraints; model.mujoco has {rows} enabled equality rows."
        )
    for name, activations in (("triangle", model.tri_activations), ("tetrahedron", model.tet_activations)):
        if activations is not None and len(activations) and np.any(activations.numpy() != 0.0):
            raise NotImplementedError(f"SolverIPC does not take {name} activations; the model has non-zero ones.")
    if model.joint_count:
        joint_type = model.joint_type.numpy()
        if np.any(joint_type != int(JointType.FREE)):
            raise NotImplementedError(
                "SolverIPC does not simulate joints; only the free joints of stand-alone bodies are allowed."
            )
        if model.joint_target_mode is not None and model.joint_dof_count:
            mode = model.joint_target_mode.numpy()
            driven = np.flatnonzero((mode != int(JointTargetMode.NONE)) & (mode != int(JointTargetMode.EFFORT)))
            if len(driven):
                raise NotImplementedError(
                    f"SolverIPC does not drive joints: joint DOF {int(driven[0])} has the target mode "
                    f"{JointTargetMode(int(mode[driven[0]])).name}. Free joints take Control.joint_f only "
                    "(JointTargetMode.NONE or EFFORT)."
                )
    _check_one_world(model)
    _check_unfiltered(model)
    _check_surface_velocity(model)


def _check_one_world(model: Model) -> None:
    """One world, besides global entities: a solve couples everything it holds through one contact space, one
    step length and one convergence test."""
    worlds = []
    if model.particle_count:
        worlds.append(model.particle_world.numpy())
    if model.body_count:
        worlds.append(model.body_world.numpy())
    if model.shape_count:
        worlds.append(model.shape_world.numpy()[_colliding_shapes(model)])
    local = np.unique(np.concatenate(worlds)) if worlds else np.zeros(0, np.int32)
    local = local[local >= 0]
    if len(local) > 1:
        raise NotImplementedError(
            f"SolverIPC simulates one world: the model has particles, bodies or colliding shapes in worlds "
            f"{local.tolist()}, which would collide with each other and share one Newton solve."
        )


def _check_unfiltered(model: Model) -> None:
    """Every shape that collides collides with every shape on another body and with every particle, as in the
    solver's single contact space."""
    if model.shape_count == 0:
        return
    flags = model.shape_flags.numpy()
    colliding = _colliding_shapes(model)
    shapes_only = np.flatnonzero(colliding & ((flags & int(ShapeFlags.COLLIDE_PARTICLES)) == 0))
    if model.particle_count and len(shapes_only):
        raise NotImplementedError(
            f"SolverIPC collides every shape with the particles; shape {int(shapes_only[0])} has "
            "ShapeFlags.COLLIDE_SHAPES without COLLIDE_PARTICLES."
        )
    _, per_body = np.unique(model.shape_body.numpy()[colliding], return_counts=True)
    n = int(per_body.sum())
    pairs = n * (n - 1) // 2 - int((per_body * (per_body - 1) // 2).sum())
    if model.shape_contact_pair_count != pairs:
        raise NotImplementedError(
            "SolverIPC does not filter contacts: every colliding shape collides with every colliding shape on "
            f"another body ({pairs} pairs), but Model.shape_contact_pairs has {model.shape_contact_pair_count}; "
            "collision groups, filter pairs or, in a model with particles, shapes with "
            "ShapeFlags.COLLIDE_PARTICLES only exclude some."
        )


def _check_surface_velocity(model: Model) -> None:
    """Refuse a collision mesh with a surface velocity: Newton's contacts sample it (a conveyor belt), and the
    solver, which finds its own contacts, would ignore it."""
    if model.shape_count == 0:
        return
    meshes = np.flatnonzero(_colliding_shapes(model) & (model.shape_type.numpy() == int(GeoType.MESH)))
    for s in meshes:
        source = model.shape_source[s]
        if isinstance(source, Mesh) and source.enable_surface_velocity:
            raise NotImplementedError(
                f"SolverIPC does not apply mesh surface velocities: shape {int(s)} is a mesh with "
                "enable_surface_velocity=True."
            )


def _quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate the rows of ``v`` by the unit quaternion ``q = (x, y, z, w)``."""
    u = np.asarray(q[:3], np.float64)
    w = float(q[3])
    return v + 2.0 * w * np.cross(u, v) + 2.0 * np.cross(u, np.cross(u, v))


def _quat_to_matrix_np(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64) / np.linalg.norm(q)
    return _quat_rotate_np(q, np.eye(3)).T


def _weld(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge coincident vertices (the seams and poles of generated meshes) and drop degenerate triangles."""
    v = np.asarray(v, np.float64).reshape(-1, 3)
    f = np.asarray(f, np.int64).reshape(-1, 3)
    if len(v) == 0 or len(f) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    tol = 1.0e-6 * max(float(np.ptp(v, axis=0).max()), 1.0e-12)
    _, first, inv = np.unique(np.round(v / tol).astype(np.int64), axis=0, return_index=True, return_inverse=True)
    f = inv.reshape(-1)[f]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])]
    _, keep = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
    f = f[np.sort(keep)]
    used, f = np.unique(f, return_inverse=True)
    return v[first][used], f.reshape(-1, 3)


def _shape_mesh(model: Model, s: int, segments: int) -> tuple[np.ndarray, np.ndarray]:
    """Closed triangle mesh of shape ``s`` in its local frame, scaled."""
    t = int(model.shape_type.numpy()[s])
    sc = model.shape_scale.numpy()[s].astype(np.float64)
    opts = {"compute_normals": False, "compute_uvs": False, "compute_inertia": False}
    if t == GeoType.BOX:
        mesh = Mesh.create_box(sc[0], sc[1], sc[2], duplicate_vertices=False, **opts)
    elif t == GeoType.SPHERE:
        mesh = Mesh.create_sphere(sc[0], num_latitudes=max(segments // 2, 2), num_longitudes=segments, **opts)
    elif t == GeoType.ELLIPSOID:
        mesh = Mesh.create_ellipsoid(
            sc[0], sc[1], sc[2], num_latitudes=max(segments // 2, 2), num_longitudes=segments, **opts
        )
    elif t == GeoType.CAPSULE:
        mesh = Mesh.create_capsule(sc[0], sc[1], up_axis=Axis.Z, segments=segments, **opts)
    elif t == GeoType.CYLINDER:
        mesh = Mesh.create_cylinder(sc[0], sc[1], up_axis=Axis.Z, segments=segments, **opts)
    elif t == GeoType.CONE:
        mesh = Mesh.create_cone(sc[0], sc[1], up_axis=Axis.Z, segments=segments, **opts)
    elif t in (GeoType.MESH, GeoType.CONVEX_MESH):
        src = model.shape_source[s]
        return _weld(np.asarray(src.vertices, np.float64) * sc, np.asarray(src.indices))
    else:
        raise NotImplementedError(f"SolverIPC does not support shape {s} of type {GeoType(t).name}.")
    return _weld(mesh.vertices, mesh.indices)


def _closed_volume(v: np.ndarray, f: np.ndarray) -> float:
    """Enclosed volume of a closed triangle mesh (divergence theorem), either orientation."""
    if len(f) == 0:
        return 0.0
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


class SolverIPC(SolverBase):
    """Incremental Potential Contact solver for soft bodies, cloth and rigid bodies.

    Wraps the `libuipc-warp <https://github.com/OrangeEarth15/libuipc-warp>`_ engine,
    a Warp implementation of libuipc, through its array interface (no libuipc
    build is needed). Each :meth:`step` is one implicit-Euler (BDF1) time step:
    a Newton iteration with a backtracking line search, a block-Jacobi PCG
    linear solve, barrier contact and continuous collision detection. The
    Newton loop, the PCG loop and the line search run on the device under
    ``wp.capture_while``, so a step never synchronizes with the host when it is
    captured in a CUDA graph. A step that is not captured captures the solve
    itself and checks how it ended (see below).

    Supported model content:

    - particles and tetrahedra. Elasticity is the SNK1 stable Neo-Hookean
      model with Lamé parameters ``tet_materials[:, 0]`` (``k_mu``) and
      ``tet_materials[:, 1]`` (``k_lambda``). ``k_damp`` is ignored.
    - cloth: triangles of :attr:`~newton.Model.tri_indices` that are not on a
      tetrahedral body become libuipc's strain-limiting Baraff-Witkin membrane,
      and bending edges of :attr:`~newton.Model.edge_indices` become
      discrete-shell hinges (see the mapping below).
    - rigid bodies with box, sphere, capsule, cylinder, cone, ellipsoid or mesh
      collision shapes become affine bodies (see below).
    - contact against static plane shapes, each treated as an infinite half
      space, and against other static shapes, which become fixed triangle
      meshes.
    - contact between all of these: point-triangle and edge-edge barriers with
      continuous collision detection. Each connected group of tetrahedra or
      cloth triangles is one body and collides with itself (as finite elements
      do in libuipc), so ``d_hat`` must stay below the spacing of
      non-adjacent surface primitives at rest, as the default ``d_hat`` does
      (see the note below). A rigid body does not collide with itself.
    - kinematic particles (zero mass or not :attr:`~newton.ParticleFlags.ACTIVE`)
      and kinematic bodies (:attr:`~newton.BodyFlags.KINEMATIC` or zero mass).
      They follow the poses written to ``state_in`` every step; the jump is not
      swept by collision detection.
    - external forces: :attr:`~newton.State.particle_f`, :attr:`~newton.State.body_f`
      and the :attr:`~newton.Control.joint_f` of free joints, held constant over
      the step as libuipc holds its external forces (a constant acceleration in
      the implicit-Euler prediction). A body wrench acts at the centre of mass;
      its torque does work on the rotations of the affine body and none on its
      stretch.
    - one gravity vector, the same for all worlds.
    - lagged friction on every contact, with one coefficient ``friction_mu``.

    Not supported, and rejected when the solver is constructed:
    :class:`NotImplementedError` for springs, particle-body attachments, mimic
    and equality constraints, tendons, muscles, non-zero triangle or
    tetrahedron activations, joints other than the free joints that
    :meth:`~newton.ModelBuilder.add_body` creates, drive targets on those free
    joints (only :class:`~newton.JointTargetMode` ``NONE`` and ``EFFORT``),
    gradients (``model.requires_grad``), content in more than one world (the
    solver has one contact space and one Newton solve), contact filtering (a
    collision group, a filter pair, or, in a model with particles, a shape that
    collides only with shapes or only with particles: every colliding shape
    collides with everything; in a model without particles, a shape that
    collides with particles only collides with nothing),
    collision meshes with a surface velocity (``Mesh.enable_surface_velocity``,
    which the solver's own contacts would ignore), and a CPU device.
    Height-field and SDF shapes and planes attached to bodies raise when the
    shapes are triangulated. The ``contacts`` argument of
    :meth:`step` is not used: the solver detects its own contacts.

    Not read: per-shape friction and contact stiffness and
    :attr:`~newton.Model.particle_radius` (the arguments ``friction_mu``,
    ``kappa`` and ``thickness`` hold for all of them), damping coefficients,
    drag, lift, :attr:`~newton.Model.tri_poses` and
    :attr:`~newton.Model.edge_rest_angle` (see below), the activations of a
    :class:`~newton.Control`, and ``joint_q`` (not written).

    **Cloth parameters.** libuipc-warp's membrane energy per rest area is

    .. math::

        \\Psi = K_a \\phi(|F a|) + K_b \\phi(|F b|) + K_g ((F a) \\cdot (F b))^2,
        \\qquad \\phi(s) = (s - 1)^2 + \\eta \\max(s - 1, 0)^3,

    with the material axes ``a``, ``b`` of each triangle's rest frame. Newton
    integrates its membrane energy over the rest area only, so
    ``tri_materials[:, 0]`` (``tri_ke`` = :math:`\\mu`) and ``tri_materials[:, 1]``
    (``tri_ka`` = :math:`\\lambda`) act as membrane Lamé moduli [N/m]. At small
    strain Newton's membrane is linear elasticity,
    :math:`\\mu\\,\\varepsilon:\\varepsilon + \\tfrac{\\lambda}{2}(\\mathrm{tr}\\,\\varepsilon)^2`,
    and the mapping matches its energy for a stretch along either material
    axis with the other held fixed and for simple shear:

    - :math:`K_a = K_b = \\mu + \\lambda / 2` [N/m];
    - :math:`K_g = \\mu / 2` [N/m], since :math:`(F a) \\cdot (F b) \\approx 2 \\varepsilon_{ab}`;
    - :math:`\\eta` = ``cloth_strain_rate`` (libuipc's strain limiting; 0 keeps the
      membrane quadratic). The Baraff-Witkin model has no Poisson coupling
      (no :math:`\\lambda\\,\\varepsilon_{aa}\\varepsilon_{bb}` term), so an equal stretch in both
      directions meets less resistance than in Newton's model.

    Newton's hinge energy is :math:`\\tfrac{1}{2} k_e L_0 (\\theta - \\bar\\theta)^2`
    (``edge_bending_properties[:, 0]`` = :math:`k_e`); libuipc's is
    :math:`\\kappa L_0 (\\theta - \\bar\\theta)^2 / \\bar h` with
    :math:`\\bar h = (A_1 + A_2) / (3 L_0)` from the two rest triangle areas, so
    each hinge gets :math:`\\kappa = k_e \\bar h / 2`. Particle masses are used as
    they are. The damping coefficients ``tri_kd`` and ``edge_kd``, drag and
    lift are ignored: implicit Euler supplies the only damping. Rest shapes
    (the membrane's rest frames, rest lengths and rest angles) are taken from
    :attr:`~newton.Model.particle_q`, as libuipc takes them from the rest
    geometry; Newton's ``tri_poses`` and ``edge_rest_angle`` are not read, so a
    cloth built in a deformed state rests in that state.

    **Rigid bodies.** Each Newton body becomes a libuipc affine body with 12
    degrees of freedom :math:`q = (p, a_1, a_2, a_3)`, the translation and the
    rows of :math:`A`, in Newton's body frame: a point with body-frame
    coordinates :math:`\\bar x` sits at :math:`x = A \\bar x + p`, starting from
    :math:`A = R(q_{body})` and :math:`p` = the body origin. Its kinetic term uses
    libuipc's dyadic mass, built from :attr:`~newton.Model.body_mass` :math:`m`,
    :attr:`~newton.Model.body_com` :math:`c` and :attr:`~newton.Model.body_inertia`
    :math:`I` (which Newton computes from the shapes' densities):
    :math:`\\sum m \\bar x = m c` and
    :math:`\\sum m \\bar x \\bar x^T = \\tfrac{1}{2}\\mathrm{tr}(I)\\,1 - I + m c c^T`, so the
    affine kinetic energy equals the rigid one for every rigid motion. The
    orthogonality potential :math:`\\kappa V \\Delta t^2 \\|A A^T - I\\|_F^2` with
    ``body_kappa`` and the volume :math:`V` of the body's collision meshes
    keeps :math:`A` within a small strain of a rotation; this is libuipc's
    near-rigid body, not an exactly rigid one. Gravity acts on :math:`p`; with
    the full dyadic mass this is the exact gravity force on the affine body.
    An external force :math:`f` at the centre of mass is the generalized force
    :math:`(f, f c^T)` on :math:`(p, A)`, and a torque :math:`\\tau` is
    :math:`(0, \\tfrac{1}{2} [\\tau]_\\times A^{-T})`. The body's collision shapes
    are triangulated (curved shapes with ``shape_segments`` segments around)
    and act as its contact surface.

    After each step the solver writes ``body_q`` and ``body_qd``. :math:`A` is
    projected onto its nearest rotation :math:`R` by the polar decomposition
    :math:`A = R S` (from the SVD :math:`A = U \\Sigma V^T`, :math:`R = U V^T`),
    since a quaternion can only hold a rotation; the residual stretch
    :math:`S - 1` is of the order of the orthogonality strain. The body origin
    is placed so that the centre of mass stays where the affine map puts it,
    :math:`x_{com} = A c + p`; the linear velocity is that point's,
    :math:`\\dot p + \\dot A c`, and the angular velocity is the axial vector of
    the skew part of :math:`\\dot A A^{-1}`.

    **State.** The solver keeps positions and velocities in float64 and reads
    ``state_in`` on every step. A particle whose ``state_in`` position and
    velocity are the float32 rounding of the solver's state, and a body whose
    pose and twist are what the solver last wrote, keep the float64 state;
    anything else is taken from ``state_in``, a body as the rigid motion of its
    pose and twist. Taking a new state for a particle or body that is not
    kinematic discards the friction of that step's contacts, as libuipc does
    after a state is written and as a solver started from that state does.
    :meth:`reset` does the same without a new state. Output is written to
    ``state_out`` in float32.

    **Status.** Request :attr:`SolverIPC.ObservableKind.STATUS` and
    ``ObservableKind.SOLVER_RESIDUAL`` with :meth:`observables` and pass the
    container to :meth:`step`: the step writes a :class:`SolverIPC.StatusBits` mask and
    the residual on the device, also during graph replay. Every world that
    owns particles or bodies receives the values of the one solve. The
    residual is the convergence measure of libuipc's Newton loop: the largest
    displacement of the last Newton direction (the rows of an affine body's
    :math:`A` scaled by ``newton_velocity_tol / newton_transrate_tol``) over
    ``newton_velocity_tol * dt``. It is dimensionless, at most one when the
    loop ended on its displacement test, may be above one when the
    semi-implicit test ended it, and infinite when the step failed on a value
    in :class:`SolverIPC.StatusBits`.

    A step that is not captured reads the status itself after the solve, with
    one synchronization. When a contact list overflowed it enlarges the lists
    and solves the step again; when the step failed otherwise (a non-finite
    value, a contact starting at or inside its thickness, an exhausted BVH
    traversal), it raises :class:`RuntimeError` and leaves ``state_out``
    unwritten. A captured step cannot do either: it sets the bit, and nothing
    else reports the failure. A caller that captures :meth:`step` must
    therefore request ``STATUS``, pass one container to each captured step
    (a container holds the bits of the last step that wrote it), and after
    every replay call :meth:`check_status` with each container. It raises
    :class:`RuntimeError` on a failed step and warns on an iteration limit;
    with ``grow=True`` it answers :attr:`SolverIPC.StatusBits.OVERFLOW` with
    :meth:`grow` and returns ``True``, and the caller then restores the
    state the replayed steps started from, captures again and repeats them.

    This solver requires the optional ``libuipc-warp`` package (``import uipc_warp``).

    Args:
        model: The model to simulate.
        kappa: Barrier stiffness [Pa], libuipc's contact resistance.
        d_hat: Barrier activation distance [m]. ``None`` takes libuipc's
            0.01 m, or less for a model with small colliding features (see
            the note below).
        thickness: Contact thickness of every particle [m]; rigid-body and
            static-shape vertices have none, as in libuipc.
        newton_max_iter: Maximum Newton iterations per step.
        newton_velocity_tol: Newton convergence tolerance on ``max |dx| / dt`` [m/s].
        newton_transrate_tol: Newton convergence tolerance on ``max |dA| / dt`` of
            rigid bodies [1/s].
        linear_tol_rate: Relative PCG tolerance on ``r^T M^-1 r``.
        line_search_max_iter: Maximum backtracking trials per Newton iteration.
        semi_implicit: Enable libuipc's semi-implicit early exit after
            ``semi_implicit_k_min`` iterations of full feasible steps.
        semi_implicit_k_min: Newton iterations before the early exit applies.
        friction_mu: Friction coefficient of every contact; 0 disables friction.
        eps_velocity: Friction smoothing velocity [m/s]; below it static friction becomes a stiff spring.
        cloth_strain_rate: Strain-limiting coefficient :math:`\\eta` of the cloth
            membrane (libuipc's default 100); 0 makes the membrane quadratic.
        body_kappa: Orthogonality stiffness :math:`\\kappa` of rigid bodies [Pa]
            (libuipc's affine-body stiffness).
        shape_segments: Segments around a sphere, capsule, cylinder, cone or
            ellipsoid when it is triangulated.

    Note:
        A step ends once the Newton update moves no vertex by more than
        ``newton_velocity_tol * dt``. When a body should gain less velocity
        per step than ``newton_velocity_tol`` (slow sliding against friction,
        for example), lower the tolerance, or the body creeps at about
        ``eps_velocity`` instead.

        With a stiff barrier (large ``kappa``) a resting contact settles just
        inside ``d_hat``, so objects rest up to ``d_hat`` apart; lower ``d_hat``
        for tighter contact. Without a ``d_hat`` argument the solver keeps it
        at or below a quarter of the smallest half-size of a colliding shape (a
        sphere's or capsule's radius, a box's smallest half extent, half of a
        mesh's thinnest extent) and half the median rest length of the edges
        of the cloth triangles and tetrahedra, chosen whenever the solve is
        built. A ``d_hat`` that spans a small shape lets it rest up to its own
        size away and puts more contact pairs inside ``d_hat`` than the contact
        lists reserve, which a captured step reports as
        :attr:`SolverIPC.StatusBits.OVERFLOW`; one as long as the edges of a
        cloth or soft body puts its non-adjacent primitives inside ``d_hat`` at
        rest, and the barrier stretches it. A smaller ``d_hat`` also leaves
        less room for the jump of a kinematic body, which is not swept.

        The barrier needs a gap at the start. When the surfaces of different
        bodies cross each other, or are at or inside their contact thickness (up
        to the rounding of float32 positions), in the model's state, the solver
        raises :class:`ValueError` at construction and names the closest pair,
        as libuipc's sanity check rejects such a scene at init. Surfaces that
        never move may touch or cross; the inside of a body is not checked, nor
        two particles that are on no cloth or tetrahedron. A gap inside
        ``d_hat`` is valid, but there the barrier starts compressed and pushes
        the surfaces apart in the first steps. The check compares the bodies
        whose bounding boxes come that close, every primitive of one against
        every primitive of the other, and skips a pair of bodies with more than
        2**31 - 1 primitive pairs with a warning. A state written between steps
        is not checked: a step that starts in contact fails
        (:attr:`SolverIPC.StatusBits.THICKNESS_VIOLATED` or
        :attr:`SolverIPC.StatusBits.INVALID_CCD_STEP`).
    """

    class ObservableKind(SolverBase.ObservableKind):
        """Standard and IPC-specific observable kinds.

        Inherits all standard names from :class:`SolverBase.ObservableKind`.

        .. experimental::
        """

        STATUS = "status"
        """Per-world bitmask of :class:`SolverIPC.StatusBits` for the most recent step."""

    @dataclass(eq=False)
    class Observables(SolverBase.Observables):
        """Standard and IPC-specific observable arrays.

        .. experimental::
        """

        status: wp.array[wp.int32] | None = SolverBase.Observables.field(  # noqa: RUF009
            dtype=wp.int32, frequency=Model.AttributeFrequency.WORLD
        )
        """Per-world :class:`SolverIPC.StatusBits` of the latest step, shape ``(world_count + 1,)``.

        Every world that owns particles or bodies gets the mask of the one solve; the final entry is global
        world ``-1``. Other entries stay zero.
        """

    class StatusBits(IntFlag):
        """Bits of the per-world ``Observables.status`` mask: how the solve of the step ended.

        The step failed, and its result must not be used, when a bit of :attr:`SolverIPC.StatusBits.FAILED` is set. The two
        iteration limits are not failures.
        """

        OVERFLOW = 1 << 0
        """A contact list overflowed and dropped pairs: the step ran without them. Set from then on until the
        lists are enlarged (:meth:`SolverIPC.grow`, or an eager step)."""

        NONFINITE_ENERGY = 1 << 1
        """The line search met a non-finite energy."""

        NONFINITE_PCG_INIT = 1 << 2
        """The linear solve started from a non-finite residual."""

        NONFINITE_PCG_ITER = 1 << 3
        """The linear solve produced a non-finite residual."""

        INVALID_CCD_STEP = 1 << 4
        """Continuous collision detection admitted no step (a fraction that is not positive), as for a vertex
        that starts at or inside its contact thickness from a plane."""

        THICKNESS_VIOLATED = 1 << 5
        """A contact pair started at or inside its contact thickness (surfaces in touching contact or
        intersecting)."""

        TRAVERSAL_EXHAUSTED = 1 << 6
        """A BVH traversal ran out of its node budget: contact candidates are missing."""

        LINE_SEARCH_LIMIT = 1 << 7
        """A line search ran out of its ``line_search_max_iter`` trials."""

        NEWTON_LIMIT = 1 << 8
        """The Newton loop ran to ``newton_max_iter`` without converging."""

        FAILED = (
            OVERFLOW
            | NONFINITE_ENERGY
            | NONFINITE_PCG_INIT
            | NONFINITE_PCG_ITER
            | INVALID_CCD_STEP
            | THICKNESS_VIOLATED
            | TRAVERSAL_EXHAUSTED
        )
        """The bits of a failed step."""

    SUPPORTED_OBSERVABLES = frozenset({SolverBase.ObservableKind.SOLVER_RESIDUAL, ObservableKind.STATUS})

    # libuipc-warp and its external-acceleration element group, imported by the first solver
    _uipc_warp: ClassVar = None
    _external_acceleration: ClassVar = None

    def __init__(
        self,
        model: Model,
        *,
        kappa: float = 1.0e9,
        d_hat: float | None = None,
        thickness: float = 0.0,
        newton_max_iter: int = 1024,
        newton_velocity_tol: float = 0.05,
        newton_transrate_tol: float = 0.1,
        linear_tol_rate: float = 1.0e-3,
        line_search_max_iter: int = 8,
        semi_implicit: bool = True,
        semi_implicit_k_min: int = 6,
        friction_mu: float = 0.5,
        eps_velocity: float = 0.01,
        cloth_strain_rate: float = 100.0,
        body_kappa: float = 1.0e8,
        shape_segments: int = 24,
    ):
        if not model.device.is_cuda:
            raise NotImplementedError(f"SolverIPC runs on CUDA devices only; the model is on {model.device}.")
        super().__init__(model)
        _check_supported(model)
        uipc_warp = self._import_uipc_warp()
        self._d_hat_given = d_hat is not None
        self._config = uipc_warp.SimConfig(
            kappa=kappa,
            d_hat=_D_HAT if d_hat is None else d_hat,
            newton_max_iter=newton_max_iter,
            newton_velocity_tol=newton_velocity_tol,
            newton_transrate_tol=newton_transrate_tol,
            linear_tol_rate=linear_tol_rate,
            line_search_max_iter=line_search_max_iter,
            semi_implicit=semi_implicit,
            semi_implicit_k_min=semi_implicit_k_min,
            friction_mu=friction_mu,
            eps_velocity=eps_velocity,
        )
        self._thickness = float(thickness)
        self._cloth_strain_rate = float(cloth_strain_rate)
        self._body_kappa = float(body_kappa)
        self._shape_segments = int(shape_segments)
        self._dt = self._config.dt  # of the last step, which grow() solves again
        self._engine = None
        self._build_engine()
        self._check_start()

    @classmethod
    def _import_uipc_warp(cls):
        """Import libuipc-warp once: raise if it is missing or lacks what the solver calls, warn on versions it was
        not written for."""
        if cls._uipc_warp is not None:
            return cls._uipc_warp
        try:
            import uipc_warp  # noqa: PLC0415
            import uipc_warp.arrays  # noqa: PLC0415
            from uipc_warp.backend.external_force import external_acceleration  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "SolverIPC requires the libuipc-warp package: pip install git+https://github.com/OrangeEarth15/libuipc-warp"
            ) from e
        version = getattr(uipc_warp, "__version__", "0")
        if not hasattr(uipc_warp.arrays, "write_frame_status"):
            raise ImportError(f"SolverIPC requires libuipc-warp 0.4 or newer; the installed one is {version}.")
        # write_frame_status writes libuipc-warp's FrameStatusFlags into the status arrays: the same bits under the
        # same names, but for the CCD bit, which libuipc-warp names after the time of impact
        flags = {int(flag): flag.name for flag in uipc_warp.arrays.FrameStatusFlags}
        bits = {int(bit): bit.name for bit in cls.StatusBits}
        renamed = int(cls.StatusBits.INVALID_CCD_STEP)
        if flags.keys() != bits.keys() or any(flags[b] != bits[b] for b in bits if b != renamed):
            raise RuntimeError(
                f"Incompatible libuipc-warp {version}: its FrameStatusFlags {flags} are not SolverIPC.StatusBits."
            )
        mismatches = _version_mismatches(version, wp.config.version, _warp_requirement())
        if mismatches:
            warnings.warn(
                "SolverIPC is not tested with " + "; ".join(mismatches) + ".",
                NewtonWarning,
                stacklevel=3,
            )
        cls._uipc_warp = uipc_warp
        cls._external_acceleration = external_acceleration.ExternalAcceleration
        return uipc_warp

    # ------------------------------------------------------------------ setup
    def _build_engine(self) -> None:
        """Lay the model out for libuipc-warp and build its engine. The next step takes its whole state from
        ``state_in``."""
        model = self.model
        device = model.device
        self._engine = self._frame_graph = self._forces = None
        if model.particle_count == 0 and model.body_count == 0:
            return
        gravity = model.gravity.numpy().reshape(-1, 3).astype(np.float64)
        if len(gravity) > 1 and not np.allclose(gravity, gravity[0]):
            raise NotImplementedError("SolverIPC supports a single gravity vector for all worlds.")
        self._config.gravity = tuple(gravity[0])
        if not self._d_hat_given:
            self._config.d_hat = _default_d_hat(model)

        scene = self._uipc_warp.arrays.ArrayScene()
        tets = self._add_particles_and_tets(scene)
        self._add_cloth(scene, tets)
        self._dyadic = self._add_shapes_and_bodies(scene)
        self._planes = self._collect_planes()
        for P, N in zip(*self._planes[:2], strict=True):
            scene.add_half_plane(P, N)
        self._scene = scene
        self._shape_geometry = self._shape_geometry_now()

        layout = scene.layout()
        # gravity acts on the finite-element vertices and on the translation of a body (ArrayScene's layout)
        self._gravity_scale = np.zeros(layout.n)
        self._gravity_scale[: layout.n_fem] = 1.0
        self._gravity_scale[layout.body_base] = 1.0
        # the external accelerations: one row per free particle, then four per dynamic body (p and the rows of A)
        free = np.flatnonzero(~self._fixed_particle).astype(np.int32)
        dynamic = np.flatnonzero(~self._kinematic_body).astype(np.int32)
        self._n_free, self._n_dynamic = len(free), len(dynamic)
        self._force_blocks = np.concatenate([free, (layout.body_base[dynamic][:, None] + np.arange(4)).reshape(-1)])
        self._free_particles = wp.array(free, dtype=wp.int32, device=device)
        self._dynamic_bodies = wp.array(dynamic, dtype=wp.int32, device=device)
        self._body_mass_inv = wp.array(self._mass_inverse(*self._dyadic), dtype=wp.mat44d, device=device)
        self._changed = wp.zeros(1, dtype=wp.int32, device=device)
        self._selected = wp.zeros(1, dtype=wp.int32, device=device)
        worlds = np.unique(
            np.concatenate([w.numpy() for w in (model.particle_world, model.body_world) if w is not None and len(w)])
        )
        self._world_entries = np.where(worlds < 0, model.world_count, worlds).astype(np.int32)
        self._world_entries_device = wp.array(self._world_entries, dtype=wp.int32, device=device)
        nb = model.body_count
        if nb:
            self._body_base = wp.array(layout.body_base.astype(np.int32), dtype=wp.int32, device=device)
            self._body_kinematic = wp.array(self._kinematic_body.astype(np.int32), dtype=wp.int32, device=device)
            self._joint_f_start = wp.array(self._free_joint_dofs(), dtype=wp.int32, device=device)
            self._body_q_written = wp.empty(nb, dtype=wp.transform, device=device)
            self._body_qd_written = wp.empty(nb, dtype=wp.spatial_vector, device=device)
            self._forget_written()
        self._make_engine(None)

    def _check_start(self) -> None:
        """Raise ``ValueError`` when the surfaces of different bodies cross each other in the model's state, or start
        at or inside their contact thickness.

        The barrier needs surfaces that start apart, by more than their contact thickness. A step fails on a contact
        pair at or inside its thickness (``THICKNESS_VIOLATED`` or ``INVALID_CCD_STEP``), which a captured step only
        reports, and surfaces that cross without such a pair (an edge through a triangle, away from its vertices) step
        on with the barrier keeping them crossed. As libuipc's sanity check does at init, this tests the model's state
        once, without the surfaces that never move (static shapes, kinematic particles and bodies), whose contact the
        solve skips. Every contact point that moves is tested against every half plane. Triangle meshes are tested body
        pair by body pair, for the pairs whose bounding boxes come within their contact thickness of each other, with
        libuipc-warp's brute-force ``SurfaceChecker``: its edge-triangle crossings and its point-triangle and edge-edge
        distances. A distance above the thickness, also inside ``d_hat``, is a valid start, unless it is within a few
        float32 steps of the thickness: Newton's positions are float32, and surfaces placed in touching contact come out
        that far apart. Not tested: the inside of a body (contact ignores crossings inside a rigid body made of
        overlapping shapes, and inside a cloth or a soft body the checker cannot tell the nearly coplanar triangles of a
        flat sheet from crossing ones) and two particles that are on no cloth or tetrahedron (the checker has no
        point-point distance). A body pair of more than ``_START_CHECK_PAIRS`` primitive pairs is skipped with a
        warning.
        """
        eng = self._engine
        if eng is None:
            return
        layout = self._scene.layout()
        still = np.zeros(layout.n, bool)  # vertices that the solve never moves
        still[: layout.n_fem] = self._fixed_particle
        still[layout.obstacle_offsets[0] : layout.obstacle_offsets[-1]] = True
        offsets = layout.body_vertex_offsets
        for b in np.flatnonzero(self._kinematic_body):
            still[offsets[b] : offsets[b + 1]] = True
        x = eng.vm.x.numpy()
        thickness = np.zeros(layout.n)
        thickness[: layout.n_fem] = self._thickness
        points = np.asarray(eng.contact_points, np.int64)
        if len(points) == 0:
            return
        # the engine's contact bodies, with the particles on no surface (each a body of its own) taken as one
        group = eng.body.numpy().astype(np.int64)
        lone = np.setdiff1d(points, np.asarray(eng.surface["vertices"], np.int64))
        if len(lone):
            group[lone] = group.max() + 1
        ids, point_group = np.unique(group[points], return_inverse=True)
        # Newton's positions are float32: surfaces placed in touching contact come out up to a few float32 steps
        # apart, which the barrier would take as a valid, nearly singular start. Such a gap counts as contact.
        slack = 4.0 * float(np.spacing(np.float32(np.abs(x[points]).max())))
        # (crossings, gap, distance, pair thickness, contact body, the half plane (as -1 - plane) or contact body)
        violations = []

        # contact points against half planes
        P, N, plane_shapes = self._planes
        moves = ~still[points]
        if moves.any() and len(P):
            d = x[points[moves]] @ N.T - np.einsum("ij,ij->i", P, N)
            gap = d - thickness[points[moves]][:, None]
            owner = point_group[moves]
            for j in range(len(P)):
                hit = np.flatnonzero(gap[:, j] <= slack)
                for g in np.unique(owner[hit]):
                    k = hit[owner[hit] == g]
                    k = k[np.argmin(gap[k, j])]
                    violations.append((0, gap[k, j], d[k, j], thickness[points[moves][k]], g, -1 - j))

        # triangle meshes, one pair of bodies at a time
        lo = np.full((len(ids), 3), np.inf)
        hi = np.full((len(ids), 3), -np.inf)
        np.minimum.at(lo, point_group, x[points])
        np.maximum.at(hi, point_group, x[points])
        reach = np.zeros(len(ids))
        np.maximum.at(reach, point_group, thickness[points])
        all_still = np.ones(len(ids), bool)
        np.logical_and.at(all_still, point_group, still[points])
        tris = np.asarray(eng.surface["triangles"], np.int64).reshape(-1, 3)
        edges = np.asarray(eng.surface["edges"], np.int64).reshape(-1, 2)
        # the primitives of each body, sorted by body: rows, whether they move, and where each body's rows start
        prims = {}
        for name, rows in (("vertices", points[:, None]), ("triangles", tris), ("edges", edges)):
            body_of = np.searchsorted(ids, group[rows[:, 0]])
            order = np.argsort(body_of, kind="stable")
            prims[name] = (
                rows[order],
                ~still[rows[order]].all(axis=1),
                np.searchsorted(body_of[order], np.arange(len(ids) + 1)),
            )
        for a, b in self._near_boxes(lo, hi, reach, slack):
            if all_still[a] and all_still[b]:
                continue
            result = self._body_pair(prims, group, a, b)
            if result is None:
                warnings.warn(
                    f"SolverIPC does not check whether {self._describe_group(group, points, ids[a])} and "
                    f"{self._describe_group(group, points, ids[b])} start in contact: their surfaces make more "
                    f"primitive pairs than the check tests ({_START_CHECK_PAIRS}).",
                    NewtonWarning,
                    stacklevel=3,
                )
                continue
            found, distance = result
            t = reach[a] + reach[b]
            if found or distance - t <= slack:
                violations.append((found, distance - t, distance, t, a, b))
        if not violations:
            return
        found, gap, distance, t, a, b = min(violations, key=lambda v: (-v[0], v[1]))
        n = len(violations) - 1
        more = f" ({n} more pair{'s' if n > 1 else ''} of bodies {'do' if n > 1 else 'does'} too)" if n else ""
        one = self._describe_group(group, points, ids[a])
        other = f"plane shape {plane_shapes[-1 - b]}" if b < 0 else self._describe_group(group, points, ids[b])
        if found:
            raise ValueError(
                f"SolverIPC: the surfaces of {one} and {other} intersect in the model's state: {found} edges of one "
                f"cross triangles of the other{more}. The barrier needs surfaces that start apart, by more than their "
                "contact thickness; a step would keep them crossed."
            )
        need = f" by more than {t - distance:.6g} m" if t - distance > 0.0 else ""
        fix = f"lift {one} off it{need}" if b < 0 else f"move them apart{need}, for example by lifting the upper one"
        rounding = ", up to the rounding of float32 positions" if distance > t else ""
        raise ValueError(
            f"SolverIPC: {one} and {other} start {distance:.6g} m apart, at or inside their contact thickness "
            f"({t:.6g} m{rounding}){more}. The barrier needs a gap larger than the thickness at the start: {fix}."
        )

    @staticmethod
    def _near_boxes(lo: np.ndarray, hi: np.ndarray, reach: np.ndarray, slack: float) -> np.ndarray:
        """The pairs ``(a, b)``, ``a < b``, of boxes ``[lo, hi]`` that come within ``reach[a] + reach[b] + slack`` of
        each other, by sweep and prune along the axis on which the boxes spread most."""
        axis = int(np.argmax(np.ptp(lo + hi, axis=0)))
        order = np.argsort(lo[:, axis], kind="stable")
        start = lo[order, axis]
        end = np.searchsorted(start, hi[order, axis] + reach[order] + reach.max() + slack, side="right")
        count = np.maximum(end - np.arange(len(order)) - 1, 0)
        first = np.repeat(np.arange(len(order)), count)
        second = first + 1 + np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
        a, b = order[first], order[second]
        apart = np.maximum(lo[b] - hi[a], lo[a] - hi[b]).max(axis=1, initial=-np.inf)
        near = apart <= reach[a] + reach[b] + slack
        a, b = a[near], b[near]
        return np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1)

    def _body_pair(self, prims: dict, body: np.ndarray, a: int, b: int) -> tuple[int, float] | None:
        """The edge-triangle crossings and the smallest distance between the surfaces of bodies ``a`` and ``b``
        (indices into ``prims``), without the primitive pairs whose vertices never move: libuipc-warp's
        ``SurfaceChecker`` on the primitives of ``a`` that move against all of ``b``, and on those of ``a`` that
        never move against those of ``b`` that move. ``body`` (the contact body of every vertex) keeps the distance
        query to pairs across the two. None when they make more than ``_START_CHECK_PAIRS`` primitive pairs."""
        from uipc_warp.validation import SurfaceChecker  # noqa: PLC0415

        def rows(name: str, g: int, moves: bool | None) -> np.ndarray:
            all_rows, moving, starts = prims[name]
            r, m = all_rows[starts[g] : starts[g + 1]], moving[starts[g] : starts[g + 1]]
            return (r if moves is None else r[m == moves]).astype(np.int32)

        halves = [
            ({n: rows(n, a, mine) for n in prims}, {n: rows(n, b, theirs) for n in prims})
            for mine, theirs in ((True, None), (False, True))
        ]
        size = 0
        for one, other in halves:
            v1, t1, e1 = (len(one[n]) for n in ("vertices", "triangles", "edges"))
            v2, t2, e2 = (len(other[n]) for n in ("vertices", "triangles", "edges"))
            size += v1 * t2 + v2 * t1 + e1 * e2 + e1 * t2 + e2 * t1
        if size > _START_CHECK_PAIRS:
            return None
        x, device = self._engine.vm.x, self.model.device
        chunk = 16384  # primitives of one side per launch: every launch stays below 2**31 threads
        found, distance = 0, np.inf
        for one, other in halves:
            for edges, tris in ((one["edges"], other["triangles"]), (other["edges"], one["triangles"])):
                for i in range(0, len(edges), chunk):
                    for j in range(0, len(tris), chunk):
                        e, f = edges[i : i + chunk], tris[j : j + chunk]
                        surface = {"edges": e, "triangles": f, "vertices": np.zeros(0, np.int32)}
                        found += SurfaceChecker(surface, body, device).intersections(x)
            pieces = [max(len(part[n]) for n in prims) for part in (one, other)]
            if min(pieces) == 0:
                continue
            for i in range(0, pieces[0], chunk):
                for j in range(0, pieces[1], chunk):
                    surface = {n: np.concatenate([one[n][i : i + chunk], other[n][j : j + chunk]]) for n in prims}
                    surface["vertices"] = surface["vertices"].reshape(-1)
                    surface["triangles"] = surface["triangles"].reshape(-1, 3)
                    surface["edges"] = surface["edges"].reshape(-1, 2)
                    distance = min(distance, SurfaceChecker(surface, body, device).min_distance(x))
        return found, distance

    def _describe_vertex(self, v: int) -> str:
        """The Newton particle, body or shape that engine vertex ``v`` belongs to."""
        layout = self._scene.layout()
        if v < layout.n_fem:
            return f"particle {v}"
        if v < layout.obstacle_offsets[-1]:
            k = int(np.searchsorted(layout.obstacle_offsets, v, side="right")) - 1
            return f"static shape {self._obstacle_shapes[k]}"
        b = int(np.searchsorted(layout.body_vertex_offsets, v, side="right")) - 1
        label = self.model.body_label[b] if b < len(self.model.body_label) else ""
        return f"body {b}" + (f" ({label!r})" if label and label != f"body_{b}" else "")

    def _describe_group(self, group: np.ndarray, points: np.ndarray, g: int) -> str:
        """The Newton body, static shape or particles of contact body ``g``."""
        first = int(points[group[points] == g].min())
        n_fem = self._scene.layout().n_fem
        if first >= n_fem:
            return self._describe_vertex(first)
        members = np.flatnonzero(group[:n_fem] == g)  # with the particles inside a soft body
        first, last = int(members.min()), int(members.max())
        if len(members) == 1:
            return f"particle {first}"
        if last - first + 1 == len(members):
            return f"particles {first} to {last}"
        return f"{len(members)} particles from particle {first}"

    def _forget_written(self) -> None:
        """Forget what the solver wrote for the bodies, so that the next step takes their state from ``state_in``
        (NaN equals nothing)."""
        nb = self.model.body_count
        if nb:
            self._body_q_written.assign(np.full((nb, 7), np.nan, np.float32))
            self._body_qd_written.assign(np.full((nb, 6), np.nan, np.float32))

    def _make_engine(self, capacity: dict | None) -> None:
        """The engine of the laid-out scene, with the current gravity and masses; ``capacity``: its list
        capacities (``SimEngine``'s argument; None for its defaults)."""
        device = self.model.device
        inputs = self._scene.engine_inputs(self._config, device)
        self._forces = None
        if len(self._force_blocks):
            self._forces = self._external_acceleration("newton_forces", self._force_blocks, device)
            inputs["elements"].append(self._forces)
        self._engine = self._uipc_warp.make_engine(self._config, None, **inputs, device=device, capacity=capacity)
        if self.model.body_count:
            # notify_model_changed may have changed them since the scene was laid out
            self._engine.set_dyadic_mass(*self._dyadic)

    def _add_particles_and_tets(self, scene) -> np.ndarray:
        """Every particle as a finite-element vertex (engine index = particle index), and the tetrahedra."""
        model = self.model
        n = model.particle_count
        self._fixed_particle = np.zeros(n, bool)
        if n == 0:
            return np.zeros((0, 4), np.int64)
        mass = model.particle_mass.numpy().astype(np.float64)
        flags = model.particle_flags.numpy()
        self._fixed_particle = (mass <= 0.0) | ((flags & int(ParticleFlags.ACTIVE)) == 0)
        scene.add_vertices(
            model.particle_q.numpy().astype(np.float64),
            model.particle_qd.numpy().astype(np.float64),
            mass,
            fixed=self._fixed_particle,
            thickness=self._thickness,
        )
        if model.tet_count == 0:
            return np.zeros((0, 4), np.int64)
        tets = model.tet_indices.numpy().reshape(-1, 4).astype(np.int64)
        Dm_inv = model.tet_poses.numpy().astype(np.float64).reshape(-1, 3, 3)
        volume = 1.0 / (6.0 * np.abs(np.linalg.det(Dm_inv)))
        materials = model.tet_materials.numpy().astype(np.float64).reshape(-1, 3)
        scene.add_tetrahedra(tets, Dm_inv, volume, materials[:, 0], materials[:, 1])
        return tets

    def _add_cloth(self, scene, tets: np.ndarray) -> None:
        """Triangles not on a tetrahedral body: Baraff-Witkin membrane and discrete-shell hinges."""
        model = self.model
        if model.tri_count == 0:
            return
        tris = model.tri_indices.numpy().reshape(-1, 3).astype(np.int64)
        on_tet = np.zeros(model.particle_count, bool)
        on_tet[tets.ravel()] = True
        cloth = ~on_tet[tris].all(axis=1)
        if not cloth.any():
            return
        tris = tris[cloth]
        mat = model.tri_materials.numpy().astype(np.float64).reshape(-1, 5)[cloth]
        k_mu, k_lambda = mat[:, 0], mat[:, 1]
        stretch = k_mu + 0.5 * k_lambda
        shear = 0.5 * k_mu

        edges, kappa = np.zeros((0, 2), np.int64), np.zeros(0)
        if model.edge_count:
            x = model.particle_q.numpy().astype(np.float64)
            e = model.edge_indices.numpy().reshape(-1, 4).astype(np.int64)
            ke = model.edge_bending_properties.numpy().astype(np.float64).reshape(-1, 2)[:, 0]
            # hinges: both opposite vertices present, the edge on the cloth
            n = model.particle_count
            cloth_keys = np.unique(
                np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1) @ [n, 1]
            )
            keys = np.sort(e[:, 2:4], axis=1) @ [n, 1]
            sel = (e[:, 0] >= 0) & (e[:, 1] >= 0) & np.isin(keys, cloth_keys)
            e, ke = e[sel], ke[sel]
            x0, x1, x2, x3 = x[e[:, 0]], x[e[:, 1]], x[e[:, 2]], x[e[:, 3]]
            L0 = np.linalg.norm(x3 - x2, axis=1)
            area = 0.5 * (
                np.linalg.norm(np.cross(x2 - x0, x3 - x0), axis=1) + np.linalg.norm(np.cross(x2 - x1, x3 - x1), axis=1)
            )
            h_bar = area / (3.0 * np.maximum(L0, 1.0e-30))
            edges, kappa = e[:, 2:4], 0.5 * ke * h_bar
        scene.add_shell(tris, stretch, stretch, shear, self._cloth_strain_rate, bending=kappa, bending_edges=edges)

    def _body_mass_moments(self, stacklevel: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The dyadic mass ``(m, sum m x_bar, sum m x_bar x_bar^T)`` of every body from Newton's mass properties,
        and which bodies are kinematic (``self._kinematic_body``)."""
        model = self.model
        nb = model.body_count
        self._kinematic_body = np.zeros(nb, bool)
        if nb == 0:
            return np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3, 3))
        body_mass = model.body_mass.numpy().astype(np.float64)
        body_com = model.body_com.numpy().astype(np.float64)
        body_inertia = model.body_inertia.numpy().astype(np.float64)
        body_flags = model.body_flags.numpy()
        self._kinematic_body = ((body_flags & int(BodyFlags.KINEMATIC)) != 0) | (body_mass <= 0.0)
        mass, first, second = np.zeros(nb), np.zeros((nb, 3)), np.zeros((nb, 3, 3))
        for b in range(nb):
            inertia = 0.5 * (body_inertia[b] + body_inertia[b].T)
            if not self._kinematic_body[b] and np.linalg.eigvalsh(inertia).min() <= 0.0:
                raise ValueError(f"SolverIPC needs a positive-definite inertia for the dynamic body {b}.")
            m, mx, S = self._uipc_warp.arrays.dyadic_mass(body_mass[b], body_com[b], inertia)
            # the second moment about the centre of mass must be PSD (the inertia's triangle inequality)
            cov = S - m * np.outer(body_com[b], body_com[b])
            lam, vec = np.linalg.eigh(cov)
            if lam.min() < 0.0:
                warnings.warn(
                    f"SolverIPC: the inertia of body {b} violates the triangle inequality; clamping its mass spread.",
                    NewtonWarning,
                    stacklevel=stacklevel,
                )
                S = (vec * np.maximum(lam, 0.0)) @ vec.T + m * np.outer(body_com[b], body_com[b])
            mass[b], first[b], second[b] = m, mx, S
        return mass, first, second

    def _mass_inverse(self, mass: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        """The inverse of the 4 x 4 block ``[[m, (sum m x_bar)^T], [sum m x_bar, sum m x_bar x_bar^T]]`` of every
        dynamic body's dyadic mass, which maps a force on ``(p_k, row k of A)`` to its acceleration."""
        dynamic = np.flatnonzero(~self._kinematic_body)
        M = np.zeros((len(dynamic), 4, 4))
        M[:, 0, 0] = mass[dynamic]
        M[:, 0, 1:] = first[dynamic]
        M[:, 1:, 0] = first[dynamic]
        M[:, 1:, 1:] = second[dynamic]
        return np.linalg.inv(M) if len(dynamic) else np.zeros((0, 4, 4))

    def _free_joint_dofs(self) -> np.ndarray:
        """The first ``joint_f`` entry of every body's free joint, -1 for a body without one."""
        model = self.model
        start = np.full(model.body_count, -1, np.int32)
        if model.joint_count:
            child = model.joint_child.numpy()
            qd_start = model.joint_qd_start.numpy()
            free = np.flatnonzero((model.joint_type.numpy() == int(JointType.FREE)) & (child >= 0))
            start[child[free]] = qd_start[free]
        return start

    def _shape_geometry_now(self) -> tuple:
        """What the triangulated shapes are made of, to tell a change of geometry from one of materials."""
        model = self.model
        if model.shape_count == 0:
            return ()
        return (model.shape_transform.numpy().copy(), model.shape_scale.numpy().copy())

    def _add_shapes_and_bodies(self, scene) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Static non-plane shapes as fixed triangle meshes; every body as an affine body. Returns the bodies'
        dyadic masses."""
        model = self.model
        meshes: dict[int, list] = {b: [] for b in range(model.body_count)}
        self._obstacle_shapes = []  # the shape of every obstacle, in the order they are added
        if model.shape_count:
            shape_type = model.shape_type.numpy()
            shape_body = model.shape_body.numpy()
            colliding = _colliding_shapes(model)
            shape_xform = model.shape_transform.numpy().astype(np.float64)
            for s in range(model.shape_count):
                if not colliding[s]:
                    continue
                b = int(shape_body[s])
                if shape_type[s] == GeoType.PLANE:
                    if b != -1:
                        raise NotImplementedError(f"SolverIPC does not support the plane shape {s} on a body.")
                    continue
                v, f = _shape_mesh(model, s, self._shape_segments)
                v = _quat_rotate_np(shape_xform[s, 3:7], v) + shape_xform[s, :3]
                if b == -1:
                    scene.add_obstacle(v, f)
                    self._obstacle_shapes.append(s)
                else:
                    meshes[b].append((v, f))

        mass, first, second = self._body_mass_moments(stacklevel=5)
        if model.body_count == 0:
            return mass, first, second
        body_q = model.body_q.numpy().astype(np.float64)
        body_qd = model.body_qd.numpy().astype(np.float64)
        body_com = model.body_com.numpy().astype(np.float64)
        body_mass = model.body_mass.numpy().astype(np.float64)
        for b in range(model.body_count):
            parts = meshes[b]
            offsets = np.cumsum([0] + [len(v) for v, _ in parts])
            xbar = np.concatenate([v for v, _ in parts]) if parts else np.zeros((0, 3))
            tris = (
                np.concatenate([f + o for (_, f), o in zip(parts, offsets[:-1], strict=True)])
                if parts
                else np.zeros((0, 3), np.int64)
            )
            volume = sum(_closed_volume(v, f) for v, f in parts)
            if volume <= 0.0:
                # no closed collision mesh: the volume of water of that mass
                volume = max(body_mass[b], 1.0e-12) / 1000.0
            R = _quat_to_matrix_np(body_q[b, 3:7])
            w = body_qd[b, 3:6]
            W = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
            scene.add_affine_body(
                xbar,
                tris,
                mass=mass[b],
                first_moment=first[b],
                second_moment=second[b],
                volume=volume,
                kappa=self._body_kappa,
                p=body_q[b, :3],
                A=R,
                p_dot=body_qd[b, :3] - np.cross(w, R @ body_com[b]),
                A_dot=W @ R,
                fixed=bool(self._kinematic_body[b]),
            )
        return mass, first, second

    def _collect_planes(self) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """The points, normals and shapes of the static planes that collide."""
        model = self.model
        if model.shape_count == 0:
            return np.zeros((0, 3)), np.zeros((0, 3)), []
        shape_type = model.shape_type.numpy()
        shape_body = model.shape_body.numpy()
        colliding = _colliding_shapes(model)
        shape_xform = model.shape_transform.numpy()
        shape_scale = model.shape_scale.numpy()
        P, N, shapes = [], [], []
        for s in range(model.shape_count):
            if shape_type[s] != int(GeoType.PLANE) or shape_body[s] != -1:
                continue
            if not colliding[s]:
                continue
            if shape_scale[s][0] > 0.0 or shape_scale[s][1] > 0.0:
                warnings.warn(
                    f"SolverIPC treats the finite plane shape {s} as an infinite half space.",
                    NewtonWarning,
                    stacklevel=4,
                )
            xf = shape_xform[s].astype(np.float64)
            P.append(xf[:3])
            N.append(_quat_rotate_np(xf[3:7], np.array([0.0, 0.0, 1.0])))
            shapes.append(s)
        return np.asarray(P).reshape(-1, 3), np.asarray(N).reshape(-1, 3), shapes

    # ------------------------------------------------------------------ SolverBase
    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
        *,
        observables: SolverBase.Observables | None = None,
    ) -> None:
        self.validate_observables(observables)
        if self._engine is None:
            return
        if control is None:
            control = self.model.control(clone_variables=False)
        self._dt = dt
        self._load(state_in)
        self._write_forces(state_in, control)
        if self.model.device.is_capturing:
            # the solve joins the caller's graph; it cannot synchronize, grow or raise (see the class docstring)
            self._engine.stop_on_overflow = False
            self._engine.advance(dt)
        else:
            status = self._advance_eager(dt)
            if status.failed:
                self._write_observables(observables)
                self._forget_written()  # the engine holds the failed solve: the next step takes state_in
                raise RuntimeError(self._failure(status))
        self._store(state_in.body_q, state_in.body_qd, state_out)
        self._write_observables(observables)

    def _load(self, state_in: State) -> None:
        """The engine's state from ``state_in`` where it differs from what the solver wrote (class docstring)."""
        model, eng = self.model, self._engine
        device, vm = model.device, eng.vm
        self._changed.zero_()
        if model.particle_count:
            wp.launch(
                _load_particles,
                model.particle_count,
                inputs=[state_in.particle_q, state_in.particle_qd, vm.fixed],
                outputs=[vm.x, vm.v, self._changed],
                device=device,
            )
        if model.body_count:
            wp.launch(
                _load_bodies,
                model.body_count,
                inputs=[
                    state_in.body_q,
                    state_in.body_qd,
                    self._body_q_written,
                    self._body_qd_written,
                    model.body_com,
                    self._body_base,
                    self._body_kinematic,
                ],
                outputs=[vm.x, vm.v, self._changed],
                device=device,
            )
            self._uipc_warp.arrays.update_body_vertices(eng)  # surface vertices x = A x_bar + p
        eng.discard_friction(where=self._changed)

    def _write_forces(self, state_in: State, control: Control) -> None:
        """The external accelerations of the step, on the device."""
        if self._forces is None:
            return
        model, acc = self.model, self._forces.a
        if self._n_free:
            particle_f = state_in.particle_f
            wp.launch(
                _particle_accelerations,
                self._n_free,
                inputs=[particle_f, int(particle_f is not None), model.particle_mass, self._free_particles],
                outputs=[acc],
                device=model.device,
            )
        if self._n_dynamic:
            body_f, joint_f = state_in.body_f, control.joint_f
            wp.launch(
                _body_accelerations,
                self._n_dynamic,
                inputs=[
                    body_f,
                    int(body_f is not None),
                    joint_f,
                    int(joint_f is not None),
                    self._joint_f_start,
                    self._dynamic_bodies,
                    model.body_com,
                    self._body_mass_inv,
                    self._body_base,
                    self._engine.vm.x,
                    self._n_free,
                ],
                outputs=[acc],
                device=model.device,
            )

    def _store(self, body_q_in, body_qd_in, state_out: State | None) -> None:
        """``state_out`` from the engine's state; without one, only the record of what the solver writes."""
        model, vm = self.model, self._engine.vm
        if model.particle_count and state_out is not None:
            wp.launch(
                _store_particles,
                model.particle_count,
                inputs=[vm.x, vm.v],
                outputs=[state_out.particle_q, state_out.particle_qd],
                device=model.device,
            )
        if model.body_count:
            body_q = self._body_q_written if state_out is None else state_out.body_q
            body_qd = self._body_qd_written if state_out is None else state_out.body_qd
            wp.launch(
                _store_bodies,
                model.body_count,
                inputs=[vm.x, vm.v, self._body_base, model.body_com, self._body_kinematic, body_q_in, body_qd_in],
                outputs=[body_q, body_qd, self._body_q_written, self._body_qd_written],
                device=model.device,
            )

    def _advance_eager(self, dt: float):
        """The solve outside a caller's capture, its status read back; on an overflow the frame again on an engine
        with longer lists (libuipc-warp's FrameRunner, in its order)."""
        status = self._solve_frame(dt)
        for _ in range(_MAX_REGROWS):
            if not status.overflowed:
                break
            self._regrow()
            status = self._solve_frame(dt)
        if not status.overflowed and self._engine.graph_outdated():
            self._frame_graph = None  # the solves ask for the other PCG loop: capture the next frame anew
        return status

    def _solve_frame(self, dt: float):
        """One frame as a graph of its own (captured at the first frame of an engine), and how it ended."""
        eng = self._engine
        eng.set_dt(dt)
        # an overflowing frame is done again, so its Newton loop may end at the overflow
        eng.stop_on_overflow = True
        if self._frame_graph is None:
            gc_enabled = gc.isenabled()
            gc.disable()  # a collection inside the capture would record the release of other arrays
            try:
                with wp.ScopedCapture(device=self.model.device) as capture:
                    eng.advance()
            finally:
                if gc_enabled:
                    gc.enable()
            self._frame_graph = capture.graph
        wp.capture_launch(self._frame_graph)
        return self._uipc_warp.arrays.frame_status(eng)

    def _regrow(self) -> None:
        """A new engine whose contact lists hold what the last frame asked for, at the start of that frame and
        with its external accelerations."""
        old = self._engine
        capacity = old.grown_capacity()
        start = old.frame_start_state()
        forces = None if self._forces is None else self._forces.a
        # release the old engine before the new one allocates
        self._engine = self._frame_graph = self._forces = old = None
        self._make_engine(capacity)
        self._engine.load_state(**start)
        if forces is not None:
            wp.copy(self._forces.a, forces)

    def _failure(self, status) -> str:
        """Why a step failed, with what libuipc would print."""
        if status.overflowed:
            return f"SolverIPC: the contact lists still overflow after {_MAX_REGROWS} enlargements."
        if status.traversal_exhausted:
            return "SolverIPC: a BVH traversal ran out of its node budget, so contact candidates are missing."
        report = self._engine.nonfinite_report()
        name = self.StatusBits(1 << status.nonfinite).name
        detail = report.get("message", "")
        if not detail:
            detail = ", ".join(f"{k}={v}" for k, v in report.items() if k not in ("kind",))
        return (
            f"SolverIPC: the step failed on {name} in Newton iteration {report['newton']}: {detail}. "
            "Surfaces must start apart (outside their contact thickness), and the state must stay finite."
        )

    def _write_observables(self, observables: SolverBase.Observables | None) -> None:
        if observables is None:
            return
        status = observables.status if observables.is_requested(self.ObservableKind.STATUS) else None
        residual = (
            observables.solver_residual if observables.is_requested(SolverBase.ObservableKind.SOLVER_RESIDUAL) else None
        )
        if status is None and residual is None:
            return
        for entry in self._world_entries:
            self._uipc_warp.arrays.write_frame_status(self._engine, status, residual, index=int(entry))

    def grow(self) -> None:
        """Enlarge the contact lists after a captured step reported :attr:`SolverIPC.StatusBits.OVERFLOW`.

        A captured step cannot enlarge them: it drops the pairs that do not fit, sets the bit and keeps it set.
        This builds the solve again with lists that hold what the last step asks for (solving that step again
        until it fits, as an eager step does) and returns to the state the last step started from, so that the
        same ``state_in`` repeats it. The new solve has new arrays: capture every graph that holds :meth:`step`
        again. Synchronizes; does nothing when no list overflowed.
        """
        if self._engine is None:
            return
        if self.model.device.is_capturing:
            raise RuntimeError("SolverIPC.grow() synchronizes with the device; call it outside graph capture.")
        if not self._uipc_warp.arrays.frame_status(self._engine).overflowed:
            return
        for _ in range(_MAX_REGROWS):
            self._regrow()
            if not self._solve_frame(self._dt).overflowed:
                break
        else:
            raise RuntimeError(f"SolverIPC: the contact lists still overflow after {_MAX_REGROWS} enlargements.")
        self._engine.load_state(**self._engine.frame_start_state())
        if self.model.body_count:
            # what the solver wrote for the restored state, so that the step's own state_in is kept as it was
            self._store(self._body_q_written, self._body_qd_written, None)

    def check_status(self, observables: SolverBase.Observables, *, grow: bool = False) -> bool:
        """Raise when a step that wrote ``observables`` failed, and warn when one reached an iteration limit.

        A step captured in a CUDA graph cannot raise: it only writes how it ended into the observables passed to it
        (see **Status** in the class docstring). Call this after replaying the graph, outside the capture, once for
        each container the replayed steps wrote. It reads the container back from the device (one synchronization).

        Args:
            observables: A container from :meth:`observables` with :attr:`SolverIPC.ObservableKind.STATUS`
                requested, as passed to :meth:`step`. With ``SOLVER_RESIDUAL`` requested as well, the messages give
                the residual.
            grow: On :attr:`SolverIPC.StatusBits.OVERFLOW`, call :meth:`grow` and return ``True`` instead of
                raising, whatever other bits are set: the steps ran without some contact pairs, which can cause the
                other failures, and the repeated steps report again.

        Returns:
            ``True`` when a contact list overflowed and ``grow`` enlarged the lists: the steps ran without some
            contact pairs, so restore the state they started from, capture every graph that holds :meth:`step`
            again and repeat them. ``False`` otherwise.

        Raises:
            RuntimeError: When a bit of :attr:`SolverIPC.StatusBits.FAILED` is set and ``grow`` does not take an
                ``OVERFLOW``: the step's result must not be used.
            ValueError: When ``observables`` does not hold the ``STATUS`` observable.

        Warns:
            NewtonWarning: When :attr:`SolverIPC.StatusBits.NEWTON_LIMIT` or
                :attr:`SolverIPC.StatusBits.LINE_SEARCH_LIMIT` is set. Neither fails the step.
        """
        if self.model.device.is_capturing:
            raise RuntimeError("SolverIPC.check_status() reads from the device; call it outside graph capture.")
        if observables is None or not observables.is_requested(self.ObservableKind.STATUS):
            raise ValueError("SolverIPC.check_status() needs observables with ObservableKind.STATUS requested.")
        bits = self.StatusBits(int(np.bitwise_or.reduce(observables.status.numpy())))
        residual = ""
        if observables.is_requested(SolverBase.ObservableKind.SOLVER_RESIDUAL):
            residual = f" (residual {float(observables.solver_residual.numpy().max()):.3g}, at most 1 when converged)"
        if bits & self.StatusBits.OVERFLOW and grow:
            self.grow()
            return True
        failed = bits & self.StatusBits.FAILED
        if failed:
            causes = []
            if failed & self.StatusBits.OVERFLOW:
                causes.append(
                    "A contact list overflowed and the step ran without some contact pairs: call grow() (or "
                    "check_status(..., grow=True)), restore the state the steps started from, capture again and repeat "
                    "them."
                )
            if failed & (self.StatusBits.THICKNESS_VIOLATED | self.StatusBits.INVALID_CCD_STEP):
                causes.append(
                    "A contact pair started at or inside its contact thickness, as surfaces in touching contact or "
                    "intersecting do (in a state written between steps, for example): surfaces must start apart."
                )
            if failed & (
                self.StatusBits.NONFINITE_ENERGY
                | self.StatusBits.NONFINITE_PCG_INIT
                | self.StatusBits.NONFINITE_PCG_ITER
            ):
                causes.append(
                    "The solve met a non-finite value: the state, the forces and the parameters must be finite."
                )
            if failed & self.StatusBits.TRAVERSAL_EXHAUSTED:
                causes.append("A BVH traversal ran out of its node budget, so contact candidates are missing.")
            last = self._uipc_warp.arrays.frame_status(self._engine)
            if last.nonfinite and failed & self.StatusBits(1 << last.nonfinite):
                report = self._engine.nonfinite_report()
                detail = report.get("message") or ", ".join(f"{k}={v}" for k, v in report.items() if k != "kind")
                causes.append(f"The last step failed in Newton iteration {report['newton']}: {detail}.")
            names = "|".join(bit.name for bit in self.StatusBits if bit & failed)
            raise RuntimeError(
                f"SolverIPC: a step failed on {names}{residual}; its result must not be used. " + " ".join(causes)
            )
        limits = []
        if bits & self.StatusBits.NEWTON_LIMIT:
            limits.append(f"the Newton loop stopped at newton_max_iter without converging{residual}")
        if bits & self.StatusBits.LINE_SEARCH_LIMIT:
            limits.append("a line search ran out of its line_search_max_iter trials")
        if limits:
            warnings.warn(
                f"SolverIPC: {' and '.join(limits)}; this does not fail the step.", NewtonWarning, stacklevel=2
            )
        return False

    @override
    def reset(
        self,
        state: State,
        world_mask: wp.array[wp.bool] | None = None,
        flags: StateFlags | int | None = None,
    ) -> None:
        """Make the next step start as the first step of a new solver.

        Besides its float64 state, which the next step compares with ``state_in`` (see the class docstring), the
        solver carries the friction of the last step's contacts from one step to the next; this discards it. The
        contacts themselves are reused only where nothing moved, where they are the ones a new solver finds.
        Launched on the device without synchronization, so it can be captured.

        Args:
            state: Unused; the next step reads its own ``state_in``.
            world_mask: Optional boolean mask of shape ``(world_count + 1,)``. The solver simulates one world
                (with the global entities), so the reset applies when the mask selects a world that owns
                particles or bodies. ``None`` resets.
            flags: Unused.
        """
        del state, flags
        world_mask = self._normalize_reset_world_mask(world_mask)
        if self._engine is None:
            return
        if world_mask is None:
            self._engine.discard_friction()
            return
        wp.launch(
            _any_selected,
            1,
            inputs=[world_mask, self._world_entries_device],
            outputs=[self._selected],
            device=self.model.device,
        )
        self._engine.discard_friction(where=self._selected)

    @override
    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        """Take changed model properties.

        Gravity (``MODEL_PROPERTIES``) and the bodies' mass, centre of mass and inertia
        (``BODY_INERTIAL_PROPERTIES``) are updated in place: graphs captured before take them. A change of the
        triangulated shapes (``SHAPE_PROPERTIES`` with new transforms or scales, ``SHAPE_TYPE_AND_SOURCE``) or of
        which bodies are kinematic (``BODY_PROPERTIES``, ``BODY_INERTIAL_PROPERTIES``) builds the solve again: the
        next step reads ``state_in`` in full, and graphs that hold :meth:`step` must be captured again. Shape
        materials are not read (class docstring). Copies from the host: not inside a graph capture.
        """
        if self.model.device.is_capturing:
            raise RuntimeError("SolverIPC.notify_model_changed() copies from the host; call it outside graph capture.")
        if self._engine is None:  # nothing moves
            return
        model = self.model
        rebuild = bool(flags & ModelFlags.SHAPE_TYPE_AND_SOURCE)
        if flags & ModelFlags.SHAPE_PROPERTIES:
            now = self._shape_geometry_now()
            rebuild |= len(now) != len(self._shape_geometry) or any(
                not np.array_equal(a, b) for a, b in zip(now, self._shape_geometry, strict=True)
            )
        kinematic = self._kinematic_body
        if flags & (ModelFlags.BODY_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES) and model.body_count:
            dyadic = self._body_mass_moments(stacklevel=3)
            rebuild |= not np.array_equal(self._kinematic_body, kinematic)
        if rebuild:
            _check_supported(model)
            self._build_engine()
            return
        if flags & ModelFlags.MODEL_PROPERTIES:
            gravity = model.gravity.numpy().reshape(-1, 3).astype(np.float64)
            if len(gravity) > 1 and not np.allclose(gravity, gravity[0]):
                raise NotImplementedError("SolverIPC supports a single gravity vector for all worlds.")
            self._config.gravity = tuple(gravity[0])
            self._engine.set_gravity(self._gravity_scale[:, None] * gravity[0])
        if flags & ModelFlags.BODY_INERTIAL_PROPERTIES and model.body_count:
            self._dyadic = dyadic
            self._engine.set_dyadic_mass(*dyadic)
            self._body_mass_inv.assign(self._mass_inverse(*dyadic))

    def frame_stats(self) -> dict:
        """Solver counters of the last step: Newton/PCG iterations, line-search trials, convergence.

        Synchronizes with the device.
        """
        return {} if self._engine is None else self._engine.frame_stats()
