# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Incremental Potential Contact (IPC) solver backed by libuipc-warp."""

from __future__ import annotations

import gc
import importlib.metadata
import inspect
import operator
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntFlag
from types import SimpleNamespace
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import CouplingInterface

from newton import (
    MAXVAL,
    Axis,
    BodyFlags,
    Contacts,
    Control,
    GeoType,
    JointTargetMode,
    JointType,
    Mesh,
    Model,
    ModelFlags,
    NewtonWarning,
    ParticleFlags,
    ShapeFlags,
    State,
    StateFlags,
    eval_fk,
)
from newton.math import quat_decompose, velocity_at_point
from newton.solvers import SolverBase

# typing.override from Python 3.12, typing_extensions' on Python 3.11, else no check, as Newton takes it
if TYPE_CHECKING:
    from typing_extensions import override
else:
    try:
        from typing import override
    except ImportError:
        try:
            from typing_extensions import override
        except ImportError:

            def override(method, /):
                return method


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

# libuipc's orthogonality stiffness of an affine body (AffineBodyConstitution's kappa, 100 MPa) [Pa]
_BODY_KAPPA = 1.0e8
# the turn [rad] of a body in one step, about the axis of its largest principal moment of inertia, that the first Newton
# iteration takes in full under the default body_kappa: libuipc's stiffness is lowered where the orthogonality term of
# a step's incremental potential would exceed 1 / (4 _BODY_TURN^2) times that moment (class docstring, **Rigid bodies**)
_BODY_TURN = 0.1

# libuipc's strength ratio of a joint (the default of its joints' apply_to): the default joint_strength_ratio
_JOINT_STRENGTH_RATIO = 100.0
# the overshoot [rad or m] at which libuipc's cubic joint limit pushes back with the force of Newton's linear limit
# spring of stiffness joint_limit_ke; it is stiffer beyond it and softer below
_LIMIT_OVERSHOOT = wp.constant(wp.float64(1.0e-3))
# a joint limit at -MAXVAL or MAXVAL is no limit: Newton stores an infinite one there
_MAXVAL = wp.constant(wp.float32(MAXVAL))
# Newton's default joint_effort_limit, which bounds no drive
_EFFORT_LIMIT = 1.0e6
_EFFORT_BOUND = wp.constant(wp.float64(_EFFORT_LIMIT))
# why libuipc has no joint for a joint type SolverIPC refuses
_UNMAPPED_JOINTS = {
    JointType.DISTANCE: "libuipc has no distance joint; its affine-body joints are revolute, prismatic, fixed and ball "
    "joints",
    JointType.D6: "libuipc has no D6 joint (constitution 28 is only registered); its affine-body joints are revolute, "
    "prismatic, fixed and ball joints",
    JointType.ROD: "libuipc has no constitution of stretch, shear, bend and twist between two affine bodies; its rods "
    "are finite elements (Kirchhoff rod bending, constitution 15) and codimensional affine bodies",
}

# threads per block of the contact rows of CONTACT_F, whose grid strides over the records of the step's contact terms
_CONTACT_ROW_BLOCK = 256


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
    quiet: wp.array[wp.int32],
    subscene: wp.array[wp.int32],
    x: wp.array[wp.vec3d],
    v: wp.array[wp.vec3d],
    changed: wp.array[wp.int32],
):
    """Keep the float64 state where ``state_in`` holds its float32 rounding, take ``state_in`` elsewhere.

    ``changed[s]`` is set where the state of a particle of subscene ``s`` was taken that ``quiet`` does not mark (one
    entry, 0, with one world): a free particle that is no coupled solver's proxy.
    """
    i = wp.tid()
    taken = int(0)
    if _differs(_to32(x[i]), q[i]):
        x[i] = _to64(q[i])
        taken = 1
    if _differs(_to32(v[i]), qd[i]):
        v[i] = _to64(qd[i])
        taken = 1
    if taken != 0 and quiet[i] == 0:
        changed[subscene[i]] = 1


@wp.kernel
def _load_bodies(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    written_q: wp.array[wp.transform],
    written_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    base: wp.array[wp.int32],
    quiet: wp.array[wp.int32],
    hold: wp.array[wp.int32],
    subscene: wp.array[wp.int32],
    x: wp.array[wp.vec3d],
    v: wp.array[wp.vec3d],
    changed: wp.array[wp.int32],
):
    """Keep the affine state of a body whose ``state_in`` pose and twist are what the solver wrote last, or that
    ``hold`` holds (a kinematic body moving toward its aim, ``kinematic_motion="aim"``); give another the rigid state
    of ``state_in``: q = (p, rows of R) and its rate from Newton's COM twist.

    ``changed[s]`` is set where a body of subscene ``s`` that ``quiet`` does not mark took ``state_in``: a dynamic
    body that is no coupled solver's proxy.
    """
    b = wp.tid()
    if hold[b] != 0:
        return
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
    if quiet[b] == 0:
        changed[subscene[b]] = 1


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
def _aim_kinematic_bodies(
    bodies: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    base: wp.array[wp.int32],
    x: wp.array[wp.vec3d],
    dt: wp.float64,
    bdf2: int,
    q_aim: wp.array[wp.vec3d],
    active: wp.array[wp.int32],
    v: wp.array[wp.vec3d],
    x_nm1: wp.array[wp.vec3d],
    v_nm1: wp.array[wp.vec3d],
    hold: wp.array[wp.int32],
):
    """The aim of every kinematic body of the soft transform constraint's rows (``kinematic_motion="aim"``, class
    docstring, **Kinematic bodies**): its ``state_in`` pose advanced by its twist over the step, the centre of mass by
    ``v dt``, the rotation by the exponential of ``omega dt``, as ``q_aim = (p, rows of R)``; the constraint on; the
    body's affine velocity ``(q_aim - q) / dt`` from its state ``q``, so that the step's prediction ``q + v dt`` is the
    aim, and with ``bdf2`` the history of a restart from that velocity (``x_nm1 = q - dt v``, ``v_nm1 = v``), whose
    prediction is the same; and the body held from the next step on, so that it starts where it ends."""
    k = wp.tid()
    b = bodies[k]
    tf = body_q[b]
    tw = body_qd[b]
    R0 = _rotation64(wp.transform_get_rotation(tf))
    c = _to64(body_com[b])
    com = _to64(wp.transform_get_translation(tf)) + R0 * c + _to64(wp.spatial_top(tw)) * dt
    w = _to64(wp.spatial_bottom(tw))
    R = R0
    turn = wp.length(w) * dt
    if turn > wp.float64(0.0):
        R = wp.quat_to_matrix(wp.quat_from_axis_angle(wp.normalize(w), turn)) * R0
    o = base[b]
    for r in range(4):
        aim = com - R * c
        if r > 0:
            aim = wp.vec3d(R[r - 1, 0], R[r - 1, 1], R[r - 1, 2])
        q_aim[4 * k + r] = aim
        rate = (aim - x[o + r]) / dt
        v[o + r] = rate
        if bdf2 != 0:
            x_nm1[o + r] = x[o + r] - dt * rate
            v_nm1[o + r] = rate
    active[k] = 1
    hold[b] = 1


@wp.kernel
def _release_held(
    world_mask: wp.array[wp.bool],
    entry: wp.array[wp.int32],
    bodies: wp.array[wp.int32],
    hold: wp.array[wp.int32],
):
    """A kinematic body of the soft transform constraint (``kinematic_motion="aim"``) whose world ``world_mask`` selects
    takes its ``state_in`` at the next step (``SolverIPC.reset``)."""
    k = wp.tid()
    b = bodies[k]
    if world_mask[entry[b]]:
        hold[b] = 0


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
    joint_wrench: wp.array[wp.spatial_vector],
    dynamic_body: wp.array[wp.int32],
    body_com: wp.array[wp.vec3],
    mass_inv: wp.array[wp.mat44d],
    base: wp.array[wp.int32],
    x: wp.array[wp.vec3d],
    row0: int,
    acc: wp.array[wp.vec3d],
):
    """The accelerations of a dynamic body's q = (p, rows of A) under the wrench ``body_f`` plus the wrench of the
    ``joint_f`` of its free and ball joints (``_joint_wrenches``): M^-1 Q of its generalized force Q, four rows from
    ``row0 + 4 j``."""
    j = wp.tid()
    b = dynamic_body[j]
    wrench = joint_wrench[b]
    if has_body_f != 0:
        wrench = wrench + body_f[b]
    f = _to64(wp.spatial_top(wrench))
    t = _to64(wp.spatial_bottom(wrench))
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
def _joint_wrenches(
    joints: wp.array[wp.int32],
    joint_type: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_X_p: wp.array[wp.transform],
    joint_qd_start: wp.array[wp.int32],
    joint_enabled: wp.array[wp.bool],
    joint_f: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    wrench: wp.array[wp.spatial_vector],
):
    """The ``joint_f`` of a free or ball joint as wrenches at the centres of mass of its bodies: a free joint's is a
    world-frame wrench at the child's centre of mass, a ball joint's a torque in the parent's joint frame; the parent
    takes the reaction (``SolverXPBD``'s ``apply_joint_forces`` for a free joint)."""
    k = wp.tid()
    j = joints[k]
    if not joint_enabled[j]:
        return
    c = joint_child[j]
    p = joint_parent[j]
    s = joint_qd_start[j]
    if joint_type[j] == JointType.FREE:
        f = wp.vec3(joint_f[s], joint_f[s + 1], joint_f[s + 2])
        t = wp.vec3(joint_f[s + 3], joint_f[s + 4], joint_f[s + 5])
        wp.atomic_add(wrench, c, wp.spatial_vector(f, t))
        if p >= 0:
            # the reaction, moved to the parent's centre of mass
            r = wp.transform_point(body_q[c], body_com[c]) - wp.transform_point(body_q[p], body_com[p])
            wp.atomic_sub(wrench, p, wp.spatial_vector(f, t + wp.cross(r, f)))
    else:
        X_wpj = joint_X_p[j]
        if p >= 0:
            X_wpj = body_q[p] * X_wpj
        t = wp.transform_vector(X_wpj, wp.vec3(joint_f[s], joint_f[s + 1], joint_f[s + 2]))
        wp.atomic_add(wrench, c, wp.spatial_vector(wp.vec3(), t))
        if p >= 0:
            wp.atomic_sub(wrench, p, wp.spatial_vector(wp.vec3(), t))


@wp.kernel
def _joint_terms(
    joints: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_enabled: wp.array[wp.bool],
    target_ke: wp.array[wp.float32],
    target_kd: wp.array[wp.float32],
    stiffness: wp.array[wp.float32],
    rest_q: wp.array[wp.float32],
    damping: wp.array[wp.float32],
    has_springs: int,
    limit_lower: wp.array[wp.float32],
    limit_upper: wp.array[wp.float32],
    limit_ke: wp.array[wp.float32],
    target_q: wp.array[wp.float32],
    has_target_q: int,
    target_qd: wp.array[wp.float32],
    has_target_qd: int,
    joint_f: wp.array[wp.float32],
    has_joint_f: int,
    strength: wp.array[wp.float64],
    mass_sum: wp.array[wp.float64],
    current: wp.array[wp.float64],
    q_prev: wp.array[wp.float64],
    bdf2: int,
    restart: wp.array[wp.int32],
    restart_rows: wp.array[wp.int32],
    per_row: int,
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    body_base: wp.array[wp.int32],
    dt: wp.float64,
    drive_scale: wp.float64,
    q_start: wp.array[wp.float64],
    ratio: wp.array[wp.float64],
    drv_on: wp.array[wp.int32],
    drv_ratio: wp.array[wp.float64],
    drv_aim: wp.array[wp.float64],
    lim_on: wp.array[wp.int32],
    lim_lower: wp.array[wp.float64],
    lim_upper: wp.array[wp.float64],
    lim_strength: wp.array[wp.float64],
    force_on: wp.array[wp.int32],
    force: wp.array[wp.float64],
):
    """The terms of a revolute or prismatic joint for the next step, from Newton's model and control (class docstring,
    **Joints**): the strength ratio of its penalty ``strength`` (0 when disabled) for the engine's ``mass_sum``, the
    masses the joint was built with; the drive, passive spring and passive damping as one quadratic potential of the
    step (libuipc's driving term, ``drive_scale`` 1 for a revolute joint, 1/2 for a prismatic one, whose driving
    energy counts its distance twice), with the velocity at the step's end of the step's integrator: BDF1, or with
    ``bdf2`` BDF2 from the coordinate ``q_prev`` at the start of the step before, unless the engine's frame starts the
    history of the joint's bodies with an implicit Euler step: that of every row (``restart[0]``), or with ``per_row``
    that of the rows of a reset or a new state of their world (``restart_rows``, at the base row of the child or of the
    parent); the limit as libuipc's cubic penalty with Newton's limit stiffness at ``_LIMIT_OVERSHOOT``; ``joint_f`` as
    libuipc's external torque or force. ``q_start`` takes the coordinate at the step's start."""
    e = wp.tid()
    j = joints[e]
    d = joint_qd_start[j]
    i = joint_q_start[j]
    on = joint_enabled[j]
    zero = wp.float64(0.0)
    m = mass_sum[e]
    r = zero
    if on and m > zero:
        r = strength[e] / m
    ratio[e] = r
    ke = wp.float64(target_ke[d])
    kd = wp.float64(target_kd[d])
    ks = zero
    c = zero
    q_rest = zero
    if has_springs != 0:
        ks = wp.float64(stiffness[d])
        c = wp.float64(damping[d])
        q_rest = wp.float64(rest_q[i])
    q_target = zero
    if has_target_q != 0:
        q_target = wp.float64(target_q[i])
    qd_target = zero
    if has_target_qd != 0:
        qd_target = wp.float64(target_qd[d])
    q_n = current[e]
    q_start[e] = q_n
    # implicit Euler: ke (q* - q) + kd (qd* - qd) - ks (q - q_rest) - c qd at the step's end, qd = (q - q_n) / dt, is
    # minus the gradient of K/2 (q - q_hat)^2; BDF2: qd = 3/2 (q - (4 q_n - q_nm1) / 3) / dt
    K = ke + ks + (kd + c) / dt
    q_aim = zero
    restarted = restart[0] != 0
    if per_row != 0:
        if restart_rows[body_base[joint_child[j]]] != 0:
            restarted = True
        p = joint_parent[j]
        if p >= 0:
            if restart_rows[body_base[p]] != 0:
                restarted = True
    if bdf2 != 0 and not restarted:
        w = wp.float64(1.5) / dt
        K = ke + ks + (kd + c) * w
        q_bar = (wp.float64(4.0) * q_n - q_prev[e]) / wp.float64(3.0)
        if K > zero:
            q_aim = (ke * q_target + ks * q_rest + kd * w * (q_bar + qd_target / w) + c * w * q_bar) / K
    elif K > zero:
        q_aim = (ke * q_target + ks * q_rest + (kd / dt) * (q_n + dt * qd_target) + (c / dt) * q_n) / K
    drive = on and K > zero and m > zero
    drv_on[e] = wp.where(drive, 1, 0)
    if drive:
        drv_aim[e] = q_aim
        drv_ratio[e] = drive_scale * K * dt * dt / m
    lo = limit_lower[d]
    hi = limit_upper[d]
    limit_k = wp.float64(limit_ke[d])
    limited = on and limit_k > zero and (lo > -_MAXVAL or hi < _MAXVAL)
    lim_on[e] = wp.where(limited, 1, 0)
    if limited:
        # libuipc's penalty s ((x - u) / (u - l))^3 is, at an overshoot delta, the energy
        # dt^2 limit_k delta^3 / (3 delta_r)
        w = wp.float64(hi) - wp.float64(lo)
        w3 = wp.where(w > zero, w * w * w, wp.float64(1.0))
        lim_lower[e] = wp.float64(lo)
        lim_upper[e] = wp.float64(hi)
        lim_strength[e] = dt * dt * limit_k * w3 / (wp.float64(3.0) * _LIMIT_OVERSHOOT)
    value = zero
    if has_joint_f != 0:
        value = wp.float64(joint_f[d])
    force_on[e] = wp.where(on and value != zero, 1, 0)
    force[e] = value


@wp.kernel
def _joint_rates(
    joints: wp.array[wp.int32],
    prismatic: int,
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_qd_start: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    rate: wp.array[wp.float64],
):
    """The rate of the coordinate of every revolute (``prismatic`` 0) or prismatic joint of a group at the step's
    start, from the bodies' twists, as ``_store_joints`` writes ``joint_qd``: the angular velocity difference, or
    the velocity difference of the joint frames' origins, along the axis in the parent's joint frame."""
    e = wp.tid()
    j = joints[e]
    p = joint_parent[j]
    c = joint_child[j]
    X_wpj = joint_X_p[j]
    w_p = wp.vec3()
    if p >= 0:
        X_wpj = body_q[p] * X_wpj
        w_p = wp.spatial_bottom(body_qd[p])
    axis = wp.quat_rotate(wp.transform_get_rotation(X_wpj), joint_axis[joint_qd_start[j]])
    if prismatic != 0:
        x_c = wp.transform_get_translation(body_q[c] * joint_X_c[j])
        v = _joint_point_velocity(c, body_q, body_qd, body_com, x_c) - _joint_point_velocity(
            p, body_q, body_qd, body_com, wp.transform_get_translation(X_wpj)
        )
        rate[e] = wp.float64(wp.dot(v, axis))
    else:
        rate[e] = wp.float64(wp.dot(wp.spatial_bottom(body_qd[c]) - w_p, axis))


@wp.kernel
def _joint_option_terms(
    joints: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_enabled: wp.array[wp.bool],
    target_ke: wp.array[wp.float32],
    target_kd: wp.array[wp.float32],
    stiffness: wp.array[wp.float32],
    rest_q: wp.array[wp.float32],
    damping: wp.array[wp.float32],
    has_springs: int,
    limit_lower: wp.array[wp.float32],
    limit_upper: wp.array[wp.float32],
    limit_ke: wp.array[wp.float32],
    limit_kd: wp.array[wp.float32],
    limit_damping: int,
    joint_armature: wp.array[wp.float32],
    armature: int,
    rate_start: wp.array[wp.float64],
    rate_prev: wp.array[wp.float64],
    effort_limit: wp.array[wp.float32],
    effort: int,
    joint_f: wp.array[wp.float32],
    has_joint_f: int,
    target_q: wp.array[wp.float32],
    has_target_q: int,
    target_qd: wp.array[wp.float32],
    has_target_qd: int,
    mass_sum: wp.array[wp.float64],
    current: wp.array[wp.float64],
    q_prev: wp.array[wp.float64],
    bdf2: int,
    restart: wp.array[wp.int32],
    restart_rows: wp.array[wp.int32],
    per_row: int,
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    body_base: wp.array[wp.int32],
    dt: wp.float64,
    drive_scale: wp.float64,
    drv_on: wp.array[wp.int32],
    drv_ratio: wp.array[wp.float64],
    drv_aim: wp.array[wp.float64],
    force_on: wp.array[wp.int32],
    force: wp.array[wp.float64],
):
    """The driving term of a revolute or prismatic joint with the terms SolverIPC adds on request (class docstring,
    **Joints**), launched after ``_joint_terms`` and in place of its driving term: the drive, passive spring and
    passive damping as ``_joint_terms`` computes them (the same expressions, so that terms of 0 leave its values bit
    for bit), and with ``limit_damping`` the damping ``-joint_limit_kd qd`` of a joint whose coordinate is past one
    of its limits at the step's start, a passive damping of the step's velocity (BDF1 or BDF2, as ``_joint_terms``
    takes it) folded into the same quadratic potential; with ``armature`` the inertia of the coordinate, the
    step's kinetic term ``a/2 |q - q~|^2 / dt^2`` (BDF2: 9/4 of it) of ``joint_armature`` a with the integrator's
    prediction q~ from the coordinate and its rate ``rate_start`` at the step's start (``_joint_rates``; BDF2 also
    the step before's, ``q_prev`` and ``rate_prev``), as libuipc writes the inertia of every degree of freedom; with
    ``effort`` the bound ``joint_effort_limit`` L (below Newton's default 1e6) of the drive's force, decided at the
    step's start: where ``ke (q* - q_n) + kd (qd* - qd_n)`` exceeds L in magnitude, the step takes the constant
    force of that sign and magnitude L as libuipc's joint torque or force (``force``, with ``joint_f``) in place of
    the drive's quadratic, whose passive spring and damping remain."""
    e = wp.tid()
    j = joints[e]
    d = joint_qd_start[j]
    i = joint_q_start[j]
    on = joint_enabled[j]
    zero = wp.float64(0.0)
    m = mass_sum[e]
    ke = wp.float64(target_ke[d])
    kd = wp.float64(target_kd[d])
    ks = zero
    c = zero
    q_rest = zero
    if has_springs != 0:
        ks = wp.float64(stiffness[d])
        c = wp.float64(damping[d])
        q_rest = wp.float64(rest_q[i])
    q_target = zero
    if has_target_q != 0:
        q_target = wp.float64(target_q[i])
    qd_target = zero
    if has_target_qd != 0:
        qd_target = wp.float64(target_qd[d])
    q_n = current[e]
    restarted = restart[0] != 0
    if per_row != 0:
        if restart_rows[body_base[joint_child[j]]] != 0:
            restarted = True
        p = joint_parent[j]
        if p >= 0:
            if restart_rows[body_base[p]] != 0:
                restarted = True
    second = bdf2 != 0 and not restarted
    # _joint_terms' potential K/2 (q - N/K)^2
    K = ke + ks + (kd + c) / dt
    N = zero
    w = wp.float64(1.5) / dt
    q_bar = q_n
    if second:
        K = ke + ks + (kd + c) * w
        q_bar = (wp.float64(4.0) * q_n - q_prev[e]) / wp.float64(3.0)
        N = ke * q_target + ks * q_rest + kd * w * (q_bar + qd_target / w) + c * w * q_bar
    else:
        N = ke * q_target + ks * q_rest + (kd / dt) * (q_n + dt * qd_target) + (c / dt) * q_n
    saturated = False
    bound = zero
    if effort != 0 and (ke != zero or kd != zero):
        L = wp.float64(effort_limit[d])
        if L < _EFFORT_BOUND:
            tau = ke * (q_target - q_n) + kd * (qd_target - rate_start[e])
            if wp.abs(tau) > L:
                # the drive's force at the step's start exceeds its bound: the bound, constant over the step, in place
                # of the drive, whose passive spring and damping remain
                saturated = True
                bound = wp.where(tau > zero, L, -L)
                if second:
                    K = ks + c * w
                    N = ks * q_rest + c * w * q_bar
                else:
                    K = ks + c / dt
                    N = ks * q_rest + (c / dt) * q_n
    K_add = zero
    N_add = zero
    if limit_damping != 0:
        # Featherstone's and SemiImplicit's -limit_kd qd while the coordinate is past a limit, decided at the step's
        # start, at the step's velocity: BDF1 (q - q_n) / dt, BDF2 w (q - q_bar)
        lo = limit_lower[d]
        hi = limit_upper[d]
        limit_k = wp.float64(limit_ke[d])
        limited = on and limit_k > zero and (lo > -_MAXVAL or hi < _MAXVAL)
        if limited and (q_n < wp.float64(lo) or q_n > wp.float64(hi)):
            kl = wp.float64(limit_kd[d])
            if second:
                K_add += kl * w
                N_add += kl * w * q_bar
            else:
                K_add += kl / dt
                N_add += (kl / dt) * q_n
    if armature != 0:
        a = wp.float64(joint_armature[d])
        if a > zero:
            # implicit Euler: q~ = q_n + dt qd_n; BDF2: q~ = q_bar + 2/9 dt (4 qd_n - qd_(n-1)), weighed 9/4
            if second:
                ka = wp.float64(2.25) * a / (dt * dt)
                K_add += ka
                N_add += ka * (
                    q_bar + wp.float64(2.0) * dt / wp.float64(9.0) * (wp.float64(4.0) * rate_start[e] - rate_prev[e])
                )
            else:
                ka = a / (dt * dt)
                K_add += ka
                N_add += ka * (q_n + dt * rate_start[e])
    K = K + K_add
    drive = on and K > zero and m > zero
    drv_on[e] = wp.where(drive, 1, 0)
    if drive:
        drv_aim[e] = (N + N_add) / K
        drv_ratio[e] = drive_scale * K * dt * dt / m
    if saturated:
        value = bound
        if has_joint_f != 0:
            value += wp.float64(joint_f[d])
        force_on[e] = wp.where(on and value != zero, 1, 0)
        force[e] = value


@wp.kernel
def _enable_anchor_joints(
    joints: wp.array[wp.int32],
    joint_enabled: wp.array[wp.bool],
    strength: wp.array[wp.float64],
    kappa: wp.array[wp.float64],
):
    """The stiffness of a fixed or ball joint for the next step, 0 when it is disabled."""
    e = wp.tid()
    kappa[e] = wp.where(joint_enabled[joints[e]], strength[e], wp.float64(0.0))


@wp.kernel
def _stitch_stiffness(
    source: wp.array[wp.int32],
    stiffness: wp.array[wp.float32],
    enabled: wp.array[wp.bool],
    hard: wp.array[wp.float64],
    spring_stiffness: wp.array[wp.float32],
    inv_dt2: wp.float64,
    kappa: wp.array[wp.float64],
):
    """The stiffness [N/m] of every row of a stitch group for the next step (class docstring, **Attachments**):
    ``source`` holds the row's attachment, or -1 less its spring. An attachment takes its stiffness, 0 when it is
    disabled, and a hard one (infinite stiffness) the penalty ``hard`` of a ball joint over ``dt^2``; a spring of
    rest length 0 takes its ``spring_stiffness``."""
    r = wp.tid()
    s = source[r]
    k = wp.float64(0.0)
    if s >= 0:
        if enabled[s]:
            ks = stiffness[s]
            if wp.isinf(ks):
                k = hard[s] * inv_dt2
            else:
                k = wp.float64(ks)
    else:
        k = wp.float64(spring_stiffness[-1 - s])
    kappa[r] = k


@wp.kernel
def _distance_stiffness(
    source: wp.array[wp.int32],
    joints: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_enabled: wp.array[wp.bool],
    joint_limit_lower: wp.array[wp.float32],
    hard: wp.array[wp.float64],
    inv_dt2: wp.float64,
    kappa: wp.array[wp.float64],
    rest_length: wp.array[wp.float64],
):
    """The stiffness [N/m] and rest length of every row of a stitch group of distance joints for the next step
    (class docstring, **Distance joints**): ``source`` holds the row's distance joint among ``joints``, whose penalty
    ``hard`` is a ball joint's, over ``dt^2``, 0 when the joint is disabled; its rest length the joint's distance,
    the bounds of its first degree of freedom."""
    r = wp.tid()
    k = source[r]
    j = joints[k]
    kappa[r] = wp.where(joint_enabled[j], hard[k] * inv_dt2, wp.float64(0.0))
    rest_length[r] = wp.float64(joint_limit_lower[joint_qd_start[j]])


@wp.kernel
def _adopt_joint_angles(
    joints: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_q: wp.array[wp.float32],
    written: wp.array[wp.float32],
    current: wp.array[wp.float64],
):
    """A revolute joint whose ``state_in.joint_q`` is not what the solver wrote takes it as the angle the next unwrap
    starts from (``compute_current_angles``), which therefore keeps its turn."""
    e = wp.tid()
    i = joint_q_start[joints[e]]
    q = joint_q[i]
    if q != written[i]:
        current[e] = wp.float64(q)


@wp.func
def _joint_point_velocity(
    b: int,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    point: wp.vec3,
):
    """The velocity of a point fixed to body ``b`` (-1: the world, at rest)."""
    if b < 0:
        return wp.vec3()
    return velocity_at_point(body_qd[b], point - wp.transform_point(body_q[b], body_com[b]))


@wp.kernel
def _store_joints(
    joint_type: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    element: wp.array[wp.int32],
    revolute_angle: wp.array[wp.float64],
    prismatic_distance: wp.array[wp.float64],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    written: wp.array[wp.float32],
):
    """``joint_q`` and ``joint_qd`` of every joint from the bodies' state, as ``newton.eval_ik`` computes them, but the
    position of a revolute or prismatic joint, which is libuipc's multi-turn angle or distance (``element``: the
    joint's row in its group); ``written`` keeps the positions."""
    j = wp.tid()
    t = joint_type[j]
    if t == JointType.FIXED:
        return
    p = joint_parent[j]
    c = joint_child[j]
    i = joint_q_start[j]
    d = joint_qd_start[j]
    X_wpj = joint_X_p[j]
    w_p = wp.vec3()
    if p >= 0:
        X_wpj = body_q[p] * X_wpj
        w_p = wp.spatial_bottom(body_qd[p])
    X_wcj = body_q[c] * joint_X_c[j]
    x_p = wp.transform_get_translation(X_wpj)
    x_c = wp.transform_get_translation(X_wcj)
    q_p = wp.transform_get_rotation(X_wpj)
    q_c = wp.transform_get_rotation(X_wcj)
    w_err = wp.spatial_bottom(body_qd[c]) - w_p
    if t == JointType.REVOLUTE:
        joint_q[i] = wp.float32(revolute_angle[element[j]])
        joint_qd[d] = wp.dot(w_err, wp.quat_rotate(q_p, joint_axis[d]))
    elif t == JointType.PRISMATIC:
        joint_q[i] = wp.float32(prismatic_distance[element[j]])
        v_err = _joint_point_velocity(c, body_q, body_qd, body_com, x_c) - _joint_point_velocity(
            p, body_q, body_qd, body_com, x_p
        )
        joint_qd[d] = wp.dot(v_err, wp.quat_rotate(q_p, joint_axis[d]))
    elif t == JointType.BALL:
        q_pc = wp.quat_inverse(q_p) * q_c
        for k in range(4):
            joint_q[i + k] = q_pc[k]
        w = wp.quat_rotate_inv(q_p, w_err)
        for k in range(3):
            joint_qd[d + k] = w[k]
    else:  # FREE: the child's joint frame relative to the parent's, its centre of mass's velocity, in that frame
        q_pc = wp.quat_inverse(q_p) * q_c
        x_err = wp.quat_rotate_inv(q_p, x_c - x_p)
        com = wp.transform_point(body_q[c], body_com[c])
        v_err = wp.quat_rotate_inv(
            q_p, wp.spatial_top(body_qd[c]) - _joint_point_velocity(p, body_q, body_qd, body_com, com)
        )
        w = wp.quat_rotate_inv(q_p, w_err)
        for k in range(3):
            joint_q[i + k] = x_err[k]
            joint_qd[d + k] = v_err[k]
            joint_qd[d + 3 + k] = w[k]
        for k in range(4):
            joint_q[i + 3 + k] = q_pc[k]
    for k in range(joint_q_start[j + 1] - i):
        written[i + k] = joint_q[i + k]


@wp.func
def _d6_angular_axes(axis_0: wp.vec3, axis_1: wp.vec3, axis_2: wp.vec3, q0: float, q1: float):
    """The three angular axes of a D6 joint at its coordinates ``(q0, q1)``, in its parent joint frame: ``axis_0`` and
    each later axis carried by the rotations before it (Newton's ``transform_3d_rotational_axes`` in
    ``newton._src.sim.articulation``, which has no public name; copied unchanged)."""
    q_0 = wp.quat_from_axis_angle(axis_0, q0)

    axis_1_w = wp.quat_rotate(q_0, axis_1)
    q_1 = wp.quat_from_axis_angle(axis_1_w, q1)

    axis_2_w = wp.quat_rotate(q_1 * q_0, axis_2)

    return axis_0, axis_1_w, axis_2_w


@wp.func
def _d6_angular_coordinates(
    axis_0: wp.vec3, axis_1: wp.vec3, axis_2: wp.vec3, q_p: wp.quat, q_c: wp.quat, w_err: wp.vec3
):
    """The three angular coordinates and rates of a D6 joint from its frames' rotations ``q_p``, ``q_c`` and the
    bodies' angular velocity difference ``w_err``, as ``newton.eval_ik`` computes them (Newton's
    ``invert_3d_rotational_dofs`` in ``newton._src.sim.articulation``, which has no public name; copied unchanged):
    the XYZ Euler angles of the relative rotation in the right-handed basis ``(axis_0, axis_1, axis_0 x axis_1)``,
    the third negated for a left-handed triple."""
    axis_2_rh = wp.cross(axis_0, axis_1)
    s = float(1.0)
    if wp.dot(axis_2_rh, axis_2) < 0.0:
        s = float(-1.0)

    q_off = wp.quat_from_matrix(wp.matrix_from_cols(axis_0, axis_1, axis_2_rh))
    q_pc = wp.quat_inverse(q_off) * wp.quat_inverse(q_p) * q_c * q_off

    # decompose to a compound rotation each axis in the canonical right-handed basis
    angles = quat_decompose(q_pc)

    # find rotation axes (canonical right-handed basis)
    local_0 = wp.quat_rotate(q_off, wp.vec3(1.0, 0.0, 0.0))
    local_1 = wp.quat_rotate(q_off, wp.vec3(0.0, 1.0, 0.0))
    local_2 = wp.quat_rotate(q_off, wp.vec3(0.0, 0.0, 1.0))

    a0 = local_0
    q_0 = wp.quat_from_axis_angle(a0, angles[0])

    a1 = wp.quat_rotate(q_0, local_1)
    q_1 = wp.quat_from_axis_angle(a1, angles[1])

    a2 = wp.quat_rotate(q_1 * q_0, local_2)

    # convert angular velocity to local space
    w_err_p = wp.quat_rotate_inv(q_p, w_err)

    # given joint axes and angular velocity error, solve for joint velocities in the canonical basis
    c12 = wp.cross(a1, a2)
    c02 = wp.cross(a0, a2)
    c01 = wp.cross(a0, a1)

    velocities = wp.vec3(
        wp.dot(w_err_p, c12) / wp.dot(a0, c12),
        wp.dot(w_err_p, c02) / wp.dot(a1, c02),
        wp.dot(w_err_p, c01) / wp.dot(a2, c01),
    )

    # Map canonical-basis result back to the user's axes: rotation/velocity around axis_2 equals
    # ``s`` times the same quantity around axis_2_rh (s = -1 iff the user's triple is left-handed).
    return wp.vec3(angles[0], angles[1], s * angles[2]), wp.vec3(velocities[0], velocities[1], s * velocities[2])


@wp.kernel
def _store_d6_joints(
    joints: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    written: wp.array[wp.float32],
):
    """``joint_q`` and ``joint_qd`` of a D6 joint of three angular axes, with or without three linear ones, from the
    bodies' state, as ``newton.eval_ik`` computes them: the projections of the joint frames' offset and of their
    velocity difference on the linear axes, in the parent joint frame, then Newton's angular coordinates
    (``_d6_angular_coordinates``); ``written`` keeps the positions."""
    k = wp.tid()
    j = joints[k]
    p = joint_parent[j]
    c = joint_child[j]
    i = joint_q_start[j]
    d = joint_qd_start[j]
    lin = joint_dof_dim[j, 0]
    X_wpj = joint_X_p[j]
    w_p = wp.vec3()
    if p >= 0:
        X_wpj = body_q[p] * X_wpj
        w_p = wp.spatial_bottom(body_qd[p])
    X_wcj = body_q[c] * joint_X_c[j]
    x_p = wp.transform_get_translation(X_wpj)
    x_c = wp.transform_get_translation(X_wcj)
    q_p = wp.transform_get_rotation(X_wpj)
    q_c = wp.transform_get_rotation(X_wcj)
    if lin > 0:
        x_err = wp.quat_rotate_inv(q_p, x_c - x_p)
        v_err = wp.quat_rotate_inv(
            q_p,
            _joint_point_velocity(c, body_q, body_qd, body_com, x_c)
            - _joint_point_velocity(p, body_q, body_qd, body_com, x_p),
        )
        for a in range(lin):
            joint_q[i + a] = wp.dot(x_err, joint_axis[d + a])
            joint_qd[d + a] = wp.dot(v_err, joint_axis[d + a])
    w_err = wp.spatial_bottom(body_qd[c]) - w_p
    coords, rates = _d6_angular_coordinates(
        joint_axis[d + lin], joint_axis[d + lin + 1], joint_axis[d + lin + 2], q_p, q_c, w_err
    )
    for a in range(3):
        joint_q[i + lin + a] = coords[a]
        joint_qd[d + lin + a] = rates[a]
    for a in range(joint_q_start[j + 1] - i):
        written[i + a] = joint_q[i + a]


@wp.kernel
def _d6_joint_wrenches(
    joints: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_enabled: wp.array[wp.bool],
    joint_f: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    wrench: wp.array[wp.spatial_vector],
):
    """The ``joint_f`` of a D6 joint of three angular axes, with or without three linear ones, as wrenches at the
    centres of mass of its bodies: Featherstone's generalized forces, a force along the linear axes (in the parent
    joint frame) on the child's joint frame origin, and the torque ``sum_k tau_k c_k`` of the angular coordinates,
    ``c_k`` the dual basis of the angular axes that ``_d6_angular_axes`` carries to the bodies' pose (the covectors of
    Newton's ``eval_angular_coordinates``, which map the angular velocity difference to the coordinates' rates). The
    parent takes the reaction on the same line."""
    k = wp.tid()
    j = joints[k]
    if not joint_enabled[j]:
        return
    p = joint_parent[j]
    c = joint_child[j]
    d = joint_qd_start[j]
    lin = joint_dof_dim[j, 0]
    X_wpj = joint_X_p[j]
    if p >= 0:
        X_wpj = body_q[p] * X_wpj
    X_wcj = body_q[c] * joint_X_c[j]
    q_p = wp.transform_get_rotation(X_wpj)
    x_c = wp.transform_get_translation(X_wcj)
    f = wp.vec3()
    for a in range(lin):
        f += joint_f[d + a] * wp.quat_rotate(q_p, joint_axis[d + a])
    axis_0 = joint_axis[d + lin]
    axis_1 = joint_axis[d + lin + 1]
    axis_2 = joint_axis[d + lin + 2]
    coords, _rates = _d6_angular_coordinates(
        axis_0, axis_1, axis_2, q_p, wp.transform_get_rotation(X_wcj), wp.vec3(0.0)
    )
    a0, a1, a2 = _d6_angular_axes(axis_0, axis_1, axis_2, coords[0], coords[1])
    a0 = wp.quat_rotate(q_p, a0)
    a1 = wp.quat_rotate(q_p, a1)
    a2 = wp.quat_rotate(q_p, a2)
    # the dual basis of the carried axes, bounded near gimbal lock as Newton bounds it
    det = wp.dot(a0, wp.cross(a1, a2))
    inv_det = 1.0 / (wp.sign(det) * wp.max(wp.abs(det), 1.0e-4))
    t = (
        joint_f[d + lin] * wp.cross(a1, a2)
        + joint_f[d + lin + 1] * wp.cross(a2, a0)
        + joint_f[d + lin + 2] * wp.cross(a0, a1)
    ) * inv_det
    arm_c = x_c - wp.transform_point(body_q[c], body_com[c])
    wp.atomic_add(wrench, c, wp.spatial_vector(f, t + wp.cross(arm_c, f)))
    if p >= 0:
        arm_p = x_c - wp.transform_point(body_q[p], body_com[p])
        wp.atomic_sub(wrench, p, wp.spatial_vector(f, t + wp.cross(arm_p, f)))


@wp.kernel
def _copy_status(
    entries: wp.array[wp.int32],
    status_in: wp.array[wp.int32],
    residual_in: wp.array[wp.float32],
    write_status: int,
    write_residual: int,
    status: wp.array[wp.int32],
    residual: wp.array[wp.float32],
):
    """The status bits and the residual of the one solve, written into every world entry that owns content."""
    k = wp.tid()
    if write_status != 0:
        status[entries[k]] = status_in[0]
    if write_residual != 0:
        residual[entries[k]] = residual_in[0]


@wp.kernel
def _keep_status(step: wp.array[wp.int32], kept: wp.array[wp.int32]):
    """The status bits of a step joined into those kept since ``check_status()`` last read them (an entry of a coupled
    solver, which steps the solver without observables): one word, or one per world entry with independent worlds."""
    k = wp.tid()
    kept[k] = kept[k] | step[k]


@wp.func
def _term_sides(kind: int):
    """(vertices of the first primitive, vertices of the term) of a contact term of libuipc-warp's ``ContactForces``
    kind PT, EE, PE, PP or PH (0 ... 4): the point, the first edge or the vertex comes first; a half-plane term's
    second side is its half plane."""
    first = int(1)
    count = int(4)
    if kind == 1:
        first = 2
    elif kind == 2:
        count = 3
    elif kind == 3:
        count = 2
    elif kind == 4:
        count = 1
    return first, count


@wp.func
def _record_force(forces: wp.array2d[wp.vec3], r: int, k: int) -> wp.vec3d:
    f = forces[r, k]
    return wp.vec3d(wp.float64(f[0]), wp.float64(f[1]), wp.float64(f[2]))


@wp.func
def _side_force(forces: wp.array2d[wp.vec3], r: int, k0: int, k1: int) -> wp.vec3d:
    """The force on the vertices ``k0 .. k1 - 1`` of record ``r`` together."""
    f = wp.vec3d()
    for k in range(k0, k1):
        f += _record_force(forces, r, k)
    return f


@wp.func
def _side_point(x: wp.array[wp.vec3d], q: wp.vec4i, forces: wp.array2d[wp.vec3], r: int, k0: int, k1: int) -> wp.vec3d:
    """The force-weighted mean of the vertices ``k0 .. k1 - 1`` of record ``r``: the closest point of their primitive
    where the force splits by the barycentric weights of that point, as a barrier's and a friction term's do (their
    plain mean where no force acts on them)."""
    p = wp.vec3d()
    mean = wp.vec3d()
    w = wp.float64(0.0)
    for k in range(k0, k1):
        a = wp.float64(wp.length(forces[r, k]))
        p += a * x[q[k]]
        mean += x[q[k]]
        w += a
    if w > wp.float64(0.0):
        return p / w
    return mean / wp.float64(k1 - k0)


@wp.func
def _side_torque(
    x: wp.array[wp.vec3d], q: wp.vec4i, forces: wp.array2d[wp.vec3], r: int, k0: int, k1: int, c: wp.vec3d
) -> wp.vec3d:
    """The torque about ``c`` of the forces on the vertices ``k0 .. k1 - 1`` of record ``r``, each at its vertex: exact
    also where a force does not split by barycentric weights (the mollified edge-edge barrier)."""
    t = wp.vec3d()
    for k in range(k0, k1):
        t += wp.cross(x[q[k]] - c, _record_force(forces, r, k))
    return t


@wp.func
def _body_frame(shape: int, shape_body: wp.array[wp.int32], body_q: wp.array[wp.transform]) -> wp.transform:
    """The frame of ``shape``'s body in ``body_q``, the world's for a static shape."""
    b = shape_body[shape]
    X = wp.transform_identity()
    if b >= 0:
        X = body_q[b]
    return X


@wp.func
def _body_centre(
    shape: int, shape_body: wp.array[wp.int32], body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3]
) -> wp.vec3d:
    """The centre of mass of ``shape``'s body at the pose ``body_q``, the world origin for a static shape."""
    b = shape_body[shape]
    c = wp.vec3d()
    if b >= 0:
        p = wp.transform_point(body_q[b], body_com[b])
        c = wp.vec3d(wp.float64(p[0]), wp.float64(p[1]), wp.float64(p[2]))
    return c


@wp.func
def _in_frame(X: wp.transform, p: wp.vec3d) -> wp.vec3:
    """World point ``p`` in the frame ``X``, its offset from the frame's origin taken in float64."""
    o = wp.transform_get_translation(X)
    d = p - wp.vec3d(wp.float64(o[0]), wp.float64(o[1]), wp.float64(o[2]))
    return wp.quat_rotate_inv(
        wp.transform_get_rotation(X), wp.vec3(wp.float32(d[0]), wp.float32(d[1]), wp.float32(d[2]))
    )


@wp.func
def _to32v(a: wp.vec3d) -> wp.vec3:
    return wp.vec3(wp.float32(a[0]), wp.float32(a[1]), wp.float32(a[2]))


@wp.func
def _row_normal(barrier: bool, push: wp.vec3d, d: wp.vec3d) -> wp.vec3:
    """The normal of a row from its first side toward its second: against ``push``, the force on the first side, for
    a barrier term, whose force is the distance's gradient (closest points of two parallel edges, which the mollified
    edge-edge barrier takes, are not unique: their difference can lie in the plane of contact); along ``d``, from the
    first side's point to the second's, for a friction term, whose force is tangential. The other where one has no
    length."""
    n = -push
    if not barrier or wp.length(n) == wp.float64(0.0):
        n = d
        if wp.length(n) == wp.float64(0.0):
            n = -push
    if wp.length(n) > wp.float64(0.0):
        n = wp.normalize(n)
    return _to32v(n)


@wp.kernel
def _contact_rows(
    rec_vertices: wp.array[wp.vec4i],
    rec_forces: wp.array2d[wp.vec3],
    rec_counts: wp.array[wp.int32],
    rec_offsets: wp.array[wp.int32],
    nthreads: int,
    x: wp.array[wp.vec3d],
    n_particles: int,
    vertex_shape: wp.array[wp.int32],
    plane_shape: wp.array[wp.int32],
    plane_P: wp.array[wp.vec3d],
    plane_N: wp.array[wp.vec3d],
    shape_body: wp.array[wp.int32],
    shape_margin: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    rigid_max: int,
    soft_max: int,
    rigid_count: wp.array[wp.int32],
    rigid_shape0: wp.array[wp.int32],
    rigid_shape1: wp.array[wp.int32],
    rigid_point0: wp.array[wp.vec3],
    rigid_point1: wp.array[wp.vec3],
    rigid_normal: wp.array[wp.vec3],
    rigid_margin0: wp.array[wp.float32],
    rigid_margin1: wp.array[wp.float32],
    rigid_offset0: wp.array[wp.vec3],
    rigid_offset1: wp.array[wp.vec3],
    rigid_surface_velocity: wp.array[wp.vec3],
    soft_count: wp.array[wp.int32],
    soft_shape: wp.array[wp.int32],
    soft_indices: wp.array[wp.vec3i],
    soft_barycentric: wp.array[wp.vec3],
    soft_particle: wp.array[wp.int32],
    soft_body_pos: wp.array[wp.vec3],
    soft_body_vel: wp.array[wp.vec3],
    soft_normal: wp.array[wp.vec3],
    soft_rigid_indices: wp.array[wp.vec3i],
    contact_f: wp.array[wp.spatial_vector],
):
    """One row of ``Contacts`` per contact term with a force, from libuipc-warp's ``ContactForces`` records (class
    docstring, **Contact output**): a rigid row for a term between two shapes, a soft row for a term between a
    particle feature and a shape, none between particle features. Grid-stride over the live records; a row takes its
    index from the rigid or the soft count, which counts every row, and is written only below its capacity."""
    tid = wp.tid()
    total = int(0)
    for k in range(10):
        total += rec_counts[k]
    for idx in range(tid, total, nthreads):
        seg = int(0)
        i = idx
        while seg < 9 and i >= rec_counts[seg]:
            i -= rec_counts[seg]
            seg += 1
        r = rec_offsets[seg] + i
        kind = seg % 5
        first, count = _term_sides(kind)
        q = rec_vertices[r]
        # the owner of each side: a shape, or a particle feature (cloth, tetrahedra, free particles)
        shape_a = vertex_shape[q[0]]
        particle_a = q[0] < n_particles
        shape_b = int(-1)
        particle_b = bool(False)
        if kind == 4:
            shape_b = plane_shape[q[1]]
        else:
            shape_b = vertex_shape[q[first]]
            particle_b = q[first] < n_particles
        if (shape_a < 0 and not particle_a) or (shape_b < 0 and not particle_b):
            continue  # a vertex of no shape and no particle: not a contact primitive
        if shape_a < 0 and shape_b < 0:
            continue  # between particle features: not exported
        f_a = _side_force(rec_forces, r, 0, first)
        f_b = wp.vec3d()
        if kind == 4:
            f_b = _record_force(rec_forces, r, 1)
        else:
            f_b = _side_force(rec_forces, r, first, count)
        if wp.length(f_a) == wp.float64(0.0) and wp.length(f_b) == wp.float64(0.0):
            continue  # no force
        p_a = _side_point(x, q, rec_forces, r, 0, first)
        p_b = wp.vec3d()
        if kind == 4:
            # the point of the half plane under the vertex, on the plane's raw surface: the engine's half plane is
            # moved out by the shape's margin
            n_p = plane_N[q[1]]
            raw = plane_P[q[1]] - wp.float64(shape_margin[shape_b]) * n_p
            p_b = p_a - wp.dot(p_a - raw, n_p) * n_p
        else:
            p_b = _side_point(x, q, rec_forces, r, first, count)
        if shape_a >= 0 and shape_b >= 0:
            row = wp.atomic_add(rigid_count, 0, 1)
            if row < rigid_max:
                X0 = _body_frame(shape_a, shape_body, body_q)
                X1 = _body_frame(shape_b, shape_body, body_q)
                n = _row_normal(seg < 5, f_a, p_b - p_a)
                m0 = shape_margin[shape_a]
                m1 = shape_margin[shape_b]
                rigid_shape0[row] = shape_a
                rigid_shape1[row] = shape_b
                rigid_point0[row] = _in_frame(X0, p_a)
                rigid_point1[row] = _in_frame(X1, p_b)
                rigid_normal[row] = n
                rigid_margin0[row] = m0
                rigid_margin1[row] = m1
                rigid_offset0[row] = wp.quat_rotate_inv(wp.transform_get_rotation(X0), m0 * n)
                rigid_offset1[row] = wp.quat_rotate_inv(wp.transform_get_rotation(X1), -m1 * n)
                if rigid_surface_velocity:
                    rigid_surface_velocity[row] = wp.vec3()
                c = _body_centre(shape_a, shape_body, body_q, body_com)
                torque = _side_torque(x, q, rec_forces, r, 0, first, c)
                contact_f[row] = wp.spatial_vector(_to32v(f_a), _to32v(torque))
        else:
            row = wp.atomic_add(soft_count, 0, 1)
            if row < soft_max:
                # the shape takes shape 0's role: the force on its body by the particle feature
                shape = shape_a
                p_shape = p_a
                p_soft = p_b
                f_shape = f_a
                k0 = first
                k1 = count
                if shape_a < 0:
                    shape = shape_b
                    p_shape = p_b
                    p_soft = p_a
                    f_shape = f_b
                    k0 = 0
                    k1 = first
                c = _body_centre(shape, shape_body, body_q, body_com)
                torque = wp.cross(p_shape - c, f_shape)
                if kind != 4:
                    if shape_a < 0:
                        torque = _side_torque(x, q, rec_forces, r, first, count, c)
                    else:
                        torque = _side_torque(x, q, rec_forces, r, 0, first, c)
                indices = wp.vec3i(-1, -1, -1)
                weights = wp.vec3()
                w = wp.float32(0.0)
                for k in range(k0, k1):
                    w += wp.length(rec_forces[r, k])
                for k in range(k0, k1):
                    indices[k - k0] = q[k]
                    if w > wp.float32(0.0):
                        weights[k - k0] = wp.length(rec_forces[r, k]) / w
                    else:
                        weights[k - k0] = wp.float32(1.0) / wp.float32(k1 - k0)
                soft_shape[row] = shape
                soft_indices[row] = indices
                soft_barycentric[row] = weights
                particle = int(-1)
                if k1 - k0 == 1:
                    particle = q[k0]
                soft_particle[row] = particle
                soft_body_pos[row] = _in_frame(_body_frame(shape, shape_body, body_q), p_shape)
                soft_body_vel[row] = wp.vec3()
                soft_normal[row] = _row_normal(seg < 5, f_shape, p_soft - p_shape)
                soft_rigid_indices[row] = wp.vec3i(-1, -1, -1)
                contact_f[rigid_max + row] = wp.spatial_vector(_to32v(f_shape), _to32v(torque))


@wp.kernel
def _proxy_wrenches(
    rec_vertices: wp.array[wp.vec4i],
    rec_forces: wp.array2d[wp.vec3],
    rec_counts: wp.array[wp.int32],
    rec_offsets: wp.array[wp.int32],
    nthreads: int,
    x: wp.array[wp.vec3d],
    vertex_body: wp.array[wp.int32],
    proxy_of_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    out: wp.array[wp.spatial_vector],
):
    """A coupled solver's proxy-body feedback (class docstring, **Coupling**): the force of every contact term of the
    step on a vertex of a proxy body, and its torque about the body's centre of mass at the end of the step, added at
    the proxy's index ``proxy_of_body[b]``. Grid-stride over the live records of libuipc-warp's ``ContactForces``."""
    tid = wp.tid()
    total = int(0)
    for k in range(10):
        total += rec_counts[k]
    for idx in range(tid, total, nthreads):
        seg = int(0)
        i = idx
        while seg < 9 and i >= rec_counts[seg]:
            i -= rec_counts[seg]
            seg += 1
        r = rec_offsets[seg] + i
        q = rec_vertices[r]
        _first, count = _term_sides(seg % 5)  # a half-plane term holds its plane after its one vertex
        for k in range(count):
            b = vertex_body[q[k]]
            if b < 0:
                continue
            g = proxy_of_body[b]
            if g < 0:
                continue
            f = _record_force(rec_forces, r, k)
            c = wp.transform_point(body_q[b], body_com[b])
            arm = x[q[k]] - wp.vec3d(wp.float64(c[0]), wp.float64(c[1]), wp.float64(c[2]))
            wp.atomic_add(out, g, wp.spatial_vector(_to32v(f), _to32v(wp.cross(arm, f))))


@wp.kernel
def _proxy_particle_forces(
    rec_vertices: wp.array[wp.vec4i],
    rec_forces: wp.array2d[wp.vec3],
    rec_counts: wp.array[wp.int32],
    rec_offsets: wp.array[wp.int32],
    nthreads: int,
    n_particles: int,
    proxy_of_particle: wp.array[int],
    out: wp.array[wp.vec3],
):
    """A coupled solver's proxy-particle feedback (class docstring, **Coupling**): the force of every contact term of
    the step on a proxy particle, added at the proxy's index ``proxy_of_particle[p]``. Grid-stride over the live
    records of libuipc-warp's ``ContactForces``."""
    tid = wp.tid()
    total = int(0)
    for k in range(10):
        total += rec_counts[k]
    for idx in range(tid, total, nthreads):
        seg = int(0)
        i = idx
        while seg < 9 and i >= rec_counts[seg]:
            i -= rec_counts[seg]
            seg += 1
        r = rec_offsets[seg] + i
        q = rec_vertices[r]
        _first, count = _term_sides(seg % 5)
        for k in range(count):
            p = q[k]
            if p >= n_particles:
                continue
            g = proxy_of_particle[p]
            if g < 0:
                continue
            wp.atomic_add(out, g, rec_forces[r, k])


@wp.kernel
def _body_stiffness(
    inertia_per_volume: wp.array[wp.float64],
    cap: wp.float64,
    scale: wp.float64,
    kappa: wp.array[wp.float64],
):
    """The orthogonality stiffness of every body for the step: ``cap`` (libuipc's), lowered to ``scale`` times the
    body's largest principal moment of inertia over its volume where that is lower; a body without one (0) keeps
    ``cap``."""
    i = wp.tid()
    k = cap
    r = inertia_per_volume[i]
    if r > wp.float64(0.0):
        k = wp.min(cap, scale * r)
    kappa[i] = k


@wp.kernel
def _any_selected(world_mask: wp.array[wp.bool], entries: wp.array[wp.int32], selected: wp.array[wp.int32]):
    """``selected[0]`` = 1 when ``world_mask`` selects a world the solver simulates."""
    s = int(0)
    for k in range(entries.shape[0]):
        if world_mask[entries[k]]:
            s = 1
    selected[0] = s


@wp.kernel
def _subscenes_selected(
    world_mask: wp.array[wp.bool], entry: wp.array[wp.int32], first: int, selected: wp.array[wp.int32]
):
    """``selected[s]`` = 1 where ``world_mask`` selects the world of subscene ``s`` (several worlds): its entry
    ``entry[s]``, the global entities' (the mask's last) for subscene 0, the k-th local world with content for
    subscene k + 1. Subscenes below ``first`` are never selected: with independent worlds the global entry selects
    nothing (``first`` 1), since the global entities are no unknowns."""
    s = wp.tid()
    selected[s] = wp.where(s >= first and world_mask[entry[s]], 1, 0)


# A copy of Newton's normalize_reset_world_mask (newton/_src/core/reset.py), which SolverBase.reset calls and which has
# no public name. test_reset_checks_world_mask_as_newton_does fails when the two check a mask differently.
def _normalize_reset_world_mask(
    world_mask: wp.array[wp.bool] | None, *, world_count: int, device: wp.Device
) -> wp.array[wp.bool] | None:
    """Validate a reset mask with shape ``(world_count + 1,)``.

    The final entry selects global entities in world ``-1``. Returns the mask
    unchanged, or ``None`` if no mask was given.
    """
    if world_mask is None:
        return None
    if not isinstance(world_mask, wp.array):
        raise TypeError("'world_mask' must be a Warp array or None.")
    if world_mask.dtype != wp.bool:
        raise TypeError("'world_mask' must have dtype bool.")
    if world_mask.ndim != 1:
        raise ValueError("'world_mask' must be one-dimensional.")
    if world_mask.device != device:
        raise ValueError(f"'world_mask' device {world_mask.device} does not match expected device {device}.")
    mask_size = world_mask.shape[0]
    if mask_size != world_count + 1:
        raise ValueError(f"'world_mask' length {mask_size} must equal model.world_count + 1 ({world_count + 1}).")
    return world_mask


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


def _check_choice(name: str, value: str, accepted: tuple) -> None:
    """``ValueError`` unless ``value`` is one of ``accepted``, the values of libuipc's config key that ``name`` sets."""
    if value not in accepted:
        raise ValueError(f"SolverIPC: {name} must be one of {', '.join(map(repr, accepted))}, got {value!r}.")


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
    and half the median rest length of the edges of the cloth triangles and tetrahedra the solver simulates
    (``_own_elements``)."""
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
    own_tris, own_tets, _ = _own_elements(model)
    if own_tris.any():
        t = model.tri_indices.numpy().reshape(-1, 3)[own_tris]
        edges.append(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]))
    if own_tets.any():
        t = model.tet_indices.numpy().reshape(-1, 4)[own_tets]
        edges.append(np.concatenate([t[:, [a, b]] for a in range(3) for b in range(a + 1, 4)]))
    if edges:
        e = np.unique(np.sort(np.concatenate(edges), axis=1), axis=0)
        x = model.particle_q.numpy().astype(np.float64)
        length = np.linalg.norm(x[e[:, 0]] - x[e[:, 1]], axis=1)
        length = length[length > 0.0]
        if len(length):
            d_hat = min(d_hat, _D_HAT_MESH_SHARE * float(np.median(length)))
    return d_hat


@dataclass(frozen=True)
class _JointOptions:
    """The joints SolverIPC maps beyond revolute, prismatic, fixed, ball and free joints, on request (class
    docstring, **Joints**): ``d6``, D6 joints that lock what one of libuipc's joints locks; ``distance``, distance
    joints of one length or none; ``limit_damping``, ``joint_limit_kd`` past a limit; ``armature``,
    ``joint_armature`` as the inertia of a revolute or prismatic joint's coordinate; ``effort_limit``,
    ``joint_effort_limit`` as a bound of the drive's force decided at the step's start."""

    d6: bool = False
    distance: bool = False
    limit_damping: bool = False
    armature: bool = False
    effort_limit: bool = False

    @property
    def drive_terms(self) -> bool:
        """Whether the driving terms take a term beyond ``_joint_terms``' (``_joint_option_terms``)."""
        return self.limit_damping or self.armature or self.effort_limit

    @property
    def rates(self) -> bool:
        """Whether those terms need the joints' rates at the step's start (``_joint_rates``)."""
        return self.armature or self.effort_limit


@dataclass(frozen=True)
class _JointMap:
    """How every joint of a model maps onto libuipc (``_map_joints``).

    ``kind``: the joint type whose libuipc joint the joint becomes, REVOLUTE, PRISMATIC, FIXED or BALL, or FREE for
    none. ``store``: the type ``_store_joints`` and ``_joint_wrenches`` take the joint for, its own but for a D6 joint:
    that of its libuipc joint, or FIXED (skipped) for one whose coordinates and ``joint_f`` the D6 kernels take.
    ``d6``: those joints, D6 joints of three angular axes, with or without three linear ones. ``distance``: the
    distance joints of one length, libuipc's stitch of that rest length between their joint frames' origins; the
    other distance joints, of no length, are no joint, and every distance joint is stored as a free joint."""

    kind: np.ndarray
    store: np.ndarray
    d6: np.ndarray
    relabeled: bool  # whether a joint is stored as another type than its own
    distance: np.ndarray


# what libuipc's joints lock, by the (linear, angular) axes a D6 joint keeps free: all six (fixed), all but one
# rotation (revolute), all but one translation (prismatic), the translations (ball) or nothing
_D6_KINDS = {
    (0, 0): JointType.FIXED,
    (0, 1): JointType.REVOLUTE,
    (1, 0): JointType.PRISMATIC,
    (0, 3): JointType.BALL,
    (3, 3): JointType.FREE,
}


def _orthonormal(axes: np.ndarray) -> bool:
    """Whether the rows of ``axes`` are orthonormal (to float32's rounding)."""
    return bool(np.abs(axes @ axes.T - np.eye(len(axes))).max() < 1.0e-5)


def _map_joints(model: Model, options: _JointOptions) -> _JointMap:
    """How every joint maps onto libuipc (``_JointMap``), or ``NotImplementedError`` for the first joint that does
    not: a joint type libuipc has no joint for (``_UNMAPPED_JOINTS``), and without ``options.d6`` every D6 joint; with
    it a D6 joint that locks other degrees of freedom than one of libuipc's joints does, or that takes Newton's
    coordinates of three axes that are not orthonormal; without ``options.distance`` every distance joint, with it one
    whose distance is bounded to a range or on one side."""
    joint_type = model.joint_type.numpy().astype(np.int32)
    kind, store = joint_type.copy(), joint_type.copy()
    supported = [int(t) for t in (JointType.FREE, JointType.REVOLUTE, JointType.PRISMATIC, JointType.FIXED)]
    supported.append(int(JointType.BALL))
    if options.d6:
        supported.append(int(JointType.D6))
    if options.distance:
        supported.append(int(JointType.DISTANCE))
    other = np.flatnonzero(~np.isin(joint_type, supported))
    if len(other):
        j = int(other[0])
        t = JointType(int(joint_type[j]))
        why = _UNMAPPED_JOINTS.get(t, "libuipc's affine-body joints are revolute, prismatic, fixed and ball joints")
        if t == JointType.D6:
            why += (
                "; with enable_d6_joints=True SolverIPC takes a D6 joint that locks what one of them locks, or nothing"
            )
        elif t == JointType.DISTANCE:
            why += (
                "; with enable_distance_joints=True SolverIPC takes a distance joint of one length, as libuipc's soft "
                "vertex stitch of that rest length, and one of no length"
            )
        raise NotImplementedError(f"SolverIPC does not simulate {t.name} joints (joint {j}): {why}.")
    d6 = np.flatnonzero(joint_type == int(JointType.D6))
    if len(d6):
        dof_dim = model.joint_dof_dim.numpy()
        qd_start = model.joint_qd_start.numpy()
        axes = model.joint_axis.numpy().astype(np.float64).reshape(-1, 3)
    for j in d6:
        lin, ang = (int(n) for n in dof_dim[j])
        t = _D6_KINDS.get((lin, ang))
        if t is None:
            raise NotImplementedError(
                f"SolverIPC does not simulate the D6 joint {j}: its {lin} linear and {ang} angular axes lock "
                f"{3 - lin} translations and {3 - ang} rotations, and libuipc's joints lock all six (fixed), all but "
                "one rotation (revolute), all but one translation (prismatic), the translations (ball) or nothing."
            )
        if ang == 3:
            s = int(qd_start[j])
            for name, rows in (("linear", axes[s : s + lin]), ("angular", axes[s + lin : s + lin + ang])):
                if len(rows) and not _orthonormal(rows):
                    raise NotImplementedError(
                        f"SolverIPC does not simulate the D6 joint {j}: its {name} axes are not orthogonal, which "
                        "Newton's coordinates of a D6 joint of three axes assume."
                    )
        kind[j] = int(t)
        store[j] = int(JointType.FIXED) if ang == 3 else int(t)
    distance = np.flatnonzero(joint_type == int(JointType.DISTANCE))
    if len(distance):
        # the bounds of a distance joint are those of its first degree of freedom, a negative one being no bound
        first = model.joint_qd_start.numpy()[distance]
        lo, hi = model.joint_limit_lower.numpy()[first], model.joint_limit_upper.numpy()[first]
        for j, a, b in zip(distance, lo, hi, strict=True):
            if (a >= 0.0 or b >= 0.0) and not (a == b):
                raise NotImplementedError(
                    f"SolverIPC does not simulate the DISTANCE joint {j}, whose distance is bounded to "
                    f"[{float(a)!r}, {float(b)!r}] (a negative bound is none): a range or a single bound is an "
                    "inequality, and libuipc's soft vertex stitch, the spring it holds a distance with, has one rest "
                    "length; libuipc has no distance limit."
                )
        kind[distance] = int(JointType.FREE)
        store[distance] = int(JointType.FREE)
        distance = distance[lo >= 0.0]
    three = (d6[dof_dim[d6, 1] == 3] if len(d6) else d6).astype(np.int32)
    return _JointMap(kind, store, three, bool(np.any(store != joint_type)), distance.astype(np.int32))


def _check_supported(model: Model, joints: _JointOptions | None = None) -> None:
    """Reject model content the solver would otherwise accept and ignore; joints as ``joints`` maps them
    (``_JointOptions``). What independent worlds refuse is checked when the solve is laid out
    (``SolverIPC._check_independent_options``, ``_check_independent_content``)."""
    if model.requires_grad:
        raise NotImplementedError("SolverIPC does not support gradients (model.requires_grad=True).")
    for name, count in (
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
    own_tris, own_tets, _ = _own_elements(model)
    for name, activations, own in (
        ("triangle", model.tri_activations, own_tris),
        ("tetrahedron", model.tet_activations, own_tets),
    ):
        if activations is not None and len(activations) and np.any(activations.numpy()[own] != 0.0):
            raise NotImplementedError(f"SolverIPC does not take {name} activations; the model has non-zero ones.")
    _check_joints(model, joints)
    _check_springs(model)
    _check_attachments(model)
    _check_surface_velocity(model)


def _check_springs(model: Model) -> None:
    """Springs that libuipc represents: no damping (libuipc has no damping element), a rest length that is not
    negative (above 0 libuipc's Hookean spring, 0 its soft vertex stitch) and a stiffness that is not negative. The
    springs of a coupled solver's other entries are theirs (``_own_elements``)."""
    if model.spring_count == 0:
        return
    own = _own_elements(model)[2]
    for name, array, bad, why in (
        ("damping", model.spring_damping, lambda v: v != 0.0, "libuipc's springs and stitches have no damping"),
        ("rest length", model.spring_rest_length, lambda v: ~(v >= 0.0), "it must not be negative"),
        ("stiffness", model.spring_stiffness, lambda v: ~(v >= 0.0), "it must not be negative"),
    ):
        values = array.numpy()
        found = np.flatnonzero(bad(values) & own)
        if len(found):
            k = int(found[0])
            raise NotImplementedError(
                f"SolverIPC does not simulate spring {k}, whose {name} is {float(values[k])!r}: {why}."
            )
    ends = model.spring_indices.numpy().reshape(-1, 2)
    same = np.flatnonzero((ends[:, 0] == ends[:, 1]) & own)
    if len(same):
        k = int(same[0])
        raise NotImplementedError(
            f"SolverIPC does not simulate spring {k}, which joins particle {int(ends[k, 0])} to itself."
        )


def _check_attachments(model: Model) -> None:
    """Particle-body attachments that libuipc's soft vertex stitch represents: no damping (libuipc's stitch has none,
    and an implicit damping cannot be folded into it the way the joint drives fold theirs, since it has no rest
    offset to carry it) and a stiffness that is not negative, finite or infinite (hard)."""
    if model.attachment_particle_body_count == 0:
        return
    for name, array, bad, why in (
        ("damping", model.attachment_particle_body_damping, lambda v: v != 0.0, "libuipc's stitch has no damping"),
        ("stiffness", model.attachment_particle_body_stiffness, lambda v: ~(v >= 0.0), "it must not be negative"),
    ):
        values = array.numpy()
        found = np.flatnonzero(bad(values))
        if len(found):
            k = int(found[0])
            raise NotImplementedError(
                f"SolverIPC does not simulate particle-body attachment {k}, whose {name} is {float(values[k])!r}: "
                f"{why}."
            )


def _check_joints(model: Model, options: _JointOptions | None = None) -> None:
    """Joints that libuipc's affine-body joints represent (class docstring, **Joints**): revolute, prismatic, fixed,
    ball and free joints, and with ``options`` the joints ``_map_joints`` maps; drives, passive springs, damping and
    limits on revolute and prismatic joints only; no armature (with ``options.armature``, none on ball and free
    joints), no joint friction, and no effort limit on a driven degree of freedom (with ``options.effort_limit``, none
    that is negative)."""
    if model.joint_count == 0:
        return
    joint_type = model.joint_type.numpy()
    options = options or _JointOptions()
    kind = _map_joints(model, options).kind
    if model.joint_dof_count == 0:
        return
    qd_start = model.joint_qd_start.numpy()
    dof_joint = np.repeat(np.arange(model.joint_count), np.diff(qd_start))
    dof_type = kind[dof_joint]

    def per_dof(name: str) -> np.ndarray:
        a = getattr(model, name)
        return np.zeros(model.joint_dof_count) if a is None else a.numpy().astype(np.float64)

    def refuse(dofs: np.ndarray, what: Callable[[int], str], why: str) -> None:
        if len(dofs):
            d = int(dofs[0])
            j = int(dof_joint[d])
            raise NotImplementedError(
                f"SolverIPC {why}: DOF {d} of the {JointType(int(joint_type[j])).name} joint {j} has {what(d)}."
            )

    friction = per_dof("joint_friction")
    refuse(
        np.flatnonzero(friction != 0.0),
        lambda d: f"joint_friction {float(friction[d])!r}",
        "does not simulate joint friction: libuipc's joints have none, and dry friction, a bounded force, is no "
        "quadratic potential that libuipc's driving term could hold",
    )
    ke, kd = per_dof("joint_target_ke"), per_dof("joint_target_kd")
    ks, c = per_dof("joint_stiffness"), per_dof("joint_damping")
    mode = model.joint_target_mode.numpy() if model.joint_target_mode is not None else np.zeros(len(ke), np.int32)
    actuated = np.isin(
        mode, [int(m) for m in (JointTargetMode.POSITION, JointTargetMode.VELOCITY, JointTargetMode.POSITION_VELOCITY)]
    )
    lo, hi, limit_k = per_dof("joint_limit_lower"), per_dof("joint_limit_upper"), per_dof("joint_limit_ke")
    limited = ((lo > -MAXVAL) | (hi < MAXVAL)) & (limit_k > 0.0)
    limited[qd_start[:-1][joint_type == int(JointType.DISTANCE)]] = False  # a distance joint's bounds
    other = np.isin(dof_type, [int(JointType.BALL), int(JointType.FREE)])
    armature = per_dof("joint_armature")
    if options.armature:
        refuse(
            np.flatnonzero(~(armature >= 0.0)),
            lambda d: f"joint_armature {float(armature[d])!r}",
            "takes no negative joint armature",
        )
        refuse(
            np.flatnonzero((armature != 0.0) & other),
            lambda d: f"joint_armature {float(armature[d])!r}",
            "gives the armature of revolute and prismatic joints only: libuipc's driving term, which holds it, is a "
            "quadratic potential of one hinge angle or one distance, and the armature of a ball or a free joint is an "
            "inertia about three axes, or a mass without weight",
        )
    else:
        refuse(
            np.flatnonzero(armature != 0.0),
            lambda d: f"joint_armature {float(armature[d])!r}",
            "does not simulate joint armature: libuipc's joints have no inertia of their own; with "
            "enable_joint_armature=True SolverIPC gives a revolute or prismatic joint's to its coordinate, through "
            "libuipc's driving term",
        )

    def drive(d: int) -> str:
        if actuated[d]:
            return f"the target mode {JointTargetMode(int(mode[d])).name}"
        return (
            f"target_ke {float(ke[d])!r}, target_kd {float(kd[d])!r}, stiffness {float(ks[d])!r} "
            f"and damping {float(c[d])!r}"
        )

    driven = actuated | (ke != 0.0) | (kd != 0.0) | (ks != 0.0) | (c != 0.0)
    refuse(
        np.flatnonzero(other & driven),
        drive,
        "drives, springs and damps revolute and prismatic joints only: libuipc's driving term (constitutions 19 "
        "and 21) is a quadratic potential of one hinge angle or one distance, its ball joint has none (constitution "
        "27 is only registered) and a free body has no joint (a D6 joint of three angular axes is a ball joint or "
        "no joint); Control.joint_f acts on every joint",
    )
    refuse(
        np.flatnonzero(other & limited),
        lambda d: f"the limits [{float(lo[d])!r}, {float(hi[d])!r}] with limit_ke {float(limit_k[d])!r}",
        "limits revolute and prismatic joints only: libuipc's limits (constitutions 669 and 670) act on one hinge "
        "angle or one distance, and its ball joint has none",
    )
    effort = per_dof("joint_effort_limit")
    gains = (ke != 0.0) | (kd != 0.0)
    if options.effort_limit:
        refuse(
            np.flatnonzero(gains & ~(effort >= 0.0)),
            lambda d: f"joint_effort_limit {float(effort[d])!r} and a drive",
            "takes no negative effort limit",
        )
    else:
        refuse(
            np.flatnonzero(gains & (effort < _EFFORT_LIMIT)),
            lambda d: f"joint_effort_limit {float(effort[d])!r} and a drive",
            "cannot bound the force of a drive: libuipc's driving term is a quadratic potential, whose force has no "
            f"bound; raise the limit to {_EFFORT_LIMIT:g}, Newton's default, to drive the joint without a bound, or "
            "pass enable_joint_effort_limit=True for a bound decided at the start of every step",
        )


def _kinematic_bodies(model: Model) -> np.ndarray:
    """The bodies the solver does not let forces move: :attr:`~newton.BodyFlags.KINEMATIC` ones, those of mass 0, and
    those of inverse mass 0. A body of positive mass has an inverse mass of 0 only in a coupled solver's view, which
    zeroes the inverse mass of the bodies another entry owns and keeps their mass (class docstring, **Coupling**)."""
    if model.body_count == 0:
        return np.zeros(0, bool)
    kinematic = ((model.body_flags.numpy() & int(BodyFlags.KINEMATIC)) != 0) | (model.body_mass.numpy() <= 0.0)
    return kinematic | (model.body_inv_mass.numpy() <= 0.0)


def _own_elements(model: Model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Which triangles, tetrahedra and springs the solver simulates, as masks over ``tri_indices``, ``tet_indices`` and
    ``spring_indices``: every one but those whose particles are all disabled, of mass 0 and not
    :attr:`~newton.ParticleFlags.ACTIVE`. A coupled solver's view disables the particles another entry owns, which
    simulates their cloth, soft bodies and springs (class docstring, **Coupling**); in a model of one solver such a
    triangle or tetrahedron holds nothing that moves, and Newton's own contacts skip its faces."""
    own = tuple(np.ones(n, bool) for n in (model.tri_count, model.tet_count, model.spring_count))
    if model.particle_count == 0 or not (model.tri_count or model.tet_count or model.spring_count):
        return own
    flags = model.particle_flags.numpy()
    disabled = (model.particle_mass.numpy() <= 0.0) & ((flags & int(ParticleFlags.ACTIVE)) == 0)
    if not disabled.any():
        return own
    indices = ((model.tri_indices, 3), (model.tet_indices, 4), (model.spring_indices, 2))
    return tuple(
        ~disabled[rows.numpy().reshape(-1, k)].all(axis=1) if len(mask) else mask
        for mask, (rows, k) in zip(own, indices, strict=True)
    )


def _proxy_bodies(model: Model) -> np.ndarray:
    """The bodies a coupled solver's view marks as proxies (:attr:`~newton.BodyFlags.PROXY`): another entry's bodies
    that this entry moves as its own, with the view's masses (class docstring, **Coupling**)."""
    if model.body_count == 0:
        return np.zeros(0, bool)
    return (model.body_flags.numpy() & int(BodyFlags.PROXY)) != 0


def _proxy_particles(model: Model) -> np.ndarray:
    """The particles a coupled solver's view marks as proxies (:attr:`~newton.ParticleFlags.PROXY`)."""
    if model.particle_count == 0:
        return np.zeros(0, bool)
    return (model.particle_flags.numpy() & int(ParticleFlags.PROXY)) != 0


def _local_worlds(model: Model) -> np.ndarray:
    """The local worlds (index >= 0) that hold particles, bodies or colliding shapes, in increasing order."""
    worlds = []
    if model.particle_count:
        worlds.append(model.particle_world.numpy())
    if model.body_count:
        worlds.append(model.body_world.numpy())
    if model.shape_count:
        worlds.append(model.shape_world.numpy()[_colliding_shapes(model)])
    local = np.unique(np.concatenate(worlds)) if worlds else np.zeros(0, np.int32)
    return local[local >= 0].astype(np.int64)


def _element_across_worlds(model: Model, free: np.ndarray) -> str | None:
    """The first triangle, tetrahedron or spring the solver simulates whose free particles (``free``) lie in two local
    worlds, named with the two worlds, or None. With independent worlds no element joins two worlds, which
    libuipc-warp refuses in its own terms; Newton's builder lets a spring, a triangle or a tetrahedron join any
    particles. A bending edge joins two worlds only where one of its triangles does. A global particle is in no world
    (a free global particle is refused before)."""
    if model.particle_count == 0:
        return None
    world = model.particle_world.numpy().astype(np.int64)
    own_tris, own_tets, own_springs = _own_elements(model)
    for name, indices, k, own in (
        ("triangle", model.tri_indices, 3, own_tris),
        ("tetrahedron", model.tet_indices, 4, own_tets),
        ("spring", model.spring_indices, 2, own_springs),
    ):
        if indices is None or len(indices) == 0:
            continue
        ids = np.flatnonzero(own)
        rows = indices.numpy().reshape(-1, k).astype(np.int64)[ids]
        index = np.maximum(rows, 0)
        mine = (rows >= 0) & free[index] & (world[index] >= 0)
        lo = np.where(mine, world[index], np.iinfo(np.int64).max).min(axis=1)
        hi = np.where(mine, world[index], -1).max(axis=1)
        joined = np.flatnonzero(mine.any(axis=1) & (lo != hi))
        if len(joined):
            i = int(joined[0])
            more = f" ({len(joined) - 1} more {name}s do too)" if len(joined) > 1 else ""
            return f"{name} {int(ids[i])} joins free particles of the worlds {int(lo[i])} and {int(hi[i])}{more}"
    return None


def _compatible_groups(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Newton's collision-group rule for two shapes that take part in shape contact (``shape_contact_pairs.py``): a
    positive group meets itself and every negative group, a negative group every group but itself. A shape of group 0
    takes part in no shape pair."""
    return np.where(a > 0, (a == b) | (b < 0), a != b) & (a != 0) & (b != 0)


def _compatible_worlds(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """A global entity (world -1) meets every world, a local one its own world only."""
    return (a == b) | (a == -1) | (b == -1)


@dataclass
class _Worlds:
    """The subscenes of a model with content in more than one local world (several worlds): subscene 0 holds the
    global entities (world -1), subscene k + 1 the k-th local world that holds content."""

    worlds: np.ndarray  # (L,): the world of subscene k + 1
    lookup: np.ndarray  # (world_count + 1,): the subscene of world w at index w, of the global world -1 at the last

    @classmethod
    def of(cls, model: Model) -> _Worlds | None:
        """The subscenes of ``model``, or None when it has content in at most one local world."""
        local = _local_worlds(model)
        if len(local) < 2:
            return None
        lookup = np.zeros(model.world_count + 1, np.int64)
        lookup[local] = 1 + np.arange(len(local))
        return cls(local, lookup)

    @property
    def count(self) -> int:
        """The subscenes, the global one included."""
        return len(self.worlds) + 1

    def subscene(self, world: np.ndarray) -> np.ndarray:
        """The subscene of every entity of ``world`` (-1 for a global one)."""
        return self.lookup[np.asarray(world, np.int64)]


@dataclass
class _ContactElements:
    """libuipc contact elements for a model and the contact model of every pair of them (class docstring,
    **Contact**). Element 0 is libuipc's default element; it holds no vertex."""

    shape: np.ndarray  # (shape_count,): the element of every shape the solver triangulates, -1 for the others
    particle: np.ndarray  # (particle_count,): the element of every particle
    on_mesh: np.ndarray  # (particle_count,): the particles of cloth and tetrahedra
    enabled: np.ndarray  # (count, count): whether the two elements collide
    mu: np.ndarray  # (count, count): their friction coefficient

    @property
    def count(self) -> int:
        return len(self.enabled)


def _filter_classes(world: np.ndarray, filters: np.ndarray, ns: int) -> np.ndarray:
    """A key for every shape in a filter pair (-1 for the other shapes) that the copies of the shape in replicated
    worlds share: its offset in its world (a global shape: ``-1 - index``) and the offsets of its filter partners,
    which are in its world or global, as Newton compares replicated worlds (``shape_contact_pairs.py``, its
    normalized filters). Two shapes get one key only where their filter partners correspond."""
    local = world >= 0
    start = np.full(max(int(world.max(initial=-1)) + 1, 1), ns, np.int64)
    np.minimum.at(start, world[local], np.flatnonzero(local))
    offset = np.where(local, np.arange(ns) - start[np.maximum(world, 0)], -1 - np.arange(ns))
    partners: dict[int, list] = {}
    for a, b in filters.tolist():
        partners.setdefault(a, []).append(int(offset[b]))
        partners.setdefault(b, []).append(int(offset[a]))
    keys: dict[tuple, int] = {}
    own = np.full(ns, -1, np.int64)
    for shape in sorted(partners):
        own[shape] = keys.setdefault((int(offset[shape]), *sorted(partners[shape])), len(keys))
    return own


def _contact_elements(model: Model, worlds: _Worlds | None = None, aims: bool = False) -> _ContactElements:
    """The contact elements of a model's shapes and particles and the models of their pairs, checked against
    :attr:`Model.shape_contact_pairs`.

    Shapes that Newton's rules cannot tell apart share an element: their signature is the collision group, the world,
    the two collide flags and the friction coefficient. A shape in a filter pair that would otherwise collide gets an
    element of its own, since the pair excludes two shapes and not two classes of them. The particles take three
    elements: those in no triangle and no tetrahedron, active (``P_free``) or not (``P_off``), and the vertices of
    cloth and tetrahedra (``P_surf``). With ``worlds`` (several worlds) the subscenes keep the worlds apart, so the
    world leaves the signature, and a shape in a filter pair shares its element with its copies in replicated worlds
    (``_filter_classes``). With ``aims`` (``kinematic_motion="aim"``) a shape's motion joins the signature: on a
    dynamic body, on a kinematic body (``_kinematic_bodies``), or static; the pairs of two shapes that are each
    kinematic or static are disabled, and ``Model.shape_contact_pairs`` less those pairs is the set compared. In a
    coupled solver's view with proxies (class docstring, **Coupling**) a shape's part joins the signature
    (``_proxy_roles``): the pairs of a shape on a proxy body with another such shape or with a static shape or one on a
    body of inverse mass 0 are disabled, and so are those of proxy particles, which take elements of their own
    (``Q_free``, ``Q_surf``), with those shapes and among themselves; the set compared leaves those shape pairs out
    too. Raises ``NotImplementedError`` when the shape pairs the elements enable on different bodies are not exactly
    that set (a pair array assigned to the model; a coupled solver's view assigns it this set). Only the triangles and
    tetrahedra the solver simulates (``_own_elements``) make vertices of cloth and tetrahedra. The engine tests the
    elements' table and the subscenes' table apart, so their sizes are not multiplied; it refuses what it cannot hold
    itself, naming the sizes.
    """
    ns, n_particles = model.shape_count, model.particle_count
    shape_element = np.full(ns, -1, np.int64)
    classes = 0
    proxy_particle = _proxy_particles(model)
    proxies = bool(proxy_particle.any() or _proxy_bodies(model).any())
    if ns:
        flags = model.shape_flags.numpy()
        group = model.shape_collision_group.numpy().astype(np.int64)
        world = model.shape_world.numpy().astype(np.int64)
        body = model.shape_body.numpy().astype(np.int64)
        shape_mu = model.shape_material_mu.numpy().astype(np.float32)
        hits_shapes = (flags & int(ShapeFlags.COLLIDE_SHAPES)) != 0
        hits_particles = (flags & int(ShapeFlags.COLLIDE_PARTICLES)) != 0
        takes_part = hits_shapes & (group != 0)
        # the filter pairs that remove a pair Newton's other rules let collide (_ShapeContactPairs._active_filters)
        filters = np.asarray(model.shape_collision_filter_pairs_array(), np.int64).reshape(-1, 2)
        fa, fb = filters[:, 0], filters[:, 1]
        effective = (
            takes_part[fa]
            & takes_part[fb]
            & (body[fa] != body[fb])
            & _compatible_worlds(world[fa], world[fb])
            & _compatible_groups(group[fa], group[fb])
        )
        filters = filters[effective]
        if worlds is None:
            own = np.full(ns, -1, np.int64)
            own[filters.reshape(-1)] = filters.reshape(-1)
        else:
            own = _filter_classes(world, filters, ns)
        colliding = np.flatnonzero(_colliding_shapes(model))
        # with subscenes the worlds are kept apart by them: the world does not split the classes
        class_world = world if worlds is None else np.zeros_like(world)
        columns = [group, class_world, hits_shapes, hits_particles, shape_mu.view(np.int32).astype(np.int64), own]
        if aims:  # the shape's motion: 0 on a dynamic body, 1 on a kinematic one, 2 static
            motion = np.where(body < 0, 2, 0)
            motion[body >= 0] = _kinematic_bodies(model)[body[body >= 0]]
            columns.append(motion)
        if proxies:  # the shape's part in a coupled solver's proxy contact
            role = _proxy_roles(model, body)
            columns.append(role)
        signature = np.stack(columns, axis=1)[colliding]
        _, first, inverse = np.unique(signature, axis=0, return_index=True, return_inverse=True)
        order = np.argsort(first)  # classes numbered in the order of their first shapes
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        shape_class = rank[inverse.reshape(-1)]
        shape_element[colliding] = 1 + shape_class
        classes = len(first)
        rep = colliding[first[order]]  # one shape of each class
    count = 1 + classes + (3 if n_particles else 0) + (2 if proxy_particle.any() else 0)
    enabled = np.zeros((count, count), bool)
    mu = np.zeros((count, count))
    shapes = slice(1, 1 + classes)
    if classes:
        g, w, m = group[rep], class_world[rep], shape_mu[rep].astype(np.float64)
        part = takes_part[rep]
        pairs = (
            part[:, None]
            & part[None, :]
            & _compatible_groups(g[:, None], g[None, :])
            & _compatible_worlds(w[:, None], w[None, :])
        )
        cls = shape_element - 1
        pairs[cls[filters[:, 0]], cls[filters[:, 1]]] = False
        pairs[cls[filters[:, 1]], cls[filters[:, 0]]] = False
        immovable = None
        if aims:  # two shapes that each move only as the caller prescribes, or not at all, do not collide
            moves = motion[rep] == 0
            pairs &= moves[:, None] | moves[None, :]
            immovable = motion != 0
        if proxies:  # a proxy meets neither another proxy nor what is static or of inverse mass 0
            proxy, still = role[rep] == 1, role[rep] == 2
            pairs &= ~(proxy[:, None] & (proxy | still)[None, :]) & ~(still[:, None] & proxy[None, :])
        enabled[shapes, shapes] = pairs
        mu[shapes, shapes] = 0.5 * (m[:, None] + m[None, :])
        _check_shape_pairs(
            model,
            colliding,
            shape_class,
            body[colliding],
            pairs,
            worlds is not None,
            immovable,
            role if proxies else None,
        )
    particle_element = np.zeros(n_particles, np.int64)
    on_mesh = np.zeros(n_particles, bool)
    if n_particles:
        p_free, p_surf, p_off = 1 + classes, 2 + classes, 3 + classes
        own_tris, own_tets, _ = _own_elements(model)
        for cells, own, k in ((model.tri_indices, own_tris, 3), (model.tet_indices, own_tets, 4)):
            if own.any():
                on_mesh[cells.numpy().reshape(-1, k)[own].reshape(-1)] = True
        active = (model.particle_flags.numpy() & int(ParticleFlags.ACTIVE)) != 0
        particle_element = np.where(on_mesh, p_surf, np.where(active, p_free, p_off))
        if proxy_particle.any():  # a coupled solver's proxy particles, by kind
            q_free, q_surf = 4 + classes, 5 + classes
            particle_element = np.where(
                proxy_particle & (particle_element == p_surf),
                q_surf,
                np.where(proxy_particle & (particle_element == p_free), q_free, particle_element),
            )
        # particles meet the shapes that collide with particles, of their own world or global (with subscenes, the
        # subscenes keep the worlds apart)
        for p in (p_free, p_surf):
            if classes:
                enabled[p, shapes] = enabled[shapes, p] = hits_particles[rep]
                mu[p, shapes] = mu[shapes, p] = 0.5 * (model.soft_contact_mu + m)
        # particles are spheres to each other where Newton's particle solvers collide them; finite elements always
        # collide with themselves and with each other
        spheres = n_particles > 1 and model.particle_max_radius > 0.0
        enabled[p_free, p_free] = enabled[p_free, p_surf] = enabled[p_surf, p_free] = spheres
        enabled[p_surf, p_surf] = True
        mu[p_free : p_surf + 1, p_free : p_surf + 1] = model.particle_mu
        if proxy_particle.any():
            # a proxy particle meets the shapes a particle of its kind meets but those a proxy does not, and the
            # entry's own particles as a particle of its kind does; no other proxy
            for q, p in ((q_free, p_free), (q_surf, p_surf)):
                if classes:
                    enabled[q, shapes] = enabled[shapes, q] = hits_particles[rep] & (role[rep] == 0)
                    mu[q, shapes] = mu[shapes, q] = 0.5 * (model.soft_contact_mu + m)
                for other in (p_free, p_surf):
                    enabled[q, other] = enabled[other, q] = enabled[p, other]
                    mu[q, other] = mu[other, q] = model.particle_mu
    return _ContactElements(shape_element, particle_element, on_mesh, enabled, mu)


def _proxy_roles(model: Model, body: np.ndarray) -> np.ndarray:
    """The part of every shape in a coupled solver's proxy contact (class docstring, **Coupling**), from its body
    ``body``: 1 on a proxy body, 2 static or on a body of inverse mass 0, 0 otherwise. A proxy meets none of the
    shapes of parts 1 and 2, as the default ``CouplingInterface.coupling_prepare_proxy_contacts`` drops Newton's
    contacts of a proxy with another proxy, a static shape or a body of inverse mass 0."""
    role = np.where(body < 0, 2, 0)
    on_body = body >= 0
    if on_body.any():
        b = body[on_body]
        inverse_mass = model.body_inv_mass.numpy()[b]
        role[on_body] = np.where(_proxy_bodies(model)[b], 1, np.where(inverse_mass <= 0.0, 2, 0))
    return role


def _enabled_pair_count(shape_class: np.ndarray, shape_body: np.ndarray, enabled: np.ndarray) -> int:
    """The pairs of shapes on different bodies (static shapes on body -1) whose classes ``enabled`` pairs, counted
    from the class sizes, less the pairs of shapes on one body."""
    sizes = np.bincount(shape_class, minlength=len(enabled)).astype(np.int64)
    upper = np.triu(enabled, 1)
    count = int((upper * np.outer(sizes, sizes)).sum() + (np.diag(enabled) * sizes * (sizes - 1) // 2).sum())
    entries, n = np.unique(np.stack([shape_body, shape_class], axis=1), axis=0, return_counts=True)
    count -= int((np.diag(enabled)[entries[:, 1]] * n * (n - 1) // 2).sum())
    starts = np.flatnonzero(np.r_[True, entries[1:, 0] != entries[:-1, 0]])
    for s, e in zip(starts, np.r_[starts[1:], len(entries)], strict=True):
        if e - s > 1:
            c, k = entries[s:e, 1], n[s:e]
            count -= int((upper[np.ix_(c, c)] * np.outer(k, k)).sum())
    return count


def _check_shape_pairs(
    model: Model,
    shapes: np.ndarray,
    shape_class: np.ndarray,
    shape_body: np.ndarray,
    enabled: np.ndarray,
    by_world: bool = False,
    immovable: np.ndarray | None = None,
    proxy_role: np.ndarray | None = None,
) -> None:
    """Raise unless the shape classes and their ``enabled`` pairs give exactly ``Model.shape_contact_pairs``: every
    listed pair enabled, and as many enabled pairs of shapes on different bodies (static shapes on body -1) as listed.
    ``shapes`` are the shapes the solver triangulates, ``shape_class`` and ``shape_body`` theirs. ``by_world``: the
    worlds are subscenes, and only shapes of one world or a global one pair; the count is taken per world, the global
    shapes among themselves and every world's shapes among themselves and with the global ones. ``immovable``: per
    shape, whether it is kinematic or static; the listed pairs of two such shapes are left out of the comparison.
    ``proxy_role``: per shape, its part in a coupled solver's proxy contact (``_proxy_roles``); the listed pairs of a
    proxy shape with another or with a shape of part 2 are left out too."""
    cls = np.full(model.shape_count, -1, np.int64)
    cls[shapes] = shape_class
    listed = model.shape_contact_pairs
    listed = np.zeros((0, 2), np.int64) if listed is None else listed.numpy().reshape(-1, 2).astype(np.int64)
    expected = model.shape_contact_pair_count
    dropped, of = np.zeros(len(listed), bool), []
    if immovable is not None:
        dropped |= immovable[listed[:, 0]] & immovable[listed[:, 1]]
        of.append("of two kinematic or static shapes")
    if proxy_role is not None:
        r0, r1 = proxy_role[listed[:, 0]], proxy_role[listed[:, 1]]
        dropped |= ((r0 == 1) & (r1 > 0)) | ((r1 == 1) & (r0 > 0))
        of.append("of a proxy with another proxy or a static shape")
    if of:
        listed = listed[~dropped]
        expected = len(listed)
    a, b = cls[listed[:, 0]], cls[listed[:, 1]]
    missing = (a < 0) | (b < 0) | ~enabled[np.maximum(a, 0), np.maximum(b, 0)]
    if not by_world:
        count = _enabled_pair_count(shape_class, shape_body, enabled)
    else:
        all_worlds = model.shape_world.numpy().astype(np.int64)
        missing |= ~_compatible_worlds(all_worlds[listed[:, 0]], all_worlds[listed[:, 1]])
        world = all_worlds[shapes]
        glob = world == -1
        count = global_count = _enabled_pair_count(shape_class[glob], shape_body[glob], enabled)
        for w in np.unique(world[~glob]):
            mine = glob | (world == w)
            count += _enabled_pair_count(shape_class[mine], shape_body[mine], enabled) - global_count
    missing = np.flatnonzero(missing)
    if len(missing) or count != expected:
        detail = (
            f"shapes {int(listed[missing[0], 0])} and {int(listed[missing[0], 1])} are listed but not in it"
            if len(missing)
            else f"they give {count} pairs, Model.shape_contact_pairs lists {expected}"
            + (f" that are not {' or '.join(of)}" if of else "")
        )
        raise NotImplementedError(
            "SolverIPC collides the shapes that Newton's collision groups, worlds, collide flags and filter pairs "
            f"let collide, which must be Model.shape_contact_pairs: {detail}. A pair array assigned to the model is "
            "supported only when it is that set, as a coupled solver's view assigns it."
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
    elif t == GeoType.HFIELD:
        return _heightfield_mesh(model, s)
    else:
        raise NotImplementedError(f"SolverIPC does not support shape {s} of type {GeoType(t).name}.")
    return _weld(mesh.vertices, mesh.indices)


def _heightfield_mesh(model: Model, s: int) -> tuple[np.ndarray, np.ndarray]:
    """The collision triangles of height field shape ``s`` in its frame, as Newton's narrow phase makes them
    (``get_triangle_shape_from_heightfield``): sample ``(r, c)`` at ``x = -hx + c dx``, ``y = -hy + r dy``,
    ``z = min_z + h (max_z - min_z)``, every cell ``(r, c)`` split into ``(p00, p10, p11)`` and ``(p00, p11, p01)``.
    The field is the shape's :class:`~newton.Heightfield` (:attr:`~newton.Model.shape_source`), its extents scaled by
    :attr:`~newton.Model.shape_scale` and rounded to float32 with its heights, as ``ModelBuilder.finalize`` stores them
    for Newton's collision. An open surface; none for a field of fewer than two rows or columns or without a source,
    which Newton's collision ignores too."""
    field = model.shape_source[s]
    if field is None or field.nrow < 2 or field.ncol < 2:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    nrow, ncol = int(field.nrow), int(field.ncol)
    sx, sy, sz = (float(k) for k in model.shape_scale.numpy()[s])
    hx, hy, min_z, max_z = (
        float(np.float32(k)) for k in (abs(field.hx * sx), abs(field.hy * sy), field.min_z * sz, field.max_z * sz)
    )
    h = np.asarray(field.data, np.float32).astype(np.float64).reshape(nrow, ncol)
    y, x = np.meshgrid(np.linspace(-hy, hy, nrow), np.linspace(-hx, hx, ncol), indexing="ij")
    v = np.stack([x, y, min_z + h * (max_z - min_z)], axis=-1).reshape(-1, 3)
    r, c = np.meshgrid(np.arange(nrow - 1), np.arange(ncol - 1), indexing="ij")
    p00, p10 = r * ncol + c, r * ncol + c + 1
    p01, p11 = p00 + ncol, p10 + ncol
    f = np.stack([np.stack([p00, p10, p11], -1), np.stack([p00, p11, p01], -1)], axis=2).reshape(-1, 3)
    return v, f.astype(np.int64)


def _winding_signs(tris: np.ndarray, rows: np.ndarray, names: np.ndarray) -> np.ndarray:
    """For bending edge rows ``(o0, o1, v1, v2)``: 1 where the triangle ``(o0, v1, v2)`` of ``tris`` runs the edge
    ``v1 -> v2``, -1 where it runs ``v2 -> v1``.

    Newton's rest angle of a row is the signed angle between the normals of ``(o0, v1, v2)`` and ``(o1, v2, v1)`` about
    ``v2 - v1`` (``edge_rest_angle``). Where ``o0``'s triangle runs ``v1 -> v2``, as in every row of ``MeshAdjacency``,
    these are the normals by the triangles' winding and the angle is the one libuipc takes; where it runs
    ``v2 -> v1``, both normals are reversed and the angle is its negative. Raises ``ValueError``, naming the edge by
    its row in ``names``, for a row whose triangles are not among ``tris`` or run its edge the same way, which has no
    angle by the winding.
    """
    record = [("a", np.int64), ("b", np.int64), ("c", np.int64)]

    def keys(t):
        return np.ascontiguousarray(np.sort(np.asarray(t, np.int64), axis=1)).view(record).reshape(-1)

    table = keys(tris)
    order = np.argsort(table, kind="stable")
    v1, v2 = rows[:, 2], rows[:, 3]

    def runs_forward(opposite):
        """Whether the triangle of ``opposite`` and the edge runs ``v1 -> v2``, and whether there is one."""
        wanted = keys(np.stack([opposite, v1, v2], axis=1))
        at = np.minimum(np.searchsorted(table[order], wanted), len(order) - 1)
        found = table[order[at]] == wanted
        t = tris[order[at]]
        return np.any((t == v1[:, None]) & (np.roll(t, -1, axis=1) == v2[:, None]), axis=1), found

    forward0, found0 = runs_forward(rows[:, 0])
    forward1, found1 = runs_forward(rows[:, 1])
    for bad, why in (
        (~(found0 & found1), "names opposite particles whose triangles with the edge are not both cloth triangles"),
        (forward0 == forward1, "has two triangles that run the edge in the same direction, so the edge has no rest "
         "angle by their winding; wind the cloth's triangles consistently"),
    ):  # fmt: skip
        k = np.flatnonzero(bad)
        if len(k):
            r = rows[k[0]]
            raise ValueError(
                f"SolverIPC: bending edge {int(names[k[0]])} (particles {int(r[2])} and {int(r[3])}, opposite "
                f"{int(r[0])} and {int(r[1])}) {why}."
            )
    return np.where(forward0, 1.0, -1.0)


# Copies of Newton's rest-shape helpers (newton/_src/geometry/rest_shape.py), which have no public name. Newton's
# builders compute tri_poses and edge_rest_angle with them; _rest_edges and _add_cloth compute them again to tell
# whether a cloth was built at rest. test_rest_shape_copies_match_newtons_builders fails when they no longer reproduce
# what the builders store.


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors along the last axis, leaving zero vectors unchanged."""
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    return vectors / lengths


def _triangle_rest_geometry(qp: np.ndarray, rp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return triangle rest bases and areas from two edges, preserving input precision."""
    normal = _normalize_vectors(np.cross(qp, rp))
    e1 = _normalize_vectors(qp)
    e2 = _normalize_vectors(np.cross(normal, e1))
    frame = np.stack((e1, e2), axis=-1)
    edges = np.stack((qp, rp), axis=-1)
    basis = np.swapaxes(frame, -2, -1) @ edges
    return basis, np.linalg.det(basis) / 2.0


def _dihedral_angles(
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    x4: np.ndarray,
    x3_other: np.ndarray,
    x4_other: np.ndarray,
) -> np.ndarray:
    """Return signed dihedrals for triangles (x1, x3, x4) and (x2, x4_other, x3_other).

    The second triangle can use separate copies of the shared edge for disjoint rest
    topology. Welded geometry passes x3 and x4 for both copies.
    """
    n1 = _normalize_vectors(np.cross(x3 - x1, x4 - x1))
    n2 = _normalize_vectors(np.cross(x4_other - x2, x3_other - x2))
    edge = _normalize_vectors(x4 - x3)
    cos_theta = np.clip((n1 * n2).sum(axis=-1), -1.0, 1.0)
    sin_theta = (np.cross(n1, n2) * edge).sum(axis=-1)
    return np.arctan2(sin_theta, cos_theta)


def _rest_edges(x: np.ndarray, tris: np.ndarray, poses: np.ndarray) -> np.ndarray | None:
    """``ArrayScene.add_shell``'s ``rest_edges`` of the cloth triangles ``tris`` from Newton's ``tri_poses`` ``poses``.

    Newton's builders compute a triangle's rest edges in float64 from ``particle_q`` (``_triangle_rest_geometry``, in
    the edge frame ``add_shell`` takes) and store their inverse in float32. A triangle whose ``poses`` are the float32
    rounding of the inverse of its rest edges at ``x`` (``particle_q`` in float64) takes those rest edges, any other
    the inverse of its ``poses``. None when every triangle has the rest shape of ``x``: the engine then takes the rest
    shapes from the rest positions, as libuipc takes them from its rest geometry.
    """
    p = x[tris[:, 0]]
    edges, area = _triangle_rest_geometry(x[tris[:, 1]] - p, x[tris[:, 2]] - p)
    start = area > 0.0
    inverse = np.zeros_like(edges)
    inverse[start] = np.linalg.inv(edges[start])
    start &= np.all(inverse.astype(np.float32) == poses, axis=(1, 2))
    if start.all():
        return None
    rest = np.linalg.inv(poses.astype(np.float64))
    rest[start] = edges[start]
    return rest


def _closed_volume(v: np.ndarray, f: np.ndarray) -> float:
    """Enclosed volume of a closed triangle mesh (divergence theorem), either orientation."""
    if len(f) == 0:
        return 0.0
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


class SolverIPC(SolverBase, CouplingInterface):
    """Incremental Potential Contact solver for soft bodies, cloth and rigid bodies.

    Wraps the `libuipc-warp <https://github.com/OrangeEarth15/libuipc-warp>`_ engine,
    a Warp implementation of libuipc, through its array interface (no libuipc
    build is needed). Each :meth:`step` is one implicit time step, by default
    of implicit Euler (BDF1): a Newton iteration with a backtracking line
    search, a block-Jacobi PCG linear solve, barrier contact and continuous
    collision detection. ``integrator="bdf2"``,
    ``contact_constitution="al-ipc"`` and ``fem_preconditioner="mas"`` take
    libuipc's BDF2, its augmented-Lagrangian contact and its MAS
    preconditioner instead (see **BDF2**, **AL-IPC contact** and **MAS**
    below). The Newton loop, the PCG loop and the line search run on the
    device under ``wp.capture_while``, so a step never synchronizes with the
    host when it is captured in a CUDA graph. A step that is not captured
    captures the solve itself and checks how it ended (see below).

    Supported model content:

    - particles (a particle on no cloth or tetrahedron is a sphere of its
      radius, see **Contact**) and tetrahedra. Elasticity is the SNK1 stable Neo-Hookean
      model with Lamé parameters ``tet_materials[:, 0]`` (``k_mu``) and
      ``tet_materials[:, 1]`` (``k_lambda``). ``k_damp`` is ignored.
    - cloth: triangles of :attr:`~newton.Model.tri_indices` that are not on a
      tetrahedral body become libuipc's strain-limiting Baraff-Witkin membrane,
      and bending edges of :attr:`~newton.Model.edge_indices` become
      discrete-shell hinges (see the mapping below).
    - springs between particles: libuipc's Hookean springs with Newton's
      energy :math:`\\tfrac{1}{2} k_e (l - L_0)^2` (``spring_stiffness``,
      ``spring_rest_length``), and libuipc's soft vertex stitch for a rest
      length of 0. A spring adds no contact primitive: the particles it joins
      collide as other particles do.
    - particle-body attachments: libuipc's soft vertex stitch between the
      particle and a point of the body (see **Attachments** below).
    - rigid bodies with box, sphere, capsule, cylinder, cone, ellipsoid or mesh
      collision shapes become affine bodies (see below).
    - contact against static plane shapes, each treated as an infinite half
      space, and against other static shapes, which become fixed triangle
      meshes; a height field with Newton's collision triangles, two per cell
      of its grid (an open surface, as Newton's).
    - contact between all of these: point-triangle and edge-edge barriers with
      continuous collision detection, between the pairs that Newton's
      collision groups, worlds, collide flags and filter pairs let collide,
      with the friction of the two materials (see **Contact** below). Each
      connected group of tetrahedra or cloth triangles is one body and
      collides with itself (as finite elements do in libuipc), so ``d_hat``
      must stay below the spacing of non-adjacent surface primitives at rest,
      as the default ``d_hat`` does (see the note below). A rigid body does
      not collide with itself.
    - kinematic particles (zero mass or not :attr:`~newton.ParticleFlags.ACTIVE`)
      and kinematic bodies (:attr:`~newton.BodyFlags.KINEMATIC`, zero mass
      or zero inverse mass, see **Coupling**).
      By default they follow the poses written to ``state_in`` every step; the
      jump is not swept by collision detection. With
      ``kinematic_motion="aim"`` kinematic bodies move inside the step toward
      the pose that ``state_in`` prescribes for its end, swept like every other
      motion (see **Kinematic bodies** below). A particle that is not active
      and on no cloth or tetrahedron collides with nothing, as in Newton.
    - revolute, prismatic, fixed and ball joints between bodies or to the
      world, with drives, passive springs and damping, limits and
      :attr:`~newton.Control.joint_f` (see **Joints** below); ``joint_q`` and
      ``joint_qd`` are written at every step. With ``enable_d6_joints``, D6
      joints that lock what one of these locks, or nothing; with
      ``enable_distance_joints``, distance joints of one length or none.
    - external forces: :attr:`~newton.State.particle_f`, :attr:`~newton.State.body_f`
      and :attr:`~newton.Control.joint_f`, held constant over the step as
      libuipc holds its external forces (a constant acceleration in the
      implicit-Euler prediction). A body wrench acts at the centre of mass;
      its torque does work on the rotations of the affine body and none on its
      stretch.
    - gravity per world: a particle or body falls with the
      :attr:`~newton.Model.gravity` of its world, a global one with the
      global world's (the last entry).
    - lagged friction on every contact.

    Not supported, and rejected when the solver is constructed:
    :class:`NotImplementedError` for springs and particle-body attachments
    with damping (libuipc has no damping element), springs of a negative
    rest length or stiffness or that join a particle to itself, mimic
    and equality constraints, tendons, muscles, non-zero triangle or
    tetrahedron activations, the joint types and properties that **Joints**
    below rejects,
    gradients (``model.requires_grad``), with independent worlds what
    **Worlds** below refuses (an unknown of the global world, an element that
    joins two worlds, and the options they do not take yet), a
    :attr:`~newton.Model.shape_contact_pairs` array assigned to the model, more
    contact elements than the engine takes (see **Contact** below),
    collision meshes with a surface velocity (``Mesh.enable_surface_velocity``,
    which the solver's own contacts would ignore), and a CPU device.
    Gaussian shapes and planes attached to bodies raise when the shapes are
    triangulated. The ``contacts`` argument of
    :meth:`step` is not read: the solver detects its own contacts, and
    writes them into it when ``CONTACT_F`` is requested (see **Contact
    output** below).

    Not read: the penalty contact parameters of shapes and particles
    (``ke``, ``kd``, ``kf``, ``ka``, ``particle_cohesion`` and
    ``particle_adhesion``: ``kappa`` holds for every pair), restitution,
    torsional and rolling friction, ``shape_material_mu_static``,
    ``shape_gap`` (``d_hat`` is the solver's detection distance), the
    :attr:`~newton.Model.particle_radius` of cloth and tetrahedron vertices
    (see **Contact** below), damping coefficients,
    ``spring_control``, drag, lift, ``edge_rest_length`` (see below), the
    activations of a :class:`~newton.Control`, ``joint_target_mode``,
    ``joint_velocity_limit``, ``joint_limit_kd`` (but with
    ``enable_joint_limit_damping``) and ``joint_act``.

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
    lift are ignored: implicit Euler supplies the only damping.

    Rest shapes are Newton's. A membrane triangle's rest edges are the inverse
    of its :attr:`~newton.Model.tri_poses` (the edges in the triangle's frame
    whose first axis runs along its first edge), laid into the plane of the
    triangle at :attr:`~newton.Model.particle_q`, first edge along first edge,
    where it takes libuipc's material axes. Newton's builders compute
    ``tri_poses`` in float64 from ``particle_q`` and store them in float32: a
    triangle whose ``tri_poses`` are that rounding takes its rest edges at
    ``particle_q`` in float64, and when every triangle does, as for a cloth
    built at rest, the engine takes them from ``particle_q`` as libuipc takes
    them from its rest geometry, so that the membrane starts without strain.
    A hinge's rest angle is its :attr:`~newton.Model.edge_rest_angle`, the
    signed angle between the normals of ``(o0, v1, v2)`` and ``(o1, v2, v1)``
    for the row ``(o0, o1, v1, v2)``: libuipc's angle by the winding of the
    two triangles where ``o0``'s triangle runs ``v1 -> v2``, as in the rows
    Newton's cloth builders make, and its negative where a row given by hand
    has ``o0``'s triangle run ``v2 -> v1``. A row whose two triangles run the edge the same way raises
    :class:`ValueError`. Where every hinge's ``edge_rest_angle`` is the angle
    Newton's builders measure at ``particle_q``, as for a cloth built at rest,
    the engine takes the rest angles from ``particle_q`` as libuipc takes them
    from its rest geometry, with the dihedral angle its bending energy
    evaluates, so that every hinge starts at its rest angle: libuipc's angle,
    the arccosine of the normals' cosine, reads up to 1.5e-8 rad on a flat
    hinge that Newton measures as 0. The hinges keep the rest lengths and
    areas of ``particle_q`` (libuipc's :math:`L_0` and :math:`\\bar h`) where
    Newton's VBD takes ``edge_rest_length``; for a rest shape equal to the
    start the two agree.

    **Contact.** Which pairs collide, and with which friction, is libuipc's
    contact tabular: every vertex and half plane is in a contact element, and
    every pair of elements has a friction coefficient and a switch. Shapes that
    Newton's rules cannot tell apart share an element (the same collision
    group, world, collide flags and friction coefficient; with several
    worlds the subscenes keep the worlds apart, see **Worlds**, and the world
    leaves the signature); a shape in a filter pair gets one of its own, which
    with several worlds its copies in replicated worlds share. Free particles
    (in no triangle and no
    tetrahedron) are in one element, or in another one that collides with
    nothing when they are not :attr:`~newton.ParticleFlags.ACTIVE`, and the
    vertices of cloth and tetrahedra in a third. Two shapes collide as
    :attr:`~newton.Model.shape_contact_pairs` says: both collide with shapes,
    their collision groups and worlds are compatible and they are not a filter
    pair; shapes of one body never collide, and the solver checks at
    construction that its pairs are exactly the model's. A particle collides
    with a shape that has :attr:`~newton.ShapeFlags.COLLIDE_PARTICLES`, two
    free particles or a free particle and a cloth or tetrahedron vertex
    collide when the model has more than one particle and a positive
    :attr:`~newton.Model.particle_max_radius` (as Newton's particle solvers
    decide), and cloth and tetrahedra always collide with themselves and with
    each other. The friction of two shapes is the arithmetic mean of their
    :attr:`~newton.Model.shape_material_mu`, of a particle and a shape the mean
    of :attr:`~newton.Model.soft_contact_mu` and the shape's, and of two
    particles :attr:`~newton.Model.particle_mu`, as in Newton's semi-implicit,
    Featherstone and XPBD solvers. The engine tests the table of the elements
    and that of the worlds (see **Worlds**) apart, as libuipc does.

    Contact thickness, libuipc's distance at which the barrier diverges: a
    free particle is a sphere of its :attr:`~newton.Model.particle_radius`, so
    it rests its radius above a shape and two free particles rest the sum of
    their radii apart; a shape's vertices take its
    :attr:`~newton.Model.shape_margin` (a static plane's surface is moved out
    by it), so that two shapes rest the sum of their margins apart, as
    Newton's margins add up; and the vertices of cloth and tetrahedra take the
    solver's ``thickness`` argument. libuipc gives a vertex's thickness to
    every edge and triangle it is on, and finite elements collide with
    themselves, so a cloth vertex of radius r would keep the cloth's
    non-adjacent triangles and any other cloth 2r away: Newton's radii are
    sphere radii, often larger than the spacing of a cloth, and Newton's cloth
    solvers do not use them for cloth self-contact either.

    Contact stiffness, the barrier's :math:`\\kappa`, is ``kappa`` for every
    pair, taken as given. libuipc does not take a stiffness as given: its
    contact manager keeps it in a range :math:`[\\kappa_{lo}, 100\\,\\kappa_{lo}]`
    with :math:`\\kappa_{lo} = c\\,\\bar m / \\Delta t^2`, :math:`\\bar m` the mean
    mass of libuipc's vertices and :math:`c` a factor of the scene's diagonal
    and ``d_hat`` (Stiff-GIPC's estimate); a scene whose default contact model
    the user has not set takes :math:`\\kappa_{lo}`, and a stiffness the user
    gives is clamped into the range. libuipc-warp's pyuipc front end does the
    same. With ``adaptive_kappa=True`` the solver does too: ``kappa=None``
    takes :math:`\\kappa_{lo}`, a given ``kappa`` is clamped into the range and
    the first step that clamps it warns with :class:`~newton.NewtonWarning`.
    The vertices are then every particle, every vertex of the triangulated
    surfaces of bodies and static shapes, and one per plane; each body's mass
    counts once, static shapes and planes have none. The range follows the
    ``dt`` of each step, also of a captured one, and is computed from the
    masses when the solve is built: a change of mass in place
    (:meth:`notify_model_changed`) keeps it, as libuipc computes it once at
    init. Every contact pair takes one value, which :meth:`contact_stiffness`
    gives for a step of any ``dt``; with independent worlds every world takes
    the range of the same world alone, from its own vertices and the global
    ones, and every pair of a world its value (see **Worlds**).

    **BDF2.** ``integrator="bdf2"`` is libuipc's BDF2 (its ``integrator/type``,
    taken as libuipc and its pyuipc front end take it): the step predicts

    .. math::

        \\tilde x = \\tfrac{4}{3} x_n - \\tfrac{1}{3} x_{n-1}
            + \\tfrac{2}{9} \\Delta t (4 v_n - v_{n-1}) + \\tfrac{4}{9} \\Delta t^2 a,

    weighs the inertia :math:`\\tfrac{9}{4}` times as much against the potentials
    as implicit Euler does, and ends at
    :math:`v_{n+1} = (3 x_{n+1} - 4 x_n + x_{n-1}) / (2 \\Delta t)`: second order,
    with less numerical damping than implicit Euler. Contact so loses less
    energy and bodies take longer to settle: a cube dropped onto the ground in
    steps of 1/1800 s rebounds at 0.93 of its impact speed, at 0.47 under
    implicit Euler, as in libuipc. The history :math:`(x_{n-1}, v_{n-1})` is
    the solver's, not the state's: a step from the state the solver wrote last
    continues it, and every other step is an implicit Euler step whose start
    becomes the history: the first step, a step from a state the solver did not
    write last (also one it wrote earlier) and the step after :meth:`reset`. A
    step from a new state so applies the whole impulse of its forces, as in
    Newton's other solvers. (libuipc starts the history from rest, without
    acceleration: a constant force from rest gets two thirds of its impulse in
    the first step, and the velocity lags half a step's impulse behind for
    good.) Within a history the velocity lags the same way behind a force that
    jumps: gravity changed by :math:`\\Delta g` through
    :meth:`notify_model_changed` leaves it :math:`\\Delta t \\, \\Delta g / 2`
    behind. With several worlds a new state in one world, or its reset, makes
    the step an implicit Euler step of that world's particles and bodies, and
    the other worlds go on with BDF2. As in libuipc the coefficients assume
    one ``dt`` for consecutive steps: a step of another ``dt`` is taken as it
    is. The drives, passive springs and damping of joints take BDF2's velocity
    at the end of the step (see **Joints**); the default body stiffness follows
    the rule of **Rigid bodies**.

    **AL-IPC contact.** ``contact_constitution="al-ipc"`` is libuipc's
    augmented-Lagrangian contact (its ``contact/constitution``): each Newton
    iteration linearizes the constraint
    :math:`d \\ge \\hat d + \\text{thickness}` of the active contact pairs, with
    a penalty that follows the masses and multipliers that converge to the
    contact forces, and a collision-free sweep moves the configuration toward
    the solution. Contact rests at ``d_hat`` beyond the contact thickness,
    where the barrier rests inside ``d_hat``. Its constraints hold between
    surfaces of vertices with mass, the penalty of a pair scaling with the
    smallest mass of its vertices: particles on no cloth or tetrahedron that
    collide are refused, and so are static mesh shapes, particles and bodies
    without mass that collide (Newton's static shapes have no mass; static
    planes are taken). A pair joins the active set only when the motion of a
    step sweeps into it: a kinematic body moved between steps pushes nothing,
    and what moves into it collides with it. A fast impact runs past the
    constraint and rebounds faster than it came, as in libuipc: a cube dropped
    0.4 m onto the ground in steps of 1/1800 s leaves at 4.5 times its impact
    speed. ``kappa`` is not used, and ``adaptive_kappa`` is refused. The step
    ends on AL-IPC's own criterion, which has no Newton residual:
    ``SOLVER_RESIDUAL`` is not offered. A change of the bodies' masses
    (:meth:`notify_model_changed`) builds the solve again, since the penalty
    follows the masses the solve was built with.

    **MAS.** ``fem_preconditioner="mas"`` is libuipc's multilevel additive
    Schwarz preconditioner (its ``linear_system/fem_preconditioner``) on the
    vertices of particles, cloth and tetrahedra (bodies
    keep their 12 x 12 blocks): it inverts the system exactly on clusters of up
    to 16 vertices and corrects between them on coarser levels made of
    neighbouring clusters. libuipc takes the clusters from METIS; libuipc-warp's
    array path makes them itself, by recursive inertial bisection of every
    connected cloth, soft body and spring network at its rest positions. The
    preconditioner is libuipc's on another partition: the PCG iterations
    differ from libuipc's, and the result is the same to the tolerance of the
    linear solve. Particles on no element keep their 3 x 3 blocks.

    **Worlds.** Particles, bodies and colliding shapes in more than one local
    world are simulated as independent worlds, Newton's meaning of worlds:
    every world takes the step it would take alone, with its own Newton
    iterations, line searches and step lengths, convergence test, linear
    solves and status, as libuipc-warp's independent worlds solve them
    (libuipc-warp's own; libuipc solves the subscenes of a scene together).
    Every local world is a libuipc subscene of its own and the global entities
    (world -1) one more, which meets every world, as Newton's collision rules
    say: no contact pair joins two worlds, also where the worlds overlap in
    space, and a global ground or static shape acts in every world. The worlds
    share one frame graph, one collision detection and one set of contact
    lists: a step runs as many Newton iterations as its slowest world, and a
    world that has converged is held where it is while the others iterate. A
    world takes the Newton iterations of the same world alone and agrees with
    it to the rounding of the engine's sums, which add in another order than in
    an engine of that world alone, as two runs of one engine already differ by
    the rounding of its atomics; where a step carries rounding far, as a stiff
    landing does at the default tolerances, the two part as far as a run alone
    parts from one started a rounding error away.

    A world that fails (a non-finite value, a contact pair at or inside its
    thickness, a CCD fraction that is not positive) fails alone: it is put
    back at its state at the step's start, its friction discarded and its BDF2
    history started anew, and the other worlds take their step. ``STATUS`` and
    ``SOLVER_RESIDUAL`` hold every world's own bits and residual at its entry;
    the entries of the global world and of worlds that hold no unknowns (only
    static or kinematic content) stay 0. A step that is not captured and fails
    in some worlds writes ``state_out``, those worlds at the step's start and
    the others at their step, and raises :class:`RuntimeError` naming the
    worlds and their bits; :meth:`check_status` names them too. The worlds
    share the contact lists, and every world has a quota of them, sized as for
    that world alone (libuipc-warp's capacity per world): a world whose
    contact asks for more than its quota fails alone in the same way, while
    the other worlds keep every contact pair and take their step. An eager
    step then grows that world's quota and solves the step again for every
    world; a captured step sets ``OVERFLOW`` in that world's entry alone, and
    :meth:`grow` grows its quota. A reset with a ``world_mask``, or
    a new state written for one world, discards that world's friction and
    starts its BDF2 history anew and leaves the other worlds as they were; the
    mask's global entry selects nothing, since the global entities are no
    unknowns. :meth:`frame_stats` gives the counts of the step's loops: the
    Newton iterations of its slowest world. With ``adaptive_kappa=True`` every
    world takes libuipc's stiffness range of the same world alone, from the
    mean mass and the extent of its own vertices and the global ones
    (libuipc-warp's stiffness per world): its own lower bound with
    ``kappa=None``, a given ``kappa`` clamped into its own range, at every
    step's ``dt``; :meth:`contact_stiffness` gives every world's.

    The global entities do not move: a free particle (mass and
    :attr:`~newton.ParticleFlags.ACTIVE`), a dynamic body or a coupled solver's
    proxy of world -1, and with ``kinematic_motion="aim"`` a global kinematic
    body with colliding shapes, would be an unknown that every world pushes,
    and are refused with :class:`NotImplementedError`, as Newton's implicit MPM
    refuses global colliders backed by dynamic bodies with
    ``separate_worlds``; so is a triangle, tetrahedron or spring that joins two
    worlds. Global static shapes and planes, kinematic particles and kinematic
    bodies (``kinematic_motion="jump"``) act in every world. Not offered with
    independent worlds yet, where two worlds or more move, and refused naming
    what to use instead: ``contact_constitution="al-ipc"`` and
    ``fem_preconditioner="mas"`` (its clusters may join two worlds). The
    engine's Newton exit check, which independent worlds do not take either,
    is not used by the solver.

    With ``coupled_worlds=True`` the worlds are one solve instead, with the
    same subscenes: one step length (the smallest time of impact over all
    worlds, and a line search on the sum of their energies), one Newton
    convergence test (the largest displacement over all worlds), one PCG solve
    and one set of contact lists. A world's result therefore depends on the
    others through the iteration counts and the step lengths, and a failure or
    an overflow in one world fails the step in every world, whose entries all
    get the bits of the one solve. Copies of one world take the iterations of
    one copy, and each follows the run of that world alone to the rounding of
    the shared sums. Global particles and bodies may move, as unknowns of the
    one solve that meet every world, and the options above are taken. A reset
    of one world, or a new state written for one world, discards the friction
    of that world's contacts and starts its BDF2 history anew, and leaves every
    other world's friction and history as they were, bit for bit
    (libuipc-warp's requests per subscene; libuipc discards friction for a
    whole scene); a reset of the global entities discards the friction of
    every pair with a global entity; with ``adaptive_kappa`` the stiffness
    range is that of the whole scene.

    A model with content in one local world is solved as one world either
    way. The engine takes up to 32,768 subscenes, and contact elements times
    the subscene count rounded up to a power of two below :math:`2^{31}`; the
    start check forms no pair of parts of two worlds.

    **Kinematic bodies.** By default (``kinematic_motion="jump"``) a
    kinematic body is a fixed affine body whose pose the solver writes from
    ``state_in`` before the step: the jump between two steps is not swept by
    collision detection, so a jump larger than the room the barrier leaves
    puts surfaces into each other, and friction never sees the motion (a
    platform sliding under a box does not carry it). With
    ``kinematic_motion="aim"`` every kinematic body with colliding shapes is
    built as libuipc builds an animated body: an affine body that is an
    unknown of the step, with no gravity and no external force, pulled by
    libuipc's soft transform constraint (constitution 16) toward its aim, the
    pose of ``state_in`` advanced by its twist over the step (the centre of
    mass by :math:`v \\Delta t`, the rotation by the exponential of
    :math:`\\omega \\Delta t`). Its velocity is set so that the step's
    prediction is the aim, so its motion is part of the Newton direction: the
    CCD sweeps it, the line search keeps every surface apart, friction sees
    it, and a body that contact blocks stops short of its aim. The aim is
    soft, with the strength ratio :math:`\\eta` = ``kinematic_strength_ratio``
    (libuipc's motors' 100) for translation and rotation alike: under a load
    :math:`F` a body of mass :math:`m` lags its aim by about
    :math:`F \\Delta t^2 / ((1 + \\eta) m)`, 2.8e-5 m for 10 N on 1 kg at
    1/60 s. A kinematic body moves with Newton's mass; one of mass 0 takes the
    mass of its collision meshes at 1000 kg/m^3. A caller who writes
    consistent ``body_q`` and ``body_qd`` every step gets the next pose as the
    aim, one who writes poses only (``body_qd`` 0) the written pose, one who
    writes nothing motion at constant twist. The step starts from the pose the
    body reached, so the motion from there to the aim is swept, a jump
    included; the first step after the solver is built and the step after a
    :meth:`reset` that selects the body's world take ``state_in`` instead,
    which is how a caller moves such a body without sweeping. ``state_out``
    holds the pose the body reached and its twist, not a copy of
    ``state_in``. Pairs of two shapes that are each kinematic or static do not
    collide, as Newton's collision pipeline can filter them
    (``include_static_kinematic_pairs=False``) and as they never collide in
    the default mode, where neither moves inside the step; the solver checks
    its pairs against :attr:`~newton.Model.shape_contact_pairs` less those
    pairs. Kinematic particles keep the jump (a soft aim would make a pinned
    vertex sag). Not with ``contact_constitution="al-ipc"``.

    **Attachments.** A particle-body attachment
    (:meth:`~newton.ModelBuilder.add_attachment_particle_body`) is libuipc's
    soft vertex stitch (constitution 22) of rest length 0 between the particle
    and an anchor of the body's affine body at ``attachment_particle_body_body_point``
    in Newton's body frame: a point that moves with the body and is no contact
    primitive, as the interior vertices of an affine-body mesh are in libuipc;
    attachments at one point of one body share it. Its energy in the step's
    incremental potential is :math:`\\tfrac{1}{2} k \\Delta t^2 |x_p - x_a|^2`, the
    spring :math:`\\tfrac{1}{2} k |x_p - x_a|^2` of implicit Euler with
    :math:`k` = ``attachment_particle_body_stiffness``, so that the particle
    and the body take equal and opposite forces, the body with the torque the
    anchor's offset gives, and a disabled attachment
    (``attachment_particle_body_enabled``) takes none. A hard attachment
    (infinite stiffness), which libuipc has no constitution for, takes the
    penalty of a ball joint (see **Joints**): :math:`K/\\Delta t^2` with
    :math:`K` = ``joint_strength_ratio`` times the masses of the particle,
    if it is not fixed, and of the body, if it is dynamic, so that it holds a
    load :math:`F` within :math:`F \\Delta t^2 / K`. The stiffness and the
    enabled flags are read at every step: a captured step takes new values at
    its next replay. A spring of rest length 0 is the same stitch between its
    two particles, with ``spring_stiffness``. Damping is rejected: libuipc's
    stitch has none, and an implicit damping cannot be folded into it the way
    the joint drives fold theirs. An attachment does not change which pairs
    collide: an attached particle collides with its body as with any other,
    so a particle attached on the body's surface starts in contact (see the
    note on the start below); place it a gap away.

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
    the stiffness :math:`\\kappa` (below) and the volume :math:`V` of the body's
    collision meshes keeps :math:`A` within a small strain of a rotation; this
    is libuipc's near-rigid body, not an exactly rigid one. Within a step a
    turning body moves along a straight line in its 12 coordinates and the
    orthogonality potential pulls it back onto the rotations, so under implicit
    Euler a free spin at :math:`\\omega` loses about :math:`(\\omega \\Delta t)^2`
    of its angular velocity per step: 1e-4 at 1 rad/s and :math:`\\Delta t` = 0.01 s.

    The stiffness is ``body_kappa`` at every step when it is given. By default
    it is libuipc's 1e8 Pa, lowered for a body and a step to
    :math:`\\kappa = 25\\, I_{max} / (V \\Delta t^2)` where that is lower,
    :math:`I_{max}` being the body's largest principal moment of inertia. The
    straight line to a turn :math:`\\theta` stretches the body by about
    :math:`\\theta^2 / 2`, and the first Newton iteration of a step takes the
    turn in full while that stretch's energy,
    :math:`2 \\theta^4 \\kappa V \\Delta t^2`, stays below the inertia term
    :math:`\\theta^2 I / 2` of the incremental potential that it removes,
    :math:`I` being the moment of inertia about the axis of the turn. With the
    lowered stiffness the first iteration takes a turn of up to 0.1 rad about
    the axis of the largest moment and of up to
    :math:`0.1 \\sqrt{I / I_{max}}` rad about an axis of moment :math:`I`:
    0.1 rad about any axis of a compact body, less about the long axis of a
    thin one, which keeps its stiffness along its length. With a stiffer body
    the line search cuts the iteration short, and the next Newton directions,
    held back by the stretched potential, become small enough to pass the
    displacement test of the Newton loop long before the turn is complete: at
    1e8 Pa a 1 m cube that a torque turns by 0.06 rad in a 0.1 s step ends at
    31% of its angular velocity, and a 10 cm cube spinning at 0.6 rad/s keeps
    33% of it over six steps of 1/60 s, where the converged solve keeps 99.9%.
    The lowered stiffness costs rigidity in proportion to :math:`\\Delta t^2`
    where it applies: the lowest of five 10 cm cubes stacked on the ground is
    compressed by 4e-5 at :math:`\\Delta t` = 1/600 s and by 4e-3 at 1/60 s
    (6e-6 at 1e8 Pa), a 30 cm by 1 cm finger swinging from a hinge stretches
    by up to 8e-6 and 7e-4, and a box of unequal sides that a torque starts
    turning runs ahead of a rigid one by 7e-4 of its turn after one step of
    0.01 s and by 1e-5 after ten. Pass ``body_kappa=1e8`` for libuipc's
    stiffness at every step.
    Gravity acts on :math:`p`; with the full dyadic mass this is the exact
    gravity force on the affine body.
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

    **Joints.** Revolute, prismatic, fixed and ball joints are libuipc's
    affine-body joints (constitutions 18, 20, 25 and 26), between the affine
    bodies of their parent and child, a parent of -1 being a fixed body
    without mass at the world's origin. A revolute or prismatic joint holds
    the two points of its axis, ``joint_X_p`` and ``joint_X_c`` and one metre
    along ``joint_axis`` from them, together; a prismatic joint also holds the
    bodies' relative rotation, a fixed joint the anchors and the rotation, a
    ball joint the anchors. The rotation they hold, and the zero of a joint's
    coordinate, is where the joint frames coincide, as in Newton, whatever the
    bodies' poses when the solver is built.

    The joints are penalties, not hard constraints: libuipc's energy
    :math:`\\tfrac{K}{2}|C|^2` of the joint's violation :math:`C`, a term of
    the incremental potential (not multiplied by :math:`\\Delta t^2`), with
    :math:`K` = ``joint_strength_ratio`` times the mass of the dynamic bodies
    the joint joins (the world and kinematic bodies count 0). Against a
    translation of the child, in a step of length :math:`\\Delta t`, a
    revolute or prismatic joint, which holds two points, is a spring of
    :math:`2K/\\Delta t^2` [N/m], a fixed or ball joint one of
    :math:`K/\\Delta t^2`: a load that would move the child by :math:`\\delta`
    in one step stretches the joint by about :math:`\\delta / (2\\gamma)` or
    :math:`\\delta / \\gamma`, :math:`\\gamma` = ``joint_strength_ratio``.
    Gravity at 1/60 s and the default 100 stretch a revolute joint by 1.4e-5 m
    and a ball joint by 2.7e-5 m. A larger ratio holds the joints closer and
    makes the linear system stiffer.

    Drives, passive springs and passive damping of a revolute or prismatic
    joint are exact for the step's integrator: the joint's
    :math:`\\tau = k_e (q^* - q) + k_d (\\dot q^* - \\dot q) - k_s (q - q_r) - c \\dot q`
    (``joint_target_ke``, ``joint_target_kd``, ``joint_stiffness``,
    ``joint_rest_q``, ``joint_damping``; targets from
    :attr:`~newton.Control.joint_target_q` and ``joint_target_qd``), taken at
    the end of the step with :math:`\\dot q = (q - q_n)/\\Delta t` (BDF2:
    :math:`(3 q - 4 q_n + q_{n-1}) / (2 \\Delta t)`), is minus the
    gradient of one quadratic potential in :math:`q`, which the solver gives
    libuipc's driving term every step. The gains act whatever
    ``joint_target_mode`` says, as in the other maximal-coordinate solvers.

    With ``enable_joint_armature=True`` a revolute or prismatic joint's
    ``joint_armature`` :math:`a`, which Featherstone and MuJoCo add to the
    joint's inertia, is the kinetic term of its coordinate in the same
    potential: :math:`\\tfrac{a}{2 \\Delta t^2} (q - \\tilde q)^2` with
    :math:`\\tilde q = q_n + \\Delta t \\dot q_n` (BDF2: 9/4 of it, with
    :math:`\\tilde q = \\tfrac{4 q_n - q_{n-1}}{3} + \\tfrac{2 \\Delta t}{9}
    (4 \\dot q_n - \\dot q_{n-1})`), as libuipc writes the inertia of every
    degree of freedom, :math:`\\dot q_n` being the rate the bodies' twists give
    at the step's start (as ``joint_qd`` is written). The armature of a ball
    or free joint, an inertia about three axes or a mass without weight,
    has no such potential and is rejected.

    The drive's force cannot be bounded inside libuipc's driving term, a
    quadratic potential. With ``enable_joint_effort_limit=True`` a
    ``joint_effort_limit`` :math:`L` below Newton's default 1e6 bounds the
    force of the gains as decided at the start of every step: where
    :math:`|k_e (q^* - q_n) + k_d (\\dot q^* - \\dot q_n)| > L`, the step takes the
    constant force :math:`\\pm L` (libuipc's joint torque or force, with
    ``joint_f``) in place of the gains' part of the potential; the passive
    spring and damping remain. The bound holds at the start of every step,
    which is where MuJoCo clamps its actuators' force; a step that starts
    below it and ends above it, or the reverse, takes the bound one step
    late, which a drive clamped at the end of the step would not (by 1.6e-4
    of the motion for a hinge driven through a step of its target).

    Limits are libuipc's cubic penalty outside ``[joint_limit_lower,
    joint_limit_upper]``, not Newton's linear spring: at an overshoot
    :math:`\\delta` it stores :math:`k_l \\delta^3 / (3 \\delta_r)` with
    :math:`k_l` = ``joint_limit_ke`` and :math:`\\delta_r` = 1e-3 (rad or m), so
    that it pushes back with the spring's :math:`k_l \\delta_r` at
    :math:`\\delta_r` and harder beyond it; a load :math:`F` rests at
    :math:`\\sqrt{F \\delta_r / k_l}` past the limit. ``joint_limit_kd`` is not
    read (libuipc's limit has no damping), a ``joint_limit_ke`` of 0 holds
    nothing, and a bound at :data:`newton.MAXVAL` is no bound. With
    ``enable_joint_limit_damping=True`` a joint whose coordinate is past one
    of its limits at the start of a step takes the damping
    :math:`-k_{ld} \\dot q` of Featherstone and SemiImplicit,
    :math:`k_{ld}` = ``joint_limit_kd``, in the same quadratic potential as
    ``joint_damping``: exact for the step's integrator, with the velocity at
    the end of the step, and decided at its start, as those solvers decide it
    (they also switch the drive off past a limit, which SolverIPC does not,
    as MuJoCo does not). A joint that reaches its limit within a step is
    damped from the next step on; a limit of ``joint_limit_ke`` 0 damps
    nothing.

    :attr:`~newton.Control.joint_f` of a revolute or prismatic joint is
    libuipc's joint torque or force between the two bodies; of a ball joint a
    torque in the parent's joint frame, on the child, the parent taking the
    reaction; of a free joint a wrench at the child's centre of mass in the
    world frame, the parent, if it is a body, taking the reaction.
    ``joint_enabled`` is read at every step, and so are the gains, targets,
    springs, damping and limits: a captured step takes new values at its next
    replay. A change of the joints' types, bodies, frames or axes rebuilds
    the solve (``notify_model_changed`` with ``JOINT_PROPERTIES``).

    After every step ``joint_q`` and ``joint_qd`` hold the coordinates
    :func:`newton.eval_ik` computes from the bodies' state, for every joint,
    but the position of a revolute joint, which is libuipc's angle and turns
    on past :math:`\\pi` as in the generalized-coordinate solvers, and of a
    prismatic joint, libuipc's distance. A revolute joint's angle continues
    from the step before; where ``state_in.joint_q`` is not what the solver
    wrote, it picks the turn of the bodies' angle (the pose decides the angle
    modulo :math:`2\\pi`).

    **D6 joints** (``enable_d6_joints=True``). libuipc has no D6 joint (its
    constitution 28 is only registered). A D6 joint whose free axes leave
    locked what one of libuipc's joints locks is that joint, exactly: no
    axis is a fixed joint, one angular axis a revolute joint about it, one
    linear axis a prismatic joint along it, with their drives, springs,
    damping, limits and ``joint_f``; three orthonormal angular axes are a
    ball joint, and three orthonormal linear and three orthonormal angular
    axes lock nothing, as a free joint. ``joint_q`` and ``joint_qd`` hold
    Newton's coordinates of the D6 joint: a revolute or prismatic one's as
    above, of three angular axes the angles :func:`newton.eval_ik` takes
    (the XYZ Euler angles in the basis of the first two axes), of linear
    axes the projections of the joint frames' offset. ``joint_f`` of a D6
    joint of three angular axes is Featherstone's generalized force: a
    torque along the dual basis of the axes as the coordinates carry them,
    and with linear axes a force along them on the child's joint frame,
    the parent taking the reaction on the same line. Other D6 joints (two
    angular axes, which lock one rotation, or linear and angular axes
    together but three and three) lock what no libuipc joint locks, and
    three axes that are not orthonormal have no coordinates in Newton's
    decomposition: both are rejected, and so are drives, springs, damping
    and limits on the D6 joints that become a ball or no joint.

    **Distance joints** (``enable_distance_joints=True``). libuipc has no
    distance joint. One whose distance has a single value ``L`` (the two
    bounds of its first degree of freedom equal and not negative) holds the
    origins of its joint frames ``L`` apart as libuipc's soft vertex stitch
    (constitution 22) of rest length ``L`` between two anchors of the
    bodies there (the world's fixed body for a parent of -1): the penalty
    of a ball joint, ``K/2 (|x_c - x_p| - L)^2`` with the same ``K``, and
    nothing when the joint is disabled. One whose bounds are both negative
    holds nothing, as Featherstone and SemiImplicit take every distance
    joint. Either takes ``joint_q``, ``joint_qd`` and ``joint_f`` as a free
    joint does. A range of distances, or a single bound, is an inequality,
    which a stitch, a spring about one rest length, cannot hold: rejected,
    and so are drives on a distance joint (SolverXPBD reads its gains as
    the compliance of the distance).

    Rejected (:class:`NotImplementedError`), each for the libuipc
    constitution it lacks: distance joints and D6 joints but those above,
    which libuipc does not have, and rod joints, which no libuipc
    constitution between two affine bodies represents; drives, springs,
    damping and limits on ball and free joints,
    since libuipc's driving term and limits act on one hinge angle or one
    distance and its ball joint has neither (its ball driving term,
    constitution 27, is only registered); ``joint_armature`` other than 0
    but with ``enable_joint_armature`` on a revolute or prismatic joint,
    since libuipc's joints have no inertia of their own; ``joint_friction``
    other than 0, since they have no friction and dry friction, a bounded
    force, is no quadratic potential; a ``joint_effort_limit`` below
    Newton's default 1e6 on a driven degree of freedom but with
    ``enable_joint_effort_limit``, since the quadratic drive cannot bound its
    force.

    **State.** The solver keeps positions and velocities in float64 and reads
    ``state_in`` on every step. A particle whose ``state_in`` position and
    velocity are the float32 rounding of the solver's state, and a body whose
    pose and twist are what the solver last wrote, keep the float64 state;
    anything else is taken from ``state_in``, a body as the rigid motion of its
    pose and twist. Taking a new state for a particle or body that is not
    kinematic discards the friction of that step's contacts (with several
    worlds, the contacts of its world), as libuipc does after a
    state is written and as a solver started from that state does.
    :meth:`reset` does the same without a new state. Output is written to
    ``state_out`` in float32.

    The solve is built at the model's state: :attr:`~newton.Model.body_q` and
    ``body_qd``, but a body whose pose there is the identity, where
    :meth:`~newton.ModelBuilder.add_link` and
    :meth:`~newton.ModelBuilder.add_body` put a body given no transform,
    starts where :func:`newton.eval_fk` puts it from
    :attr:`~newton.Model.joint_q` and :attr:`~newton.Model.joint_qd`. As for
    Newton's other maximal-coordinate solvers, the caller writes those poses
    into the state before the first step, also after building the solver;
    the model of an articulation built without poses keeps the identity
    until then, and its links may cross each other or the ground there. The
    start is checked where it is known (see the note below): on the model's
    state, and for the pairs with a kinematic body or such a link, whose
    start the caller writes, on the first step's ``state_in``.

    **Status.** Request :attr:`SolverIPC.ObservableKind.STATUS` and
    ``ObservableKind.SOLVER_RESIDUAL`` (not with ``contact_constitution="al-ipc"``)
    with :meth:`observables` and pass the
    container to :meth:`step`: the step writes a :class:`SolverIPC.StatusBits` mask and
    the residual on the device, also during graph replay. With independent
    worlds every world that holds unknowns receives its own (see **Worlds**);
    with one world, or ``coupled_worlds``, every world that owns particles or
    bodies receives the values of the one solve. The
    residual is the convergence measure of libuipc's Newton loop: the largest
    displacement of the last Newton direction (the rows of an affine body's
    :math:`A` scaled by ``newton_velocity_tol / newton_transrate_tol``) over
    ``newton_velocity_tol * dt``. It is dimensionless, at most one when the
    loop ended on its displacement test, may be above one when the
    semi-implicit test ended it, and infinite when the step failed on a value
    in :class:`SolverIPC.StatusBits`.

    A step that is not captured reads the status itself after the solve, with
    one synchronization. When a contact list overflowed (with independent
    worlds, a world's quota of the lists) it enlarges the lists and solves the
    step again; when the step failed otherwise (a non-finite
    value, a contact starting at or inside its thickness, an exhausted BVH
    traversal), it raises :class:`RuntimeError` and leaves ``state_out``
    unwritten, but where independent worlds failed alone: there it writes
    ``state_out`` first, the failed worlds at the step's start and the others
    at their step (see **Worlds**). A captured step cannot do either: it sets
    the bit, and nothing else reports the failure. A caller that captures
    :meth:`step` must therefore request ``STATUS``, pass one container to each
    captured step (a container holds the bits of the last step that wrote
    it), and after every replay call :meth:`check_status` with each
    container. It raises
    :class:`RuntimeError` on a failed step and warns on an iteration limit;
    with ``grow=True`` it answers :attr:`SolverIPC.StatusBits.OVERFLOW` with
    :meth:`grow` and returns ``True``, and the caller then restores the
    state the replayed steps started from, captures again and repeats them.
    The PCG loop of a captured step re-reads its stopping test every 8 or 32
    iterations, and work whose size only the device knows starts at a guessed
    size; both are chosen when the step is captured. :meth:`graph_outdated`,
    called after replays, returns ``True`` when the steps replayed since asked
    for the other length or for other guesses, and the caller then captures
    again to run faster (the old graphs stay valid and give the same results).

    **Contact output.** Request ``SolverBase.ObservableKind.CONTACT_F`` with
    :meth:`observables`, passing the :class:`~newton.Contacts` that sizes
    ``contact_f``, and pass the container and a ``Contacts`` to :meth:`step`:
    the step writes its own contacts into that buffer and their forces into
    ``contact_f``, as a native collision backend does (SolverMuJoCo with its
    own collision detection). Do not run ``CollisionPipeline.collide`` into
    the buffer; the step overwrites its rows. Each contact term of the step
    with a force is one row: a point-triangle, edge-edge, point-edge,
    point-point or vertex / half-plane pair of the final active set (its
    barrier) or of the step's friction pairs (its lagged friction), so that a
    pair in both gives two rows, one normal and one tangential. A term between
    two shapes is a rigid row. A term between a particle feature (a particle,
    or an edge or a face of cloth or tetrahedra) and a shape is a soft row,
    with the particle, edge or face record of
    :attr:`~newton.Contacts.soft_contact_indices` and the weights of its
    contact point. Terms between particle features (cloth self-contact,
    particles among themselves) are not exported, as SolverVBD exports no
    soft self-contact.

    The forces are those at the end of the step: minus the gradient of the
    term's energy in the step's incremental potential at the final positions,
    over :math:`\\Delta t^2`, which the step balanced to the Newton tolerance.
    A box resting on the ground reports its weight. On a body of mass ``m``
    whose step ended on the displacement test the forces can be off by up to
    about ``m * newton_velocity_tol / dt``: a box creeping down a ramp under
    friction reports its weight within 1.2% at the default tolerances and
    within 1e-4 with ``newton_velocity_tol=1e-4``, ``linear_tol_rate=1e-10``
    and ``semi_implicit=False``. A rigid row's
    ``contact_f`` is the force on the body of shape 0 by the body of shape 1,
    the sum of the forces on its vertices, with their torque about that body's
    centre of mass at the end of the step (about the world origin for a static
    shape), in the world frame; a soft row's is the force on the shape's body
    by the particle feature, with its torque about that body's centre of mass.
    The points are written in the body frames of ``state_out``, so pass
    ``state_out`` to a consumer such as :class:`~newton.sensors.SensorContact`.
    Each side's point is the force-weighted mean of its vertices, the closest
    point of its primitive on the triangulated surface (a half plane's: the
    point under the vertex, on the plane's surface). The normal points from
    shape 0 toward shape 1, a soft row's from the shape toward the particle
    feature: against the force on shape 0 for a barrier term, whose force is
    the gradient of the distance, and from shape 0's point to shape 1's for a
    friction term, whose force is tangential. The margins are the shapes'
    ``shape_margin``, their contact thickness, and the offsets the normal
    times each margin toward the other side, as Newton's collision pipeline
    writes them.
    ``rigid_contact_surface_velocity`` (when allocated) and
    ``soft_contact_body_vel`` are zero, and ``soft_contact_rigid_indices`` is
    ``-1``: the solver has no surface velocity, and it triangulates and welds
    the shapes' meshes. The counts are true counts: rows past the buffer's
    capacity are dropped, as Newton's collision pipeline drops them, and a
    ``rigid_contact_count`` or ``soft_contact_count`` above the capacity tells
    it; :meth:`get_max_contact_count` and :meth:`get_max_soft_contact_count`
    give capacities that drop none, a worst case several times the usual
    count. A step that failed writes no rows. The order of the rows changes
    from run to run (atomic counters), their set and values do not. With
    several worlds the rows of every world share the buffer, and the shapes
    tell the worlds apart. With independent worlds a step in which some worlds
    failed writes no row of those worlds and the rows of the others:
    libuipc-warp's records give the terms of a world that failed no force, so
    the feedback forces of a coupled solver's proxies (see **Coupling**) in
    such a world are zero in that step, and those in the other worlds are
    their step's. Not with ``contact_constitution="al-ipc"``.
    The output costs, only when requested, one pass over the step's contact
    terms and one over its rows, and 64 bytes per slot of the contact lists.

    **Coupling.** The solver is a
    :class:`~newton.solvers.experimental.coupled.CouplingInterface`: an entry
    of Newton's coupled solvers (``SolverCoupled``, ``SolverCoupledProxy``),
    built on the entry's ``ModelView``. It reads a view as a model; three
    things that only a view holds take their meaning from the coupler. A body
    of positive mass whose inverse mass is 0, which the view makes of a body
    another entry owns, is a kinematic body: a fixed affine body whose pose
    follows ``state_in``, and whose shapes, which the view hides, collide with
    nothing. A triangle, tetrahedron or spring whose particles are all disabled
    (mass 0 and not :attr:`~newton.ParticleFlags.ACTIVE`), as the view disables
    another entry's particles, is that entry's: neither simulated nor a contact
    surface here, its particles fixed points that collide with nothing. And the
    view's :attr:`~newton.Model.shape_contact_pairs`, which the coupler filters
    by the view's collide flags, is the set the solver's rules give (see
    **Contact**). An entry therefore steps its own content as the solver alone
    does, to the rounding of its sums.

    A proxy body or particle (:attr:`~newton.BodyFlags.PROXY`,
    :attr:`~newton.ParticleFlags.PROXY`), another entry's that this entry moves
    with the view's scaled mass, is an unknown of the step. It does not collide
    with another proxy, with a static shape or with a body of inverse mass 0,
    as the interface's default ``coupling_prepare_proxy_contacts`` drops
    Newton's contacts of those pairs (the solver reads no
    :class:`~newton.Contacts`, and returns them as they are). The coupler
    writes a proxy's pose and twist at every step; unlike any other new state,
    that write keeps the friction of the step's contacts, where libuipc
    discards friction after any write of a state: a proxy would otherwise
    never have friction. A proxy's new pose is not swept: a source solver that
    moves a proxy into this entry's content by more than the room the barrier
    leaves fails the step. A proxy's feedback is the force of the step's
    contact terms on it at the end of the step, a wrench about its centre of
    mass for a body (``coupling_harvest_proxy_wrenches``,
    ``coupling_harvest_proxy_particle_forces``), as SolverVBD harvests it,
    rather than its change of momentum. The barrier's force is steep in the
    distance, so the feedback is as accurate as the step's convergence: give
    an entry with proxies converged steps (``semi_implicit=False`` and a
    ``newton_velocity_tol`` of about 1e-4 m/s). At the default tolerances the
    step of an impact can end before the barrier has relaxed, with a force
    tens of times the change of momentum it caused. With
    ``SolverCoupledProxy``'s lagged transfer, the default, a proxy resting on
    this entry's content does not come to rest: the feedback of a contact this
    stiff reaches the source one step late and keeps the proxy bouncing,
    though its average holds the proxy up; the staggered transfer
    (``mode="staggered"``) brings it to rest. Proxies are refused with
    ``integrator="bdf2"``, whose history the coupler's writes would leave
    behind, and with ``contact_constitution="al-ipc"``, which neither sweeps a
    proxy's synced pose into its active set nor exports its contact forces.
    The effective masses, gravity and the rewinds of the proxies take the
    interface's defaults.

    A coupler that solves a step more than once (a ``SolverCoupledProxy`` or
    ``SolverCoupledADMM`` with more than one iteration) starts every pass after
    the first from the step's start. The solver keeps that start on the
    device: libuipc-warp's frame restart (the float64 positions and
    velocities, the BDF2 history, the active contact set the step takes its
    friction from, the step's requests to discard friction and restart the
    history, the joints' angles) and what the solver carries from step to step
    (what it wrote of the bodies and joints, the kinematic bodies it holds
    toward their aims, the joints' BDF2 history). The step after a coupler's
    write of the input state copies the start, and the step after a restart's
    notification copies it back: device copies, inside a captured graph too.
    Every pass therefore starts from the same float64 state with the same
    friction, where taking ``state_in``'s float32 rounding would discard the
    friction, and a pass with the inputs of the one before repeats it, bit for
    bit where a step repeats itself (where sums add with atomics, to their
    rounding). With independent worlds every world starts again, also one
    that failed in the pass before. An entry with several substeps per coupled
    step keeps the start of the first. The copy holds the vertices' state, the
    active contact lists and the joints' coordinates, and a solver whose
    coupler does not restart allocates none of it. The coupler steps its
    entries without observables, so the solver as an entry keeps the status
    bits of its steps, captured or not, with independent worlds a word for
    every world entry, and :meth:`check_status` without a container reports
    them, naming the worlds, and clears them; a restarted pass replaces the
    bits of the pass before it, so a coupled step reports its last pass. A
    reset through the coupler is :meth:`reset` with the coupler's world mask.
    Of a view that the coupler cannot compact, the other entries' joints are
    in the view, disabled, and are checked as the solver's own.

    This solver requires the optional ``libuipc-warp`` package (``import uipc_warp``).

    Args:
        model: The model to simulate.
        kappa: Barrier stiffness [Pa] of every contact pair, taken as given:
            libuipc's contact resistance, which libuipc and its pyuipc front
            end would clamp into libuipc's range (see ``adaptive_kappa``).
            ``None``, with ``adaptive_kappa``, takes the range's lower bound,
            libuipc's default. Not used with ``contact_constitution="al-ipc"``.
        adaptive_kappa: Take the contact stiffness as libuipc and its pyuipc
            front end do: inside libuipc's scene-adaptive range, which
            follows each step's ``dt`` (see **Contact**); with independent
            worlds every world's own range, that of the world alone (see
            **Worlds**).
        d_hat: Barrier activation distance [m]. ``None`` takes libuipc's
            0.01 m, or less for a model with small colliding features (see
            the note below).
        thickness: Contact thickness of the vertices of cloth and tetrahedra
            [m]; free particles take their radius and shapes their margin
            (see **Contact**).
        newton_max_iter: Maximum Newton iterations per step.
        newton_velocity_tol: Newton convergence tolerance on ``max |dx| / dt`` [m/s].
        newton_transrate_tol: Newton convergence tolerance on ``max |dA| / dt`` of
            rigid bodies [1/s].
        linear_tol_rate: Relative PCG tolerance on ``r^T M^-1 r``.
        line_search_max_iter: Maximum backtracking trials per Newton iteration.
        semi_implicit: Enable libuipc's semi-implicit early exit after
            ``semi_implicit_k_min`` iterations of full feasible steps.
        semi_implicit_k_min: Newton iterations before the early exit applies.
        eps_velocity: Friction smoothing velocity [m/s]; below it static friction becomes a stiff spring.
        cloth_strain_rate: Strain-limiting coefficient :math:`\\eta` of the cloth
            membrane (libuipc's default 100); 0 makes the membrane quadratic.
        body_kappa: Orthogonality stiffness :math:`\\kappa` of rigid bodies [Pa]
            (libuipc's affine-body stiffness), taken as given at every step.
            ``None`` takes libuipc's 1e8 Pa, lowered for a body and a step
            whose turn of 0.1 rad about the axis of its largest moment of
            inertia it would not take in one Newton iteration (see
            **Rigid bodies**).
        joint_strength_ratio: Penalty of the joints relative to the mass of the
            dynamic bodies they join (libuipc's ``strength_ratio``; see **Joints**).
        shape_segments: Segments around a sphere, capsule, cylinder, cone or
            ellipsoid when it is triangulated.
        coupled_worlds: Simulate content in more than one world as one solve
            in which no world collides with another, where by default every
            world is solved apart from the others (see **Worlds**).
        integrator: ``"bdf1"`` (implicit Euler) or ``"bdf2"`` (see **BDF2**).
        contact_constitution: ``"ipc"`` (barrier contact) or ``"al-ipc"`` (see
            **AL-IPC contact**).
        fem_preconditioner: ``"diag"`` (block diagonal) or ``"mas"`` for
            particles, cloth and tetrahedra (see **MAS**).
        kinematic_motion: ``"jump"`` (a kinematic body takes the pose of
            ``state_in`` before the step) or ``"aim"`` (it moves inside the
            step toward the pose prescribed for its end; see **Kinematic
            bodies**).
        kinematic_strength_ratio: The strength ratio of the aims of
            ``kinematic_motion="aim"`` (libuipc's soft transform constraint).
        enable_d6_joints: Simulate the D6 joints that lock what one of
            libuipc's joints locks, as that joint (see **D6 joints**); without
            it every D6 joint is rejected.
        enable_distance_joints: Simulate distance joints of one length, as
            libuipc's soft vertex stitch of that rest length, and those of
            none (see **Distance joints**); without it every distance joint is
            rejected.
        enable_joint_limit_damping: Damp a revolute or prismatic joint that
            starts a step past one of its limits with ``joint_limit_kd``, as
            Featherstone and SemiImplicit do (see **Joints**); without it
            ``joint_limit_kd`` is not read.
        enable_joint_armature: Give a revolute or prismatic joint's
            ``joint_armature`` to its coordinate as an inertia (see
            **Joints**); without it an armature other than 0 is rejected.
        enable_joint_effort_limit: Bound a drive's force by its
            ``joint_effort_limit`` as decided at the start of every step (see
            **Joints**); without it a limit below 1e6 on a driven degree of
            freedom is rejected.

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
        less room for the jump of a kinematic body, which is not swept (see
        ``kinematic_motion``).

        The barrier needs a gap at the start. When the surfaces of different
        bodies cross each other, or are at or inside their contact thickness (up
        to the rounding of float32 positions), at the start, the solver raises
        :class:`ValueError` and names the closest pair, as libuipc's sanity
        check rejects such a scene at init. The start is the state the first
        step takes. Where the model holds it, the solver checks it at
        construction, on the model's state (with forward kinematics for bodies
        at the identity, see **State**). A kinematic body, whose pose the
        caller writes at every step, and a link that the model leaves at the
        identity, which the caller places by writing forward kinematics into
        the state, start where the caller puts them: the pairs with such a body
        are checked by the first step on its ``state_in``, before anything runs
        from it. A first step captured in a graph cannot read its ``state_in``:
        it raises when such a pair fails in the model's state, and otherwise
        takes its start as it is; take the first step outside graph capture to
        have that start checked. The thickness of a pair is the sum of its two
        sides' (see **Contact**). Surfaces that never move may touch or cross,
        and pairs that do not collide may overlap; the inside of a body is not
        checked. A gap inside ``d_hat`` is valid, but there the barrier starts
        compressed and pushes the surfaces apart in the first steps. The check
        compares the bodies whose bounding boxes come that close, every
        primitive of one against every primitive of the other, and skips a pair
        of bodies with more than 2**31 - 1 primitive pairs with a warning. A
        state written after the first step is not checked: a step that starts
        in contact fails (:attr:`SolverIPC.StatusBits.THICKNESS_VIOLATED` or
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

        status: wp.array[wp.int32] | None = SolverBase.Observables.field(
            dtype=wp.int32, frequency=Model.AttributeFrequency.WORLD
        )
        """Per-world :class:`SolverIPC.StatusBits` of the latest step, shape ``(world_count + 1,)``.

        With independent worlds (several worlds, the default) every world that holds unknowns gets its own mask, and
        the final entry, global world ``-1``, stays zero. With one world, or ``coupled_worlds``, every world that owns
        particles or bodies gets the mask of the one solve, the final entry too when global particles or bodies are
        in it. Other entries stay zero.
        """

    class StatusBits(IntFlag):
        """Bits of the per-world ``Observables.status`` mask: how the solve of the step ended.

        The step failed, and its result must not be used, when a bit of :attr:`SolverIPC.StatusBits.FAILED` is set. The two
        iteration limits are not failures.
        """

        OVERFLOW = 1 << 0
        """A contact list overflowed and dropped pairs: the step ran without them. With independent worlds a world
        asked for more than its quota of the lists: the bit is in that world's entry alone, and that world is back at
        its state at the step's start. Set from then on until the lists are enlarged (:meth:`SolverIPC.grow`, or an
        eager step)."""

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

    SUPPORTED_OBSERVABLES = frozenset(
        {SolverBase.ObservableKind.SOLVER_RESIDUAL, SolverBase.ObservableKind.CONTACT_F, ObservableKind.STATUS}
    )

    # libuipc-warp, imported by the first solver
    _uipc_warp: ClassVar = None

    def __init__(
        self,
        model: Model,
        *,
        kappa: float | None = 1.0e9,
        adaptive_kappa: bool = False,
        d_hat: float | None = None,
        thickness: float = 0.0,
        newton_max_iter: int = 1024,
        newton_velocity_tol: float = 0.05,
        newton_transrate_tol: float = 0.1,
        linear_tol_rate: float = 1.0e-3,
        line_search_max_iter: int = 8,
        semi_implicit: bool = True,
        semi_implicit_k_min: int = 6,
        eps_velocity: float = 0.01,
        cloth_strain_rate: float = 100.0,
        body_kappa: float | None = None,
        joint_strength_ratio: float = _JOINT_STRENGTH_RATIO,
        shape_segments: int = 24,
        coupled_worlds: bool = False,
        integrator: str = "bdf1",
        contact_constitution: str = "ipc",
        fem_preconditioner: str = "diag",
        kinematic_motion: str = "jump",
        kinematic_strength_ratio: float = 100.0,
        enable_d6_joints: bool = False,
        enable_distance_joints: bool = False,
        enable_joint_limit_damping: bool = False,
        enable_joint_armature: bool = False,
        enable_joint_effort_limit: bool = False,
    ):
        if not model.device.is_cuda:
            raise NotImplementedError(f"SolverIPC runs on CUDA devices only; the model is on {model.device}.")
        super().__init__(model)
        self._coupled_worlds = bool(coupled_worlds)
        self._joint_options = _JointOptions(
            d6=bool(enable_d6_joints),
            distance=bool(enable_distance_joints),
            limit_damping=bool(enable_joint_limit_damping),
            armature=bool(enable_joint_armature),
            effort_limit=bool(enable_joint_effort_limit),
        )
        _check_supported(model, self._joint_options)
        _check_choice("integrator", integrator, ("bdf1", "bdf2"))
        _check_choice("contact_constitution", contact_constitution, ("ipc", "al-ipc"))
        _check_choice("fem_preconditioner", fem_preconditioner, ("diag", "mas"))
        _check_choice("kinematic_motion", kinematic_motion, ("jump", "aim"))
        if not (np.isfinite(kinematic_strength_ratio) and kinematic_strength_ratio >= 0.0):
            raise ValueError(
                f"SolverIPC: kinematic_strength_ratio must be finite and not negative, got {kinematic_strength_ratio}."
            )
        if kinematic_motion == "aim" and contact_constitution == "al-ipc":
            # libuipc's AL-IPC frame checks animated aims with a test of its own (animation_reach_target), which has
            # not been tested on this path
            raise NotImplementedError(
                "SolverIPC: kinematic_motion='aim' is not offered with contact_constitution='al-ipc'."
            )
        if adaptive_kappa and contact_constitution == "al-ipc":
            raise ValueError(
                "SolverIPC: adaptive_kappa sets the stiffness of the barrier, which contact_constitution='al-ipc' "
                "does not use."
            )
        if kappa is None and not adaptive_kappa:
            raise ValueError("SolverIPC: kappa=None, libuipc's default stiffness, needs adaptive_kappa=True.")
        if kappa is not None and not (np.isfinite(kappa) and kappa > 0.0):
            raise ValueError(f"SolverIPC: kappa must be a finite positive stiffness, got {kappa}.")
        if body_kappa is not None and not (np.isfinite(body_kappa) and body_kappa > 0.0):
            raise ValueError(f"SolverIPC: body_kappa must be a finite positive stiffness, got {body_kappa}.")
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
            eps_velocity=eps_velocity,
            integrator=integrator,
            contact_constitution=contact_constitution,
            fem_preconditioner=fem_preconditioner,
        )
        # a BDF2 history starts with an implicit Euler step: at the first step, and at every step from a state the
        # solver did not write (class docstring, **BDF2**)
        self._options = uipc_warp.EngineOptions(bdf2_start="bdf1") if integrator == "bdf2" else None
        self._adaptive_kappa = bool(adaptive_kappa)
        self._thickness = float(thickness)
        self._cloth_strain_rate = float(cloth_strain_rate)
        self._body_kappa = None if body_kappa is None else float(body_kappa)
        self._joint_strength_ratio = float(joint_strength_ratio)
        self._shape_segments = int(shape_segments)
        self._aims = kinematic_motion == "aim"
        self._kinematic_strength_ratio = float(kinematic_strength_ratio)
        self._dt = self._config.dt  # of the last step, which grow() solves again
        self._clamp_checked_dt = None  # the step size whose stiffness the clamp warning was last checked for
        self._clamp_reported = False
        self._engine = None
        # several local worlds (_Worlds), solved apart from each other unless coupled_worlds (_build_engine)
        self._worlds = None
        self._independent = False
        self._ortho = None  # the engine's orthogonality potentials, whose stiffness follows dt (default body_kappa)
        self._default_control = None  # the Control of a step given none, built once (_control_of_model)
        # CONTACT_F: whether a container requested it, and libuipc-warp's ContactForces of the engine then
        # (allocated with the request and with every engine built after it, outside any capture)
        self._contact_output = False
        self._contact_export = None
        self._steps = 0  # steps taken, which the contact export's records follow (_contact_records)
        # an entry of a coupled solver (coupling_notify_iteration_restarts): the status bits of its steps since
        # check_status() last read them, and the scratch word of one step's bits
        self._sticky_status = self._step_status = None
        # an entry of a coupler that restarts its steps (coupling_notify_iteration_restarts): the start of a coupled step
        # kept on the device, saved by the step after the coupler wrote the input state and restored by the step after
        # a restart's notification (_save_step, _restore_step); the (array, copy) of what the solver carries from step
        # to step (_allocate_restart)
        self._restarts = False
        self._save_pending = self._restore_pending = False
        self._carried = []
        # the failure, in the model's state, of the start of a pair with a body whose start the caller writes, which
        # a captured first step raises (_check_start)
        self._start_pending = None
        self._build_engine()
        self._check_start()
        # the first step checks again the start of the pairs with a body whose start the caller writes
        self._start_unchecked = self._engine is not None and bool(self._posed_body.any())

    @property
    def supported_observables(self) -> frozenset[str]:
        """``STATUS``, ``SOLVER_RESIDUAL`` and ``CONTACT_F``; with ``contact_constitution="al-ipc"`` ``STATUS`` only: the
        AL-IPC frame ends on its own criterion, without a Newton residual, and its contact is an active set with
        multipliers, not the barrier and friction terms the contact output exports."""
        config = getattr(self, "_config", None)
        if config is not None and config.contact_constitution == "al-ipc":
            return frozenset({self.ObservableKind.STATUS})
        return self.SUPPORTED_OBSERVABLES

    @override
    def observables(
        self, kinds, *, contacts: Contacts | None = None, requires_grad: bool | None = None
    ) -> SolverIPC.Observables:
        """Allocate solver observables (see :meth:`SolverBase.observables`).

        With ``CONTACT_F`` requested this also allocates what a step needs to write the solver's own contacts into the
        step's :class:`~newton.Contacts` (see **Contact output** in the class docstring): libuipc-warp's records of
        every slot of the solver's contact lists, about 64 bytes per slot, kept from then on and allocated again when
        the solve is built again. ``contacts`` sizes ``contact_f``; any capacity is taken, the rows past it are
        dropped (:meth:`get_max_contact_count` and :meth:`get_max_soft_contact_count` give the capacities that drop
        none). Allocate before graph capture.
        """
        observables = super().observables(kinds, contacts=contacts, requires_grad=requires_grad)
        if observables.is_requested(SolverBase.ObservableKind.CONTACT_F) and not self._contact_output:
            self._contact_output = True
            self._make_contact_export()
        return observables

    def _contact_term_capacity(self) -> int:
        """The most contact terms a step can export: every slot of the engine's contact lists (libuipc-warp's
        ``ContactForces.segment_capacities``)."""
        if self._engine is None:
            return 0
        return int(self._uipc_warp.arrays.ContactForces.segment_capacities(self._engine).sum())

    def get_max_contact_count(self) -> int:
        """The rigid rows a step with ``CONTACT_F`` can write: the size of ``rigid_contact_max`` that drops none.

        Every contact term between two shapes is a row (see **Contact output** in the class docstring), so this is the
        number of slots of the solver's contact lists, active and friction terms together, or 0 when fewer than two
        shapes collide. It is a worst case, several times the rows a step usually writes; a smaller buffer is taken,
        and the rows past it are dropped while ``rigid_contact_count`` counts them. :meth:`grow` and an eager step
        that overflows enlarge the lists, and this with them. 0 for a model without particles and bodies.
        """
        if self._engine is None or len(self._contact_shapes) < 2:
            return 0
        return self._contact_term_capacity()

    def get_max_soft_contact_count(self) -> int:
        """The soft rows a step with ``CONTACT_F`` can write: the size of ``soft_contact_max`` that drops none.

        Every contact term between a particle feature (a particle, an edge or a face of cloth or tetrahedra) and a
        shape is a row, so this is the number of slots of the solver's contact lists, or 0 without particles or
        without colliding shapes. As :meth:`get_max_contact_count`, it is a worst case, and a smaller buffer is taken.
        """
        if self._engine is None or self.model.particle_count == 0 or len(self._contact_shapes) == 0:
            return 0
        return self._contact_term_capacity()

    @classmethod
    def _import_uipc_warp(cls):
        """Import libuipc-warp once: raise if it is missing or lacks what the solver calls, warn on versions it was
        not written for."""
        if cls._uipc_warp is not None:
            return cls._uipc_warp
        try:
            import uipc_warp  # noqa: PLC0415
            import uipc_warp.arrays  # noqa: PLC0415
            import uipc_warp.validation  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "SolverIPC requires the libuipc-warp package: pip install git+https://github.com/OrangeEarth15/libuipc-warp"
            ) from e
        version = getattr(uipc_warp, "__version__", "0")
        # the newest calls of each kind the solver makes, all of them public names of libuipc-warp: the frame status,
        # the contact and subscene tabulars and the stiffness policy, the start check, the joints, the iterations of
        # captured steps, the restart of a BDF2 history and the friction discard per subscene (with the rows a restart
        # per subscene takes), the stitches (with the anchors and stitch_groups of the same release), the soft
        # transform constraint (with constraint_groups), the contact forces, the external accelerations, the bodies'
        # orthogonality stiffness, the redo of an overflowing step (with the documented engine members of the same
        # release) and the frame start kept on the device for a coupler's restarts
        needed = (
            (uipc_warp.arrays, "write_frame_status"),
            (uipc_warp.arrays.ArrayScene, "contact_model"),
            (uipc_warp.arrays.ArrayScene, "subscene_model"),
            (uipc_warp.arrays, "ContactStiffness"),
            (uipc_warp.validation.SurfaceChecker, "min_gap"),
            (uipc_warp.arrays, "joint_groups"),
            (uipc_warp.arrays, "add_frame_iterations"),
            (uipc_warp.SimEngine, "prepare_capture"),
            (uipc_warp.SimEngine, "restart_history"),
            (uipc_warp.SimEngine, "restart_rows"),
            (uipc_warp.arrays.ArrayScene, "add_stitches"),
            (uipc_warp.arrays.ArrayScene, "add_soft_transform_constraints"),
            (uipc_warp.arrays, "ContactForces"),
            (uipc_warp.arrays, "ExternalAcceleration"),
            (uipc_warp.arrays, "orthogonality"),
            (uipc_warp.SimEngine, "redo_state"),
            (uipc_warp.SimEngine, "enable_frame_restart"),
        )
        # and the keywords of the newest calls: independent worlds, the scene's flag and the status of every world
        keywords = (
            ("ArrayScene(independent_subscenes=)", "independent_subscenes", uipc_warp.arrays.ArrayScene),
            ("write_frame_status(worlds=)", "worlds", uipc_warp.arrays.write_frame_status),
        )
        missing = [f"{owner.__name__}.{name}" for owner, name in needed if not hasattr(owner, name)]
        missing += [call for call, keyword, owner in keywords if keyword not in inspect.signature(owner).parameters]
        if missing:
            calls = [f"{o.__name__}.{n}" for o, n in needed] + [call for call, _, _ in keywords]
            raise ImportError(
                f"SolverIPC requires libuipc-warp 0.4 or newer with {', '.join(calls)}; the installed one ({version}) "
                f"lacks {', '.join(missing)}."
            )
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
        return uipc_warp

    # ------------------------------------------------------------------ setup
    def _build_engine(self) -> None:
        """Lay the model out for libuipc-warp and build its engine. The next step takes its whole state from
        ``state_in``."""
        model = self.model
        device = model.device
        self._engine = self._frame_graph = self._forces = self._contact_export = None
        # a coupled solver's proxies (class docstring, **Coupling**)
        self._proxy_body, self._proxy_particle = _proxy_bodies(model), _proxy_particles(model)
        self._proxies = bool(self._proxy_body.any() or self._proxy_particle.any())
        if model.particle_count == 0 and model.body_count == 0:
            return
        if self._proxies:
            self._check_proxies()
        # the global world's gravity, which every part overrides with the gravity of its own world
        self._config.gravity = tuple(model.gravity.numpy()[-1].astype(np.float64))
        if not self._d_hat_given:
            self._config.d_hat = _default_d_hat(model)

        # several local worlds: one subscene each, all of them meeting the global entities' subscene 0; libuipc-warp
        # solves them apart from each other (independent worlds) unless coupled_worlds asks for one solve
        self._worlds = _Worlds.of(model)
        self._independent = self._worlds is not None and not self._coupled_worlds
        self._status_entries = None
        scene = self._uipc_warp.arrays.ArrayScene(
            kappa_corridor=self._adaptive_kappa, independent_subscenes=self._independent
        )
        elements = _contact_elements(model, self._worlds, self._aims)
        for _ in range(1, elements.count):
            scene.contact_element()
        if self._worlds is not None:
            for _ in range(1, self._worlds.count):
                scene.subscene_element()
            scene.subscene_model(np.zeros(self._worlds.count - 1, np.int64), np.arange(1, self._worlds.count), True)
        tets = self._add_particles_and_tets(scene, elements)
        self._add_cloth(scene, tets)
        self._add_springs(scene)
        self._plan_attachments()
        self._plan_distance_joints()
        self._dyadic = self._add_shapes_and_bodies(scene, elements)
        if self._independent:
            self._check_independent_content()
            if len(self._status_entries) > 1:  # one world that moves is solved as one world: no option to refuse
                self._check_independent_options()
        if self._config.contact_constitution == "al-ipc":
            self._check_al_ipc(elements)
        self._add_joints(scene)
        self._planes = self._collect_planes()
        for P, N, s in zip(*self._planes, strict=True):
            scene.add_half_plane(P, N, contact_element=elements.shape[s], subscene_element=self._subscene_of_shape(s))
        if elements.count > 1:
            # one model for every pair of the elements that hold vertices, with the default model's stiffness: the
            # solver's kappa, as given or as libuipc's policy resolves it (adaptive_kappa), one value for every pair
            a, b = np.triu_indices(elements.count - 1)
            a, b = a + 1, b + 1
            scene.contact_model(a, b, elements.mu[a, b], None, elements.enabled[a, b])
        self._attachment_hard = wp.array(self._attachment_penalties(self._dyadic[0]), dtype=wp.float64, device=device)
        self._distance_hard = wp.array(self._distance_penalties(self._dyadic[0]), dtype=wp.float64, device=device)
        self._add_stitches(scene)
        if len(self._aim_bodies):  # libuipc's soft transform constraint on the kinematic bodies, in their order
            ratio = self._kinematic_strength_ratio
            scene.add_soft_transform_constraints(self._aim_bodies, (ratio, ratio))
        self._scene = scene
        self._scene_inputs = self._scene_inputs_now()
        self._joint_inputs = self._joint_inputs_now()
        self._control = None  # of the last step, whose joint terms a redone frame takes again

        layout = scene.layout()
        # CONTACT_F's contact rows: the shape of every engine vertex on a shape (-1 elsewhere: particles, the bodies'
        # pseudo-vertices and anchors), and the shapes that hold contact primitives, the planes' among them
        vertex_shape = np.full(layout.n, -1, np.int32)
        for k, shape in enumerate(self._obstacle_shapes):
            vertex_shape[layout.obstacle_offsets[k] : layout.obstacle_offsets[k + 1]] = shape
        for b, shapes in enumerate(self._body_vertex_shape):
            vertex_shape[layout.body_vertex_offsets[b] : layout.body_vertex_offsets[b + 1]] = shapes
        self._vertex_shape = vertex_shape
        self._contact_shapes = np.union1d(vertex_shape[vertex_shape >= 0], np.asarray(self._planes[2], np.int32))
        # the external accelerations: one row per free particle, then four per dynamic body (p and the rows of A)
        free = np.flatnonzero(~self._fixed_particle).astype(np.int32)
        dynamic = np.flatnonzero(~self._kinematic_body).astype(np.int32)
        self._n_free, self._n_dynamic = len(free), len(dynamic)
        self._force_blocks = np.concatenate([free, (layout.body_base[dynamic][:, None] + np.arange(4)).reshape(-1)])
        self._free_particles = wp.array(free, dtype=wp.int32, device=device)
        self._dynamic_bodies = wp.array(dynamic, dtype=wp.int32, device=device)
        self._body_mass_inv = wp.array(self._mass_inverse(*self._dyadic), dtype=wp.mat44d, device=device)
        # the requests of a new state or a reset, one flag per subscene (one with one world): which subscenes
        # discard their friction and start their BDF2 history anew, libuipc-warp's requests per subscene
        n_sub = 1 if self._worlds is None else self._worlds.count
        self._changed = wp.zeros(n_sub, dtype=wp.int32, device=device)
        self._selected = wp.zeros(n_sub, dtype=wp.int32, device=device)
        for name, entity_world in (("_particle_subscene", model.particle_world), ("_body_subscene", model.body_world)):
            w = np.zeros(0, np.int64) if entity_world is None else entity_world.numpy()
            subscene = np.zeros(len(w)) if self._worlds is None else self._worlds.subscene(w)
            setattr(self, name, wp.array(np.asarray(subscene, np.int32), dtype=wp.int32, device=device))
        if self._worlds is not None:  # the world_mask entry of every subscene: the global entities', then the worlds'
            entry = np.concatenate([[model.world_count], self._worlds.worlds]).astype(np.int32)
            self._subscene_entry = wp.array(entry, dtype=wp.int32, device=device)
        worlds = np.unique(
            np.concatenate([w.numpy() for w in (model.particle_world, model.body_world) if w is not None and len(w)])
        )
        self._world_entries = np.where(worlds < 0, model.world_count, worlds).astype(np.int32)
        self._world_entries_device = wp.array(self._world_entries, dtype=wp.int32, device=device)
        if self._independent:  # the entry of every independent world, and a scratch word for each world entry
            self._status_entries_device = wp.array(self._status_entries, dtype=wp.int32, device=device)
            self._world_words = wp.zeros(model.world_count + 1, dtype=wp.int32, device=device)
        if self._sticky_status is not None and self._sticky_status.shape[0] != self._kept_words():
            self._allocate_kept_status()  # an entry of a coupled solver whose rebuilt solve has other worlds
        # the particles whose new state from state_in discards no friction: the fixed ones (the engine's flags), and
        # a coupled solver's proxy particles, whose state the coupler writes at every step
        self._particle_quiet = None
        if self._proxy_particle.any():
            quiet = (self._fixed_particle | self._proxy_particle).astype(np.int32)
            self._particle_quiet = wp.array(quiet, dtype=wp.int32, device=device)
        self._status_scratch = wp.zeros(1, dtype=wp.int32, device=device)
        self._residual_scratch = wp.zeros(1, dtype=wp.float32, device=device)
        self._no_restart = wp.zeros(1, dtype=wp.int32, device=device)  # BDF1: no history to restart
        self._no_float32 = wp.zeros(1, dtype=wp.float32, device=device)  # a model array the model does not have
        self._no_bool = wp.zeros(1, dtype=wp.bool, device=device)
        nb = model.body_count
        if nb:
            self._body_base = wp.array(layout.body_base.astype(np.int32), dtype=wp.int32, device=device)
            self._body_kinematic = wp.array(self._kinematic_body.astype(np.int32), dtype=wp.int32, device=device)
            # the bodies whose new state from state_in discards no friction: the kinematic ones, and a coupled
            # solver's proxy bodies
            self._body_quiet = self._body_kinematic
            if self._proxy_body.any():
                quiet = (self._kinematic_body | self._proxy_body).astype(np.int32)
                self._body_quiet = wp.array(quiet, dtype=wp.int32, device=device)
            # kinematic_motion="aim": the bodies whose state the solver keeps (held), and those whose state_out copies
            # state_in (passed through: every kinematic body, but those moving toward their aims)
            self._body_hold = wp.zeros(nb, dtype=wp.int32, device=device)
            self._body_passthrough = self._body_kinematic
            if len(self._aim_bodies):
                through = self._kinematic_body.copy()
                through[self._aim_bodies] = False
                self._body_passthrough = wp.array(through.astype(np.int32), dtype=wp.int32, device=device)
            self._aim_rows = wp.array(self._aim_bodies.astype(np.int32), dtype=wp.int32, device=device)
            entry = np.where(model.body_world.numpy() < 0, model.world_count, model.body_world.numpy())
            self._body_entry = wp.array(entry.astype(np.int32), dtype=wp.int32, device=device)
            self._joint_wrench = wp.zeros(nb, dtype=wp.spatial_vector, device=device)
            self._body_q_written = wp.empty(nb, dtype=wp.transform, device=device)
            self._body_qd_written = wp.empty(nb, dtype=wp.spatial_vector, device=device)
            self._forget_written()
        if self._proxy_body.any():  # the body of every surface vertex, for the proxy feedback (-1 elsewhere)
            vertex_body = np.full(layout.n, -1, np.int32)
            for b in range(nb):
                vertex_body[layout.body_vertex_offsets[b] : layout.body_vertex_offsets[b + 1]] = b
            self._vertex_body = wp.array(vertex_body, dtype=wp.int32, device=device)
        self._make_engine(None)
        if self._restarts:  # an entry of a coupler that restarts its steps: the new solve's copies
            self._allocate_restart()

    def _check_proxies(self) -> None:
        """``NotImplementedError`` for the options a coupled solver's proxies cannot take (class docstring,
        **Coupling**): BDF2, whose history a proxy's state, written by the coupler at every step, would leave behind,
        and AL-IPC, which has no contact forces to harvest the proxies' feedback from."""
        if self._config.integrator == "bdf2":
            raise NotImplementedError(
                "SolverIPC: integrator='bdf2' is not offered to an entry of a coupled solver with proxies. The coupler "
                "writes a proxy's state at every step, so that the step would start a proxy's BDF2 history anew every "
                "time; use integrator='bdf1'."
            )
        if self._config.contact_constitution == "al-ipc":
            raise NotImplementedError(
                "SolverIPC: contact_constitution='al-ipc' is not offered to an entry of a coupled solver with proxies: "
                "the proxies' feedback is the force of the step's contact terms on them, which AL-IPC does not export."
            )

    def _check_al_ipc(self, elements: _ContactElements) -> None:
        """``NotImplementedError`` for contact that libuipc's AL-IPC does not constrain. Its active set holds the
        vertex / half-plane, point / triangle and edge / edge pairs of surfaces, so a particle on no cloth or
        tetrahedron that collides has no constraint; and the penalty of a pair is the smallest of its vertices'
        mass-scaled penalties, so a pair with a vertex without mass has none (a static mesh shape, a particle of no
        mass, a body of no mass), where libuipc's own obstacles are bodies or vertices with mass."""
        model = self.model
        colliding = elements.enabled[elements.particle].any(axis=1) if model.particle_count else np.zeros(0, bool)
        massless = []
        if np.any(colliding & ~elements.on_mesh):
            massless.append("particles on no cloth or tetrahedron, which are points of no surface")
        if np.any(colliding & elements.on_mesh & (model.particle_mass.numpy() <= 0.0)):
            massless.append("cloth or tetrahedron particles without mass")
        if self._obstacle_shapes:
            massless.append(f"the static shapes {sorted(self._obstacle_shapes)}, which have no mass")
        if model.shape_count and model.body_count:
            shape_body = model.shape_body.numpy()[_colliding_shapes(model)]
            bodies = np.unique(shape_body[shape_body >= 0])
            light = bodies[self._dyadic[0][bodies] <= 0.0]
            if len(light):
                massless.append(f"the bodies {light.tolist()} without mass")
        if massless:
            raise NotImplementedError(
                "SolverIPC: contact_constitution='al-ipc' constrains contact between surfaces of vertices with mass "
                "(libuipc's AL-IPC scales the penalty of a pair by the smallest mass of its vertices); this model "
                "collides " + "; ".join(massless) + ". Static planes are taken."
            )

    def _check_independent_options(self) -> None:
        """``NotImplementedError`` for the options that independent worlds do not take yet (class docstring,
        **Worlds**), each naming what to use instead, when two worlds or more hold unknowns: libuipc-warp solves the
        worlds apart from each other with libuipc's barrier contact and the block-diagonal preconditioner."""
        worlds = self._worlds.worlds
        which = f"the worlds {worlds.tolist()}" if len(worlds) <= 8 else f"{len(worlds)} worlds"
        instead = (
            f"or pass coupled_worlds=True to solve {which} as one (see Worlds in the class docstring), or give the "
            "solver a model of one world"
        )
        if self._config.contact_constitution == "al-ipc":
            raise NotImplementedError(
                "SolverIPC: contact_constitution='al-ipc' is not offered with independent worlds yet: libuipc-warp "
                f"solves {which} apart from each other with libuipc's barrier contact only, AL-IPC's frame being one "
                f"solve of the whole scene. Use contact_constitution='ipc', {instead}."
            )
        if self._config.fem_preconditioner == "mas":
            raise NotImplementedError(
                "SolverIPC: fem_preconditioner='mas' is not offered with independent worlds yet: MAS's clusters may "
                f"join vertices of different worlds, whose linear solves are apart. Use fem_preconditioner='diag', "
                f"{instead}."
            )

    def _check_independent_content(self) -> None:
        """With independent worlds: ``NotImplementedError`` for an unknown of the global world -1, which every world
        would push and so join (a particle with mass that is ACTIVE, a dynamic body, a coupled solver's proxy, or with
        ``kinematic_motion="aim"`` a kinematic body with colliding shapes), as Newton's implicit MPM refuses global
        colliders backed by dynamic bodies with ``separate_worlds``, and for a triangle, tetrahedron or spring that
        joins two worlds. Then ``_status_entries``: the worlds that hold unknowns, in increasing order, which are
        libuipc-warp's independent worlds in their order (its worlds are the subscenes from 1 on that hold unknowns,
        in subscene order, and subscene k + 1 is the k-th local world with content, ``_Worlds``)."""
        model = self.model
        world = model.particle_world.numpy() if model.particle_count else np.zeros(0, np.int64)
        free = ~self._fixed_particle
        particle_sub = self._worlds.subscene(world)
        moving = ~self._kinematic_body
        moving[self._aim_bodies] = True
        body_sub = self._body_row_subscene
        instead = "or pass coupled_worlds=True to solve the worlds as one, in which global entities may move"
        lost = np.flatnonzero(free & (particle_sub == 0))
        if len(lost):
            more = f" ({len(lost) - 1} more global particles are too)" if len(lost) > 1 else ""
            raise NotImplementedError(
                f"SolverIPC: particle {int(lost[0])} of the global world -1 is free (it has mass and is ACTIVE){more}: "
                "with independent worlds it would be an unknown that every world pushes, which would join the worlds. "
                f"Give it a world or make it kinematic (mass 0), {instead}."
            )
        lost = np.flatnonzero(moving & (body_sub == 0))
        if len(lost):
            b = int(lost[0])
            label = model.body_label[b] if b < len(model.body_label) else ""
            name = f"body {b}" + (f" ({label!r})" if label and label != f"body_{b}" else "")
            more = f" ({len(lost) - 1} more global bodies are too)" if len(lost) > 1 else ""
            if self._kinematic_body[b]:
                what, fix = (
                    "a kinematic body with colliding shapes, which kinematic_motion='aim' moves toward its aim as an "
                    "unknown of the step",
                    "Use kinematic_motion='jump', under which it is a fixed body that acts in every world, give it a "
                    "world",
                )
            else:
                what, fix = "a dynamic body", "Give it a world or make it kinematic"
            raise NotImplementedError(
                f"SolverIPC: {name} of the global world -1 is {what}{more}: with independent worlds it would be an "
                f"unknown that every world pushes, which would join the worlds. {fix}, {instead}."
            )
        across = _element_across_worlds(model, free)
        if across is not None:
            raise NotImplementedError(
                f"SolverIPC: {across}: with independent worlds no element joins two worlds. Keep it inside one world, "
                f"{instead}."
            )
        subscenes = np.unique(np.concatenate([particle_sub[free], body_sub[moving]]).astype(np.int64))
        self._status_entries = self._worlds.worlds[subscenes - 1].astype(np.int32)

    def _subscene_of_shape(self, s: int) -> int:
        """The subscene of shape ``s``'s vertices: its world's (several worlds), else the one subscene, 0."""
        return 0 if self._worlds is None else int(self._worlds.subscene(self.model.shape_world.numpy()[s]))

    def _subscene_of_body(self, b: int) -> int:
        """The subscene of body ``b``, for the anchors of a body without shapes: its world's (several worlds),
        else the one subscene, 0."""
        return 0 if self._worlds is None else int(self._worlds.subscene(self.model.body_world.numpy()[b]))

    def _world_gravity(self) -> np.ndarray:
        """``Model.gravity`` in float64: one row per world, the global world's last, so that the world of an entity
        (-1 for a global one) indexes its row."""
        return self.model.gravity.numpy().reshape(-1, 3).astype(np.float64)

    def _vertex_gravity(self) -> np.ndarray:
        """The gravity of every engine vertex, as the scene lays it out: a particle's world's on the particle, a
        body's world's on its translation, none elsewhere (``SimEngine.set_gravity``)."""
        layout = self._scene.layout()
        g = self._world_gravity()
        out = np.zeros((layout.n, 3))
        if self.model.particle_count:
            out[: layout.n_fem] = g[self.model.particle_world.numpy()]
        if self.model.body_count:
            out[layout.body_base[: self.model.body_count]] = g[self.model.body_world.numpy()]
            out[layout.body_base[self._aim_bodies]] = 0.0  # gravity does not move a kinematic body toward its aim
        return out

    def _check_start(self, state: str | None = None) -> None:
        """Raise ``ValueError`` when the surfaces of different bodies cross each other at the start, or start at or
        inside their contact thickness.

        The barrier needs surfaces that start apart, by more than their contact thickness. A step fails on a contact
        pair at or inside its thickness (``THICKNESS_VIOLATED`` or ``INVALID_CCD_STEP``), which a captured step only
        reports, and surfaces that cross without such a pair (an edge through a triangle, away from its vertices) step
        on with the barrier keeping them crossed. As libuipc's sanity check does at init, this tests the start
        state, without the surfaces that never move (static shapes, kinematic particles and bodies), whose contact the
        solve skips, and without the pairs whose contact elements do not collide or whose subscenes do not interact (the
        two tables tested apart, as the engine tests them). The gap of a pair is its distance less its contact
        thickness, the sum of its two primitives' thicknesses, each that of its vertices (a free particle's radius, a
        shape's margin, the ``thickness`` of cloth and tetrahedra). Every contact point that moves is tested against
        every half plane. Triangle meshes and particles are tested part pair by part pair, a part being a contact body's
        primitives in one contact element and one subscene, with the particles on no cloth or tetrahedron taken together
        per element and subscene and also tested among themselves: for the pairs of parts whose subscenes interact (with
        several worlds no pair of two worlds is formed) and whose bounding boxes come within their thickness of each
        other, libuipc-warp's brute-force ``SurfaceChecker`` counts the edge-triangle crossings and finds the smallest
        gap over the point-triangle, point-edge, edge-edge and point-point pairs. A gap above 0, also inside ``d_hat``,
        is a valid start, unless it is within a few float32 steps of 0: Newton's positions are float32, and surfaces
        placed in touching contact come out that far apart. Not tested: the inside of a body (contact ignores crossings
        inside a rigid body made of overlapping shapes, and inside a cloth or a soft body the checker cannot tell the
        nearly coplanar triangles of a flat sheet from crossing ones). A pair of parts of more than
        ``_START_CHECK_PAIRS`` primitive pairs is skipped with a warning.

        The start is the state the first step takes (class docstring, **State**). At construction (``state`` None)
        the engine holds the model's state, and a pair that fails there raises, unless one of its bodies is one whose
        start the caller writes into the state (``_posed_body``: a kinematic body, and a link that the model leaves at
        the identity where forward kinematics puts it elsewhere). The failure of such a pair is kept
        (``_start_pending``), and the first step tests the pairs with such a body again: a first step that is not
        captured on the engine's positions it has just loaded from its ``state_in``, a captured one, which cannot read
        its ``state_in``, by raising the kept failure (``_start_in_capture``). With ``state``, the name of the state the
        engine holds in the message, only the pairs with such a body are tested, on the engine's positions as they
        stand, and a failure raises: the test holds for the state of any step loaded into the engine (it synchronizes).
        """
        eng = self._engine
        if eng is None:
            return
        posed_only = state is not None
        layout = self._scene.layout()
        still = np.zeros(layout.n, bool)  # vertices that the solve never moves
        still[: layout.n_fem] = self._fixed_particle
        still[layout.obstacle_offsets[0] : layout.obstacle_offsets[-1]] = True
        offsets = layout.body_vertex_offsets
        held = self._kinematic_body.copy()
        held[self._aim_bodies] = False  # kinematic_motion="aim": these move in the step
        for b in np.flatnonzero(held):
            still[offsets[b] : offsets[b + 1]] = True
        posed_vertex = np.zeros(layout.n, bool)  # the surface vertices of the bodies whose start the caller writes
        for b in np.flatnonzero(self._posed_body):
            posed_vertex[offsets[b] : offsets[b + 1]] = True
        x = eng.vm.x.numpy()
        thickness = self._vertex_thickness
        points = np.asarray(eng.contact_points, np.int64)
        if len(points) == 0:
            return
        # the engine's contact bodies, with the particles on no surface (each a body of its own) taken as one. The
        # surface vertices are those of the contact triangles and edges: the engine's own list holds the particles too
        # where the scene gives it the contact surface (with springs or stitches)
        body = eng.body.numpy().astype(np.int64)
        tris = np.asarray(eng.surface["triangles"], np.int64).reshape(-1, 3)
        edges = np.asarray(eng.surface["edges"], np.int64).reshape(-1, 2)
        surface_vertices = np.unique(np.concatenate([tris.reshape(-1), edges.reshape(-1)]))
        lone = np.setdiff1d(points, surface_vertices)
        group = body.copy()
        lone_group = int(group.max()) + 1
        group[lone] = lone_group
        # parts: the groups split by contact element and subscene, compared only where both tables let them collide
        cid, sid, enabled, subscenes = self._cids, self._scids, self._enabled, self._subscene_enabled
        n_el, n_sub = len(enabled), len(subscenes)
        key = (group * n_el + cid) * n_sub + sid
        ids, point_group = np.unique(key[points], return_inverse=True)
        ids_group, ids_element, ids_subscene = ids // (n_el * n_sub), ids // n_sub % n_el, ids % n_sub
        posed = np.zeros(len(ids), bool)  # the parts of the bodies whose start the caller writes
        np.logical_or.at(posed, point_group, posed_vertex[points])
        if posed_only and not posed.any():
            return
        # Newton's positions are float32: surfaces placed in touching contact come out up to a few float32 steps
        # apart, which the barrier would take as a valid, nearly singular start. Such a gap counts as contact.
        slack = 4.0 * float(np.spacing(np.float32(np.abs(x[points]).max())))
        # (crossings, gap, distance, pair thickness, part, the half plane (as -1 - plane) or part, the pair's vertices,
        # whether a body of the pair is one whose start the caller writes)
        violations = []

        # contact points against half planes, each moved out by its shape's margin: the distance from the plane's own
        # surface and the margin's share of the pair's thickness (with state, the points of posed bodies alone)
        P, N, plane_shapes = self._planes
        moves = ~still[points]
        if posed_only:
            moves &= posed[point_group]
        if moves.any() and len(P):
            moving = points[moves]
            margin = self.model.shape_margin.numpy().astype(np.float64)[plane_shapes]
            d = x[moving] @ N.T - np.einsum("ij,ij->i", P, N) + margin
            t = thickness[moving][:, None] + margin
            gap = d - t
            meets = enabled[cid[moving][:, None], self._plane_cids[None, :]]
            meets &= subscenes[sid[moving][:, None], self._plane_scids[None, :]]
            owner = point_group[moves]
            for j in range(len(P)):
                hit = np.flatnonzero(meets[:, j] & (gap[:, j] <= slack))
                for g in np.unique(owner[hit]):
                    k = hit[owner[hit] == g]
                    k = k[np.argmin(gap[k, j])]
                    violations.append((0, gap[k, j], d[k, j], t[k, j], g, -1 - j, (int(moving[k]),), posed[g]))

        # triangle meshes and particles, one pair of parts at a time
        lo = np.full((len(ids), 3), np.inf)
        hi = np.full((len(ids), 3), -np.inf)
        np.minimum.at(lo, point_group, x[points])
        np.maximum.at(hi, point_group, x[points])
        reach = np.zeros(len(ids))
        np.maximum.at(reach, point_group, thickness[points])
        all_still = np.ones(len(ids), bool)
        np.logical_and.at(all_still, point_group, still[points])
        # the primitives of each part, sorted by part: rows, whether they move, and where each part's rows start (a
        # primitive's vertices are in one contact body and one element)
        prims = {}
        kinds = (
            ("vertices", surface_vertices[:, None]),
            ("codim", lone[:, None]),
            ("triangles", tris),
            ("edges", edges),
        )
        for name, rows in kinds:
            part_of = np.searchsorted(ids, key[rows[:, 0]])
            order = np.argsort(part_of, kind="stable")
            prims[name] = (
                rows[order],
                ~still[rows[order]].all(axis=1),
                np.searchsorted(part_of[order], np.arange(len(ids) + 1)),
            )
        pairs = self._near_parts(lo, hi, reach, slack, ids_subscene)
        # the particles on no surface meet each other where their element and their subscene meet themselves
        lone_parts = np.flatnonzero(
            (ids_group == lone_group) & enabled[ids_element, ids_element] & subscenes[ids_subscene, ids_subscene]
        )
        pairs = np.concatenate([pairs, np.stack([lone_parts, lone_parts], axis=1)])
        # not tested: two parts that never move, the inside of a body, two parts whose elements do not collide or
        # whose subscenes do not interact (worlds that overlap in space); with state, pairs without a posed body
        a, b = pairs[:, 0], pairs[:, 1]
        keep = (
            ~(all_still[a] & all_still[b])
            & ~((a != b) & (ids_group[a] == ids_group[b]))
            & enabled[ids_element[a], ids_element[b]]
            & subscenes[ids_subscene[a], ids_subscene[b]]
        )
        if posed_only:
            keep &= posed[a] | posed[b]
        pairs = pairs[keep]
        device = self.model.device
        arrays = (wp.array(body.astype(np.int32), device=device), wp.array(thickness, dtype=wp.float64, device=device))
        for a, b in pairs:
            result = self._part_pair(prims, arrays, a, b)
            if result is None:
                warnings.warn(
                    f"SolverIPC does not check whether {self._describe_group(group, points, ids_group[a])} and "
                    f"{self._describe_group(group, points, ids_group[b])} start in contact: their surfaces make more "
                    f"primitive pairs than the check tests ({_START_CHECK_PAIRS}).",
                    NewtonWarning,
                    stacklevel=3,
                )
                continue
            found, gap = result
            if found or gap.gap <= slack:
                violations.append(
                    (found, gap.gap, gap.distance, gap.thickness, a, b, gap.vertices, posed[a] | posed[b])
                )
        if not posed_only:
            self._start_pending = None
        if not violations:
            return
        lone_set = set(lone.tolist())

        def describe(part: int, vertices, other_than: int | None = None) -> tuple[str, int | None]:
            """A part's name, and its particle in the closest pair when it is the particles on no surface."""
            if ids_group[part] == lone_group:
                v = next((v for v in vertices if v in lone_set and v != other_than), None)
                if v is not None:
                    return f"particle {v}", v
            return self._describe_group(group, points, ids_group[part]), None

        def failure(violations: list, state: str | None = None) -> str:
            """Why the start fails: the pair that crosses most, or else the closest pair, and how many more fail;
            ``state``: where (None: the model's state, said for crossings only, as the construction always said)."""
            found, _, distance, t, a, b, vertices, _ = min(violations, key=lambda v: (-v[0], v[1]))
            n = len(violations) - 1
            more = f" ({n} more pair{'s' if n > 1 else ''} of bodies {'do' if n > 1 else 'does'} too)" if n else ""
            one, particle = describe(a, vertices)
            other = (
                f"plane shape {plane_shapes[-1 - b]}"
                if b < 0
                else describe(b, vertices, particle if a == b else None)[0]
            )
            if found:
                where = state or "the model's state"
                return (
                    f"SolverIPC: the surfaces of {one} and {other} intersect in {where}: {found} edges of one cross "
                    f"triangles of the other{more}. The barrier needs surfaces that start apart, by more than their "
                    "contact thickness; a step would keep them crossed."
                )
            need = f" by more than {t - distance:.6g} m" if t - distance > 0.0 else ""
            fix = (
                f"lift {one} off it{need}" if b < 0 else f"move them apart{need}, for example by lifting the upper one"
            )
            rounding = ", up to the rounding of float32 positions" if distance > t else ""
            at = f" in {state}" if state else ""
            return (
                f"SolverIPC: {one} and {other} start {distance:.6g} m apart, at or inside their contact thickness "
                f"({t:.6g} m{rounding}){more}{at}. The barrier needs a gap larger than the thickness at the start: "
                f"{fix}."
            )

        if posed_only:
            raise ValueError(failure(violations, state))
        now = [v for v in violations if not v[-1]]
        if now:
            raise ValueError(failure(now))
        # only pairs with a body whose start the caller writes fail, in the model's state: the first step tests them
        # on its state_in, or raises this when it is captured
        self._start_pending = failure(violations, "the model's state")

    def _near_parts(self, lo: np.ndarray, hi: np.ndarray, reach: np.ndarray, slack: float, sub: np.ndarray):
        """The pairs ``(a, b)``, ``a < b``, of the parts ``_near_boxes`` finds, but where the subscene table has the
        form of separate worlds (several worlds): there no pair of two worlds is formed, which with worlds at one
        place would be every pair of their parts. The parts of the subscenes that meet every subscene (the global
        entities) are tested against every part, those of the others in one band per subscene (``_near_boxes``'s
        ``band``). Other tables keep every near pair, which the caller tests against the table."""
        table = self._subscene_enabled
        meets_all = table.all(axis=1)
        diagonal = np.diagonal(table)
        separate = np.array_equal(
            table, meets_all[:, None] | meets_all[None, :] | (np.eye(len(table), dtype=bool) & diagonal[:, None])
        )
        if len(table) == 1 or not separate:
            return self._near_boxes(lo, hi, reach, slack)
        wide = np.flatnonzero(meets_all[sub])
        local = np.flatnonzero(~meets_all[sub])
        out = [local[self._near_boxes(lo[local], hi[local], reach[local], slack, band=sub[local])]]
        chunk = max(1, 1_000_000 // max(len(lo), 1))  # the wide parts' tests against every part, a block at a time
        for k in range(0, len(wide), chunk):
            a = wide[k : k + chunk]
            apart = np.maximum(lo[None, :, :] - hi[a][:, None, :], lo[a][:, None, :] - hi[None, :, :]).max(axis=2)
            i, b = np.nonzero(apart <= reach[a][:, None] + reach[None, :] + slack)
            b_wide = meets_all[sub[b]]
            keep = (a[i] != b) & (~b_wide | (a[i] < b))  # a pair of two wide parts once
            out.append(np.stack([np.minimum(a[i], b), np.maximum(a[i], b)], axis=1)[keep])
        return np.concatenate(out).reshape(-1, 2)

    @staticmethod
    def _near_boxes(
        lo: np.ndarray, hi: np.ndarray, reach: np.ndarray, slack: float, band: np.ndarray | None = None
    ) -> np.ndarray:
        """The pairs ``(a, b)``, ``a < b``, of boxes ``[lo, hi]`` that come within ``reach[a] + reach[b] + slack`` of
        each other, by sweep and prune along the axis on which the boxes spread most. With ``band`` (an integer per
        box) only boxes of one band pair: the sweep runs over the bands laid end to end along that axis, each farther
        from the next than any box reaches."""
        if len(lo) == 0:
            return np.zeros((0, 2), np.int64)
        axis = int(np.argmax(np.ptp(lo + hi, axis=0)))
        key_lo, key_hi = lo[:, axis], hi[:, axis]
        pad = 0.0
        if band is not None:  # the gap between two bands exceeds the reach of any box by 1 (m)
            width = float(key_hi.max() - key_lo.min()) + 2.0 * float(reach.max()) + slack + 1.0
            shift = (band - band.min()).astype(np.float64) * width
            key_lo, key_hi = key_lo + shift, key_hi + shift
            pad = 4.0 * float(np.spacing(np.abs(key_hi).max()))  # the rounding of the shifted ends
        order = np.argsort(key_lo, kind="stable")
        start = key_lo[order]
        end = np.searchsorted(start, key_hi[order] + reach[order] + reach.max() + slack + pad, side="right")
        count = np.maximum(end - np.arange(len(order)) - 1, 0)
        first = np.repeat(np.arange(len(order)), count)
        second = first + 1 + np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
        a, b = order[first], order[second]
        apart = np.maximum(lo[b] - hi[a], lo[a] - hi[b]).max(axis=1, initial=-np.inf)
        near = apart <= reach[a] + reach[b] + slack
        a, b = a[near], b[near]
        return np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1)

    def _part_pair(self, prims: dict, arrays: tuple, a: int, b: int):
        """The edge-triangle crossings between parts ``a`` and ``b`` (indices into ``prims``, which may be one part)
        and their primitive pair of the smallest gap (``uipc_warp.validation.Gap``), without the pairs whose vertices
        never move: libuipc-warp's ``SurfaceChecker`` on the primitives of ``a`` that move against all of ``b``, and
        on those of ``a`` that never move against those of ``b`` that move. ``arrays``: the engine's contact body of
        every vertex (the checker compares no pair within one) and the vertices' thickness, on the device. None when
        the parts make more than ``_START_CHECK_PAIRS`` primitive pairs."""
        from uipc_warp.validation import Gap, SurfaceChecker  # noqa: PLC0415

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
            v1, c1, t1, e1 = (len(one[n]) for n in ("vertices", "codim", "triangles", "edges"))
            v2, c2, t2, e2 = (len(other[n]) for n in ("vertices", "codim", "triangles", "edges"))
            # point-triangle, point-edge, edge-edge and point-point pairs, and the crossings, edges against triangles
            size += (v1 + c1) * t2 + (v2 + c2) * t1 + c1 * e2 + c2 * e1 + e1 * e2 + c1 * c2 + e1 * t2 + e2 * t1
        if size > _START_CHECK_PAIRS:
            return None
        x, device = self._engine.vm.x, self.model.device
        chunk = 16384  # primitives of a kind on one side per launch: every launch stays below 2**31 threads

        body, thickness = arrays

        def checker(part: dict, k: int) -> SurfaceChecker:
            surface = {n: part[n][k : k + chunk] for n in prims}
            return SurfaceChecker(surface, body, device, thickness=thickness)

        found, best = 0, Gap(np.inf, np.inf, 0.0, "", ())
        for one, other in halves:
            for e, f in ((one["edges"], other["triangles"]), (other["edges"], one["triangles"])):
                for i in range(0, len(e), chunk):
                    for j in range(0, len(f), chunk):
                        surface = {"edges": e[i : i + chunk], "triangles": f[j : j + chunk], "vertices": np.zeros(0)}
                        found += SurfaceChecker(surface, body, device, thickness=thickness).intersections(x)
            pieces = [max(len(part[n]) for n in prims) for part in (one, other)]
            for i in range(0, pieces[0], chunk):
                for j in range(0, pieces[1], chunk):
                    gap = checker(one, i).min_gap(x, checker(other, j))
                    if gap.gap < best.gap:
                        best = gap
        return found, best

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
        """Forget what the solver wrote for the bodies and the joints, so that the next step takes their state from
        ``state_in`` (NaN equals nothing)."""
        nb = self.model.body_count
        if nb:
            self._body_q_written.assign(np.full((nb, 7), np.nan, np.float32))
            self._body_qd_written.assign(np.full((nb, 6), np.nan, np.float32))
            self._body_hold.zero_()  # the kinematic bodies moving toward their aims take state_in too
        if self.model.joint_count:
            self._joint_q_written.fill_(np.nan)

    def _make_engine(self, capacity: dict | None) -> None:
        """The engine of the laid-out scene, with the current gravity and masses; ``capacity``: its list
        capacities (``SimEngine``'s argument; None for its defaults)."""
        device = self.model.device
        inputs = self._scene.engine_inputs(self._config, device)
        self._forces = None
        if len(self._force_blocks):
            self._forces = self._uipc_warp.arrays.ExternalAcceleration("newton_forces", self._force_blocks, device)
            inputs["elements"].append(self._forces)
        self._engine = self._uipc_warp.make_engine(
            self._config, self._options, **inputs, device=device, capacity=capacity
        )
        # the Newton and PCG iterations of the captured steps since the last graph_outdated()
        self._iterations = wp.zeros(2, dtype=wp.int64, device=device)
        # what the start check needs of the tabular, its two tables apart as the engine tests them: the contact
        # element and the subscene of every vertex and half plane, the elements' mask and the subscenes' mask
        self._vertex_thickness = np.asarray(inputs["thickness"], np.float64)
        tabular = inputs["contact_tabular"]
        n, n_planes = len(inputs["x"]), len(inputs["plane_P"])
        if tabular is None:
            self._cids, self._scids = np.zeros(n, np.int64), np.zeros(n, np.int64)
            self._plane_cids, self._plane_scids = np.zeros(n_planes, np.int64), np.zeros(n_planes, np.int64)
            self._enabled, self._subscene_enabled = np.ones((1, 1), bool), np.ones((1, 1), bool)
        else:
            self._cids, self._scids = (np.asarray(tabular[k], np.int64) for k in ("cids", "scids"))
            self._plane_cids, self._plane_scids = (
                np.asarray(tabular[k], np.int64) for k in ("plane_cids", "plane_scids")
            )
            self._enabled = np.asarray(tabular["enabled"]) != 0
            self._subscene_enabled = np.asarray(tabular["subscene_enabled"]) != 0
        if self.model.body_count and self._config.contact_constitution != "al-ipc":
            # notify_model_changed may have changed them since the scene was laid out (AL-IPC: it builds it again)
            self._engine.set_dyadic_mass(*self._engine_dyadic(self._dyadic))
        self._joint_groups = self._uipc_warp.arrays.joint_groups(self._engine)
        self._aim_group = self._uipc_warp.arrays.constraint_groups(self._engine).transform  # None without aims
        # the stiffness of every stitch group and the attachment or spring of each of its rows (_stitch_stiffness);
        # the groups of the distance joints' stitches, which join two bodies and so never share a group with an
        # attachment or a spring, apart (_distance_stiffness)
        self._stitch_rows, self._distance_rows = [], []
        for g in self._uipc_warp.arrays.stitch_groups(self._engine):
            distance = g.rows >= self._distance_first
            if not distance.any():
                self._stitch_rows.append(
                    (g.kappa, wp.array(self._stitch_source[g.rows], dtype=wp.int32, device=device))
                )
            elif distance.all():
                rows = wp.array((g.rows - self._distance_first).astype(np.int32), dtype=wp.int32, device=device)
                self._distance_rows.append((g.kappa, g.rest_length, rows))
            else:
                raise RuntimeError("SolverIPC: a stitch group holds distance joints and attachments or springs.")
        if self._distance_rows:
            self._distance_joints_device = wp.array(self._distance_joints, dtype=wp.int32, device=device)
        self._find_body_stiffness()
        self._make_contact_export()

    def _make_contact_export(self) -> None:
        """libuipc-warp's ``ContactForces`` of the engine, at the request and whenever the engine is built again,
        outside any capture: once a container requested ``CONTACT_F`` (class docstring, **Contact output**), with
        what the contact rows read besides it, for a model whose contact holds a shape (one without has no rows); and
        for a coupled solver's proxies, whose feedback is the force of the contact terms on them (**Coupling**). None
        otherwise."""
        self._contact_export = None
        self._export_step = -1  # the step whose records the export holds
        if self._engine is None:
            return
        rows = self._contact_output and len(self._contact_shapes) > 0
        if not (rows or self._proxies):
            return
        device = self.model.device
        self._contact_export = self._uipc_warp.arrays.ContactForces(self._engine)
        records = self._contact_export.capacity
        blocks = max(1, min(-(-records // _CONTACT_ROW_BLOCK), 4 * device.sm_count))
        self._contact_threads = blocks * _CONTACT_ROW_BLOCK
        if not rows:
            return
        self._contact_vertex_shape = wp.array(self._vertex_shape, dtype=wp.int32, device=device)
        P, N, shapes = self._planes
        n_planes = max(len(shapes), 1)
        self._contact_plane_shape = wp.array(
            np.asarray(shapes, np.int32) if len(shapes) else np.full(1, -1, np.int32), dtype=wp.int32, device=device
        )
        self._contact_plane_P = wp.array(np.asarray(P, np.float64).reshape(-1, 3), dtype=wp.vec3d, device=device)
        self._contact_plane_N = wp.array(np.asarray(N, np.float64).reshape(-1, 3), dtype=wp.vec3d, device=device)
        if not len(shapes):
            self._contact_plane_P = wp.zeros(n_planes, dtype=wp.vec3d, device=device)
            self._contact_plane_N = wp.zeros(n_planes, dtype=wp.vec3d, device=device)
        self._no_transform = wp.zeros(1, dtype=wp.transform, device=device)
        self._no_vec3 = wp.zeros(1, dtype=wp.vec3, device=device)

    def _find_body_stiffness(self) -> None:
        """With the default ``body_kappa``: the engine's orthogonality potential (libuipc-warp's ``orthogonality``),
        whose stiffness array a kernel writes before every step, the body of each of its rows, and the largest
        principal moment of inertia over the volume of that body (0 for a kinematic body and the world's). None
        without bodies or with a given ``body_kappa``, which the engine is built with."""
        self._ortho = self._ortho_inertia = None
        if self._body_kappa is not None or self.model.body_count == 0:
            return
        self._ortho = self._uipc_warp.arrays.orthogonality(self._engine)
        self._ortho_body = self._ortho.bodies
        self._ortho_inertia = wp.array(self._ortho_ratios(), dtype=wp.float64, device=self.model.device)

    def _ortho_ratios(self) -> np.ndarray:
        """For every row of the orthogonality group, its body's largest principal moment of inertia about the
        centre of mass over its volume, from the dyadic mass the engine holds; 0 for a kinematic body and the
        world's."""
        mass, first, second = self._dyadic
        nb = self.model.body_count
        ratio = np.zeros(len(self._ortho_body))
        for i, b in enumerate(self._ortho_body):
            if b >= nb or self._kinematic_body[b] or mass[b] <= 0.0:
                continue
            c = first[b] / mass[b]
            spread = np.linalg.eigvalsh(second[b] - mass[b] * np.outer(c, c))
            ratio[i] = (spread.sum() - spread.min()) / self._body_volume[b]
        return ratio

    def _write_body_stiffness(self, dt: float) -> None:
        """The orthogonality stiffness of every body for a step of ``dt`` (default ``body_kappa``, class docstring,
        **Rigid bodies**): libuipc's, lowered to 1 / (4 _BODY_TURN^2) times the largest principal moment of inertia
        over ``V dt^2``, on the device, so that a captured step takes the stiffness of its own ``dt``."""
        if self._ortho is None:
            return
        wp.launch(
            _body_stiffness,
            len(self._ortho_body),
            inputs=[self._ortho_inertia, wp.float64(_BODY_KAPPA), wp.float64(1.0 / (4.0 * _BODY_TURN**2 * dt * dt))],
            outputs=[self._ortho.kappa],
            device=self.model.device,
        )

    def _add_particles_and_tets(self, scene, elements: _ContactElements) -> np.ndarray:
        """Every particle as a finite-element vertex (engine index = particle index), and the tetrahedra."""
        model = self.model
        n = model.particle_count
        self._fixed_particle = np.zeros(n, bool)
        if n == 0:
            return np.zeros((0, 4), np.int64)
        mass = model.particle_mass.numpy().astype(np.float64)
        flags = model.particle_flags.numpy()
        self._fixed_particle = (mass <= 0.0) | ((flags & int(ParticleFlags.ACTIVE)) == 0)
        # a free particle is a sphere of its radius; the vertices of cloth and tetrahedra take the solver's thickness
        radius = model.particle_radius.numpy().astype(np.float64)
        scene.add_vertices(
            model.particle_q.numpy().astype(np.float64),
            model.particle_qd.numpy().astype(np.float64),
            mass,
            fixed=self._fixed_particle,
            thickness=np.where(elements.on_mesh, self._thickness, radius),
            contact_element=elements.particle,
            subscene_element=0 if self._worlds is None else self._worlds.subscene(model.particle_world.numpy()),
            gravity=self._world_gravity()[model.particle_world.numpy()],
        )
        own = _own_elements(model)[1]  # in a coupled solver's view, without another entry's tetrahedra
        if not own.any():
            return np.zeros((0, 4), np.int64)
        tets = model.tet_indices.numpy().reshape(-1, 4).astype(np.int64)[own]
        Dm_inv = model.tet_poses.numpy().astype(np.float64).reshape(-1, 3, 3)[own]
        volume = 1.0 / (6.0 * np.abs(np.linalg.det(Dm_inv)))
        materials = model.tet_materials.numpy().astype(np.float64).reshape(-1, 3)[own]
        # k_mu and k_lambda are Lame parameters; libuipc's StableNeoHookean.apply_to stores mu and lambda + mu, the
        # SNK1 parameters whose linearization is Hooke's law with those Lame parameters (stable_neo_hookean.cpp)
        k_mu, k_lambda = materials[:, 0], materials[:, 1]
        scene.add_tetrahedra(tets, Dm_inv, volume, k_mu, k_lambda + k_mu)
        return tets

    def _add_cloth(self, scene, tets: np.ndarray) -> None:
        """Triangles not on a tetrahedral body: Baraff-Witkin membrane and discrete-shell hinges; in a coupled solver's
        view, not another entry's (``_own_elements``)."""
        model = self.model
        if model.tri_count == 0:
            return
        tris = model.tri_indices.numpy().reshape(-1, 3).astype(np.int64)
        on_tet = np.zeros(model.particle_count, bool)
        on_tet[tets.ravel()] = True
        cloth = ~on_tet[tris].all(axis=1) & _own_elements(model)[0]
        if not cloth.any():
            return
        tris = tris[cloth]
        mat = model.tri_materials.numpy().astype(np.float64).reshape(-1, 5)[cloth]
        k_mu, k_lambda = mat[:, 0], mat[:, 1]
        stretch = k_mu + 0.5 * k_lambda
        shear = 0.5 * k_mu
        # Newton's tri_poses are the inverses of the rest edges in each triangle's edge frame, the frame add_shell
        # takes them in, computed by Newton's builders in float64 from particle_q and stored in float32: a triangle
        # whose tri_poses are the rounding of the rest edges of particle_q takes those in float64 (_rest_edges)
        x = model.particle_q.numpy().astype(np.float64)
        rest_edges = _rest_edges(x, tris, model.tri_poses.numpy().reshape(-1, 2, 2)[cloth])

        edges, kappa, angles = np.zeros((0, 2), np.int64), np.zeros(0), None
        if model.edge_count:
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
            # Newton's rest angle of a row by the winding of its triangles (add_shell's convention)
            signs = _winding_signs(tris, e, np.flatnonzero(sel))
            x0, x1, x2, x3 = x[e[:, 0]], x[e[:, 1]], x[e[:, 2]], x[e[:, 3]]
            # Newton's builders measure a row's rest angle at particle_q in float64 and store it in float32. When every
            # edge_rest_angle is that measure, the rest angles are those of the start, and the engine takes them from
            # the rest positions, as libuipc takes them from its rest geometry, with the dihedral angle its bending
            # energy evaluates: libuipc's arccosine of the normals' cosine reads up to 1.5e-8 rad on a flat hinge,
            # which Newton measures as 0, so that a rest angle of 0 would push a flat cloth out of its plane
            rest = model.edge_rest_angle.numpy()[sel]
            at_start = _dihedral_angles(x0, x1, x2, x3, x2, x3).astype(np.float32) == rest
            angles = None if at_start.all() else signs * rest
            L0 = np.linalg.norm(x3 - x2, axis=1)
            area = 0.5 * (
                np.linalg.norm(np.cross(x2 - x0, x3 - x0), axis=1) + np.linalg.norm(np.cross(x2 - x1, x3 - x1), axis=1)
            )
            h_bar = area / (3.0 * np.maximum(L0, 1.0e-30))
            edges, kappa = e[:, 2:4], 0.5 * ke * h_bar
        scene.add_shell(
            tris,
            stretch,
            stretch,
            shear,
            self._cloth_strain_rate,
            bending=kappa,
            bending_edges=edges,
            rest_edges=rest_edges,
            rest_angles=angles,
        )

    def _add_springs(self, scene) -> None:
        """Newton's springs of positive rest length as libuipc's Hookean springs: ``0.5 ke (l - L0)^2`` each
        (``spring_kappa`` with a radius of 1 m, which only scales the energy, since a spring is no contact
        primitive). Those of rest length 0 become stitches (``_add_stitches``)."""
        model = self.model
        self._zero_springs = np.zeros(0, np.int64)
        if model.spring_count == 0:
            return
        edges = model.spring_indices.numpy().reshape(-1, 2).astype(np.int64)
        rest = model.spring_rest_length.numpy().astype(np.float64)
        ke = model.spring_stiffness.numpy().astype(np.float64)
        own = _own_elements(model)[2]  # in a coupled solver's view, not another entry's
        self._zero_springs = np.flatnonzero((rest == 0.0) & own)
        hookean = (rest > 0.0) & own
        if not hookean.any():
            return
        kappa = self._uipc_warp.arrays.spring_kappa(ke[hookean], rest[hookean], 1.0)
        scene.add_springs(edges[hookean], kappa, rest_lengths=rest[hookean], radius=1.0)

    def _plan_attachments(self) -> None:
        """The anchors of every body that attachments hold (class docstring, **Attachments**): the distinct
        ``attachment_particle_body_body_point`` of its attachments in the order they first appear, and for every
        attachment its anchor among its body's."""
        model = self.model
        count = model.attachment_particle_body_count
        self._body_anchors = {}
        self._attachment_anchor = np.zeros(count, np.int64)
        self._attachment_inputs = ()
        if count == 0:
            return
        body = model.attachment_particle_body_body.numpy().astype(np.int64)
        particle = model.attachment_particle_body_particle.numpy().astype(np.int64)
        point = model.attachment_particle_body_body_point.numpy()
        self._attachment_inputs = (body, particle, point.copy())
        for b in np.unique(body):
            rows = np.flatnonzero(body == b)
            points, first, inverse = np.unique(point[rows], axis=0, return_index=True, return_inverse=True)
            order = np.argsort(first)
            rank = np.empty_like(order)
            rank[order] = np.arange(len(order))
            self._body_anchors[int(b)] = points[order].astype(np.float64)
            self._attachment_anchor[rows] = rank[inverse.reshape(-1)]

    def _plan_distance_joints(self) -> None:
        """The anchors of the distance joints of one length (``enable_distance_joints``, class docstring, **Distance
        joints**): one on either side at the origin of its joint frame, ``joint_X_p`` in the parent's frame, the
        fixed body that stands for the world for a parent of -1, and ``joint_X_c`` in the child's, after the anchors
        of the attachments; for every such joint the body and the anchor of its two sides."""
        model = self.model
        self._distance_joints = np.zeros(0, np.int32)
        self._distance_anchor = np.zeros((0, 2, 2), np.int64)  # joint, side (parent, child), (body, anchor)
        if model.joint_count == 0 or not self._joint_options.distance:
            return
        joints = _map_joints(model, self._joint_options).distance
        parent, child = model.joint_parent.numpy()[joints], model.joint_child.numpy()[joints]
        X_p, X_c = model.joint_X_p.numpy().astype(np.float64), model.joint_X_c.numpy().astype(np.float64)
        anchor = np.zeros((len(joints), 2, 2), np.int64)
        for k, j in enumerate(joints):
            sides = ((parent[k] if parent[k] >= 0 else model.body_count, X_p[j, :3]), (child[k], X_c[j, :3]))
            for side, (b, point) in enumerate(sides):
                points = self._body_anchors.get(int(b), np.zeros((0, 3)))
                anchor[k, side] = (b, len(points))
                self._body_anchors[int(b)] = np.vstack([points, point])
        self._distance_joints, self._distance_anchor = joints, anchor

    def _distance_penalties(self, mass: np.ndarray) -> np.ndarray:
        """The penalty of every distance joint of one length: a ball joint's, ``joint_strength_ratio`` times the
        mass of the dynamic bodies it joins, in the units of the incremental potential, so that its stitch's
        stiffness is that over ``dt^2``."""
        if not len(self._distance_joints):
            return np.zeros(1)
        dynamic = np.append(np.where(self._kinematic_body, 0.0, mass), 0.0)  # the world's after Newton's bodies
        bodies = self._distance_anchor[:, :, 0]
        return self._joint_strength_ratio * dynamic[bodies].sum(axis=1)

    def _attachment_penalties(self, mass: np.ndarray) -> np.ndarray:
        """The penalty of every attachment if it is hard (infinite stiffness): a ball joint's, ``joint_strength_ratio``
        times the dynamic masses it joins (a particle that is not fixed, a dynamic body), in the units of the
        incremental potential, so that its stiffness is that over ``dt^2``."""
        model = self.model
        count = model.attachment_particle_body_count
        if count == 0:
            return np.zeros(1)
        body = model.attachment_particle_body_body.numpy().astype(np.int64)
        particle = model.attachment_particle_body_particle.numpy().astype(np.int64)
        particle_mass = np.where(self._fixed_particle, 0.0, model.particle_mass.numpy().astype(np.float64))
        body_mass = np.where(self._kinematic_body, 0.0, mass)
        return self._joint_strength_ratio * (particle_mass[particle] + body_mass[body])

    def _add_stitches(self, scene) -> None:
        """Newton's particle-body attachments and springs of rest length 0 as libuipc's soft vertex stitches of rest
        length 0 (class docstring, **Attachments**): an attachment between its particle and its body's anchor, a
        spring between its two particles. Keeps the attachment of every stitch, or -1 less its spring, for the
        stiffness written at every step."""
        model = self.model
        pairs, source = [], []
        if model.attachment_particle_body_count:
            body, particle, _ = self._attachment_inputs
            anchor = scene.layout().body_anchor_offsets[body] + self._attachment_anchor
            pairs.append(np.stack([particle, anchor], axis=1))
            source.append(np.arange(len(body)))
        if len(self._zero_springs):
            pairs.append(model.spring_indices.numpy().reshape(-1, 2).astype(np.int64)[self._zero_springs])
            source.append(-1 - self._zero_springs)
        self._stitch_source = np.concatenate(source).astype(np.int32) if source else np.zeros(0, np.int32)
        if pairs:
            scene.add_stitches(np.concatenate(pairs), self._stitch_kappa(self._config.dt), 0.0)
        # the distance joints of one length: a stitch of that rest length between the anchors of their two sides,
        # after the other stitches (class docstring, **Distance joints**)
        self._distance_first = len(self._stitch_source)
        if len(self._distance_joints):
            offsets = scene.layout().body_anchor_offsets
            a = self._distance_anchor
            ends = np.stack([offsets[a[:, 0, 0]] + a[:, 0, 1], offsets[a[:, 1, 0]] + a[:, 1, 1]], axis=1)
            dt = self._config.dt
            model = self.model
            first = model.joint_qd_start.numpy()[self._distance_joints]
            length = model.joint_limit_lower.numpy()[first].astype(np.float64)
            kappa = self._distance_hard.numpy() / (dt * dt)
            kappa[~model.joint_enabled.numpy()[self._distance_joints]] = 0.0
            self._distance_first = scene.add_stitches(ends, kappa, length)

    def _stitch_kappa(self, dt: float) -> np.ndarray:
        """The stiffness of every stitch for a step of ``dt``, as ``_stitch_stiffness`` writes it, on the host."""
        model, source = self.model, self._stitch_source
        kappa = np.zeros(len(source))
        attached = source >= 0
        if attached.any():
            k = model.attachment_particle_body_stiffness.numpy().astype(np.float64)
            hard = np.isinf(k)
            k[hard] = self._attachment_hard.numpy()[hard] / (dt * dt)
            k[~model.attachment_particle_body_enabled.numpy()] = 0.0
            kappa[attached] = k[source[attached]]
        if (~attached).any():
            kappa[~attached] = model.spring_stiffness.numpy().astype(np.float64)[-1 - source[~attached]]
        return kappa

    def _write_stitch_stiffness(self, dt: float) -> None:
        """The stiffness of every stitch for a step of ``dt`` (``_stitch_stiffness``), on the device, so that a
        captured step takes the attachments' stiffness and enabled flags of its replay, and the hard ones' penalty
        for its own ``dt``; and of the distance joints' stitches (``_distance_stiffness``)."""
        model = self.model
        for kappa, rest_length, source in self._distance_rows:
            wp.launch(
                _distance_stiffness,
                source.shape[0],
                inputs=[
                    source,
                    self._distance_joints_device,
                    model.joint_qd_start,
                    model.joint_enabled,
                    model.joint_limit_lower,
                    self._distance_hard,
                    wp.float64(1.0 / (dt * dt)),
                ],
                outputs=[kappa, rest_length],
                device=model.device,
            )
        if not self._stitch_rows:
            return
        model = self.model
        attached = model.attachment_particle_body_count > 0
        springs = model.spring_count > 0
        for kappa, source in self._stitch_rows:
            wp.launch(
                _stitch_stiffness,
                source.shape[0],
                inputs=[
                    source,
                    model.attachment_particle_body_stiffness if attached else self._no_float32,
                    model.attachment_particle_body_enabled if attached else self._no_bool,
                    self._attachment_hard,
                    model.spring_stiffness if springs else self._no_float32,
                    wp.float64(1.0 / (dt * dt)),
                ],
                outputs=[kappa],
                device=model.device,
            )

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
        self._kinematic_body = _kinematic_bodies(model)
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

    def _scene_inputs_now(self) -> tuple:
        """What the scene was laid out from besides gravity and the bodies' mass properties, which change in place:
        the triangulated shapes, the friction coefficients of the contact models, the contact thicknesses, the
        particles' masses and the springs."""
        model = self.model
        materials = (float(model.soft_contact_mu), float(model.particle_mu))
        arrays = [model.particle_radius, model.particle_mass] if model.particle_count else []
        if model.spring_count:
            arrays += [model.spring_rest_length, model.spring_stiffness, model.spring_damping]
        if model.shape_count:
            arrays += [model.shape_transform, model.shape_scale, model.shape_material_mu, model.shape_margin]
        return (*materials, *(a.numpy().copy() for a in arrays))

    def _joint_inputs_now(self) -> tuple:
        """What the joints were laid out from: their types, bodies and frames. The other joint data are read at every
        step."""
        model = self.model
        if model.joint_count == 0:
            return ()
        arrays = (model.joint_type, model.joint_parent, model.joint_child, model.joint_X_p, model.joint_X_c)
        arrays = (*arrays, model.joint_axis, model.joint_qd_start, model.joint_dof_dim)
        return tuple(a.numpy().copy() for a in arrays)

    def _add_shapes_and_bodies(self, scene, elements: _ContactElements) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Static non-plane shapes as fixed triangle meshes; every body as an affine body. Returns the bodies'
        dyadic masses."""
        model = self.model
        meshes: dict[int, list] = {b: [] for b in range(model.body_count)}
        self._obstacle_shapes = []  # the shape of every obstacle, in the order they are added
        # the subscene of every body's rows: that of its surface vertices, as the engine takes it (one subscene)
        self._body_row_subscene = np.zeros(model.body_count, np.int64)
        if model.shape_count:
            shape_type = model.shape_type.numpy()
            shape_body = model.shape_body.numpy()
            colliding = _colliding_shapes(model)
            shape_xform = model.shape_transform.numpy().astype(np.float64)
            margin = model.shape_margin.numpy().astype(np.float64)
            for s in range(model.shape_count):
                if not colliding[s]:
                    continue
                b = int(shape_body[s])
                if shape_type[s] == GeoType.PLANE:
                    if b != -1:
                        raise NotImplementedError(f"SolverIPC does not support the plane shape {s} on a body.")
                    continue
                v, f = _shape_mesh(model, s, self._shape_segments)
                if len(f) == 0:
                    continue  # a height field without a cell
                v = _quat_rotate_np(shape_xform[s, 3:7], v) + shape_xform[s, :3]
                if b == -1:
                    scene.add_obstacle(
                        v,
                        f,
                        thickness=margin[s],
                        contact_element=elements.shape[s],
                        subscene_element=self._subscene_of_shape(s),
                    )
                    self._obstacle_shapes.append(s)
                else:
                    meshes[b].append((v, f, s))
        # the shape of every surface vertex of a body, in the order of its parts (the contact rows of CONTACT_F)
        self._body_vertex_shape = [
            np.concatenate([np.full(len(v), s, np.int32) for v, _, s in meshes[b]])
            if meshes[b]
            else np.zeros(0, np.int32)
            for b in range(model.body_count)
        ]

        mass, first, second = self._body_mass_moments(stacklevel=5)
        # kinematic_motion="aim": the kinematic bodies with colliding shapes, which move toward their aims
        self._aim_bodies = np.zeros(0, np.int64)
        self._aim_mesh_moments = {}
        # the bodies whose start the caller writes into the state, which the first step checks (_check_start)
        self._posed_body = np.zeros(model.body_count, bool)
        if model.body_count == 0:
            return mass, first, second
        if self._aims:
            self._aim_bodies = np.flatnonzero(self._kinematic_body & np.array([bool(meshes[b]) for b in meshes]))
            for b in self._aim_bodies:
                if mass[b] <= 0.0:  # the mass of its collision meshes at 1000 kg/m^3
                    self._aim_mesh_moments[int(b)] = self._mesh_moments(int(b), meshes[b])
            mass, first, second = self._with_aim_moments((mass, first, second))
        body_q, body_qd, placed = self._start_bodies()
        # a kinematic body, whose pose the caller writes at every step, and a link the model leaves at the identity,
        # which the caller places by writing forward kinematics into the state (class docstring, **State**)
        self._posed_body = self._kinematic_body | placed
        body_com = model.body_com.numpy().astype(np.float64)
        body_mass = model.body_mass.numpy().astype(np.float64)
        body_gravity = self._world_gravity()[model.body_world.numpy()]
        self._body_volume = np.zeros(model.body_count)
        for b in range(model.body_count):
            parts = meshes[b]
            offsets = np.cumsum([0] + [len(v) for v, _, _ in parts])
            xbar = np.concatenate([v for v, _, _ in parts]) if parts else np.zeros((0, 3))
            tris = (
                np.concatenate([f + o for (_, f, _), o in zip(parts, offsets[:-1], strict=True)])
                if parts
                else np.zeros((0, 3), np.int64)
            )
            # every vertex in the element and the subscene of its shape, with the shape's margin as its thickness
            element = np.concatenate([np.full(len(v), elements.shape[s]) for v, _, s in parts]) if parts else 0
            subscene = (
                np.concatenate([np.full(len(v), self._subscene_of_shape(s)) for v, _, s in parts])
                if parts
                else self._subscene_of_body(b)
            )
            self._body_row_subscene[b] = np.atleast_1d(subscene)[0]
            thickness = np.concatenate([np.full(len(v), margin[s]) for v, _, s in parts]) if parts else 0.0
            volume = sum(_closed_volume(v, f) for v, f, _ in parts)
            if volume <= 0.0:
                # no closed collision mesh: the volume of water of that mass
                volume = max(body_mass[b], 1.0e-12) / 1000.0
            self._body_volume[b] = volume
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
                kappa=_BODY_KAPPA if self._body_kappa is None else self._body_kappa,
                p=body_q[b, :3],
                A=R,
                p_dot=body_qd[b, :3] - np.cross(w, R @ body_com[b]),
                A_dot=W @ R,
                fixed=bool(self._kinematic_body[b]) and b not in self._aim_bodies,
                thickness=thickness,
                contact_element=element,
                subscene_element=subscene,
                # a kinematic body moving toward its aim falls with nothing: gravity does not move it
                gravity=np.zeros(3) if b in self._aim_bodies else body_gravity[b],
                anchors=self._body_anchors.get(b),
            )
        return mass, first, second

    def _mesh_moments(self, b: int, parts: list) -> tuple[float, np.ndarray, np.ndarray]:
        """The dyadic mass of body ``b``'s collision meshes ``parts`` at 1000 kg/m^3, for a kinematic body of mass 0
        that moves toward its aim (``kinematic_motion="aim"``); ``NotImplementedError`` when they enclose no volume."""
        m, mx, S = 0.0, np.zeros(3), np.zeros((3, 3))
        for v, f, _ in parts:
            pm, pmx, pS, vol = self._uipc_warp.arrays.mesh_dyadic_mass(v, f, 1000.0)
            sign = -1.0 if vol < 0.0 else 1.0  # a mesh wound inward
            m, mx, S = m + sign * pm, mx + sign * pmx, S + sign * pS
        if not m > 0.0:
            raise NotImplementedError(
                f"SolverIPC: kinematic_motion='aim' moves the kinematic body {b} with its mass; it has none, and its "
                "collision shapes enclose no volume to give it one at 1000 kg/m^3."
            )
        return m, mx, S

    def _with_aim_moments(self, dyadic: tuple) -> tuple:
        """``dyadic`` with the mass moments of the kinematic bodies that move toward their aims (``kinematic_motion=
        "aim"``): Newton's for a body of positive mass, which needs a positive-definite inertia, else those of its
        collision meshes (``_mesh_moments``)."""
        if not len(self._aim_bodies):
            return dyadic
        mass, first, second = (a.copy() for a in dyadic)
        inertia = self.model.body_inertia.numpy().astype(np.float64)
        for b in self._aim_bodies:
            if int(b) in self._aim_mesh_moments:
                mass[b], first[b], second[b] = self._aim_mesh_moments[int(b)]
            elif np.linalg.eigvalsh(0.5 * (inertia[b] + inertia[b].T)).min() <= 0.0:
                raise ValueError(
                    f"SolverIPC needs a positive-definite inertia for the kinematic body {b}, which "
                    "kinematic_motion='aim' moves with its mass."
                )
        return mass, first, second

    def _start_bodies(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The poses and twists the bodies are built at and the start check tests (class docstring, **State**):
        ``model.body_q`` and ``model.body_qd``, but for a body whose pose there is the identity, where
        ``ModelBuilder.add_link`` and ``add_body`` put a body given no transform: it starts where ``newton.eval_fk``
        puts it from ``model.joint_q`` and ``model.joint_qd``, the pose with which a maximal-coordinate solver starts
        once the caller has written forward kinematics into the state. Also the bodies that forward kinematics put
        elsewhere than the identity: the links whose start the caller places."""
        model = self.model
        body_q = model.body_q.numpy().astype(np.float64)
        body_qd = model.body_qd.numpy().astype(np.float64)
        identity = np.array([0, 0, 0, 0, 0, 0, 1], np.float32)
        unplaced = np.all(model.body_q.numpy() == identity, axis=1)
        if not unplaced.any() or model.articulation_count == 0:
            return body_q, body_qd, np.zeros(model.body_count, bool)
        fk = SimpleNamespace(body_q=wp.clone(model.body_q), body_qd=wp.clone(model.body_qd))
        eval_fk(model, model.joint_q, model.joint_qd, fk)
        fk_q = fk.body_q.numpy()
        body_q[unplaced] = fk_q.astype(np.float64)[unplaced]
        body_qd[unplaced] = fk.body_qd.numpy().astype(np.float64)[unplaced]
        return body_q, body_qd, unplaced & ~np.all(fk_q == identity, axis=1)

    def _add_joints(self, scene) -> None:
        """Newton's revolute, prismatic, fixed and ball joints as libuipc's (class docstring, **Joints**), and with
        ``enable_d6_joints`` the D6 joints that lock what one of them locks (``_map_joints``), joint frames taken as the
        reference rotations, a fixed affine body without mass standing for the world. Sets up what the
        per-step kernels read: per kind the joint of every group row and the row's penalty (``_joint_strength``,
        0 without a dynamic body), per joint its row (``_joint_element``), and the free and ball joints, whose
        ``joint_f`` acts through the bodies' wrenches."""
        model, device = self.model, self.model.device
        self._world_body = -1
        self._joint_rows, self._joint_strength, self._joint_history, self._joint_rate = {}, {}, {}, {}
        self._joint_bodies = {}
        if model.joint_count == 0:
            return
        # the libuipc joint of every joint: its own type, but for a D6 joint (enable_d6_joints) that of the joint
        # that locks what it locks; the type the kernels that write the coordinates and joint_f take it for
        mapping = _map_joints(model, self._joint_options)
        joint_type = mapping.kind
        self._joint_store_type = model.joint_type
        if mapping.relabeled:
            self._joint_store_type = wp.array(mapping.store, dtype=wp.int32, device=device)
        self._d6_joints = wp.array(mapping.d6, dtype=wp.int32, device=device)
        parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
        X_p, X_c = model.joint_X_p.numpy().astype(np.float64), model.joint_X_c.numpy().astype(np.float64)
        axis = model.joint_axis.numpy().astype(np.float64).reshape(-1, 3)
        qd_start = model.joint_qd_start.numpy()
        wrenched = np.isin(mapping.store, [int(JointType.FREE), int(JointType.BALL)])
        self._wrenched_joints = wp.array(np.flatnonzero(wrenched).astype(np.int32), dtype=wp.int32, device=device)
        element = np.full(model.joint_count, -1, np.int32)
        joints = np.flatnonzero(joint_type != int(JointType.FREE))
        world = np.concatenate([joints, mapping.distance])  # the distance joints' stitches need it too
        if len(world) and np.any(parent[world] < 0):
            self._world_body = model.body_count
            scene.add_affine_body(
                np.zeros((0, 3)),
                np.zeros((0, 3), np.int64),
                mass=0.0,
                first_moment=np.zeros(3),
                second_moment=np.zeros((3, 3)),
                volume=0.0,
                fixed=True,
                anchors=self._body_anchors.get(model.body_count),
            )
        for kind, t in (
            ("revolute", JointType.REVOLUTE),
            ("prismatic", JointType.PRISMATIC),
            ("fixed", JointType.FIXED),
            ("spherical", JointType.BALL),
        ):
            rows = joints[joint_type[joints] == int(t)]
            if not len(rows):
                continue
            element[rows] = np.arange(len(rows), dtype=np.int32)
            left = np.where(parent[rows] >= 0, parent[rows], self._world_body)
            bodies = np.stack([left, child[rows]], axis=1)
            self._joint_bodies[kind] = bodies
            R_p = np.array([_quat_to_matrix_np(q) for q in X_p[rows, 3:7]]).reshape(-1, 3, 3)
            R_c = np.array([_quat_to_matrix_np(q) for q in X_c[rows, 3:7]]).reshape(-1, 3, 3)
            p_p, p_c = X_p[rows, :3], X_c[rows, :3]
            # the rotations at which the joint frames coincide, in which Newton's joint coordinates are 0
            reference = np.stack([np.swapaxes(R_p, 1, 2), np.swapaxes(R_c, 1, 2)], axis=1)
            ratio, strength = self._joint_strengths(kind, self._dyadic[0])
            if kind in ("revolute", "prismatic"):
                a = axis[qd_start[rows]]
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                # the axis points: the joint origin and one metre along the axis, in either body's frame
                ends_p = np.stack([p_p, p_p + np.einsum("kij,kj->ki", R_p, a)], axis=1)
                ends_c = np.stack([p_c, p_c + np.einsum("kij,kj->ki", R_c, a)], axis=1)
                add = scene.add_revolute_joints if kind == "revolute" else scene.add_prismatic_joints
                add(bodies, ends_p, ends_c, strength_ratio=ratio, reference_rotations=reference)
            elif kind == "fixed":
                scene.add_fixed_joints(bodies, p_p, p_c, strength_ratio=ratio, reference_rotations=reference)
            else:
                scene.add_spherical_joints(bodies, p_p, p_c, strength_ratio=ratio)
            self._joint_rows[kind] = wp.array(rows.astype(np.int32), dtype=wp.int32, device=device)
            self._joint_strength[kind] = wp.array(strength, dtype=wp.float64, device=device)
            if kind in ("revolute", "prismatic"):
                # the coordinate at the start of this step, of the step before (BDF2's drives), and the one before
                # that step's, which grow() takes back when it solves that step again
                self._joint_history[kind] = tuple(
                    wp.zeros(len(rows), dtype=wp.float64, device=device) for _ in range(3)
                )
                if self._joint_options.rates:  # their rates likewise (_joint_rates)
                    self._joint_rate[kind] = tuple(
                        wp.zeros(len(rows), dtype=wp.float64, device=device) for _ in range(3)
                    )
        self._no_joint_state = wp.zeros(1, dtype=wp.float64, device=device)  # the state of a kind without joints
        self._joint_element = wp.array(element, dtype=wp.int32, device=device)
        self._joint_q_written = wp.full(model.joint_coord_count, np.nan, dtype=wp.float32, device=device)

    def _joint_strengths(self, kind: str, mass: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For the bodies' masses ``mass``, the strength ratio the joints of ``kind`` are built with and their
        penalty, ``joint_strength_ratio`` times the mass of the dynamic bodies a joint joins (the world and kinematic
        bodies count 0), where libuipc's ratio multiplies the masses of both bodies. The per-step kernels take the
        penalty: a fixed or ball joint's as it is, a revolute or prismatic joint's divided by the masses it was built
        with, which libuipc keeps."""
        bodies = self._joint_bodies[kind]
        everything = np.append(mass, 0.0)  # the world's mass after Newton's bodies
        dynamic = np.append(np.where(self._kinematic_body, 0.0, mass), 0.0)
        mass_sum, dynamic_sum = everything[bodies].sum(axis=1), dynamic[bodies].sum(axis=1)
        stiffness = self._joint_strength_ratio * dynamic_sum
        ratio = np.divide(stiffness, mass_sum, out=np.zeros(len(bodies)), where=mass_sum > 0.0)
        return ratio, stiffness

    def _engine_dyadic(self, dyadic: tuple) -> tuple:
        """The bodies' dyadic masses as the engine holds them: the world's (zero) after Newton's bodies."""
        if self._world_body < 0:
            return dyadic
        mass, first, second = dyadic
        return np.append(mass, 0.0), np.vstack([first, np.zeros((1, 3))]), np.concatenate([second, np.zeros((1, 3, 3))])

    def _collect_planes(self) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """The points, normals and shapes of the static planes that collide, each point moved out by the shape's
        margin."""
        model = self.model
        if model.shape_count == 0:
            return np.zeros((0, 3)), np.zeros((0, 3)), []
        shape_type = model.shape_type.numpy()
        shape_body = model.shape_body.numpy()
        colliding = _colliding_shapes(model)
        shape_xform = model.shape_transform.numpy()
        shape_scale = model.shape_scale.numpy()
        margin = model.shape_margin.numpy().astype(np.float64)
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
            n = _quat_rotate_np(xf[3:7], np.array([0.0, 0.0, 1.0]))
            # a half plane has no thickness: its margin moves its surface out along the normal
            P.append(xf[:3] + margin[s] * n)
            N.append(n)
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
        self.validate_observables(observables, contacts)
        if self._engine is None:
            self._write_contacts(observables, None)  # no particles and no bodies: no contact
            return
        if self._start_unchecked and self.model.device.is_capturing:
            self._start_in_capture()  # before anything is captured
        if self._restarts:  # an entry of a coupler that restarts its steps (class docstring, **Coupling**)
            if self._restore_pending:
                self._restore_step()
            elif self._save_pending:
                self._save_step()
            self._save_pending = self._restore_pending = False
        if control is None:
            control = self._control_of_model()
        self._dt, self._control = dt, control
        self._report_clamp(dt)
        self._load(state_in)
        if self._start_unchecked:
            # the first step, not captured: the start of the pairs with a body whose start the caller writes, on the
            # state just loaded from state_in, before anything runs from it (_check_start)
            self._check_start("the state of the first step (its state_in)")
            self._start_unchecked = False
        self._steps += 1
        self._write_forces(state_in, control)
        self._write_joint_controls(control, dt)
        self._write_stitch_stiffness(dt)
        self._write_body_stiffness(dt)
        if self.model.device.is_capturing:
            # the solve joins the caller's graph; it cannot synchronize, grow or raise (see the class docstring)
            self._engine.stop_on_overflow = False
            self._engine.advance(dt)
            self._uipc_warp.arrays.add_frame_iterations(self._engine, self._iterations)
            self._keep_status()
        else:
            status = self._advance_eager(dt)
            self._keep_status()
            if status.failed and self._failed_alone(status):
                # libuipc-warp put the worlds that failed back at their state at the step's start and stepped the
                # others: state_out holds both, then the step raises, naming the worlds that failed
                failed = self._failed_worlds()
                self._shift_joint_history()
                self._store(state_in.body_q, state_in.body_qd, state_out)
                self._write_observables(observables, state_out)
                raise RuntimeError(self._world_failure(failed))
            if status.failed:
                self._write_observables(observables, state_out)
                self._forget_written()  # the engine holds the failed solve: the next step takes state_in
                raise RuntimeError(self._failure(status))
        self._shift_joint_history()
        self._store(state_in.body_q, state_in.body_qd, state_out)
        self._write_observables(observables, state_out)

    def _start_in_capture(self) -> None:
        """The first step, captured in a graph: it cannot read its ``state_in``, so the pairs with a body whose start
        the caller writes keep the test of the construction, on the model's state; raise its failure, if any, before
        anything is captured (``_check_start``)."""
        if self._start_pending is not None:
            raise ValueError(
                f"{self._start_pending} The first step is captured in a graph, which cannot read its state_in, so the "
                "solver cannot check the start of kinematic bodies and of links that the model leaves at the identity "
                "on it, as a first step outside capture does. Take the first step outside graph capture, or put the "
                "start poses into model.body_q before building the solver."
            )
        self._start_unchecked = False

    def _control_of_model(self) -> Control:
        """The control of a step given none: the model's own control inputs, ``model.control(clone_variables=False)``
        built once, and again when the model holds other arrays for the inputs the solver reads."""
        model, control = self.model, self._default_control
        if control is None or any(
            getattr(control, name) is not getattr(model, name)
            for name in ("joint_target_q", "joint_target_qd", "joint_f")
        ):
            control = self._default_control = model.control(clone_variables=False)
        return control

    def _shift_joint_history(self) -> None:
        """BDF2: the joint coordinates at the start of this step become those of the step before, whose own are kept
        for grow() (on the device, also inside a captured graph); and so do their rates, where the joint options need
        them (``_joint_rates``)."""
        if self._config.integrator != "bdf2":
            return
        for q_start, q_prev, q_kept in (*self._joint_history.values(), *self._joint_rate.values()):
            wp.copy(q_kept, q_prev)
            wp.copy(q_prev, q_start)

    def _report_clamp(self, dt: float) -> None:
        """Warn once when a step's ``dt`` clamps the given ``kappa`` into libuipc's stiffness range, with independent
        worlds into the range of a world (each world has its own)."""
        kappa = self._config.kappa
        if not self._adaptive_kappa or self._clamp_reported or kappa is None or self._engine is None:
            return
        if dt == self._clamp_checked_dt:
            return
        self._clamp_checked_dt = dt
        policies = self._engine.world_contact_stiffness
        if policies is None:
            _, used, lo, hi, _ = self._engine.contact_stiffness.resolve(dt)
            if used != kappa:
                self._clamp_reported = True
                warnings.warn(
                    f"SolverIPC: kappa={kappa:g} is outside libuipc's contact stiffness range [{lo:.6g}, {hi:.6g}] Pa "
                    f"of this model at dt={dt:g} s; the steps use {used:.6g} Pa, as libuipc clamps a stiffness it is "
                    "given. The range scales with the mean vertex mass and with 1/dt^2 (contact_stiffness(dt) gives "
                    "the stiffness of any dt); kappa=None takes its lower bound, libuipc's default. Not repeated.",
                    NewtonWarning,
                    stacklevel=3,
                )
            return
        clamped = []
        for world, policy in zip(self._status_entries.tolist(), policies, strict=True):
            _, used, lo, hi, _ = policy.resolve(dt)
            if used != kappa:
                clamped.append(f"world {world} uses {used:.6g} Pa (range [{lo:.6g}, {hi:.6g}] Pa)")
        if clamped:
            self._clamp_reported = True
            shown = "; ".join(clamped[:4]) + (f"; and {len(clamped) - 4} more worlds" if len(clamped) > 4 else "")
            warnings.warn(
                f"SolverIPC: kappa={kappa:g} is outside libuipc's contact stiffness range of {len(clamped)} of the "
                f"{len(policies)} independent worlds at dt={dt:g} s, each world's range that of the world alone, and "
                f"those worlds' steps use their range's bound, as libuipc clamps a stiffness it is given: {shown}. "
                "A range scales with the world's mean vertex mass and with 1/dt^2 (contact_stiffness(dt) gives every "
                "world's stiffness of any dt); kappa=None takes every world's lower bound, libuipc's default. Not "
                "repeated.",
                NewtonWarning,
                stacklevel=3,
            )

    def _load(self, state_in: State) -> None:
        """The engine's state from ``state_in`` where it differs from what the solver wrote (class docstring)."""
        model, eng = self.model, self._engine
        device, vm = model.device, eng.vm
        self._changed.zero_()
        if model.particle_count:
            quiet = vm.fixed if self._particle_quiet is None else self._particle_quiet
            wp.launch(
                _load_particles,
                model.particle_count,
                inputs=[state_in.particle_q, state_in.particle_qd, quiet, self._particle_subscene],
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
                    self._body_quiet,
                    self._body_hold,
                    self._body_subscene,
                ],
                outputs=[vm.x, vm.v, self._changed],
                device=device,
            )
            self._uipc_warp.arrays.update_body_vertices(eng)  # surface vertices x = A x_bar + p
            self._aim_bodies_of(state_in)
            self._load_joints(state_in)
        # the subscenes of the particles and bodies that took state_in discard their friction, and under BDF2 their
        # step is an implicit Euler step (BDF1: nothing); with several worlds the other worlds go on as they were
        eng.discard_friction(where=self._changed)
        eng.restart_history(where=self._changed)

    def _aim_bodies_of(self, state_in: State) -> None:
        """``kinematic_motion="aim"``: the aims of the kinematic bodies for the step, from ``state_in`` (the poses
        advanced by the twists), with the velocities that put the step's prediction on them (``_aim_kinematic_bodies``);
        ``dt`` is the step's, frozen in a captured step as the joints' kernels freeze it."""
        if self._aim_group is None:
            return
        eng = self._engine
        bdf2 = self._config.integrator == "bdf2"
        wp.launch(
            _aim_kinematic_bodies,
            len(self._aim_bodies),
            inputs=[
                self._aim_rows,
                state_in.body_q,
                state_in.body_qd,
                self.model.body_com,
                self._body_base,
                eng.vm.x,
                wp.float64(self._dt),
                int(bdf2),
            ],
            outputs=[
                self._aim_group.q_aim,
                self._aim_group.active,
                eng.vm.v,
                eng.vm.x_nm1,
                eng.vm.v_nm1,
                self._body_hold,
            ],
            device=self.model.device,
        )

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
            self._joint_wrench.zero_()
            if model.joint_count and control.joint_f is not None and self._wrenched_joints.shape[0]:
                wp.launch(
                    _joint_wrenches,
                    self._wrenched_joints.shape[0],
                    inputs=[
                        self._wrenched_joints,
                        self._joint_store_type,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_X_p,
                        model.joint_qd_start,
                        model.joint_enabled,
                        control.joint_f,
                        state_in.body_q,
                        model.body_com,
                    ],
                    outputs=[self._joint_wrench],
                    device=model.device,
                )
            if model.joint_count and control.joint_f is not None and self._d6_joints.shape[0]:
                wp.launch(
                    _d6_joint_wrenches,
                    self._d6_joints.shape[0],
                    inputs=[
                        self._d6_joints,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_X_p,
                        model.joint_X_c,
                        model.joint_axis,
                        model.joint_dof_dim,
                        model.joint_qd_start,
                        model.joint_enabled,
                        control.joint_f,
                        state_in.body_q,
                        model.body_com,
                    ],
                    outputs=[self._joint_wrench],
                    device=model.device,
                )
            body_f = state_in.body_f
            wp.launch(
                _body_accelerations,
                self._n_dynamic,
                inputs=[
                    body_f,
                    int(body_f is not None),
                    self._joint_wrench,
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

    def _load_joints(self, state_in: State) -> None:
        """The joints' angles and distances at the step's start, from the bodies' state just loaded. A revolute joint
        unwraps its angle against the one it had, or against ``state_in.joint_q`` where that is not what the solver
        wrote, so that a caller who wrote the angle picks its turn (class docstring, **Joints**). Where the joint
        options need them, the rates of the coordinates from the bodies' twists (``_joint_rates``)."""
        groups, x = self._joint_groups, self._engine.vm.x
        if groups.revolute is not None:
            if state_in.joint_q is not None:
                wp.launch(
                    _adopt_joint_angles,
                    groups.revolute.n,
                    inputs=[
                        self._joint_rows["revolute"],
                        self.model.joint_q_start,
                        state_in.joint_q,
                        self._joint_q_written,
                    ],
                    outputs=[groups.revolute.current],
                    device=self.model.device,
                )
            groups.revolute.compute_current_angles(x)
        if groups.prismatic is not None:
            groups.prismatic.compute_current_distances(x)
        model = self.model
        for kind, (rate, _, _) in self._joint_rate.items():  # the rates at the step's start (_joint_rates)
            wp.launch(
                _joint_rates,
                rate.shape[0],
                inputs=[
                    self._joint_rows[kind],
                    int(kind == "prismatic"),
                    model.joint_parent,
                    model.joint_child,
                    model.joint_X_p,
                    model.joint_X_c,
                    model.joint_axis,
                    model.joint_qd_start,
                    state_in.body_q,
                    state_in.body_qd,
                    model.body_com,
                ],
                outputs=[rate],
                device=model.device,
            )

    def _write_joint_controls(self, control: Control, dt: float) -> None:
        """The joints' terms of the step from the model and ``control``, on the device (``_joint_terms``), and with the
        joint options that add to the driving term their terms (``_joint_option_terms``)."""
        model, groups = self.model, self._joint_groups
        if self._engine is None or model.joint_count == 0:
            return
        springs = all(a is not None for a in (model.joint_stiffness, model.joint_rest_q, model.joint_damping))
        target_q, target_qd, joint_f = control.joint_target_q, control.joint_target_qd, control.joint_f
        bdf2 = self._config.integrator == "bdf2"
        restart = self._engine.bdf1_frame[0:1] if bdf2 else self._no_restart
        # with several worlds a reset or a new state restarts the rows of its worlds only (reset, _load)
        rows = self._engine.restart_rows if bdf2 else None
        restart_rows = self._no_restart if rows is None else rows
        for kind, group, scale in (("revolute", groups.revolute, 1.0), ("prismatic", groups.prismatic, 0.5)):
            if group is None:
                continue
            q_start, q_prev, _ = self._joint_history[kind]
            force_on, force = (group.tq_on, group.tq) if kind == "revolute" else (group.f_on, group.f)
            wp.launch(
                _joint_terms,
                group.n,
                inputs=[
                    self._joint_rows[kind],
                    model.joint_q_start,
                    model.joint_qd_start,
                    model.joint_enabled,
                    model.joint_target_ke,
                    model.joint_target_kd,
                    model.joint_stiffness,
                    model.joint_rest_q,
                    model.joint_damping,
                    int(springs),
                    model.joint_limit_lower,
                    model.joint_limit_upper,
                    model.joint_limit_ke,
                    target_q,
                    int(target_q is not None),
                    target_qd,
                    int(target_qd is not None),
                    joint_f,
                    int(joint_f is not None),
                    self._joint_strength[kind],
                    group.mass_sum,
                    group.current,
                    q_prev,
                    int(bdf2),
                    restart,
                    restart_rows,
                    int(rows is not None),
                    model.joint_parent,
                    model.joint_child,
                    self._body_base,
                    wp.float64(dt),
                    wp.float64(scale),
                ],
                outputs=[
                    q_start,
                    group.ratio,
                    group.drv_on,
                    group.drv_ratio,
                    group.drv_aim,
                    group.lim_on,
                    group.lim_lower,
                    group.lim_upper,
                    group.lim_strength,
                    force_on,
                    force,
                ],
                device=model.device,
            )
            if self._joint_options.drive_terms:
                wp.launch(
                    _joint_option_terms,
                    group.n,
                    inputs=[
                        self._joint_rows[kind],
                        model.joint_q_start,
                        model.joint_qd_start,
                        model.joint_enabled,
                        model.joint_target_ke,
                        model.joint_target_kd,
                        model.joint_stiffness,
                        model.joint_rest_q,
                        model.joint_damping,
                        int(springs),
                        model.joint_limit_lower,
                        model.joint_limit_upper,
                        model.joint_limit_ke,
                        model.joint_limit_kd,
                        int(self._joint_options.limit_damping),
                        model.joint_armature,
                        int(self._joint_options.armature),
                        *(self._joint_rate[kind][:2] if self._joint_options.rates else (q_prev, q_prev)),
                        model.joint_effort_limit,
                        int(self._joint_options.effort_limit),
                        joint_f,
                        int(joint_f is not None),
                        target_q,
                        int(target_q is not None),
                        target_qd,
                        int(target_qd is not None),
                        group.mass_sum,
                        group.current,
                        q_prev,
                        int(bdf2),
                        restart,
                        restart_rows,
                        int(rows is not None),
                        model.joint_parent,
                        model.joint_child,
                        self._body_base,
                        wp.float64(dt),
                        wp.float64(scale),
                    ],
                    outputs=[group.drv_on, group.drv_ratio, group.drv_aim, force_on, force],
                    device=model.device,
                )
        for kind, group in (("fixed", groups.fixed), ("spherical", groups.spherical)):
            if group is not None:
                wp.launch(
                    _enable_anchor_joints,
                    group.n,
                    inputs=[self._joint_rows[kind], model.joint_enabled, self._joint_strength[kind]],
                    outputs=[group.kappa],
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
                inputs=[vm.x, vm.v, self._body_base, model.body_com, self._body_passthrough, body_q_in, body_qd_in],
                outputs=[body_q, body_qd, self._body_q_written, self._body_qd_written],
                device=model.device,
            )
        if model.joint_count and state_out is not None and state_out.joint_q is not None:
            groups = self._joint_groups
            angle = self._no_joint_state if groups.revolute is None else groups.revolute.current
            distance = self._no_joint_state if groups.prismatic is None else groups.prismatic.current
            wp.launch(
                _store_joints,
                model.joint_count,
                inputs=[
                    self._joint_store_type,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_X_p,
                    model.joint_X_c,
                    model.joint_axis,
                    model.joint_q_start,
                    model.joint_qd_start,
                    self._joint_element,
                    angle,
                    distance,
                    state_out.body_q,
                    state_out.body_qd,
                    model.body_com,
                ],
                outputs=[state_out.joint_q, state_out.joint_qd, self._joint_q_written],
                device=model.device,
            )
            if self._d6_joints.shape[0]:
                wp.launch(
                    _store_d6_joints,
                    self._d6_joints.shape[0],
                    inputs=[
                        self._d6_joints,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_X_p,
                        model.joint_X_c,
                        model.joint_axis,
                        model.joint_dof_dim,
                        model.joint_q_start,
                        model.joint_qd_start,
                        state_out.body_q,
                        state_out.body_qd,
                        model.body_com,
                    ],
                    outputs=[state_out.joint_q, state_out.joint_qd, self._joint_q_written],
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
        if not status.overflowed:
            self._engine_outdated()  # the next frame is captured anew when the solves or the size ladders ask
        return status

    def _engine_outdated(self, iterations=None) -> bool:
        """``engine.graph_outdated``: whether a frame captured from now on would hold a better PCG loop or better
        guesses of the size ladders. When it would, the engine takes the ladders' guesses for that capture
        (``prepare_capture``, as libuipc-warp's ``FrameRunner`` does before it captures) and the solver's own graph of
        the solve is dropped. ``iterations``: the sums of the captured steps (:meth:`graph_outdated`)."""
        if not self._engine.graph_outdated(iterations=iterations):
            return False
        self._engine.prepare_capture()
        self._frame_graph = None
        return True

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
        """A new engine whose contact lists hold what the last frame asked for, at the start of that frame, with its
        element groups' state (the joints' angles) and the PCG loop the steps so far chose (libuipc-warp's
        ``redo_state``, as its ``FrameRunner`` hands them over), and with the frame's external accelerations and
        aims."""
        old = self._engine
        capacity = old.grown_capacity()
        redo = old.redo_state()
        # an entry of a coupler that restarts its steps: the start of the coupled step, which the new engine keeps
        restart = None if old.frame_restart is None else old.frame_restart.saved()
        forces = None if self._forces is None else self._forces.a
        aims = None if self._aim_group is None else self._aim_group
        # release the old engine before the new one allocates (the contact export holds it too)
        self._engine = self._frame_graph = self._forces = self._contact_export = old = None
        self._make_engine(capacity)
        self._engine.load_redo_state(redo)
        if restart is not None:
            self._engine.enable_frame_restart().load_saved(restart)
        if forces is not None:
            wp.copy(self._forces.a, forces)
        if aims is not None:  # the aims of the step, which the new engine solves again
            wp.copy(self._aim_group.q_aim, aims.q_aim)
            wp.copy(self._aim_group.active, aims.active)
            wp.copy(self._aim_group.strength, aims.strength)
        self._write_joint_controls(self._control or self._control_of_model(), self._dt)
        self._write_stitch_stiffness(self._dt)
        self._write_body_stiffness(self._dt)

    def _failed_alone(self, status) -> bool:
        """Whether a failed step failed in some independent worlds alone (``status``: the step's ``frame_status``):
        libuipc-warp's frame of worlds puts a world that fails on a value libuipc asserts on (a non-finite value, a CCD
        fraction that is not positive, a pair at or inside its thickness) back at its frame start, and the others take
        their step. An exhausted traversal is every world's, the traversal being shared; an overflow fails only the
        world over its quota of the contact lists, but an eager step grows that quota and solves the step again, so
        one that is left after the last enlargement is the solver's failure, as without independent worlds."""
        return (
            self._independent
            and len(self._status_entries) > 1
            and not (status.overflowed or status.traversal_exhausted)
        )

    def _world_failure(self, failed: dict[int, int]) -> str:
        """Why a step failed in some of the independent worlds (``_failed_worlds``), and what ``state_out`` holds."""
        return (
            f"SolverIPC: the step failed in {len(failed)} of its {len(self._status_entries)} worlds, "
            f"{self._name_worlds(failed)}. Those worlds are back at their state at the step's start, with their "
            "friction discarded and their BDF2 history started anew, and every other world took the step: state_out "
            "holds both. Surfaces must start apart (outside their contact thickness), and the state must stay finite."
        )

    def _name_worlds(self, failed: dict[int, int]) -> str:
        """``failed`` (bits by world entry) as ``world 1 on THICKNESS_VIOLATED and world 3 on ...``, the first eight."""
        names = [
            ("the global world" if k == self.model.world_count else f"world {k}")
            + " on "
            + "|".join(bit.name for bit in self.StatusBits if bit & self.StatusBits.FAILED and bit & failed[k])
            for k in sorted(failed)
        ]
        if len(names) > 8:
            names = [*names[:8], f"{len(names) - 8} more"]
        return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]

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

    def _write_observables(self, observables: SolverBase.Observables | None, state_out: State | None) -> None:
        if observables is None:
            return
        self._write_contacts(observables, state_out)
        status = observables.status if observables.is_requested(self.ObservableKind.STATUS) else None
        residual = (
            observables.solver_residual if observables.is_requested(SolverBase.ObservableKind.SOLVER_RESIDUAL) else None
        )
        if status is None and residual is None:
            return
        if self._independent:  # every world its own, the entries of the global world and of worlds without unknowns 0
            self._write_world_status(status, residual)
            return
        if len(self._world_entries) <= 2:
            for entry in self._world_entries:
                self._uipc_warp.arrays.write_frame_status(self._engine, status, residual, index=int(entry))
            return
        # many worlds (coupled_worlds): the one solve's status written once, then copied to every entry
        self._uipc_warp.arrays.write_frame_status(self._engine, self._status_scratch, self._residual_scratch)
        wp.launch(
            _copy_status,
            len(self._world_entries),
            inputs=[
                self._world_entries_device,
                self._status_scratch,
                self._residual_scratch,
                int(status is not None),
                int(residual is not None),
            ],
            outputs=[
                self._status_scratch if status is None else status,
                self._residual_scratch if residual is None else residual,
            ],
            device=self.model.device,
        )

    def _write_world_status(self, flags, residual) -> None:
        """With independent worlds: every world's status bits and Newton residual into its entry of ``flags`` and
        ``residual`` (libuipc-warp's ``write_frame_status`` with ``worlds``, the record of each world), or the one
        solve's when one world holds unknowns, the engine then being that world's alone; nothing when none does. The
        entries of the global world and of worlds without unknowns are not written: they take no step."""
        entries = self._status_entries
        if len(entries) > 1:
            self._uipc_warp.arrays.write_frame_status(self._engine, flags, residual, worlds=self._status_entries_device)
        elif len(entries) == 1:
            self._uipc_warp.arrays.write_frame_status(self._engine, flags, residual, index=int(entries[0]))

    def _failed_worlds(self) -> dict[int, int]:
        """With independent worlds, after a step that is not captured: the status bits of every world whose step
        failed, by world (reads every world's word back; one synchronization)."""
        self._world_words.zero_()
        self._write_world_status(self._world_words, None)
        words = self._world_words.numpy()
        failed = int(self.StatusBits.FAILED)
        return {int(k): int(words[k]) for k in np.flatnonzero(words & failed)}

    def _write_contacts(self, observables: SolverBase.Observables | None, state_out: State | None) -> None:
        """With ``CONTACT_F`` requested: the solver's contact terms of the step into the step's ``Contacts``, one row
        per term with a force, and their forces into ``contact_f`` (class docstring, **Contact output**), on the
        device. The counts and the forces are zeroed first; a step that failed has no rows."""
        if observables is None or not observables.is_requested(SolverBase.ObservableKind.CONTACT_F):
            return
        contacts, contact_f = observables.contacts, observables.contact_f
        contacts.rigid_contact_count.zero_()
        contacts.soft_contact_count.zero_()
        contact_f.zero_()
        if self._contact_export is None or state_out is None or len(self._contact_shapes) == 0:
            return
        export = self._contact_records()
        model = self.model
        body_q = state_out.body_q if model.body_count and state_out.body_q is not None else self._no_transform
        body_com = model.body_com if model.body_count else self._no_vec3
        wp.launch(
            _contact_rows,
            self._contact_threads,
            inputs=[
                export.vertices,
                export.forces,
                export.counts,
                export.segment_offsets,
                self._contact_threads,
                self._engine.vm.x,
                model.particle_count,
                self._contact_vertex_shape,
                self._contact_plane_shape,
                self._contact_plane_P,
                self._contact_plane_N,
                model.shape_body,
                model.shape_margin,
                body_q,
                body_com,
                contacts.rigid_contact_max,
                contacts.soft_contact_max,
            ],
            outputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
                contacts.rigid_contact_surface_velocity,
                contacts.soft_contact_count,
                contacts.soft_contact_shape,
                contacts.soft_contact_indices,
                contacts.soft_contact_barycentric,
                contacts.soft_contact_particle,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
                contacts.soft_contact_rigid_indices,
                contact_f,
            ],
            block_dim=_CONTACT_ROW_BLOCK,
            device=model.device,
        )

    def _contact_records(self):
        """libuipc-warp's ``ContactForces`` with the records of the last step: its ``update()`` enqueued once per step,
        for the contact rows and the proxies' feedback alike."""
        if self._export_step != self._steps:
            self._contact_export.update()
            self._export_step = self._steps
        return self._contact_export

    def grow(self) -> None:
        """Enlarge the contact lists after a captured step reported :attr:`SolverIPC.StatusBits.OVERFLOW`.

        A captured step cannot enlarge them: it drops the pairs that do not fit, sets the bit and keeps it set.
        This builds the solve again with lists that hold what the last step asks for (solving that step again
        until it fits, as an eager step does) and returns to the state the last step started from, so that the
        same ``state_in`` repeats it. With independent worlds the quotas of the worlds that went over theirs grow,
        and the other worlds keep theirs. The new solve has new arrays: capture every graph that holds
        :meth:`step` again. Synchronizes; does nothing when no list overflowed.
        """
        if self._engine is None:
            return
        if self.model.device.is_capturing:
            raise RuntimeError("SolverIPC.grow() synchronizes with the device; call it outside graph capture.")
        if not self._uipc_warp.arrays.frame_status(self._engine).overflowed:
            return
        if self._config.integrator == "bdf2":  # the joints' history of the last step, which is solved again
            for _, q_prev, q_kept in (*self._joint_history.values(), *self._joint_rate.values()):
                wp.copy(q_prev, q_kept)
        for _ in range(_MAX_REGROWS):
            self._regrow()
            if not self._solve_frame(self._dt).overflowed:
                break
        else:
            raise RuntimeError(f"SolverIPC: the contact lists still overflow after {_MAX_REGROWS} enlargements.")
        self._engine.load_redo_state(self._engine.redo_state())
        if self.model.body_count:
            # what the solver wrote for the restored state, so that the step's own state_in is kept as it was
            self._store(self._body_q_written, self._body_qd_written, None)

    def graph_outdated(self) -> bool:
        """Whether graphs that hold :meth:`step` would run faster captured again.

        The PCG loop of a step re-reads its stopping test after a fixed number of iterations, 8 or 32, and work whose
        size only the device knows (the matrix conversion, the sort of body pairs) runs on a ladder of sizes that
        starts at a guess. Both are chosen when the step is enqueued: a captured step keeps what it was captured with.
        A step that is not captured chooses for itself (libuipc-warp's ``SimEngine.graph_outdated``), from the steps
        before. For captured steps, this takes the Newton and PCG iterations of every captured step replayed since the
        last call, which the steps count on the device, and the sizes the ladders took, and returns ``True`` when
        they ask for the other number or for other guesses; the engine then takes the guesses
        (``SimEngine.prepare_capture``), and the steps captured from then on take both. The graphs captured before
        stay valid and give the same results; only the loop and the ladders cost more. Call it outside graph
        capture, after replays (after :meth:`check_status`, for example); a caller that never calls it keeps 8 and
        no guesses. Reads the two counters and the ladders' counts back: two or three synchronizations.

        Returns:
            ``True`` when graphs that hold :meth:`step` should be captured again.
        """
        if self._engine is None:
            return False
        if self.model.device.is_capturing:
            raise RuntimeError("SolverIPC.graph_outdated() reads from the device; call it outside graph capture.")
        newton, pcg = self._iterations.numpy().tolist()
        self._iterations.zero_()
        return self._engine_outdated(iterations=(newton, pcg))

    def check_status(self, observables: SolverBase.Observables | None = None, *, grow: bool = False) -> bool:
        """Raise when a step that wrote ``observables`` failed, and warn when one reached an iteration limit.

        A step captured in a CUDA graph cannot raise: it only writes how it ended into the observables passed to it
        (see **Status** in the class docstring). Call this after replaying the graph, outside the capture, once for
        each container the replayed steps wrote. It reads the container back from the device (one synchronization).

        A coupled solver steps its entries without observables. The solver as an entry (class docstring,
        **Coupling**) keeps the status bits of every step it takes, captured or not, until a call without a container
        reads them: that call reports every step since the last such call, and clears the bits. A pass of a coupled step
        that the coupler solves again is replaced by the next pass: a coupled step reports its last pass.

        Args:
            observables: A container from :meth:`observables` with :attr:`SolverIPC.ObservableKind.STATUS`
                requested, as passed to :meth:`step`. With ``SOLVER_RESIDUAL`` requested as well, the messages give
                the residual. ``None`` for the solver as an entry of a coupled solver: the bits it kept.
            grow: On :attr:`SolverIPC.StatusBits.OVERFLOW`, call :meth:`grow` and return ``True`` instead of
                raising, whatever other bits are set: the steps ran without some contact pairs, which can cause the
                other failures, and the repeated steps report again.

        Returns:
            ``True`` when a contact list overflowed and ``grow`` enlarged the lists: the steps ran without some
            contact pairs, so restore the state they started from, capture every graph that holds :meth:`step`
            again and repeat them. ``False`` otherwise.

        Raises:
            RuntimeError: When a bit of :attr:`SolverIPC.StatusBits.FAILED` is set and ``grow`` does not take an
                ``OVERFLOW``: the step's result must not be used. With independent worlds the bits are every world's
                own, and the message names the worlds that failed, whose results alone must not be used.
            ValueError: When ``observables`` does not hold the ``STATUS`` observable, or is ``None`` and the solver
                is no entry of a coupled solver.

        Warns:
            NewtonWarning: When :attr:`SolverIPC.StatusBits.NEWTON_LIMIT` or
                :attr:`SolverIPC.StatusBits.LINE_SEARCH_LIMIT` is set. Neither fails the step.
        """
        if self.model.device.is_capturing:
            raise RuntimeError("SolverIPC.check_status() reads from the device; call it outside graph capture.")
        residual = ""
        if observables is None and self._sticky_status is not None:  # an entry of a coupled solver: its kept bits
            words = self._sticky_status.numpy()
            self._sticky_status.zero_()
        elif observables is None or not observables.is_requested(self.ObservableKind.STATUS):
            raise ValueError(
                "SolverIPC.check_status() needs observables with ObservableKind.STATUS requested, or none for an "
                "entry of a coupled solver."
            )
        else:
            words = observables.status.numpy()
            if observables.is_requested(SolverBase.ObservableKind.SOLVER_RESIDUAL):
                residual = (
                    f" (residual {float(observables.solver_residual.numpy().max()):.3g}, at most 1 when converged)"
                )
        bits = self.StatusBits(int(np.bitwise_or.reduce(words)))
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
            if self._independent:  # every world its own bits: the worlds that failed, the others took their step
                worlds = {int(k): int(words[k]) for k in np.flatnonzero(words & int(self.StatusBits.FAILED))}
                which = self._name_worlds(worlds)
                if len(worlds) < len(self._status_entries):
                    subject = f"in {which}{residual}; the result of those worlds must not be used, the other worlds"
                    subject += " took their step"
                else:
                    subject = f"in every world ({which}){residual}; its result must not be used"
                raise RuntimeError(f"SolverIPC: a step failed {subject}. " + " ".join(causes))
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
        solver carries the friction of the last step's contacts from one step to the next, and with
        ``integrator="bdf2"`` the integrator's history; this discards them, so that the next step is an implicit
        Euler step without friction. The contacts themselves are reused only where nothing moved, where they are
        the ones a new solver finds. Launched on the device without synchronization, so it can be captured.

        Args:
            state: Unused; the next step reads its own ``state_in``.
            world_mask: Optional boolean mask of shape ``(world_count + 1,)``, the last entry the global entities.
                With several worlds the reset applies to the worlds it selects: their friction is discarded and their
                BDF2 history starts anew, and every other world keeps its own, bit for bit. With independent worlds
                (the default) the last entry selects nothing, since the global entities are no unknowns, as Newton
                lets a solver without global dynamic objects take it; with ``coupled_worlds`` it discards the friction
                of every pair with a global entity, a global ground plane among them (see **Worlds**). With one world
                the reset applies when the mask selects a world that owns particles or bodies. ``None`` resets.
            flags: Unused.
        """
        del state, flags
        world_mask = _normalize_reset_world_mask(
            world_mask, world_count=int(self.model.world_count), device=self.model.device
        )
        self._restore_pending = False  # the next step starts from the reset, not from a coupled step's start
        if self._engine is None:
            return
        if world_mask is None:
            self._engine.discard_friction()
            self._engine.restart_history()
            if self.model.body_count:
                self._body_hold.zero_()
            return
        if len(self._aim_bodies):
            wp.launch(
                _release_held,
                len(self._aim_bodies),
                inputs=[world_mask, self._body_entry, self._aim_rows],
                outputs=[self._body_hold],
                device=self.model.device,
            )
        if self._worlds is None:
            wp.launch(
                _any_selected,
                1,
                inputs=[world_mask, self._world_entries_device],
                outputs=[self._selected],
                device=self.model.device,
            )
        else:  # the subscenes of the selected worlds (libuipc-warp's requests per subscene)
            wp.launch(
                _subscenes_selected,
                self._worlds.count,
                inputs=[world_mask, self._subscene_entry, int(self._independent)],
                outputs=[self._selected],
                device=self.model.device,
            )
        self._engine.discard_friction(where=self._selected)
        self._engine.restart_history(where=self._selected)

    @override
    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        """Take changed model properties.

        Gravity, of every world (``MODEL_PROPERTIES``), and the bodies' mass, centre of mass and inertia
        (``BODY_INERTIAL_PROPERTIES``), with the joints' penalties, which follow the masses, are updated in place:
        graphs captured before take them. A change of the triangulated shapes (``SHAPE_PROPERTIES`` with new
        transforms or scales, ``SHAPE_TYPE_AND_SOURCE``), of the friction coefficients and the contact thicknesses
        (``SHAPE_PROPERTIES`` with a new ``shape_material_mu`` or ``shape_margin``, ``MODEL_PROPERTIES`` with a new
        ``soft_contact_mu``, ``particle_mu`` or ``particle_radius``), of the particles' masses (``MODEL_PROPERTIES``
        with a new ``particle_mass``, as a coupled solver gives its proxy particles their masses), of the springs
        (``MODEL_PROPERTIES`` with a new ``spring_rest_length``, ``spring_stiffness`` or ``spring_damping``), of the
        joints' types, bodies, frames or axes (``JOINT_PROPERTIES``) or of which bodies are kinematic
        (``BODY_PROPERTIES``, ``BODY_INERTIAL_PROPERTIES``) builds the solve again: the next step reads ``state_in`` in
        full, and graphs that
        hold :meth:`step` must be captured again. The joints' other data are read at every step; the joint flags only
        check them again against what **Joints** in the class docstring rejects. So are the stiffness and the enabled
        flags of the particle-body attachments (``CONSTRAINT_PROPERTIES``), which that flag checks again against what
        **Attachments** rejects; a new body, particle or point of an attachment builds the solve again. With
        ``kinematic_motion="aim"`` a change of a kinematic body's mass properties builds the solve again too, since
        its soft transform constraint follows the masses it was built with. Copies from the host: not inside a graph
        capture. With ``contact_constitution="al-ipc"`` new masses build the solve again too, since the AL-IPC
        penalty follows the masses it was built with.
        """
        if self.model.device.is_capturing:
            raise RuntimeError("SolverIPC.notify_model_changed() copies from the host; call it outside graph capture.")
        if self._engine is None:  # nothing moves
            return
        model = self.model
        rebuild = bool(flags & ModelFlags.SHAPE_TYPE_AND_SOURCE)
        if flags & (ModelFlags.SHAPE_PROPERTIES | ModelFlags.MODEL_PROPERTIES):
            now = self._scene_inputs_now()
            rebuild |= len(now) != len(self._scene_inputs) or any(
                not np.array_equal(a, b) for a, b in zip(now, self._scene_inputs, strict=True)
            )
        if flags & ModelFlags.JOINT_PROPERTIES:
            now = self._joint_inputs_now()
            rebuild |= len(now) != len(self._joint_inputs) or any(
                not np.array_equal(a, b) for a, b in zip(now, self._joint_inputs, strict=True)
            )
        dof_flags = (
            ModelFlags.JOINT_DOF_PROPERTIES
            | ModelFlags.JOINT_DOF_FORCE_PROPERTIES
            | ModelFlags.JOINT_DOF_INERTIAL_PROPERTIES
            | ModelFlags.JOINT_REFERENCE_POSE_PROPERTIES
        )
        if flags & dof_flags:
            _check_joints(model, self._joint_options)  # the other joint data are read at every step
            if self._joint_options.distance and model.joint_count:  # a distance joint gains or loses its length
                rebuild |= not np.array_equal(_map_joints(model, self._joint_options).distance, self._distance_joints)
        if flags & ModelFlags.CONSTRAINT_PROPERTIES and model.attachment_particle_body_count:
            _check_attachments(model)  # their stiffness and enabled flags are read at every step
            body, particle, point = self._attachment_inputs or (None, None, None)
            rebuild |= (
                body is None
                or not np.array_equal(model.attachment_particle_body_body.numpy(), body)
                or not np.array_equal(model.attachment_particle_body_particle.numpy(), particle)
                or not np.array_equal(model.attachment_particle_body_body_point.numpy(), point)
            )
        kinematic = self._kinematic_body
        if flags & (ModelFlags.BODY_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES) and model.body_count:
            dyadic = self._body_mass_moments(stacklevel=3)
            rebuild |= not np.array_equal(self._kinematic_body, kinematic)
            if not rebuild:
                dyadic = self._with_aim_moments(dyadic)
            changed = any(not np.array_equal(a, b) for a, b in zip(dyadic, self._dyadic, strict=True))
            if self._config.contact_constitution == "al-ipc":
                # the AL-IPC contact penalty follows the masses the engine was built with
                rebuild |= changed
            if changed and len(self._aim_bodies):
                # kinematic_motion="aim": the soft transform constraint of a kinematic body follows the masses the
                # engine was built with
                aims = self._aim_bodies
                rebuild |= any(not np.array_equal(a[aims], b[aims]) for a, b in zip(dyadic, self._dyadic, strict=True))
        if rebuild:
            _check_supported(model, self._joint_options)
            self._build_engine()
            return
        if flags & ModelFlags.MODEL_PROPERTIES:
            self._config.gravity = tuple(self._world_gravity()[-1])
            self._engine.set_gravity(self._vertex_gravity())
        if flags & ModelFlags.BODY_INERTIAL_PROPERTIES and model.body_count:
            self._dyadic = dyadic
            self._engine.set_dyadic_mass(*self._engine_dyadic(dyadic))
            self._body_mass_inv.assign(self._mass_inverse(*dyadic))
            for kind in self._joint_bodies:  # the joints' penalties follow the masses
                self._joint_strength[kind].assign(self._joint_strengths(kind, dyadic[0])[1])
            self._attachment_hard.assign(self._attachment_penalties(dyadic[0]))  # and so do the hard attachments'
            self._distance_hard.assign(self._distance_penalties(dyadic[0]))  # and the distance joints'
            if self._ortho is not None:  # and the default body stiffness the inertia
                self._ortho_inertia.assign(self._ortho_ratios())

    # ------------------------------------------------------------------ CouplingInterface
    @override
    def coupling_notify_iteration_restarts(self, enabled: bool) -> None:
        """Become an entry of a coupled solver (class docstring, **Coupling**): keep the status bits of every step for
        :meth:`check_status`, and when the coupler may solve a step again from its start, keep that start on the
        device. Outside graph capture: the coupler calls this once, after construction.

        Args:
            enabled: Whether the coupler may restart a step's solve: a ``SolverCoupledProxy`` with more than one
                iteration per step, or a ``SolverCoupledADMM`` with more than one. The solver then allocates the
                copy of a step's start (libuipc-warp's frame restart, and what the solver carries from step to step).
        """
        if self._sticky_status is None:
            self._allocate_kept_status()
        self._restarts = bool(enabled)
        if self._restarts:
            self._allocate_restart()

    @override
    def coupling_notify_input_state_update(
        self,
        state: State,
        flags: StateFlags | int,
        *,
        iteration_restart: bool = False,
        dt: float = 0.0,
    ) -> None:
        """Take note of a coupled solver's write of the entry's input state (class docstring, **Coupling**).

        Every step compares ``state_in`` with what the solver wrote last and takes what differs (class docstring,
        **State**), a proxy's pose and twist without discarding friction, so a write needs nothing of its own. When the
        coupler restarts steps (:meth:`coupling_notify_iteration_restarts`), the next step after a write (``dt`` above
        0) keeps its start on the device before it takes ``state_in``, and the next step after a restart's notification
        (``iteration_restart`` with ``dt`` above 0) starts from that start again instead: device copies, also inside a
        graph capture. A reset's notification (``dt`` 0) needs nothing: the coupled solver's ``reset`` calls
        :meth:`reset` next, with its world mask, which discards the friction and releases the kinematic aims of the
        worlds it selects.

        Raises:
            NotImplementedError: For a restart's notification from a coupler that did not announce restarts through
                :meth:`coupling_notify_iteration_restarts`: the solver keeps no start to restart from.
        """
        del state, flags
        if dt <= 0.0:
            return
        if not self._restarts:
            if iteration_restart:
                raise NotImplementedError(
                    "SolverIPC cannot solve a step again from its start: the coupler did not announce restarts "
                    "(coupling_notify_iteration_restarts(True)), so the solver keeps no start to restart from."
                )
            return
        if iteration_restart:
            self._restore_pending = True
        else:
            self._save_pending = True

    @override
    def coupling_supports_full_surface_soft_contacts(self) -> bool:
        """``True``: the solver reads no :class:`~newton.Contacts`, of whatever kind, and finds its own contacts."""
        return True

    @override
    def coupling_prepare_proxy_contacts(
        self,
        state: State,
        contacts: Contacts | None,
        *,
        contacts_freshly_detected: bool = False,
    ) -> Contacts | None:
        """Return ``contacts`` as they are: the solver reads no :class:`~newton.Contacts`, and the contacts of a proxy
        with another proxy or with a static shape do not exist in its own (class docstring, **Coupling**)."""
        del state, contacts_freshly_detected
        return contacts

    @override
    def coupling_harvest_proxy_wrenches(
        self,
        body_local_to_proxy_global: wp.array[int],
        out_body_f: wp.array[wp.spatial_vector],
        *,
        body_qd_before: wp.array[wp.spatial_vector],
        state: State,
        state_out: State,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """The proxy bodies' feedback of the step, from its contact forces (class docstring, **Coupling**).

        Zeroes ``out_body_f`` and adds, at every proxy body's index, the force at the end of the step of every contact
        term on the body's vertices, with its torque about the body's centre of mass in ``state_out``: the force this
        entry's content exerts on the proxy, as SolverVBD harvests it, rather than its change of momentum. Device work
        only; call it after :meth:`step`.
        """
        del body_qd_before, state, contacts, dt
        out_body_f.zero_()
        if self._engine is None or not self._proxy_body.any() or body_local_to_proxy_global.shape[0] == 0:
            return
        export = self._contact_records()
        wp.launch(
            _proxy_wrenches,
            self._contact_threads,
            inputs=[
                export.vertices,
                export.forces,
                export.counts,
                export.segment_offsets,
                self._contact_threads,
                self._engine.vm.x,
                self._vertex_body,
                body_local_to_proxy_global,
                state_out.body_q,
                self.model.body_com,
            ],
            outputs=[out_body_f],
            block_dim=_CONTACT_ROW_BLOCK,
            device=self.model.device,
        )

    @override
    def coupling_harvest_proxy_particle_forces(
        self,
        particle_local_to_proxy_global: wp.array[int],
        out_particle_f: wp.array[wp.vec3],
        *,
        particle_qd_before: wp.array[wp.vec3],
        state: State,
        state_out: State,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """The proxy particles' feedback of the step, from its contact forces (class docstring, **Coupling**): zeroes
        ``out_particle_f`` and adds, at every proxy particle's index, the force at the end of the step of every contact
        term on the particle. Device work only; call it after :meth:`step`."""
        del particle_qd_before, state, state_out, contacts, dt
        out_particle_f.zero_()
        if self._engine is None or not self._proxy_particle.any() or particle_local_to_proxy_global.shape[0] == 0:
            return
        export = self._contact_records()
        wp.launch(
            _proxy_particle_forces,
            self._contact_threads,
            inputs=[
                export.vertices,
                export.forces,
                export.counts,
                export.segment_offsets,
                self._contact_threads,
                self.model.particle_count,
                particle_local_to_proxy_global,
            ],
            outputs=[out_particle_f],
            block_dim=_CONTACT_ROW_BLOCK,
            device=self.model.device,
        )

    def _kept_words(self) -> int:
        """The status words an entry of a coupled solver keeps: one per world entry with independent worlds, else
        one, the solve's."""
        return self.model.world_count + 1 if self._independent else 1

    def _allocate_kept_status(self) -> None:
        """The status words an entry of a coupled solver keeps for :meth:`check_status` (``_kept_words``), zeroed,
        and the scratch words of one step. Outside graph capture."""
        device = self.model.device
        self._sticky_status = wp.zeros(self._kept_words(), dtype=wp.int32, device=device)
        self._step_status = wp.zeros(self._kept_words(), dtype=wp.int32, device=device)

    def _keep_status(self) -> None:
        """An entry of a coupled solver: the step's status bits joined into those kept for :meth:`check_status`, with
        independent worlds every world's into its entry."""
        if self._sticky_status is None:
            return
        if self._independent:
            self._step_status.zero_()
            self._write_world_status(self._step_status, None)
        else:
            self._uipc_warp.arrays.write_frame_status(self._engine, self._step_status)
        wp.launch(
            _keep_status,
            self._step_status.shape[0],
            inputs=[self._step_status],
            outputs=[self._sticky_status],
            device=self.model.device,
        )

    def _allocate_restart(self) -> None:
        """An entry of a coupler that restarts its steps: the engine keeps its frame start on the device
        (libuipc-warp's ``enable_frame_restart``), and the solver a copy of what it carries from one step to the next:
        what it wrote of the bodies and the joints, the kinematic bodies it holds toward their aims, the joints'
        history under BDF2, and the status bits it keeps for :meth:`check_status`, so that a restarted step replaces
        the bits of the pass before it. Outside graph capture."""
        self._carried = []
        if self._engine is None:
            return
        self._engine.enable_frame_restart()
        carried = [] if self._sticky_status is None else [self._sticky_status]
        if self.model.body_count:
            carried += [self._body_q_written, self._body_qd_written, self._body_hold]
        if self.model.joint_count:
            carried.append(self._joint_q_written)
        for _, q_prev, q_kept in (*self._joint_history.values(), *self._joint_rate.values()):
            carried += [q_prev, q_kept]
        self._carried = [(a, wp.clone(a)) for a in carried if a.size]

    def _save_step(self) -> None:
        """Keep the start of a coupled step: the engine's frame start and what the solver carries (device copies)."""
        self._engine.save_frame_start()
        for live, kept in self._carried:
            wp.copy(kept, live)

    def _restore_step(self) -> None:
        """Start the step again where the last ``_save_step`` kept it (device copies)."""
        self._engine.restore_frame_start()
        for live, kept in self._carried:
            wp.copy(live, kept)

    def contact_stiffness(self, dt: float) -> float | np.ndarray:
        """The contact stiffness [Pa] of every contact pair in a step of length ``dt`` [s]: ``kappa``, or with
        ``adaptive_kappa`` ``kappa`` clamped into libuipc's stiffness range for this model and ``dt``, the range's lower
        bound when ``kappa`` is None (see **Contact** in the class docstring). 0 for a model without particles and
        bodies, which has no contact.

        With ``adaptive_kappa`` and independent worlds every world has a range of its own, that of the world alone
        (see **Worlds**), and the stiffness is one value per world, a float64 array of ``model.world_count``: world
        ``k``'s pairs take entry ``k``, and a world in which nothing moves, which has no contact, 0."""
        if self._engine is None:
            return 0.0
        policies = self._engine.world_contact_stiffness
        if policies is not None:  # every world's own range (libuipc-warp's independent worlds)
            out = np.zeros(self.model.world_count, np.float64)
            out[self._status_entries] = [policy.resolve(dt)[1] for policy in policies]
            return out
        policy = self._engine.contact_stiffness
        return float(self._config.kappa if policy is None else policy.resolve(dt)[1])

    def frame_stats(self) -> dict:
        """Solver counters of the last step: Newton/PCG iterations, line-search trials, convergence.

        With independent worlds the counts are those of the step's loops, which run as long as a world is in them: the
        Newton iterations of the slowest world. Synchronizes with the device.
        """
        return {} if self._engine is None else self._engine.frame_stats()
