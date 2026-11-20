# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Incremental Potential Contact (IPC) solver backed by libuipc-warp."""

from __future__ import annotations

import gc
import importlib.metadata
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
from newton.math import velocity_at_point
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
    hold: wp.array[wp.int32],
    x: wp.array[wp.vec3d],
    v: wp.array[wp.vec3d],
    changed: wp.array[wp.int32],
):
    """Keep the affine state of a body whose ``state_in`` pose and twist are what the solver wrote last, or that
    ``hold`` holds (a kinematic body moving toward its aim, ``kinematic_motion="aim"``); give another the rigid state
    of ``state_in``: q = (p, rows of R) and its rate from Newton's COM twist.

    ``changed[0]`` is set where a dynamic body took ``state_in``.
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
    ``bdf2`` BDF2 from the coordinate ``q_prev`` at the start of the step before, unless the engine's frame starts its
    history (``restart[0]``) with an implicit Euler step; the limit as libuipc's cubic penalty with Newton's limit
    stiffness at ``_LIMIT_OVERSHOOT``; ``joint_f`` as libuipc's external torque or force. ``q_start`` takes the
    coordinate at the step's start."""
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
    if bdf2 != 0 and restart[0] == 0:
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


def _check_supported(model: Model, coupled_worlds: bool = False) -> None:
    """Reject model content the solver would otherwise accept and ignore; content in more than one world unless
    ``coupled_worlds``."""
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
    for name, activations in (("triangle", model.tri_activations), ("tetrahedron", model.tet_activations)):
        if activations is not None and len(activations) and np.any(activations.numpy() != 0.0):
            raise NotImplementedError(f"SolverIPC does not take {name} activations; the model has non-zero ones.")
    _check_joints(model)
    _check_springs(model)
    _check_attachments(model)
    if not coupled_worlds:
        _check_one_world(model)
    _check_surface_velocity(model)


def _check_springs(model: Model) -> None:
    """Springs that libuipc represents: no damping (libuipc has no damping element), a rest length that is not
    negative (above 0 libuipc's Hookean spring, 0 its soft vertex stitch) and a stiffness that is not negative."""
    if model.spring_count == 0:
        return
    for name, array, bad, why in (
        ("damping", model.spring_damping, lambda v: v != 0.0, "libuipc's springs and stitches have no damping"),
        ("rest length", model.spring_rest_length, lambda v: ~(v >= 0.0), "it must not be negative"),
        ("stiffness", model.spring_stiffness, lambda v: ~(v >= 0.0), "it must not be negative"),
    ):
        values = array.numpy()
        found = np.flatnonzero(bad(values))
        if len(found):
            k = int(found[0])
            raise NotImplementedError(
                f"SolverIPC does not simulate spring {k}, whose {name} is {float(values[k])!r}: {why}."
            )
    ends = model.spring_indices.numpy().reshape(-1, 2)
    same = np.flatnonzero(ends[:, 0] == ends[:, 1])
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


def _check_joints(model: Model) -> None:
    """Joints that libuipc's affine-body joints represent (class docstring, **Joints**): revolute, prismatic, fixed,
    ball and free joints; drives, passive springs, damping and limits on revolute and prismatic joints only; no
    armature, no joint friction, and no effort limit on a driven degree of freedom."""
    if model.joint_count == 0:
        return
    joint_type = model.joint_type.numpy()
    supported = [
        int(t) for t in (JointType.FREE, JointType.REVOLUTE, JointType.PRISMATIC, JointType.FIXED, JointType.BALL)
    ]
    other = np.flatnonzero(~np.isin(joint_type, supported))
    if len(other):
        j = int(other[0])
        raise NotImplementedError(
            f"SolverIPC does not simulate {JointType(int(joint_type[j])).name} joints (joint {j}): "
            "libuipc's affine-body joints are revolute, prismatic, fixed and ball joints."
        )
    if model.joint_dof_count == 0:
        return
    qd_start = model.joint_qd_start.numpy()
    dof_joint = np.repeat(np.arange(model.joint_count), np.diff(qd_start))
    dof_type = joint_type[dof_joint]

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

    for name, why in (
        ("joint_armature", "does not simulate joint armature"),
        ("joint_friction", "does not simulate joint friction"),
    ):
        values = per_dof(name)
        refuse(
            np.flatnonzero(values != 0.0),
            lambda d, v=values, n=name: f"{n} {float(v[d])!r}",
            f"{why} (libuipc's joints have none)",
        )
    ke, kd = per_dof("joint_target_ke"), per_dof("joint_target_kd")
    ks, c = per_dof("joint_stiffness"), per_dof("joint_damping")
    mode = model.joint_target_mode.numpy() if model.joint_target_mode is not None else np.zeros(len(ke), np.int32)
    actuated = np.isin(
        mode, [int(m) for m in (JointTargetMode.POSITION, JointTargetMode.VELOCITY, JointTargetMode.POSITION_VELOCITY)]
    )
    lo, hi, limit_k = per_dof("joint_limit_lower"), per_dof("joint_limit_upper"), per_dof("joint_limit_ke")
    limited = ((lo > -MAXVAL) | (hi < MAXVAL)) & (limit_k > 0.0)
    other = np.isin(dof_type, [int(JointType.BALL), int(JointType.FREE)])

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
        "drives, springs and damps revolute and prismatic joints only (libuipc's ball joint has no driving term, a "
        "free body no joint); Control.joint_f acts on every joint",
    )
    refuse(
        np.flatnonzero(other & limited),
        lambda d: f"the limits [{float(lo[d])!r}, {float(hi[d])!r}] with limit_ke {float(limit_k[d])!r}",
        "limits revolute and prismatic joints only (libuipc's ball joint has no limit)",
    )
    effort = per_dof("joint_effort_limit")
    refuse(
        np.flatnonzero(((ke != 0.0) | (kd != 0.0)) & (effort < _EFFORT_LIMIT)),
        lambda d: f"joint_effort_limit {float(effort[d])!r} and a drive",
        "cannot bound the torque of a drive (libuipc's driving term is a quadratic potential); raise the limit to "
        f"{_EFFORT_LIMIT:g}, Newton's default, to drive the joint without a bound",
    )


def _kinematic_bodies(model: Model) -> np.ndarray:
    """The bodies the solver does not let forces move: :attr:`~newton.BodyFlags.KINEMATIC` ones and those of mass 0."""
    if model.body_count == 0:
        return np.zeros(0, bool)
    return ((model.body_flags.numpy() & int(BodyFlags.KINEMATIC)) != 0) | (model.body_mass.numpy() <= 0.0)


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


def _check_one_world(model: Model) -> None:
    """One world, besides global entities, unless ``coupled_worlds``: one solve couples everything it holds through
    one step length, one line search and one convergence test."""
    local = _local_worlds(model)
    if len(local) > 1:
        raise NotImplementedError(
            f"SolverIPC simulates one world: the model has particles, bodies or colliding shapes in worlds "
            f"{local.tolist()}, which would share one Newton solve. With coupled_worlds=True the worlds are solved "
            "together without colliding with each other, and each world's result depends on the others' iterations."
        )


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
    """The subscenes of a model with content in more than one local world (``coupled_worlds``): subscene 0 holds the
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


# libuipc-warp's engine refuses contact tables of more elements (a dense mask of contact x subscene elements)
_MAX_CONTACT_ELEMENTS = 2048


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
    cloth and tetrahedra (``P_surf``). With ``worlds`` (``coupled_worlds``) the subscenes keep the worlds apart, so the
    world leaves the signature, and a shape in a filter pair shares its element with its copies in replicated worlds
    (``_filter_classes``). With ``aims`` (``kinematic_motion="aim"``) a shape's motion joins the signature: on a
    dynamic body, on a kinematic body (``_kinematic_bodies``), or static; the pairs of two shapes that are each
    kinematic or static are disabled, and ``Model.shape_contact_pairs`` less those pairs is the set compared. Raises
    ``NotImplementedError`` when the shape pairs the elements enable on different bodies are not exactly that set (a
    pair array assigned to the model), or when there are more elements, times the subscenes, than the engine takes.
    """
    ns, n_particles = model.shape_count, model.particle_count
    shape_element = np.full(ns, -1, np.int64)
    classes = 0
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
        signature = np.stack(columns, axis=1)[colliding]
        _, first, inverse = np.unique(signature, axis=0, return_index=True, return_inverse=True)
        order = np.argsort(first)  # classes numbered in the order of their first shapes
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        shape_class = rank[inverse.reshape(-1)]
        shape_element[colliding] = 1 + shape_class
        classes = len(first)
        rep = colliding[first[order]]  # one shape of each class
    count = 1 + classes + (3 if n_particles else 0)
    subscenes = 1 if worlds is None else worlds.count
    if count * subscenes > _MAX_CONTACT_ELEMENTS:
        detail = f" times {subscenes} subscenes (the worlds and the global one)" if worlds is not None else ""
        raise NotImplementedError(
            f"SolverIPC needs {count} contact elements for this model (one per class of shapes that collide alike, "
            f"one per shape in a filter pair){detail}, more than libuipc-warp's engine takes "
            f"({_MAX_CONTACT_ELEMENTS})."
        )
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
        enabled[shapes, shapes] = pairs
        mu[shapes, shapes] = 0.5 * (m[:, None] + m[None, :])
        _check_shape_pairs(model, colliding, shape_class, body[colliding], pairs, worlds is not None, immovable)
    particle_element = np.zeros(n_particles, np.int64)
    on_mesh = np.zeros(n_particles, bool)
    if n_particles:
        p_free, p_surf, p_off = 1 + classes, 2 + classes, 3 + classes
        for cells in (model.tri_indices, model.tet_indices):
            if cells is not None and len(cells):
                on_mesh[cells.numpy().reshape(-1)] = True
        active = (model.particle_flags.numpy() & int(ParticleFlags.ACTIVE)) != 0
        particle_element = np.where(on_mesh, p_surf, np.where(active, p_free, p_off))
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
    return _ContactElements(shape_element, particle_element, on_mesh, enabled, mu)


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
) -> None:
    """Raise unless the shape classes and their ``enabled`` pairs give exactly ``Model.shape_contact_pairs``: every
    listed pair enabled, and as many enabled pairs of shapes on different bodies (static shapes on body -1) as listed.
    ``shapes`` are the shapes the solver triangulates, ``shape_class`` and ``shape_body`` theirs. ``by_world``: the
    worlds are subscenes, and only shapes of one world or a global one pair; the count is taken per world, the global
    shapes among themselves and every world's shapes among themselves and with the global ones. ``immovable``: per
    shape, whether it is kinematic or static; the listed pairs of two such shapes are left out of the comparison."""
    cls = np.full(model.shape_count, -1, np.int64)
    cls[shapes] = shape_class
    listed = model.shape_contact_pairs
    listed = np.zeros((0, 2), np.int64) if listed is None else listed.numpy().reshape(-1, 2).astype(np.int64)
    expected = model.shape_contact_pair_count
    if immovable is not None:
        listed = listed[~(immovable[listed[:, 0]] & immovable[listed[:, 1]])]
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
            + (" that are not of two kinematic or static shapes" if immovable is not None else "")
        )
        raise NotImplementedError(
            "SolverIPC collides the shapes that Newton's collision groups, worlds, collide flags and filter pairs "
            f"let collide, which must be Model.shape_contact_pairs: {detail}. A pair array assigned to the model is "
            "not supported."
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


class SolverIPC(SolverBase):
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
      and kinematic bodies (:attr:`~newton.BodyFlags.KINEMATIC` or zero mass).
      By default they follow the poses written to ``state_in`` every step; the
      jump is not swept by collision detection. With
      ``kinematic_motion="aim"`` kinematic bodies move inside the step toward
      the pose that ``state_in`` prescribes for its end, swept like every other
      motion (see **Kinematic bodies** below). A particle that is not active
      and on no cloth or tetrahedron collides with nothing, as in Newton.
    - revolute, prismatic, fixed and ball joints between bodies or to the
      world, with drives, passive springs and damping, limits and
      :attr:`~newton.Control.joint_f` (see **Joints** below); ``joint_q`` and
      ``joint_qd`` are written at every step.
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
    gradients (``model.requires_grad``), content in more than one world unless
    ``coupled_worlds`` (the worlds would share one Newton solve; see
    **Worlds**), a
    :attr:`~newton.Model.shape_contact_pairs` array assigned to the model, more
    contact elements than the engine takes (see **Contact** below),
    collision meshes with a surface velocity (``Mesh.enable_surface_velocity``,
    which the solver's own contacts would ignore), and a CPU device.
    Gaussian shapes and planes attached to bodies raise when the shapes are
    triangulated. The ``contacts`` argument of
    :meth:`step` is not used: the solver detects its own contacts.

    Not read: the penalty contact parameters of shapes and particles
    (``ke``, ``kd``, ``kf``, ``ka``, ``particle_cohesion`` and
    ``particle_adhesion``: ``kappa`` holds for every pair), restitution,
    torsional and rolling friction, ``shape_material_mu_static``,
    ``shape_gap`` (``d_hat`` is the solver's detection distance), the
    :attr:`~newton.Model.particle_radius` of cloth and tetrahedron vertices
    (see **Contact** below), damping coefficients,
    ``spring_control``, drag, lift, ``edge_rest_length`` (see below), the
    activations of a :class:`~newton.Control`, ``joint_target_mode``,
    ``joint_velocity_limit``, ``joint_limit_kd`` and ``joint_act``.

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
    group, world, collide flags and friction coefficient; with
    ``coupled_worlds`` the subscenes keep the worlds apart, see **Worlds**,
    and the world leaves the signature); a shape in a filter pair gets one of
    its own, which with ``coupled_worlds`` its copies in replicated worlds
    share. Free particles (in no triangle and no
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
    Featherstone and XPBD solvers. The engine takes at most 2048 elements.

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
    gives for a step of any ``dt``.

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
    behind. With ``coupled_worlds`` a new state in one world makes the step an
    implicit Euler step in every world. As in libuipc the coefficients assume
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
    world are refused unless ``coupled_worlds=True``. With it every local world
    is a libuipc subscene of its own and the global entities (world -1) one
    more, which meets every world, as Newton's collision rules say: no contact
    pair joins two worlds, also where the worlds overlap in space, and a global
    ground or static shape acts in every world. The worlds remain one solve:
    one step length (the smallest time of impact over all worlds, and a line
    search on the sum of their energies), one Newton convergence test (the
    largest displacement over all worlds), one PCG solve and one set of contact
    lists. A world's result therefore depends on the others through the
    iteration counts and the step lengths, and a failure or an overflow in one
    world fails the step in every world. Copies of one world take the
    iterations of one copy, and each follows the run of that world alone to
    the rounding of the shared sums. A reset of one world, or a new state
    written for one world, discards the friction of the next step in every
    world, since libuipc discards friction for a whole scene; with
    ``adaptive_kappa`` the stiffness range is that of the whole scene. A model
    with content in one local world is solved as without the option. The
    engine takes at most 2048 contact elements times subscenes.

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

    Limits are libuipc's cubic penalty outside ``[joint_limit_lower,
    joint_limit_upper]``, not Newton's linear spring: at an overshoot
    :math:`\\delta` it stores :math:`k_l \\delta^3 / (3 \\delta_r)` with
    :math:`k_l` = ``joint_limit_ke`` and :math:`\\delta_r` = 1e-3 (rad or m), so
    that it pushes back with the spring's :math:`k_l \\delta_r` at
    :math:`\\delta_r` and harder beyond it; a load :math:`F` rests at
    :math:`\\sqrt{F \\delta_r / k_l}` past the limit. ``joint_limit_kd`` is not
    read (libuipc's limit has no damping), a ``joint_limit_ke`` of 0 holds
    nothing, and a bound at :data:`newton.MAXVAL` is no bound.

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

    Rejected (:class:`NotImplementedError`): distance, D6 and rod joints;
    drives, springs, damping and limits on ball and free joints (libuipc's
    ball joint has none); ``joint_armature`` and ``joint_friction`` other
    than 0; a ``joint_effort_limit`` below Newton's default 1e6 on a driven
    degree of freedom, since the quadratic drive cannot bound its torque.

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

    The solve is built at, and the start is checked on (see the note below),
    the model's state: :attr:`~newton.Model.body_q` and ``body_qd``, but a
    body whose pose there is the identity, where
    :meth:`~newton.ModelBuilder.add_link` and
    :meth:`~newton.ModelBuilder.add_body` put a body given no transform,
    starts where :func:`newton.eval_fk` puts it from
    :attr:`~newton.Model.joint_q` and :attr:`~newton.Model.joint_qd`. As for
    Newton's other maximal-coordinate solvers, the caller writes those poses
    into the state before the first step, also after building the solver;
    the model of an articulation built without poses keeps the identity
    until then, and its links may cross each other or the ground there.

    **Status.** Request :attr:`SolverIPC.ObservableKind.STATUS` and
    ``ObservableKind.SOLVER_RESIDUAL`` (not with ``contact_constitution="al-ipc"``)
    with :meth:`observables` and pass the
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
    The PCG loop of a captured step re-reads its stopping test every 8 or 32
    iterations, and work whose size only the device knows starts at a guessed
    size; both are chosen when the step is captured. :meth:`graph_outdated`,
    called after replays, returns ``True`` when the steps replayed since asked
    for the other length or for other guesses, and the caller then captures
    again to run faster (the old graphs stay valid and give the same results).

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
            follows each step's ``dt`` (see **Contact**).
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
            in which no world collides with another (see **Worlds**); without
            it such a model is refused.
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
        to the rounding of float32 positions), in the model's state (with
        forward kinematics for bodies at the identity, see **State**), the solver
        raises :class:`ValueError` at construction and names the closest pair,
        as libuipc's sanity check rejects such a scene at init. The thickness of
        a pair is the sum of its two sides' (see **Contact**). Surfaces that
        never move may touch or cross, and pairs that do not collide may
        overlap; the inside of a body is not checked. A gap inside
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

        status: wp.array[wp.int32] | None = SolverBase.Observables.field(
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

    # libuipc-warp, its external-acceleration element group and the element group of the bodies' orthogonality
    # potential, imported by the first solver
    _uipc_warp: ClassVar = None
    _external_acceleration: ClassVar = None
    _ortho_potential: ClassVar = None

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
    ):
        if not model.device.is_cuda:
            raise NotImplementedError(f"SolverIPC runs on CUDA devices only; the model is on {model.device}.")
        super().__init__(model)
        self._coupled_worlds = bool(coupled_worlds)
        _check_supported(model, self._coupled_worlds)
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
        self._worlds = None
        self._ortho = None  # the engine's orthogonality potentials, whose stiffness follows dt (default body_kappa)
        self._default_control = None  # the Control of a step given none, built once (_control_of_model)
        self._build_engine()
        self._check_start()

    @property
    def supported_observables(self) -> frozenset[str]:
        """``STATUS`` and ``SOLVER_RESIDUAL``; with ``contact_constitution="al-ipc"`` ``STATUS`` only: the AL-IPC
        frame ends on its own criterion and has no Newton residual."""
        config = getattr(self, "_config", None)
        if config is not None and config.contact_constitution == "al-ipc":
            return frozenset({self.ObservableKind.STATUS})
        return self.SUPPORTED_OBSERVABLES

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
            from uipc_warp.backend.affine_body.constitutions import ortho_potential  # noqa: PLC0415
            from uipc_warp.backend.external_force import external_acceleration  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "SolverIPC requires the libuipc-warp package: pip install git+https://github.com/OrangeEarth15/libuipc-warp"
            ) from e
        version = getattr(uipc_warp, "__version__", "0")
        # the newest calls of each kind the solver makes: the frame status, the contact and subscene tabulars and the
        # stiffness policy, the start check, the joints, the iterations of captured steps, the restart of a BDF2
        # history, the stitches (with the anchors and stitch_groups of the same release), the soft transform constraint
        # (with constraint_groups)
        needed = (
            (uipc_warp.arrays, "write_frame_status"),
            (uipc_warp.arrays.ArrayScene, "contact_model"),
            (uipc_warp.arrays.ArrayScene, "subscene_model"),
            (uipc_warp.backend.contact_system, "contact_stiffness"),
            (uipc_warp.validation.SurfaceChecker, "min_gap"),
            (uipc_warp.arrays, "joint_groups"),
            (uipc_warp.arrays, "add_frame_iterations"),
            (uipc_warp.SimEngine, "prepare_capture"),
            (uipc_warp.SimEngine, "restart_history"),
            (uipc_warp.arrays.ArrayScene, "add_stitches"),
            (uipc_warp.arrays.ArrayScene, "add_soft_transform_constraints"),
        )
        missing = [f"{owner.__name__}.{name}" for owner, name in needed if not hasattr(owner, name)]
        if missing:
            raise ImportError(
                f"SolverIPC requires libuipc-warp 0.4 or newer with {', '.join(f'{o.__name__}.{n}' for o, n in needed)};"
                f" the installed one ({version}) lacks {', '.join(missing)}."
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
        cls._external_acceleration = external_acceleration.ExternalAcceleration
        cls._ortho_potential = ortho_potential.OrthoPotential
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
        # the global world's gravity, which every part overrides with the gravity of its own world
        self._config.gravity = tuple(model.gravity.numpy()[-1].astype(np.float64))
        if not self._d_hat_given:
            self._config.d_hat = _default_d_hat(model)

        scene = self._uipc_warp.arrays.ArrayScene(kappa_corridor=self._adaptive_kappa)
        # several local worlds (coupled_worlds): one subscene each, all of them meeting the global entities' subscene 0
        self._worlds = _Worlds.of(model) if self._coupled_worlds else None
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
        self._dyadic = self._add_shapes_and_bodies(scene, elements)
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
        self._add_stitches(scene)
        if len(self._aim_bodies):  # libuipc's soft transform constraint on the kinematic bodies, in their order
            ratio = self._kinematic_strength_ratio
            scene.add_soft_transform_constraints(self._aim_bodies, (ratio, ratio))
        self._scene = scene
        self._scene_inputs = self._scene_inputs_now()
        self._joint_inputs = self._joint_inputs_now()
        self._control = None  # of the last step, whose joint terms a redone frame takes again

        layout = scene.layout()
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
        self._status_scratch = wp.zeros(1, dtype=wp.int32, device=device)
        self._residual_scratch = wp.zeros(1, dtype=wp.float32, device=device)
        self._no_restart = wp.zeros(1, dtype=wp.int32, device=device)  # BDF1: no history to restart
        self._no_float32 = wp.zeros(1, dtype=wp.float32, device=device)  # a model array the model does not have
        self._no_bool = wp.zeros(1, dtype=wp.bool, device=device)
        nb = model.body_count
        if nb:
            self._body_base = wp.array(layout.body_base.astype(np.int32), dtype=wp.int32, device=device)
            self._body_kinematic = wp.array(self._kinematic_body.astype(np.int32), dtype=wp.int32, device=device)
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
        self._make_engine(None)

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

    def _subscene_of_shape(self, s: int) -> int:
        """The subscene of shape ``s``'s vertices: its world's (``coupled_worlds``), else the one subscene, 0."""
        return 0 if self._worlds is None else int(self._worlds.subscene(self.model.shape_world.numpy()[s]))

    def _subscene_of_body(self, b: int) -> int:
        """The subscene of body ``b``, for the anchors of a body without shapes: its world's (``coupled_worlds``),
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

    def _check_start(self) -> None:
        """Raise ``ValueError`` when the surfaces of different bodies cross each other in the model's state, or start
        at or inside their contact thickness.

        The barrier needs surfaces that start apart, by more than their contact thickness. A step fails on a contact
        pair at or inside its thickness (``THICKNESS_VIOLATED`` or ``INVALID_CCD_STEP``), which a captured step only
        reports, and surfaces that cross without such a pair (an edge through a triangle, away from its vertices) step
        on with the barrier keeping them crossed. As libuipc's sanity check does at init, this tests the model's state
        once, without the surfaces that never move (static shapes, kinematic particles and bodies), whose contact the
        solve skips, and without the pairs whose contact elements do not collide. The gap of a pair is its distance
        less its contact thickness, the sum of its two primitives' thicknesses, each that of its vertices (a free
        particle's radius, a shape's margin, the ``thickness`` of cloth and tetrahedra). Every contact point that moves
        is tested against every half plane. Triangle meshes and particles are tested part pair by part pair, a part
        being a contact body's primitives in one contact element, with the particles on no cloth or tetrahedron taken
        together per element and also tested among themselves: for the pairs of parts whose bounding boxes come within
        their thickness of each other, libuipc-warp's brute-force ``SurfaceChecker`` counts the edge-triangle crossings
        and finds the smallest gap over the point-triangle, point-edge, edge-edge and point-point pairs. A gap above
        0, also inside ``d_hat``, is a valid start, unless it is within a few float32 steps of 0: Newton's positions
        are float32, and surfaces placed in touching contact come out that far apart. Not tested: the inside of a body
        (contact ignores crossings inside a rigid body made of overlapping shapes, and inside a cloth or a soft body the
        checker cannot tell the nearly coplanar triangles of a flat sheet from crossing ones). A pair of parts of more
        than ``_START_CHECK_PAIRS`` primitive pairs is skipped with a warning.
        """
        eng = self._engine
        if eng is None:
            return
        layout = self._scene.layout()
        still = np.zeros(layout.n, bool)  # vertices that the solve never moves
        still[: layout.n_fem] = self._fixed_particle
        still[layout.obstacle_offsets[0] : layout.obstacle_offsets[-1]] = True
        offsets = layout.body_vertex_offsets
        held = self._kinematic_body.copy()
        held[self._aim_bodies] = False  # kinematic_motion="aim": these move in the step
        for b in np.flatnonzero(held):
            still[offsets[b] : offsets[b + 1]] = True
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
        # parts: the groups split by contact element, compared only where their elements collide
        cid, enabled = self._cids, self._enabled
        key = group * len(enabled) + cid
        ids, point_group = np.unique(key[points], return_inverse=True)
        ids_group, ids_element = ids // len(enabled), ids % len(enabled)
        # Newton's positions are float32: surfaces placed in touching contact come out up to a few float32 steps
        # apart, which the barrier would take as a valid, nearly singular start. Such a gap counts as contact.
        slack = 4.0 * float(np.spacing(np.float32(np.abs(x[points]).max())))
        # (crossings, gap, distance, pair thickness, part, the half plane (as -1 - plane) or part, the pair's vertices)
        violations = []

        # contact points against half planes, each moved out by its shape's margin: the distance from the plane's own
        # surface and the margin's share of the pair's thickness
        P, N, plane_shapes = self._planes
        moves = ~still[points]
        if moves.any() and len(P):
            moving = points[moves]
            margin = self.model.shape_margin.numpy().astype(np.float64)[plane_shapes]
            d = x[moving] @ N.T - np.einsum("ij,ij->i", P, N) + margin
            t = thickness[moving][:, None] + margin
            gap = d - t
            meets = enabled[cid[moving][:, None], self._plane_cids[None, :]]
            owner = point_group[moves]
            for j in range(len(P)):
                hit = np.flatnonzero(meets[:, j] & (gap[:, j] <= slack))
                for g in np.unique(owner[hit]):
                    k = hit[owner[hit] == g]
                    k = k[np.argmin(gap[k, j])]
                    violations.append((0, gap[k, j], d[k, j], t[k, j], g, -1 - j, (int(moving[k]),)))

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
        pairs = self._near_boxes(lo, hi, reach, slack)
        # the particles on no surface meet each other where their element meets itself
        lone_parts = np.flatnonzero((ids_group == lone_group) & enabled[ids_element, ids_element])
        pairs = np.concatenate([pairs, np.stack([lone_parts, lone_parts], axis=1)])
        # not tested: two parts that never move, the inside of a body, two parts whose elements do not collide (such
        # as worlds that overlap in space)
        a, b = pairs[:, 0], pairs[:, 1]
        pairs = pairs[
            ~(all_still[a] & all_still[b])
            & ~((a != b) & (ids_group[a] == ids_group[b]))
            & enabled[ids_element[a], ids_element[b]]
        ]
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
                violations.append((found, gap.gap, gap.distance, gap.thickness, a, b, gap.vertices))
        if not violations:
            return
        found, gap, distance, t, a, b, vertices = min(violations, key=lambda v: (-v[0], v[1]))
        n = len(violations) - 1
        more = f" ({n} more pair{'s' if n > 1 else ''} of bodies {'do' if n > 1 else 'does'} too)" if n else ""
        lone_set = set(lone.tolist())

        def describe(part: int, other_than: int | None = None) -> tuple[str, int | None]:
            """A part's name, and its particle in the closest pair when it is the particles on no surface."""
            if ids_group[part] == lone_group:
                v = next((v for v in vertices if v in lone_set and v != other_than), None)
                if v is not None:
                    return f"particle {v}", v
            return self._describe_group(group, points, ids_group[part]), None

        one, particle = describe(a)
        other = f"plane shape {plane_shapes[-1 - b]}" if b < 0 else describe(b, particle if a == b else None)[0]
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
            self._forces = self._external_acceleration("newton_forces", self._force_blocks, device)
            inputs["elements"].append(self._forces)
        self._engine = self._uipc_warp.make_engine(
            self._config, self._options, **inputs, device=device, capacity=capacity
        )
        # the Newton and PCG iterations of the captured steps since the last graph_outdated()
        self._iterations = wp.zeros(2, dtype=wp.int64, device=device)
        # what the start check needs of the tabular: the element of every vertex and half plane, and their pairs
        self._vertex_thickness = np.asarray(inputs["thickness"], np.float64)
        tabular = inputs["contact_tabular"]
        if tabular is None:
            self._cids, self._plane_cids = (
                np.zeros(len(inputs["x"]), np.int64),
                np.zeros(len(inputs["plane_P"]), np.int64),
            )
            self._enabled = np.ones((1, 1), bool)
        else:
            # the engine's ids, contact element * S + subscene, and the product of the two masks: with one subscene
            # the contact elements and their mask
            n_sub = len(tabular["subscene_enabled"])
            self._cids = np.asarray(tabular["cids"], np.int64) * n_sub + np.asarray(tabular["scids"], np.int64)
            self._plane_cids = np.asarray(tabular["plane_cids"], np.int64) * n_sub + np.asarray(
                tabular["plane_scids"], np.int64
            )
            self._enabled = np.kron(np.asarray(tabular["enabled"]) != 0, np.asarray(tabular["subscene_enabled"]) != 0)
        if self.model.body_count and self._config.contact_constitution != "al-ipc":
            # notify_model_changed may have changed them since the scene was laid out (AL-IPC: it builds it again)
            self._engine.set_dyadic_mass(*self._engine_dyadic(self._dyadic))
        self._joint_groups = self._uipc_warp.arrays.joint_groups(self._engine)
        self._aim_group = self._uipc_warp.arrays.constraint_groups(self._engine).transform  # None without aims
        # the stiffness of every stitch group and the attachment or spring of each of its rows (_stitch_stiffness)
        self._stitch_rows = [
            (g.kappa, wp.array(self._stitch_source[g.rows], dtype=wp.int32, device=device))
            for g in self._uipc_warp.arrays.stitch_groups(self._engine)
        ]
        self._find_body_stiffness()

    def _find_body_stiffness(self) -> None:
        """With the default ``body_kappa``: the engine's group of orthogonality potentials, whose stiffness array a
        kernel writes before every step, and the largest principal moment of inertia over the volume of the body
        of each of its rows (0 for a kinematic body and the world's). None without bodies or with a given
        ``body_kappa``, which the engine is built with."""
        self._ortho = self._ortho_inertia = None
        if self._body_kappa is not None or self.model.body_count == 0:
            return
        groups = [e for e in self._engine.elements if isinstance(e, self._ortho_potential)]
        if len(groups) != 1:
            raise RuntimeError(f"SolverIPC: the engine has {len(groups)} groups of orthogonality potentials, not 1.")
        self._ortho = groups[0]
        base = self._scene.layout().body_base
        rows = self._ortho.rows.numpy().astype(np.int64) - 1  # the rows of A follow the body's translation
        order = np.argsort(base)
        body = order[np.searchsorted(base[order], rows)]
        if not np.array_equal(base[body], rows):
            raise RuntimeError("SolverIPC: the orthogonality potentials do not match the engine's bodies.")
        self._ortho_body = body
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
        if model.tet_count == 0:
            return np.zeros((0, 4), np.int64)
        tets = model.tet_indices.numpy().reshape(-1, 4).astype(np.int64)
        Dm_inv = model.tet_poses.numpy().astype(np.float64).reshape(-1, 3, 3)
        volume = 1.0 / (6.0 * np.abs(np.linalg.det(Dm_inv)))
        materials = model.tet_materials.numpy().astype(np.float64).reshape(-1, 3)
        # k_mu and k_lambda are Lame parameters; libuipc's StableNeoHookean.apply_to stores mu and lambda + mu, the
        # SNK1 parameters whose linearization is Hooke's law with those Lame parameters (stable_neo_hookean.cpp)
        k_mu, k_lambda = materials[:, 0], materials[:, 1]
        scene.add_tetrahedra(tets, Dm_inv, volume, k_mu, k_lambda + k_mu)
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
        self._zero_springs = np.flatnonzero(rest == 0.0)
        hookean = rest > 0.0
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
        for its own ``dt``."""
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

    def _scene_inputs_now(self) -> tuple:
        """What the scene was laid out from besides gravity and the bodies' mass properties, which change in place:
        the triangulated shapes, the friction coefficients of the contact models, the contact thicknesses and the
        springs."""
        model = self.model
        materials = (float(model.soft_contact_mu), float(model.particle_mu))
        arrays = [model.particle_radius] if model.particle_count else []
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
        return tuple(a.numpy().copy() for a in (*arrays, model.joint_axis, model.joint_qd_start))

    def _add_shapes_and_bodies(self, scene, elements: _ContactElements) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        mass, first, second = self._body_mass_moments(stacklevel=5)
        # kinematic_motion="aim": the kinematic bodies with colliding shapes, which move toward their aims
        self._aim_bodies = np.zeros(0, np.int64)
        self._aim_mesh_moments = {}
        if model.body_count == 0:
            return mass, first, second
        if self._aims:
            self._aim_bodies = np.flatnonzero(self._kinematic_body & np.array([bool(meshes[b]) for b in meshes]))
            for b in self._aim_bodies:
                if mass[b] <= 0.0:  # the mass of its collision meshes at 1000 kg/m^3
                    self._aim_mesh_moments[int(b)] = self._mesh_moments(int(b), meshes[b])
            mass, first, second = self._with_aim_moments((mass, first, second))
        body_q, body_qd = self._start_bodies()
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

    def _start_bodies(self) -> tuple[np.ndarray, np.ndarray]:
        """The poses and twists the bodies are built at and the start check tests (class docstring, **State**):
        ``model.body_q`` and ``model.body_qd``, but for a body whose pose there is the identity, where
        ``ModelBuilder.add_link`` and ``add_body`` put a body given no transform: it starts where ``newton.eval_fk``
        puts it from ``model.joint_q`` and ``model.joint_qd``, the pose with which a maximal-coordinate solver starts
        once the caller has written forward kinematics into the state."""
        model = self.model
        body_q = model.body_q.numpy().astype(np.float64)
        body_qd = model.body_qd.numpy().astype(np.float64)
        unplaced = np.all(model.body_q.numpy() == np.array([0, 0, 0, 0, 0, 0, 1], np.float32), axis=1)
        if not unplaced.any() or model.articulation_count == 0:
            return body_q, body_qd
        fk = SimpleNamespace(body_q=wp.clone(model.body_q), body_qd=wp.clone(model.body_qd))
        eval_fk(model, model.joint_q, model.joint_qd, fk)
        body_q[unplaced] = fk.body_q.numpy().astype(np.float64)[unplaced]
        body_qd[unplaced] = fk.body_qd.numpy().astype(np.float64)[unplaced]
        return body_q, body_qd

    def _add_joints(self, scene) -> None:
        """Newton's revolute, prismatic, fixed and ball joints as libuipc's (class docstring, **Joints**), joint frames
        taken as the reference rotations, a fixed affine body without mass standing for the world. Sets up what the
        per-step kernels read: per kind the joint of every group row and the row's penalty (``_joint_strength``,
        0 without a dynamic body), per joint its row (``_joint_element``), and the free and ball joints, whose
        ``joint_f`` acts through the bodies' wrenches."""
        model, device = self.model, self.model.device
        self._world_body = -1
        self._joint_rows, self._joint_strength, self._joint_history = {}, {}, {}
        if model.joint_count == 0:
            return
        joint_type = model.joint_type.numpy()
        parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
        X_p, X_c = model.joint_X_p.numpy().astype(np.float64), model.joint_X_c.numpy().astype(np.float64)
        axis = model.joint_axis.numpy().astype(np.float64).reshape(-1, 3)
        qd_start = model.joint_qd_start.numpy()
        wrenched = np.isin(joint_type, [int(JointType.FREE), int(JointType.BALL)])
        self._wrenched_joints = wp.array(np.flatnonzero(wrenched).astype(np.int32), dtype=wp.int32, device=device)
        element = np.full(model.joint_count, -1, np.int32)
        joints = np.flatnonzero(joint_type != int(JointType.FREE))
        if len(joints) and np.any(parent[joints] < 0):
            self._world_body = model.body_count
            scene.add_affine_body(
                np.zeros((0, 3)),
                np.zeros((0, 3), np.int64),
                mass=0.0,
                first_moment=np.zeros(3),
                second_moment=np.zeros((3, 3)),
                volume=0.0,
                fixed=True,
            )
        self._joint_bodies = {}
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
        self.validate_observables(observables)
        if self._engine is None:
            return
        if control is None:
            control = self._control_of_model()
        self._dt, self._control = dt, control
        self._report_clamp(dt)
        self._load(state_in)
        self._write_forces(state_in, control)
        self._write_joint_controls(control, dt)
        self._write_stitch_stiffness(dt)
        self._write_body_stiffness(dt)
        if self.model.device.is_capturing:
            # the solve joins the caller's graph; it cannot synchronize, grow or raise (see the class docstring)
            self._engine.stop_on_overflow = False
            self._engine.advance(dt)
            self._uipc_warp.arrays.add_frame_iterations(self._engine, self._iterations)
        else:
            status = self._advance_eager(dt)
            if status.failed:
                self._write_observables(observables)
                self._forget_written()  # the engine holds the failed solve: the next step takes state_in
                raise RuntimeError(self._failure(status))
        self._shift_joint_history()
        self._store(state_in.body_q, state_in.body_qd, state_out)
        self._write_observables(observables)

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
        for grow() (on the device, also inside a captured graph)."""
        if self._config.integrator != "bdf2":
            return
        for q_start, q_prev, q_kept in self._joint_history.values():
            wp.copy(q_kept, q_prev)
            wp.copy(q_prev, q_start)

    def _report_clamp(self, dt: float) -> None:
        """Warn once when a step's ``dt`` clamps the given ``kappa`` into libuipc's stiffness range."""
        kappa = self._config.kappa
        if not self._adaptive_kappa or self._clamp_reported or kappa is None or self._engine is None:
            return
        if dt == self._clamp_checked_dt:
            return
        self._clamp_checked_dt = dt
        _, used, lo, hi, _ = self._engine.contact_stiffness.resolve(dt)
        if used != kappa:
            self._clamp_reported = True
            warnings.warn(
                f"SolverIPC: kappa={kappa:g} is outside libuipc's contact stiffness range [{lo:.6g}, {hi:.6g}] Pa of "
                f"this model at dt={dt:g} s; the steps use {used:.6g} Pa, as libuipc clamps a stiffness it is given. "
                "The range scales with the mean vertex mass and with 1/dt^2 (contact_stiffness(dt) gives the stiffness "
                "of any dt); kappa=None takes its lower bound, libuipc's default. Not repeated.",
                NewtonWarning,
                stacklevel=3,
            )

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
                    self._body_hold,
                ],
                outputs=[vm.x, vm.v, self._changed],
                device=device,
            )
            self._uipc_warp.arrays.update_body_vertices(eng)  # surface vertices x = A x_bar + p
            self._aim_bodies_of(state_in)
            self._load_joints(state_in)
        eng.discard_friction(where=self._changed)
        eng.restart_history(where=self._changed)  # BDF2: the step is an implicit Euler step (BDF1: nothing)

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
                        model.joint_type,
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
        wrote, so that a caller who wrote the angle picks its turn (class docstring, **Joints**)."""
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

    def _write_joint_controls(self, control: Control, dt: float) -> None:
        """The joints' terms of the step from the model and ``control``, on the device (``_joint_terms``)."""
        model, groups = self.model, self._joint_groups
        if self._engine is None or model.joint_count == 0:
            return
        springs = all(a is not None for a in (model.joint_stiffness, model.joint_rest_q, model.joint_damping))
        target_q, target_qd, joint_f = control.joint_target_q, control.joint_target_qd, control.joint_f
        bdf2 = self._config.integrator == "bdf2"
        restart = self._engine.bdf1_frame[0:1] if bdf2 else self._no_restart
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
                    model.joint_type,
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
        """A new engine whose contact lists hold what the last frame asked for, at the start of that frame and
        with its external accelerations and its element groups' state (the joints' angles)."""
        old = self._engine
        capacity = old.grown_capacity()
        start = old.frame_start_state()
        kept = [e.frame_start_state() for e in old.elements]
        loop = old.solver.pcg.loop
        forces = None if self._forces is None else self._forces.a
        aims = None if self._aim_group is None else self._aim_group
        # release the old engine before the new one allocates
        self._engine = self._frame_graph = self._forces = old = None
        self._make_engine(capacity)
        self._engine.load_state(**start)
        for e, group_state in zip(self._engine.elements, kept, strict=True):
            if group_state:
                e.load_state(group_state)
        if forces is not None:
            wp.copy(self._forces.a, forces)
        if aims is not None:  # the aims of the step, which the new engine solves again
            for name in ("q_aim", "active", "strength"):
                wp.copy(getattr(self._aim_group, name), getattr(aims, name))
        self._engine.solver.pcg.loop = loop  # the PCG loop the steps so far chose, as FrameRunner carries it
        self._write_joint_controls(self._control or self._control_of_model(), self._dt)
        self._write_stitch_stiffness(self._dt)
        self._write_body_stiffness(self._dt)

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
        if self._config.integrator == "bdf2":  # the joints' history of the last step, which is solved again
            for _, q_prev, q_kept in self._joint_history.values():
                wp.copy(q_prev, q_kept)
        for _ in range(_MAX_REGROWS):
            self._regrow()
            if not self._solve_frame(self._dt).overflowed:
                break
        else:
            raise RuntimeError(f"SolverIPC: the contact lists still overflow after {_MAX_REGROWS} enlargements.")
        self._engine.load_state(**self._engine.frame_start_state())
        for e in self._engine.elements:
            group_state = e.frame_start_state()
            if group_state:
                e.load_state(group_state)
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
        solver carries the friction of the last step's contacts from one step to the next, and with
        ``integrator="bdf2"`` the integrator's history; this discards them, so that the next step is an implicit
        Euler step without friction. The contacts themselves are reused only where nothing moved, where they are
        the ones a new solver finds. Launched on the device without synchronization, so it can be captured.

        Args:
            state: Unused; the next step reads its own ``state_in``.
            world_mask: Optional boolean mask of shape ``(world_count + 1,)``. The worlds share one solve, so
                the reset applies to all of them when the mask selects a world that owns particles or bodies.
                ``None`` resets.
            flags: Unused.
        """
        del state, flags
        world_mask = _normalize_reset_world_mask(
            world_mask, world_count=int(self.model.world_count), device=self.model.device
        )
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
        wp.launch(
            _any_selected,
            1,
            inputs=[world_mask, self._world_entries_device],
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
        ``soft_contact_mu``, ``particle_mu`` or ``particle_radius``), of the springs (``MODEL_PROPERTIES`` with a new
        ``spring_rest_length``, ``spring_stiffness`` or ``spring_damping``), of the joints' types, bodies, frames or
        axes (``JOINT_PROPERTIES``) or of which bodies are kinematic (``BODY_PROPERTIES``,
        ``BODY_INERTIAL_PROPERTIES``) builds the solve again: the next step reads ``state_in`` in full, and graphs that
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
            _check_joints(model)  # the other joint data are read at every step
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
            _check_supported(model, self._coupled_worlds)
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
            if self._ortho is not None:  # and the default body stiffness the inertia
                self._ortho_inertia.assign(self._ortho_ratios())

    def contact_stiffness(self, dt: float) -> float:
        """The contact stiffness [Pa] of every contact pair in a step of length ``dt`` [s]: ``kappa``, or with
        ``adaptive_kappa`` ``kappa`` clamped into libuipc's stiffness range for this model and ``dt``, the range's lower
        bound when ``kappa`` is None (see **Contact** in the class docstring). 0 for a model without particles and
        bodies, which has no contact."""
        if self._engine is None:
            return 0.0
        policy = self._engine.contact_stiffness
        return float(self._config.kappa if policy is None else policy.resolve(dt)[1])

    def frame_stats(self) -> dict:
        """Solver counters of the last step: Newton/PCG iterations, line-search trials, convergence.

        Synchronizes with the device.
        """
        return {} if self._engine is None else self._engine.frame_stats()
