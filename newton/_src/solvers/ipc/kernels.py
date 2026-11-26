# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The Warp kernels and functions of SolverIPC.

They move Newton's state into libuipc-warp's float64 engine and back (particles, the affine bodies of rigid bodies,
joint coordinates), turn the external forces into the engine's accelerations, write the per-step terms of joints,
stitches and bodies, write the contact rows and the coupled solvers' proxy feedback from libuipc-warp's contact
records, and carry the solver's flags of new states, resets and status.
"""

from __future__ import annotations

import warp as wp

from newton import MAXVAL, JointType
from newton.math import quat_decompose, velocity_at_point

# the overshoot [rad or m] at which libuipc's cubic joint limit pushes back with the force of Newton's linear limit
# spring of stiffness joint_limit_ke; it is stiffer beyond it and softer below
_LIMIT_OVERSHOOT = wp.constant(wp.float64(1.0e-3))
# a joint limit at -MAXVAL or MAXVAL is no limit: Newton stores an infinite one there
_MAXVAL = wp.constant(wp.float32(MAXVAL))
# Newton's default joint_effort_limit, which bounds no drive
_EFFORT_LIMIT = 1.0e6
_EFFORT_BOUND = wp.constant(wp.float64(_EFFORT_LIMIT))
# MuJoCo's equality types (mjtEq: CONNECT, WELD, JOINT), and the target kinds with which Newton's importers link an
# equality row to the joint or the mimic row they made of it (MjcEqualityTargetKind.JOINT, MIMIC); none has a public
# name in Newton
_EQ_CONNECT, _EQ_WELD, _EQ_JOINT = 0, 1, 2
_EQ_TARGET_JOINT, _EQ_TARGET_MIMIC = 1, 2


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
    placing: wp.array[wp.int32],
    x: wp.array[wp.vec3d],
    v: wp.array[wp.vec3d],
    changed: wp.array[wp.int32],
    written: wp.array[wp.int32],
    request: wp.array[wp.int32],
):
    """Keep the float64 state where ``state_in`` holds its float32 rounding, take ``state_in`` elsewhere.

    ``changed[s]`` is set where the state of a particle of subscene ``s`` was taken that ``quiet`` does not mark (one
    entry, 0, with one world): a free particle that is no coupled solver's proxy. ``written[i]`` says whether the step
    takes particle ``i``'s position from ``state_in`` without sweeping its move, so that the frame start checks it
    (libuipc-warp's ``StartCheck``, which ``request[0]`` asks for): a particle that ``quiet`` does not mark, whose moves
    nothing sweeps, or a fixed particle or proxy that the step places (``placing`` of its subscene).
    """
    i = wp.tid()
    taken = int(0)
    moved = int(0)
    if _differs(_to32(x[i]), q[i]):
        x[i] = _to64(q[i])
        taken = 1
        moved = 1
    if _differs(_to32(v[i]), qd[i]):
        v[i] = _to64(qd[i])
        taken = 1
    if taken != 0 and quiet[i] == 0:
        changed[subscene[i]] = 1
    if moved != 0 and (quiet[i] == 0 or placing[subscene[i]] != 0):
        written[i] = 1
        request[0] = 1
    else:
        written[i] = 0


@wp.kernel
def _load_bodies(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    written_q: wp.array[wp.transform],
    written_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    base: wp.array[wp.int32],
    quiet: wp.array[wp.int32],
    swept: wp.array[wp.int32],
    hold: wp.array[wp.int32],
    subscene: wp.array[wp.int32],
    placing: wp.array[wp.int32],
    x: wp.array[wp.vec3d],
    v: wp.array[wp.vec3d],
    changed: wp.array[wp.int32],
    written: wp.array[wp.int32],
    request: wp.array[wp.int32],
):
    """Keep the affine state of a body whose ``state_in`` pose and twist are what the solver wrote last, or that
    ``hold`` holds (a kinematic body moving toward its aim, ``kinematic_motion="aim"``); give another the rigid state
    of ``state_in``: q = (p, rows of R) and its rate from Newton's COM twist.

    ``changed[s]`` is set where a body of subscene ``s`` that ``quiet`` does not mark took ``state_in``: a dynamic
    body that is no coupled solver's proxy. ``written`` at the body's first pseudo-vertex says whether the step takes
    its pose from ``state_in`` without sweeping its move, so that the frame start checks it (libuipc-warp's
    ``StartCheck``, which ``request[0]`` asks for): a body that ``swept`` does not mark (a dynamic one, or a kinematic
    one moving toward its aims), or a kinematic body or proxy that the step places (``placing`` of its subscene).
    """
    b = wp.tid()
    o = base[b]
    written[o] = 0
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
    pose = same
    for k in range(6):
        if tw[k] != seen_tw[k]:
            same = 0
    if same != 0:
        return
    if pose == 0 and (swept[b] == 0 or placing[subscene[b]] != 0):
        written[o] = 1
        request[0] = 1
    R = _rotation64(wp.transform_get_rotation(tf))
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
    through: wp.array[wp.int32],
    body_q_in: wp.array[wp.transform],
    body_qd_in: wp.array[wp.spatial_vector],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    written_q: wp.array[wp.transform],
    written_qd: wp.array[wp.spatial_vector],
):
    """Newton's body pose and COM twist from the affine state, A projected to its nearest rotation; also kept in
    ``written_q`` / ``written_qd`` for the next load. ``through[b]`` 1 takes the pose and the twist of ``state_in``
    instead (a kinematic body that jumps), 2 its twist (a kinematic body moving toward its aims): a kinematic body's
    twist is the caller's, which Newton's solvers pass through."""
    b = wp.tid()
    if through[b] == 1:
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
    if through[b] == 2:
        tw = body_qd_in[b]
    body_q[b] = tf
    body_qd[b] = tw
    written_q[b] = tf
    written_qd[b] = tw


@wp.kernel
def _store_body_momentum(
    x: wp.array[wp.vec3d],
    v: wp.array[wp.vec3d],
    base: wp.array[wp.int32],
    kinematic: wp.array[wp.int32],
    spread: wp.array[wp.mat33d],
    body_qd: wp.array[wp.spatial_vector],
    written_qd: wp.array[wp.spatial_vector],
):
    """``body_rotation="rigid"`` (:ref:`ipc-rigid-bodies`): the angular velocity of ``body_qd`` (and of ``written_qd``)
    that carries the body's angular momentum about its centre of mass with its rigid inertia, w = (R I R^T)^-1 L, with L
    = vee(A_dot S_c A^T - A S_c A_dot^T), I = tr(S_c) 1 - S_c and ``spread`` the second moment S_c about the centre of
    mass in the body frame; written over what ``_store_bodies`` wrote."""
    b = wp.tid()
    if kinematic[b] != 0:
        return
    o = base[b]
    A = wp.matrix_from_rows(x[o + 1], x[o + 2], x[o + 3])
    A_dot = wp.matrix_from_rows(v[o + 1], v[o + 2], v[o + 3])
    Sc = spread[b]
    M = A_dot * Sc * wp.transpose(A)
    L = wp.vec3d(M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1])
    R = _nearest_rotation(A)
    S = R * Sc * wp.transpose(R)
    inertia = wp.trace(S) * wp.identity(n=3, dtype=wp.float64) - S
    if not (wp.determinant(inertia) > wp.float64(0.0)):
        return
    tw = wp.spatial_vector(wp.spatial_top(body_qd[b]), _to32(wp.inverse(inertia) * L))
    body_qd[b] = tw
    written_qd[b] = tw


@wp.kernel
def _forget_kinematic_written(
    kinematic: wp.array[wp.int32],
    written_q: wp.array[wp.transform],
    written_qd: wp.array[wp.spatial_vector],
):
    """AL-IPC's swept kinematic bodies: forget what the solver wrote for them, so that the next step loads their pose
    from ``state_in`` again (one that contact held short of its pose goes on toward it)."""
    b = wp.tid()
    if kinematic[b] != 0:
        nan = wp.nan
        written_q[b] = wp.transform(wp.vec3(nan, nan, nan), wp.quat(nan, nan, nan, nan))
        written_qd[b] = wp.spatial_vector(nan, nan, nan, nan, nan, nan)


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
    residual: wp.array[wp.spatial_vector],
):
    """The aim of every kinematic body of the soft transform constraint's rows (``kinematic_motion="aim"``,
    :ref:`ipc-kinematic-bodies`): its ``state_in`` pose advanced by its twist over the step, the centre of mass by ``v
    dt``, the rotation by the exponential of ``omega dt``, as ``q_aim = (p, rows of R)``; the constraint on; the body's
    affine velocity ``(q_aim - q) / dt`` from its state ``q``, so that the step's prediction ``q + v dt`` is the aim,
    and with ``bdf2`` the history of a restart from that velocity (``x_nm1 = q - dt v``, ``v_nm1 = v``), whose
    prediction is the same; and the body held from the next step on, so that it starts where it ends.

    ``residual[k]`` is the twist the solver wrote for the body while it was dynamic, before a notification made it
    kinematic (NaN for none): a ``state_in`` twist equal to it is no twist the caller wrote, and the body takes a zero
    twist instead; any other twist ends that, and ``residual[k]`` becomes NaN."""
    k = wp.tid()
    b = bodies[k]
    tf = body_q[b]
    tw = body_qd[b]
    left = residual[k]
    same = int(1)
    for i in range(6):
        if tw[i] != left[i]:
            same = 0
    if same != 0:
        tw = wp.spatial_vector()
    else:
        nan = wp.nan
        residual[k] = wp.spatial_vector(nan, nan, nan, nan, nan, nan)
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
    particle_mass: wp.array[wp.float32],
    free: wp.array[wp.int32],
    acc: wp.array[wp.vec3d],
):
    """``f / m`` of the free particles, the first rows of the external accelerations; 0 without ``particle_f``."""
    k = wp.tid()
    a = wp.vec3d()
    if particle_f:
        i = free[k]
        a = _to64(particle_f[i]) / wp.float64(particle_mass[i])
    acc[k] = a


@wp.kernel
def _body_accelerations(
    body_f: wp.array[wp.spatial_vector],
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
    ``joint_f`` of its free and ball joints (``_joint_wrenches``), ``body_f`` None being none: M^-1 Q of its
    generalized force Q, four rows from ``row0 + 4 j``."""
    j = wp.tid()
    b = dynamic_body[j]
    wrench = joint_wrench[b]
    if body_f:
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


@wp.func
def _drive_inputs(
    d: int,
    i: int,
    target_ke: wp.array[wp.float32],
    target_kd: wp.array[wp.float32],
    stiffness: wp.array[wp.float32],
    rest_q: wp.array[wp.float32],
    damping: wp.array[wp.float32],
    target_q: wp.array[wp.float32],
    target_qd: wp.array[wp.float32],
):
    """The target gains ``ke`` and ``kd``, the passive stiffness ``ks`` and damping ``c``, the rest coordinate and the
    targets of the degree of freedom ``d`` (coordinate ``i``) of a revolute or prismatic joint, in float64: 0 where the
    model or the control has none (None)."""
    zero = wp.float64(0.0)
    ke = wp.float64(target_ke[d])
    kd = wp.float64(target_kd[d])
    ks = zero
    c = zero
    q_rest = zero
    if stiffness and rest_q and damping:
        ks = wp.float64(stiffness[d])
        c = wp.float64(damping[d])
        q_rest = wp.float64(rest_q[i])
    q_target = zero
    if target_q:
        q_target = wp.float64(target_q[i])
    qd_target = zero
    if target_qd:
        qd_target = wp.float64(target_qd[d])
    return ke, kd, ks, c, q_rest, q_target, qd_target


@wp.func
def _joint_restarted(
    j: int,
    restart: wp.array[wp.int32],
    restart_rows: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    body_base: wp.array[wp.int32],
) -> bool:
    """Whether the engine's frame starts the BDF2 history of joint ``j``'s bodies with an implicit Euler step: that of
    every row (``restart[0]``), or that of the rows of a reset or a new state of their world (``restart_rows``, at the
    base row of the child or of the parent); neither when they are None (implicit Euler, no restarts per row)."""
    restarted = False
    if restart:
        restarted = restart[0] != 0
    if restart_rows:
        if restart_rows[body_base[joint_child[j]]] != 0:
            restarted = True
        p = joint_parent[j]
        if p >= 0:
            if restart_rows[body_base[p]] != 0:
                restarted = True
    return restarted


# A joint's drive, passive spring and passive damping are one quadratic potential K/2 (q - N/K)^2 of the step's
# coordinate q, K/2 q^2 - N q up to a constant: minus its gradient is ke (q* - q) + kd (qd* - qd) - ks (q - q_rest) - c qd
# at the step's end, with implicit Euler's qd = (q - q_n) / dt or BDF2's qd = w (q - q_bar), w = 3 / (2 dt),
# q_bar = (4 q_n - q_nm1) / 3. Its stiffness K and load N under each integrator follow; the kernels call each where
# they computed it, which keeps their machine code.


@wp.func
def _drive_stiffness_bdf1(ke: wp.float64, kd: wp.float64, ks: wp.float64, c: wp.float64, dt: wp.float64) -> wp.float64:
    """The stiffness K of a drive's potential under implicit Euler."""
    return ke + ks + (kd + c) / dt


@wp.func
def _drive_stiffness_bdf2(ke: wp.float64, kd: wp.float64, ks: wp.float64, c: wp.float64, w: wp.float64) -> wp.float64:
    """The stiffness K of a drive's potential under BDF2, ``w`` = 3 / (2 dt)."""
    return ke + ks + (kd + c) * w


@wp.func
def _drive_load_bdf1(
    ke: wp.float64,
    kd: wp.float64,
    ks: wp.float64,
    c: wp.float64,
    q_rest: wp.float64,
    q_target: wp.float64,
    qd_target: wp.float64,
    q_n: wp.float64,
    dt: wp.float64,
) -> wp.float64:
    """The load N of a drive's potential under implicit Euler, from the coordinate ``q_n`` at the step's start."""
    return ke * q_target + ks * q_rest + (kd / dt) * (q_n + dt * qd_target) + (c / dt) * q_n


@wp.func
def _drive_load_bdf2(
    ke: wp.float64,
    kd: wp.float64,
    ks: wp.float64,
    c: wp.float64,
    q_rest: wp.float64,
    q_target: wp.float64,
    qd_target: wp.float64,
    q_bar: wp.float64,
    w: wp.float64,
) -> wp.float64:
    """The load N of a drive's potential under BDF2, ``w`` = 3 / (2 dt) and ``q_bar`` = (4 q_n - q_nm1) / 3."""
    return ke * q_target + ks * q_rest + kd * w * (q_bar + qd_target / w) + c * w * q_bar


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
    limit_lower: wp.array[wp.float32],
    limit_upper: wp.array[wp.float32],
    limit_ke: wp.array[wp.float32],
    limit_kd: wp.array[wp.float32],
    limit_damping: int,
    joint_armature: wp.array[wp.float32],
    rate_start: wp.array[wp.float64],
    rate_prev: wp.array[wp.float64],
    effort_limit: wp.array[wp.float32],
    effort: int,
    target_q: wp.array[wp.float32],
    target_qd: wp.array[wp.float32],
    joint_f: wp.array[wp.float32],
    strength: wp.array[wp.float64],
    mass_sum: wp.array[wp.float64],
    current: wp.array[wp.float64],
    q_prev: wp.array[wp.float64],
    bdf2: int,
    restart: wp.array[wp.int32],
    restart_rows: wp.array[wp.int32],
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
    """The terms of a revolute or prismatic joint for the next step, from Newton's model and control
    (:ref:`ipc-joints`): the strength ratio of its penalty ``strength`` (0 when disabled) for the engine's ``mass_sum``,
    the masses the joint was built with; the drive, passive spring and passive damping as one quadratic potential of the
    step (libuipc's driving term, ``drive_scale`` 1 for a revolute joint, 1/2 for a prismatic one, whose driving energy
    counts its distance twice), with the velocity at the step's end of the step's integrator: BDF1, or with ``bdf2``
    BDF2 from the coordinate ``q_prev`` at the start of the step before, unless the engine's frame starts the history of
    the joint's bodies with an implicit Euler step: that of every row (``restart[0]``), or that of the rows of a reset
    or a new state of their world (``restart_rows``, at the base row of the child or of the parent); in the same
    potential the inertia of the coordinate, the step's kinetic term ``a/2 |q - q~|^2 / dt^2`` (BDF2: 9/4 of it) of
    ``joint_armature`` a with the integrator's prediction q~ from the coordinate and its rate ``rate_start`` at the
    step's start (``_joint_rates``; BDF2 also the step before's, ``q_prev`` and ``rate_prev``), as libuipc writes the
    inertia of every degree of freedom, and with ``limit_damping`` the damping ``-joint_limit_kd qd`` of a joint whose
    coordinate is past one of its limits at the step's start, a passive damping of the step's velocity; with ``effort``
    the bound ``joint_effort_limit`` L (below Newton's default 1e6) of the drive's force, decided at the step's start:
    where ``ke (q* - q_n) + kd (qd* - qd_n)`` exceeds L in magnitude, the step takes the constant force of that sign and
    magnitude L as libuipc's joint torque or force (``force``, with ``joint_f``) in place of the drive's quadratic,
    whose passive spring and damping remain; the limit as libuipc's cubic penalty with Newton's limit stiffness at
    ``_LIMIT_OVERSHOOT``; ``joint_f`` as libuipc's external torque or force. ``q_start`` takes the coordinate at the
    step's start. Terms of 0 leave the potential of the drive, spring and damping bit for bit."""
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
    ke, kd, ks, c, q_rest, q_target, qd_target = _drive_inputs(
        d, i, target_ke, target_kd, stiffness, rest_q, damping, target_q, target_qd
    )
    q_n = current[e]
    q_start[e] = q_n
    restarted = _joint_restarted(j, restart, restart_rows, joint_parent, joint_child, body_base)
    second = bdf2 != 0 and not restarted
    # the drive, passive spring and passive damping as one quadratic potential of the step, K/2 (q - N/K)^2
    K = _drive_stiffness_bdf1(ke, kd, ks, c, dt)
    N = zero
    w = wp.float64(1.5) / dt
    q_bar = q_n
    if second:
        K = _drive_stiffness_bdf2(ke, kd, ks, c, w)
        q_bar = (wp.float64(4.0) * q_n - q_prev[e]) / wp.float64(3.0)
        N = _drive_load_bdf2(ke, kd, ks, c, q_rest, q_target, qd_target, q_bar, w)
    else:
        N = _drive_load_bdf1(ke, kd, ks, c, q_rest, q_target, qd_target, q_n, dt)
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
    lo = limit_lower[d]
    hi = limit_upper[d]
    limit_k = wp.float64(limit_ke[d])
    limited = on and limit_k > zero and (lo > -_MAXVAL or hi < _MAXVAL)
    if limit_damping != 0:
        # Featherstone's and SemiImplicit's -limit_kd qd while the coordinate is past a limit, decided at the step's
        # start, at the step's velocity: BDF1 (q - q_n) / dt, BDF2 w (q - q_bar)
        if limited and (q_n < wp.float64(lo) or q_n > wp.float64(hi)):
            kl = wp.float64(limit_kd[d])
            if second:
                K_add += kl * w
                N_add += kl * w * q_bar
            else:
                K_add += kl / dt
                N_add += (kl / dt) * q_n
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
    lim_on[e] = wp.where(limited, 1, 0)
    if limited:
        # libuipc's penalty s ((x - u) / (u - l))^3 is, at an overshoot delta, the energy
        # dt^2 limit_k delta^3 / (3 delta_r)
        width = wp.float64(hi) - wp.float64(lo)
        w3 = wp.where(width > zero, width * width * width, wp.float64(1.0))
        lim_lower[e] = wp.float64(lo)
        lim_upper[e] = wp.float64(hi)
        lim_strength[e] = dt * dt * limit_k * w3 / (wp.float64(3.0) * _LIMIT_OVERSHOOT)
    value = zero
    if joint_f:
        value = wp.float64(joint_f[d])
    if saturated:
        value = bound + value
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
    """The stiffness [N/m] of every row of a stitch group for the next step (:ref:`ipc-attachments`): ``source`` holds
    the row's attachment, or -1 less its spring. An attachment takes its stiffness, 0 when it is disabled, and a hard
    one (infinite stiffness) the penalty ``hard`` of a ball joint over ``dt^2``; a spring of rest length 0 takes its
    ``spring_stiffness``."""
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
    (:ref:`ipc-distance-joints`): ``source`` holds the row's distance joint among ``joints``, whose penalty ``hard`` is
    a ball joint's, over ``dt^2``, 0 when the joint is disabled; its rest length the joint's distance, the bounds of its
    first degree of freedom."""
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
    status: wp.array[wp.int32],
    residual: wp.array[wp.float32],
):
    """The status bits and the residual of the one solve, written into every world entry that owns content; ``status``
    or ``residual`` None: not requested."""
    k = wp.tid()
    if status:
        status[entries[k]] = status_in[0]
    if residual:
        residual[entries[k]] = residual_in[0]


@wp.kernel
def _keep_status(step: wp.array[wp.int32], kept: wp.array[wp.int32]):
    """The status bits of a step joined into those kept since ``check_status()`` last read them (an entry of a coupled
    solver, which steps the solver without observables): one word, or one per world entry with independent worlds."""
    k = wp.tid()
    kept[k] = kept[k] | step[k]


@wp.kernel
def _place_where(selected: wp.array[wp.int32], placing: wp.array[wp.int32]):
    """The subscenes a reset selects place their kinematic bodies and particles at the next step."""
    k = wp.tid()
    if selected[k] != 0:
        placing[k] = 1


@wp.kernel
def _count_step(count: wp.array[wp.int64]):
    """One more step taken, counted on the device, so that a step captured in a graph counts at every replay."""
    count[0] = count[0] + wp.int64(1)


@wp.kernel
def _records_due(count: wp.array[wp.int64], records_of: wp.array[wp.int64], due: wp.array[wp.int32]):
    """``due[0]`` 1 when the contact records are not those of the last step (``records_of``, the step count when they
    were written), else 0."""
    due[0] = wp.where(records_of[0] != count[0], 1, 0)


@wp.kernel
def _records_written(count: wp.array[wp.int64], records_of: wp.array[wp.int64]):
    """The contact records are those of the last step."""
    records_of[0] = count[0]


@wp.kernel
def _nonzero_activations(
    tris: wp.array[wp.int32],
    tri_activations: wp.array[wp.float32],
    tets: wp.array[wp.int32],
    tet_activations: wp.array[wp.float32],
    found: wp.array[wp.int32],
):
    """Set ``found[0]`` when a triangle of ``tris`` has an activation other than 0 in ``tri_activations``, ``found[1]``
    when a tetrahedron of ``tets`` has one in ``tet_activations`` (an absent array has none); one thread per triangle,
    then one per tetrahedron."""
    k = wp.tid()
    n = tris.shape[0]
    if k < n:
        if tri_activations:
            if tri_activations[tris[k]] != 0.0:
                found[0] = 1
    elif tet_activations:
        if tet_activations[tets[k - n]] != 0.0:
            found[1] = 1


@wp.func
def _record_count(rec_counts: wp.array[wp.int32]) -> int:
    """The live records of libuipc-warp's ``ContactForces``: the sum of the counts of its 10 segments, the active and
    the friction terms of kinds PT, EE, PE, PP and PH."""
    total = int(0)
    for k in range(10):
        total += rec_counts[k]
    return total


@wp.func
def _record_at(rec_counts: wp.array[wp.int32], rec_offsets: wp.array[wp.int32], idx: int):
    """The segment of the ``idx``-th live record of libuipc-warp's ``ContactForces``, counting the live records of the
    segments in their order, and its row among the records."""
    seg = int(0)
    i = idx
    while seg < 9 and i >= rec_counts[seg]:
        i -= rec_counts[seg]
        seg += 1
    return seg, rec_offsets[seg] + i


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
    """One row of ``Contacts`` per contact term with a force, from libuipc-warp's ``ContactForces`` records
    (:ref:`ipc-contact-output`): a rigid row for a term between two shapes, a soft row for a term between a particle
    feature and a shape, none between particle features. Grid-stride over the live records; a row takes its index from
    the rigid or the soft count, which counts every row, and is written only below its capacity."""
    tid = wp.tid()
    for idx in range(tid, _record_count(rec_counts), nthreads):
        seg, r = _record_at(rec_counts, rec_offsets, idx)
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
    """A coupled solver's proxy-body feedback (:ref:`ipc-coupling`): the force of every contact term of the step on a
    vertex of a proxy body, and its torque about the body's centre of mass at the end of the step, added at the proxy's
    index ``proxy_of_body[b]``. Grid-stride over the live records of libuipc-warp's ``ContactForces``."""
    tid = wp.tid()
    for idx in range(tid, _record_count(rec_counts), nthreads):
        seg, r = _record_at(rec_counts, rec_offsets, idx)
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
    """A coupled solver's proxy-particle feedback (:ref:`ipc-coupling`): the force of every contact term of the step on
    a proxy particle, added at the proxy's index ``proxy_of_particle[p]``. Grid-stride over the live records of
    libuipc-warp's ``ContactForces``."""
    tid = wp.tid()
    for idx in range(tid, _record_count(rec_counts), nthreads):
        seg, r = _record_at(rec_counts, rec_offsets, idx)
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
def _gravity_changed(
    now: wp.array[wp.vec3d],
    subscene: wp.array[wp.int32],
    last: wp.array[wp.vec3d],
    restart: wp.array[wp.int32],
):
    """``integrator="bdf2"``: ``restart[subscene[w]]`` = 1 where the gravity of world entry ``w`` that the step takes,
    ``now[w]``, differs from the one the step before took, ``last[w]``, which then takes it; ``subscene[w]`` is -1 for
    an entry without particles or bodies."""
    w = wp.tid()
    g, h = now[w], last[w]
    if g[0] != h[0] or g[1] != h[1] or g[2] != h[2]:
        last[w] = g
        if subscene[w] >= 0:
            restart[subscene[w]] = 1


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
