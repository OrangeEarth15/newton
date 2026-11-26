# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Universal joints, and distance joints whose distance has a range or one bound.

libuipc's joints lock no rotation, two or three, and its stitch holds one distance. libuipc-warp's extension
``uipc_warp.ext.joints`` has the two joints in between (:ref:`ipc-universal-joints`, :ref:`ipc-distance-joints`):

- a D6 joint of two angular axes and no linear one, a universal joint, is a ``UniversalJoints`` element: the
  anchors held as a ball joint holds them, the rotation about ``axis_0 x axis_1`` locked, and per angle the drive,
  passive spring and damping, armature, effort limit, limits with their damping, friction and ``joint_f``, read from
  the joint's two degrees of freedom at every step;
- a distance joint whose bounds are a range, or one bound (a negative bound is none), is a ``RangeStitches`` stitch
  between the origins of its joint frames: free inside the range and held to it by libuipc's cubic joint limit, of
  DOF 0's ``joint_limit_ke`` as SolverIPC takes every limit, with ``joint_limit_kd`` past it.

``joints._map_joints`` offers both here (``take_d6``, ``take_distance``) and builds, drives and stores nothing of
them: the groups are made for every engine (``engine_inputs``), their terms written at every step (``prepare``), a
universal joint's ``joint_q`` and ``joint_qd`` after it (``finish``), and both give ``BODY_PARENT_F`` their terms
(``parent_terms``). Other D6 joints that lock what no libuipc joint locks, two angular axes that are not
orthonormal, and a distance whose lower bound is above its upper are refused. The other hooks are those of
``features`` (its docstring).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from newton import MAXVAL, JointType, ModelFlags

from ..kernels import _EFFORT_BOUND, _LIMIT_OVERSHOOT, _MAXVAL, _d6_angular_coordinates, _drive_inputs
from ..rotations import _quat_to_matrix_np

if TYPE_CHECKING:
    from newton import Control, Model, State
    from newton.solvers import SolverBase

    from ..solver_ipc import SolverIPC

_INFINITY = wp.constant(wp.float64(np.inf))
# a distance joint's upper bound that is none, Newton's MAXVAL: the cubic's 1 / w^3 cancels the strength's w^3
_NO_BOUND = wp.constant(wp.float64(MAXVAL))
# the DOF flags of notify_model_changed that change a distance joint's bounds
_DOF_FLAGS = (
    ModelFlags.JOINT_DOF_PROPERTIES
    | ModelFlags.JOINT_DOF_FORCE_PROPERTIES
    | ModelFlags.JOINT_DOF_INERTIAL_PROPERTIES
    | ModelFlags.JOINT_REFERENCE_POSE_PROPERTIES
)


def _orthonormal(axes: np.ndarray) -> bool:
    """Whether the rows of ``axes`` are orthonormal (to float32's rounding), as ``joints._map_joints`` requires of a D6
    joint's axes."""
    return bool(np.abs(axes @ axes.T - np.eye(len(axes))).max() < 1.0e-5)


def take_d6(model: Model, joint: int, linear: int, angular: int) -> None:
    """Take the D6 joint ``joint`` of ``linear`` linear and ``angular`` angular axes, which lock what no libuipc joint
    locks: a universal joint (two orthonormal angular axes, no linear one); ``NotImplementedError`` for the others."""
    if (linear, angular) == (0, 2):
        s = int(model.joint_qd_start.numpy()[joint])
        axes = model.joint_axis.numpy()[s : s + 2].astype(np.float64)
        if not _orthonormal(axes):
            raise NotImplementedError(
                f"SolverIPC: the angular axes of the D6 joint {joint} are not orthonormal; make them orthonormal. "
                "Newton's coordinates of a universal joint assume they are."
            )
        return
    raise NotImplementedError(
        f"SolverIPC: the D6 joint {joint} has {linear} linear and {angular} angular axes; give it no axis, one axis, "
        f"two or three angular axes, or three linear and three angular axes. Its axes lock {3 - linear} translations "
        f"and {3 - angular} rotations, which no joint of SolverIPC locks."
    )


def take_distance(model: Model, joint: int, lower, upper) -> None:
    """Take the distance joint ``joint``, whose distance has the range ``[lower, upper]`` (a negative bound is none),
    or ``NotImplementedError`` for a lower bound above the upper."""
    if lower >= 0.0 and upper >= 0.0 and lower > upper:
        raise NotImplementedError(
            f"SolverIPC: the DISTANCE joint {joint} bounds its distance to [{float(lower)!r}, {float(upper)!r}], the "
            "lower bound above the upper; make the lower bound at most the upper, or negative for none."
        )


# ---------------------------------------------------------------- kernels
@wp.kernel
def _universal_terms(
    joints: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_enabled: wp.array[wp.bool],
    target_ke: wp.array[wp.float32],
    target_kd: wp.array[wp.float32],
    stiffness: wp.array[wp.float32],
    rest_q: wp.array[wp.float32],
    damping: wp.array[wp.float32],
    joint_armature: wp.array[wp.float32],
    joint_friction: wp.array[wp.float32],
    effort_limit: wp.array[wp.float32],
    effort: int,
    limit_lower: wp.array[wp.float32],
    limit_upper: wp.array[wp.float32],
    limit_ke: wp.array[wp.float32],
    limit_kd: wp.array[wp.float32],
    limit_damping: int,
    target_q: wp.array[wp.float32],
    target_qd: wp.array[wp.float32],
    joint_f: wp.array[wp.float32],
    strength: wp.array[wp.float64],
    dt: wp.float64,
    kappa: wp.array[wp.float64],
    ke_out: wp.array[wp.vec2d],
    target_out: wp.array[wp.vec2d],
    kd_out: wp.array[wp.vec2d],
    target_velocity_out: wp.array[wp.vec2d],
    ks_out: wp.array[wp.vec2d],
    rest_out: wp.array[wp.vec2d],
    damping_out: wp.array[wp.vec2d],
    armature_out: wp.array[wp.vec2d],
    friction_out: wp.array[wp.vec2d],
    effort_limit_out: wp.array[wp.vec2d],
    limit_lower_out: wp.array[wp.vec2d],
    limit_upper_out: wp.array[wp.vec2d],
    limit_strength_out: wp.array[wp.vec2d],
    limit_damping_out: wp.array[wp.vec2d],
    torque_out: wp.array[wp.vec2d],
):
    """The terms of universal joint ``e`` (``joints[e]``) for the next step, from Newton's model and control, per angle
    from its degree of freedom (:ref:`ipc-universal-joints`): the penalty ``strength`` of the anchor and the lock, the
    drive's gains and targets, the passive spring and damping, the armature, the friction, the effort limit (below
    Newton's default 1e6, with ``effort``; else none), the limit as libuipc's cubic penalty with Newton's limit
    stiffness at ``_LIMIT_OVERSHOOT`` and its ``joint_limit_kd`` (with ``limit_damping``), and ``joint_f``; nothing
    for a disabled joint."""
    e = wp.tid()
    j = joints[e]
    on = joint_enabled[j]
    zero = wp.float64(0.0)
    kappa[e] = wp.where(on, strength[e], zero)
    ke = wp.vec2d()
    target = wp.vec2d()
    kd = wp.vec2d()
    target_velocity = wp.vec2d()
    ks = wp.vec2d()
    rest = wp.vec2d()
    c = wp.vec2d()
    armature = wp.vec2d()
    friction = wp.vec2d()
    bound = wp.vec2d(_INFINITY, _INFINITY)
    lower = wp.vec2d()
    upper = wp.vec2d()
    s = wp.vec2d()
    kl = wp.vec2d()
    tau = wp.vec2d()
    if on:
        for a in range(2):
            d = joint_qd_start[j] + a
            i = joint_q_start[j] + a
            ke_a, kd_a, ks_a, c_a, rest_a, target_a, target_velocity_a = _drive_inputs(
                d, i, target_ke, target_kd, stiffness, rest_q, damping, target_q, target_qd
            )
            ke[a] = ke_a
            kd[a] = kd_a
            ks[a] = ks_a
            c[a] = c_a
            rest[a] = rest_a
            target[a] = target_a
            target_velocity[a] = target_velocity_a
            armature[a] = wp.float64(joint_armature[d])
            if joint_friction:
                friction[a] = wp.float64(joint_friction[d])
            if effort != 0 and wp.float64(effort_limit[d]) < _EFFORT_BOUND:
                bound[a] = wp.float64(effort_limit[d])
            lo = limit_lower[d]
            hi = limit_upper[d]
            k = wp.float64(limit_ke[d])
            if k > zero and (lo > -_MAXVAL or hi < _MAXVAL):
                # libuipc's penalty s ((q - u) / (u - l))^3 is, at an overshoot delta, the energy
                # dt^2 limit_ke delta^3 / (3 delta_r)
                width = wp.float64(hi) - wp.float64(lo)
                w3 = wp.where(width > zero, width * width * width, wp.float64(1.0))
                lower[a] = wp.float64(lo)
                upper[a] = wp.float64(hi)
                s[a] = dt * dt * k * w3 / (wp.float64(3.0) * _LIMIT_OVERSHOOT)
                if limit_damping != 0:
                    kl[a] = wp.float64(limit_kd[d])
            if joint_f:
                tau[a] = wp.float64(joint_f[d])
    ke_out[e] = ke
    target_out[e] = target
    kd_out[e] = kd
    target_velocity_out[e] = target_velocity
    ks_out[e] = ks
    rest_out[e] = rest
    damping_out[e] = c
    armature_out[e] = armature
    friction_out[e] = friction
    effort_limit_out[e] = bound
    limit_lower_out[e] = lower
    limit_upper_out[e] = upper
    limit_strength_out[e] = s
    limit_damping_out[e] = kl
    torque_out[e] = tau


@wp.kernel
def _range_terms(
    joints: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_enabled: wp.array[wp.bool],
    limit_lower: wp.array[wp.float32],
    limit_upper: wp.array[wp.float32],
    limit_ke: wp.array[wp.float32],
    limit_kd: wp.array[wp.float32],
    limit_damping: int,
    dt: wp.float64,
    lower: wp.array[wp.float64],
    upper: wp.array[wp.float64],
    strength: wp.array[wp.float64],
    damping: wp.array[wp.float64],
):
    """The range of distance joint ``k`` (``joints[k]``) for the next step (:ref:`ipc-distance-joints`): its DOF 0's
    bounds, 0 for a negative lower one and ``MAXVAL`` for a negative upper one; libuipc's cubic limit of strength
    ``dt^2 joint_limit_ke w^3 / (3 delta_r)`` (``w`` the range's width), and ``joint_limit_kd`` past it (with
    ``limit_damping``); nothing for a disabled joint or one whose bounds are both negative."""
    k = wp.tid()
    j = joints[k]
    d = joint_qd_start[j]
    zero = wp.float64(0.0)
    a = wp.float64(limit_lower[d])
    b = wp.float64(limit_upper[d])
    lo = wp.where(a >= zero, a, zero)
    hi = wp.where(b >= zero, b, _NO_BOUND)
    ke = wp.float64(limit_ke[d])
    on = joint_enabled[j] and ke > zero and (a >= zero or b >= zero)
    width = hi - lo
    w3 = wp.where(width > zero, width * width * width, wp.float64(1.0))
    lower[k] = lo
    upper[k] = hi
    strength[k] = wp.where(on, dt * dt * ke * w3 / (wp.float64(3.0) * _LIMIT_OVERSHOOT), zero)
    damping[k] = wp.where(on and limit_damping != 0, wp.float64(limit_kd[d]), zero)


@wp.kernel
def _adopt_universal_angles(
    joints: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_q: wp.array[wp.float32],
    written: wp.array[wp.float32],
    current: wp.array[wp.vec2d],
):
    """A universal joint whose ``state_in.joint_q`` is not what the solver wrote takes it as the angles the next unwrap
    starts from (``compute_current_angles``), which therefore keeps their turns."""
    e = wp.tid()
    i = joint_q_start[joints[e]]
    c = current[e]
    for a in range(2):
        q = joint_q[i + a]
        if q != written[i + a]:
            c[a] = wp.float64(q)
    current[e] = c


@wp.kernel
def _store_universal(
    joints: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    current: wp.array[wp.vec2d],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    written: wp.array[wp.float32],
):
    """``joint_q`` and ``joint_qd`` of universal joint ``e``: its multi-turn angles, and the rates
    :func:`newton.eval_ik` computes from the bodies' state (Newton's ``invert_2d_rotational_dofs``: the decomposition
    in the order ``(axis_0, axis_0 x axis_1, axis_1)``, of which the first and the last); ``written`` keeps the
    angles."""
    e = wp.tid()
    j = joints[e]
    p = joint_parent[j]
    i = joint_q_start[j]
    d = joint_qd_start[j]
    X_wpj = joint_X_p[j]
    w_p = wp.vec3()
    if p >= 0:
        X_wpj = body_q[p] * X_wpj
        w_p = wp.spatial_bottom(body_qd[p])
    c = joint_child[j]
    X_wcj = body_q[c] * joint_X_c[j]
    a0 = joint_axis[d]
    a1 = joint_axis[d + 1]
    _coords, rates = _d6_angular_coordinates(
        a0,
        wp.cross(a0, a1),
        a1,
        wp.transform_get_rotation(X_wpj),
        wp.transform_get_rotation(X_wcj),
        wp.spatial_bottom(body_qd[c]) - w_p,
    )
    q = current[e]
    joint_q[i] = wp.float32(q[0])
    joint_q[i + 1] = wp.float32(q[1])
    joint_qd[d] = rates[0]
    joint_qd[d + 1] = rates[2]
    written[i] = joint_q[i]
    written[i + 1] = joint_q[i + 1]


# ---------------------------------------------------------------- the plan of a solve
@dataclass
class _Universal:
    """The universal joints of a solve: the joint of every element (``joints``, also on the device as ``rows``), the
    scene's bodies of its two sides (the parent, or the fixed body that stands for the world, and the child), its
    origin and two axes in their frames, and its penalty ``strength`` (the ball joint's, ``joint_strength_ratio``
    times the mass of the dynamic bodies it joins); ``group``, the ``UniversalJoints`` of the last engine made."""

    joints: np.ndarray
    rows: wp.array
    bodies: np.ndarray
    positions: np.ndarray  # (n, 2, 3)
    axes: np.ndarray  # (n, 2, 2, 3)
    strength: wp.array
    group: object = None


@dataclass
class _Ranges:
    """The distance joints with a range of a solve: the joint of every stitch (``joints``, ``rows``), the bodies of
    its two ends (-1 for the world) and the origins of its joint frames in their frames; ``stitches``, the
    ``RangeStitches`` of the last engine made."""

    joints: np.ndarray
    rows: wp.array
    pairs: np.ndarray
    points: np.ndarray
    stitches: object = None


def _universal_joints(model: Model) -> np.ndarray:
    """The universal joints of ``model``: its D6 joints of two angular axes and no linear one."""
    if model.joint_count == 0:
        return np.zeros(0, np.int64)
    d6 = model.joint_type.numpy() == int(JointType.D6)
    dim = model.joint_dof_dim.numpy()
    return np.flatnonzero(d6 & (dim[:, 0] == 0) & (dim[:, 1] == 2))


def _range_joints(model: Model) -> np.ndarray:
    """The distance joints of ``model`` whose bounds are a range or one bound: those ``joints._map_joints`` offers."""
    if model.joint_count == 0:
        return np.zeros(0, np.int64)
    distance = np.flatnonzero(model.joint_type.numpy() == int(JointType.DISTANCE))
    first = model.joint_qd_start.numpy()[distance]
    lo, hi = model.joint_limit_lower.numpy()[first], model.joint_limit_upper.numpy()[first]
    return distance[((lo >= 0.0) | (hi >= 0.0)) & (lo != hi)]


def _universal_strengths(solver: SolverIPC, bodies: np.ndarray) -> np.ndarray:
    """The penalty of every universal joint of the scene's ``bodies``: ``joint_strength_ratio`` times the mass of the
    dynamic bodies it joins (the world and kinematic bodies count 0), as a ball joint's."""
    model = solver.model
    mass = model.body_mass.numpy().astype(np.float64) if model.body_count else np.zeros(0)
    dynamic = np.append(np.where(solver._parts.kinematic_body, 0.0, mass), 0.0)  # the world's after Newton's bodies
    return solver._joint_strength_ratio * dynamic[bodies].sum(axis=1)


@dataclass
class _State:
    """What the module keeps of a solve (``solver._features.universal``): its universal joints and its distance joints
    with a range, None for a solve without them."""

    universal: _Universal | None = None
    ranges: _Ranges | None = None


def _planned(solver: SolverIPC) -> tuple[_Universal | None, _Ranges | None]:
    """The universal joints and the distance ranges of the solve, None where it has none or was not built."""
    state = solver._features.universal
    return (None, None) if state is None else (state.universal, state.ranges)


def build(solver: SolverIPC, scene) -> None:
    """Plan the universal joints and the distance joints with a range (``_Universal``, ``_Ranges``): their bodies, and
    their frames' origins and axes in the bodies' frames, from the model's joint frames."""
    model, device = solver.model, solver.model.device
    state = solver._features.universal = _State()
    if model.joint_count == 0:
        return
    parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
    X_p, X_c = model.joint_X_p.numpy().astype(np.float64), model.joint_X_c.numpy().astype(np.float64)
    joints = _universal_joints(model)
    if len(joints):
        qd_start = model.joint_qd_start.numpy()
        axis = model.joint_axis.numpy().astype(np.float64)
        bodies = np.stack(
            [np.where(parent[joints] >= 0, parent[joints], solver._joints.world_body), child[joints]], axis=1
        )
        positions = np.stack([X_p[joints, :3], X_c[joints, :3]], axis=1)
        axes = np.zeros((len(joints), 2, 2, 3))
        for k, j in enumerate(joints):
            a = axis[qd_start[j] : qd_start[j] + 2]
            for side, X in enumerate((X_p[j], X_c[j])):
                axes[k, side] = a @ _quat_to_matrix_np(X[3:7]).T  # the joint frame's axes in the body's frame
        state.universal = _Universal(
            joints,
            wp.array(joints.astype(np.int32), dtype=wp.int32, device=device),
            bodies,
            positions,
            axes,
            wp.array(_universal_strengths(solver, bodies), dtype=wp.float64, device=device),
        )
    joints = _range_joints(model)
    if len(joints):
        pairs = np.stack([np.where(parent[joints] >= 0, parent[joints], -1), child[joints]], axis=1)
        points = np.stack([X_p[joints, :3], X_c[joints, :3]], axis=1)  # the parent's in the world for -1
        state.ranges = _Ranges(joints, wp.array(joints.astype(np.int32), dtype=wp.int32, device=device), pairs, points)


def engine_inputs(solver: SolverIPC, inputs: dict) -> None:
    """The universal joints' and the distance ranges' element groups of an engine made from the scene, with the terms
    of the last step (none before the first)."""
    universal, ranges = _planned(solver)
    if universal is None and ranges is None:
        return
    from uipc_warp.ext import joints as ext  # noqa: PLC0415 (libuipc-warp is imported lazily)

    device = solver.model.device
    if universal is not None:
        p, a = universal.positions, universal.axes
        universal.group = ext.UniversalJoints(
            universal.bodies, p[:, 0], p[:, 1], a[:, 0], a[:, 1], inputs["affine_bodies"], device
        )
        inputs["elements"].append(universal.group)
    if ranges is not None:
        ranges.stitches = ext.RangeStitches(
            ranges.pairs, 0.0, 0.0, 0.0, inputs["affine_bodies"], device, points=ranges.points
        )
        inputs["elements"] += ranges.stitches.groups
    if solver._control is not None:  # an engine built again within a step, or after one: that step's terms
        _write_terms(solver, solver._control, solver._dt)


def _write_terms(solver: SolverIPC, control: Control, dt: float) -> None:
    """The terms of the universal joints and the ranges of the distance joints for the next step, on the device."""
    model, options = solver.model, solver._joint_options
    universal, ranges = _planned(solver)
    if universal is not None:
        g = universal.group
        wp.launch(
            _universal_terms,
            len(universal.joints),
            inputs=[
                universal.rows,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_enabled,
                model.joint_target_ke,
                model.joint_target_kd,
                model.joint_stiffness,
                model.joint_rest_q,
                model.joint_damping,
                model.joint_armature,
                model.joint_friction,
                model.joint_effort_limit,
                int(options.effort_limit),
                model.joint_limit_lower,
                model.joint_limit_upper,
                model.joint_limit_ke,
                model.joint_limit_kd,
                int(options.limit_damping),
                control.joint_target_q,
                control.joint_target_qd,
                control.joint_f,
                universal.strength,
                wp.float64(dt),
            ],
            outputs=[
                g.kappa, g.ke, g.target, g.kd, g.target_velocity, g.ks, g.rest, g.damping, g.armature, g.friction,
                g.effort_limit, g.limit_lower, g.limit_upper, g.limit_strength, g.limit_damping, g.torque,
            ],
            device=model.device,
        )  # fmt: skip
    if ranges is not None:
        s = ranges.stitches
        wp.launch(
            _range_terms,
            len(ranges.joints),
            inputs=[
                ranges.rows,
                model.joint_qd_start,
                model.joint_enabled,
                model.joint_limit_lower,
                model.joint_limit_upper,
                model.joint_limit_ke,
                model.joint_limit_kd,
                int(options.limit_damping),
                wp.float64(dt),
            ],
            outputs=[s.lower, s.upper, s.strength, s.damping],
            device=model.device,
        )


def prepare(solver: SolverIPC, state_in: State, control: Control, dt: float) -> None:
    """The step's terms; a universal joint's angles at the step's start: those of the bodies' state just loaded,
    unwrapped against the angles it had, or against ``state_in.joint_q`` where that is not what the solver wrote."""
    universal, ranges = _planned(solver)
    if universal is None and ranges is None:
        return
    if universal is not None:
        group, model = universal.group, solver.model
        if state_in.joint_q is not None:
            wp.launch(
                _adopt_universal_angles,
                len(universal.joints),
                inputs=[universal.rows, model.joint_q_start, state_in.joint_q, solver._joints.q_written],
                outputs=[group.current],
                device=model.device,
            )
        group.compute_current_angles(solver._engine.vm.x)
    _write_terms(solver, control, dt)


def finish(solver: SolverIPC, state_out: State, observables: SolverBase.Observables | None) -> None:
    """``joint_q`` and ``joint_qd`` of the universal joints (``_store_universal``)."""
    universal, _ = _planned(solver)
    if universal is None or state_out is None or state_out.joint_q is None:
        return
    model = solver.model
    wp.launch(
        _store_universal,
        len(universal.joints),
        inputs=[
            universal.rows,
            model.joint_parent,
            model.joint_child,
            model.joint_X_p,
            model.joint_X_c,
            model.joint_axis,
            model.joint_q_start,
            model.joint_qd_start,
            universal.group.current,
            state_out.body_q,
            state_out.body_qd,
        ],
        outputs=[state_out.joint_q, state_out.joint_qd, solver._joints.q_written],
        device=model.device,
    )


def notify(solver: SolverIPC, flags: ModelFlags | int) -> bool:
    """New masses give the universal joints new penalties (``BODY_INERTIAL_PROPERTIES``); a distance joint that gains
    or loses its range (a DOF flag) needs the solve built again."""
    model = solver.model
    universal, ranges = _planned(solver)
    if flags & ModelFlags.BODY_INERTIAL_PROPERTIES and universal is not None:
        universal.strength.assign(_universal_strengths(solver, universal.bodies))
    if flags & _DOF_FLAGS and model.joint_count:
        planned = np.zeros(0, np.int64) if ranges is None else ranges.joints
        return not np.array_equal(_range_joints(model), planned)
    return False


def parent_terms(solver: SolverIPC, engine) -> list:
    """The element groups of the joints of this module in ``engine``, the engine being made, with the joint of every
    element: the universal joints, and the distance joints' range stitches by stencil size."""
    terms = []
    universal, ranges = _planned(solver)
    if universal is not None:
        terms.append((universal.group, universal.joints))
    if ranges is not None:
        for group in ranges.stitches.groups:
            terms.append((group, ranges.joints[group.rows]))
    return terms
