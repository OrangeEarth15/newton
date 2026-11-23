# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The drives, springs, damping, armature and friction of ball joints, and the refusals of what SolverIPC does not take
of ball, free and distance joints.

libuipc's ball joint holds the anchors only. libuipc-warp's ball terms (``uipc_warp.ext.ball.BallTerms``) add, per
axis of the parent's joint frame, a drive on the rotation vector of the joint's rotation, a spring toward a rest
rotation, damping, an armature and friction on the rotation of the step, and the drive's effort limit (:ref:`ipc-ball`).
A BALL joint whose degrees of freedom have a term gets them: from its degree of freedom ``a``, ``joint_target_ke``,
``joint_target_kd``, ``joint_stiffness``, ``joint_damping``, ``joint_armature``, ``joint_friction`` and
``joint_effort_limit``, read at every step, from its coordinates the targets ``joint_target_q`` (a quaternion, of which
the terms take the rotation vector, as SolverMuJoCo does) and ``joint_rest_q``, and from its degrees of freedom
``joint_target_qd``; ``joint_target_mode`` is not read, as for revolute joints. Every step writes them on the device,
so a captured step takes new values at its next replay.

SolverIPC refuses, on a degree of freedom of a D6 joint that becomes a ball joint, joint friction, armature, drives,
passive springs and damping, a target mode other than NONE, and limits: a D6 joint's coordinates are angles about its
axes, on which no ball term acts. On a free joint (and a D6 joint that becomes one) it refuses armature, drives,
passive springs and damping, a target mode other than NONE and limits, and on a ball joint limits, which Newton does
not define for it. The refusals of free and distance joints share their messages with those of ball joints. The hooks
are those of ``features`` (its docstring).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from newton import MAXVAL, JointTargetMode, JointType, ModelFlags

from ..kernels import _EFFORT_BOUND
from ..scene import _quat_to_matrix_np

if TYPE_CHECKING:
    from newton import Control, Model, State
    from newton.solvers import SolverBase

    from ..joints import _JointDofs
    from ..solver_ipc import SolverIPC
    from . import FeatureOptions

# the model's per-DOF arrays of a ball joint whose value other than 0 gives it a term
_TERMS = ("joint_target_ke", "joint_target_kd", "joint_stiffness", "joint_damping", "joint_armature", "joint_friction")
_D6 = (
    "SolverIPC takes the drives, springs, damping, armature and friction of a BALL joint, on the rotation vector of its "
    "rotation, and those of no D6 joint of three angular axes, whose coordinates are angles about its axes"
)


def check(model: Model, options: FeatureOptions, at: str) -> None:
    """Nothing: the joints' degrees of freedom are checked through ``joint_refusal``."""


def _lowest(*refusals: tuple[int, str] | None) -> tuple[int, str] | None:
    """The refusal of the lowest degree of freedom among ``refusals``, or None."""
    found = [r for r in refusals if r is not None]
    return min(found, key=lambda r: r[0]) if found else None


def joint_refusal(category: str, dofs: _JointDofs) -> tuple[int, str] | None:
    """The first degree of freedom refused in ``category``: ``"friction"``, a ``joint_friction`` other than 0 on a D6
    joint that becomes a ball joint; ``"armature"``, a ``joint_armature`` other than 0, and ``"drive"``, gains, a
    passive spring or damping or a target mode other than NONE, on a free joint or a D6 joint that becomes a ball or
    free joint; ``"limit"``, a finite bound with a ``joint_limit_ke`` above 0 on a ball or free joint (a distance
    joint's bounds are its distance, not a limit)."""
    own = dofs.joint_type[dofs.joint]
    d6_ball = (dofs.kind == int(JointType.BALL)) & (own != int(JointType.BALL))
    free = dofs.kind == int(JointType.FREE)
    if category == "friction":
        friction = dofs.value("joint_friction")
        return dofs.refusal(
            np.flatnonzero((friction != 0.0) & d6_ball),
            lambda d: f"joint_friction {float(friction[d])!r}",
            f"set it to 0. {_D6}",
        )
    if category == "armature":
        armature = dofs.value("joint_armature")

        def what(d: int) -> str:
            return f"joint_armature {float(armature[d])!r}"

        return _lowest(
            dofs.refusal(
                np.flatnonzero((armature != 0.0) & free),
                what,
                "set it to 0. SolverIPC takes the armature of revolute, prismatic, ball and universal joints "
                "only; that of a free joint, a mass without weight and an inertia, has no term",
            ),
            dofs.refusal(np.flatnonzero((armature != 0.0) & d6_ball), what, f"set it to 0. {_D6}"),
        )
    if category == "drive":
        ke, kd = dofs.value("joint_target_ke"), dofs.value("joint_target_kd")
        ks, c = dofs.value("joint_stiffness"), dofs.value("joint_damping")
        model = dofs.model
        mode = model.joint_target_mode.numpy() if model.joint_target_mode is not None else np.zeros(len(ke), np.int32)
        actuated = np.isin(
            mode,
            [int(m) for m in (JointTargetMode.POSITION, JointTargetMode.VELOCITY, JointTargetMode.POSITION_VELOCITY)],
        )

        def drive(d: int) -> str:
            if actuated[d]:
                return f"the target mode {JointTargetMode(int(mode[d])).name}"
            return (
                f"target_ke {float(ke[d])!r}, target_kd {float(kd[d])!r}, stiffness {float(ks[d])!r} "
                f"and damping {float(c[d])!r}"
            )

        driven = actuated | (ke != 0.0) | (kd != 0.0) | (ks != 0.0) | (c != 0.0)
        return _lowest(
            dofs.refusal(
                np.flatnonzero(free & driven),
                drive,
                "set the gains, stiffness and damping to 0 and the target mode to NONE, and apply forces through "
                "Control.joint_f, which acts on every joint. SolverIPC drives, springs and damps revolute, prismatic, "
                "ball and universal joints only",
            ),
            dofs.refusal(
                np.flatnonzero(d6_ball & driven),
                drive,
                f"set the gains, stiffness and damping to 0 and the target mode to NONE. {_D6}",
            ),
        )
    if category == "limit":
        lo, hi = dofs.value("joint_limit_lower"), dofs.value("joint_limit_upper")
        limit_k = dofs.value("joint_limit_ke")
        limited = ((lo > -MAXVAL) | (hi < MAXVAL)) & (limit_k > 0.0)
        limited[dofs.qd_start[:-1][dofs.joint_type == int(JointType.DISTANCE)]] = False  # a distance joint's bounds

        def what(d: int) -> str:
            return f"the limits [{float(lo[d])!r}, {float(hi[d])!r}] with limit_ke {float(limit_k[d])!r}"

        remedy = "set limit_ke to 0, or the bounds to -MAXVAL and MAXVAL"
        return _lowest(
            dofs.refusal(
                np.flatnonzero(limited & (free | d6_ball)),
                what,
                f"{remedy}. libuipc limits revolute and prismatic joints only",
            ),
            dofs.refusal(
                np.flatnonzero(limited & (dofs.kind == int(JointType.BALL)) & ~d6_ball),
                what,
                f"{remedy}. Newton defines no limits of a ball joint: add_joint_ball takes none, its importers drop "
                "them and its solvers do not read them",
            ),
        )
    return None


def _with_terms(model: Model) -> np.ndarray:
    """The BALL joints (of their own type, not D6 joints that become one) with a term on one of their degrees of
    freedom (``_TERMS`` other than 0), in their order."""
    if model.joint_count == 0:
        return np.zeros(0, np.int32)
    balls = np.flatnonzero(model.joint_type.numpy() == int(JointType.BALL))
    if not len(balls):
        return np.zeros(0, np.int32)
    term = np.zeros(model.joint_dof_count, bool)
    for name in _TERMS:
        values = getattr(model, name)
        if values is not None:
            term |= values.numpy() != 0.0
    start = model.joint_qd_start.numpy()
    return np.array([j for j in balls if term[start[j] : start[j] + 3].any()], np.int32)


def build(solver: SolverIPC, scene) -> None:
    """The ball joints with terms (``_with_terms``): their bodies, as the spherical joints laid them out, and their joint
    frames in the bodies' frames."""
    model = solver.model
    solver._ball_terms = None  # the group of the last engine made from the scene
    solver._ball_joints = joints = _with_terms(model)
    if not len(joints):
        return
    try:
        import uipc_warp.ext.ball  # noqa: F401, PLC0415 (libuipc-warp is imported lazily)
    except ImportError as e:
        raise ImportError(
            "SolverIPC: the drives, springs, damping, armature and friction of ball joints need libuipc-warp's ball "
            "terms (uipc_warp.ext.ball); install a libuipc-warp that has them, or set those of every ball joint to 0."
        ) from e
    element = solver._joint_element.numpy()[joints]
    solver._ball_bodies = solver._joint_bodies["spherical"][element]
    X_p, X_c = model.joint_X_p.numpy().astype(np.float64), model.joint_X_c.numpy().astype(np.float64)
    solver._ball_frames = tuple(np.array([_quat_to_matrix_np(q) for q in X[joints, 3:7]]) for X in (X_p, X_c))
    solver._ball_rows = wp.array(joints, dtype=wp.int32, device=model.device)


def engine_inputs(solver: SolverIPC, inputs: dict) -> None:
    """The ball terms of the joints with terms (``uipc_warp.ext.ball.BallTerms``), appended to ``inputs["elements"]``;
    for an engine made again with longer lists, with the inputs the last step wrote into those of the engine before."""
    if not len(solver._ball_joints):
        return
    from uipc_warp.ext.ball import BallTerms  # noqa: PLC0415 (libuipc-warp is imported lazily)

    terms = BallTerms(solver._scene.layout(), solver._ball_bodies, *solver._ball_frames, solver.model.device)
    before = solver._ball_terms
    if before is not None:
        for new, old in zip(_inputs(terms), _inputs(before), strict=True):
            wp.copy(new, old)
    inputs["elements"].append(terms)
    solver._ball_terms = terms


def _inputs(terms) -> tuple:
    """The inputs of libuipc-warp's ``BallTerms`` that a step writes, in the order of ``_ball_inputs``' outputs."""
    return (
        terms.ke, terms.target, terms.ks, terms.rest, terms.kd, terms.target_velocity, terms.damping, terms.armature,
        terms.friction, terms.effort_limit,
    )  # fmt: skip


def parent_terms(solver: SolverIPC, engine) -> list:
    """The ball terms of ``engine`` and the joint of each of their elements, for ``BODY_PARENT_F`` (``observables``):
    they are terms of the ball joint, whose wrench on its child they are part of."""
    terms = solver._ball_terms
    if terms is None:
        return []
    return [(terms, solver._ball_joints)]


@wp.func
def _rotation_vector(q: wp.array[wp.float32], i: int) -> wp.vec3d:
    """The principal rotation vector of the quaternion ``q[i : i + 4]`` (x, y, z, w), as SolverMuJoCo converts a ball
    joint's target (``_target_quat_to_axis_angle``), in float64; 0 for a quaternion of length 0."""
    x = wp.float64(q[i])
    y = wp.float64(q[i + 1])
    z = wp.float64(q[i + 2])
    w = wp.float64(q[i + 3])
    out = wp.vec3d()
    length = wp.sqrt(x * x + y * y + z * z + w * w)
    if length > wp.float64(1.0e-20):
        v = wp.vec3d(x, y, z) / length
        s = wp.length(v)
        if s > wp.float64(0.0):
            angle = wp.float64(2.0) * wp.atan2(s, w / length)
            if angle > wp.float64(wp.pi):
                angle = angle - wp.float64(2.0 * wp.pi)
            out = v * (angle / s)
    return out


@wp.func
def _axes(values: wp.array[wp.float32], d: int, on: bool) -> wp.vec3d:
    """``values[d : d + 3]`` in float64, 0 where ``values`` is None or the joint is disabled."""
    out = wp.vec3d()
    if values and on:
        out = wp.vec3d(wp.float64(values[d]), wp.float64(values[d + 1]), wp.float64(values[d + 2]))
    return out


@wp.kernel
def _ball_inputs(
    joints: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_enabled: wp.array[wp.bool],
    target_ke: wp.array[wp.float32],
    target_kd: wp.array[wp.float32],
    stiffness: wp.array[wp.float32],
    rest_q: wp.array[wp.float32],
    damping: wp.array[wp.float32],
    armature: wp.array[wp.float32],
    friction: wp.array[wp.float32],
    effort_limit: wp.array[wp.float32],
    effort: int,
    target_q: wp.array[wp.float32],
    target_qd: wp.array[wp.float32],
    ke: wp.array[wp.vec3d],
    target: wp.array[wp.vec3d],
    ks: wp.array[wp.vec3d],
    rest: wp.array[wp.vec3d],
    kd: wp.array[wp.vec3d],
    target_velocity: wp.array[wp.vec3d],
    c: wp.array[wp.vec3d],
    a: wp.array[wp.vec3d],
    F: wp.array[wp.vec3d],
    bound: wp.array[wp.vec3d],
):
    """The ball terms of joint ``joints[k]`` for the next step, from Newton's model and control: its degrees of
    freedom's gains, stiffness, damping, armature and friction (0 when the joint is disabled), the rotation vectors of
    its target and rest quaternions, its target velocities, and with ``effort`` its effort limits below Newton's default
    (infinite otherwise: no bound)."""
    k = wp.tid()
    j = joints[k]
    d = joint_qd_start[j]
    i = joint_q_start[j]
    on = joint_enabled[j]
    ke[k] = _axes(target_ke, d, on)
    kd[k] = _axes(target_kd, d, on)
    ks[k] = _axes(stiffness, d, on)
    c[k] = _axes(damping, d, on)
    a[k] = _axes(armature, d, on)
    F[k] = _axes(friction, d, on)
    t = wp.vec3d()
    if target_q:
        t = _rotation_vector(target_q, i)
    target[k] = t
    r = wp.vec3d()
    if rest_q:
        r = _rotation_vector(rest_q, i)
    rest[k] = r
    target_velocity[k] = _axes(target_qd, d, True)
    limit = wp.vec3d(wp.float64(wp.inf), wp.float64(wp.inf), wp.float64(wp.inf))
    if effort != 0:
        for axis in range(3):
            value = wp.float64(effort_limit[d + axis])
            if value < _EFFORT_BOUND:
                limit[axis] = value
    bound[k] = limit


def prepare(solver: SolverIPC, state_in: State, control: Control, dt: float) -> None:
    """The ball terms of the step (``_ball_inputs``), on the device."""
    terms = solver._ball_terms
    if terms is None:
        return
    model = solver.model
    wp.launch(
        _ball_inputs,
        terms.n,
        inputs=[
            solver._ball_rows,
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
            int(solver._joint_options.effort_limit),
            control.joint_target_q,
            control.joint_target_qd,
        ],
        outputs=list(_inputs(terms)),
        device=model.device,
    )


def finish(solver: SolverIPC, state_out: State, observables: SolverBase.Observables | None) -> None:
    """Nothing."""


def notify(solver: SolverIPC, flags: ModelFlags | int) -> bool:
    """Whether a ball joint without terms when the solve was built has one now (a flag of the joints' degrees of
    freedom): the solve is built again with its terms. The solver checks the joints again (``joint_refusal``) when such
    a flag is reported."""
    dof_flags = (
        ModelFlags.JOINT_DOF_PROPERTIES
        | ModelFlags.JOINT_DOF_FORCE_PROPERTIES
        | ModelFlags.JOINT_DOF_INERTIAL_PROPERTIES
        | ModelFlags.JOINT_REFERENCE_POSE_PROPERTIES
    )
    if not flags & dof_flags:
        return False
    return not set(_with_terms(solver.model).tolist()) <= set(solver._ball_joints.tolist())
