# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Newton's joints on libuipc's affine-body joints.

Which libuipc joint every Newton joint becomes, and the joints SolverIPC refuses (``_map_joints``, ``_check_joints``,
with the feature modules' refusals of what libuipc's joints do not hold: ``features``); how the joints are laid out in
the scene with their penalties (``_add_joints``, the distance joints' anchors); and the joints' work at every step:
their angles and distances at the step's start, the drives, springs, damping, limits and ``joint_f`` as libuipc's
driving terms (``_write_joint_controls``), and the BDF2 history of their coordinates.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from newton import Control, JointType, Model, NewtonWarning, State

from . import features
from .kernels import (
    _EFFORT_LIMIT,
    _adopt_joint_angles,
    _enable_anchor_joints,
    _joint_rates,
    _joint_terms,
)
from .scene import _quat_to_matrix_np

if TYPE_CHECKING:
    from .solver_ipc import SolverIPC


# Newton's default joint_velocity_limit (ModelBuilder.JointDofConfig's), which bounds nothing; SolverIPC warns of one
# below it (_warn_of_velocity_limits)
_VELOCITY_LIMIT = 1.0e6
# what to do with a joint of a type SolverIPC refuses, and why libuipc has no joint for it
_UNMAPPED_JOINTS = {
    JointType.ROD: "use SolverVBD or SolverKamino for rods. libuipc's rods are finite elements, not joints between two "
    "affine bodies",
}


@dataclass(frozen=True)
class _JointOptions:
    """The terms SolverIPC adds to libuipc's joints that a caller may switch off (:ref:`ipc-joints`), both on by
    default: ``limit_damping``, ``joint_limit_kd`` past a limit; ``effort_limit``, ``joint_effort_limit`` as a bound of
    the drive's force decided at the step's start."""

    limit_damping: bool = True
    effort_limit: bool = True


@dataclass(frozen=True)
class _JointMap:
    """How every joint of a model maps onto libuipc (``_map_joints``).

    ``kind``: the joint type whose libuipc joint the joint becomes, REVOLUTE, PRISMATIC, FIXED or BALL, or FREE for
    none; D6 for a D6 joint that a feature module took (``features.universal.take_d6``), of which this module builds,
    drives and stores nothing. ``store``: the type ``_store_joints`` and ``_joint_wrenches`` take the joint for, its own
    but for a D6 joint: that of its libuipc joint, or FIXED (skipped) for one whose coordinates and ``joint_f`` the D6
    kernels take or a feature module took. ``d6``: those joints, D6 joints of three angular axes, with or without three
    linear ones. ``distance``: the distance joints of one length, libuipc's stitch of that rest length between their
    joint frames' origins; the other distance joints, of no length or taken by a feature module
    (``features.universal.take_distance``), are no joint here, and every distance joint is stored as a free joint."""

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


def _map_joints(model: Model) -> _JointMap:
    """How every joint maps onto libuipc (``_JointMap``), or ``NotImplementedError`` for the first joint that does
    not: a joint type libuipc has no joint for (``_UNMAPPED_JOINTS``) and a D6 joint that takes Newton's coordinates of
    three axes that are not orthonormal. A D6 joint that locks other degrees of freedom than one of libuipc's joints
    does, and a distance joint whose distance is bounded to a range or on one side, are offered to the universal
    module in the order of the joints, which takes them or raises (``features.universal``)."""
    joint_type = model.joint_type.numpy().astype(np.int32)
    kind, store = joint_type.copy(), joint_type.copy()
    supported = [int(t) for t in (JointType.FREE, JointType.REVOLUTE, JointType.PRISMATIC, JointType.FIXED)]
    supported += [int(JointType.BALL), int(JointType.D6), int(JointType.DISTANCE)]
    other = np.flatnonzero(~np.isin(joint_type, supported))
    if len(other):
        j = int(other[0])
        t = JointType(int(joint_type[j]))
        why = _UNMAPPED_JOINTS.get(
            t, "use revolute, prismatic, fixed, ball, free, D6 or distance joints. libuipc has no joint of this type"
        )
        raise NotImplementedError(f"SolverIPC: joint {j} is a {t.name} joint; {why}.")
    d6 = np.flatnonzero(joint_type == int(JointType.D6))
    if len(d6):
        dof_dim = model.joint_dof_dim.numpy()
        qd_start = model.joint_qd_start.numpy()
        axes = model.joint_axis.numpy().astype(np.float64).reshape(-1, 3)
    for j in d6:
        lin, ang = (int(n) for n in dof_dim[j])
        t = _D6_KINDS.get((lin, ang))
        if t is None:  # what no libuipc joint locks: the universal module takes the joint, or raises
            features.universal.take_d6(model, int(j), lin, ang)
            kind[j], store[j] = int(JointType.D6), int(JointType.FIXED)
            continue
        if ang == 3:
            s = int(qd_start[j])
            for name, rows in (("linear", axes[s : s + lin]), ("angular", axes[s + lin : s + lin + ang])):
                if len(rows) and not _orthonormal(rows):
                    raise NotImplementedError(
                        f"SolverIPC: the {name} axes of the D6 joint {j} are not orthonormal; make them orthonormal. "
                        "Newton's coordinates of a D6 joint of three axes assume they are."
                    )
        kind[j] = int(t)
        store[j] = int(JointType.FIXED) if ang == 3 else int(t)
    distance = np.flatnonzero(joint_type == int(JointType.DISTANCE))
    if len(distance):
        # the bounds of a distance joint are those of its first degree of freedom, a negative one being no bound
        first = model.joint_qd_start.numpy()[distance]
        lo, hi = model.joint_limit_lower.numpy()[first], model.joint_limit_upper.numpy()[first]
        taken = np.zeros(len(distance), bool)
        for k, (j, a, b) in enumerate(zip(distance, lo, hi, strict=True)):
            if (a >= 0.0 or b >= 0.0) and not (a == b):  # a range: the universal module takes the joint, or raises
                features.universal.take_distance(model, int(j), a, b)
                taken[k] = True
        kind[distance] = int(JointType.FREE)
        store[distance] = int(JointType.FREE)
        distance = distance[(lo >= 0.0) & ~taken]
    three = (d6[dof_dim[d6, 1] == 3] if len(d6) else d6).astype(np.int32)
    return _JointMap(kind, store, three, bool(np.any(store != joint_type)), distance.astype(np.int32))


class _JointDofs:
    """The degrees of freedom of a model's joints as the refusals of ``_check_joints`` and of the feature modules
    (``features.joint_refusal``) read them: the joint of every degree of freedom (``joint``) and the type that joint
    maps onto (``kind``, ``_JointMap.kind``), the joints' types and first degrees of freedom (``joint_type``,
    ``qd_start``), the model's per-DOF arrays (``value``) and the message of a refusal (``refusal``)."""

    def __init__(self, model: Model, kind: np.ndarray):
        self.model = model
        self.joint_type = model.joint_type.numpy()
        self.qd_start = model.joint_qd_start.numpy()
        self.joint = np.repeat(np.arange(model.joint_count), np.diff(self.qd_start))
        self.kind = kind[self.joint]
        self._values = {}

    def value(self, name: str) -> np.ndarray:
        """The model's per-DOF array ``name`` in float64, zeros where the model has none."""
        if name not in self._values:
            a = getattr(self.model, name)
            self._values[name] = np.zeros(self.model.joint_dof_count) if a is None else a.numpy().astype(np.float64)
        return self._values[name]

    def refusal(self, dofs: np.ndarray, what: Callable[[int], str], remedy: str) -> tuple[int, str] | None:
        """The refusal of the first of ``dofs``, ``(dof, message)``, or None for none: the message says what it has
        (``what(dof)``), what to do and why (``remedy``)."""
        if not len(dofs):
            return None
        d = int(dofs[0])
        j = int(self.joint[d])
        name = JointType(int(self.joint_type[j])).name
        return d, f"SolverIPC: DOF {d} of the {name} joint {j} has {what(d)}; {remedy}."


def _refuse(refusal: tuple[int, str] | None) -> None:
    """Raise ``NotImplementedError`` with the message of ``refusal`` (``_JointDofs.refusal``), if any."""
    if refusal is not None:
        raise NotImplementedError(refusal[1])


def _check_joints(model: Model, options: _JointOptions | None = None) -> None:
    """Joints that libuipc's affine-body joints represent (:ref:`ipc-joints`): the joints ``_map_joints`` maps, no
    negative armature, and no negative effort limit on a driven degree of freedom (without ``options.effort_limit``,
    none below Newton's default). What the feature modules refuse of the degrees of freedom (``features``: joint
    friction, and armature, drives, springs, damping and limits on ball and free joints) is checked in the order
    these checks always had: friction, a negative armature, the modules' armature, drives and limits, the effort
    limit."""
    if model.joint_count == 0:
        return
    options = options or _JointOptions()
    kind = _map_joints(model).kind
    if model.joint_dof_count == 0:
        return
    dofs = _JointDofs(model, kind)
    _refuse(features.joint_refusal("friction", dofs))
    armature = dofs.value("joint_armature")
    _refuse(
        dofs.refusal(
            np.flatnonzero(~(armature >= 0.0)),
            lambda d: f"joint_armature {float(armature[d])!r}",
            "make it 0 or positive. An armature is an inertia",
        )
    )
    for category in ("armature", "drive", "limit"):
        _refuse(features.joint_refusal(category, dofs))
    ke, kd, effort = (dofs.value(name) for name in ("joint_target_ke", "joint_target_kd", "joint_effort_limit"))
    gains = (ke != 0.0) | (kd != 0.0)
    if options.effort_limit:
        _refuse(
            dofs.refusal(
                np.flatnonzero(gains & ~(effort >= 0.0)),
                lambda d: f"joint_effort_limit {float(effort[d])!r} and a drive",
                "make it 0 or positive. It bounds the drive's force",
            )
        )
    else:
        _refuse(
            dofs.refusal(
                np.flatnonzero(gains & (effort < _EFFORT_LIMIT)),
                lambda d: f"joint_effort_limit {float(effort[d])!r} and a drive",
                f"raise it to {_EFFORT_LIMIT:g}, Newton's default, or pass enable_joint_effort_limit=True for a bound "
                "decided at the start of every step. libuipc's quadratic drive cannot bound its force",
            )
        )


def _warn_of_velocity_limits(solver: SolverIPC, stacklevel: int) -> None:
    """Warn once per solver, with ``NewtonWarning``, of the joint velocity limits below Newton's default, which
    SolverIPC does not read (:ref:`ipc-not-read`): how many degrees of freedom have one, and the first of them.
    ``stacklevel`` is ``warnings.warn``'s."""
    model = solver.model
    if solver._velocity_limits_warned or model.joint_dof_count == 0:
        return
    velocity = model.joint_velocity_limit.numpy()
    dofs = np.flatnonzero(~(velocity >= _VELOCITY_LIMIT))
    if not len(dofs):
        return
    d = int(dofs[0])
    j = int(np.searchsorted(model.joint_qd_start.numpy(), d, side="right")) - 1
    many = "DOFs have" if len(dofs) > 1 else "DOF has"
    warnings.warn(
        f"SolverIPC does not limit joint velocities: {len(dofs)} joint {many} a joint_velocity_limit below Newton's "
        f"default {_VELOCITY_LIMIT:g}, the first DOF {d} of the {JointType(int(model.joint_type.numpy()[j])).name} "
        f"joint {j} ({float(velocity[d])!r}).",
        NewtonWarning,
        stacklevel=stacklevel,
    )
    solver._velocity_limits_warned = True


def _plan_distance_joints(solver: SolverIPC) -> None:
    """The anchors of the distance joints of one length (:ref:`ipc-distance-joints`): one on either side at the origin
    of its joint frame, ``joint_X_p`` in the parent's frame, the fixed body that stands for the world for a parent of
    -1, and ``joint_X_c`` in the child's, after the anchors of the attachments; for every such joint the body and the
    anchor of its two sides."""
    model = solver.model
    solver._distance_joints = np.zeros(0, np.int32)
    solver._distance_anchor = np.zeros((0, 2, 2), np.int64)  # joint, side (parent, child), (body, anchor)
    if model.joint_count == 0:
        return
    joints = _map_joints(model).distance
    parent, child = model.joint_parent.numpy()[joints], model.joint_child.numpy()[joints]
    X_p, X_c = model.joint_X_p.numpy().astype(np.float64), model.joint_X_c.numpy().astype(np.float64)
    anchor = np.zeros((len(joints), 2, 2), np.int64)
    for k, j in enumerate(joints):
        sides = ((parent[k] if parent[k] >= 0 else model.body_count, X_p[j, :3]), (child[k], X_c[j, :3]))
        for side, (b, point) in enumerate(sides):
            points = solver._body_anchors.get(int(b), np.zeros((0, 3)))
            anchor[k, side] = (b, len(points))
            solver._body_anchors[int(b)] = np.vstack([points, point])
    solver._distance_joints, solver._distance_anchor = joints, anchor


def _distance_penalties(solver: SolverIPC, mass: np.ndarray) -> np.ndarray:
    """The penalty of every distance joint of one length: a ball joint's, ``joint_strength_ratio`` times the
    mass of the dynamic bodies it joins, in the units of the incremental potential, so that its stitch's
    stiffness is that over ``dt^2``."""
    if not len(solver._distance_joints):
        return np.zeros(1)
    dynamic = np.append(np.where(solver._kinematic_body, 0.0, mass), 0.0)  # the world's after Newton's bodies
    bodies = solver._distance_anchor[:, :, 0]
    return solver._joint_strength_ratio * dynamic[bodies].sum(axis=1)


def _joint_inputs_now(solver: SolverIPC) -> tuple:
    """What the joints were laid out from: their types, bodies and frames. The other joint data are read at every
    step."""
    model = solver.model
    if model.joint_count == 0:
        return ()
    arrays = (model.joint_type, model.joint_parent, model.joint_child, model.joint_X_p, model.joint_X_c)
    arrays = (*arrays, model.joint_axis, model.joint_qd_start, model.joint_dof_dim)
    return tuple(a.numpy().copy() for a in arrays)


def _add_joints(solver: SolverIPC, scene) -> None:
    """Newton's revolute, prismatic, fixed and ball joints as libuipc's (:ref:`ipc-joints`), and the D6 joints that lock
    what one of them locks (``_map_joints``), joint frames taken as the reference rotations, a fixed affine body without
    mass standing for the world. Sets up what the per-step kernels
    read: per kind the joint of every group row and the row's penalty (``_joint_strength``, 0 without a dynamic body),
    per joint its row (``_joint_element``), and the free and ball joints, whose ``joint_f`` acts through the bodies'
    wrenches."""
    model, device = solver.model, solver.model.device
    solver._world_body = -1
    solver._joint_rows, solver._joint_strength, solver._joint_history, solver._joint_rate = {}, {}, {}, {}
    solver._joint_bodies = {}
    if model.joint_count == 0:
        return
    # the libuipc joint of every joint: its own type, but for a D6 joint that of the joint that locks what it locks;
    # the type the kernels that write the coordinates and joint_f take it for
    mapping = _map_joints(model)
    joint_type = mapping.kind
    solver._joint_store_type = model.joint_type
    if mapping.relabeled:
        solver._joint_store_type = wp.array(mapping.store, dtype=wp.int32, device=device)
    solver._d6_joints = wp.array(mapping.d6, dtype=wp.int32, device=device)
    parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
    X_p, X_c = model.joint_X_p.numpy().astype(np.float64), model.joint_X_c.numpy().astype(np.float64)
    axis = model.joint_axis.numpy().astype(np.float64).reshape(-1, 3)
    qd_start = model.joint_qd_start.numpy()
    wrenched = np.isin(mapping.store, [int(JointType.FREE), int(JointType.BALL)])
    solver._wrenched_joints = wp.array(np.flatnonzero(wrenched).astype(np.int32), dtype=wp.int32, device=device)
    element = np.full(model.joint_count, -1, np.int32)
    joints = np.flatnonzero(joint_type != int(JointType.FREE))
    # the distance joints' stitches need the world's body too, and so may the joints a feature module took (D6)
    world = np.concatenate([joints, mapping.distance])
    if len(world) and np.any(parent[world] < 0):
        solver._world_body = model.body_count
        scene.add_affine_body(
            np.zeros((0, 3)),
            np.zeros((0, 3), np.int64),
            mass=0.0,
            first_moment=np.zeros(3),
            second_moment=np.zeros((3, 3)),
            volume=0.0,
            fixed=True,
            anchors=solver._body_anchors.get(model.body_count),
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
        left = np.where(parent[rows] >= 0, parent[rows], solver._world_body)
        bodies = np.stack([left, child[rows]], axis=1)
        solver._joint_bodies[kind] = bodies
        R_p = np.array([_quat_to_matrix_np(q) for q in X_p[rows, 3:7]]).reshape(-1, 3, 3)
        R_c = np.array([_quat_to_matrix_np(q) for q in X_c[rows, 3:7]]).reshape(-1, 3, 3)
        p_p, p_c = X_p[rows, :3], X_c[rows, :3]
        # the rotations at which the joint frames coincide, in which Newton's joint coordinates are 0
        reference = np.stack([np.swapaxes(R_p, 1, 2), np.swapaxes(R_c, 1, 2)], axis=1)
        ratio, strength = _joint_strengths(solver, kind, solver._dyadic[0])
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
        solver._joint_rows[kind] = wp.array(rows.astype(np.int32), dtype=wp.int32, device=device)
        solver._joint_strength[kind] = wp.array(strength, dtype=wp.float64, device=device)
        if kind in ("revolute", "prismatic"):
            # the coordinate at the start of this step, of the step before (BDF2's drives), and the one before
            # that step's, which grow() takes back when it solves that step again
            solver._joint_history[kind] = tuple(wp.zeros(len(rows), dtype=wp.float64, device=device) for _ in range(3))
            # their rates likewise, for the armature and the effort limit (_joint_rates)
            solver._joint_rate[kind] = tuple(wp.zeros(len(rows), dtype=wp.float64, device=device) for _ in range(3))
    solver._joint_element = wp.array(element, dtype=wp.int32, device=device)
    solver._joint_q_written = wp.full(model.joint_coord_count, np.nan, dtype=wp.float32, device=device)


def _joint_strengths(solver: SolverIPC, kind: str, mass: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For the bodies' masses ``mass``, the strength ratio the joints of ``kind`` are built with and their
    penalty, ``joint_strength_ratio`` times the mass of the dynamic bodies a joint joins (the world and kinematic
    bodies count 0), where libuipc's ratio multiplies the masses of both bodies. The per-step kernels take the
    penalty: a fixed or ball joint's as it is, a revolute or prismatic joint's divided by the masses it was built
    with, which libuipc keeps."""
    bodies = solver._joint_bodies[kind]
    everything = np.append(mass, 0.0)  # the world's mass after Newton's bodies
    dynamic = np.append(np.where(solver._kinematic_body, 0.0, mass), 0.0)
    mass_sum, dynamic_sum = everything[bodies].sum(axis=1), dynamic[bodies].sum(axis=1)
    stiffness = solver._joint_strength_ratio * dynamic_sum
    ratio = np.divide(stiffness, mass_sum, out=np.zeros(len(bodies)), where=mass_sum > 0.0)
    return ratio, stiffness


def _shift_joint_history(solver: SolverIPC) -> None:
    """BDF2: the joint coordinates at the start of this step become those of the step before, whose own are kept
    for grow() (on the device, also inside a captured graph); and so do their rates (``_joint_rates``)."""
    if solver._config.integrator != "bdf2":
        return
    for q_start, q_prev, q_kept in (*solver._joint_history.values(), *solver._joint_rate.values()):
        wp.copy(q_kept, q_prev)
        wp.copy(q_prev, q_start)


def _load_joints(solver: SolverIPC, state_in: State) -> None:
    """The joints' angles and distances at the step's start, from the bodies' state just loaded. A revolute joint
    unwraps its angle against the one it had, or against ``state_in.joint_q`` where that is not what the solver wrote,
    so that a caller who wrote the angle picks its turn (:ref:`ipc-joints`); and the rates of the coordinates from the
    bodies' twists (``_joint_rates``)."""
    groups, x = solver._joint_groups, solver._engine.vm.x
    if groups.revolute is not None:
        if state_in.joint_q is not None:
            wp.launch(
                _adopt_joint_angles,
                groups.revolute.n,
                inputs=[
                    solver._joint_rows["revolute"],
                    solver.model.joint_q_start,
                    state_in.joint_q,
                    solver._joint_q_written,
                ],
                outputs=[groups.revolute.current],
                device=solver.model.device,
            )
        groups.revolute.compute_current_angles(x)
    if groups.prismatic is not None:
        groups.prismatic.compute_current_distances(x)
    model = solver.model
    for kind, (rate, _, _) in solver._joint_rate.items():  # the rates at the step's start (_joint_rates)
        wp.launch(
            _joint_rates,
            rate.shape[0],
            inputs=[
                solver._joint_rows[kind],
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


def _write_joint_controls(solver: SolverIPC, control: Control, dt: float) -> None:
    """The joints' terms of the step from the model and ``control``, on the device (``_joint_terms``)."""
    model, groups = solver.model, solver._joint_groups
    if solver._engine is None or model.joint_count == 0:
        return
    target_q, target_qd, joint_f = control.joint_target_q, control.joint_target_qd, control.joint_f
    bdf2 = solver._config.integrator == "bdf2"
    restart = solver._engine.bdf1_frame[0:1] if bdf2 else None
    # with several worlds a reset or a new state restarts the rows of its worlds only (reset, _load)
    rows = solver._engine.restart_rows if bdf2 else None
    options = solver._joint_options
    for kind, group, scale in (("revolute", groups.revolute, 1.0), ("prismatic", groups.prismatic, 0.5)):
        if group is None:
            continue
        q_start, q_prev, _ = solver._joint_history[kind]
        rate_start, rate_prev, _ = solver._joint_rate[kind]
        force_on, force = (group.tq_on, group.tq) if kind == "revolute" else (group.f_on, group.f)
        wp.launch(
            _joint_terms,
            group.n,
            inputs=[
                solver._joint_rows[kind],
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_enabled,
                model.joint_target_ke,
                model.joint_target_kd,
                model.joint_stiffness,
                model.joint_rest_q,
                model.joint_damping,
                model.joint_limit_lower,
                model.joint_limit_upper,
                model.joint_limit_ke,
                model.joint_limit_kd,
                int(options.limit_damping),
                model.joint_armature,
                rate_start,
                rate_prev,
                model.joint_effort_limit,
                int(options.effort_limit),
                target_q,
                target_qd,
                joint_f,
                solver._joint_strength[kind],
                group.mass_sum,
                group.current,
                q_prev,
                int(bdf2),
                restart,
                rows,
                model.joint_parent,
                model.joint_child,
                solver._body_base,
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
                inputs=[solver._joint_rows[kind], model.joint_enabled, solver._joint_strength[kind]],
                outputs=[group.kappa],
                device=model.device,
            )
