# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Newton's joints on libuipc's affine-body joints.

Which libuipc joint every Newton joint becomes, and the joints SolverIPC refuses (``_map_joints``, ``_check_joints``);
how the joints are laid out in the scene with their penalties (``_add_joints``, the distance joints' anchors); and the
joints' work at every step: their angles and distances at the step's start, the drives, springs, damping, limits and
``joint_f`` as libuipc's driving terms (``_write_joint_controls``), and the BDF2 history of their coordinates.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from newton import MAXVAL, Control, JointTargetMode, JointType, Model, State

from .kernels import (
    _EFFORT_LIMIT,
    _adopt_joint_angles,
    _enable_anchor_joints,
    _joint_option_terms,
    _joint_rates,
    _joint_terms,
)
from .scene import _quat_to_matrix_np

if TYPE_CHECKING:
    from .solver_ipc import SolverIPC


# why libuipc has no joint for a joint type SolverIPC refuses
_UNMAPPED_JOINTS = {
    JointType.DISTANCE: "libuipc has no distance joint; its affine-body joints are revolute, prismatic, fixed and ball "
    "joints",
    JointType.D6: "libuipc has no D6 joint (constitution 28 is only registered); its affine-body joints are revolute, "
    "prismatic, fixed and ball joints",
    JointType.ROD: "libuipc has no constitution of stretch, shear, bend and twist between two affine bodies; its rods "
    "are finite elements (Kirchhoff rod bending, constitution 15) and codimensional affine bodies",
}


@dataclass(frozen=True)
class _JointOptions:
    """The joints SolverIPC maps beyond revolute, prismatic, fixed, ball and free joints, on request
    (:ref:`ipc-joints`): ``d6``, D6 joints that lock what one of libuipc's joints locks; ``distance``, distance joints
    of one length or none; ``limit_damping``, ``joint_limit_kd`` past a limit; ``armature``, ``joint_armature`` as the
    inertia of a revolute or prismatic joint's coordinate; ``effort_limit``, ``joint_effort_limit`` as a bound of the
    drive's force decided at the step's start."""

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


def _check_joints(model: Model, options: _JointOptions | None = None) -> None:
    """Joints that libuipc's affine-body joints represent (:ref:`ipc-joints`): revolute, prismatic, fixed, ball and free
    joints, and with ``options`` the joints ``_map_joints`` maps; drives, passive springs, damping and limits on
    revolute and prismatic joints only; no armature (with ``options.armature``, none on ball and free joints), no joint
    friction, and no effort limit on a driven degree of freedom (with ``options.effort_limit``, none that is
    negative)."""
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


def _plan_distance_joints(solver: SolverIPC) -> None:
    """The anchors of the distance joints of one length (``enable_distance_joints``, :ref:`ipc-distance-joints`): one on
    either side at the origin of its joint frame, ``joint_X_p`` in the parent's frame, the fixed body that stands for
    the world for a parent of -1, and ``joint_X_c`` in the child's, after the anchors of the attachments; for every such
    joint the body and the anchor of its two sides."""
    model = solver.model
    solver._distance_joints = np.zeros(0, np.int32)
    solver._distance_anchor = np.zeros((0, 2, 2), np.int64)  # joint, side (parent, child), (body, anchor)
    if model.joint_count == 0 or not solver._joint_options.distance:
        return
    joints = _map_joints(model, solver._joint_options).distance
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
    """Newton's revolute, prismatic, fixed and ball joints as libuipc's (:ref:`ipc-joints`), and with
    ``enable_d6_joints`` the D6 joints that lock what one of them locks (``_map_joints``), joint frames taken as the
    reference rotations, a fixed affine body without mass standing for the world. Sets up what the per-step kernels
    read: per kind the joint of every group row and the row's penalty (``_joint_strength``, 0 without a dynamic body),
    per joint its row (``_joint_element``), and the free and ball joints, whose ``joint_f`` acts through the bodies'
    wrenches."""
    model, device = solver.model, solver.model.device
    solver._world_body = -1
    solver._joint_rows, solver._joint_strength, solver._joint_history, solver._joint_rate = {}, {}, {}, {}
    solver._joint_bodies = {}
    if model.joint_count == 0:
        return
    # the libuipc joint of every joint: its own type, but for a D6 joint (enable_d6_joints) that of the joint
    # that locks what it locks; the type the kernels that write the coordinates and joint_f take it for
    mapping = _map_joints(model, solver._joint_options)
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
    world = np.concatenate([joints, mapping.distance])  # the distance joints' stitches need it too
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
            if solver._joint_options.rates:  # their rates likewise (_joint_rates)
                solver._joint_rate[kind] = tuple(wp.zeros(len(rows), dtype=wp.float64, device=device) for _ in range(3))
    solver._no_joint_state = wp.zeros(1, dtype=wp.float64, device=device)  # the state of a kind without joints
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
    for grow() (on the device, also inside a captured graph); and so do their rates, where the joint options need
    them (``_joint_rates``)."""
    if solver._config.integrator != "bdf2":
        return
    for q_start, q_prev, q_kept in (*solver._joint_history.values(), *solver._joint_rate.values()):
        wp.copy(q_kept, q_prev)
        wp.copy(q_prev, q_start)


def _load_joints(solver: SolverIPC, state_in: State) -> None:
    """The joints' angles and distances at the step's start, from the bodies' state just loaded. A revolute joint
    unwraps its angle against the one it had, or against ``state_in.joint_q`` where that is not what the solver wrote,
    so that a caller who wrote the angle picks its turn (:ref:`ipc-joints`). Where the joint options need them, the
    rates of the coordinates from the bodies' twists (``_joint_rates``)."""
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
    """The joints' terms of the step from the model and ``control``, on the device (``_joint_terms``), and with the
    joint options that add to the driving term their terms (``_joint_option_terms``)."""
    model, groups = solver.model, solver._joint_groups
    if solver._engine is None or model.joint_count == 0:
        return
    springs = all(a is not None for a in (model.joint_stiffness, model.joint_rest_q, model.joint_damping))
    target_q, target_qd, joint_f = control.joint_target_q, control.joint_target_qd, control.joint_f
    bdf2 = solver._config.integrator == "bdf2"
    restart = solver._engine.bdf1_frame[0:1] if bdf2 else solver._no_restart
    # with several worlds a reset or a new state restarts the rows of its worlds only (reset, _load)
    rows = solver._engine.restart_rows if bdf2 else None
    restart_rows = solver._no_restart if rows is None else rows
    for kind, group, scale in (("revolute", groups.revolute, 1.0), ("prismatic", groups.prismatic, 0.5)):
        if group is None:
            continue
        q_start, q_prev, _ = solver._joint_history[kind]
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
                solver._joint_strength[kind],
                group.mass_sum,
                group.current,
                q_prev,
                int(bdf2),
                restart,
                restart_rows,
                int(rows is not None),
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
        if solver._joint_options.drive_terms:
            wp.launch(
                _joint_option_terms,
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
                    int(springs),
                    model.joint_limit_lower,
                    model.joint_limit_upper,
                    model.joint_limit_ke,
                    model.joint_limit_kd,
                    int(solver._joint_options.limit_damping),
                    model.joint_armature,
                    int(solver._joint_options.armature),
                    *(solver._joint_rate[kind][:2] if solver._joint_options.rates else (q_prev, q_prev)),
                    model.joint_effort_limit,
                    int(solver._joint_options.effort_limit),
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
                    solver._body_base,
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
                inputs=[solver._joint_rows[kind], model.joint_enabled, solver._joint_strength[kind]],
                outputs=[group.kappa],
                device=model.device,
            )
