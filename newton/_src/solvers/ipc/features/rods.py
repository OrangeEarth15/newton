# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rod joints: the stretch, shear, bend and twist of Newton's rods and cables and their damping.

:meth:`newton.ModelBuilder.add_rod` makes a rod of capsule bodies joined by joints of type ROD
(:meth:`~newton.ModelBuilder.add_joint_rod`). libuipc's rods are finite elements of points without a twist;
libuipc-warp's rod joints (``uipc_warp.ext.rods.RodJoints``) act between two affine bodies with Newton's strain:

- the child anchor's offset in the parent's anchor frame, ``joint_target_ke`` of the DOFs ``shear_x``, ``shear_y`` and
  ``stretch`` its springs and ``joint_target_kd`` their damping, the translation of ``joint_rest_q`` its rest;
- the bend and twist of the child's anchor frame relative to the parent's, Newton's discrete elastic rod strain without
  its cap, the DOFs ``bend_x``, ``bend_y`` and ``twist`` their stiffness and damping, the rotation of ``joint_rest_q``
  its rest.

``joints._map_joints`` offers the rod joints here (``take_rods``) and maps them onto no libuipc joint: its kind is ROD,
stored as a FREE joint, so that ``joint_q`` and ``joint_qd`` are written as a free joint's (the child's anchor frame
in the parent's) and ``joint_f`` is a free joint's world wrench at the child's centre of mass, as SolverVBD applies it.
Every step writes the joints' inputs from the model on the device (``prepare``), so a captured step takes new gains,
rests and enabled flags at its next replay. The joints give ``BODY_PARENT_F`` their terms (``parent_terms``).

SolverIPC refuses, on a rod's degree of freedom, joint friction, armature, a ``joint_stiffness`` or ``joint_damping``
other than 0, a target mode other than NONE, a negative gain and a finite limit, as SolverKamino does. The other hooks
are those of ``features`` (its docstring).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from newton import MAXVAL, JointTargetMode, JointType, ModelFlags

from ..scene import _quat_to_matrix_np

if TYPE_CHECKING:
    from newton import Control, Model, State
    from newton.solvers import SolverBase

    from ..joints import _JointDofs
    from ..solver_ipc import SolverIPC
    from . import FeatureOptions

# a rod joint's degrees of freedom, in the order of add_joint_rod's axes
_DOFS = ("shear_x", "shear_y", "stretch", "bend_x", "bend_y", "twist")
_REMEDY = "A rod's stiffness and damping are its joint_target_ke and joint_target_kd, its rest joint_rest_q"


def take_rods(model: Model, joints: np.ndarray) -> None:
    """Take the rod joints ``joints``, or ``NotImplementedError`` for the first that is not of the layout
    :meth:`~newton.ModelBuilder.add_joint_rod` makes (three linear and three angular axes, X, Y and Z in order), which
    SolverVBD refuses too."""
    if not len(joints):
        return
    dim = model.joint_dof_dim.numpy()[joints]
    start = model.joint_qd_start.numpy()[joints]
    axis = model.joint_axis.numpy().astype(np.float64)
    canonical = np.concatenate([np.eye(3), np.eye(3)])
    for j, (lin, ang), s in zip(joints, dim, start, strict=True):
        if (int(lin), int(ang)) != (3, 3) or np.abs(axis[s : s + 6] - canonical).max() > 1.0e-6:
            raise NotImplementedError(
                f"SolverIPC: the ROD joint {int(j)} does not have three linear and three angular axes X, Y and Z in "
                "order; build it with ModelBuilder.add_joint_rod(), whose canonical layout a rod joint requires."
            )


def check(model: Model, options: FeatureOptions, at: str) -> None:
    """Nothing: the rod joints' degrees of freedom are checked through ``joint_refusal`` and their layout by
    ``take_rods``."""


def joint_refusal(category: str, dofs: _JointDofs) -> tuple[int, str] | None:
    """The first degree of freedom of a rod joint refused in ``category``: ``"friction"``, a ``joint_friction`` other
    than 0; ``"armature"``, a ``joint_armature`` other than 0; ``"drive"``, a ``joint_stiffness`` or ``joint_damping``
    other than 0, a target mode other than NONE, or a ``joint_target_ke`` or ``joint_target_kd`` that is negative or
    not a number; ``"limit"``, a finite bound with a ``joint_limit_ke`` above 0."""
    rod = dofs.kind == int(JointType.ROD)
    if not rod.any():
        return None
    takes = "SolverIPC's rod joints take the rod's material response only, as SolverKamino's do"
    if category in ("friction", "armature"):
        name = f"joint_{category}"
        value = dofs.value(name)
        return dofs.refusal(
            np.flatnonzero(rod & (value != 0.0)), lambda d: f"{name} {float(value[d])!r}", f"set it to 0. {takes}"
        )
    if category == "drive":
        found = []
        for name in ("joint_stiffness", "joint_damping"):
            value = dofs.value(name)
            found.append(
                dofs.refusal(
                    np.flatnonzero(rod & (value != 0.0)),
                    lambda d, name=name, value=value: f"{name} {float(value[d])!r}",
                    f"set it to 0. {_REMEDY}",
                )
            )
        model = dofs.model
        if model.joint_target_mode is not None:
            mode = model.joint_target_mode.numpy()
            found.append(
                dofs.refusal(
                    np.flatnonzero(rod & (mode != int(JointTargetMode.NONE))),
                    lambda d: f"the target mode {JointTargetMode(int(mode[d])).name}",
                    f"set it to NONE. A rod joint has no drive. {_REMEDY}",
                )
            )
        for name in ("joint_target_ke", "joint_target_kd"):
            value = dofs.value(name)
            found.append(
                dofs.refusal(
                    np.flatnonzero(rod & ~(value >= 0.0)),
                    lambda d, name=name, value=value: f"{name} {float(value[d])!r}",
                    "make it 0 or positive. A rod's stiffness and damping are not negative",
                )
            )
        found = [r for r in found if r is not None]
        return min(found, key=lambda r: r[0]) if found else None
    if category == "limit":
        lo, hi = dofs.value("joint_limit_lower"), dofs.value("joint_limit_upper")
        limit_k = dofs.value("joint_limit_ke")
        limited = ((lo > -MAXVAL) | (hi < MAXVAL)) & (limit_k > 0.0)
        return dofs.refusal(
            np.flatnonzero(rod & limited),
            lambda d: f"the limits [{float(lo[d])!r}, {float(hi[d])!r}] with limit_ke {float(limit_k[d])!r}",
            f"set limit_ke to 0, or the bounds to -MAXVAL and MAXVAL. {takes}",
        )
    return None


def _rod_joints(model: Model) -> np.ndarray:
    """The joints of type ROD of ``model``, in their order."""
    if model.joint_count == 0:
        return np.zeros(0, np.int64)
    return np.flatnonzero(model.joint_type.numpy() == int(JointType.ROD))


def build(solver: SolverIPC, scene) -> None:
    """Plan the rod joints: per joint its parent (or the fixed body that stands for the world) and child, its anchor
    frames' rotations and anchors in the bodies' frames, from ``joint_X_p`` and ``joint_X_c``."""
    model = solver.model
    solver._rods = None  # the RodJoints of the last engine made from the scene
    solver._rod_joints = joints = _rod_joints(model)
    if not len(joints):
        return
    try:
        import uipc_warp.ext.rods  # noqa: F401, PLC0415 (libuipc-warp is imported lazily)
    except ImportError as e:
        raise ImportError(
            "SolverIPC: rod joints need libuipc-warp's rod joints (uipc_warp.ext.rods); install a libuipc-warp that "
            "has them, or use SolverVBD or SolverKamino for rods."
        ) from e
    parent, child = model.joint_parent.numpy()[joints], model.joint_child.numpy()[joints]
    solver._rod_bodies = np.stack([np.where(parent >= 0, parent, solver._world_body), child], axis=1)
    X_p, X_c = model.joint_X_p.numpy().astype(np.float64), model.joint_X_c.numpy().astype(np.float64)
    solver._rod_frames = tuple(np.array([_quat_to_matrix_np(q) for q in X[joints, 3:7]]) for X in (X_p, X_c))
    solver._rod_anchors = (X_p[joints, :3], X_c[joints, :3])
    solver._rod_rows = wp.array(joints.astype(np.int32), dtype=wp.int32, device=model.device)


@wp.func
def _rotation(q: wp.array[wp.float32], i: int) -> wp.mat33d:
    """The rotation of the quaternion ``q[i : i + 4]`` (x, y, z, w), normalized, in float64; the identity for one of
    length 0."""
    x = wp.float64(q[i])
    y = wp.float64(q[i + 1])
    z = wp.float64(q[i + 2])
    w = wp.float64(q[i + 3])
    length = wp.sqrt(x * x + y * y + z * z + w * w)
    R = wp.identity(n=3, dtype=wp.float64)
    if length > wp.float64(1.0e-20):
        x = x / length
        y = y / length
        z = z / length
        w = w / length
        one = wp.float64(1.0)
        two = wp.float64(2.0)
        R = wp.mat33d(
            one - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w),
            two * (x * y + z * w), one - two * (x * x + z * z), two * (y * z - x * w),
            two * (x * z - y * w), two * (y * z + x * w), one - two * (x * x + y * y),
        )  # fmt: skip
    return R


@wp.kernel
def _rod_inputs(
    joints: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_enabled: wp.array[wp.bool],
    target_ke: wp.array[wp.float32],
    target_kd: wp.array[wp.float32],
    rest_q: wp.array[wp.float32],
    on: wp.array[wp.int32],
    stretch: wp.array[wp.float64],
    shear: wp.array[wp.vec2d],
    bend: wp.array[wp.vec2d],
    twist: wp.array[wp.float64],
    stretch_damping: wp.array[wp.float64],
    shear_damping: wp.array[wp.vec2d],
    bend_damping: wp.array[wp.vec2d],
    twist_damping: wp.array[wp.float64],
    rest_translation: wp.array[wp.vec3d],
    rest_rotation: wp.array[wp.mat33d],
):
    """The inputs of rod joint ``joints[k]`` for the next step, from Newton's model: per degree of freedom ``[shear_x,
    shear_y, stretch, bend_x, bend_y, twist]`` its ``joint_target_ke`` and ``joint_target_kd``, the translation and the
    rotation of ``joint_rest_q``, and whether it is enabled."""
    k = wp.tid()
    j = joints[k]
    d = joint_qd_start[j]
    i = joint_q_start[j]
    on[k] = wp.where(joint_enabled[j], 1, 0)
    shear[k] = wp.vec2d(wp.float64(target_ke[d]), wp.float64(target_ke[d + 1]))
    stretch[k] = wp.float64(target_ke[d + 2])
    bend[k] = wp.vec2d(wp.float64(target_ke[d + 3]), wp.float64(target_ke[d + 4]))
    twist[k] = wp.float64(target_ke[d + 5])
    shear_damping[k] = wp.vec2d(wp.float64(target_kd[d]), wp.float64(target_kd[d + 1]))
    stretch_damping[k] = wp.float64(target_kd[d + 2])
    bend_damping[k] = wp.vec2d(wp.float64(target_kd[d + 3]), wp.float64(target_kd[d + 4]))
    twist_damping[k] = wp.float64(target_kd[d + 5])
    rest_translation[k] = wp.vec3d(wp.float64(rest_q[i]), wp.float64(rest_q[i + 1]), wp.float64(rest_q[i + 2]))
    rest_rotation[k] = _rotation(rest_q, i + 3)


def _write_inputs(solver: SolverIPC) -> None:
    """The rod joints' inputs of the next step, on the device (``_rod_inputs``)."""
    rods, model = solver._rods, solver.model
    wp.launch(
        _rod_inputs,
        rods.n,
        inputs=[
            solver._rod_rows,
            model.joint_q_start,
            model.joint_qd_start,
            model.joint_enabled,
            model.joint_target_ke,
            model.joint_target_kd,
            model.joint_rest_q,
        ],
        outputs=[
            rods.on, rods.stretch, rods.shear, rods.bend, rods.twist, rods.stretch_damping, rods.shear_damping,
            rods.bend_damping, rods.twist_damping, rods.rest_translation, rods.rest_rotation,
        ],
        device=model.device,
    )  # fmt: skip


def engine_inputs(solver: SolverIPC, inputs: dict) -> None:
    """The rod joints of an engine made from the scene (``uipc_warp.ext.rods.RodJoints``), their two element groups
    appended to ``inputs["elements"]``, with the model's inputs."""
    if not len(solver._rod_joints):
        return
    from uipc_warp.ext.rods import RodJoints  # noqa: PLC0415 (libuipc-warp is imported lazily)

    solver._rods = RodJoints(
        solver._scene.layout(), solver._rod_bodies, *solver._rod_frames, *solver._rod_anchors, solver.model.device
    )
    inputs["elements"] += solver._rods.groups
    _write_inputs(solver)


def prepare(solver: SolverIPC, state_in: State, control: Control, dt: float) -> None:
    """The rod joints' inputs of the step (``_rod_inputs``), on the device."""
    if solver._rods is not None:
        _write_inputs(solver)


def finish(solver: SolverIPC, state_out: State, observables: SolverBase.Observables | None) -> None:
    """Nothing: ``joint_q`` and ``joint_qd`` of a rod joint are written as a free joint's."""


def notify(solver: SolverIPC, flags: ModelFlags | int) -> bool:
    """Nothing to build again: the gains, rests and enabled flags are read at every step, and a change
    of the joints' types, bodies or frames builds the solve again by the solver's own check."""
    return False


def parent_terms(solver: SolverIPC, engine) -> list:
    """The rod joints' two element groups of ``engine`` and the joint of each of their elements, for ``BODY_PARENT_F``
    (``observables``): a rod joint's terms are the wrench it passes to its child."""
    rods = solver._rods
    if rods is None:
        return []
    return [(group, solver._rod_joints) for group in rods.groups]
