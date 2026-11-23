# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The drives, springs, damping, armature, friction and limits of ball joints, and those of the joints libuipc holds by
no coordinate: free and distance joints.

libuipc's ball joint holds the anchors only, and it has no free or distance joint. SolverIPC refuses, on a degree of
freedom of a ball joint (and of a D6 joint that becomes one), joint friction; and on one of a ball, free or distance
joint (and of a D6 joint that becomes one of these), armature, drives, passive springs and damping, a target mode
other than NONE, and limits. These refusals of free and distance joints share their messages with those of ball
joints. The hooks are those of ``features`` (its docstring).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from newton import MAXVAL, JointTargetMode, JointType

if TYPE_CHECKING:
    from newton import Control, Model, ModelFlags, State
    from newton.solvers import SolverBase

    from ..joints import _JointDofs
    from ..solver_ipc import SolverIPC
    from . import FeatureOptions


def check(model: Model, options: FeatureOptions, at: str) -> None:
    """Nothing: the joints' degrees of freedom are checked through ``joint_refusal``."""


def joint_refusal(category: str, dofs: _JointDofs) -> tuple[int, str] | None:
    """The first degree of freedom refused in ``category``: ``"friction"``, a ``joint_friction`` other than 0 on a ball
    joint; ``"armature"``, a ``joint_armature`` other than 0, ``"drive"``, gains, a passive spring or damping or a
    target mode other than NONE, and ``"limit"``, a finite bound with a ``joint_limit_ke`` above 0, on a ball or free
    joint (a distance joint's bounds are its distance, not a limit)."""
    other = np.isin(dofs.kind, [int(JointType.BALL), int(JointType.FREE)])
    if category == "friction":
        friction = dofs.value("joint_friction")
        return dofs.refusal(
            np.flatnonzero((friction != 0.0) & (dofs.kind == int(JointType.BALL))),
            lambda d: f"joint_friction {float(friction[d])!r}",
            "set it to 0. libuipc's joints have no friction",
        )
    if category == "armature":
        armature = dofs.value("joint_armature")
        return dofs.refusal(
            np.flatnonzero((armature != 0.0) & other),
            lambda d: f"joint_armature {float(armature[d])!r}",
            "set it to 0. SolverIPC takes the armature of revolute and prismatic joints only, as the inertia of their "
            "one coordinate",
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
        return dofs.refusal(
            np.flatnonzero(other & driven),
            drive,
            "set the gains, stiffness and damping to 0 and the target mode to NONE, and apply forces through "
            "Control.joint_f, which acts on every joint. libuipc drives, springs and damps revolute and prismatic "
            "joints only",
        )
    if category == "limit":
        lo, hi = dofs.value("joint_limit_lower"), dofs.value("joint_limit_upper")
        limit_k = dofs.value("joint_limit_ke")
        limited = ((lo > -MAXVAL) | (hi < MAXVAL)) & (limit_k > 0.0)
        limited[dofs.qd_start[:-1][dofs.joint_type == int(JointType.DISTANCE)]] = False  # a distance joint's bounds
        return dofs.refusal(
            np.flatnonzero(other & limited),
            lambda d: f"the limits [{float(lo[d])!r}, {float(hi[d])!r}] with limit_ke {float(limit_k[d])!r}",
            "set limit_ke to 0, or the bounds to -MAXVAL and MAXVAL. libuipc limits revolute and prismatic joints only",
        )
    return None


def build(solver: SolverIPC, scene) -> None:
    """Nothing."""


def engine_inputs(solver: SolverIPC, inputs: dict) -> None:
    """Nothing."""


def prepare(solver: SolverIPC, state_in: State, control: Control, dt: float) -> None:
    """Nothing."""


def finish(solver: SolverIPC, state_out: State, observables: SolverBase.Observables | None) -> None:
    """Nothing."""


def notify(solver: SolverIPC, flags: ModelFlags | int) -> bool:
    """Nothing: the solver checks the joints again (``joint_refusal``) when a flag of their degrees of freedom is
    reported."""
    return False
