# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Terms on joint coordinates: mimic joints, tendons, and the friction of revolute, prismatic and free joints.

libuipc has none of them. SolverIPC refuses a model with tendons (Newton's own and MuJoCo's) or mimic constraints, and
joint friction on a degree of freedom of a revolute, prismatic, free or distance joint (and of a D6 joint that becomes
one of these); the ball module refuses the friction of ball joints. The hooks are those of ``features`` (its
docstring).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from newton import JointType

if TYPE_CHECKING:
    from newton import Model

    from ..joints import _JointDofs
    from . import FeatureOptions


def check(model: Model, options: FeatureOptions, at: str) -> None:
    """``NotImplementedError`` for the tendons of the model, Newton's and MuJoCo's (``"tendons"``), and for its mimic
    constraints, the rows of ``constraint_mimic_*`` and the joints of ``joint_mimic_joint`` (``"constraints"``)."""
    if at == "tendons":
        count = model.tendon_count + model.custom_frequency_counts.get("mujoco:tendon", 0)
        if count:
            raise NotImplementedError(
                f"SolverIPC: the model has {count} tendons; remove them. SolverIPC does not simulate tendons."
            )
    if at == "constraints":
        mimic = int(model.constraint_mimic_count)
        if model.joint_count and model.joint_mimic_joint is not None:
            mimic += int(np.count_nonzero(model.joint_mimic_joint.numpy() >= 0))
        if mimic:
            raise NotImplementedError(
                f"SolverIPC: the model has {mimic} mimic constraints; remove them. SolverIPC does not enforce mimic "
                "constraints."
            )


def joint_refusal(category: str, dofs: _JointDofs) -> tuple[int, str] | None:
    """``"friction"``: the first degree of freedom of a joint that maps onto a revolute, prismatic or free joint with
    a ``joint_friction`` other than 0."""
    if category != "friction":
        return None
    friction = dofs.value("joint_friction")
    mine = np.isin(dofs.kind, [int(t) for t in (JointType.REVOLUTE, JointType.PRISMATIC, JointType.FREE)])
    return dofs.refusal(
        np.flatnonzero((friction != 0.0) & mine),
        lambda d: f"joint_friction {float(friction[d])!r}",
        "set it to 0. libuipc's joints have no friction",
    )
