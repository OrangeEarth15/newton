# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Joints that lock what no libuipc joint locks: D6 joints of other axes than libuipc's joints (two angular axes, a
universal joint, among them), and distance joints whose distance has a range or one bound.

``joints._map_joints`` offers every such joint here, in the order of the joints (``take_d6``, ``take_distance``): a
call that returns takes the joint, and ``joints.py`` then builds, drives and stores nothing of it; a call that raises
refuses it. SolverIPC takes none of them. The other hooks are those of ``features`` (its docstring).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from newton import Model

    from ..joints import _JointDofs
    from . import FeatureOptions


def take_d6(model: Model, joint: int, linear: int, angular: int) -> None:
    """Take the D6 joint ``joint`` of ``linear`` linear and ``angular`` angular axes, which lock what no libuipc joint
    locks, or raise ``NotImplementedError``: SolverIPC takes none."""
    raise NotImplementedError(
        f"SolverIPC: the D6 joint {joint} has {linear} linear and {angular} angular axes; give it no axis, one axis, "
        f"three angular axes, or three linear and three angular axes. Its axes lock {3 - linear} translations and "
        f"{3 - angular} rotations, which no libuipc joint locks."
    )


def take_distance(model: Model, joint: int, lower, upper) -> None:
    """Take the distance joint ``joint``, whose distance has the range ``[lower, upper]`` (a negative bound is none),
    or raise ``NotImplementedError``: SolverIPC takes none."""
    raise NotImplementedError(
        f"SolverIPC: the DISTANCE joint {joint} bounds its distance to [{float(lower)!r}, {float(upper)!r}] (a "
        "negative bound is none); give both bounds one value, or make both negative. libuipc's stitch, which holds the "
        "distance, has one rest length and no range."
    )


def check(model: Model, options: FeatureOptions, at: str) -> None:
    """Nothing: the joints are offered through ``take_d6`` and ``take_distance``."""


def joint_refusal(category: str, dofs: _JointDofs) -> tuple[int, str] | None:
    """Nothing: the module takes no joint."""
    return None
