# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The damping of springs, particle-body attachments, triangles, tetrahedra and bending edges, and drag and lift.

libuipc has no damping element. SolverIPC refuses a damped spring and a damped particle-body attachment, and does not
read the damping of triangles, tetrahedra and bending edges, nor drag and lift (:ref:`ipc-not-read`). The hooks are
those of ``features`` (its docstring).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..scene import _own_elements

if TYPE_CHECKING:
    from newton import Control, Model, ModelFlags, State
    from newton.solvers import SolverBase

    from ..joints import _JointDofs
    from ..solver_ipc import SolverIPC
    from . import FeatureOptions


def check(model: Model, options: FeatureOptions, at: str) -> None:
    """``NotImplementedError`` for the first spring the solver simulates (``_own_elements``) with a damping
    (``"springs"``), and for the first particle-body attachment with a damping (``"attachments"``): libuipc's springs
    and its stitch have none, and an implicit damping cannot be folded into an attachment's stitch the way the joint
    drives fold theirs, since the stitch has no rest offset to carry it."""
    if at == "springs" and model.spring_count:
        values = model.spring_damping.numpy()
        found = np.flatnonzero((values != 0.0) & _own_elements(model)[2])
        if len(found):
            k = int(found[0])
            raise NotImplementedError(
                f"SolverIPC: spring {k} has spring_damping {float(values[k])!r}; set it to 0. libuipc's springs have "
                "no damping."
            )
    if at == "attachments" and model.attachment_particle_body_count:
        values = model.attachment_particle_body_damping.numpy()
        found = np.flatnonzero(values != 0.0)
        if len(found):
            k = int(found[0])
            raise NotImplementedError(
                f"SolverIPC: particle-body attachment {k} has damping {float(values[k])!r}; set it to 0. libuipc's "
                "stitch has no damping."
            )


def joint_refusal(category: str, dofs: _JointDofs) -> tuple[int, str] | None:
    """Nothing of the joints."""
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
    """Nothing: the solver checks the attachments again (``check``) when ``CONSTRAINT_PROPERTIES`` is reported, and
    builds the solve again, which checks the springs, when their damping changes."""
    return False
