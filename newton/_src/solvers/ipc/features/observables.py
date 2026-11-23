# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The solver observables SolverIPC writes (``SolverBase.observables``): its ``STATUS``, the Newton residual
(``SOLVER_RESIDUAL``) and its contacts and their forces (``CONTACT_F``).

Newton's ``SolverBase`` refuses an observable that a solver does not support. The hooks are those of ``features``
(its docstring).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from newton import Control, Model, ModelFlags, State
    from newton.solvers import SolverBase

    from ..joints import _JointDofs
    from ..solver_ipc import SolverIPC
    from . import FeatureOptions


def check(model: Model, options: FeatureOptions, at: str) -> None:
    """Nothing of the model."""


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
    """Nothing."""
    return False
