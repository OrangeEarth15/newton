# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The solver observables SolverIPC writes (``SolverBase.observables``): its ``STATUS``, the Newton residual
(``SOLVER_RESIDUAL``) and its contacts and their forces (``CONTACT_F``).

The solver's ``SUPPORTED_OBSERVABLES`` holds ``KINDS`` and ``STATUS``, and its ``supported_observables`` is
``supported(solver)``; Newton's ``SolverBase`` refuses an observable outside it. ``SolverIPC.observables`` calls
``allocate`` for every container it allocates. The hooks are those of ``features`` (its docstring).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from newton.solvers import SolverBase

if TYPE_CHECKING:
    from newton import Control, Model, ModelFlags, State

    from ..joints import _JointDofs
    from ..solver_ipc import SolverIPC
    from . import FeatureOptions

# the standard observable kinds SolverIPC writes, besides its own STATUS (SolverIPC.ObservableKind.STATUS)
KINDS = frozenset({SolverBase.ObservableKind.SOLVER_RESIDUAL, SolverBase.ObservableKind.CONTACT_F})


def supported(solver: SolverIPC) -> frozenset[str]:
    """The observable kinds ``solver`` writes: those of ``SolverIPC.SUPPORTED_OBSERVABLES`` (``KINDS`` and
    ``STATUS``), and with ``contact_constitution="al-ipc"`` ``STATUS`` only, since the AL-IPC frame ends on its own
    criterion, without a Newton residual, and its contact is an active set with multipliers, not the barrier and
    friction terms the contact output exports."""
    config = getattr(solver, "_config", None)
    if config is not None and config.contact_constitution == "al-ipc":
        return frozenset({solver.ObservableKind.STATUS})
    return solver.SUPPORTED_OBSERVABLES


def allocate(solver: SolverIPC, observables: SolverBase.Observables) -> None:
    """Nothing: what a container requests is allocated by ``SolverBase.observables`` and, for ``CONTACT_F``, by the
    solver. ``SolverIPC.observables`` calls this for every container it allocates, after them and outside any graph
    capture, with the container (``observables.is_requested`` gives what it asks for)."""


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
