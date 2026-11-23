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
    from newton import Model

    from ..joints import _JointDofs
    from . import FeatureOptions


def check(model: Model, options: FeatureOptions, at: str) -> None:
    """Nothing of the model."""


def joint_refusal(category: str, dofs: _JointDofs) -> tuple[int, str] | None:
    """Nothing of the joints."""
    return None
