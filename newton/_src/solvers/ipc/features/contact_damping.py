# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact damping: a damping of the motion of surfaces in contact along their normal (the constructor's
``contact_damping``).

libuipc's contact has no damping term: its barrier and friction are its only forces. SolverIPC takes
``contact_damping=None``, which adds none, and refuses any other value. The hooks are those of ``features`` (its
docstring).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from newton import Model

    from ..joints import _JointDofs
    from . import FeatureOptions


def check(model: Model, options: FeatureOptions, at: str) -> None:
    """``NotImplementedError`` for a ``contact_damping`` other than None (``"options"``)."""
    if at == "options" and options.contact_damping is not None:
        raise NotImplementedError(
            f"SolverIPC: contact_damping={options.contact_damping!r} is not offered yet: the solver's contact has no "
            "damping term. Leave contact_damping at None, its default."
        )


def joint_refusal(category: str, dofs: _JointDofs) -> tuple[int, str] | None:
    """Nothing of the joints."""
    return None
