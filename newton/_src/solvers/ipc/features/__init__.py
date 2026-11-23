# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC's features beyond libuipc's own, one module each, and the points at which the solver calls them.

A feature is Newton input that libuipc has no term for: the damping of springs, attachments and elements
(``damping``), mimic joints, tendons and joint friction (``joint_terms``), universal joints and distance joints with a
range (``universal``), the drives, springs, damping, armature, friction and limits of ball joints (``ball``), the
surface velocity of a mesh (``surface_velocity``), contact damping (``contact_damping``) and the solver observables
(``observables``). A module holds what SolverIPC does with its feature, and while SolverIPC has no term for it, the
refusal of it: each refusal keeps its message, and fires at the point and in the order it fired before the modules
held them, so that a model refused before is refused with the same message.

The solver calls every module, in the order of ``FEATURES``, at these points; a module with nothing to do there does
nothing:

- ``check(model, options, at)``: raise ``NotImplementedError`` for what the module refuses of ``model`` at the check
  ``at`` names, among the solver's own checks: ``"tendons"`` (after the body variants, before the muscles),
  ``"constraints"`` (before MuJoCo's equality rows), ``"springs"`` and ``"attachments"`` (before the other checks of
  springs and of particle-body attachments) and ``"shapes"`` (the last check of the model), when the solver is built
  and when it builds its solve again; the constraints and the attachments also when ``notify_model_changed`` reports
  ``CONSTRAINT_PROPERTIES``; and ``"options"``, among the checks of the constructor's options. ``options`` is
  ``FeatureOptions``.
- ``joint_refusal(category, dofs)``: what the module refuses of the joints' degrees of freedom in ``category``, as
  ``(dof, message)`` for the first such degree of freedom, or None. ``joints._check_joints`` asks for the categories
  ``"friction"``, then ``"armature"``, ``"drive"`` and ``"limit"``, among its own checks, and raises the refusal of
  the lowest degree of freedom of all modules. ``dofs`` is ``joints._JointDofs``: per degree of freedom its joint and
  the type that joint maps onto, the model's per-DOF arrays, and the message of a refusal.

The universal module is also offered the joints that ``joints._map_joints`` cannot map onto a libuipc joint
(``universal.take_d6``, ``universal.take_distance``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import ball, contact_damping, damping, joint_terms, observables, surface_velocity, universal

if TYPE_CHECKING:
    from newton import Model

    from ..joints import _JointDofs

# the feature modules, in the order the solver calls them
FEATURES = (damping, joint_terms, universal, ball, surface_velocity, contact_damping, observables)


@dataclass(frozen=True)
class FeatureOptions:
    """The constructor options of :class:`~newton.solvers.SolverIPC` that the feature modules' checks read."""

    contact_damping: float | str | None = None


def check(model: Model, options: FeatureOptions, at: str) -> None:
    """Every module's refusals of ``model`` at the check ``at``, in the order of ``FEATURES``."""
    for feature in FEATURES:
        feature.check(model, options, at)


def joint_refusal(category: str, dofs: _JointDofs) -> tuple[int, str] | None:
    """Of the modules' refusals of the joints' degrees of freedom in ``category``, the one of the lowest degree of
    freedom, as ``(dof, message)``, or None."""
    found = [refusal for feature in FEATURES if (refusal := feature.joint_refusal(category, dofs)) is not None]
    return min(found, key=lambda refusal: refusal[0]) if found else None
