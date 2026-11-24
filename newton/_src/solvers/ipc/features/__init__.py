# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC's features beyond libuipc's own, one module each, and the points at which the solver calls them.

A feature is Newton input that libuipc has no term for: the damping of springs, attachments and elements (``damping``),
MuJoCo's spatial tendons (``tendons``), mimic joints, fixed tendons and joint friction (``joint_terms``), universal
joints and distance joints with a range (``universal``), the drives, springs, damping, armature, friction and limits of
ball joints (``ball``), rod joints (``rods``), the surface velocity of a mesh (``surface_velocity``), contact damping
(``contact_damping``) and the solver observables (``observables``). A module holds what SolverIPC does with its
feature, and while SolverIPC has no term for it, the refusal of it: each refusal keeps its message, and fires at the
point and in the order it fired before the modules held them, so that a model refused before is refused with the same
message.

The solver calls every module, in the order of ``FEATURES``, at these points; a module with nothing to do there does
nothing:

- ``check(model, options, at)``: raise ``NotImplementedError`` for what the module refuses of ``model`` at the check
  ``at`` names, among the solver's own checks: ``"tendons"`` (after the body variants, before the muscles),
  ``"constraints"`` (before MuJoCo's equality rows), ``"joints"`` (after the joints' own checks), ``"springs"`` and
  ``"attachments"`` (before the other checks of springs and of particle-body attachments) and ``"shapes"`` (the last
  check of the model), when the solver is built and when it builds its solve again; the constraints and the attachments
  also when ``notify_model_changed`` reports ``CONSTRAINT_PROPERTIES``; and ``"options"``, among the checks of the
  constructor's options. ``options`` is ``FeatureOptions``.
- ``joint_refusal(category, dofs)``: what the module refuses of the joints' degrees of freedom in ``category``, as
  ``(dof, message)`` for the first such degree of freedom, or None. ``joints._check_joints`` asks for the categories
  ``"friction"``, then ``"armature"``, ``"drive"`` and ``"limit"``, among its own checks, and raises the refusal of
  the lowest degree of freedom of all modules. ``dofs`` is ``joints._JointDofs``: per degree of freedom its joint and
  the type that joint maps onto, the model's per-DOF arrays, and the message of a refusal.
- ``build(solver, scene)``: when the solve is built, after the solver has laid its own content out in ``scene``
  (libuipc-warp's ``ArrayScene``) and before the scene's layout and the engine are made from it.
- ``engine_inputs(solver, inputs)``: every time an engine is made from the scene, when the solve is built and when an
  eager step or ``grow()`` enlarges the contact lists, with the engine's keyword arguments (the dict that
  ``ArrayScene.engine_inputs`` returns, the solver's external accelerations appended to ``inputs["elements"]``) before
  the engine is made from them: a module adds its element groups to ``inputs["elements"]``, and copies what it wrote
  into the groups of an engine before into those of the new one.
- ``prepare(solver, state_in, control, dt)``: in every step, after the solver has written the step's inputs (external
  forces, joint terms, stitch and body stiffness) and before the frame is enqueued; inside a graph capture too, so a
  module only enqueues device work there. ``solver._observables`` is then the observables container the step was
  given (None for none).
- ``finish(solver, state_out, observables)``: in every step whose ``state_out`` is written (a step that fails in some
  independent worlds alone among them), after ``state_out`` and the observables.
- ``notify(solver, flags)``: in ``notify_model_changed``, after the solver's own checks and before it decides whether
  to build the solve again: returns whether the solve must be built again.
- ``reset(solver, world_mask)``, of the modules that keep a state across steps: in ``SolverIPC.reset`` once the solve
  is built, with its ``world_mask``; on the device only, since a reset may be captured.

The universal module is also offered the joints that ``joints._map_joints`` cannot map onto a libuipc joint
(``universal.take_d6``, ``universal.take_distance``), the rods module the rod joints (``rods.take_rods``), and the
observables module gives the supported observables (``observables.supported``) and allocates, in
``SolverIPC.observables`` after Newton's allocation and outside any graph capture, what the kinds a container requests
need of it (``observables.allocate``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import ball, contact_damping, damping, joint_terms, observables, rods, surface_velocity, tendons, universal

if TYPE_CHECKING:
    import warp as wp

    from newton import Control, Model, ModelFlags, State
    from newton.solvers import SolverBase

    from ..joints import _JointDofs
    from ..solver_ipc import SolverIPC

# the feature modules, in the order the solver calls them
FEATURES = (damping, tendons, joint_terms, universal, ball, rods, surface_velocity, contact_damping, observables)


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


def build(solver: SolverIPC, scene) -> None:
    """Every module's work when the solve is built (``scene``: the solver's ``ArrayScene``)."""
    for feature in FEATURES:
        feature.build(solver, scene)


def engine_inputs(solver: SolverIPC, inputs: dict) -> None:
    """Every module's additions to the inputs of an engine made from the scene."""
    for feature in FEATURES:
        feature.engine_inputs(solver, inputs)


def prepare(solver: SolverIPC, state_in: State, control: Control, dt: float) -> None:
    """Every module's work of a step before its frame is enqueued."""
    for feature in FEATURES:
        feature.prepare(solver, state_in, control, dt)


def finish(solver: SolverIPC, state_out: State, observables: SolverBase.Observables | None) -> None:
    """Every module's work of a step after ``state_out`` and the observables are written."""
    for feature in FEATURES:
        feature.finish(solver, state_out, observables)


def notify(solver: SolverIPC, flags: ModelFlags | int) -> bool:
    """Every module's response to ``notify_model_changed(flags)``: whether one of them needs the solve built again."""
    rebuild = False
    for feature in FEATURES:
        rebuild |= bool(feature.notify(solver, flags))
    return rebuild


def reset(solver: SolverIPC, world_mask: wp.array | None) -> None:
    """The work of the modules that keep a state across steps in ``SolverIPC.reset(state, world_mask)``."""
    for feature in FEATURES:
        hook = getattr(feature, "reset", None)
        if hook is not None:
            hook(solver, world_mask)
