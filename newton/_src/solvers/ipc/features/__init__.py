# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC's features beyond libuipc's own, one module each, and the points at which the solver calls them.

A feature is Newton input that libuipc has no term for. Each module extends one of the solver's core modules, whose
content it completes with libuipc-warp's extensions:

- ``damping``: the damping of springs, particle-body attachments, triangles, tetrahedra and bending edges, and drag and
  lift; extends ``scene``, which lays out those elements;
- ``joint_terms``: mimic joints, MuJoCo's fixed tendons and joint friction, as terms on joint coordinates; extends
  ``joints``;
- ``universal``: universal joints and distance joints with a range; extends ``joints``, which offers it the joints it
  cannot map onto a libuipc joint;
- ``ball``: the drives, springs, damping, armature and friction of ball joints, and what it refuses of ball, free and
  distance joints; extends ``joints``;
- ``rods``: rod joints; extends ``joints``, which offers it the rod joints;
- ``tendons``: MuJoCo's spatial tendons, paths between the bodies' sites; extends ``joints``, as terms between bodies;
- ``surface_velocity``: the surface velocity of a mesh, which friction measures slip against; extends ``contacts``;
- ``contact_damping``: a viscous force on the normal motion of surfaces in contact; extends ``contacts``.

A module holds what SolverIPC does with its feature, and what it refuses of it.

The solver calls every module that defines a hook, in the order of ``FEATURES``, at these points; every hook is
optional, and a module without it has nothing to do there:

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
- ``reset(solver, world_mask)``: in ``SolverIPC.reset`` once the solve is built, with its ``world_mask``, for a module
  that keeps a state across steps; on the device only, since a reset may be captured.
- ``parent_terms(solver, engine)``: when ``BODY_PARENT_F`` reads the joints' terms of ``engine`` (``observables``): a
  module whose element groups are terms of a joint returns them, as ``(group, the joint of every element)`` pairs
  (-1 for an element of no joint).

A module keeps what it needs from one hook to the next in its field of ``solver._features`` (``FeatureState``), which
its ``build`` sets; nothing else of the module lives on the solver.

The universal module is also offered the joints that ``joints._map_joints`` cannot map onto a libuipc joint
(``universal.take_d6``, ``universal.take_distance``), and the rods module the rod joints (``rods.take_rods``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import ball, contact_damping, damping, joint_terms, rods, surface_velocity, tendons, universal

if TYPE_CHECKING:
    import warp as wp

    from newton import Control, Model, ModelFlags, State
    from newton.solvers import SolverBase

    from ..joints import _JointDofs
    from ..solver_ipc import SolverIPC

# the feature modules, in the order the solver calls them
FEATURES = (damping, tendons, joint_terms, universal, ball, rods, surface_velocity, contact_damping)


@dataclass
class FeatureState:
    """What the feature modules keep of a solver (``SolverIPC._features``), one field per module: None until the
    module's ``build`` makes it, and for a module with nothing to keep in that solve."""

    damping: damping._Plan | None = None
    tendons: tendons._State | None = None
    joint_terms: joint_terms._State | None = None
    universal: universal._State | None = None
    ball: ball._State | None = None
    rods: rods._State | None = None
    surface_velocity: surface_velocity._Belts | None = None
    contact_damping: contact_damping._Damping | None = None


@dataclass(frozen=True)
class FeatureOptions:
    """The constructor options of :class:`~newton.solvers.SolverIPC` that the feature modules' checks read."""

    contact_damping: float | str | None = None


def _hooks(name: str) -> list:
    """The modules' hook ``name``, of the modules that define it, in the order of ``FEATURES``."""
    return [hook for feature in FEATURES if (hook := getattr(feature, name, None)) is not None]


def check(model: Model, options: FeatureOptions, at: str) -> None:
    """Every module's refusals of ``model`` at the check ``at``, in the order of ``FEATURES``."""
    for hook in _hooks("check"):
        hook(model, options, at)


def joint_refusal(category: str, dofs: _JointDofs) -> tuple[int, str] | None:
    """Of the modules' refusals of the joints' degrees of freedom in ``category``, the one of the lowest degree of
    freedom, as ``(dof, message)``, or None."""
    found = [refusal for hook in _hooks("joint_refusal") if (refusal := hook(category, dofs)) is not None]
    return min(found, key=lambda refusal: refusal[0]) if found else None


def build(solver: SolverIPC, scene) -> None:
    """Every module's work when the solve is built (``scene``: the solver's ``ArrayScene``)."""
    for hook in _hooks("build"):
        hook(solver, scene)


def engine_inputs(solver: SolverIPC, inputs: dict) -> None:
    """Every module's additions to the inputs of an engine made from the scene."""
    for hook in _hooks("engine_inputs"):
        hook(solver, inputs)


def prepare(solver: SolverIPC, state_in: State, control: Control, dt: float) -> None:
    """Every module's work of a step before its frame is enqueued."""
    for hook in _hooks("prepare"):
        hook(solver, state_in, control, dt)


def finish(solver: SolverIPC, state_out: State, observables: SolverBase.Observables | None) -> None:
    """Every module's work of a step after ``state_out`` and the observables are written."""
    for hook in _hooks("finish"):
        hook(solver, state_out, observables)


def notify(solver: SolverIPC, flags: ModelFlags | int) -> bool:
    """Every module's response to ``notify_model_changed(flags)``: whether one of them needs the solve built again."""
    rebuild = False
    for hook in _hooks("notify"):
        rebuild |= bool(hook(solver, flags))
    return rebuild


def reset(solver: SolverIPC, world_mask: wp.array | None) -> None:
    """The work of the modules that keep a state across steps in ``SolverIPC.reset(state, world_mask)``."""
    for hook in _hooks("reset"):
        hook(solver, world_mask)


def parent_terms(solver: SolverIPC, engine) -> list:
    """Every module's joint terms of ``engine`` for ``BODY_PARENT_F``, as ``(element group, the joint of every
    element)`` pairs, in the order of ``FEATURES``."""
    return [term for hook in _hooks("parent_terms") for term in hook(solver, engine)]
