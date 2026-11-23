# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Incremental Potential Contact (IPC) solver backed by libuipc-warp."""

from __future__ import annotations

import gc
import re
import warnings
from dataclasses import dataclass
from enum import IntFlag
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import CouplingInterface

from newton import Contacts, Control, JointType, Model, ModelFlags, NewtonWarning, State, StateFlags
from newton.solvers import SolverBase

from .contacts import (
    _CONTACT_ROW_BLOCK,
    _check_al_ipc,
    _check_surface_velocity,
    _contact_elements,
    _contact_records,
    _make_contact_export,
    _unread_materials,
    _write_contacts,
)
from .joints import (
    _add_joints,
    _check_joints,
    _distance_penalties,
    _joint_inputs_now,
    _joint_strengths,
    _JointOptions,
    _load_joints,
    _map_joints,
    _plan_distance_joints,
    _shift_joint_history,
    _warn_of_velocity_limits,
    _write_joint_controls,
)
from .kernels import (
    _aim_kinematic_bodies,
    _any_selected,
    _body_accelerations,
    _copy_status,
    _d6_joint_wrenches,
    _forget_kinematic_written,
    _gravity_changed,
    _joint_wrenches,
    _keep_status,
    _load_bodies,
    _load_particles,
    _nonzero_activations,
    _particle_accelerations,
    _proxy_particle_forces,
    _proxy_wrenches,
    _release_held,
    _store_bodies,
    _store_body_momentum,
    _store_d6_joints,
    _store_joints,
    _store_particles,
    _subscenes_selected,
)
from .scene import (
    _D_HAT,
    _add_cloth,
    _add_particles_and_tets,
    _add_shapes_and_bodies,
    _add_springs,
    _add_stitches,
    _aiming_bodies,
    _attachment_penalties,
    _body_mass_moments,
    _check_attachments,
    _check_independent_content,
    _check_springs,
    _collect_planes,
    _default_d_hat,
    _element_inputs_now,
    _engine_dyadic,
    _find_body_stiffness,
    _mass_inverse,
    _ortho_ratios,
    _own_elements,
    _plan_attachments,
    _proxy_bodies,
    _proxy_particles,
    _scene_inputs_now,
    _spreads,
    _subscene_of_shape,
    _vertex_gravity,
    _world_gravity,
    _Worlds,
    _write_body_stiffness,
    _write_stitch_stiffness,
)
from .start_check import _check_start

# typing.override from Python 3.12, typing_extensions' on Python 3.11, else no check, as Newton takes it
if TYPE_CHECKING:
    from typing_extensions import override
else:
    try:
        from typing import override
    except ImportError:
        try:
            from typing_extensions import override
        except ImportError:

            def override(method, /):
                return method


# The oldest libuipc-warp SolverIPC takes, its development builds counted as their release ("0.4.4.dev0" as 0.4.4), and
# where to install it from
_UIPC_WARP_MINIMUM = (0, 4, 4)
_UIPC_WARP_INSTALL = "pip install git+https://github.com/OrangeEarth15/libuipc-warp"
# grow-and-redo attempts of an eager step whose contact lists keep overflowing, as libuipc-warp's FrameRunner
_MAX_REGROWS = 8
# libuipc's strength ratio of a joint (the default of its joints' apply_to): the default joint_strength_ratio
_JOINT_STRENGTH_RATIO = 100.0
# the default barrier stiffness kappa [Pa], and libuipc's strength ratio of a soft transform constraint, the default
# kinematic_strength_ratio
_KAPPA = 1.0e9
_KINEMATIC_STRENGTH_RATIO = 100.0
# the Newton loop's defaults: libuipc's tolerances with its semi-implicit exit, and for an entry of a coupled solver
# with proxies converged steps, at which the feedback of an impact stays within 1% of the converged one in the
# measured scenes (docs/solvers/ipc.rst > "Coupling")
_TOLERANCES = {"newton_velocity_tol": 0.05, "newton_transrate_tol": 0.1, "semi_implicit": True}
_PROXY_TOLERANCES = {"newton_velocity_tol": 5.0e-4, "newton_transrate_tol": 1.0e-3, "semi_implicit": False}
# AL-IPC's decay of the weight of a pair that has opened: the paper's 0.9 (Zheng, Luo and Li), under which a stack of
# boxes comes to rest, where libuipc's default 0.3 drops the pair within three outer updates (docs/solvers/ipc.rst >
# "AL-IPC contact")
_AL_DECAY_FACTOR = 0.9
# MuJoCo's CONNECT and WELD equality types (mjtEq), and the target kind with which Newton's importers link an equality
# row to the loop joint they turned it into (MjcEqualityTargetKind.JOINT); neither has a public name in Newton
_EQ_CONNECT, _EQ_WELD = 0, 1
_EQ_TARGET_JOINT = 1


def _release(version: str) -> tuple[int, ...]:
    """The release numbers at the start of a version string (``"0.4.4.dev0"`` gives ``(0, 4, 4)``)."""
    match = re.match(r"\d+(?:\.\d+)*", version)
    return tuple(int(part) for part in match.group(0).split(".")) if match else ()


def _capabilities(uipc_warp) -> dict[str, bool]:
    """Whether libuipc-warp has the capabilities SolverIPC calls that a build of ``_UIPC_WARP_MINIMUM`` may lack, a
    development build among them, by name: the rotation of rigid bodies as a rigid body's (``body_rotation="rigid"``,
    the default), the sweeps of kinematic moves (the barrier's, ``sweep_kinematic``, and AL-IPC's) and the frame
    restart (an entry of a coupler that restarts its steps)."""
    options = uipc_warp.EngineOptions.names()
    engine = uipc_warp.SimEngine
    return {
        "EngineOptions(affine_rotation=)": "affine_rotation" in options,
        "EngineOptions(sweep_kinematic=)": "sweep_kinematic" in options,
        "EngineOptions(al_sweep_kinematic=) with SimEngine.place_kinematic": "al_sweep_kinematic" in options
        and hasattr(engine, "place_kinematic"),
        "the frame restart (SimEngine.enable_frame_restart)": hasattr(engine, "enable_frame_restart"),
    }


def _check_choice(name: str, value: str, accepted: tuple) -> None:
    """``ValueError`` unless ``value`` is one of ``accepted``, the values of libuipc's config key that ``name`` sets."""
    if value not in accepted:
        raise ValueError(f"SolverIPC: {name} must be one of {', '.join(map(repr, accepted))}, got {value!r}.")


def _unprojected_equality_rows(model: Model) -> np.ndarray:
    """The enabled rows of ``model.mujoco``'s equality constraints that no loop joint holds, by SolverSteel's rule.
    Newton's importers turn a CONNECT row into a BALL joint and a WELD row into a FIXED joint (a BALL joint when its
    torquescale is 0), outside every articulation and between the row's two bodies (or its body and the world), keep
    the row and link it to the joint (``equality_constraint_target_kind`` JOINT and ``equality_constraint_target``);
    SolverIPC simulates the joint as every other. A row that names another joint, or none, is not held; neither is a
    JOINT row, whose mimic constraint SolverIPC refuses."""
    mujoco = getattr(model, "mujoco", None)
    count = int(getattr(mujoco, "equality_constraint_count", 0) or 0) if mujoco is not None else 0
    if count == 0:
        return np.zeros(0, np.int64)

    def field(name: str, missing) -> np.ndarray:
        values = getattr(mujoco, f"equality_constraint_{name}", None)
        return np.full(count, missing) if values is None else values.numpy()

    enabled = field("enabled", True).astype(bool)
    target_kind, target = field("target_kind", 0), field("target", -1)
    linked = np.flatnonzero(enabled & (target_kind == _EQ_TARGET_JOINT) & (target >= 0) & (target < model.joint_count))
    projected = np.zeros(count, bool)
    if len(linked):
        eq_type, torquescale = field("type", -1), field("torquescale", 1.0)
        body1, body2 = field("body1", -2), field("body2", -2)
        joint_type, parent, child = (a.numpy() for a in (model.joint_type, model.joint_parent, model.joint_child))
        articulation = model.joint_articulation.numpy()
        for row in linked:
            j = int(target[row])
            expected = {_EQ_CONNECT: JointType.BALL, _EQ_WELD: JointType.FIXED}.get(int(eq_type[row]))
            if expected == JointType.FIXED and float(torquescale[row]) == 0.0:
                expected = JointType.BALL  # a weld without its rotation, as the importers turn it
            projected[row] = (
                expected is not None
                and int(joint_type[j]) == int(expected)
                and articulation[j] < 0
                and child[j] >= 0
                and {int(parent[j]), int(child[j])} == {int(body1[row]), int(body2[row])}
            )
    return np.flatnonzero(enabled & ~projected)


def _check_constraints(model: Model) -> None:
    """``NotImplementedError`` for a mimic constraint, and for an enabled equality row that no loop joint holds
    (``_unprojected_equality_rows``)."""
    mimic = int(model.constraint_mimic_count)
    if model.joint_count and model.joint_mimic_joint is not None:
        mimic += int(np.count_nonzero(model.joint_mimic_joint.numpy() >= 0))
    if mimic:
        raise NotImplementedError(
            f"SolverIPC: the model has {mimic} mimic constraints; remove them. SolverIPC does not enforce mimic "
            "constraints."
        )
    rows = _unprojected_equality_rows(model)
    if len(rows):
        shown = rows[:8].tolist()
        raise NotImplementedError(
            f"SolverIPC: model.mujoco has {len(rows)} enabled equality rows that no loop joint holds (rows {shown}"
            f"{' and more' if len(rows) > 8 else ''}); import CONNECT and WELD equalities with "
            "convert_mjc_equality_constraints=True, model the constraints with Newton's joints, or disable the rows "
            "(equality_constraint_enabled). SolverIPC enforces only the rows that Newton's importers turn into loop "
            "joints, through those joints."
        )


def _check_supported(model: Model, joints: _JointOptions | None = None) -> None:
    """Reject model content the solver would otherwise accept and ignore; joints as ``joints`` maps them
    (``_JointOptions``). What independent worlds refuse is checked when the solve is laid out
    (``SolverIPC._check_independent_options``, ``_check_independent_content``)."""
    if model.requires_grad:
        raise NotImplementedError(
            "SolverIPC: the model requires gradients (requires_grad=True); finalize it without them. SolverIPC "
            "computes no gradients."
        )
    variant = model.body_variant.numpy() if model.body_count and model.body_variant is not None else np.zeros(0)
    if np.any(variant >= 0):
        bodies = np.flatnonzero(variant >= 0)
        raise NotImplementedError(
            f"SolverIPC: {len(bodies)} bodies have shape variants (ModelBuilder.add_body_variants; bodies "
            f"{bodies[:8].tolist()}{' and more' if len(bodies) > 8 else ''}); build each body with one set of shapes. "
            "SolverIPC triangulates the shapes when it builds the solve and finds its own contacts, so it cannot "
            "follow a variant row's shape."
        )
    for name, count in (
        ("tendons", model.tendon_count + model.custom_frequency_counts.get("mujoco:tendon", 0)),
        ("muscles", model.muscle_count),
    ):
        if count:
            raise NotImplementedError(
                f"SolverIPC: the model has {count} {name}; remove them. SolverIPC does not simulate {name}."
            )
    _check_constraints(model)
    own_tris, own_tets, _ = _own_elements(model)
    for name, activations, own in (
        ("triangle", model.tri_activations, own_tris),
        ("tetrahedron", model.tet_activations, own_tets),
    ):
        if activations is not None and len(activations) and np.any(activations.numpy()[own] != 0.0):
            raise NotImplementedError(
                f"SolverIPC: the model has non-zero {name} activations; set them to 0. SolverIPC does not read "
                "activations."
            )
    _check_joints(model, joints)
    _check_springs(model)
    _check_attachments(model)
    _check_surface_velocity(model)


class SolverIPC(SolverBase, CouplingInterface):
    """Incremental Potential Contact solver for soft bodies, cloth and rigid bodies.

    .. experimental::

        :class:`SolverIPC`, its constructor options and its behavior may change without the normal deprecation
        period.

    Wraps the `libuipc-warp <https://github.com/OrangeEarth15/libuipc-warp>`_ engine,
    a Warp implementation of libuipc, through its array interface (no libuipc
    build is needed). Each :meth:`step` is one implicit time step, by default
    of implicit Euler (BDF1): a Newton iteration with a backtracking line
    search, a block-Jacobi PCG linear solve, barrier contact and continuous
    collision detection. ``integrator="bdf2"``,
    ``contact_constitution="al-ipc"`` and ``fem_preconditioner="mas"`` take
    libuipc's BDF2, its augmented-Lagrangian contact and its MAS
    preconditioner instead. Under implicit Euler the rotation of a rigid body
    is by default integrated as a rigid body's (``body_rotation``). The Newton
    loop, the PCG loop and the line search run on the device under
    ``wp.capture_while``, so a step never synchronizes with the host when it
    is captured in a CUDA graph.

    The IPC solver page, :doc:`/solvers/ipc`, describes how the solver takes
    every input of Newton's model, state and control. In short, it simulates:

    - particles and tetrahedra, with libuipc's stable Neo-Hookean elasticity
      (:ref:`ipc-model-content`);
    - cloth: libuipc's strain-limiting Baraff-Witkin membrane and
      discrete-shell bending, from Newton's rest shapes (:ref:`ipc-cloth`);
    - springs, and particle-body attachments as libuipc's soft vertex stitch
      (:ref:`ipc-attachments`);
    - rigid bodies with box, sphere, capsule, cylinder, cone, ellipsoid or
      mesh shapes as libuipc's affine bodies (:ref:`ipc-rigid-bodies`);
    - static shapes as fixed triangle meshes, static planes as half spaces,
      and height fields;
    - contact between all of these, with friction, between the pairs that
      Newton's collision groups, worlds, collide flags and filter pairs let
      collide (:ref:`ipc-contact`);
    - kinematic particles and bodies, which follow ``state_in``, or with
      ``kinematic_motion="aim"`` move inside the step toward the pose it
      prescribes (:ref:`ipc-kinematic-bodies`);
    - revolute, prismatic, fixed and ball joints with drives, springs,
      damping, limits, armature and :attr:`~newton.Control.joint_f`, D6 joints
      that lock what one of them locks, and distance joints of one length
      (:ref:`ipc-joints`);
    - external forces (:attr:`~newton.State.particle_f`,
      :attr:`~newton.State.body_f`) and gravity per world;
    - content in several worlds, every world stepping as it would alone, or
      with ``coupled_worlds`` as one solve (:ref:`ipc-worlds`).

    What libuipc has no form for raises :class:`NotImplementedError` when the
    solver is constructed (:ref:`ipc-refused`), and so does a model on the CPU;
    :ref:`ipc-not-read` lists the inputs the solver does not read. The
    ``contacts`` argument of :meth:`step` is not read: the solver detects its
    own contacts. Surfaces that cross each other, or that start at or inside
    their contact thickness, raise :class:`ValueError` (:ref:`ipc-start`).

    A step that is not captured checks how its solve ended: when a contact list
    overflowed it enlarges the lists and solves the step again, and when the
    step failed it raises :class:`RuntimeError`. A captured step cannot do
    either: request :attr:`SolverIPC.ObservableKind.STATUS` with
    :meth:`observables`, pass the container to :meth:`step` and call
    :meth:`check_status` after every replay (:ref:`ipc-status`). The solver
    keeps positions and velocities in float64 and takes from ``state_in`` only
    what differs from what it wrote (:ref:`ipc-state`). With
    ``SolverBase.ObservableKind.CONTACT_F`` a step writes its own contacts and
    their forces into the step's :class:`~newton.Contacts`
    (:ref:`ipc-contact-output`), and the solver is an entry of Newton's coupled
    solvers (:ref:`ipc-coupling`).

    This solver requires the optional ``libuipc-warp`` package, 0.4.4 or newer (``import uipc_warp``).

    Args:
        model: The model to simulate.
        kappa: Barrier stiffness [Pa] of every contact pair, taken as given:
            libuipc's contact resistance, which libuipc and its pyuipc front
            end would clamp into libuipc's range (see ``adaptive_kappa``).
            ``None``, with ``adaptive_kappa``, takes the range's lower bound,
            libuipc's default. Not used with ``contact_constitution="al-ipc"``,
            which refuses a value other than the default.
        adaptive_kappa: Take the contact stiffness as libuipc and its pyuipc
            front end do: inside libuipc's scene-adaptive range, which
            follows each step's ``dt`` (see :ref:`ipc-contact`); with independent
            worlds every world's own range, that of the world alone (see
            :ref:`ipc-worlds`).
        d_hat: Barrier activation distance [m]. ``None`` takes libuipc's
            0.01 m, or less for a model with small colliding features (see
            :ref:`ipc-d-hat`).
        thickness: Contact thickness of the vertices of cloth and tetrahedra
            [m]; free particles take their radius and shapes their margin
            (see :ref:`ipc-contact`).
        newton_max_iter: Maximum Newton iterations per step.
        newton_velocity_tol: Newton convergence tolerance on ``max |dx| / dt``
            [m/s]. ``None`` takes 0.05, or 5e-4 for an entry of a coupled
            solver with proxies (see :ref:`ipc-coupling`).
        newton_transrate_tol: Newton convergence tolerance on ``max |dA| / dt`` of
            rigid bodies [1/s]. ``None`` takes 0.1, or 1e-3 with proxies.
        linear_tol_rate: Relative PCG tolerance on ``r^T M^-1 r``.
        line_search_max_iter: Maximum backtracking trials per Newton iteration.
        semi_implicit: Enable libuipc's semi-implicit early exit after
            ``semi_implicit_k_min`` iterations of full feasible steps; ``None``
            enables it but with proxies. Under
            ``contact_constitution="al-ipc"``, whose step has no Newton
            tolerance, it enables the delay of AL-IPC's progress weight
            instead: without contact a step takes ``semi_implicit_k_min``
            Newton iterations, and one when False (see :ref:`ipc-al-ipc`).
        semi_implicit_k_min: Newton iterations before the early exit applies.
        eps_velocity: Friction smoothing velocity [m/s]; below it static friction becomes a stiff spring.
        cloth_strain_rate: Strain-limiting coefficient :math:`\\eta` of the cloth
            membrane (libuipc's default 100); 0 makes the membrane quadratic.
        body_kappa: Orthogonality stiffness :math:`\\kappa` of rigid bodies [Pa]
            (libuipc's affine-body stiffness), taken as given at every step.
            ``None`` takes libuipc's 1e8 Pa, lowered for a body and a step
            whose turn of 0.1 rad about the axis of its largest moment of
            inertia it would not take in one Newton iteration (see
            :ref:`ipc-rigid-bodies`).
        joint_strength_ratio: Penalty of the joints relative to the mass of the
            dynamic bodies they join (libuipc's ``strength_ratio``; see :ref:`ipc-joints`).
        shape_segments: Segments around a sphere, capsule, cylinder, cone or
            ellipsoid when it is triangulated.
        coupled_worlds: Simulate content in more than one world as one solve
            in which no world collides with another, where by default every
            world is solved apart from the others (see :ref:`ipc-worlds`).
        integrator: ``"bdf1"`` (implicit Euler) or ``"bdf2"`` (see :ref:`ipc-bdf2`).
        body_rotation: ``"rigid"``, the rotation of rigid bodies integrated as
            a rigid body's, or ``"linear"``, libuipc's step, in which a rigid
            body moves along a straight line in its 12 coordinates and a spin
            loses about :math:`(\\omega \\Delta t)^2` per step (see
            :ref:`ipc-rigid-bodies`). ``None`` (the default) is ``"rigid"`` under
            ``integrator="bdf1"`` and ``"linear"`` under ``"bdf2"``, which
            refuses ``"rigid"``.
        contact_constitution: ``"ipc"`` (barrier contact) or ``"al-ipc"`` (see
            :ref:`ipc-al-ipc`).
        fem_preconditioner: ``"diag"`` (block diagonal) or ``"mas"`` for
            particles, cloth and tetrahedra (see :ref:`ipc-mas`).
        kinematic_motion: ``"jump"`` (a kinematic body takes the pose of
            ``state_in`` before the step, and a jump that meets a surface
            fails the step) or ``"aim"`` (it moves inside the step toward the
            pose prescribed for its end; see :ref:`ipc-kinematic-bodies`).
        kinematic_strength_ratio: The strength ratio of the aims of
            ``kinematic_motion="aim"`` (libuipc's soft transform constraint);
            ``"jump"`` refuses a value other than the default.
        enable_joint_limit_damping: Damp a revolute or prismatic joint that
            starts a step past one of its limits with ``joint_limit_kd``, as
            Featherstone and SemiImplicit do (see :ref:`ipc-joints`); with False
            ``joint_limit_kd`` is not read.
        enable_joint_effort_limit: Bound a drive's force by its
            ``joint_effort_limit`` as decided at the start of every step (see
            :ref:`ipc-joints`); with False a limit below 1e6 on a driven degree of
            freedom is rejected.

    Note:
        A step ends once the Newton update moves no vertex by more than
        ``newton_velocity_tol * dt``: lower the tolerance where a body
        should gain less velocity per step (:ref:`ipc-tolerances`). A
        resting contact settles just inside ``d_hat``, which by default
        stays below the smallest colliding features (:ref:`ipc-d-hat`). The
        barrier needs a gap at the start (:ref:`ipc-start`).
    """

    class ObservableKind(SolverBase.ObservableKind):
        """Standard and IPC-specific observable kinds.

        Inherits all standard names from :class:`SolverBase.ObservableKind`.

        .. experimental::
        """

        STATUS = "status"
        """Per-world bitmask of :class:`SolverIPC.StatusBits` for the most recent step."""

    @dataclass(eq=False)
    class Observables(SolverBase.Observables):
        """Standard and IPC-specific observable arrays.

        .. experimental::
        """

        status: wp.array[wp.int32] | None = SolverBase.Observables.field(
            dtype=wp.int32, frequency=Model.AttributeFrequency.WORLD
        )
        """Per-world :class:`SolverIPC.StatusBits` of the latest step, shape ``(world_count + 1,)``.

        With independent worlds (several worlds, the default) every world that holds unknowns gets its own mask, and
        the final entry, global world ``-1``, stays zero. With one world, or ``coupled_worlds``, every world that owns
        particles or bodies gets the mask of the one solve, the final entry too when global particles or bodies are
        in it. Other entries stay zero.
        """

    class StatusBits(IntFlag):
        """Bits of the per-world ``Observables.status`` mask: how the solve of the step ended.

        The step failed, and its result must not be used, when a bit of :attr:`SolverIPC.StatusBits.FAILED` is set. The two
        iteration limits are not failures, and neither is a bit of a later libuipc-warp outside
        :attr:`SolverIPC.StatusBits.KNOWN`.
        """

        OVERFLOW = 1 << 0
        """A contact list overflowed and dropped pairs: the step ran without them. With independent worlds a world
        asked for more than its quota of the lists: the bit is in that world's entry alone, and that world is back at
        its state at the step's start. Set from then on until the lists are enlarged (:meth:`SolverIPC.grow`, or an
        eager step)."""

        NONFINITE_ENERGY = 1 << 1
        """The line search met a non-finite energy."""

        NONFINITE_PCG_INIT = 1 << 2
        """The linear solve started from a non-finite residual."""

        NONFINITE_PCG_ITER = 1 << 3
        """The linear solve produced a non-finite residual."""

        INVALID_CCD_STEP = 1 << 4
        """Continuous collision detection admitted no step (a fraction that is not positive), as for a vertex
        that starts at or inside its contact thickness from a plane; or the sweep of a move written since the last
        step (a kinematic body's or particle's jump, a coupled solver's proxy's new pose) met a surface, and the step
        did not run (:ref:`ipc-kinematic-bodies`)."""

        THICKNESS_VIOLATED = 1 << 5
        """A contact pair started at or inside its contact thickness (surfaces in touching contact or
        intersecting)."""

        TRAVERSAL_EXHAUSTED = 1 << 6
        """A BVH traversal ran out of its node budget: contact candidates are missing."""

        LINE_SEARCH_LIMIT = 1 << 7
        """A line search ran out of its ``line_search_max_iter`` trials."""

        NEWTON_LIMIT = 1 << 8
        """The Newton loop ran to ``newton_max_iter`` without converging."""

        FAILED = (
            OVERFLOW
            | NONFINITE_ENERGY
            | NONFINITE_PCG_INIT
            | NONFINITE_PCG_ITER
            | INVALID_CCD_STEP
            | THICKNESS_VIOLATED
            | TRAVERSAL_EXHAUSTED
        )
        """The bits of a failed step."""

        KNOWN = FAILED | LINE_SEARCH_LIMIT | NEWTON_LIMIT
        """Every bit this class names. A later libuipc-warp may set bits above them in the status mask (its
        ``FrameStatusFlags``): such a bit does not fail the step, and :meth:`SolverIPC.check_status` warns, naming it
        as libuipc-warp does."""

    SUPPORTED_OBSERVABLES = frozenset(
        {SolverBase.ObservableKind.SOLVER_RESIDUAL, SolverBase.ObservableKind.CONTACT_F, ObservableKind.STATUS}
    )

    # libuipc-warp, imported by the first solver
    _uipc_warp: ClassVar = None

    def __init__(
        self,
        model: Model,
        *,
        kappa: float | None = _KAPPA,
        adaptive_kappa: bool = False,
        d_hat: float | None = None,
        thickness: float = 0.0,
        newton_max_iter: int = 1024,
        newton_velocity_tol: float | None = None,
        newton_transrate_tol: float | None = None,
        linear_tol_rate: float = 1.0e-3,
        line_search_max_iter: int = 8,
        semi_implicit: bool | None = None,
        semi_implicit_k_min: int = 6,
        eps_velocity: float = 0.01,
        cloth_strain_rate: float = 100.0,
        body_kappa: float | None = None,
        joint_strength_ratio: float = _JOINT_STRENGTH_RATIO,
        shape_segments: int = 24,
        coupled_worlds: bool = False,
        integrator: str = "bdf1",
        body_rotation: str | None = None,
        contact_constitution: str = "ipc",
        fem_preconditioner: str = "diag",
        kinematic_motion: str = "jump",
        kinematic_strength_ratio: float = _KINEMATIC_STRENGTH_RATIO,
        enable_joint_limit_damping: bool = True,
        enable_joint_effort_limit: bool = True,
    ):
        if not model.device.is_cuda:
            raise NotImplementedError(
                f"SolverIPC: the model is on {model.device}; finalize it on a CUDA device. libuipc-warp runs on CUDA "
                "devices only."
            )
        super().__init__(model)
        self._coupled_worlds = bool(coupled_worlds)
        self._joint_options = _JointOptions(
            limit_damping=bool(enable_joint_limit_damping), effort_limit=bool(enable_joint_effort_limit)
        )
        _check_supported(model, self._joint_options)
        self._velocity_limits_warned = False
        _warn_of_velocity_limits(self, stacklevel=3)
        unread = _unread_materials(model)
        if unread:
            warnings.warn(
                f"SolverIPC does not read {', '.join(unread)}: its contact has no restitution and one friction "
                "coefficient, without torsional or rolling friction (see Inputs not read in Newton's IPC solver "
                "guide).",
                NewtonWarning,
                stacklevel=2,
            )
        _check_choice("integrator", integrator, ("bdf1", "bdf2"))
        # the rigid flight under implicit Euler, libuipc's step under BDF2 (docs/solvers/ipc.rst > "Rigid bodies")
        if body_rotation is None:
            body_rotation = "linear" if integrator == "bdf2" else "rigid"
        _check_choice("body_rotation", body_rotation, ("linear", "rigid"))
        if body_rotation == "rigid" and integrator == "bdf2":
            # libuipc-warp's rigid flight replaces the implicit Euler prediction; BDF2 predicts from the step before too
            raise ValueError(
                "SolverIPC: body_rotation='rigid' integrates the rigid bodies with implicit Euler (integrator='bdf1'); "
                "BDF2 predicts from the step before as well, which the rigid flight does not turn. With "
                "integrator='bdf2' leave body_rotation unset, or pass 'linear', libuipc's step."
            )
        _check_choice("contact_constitution", contact_constitution, ("ipc", "al-ipc"))
        _check_choice("fem_preconditioner", fem_preconditioner, ("diag", "mas"))
        _check_choice("kinematic_motion", kinematic_motion, ("jump", "aim"))
        for name, value in (
            ("thickness", thickness),
            ("cloth_strain_rate", cloth_strain_rate),
            ("joint_strength_ratio", joint_strength_ratio),
            ("kinematic_strength_ratio", kinematic_strength_ratio),
        ):
            if not (np.isfinite(value) and value >= 0.0):
                raise ValueError(f"SolverIPC: {name} must be finite and not negative, got {value}.")
        if not (np.isfinite(shape_segments) and shape_segments >= 3 and shape_segments == int(shape_segments)):
            raise ValueError(f"SolverIPC: shape_segments must be a whole number of at least 3, got {shape_segments}.")
        # a value other than the default of an argument that the other options leave unread
        if adaptive_kappa and contact_constitution == "al-ipc":
            raise ValueError(
                "SolverIPC: adaptive_kappa sets the stiffness of the barrier, which contact_constitution='al-ipc' "
                "does not use; leave it False."
            )
        if kappa != _KAPPA and contact_constitution == "al-ipc":
            raise ValueError(
                f"SolverIPC: kappa sets the stiffness of the barrier, which contact_constitution='al-ipc' does not use; "
                f"leave it at its default {_KAPPA:g}, got {kappa}."
            )
        if kinematic_strength_ratio != _KINEMATIC_STRENGTH_RATIO and kinematic_motion == "jump":
            raise ValueError(
                "SolverIPC: kinematic_strength_ratio sets the strength of the aims of kinematic_motion='aim', which "
                f"kinematic_motion='jump' does not use; leave it at its default {_KINEMATIC_STRENGTH_RATIO:g}, got "
                f"{kinematic_strength_ratio}."
            )
        if kappa is None and not adaptive_kappa:
            raise ValueError("SolverIPC: kappa=None, libuipc's default stiffness, needs adaptive_kappa=True.")
        if kappa is not None and not (np.isfinite(kappa) and kappa > 0.0):
            raise ValueError(f"SolverIPC: kappa must be a finite positive stiffness, got {kappa}.")
        if body_kappa is not None and not (np.isfinite(body_kappa) and body_kappa > 0.0):
            raise ValueError(f"SolverIPC: body_kappa must be a finite positive stiffness, got {body_kappa}.")
        uipc_warp = self._import_uipc_warp()
        # an entry of a coupled solver with proxies takes converged steps unless told otherwise: a proxy's feedback is
        # the force of the step's contact terms at its end (docs/solvers/ipc.rst > "Coupling")
        tolerances = _PROXY_TOLERANCES if _proxy_bodies(model).any() or _proxy_particles(model).any() else _TOLERANCES
        if newton_velocity_tol is None:
            newton_velocity_tol = tolerances["newton_velocity_tol"]
        if newton_transrate_tol is None:
            newton_transrate_tol = tolerances["newton_transrate_tol"]
        if semi_implicit is None:
            semi_implicit = tolerances["semi_implicit"]
        self._d_hat_given = d_hat is not None
        al = {"al_decay_factor": _AL_DECAY_FACTOR} if contact_constitution == "al-ipc" else {}
        self._config = uipc_warp.SimConfig(
            kappa=kappa,
            d_hat=_D_HAT if d_hat is None else d_hat,
            newton_max_iter=newton_max_iter,
            newton_velocity_tol=newton_velocity_tol,
            newton_transrate_tol=newton_transrate_tol,
            linear_tol_rate=linear_tol_rate,
            line_search_max_iter=line_search_max_iter,
            semi_implicit=semi_implicit,
            semi_implicit_k_min=semi_implicit_k_min,
            eps_velocity=eps_velocity,
            integrator=integrator,
            contact_constitution=contact_constitution,
            fem_preconditioner=fem_preconditioner,
            **al,
        )
        # a BDF2 history starts with an implicit Euler step: at the first step, and at every step from a state the
        # solver did not write (docs/solvers/ipc.rst > "BDF2")
        options = {"bdf2_start": "bdf1"} if integrator == "bdf2" else {}
        # the moves of kinematic bodies and particles between steps, and of a coupled solver's proxies, are swept from
        # where the last step left them: the barrier's step sweeps them at its start and fails on one that meets a
        # surface (docs/solvers/ipc.rst > "Kinematic bodies"); AL-IPC's collision-free sweeps carry the kinematic
        # bodies along and push what they meet (docs/solvers/ipc.rst > "AL-IPC contact")
        self._al_sweep = contact_constitution == "al-ipc"
        options["al_sweep_kinematic" if self._al_sweep else "sweep_kinematic"] = True
        # body_rotation="rigid": the rigid bodies' rotation as a rigid body's flight (docs/solvers/ipc.rst >
        # "Rigid bodies")
        self._rigid_rotation = body_rotation == "rigid"
        if self._rigid_rotation:
            options["affine_rotation"] = "rigid"
        self._options = uipc_warp.EngineOptions(**options) if options else None
        self._adaptive_kappa = bool(adaptive_kappa)
        self._thickness = float(thickness)
        self._cloth_strain_rate = float(cloth_strain_rate)
        self._body_kappa = None if body_kappa is None else float(body_kappa)
        self._joint_strength_ratio = float(joint_strength_ratio)
        self._shape_segments = int(shape_segments)
        self._aims = kinematic_motion == "aim"
        self._kinematic_strength_ratio = float(kinematic_strength_ratio)
        self._dt = self._config.dt  # of the last step, which grow() solves again
        self._clamp_checked_dt = None  # the step size whose stiffness the clamp warning was last checked for
        self._clamp_reported = False
        self._engine = None
        # several local worlds (_Worlds), solved apart from each other unless coupled_worlds (_build_engine)
        self._worlds = None
        self._independent = False
        self._ortho = None  # the engine's orthogonality potentials, whose stiffness follows dt (default body_kappa)
        self._body_spread = None  # body_rotation="rigid": the bodies' second moments about their centres of mass
        self._default_control = None  # the Control of a step given none, built once (_control_of_model)
        # CONTACT_F: whether a container requested it, and libuipc-warp's ContactForces of the engine then
        # (allocated with the request and with every engine built after it, outside any capture)
        self._contact_output = False
        self._contact_export = None
        self._steps = 0  # steps taken, which the contact export's records follow (_contact_records)
        # an entry of a coupled solver (coupling_notify_iteration_restarts): the status bits of its steps since
        # check_status() last read them, and the scratch word of one step's bits
        self._sticky_status = self._step_status = None
        # an entry of a coupler that restarts its steps (coupling_notify_iteration_restarts): the start of a coupled step
        # kept on the device, saved by the step after the coupler wrote the input state and restored by the step after
        # a restart's notification (_save_step, _restore_step); the (array, copy) of what the solver carries from step
        # to step (_allocate_restart)
        self._restarts = False
        self._save_pending = self._restore_pending = False
        self._carried = []
        # integrator="bdf2": the gravity of the steps and of the step before, on the device (_build_engine)
        self._gravity_now = self._gravity_last = None
        # the failure, in the model's state, of the start of a pair with a body whose start the caller writes, which
        # a captured first step raises (_check_start)
        self._start_pending = None
        self._build_engine()
        _check_start(self)
        # the first step checks again the start of the pairs with a body whose start the caller writes
        self._start_unchecked = self._engine is not None and bool(self._posed_body.any())

    @property
    def supported_observables(self) -> frozenset[str]:
        """``STATUS``, ``SOLVER_RESIDUAL`` and ``CONTACT_F``; with ``contact_constitution="al-ipc"`` ``STATUS`` only: the
        AL-IPC frame ends on its own criterion, without a Newton residual, and its contact is an active set with
        multipliers, not the barrier and friction terms the contact output exports."""
        config = getattr(self, "_config", None)
        if config is not None and config.contact_constitution == "al-ipc":
            return frozenset({self.ObservableKind.STATUS})
        return self.SUPPORTED_OBSERVABLES

    @override
    def observables(
        self, kinds, *, contacts: Contacts | None = None, requires_grad: bool | None = None
    ) -> SolverIPC.Observables:
        """Allocate solver observables (see :meth:`SolverBase.observables`).

        With ``CONTACT_F`` requested this also allocates what a step needs to write the solver's own contacts into the
        step's :class:`~newton.Contacts` (see :ref:`ipc-contact-output`): libuipc-warp's records of every slot of the
        solver's contact lists, about 64 bytes per slot, kept from then on and allocated again when the solve is built
        again. ``contacts`` sizes ``contact_f``; any capacity is taken, the rows past it are dropped
        (:meth:`get_max_contact_count` and :meth:`get_max_soft_contact_count` give the capacities that drop none).
        Allocate before graph capture.
        """
        observables = super().observables(kinds, contacts=contacts, requires_grad=requires_grad)
        if observables.is_requested(SolverBase.ObservableKind.CONTACT_F) and not self._contact_output:
            self._contact_output = True
            _make_contact_export(self)
        return observables

    def _contact_term_capacity(self) -> int:
        """The most contact terms a step can export: every slot of the engine's contact lists (libuipc-warp's
        ``ContactForces.segment_capacities``)."""
        if self._engine is None:
            return 0
        return int(self._uipc_warp.arrays.ContactForces.segment_capacities(self._engine).sum())

    def get_max_contact_count(self) -> int:
        """The rigid rows a step with ``CONTACT_F`` can write: the size of ``rigid_contact_max`` that drops none.

        Every contact term between two shapes is a row (see :ref:`ipc-contact-output`), so this is the number of slots
        of the solver's contact lists, active and friction terms together, or 0 when fewer than two shapes collide. It
        is a worst case, several times the rows a step usually writes; a smaller buffer is taken, and the rows past it
        are dropped while ``rigid_contact_count`` counts them. :meth:`grow` and an eager step that overflows enlarge the
        lists, and this with them. 0 for a model without particles and bodies.
        """
        if self._engine is None or len(self._contact_shapes) < 2:
            return 0
        return self._contact_term_capacity()

    def get_max_soft_contact_count(self) -> int:
        """The soft rows a step with ``CONTACT_F`` can write: the size of ``soft_contact_max`` that drops none.

        Every contact term between a particle feature (a particle, an edge or a face of cloth or tetrahedra) and a
        shape is a row, so this is the number of slots of the solver's contact lists, or 0 without particles or
        without colliding shapes. As :meth:`get_max_contact_count`, it is a worst case, and a smaller buffer is taken.
        """
        if self._engine is None or self.model.particle_count == 0 or len(self._contact_shapes) == 0:
            return 0
        return self._contact_term_capacity()

    @classmethod
    def _import_uipc_warp(cls):
        """Import libuipc-warp once: raise when it is missing, older than ``_UIPC_WARP_MINIMUM``, without one of the
        capabilities SolverIPC calls (``_capabilities``) or with other status bits than ``StatusBits``."""
        if cls._uipc_warp is not None:
            return cls._uipc_warp
        try:
            import uipc_warp  # noqa: PLC0415
            import uipc_warp.arrays  # noqa: PLC0415
            import uipc_warp.validation  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(f"SolverIPC requires the libuipc-warp package: {_UIPC_WARP_INSTALL}") from e
        version = getattr(uipc_warp, "__version__", "0")
        minimum = ".".join(map(str, _UIPC_WARP_MINIMUM))
        if _release(version) < _UIPC_WARP_MINIMUM:
            raise ImportError(
                f"SolverIPC requires libuipc-warp {minimum} or newer, and the installed one is {version}: "
                f"{_UIPC_WARP_INSTALL}"
            )
        missing = [name for name, present in _capabilities(uipc_warp).items() if not present]
        if missing:
            raise ImportError(
                f"SolverIPC requires libuipc-warp {minimum} or newer with {', '.join(_capabilities(uipc_warp))}; "
                f"the installed one ({version}) lacks {', '.join(missing)}: {_UIPC_WARP_INSTALL}"
            )
        # write_frame_status writes libuipc-warp's FrameStatusFlags into the status arrays: every StatusBits bit must be
        # one of them under its name (but the CCD bit, which libuipc-warp names after the time of impact); a bit that
        # a later libuipc-warp adds is taken as no failure (StatusBits)
        flags = {int(flag): flag.name for flag in uipc_warp.arrays.FrameStatusFlags}
        renamed = int(cls.StatusBits.INVALID_CCD_STEP)
        unlike = [
            bit.name for bit in cls.StatusBits if int(bit) not in flags or (flags[bit] != bit.name and bit != renamed)
        ]
        if unlike:
            raise RuntimeError(
                f"Incompatible libuipc-warp {version}: its FrameStatusFlags {flags} lack SolverIPC.StatusBits "
                f"{', '.join(unlike)} under their names and values."
            )
        cls._uipc_warp = uipc_warp
        return uipc_warp

    # ------------------------------------------------------------------ setup
    def _build_engine(self) -> None:
        """Lay the model out for libuipc-warp and build its engine. The next step takes its whole state from
        ``state_in``."""
        model = self.model
        device = model.device
        self._engine = self._frame_graph = self._forces = self._contact_export = None
        # a coupled solver's proxies (docs/solvers/ipc.rst > "Coupling")
        self._proxy_body, self._proxy_particle = _proxy_bodies(model), _proxy_particles(model)
        self._proxies = bool(self._proxy_body.any() or self._proxy_particle.any())
        if model.particle_count == 0 and model.body_count == 0:
            return
        if self._proxies:
            self._check_proxies()
        # the gravity of every world, the global world's last, which the parts are laid out with; the config takes the
        # global world's, which every part overrides with the gravity of its own world
        self._gravity = self._layout_gravity = _world_gravity(self)
        self._config.gravity = tuple(self._gravity[-1])
        if not self._d_hat_given:
            self._config.d_hat = _default_d_hat(model)

        # several local worlds: one subscene each, all of them meeting the global entities' subscene 0; libuipc-warp
        # solves them apart from each other (independent worlds) unless coupled_worlds asks for one solve
        self._worlds = _Worlds.of(model)
        self._independent = self._worlds is not None and not self._coupled_worlds
        self._status_entries = None
        scene = self._uipc_warp.arrays.ArrayScene(
            kappa_corridor=self._adaptive_kappa, independent_subscenes=self._independent
        )
        elements = _contact_elements(model, self._worlds, self._aims)
        for _ in range(1, elements.count):
            scene.contact_element()
        if self._worlds is not None:
            for _ in range(1, self._worlds.count):
                scene.subscene_element()
            scene.subscene_model(np.zeros(self._worlds.count - 1, np.int64), np.arange(1, self._worlds.count), True)
        tets = _add_particles_and_tets(self, scene, elements)
        _add_cloth(self, scene, tets)
        _add_springs(self, scene)
        _plan_attachments(self)
        _plan_distance_joints(self)
        self._dyadic = _add_shapes_and_bodies(self, scene, elements)
        if self._independent:
            _check_independent_content(self)
            if len(self._status_entries) > 1:  # one world that moves is solved as one world: no option to refuse
                self._check_independent_options()
        if self._config.contact_constitution == "al-ipc":
            _check_al_ipc(self, elements)
        _add_joints(self, scene)
        self._planes = _collect_planes(self)
        for P, N, s in zip(*self._planes, strict=True):
            scene.add_half_plane(P, N, contact_element=elements.shape[s], subscene_element=_subscene_of_shape(self, s))
        if elements.count > 1:
            # one model for every pair of the elements that hold vertices, with the default model's stiffness: the
            # solver's kappa, as given or as libuipc's policy resolves it (adaptive_kappa), one value for every pair
            a, b = np.triu_indices(elements.count - 1)
            a, b = a + 1, b + 1
            scene.contact_model(a, b, elements.mu[a, b], None, elements.enabled[a, b])
        self._attachment_hard = wp.array(_attachment_penalties(self, self._dyadic[0]), dtype=wp.float64, device=device)
        self._distance_hard = wp.array(_distance_penalties(self, self._dyadic[0]), dtype=wp.float64, device=device)
        _add_stitches(self, scene)
        if len(self._aim_bodies):  # libuipc's soft transform constraint on the kinematic bodies, in their order
            ratio = self._kinematic_strength_ratio
            scene.add_soft_transform_constraints(self._aim_bodies, (ratio, ratio))
        self._scene = scene
        self._scene_inputs = _scene_inputs_now(self)
        self._element_inputs = _element_inputs_now(self)
        self._joint_inputs = _joint_inputs_now(self)
        self._control = None  # of the last step, whose joint terms a redone frame takes again

        layout = scene.layout()
        # CONTACT_F's contact rows: the shape of every engine vertex on a shape (-1 elsewhere: particles, the bodies'
        # pseudo-vertices and anchors), and the shapes that hold contact primitives, the planes' among them
        vertex_shape = np.full(layout.n, -1, np.int32)
        for k, shape in enumerate(self._obstacle_shapes):
            vertex_shape[layout.obstacle_offsets[k] : layout.obstacle_offsets[k + 1]] = shape
        for b, shapes in enumerate(self._body_vertex_shape):
            vertex_shape[layout.body_vertex_offsets[b] : layout.body_vertex_offsets[b + 1]] = shapes
        self._vertex_shape = vertex_shape
        self._contact_shapes = np.union1d(vertex_shape[vertex_shape >= 0], np.asarray(self._planes[2], np.int32))
        # the external accelerations: one row per free particle, then four per dynamic body (p and the rows of A)
        free = np.flatnonzero(~self._fixed_particle).astype(np.int32)
        dynamic = np.flatnonzero(~self._kinematic_body).astype(np.int32)
        self._n_free, self._n_dynamic = len(free), len(dynamic)
        self._force_blocks = np.concatenate([free, (layout.body_base[dynamic][:, None] + np.arange(4)).reshape(-1)])
        self._free_particles = wp.array(free, dtype=wp.int32, device=device)
        self._dynamic_bodies = wp.array(dynamic, dtype=wp.int32, device=device)
        self._body_mass_inv = wp.array(_mass_inverse(self, *self._dyadic), dtype=wp.mat44d, device=device)
        # the requests of a new state or a reset, one flag per subscene (one with one world): which subscenes
        # discard their friction and start their BDF2 history anew, libuipc-warp's requests per subscene
        n_sub = 1 if self._worlds is None else self._worlds.count
        self._changed = wp.zeros(n_sub, dtype=wp.int32, device=device)
        self._selected = wp.zeros(n_sub, dtype=wp.int32, device=device)
        for name, entity_world in (("_particle_subscene", model.particle_world), ("_body_subscene", model.body_world)):
            w = np.zeros(0, np.int64) if entity_world is None else entity_world.numpy()
            subscene = np.zeros(len(w)) if self._worlds is None else self._worlds.subscene(w)
            setattr(self, name, wp.array(np.asarray(subscene, np.int32), dtype=wp.int32, device=device))
        if self._worlds is not None:  # the world_mask entry of every subscene: the global entities', then the worlds'
            entry = np.concatenate([[model.world_count], self._worlds.worlds]).astype(np.int32)
            self._subscene_entry = wp.array(entry, dtype=wp.int32, device=device)
        worlds = np.unique(
            np.concatenate([w.numpy() for w in (model.particle_world, model.body_world) if w is not None and len(w)])
        )
        self._world_entries = np.where(worlds < 0, model.world_count, worlds).astype(np.int32)
        self._world_entries_device = wp.array(self._world_entries, dtype=wp.int32, device=device)
        # integrator="bdf2": the gravity of every world entry that the steps take (_gravity) and the one the step
        # before took, on the device, and the subscene whose history a change restarts (-1: an entry without particles
        # or bodies); a step whose gravity differs restarts the histories of those subscenes (_load)
        self._gravity_now = self._gravity_last = None
        if self._config.integrator == "bdf2":
            self._gravity_now = wp.array(self._gravity, dtype=wp.vec3d, device=device)
            self._gravity_last = wp.clone(self._gravity_now)
            self._gravity_restart = wp.zeros(n_sub, dtype=wp.int32, device=device)
            subscene = np.full(model.world_count + 1, -1, np.int32)
            if self._worlds is None:
                subscene[self._world_entries] = 0
            else:
                subscene[self._worlds.worlds] = np.arange(1, self._worlds.count)
                subscene[-1] = 0
            self._gravity_subscene = wp.array(subscene, dtype=wp.int32, device=device)
        if self._independent:  # the entry of every independent world, and a scratch word for each world entry
            self._status_entries_device = wp.array(self._status_entries, dtype=wp.int32, device=device)
            self._world_words = wp.zeros(model.world_count + 1, dtype=wp.int32, device=device)
        if self._sticky_status is not None and self._sticky_status.shape[0] != self._kept_words():
            self._allocate_kept_status()  # an entry of a coupled solver whose rebuilt solve has other worlds
        # the particles whose new state from state_in discards no friction: the fixed ones (the engine's flags), and
        # a coupled solver's proxy particles, whose state the coupler writes at every step
        self._particle_quiet = None
        if self._proxy_particle.any():
            quiet = (self._fixed_particle | self._proxy_particle).astype(np.int32)
            self._particle_quiet = wp.array(quiet, dtype=wp.int32, device=device)
        self._status_scratch = wp.zeros(1, dtype=wp.int32, device=device)
        self._residual_scratch = wp.zeros(1, dtype=wp.float32, device=device)
        # the triangles and tetrahedra whose activations in a step's control an eager step checks (_check_activations)
        own_tris, own_tets, _ = _own_elements(model)
        self._activation_tris = wp.array(np.flatnonzero(own_tris).astype(np.int32), dtype=wp.int32, device=device)
        self._activation_tets = wp.array(np.flatnonzero(own_tets).astype(np.int32), dtype=wp.int32, device=device)
        self._activations_found = wp.zeros(2, dtype=wp.int32, device=device)
        nb = model.body_count
        if nb:
            self._body_base = wp.array(layout.body_base.astype(np.int32), dtype=wp.int32, device=device)
            self._body_kinematic = wp.array(self._kinematic_body.astype(np.int32), dtype=wp.int32, device=device)
            # the bodies whose new state from state_in discards no friction: the kinematic ones, and a coupled
            # solver's proxy bodies
            self._body_quiet = self._body_kinematic
            if self._proxy_body.any():
                quiet = (self._kinematic_body | self._proxy_body).astype(np.int32)
                self._body_quiet = wp.array(quiet, dtype=wp.int32, device=device)
            # kinematic_motion="aim": the bodies whose state the solver keeps (held); and what state_out takes from
            # state_in (_store_bodies): the pose and the twist of a kinematic body that jumps (1), the twist of one
            # moving toward its aims (2), whose pose is the one it reached
            self._body_hold = wp.zeros(nb, dtype=wp.int32, device=device)
            self._body_passthrough = self._body_kinematic
            if len(self._aim_bodies):
                through = self._kinematic_body.astype(np.int32)
                through[self._aim_bodies] = 2
                self._body_passthrough = wp.array(through, dtype=wp.int32, device=device)
            self._aim_rows = wp.array(self._aim_bodies.astype(np.int32), dtype=wp.int32, device=device)
            # the twist of every such body that it had as a dynamic body before a notification made it kinematic,
            # which its aims do not take (NaN: none; _drop_residual_twists)
            self._aim_residual = wp.array(
                np.full((len(self._aim_bodies), 6), np.nan, np.float32), dtype=wp.spatial_vector, device=device
            )
            entry = np.where(model.body_world.numpy() < 0, model.world_count, model.body_world.numpy())
            self._body_entry = wp.array(entry.astype(np.int32), dtype=wp.int32, device=device)
            self._joint_wrench = wp.zeros(nb, dtype=wp.spatial_vector, device=device)
            self._body_q_written = wp.empty(nb, dtype=wp.transform, device=device)
            self._body_qd_written = wp.empty(nb, dtype=wp.spatial_vector, device=device)
            self._forget_written()
            # body_rotation="rigid": the second moment about the centre of mass of every body, from which the stored
            # angular velocity is that of the body's angular momentum (_store_body_momentum)
            self._body_spread = None
            if self._rigid_rotation:
                self._body_spread = wp.array(_spreads(self._dyadic), dtype=wp.mat33d, device=device)
        if self._proxy_body.any():  # the body of every surface vertex, for the proxy feedback (-1 elsewhere)
            vertex_body = np.full(layout.n, -1, np.int32)
            for b in range(nb):
                vertex_body[layout.body_vertex_offsets[b] : layout.body_vertex_offsets[b + 1]] = b
            self._vertex_body = wp.array(vertex_body, dtype=wp.int32, device=device)
        self._make_engine(None)
        if self._restarts:  # an entry of a coupler that restarts its steps: the new solve's copies
            self._allocate_restart()

    def _check_proxies(self) -> None:
        """``NotImplementedError`` for the options a coupled solver's proxies cannot take (:ref:`ipc-coupling`): BDF2,
        whose history a proxy's state, written by the coupler at every step, would leave behind, and AL-IPC, which has
        no contact forces to harvest the proxies' feedback from."""
        if self._config.integrator == "bdf2":
            raise NotImplementedError(
                "SolverIPC: integrator='bdf2' is not offered to an entry of a coupled solver with proxies. The coupler "
                "writes a proxy's state at every step, so that the step would start a proxy's BDF2 history anew every "
                "time; use integrator='bdf1'."
            )
        if self._config.contact_constitution == "al-ipc":
            raise NotImplementedError(
                "SolverIPC: contact_constitution='al-ipc' is not offered to an entry of a coupled solver with proxies: "
                "the proxies' feedback is the force of the step's contact terms on them, which AL-IPC does not export."
            )

    def _check_independent_options(self) -> None:
        """``NotImplementedError`` for the options that independent worlds do not take yet (:ref:`ipc-worlds`), each
        naming what to use instead, when two worlds or more hold unknowns: libuipc-warp solves the worlds apart from
        each other with libuipc's barrier contact and the block-diagonal preconditioner."""
        worlds = self._worlds.worlds
        which = f"the worlds {worlds.tolist()}" if len(worlds) <= 8 else f"{len(worlds)} worlds"
        instead = (
            f"or pass coupled_worlds=True to solve {which} as one (see Worlds in Newton's IPC solver guide), or give "
            "the solver a model of one world"
        )
        if self._config.contact_constitution == "al-ipc":
            raise NotImplementedError(
                "SolverIPC: contact_constitution='al-ipc' is not offered with independent worlds yet: libuipc-warp "
                f"solves {which} apart from each other with libuipc's barrier contact only, AL-IPC's frame being one "
                f"solve of the whole scene. Use contact_constitution='ipc', {instead}."
            )
        if self._config.fem_preconditioner == "mas":
            raise NotImplementedError(
                "SolverIPC: fem_preconditioner='mas' is not offered with independent worlds yet: MAS's clusters may "
                f"join vertices of different worlds, whose linear solves are apart. Use fem_preconditioner='diag', "
                f"{instead}."
            )

    def _forget_written(self) -> None:
        """Forget what the solver wrote for the bodies and the joints, so that the next step takes their state from
        ``state_in`` (NaN equals nothing)."""
        nb = self.model.body_count
        if nb:
            self._body_q_written.assign(np.full((nb, 7), np.nan, np.float32))
            self._body_qd_written.assign(np.full((nb, 6), np.nan, np.float32))
            self._body_hold.zero_()  # the kinematic bodies moving toward their aims take state_in too
        if self._engine is not None:  # and the kinematic bodies, particles and proxies are placed there, not swept
            self._engine.place_kinematic()
        if self.model.joint_count:
            self._joint_q_written.fill_(np.nan)

    def _make_engine(self, capacity: dict | None) -> None:
        """The engine of the laid-out scene, with the current gravity and masses; ``capacity``: its list
        capacities (``SimEngine``'s argument; None for its defaults)."""
        device = self.model.device
        inputs = self._scene.engine_inputs(self._config, device)
        self._forces = None
        if len(self._force_blocks):
            self._forces = self._uipc_warp.arrays.ExternalAcceleration("newton_forces", self._force_blocks, device)
            inputs["elements"].append(self._forces)
        self._engine = self._uipc_warp.make_engine(
            self._config, self._options, **inputs, device=device, capacity=capacity
        )
        if not np.array_equal(self._gravity, self._layout_gravity):
            # notify_model_changed changed gravity since the scene's parts were laid out with theirs: an engine built
            # again with longer lists (_regrow) takes the gravity of now
            self._engine.set_gravity(_vertex_gravity(self))
        # the Newton and PCG iterations of the captured steps since the last graph_outdated()
        self._iterations = wp.zeros(2, dtype=wp.int64, device=device)
        # what the start check needs of the tabular, its two tables apart as the engine tests them: the contact
        # element and the subscene of every vertex and half plane, the elements' mask and the subscenes' mask
        self._vertex_thickness = np.asarray(inputs["thickness"], np.float64)
        tabular = inputs["contact_tabular"]
        n, n_planes = len(inputs["x"]), len(inputs["plane_P"])
        if tabular is None:
            self._cids, self._scids = np.zeros(n, np.int64), np.zeros(n, np.int64)
            self._plane_cids, self._plane_scids = np.zeros(n_planes, np.int64), np.zeros(n_planes, np.int64)
            self._enabled, self._subscene_enabled = np.ones((1, 1), bool), np.ones((1, 1), bool)
        else:
            self._cids, self._scids = (np.asarray(tabular[k], np.int64) for k in ("cids", "scids"))
            self._plane_cids, self._plane_scids = (
                np.asarray(tabular[k], np.int64) for k in ("plane_cids", "plane_scids")
            )
            self._enabled = np.asarray(tabular["enabled"]) != 0
            self._subscene_enabled = np.asarray(tabular["subscene_enabled"]) != 0
        if self.model.body_count and self._config.contact_constitution != "al-ipc":
            # notify_model_changed may have changed them since the scene was laid out (AL-IPC: it builds it again)
            self._engine.set_dyadic_mass(*_engine_dyadic(self, self._dyadic))
        if self._proxies and self._engine.vm.swept is not None:
            # a coupled solver's proxies: unknowns whose state the coupler writes at every step, whose moves the step
            # sweeps as it sweeps the kinematic ones (docs/solvers/ipc.rst > "Coupling")
            swept, layout = self._engine.vm.swept.numpy(), self._scene.layout()
            swept[: self.model.particle_count][self._proxy_particle] = 1
            for b in np.flatnonzero(self._proxy_body):
                swept[layout.body_vertex_offsets[b] : layout.body_vertex_offsets[b + 1]] = 1
            self._engine.vm.swept.assign(swept)
        self._joint_groups = self._uipc_warp.arrays.joint_groups(self._engine)
        self._aim_group = self._uipc_warp.arrays.constraint_groups(self._engine).transform  # None without aims
        # the stiffness of every stitch group and the attachment or spring of each of its rows (_stitch_stiffness);
        # the groups of the distance joints' stitches, which join two bodies and so never share a group with an
        # attachment or a spring, apart (_distance_stiffness)
        self._stitch_rows, self._distance_rows = [], []
        for g in self._uipc_warp.arrays.stitch_groups(self._engine):
            distance = g.rows >= self._distance_first
            if not distance.any():
                self._stitch_rows.append(
                    (g.kappa, wp.array(self._stitch_source[g.rows], dtype=wp.int32, device=device))
                )
            elif distance.all():
                rows = wp.array((g.rows - self._distance_first).astype(np.int32), dtype=wp.int32, device=device)
                self._distance_rows.append((g.kappa, g.rest_length, rows))
            else:
                raise RuntimeError("SolverIPC: a stitch group holds distance joints and attachments or springs.")
        if self._distance_rows:
            self._distance_joints_device = wp.array(self._distance_joints, dtype=wp.int32, device=device)
        _find_body_stiffness(self)
        _make_contact_export(self)

    # ------------------------------------------------------------------ SolverBase
    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
        *,
        observables: SolverBase.Observables | None = None,
    ) -> None:
        self.validate_observables(observables, contacts)
        if self._engine is None:
            _write_contacts(self, observables, None)  # no particles and no bodies: no contact
            return
        if self._start_unchecked and self.model.device.is_capturing:
            self._start_in_capture()  # before anything is captured
        if self._restarts:  # an entry of a coupler that restarts its steps (docs/solvers/ipc.rst > "Coupling")
            if self._restore_pending:
                self._restore_step()
            elif self._save_pending:
                self._save_step()
            self._save_pending = self._restore_pending = False
        if control is None:
            control = self._control_of_model()
        if not self.model.device.is_capturing:
            self._check_activations(control)
        self._dt, self._control = dt, control
        self._report_clamp(dt)
        self._load(state_in)
        if self._start_unchecked:
            # the first step, not captured: the start of the pairs with a body whose start the caller writes, on the
            # state just loaded from state_in, before anything runs from it (_check_start)
            try:
                _check_start(self, "the state of the first step (its state_in)")
            except ValueError:
                if self.model.body_count:
                    # nothing ran, and the next step is the first again: the kinematic bodies moving toward their
                    # aims take its state_in too, not the poses this step loaded (_aim_kinematic_bodies held them)
                    self._body_hold.zero_()
                raise
            self._start_unchecked = False
        self._steps += 1
        self._write_forces(state_in, control)
        _write_joint_controls(self, control, dt)
        _write_stitch_stiffness(self, dt)
        _write_body_stiffness(self, dt)
        if self.model.device.is_capturing:
            # the solve joins the caller's graph; it cannot synchronize, grow or raise (docs/solvers/ipc.rst > "Status")
            self._engine.stop_on_overflow = False
            self._engine.advance(dt)
            self._uipc_warp.arrays.add_frame_iterations(self._engine, self._iterations)
            self._keep_status()
        else:
            status = self._advance_eager(dt)
            self._keep_status()
            if status.failed and self._failed_alone(status):
                # libuipc-warp put the worlds that failed back at their state at the step's start and stepped the
                # others: state_out holds both, then the step raises, naming the worlds that failed
                failed = self._failed_worlds()
                _shift_joint_history(self)
                self._store(state_in.body_q, state_in.body_qd, state_out)
                self._write_observables(observables, state_out)
                raise RuntimeError(self._world_failure(failed))
            if status.failed:
                self._write_observables(observables, state_out)
                self._forget_written()  # the engine holds the failed solve: the next step takes state_in
                raise RuntimeError(self._failure(status))
        _shift_joint_history(self)
        self._store(state_in.body_q, state_in.body_qd, state_out)
        self._write_observables(observables, state_out)

    def _start_in_capture(self) -> None:
        """The first step, captured in a graph: it cannot read its ``state_in``, so the pairs with a body whose start
        the caller writes keep the test of the construction, on the model's state; raise its failure, if any, before
        anything is captured (``_check_start``)."""
        if self._start_pending is not None:
            raise ValueError(
                f"{self._start_pending} The first step is captured in a graph, which cannot read its state_in, so the "
                "solver cannot check the start of kinematic bodies and of links that the model leaves at the identity "
                "on it, as a first step outside capture does. Take the first step outside graph capture, or put the "
                "start poses into model.body_q before building the solver."
            )
        self._start_unchecked = False

    def _control_of_model(self) -> Control:
        """The control of a step given none: the model's own control inputs, ``model.control(clone_variables=False)``
        built once, and again when the model holds other arrays for the inputs the solver reads."""
        model, control = self.model, self._default_control
        if control is None or any(
            getattr(control, name) is not getattr(model, name)
            for name in ("joint_target_q", "joint_target_qd", "joint_f")
        ):
            control = self._default_control = model.control(clone_variables=False)
        return control

    def _check_activations(self, control: Control) -> None:
        """A step that is not captured: ``NotImplementedError`` when ``control`` holds an activation other than 0 of a
        triangle or tetrahedron the solver simulates, which it does not read (one kernel, one read-back of two
        words). A captured step cannot raise, and does not check them."""
        n_tri, n_tet = self._activation_tris.shape[0], self._activation_tets.shape[0]
        tris = control.tri_activations if n_tri else None
        tets = control.tet_activations if n_tet else None
        if tris is None and tets is None:
            return
        self._activations_found.zero_()
        wp.launch(
            _nonzero_activations,
            n_tri + n_tet,
            inputs=[self._activation_tris, tris, self._activation_tets, tets],
            outputs=[self._activations_found],
            device=self.model.device,
        )
        found = self._activations_found.numpy()
        names = [name for name, k in (("tri_activations", 0), ("tet_activations", 1)) if found[k]]
        if names:
            raise NotImplementedError(
                f"SolverIPC: the step's control has non-zero {' and '.join(names)}; set them to 0. SolverIPC does not "
                "read activations."
            )

    def _report_clamp(self, dt: float) -> None:
        """Warn once when a step's ``dt`` clamps the given ``kappa`` into libuipc's stiffness range, with independent
        worlds into the range of a world (each world has its own)."""
        kappa = self._config.kappa
        if not self._adaptive_kappa or self._clamp_reported or kappa is None or self._engine is None:
            return
        if dt == self._clamp_checked_dt:
            return
        self._clamp_checked_dt = dt
        policies = self._engine.world_contact_stiffness
        if policies is None:
            _, used, lo, hi, _ = self._engine.contact_stiffness.resolve(dt)
            if used != kappa:
                self._clamp_reported = True
                warnings.warn(
                    f"SolverIPC: kappa={kappa:g} is outside libuipc's contact stiffness range [{lo:.6g}, {hi:.6g}] Pa "
                    f"of this model at dt={dt:g} s; the steps use {used:.6g} Pa, as libuipc clamps a stiffness it is "
                    "given. The range scales with the mean vertex mass and with 1/dt^2 (contact_stiffness(dt) gives "
                    "the stiffness of any dt); kappa=None takes its lower bound, libuipc's default. Not repeated.",
                    NewtonWarning,
                    stacklevel=3,
                )
            return
        clamped = []
        for world, policy in zip(self._status_entries.tolist(), policies, strict=True):
            _, used, lo, hi, _ = policy.resolve(dt)
            if used != kappa:
                clamped.append(f"world {world} uses {used:.6g} Pa (range [{lo:.6g}, {hi:.6g}] Pa)")
        if clamped:
            self._clamp_reported = True
            shown = "; ".join(clamped[:4]) + (f"; and {len(clamped) - 4} more worlds" if len(clamped) > 4 else "")
            warnings.warn(
                f"SolverIPC: kappa={kappa:g} is outside libuipc's contact stiffness range of {len(clamped)} of the "
                f"{len(policies)} independent worlds at dt={dt:g} s, each world's range that of the world alone, and "
                f"those worlds' steps use their range's bound, as libuipc clamps a stiffness it is given: {shown}. "
                "A range scales with the world's mean vertex mass and with 1/dt^2 (contact_stiffness(dt) gives every "
                "world's stiffness of any dt); kappa=None takes every world's lower bound, libuipc's default. Not "
                "repeated.",
                NewtonWarning,
                stacklevel=3,
            )

    def _load(self, state_in: State) -> None:
        """The engine's state from ``state_in`` where it differs from what the solver wrote (:ref:`ipc-state`)."""
        model, eng = self.model, self._engine
        device, vm = model.device, eng.vm
        self._changed.zero_()
        if model.particle_count:
            quiet = vm.fixed if self._particle_quiet is None else self._particle_quiet
            wp.launch(
                _load_particles,
                model.particle_count,
                inputs=[state_in.particle_q, state_in.particle_qd, quiet, self._particle_subscene],
                outputs=[vm.x, vm.v, self._changed],
                device=device,
            )
        if model.body_count:
            wp.launch(
                _load_bodies,
                model.body_count,
                inputs=[
                    state_in.body_q,
                    state_in.body_qd,
                    self._body_q_written,
                    self._body_qd_written,
                    model.body_com,
                    self._body_base,
                    self._body_quiet,
                    self._body_hold,
                    self._body_subscene,
                ],
                outputs=[vm.x, vm.v, self._changed],
                device=device,
            )
            self._uipc_warp.arrays.update_body_vertices(eng)  # surface vertices x = A x_bar + p
            self._aim_bodies_of(state_in)
            _load_joints(self, state_in)
        # the subscenes of the particles and bodies that took state_in discard their friction, and under BDF2 their
        # step is an implicit Euler step (BDF1: nothing); with several worlds the other worlds go on as they were
        eng.discard_friction(where=self._changed)
        eng.restart_history(where=self._changed)
        if self._gravity_now is not None:
            # BDF2: so is the step of the subscenes whose gravity changed since the step before (docs/solvers/ipc.rst >
            # "BDF2")
            self._gravity_restart.zero_()
            wp.launch(
                _gravity_changed,
                self._gravity_now.shape[0],
                inputs=[self._gravity_now, self._gravity_subscene],
                outputs=[self._gravity_last, self._gravity_restart],
                device=device,
            )
            eng.restart_history(where=self._gravity_restart)

    def _aim_bodies_of(self, state_in: State) -> None:
        """``kinematic_motion="aim"``: the aims of the kinematic bodies for the step, from ``state_in`` (the poses
        advanced by the twists), with the velocities that put the step's prediction on them (``_aim_kinematic_bodies``);
        ``dt`` is the step's, frozen in a captured step as the joints' kernels freeze it."""
        if self._aim_group is None:
            return
        eng = self._engine
        bdf2 = self._config.integrator == "bdf2"
        wp.launch(
            _aim_kinematic_bodies,
            len(self._aim_bodies),
            inputs=[
                self._aim_rows,
                state_in.body_q,
                state_in.body_qd,
                self.model.body_com,
                self._body_base,
                eng.vm.x,
                wp.float64(self._dt),
                int(bdf2),
            ],
            outputs=[
                self._aim_group.q_aim,
                self._aim_group.active,
                eng.vm.v,
                eng.vm.x_nm1,
                eng.vm.v_nm1,
                self._body_hold,
                self._aim_residual,
            ],
            device=self.model.device,
        )

    def _drop_residual_twists(self, became: np.ndarray, written: np.ndarray | None) -> None:
        """``kinematic_motion="aim"``, after a notification that made the bodies ``became`` kinematic and built the
        solve again: such a body moving toward its aims does not take the twist the solver wrote for it as a dynamic
        body (``written``, the ``_body_qd_written`` of the old solve), which ``state_in`` still holds until the caller
        writes another; its aims take a zero twist until then (``_aim_kinematic_bodies``), as Newton's other solvers
        leave a body that becomes kinematic where it is."""
        if written is None or not len(self._aim_bodies):
            return
        rows = np.flatnonzero(became[self._aim_bodies])
        if not len(rows):
            return
        residual = self._aim_residual.numpy()
        residual[rows] = written[self._aim_bodies[rows]]
        self._aim_residual.assign(residual)

    def _write_forces(self, state_in: State, control: Control) -> None:
        """The external accelerations of the step, on the device."""
        if self._forces is None:
            return
        model, acc = self.model, self._forces.a
        if self._n_free:
            particle_f = state_in.particle_f
            wp.launch(
                _particle_accelerations,
                self._n_free,
                inputs=[particle_f, model.particle_mass, self._free_particles],
                outputs=[acc],
                device=model.device,
            )
        if self._n_dynamic:
            self._joint_wrench.zero_()
            if model.joint_count and control.joint_f is not None and self._wrenched_joints.shape[0]:
                wp.launch(
                    _joint_wrenches,
                    self._wrenched_joints.shape[0],
                    inputs=[
                        self._wrenched_joints,
                        self._joint_store_type,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_X_p,
                        model.joint_qd_start,
                        model.joint_enabled,
                        control.joint_f,
                        state_in.body_q,
                        model.body_com,
                    ],
                    outputs=[self._joint_wrench],
                    device=model.device,
                )
            if model.joint_count and control.joint_f is not None and self._d6_joints.shape[0]:
                wp.launch(
                    _d6_joint_wrenches,
                    self._d6_joints.shape[0],
                    inputs=[
                        self._d6_joints,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_X_p,
                        model.joint_X_c,
                        model.joint_axis,
                        model.joint_dof_dim,
                        model.joint_qd_start,
                        model.joint_enabled,
                        control.joint_f,
                        state_in.body_q,
                        model.body_com,
                    ],
                    outputs=[self._joint_wrench],
                    device=model.device,
                )
            body_f = state_in.body_f
            wp.launch(
                _body_accelerations,
                self._n_dynamic,
                inputs=[
                    body_f,
                    self._joint_wrench,
                    self._dynamic_bodies,
                    model.body_com,
                    self._body_mass_inv,
                    self._body_base,
                    self._engine.vm.x,
                    self._n_free,
                ],
                outputs=[acc],
                device=model.device,
            )

    def _store(self, body_q_in, body_qd_in, state_out: State | None) -> None:
        """``state_out`` from the engine's state; without one, only the record of what the solver writes."""
        model, vm = self.model, self._engine.vm
        if model.particle_count and state_out is not None:
            wp.launch(
                _store_particles,
                model.particle_count,
                inputs=[vm.x, vm.v],
                outputs=[state_out.particle_q, state_out.particle_qd],
                device=model.device,
            )
        if model.body_count:
            body_q = self._body_q_written if state_out is None else state_out.body_q
            body_qd = self._body_qd_written if state_out is None else state_out.body_qd
            wp.launch(
                _store_bodies,
                model.body_count,
                inputs=[vm.x, vm.v, self._body_base, model.body_com, self._body_passthrough, body_q_in, body_qd_in],
                outputs=[body_q, body_qd, self._body_q_written, self._body_qd_written],
                device=model.device,
            )
            if self._body_spread is not None:  # body_rotation="rigid": the angular velocity of the angular momentum
                wp.launch(
                    _store_body_momentum,
                    model.body_count,
                    inputs=[vm.x, vm.v, self._body_base, self._body_passthrough, self._body_spread],
                    outputs=[body_qd, self._body_qd_written],
                    device=model.device,
                )
            if self._al_sweep:
                wp.launch(
                    _forget_kinematic_written,
                    model.body_count,
                    inputs=[self._body_kinematic],
                    outputs=[self._body_q_written, self._body_qd_written],
                    device=model.device,
                )
        if model.joint_count and state_out is not None and state_out.joint_q is not None:
            groups = self._joint_groups
            angle = None if groups.revolute is None else groups.revolute.current
            distance = None if groups.prismatic is None else groups.prismatic.current
            wp.launch(
                _store_joints,
                model.joint_count,
                inputs=[
                    self._joint_store_type,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_X_p,
                    model.joint_X_c,
                    model.joint_axis,
                    model.joint_q_start,
                    model.joint_qd_start,
                    self._joint_element,
                    angle,
                    distance,
                    state_out.body_q,
                    state_out.body_qd,
                    model.body_com,
                ],
                outputs=[state_out.joint_q, state_out.joint_qd, self._joint_q_written],
                device=model.device,
            )
            if self._d6_joints.shape[0]:
                wp.launch(
                    _store_d6_joints,
                    self._d6_joints.shape[0],
                    inputs=[
                        self._d6_joints,
                        model.joint_parent,
                        model.joint_child,
                        model.joint_X_p,
                        model.joint_X_c,
                        model.joint_axis,
                        model.joint_dof_dim,
                        model.joint_q_start,
                        model.joint_qd_start,
                        state_out.body_q,
                        state_out.body_qd,
                        model.body_com,
                    ],
                    outputs=[state_out.joint_q, state_out.joint_qd, self._joint_q_written],
                    device=model.device,
                )

    def _advance_eager(self, dt: float):
        """The solve outside a caller's capture, its status read back; on an overflow the frame again on an engine
        with longer lists (libuipc-warp's FrameRunner, in its order)."""
        status = self._solve_frame(dt)
        for _ in range(_MAX_REGROWS):
            if not status.overflowed:
                break
            self._regrow()
            status = self._solve_frame(dt)
        if not status.overflowed:
            self._engine_outdated()  # the next frame is captured anew when the solves or the size ladders ask
        return status

    def _engine_outdated(self, iterations=None) -> bool:
        """``engine.graph_outdated``: whether a frame captured from now on would hold a better PCG loop or better
        guesses of the size ladders. When it would, the engine takes the ladders' guesses for that capture
        (``prepare_capture``, as libuipc-warp's ``FrameRunner`` does before it captures) and the solver's own graph of
        the solve is dropped. ``iterations``: the sums of the captured steps (:meth:`graph_outdated`)."""
        if not self._engine.graph_outdated(iterations=iterations):
            return False
        self._engine.prepare_capture()
        self._frame_graph = None
        return True

    def _solve_frame(self, dt: float):
        """One frame as a graph of its own (captured at the first frame of an engine), and how it ended."""
        eng = self._engine
        eng.set_dt(dt)
        # an overflowing frame is done again, so its Newton loop may end at the overflow
        eng.stop_on_overflow = True
        if self._frame_graph is None:
            gc_enabled = gc.isenabled()
            gc.disable()  # a collection inside the capture would record the release of other arrays
            try:
                with wp.ScopedCapture(device=self.model.device) as capture:
                    eng.advance()
            finally:
                if gc_enabled:
                    gc.enable()
            self._frame_graph = capture.graph
        wp.capture_launch(self._frame_graph)
        return self._uipc_warp.arrays.frame_status(eng)

    def _regrow(self) -> None:
        """A new engine whose contact lists hold what the last frame asked for, at the start of that frame, with its
        element groups' state (the joints' angles) and the PCG loop the steps so far chose (libuipc-warp's
        ``redo_state``, as its ``FrameRunner`` hands them over), and with the frame's external accelerations and
        aims."""
        old = self._engine
        capacity = old.grown_capacity()
        redo = old.redo_state()
        # an entry of a coupler that restarts its steps: the start of the coupled step, which the new engine keeps
        restart = None if old.frame_restart is None else old.frame_restart.saved()
        forces = None if self._forces is None else self._forces.a
        aims = None if self._aim_group is None else self._aim_group
        # release the old engine before the new one allocates (the contact export holds it too)
        self._engine = self._frame_graph = self._forces = self._contact_export = old = None
        self._make_engine(capacity)
        self._engine.load_redo_state(redo)
        if restart is not None:
            self._engine.enable_frame_restart().load_saved(restart)
        if forces is not None:
            wp.copy(self._forces.a, forces)
        if aims is not None:  # the aims of the step, which the new engine solves again
            wp.copy(self._aim_group.q_aim, aims.q_aim)
            wp.copy(self._aim_group.active, aims.active)
            wp.copy(self._aim_group.strength, aims.strength)
        _write_joint_controls(self, self._control or self._control_of_model(), self._dt)
        _write_stitch_stiffness(self, self._dt)
        _write_body_stiffness(self, self._dt)

    def _failed_alone(self, status) -> bool:
        """Whether a failed step failed in some independent worlds alone (``status``: the step's ``frame_status``):
        libuipc-warp's frame of worlds puts a world that fails on a value libuipc asserts on (a non-finite value, a CCD
        fraction that is not positive, a pair at or inside its thickness) back at its frame start, and the others take
        their step. An exhausted traversal is every world's, the traversal being shared; an overflow fails only the
        world over its quota of the contact lists, but an eager step grows that quota and solves the step again, so
        one that is left after the last enlargement is the solver's failure, as without independent worlds."""
        return (
            self._independent
            and len(self._status_entries) > 1
            and not (status.overflowed or status.traversal_exhausted)
        )

    def _world_failure(self, failed: dict[int, int]) -> str:
        """Why a step failed in some of the independent worlds (``_failed_worlds``), and what ``state_out`` holds."""
        return (
            f"SolverIPC: the step failed in {len(failed)} of its {len(self._status_entries)} worlds, "
            f"{self._name_worlds(failed)}. Those worlds are back at their state at the step's start, with their "
            "friction discarded and their BDF2 history started anew, and every other world took the step: state_out "
            "holds both. Surfaces must start apart (outside their contact thickness), and the state must stay finite."
        )

    def _name_worlds(self, failed: dict[int, int]) -> str:
        """``failed`` (bits by world entry) as ``world 1 on THICKNESS_VIOLATED and world 3 on ...``, the first eight."""
        names = [
            ("the global world" if k == self.model.world_count else f"world {k}")
            + " on "
            + "|".join(bit.name for bit in self.StatusBits if bit & self.StatusBits.FAILED and bit & failed[k])
            for k in sorted(failed)
        ]
        if len(names) > 8:
            names = [*names[:8], f"{len(names) - 8} more"]
        return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]

    def _failure(self, status) -> str:
        """Why a step failed, with what libuipc would print."""
        if status.overflowed:
            return f"SolverIPC: the contact lists still overflow after {_MAX_REGROWS} enlargements."
        if status.traversal_exhausted:
            return "SolverIPC: a BVH traversal ran out of its node budget, so contact candidates are missing."
        report = self._engine.nonfinite_report()
        name = self._bit_names(1 << status.nonfinite)
        detail = report.get("message", "")
        if not detail:
            detail = ", ".join(f"{k}={v}" for k, v in report.items() if k not in ("kind",))
        return (
            f"SolverIPC: the step failed on {name} in Newton iteration {report['newton']}: {detail}. "
            "Surfaces must start apart (outside their contact thickness), and the state must stay finite."
        )

    def _write_observables(self, observables: SolverBase.Observables | None, state_out: State | None) -> None:
        if observables is None:
            return
        _write_contacts(self, observables, state_out)
        status = observables.status if observables.is_requested(self.ObservableKind.STATUS) else None
        residual = (
            observables.solver_residual if observables.is_requested(SolverBase.ObservableKind.SOLVER_RESIDUAL) else None
        )
        if status is None and residual is None:
            return
        if self._independent:  # every world its own, the entries of the global world and of worlds without unknowns 0
            self._write_world_status(status, residual)
            return
        if len(self._world_entries) <= 2:
            for entry in self._world_entries:
                self._uipc_warp.arrays.write_frame_status(self._engine, status, residual, index=int(entry))
            return
        # many worlds (coupled_worlds): the one solve's status written once, then copied to every entry
        self._uipc_warp.arrays.write_frame_status(self._engine, self._status_scratch, self._residual_scratch)
        wp.launch(
            _copy_status,
            len(self._world_entries),
            inputs=[
                self._world_entries_device,
                self._status_scratch,
                self._residual_scratch,
            ],
            outputs=[status, residual],
            device=self.model.device,
        )

    def _write_world_status(self, flags, residual) -> None:
        """With independent worlds: every world's status bits and Newton residual into its entry of ``flags`` and
        ``residual`` (libuipc-warp's ``write_frame_status`` with ``worlds``, the record of each world), or the one
        solve's when one world holds unknowns, the engine then being that world's alone; nothing when none does. The
        entries of the global world and of worlds without unknowns are not written: they take no step."""
        entries = self._status_entries
        if len(entries) > 1:
            self._uipc_warp.arrays.write_frame_status(self._engine, flags, residual, worlds=self._status_entries_device)
        elif len(entries) == 1:
            self._uipc_warp.arrays.write_frame_status(self._engine, flags, residual, index=int(entries[0]))

    def _failed_worlds(self) -> dict[int, int]:
        """With independent worlds, after a step that is not captured: the status bits of every world whose step
        failed, by world (reads every world's word back; one synchronization)."""
        self._world_words.zero_()
        self._write_world_status(self._world_words, None)
        words = self._world_words.numpy()
        failed = int(self.StatusBits.FAILED)
        return {int(k): int(words[k]) for k in np.flatnonzero(words & failed)}

    def grow(self) -> None:
        """Enlarge the contact lists after a captured step reported :attr:`SolverIPC.StatusBits.OVERFLOW`.

        A captured step cannot enlarge them: it drops the pairs that do not fit, sets the bit and keeps it set.
        This builds the solve again with lists that hold what the last step asks for (solving that step again
        until it fits, as an eager step does) and returns to the state the last step started from, so that the
        same ``state_in`` repeats it. With independent worlds the quotas of the worlds that went over theirs grow,
        and the other worlds keep theirs. The new solve has new arrays: capture every graph that holds
        :meth:`step` again. Synchronizes; does nothing when no list overflowed.
        """
        if self._engine is None:
            return
        if self.model.device.is_capturing:
            raise RuntimeError("SolverIPC.grow() synchronizes with the device; call it outside graph capture.")
        if not self._uipc_warp.arrays.frame_status(self._engine).overflowed:
            return
        if self._config.integrator == "bdf2":  # the joints' history of the last step, which is solved again
            for _, q_prev, q_kept in (*self._joint_history.values(), *self._joint_rate.values()):
                wp.copy(q_prev, q_kept)
        for _ in range(_MAX_REGROWS):
            self._regrow()
            if not self._solve_frame(self._dt).overflowed:
                break
        else:
            raise RuntimeError(f"SolverIPC: the contact lists still overflow after {_MAX_REGROWS} enlargements.")
        self._engine.load_redo_state(self._engine.redo_state())
        if self.model.body_count:
            # what the solver wrote for the restored state, so that the step's own state_in is kept as it was
            self._store(self._body_q_written, self._body_qd_written, None)

    def graph_outdated(self) -> bool:
        """Whether graphs that hold :meth:`step` would run faster captured again.

        The PCG loop of a step re-reads its stopping test after a fixed number of iterations, 8 or 32, and work whose
        size only the device knows (the matrix conversion, the sort of body pairs) runs on a ladder of sizes that
        starts at a guess. Both are chosen when the step is enqueued: a captured step keeps what it was captured with.
        A step that is not captured chooses for itself (libuipc-warp's ``SimEngine.graph_outdated``), from the steps
        before. For captured steps, this takes the Newton and PCG iterations of every captured step replayed since the
        last call, which the steps count on the device, and the sizes the ladders took, and returns ``True`` when
        they ask for the other number or for other guesses; the engine then takes the guesses
        (``SimEngine.prepare_capture``), and the steps captured from then on take both. The graphs captured before
        stay valid and give the same results; only the loop and the ladders cost more. Call it outside graph
        capture, after replays (after :meth:`check_status`, for example); a caller that never calls it keeps 8 and
        no guesses. Reads the two counters and the ladders' counts back: two or three synchronizations.

        Returns:
            ``True`` when graphs that hold :meth:`step` should be captured again.
        """
        if self._engine is None:
            return False
        if self.model.device.is_capturing:
            raise RuntimeError("SolverIPC.graph_outdated() reads from the device; call it outside graph capture.")
        newton, pcg = self._iterations.numpy().tolist()
        self._iterations.zero_()
        return self._engine_outdated(iterations=(newton, pcg))

    def check_status(self, observables: SolverBase.Observables | None = None, *, grow: bool = False) -> bool:
        """Raise when a step that wrote ``observables`` failed, and warn when one reached an iteration limit.

        A step captured in a CUDA graph cannot raise: it only writes how it ended into the observables passed to it (see
        :ref:`ipc-status`). Call this after replaying the graph, outside the capture, once for each container the
        replayed steps wrote. It reads the container back from the device (one synchronization).

        A coupled solver steps its entries without observables. The solver as an entry (:ref:`ipc-coupling`) keeps the
        status bits of every step it takes, captured or not, until a call without a container reads them: that call
        reports every step since the last such call, and clears the bits. A pass of a coupled step that the coupler
        solves again is replaced by the next pass: a coupled step reports its last pass.

        Args:
            observables: A container from :meth:`observables` with :attr:`SolverIPC.ObservableKind.STATUS`
                requested, as passed to :meth:`step`. With ``SOLVER_RESIDUAL`` requested as well, the messages give
                the residual. ``None`` for the solver as an entry of a coupled solver: the bits it kept.
            grow: On :attr:`SolverIPC.StatusBits.OVERFLOW`, call :meth:`grow` and return ``True`` instead of
                raising, whatever other bits are set: the steps ran without some contact pairs, which can cause the
                other failures, and the repeated steps report again.

        Returns:
            ``True`` when a contact list overflowed and ``grow`` enlarged the lists: the steps ran without some
            contact pairs, so restore the state they started from, capture every graph that holds :meth:`step`
            again and repeat them. ``False`` otherwise.

        Raises:
            RuntimeError: When a bit of :attr:`SolverIPC.StatusBits.FAILED` is set and ``grow`` does not take an
                ``OVERFLOW``: the step's result must not be used. With independent worlds the bits are every world's
                own, and the message names the worlds that failed, whose results alone must not be used.
            ValueError: When ``observables`` does not hold the ``STATUS`` observable, or is ``None`` and the solver
                is no entry of a coupled solver.

        Warns:
            NewtonWarning: When :attr:`SolverIPC.StatusBits.NEWTON_LIMIT` or
                :attr:`SolverIPC.StatusBits.LINE_SEARCH_LIMIT` is set. Neither fails the step.
        """
        if self.model.device.is_capturing:
            raise RuntimeError("SolverIPC.check_status() reads from the device; call it outside graph capture.")
        residual = ""
        if observables is None and self._sticky_status is not None:  # an entry of a coupled solver: its kept bits
            words = self._sticky_status.numpy()
            self._sticky_status.zero_()
        elif observables is None or not observables.is_requested(self.ObservableKind.STATUS):
            raise ValueError(
                "SolverIPC.check_status() needs observables with ObservableKind.STATUS requested, or none for an "
                "entry of a coupled solver."
            )
        else:
            words = observables.status.numpy()
            if observables.is_requested(SolverBase.ObservableKind.SOLVER_RESIDUAL):
                residual = (
                    f" (residual {float(observables.solver_residual.numpy().max()):.3g}, at most 1 when converged)"
                )
        word = int(np.bitwise_or.reduce(words))
        known = int(self.StatusBits.KNOWN)
        bits = self.StatusBits(word & known)
        if bits & self.StatusBits.OVERFLOW and grow:
            self.grow()
            return True
        failed = bits & self.StatusBits.FAILED
        if failed:
            causes = []
            if failed & self.StatusBits.OVERFLOW:
                causes.append(
                    "A contact list overflowed and the step ran without some contact pairs: call grow() (or "
                    "check_status(..., grow=True)), restore the state the steps started from, capture again and repeat "
                    "them."
                )
            if failed & (self.StatusBits.THICKNESS_VIOLATED | self.StatusBits.INVALID_CCD_STEP):
                causes.append(
                    "A contact pair started at or inside its contact thickness, as surfaces in touching contact or "
                    "intersecting do (in a state written between steps, for example): surfaces must start apart."
                )
            if failed & (
                self.StatusBits.NONFINITE_ENERGY
                | self.StatusBits.NONFINITE_PCG_INIT
                | self.StatusBits.NONFINITE_PCG_ITER
            ):
                causes.append(
                    "The solve met a non-finite value: the state, the forces and the parameters must be finite."
                )
            if failed & self.StatusBits.TRAVERSAL_EXHAUSTED:
                causes.append("A BVH traversal ran out of its node budget, so contact candidates are missing.")
            if self._independent:  # every world its own bits: the worlds that failed, the others took their step
                worlds = {int(k): int(words[k]) for k in np.flatnonzero(words & int(self.StatusBits.FAILED))}
                which = self._name_worlds(worlds)
                if len(worlds) < len(self._status_entries):
                    subject = f"in {which}{residual}; the result of those worlds must not be used, the other worlds"
                    subject += " took their step"
                else:
                    subject = f"in every world ({which}){residual}; its result must not be used"
                raise RuntimeError(f"SolverIPC: a step failed {subject}. " + " ".join(causes))
            last = self._uipc_warp.arrays.frame_status(self._engine)
            if last.nonfinite and failed & self.StatusBits(1 << last.nonfinite):
                report = self._engine.nonfinite_report()
                detail = report.get("message") or ", ".join(f"{k}={v}" for k, v in report.items() if k != "kind")
                causes.append(f"The last step failed in Newton iteration {report['newton']}: {detail}.")
            names = "|".join(bit.name for bit in self.StatusBits if bit & failed)
            raise RuntimeError(
                f"SolverIPC: a step failed on {names}{residual}; its result must not be used. " + " ".join(causes)
            )
        limits = []
        if bits & self.StatusBits.NEWTON_LIMIT:
            limits.append(f"the Newton loop stopped at newton_max_iter without converging{residual}")
        if bits & self.StatusBits.LINE_SEARCH_LIMIT:
            limits.append("a line search ran out of its line_search_max_iter trials")
        if word & ~known:
            limits.append(f"libuipc-warp set {self._bit_names(word & ~known)}, which StatusBits do not name")
        if limits:
            warnings.warn(
                f"SolverIPC: {' and '.join(limits)}; this does not fail the step.", NewtonWarning, stacklevel=2
            )
        return False

    def _bit_names(self, bits: int) -> str:
        """``bits`` of a status mask by name: ``StatusBits``' names, and libuipc-warp's (its ``FrameStatusFlags``) for
        the bits of a later libuipc-warp, or ``1 << k``."""
        names = {int(flag): flag.name for flag in self._uipc_warp.arrays.FrameStatusFlags}
        names.update({int(bit): bit.name for bit in self.StatusBits})
        return "|".join(names.get(1 << k, f"1 << {k}") for k in range(bits.bit_length()) if bits >> k & 1)

    @override
    def reset(
        self,
        state: State,
        world_mask: wp.array[wp.bool] | None = None,
        flags: StateFlags | int | None = None,
    ) -> None:
        """Make the next step start as the first step of a new solver.

        Besides its float64 state, which the next step compares with ``state_in`` (see :ref:`ipc-state`), the solver
        carries the friction of the last step's contacts from one step to the next, and with ``integrator="bdf2"`` the
        integrator's history; this discards them, so that the next step is an implicit Euler step without friction. The
        contacts themselves are reused only where nothing moved, where they are the ones a new solver finds. Launched on
        the device without synchronization, so it can be captured.

        Args:
            state: Unused; the next step reads its own ``state_in``.
            world_mask: Optional boolean mask of shape ``(world_count + 1,)``, the last entry the global entities.
                With several worlds the reset applies to the worlds it selects: their friction is discarded and their
                BDF2 history starts anew, and every other world keeps its own, bit for bit. With independent worlds
                (the default) the last entry selects nothing, since the global entities are no unknowns, as Newton
                lets a solver without global dynamic objects take it; with ``coupled_worlds`` it discards the friction
                of every pair with a global entity, a global ground plane among them (see :ref:`ipc-worlds`). With
                one world the reset applies when the mask selects a world that owns particles or bodies. ``None``
                resets.
            flags: Unused.
        """
        super().reset(state, world_mask, flags)  # checks world_mask
        self._restore_pending = False  # the next step starts from the reset, not from a coupled step's start
        if self._engine is None:
            return
        if world_mask is None:
            self._engine.discard_friction()
            self._engine.restart_history()
            # the next step places the kinematic bodies and particles and the proxies at their state_in poses, without
            # a sweep
            self._engine.place_kinematic()
            if self.model.body_count:
                self._body_hold.zero_()
            return
        if len(self._aim_bodies):
            wp.launch(
                _release_held,
                len(self._aim_bodies),
                inputs=[world_mask, self._body_entry, self._aim_rows],
                outputs=[self._body_hold],
                device=self.model.device,
            )
        if self._worlds is None:
            wp.launch(
                _any_selected,
                1,
                inputs=[world_mask, self._world_entries_device],
                outputs=[self._selected],
                device=self.model.device,
            )
        else:  # the subscenes of the selected worlds (libuipc-warp's requests per subscene)
            wp.launch(
                _subscenes_selected,
                self._worlds.count,
                inputs=[world_mask, self._subscene_entry, int(self._independent)],
                outputs=[self._selected],
                device=self.model.device,
            )
        self._engine.discard_friction(where=self._selected)
        self._engine.restart_history(where=self._selected)
        self._engine.place_kinematic(where=self._selected)

    @override
    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        """Take changed model properties.

        Gravity, of every world (``MODEL_PROPERTIES``), and the bodies' mass, centre of mass and inertia
        (``BODY_INERTIAL_PROPERTIES``), with the joints' penalties, which follow the masses, are updated in place:
        graphs captured before take them. With ``integrator="bdf2"`` the next step starts the history of the worlds
        whose gravity changed anew (see :ref:`ipc-bdf2`). A change of the triangulated shapes (``SHAPE_PROPERTIES`` with
        new transforms or scales, ``SHAPE_TYPE_AND_SOURCE``), of the friction coefficients and the contact thicknesses
        (``SHAPE_PROPERTIES`` with a new ``shape_material_mu`` or ``shape_margin``, ``MODEL_PROPERTIES`` with a new
        ``soft_contact_mu``, ``particle_mu`` or ``particle_radius``, or a ``particle_grid`` set to None or back), of the particles' masses (``MODEL_PROPERTIES``
        with a new ``particle_mass``, as a coupled solver gives its proxy particles their masses), of the springs
        (``MODEL_PROPERTIES`` with a new ``spring_rest_length``, ``spring_stiffness`` or ``spring_damping``), of the
        joints' types, bodies, frames or axes (``JOINT_PROPERTIES``) or of which bodies are kinematic
        (``BODY_PROPERTIES``, ``BODY_INERTIAL_PROPERTIES``) builds the solve again: the next step reads ``state_in`` in
        full, and graphs that hold :meth:`step` must be captured again. The joints' other data are read at every step;
        the joint flags only check them again against what :ref:`ipc-joints` rejects and, once per solver, warn of
        velocity limits below Newton's default (:ref:`ipc-not-read`). So are the stiffness and the
        enabled flags of the particle-body attachments (``CONSTRAINT_PROPERTIES``), which that flag checks again against
        what :ref:`ipc-attachments` rejects; a new body, particle or point of an attachment builds the solve again.
        ``CONSTRAINT_PROPERTIES`` also checks the mimic constraints and MuJoCo's equality rows again, so that a row
        enabled since the solver was built that no loop joint holds is refused (:ref:`ipc-joints`). With
        ``kinematic_motion="aim"`` a change of a kinematic body's mass properties builds the solve again too, since its
        soft transform constraint follows the masses it was built with. Every notification, whatever its flags, compares
        the arrays the elements were built from that Newton has no flag of its own for: ``tet_materials``,
        ``tri_materials``, ``edge_bending_properties``, the rest shapes ``tet_poses``, ``tri_poses`` and
        ``edge_rest_angle``, and ``particle_flags``; a change of any builds the solve again. Copies from the host: not inside a graph
        capture. With ``contact_constitution="al-ipc"`` new masses build the solve again too, since the AL-IPC penalty
        follows the masses it was built with.
        """
        if self.model.device.is_capturing:
            raise RuntimeError("SolverIPC.notify_model_changed() copies from the host; call it outside graph capture.")
        if self._engine is None:  # nothing moves
            return
        model = self.model
        rebuild = bool(flags & ModelFlags.SHAPE_TYPE_AND_SOURCE)
        # the elements' materials and rest shapes and the particles' flags, which no flag names, at every notification
        rebuild |= any(
            (a is None) != (b is None) or (a is not None and not np.array_equal(a, b))
            for a, b in zip(_element_inputs_now(self), self._element_inputs, strict=True)
        )
        if flags & (ModelFlags.SHAPE_PROPERTIES | ModelFlags.MODEL_PROPERTIES):
            now = _scene_inputs_now(self)
            rebuild |= len(now) != len(self._scene_inputs) or any(
                not np.array_equal(a, b) for a, b in zip(now, self._scene_inputs, strict=True)
            )
        if flags & ModelFlags.JOINT_PROPERTIES:
            now = _joint_inputs_now(self)
            rebuild |= len(now) != len(self._joint_inputs) or any(
                not np.array_equal(a, b) for a, b in zip(now, self._joint_inputs, strict=True)
            )
        dof_flags = (
            ModelFlags.JOINT_DOF_PROPERTIES
            | ModelFlags.JOINT_DOF_FORCE_PROPERTIES
            | ModelFlags.JOINT_DOF_INERTIAL_PROPERTIES
            | ModelFlags.JOINT_REFERENCE_POSE_PROPERTIES
        )
        if flags & dof_flags:
            _check_joints(model, self._joint_options)  # the other joint data are read at every step
            _warn_of_velocity_limits(self, stacklevel=3)
            if model.joint_count:  # a distance joint gains or loses its length
                rebuild |= not np.array_equal(_map_joints(model).distance, self._distance_joints)
        if flags & ModelFlags.CONSTRAINT_PROPERTIES:
            _check_constraints(model)  # an equality row enabled since, a mimic relation set since
        if flags & ModelFlags.CONSTRAINT_PROPERTIES and model.attachment_particle_body_count:
            _check_attachments(model)  # their stiffness and enabled flags are read at every step
            body, particle, point = self._attachment_inputs or (None, None, None)
            rebuild |= (
                body is None
                or not np.array_equal(model.attachment_particle_body_body.numpy(), body)
                or not np.array_equal(model.attachment_particle_body_particle.numpy(), particle)
                or not np.array_equal(model.attachment_particle_body_body_point.numpy(), point)
            )
        kinematic = self._kinematic_body
        if flags & (ModelFlags.BODY_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES) and model.body_count:
            dyadic = _body_mass_moments(self, stacklevel=3)
            rebuild |= not np.array_equal(self._kinematic_body, kinematic)
            # kinematic_motion="aim": a kinematic body that gains or loses its mass starts or stops moving toward its
            # aims
            rebuild |= not np.array_equal(_aiming_bodies(self, dyadic[0]), self._aim_bodies)
            changed = any(not np.array_equal(a, b) for a, b in zip(dyadic, self._dyadic, strict=True))
            if self._config.contact_constitution == "al-ipc":
                # the AL-IPC contact penalty follows the masses the engine was built with
                rebuild |= changed
            if changed and len(self._aim_bodies):
                # kinematic_motion="aim": the soft transform constraint of a kinematic body follows the masses the
                # engine was built with
                aims = self._aim_bodies
                rebuild |= any(not np.array_equal(a[aims], b[aims]) for a, b in zip(dyadic, self._dyadic, strict=True))
        if rebuild:
            # what the solver wrote for the bodies, whose twists a body that becomes kinematic does not take
            written = self._body_qd_written.numpy() if model.body_count else None
            _check_supported(model, self._joint_options)
            self._build_engine()
            if written is not None:
                self._drop_residual_twists(self._kinematic_body & ~kinematic, written)
            return
        if flags & ModelFlags.MODEL_PROPERTIES:
            self._gravity = _world_gravity(self)
            self._config.gravity = tuple(self._gravity[-1])
            self._engine.set_gravity(_vertex_gravity(self))
            if self._gravity_now is not None:  # BDF2: the next step compares it with the gravity of the step before
                self._gravity_now.assign(self._gravity)
        if flags & ModelFlags.BODY_INERTIAL_PROPERTIES and model.body_count:
            self._dyadic = dyadic
            self._engine.set_dyadic_mass(*_engine_dyadic(self, dyadic))
            self._body_mass_inv.assign(_mass_inverse(self, *dyadic))
            if self._body_spread is not None:  # and the angular velocity that body_rotation="rigid" stores
                self._body_spread.assign(_spreads(dyadic))
            for kind in self._joint_bodies:  # the joints' penalties follow the masses
                self._joint_strength[kind].assign(_joint_strengths(self, kind, dyadic[0])[1])
            self._attachment_hard.assign(_attachment_penalties(self, dyadic[0]))  # and so do the hard attachments'
            self._distance_hard.assign(_distance_penalties(self, dyadic[0]))  # and the distance joints'
            if self._ortho is not None:  # and the default body stiffness the inertia
                self._ortho_inertia.assign(_ortho_ratios(self))

    # ------------------------------------------------------------------ CouplingInterface
    @override
    def coupling_notify_iteration_restarts(self, enabled: bool) -> None:
        """Become an entry of a coupled solver (:ref:`ipc-coupling`): keep the status bits of every step for
        :meth:`check_status`, and when the coupler may solve a step again from its start, keep that start on the device.
        Outside graph capture: the coupler calls this once, after construction.

        Args:
            enabled: Whether the coupler may restart a step's solve: a ``SolverCoupledProxy`` with more than one
                iteration per step, or a ``SolverCoupledADMM`` with more than one. The solver then allocates the
                copy of a step's start (libuipc-warp's frame restart, and what the solver carries from step to step).
        """
        if self._sticky_status is None:
            self._allocate_kept_status()
        self._restarts = bool(enabled)
        if self._restarts:
            self._allocate_restart()

    @override
    def coupling_notify_input_state_update(
        self,
        state: State,
        flags: StateFlags | int,
        *,
        iteration_restart: bool = False,
        dt: float = 0.0,
    ) -> None:
        """Take note of a coupled solver's write of the entry's input state (:ref:`ipc-coupling`).

        Every step compares ``state_in`` with what the solver wrote last and takes what differs (:ref:`ipc-state`), a
        proxy's pose and twist without discarding friction, so a write needs nothing of its own. When the coupler
        restarts steps (:meth:`coupling_notify_iteration_restarts`), the next step after a write (``dt`` above 0) keeps
        its start on the device before it takes ``state_in``, and the next step after a restart's notification
        (``iteration_restart`` with ``dt`` above 0) starts from that start again instead: device copies, also inside a
        graph capture. A reset's notification (``dt`` 0) needs nothing: the coupled solver's ``reset`` calls
        :meth:`reset` next, with its world mask, which discards the friction and releases the kinematic aims of the
        worlds it selects.

        Raises:
            NotImplementedError: For a restart's notification from a coupler that did not announce restarts through
                :meth:`coupling_notify_iteration_restarts`: the solver keeps no start to restart from.
        """
        del state, flags
        if dt <= 0.0:
            return
        if not self._restarts:
            if iteration_restart:
                raise NotImplementedError(
                    "SolverIPC cannot solve a step again from its start: the coupler did not announce restarts "
                    "(coupling_notify_iteration_restarts(True)), so the solver keeps no start to restart from."
                )
            return
        if iteration_restart:
            self._restore_pending = True
        else:
            self._save_pending = True

    @override
    def coupling_supports_full_surface_soft_contacts(self) -> bool:
        """``True``: the solver reads no :class:`~newton.Contacts`, of whatever kind, and finds its own contacts."""
        return True

    @override
    def coupling_supports_particle_body_attachments(self) -> bool:
        """``True``: the solver applies the particle-body attachments of its model view as libuipc's soft vertex
        stitch, a hard one with a ball joint's penalty (:ref:`ipc-attachments`)."""
        return True

    @override
    def coupling_prepare_proxy_contacts(
        self,
        state: State,
        contacts: Contacts | None,
        *,
        contacts_freshly_detected: bool = False,
    ) -> Contacts | None:
        """Return ``contacts`` as they are: the solver reads no :class:`~newton.Contacts`, and the contacts of a proxy
        with another proxy or with a static shape do not exist in its own (:ref:`ipc-coupling`)."""
        del state, contacts_freshly_detected
        return contacts

    @override
    def coupling_harvest_proxy_wrenches(
        self,
        body_local_to_proxy_global: wp.array[int],
        out_body_f: wp.array[wp.spatial_vector],
        *,
        body_qd_before: wp.array[wp.spatial_vector],
        state: State,
        state_out: State,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """The proxy bodies' feedback of the step, from its contact forces (:ref:`ipc-coupling`).

        Zeroes ``out_body_f`` and adds, at every proxy body's index, the force at the end of the step of every contact
        term on the body's vertices, with its torque about the body's centre of mass in ``state_out``: the force this
        entry's content exerts on the proxy, as SolverVBD harvests it, rather than its change of momentum. Device work
        only; call it after :meth:`step`.
        """
        del body_qd_before, state, contacts, dt
        out_body_f.zero_()
        if self._engine is None or not self._proxy_body.any() or body_local_to_proxy_global.shape[0] == 0:
            return
        export = _contact_records(self)
        wp.launch(
            _proxy_wrenches,
            self._contact_threads,
            inputs=[
                export.vertices,
                export.forces,
                export.counts,
                export.segment_offsets,
                self._contact_threads,
                self._engine.vm.x,
                self._vertex_body,
                body_local_to_proxy_global,
                state_out.body_q,
                self.model.body_com,
            ],
            outputs=[out_body_f],
            block_dim=_CONTACT_ROW_BLOCK,
            device=self.model.device,
        )

    @override
    def coupling_harvest_proxy_particle_forces(
        self,
        particle_local_to_proxy_global: wp.array[int],
        out_particle_f: wp.array[wp.vec3],
        *,
        particle_qd_before: wp.array[wp.vec3],
        state: State,
        state_out: State,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """The proxy particles' feedback of the step, from its contact forces (:ref:`ipc-coupling`): zeroes
        ``out_particle_f`` and adds, at every proxy particle's index, the force at the end of the step of every contact
        term on the particle. Device work only; call it after :meth:`step`."""
        del particle_qd_before, state, state_out, contacts, dt
        out_particle_f.zero_()
        if self._engine is None or not self._proxy_particle.any() or particle_local_to_proxy_global.shape[0] == 0:
            return
        export = _contact_records(self)
        wp.launch(
            _proxy_particle_forces,
            self._contact_threads,
            inputs=[
                export.vertices,
                export.forces,
                export.counts,
                export.segment_offsets,
                self._contact_threads,
                self.model.particle_count,
                particle_local_to_proxy_global,
            ],
            outputs=[out_particle_f],
            block_dim=_CONTACT_ROW_BLOCK,
            device=self.model.device,
        )

    def _kept_words(self) -> int:
        """The status words an entry of a coupled solver keeps: one per world entry with independent worlds, else
        one, the solve's."""
        return self.model.world_count + 1 if self._independent else 1

    def _allocate_kept_status(self) -> None:
        """The status words an entry of a coupled solver keeps for :meth:`check_status` (``_kept_words``), zeroed,
        and the scratch words of one step. Outside graph capture."""
        device = self.model.device
        self._sticky_status = wp.zeros(self._kept_words(), dtype=wp.int32, device=device)
        self._step_status = wp.zeros(self._kept_words(), dtype=wp.int32, device=device)

    def _keep_status(self) -> None:
        """An entry of a coupled solver: the step's status bits joined into those kept for :meth:`check_status`, with
        independent worlds every world's into its entry."""
        if self._sticky_status is None:
            return
        if self._independent:
            self._step_status.zero_()
            self._write_world_status(self._step_status, None)
        else:
            self._uipc_warp.arrays.write_frame_status(self._engine, self._step_status)
        wp.launch(
            _keep_status,
            self._step_status.shape[0],
            inputs=[self._step_status],
            outputs=[self._sticky_status],
            device=self.model.device,
        )

    def _allocate_restart(self) -> None:
        """An entry of a coupler that restarts its steps: the engine keeps its frame start on the device
        (libuipc-warp's ``enable_frame_restart``), and the solver a copy of what it carries from one step to the next:
        what it wrote of the bodies and the joints, the kinematic bodies it holds toward their aims, the joints'
        history and the gravity of the step before under BDF2, and the status bits it keeps for :meth:`check_status`,
        so that a restarted step replaces the bits of the pass before it. Outside graph capture."""
        self._carried = []
        if self._engine is None:
            return
        self._engine.enable_frame_restart()
        carried = [] if self._sticky_status is None else [self._sticky_status]
        if self.model.body_count:
            carried += [self._body_q_written, self._body_qd_written, self._body_hold]
        if self.model.joint_count:
            carried.append(self._joint_q_written)
        for _, q_prev, q_kept in (*self._joint_history.values(), *self._joint_rate.values()):
            carried += [q_prev, q_kept]
        if self._gravity_last is not None:  # BDF2: the gravity of the step before, which a change of gravity restarts
            carried.append(self._gravity_last)
        self._carried = [(a, wp.clone(a)) for a in carried if a.size]

    def _save_step(self) -> None:
        """Keep the start of a coupled step: the engine's frame start and what the solver carries (device copies)."""
        self._engine.save_frame_start()
        for live, kept in self._carried:
            wp.copy(kept, live)

    def _restore_step(self) -> None:
        """Start the step again where the last ``_save_step`` kept it (device copies)."""
        self._engine.restore_frame_start()
        for live, kept in self._carried:
            wp.copy(live, kept)

    def contact_stiffness(self, dt: float) -> float | np.ndarray:
        """The contact stiffness [Pa] of every contact pair in a step of length ``dt`` [s]: ``kappa``, or with
        ``adaptive_kappa`` ``kappa`` clamped into libuipc's stiffness range for this model and ``dt``, the range's lower
        bound when ``kappa`` is None (see :ref:`ipc-contact`). 0 for a model without particles and bodies, which has no
        contact.

        With ``adaptive_kappa`` and independent worlds every world has a range of its own, that of the world alone (see
        :ref:`ipc-worlds`), and the stiffness is one value per world, a float64 array of ``model.world_count``: world
        ``k``'s pairs take entry ``k``, and a world in which nothing moves, which has no contact, 0."""
        if self._engine is None:
            return 0.0
        policies = self._engine.world_contact_stiffness
        if policies is not None:  # every world's own range (libuipc-warp's independent worlds)
            out = np.zeros(self.model.world_count, np.float64)
            out[self._status_entries] = [policy.resolve(dt)[1] for policy in policies]
            return out
        policy = self._engine.contact_stiffness
        return float(self._config.kappa if policy is None else policy.resolve(dt)[1])

    def frame_stats(self) -> dict:
        """Solver counters of the last step: Newton/PCG iterations, line-search trials, convergence.

        With independent worlds the counts are those of the step's loops, which run as long as a world is in them: the
        Newton iterations of the slowest world. Synchronizes with the device.
        """
        return {} if self._engine is None else self._engine.frame_stats()
