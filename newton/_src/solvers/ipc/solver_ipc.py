# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Incremental Potential Contact (IPC) solver backed by libuipc-warp."""

from __future__ import annotations

import gc
import importlib.metadata
import inspect
import operator
import re
import warnings
from dataclasses import dataclass
from enum import IntFlag
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import CouplingInterface

from newton import Contacts, Control, Model, ModelFlags, NewtonWarning, State, StateFlags
from newton.solvers import SolverBase

from .contacts import (
    _CONTACT_ROW_BLOCK,
    _check_al_ipc,
    _check_surface_velocity,
    _contact_elements,
    _contact_records,
    _make_contact_export,
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
    _attachment_penalties,
    _body_mass_moments,
    _check_attachments,
    _check_independent_content,
    _check_springs,
    _collect_planes,
    _default_d_hat,
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
    _with_aim_moments,
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


# libuipc-warp releases the solver is written for: from the first to before the second
_UIPC_WARP_RELEASES = ((0, 4), (0, 5))
# grow-and-redo attempts of an eager step whose contact lists keep overflowing, as libuipc-warp's FrameRunner
_MAX_REGROWS = 8
# libuipc's strength ratio of a joint (the default of its joints' apply_to): the default joint_strength_ratio
_JOINT_STRENGTH_RATIO = 100.0


# A copy of Newton's normalize_reset_world_mask (newton/_src/core/reset.py), which SolverBase.reset calls and which has
# no public name. test_reset_checks_world_mask_as_newton_does fails when the two check a mask differently.
def _normalize_reset_world_mask(
    world_mask: wp.array[wp.bool] | None, *, world_count: int, device: wp.Device
) -> wp.array[wp.bool] | None:
    """Validate a reset mask with shape ``(world_count + 1,)``.

    The final entry selects global entities in world ``-1``. Returns the mask
    unchanged, or ``None`` if no mask was given.
    """
    if world_mask is None:
        return None
    if not isinstance(world_mask, wp.array):
        raise TypeError("'world_mask' must be a Warp array or None.")
    if world_mask.dtype != wp.bool:
        raise TypeError("'world_mask' must have dtype bool.")
    if world_mask.ndim != 1:
        raise ValueError("'world_mask' must be one-dimensional.")
    if world_mask.device != device:
        raise ValueError(f"'world_mask' device {world_mask.device} does not match expected device {device}.")
    mask_size = world_mask.shape[0]
    if mask_size != world_count + 1:
        raise ValueError(f"'world_mask' length {mask_size} must equal model.world_count + 1 ({world_count + 1}).")
    return world_mask


def _release(version: str) -> tuple[int, ...]:
    """The release numbers at the start of a version string (``"0.4.0.dev0"`` gives ``(0, 4, 0)``)."""
    match = re.match(r"\d+(?:\.\d+)*", version)
    return tuple(int(part) for part in match.group(0).split(".")) if match else ()


_VERSION_OPERATORS = {
    ">=": operator.ge,
    ">": operator.gt,
    "<=": operator.le,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}


def _satisfies(version: str, specifier: str) -> bool:
    """Whether ``version`` meets every clause of a requirement specifier such as ``>=1.19,<1.20``, comparing
    release numbers only (a pre-release counts as its release)."""
    have = _release(version)
    for clause in specifier.split(","):
        match = re.fullmatch(r"\s*(~=|>=|<=|==|!=|>|<)\s*([0-9][0-9.]*)\S*\s*", clause)
        if match is None:
            continue
        op, bound = match.group(1), _release(match.group(2))
        width = max(len(have), len(bound))
        a, b = have + (0,) * (width - len(have)), bound + (0,) * (width - len(bound))
        ok = a >= b and have[: len(bound) - 1] == bound[:-1] if op == "~=" else _VERSION_OPERATORS[op](a, b)
        if not ok:
            return False
    return True


def _version_mismatches(uipc_warp_version: str, warp_version: str, warp_requirement: str | None) -> list[str]:
    """The installed versions outside the range this solver is written for, and Warp outside libuipc-warp's."""
    low, high = _UIPC_WARP_RELEASES
    out = []
    if not low <= _release(uipc_warp_version) < high:
        out.append(
            f"libuipc-warp {uipc_warp_version} (SolverIPC is written for libuipc-warp >={'.'.join(map(str, low))},"
            f"<{'.'.join(map(str, high))})"
        )
    if warp_requirement and not _satisfies(warp_version, warp_requirement):
        out.append(f"warp-lang {warp_version} (libuipc-warp requires warp-lang{warp_requirement})")
    return out


def _warp_requirement() -> str | None:
    """libuipc-warp's requirement on warp-lang, from its installed metadata (None without one)."""
    try:
        requirements = importlib.metadata.requires("libuipc-warp") or []
    except importlib.metadata.PackageNotFoundError:
        return None
    for requirement in requirements:
        match = re.match(r"warp-lang\s*\(?([^;)]*)\)?", requirement)
        if match:
            return match.group(1).strip() or None
    return None


def _check_choice(name: str, value: str, accepted: tuple) -> None:
    """``ValueError`` unless ``value`` is one of ``accepted``, the values of libuipc's config key that ``name`` sets."""
    if value not in accepted:
        raise ValueError(f"SolverIPC: {name} must be one of {', '.join(map(repr, accepted))}, got {value!r}.")


def _check_supported(model: Model, joints: _JointOptions | None = None) -> None:
    """Reject model content the solver would otherwise accept and ignore; joints as ``joints`` maps them
    (``_JointOptions``). What independent worlds refuse is checked when the solve is laid out
    (``SolverIPC._check_independent_options``, ``_check_independent_content``)."""
    if model.requires_grad:
        raise NotImplementedError("SolverIPC does not support gradients (model.requires_grad=True).")
    for name, count in (
        ("tendons", model.tendon_count + model.custom_frequency_counts.get("mujoco:tendon", 0)),
        ("muscles", model.muscle_count),
    ):
        if count:
            raise NotImplementedError(f"SolverIPC does not simulate {name}; the model has {count}.")
    mimic = int(model.constraint_mimic_count)
    if model.joint_count and model.joint_mimic_joint is not None:
        mimic += int(np.count_nonzero(model.joint_mimic_joint.numpy() >= 0))
    if mimic:
        raise NotImplementedError(f"SolverIPC does not enforce mimic constraints; the model has {mimic}.")
    mujoco = getattr(model, "mujoco", None)
    rows = int(getattr(mujoco, "equality_constraint_count", 0) or 0) if mujoco is not None else 0
    if rows:
        enabled = getattr(mujoco, "equality_constraint_enabled", None)
        rows = int(np.count_nonzero(enabled.numpy())) if enabled is not None else rows
    if rows:
        raise NotImplementedError(
            f"SolverIPC does not enforce equality constraints; model.mujoco has {rows} enabled equality rows."
        )
    own_tris, own_tets, _ = _own_elements(model)
    for name, activations, own in (
        ("triangle", model.tri_activations, own_tris),
        ("tetrahedron", model.tet_activations, own_tets),
    ):
        if activations is not None and len(activations) and np.any(activations.numpy()[own] != 0.0):
            raise NotImplementedError(f"SolverIPC does not take {name} activations; the model has non-zero ones.")
    _check_joints(model, joints)
    _check_springs(model)
    _check_attachments(model)
    _check_surface_velocity(model)


class SolverIPC(SolverBase, CouplingInterface):
    """Incremental Potential Contact solver for soft bodies, cloth and rigid bodies.

    Wraps the `libuipc-warp <https://github.com/OrangeEarth15/libuipc-warp>`_ engine,
    a Warp implementation of libuipc, through its array interface (no libuipc
    build is needed). Each :meth:`step` is one implicit time step, by default
    of implicit Euler (BDF1): a Newton iteration with a backtracking line
    search, a block-Jacobi PCG linear solve, barrier contact and continuous
    collision detection. ``integrator="bdf2"``,
    ``contact_constitution="al-ipc"`` and ``fem_preconditioner="mas"`` take
    libuipc's BDF2, its augmented-Lagrangian contact and its MAS
    preconditioner instead (see **BDF2**, **AL-IPC contact** and **MAS**
    below). Under implicit Euler the rotation of a rigid body is by default
    integrated as a rigid body's, where libuipc's step moves the body along a
    straight line in its affine coordinates and damps its spin
    (``body_rotation``, see **Rigid bodies**). The Newton loop, the PCG loop
    and the line search run on the device under ``wp.capture_while``, so a
    step never synchronizes with the host when it is captured in a CUDA
    graph. A step that is not captured captures the solve itself and checks
    how it ended (see below).

    Supported model content:

    - particles (a particle on no cloth or tetrahedron is a sphere of its
      radius, see **Contact**) and tetrahedra. Elasticity is the SNK1 stable Neo-Hookean
      model with Lamé parameters ``tet_materials[:, 0]`` (``k_mu``) and
      ``tet_materials[:, 1]`` (``k_lambda``). ``k_damp`` is ignored.
    - cloth: triangles of :attr:`~newton.Model.tri_indices` that are not on a
      tetrahedral body become libuipc's strain-limiting Baraff-Witkin membrane,
      and bending edges of :attr:`~newton.Model.edge_indices` become
      discrete-shell hinges (see the mapping below).
    - springs between particles: libuipc's Hookean springs with Newton's
      energy :math:`\\tfrac{1}{2} k_e (l - L_0)^2` (``spring_stiffness``,
      ``spring_rest_length``), and libuipc's soft vertex stitch for a rest
      length of 0. A spring adds no contact primitive: the particles it joins
      collide as other particles do.
    - particle-body attachments: libuipc's soft vertex stitch between the
      particle and a point of the body (see **Attachments** below).
    - rigid bodies with box, sphere, capsule, cylinder, cone, ellipsoid or mesh
      collision shapes become affine bodies (see below).
    - contact against static plane shapes, each treated as an infinite half
      space, and against other static shapes, which become fixed triangle
      meshes; a height field with Newton's collision triangles, two per cell
      of its grid (an open surface, as Newton's).
    - contact between all of these: point-triangle and edge-edge barriers with
      continuous collision detection, between the pairs that Newton's
      collision groups, worlds, collide flags and filter pairs let collide,
      with the friction of the two materials (see **Contact** below). Each
      connected group of tetrahedra or cloth triangles is one body and
      collides with itself (as finite elements do in libuipc), so ``d_hat``
      must stay below the spacing of non-adjacent surface primitives at rest,
      as the default ``d_hat`` does (see the note below). A rigid body does
      not collide with itself.
    - kinematic particles (zero mass or not :attr:`~newton.ParticleFlags.ACTIVE`)
      and kinematic bodies (:attr:`~newton.BodyFlags.KINEMATIC`, zero mass
      or zero inverse mass, see **Coupling**).
      By default they follow the poses written to ``state_in`` every step; the
      jump is not swept by collision detection. With
      ``kinematic_motion="aim"`` kinematic bodies move inside the step toward
      the pose that ``state_in`` prescribes for its end, swept like every other
      motion (see **Kinematic bodies** below). A particle that is not active
      and on no cloth or tetrahedron collides with nothing, as in Newton.
    - revolute, prismatic, fixed and ball joints between bodies or to the
      world, with drives, passive springs and damping, limits and
      :attr:`~newton.Control.joint_f` (see **Joints** below); ``joint_q`` and
      ``joint_qd`` are written at every step. With ``enable_d6_joints``, D6
      joints that lock what one of these locks, or nothing; with
      ``enable_distance_joints``, distance joints of one length or none.
    - external forces: :attr:`~newton.State.particle_f`, :attr:`~newton.State.body_f`
      and :attr:`~newton.Control.joint_f`, held constant over the step as
      libuipc holds its external forces (a constant acceleration in the
      implicit-Euler prediction). A body wrench acts at the centre of mass;
      its torque does work on the rotations of the affine body and none on its
      stretch.
    - gravity per world: a particle or body falls with the
      :attr:`~newton.Model.gravity` of its world, a global one with the
      global world's (the last entry).
    - lagged friction on every contact.

    Not supported, and rejected when the solver is constructed:
    :class:`NotImplementedError` for springs and particle-body attachments
    with damping (libuipc has no damping element), springs of a negative
    rest length or stiffness or that join a particle to itself, mimic
    and equality constraints, tendons, muscles, non-zero triangle or
    tetrahedron activations, the joint types and properties that **Joints**
    below rejects,
    gradients (``model.requires_grad``), with independent worlds what
    **Worlds** below refuses (an unknown of the global world, an element that
    joins two worlds, and the options they do not take yet), a
    :attr:`~newton.Model.shape_contact_pairs` array assigned to the model, more
    contact elements than the engine takes (see **Contact** below),
    collision meshes with a surface velocity (``Mesh.enable_surface_velocity``,
    which the solver's own contacts would ignore), and a CPU device.
    Gaussian shapes and planes attached to bodies raise when the shapes are
    triangulated. The ``contacts`` argument of
    :meth:`step` is not read: the solver detects its own contacts, and
    writes them into it when ``CONTACT_F`` is requested (see **Contact
    output** below).

    Not read: the penalty contact parameters of shapes and particles
    (``ke``, ``kd``, ``kf``, ``ka``, ``particle_cohesion`` and
    ``particle_adhesion``: ``kappa`` holds for every pair), restitution,
    torsional and rolling friction, ``shape_material_mu_static``,
    ``shape_gap`` (``d_hat`` is the solver's detection distance), the
    :attr:`~newton.Model.particle_radius` of cloth and tetrahedron vertices
    (see **Contact** below), damping coefficients,
    ``spring_control``, drag, lift, ``edge_rest_length`` (see below), the
    activations of a :class:`~newton.Control`, ``joint_target_mode``,
    ``joint_velocity_limit``, ``joint_limit_kd`` (but with
    ``enable_joint_limit_damping``) and ``joint_act``.

    **Cloth parameters.** libuipc-warp's membrane energy per rest area is

    .. math::

        \\Psi = K_a \\phi(|F a|) + K_b \\phi(|F b|) + K_g ((F a) \\cdot (F b))^2,
        \\qquad \\phi(s) = (s - 1)^2 + \\eta \\max(s - 1, 0)^3,

    with the material axes ``a``, ``b`` of each triangle's rest frame. Newton
    integrates its membrane energy over the rest area only, so
    ``tri_materials[:, 0]`` (``tri_ke`` = :math:`\\mu`) and ``tri_materials[:, 1]``
    (``tri_ka`` = :math:`\\lambda`) act as membrane Lamé moduli [N/m]. At small
    strain Newton's membrane is linear elasticity,
    :math:`\\mu\\,\\varepsilon:\\varepsilon + \\tfrac{\\lambda}{2}(\\mathrm{tr}\\,\\varepsilon)^2`,
    and the mapping matches its energy for a stretch along either material
    axis with the other held fixed and for simple shear:

    - :math:`K_a = K_b = \\mu + \\lambda / 2` [N/m];
    - :math:`K_g = \\mu / 2` [N/m], since :math:`(F a) \\cdot (F b) \\approx 2 \\varepsilon_{ab}`;
    - :math:`\\eta` = ``cloth_strain_rate`` (libuipc's strain limiting; 0 keeps the
      membrane quadratic). The Baraff-Witkin model has no Poisson coupling
      (no :math:`\\lambda\\,\\varepsilon_{aa}\\varepsilon_{bb}` term), so an equal stretch in both
      directions meets less resistance than in Newton's model.

    Newton's hinge energy is :math:`\\tfrac{1}{2} k_e L_0 (\\theta - \\bar\\theta)^2`
    (``edge_bending_properties[:, 0]`` = :math:`k_e`); libuipc's is
    :math:`\\kappa L_0 (\\theta - \\bar\\theta)^2 / \\bar h` with
    :math:`\\bar h = (A_1 + A_2) / (3 L_0)` from the two rest triangle areas, so
    each hinge gets :math:`\\kappa = k_e \\bar h / 2`. Particle masses are used as
    they are. The damping coefficients ``tri_kd`` and ``edge_kd``, drag and
    lift are ignored: implicit Euler supplies the only damping.

    Rest shapes are Newton's. A membrane triangle's rest edges are the inverse
    of its :attr:`~newton.Model.tri_poses` (the edges in the triangle's frame
    whose first axis runs along its first edge), laid into the plane of the
    triangle at :attr:`~newton.Model.particle_q`, first edge along first edge,
    where it takes libuipc's material axes. Newton's builders compute
    ``tri_poses`` in float64 from ``particle_q`` and store them in float32: a
    triangle whose ``tri_poses`` are that rounding takes its rest edges at
    ``particle_q`` in float64, and when every triangle does, as for a cloth
    built at rest, the engine takes them from ``particle_q`` as libuipc takes
    them from its rest geometry, so that the membrane starts without strain.
    A hinge's rest angle is its :attr:`~newton.Model.edge_rest_angle`, the
    signed angle between the normals of ``(o0, v1, v2)`` and ``(o1, v2, v1)``
    for the row ``(o0, o1, v1, v2)``: libuipc's angle by the winding of the
    two triangles where ``o0``'s triangle runs ``v1 -> v2``, as in the rows
    Newton's cloth builders make, and its negative where a row given by hand
    has ``o0``'s triangle run ``v2 -> v1``. A row whose two triangles run the edge the same way raises
    :class:`ValueError`. Where every hinge's ``edge_rest_angle`` is the angle
    Newton's builders measure at ``particle_q``, as for a cloth built at rest,
    the engine takes the rest angles from ``particle_q`` as libuipc takes them
    from its rest geometry, with the dihedral angle its bending energy
    evaluates, so that every hinge starts at its rest angle: libuipc's angle,
    the arccosine of the normals' cosine, reads up to 1.5e-8 rad on a flat
    hinge that Newton measures as 0. The hinges keep the rest lengths and
    areas of ``particle_q`` (libuipc's :math:`L_0` and :math:`\\bar h`) where
    Newton's VBD takes ``edge_rest_length``; for a rest shape equal to the
    start the two agree.

    **Contact.** Which pairs collide, and with which friction, is libuipc's
    contact tabular: every vertex and half plane is in a contact element, and
    every pair of elements has a friction coefficient and a switch. Shapes that
    Newton's rules cannot tell apart share an element (the same collision
    group, world, collide flags and friction coefficient; with several
    worlds the subscenes keep the worlds apart, see **Worlds**, and the world
    leaves the signature); a shape in a filter pair gets one of its own, which
    with several worlds its copies in replicated worlds share. Free particles
    (in no triangle and no
    tetrahedron) are in one element, or in another one that collides with
    nothing when they are not :attr:`~newton.ParticleFlags.ACTIVE`, and the
    vertices of cloth and tetrahedra in a third. Two shapes collide as
    :attr:`~newton.Model.shape_contact_pairs` says: both collide with shapes,
    their collision groups and worlds are compatible and they are not a filter
    pair; shapes of one body never collide, and the solver checks at
    construction that its pairs are exactly the model's. A particle collides
    with a shape that has :attr:`~newton.ShapeFlags.COLLIDE_PARTICLES`, two
    free particles or a free particle and a cloth or tetrahedron vertex
    collide when the model has more than one particle and a positive
    :attr:`~newton.Model.particle_max_radius` (as Newton's particle solvers
    decide), and cloth and tetrahedra always collide with themselves and with
    each other. The friction of two shapes is the arithmetic mean of their
    :attr:`~newton.Model.shape_material_mu`, of a particle and a shape the mean
    of :attr:`~newton.Model.soft_contact_mu` and the shape's, and of two
    particles :attr:`~newton.Model.particle_mu`, as in Newton's semi-implicit,
    Featherstone and XPBD solvers. The engine tests the table of the elements
    and that of the worlds (see **Worlds**) apart, as libuipc does.

    Contact thickness, libuipc's distance at which the barrier diverges: a
    free particle is a sphere of its :attr:`~newton.Model.particle_radius`, so
    it rests its radius above a shape and two free particles rest the sum of
    their radii apart; a shape's vertices take its
    :attr:`~newton.Model.shape_margin` (a static plane's surface is moved out
    by it), so that two shapes rest the sum of their margins apart, as
    Newton's margins add up; and the vertices of cloth and tetrahedra take the
    solver's ``thickness`` argument. libuipc gives a vertex's thickness to
    every edge and triangle it is on, and finite elements collide with
    themselves, so a cloth vertex of radius r would keep the cloth's
    non-adjacent triangles and any other cloth 2r away: Newton's radii are
    sphere radii, often larger than the spacing of a cloth, and Newton's cloth
    solvers do not use them for cloth self-contact either.

    Contact stiffness, the barrier's :math:`\\kappa`, is ``kappa`` for every
    pair, taken as given. libuipc does not take a stiffness as given: its
    contact manager keeps it in a range :math:`[\\kappa_{lo}, 100\\,\\kappa_{lo}]`
    with :math:`\\kappa_{lo} = c\\,\\bar m / \\Delta t^2`, :math:`\\bar m` the mean
    mass of libuipc's vertices and :math:`c` a factor of the scene's diagonal
    and ``d_hat`` (Stiff-GIPC's estimate); a scene whose default contact model
    the user has not set takes :math:`\\kappa_{lo}`, and a stiffness the user
    gives is clamped into the range. libuipc-warp's pyuipc front end does the
    same. With ``adaptive_kappa=True`` the solver does too: ``kappa=None``
    takes :math:`\\kappa_{lo}`, a given ``kappa`` is clamped into the range and
    the first step that clamps it warns with :class:`~newton.NewtonWarning`.
    The vertices are then every particle, every vertex of the triangulated
    surfaces of bodies and static shapes, and one per plane; each body's mass
    counts once, static shapes and planes have none. The range follows the
    ``dt`` of each step, also of a captured one, and is computed from the
    masses when the solve is built: a change of mass in place
    (:meth:`notify_model_changed`) keeps it, as libuipc computes it once at
    init. Every contact pair takes one value, which :meth:`contact_stiffness`
    gives for a step of any ``dt``; with independent worlds every world takes
    the range of the same world alone, from its own vertices and the global
    ones, and every pair of a world its value (see **Worlds**).

    **BDF2.** ``integrator="bdf2"`` is libuipc's BDF2 (its ``integrator/type``,
    taken as libuipc and its pyuipc front end take it): the step predicts

    .. math::

        \\tilde x = \\tfrac{4}{3} x_n - \\tfrac{1}{3} x_{n-1}
            + \\tfrac{2}{9} \\Delta t (4 v_n - v_{n-1}) + \\tfrac{4}{9} \\Delta t^2 a,

    weighs the inertia :math:`\\tfrac{9}{4}` times as much against the potentials
    as implicit Euler does, and ends at
    :math:`v_{n+1} = (3 x_{n+1} - 4 x_n + x_{n-1}) / (2 \\Delta t)`: second order,
    with less numerical damping than implicit Euler. Contact so loses less
    energy and bodies take longer to settle: a cube dropped onto the ground in
    steps of 1/1800 s rebounds at 0.93 of its impact speed, at 0.47 under
    implicit Euler, as in libuipc. The history :math:`(x_{n-1}, v_{n-1})` is
    the solver's, not the state's: a step from the state the solver wrote last
    continues it, and every other step is an implicit Euler step whose start
    becomes the history: the first step, a step from a state the solver did not
    write last (also one it wrote earlier) and the step after :meth:`reset`. A
    step from a new state so applies the whole impulse of its forces, as in
    Newton's other solvers. (libuipc starts the history from rest, without
    acceleration: a constant force from rest gets two thirds of its impulse in
    the first step, and the velocity lags half a step's impulse behind for
    good.) A change of gravity (:meth:`notify_model_changed` with
    ``MODEL_PROPERTIES``) starts the history anew too: the next step compares
    the gravity of every world with the step before's, on the device, also
    inside a captured graph, and is an implicit Euler step of the worlds
    whose gravity changed, so that their velocity follows the new gravity at
    once; without it the velocity would lag :math:`\\Delta t \\, \\Delta g / 2`
    behind for good, as above. Gravity is a parameter of the model, changed
    between steps by a jump that a history taken under the old gravity
    cannot hold. The external forces (``particle_f``, ``body_f``,
    ``joint_f``) and the drives are inputs of every step that may change at
    every step, and BDF2 takes them as they come, as libuipc does: a force
    that jumps within a history leaves the velocity half a step's impulse of
    the jump behind. With several worlds a new state in one world, or its
    reset, makes the step an implicit Euler step of that world's particles
    and bodies, and the other worlds go on with BDF2. As in libuipc the
    coefficients assume one ``dt`` for consecutive steps: a step of another
    ``dt`` is taken as it is. The drives, passive springs and damping of
    joints take BDF2's velocity at the end of the step (see **Joints**); the
    default body stiffness follows the rule of **Rigid bodies**. Rigid bodies
    take libuipc's straight-line step under BDF2: ``body_rotation`` is
    ``"linear"`` there by default, and ``"rigid"``, whose flight replaces the
    implicit Euler prediction, is refused (see **Rigid bodies**).

    **AL-IPC contact.** ``contact_constitution="al-ipc"`` is libuipc's
    augmented-Lagrangian contact (its ``contact/constitution``; the method of
    Zheng, Luo and Li, "Robust and Efficient Penetration-Free Elastodynamics
    without Barriers"): each Newton
    iteration linearizes the constraint
    :math:`d \\ge \\hat d + \\text{thickness}` of the active contact pairs, with
    a penalty that follows the masses and multipliers that converge to the
    contact forces, and a collision-free sweep moves the configuration toward
    the solution. Contact rests at ``d_hat`` beyond the contact thickness,
    where the barrier rests inside ``d_hat``. Its constraints hold between
    surfaces of vertices with mass, the penalty of a pair scaling with the
    smallest mass of its vertices: particles on no cloth or tetrahedron that
    collide are refused, and so are static mesh shapes, particles and bodies
    without mass that collide (Newton's static shapes have no mass; static
    planes are taken). A pair joins the active set only when a sweep finds its
    two primitives touching, and the step that solves its constraint ends with
    the gap at ``d_hat``: a body whose step began a gap :math:`g` from a
    surface leaves it at about :math:`(\\hat d - g) / \\Delta t`, whatever its
    speed of impact. This is the method's own (libuipc and the authors' code do
    the same), and it gains energy where ``d_hat`` exceeds the distance a body
    travels in a step: a cube dropped 0.4 m onto the ground in steps of
    1/1800 s leaves at 4.4 times its impact speed of 2.8 m/s with ``d_hat``
    1 cm, at 0.7 times with 1 mm, and in steps of 1/100 s at 0.34 times with
    1 cm. libuipc's AL-IPC drops a pair from the active set three outer
    updates after it opens (its default ``decay_factor`` 0.3), so a resting
    contact that opens for a moment joins again at the surface and is pushed
    out again: a stack of two 0.1 m boxes in steps of 1/100 s keeps rocking
    with ``d_hat`` 5 mm or 2 mm, in most runs until its upper box falls off,
    as on libuipc's own backend, where the paper's decay factor 0.9, which
    this solver does not set, brought it to rest in most runs. Keep ``d_hat``
    well below the distance a body moves in a step. A kinematic body moved
    between steps is swept from where the last step left it to its pose in
    ``state_in`` (the method's moving boundaries, which libuipc's frame lacks:
    there nothing a kinematic body moves into collides with it): what it meets
    joins the active set and is pushed, and one that contact holds back ends
    the step short of its pose and goes on toward it in the next. The first
    step after the solver is built, the step after a :meth:`reset` that
    selects its world and the step after a failed one place it at its pose
    instead, without a sweep.
    ``kappa`` is not used, and ``adaptive_kappa`` is refused. The step
    ends on AL-IPC's own criterion, the paper's progress weight, which has no
    Newton residual: ``SOLVER_RESIDUAL`` is not offered, and without contact
    a step takes ``semi_implicit_k_min`` Newton iterations (the paper's
    ``K_min``), and one with ``semi_implicit=False``, which solves the
    barrier's steps to the tolerance but makes AL-IPC's the least converged
    (with libuipc's body stiffness a free cube spinning at 1.08 rad/s keeps
    6% of its spin after 100 steps of 0.01 s). A larger
    ``semi_implicit_k_min`` converges AL-IPC's steps further. A change of the
    bodies' masses (:meth:`notify_model_changed`) builds the solve again,
    since the penalty follows the masses the solve was built with.

    **MAS.** ``fem_preconditioner="mas"`` is libuipc's multilevel additive
    Schwarz preconditioner (its ``linear_system/fem_preconditioner``) on the
    vertices of particles, cloth and tetrahedra (bodies
    keep their 12 x 12 blocks): it inverts the system exactly on clusters of up
    to 16 vertices and corrects between them on coarser levels made of
    neighbouring clusters. libuipc takes the clusters from METIS; libuipc-warp's
    array path makes them itself, by recursive inertial bisection of every
    connected cloth, soft body and spring network at its rest positions. The
    preconditioner is libuipc's on another partition: the PCG iterations
    differ from libuipc's, and the result is the same to the tolerance of the
    linear solve. Particles on no element keep their 3 x 3 blocks.

    **Worlds.** Particles, bodies and colliding shapes in more than one local
    world are simulated as independent worlds, Newton's meaning of worlds:
    every world takes the step it would take alone, with its own Newton
    iterations, line searches and step lengths, convergence test, linear
    solves and status, as libuipc-warp's independent worlds solve them
    (libuipc-warp's own; libuipc solves the subscenes of a scene together).
    Every local world is a libuipc subscene of its own and the global entities
    (world -1) one more, which meets every world, as Newton's collision rules
    say: no contact pair joins two worlds, also where the worlds overlap in
    space, and a global ground or static shape acts in every world. The worlds
    share one frame graph, one collision detection and one set of contact
    lists: a step runs as many Newton iterations as its slowest world, and a
    world that has converged is held where it is while the others iterate. A
    world takes the Newton iterations of the same world alone and agrees with
    it to the rounding of the engine's sums, which add in another order than in
    an engine of that world alone, as two runs of one engine already differ by
    the rounding of its atomics; where a step carries rounding far, as a stiff
    landing does at the default tolerances, the two part as far as a run alone
    parts from one started a rounding error away.

    A world that fails (a non-finite value, a contact pair at or inside its
    thickness, a CCD fraction that is not positive) fails alone: it is put
    back at its state at the step's start, its friction discarded and its BDF2
    history started anew, and the other worlds take their step. ``STATUS`` and
    ``SOLVER_RESIDUAL`` hold every world's own bits and residual at its entry;
    the entries of the global world and of worlds that hold no unknowns (only
    static or kinematic content) stay 0. A step that is not captured and fails
    in some worlds writes ``state_out``, those worlds at the step's start and
    the others at their step, and raises :class:`RuntimeError` naming the
    worlds and their bits; :meth:`check_status` names them too. The worlds
    share the contact lists, and every world has a quota of them, sized as for
    that world alone (libuipc-warp's capacity per world): a world whose
    contact asks for more than its quota fails alone in the same way, while
    the other worlds keep every contact pair and take their step. An eager
    step then grows that world's quota and solves the step again for every
    world; a captured step sets ``OVERFLOW`` in that world's entry alone, and
    :meth:`grow` grows its quota. A reset with a ``world_mask``, or
    a new state written for one world, discards that world's friction and
    starts its BDF2 history anew and leaves the other worlds as they were; the
    mask's global entry selects nothing, since the global entities are no
    unknowns. :meth:`frame_stats` gives the counts of the step's loops: the
    Newton iterations of its slowest world. With ``adaptive_kappa=True`` every
    world takes libuipc's stiffness range of the same world alone, from the
    mean mass and the extent of its own vertices and the global ones
    (libuipc-warp's stiffness per world): its own lower bound with
    ``kappa=None``, a given ``kappa`` clamped into its own range, at every
    step's ``dt``; :meth:`contact_stiffness` gives every world's.

    The global entities do not move: a free particle (mass and
    :attr:`~newton.ParticleFlags.ACTIVE`), a dynamic body or a coupled solver's
    proxy of world -1, and with ``kinematic_motion="aim"`` a global kinematic
    body with colliding shapes, would be an unknown that every world pushes,
    and are refused with :class:`NotImplementedError`, as Newton's implicit MPM
    refuses global colliders backed by dynamic bodies with
    ``separate_worlds``; so is a triangle, tetrahedron or spring that joins two
    worlds. Global static shapes and planes, kinematic particles and kinematic
    bodies (``kinematic_motion="jump"``) act in every world. Not offered with
    independent worlds yet, where two worlds or more move, and refused naming
    what to use instead: ``contact_constitution="al-ipc"`` and
    ``fem_preconditioner="mas"`` (its clusters may join two worlds). The
    engine's Newton exit check, which independent worlds do not take either,
    is not used by the solver.

    With ``coupled_worlds=True`` the worlds are one solve instead, with the
    same subscenes: one step length (the smallest time of impact over all
    worlds, and a line search on the sum of their energies), one Newton
    convergence test (the largest displacement over all worlds), one PCG solve
    and one set of contact lists. A world's result therefore depends on the
    others through the iteration counts and the step lengths, and a failure or
    an overflow in one world fails the step in every world, whose entries all
    get the bits of the one solve. Copies of one world take the iterations of
    one copy, and each follows the run of that world alone to the rounding of
    the shared sums. Global particles and bodies may move, as unknowns of the
    one solve that meet every world, and the options above are taken. A reset
    of one world, or a new state written for one world, discards the friction
    of that world's contacts and starts its BDF2 history anew, and leaves every
    other world's friction and history as they were, bit for bit
    (libuipc-warp's requests per subscene; libuipc discards friction for a
    whole scene); a reset of the global entities discards the friction of
    every pair with a global entity; with ``adaptive_kappa`` the stiffness
    range is that of the whole scene.

    A model with content in one local world is solved as one world either
    way. The engine takes up to 32,768 subscenes, and contact elements times
    the subscene count rounded up to a power of two below :math:`2^{31}`; the
    start check forms no pair of parts of two worlds.

    **Kinematic bodies.** By default (``kinematic_motion="jump"``) a
    kinematic body is a fixed affine body whose pose the solver writes from
    ``state_in`` before the step: under the barrier the jump between two steps
    is not swept by collision detection, so a jump larger than the room the
    barrier leaves puts surfaces into each other (under AL-IPC the step sweeps
    it, see **AL-IPC contact**), and friction never sees the motion (a
    platform sliding under a box does not carry it). With
    ``kinematic_motion="aim"`` every kinematic body with colliding shapes is
    built as libuipc builds an animated body: an affine body that is an
    unknown of the step, with no gravity and no external force, pulled by
    libuipc's soft transform constraint (constitution 16) toward its aim, the
    pose of ``state_in`` advanced by its twist over the step (the centre of
    mass by :math:`v \\Delta t`, the rotation by the exponential of
    :math:`\\omega \\Delta t`). Its velocity is set so that the step's
    prediction is the aim, so its motion is part of the Newton direction: the
    CCD sweeps it, the line search keeps every surface apart, friction sees
    it, and a body that contact blocks stops short of its aim. The aim is
    soft, with the strength ratio :math:`\\eta` = ``kinematic_strength_ratio``
    (libuipc's motors' 100) for translation and rotation alike: under a load
    :math:`F` a body of mass :math:`m` lags its aim by about
    :math:`F \\Delta t^2 / ((1 + \\eta) m)`, 2.8e-5 m for 10 N on 1 kg at
    1/60 s. A kinematic body moves with Newton's mass; one of mass 0 takes the
    mass of its collision meshes at 1000 kg/m^3. A caller who writes
    consistent ``body_q`` and ``body_qd`` every step gets the next pose as the
    aim, one who writes poses only (``body_qd`` 0) the written pose, one who
    writes nothing motion at constant twist. The step starts from the pose the
    body reached, so the motion from there to the aim is swept, a jump
    included; the first step after the solver is built and the step after a
    :meth:`reset` that selects the body's world take ``state_in`` instead,
    which is how a caller moves such a body without sweeping. ``state_out``
    holds the pose the body reached and its twist, not a copy of
    ``state_in``. Pairs of two shapes that are each kinematic or static do not
    collide, as Newton's collision pipeline can filter them
    (``include_static_kinematic_pairs=False``) and as they never collide in
    the default mode, where neither moves inside the step; the solver checks
    its pairs against :attr:`~newton.Model.shape_contact_pairs` less those
    pairs. Kinematic particles keep the jump (a soft aim would make a pinned
    vertex sag). Not with ``contact_constitution="al-ipc"``.

    **Attachments.** A particle-body attachment
    (:meth:`~newton.ModelBuilder.add_attachment_particle_body`) is libuipc's
    soft vertex stitch (constitution 22) of rest length 0 between the particle
    and an anchor of the body's affine body at ``attachment_particle_body_body_point``
    in Newton's body frame: a point that moves with the body and is no contact
    primitive, as the interior vertices of an affine-body mesh are in libuipc;
    attachments at one point of one body share it. Its energy in the step's
    incremental potential is :math:`\\tfrac{1}{2} k \\Delta t^2 |x_p - x_a|^2`, the
    spring :math:`\\tfrac{1}{2} k |x_p - x_a|^2` of implicit Euler with
    :math:`k` = ``attachment_particle_body_stiffness``, so that the particle
    and the body take equal and opposite forces, the body with the torque the
    anchor's offset gives, and a disabled attachment
    (``attachment_particle_body_enabled``) takes none. A hard attachment
    (infinite stiffness), which libuipc has no constitution for, takes the
    penalty of a ball joint (see **Joints**): :math:`K/\\Delta t^2` with
    :math:`K` = ``joint_strength_ratio`` times the masses of the particle,
    if it is not fixed, and of the body, if it is dynamic, so that it holds a
    load :math:`F` within :math:`F \\Delta t^2 / K`. The stiffness and the
    enabled flags are read at every step: a captured step takes new values at
    its next replay. A spring of rest length 0 is the same stitch between its
    two particles, with ``spring_stiffness``. Damping is rejected: libuipc's
    stitch has none, and an implicit damping cannot be folded into it the way
    the joint drives fold theirs. An attachment does not change which pairs
    collide: an attached particle collides with its body as with any other,
    so a particle attached on the body's surface starts in contact (see the
    note on the start below); place it a gap away.

    **Rigid bodies.** Each Newton body becomes a libuipc affine body with 12
    degrees of freedom :math:`q = (p, a_1, a_2, a_3)`, the translation and the
    rows of :math:`A`, in Newton's body frame: a point with body-frame
    coordinates :math:`\\bar x` sits at :math:`x = A \\bar x + p`, starting from
    :math:`A = R(q_{body})` and :math:`p` = the body origin. Its kinetic term uses
    libuipc's dyadic mass, built from :attr:`~newton.Model.body_mass` :math:`m`,
    :attr:`~newton.Model.body_com` :math:`c` and :attr:`~newton.Model.body_inertia`
    :math:`I` (which Newton computes from the shapes' densities):
    :math:`\\sum m \\bar x = m c` and
    :math:`\\sum m \\bar x \\bar x^T = \\tfrac{1}{2}\\mathrm{tr}(I)\\,1 - I + m c c^T`, so the
    affine kinetic energy equals the rigid one for every rigid motion. The
    orthogonality potential :math:`\\kappa V \\Delta t^2 \\|A A^T - I\\|_F^2` with
    the stiffness :math:`\\kappa` (below) and the volume :math:`V` of the body's
    collision meshes keeps :math:`A` within a small strain of a rotation; this
    is libuipc's near-rigid body, not an exactly rigid one. In libuipc's step
    (``body_rotation="linear"``) a turning body moves along a straight line in
    its 12 coordinates within a step and the orthogonality potential pulls it
    back onto the rotations, so under implicit Euler a free spin at
    :math:`\\omega` loses about :math:`(\\omega \\Delta t)^2` of its angular
    velocity per step: 1e-4 at 1 rad/s and :math:`\\Delta t` = 0.01 s, and a body
    tumbling about no principal axis loses angular momentum and energy alike.
    By default under implicit Euler (``body_rotation="rigid"``, libuipc-warp's
    ``affine_rotation="rigid"``) every free body is predicted instead by the
    flight of a rigid body, which keeps its angular momentum :math:`L` about
    the centre of mass and turns it at the implicit midpoint of the free rigid
    body, and its velocity after the step carries that momentum plus the
    angular impulse of the step's contact and joint forces. A free body then
    keeps its angular momentum and kinetic energy, and a spin about a
    principal axis, or of a body with equal moments, is the exact motion; a
    torque spins a body up by :math:`\\tau / I` per second, which libuipc's
    step falls short of by the same loss. A pendulum on a hinge away from its
    centre of mass keeps implicit Euler's loss of its centre of mass's
    circular motion, and the Newton loop's displacement test leaves a little
    of each turn at the default tolerances (a 20 cm cube at 2.24 rad/s keeps
    0.995 of its spin over 200 steps of 0.01 s, 0.910 in libuipc's step). The
    body's twist then holds the angular velocity that carries its angular
    momentum, :math:`\\omega = (R I R^T)^{-1} L`, rather than the skew part of
    :math:`\\dot A A^{-1}`, which also counts the spin of the body's strain.
    Fixed, kinematic and massless bodies keep libuipc's prediction, and a
    body whose rows of :math:`A` have no velocity steps bit for bit as in
    libuipc's step. Under ``integrator="bdf2"`` the default is ``"linear"``:
    BDF2 predicts from the step before as well, which the flight does not
    turn, and ``body_rotation="rigid"`` with it raises ``ValueError``.

    The stiffness is ``body_kappa`` at every step when it is given. By default
    it is libuipc's 1e8 Pa, lowered for a body and a step to
    :math:`\\kappa = 25\\, I_{max} / (V \\Delta t^2)` where that is lower,
    :math:`I_{max}` being the body's largest principal moment of inertia. The
    straight line to a turn :math:`\\theta` stretches the body by about
    :math:`\\theta^2 / 2`, and the first Newton iteration of a step takes the
    turn in full while that stretch's energy,
    :math:`2 \\theta^4 \\kappa V \\Delta t^2`, stays below the inertia term
    :math:`\\theta^2 I / 2` of the incremental potential that it removes,
    :math:`I` being the moment of inertia about the axis of the turn. With the
    lowered stiffness the first iteration takes a turn of up to 0.1 rad about
    the axis of the largest moment and of up to
    :math:`0.1 \\sqrt{I / I_{max}}` rad about an axis of moment :math:`I`:
    0.1 rad about any axis of a compact body, less about the long axis of a
    thin one, which keeps its stiffness along its length. With a stiffer body
    the line search cuts the iteration short, and the next Newton directions,
    held back by the stretched potential, become small enough to pass the
    displacement test of the Newton loop long before the turn is complete: at
    1e8 Pa a 1 m cube that a torque turns by 0.06 rad in a 0.1 s step ends at
    31% of its angular velocity, and a 10 cm cube spinning at 0.6 rad/s keeps
    33% of it over six steps of 1/60 s, where the converged solve keeps 99.9%.
    The lowered stiffness costs rigidity in proportion to :math:`\\Delta t^2`
    where it applies: the lowest of five 10 cm cubes stacked on the ground is
    compressed by 4e-5 at :math:`\\Delta t` = 1/600 s and by 4e-3 at 1/60 s
    (6e-6 at 1e8 Pa), a 30 cm by 1 cm finger swinging from a hinge stretches
    by up to 8e-6 and 7e-4, and a box of unequal sides that a torque starts
    turning runs ahead of a rigid one by 7e-4 of its turn after one step of
    0.01 s and by 1e-5 after ten. Pass ``body_kappa=1e8`` for libuipc's
    stiffness at every step.
    Gravity acts on :math:`p`; with the full dyadic mass this is the exact
    gravity force on the affine body.
    An external force :math:`f` at the centre of mass is the generalized force
    :math:`(f, f c^T)` on :math:`(p, A)`, and a torque :math:`\\tau` is
    :math:`(0, \\tfrac{1}{2} [\\tau]_\\times A^{-T})`. The body's collision shapes
    are triangulated (curved shapes with ``shape_segments`` segments around)
    and act as its contact surface.

    After each step the solver writes ``body_q`` and ``body_qd``. :math:`A` is
    projected onto its nearest rotation :math:`R` by the polar decomposition
    :math:`A = R S` (from the SVD :math:`A = U \\Sigma V^T`, :math:`R = U V^T`),
    since a quaternion can only hold a rotation; the residual stretch
    :math:`S - 1` is of the order of the orthogonality strain. The body origin
    is placed so that the centre of mass stays where the affine map puts it,
    :math:`x_{com} = A c + p`; the linear velocity is that point's,
    :math:`\\dot p + \\dot A c`, and the angular velocity is the one that
    carries the body's angular momentum, :math:`(R I R^T)^{-1} L`, when its
    rotation is a rigid body's flight (``body_rotation="rigid"``), and the
    axial vector of the skew part of :math:`\\dot A A^{-1}` in libuipc's step.

    **Joints.** Revolute, prismatic, fixed and ball joints are libuipc's
    affine-body joints (constitutions 18, 20, 25 and 26), between the affine
    bodies of their parent and child, a parent of -1 being a fixed body
    without mass at the world's origin. A revolute or prismatic joint holds
    the two points of its axis, ``joint_X_p`` and ``joint_X_c`` and one metre
    along ``joint_axis`` from them, together; a prismatic joint also holds the
    bodies' relative rotation, a fixed joint the anchors and the rotation, a
    ball joint the anchors. The rotation they hold, and the zero of a joint's
    coordinate, is where the joint frames coincide, as in Newton, whatever the
    bodies' poses when the solver is built.

    The joints are penalties, not hard constraints: libuipc's energy
    :math:`\\tfrac{K}{2}|C|^2` of the joint's violation :math:`C`, a term of
    the incremental potential (not multiplied by :math:`\\Delta t^2`), with
    :math:`K` = ``joint_strength_ratio`` times the mass of the dynamic bodies
    the joint joins (the world and kinematic bodies count 0). Against a
    translation of the child, in a step of length :math:`\\Delta t`, a
    revolute or prismatic joint, which holds two points, is a spring of
    :math:`2K/\\Delta t^2` [N/m], a fixed or ball joint one of
    :math:`K/\\Delta t^2`: a load that would move the child by :math:`\\delta`
    in one step stretches the joint by about :math:`\\delta / (2\\gamma)` or
    :math:`\\delta / \\gamma`, :math:`\\gamma` = ``joint_strength_ratio``.
    Gravity at 1/60 s and the default 100 stretch a revolute joint by 1.4e-5 m
    and a ball joint by 2.7e-5 m. A larger ratio holds the joints closer and
    makes the linear system stiffer.

    Drives, passive springs and passive damping of a revolute or prismatic
    joint are exact for the step's integrator: the joint's
    :math:`\\tau = k_e (q^* - q) + k_d (\\dot q^* - \\dot q) - k_s (q - q_r) - c \\dot q`
    (``joint_target_ke``, ``joint_target_kd``, ``joint_stiffness``,
    ``joint_rest_q``, ``joint_damping``; targets from
    :attr:`~newton.Control.joint_target_q` and ``joint_target_qd``), taken at
    the end of the step with :math:`\\dot q = (q - q_n)/\\Delta t` (BDF2:
    :math:`(3 q - 4 q_n + q_{n-1}) / (2 \\Delta t)`), is minus the
    gradient of one quadratic potential in :math:`q`, which the solver gives
    libuipc's driving term every step. The gains act whatever
    ``joint_target_mode`` says, as in the other maximal-coordinate solvers.

    With ``enable_joint_armature=True`` a revolute or prismatic joint's
    ``joint_armature`` :math:`a`, which Featherstone and MuJoCo add to the
    joint's inertia, is the kinetic term of its coordinate in the same
    potential: :math:`\\tfrac{a}{2 \\Delta t^2} (q - \\tilde q)^2` with
    :math:`\\tilde q = q_n + \\Delta t \\dot q_n` (BDF2: 9/4 of it, with
    :math:`\\tilde q = \\tfrac{4 q_n - q_{n-1}}{3} + \\tfrac{2 \\Delta t}{9}
    (4 \\dot q_n - \\dot q_{n-1})`), as libuipc writes the inertia of every
    degree of freedom, :math:`\\dot q_n` being the rate the bodies' twists give
    at the step's start (as ``joint_qd`` is written). The armature of a ball
    or free joint, an inertia about three axes or a mass without weight,
    has no such potential and is rejected.

    The drive's force cannot be bounded inside libuipc's driving term, a
    quadratic potential. With ``enable_joint_effort_limit=True`` a
    ``joint_effort_limit`` :math:`L` below Newton's default 1e6 bounds the
    force of the gains as decided at the start of every step: where
    :math:`|k_e (q^* - q_n) + k_d (\\dot q^* - \\dot q_n)| > L`, the step takes the
    constant force :math:`\\pm L` (libuipc's joint torque or force, with
    ``joint_f``) in place of the gains' part of the potential; the passive
    spring and damping remain. The bound holds at the start of every step,
    which is where MuJoCo clamps its actuators' force; a step that starts
    below it and ends above it, or the reverse, takes the bound one step
    late, which a drive clamped at the end of the step would not (by 1.6e-4
    of the motion for a hinge driven through a step of its target).

    Limits are libuipc's cubic penalty outside ``[joint_limit_lower,
    joint_limit_upper]``, not Newton's linear spring: at an overshoot
    :math:`\\delta` it stores :math:`k_l \\delta^3 / (3 \\delta_r)` with
    :math:`k_l` = ``joint_limit_ke`` and :math:`\\delta_r` = 1e-3 (rad or m), so
    that it pushes back with the spring's :math:`k_l \\delta_r` at
    :math:`\\delta_r` and harder beyond it; a load :math:`F` rests at
    :math:`\\sqrt{F \\delta_r / k_l}` past the limit. ``joint_limit_kd`` is not
    read (libuipc's limit has no damping), a ``joint_limit_ke`` of 0 holds
    nothing, and a bound at :data:`newton.MAXVAL` is no bound. With
    ``enable_joint_limit_damping=True`` a joint whose coordinate is past one
    of its limits at the start of a step takes the damping
    :math:`-k_{ld} \\dot q` of Featherstone and SemiImplicit,
    :math:`k_{ld}` = ``joint_limit_kd``, in the same quadratic potential as
    ``joint_damping``: exact for the step's integrator, with the velocity at
    the end of the step, and decided at its start, as those solvers decide it
    (they also switch the drive off past a limit, which SolverIPC does not,
    as MuJoCo does not). A joint that reaches its limit within a step is
    damped from the next step on; a limit of ``joint_limit_ke`` 0 damps
    nothing.

    :attr:`~newton.Control.joint_f` of a revolute or prismatic joint is
    libuipc's joint torque or force between the two bodies; of a ball joint a
    torque in the parent's joint frame, on the child, the parent taking the
    reaction; of a free joint a wrench at the child's centre of mass in the
    world frame, the parent, if it is a body, taking the reaction.
    ``joint_enabled`` is read at every step, and so are the gains, targets,
    springs, damping and limits: a captured step takes new values at its next
    replay. A change of the joints' types, bodies, frames or axes rebuilds
    the solve (``notify_model_changed`` with ``JOINT_PROPERTIES``).

    After every step ``joint_q`` and ``joint_qd`` hold the coordinates
    :func:`newton.eval_ik` computes from the bodies' state, for every joint,
    but the position of a revolute joint, which is libuipc's angle and turns
    on past :math:`\\pi` as in the generalized-coordinate solvers, and of a
    prismatic joint, libuipc's distance. A revolute joint's angle continues
    from the step before; where ``state_in.joint_q`` is not what the solver
    wrote, it picks the turn of the bodies' angle (the pose decides the angle
    modulo :math:`2\\pi`).

    **D6 joints** (``enable_d6_joints=True``). libuipc has no D6 joint (its
    constitution 28 is only registered). A D6 joint whose free axes leave
    locked what one of libuipc's joints locks is that joint, exactly: no
    axis is a fixed joint, one angular axis a revolute joint about it, one
    linear axis a prismatic joint along it, with their drives, springs,
    damping, limits and ``joint_f``; three orthonormal angular axes are a
    ball joint, and three orthonormal linear and three orthonormal angular
    axes lock nothing, as a free joint. ``joint_q`` and ``joint_qd`` hold
    Newton's coordinates of the D6 joint: a revolute or prismatic one's as
    above, of three angular axes the angles :func:`newton.eval_ik` takes
    (the XYZ Euler angles in the basis of the first two axes), of linear
    axes the projections of the joint frames' offset. ``joint_f`` of a D6
    joint of three angular axes is Featherstone's generalized force: a
    torque along the dual basis of the axes as the coordinates carry them,
    and with linear axes a force along them on the child's joint frame,
    the parent taking the reaction on the same line. Other D6 joints (two
    angular axes, which lock one rotation, or linear and angular axes
    together but three and three) lock what no libuipc joint locks, and
    three axes that are not orthonormal have no coordinates in Newton's
    decomposition: both are rejected, and so are drives, springs, damping
    and limits on the D6 joints that become a ball or no joint.

    **Distance joints** (``enable_distance_joints=True``). libuipc has no
    distance joint. One whose distance has a single value ``L`` (the two
    bounds of its first degree of freedom equal and not negative) holds the
    origins of its joint frames ``L`` apart as libuipc's soft vertex stitch
    (constitution 22) of rest length ``L`` between two anchors of the
    bodies there (the world's fixed body for a parent of -1): the penalty
    of a ball joint, ``K/2 (|x_c - x_p| - L)^2`` with the same ``K``, and
    nothing when the joint is disabled. One whose bounds are both negative
    holds nothing, as Featherstone and SemiImplicit take every distance
    joint. Either takes ``joint_q``, ``joint_qd`` and ``joint_f`` as a free
    joint does. A range of distances, or a single bound, is an inequality,
    which a stitch, a spring about one rest length, cannot hold: rejected,
    and so are drives on a distance joint (SolverXPBD reads its gains as
    the compliance of the distance).

    Rejected (:class:`NotImplementedError`), each for the libuipc
    constitution it lacks: distance joints and D6 joints but those above,
    which libuipc does not have, and rod joints, which no libuipc
    constitution between two affine bodies represents; drives, springs,
    damping and limits on ball and free joints,
    since libuipc's driving term and limits act on one hinge angle or one
    distance and its ball joint has neither (its ball driving term,
    constitution 27, is only registered); ``joint_armature`` other than 0
    but with ``enable_joint_armature`` on a revolute or prismatic joint,
    since libuipc's joints have no inertia of their own; ``joint_friction``
    other than 0, since they have no friction and dry friction, a bounded
    force, is no quadratic potential; a ``joint_effort_limit`` below
    Newton's default 1e6 on a driven degree of freedom but with
    ``enable_joint_effort_limit``, since the quadratic drive cannot bound its
    force.

    **State.** The solver keeps positions and velocities in float64 and reads
    ``state_in`` on every step. A particle whose ``state_in`` position and
    velocity are the float32 rounding of the solver's state, and a body whose
    pose and twist are what the solver last wrote, keep the float64 state;
    anything else is taken from ``state_in``, a body as the rigid motion of its
    pose and twist. Taking a new state for a particle or body that is not
    kinematic discards the friction of that step's contacts (with several
    worlds, the contacts of its world), as libuipc does after a
    state is written and as a solver started from that state does.
    :meth:`reset` does the same without a new state. Output is written to
    ``state_out`` in float32.

    The solve is built at the model's state: :attr:`~newton.Model.body_q` and
    ``body_qd``, but a body whose pose there is the identity, where
    :meth:`~newton.ModelBuilder.add_link` and
    :meth:`~newton.ModelBuilder.add_body` put a body given no transform,
    starts where :func:`newton.eval_fk` puts it from
    :attr:`~newton.Model.joint_q` and :attr:`~newton.Model.joint_qd`. As for
    Newton's other maximal-coordinate solvers, the caller writes those poses
    into the state before the first step, also after building the solver;
    the model of an articulation built without poses keeps the identity
    until then, and its links may cross each other or the ground there. The
    start is checked where it is known (see the note below): on the model's
    state, and for the pairs with a kinematic body or such a link, whose
    start the caller writes, on the first step's ``state_in``.

    **Status.** Request :attr:`SolverIPC.ObservableKind.STATUS` and
    ``ObservableKind.SOLVER_RESIDUAL`` (not with ``contact_constitution="al-ipc"``)
    with :meth:`observables` and pass the
    container to :meth:`step`: the step writes a :class:`SolverIPC.StatusBits` mask and
    the residual on the device, also during graph replay. With independent
    worlds every world that holds unknowns receives its own (see **Worlds**);
    with one world, or ``coupled_worlds``, every world that owns particles or
    bodies receives the values of the one solve. The
    residual is the convergence measure of libuipc's Newton loop: the largest
    displacement of the last Newton direction (the rows of an affine body's
    :math:`A` scaled by ``newton_velocity_tol / newton_transrate_tol``) over
    ``newton_velocity_tol * dt``. It is dimensionless, at most one when the
    loop ended on its displacement test, may be above one when the
    semi-implicit test ended it, and infinite when the step failed on a value
    in :class:`SolverIPC.StatusBits`.

    A step that is not captured reads the status itself after the solve, with
    one synchronization. When a contact list overflowed (with independent
    worlds, a world's quota of the lists) it enlarges the lists and solves the
    step again; when the step failed otherwise (a non-finite
    value, a contact starting at or inside its thickness, an exhausted BVH
    traversal), it raises :class:`RuntimeError` and leaves ``state_out``
    unwritten, but where independent worlds failed alone: there it writes
    ``state_out`` first, the failed worlds at the step's start and the others
    at their step (see **Worlds**). A captured step cannot do either: it sets
    the bit, and nothing else reports the failure. A caller that captures
    :meth:`step` must therefore request ``STATUS``, pass one container to each
    captured step (a container holds the bits of the last step that wrote
    it), and after every replay call :meth:`check_status` with each
    container. It raises
    :class:`RuntimeError` on a failed step and warns on an iteration limit;
    with ``grow=True`` it answers :attr:`SolverIPC.StatusBits.OVERFLOW` with
    :meth:`grow` and returns ``True``, and the caller then restores the
    state the replayed steps started from, captures again and repeats them.
    The PCG loop of a captured step re-reads its stopping test every 8 or 32
    iterations, and work whose size only the device knows starts at a guessed
    size; both are chosen when the step is captured. :meth:`graph_outdated`,
    called after replays, returns ``True`` when the steps replayed since asked
    for the other length or for other guesses, and the caller then captures
    again to run faster (the old graphs stay valid and give the same results).

    **Contact output.** Request ``SolverBase.ObservableKind.CONTACT_F`` with
    :meth:`observables`, passing the :class:`~newton.Contacts` that sizes
    ``contact_f``, and pass the container and a ``Contacts`` to :meth:`step`:
    the step writes its own contacts into that buffer and their forces into
    ``contact_f``, as a native collision backend does (SolverMuJoCo with its
    own collision detection). Do not run ``CollisionPipeline.collide`` into
    the buffer; the step overwrites its rows. Each contact term of the step
    with a force is one row: a point-triangle, edge-edge, point-edge,
    point-point or vertex / half-plane pair of the final active set (its
    barrier) or of the step's friction pairs (its lagged friction), so that a
    pair in both gives two rows, one normal and one tangential. A term between
    two shapes is a rigid row. A term between a particle feature (a particle,
    or an edge or a face of cloth or tetrahedra) and a shape is a soft row,
    with the particle, edge or face record of
    :attr:`~newton.Contacts.soft_contact_indices` and the weights of its
    contact point. Terms between particle features (cloth self-contact,
    particles among themselves) are not exported, as SolverVBD exports no
    soft self-contact.

    The forces are those at the end of the step: minus the gradient of the
    term's energy in the step's incremental potential at the final positions,
    over :math:`\\Delta t^2`, which the step balanced to the Newton tolerance.
    A box resting on the ground reports its weight. On a body of mass ``m``
    whose step ended on the displacement test the forces can be off by up to
    about ``m * newton_velocity_tol / dt``: a box creeping down a ramp under
    friction reports its weight within 1.2% at the default tolerances and
    within 1e-4 with ``newton_velocity_tol=1e-4``, ``linear_tol_rate=1e-10``
    and ``semi_implicit=False``. A rigid row's
    ``contact_f`` is the force on the body of shape 0 by the body of shape 1,
    the sum of the forces on its vertices, with their torque about that body's
    centre of mass at the end of the step (about the world origin for a static
    shape), in the world frame; a soft row's is the force on the shape's body
    by the particle feature, with its torque about that body's centre of mass.
    The points are written in the body frames of ``state_out``, so pass
    ``state_out`` to a consumer such as :class:`~newton.sensors.SensorContact`.
    Each side's point is the force-weighted mean of its vertices, the closest
    point of its primitive on the triangulated surface (a half plane's: the
    point under the vertex, on the plane's surface). The normal points from
    shape 0 toward shape 1, a soft row's from the shape toward the particle
    feature: against the force on shape 0 for a barrier term, whose force is
    the gradient of the distance, and from shape 0's point to shape 1's for a
    friction term, whose force is tangential. The margins are the shapes'
    ``shape_margin``, their contact thickness, and the offsets the normal
    times each margin toward the other side, as Newton's collision pipeline
    writes them.
    ``rigid_contact_surface_velocity`` (when allocated) and
    ``soft_contact_body_vel`` are zero, and ``soft_contact_rigid_indices`` is
    ``-1``: the solver has no surface velocity, and it triangulates and welds
    the shapes' meshes. The counts are true counts: rows past the buffer's
    capacity are dropped, as Newton's collision pipeline drops them, and a
    ``rigid_contact_count`` or ``soft_contact_count`` above the capacity tells
    it; :meth:`get_max_contact_count` and :meth:`get_max_soft_contact_count`
    give capacities that drop none, a worst case several times the usual
    count. A step that failed writes no rows. The order of the rows changes
    from run to run (atomic counters), their set and values do not. With
    several worlds the rows of every world share the buffer, and the shapes
    tell the worlds apart. With independent worlds a step in which some worlds
    failed writes no row of those worlds and the rows of the others:
    libuipc-warp's records give the terms of a world that failed no force, so
    the feedback forces of a coupled solver's proxies (see **Coupling**) in
    such a world are zero in that step, and those in the other worlds are
    their step's. Not with ``contact_constitution="al-ipc"``.
    The output costs, only when requested, one pass over the step's contact
    terms and one over its rows, and 64 bytes per slot of the contact lists.

    **Coupling.** The solver is a
    :class:`~newton.solvers.experimental.coupled.CouplingInterface`: an entry
    of Newton's coupled solvers (``SolverCoupled``, ``SolverCoupledProxy``),
    built on the entry's ``ModelView``. It reads a view as a model; three
    things that only a view holds take their meaning from the coupler. A body
    of positive mass whose inverse mass is 0, which the view makes of a body
    another entry owns, is a kinematic body: a fixed affine body whose pose
    follows ``state_in``, and whose shapes, which the view hides, collide with
    nothing. A triangle, tetrahedron or spring whose particles are all disabled
    (mass 0 and not :attr:`~newton.ParticleFlags.ACTIVE`), as the view disables
    another entry's particles, is that entry's: neither simulated nor a contact
    surface here, its particles fixed points that collide with nothing. And the
    view's :attr:`~newton.Model.shape_contact_pairs`, which the coupler filters
    by the view's collide flags, is the set the solver's rules give (see
    **Contact**). An entry therefore steps its own content as the solver alone
    does, to the rounding of its sums.

    A proxy body or particle (:attr:`~newton.BodyFlags.PROXY`,
    :attr:`~newton.ParticleFlags.PROXY`), another entry's that this entry moves
    with the view's scaled mass, is an unknown of the step. It does not collide
    with another proxy, with a static shape or with a body of inverse mass 0,
    as the interface's default ``coupling_prepare_proxy_contacts`` drops
    Newton's contacts of those pairs (the solver reads no
    :class:`~newton.Contacts`, and returns them as they are). The coupler
    writes a proxy's pose and twist at every step; unlike any other new state,
    that write keeps the friction of the step's contacts, where libuipc
    discards friction after any write of a state: a proxy would otherwise
    never have friction. A proxy's new pose is not swept: a source solver that
    moves a proxy into this entry's content by more than the room the barrier
    leaves fails the step. A proxy's feedback is the force of the step's
    contact terms on it at the end of the step, a wrench about its centre of
    mass for a body (``coupling_harvest_proxy_wrenches``,
    ``coupling_harvest_proxy_particle_forces``), as SolverVBD harvests it,
    rather than its change of momentum. The barrier's force is steep in the
    distance, so the feedback is as accurate as the step's convergence: give
    an entry with proxies converged steps (``semi_implicit=False`` and a
    ``newton_velocity_tol`` of about 1e-4 m/s). At the default tolerances the
    step of an impact can end before the barrier has relaxed, with a force
    tens of times the change of momentum it caused. With
    ``SolverCoupledProxy``'s lagged transfer, the default, a proxy resting on
    this entry's content does not come to rest: the feedback of a contact this
    stiff reaches the source one step late and keeps the proxy bouncing,
    though its average holds the proxy up; the staggered transfer
    (``mode="staggered"``) brings it to rest. Proxies are refused with
    ``integrator="bdf2"``, whose history the coupler's writes would leave
    behind, and with ``contact_constitution="al-ipc"``, which neither sweeps a
    proxy's synced pose into its active set nor exports its contact forces.
    The effective masses, gravity and the rewinds of the proxies take the
    interface's defaults.

    A coupler that solves a step more than once (a ``SolverCoupledProxy`` or
    ``SolverCoupledADMM`` with more than one iteration) starts every pass after
    the first from the step's start. The solver keeps that start on the
    device: libuipc-warp's frame restart (the float64 positions and
    velocities, the BDF2 history, the active contact set the step takes its
    friction from, the step's requests to discard friction and restart the
    history, the joints' angles) and what the solver carries from step to step
    (what it wrote of the bodies and joints, the kinematic bodies it holds
    toward their aims, the joints' BDF2 history). The step after a coupler's
    write of the input state copies the start, and the step after a restart's
    notification copies it back: device copies, inside a captured graph too.
    Every pass therefore starts from the same float64 state with the same
    friction, where taking ``state_in``'s float32 rounding would discard the
    friction, and a pass with the inputs of the one before repeats it, bit for
    bit where a step repeats itself (where sums add with atomics, to their
    rounding). With independent worlds every world starts again, also one
    that failed in the pass before. An entry with several substeps per coupled
    step keeps the start of the first. The copy holds the vertices' state, the
    active contact lists and the joints' coordinates, and a solver whose
    coupler does not restart allocates none of it. The coupler steps its
    entries without observables, so the solver as an entry keeps the status
    bits of its steps, captured or not, with independent worlds a word for
    every world entry, and :meth:`check_status` without a container reports
    them, naming the worlds, and clears them; a restarted pass replaces the
    bits of the pass before it, so a coupled step reports its last pass. A
    reset through the coupler is :meth:`reset` with the coupler's world mask.
    Of a view that the coupler cannot compact, the other entries' joints are
    in the view, disabled, and are checked as the solver's own.

    This solver requires the optional ``libuipc-warp`` package (``import uipc_warp``).

    Args:
        model: The model to simulate.
        kappa: Barrier stiffness [Pa] of every contact pair, taken as given:
            libuipc's contact resistance, which libuipc and its pyuipc front
            end would clamp into libuipc's range (see ``adaptive_kappa``).
            ``None``, with ``adaptive_kappa``, takes the range's lower bound,
            libuipc's default. Not used with ``contact_constitution="al-ipc"``.
        adaptive_kappa: Take the contact stiffness as libuipc and its pyuipc
            front end do: inside libuipc's scene-adaptive range, which
            follows each step's ``dt`` (see **Contact**); with independent
            worlds every world's own range, that of the world alone (see
            **Worlds**).
        d_hat: Barrier activation distance [m]. ``None`` takes libuipc's
            0.01 m, or less for a model with small colliding features (see
            the note below).
        thickness: Contact thickness of the vertices of cloth and tetrahedra
            [m]; free particles take their radius and shapes their margin
            (see **Contact**).
        newton_max_iter: Maximum Newton iterations per step.
        newton_velocity_tol: Newton convergence tolerance on ``max |dx| / dt`` [m/s].
        newton_transrate_tol: Newton convergence tolerance on ``max |dA| / dt`` of
            rigid bodies [1/s].
        linear_tol_rate: Relative PCG tolerance on ``r^T M^-1 r``.
        line_search_max_iter: Maximum backtracking trials per Newton iteration.
        semi_implicit: Enable libuipc's semi-implicit early exit after
            ``semi_implicit_k_min`` iterations of full feasible steps. Under
            ``contact_constitution="al-ipc"``, whose step has no Newton
            tolerance, it enables the delay of AL-IPC's progress weight
            instead: without contact a step takes ``semi_implicit_k_min``
            Newton iterations, and one when False (see **AL-IPC contact**).
        semi_implicit_k_min: Newton iterations before the early exit applies.
        eps_velocity: Friction smoothing velocity [m/s]; below it static friction becomes a stiff spring.
        cloth_strain_rate: Strain-limiting coefficient :math:`\\eta` of the cloth
            membrane (libuipc's default 100); 0 makes the membrane quadratic.
        body_kappa: Orthogonality stiffness :math:`\\kappa` of rigid bodies [Pa]
            (libuipc's affine-body stiffness), taken as given at every step.
            ``None`` takes libuipc's 1e8 Pa, lowered for a body and a step
            whose turn of 0.1 rad about the axis of its largest moment of
            inertia it would not take in one Newton iteration (see
            **Rigid bodies**).
        joint_strength_ratio: Penalty of the joints relative to the mass of the
            dynamic bodies they join (libuipc's ``strength_ratio``; see **Joints**).
        shape_segments: Segments around a sphere, capsule, cylinder, cone or
            ellipsoid when it is triangulated.
        coupled_worlds: Simulate content in more than one world as one solve
            in which no world collides with another, where by default every
            world is solved apart from the others (see **Worlds**).
        integrator: ``"bdf1"`` (implicit Euler) or ``"bdf2"`` (see **BDF2**).
        body_rotation: ``"rigid"``, the rotation of rigid bodies integrated as
            a rigid body's, or ``"linear"``, libuipc's step, in which a rigid
            body moves along a straight line in its 12 coordinates and a spin
            loses about :math:`(\\omega \\Delta t)^2` per step (see **Rigid
            bodies**). ``None`` (the default) is ``"rigid"`` under
            ``integrator="bdf1"`` and ``"linear"`` under ``"bdf2"``, which
            refuses ``"rigid"``.
        contact_constitution: ``"ipc"`` (barrier contact) or ``"al-ipc"`` (see
            **AL-IPC contact**).
        fem_preconditioner: ``"diag"`` (block diagonal) or ``"mas"`` for
            particles, cloth and tetrahedra (see **MAS**).
        kinematic_motion: ``"jump"`` (a kinematic body takes the pose of
            ``state_in`` before the step) or ``"aim"`` (it moves inside the
            step toward the pose prescribed for its end; see **Kinematic
            bodies**).
        kinematic_strength_ratio: The strength ratio of the aims of
            ``kinematic_motion="aim"`` (libuipc's soft transform constraint).
        enable_d6_joints: Simulate the D6 joints that lock what one of
            libuipc's joints locks, as that joint (see **D6 joints**); without
            it every D6 joint is rejected.
        enable_distance_joints: Simulate distance joints of one length, as
            libuipc's soft vertex stitch of that rest length, and those of
            none (see **Distance joints**); without it every distance joint is
            rejected.
        enable_joint_limit_damping: Damp a revolute or prismatic joint that
            starts a step past one of its limits with ``joint_limit_kd``, as
            Featherstone and SemiImplicit do (see **Joints**); without it
            ``joint_limit_kd`` is not read.
        enable_joint_armature: Give a revolute or prismatic joint's
            ``joint_armature`` to its coordinate as an inertia (see
            **Joints**); without it an armature other than 0 is rejected.
        enable_joint_effort_limit: Bound a drive's force by its
            ``joint_effort_limit`` as decided at the start of every step (see
            **Joints**); without it a limit below 1e6 on a driven degree of
            freedom is rejected.

    Note:
        A step ends once the Newton update moves no vertex by more than
        ``newton_velocity_tol * dt``. When a body should gain less velocity
        per step than ``newton_velocity_tol`` (slow sliding against friction,
        for example), lower the tolerance, or the body creeps at about
        ``eps_velocity`` instead.

        With a stiff barrier (large ``kappa``) a resting contact settles just
        inside ``d_hat``, so objects rest up to ``d_hat`` apart; lower ``d_hat``
        for tighter contact. Without a ``d_hat`` argument the solver keeps it
        at or below a quarter of the smallest half-size of a colliding shape (a
        sphere's or capsule's radius, a box's smallest half extent, half of a
        mesh's thinnest extent) and half the median rest length of the edges
        of the cloth triangles and tetrahedra, chosen whenever the solve is
        built. A ``d_hat`` that spans a small shape lets it rest up to its own
        size away and puts more contact pairs inside ``d_hat`` than the contact
        lists reserve, which a captured step reports as
        :attr:`SolverIPC.StatusBits.OVERFLOW`; one as long as the edges of a
        cloth or soft body puts its non-adjacent primitives inside ``d_hat`` at
        rest, and the barrier stretches it. A smaller ``d_hat`` also leaves
        less room for the jump of a kinematic body, which is not swept (see
        ``kinematic_motion``).

        The barrier needs a gap at the start. When the surfaces of different
        bodies cross each other, or are at or inside their contact thickness (up
        to the rounding of float32 positions), at the start, the solver raises
        :class:`ValueError` and names the closest pair, as libuipc's sanity
        check rejects such a scene at init. The start is the state the first
        step takes. Where the model holds it, the solver checks it at
        construction, on the model's state (with forward kinematics for bodies
        at the identity, see **State**). A kinematic body, whose pose the
        caller writes at every step, and a link that the model leaves at the
        identity, which the caller places by writing forward kinematics into
        the state, start where the caller puts them: the pairs with such a body
        are checked by the first step on its ``state_in``, before anything runs
        from it. A first step captured in a graph cannot read its ``state_in``:
        it raises when such a pair fails in the model's state, and otherwise
        takes its start as it is; take the first step outside graph capture to
        have that start checked. The thickness of a pair is the sum of its two
        sides' (see **Contact**). Surfaces that never move may touch or cross,
        and pairs that do not collide may overlap; the inside of a body is not
        checked. A gap inside ``d_hat`` is valid, but there the barrier starts
        compressed and pushes the surfaces apart in the first steps. The check
        compares the bodies whose bounding boxes come that close, every
        primitive of one against every primitive of the other, and skips a pair
        of bodies with more than 2**31 - 1 primitive pairs with a warning. A
        state written after the first step is not checked: a step that starts
        in contact fails (:attr:`SolverIPC.StatusBits.THICKNESS_VIOLATED` or
        :attr:`SolverIPC.StatusBits.INVALID_CCD_STEP`).
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
        iteration limits are not failures.
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
        that starts at or inside its contact thickness from a plane."""

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

    SUPPORTED_OBSERVABLES = frozenset(
        {SolverBase.ObservableKind.SOLVER_RESIDUAL, SolverBase.ObservableKind.CONTACT_F, ObservableKind.STATUS}
    )

    # libuipc-warp, imported by the first solver
    _uipc_warp: ClassVar = None

    def __init__(
        self,
        model: Model,
        *,
        kappa: float | None = 1.0e9,
        adaptive_kappa: bool = False,
        d_hat: float | None = None,
        thickness: float = 0.0,
        newton_max_iter: int = 1024,
        newton_velocity_tol: float = 0.05,
        newton_transrate_tol: float = 0.1,
        linear_tol_rate: float = 1.0e-3,
        line_search_max_iter: int = 8,
        semi_implicit: bool = True,
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
        kinematic_strength_ratio: float = 100.0,
        enable_d6_joints: bool = False,
        enable_distance_joints: bool = False,
        enable_joint_limit_damping: bool = False,
        enable_joint_armature: bool = False,
        enable_joint_effort_limit: bool = False,
    ):
        if not model.device.is_cuda:
            raise NotImplementedError(f"SolverIPC runs on CUDA devices only; the model is on {model.device}.")
        super().__init__(model)
        self._coupled_worlds = bool(coupled_worlds)
        self._joint_options = _JointOptions(
            d6=bool(enable_d6_joints),
            distance=bool(enable_distance_joints),
            limit_damping=bool(enable_joint_limit_damping),
            armature=bool(enable_joint_armature),
            effort_limit=bool(enable_joint_effort_limit),
        )
        _check_supported(model, self._joint_options)
        _check_choice("integrator", integrator, ("bdf1", "bdf2"))
        if body_rotation is None:  # the rigid flight under implicit Euler, libuipc's step under BDF2 (**Rigid bodies**)
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
        if not (np.isfinite(kinematic_strength_ratio) and kinematic_strength_ratio >= 0.0):
            raise ValueError(
                f"SolverIPC: kinematic_strength_ratio must be finite and not negative, got {kinematic_strength_ratio}."
            )
        if kinematic_motion == "aim" and contact_constitution == "al-ipc":
            # libuipc's AL-IPC frame checks animated aims with a test of its own (animation_reach_target), which has
            # not been tested on this path
            raise NotImplementedError(
                "SolverIPC: kinematic_motion='aim' is not offered with contact_constitution='al-ipc'."
            )
        if adaptive_kappa and contact_constitution == "al-ipc":
            raise ValueError(
                "SolverIPC: adaptive_kappa sets the stiffness of the barrier, which contact_constitution='al-ipc' "
                "does not use."
            )
        if kappa is None and not adaptive_kappa:
            raise ValueError("SolverIPC: kappa=None, libuipc's default stiffness, needs adaptive_kappa=True.")
        if kappa is not None and not (np.isfinite(kappa) and kappa > 0.0):
            raise ValueError(f"SolverIPC: kappa must be a finite positive stiffness, got {kappa}.")
        if body_kappa is not None and not (np.isfinite(body_kappa) and body_kappa > 0.0):
            raise ValueError(f"SolverIPC: body_kappa must be a finite positive stiffness, got {body_kappa}.")
        uipc_warp = self._import_uipc_warp()
        self._d_hat_given = d_hat is not None
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
        )
        # a BDF2 history starts with an implicit Euler step: at the first step, and at every step from a state the
        # solver did not write (class docstring, **BDF2**)
        # AL-IPC sweeps the moves of kinematic bodies from where the last step left them (class docstring, **AL-IPC**)
        self._sweep = contact_constitution == "al-ipc"
        options = {"bdf2_start": "bdf1"} if integrator == "bdf2" else {}
        if self._sweep:
            options["al_sweep_kinematic"] = True
        # body_rotation="rigid": the rigid bodies' rotation as a rigid body's flight (class docstring, **Rigid bodies**)
        self._rigid_rotation = body_rotation == "rigid"
        if self._rigid_rotation:
            if "affine_rotation" not in uipc_warp.EngineOptions.names():
                raise NotImplementedError(
                    "SolverIPC: body_rotation='rigid', the default under integrator='bdf1', needs libuipc-warp's "
                    "option affine_rotation (libuipc-warp 0.4.4 or later); the installed libuipc-warp is "
                    f"{uipc_warp.__version__}. body_rotation='linear' takes libuipc's step without it."
                )
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
        step's :class:`~newton.Contacts` (see **Contact output** in the class docstring): libuipc-warp's records of
        every slot of the solver's contact lists, about 64 bytes per slot, kept from then on and allocated again when
        the solve is built again. ``contacts`` sizes ``contact_f``; any capacity is taken, the rows past it are
        dropped (:meth:`get_max_contact_count` and :meth:`get_max_soft_contact_count` give the capacities that drop
        none). Allocate before graph capture.
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

        Every contact term between two shapes is a row (see **Contact output** in the class docstring), so this is the
        number of slots of the solver's contact lists, active and friction terms together, or 0 when fewer than two
        shapes collide. It is a worst case, several times the rows a step usually writes; a smaller buffer is taken,
        and the rows past it are dropped while ``rigid_contact_count`` counts them. :meth:`grow` and an eager step
        that overflows enlarge the lists, and this with them. 0 for a model without particles and bodies.
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
        """Import libuipc-warp once: raise if it is missing or lacks what the solver calls, warn on versions it was
        not written for."""
        if cls._uipc_warp is not None:
            return cls._uipc_warp
        try:
            import uipc_warp  # noqa: PLC0415
            import uipc_warp.arrays  # noqa: PLC0415
            import uipc_warp.validation  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "SolverIPC requires the libuipc-warp package: pip install git+https://github.com/OrangeEarth15/libuipc-warp"
            ) from e
        version = getattr(uipc_warp, "__version__", "0")
        # the newest calls of each kind the solver makes, all of them public names of libuipc-warp: the frame status,
        # the contact and subscene tabulars and the stiffness policy, the start check, the joints, the iterations of
        # captured steps, the restart of a BDF2 history and the friction discard per subscene (with the rows a restart
        # per subscene takes), the stitches (with the anchors and stitch_groups of the same release), the soft
        # transform constraint (with constraint_groups), the contact forces, the external accelerations, the bodies'
        # orthogonality stiffness, the redo of an overflowing step (with the documented engine members of the same
        # release), the frame start kept on the device for a coupler's restarts and AL-IPC's sweep of kinematic moves
        needed = (
            (uipc_warp.arrays, "write_frame_status"),
            (uipc_warp.arrays.ArrayScene, "contact_model"),
            (uipc_warp.arrays.ArrayScene, "subscene_model"),
            (uipc_warp.arrays, "ContactStiffness"),
            (uipc_warp.validation.SurfaceChecker, "min_gap"),
            (uipc_warp.arrays, "joint_groups"),
            (uipc_warp.arrays, "add_frame_iterations"),
            (uipc_warp.SimEngine, "prepare_capture"),
            (uipc_warp.SimEngine, "restart_history"),
            (uipc_warp.SimEngine, "restart_rows"),
            (uipc_warp.arrays.ArrayScene, "add_stitches"),
            (uipc_warp.arrays.ArrayScene, "add_soft_transform_constraints"),
            (uipc_warp.arrays, "ContactForces"),
            (uipc_warp.arrays, "ExternalAcceleration"),
            (uipc_warp.arrays, "orthogonality"),
            (uipc_warp.SimEngine, "redo_state"),
            (uipc_warp.SimEngine, "enable_frame_restart"),
            (uipc_warp.SimEngine, "place_kinematic"),
        )
        # and the keywords of the newest calls: independent worlds, the scene's flag and the status of every world
        keywords = (
            ("ArrayScene(independent_subscenes=)", "independent_subscenes", uipc_warp.arrays.ArrayScene),
            ("write_frame_status(worlds=)", "worlds", uipc_warp.arrays.write_frame_status),
            ("EngineOptions(al_sweep_kinematic=)", "al_sweep_kinematic", uipc_warp.EngineOptions),
        )
        missing = [f"{owner.__name__}.{name}" for owner, name in needed if not hasattr(owner, name)]
        missing += [call for call, keyword, owner in keywords if keyword not in inspect.signature(owner).parameters]
        if missing:
            calls = [f"{o.__name__}.{n}" for o, n in needed] + [call for call, _, _ in keywords]
            raise ImportError(
                f"SolverIPC requires libuipc-warp 0.4 or newer with {', '.join(calls)}; the installed one ({version}) "
                f"lacks {', '.join(missing)}."
            )
        # write_frame_status writes libuipc-warp's FrameStatusFlags into the status arrays: the same bits under the
        # same names, but for the CCD bit, which libuipc-warp names after the time of impact
        flags = {int(flag): flag.name for flag in uipc_warp.arrays.FrameStatusFlags}
        bits = {int(bit): bit.name for bit in cls.StatusBits}
        renamed = int(cls.StatusBits.INVALID_CCD_STEP)
        if flags.keys() != bits.keys() or any(flags[b] != bits[b] for b in bits if b != renamed):
            raise RuntimeError(
                f"Incompatible libuipc-warp {version}: its FrameStatusFlags {flags} are not SolverIPC.StatusBits."
            )
        mismatches = _version_mismatches(version, wp.config.version, _warp_requirement())
        if mismatches:
            warnings.warn(
                "SolverIPC is not tested with " + "; ".join(mismatches) + ".",
                NewtonWarning,
                stacklevel=3,
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
        # a coupled solver's proxies (class docstring, **Coupling**)
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
        self._no_restart = wp.zeros(1, dtype=wp.int32, device=device)  # BDF1: no history to restart
        self._no_float32 = wp.zeros(1, dtype=wp.float32, device=device)  # a model array the model does not have
        self._no_bool = wp.zeros(1, dtype=wp.bool, device=device)
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
            # kinematic_motion="aim": the bodies whose state the solver keeps (held), and those whose state_out copies
            # state_in (passed through: every kinematic body, but those moving toward their aims)
            self._body_hold = wp.zeros(nb, dtype=wp.int32, device=device)
            self._body_passthrough = self._body_kinematic
            if len(self._aim_bodies):
                through = self._kinematic_body.copy()
                through[self._aim_bodies] = False
                self._body_passthrough = wp.array(through.astype(np.int32), dtype=wp.int32, device=device)
            self._aim_rows = wp.array(self._aim_bodies.astype(np.int32), dtype=wp.int32, device=device)
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
        """``NotImplementedError`` for the options a coupled solver's proxies cannot take (class docstring,
        **Coupling**): BDF2, whose history a proxy's state, written by the coupler at every step, would leave behind,
        and AL-IPC, which has no contact forces to harvest the proxies' feedback from."""
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
        """``NotImplementedError`` for the options that independent worlds do not take yet (class docstring,
        **Worlds**), each naming what to use instead, when two worlds or more hold unknowns: libuipc-warp solves the
        worlds apart from each other with libuipc's barrier contact and the block-diagonal preconditioner."""
        worlds = self._worlds.worlds
        which = f"the worlds {worlds.tolist()}" if len(worlds) <= 8 else f"{len(worlds)} worlds"
        instead = (
            f"or pass coupled_worlds=True to solve {which} as one (see Worlds in the class docstring), or give the "
            "solver a model of one world"
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
            if self._sweep and self._engine is not None:  # and the swept ones are placed there, not swept
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
        if self._restarts:  # an entry of a coupler that restarts its steps (class docstring, **Coupling**)
            if self._restore_pending:
                self._restore_step()
            elif self._save_pending:
                self._save_step()
            self._save_pending = self._restore_pending = False
        if control is None:
            control = self._control_of_model()
        self._dt, self._control = dt, control
        self._report_clamp(dt)
        self._load(state_in)
        if self._start_unchecked:
            # the first step, not captured: the start of the pairs with a body whose start the caller writes, on the
            # state just loaded from state_in, before anything runs from it (_check_start)
            _check_start(self, "the state of the first step (its state_in)")
            self._start_unchecked = False
        self._steps += 1
        self._write_forces(state_in, control)
        _write_joint_controls(self, control, dt)
        _write_stitch_stiffness(self, dt)
        _write_body_stiffness(self, dt)
        if self.model.device.is_capturing:
            # the solve joins the caller's graph; it cannot synchronize, grow or raise (see the class docstring)
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
        """The engine's state from ``state_in`` where it differs from what the solver wrote (class docstring)."""
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
            # BDF2: so is the step of the subscenes whose gravity changed since the step before (class docstring)
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
            ],
            device=self.model.device,
        )

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
                inputs=[particle_f, int(particle_f is not None), model.particle_mass, self._free_particles],
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
                    int(body_f is not None),
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
            if self._sweep:
                wp.launch(
                    _forget_kinematic_written,
                    model.body_count,
                    inputs=[self._body_kinematic],
                    outputs=[self._body_q_written, self._body_qd_written],
                    device=model.device,
                )
        if model.joint_count and state_out is not None and state_out.joint_q is not None:
            groups = self._joint_groups
            angle = self._no_joint_state if groups.revolute is None else groups.revolute.current
            distance = self._no_joint_state if groups.prismatic is None else groups.prismatic.current
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
        name = self.StatusBits(1 << status.nonfinite).name
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
                int(status is not None),
                int(residual is not None),
            ],
            outputs=[
                self._status_scratch if status is None else status,
                self._residual_scratch if residual is None else residual,
            ],
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

        A step captured in a CUDA graph cannot raise: it only writes how it ended into the observables passed to it
        (see **Status** in the class docstring). Call this after replaying the graph, outside the capture, once for
        each container the replayed steps wrote. It reads the container back from the device (one synchronization).

        A coupled solver steps its entries without observables. The solver as an entry (class docstring,
        **Coupling**) keeps the status bits of every step it takes, captured or not, until a call without a container
        reads them: that call reports every step since the last such call, and clears the bits. A pass of a coupled step
        that the coupler solves again is replaced by the next pass: a coupled step reports its last pass.

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
        bits = self.StatusBits(int(np.bitwise_or.reduce(words)))
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
        if limits:
            warnings.warn(
                f"SolverIPC: {' and '.join(limits)}; this does not fail the step.", NewtonWarning, stacklevel=2
            )
        return False

    @override
    def reset(
        self,
        state: State,
        world_mask: wp.array[wp.bool] | None = None,
        flags: StateFlags | int | None = None,
    ) -> None:
        """Make the next step start as the first step of a new solver.

        Besides its float64 state, which the next step compares with ``state_in`` (see the class docstring), the
        solver carries the friction of the last step's contacts from one step to the next, and with
        ``integrator="bdf2"`` the integrator's history; this discards them, so that the next step is an implicit
        Euler step without friction. The contacts themselves are reused only where nothing moved, where they are
        the ones a new solver finds. Launched on the device without synchronization, so it can be captured.

        Args:
            state: Unused; the next step reads its own ``state_in``.
            world_mask: Optional boolean mask of shape ``(world_count + 1,)``, the last entry the global entities.
                With several worlds the reset applies to the worlds it selects: their friction is discarded and their
                BDF2 history starts anew, and every other world keeps its own, bit for bit. With independent worlds
                (the default) the last entry selects nothing, since the global entities are no unknowns, as Newton
                lets a solver without global dynamic objects take it; with ``coupled_worlds`` it discards the friction
                of every pair with a global entity, a global ground plane among them (see **Worlds**). With one world
                the reset applies when the mask selects a world that owns particles or bodies. ``None`` resets.
            flags: Unused.
        """
        del state, flags
        world_mask = _normalize_reset_world_mask(
            world_mask, world_count=int(self.model.world_count), device=self.model.device
        )
        self._restore_pending = False  # the next step starts from the reset, not from a coupled step's start
        if self._engine is None:
            return
        if world_mask is None:
            self._engine.discard_friction()
            self._engine.restart_history()
            if self._sweep:  # the next step places the kinematic bodies at their state_in poses, without a sweep
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
        if self._sweep:
            self._engine.place_kinematic(where=self._selected)

    @override
    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        """Take changed model properties.

        Gravity, of every world (``MODEL_PROPERTIES``), and the bodies' mass, centre of mass and inertia
        (``BODY_INERTIAL_PROPERTIES``), with the joints' penalties, which follow the masses, are updated in place:
        graphs captured before take them. With ``integrator="bdf2"`` the next step starts the history of the worlds
        whose gravity changed anew (see **BDF2** in the class docstring). A change of the triangulated shapes
        (``SHAPE_PROPERTIES`` with new
        transforms or scales, ``SHAPE_TYPE_AND_SOURCE``), of the friction coefficients and the contact thicknesses
        (``SHAPE_PROPERTIES`` with a new ``shape_material_mu`` or ``shape_margin``, ``MODEL_PROPERTIES`` with a new
        ``soft_contact_mu``, ``particle_mu`` or ``particle_radius``), of the particles' masses (``MODEL_PROPERTIES``
        with a new ``particle_mass``, as a coupled solver gives its proxy particles their masses), of the springs
        (``MODEL_PROPERTIES`` with a new ``spring_rest_length``, ``spring_stiffness`` or ``spring_damping``), of the
        joints' types, bodies, frames or axes (``JOINT_PROPERTIES``) or of which bodies are kinematic
        (``BODY_PROPERTIES``, ``BODY_INERTIAL_PROPERTIES``) builds the solve again: the next step reads ``state_in`` in
        full, and graphs that
        hold :meth:`step` must be captured again. The joints' other data are read at every step; the joint flags only
        check them again against what **Joints** in the class docstring rejects. So are the stiffness and the enabled
        flags of the particle-body attachments (``CONSTRAINT_PROPERTIES``), which that flag checks again against what
        **Attachments** rejects; a new body, particle or point of an attachment builds the solve again. With
        ``kinematic_motion="aim"`` a change of a kinematic body's mass properties builds the solve again too, since
        its soft transform constraint follows the masses it was built with. Copies from the host: not inside a graph
        capture. With ``contact_constitution="al-ipc"`` new masses build the solve again too, since the AL-IPC
        penalty follows the masses it was built with.
        """
        if self.model.device.is_capturing:
            raise RuntimeError("SolverIPC.notify_model_changed() copies from the host; call it outside graph capture.")
        if self._engine is None:  # nothing moves
            return
        model = self.model
        rebuild = bool(flags & ModelFlags.SHAPE_TYPE_AND_SOURCE)
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
            if self._joint_options.distance and model.joint_count:  # a distance joint gains or loses its length
                rebuild |= not np.array_equal(_map_joints(model, self._joint_options).distance, self._distance_joints)
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
            if not rebuild:
                dyadic = _with_aim_moments(self, dyadic)
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
            _check_supported(model, self._joint_options)
            self._build_engine()
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
        """Become an entry of a coupled solver (class docstring, **Coupling**): keep the status bits of every step for
        :meth:`check_status`, and when the coupler may solve a step again from its start, keep that start on the
        device. Outside graph capture: the coupler calls this once, after construction.

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
        """Take note of a coupled solver's write of the entry's input state (class docstring, **Coupling**).

        Every step compares ``state_in`` with what the solver wrote last and takes what differs (class docstring,
        **State**), a proxy's pose and twist without discarding friction, so a write needs nothing of its own. When the
        coupler restarts steps (:meth:`coupling_notify_iteration_restarts`), the next step after a write (``dt`` above
        0) keeps its start on the device before it takes ``state_in``, and the next step after a restart's notification
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
    def coupling_prepare_proxy_contacts(
        self,
        state: State,
        contacts: Contacts | None,
        *,
        contacts_freshly_detected: bool = False,
    ) -> Contacts | None:
        """Return ``contacts`` as they are: the solver reads no :class:`~newton.Contacts`, and the contacts of a proxy
        with another proxy or with a static shape do not exist in its own (class docstring, **Coupling**)."""
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
        """The proxy bodies' feedback of the step, from its contact forces (class docstring, **Coupling**).

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
        """The proxy particles' feedback of the step, from its contact forces (class docstring, **Coupling**): zeroes
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
        bound when ``kappa`` is None (see **Contact** in the class docstring). 0 for a model without particles and
        bodies, which has no contact.

        With ``adaptive_kappa`` and independent worlds every world has a range of its own, that of the world alone
        (see **Worlds**), and the stiffness is one value per world, a float64 array of ``model.world_count``: world
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
