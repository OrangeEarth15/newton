# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The solver observables SolverIPC writes (``SolverBase.observables``): its ``STATUS``, the Newton residual
(``SOLVER_RESIDUAL``), its contacts and their forces (``CONTACT_F``), the wrench every body's parent joint passes to it
(``BODY_PARENT_F``) and the bodies' accelerations (``BODY_QDD``).

The solver's ``SUPPORTED_OBSERVABLES`` holds ``KINDS`` and ``STATUS``, and its ``supported_observables`` is
``supported(solver)``; Newton's ``SolverBase`` refuses an observable outside it. ``SolverIPC.observables`` calls
``allocate`` for every container it allocates. The hooks are those of ``features`` (its docstring).

``BODY_PARENT_F``. A joint of SolverIPC is a penalty, so the wrench it passes to its child body is the force of its
terms at the end of the step: libuipc-warp's ``ElementForces`` gives the force of every term of the joints' element
groups on each block of its element (``-g / dt^2``, a revolute or prismatic joint's ``joint_f`` included, which the
engine applies as libuipc's joint torque or force). A body's entry sums the records of its parent joint's terms on the
body's four blocks, ``Q_p`` on its translation and ``Q_A`` (one row per row of ``A``), and turns them into the world
wrench about its centre of mass, ``F = Q_p`` and ``tau = vee(Q_A A^T - A Q_A^T) - (A c) x F``, ``A`` the body's matrix
at the end of the step and ``c`` its centre of mass in its frame. To it is added the part of ``joint_f`` that the
solver applies to the body as a wrench (free and ball joints, the D6 joints that are one of them, and distance joints),
taken at the step's start as the step applies it. A body's parent joint is the joint of an articulation whose child it
is (else the first joint whose child it is); a body that is no joint's child gets zero. The terms of the joint are the
libuipc joint it maps onto (its penalty, drive and limit) or the stitch of a distance joint, and the terms another
feature module adds for the joint: a module that has element groups of a joint's own gives them with
``parent_terms(solver, engine)``, a list of ``(group, joint)`` pairs, ``joint`` the joint of every element of the group
(-1 for none). The records are allocated when a container first requests the kind, for the engine then and every engine
built after it (an extension of each engine, ``_Records``).

``BODY_QDD``. The finite difference ``(qd_out - qd_in) / dt`` of the bodies' twists, as Kamino computes it: the
container's array holds ``state_in.body_qd`` from the step's ``prepare`` to its ``finish``, which also works where
``state_in`` is ``state_out``.

A body of a world whose step failed gets zero in both (a world that fails is back at its start).
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from newton.solvers import SolverBase

from ..kernels import _d6_joint_wrenches, _joint_wrenches

if TYPE_CHECKING:
    from newton import Control, Model, ModelFlags, State

    from ..joints import _JointDofs
    from ..solver_ipc import SolverIPC
    from . import FeatureOptions

# the standard observable kinds SolverIPC writes, besides its own STATUS (SolverIPC.ObservableKind.STATUS)
KINDS = frozenset(
    {
        SolverBase.ObservableKind.SOLVER_RESIDUAL,
        SolverBase.ObservableKind.CONTACT_F,
        SolverBase.ObservableKind.BODY_PARENT_F,
        SolverBase.ObservableKind.BODY_QDD,
    }
)
# the kinds of this module that AL-IPC takes too: they need neither a Newton residual nor the barrier's contact terms
_BODY_KINDS = frozenset({SolverBase.ObservableKind.BODY_PARENT_F, SolverBase.ObservableKind.BODY_QDD})


@wp.kernel
def _world_joint_frames(
    joints: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_X_p: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    world_X_p: wp.array[wp.transform],
):
    """The parent joint frame of every joint of ``joints`` in the world, ``body_q[p] * joint_X_p[j]`` (``joint_X_p[j]``
    for a parent of -1), at ``world_X_p[j]``: with a parent of -1 and these frames the solver's joint-force kernels
    give the child's share of a joint's ``joint_f`` alone."""
    k = wp.tid()
    j = joints[k]
    X = joint_X_p[j]
    p = joint_parent[j]
    if p >= 0:
        X = body_q[p] * X
    world_X_p[j] = X


@wp.kernel
def _parent_wrenches(
    start: wp.array[wp.int32],
    first: wp.array[wp.int32],
    count: wp.array[wp.int32],
    blocks: wp.array[wp.int32],
    forces: wp.array[wp.vec3f],
    base: wp.array[wp.int32],
    x: wp.array[wp.vec3d],
    body_com: wp.array[wp.vec3],
    share: wp.array[wp.spatial_vector],
    entry: wp.array[wp.int32],
    per_world: int,
    flags: wp.array[wp.int32],
    failed: int,
    body_parent_f: wp.array[wp.spatial_vector],
):
    """The wrench body ``b``'s parent joint passes to it, about its centre of mass in the world frame: the records of
    the joint's terms (``first[t] .. first[t] + count[t]`` for ``t`` in ``start[b] .. start[b + 1]``) on the body's
    blocks ``base[b] .. base[b] + 3`` as ``(Q_p, Q_A)``, ``F = Q_p``, ``tau = vee(Q_A A^T - A Q_A^T) - (A c) x F``, plus
    the body's share of ``joint_f`` (``share``); zero where the step failed (``flags`` of the body's world ``entry``
    with ``per_world``, else ``flags[0]``)."""
    b = wp.tid()
    w = int(0)
    if per_world != 0:
        w = entry[b]
    if (flags[w] & failed) != 0:
        body_parent_f[b] = wp.spatial_vector()
        return
    o = base[b]
    Qp = wp.vec3d()
    Q0 = wp.vec3d()
    Q1 = wp.vec3d()
    Q2 = wp.vec3d()
    for t in range(start[b], start[b + 1]):
        for r in range(first[t], first[t] + count[t]):
            f = forces[r]
            q = wp.vec3d(wp.float64(f[0]), wp.float64(f[1]), wp.float64(f[2]))
            k = blocks[r] - o
            if k == 0:
                Qp = Qp + q
            elif k == 1:
                Q0 = Q0 + q
            elif k == 2:
                Q1 = Q1 + q
            elif k == 3:
                Q2 = Q2 + q
    a0 = x[o + 1]
    a1 = x[o + 2]
    a2 = x[o + 3]
    # vee(Q_A A^T - A Q_A^T): (Q_A A^T)_ij = Q_i . a_j, Q_i the record of row i and a_j row j of A; the torque about
    # the body's origin p, the symmetric part of Q_A A^T (a stretch of A) dropping out
    tau = wp.vec3d(
        wp.dot(Q2, a1) - wp.dot(a2, Q1),
        wp.dot(Q0, a2) - wp.dot(a0, Q2),
        wp.dot(Q1, a0) - wp.dot(a1, Q0),
    )
    c = body_com[b]
    cd = wp.vec3d(wp.float64(c[0]), wp.float64(c[1]), wp.float64(c[2]))
    arm = wp.vec3d(wp.dot(a0, cd), wp.dot(a1, cd), wp.dot(a2, cd))  # A c, from the origin to the centre of mass
    tau = tau - wp.cross(arm, Qp)
    s = share[b]
    body_parent_f[b] = wp.spatial_vector(
        wp.vec3(wp.float32(Qp[0]), wp.float32(Qp[1]), wp.float32(Qp[2])) + wp.spatial_top(s),
        wp.vec3(wp.float32(tau[0]), wp.float32(tau[1]), wp.float32(tau[2])) + wp.spatial_bottom(s),
    )


@wp.kernel
def _accelerations(
    body_qd: wp.array[wp.spatial_vector],
    dt: float,
    entry: wp.array[wp.int32],
    per_world: int,
    flags: wp.array[wp.int32],
    failed: int,
    body_qdd: wp.array[wp.spatial_vector],
):
    """``(body_qd - body_qdd) / dt`` into ``body_qdd``, which holds the twists of the step's start; zero where the step
    failed (``_parent_wrenches``)."""
    b = wp.tid()
    w = int(0)
    if per_world != 0:
        w = entry[b]
    if (flags[w] & failed) != 0:
        body_qdd[b] = wp.spatial_vector()
    else:
        body_qdd[b] = (body_qd[b] - body_qdd[b]) / dt


class _Records:
    """An extension of every engine built once a container has requested ``BODY_PARENT_F`` (``engine_inputs``): its
    ``attach`` makes the records of that engine's joints. libuipc-warp calls it when the engine is built, before its
    first frame and outside any capture."""

    lagged_terms = ()

    def __init__(self, solver: SolverIPC):
        self._solver = weakref.ref(solver)  # the solver holds the engine that holds this: no cycle through the solver

    def attach(self, engine):
        solver = self._solver()
        if solver is not None:
            solver._body_observables.parents = _Parents.of(solver, engine)


@dataclass
class _Parents:
    """What ``BODY_PARENT_F`` reads of one solve: the records of its engine's joints, per body the ranges of its parent
    joint's records (``start``, ``first``, ``count``, as ``_parent_wrenches`` reads them), and the parent joints whose
    ``joint_f`` the solver applies as a wrench (``wrenched``: free and ball joints and distance joints; ``d6``: the D6
    joints of three angular axes)."""

    forces: object  # libuipc-warp's ElementForces
    start: wp.array
    first: wp.array
    count: wp.array
    wrenched: wp.array
    d6: wp.array

    @classmethod
    def of(cls, solver: SolverIPC, engine) -> _Parents:
        """The records of ``engine``'s joint terms: the libuipc joints, the distance joints' stitches and the groups
        that feature modules give (``parent_terms``), and the ranges of every body's parent joint."""
        from uipc_warp.ext.element_forces import ElementForces  # noqa: PLC0415 (libuipc-warp is imported lazily)

        from . import FEATURES  # noqa: PLC0415 (the package imports this module)

        arrays = solver._uipc_warp.arrays
        terms = []  # (element group, the joint of every element)
        if solver.model.joint_count:
            found = arrays.joint_groups(engine)
            # libuipc-warp's joint groups by the solver's names of its joint kinds
            groups = {
                "revolute": found.revolute,
                "prismatic": found.prismatic,
                "fixed": found.fixed,
                "spherical": found.spherical,
            }
            for kind, group in groups.items():
                if group is not None:
                    terms.append((group, solver._joint_rows[kind].numpy()))
            for stitches in arrays.stitch_groups(engine):
                rows = stitches.rows - solver._distance_first
                if len(rows) and (rows >= 0).all():  # the stitches of distance joints, a group of their own
                    terms.append((stitches.group, solver._distance_joints[rows]))
            for feature in FEATURES:
                more = getattr(feature, "parent_terms", None)
                if more is not None:
                    terms.extend(more(solver, engine))
        forces = ElementForces(engine, [group for group, _ in terms])
        model = solver.model
        parent = _parent_joints(model)
        child = model.joint_child.numpy() if model.joint_count else np.zeros(0, np.int64)
        ranges = [[] for _ in range(model.body_count)]
        for (_, joints), offset, k in zip(terms, forces.offsets[:-1], forces.stencils, strict=True):
            for e, j in enumerate(np.asarray(joints, np.int64)):
                if j >= 0 and parent[child[j]] == j:
                    ranges[child[j]].append((int(offset) + e * k, k))
        start = np.concatenate([[0], np.cumsum([len(r) for r in ranges])]).astype(np.int32)
        flat = np.array([r for rs in ranges for r in rs], np.int32).reshape(-1, 2)
        wrenched = d6 = np.zeros(0, np.int32)
        if model.joint_count:
            own = np.zeros(model.joint_count, bool)
            own[parent[parent >= 0]] = True
            wrenched, d6 = solver._wrenched_joints.numpy(), solver._d6_joints.numpy()
            wrenched, d6 = wrenched[own[wrenched]], d6[own[d6]]
        device = model.device

        def ints(a):
            return wp.array(np.ascontiguousarray(a, np.int32), dtype=wp.int32, device=device)

        return cls(forces, ints(start), ints(flat[:, 0]), ints(flat[:, 1]), ints(wrenched), ints(d6))


@dataclass
class _BodyObservables:
    """What the body observables of one solver keep (``solver._body_observables``), made by the first container that
    requests one: the step's failures (``flags``, one word per world entry, the global world's last); for
    ``BODY_PARENT_F``, what it reads of the solve (``parents``, None until requested), the bodies' shares of their
    parent joints' ``joint_f`` (``share``), and for the solver's joint-force kernels a parent of -1 for every joint and
    the joints' parent frames in the world at the step's start (``no_parent``, ``world_X_p``)."""

    flags: wp.array
    parents: _Parents | None = None
    share: wp.array | None = None
    no_parent: wp.array | None = None
    world_X_p: wp.array | None = None


def _parent_joints(model: Model) -> np.ndarray:
    """The parent joint of every body, -1 for none: the joint of an articulation whose child it is, else the first joint
    whose child it is (loop-closing joints are in no articulation)."""
    parent = np.full(model.body_count, -1, np.int64)
    if model.joint_count == 0:
        return parent
    child = model.joint_child.numpy()
    tree = model.joint_articulation.numpy() >= 0 if model.joint_articulation is not None else np.zeros(len(child), bool)
    for j in range(model.joint_count - 1, -1, -1):  # the first joint wins: backwards, each overwrites the later
        if child[j] >= 0 and not tree[j]:
            parent[child[j]] = j
    for j in range(model.joint_count - 1, -1, -1):
        if child[j] >= 0 and tree[j]:
            parent[child[j]] = j
    return parent


def _parents_requested(solver: SolverIPC) -> None:
    """Make what ``BODY_PARENT_F`` needs, once: the records of the solver's engine, and the joints whose ``joint_f``
    acts as a wrench."""
    state = solver._body_observables
    model, device = solver.model, solver.model.device
    try:
        import uipc_warp.ext.element_forces  # noqa: F401, PLC0415
    except ImportError as e:
        raise ImportError(
            f"SolverIPC: BODY_PARENT_F takes the joints' forces from libuipc-warp's uipc_warp.ext.element_forces, which "
            f"the installed libuipc-warp ({getattr(solver._uipc_warp, '__version__', '?')}) lacks; install a newer one."
        ) from e
    state.share = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=device)
    if model.joint_count:
        state.no_parent = wp.full(model.joint_count, -1, dtype=wp.int32, device=device)
        state.world_X_p = wp.clone(model.joint_X_p)
    state.parents = _Parents.of(solver, solver._engine)


def supported(solver: SolverIPC) -> frozenset[str]:
    """The observable kinds ``solver`` writes: those of ``SolverIPC.SUPPORTED_OBSERVABLES`` (``KINDS`` and
    ``STATUS``), and with ``contact_constitution="al-ipc"`` ``STATUS``, ``BODY_PARENT_F`` and ``BODY_QDD`` only, since
    the AL-IPC frame ends on its own criterion, without a Newton residual, and its contact is an active set with
    multipliers, not the barrier and friction terms the contact output exports."""
    config = getattr(solver, "_config", None)
    if config is not None and config.contact_constitution == "al-ipc":
        return frozenset({solver.ObservableKind.STATUS, *_BODY_KINDS})
    return solver.SUPPORTED_OBSERVABLES


def allocate(solver: SolverIPC, observables: SolverBase.Observables) -> None:
    """What ``BODY_PARENT_F`` and ``BODY_QDD`` need beyond the container's arrays, made when a container first requests
    them, outside any graph capture: the records of the joints' forces and the step's failures (``_BodyObservables``).
    Nothing for a model without bodies, or for a container that requests neither."""
    if solver._engine is None or solver.model.body_count == 0 or not (observables.kinds & _BODY_KINDS):
        return
    if getattr(solver, "_body_observables", None) is None:
        entries = solver.model.world_count + 1
        solver._body_observables = _BodyObservables(wp.zeros(entries, dtype=wp.int32, device=solver.model.device))
    if observables.is_requested(SolverBase.ObservableKind.BODY_PARENT_F) and solver._body_observables.share is None:
        _parents_requested(solver)


def check(model: Model, options: FeatureOptions, at: str) -> None:
    """Nothing of the model."""


def joint_refusal(category: str, dofs: _JointDofs) -> tuple[int, str] | None:
    """Nothing of the joints."""
    return None


def build(solver: SolverIPC, scene) -> None:
    """Nothing."""


def engine_inputs(solver: SolverIPC, inputs: dict) -> None:
    """Once a container has requested ``BODY_PARENT_F``, the extension that makes the records of the engine's joints
    when it is built (``_Records``)."""
    state = getattr(solver, "_body_observables", None)
    if state is not None and state.share is not None:
        inputs["extensions"].append(_Records(solver))


def prepare(solver: SolverIPC, state_in: State, control: Control, dt: float) -> None:
    """What the step's start holds for the step's container (``solver._observables``): the twists of ``state_in`` in
    its ``body_qdd`` (``BODY_QDD``), and the bodies' shares of their parent joints' ``joint_f`` at ``state_in``'s poses,
    as the solver applies it (``BODY_PARENT_F``)."""
    observables, state = solver._observables, getattr(solver, "_body_observables", None)
    if observables is None or state is None:
        return
    if observables.is_requested(SolverBase.ObservableKind.BODY_QDD):
        wp.copy(observables.body_qdd, state_in.body_qd)
    if not observables.is_requested(SolverBase.ObservableKind.BODY_PARENT_F):
        return
    state.share.zero_()
    model, parents = solver.model, state.parents
    joint_f = control.joint_f if control is not None else None
    if joint_f is None or model.joint_count == 0:
        return
    for joints in (parents.wrenched, parents.d6):
        if joints.shape[0]:
            wp.launch(
                _world_joint_frames,
                joints.shape[0],
                inputs=[joints, model.joint_parent, model.joint_X_p, state_in.body_q],
                outputs=[state.world_X_p],
                device=model.device,
            )
    # the solver's own kernels with every parent at -1 and the parent frames in the world: the child's share alone
    if parents.wrenched.shape[0]:
        wp.launch(
            _joint_wrenches,
            parents.wrenched.shape[0],
            inputs=[
                parents.wrenched,
                solver._joint_store_type,
                state.no_parent,
                model.joint_child,
                state.world_X_p,
                model.joint_qd_start,
                model.joint_enabled,
                joint_f,
                state_in.body_q,
                model.body_com,
            ],
            outputs=[state.share],
            device=model.device,
        )
    if parents.d6.shape[0]:
        wp.launch(
            _d6_joint_wrenches,
            parents.d6.shape[0],
            inputs=[
                parents.d6,
                state.no_parent,
                model.joint_child,
                state.world_X_p,
                model.joint_X_c,
                model.joint_axis,
                model.joint_dof_dim,
                model.joint_qd_start,
                model.joint_enabled,
                joint_f,
                state_in.body_q,
                model.body_com,
            ],
            outputs=[state.share],
            device=model.device,
        )


def finish(solver: SolverIPC, state_out: State, observables: SolverBase.Observables | None) -> None:
    """``BODY_PARENT_F`` and ``BODY_QDD`` of the step, when the container requests them (module docstring), zero in a
    world whose step failed."""
    state = getattr(solver, "_body_observables", None)
    if observables is None or state is None or state_out is None:
        return
    qdd = observables.is_requested(SolverBase.ObservableKind.BODY_QDD)
    parent_f = observables.is_requested(SolverBase.ObservableKind.BODY_PARENT_F)
    if not (qdd or parent_f):
        return
    model = solver.model
    failed = int(solver.StatusBits.FAILED)
    per_world = int(solver._independent)
    if solver._independent:  # every world's word at its entry
        solver._write_world_status(state.flags, None)
    else:  # the one solve's word
        solver._uipc_warp.arrays.write_frame_status(solver._engine, state.flags, None, index=0)
    if parent_f:
        records = state.parents
        records.forces.update()
        wp.launch(
            _parent_wrenches,
            model.body_count,
            inputs=[
                records.start,
                records.first,
                records.count,
                records.forces.blocks,
                records.forces.forces,
                solver._body_base,
                solver._engine.vm.x,
                model.body_com,
                state.share,
                solver._body_entry,
                per_world,
                state.flags,
                failed,
            ],
            outputs=[observables.body_parent_f],
            device=model.device,
        )
    if qdd:
        wp.launch(
            _accelerations,
            model.body_count,
            inputs=[state_out.body_qd, float(solver._dt), solver._body_entry, per_world, state.flags, failed],
            outputs=[observables.body_qdd],
            device=model.device,
        )


def notify(solver: SolverIPC, flags: ModelFlags | int) -> bool:
    """Nothing: the records of a solve built again are made by its engine's extension (``engine_inputs``), and every
    step reads the bodies' centres of mass and the joints' inputs from the model."""
    return False
