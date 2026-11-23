# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Terms on joint coordinates: mimic joints, MuJoCo's fixed tendons, and the friction of revolute and prismatic joints
(:ref:`ipc-joint-terms`).

libuipc's joints act on one coordinate each and have no friction. SolverIPC gives each of these terms to libuipc-warp
as an item of its joint coordinate terms (``uipc_warp.ext.coordinate_terms``): a coordinate ``L = c_0 + sum_t c_t
q_t`` over the coordinates ``q_t`` of the joints that become libuipc's revolute and prismatic joints, which are
Newton's ``joint_q``, with convex potentials of ``L``:

- a mimic joint (``joint_mimic_joint`` and ``joint_mimic_coeffs``, and the rows of the deprecated
  ``constraint_mimic_*``): ``L = q_f - c_1 q_r - c_0`` held at 0 by a penalty of a joint's strength,
  ``joint_strength_ratio`` times the mass of the dynamic bodies the two joints join, while both joints (and the row)
  are enabled;
- a fixed tendon of MuJoCo (``model.mujoco.tendon_*`` of type 0): ``L = sum coef_i q_i`` with its stiffness toward its
  spring length (a band when the pair differs, the model's length for -1), its damping, its armature, its friction
  loss and its limit, a cubic penalty as SolverIPC takes a joint's limit, with the damping of the limit past it;
- the friction of a joint: the ``joint_friction`` of a revolute or prismatic joint's degree of freedom, an item of
  that joint alone.

A model with friction on any degree of freedom of any joint (``joint_friction`` over all of them, revolute, prismatic,
ball or other, whichever module holds it) or on a fixed tendon takes Newton tolerances of at most
``_FRICTION_TOLERANCE`` times ``eps_velocity`` where it leaves them at their default (``_friction_tolerances``).
libuipc's Newton loop ends when its step is below the tolerance, and within its smoothing band the friction is so stiff
that Newton's steps there are short, the first from rest about ``f / (2 F)`` times ``eps_velocity`` for a load ``f``
above the friction ``F``: at the default tolerances the loop would end inside the band, and a joint loaded beyond its
friction would creep instead of sliding. A tolerance given above the bound gets a ``NewtonWarning``. MuJoCo's JOINT equality rows that repeat the
mimic rows Newton's importers made of them are held by those (``held_equality_rows``).

Refused: Newton's own tendons (``ModelBuilder.add_tendon``, spatial), MuJoCo's spatial tendons and actuators that
drive a tendon, a mimic or tendon over a joint that becomes no revolute or prismatic joint, and the friction of free
and distance joints (the ball module refuses that of ball joints). The hooks are those of ``features`` (its
docstring); ``parent_terms`` gives the terms' forces to ``BODY_PARENT_F``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

import newton
from newton import JointType, ModelFlags, NewtonWarning

if TYPE_CHECKING:
    from newton import Control, Model, State
    from newton.solvers import SolverBase

    from ..joints import _JointDofs
    from ..solver_ipc import SolverIPC
    from . import FeatureOptions

# the transmission of a MuJoCo actuator that drives a tendon (SolverMuJoCo.TrnType.TENDON) and the solref modes of a
# tendon's limit (newton/_src/solvers/mujoco/constants.py): gains given in force space, or MuJoCo's solref
_TRN_TENDON = 2
_SOLREF_FORCE_SPACE = 0
# MuJoCo's default solref of a limit (time constant [s], damping ratio)
_DEFAULT_SOLREF = (0.02, 1.0)
# MuJoCo's JOINT equality (EqType.JOINT) and the target kind with which Newton's importers link it to the mimic row
# they made of it (MjcEqualityTargetKind.MIMIC); neither has a public name in Newton
_EQ_JOINT, _EQ_TARGET_MIMIC = 2, 2
# the Newton tolerances, as a fraction of eps_velocity, that a model with friction takes by default
# (``_friction_tolerances``): the largest at which a joint under 1.5 times its friction slides at (f - F) / I to 1%
# whatever its axis. Newton's steps inside the smoothing band are short, the first about f / (2 F) eps_velocity, the
# smallest about sqrt(f / F - 1) eps_velocity, and the test takes a step's largest component, 1 / sqrt(3) of it for an
# axis along a diagonal: measured, 1.5 F slides up to 0.7 eps_velocity along a coordinate axis and up to 0.4 along the
# diagonal. A load nearer the friction needs less (1.2 F slides up to 0.45 and 0.25, 1.1 F up to 0.3 and 0.1).
_FRICTION_TOLERANCE = 0.4
# the overshoot at which SolverIPC's cubic limit has a limit's stiffness (kernels._LIMIT_OVERSHOOT) [m or rad]
_LIMIT_OVERSHOOT = wp.constant(wp.float64(1.0e-3))


def _one_coordinate(model: Model) -> np.ndarray:
    """Per joint, the kind of libuipc joint of one coordinate it becomes: REVOLUTE, PRISMATIC (a D6 joint of one free
    angular or linear axis among them, as ``joints._map_joints`` maps it), else -1."""
    if model.joint_count == 0:
        return np.zeros(0, np.int32)
    joint_type = model.joint_type.numpy()
    dof_dim = model.joint_dof_dim.numpy()
    out = np.full(model.joint_count, -1, np.int32)
    out[joint_type == int(JointType.REVOLUTE)] = int(JointType.REVOLUTE)
    out[joint_type == int(JointType.PRISMATIC)] = int(JointType.PRISMATIC)
    d6 = joint_type == int(JointType.D6)
    out[d6 & (dof_dim[:, 0] == 0) & (dof_dim[:, 1] == 1)] = int(JointType.REVOLUTE)
    out[d6 & (dof_dim[:, 0] == 1) & (dof_dim[:, 1] == 0)] = int(JointType.PRISMATIC)
    return out


def _mimics(model: Model) -> list[tuple[int, int, float, float, int]]:
    """The model's mimic relations ``(follower, reference, offset, multiplier, row)``: those of ``joint_mimic_joint``
    (``row`` -1) and the rows of ``constraint_mimic_*`` (``row`` their index)."""
    out = []
    if model.joint_count and model.joint_mimic_joint is not None:
        reference = model.joint_mimic_joint.numpy()
        coeffs = model.joint_mimic_coeffs.numpy().astype(np.float64)
        for f in np.flatnonzero(reference >= 0):
            out.append((int(f), int(reference[f]), float(coeffs[f, 0]), float(coeffs[f, 1]), -1))
    if model.constraint_mimic_count:
        rows = (model.constraint_mimic_joint0, model.constraint_mimic_joint1, model.constraint_mimic_coef0)
        joint0, joint1, coef0 = (a.numpy() for a in rows)
        coef1 = model.constraint_mimic_coef1.numpy()
        for row in range(int(model.constraint_mimic_count)):
            out.append((int(joint0[row]), int(joint1[row]), float(coef0[row]), float(coef1[row]), row))
    return out


def _tendons(model: Model) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """MuJoCo's tendons of ``model.mujoco`` as ``(tendon, joints, coefficients)``, spatial ones with no joints."""
    mujoco = getattr(model, "mujoco", None)
    count = model.custom_frequency_counts.get("mujoco:tendon", 0)
    if mujoco is None or count == 0:
        return []
    kind = mujoco.tendon_type.numpy() if hasattr(mujoco, "tendon_type") else np.zeros(count, np.int32)
    adr, num = mujoco.tendon_joint_adr.numpy(), mujoco.tendon_joint_num.numpy()
    joints = mujoco.tendon_joint.numpy() if num.sum() else np.zeros(0, np.int32)
    coef = mujoco.tendon_coef.numpy().astype(np.float64) if num.sum() else np.zeros(0)
    out = []
    for t in range(count):
        rows = slice(int(adr[t]), int(adr[t]) + int(num[t]))
        fixed = int(kind[t]) == 0
        out.append((t, joints[rows] if fixed else np.zeros(0, np.int32), coef[rows] if fixed else np.zeros(0)))
    return out


def _tendon_actuators(model: Model) -> int:
    """The MuJoCo actuators of ``model.mujoco`` that drive a tendon."""
    mujoco = getattr(model, "mujoco", None)
    if mujoco is None or model.custom_frequency_counts.get("mujoco:actuator", 0) == 0:
        return 0
    return int(np.count_nonzero(mujoco.actuator_trntype.numpy() == _TRN_TENDON))


def check(model: Model, options: FeatureOptions, at: str) -> None:
    """``NotImplementedError`` for what SolverIPC cannot hold of the tendons (``"tendons"``): Newton's own tendons,
    MuJoCo's spatial tendons, actuators that drive a tendon, and a fixed tendon over a joint that becomes no revolute
    or prismatic joint; and of the mimic relations (``"constraints"``): one between joints that are not both
    revolute or prismatic joints (or D6 joints of one coordinate)."""
    one = _one_coordinate(model)
    if at == "tendons":
        if model.tendon_count:
            raise NotImplementedError(
                f"SolverIPC: the model has {model.tendon_count} tendons of Newton's own (ModelBuilder.add_tendon); "
                "remove them. SolverIPC simulates MuJoCo's fixed tendons, whose length is a linear combination of joint "
                "coordinates, and no tendon routed through space."
            )
        for t, joints, _ in _tendons(model):
            if not len(joints):
                raise NotImplementedError(
                    f"SolverIPC: MuJoCo's tendon {t} of the model is a spatial tendon; remove it. SolverIPC simulates "
                    "fixed tendons only, whose length is a linear combination of joint coordinates."
                )
            bad = [int(j) for j in joints if not (0 <= j < model.joint_count) or one[j] < 0]
            if bad:
                raise NotImplementedError(
                    f"SolverIPC: MuJoCo's fixed tendon {t} runs over joint {bad[0]}, which is no revolute or prismatic "
                    "joint; remove it from the tendon. SolverIPC takes the coordinates of revolute and prismatic joints "
                    "(and of D6 joints of one axis) only."
                )
        mujoco = getattr(model, "mujoco", None)
        if mujoco is not None and model.custom_frequency_counts.get("mujoco:tendon", 0):
            margin = mujoco.tendon_margin.numpy()
            if np.any(margin != 0.0):
                t = int(np.flatnonzero(margin != 0.0)[0])
                raise NotImplementedError(
                    f"SolverIPC: MuJoCo's tendon {t} has a limit margin of {float(margin[t]):g}; set it to 0. SolverIPC's "
                    "limit acts from the bound on, as a joint's does."
                )
        actuators = _tendon_actuators(model)
        if actuators:
            raise NotImplementedError(
                f"SolverIPC: the model has {actuators} MuJoCo actuators that drive a tendon; remove them. SolverIPC "
                "reads no MuJoCo actuator controls."
            )
    if at == "constraints":
        for f, r, _, _, row in _mimics(model):
            for role, j in (("follower", f), ("reference", r)):
                if one[j] < 0:
                    what = "the mimic constraint row " + str(row) if row >= 0 else f"the mimic joint {f}"
                    name = JointType(int(model.joint_type.numpy()[j])).name
                    raise NotImplementedError(
                        f"SolverIPC: the {role} of {what} is the {name} joint {j}, which has more or other coordinates "
                        "than one angle or one distance; mimic revolute and prismatic joints (or D6 joints of one "
                        "axis). SolverIPC holds a mimic relation of one coordinate."
                    )


def held_equality_rows(model: Model) -> np.ndarray:
    """Per row of ``model.mujoco``'s equality constraints, whether this module holds it: a JOINT row that Newton's
    importers turned into a row of ``constraint_mimic_*`` (target kind MIMIC) and that repeats it, the same two whole
    joints (follower ``joint1``, reference ``joint2``) and the same linear coefficients (``polycoef`` = ``coef0``,
    ``coef1`` and no term of higher order), while that mimic row is enabled. Any other JOINT row, a polynomial of
    higher order or other coefficients among them, is not held (``solver_ipc._unprojected_equality_rows``)."""
    mujoco = getattr(model, "mujoco", None)
    count = int(getattr(mujoco, "equality_constraint_count", 0) or 0) if mujoco is not None else 0
    held = np.zeros(count, bool)
    rows = int(model.constraint_mimic_count)
    if count == 0 or rows == 0:
        return held
    kind, target = mujoco.equality_constraint_type.numpy(), mujoco.equality_constraint_target.numpy()
    target_kind = mujoco.equality_constraint_target_kind.numpy()
    joint1, joint2 = mujoco.equality_constraint_joint1.numpy(), mujoco.equality_constraint_joint2.numpy()
    dof1, dof2 = mujoco.equality_constraint_joint1_dof.numpy(), mujoco.equality_constraint_joint2_dof.numpy()
    polycoef = mujoco.equality_constraint_polycoef.numpy()
    follower, reference = model.constraint_mimic_joint0.numpy(), model.constraint_mimic_joint1.numpy()
    coef0, coef1 = model.constraint_mimic_coef0.numpy(), model.constraint_mimic_coef1.numpy()
    enabled = model.constraint_mimic_enabled.numpy()
    qd_start = model.joint_qd_start.numpy()
    for e in np.flatnonzero((kind == _EQ_JOINT) & (target_kind == _EQ_TARGET_MIMIC)):
        m = int(target[e])
        if not (0 <= m < rows and enabled[m]):
            continue
        j1, j2 = int(joint1[e]), int(joint2[e])
        whole = all(d in (-1, int(qd_start[j])) for d, j in ((int(dof1[e]), j1), (int(dof2[e]), j2)) if j >= 0)
        held[e] = (
            whole
            and (j1, j2) == (int(follower[m]), int(reference[m]))
            and polycoef[e][0] == coef0[m]
            and polycoef[e][1] == coef1[m]
            and not np.any(polycoef[e][2:])
        )
    return held


def joint_refusal(category: str, dofs: _JointDofs) -> tuple[int, str] | None:
    """``"friction"``: the first degree of freedom of a joint that maps onto a free joint (a free, distance or D6
    joint of six axes) with a ``joint_friction`` other than 0."""
    if category != "friction":
        return None
    friction = dofs.value("joint_friction")
    return dofs.refusal(
        np.flatnonzero((friction != 0.0) & (dofs.kind == int(JointType.FREE))),
        lambda d: f"joint_friction {float(friction[d])!r}",
        "set it to 0. SolverIPC holds the friction of revolute, prismatic, ball and universal joints only, as a force "
        "on one coordinate or a torque about each axis of a ball joint or each angle of a universal joint",
    )


# ================================================================================================ the solve's items
@dataclass
class _Plan:
    """The coordinate terms of a solve (``solver._joint_terms``, None for a model without any).

    Per item: its terms (``items``, ``CoordinateTerms``' form), its coefficients as the terms are made with them
    (``values``), and what a step writes into them: the joints whose ``joint_enabled`` it takes and its row of
    ``constraint_mimic_*`` (``gate``, -1 for none), the degree of freedom whose ``joint_friction`` it takes (``dof``),
    and a tendon's limit stiffness and damping in force space with its base damping (``limit_ke``, ``limit_kd``,
    ``damping``; with the coordinates of the tendon's joints, ``tendon_first``, ``tendon_kind``, ``tendon_row``,
    ``tendon_coef``). Per term its joint (``joints``). ``terms``: the ``CoordinateTerms`` of the solve's engine."""

    items: list
    joints: np.ndarray
    values: dict
    gate: wp.array
    dof: wp.array
    limit_ke: wp.array
    limit_kd: wp.array
    damping: wp.array
    tendon_first: wp.array
    tendon_kind: wp.array
    tendon_row: wp.array
    tendon_coef: wp.array
    none: wp.array
    mimic: list = field(default_factory=list)  # (item, the bodies the two joints join)
    terms: object = None


def _has_friction(model: Model) -> bool:
    """Whether a degree of freedom of any joint, of whatever kind, has a ``joint_friction`` other than 0, or one of
    MuJoCo's fixed tendons a friction loss: a model whose default Newton tolerances are lowered
    (``_friction_tolerances``)."""
    if model.joint_dof_count and model.joint_friction is not None and np.any(model.joint_friction.numpy() != 0.0):
        return True
    fixed = [t for t, joints, _ in _tendons(model) if len(joints)]
    return bool(fixed) and bool(np.any(model.mujoco.tendon_frictionloss.numpy()[fixed] != 0.0))


def _signature(model: Model) -> tuple:
    """What the solve is built from here: the mimic relations' joints, the tendons' joints and coefficients, the
    degrees of freedom with friction, and whether the model has friction (the Newton tolerances). A change of it builds
    the solve again (``notify``)."""
    friction = np.zeros(0)
    if model.joint_dof_count and model.joint_friction is not None:
        friction = np.flatnonzero(model.joint_friction.numpy() != 0.0)
    mimics = tuple((f, r, row) for f, r, _, _, row in _mimics(model))
    tendons = tuple((t, tuple(j.tolist()), tuple(c.tolist())) for t, j, c in _tendons(model))
    return mimics, tendons, tuple(friction.tolist()), _has_friction(model)


def _rows(solver: SolverIPC) -> dict:
    """Per joint kind the row of every Newton joint in its libuipc group (-1 where it has none)."""
    out = {}
    for kind in ("revolute", "prismatic"):
        row = np.full(solver.model.joint_count, -1, np.int64)
        if kind in solver._joint_rows:
            joints = solver._joint_rows[kind].numpy()
            row[joints] = np.arange(len(joints))
        out[kind] = row
    return out


def _mimic_stiffness(solver: SolverIPC, bodies: np.ndarray) -> float:
    """A mimic relation's penalty: ``joint_strength_ratio`` times the mass of the dynamic bodies among ``bodies``."""
    mass = solver.model.body_mass.numpy().astype(np.float64)
    dynamic = np.where(solver._kinematic_body, 0.0, mass)
    return solver._joint_strength_ratio * float(dynamic[bodies[bodies >= 0]].sum())


def _inverse_weights(model: Model, tendons: list) -> np.ndarray:
    """MuJoCo's ``tendon_invweight0`` of every fixed tendon, ``coef^T H^-1 coef`` with the joint-space inertia ``H``
    of its articulation at ``model.joint_q`` (armature included, ``newton.eval_mass_matrix``); 0 for a tendon over
    joints of no articulation or of two."""
    out = np.zeros(len(tendons))
    if not tendons or model.articulation_count == 0:
        return out
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    H = newton.eval_mass_matrix(model, state).numpy()
    articulation = model.joint_articulation.numpy()
    start = model.articulation_start.numpy()
    qd_start = model.joint_qd_start.numpy()
    for k, (_, joints, coef) in enumerate(tendons):
        arts = {int(articulation[j]) for j in joints}
        if len(arts) != 1 or -1 in arts:
            continue
        a = arts.pop()
        first = int(qd_start[start[a]])
        n = int(qd_start[start[a + 1]]) - first
        c = np.zeros(n)
        for j, cj in zip(joints, coef, strict=True):
            c[int(qd_start[j]) - first] += cj
        out[k] = float(c @ np.linalg.solve(H[a, :n, :n].astype(np.float64), c))
    return out


def _tendon_limits(model: Model, tendons: list) -> tuple[np.ndarray, np.ndarray]:
    """The force-space stiffness and damping of every fixed tendon's limit: its ``tendon_limit_ke`` and
    ``tendon_limit_kd`` when they are given in force space, else from its solref ``(T, zeta)`` and the width ``d_max``
    of its solimp as Newton's MuJoCo solver reports them, ``1 / (T^2 zeta^2 w)`` and ``2 / (T w)`` with ``w = invweight
    (1 - d_max)``."""
    mujoco = model.mujoco
    mode = mujoco.tendon_solref_limit_mode.numpy()
    ke, kd = mujoco.tendon_limit_ke.numpy().astype(np.float64), mujoco.tendon_limit_kd.numpy().astype(np.float64)
    solref, solimp = mujoco.tendon_solref_limit.numpy().astype(np.float64), mujoco.tendon_solimp_limit.numpy()
    invweight = _inverse_weights(model, tendons)
    out_ke, out_kd = np.zeros(len(tendons)), np.zeros(len(tendons))
    for k, (t, _, _) in enumerate(tendons):
        if int(mode[t]) == _SOLREF_FORCE_SPACE:
            out_ke[k], out_kd[k] = ke[t], kd[t]
            continue
        T, zeta = solref[t] if (solref[t] > 0.0).all() else _DEFAULT_SOLREF
        w = invweight[k] * (1.0 - float(solimp[t][1]))
        if w > 0.0:
            out_ke[k], out_kd[k] = 1.0 / (T * T * zeta * zeta * w), 2.0 / (T * w)
    return out_ke, out_kd


def _make_plan(solver: SolverIPC) -> _Plan | None:
    """The items of the model's mimic relations, fixed tendons and joint friction (``_Plan``), or None for none."""
    model = solver.model
    if model.joint_count == 0:
        return None
    one, rows = _one_coordinate(model), _rows(solver)
    kind_of = {int(JointType.REVOLUTE): "revolute", int(JointType.PRISMATIC): "prismatic"}
    parent, child = model.joint_parent.numpy(), model.joint_child.numpy()
    q0 = model.joint_q.numpy().astype(np.float64)
    q_start, qd_start = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
    items, joints, values, gate, dof, mimic = [], [], [], [], [], []
    tendon_terms, tendon_items = [], []

    def term(j: int, c: float):
        kind = kind_of[int(one[j])]
        joints.append(j)
        return kind, int(rows[kind][j]), c

    def item(terms: list, gates=(-1, -1, -1), friction_dof=-1, **coefficients):
        items.append(terms)
        values.append(coefficients)
        gate.append(gates)
        dof.append(friction_dof)
        return len(items) - 1

    for f, r, c0, c1, row in _mimics(model):
        bodies = np.unique([parent[f], child[f], parent[r], child[r]])
        i = item([term(f, 1.0), term(r, -c1)], (f, r, row), offset=-c0)
        mimic.append((i, bodies))
    tendons = [t for t in _tendons(model) if len(t[1])]
    if tendons:
        mujoco = model.mujoco
        stiffness, damping = mujoco.tendon_stiffness.numpy(), mujoco.tendon_damping.numpy()
        friction, armature = mujoco.tendon_frictionloss.numpy(), mujoco.tendon_armature.numpy()
        length, limited, bounds = (
            mujoco.tendon_springlength.numpy(),
            mujoco.tendon_limited.numpy(),
            mujoco.tendon_range.numpy(),
        )
        limit_ke, limit_kd = _tendon_limits(model, tendons)
        for k, (t, tj, coef) in enumerate(tendons):
            # SolverMuJoCo's reading of the spring length: -1 first, the length at the model's coordinates; a pair in
            # order, a band; else one length
            lo, hi = (float(v) for v in length[t])
            if lo == -1.0:
                lo = hi = float(sum(c * q0[q_start[j]] for j, c in zip(tj, coef, strict=True)))
            elif hi < lo:
                hi = lo
            on_limit = int(limited[t]) == 1 or (int(limited[t]) == 2 and bounds[t][0] < bounds[t][1])
            i = item(
                [term(int(j), float(c)) for j, c in zip(tj, coef, strict=True)],
                spring=float(stiffness[t]), spring_lower=min(lo, hi), spring_upper=max(lo, hi),
                damping=float(damping[t]), armature=float(armature[t]), friction=float(friction[t]),
                limit_lower=float(bounds[t][0]) if on_limit else 0.0, limit_upper=float(bounds[t][1]) if on_limit else 0.0,
            )  # fmt: skip
            tendon_items.append((i, float(limit_ke[k]) if on_limit else 0.0, float(limit_kd[k]) if on_limit else 0.0))
            tendon_terms.append([(1 if one[int(j)] == int(JointType.PRISMATIC) else 0, rows[kind_of[int(one[int(j)])]][int(j)],
                                  float(c)) for j, c in zip(tj, coef, strict=True)])  # fmt: skip
    if model.joint_dof_count and model.joint_friction is not None:
        joint_friction = model.joint_friction.numpy()
        for j in np.flatnonzero(one >= 0):
            d = int(qd_start[j])
            if joint_friction[d] != 0.0:
                item([term(int(j), 1.0)], (int(j), -1, -1), friction_dof=d)
    if not items:
        return None
    n, device = len(items), model.device
    names = ("offset", "stiffness", "spring", "spring_lower", "spring_upper", "limit_lower", "limit_upper", "damping",
             "armature", "friction")  # fmt: skip
    table = {name: np.array([v.get(name, 0.0) for v in values]) for name in names}
    for i, bodies in mimic:
        table["stiffness"][i] = _mimic_stiffness(solver, bodies)
    ke, kd, base = np.zeros(n), np.zeros(n), np.zeros(n)
    first = np.full(n + 1, 0, np.int64)
    flat = []
    for (i, k_e, k_d), terms in zip(tendon_items, tendon_terms, strict=True):
        ke[i], kd[i], base[i] = k_e, k_d, table["damping"][i]
        flat.extend(terms)
        first[i + 1 :] += len(terms)
    flat = np.array(flat, np.float64).reshape(-1, 3) if flat else np.zeros((1, 3))

    def arr(a, dtype):
        a = np.asarray(a)
        return wp.array(a if a.size else np.zeros((1, *a.shape[1:]), a.dtype), dtype=dtype, device=device)

    return _Plan(
        items=items, joints=np.asarray(joints, np.int64), values=table,
        gate=arr(np.asarray(gate, np.int32), wp.vec3i), dof=arr(np.asarray(dof, np.int32), wp.int32),
        limit_ke=arr(ke, wp.float64), limit_kd=arr(kd, wp.float64), damping=arr(base, wp.float64),
        tendon_first=arr(first.astype(np.int32), wp.int32), tendon_kind=arr(flat[:, 0].astype(np.int32), wp.int32),
        tendon_row=arr(flat[:, 1].astype(np.int32), wp.int32), tendon_coef=arr(flat[:, 2], wp.float64),
        none=wp.zeros(1, dtype=wp.float64, device=device), mimic=mimic,
    )  # fmt: skip


def build(solver: SolverIPC, scene) -> None:
    """The items of the model's mimic relations, fixed tendons and joint friction (``_Plan``), when the solve is
    built, and the Newton tolerances of a model with friction (``_friction_tolerances``): here, since the scene's
    engine inputs read them (the weight of a rigid body's rows of A in Newton's test) before the ``engine_inputs`` hook
    runs. A ``NewtonWarning`` when the model has friction and a tolerance the caller gave is above
    ``_FRICTION_TOLERANCE`` times ``eps_velocity``."""
    model = solver.model
    solver._joint_terms = _make_plan(solver)
    solver._joint_terms_signature = _signature(model)
    loose = _friction_tolerances(solver, _has_friction(model))
    if loose:
        bound = _FRICTION_TOLERANCE * solver._config.eps_velocity
        warnings.warn(
            f"SolverIPC: the model has joint or tendon friction, and the Newton tolerances given ({', '.join(loose)}) "
            f"are above {_FRICTION_TOLERANCE:g} eps_velocity = {bound:g}: a step within the friction's smoothing band "
            "ends the Newton loop, so a joint loaded beyond its friction creeps instead of sliding. Leave them at their "
            "default, which a model with friction takes as that bound, or pass them at it or below (see Mimic joints, "
            "fixed tendons and joint friction in Newton's IPC solver guide).",
            NewtonWarning,
            stacklevel=5,
        )


def _default_tolerances() -> dict:
    """SolverIPC's default Newton tolerances, without and with a coupled solver's proxies, by name."""
    from ..solver_ipc import _PROXY_TOLERANCES, _TOLERANCES  # noqa: PLC0415 (the solver imports this package)

    names = ("newton_velocity_tol", "newton_transrate_tol")
    return {name: (_TOLERANCES[name], _PROXY_TOLERANCES[name]) for name in names}


def _friction_tolerances(solver: SolverIPC, friction: bool) -> list[str]:
    """With ``friction`` (``_has_friction``), the Newton tolerances that are at SolverIPC's default and above
    ``_FRICTION_TOLERANCE`` times ``eps_velocity`` lowered to it in the solver's config, which the solve about to be
    built takes; without, the defaults this function lowered before put back. A tolerance the caller gave, which is not
    a default, is left as it is (a value equal to a default is taken as the default); returned are those of them above
    the bound, as ``"name value"``."""
    config = solver._config
    lowered = getattr(solver, "_friction_tolerances_of", None)  # the defaults this function lowered
    if not friction:
        if lowered is not None:
            config.newton_velocity_tol, config.newton_transrate_tol = lowered
            solver._friction_tolerances_of = None
        return []
    if lowered is not None:
        return []
    defaults, bound = _default_tolerances(), _FRICTION_TOLERANCE * config.eps_velocity
    velocity, transrate = config.newton_velocity_tol, config.newton_transrate_tol
    loose = []
    for name, value in (("newton_velocity_tol", velocity), ("newton_transrate_tol", transrate)):
        if value <= bound:
            continue
        if value not in defaults[name]:
            loose.append(f"{name} {value:g}")
        elif name == "newton_velocity_tol":
            config.newton_velocity_tol = bound
        else:
            config.newton_transrate_tol = bound
    if (velocity, transrate) != (config.newton_velocity_tol, config.newton_transrate_tol):
        solver._friction_tolerances_of = (velocity, transrate)
    return loose


def engine_inputs(solver: SolverIPC, inputs: dict) -> None:
    """The coordinate terms of the plan, made from ``inputs``' joint groups and appended to its element groups; for an
    engine made again (an eager step's or ``grow()``'s longer lists), with what the steps wrote into the old terms."""
    plan = solver._joint_terms
    if plan is None:
        return
    from uipc_warp.ext.coordinate_terms import CoordinateTerms  # noqa: PLC0415 (libuipc-warp is imported lazily)

    old, plan.terms = plan.terms, CoordinateTerms(inputs["elements"], plan.items, **plan.values)
    inputs["elements"] += plan.terms.groups
    if old is not None:  # the step's inputs (prepare), which the frame that the new engine redoes takes
        new = plan.terms
        wp.copy(new.on, old.on)
        wp.copy(new.friction, old.friction)
        wp.copy(new.limit_strength, old.limit_strength)
        wp.copy(new.damping, old.damping)


def parent_terms(solver: SolverIPC, engine) -> list:
    """The terms' group of ``engine`` whose elements are the items' forces through each joint, with the joint of every
    element, for ``BODY_PARENT_F``."""
    plan = solver._joint_terms
    if plan is None or plan.terms is None:
        return []
    group = plan.terms.groups[0]
    if not any(g is group for g in engine.elements):
        return []
    return [(group, plan.joints)]


@wp.kernel
def _write_terms(
    gate: wp.array[wp.vec3i],
    dof: wp.array[wp.int32],
    joint_enabled: wp.array[wp.bool],
    mimic_enabled: wp.array[wp.bool],
    joint_friction: wp.array[wp.float32],
    limit_ke: wp.array[wp.float64],
    limit_kd: wp.array[wp.float64],
    base_damping: wp.array[wp.float64],
    tendon_first: wp.array[wp.int32],
    tendon_kind: wp.array[wp.int32],
    tendon_row: wp.array[wp.int32],
    tendon_coef: wp.array[wp.float64],
    angles: wp.array[wp.float64],
    distances: wp.array[wp.float64],
    limit_lower: wp.array[wp.float64],
    limit_upper: wp.array[wp.float64],
    limit_damping: int,
    dt: wp.float64,
    on: wp.array[wp.int32],
    friction: wp.array[wp.float64],
    limit_strength: wp.array[wp.float64],
    damping: wp.array[wp.float64],
):
    """The step's inputs of item ``i``: whether it acts (its joints' ``joint_enabled`` and its mimic row's
    ``constraint_mimic_enabled``), a joint's friction (``joint_friction`` of its degree of freedom), and a tendon's
    limit, its force-space stiffness as SolverIPC's cubic of a joint's limit (``_joint_terms``) and, with
    ``limit_damping``, its damping added to the tendon's while its length at the step's start is past a bound."""
    i = wp.tid()
    g = gate[i]
    act = True
    for k in range(2):
        if g[k] >= 0:
            act = act and joint_enabled[g[k]]
    if g[2] >= 0 and mimic_enabled:
        act = act and mimic_enabled[g[2]]
    on[i] = wp.where(act, 1, 0)
    d = dof[i]
    if d >= 0:
        friction[i] = wp.float64(joint_friction[d])
    ke = limit_ke[i]
    if ke > wp.float64(0.0):
        lo = limit_lower[i]
        hi = limit_upper[i]
        width = hi - lo
        w3 = wp.where(width > wp.float64(0.0), width * width * width, wp.float64(1.0))
        limit_strength[i] = dt * dt * ke * w3 / (wp.float64(3.0) * _LIMIT_OVERSHOOT)
        c = base_damping[i]
        if limit_damping != 0:
            L = wp.float64(0.0)
            for t in range(tendon_first[i], tendon_first[i + 1]):
                q = angles[tendon_row[t]]
                if tendon_kind[t] == 1:
                    q = distances[tendon_row[t]]
                L = L + tendon_coef[t] * q
            if L < lo or L > hi:
                c = c + limit_kd[i]
        damping[i] = c


def prepare(solver: SolverIPC, state_in: State, control: Control, dt: float) -> None:
    """The step's inputs of the items (``_write_terms``), on the device."""
    plan = solver._joint_terms
    if plan is None:
        return
    model, terms, groups, none = solver.model, plan.terms, solver._joint_groups, plan.none
    wp.launch(
        _write_terms,
        len(plan.items),
        inputs=[
            plan.gate, plan.dof, model.joint_enabled,
            model.constraint_mimic_enabled if model.constraint_mimic_count else None,
            model.joint_friction, plan.limit_ke, plan.limit_kd, plan.damping, plan.tendon_first, plan.tendon_kind,
            plan.tendon_row, plan.tendon_coef,
            none if groups.revolute is None else groups.revolute.current,
            none if groups.prismatic is None else groups.prismatic.current,
            terms.limit_lower, terms.limit_upper, int(solver._joint_options.limit_damping), wp.float64(dt),
        ],
        outputs=[terms.on, terms.friction, terms.limit_strength, terms.damping],
        device=model.device,
    )  # fmt: skip


def finish(solver: SolverIPC, state_out: State, observables: SolverBase.Observables | None) -> None:
    """Nothing."""


def notify(solver: SolverIPC, flags: ModelFlags | int) -> bool:
    """Whether the solve must be built again: when the degrees of freedom with friction, the mimic relations or the
    fixed tendons have changed which joints they hold, or the model has gained or lost friction. New coefficients of the mimic relations and the tendons
    (``CONSTRAINT_PROPERTIES``, ``TENDON_PROPERTIES``), and the mimic penalties, which follow the bodies' masses
    (``BODY_INERTIAL_PROPERTIES``, ``BODY_PROPERTIES``), are written into the terms in place."""
    plan = solver._joint_terms
    joints = ModelFlags.JOINT_PROPERTIES | ModelFlags.JOINT_DOF_PROPERTIES | ModelFlags.JOINT_DOF_FORCE_PROPERTIES
    values = ModelFlags.CONSTRAINT_PROPERTIES | ModelFlags.TENDON_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES
    if not flags & (joints | values | ModelFlags.BODY_PROPERTIES):
        return False
    if solver._joint_terms_signature != _signature(solver.model):
        return True
    if plan is None or plan.terms is None:
        return False
    now = _make_plan(solver)
    plan.items, plan.values = now.items, now.values
    v, terms = now.values, plan.terms
    terms.offset.assign(v["offset"])
    terms.stiffness.assign(v["stiffness"])
    terms.spring.assign(v["spring"])
    terms.spring_lower.assign(v["spring_lower"])
    terms.spring_upper.assign(v["spring_upper"])
    terms.limit_lower.assign(v["limit_lower"])
    terms.limit_upper.assign(v["limit_upper"])
    terms.damping.assign(v["damping"])
    terms.armature.assign(v["armature"])
    terms.friction.assign(v["friction"])
    terms.coefficient.assign(np.array([c for item in now.items for _, _, c in item], np.float64))
    plan.limit_ke.assign(now.limit_ke.numpy())
    plan.limit_kd.assign(now.limit_kd.numpy())
    plan.damping.assign(now.damping.numpy())
    return False
