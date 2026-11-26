# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo's spatial tendons (:ref:`ipc-spatial-tendons`), and the refusal of Newton's own tendons.

A spatial tendon of MuJoCo (``model.mujoco.tendon_*`` of type 1, which :func:`newton.solvers.mujoco.add_tendon_spatial`
and the MJCF and USD importers write) is a path through sites, around spheres and cylinders between two sites, and over
pulleys that divide the length of the branch after them. SolverIPC gives every such tendon to libuipc-warp's spatial
tendons (``uipc_warp.ext.tendons.SpatialTendons``), whose length and moment arms are MuJoCo's, with the forces it gives
a fixed tendon (``joint_terms``): the stiffness toward the spring length (a band where the pair differs, the length at
the model's pose for -1), the damping, the armature, the friction loss, and the limit as SolverIPC takes a joint's. The
limit's stiffness is ``tendon_limit_ke`` where it is given in force space, else what Newton's MuJoCo solver makes of
``solreflimit`` and ``solimplimit`` with the tendon's inverse weight ``J H^-1 J^T`` at the model's pose (``J`` the
tendon's moment arms in joint space, ``H`` the joint-space inertia), and the limit's damping acts past a bound with
``enable_joint_limit_damping``. A site or a shape of body -1 is a point of the world.

Refused, each with its reason: Newton's own tendons (``ModelBuilder.add_tendon``); a spatial tendon that wraps a shape
other than a sphere or a cylinder, whose path MuJoCo's compiler refuses, or whose side site lies inside the shape it
wraps at the model's pose, where MuJoCo wraps inside the shape. ``joint_terms`` refuses actuators that drive a tendon and
a limit margin other than 0, of every tendon. A spatial tendon whose sites and shapes are all of the world has a
constant length and so no force, as in MuJoCo, and gets no term. The hooks are those of ``features`` (its docstring).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

import newton
from newton import GeoType, ModelFlags, ShapeFlags

from ..scene import _quat_to_matrix_np
from .joint_terms import _DEFAULT_SOLREF, _LIMIT_OVERSHOOT, _SOLREF_FORCE_SPACE

if TYPE_CHECKING:
    from newton import Control, Model, State

    from ..solver_ipc import SolverIPC
    from . import FeatureOptions

# MuJoCo's kinds of path entries (mujoco:tendon_wrap_type), and the shapes it wraps a tendon around
_SITE, _GEOM, _PULLEY = 0, 1, 2
_SHAPES = (int(GeoType.SPHERE), int(GeoType.CYLINDER))
_RULES = (
    "put every wrapping shape between two sites and every site next to a site or a shape, end the path with a site, "
    "keep pulleys apart and give them a positive divisor"
)


def _spatial(model: Model) -> list[tuple[int, list | None]]:
    """MuJoCo's spatial tendons of ``model`` as ``(tendon, entries)``, every entry of its path ``(kind, shape, side
    site, divisor)``; None for a path whose entries lie outside the model's."""
    mujoco = getattr(model, "mujoco", None)
    if (
        mujoco is None
        or model.custom_frequency_counts.get("mujoco:tendon", 0) == 0
        or not hasattr(mujoco, "tendon_type")
    ):
        return []
    spatial = np.flatnonzero(mujoco.tendon_type.numpy() == 1)
    if not len(spatial):
        return []
    n_wraps = model.custom_frequency_counts.get("mujoco:tendon_wrap", 0)
    adr, num = mujoco.tendon_wrap_adr.numpy(), mujoco.tendon_wrap_num.numpy()
    kind, shape = mujoco.tendon_wrap_type.numpy(), mujoco.tendon_wrap_shape.numpy()
    side, prm = mujoco.tendon_wrap_sidesite.numpy(), mujoco.tendon_wrap_prm.numpy()
    out = []
    for t in spatial:
        first, count = int(adr[t]), int(num[t])
        if first < 0 or count < 0 or first + count > n_wraps:
            out.append((int(t), None))
            continue
        rows = range(first, first + count)
        out.append((int(t), [(int(kind[w]), int(shape[w]), int(side[w]), float(prm[w])) for w in rows]))
    return out


def _path_fault(entries: list) -> tuple[int, str] | None:
    """The first entry of a path that MuJoCo's compiler refuses (``mjCTendon::Compile``), as ``(entry, what)``, or
    None."""
    kinds = [e[0] for e in entries]
    n = len(kinds)
    for w, kind in enumerate(kinds):
        if kind not in (_SITE, _GEOM, _PULLEY):
            return w, f"an entry of the unknown kind {kind}"
        if kind == _GEOM and (w == 0 or w == n - 1 or kinds[w - 1] != _SITE or kinds[w + 1] != _SITE):
            return w, "a geom that is not between two sites"
        if kind == _PULLEY:
            if w == n - 1:
                return w, "a pulley at the end"
            if w > 0 and kinds[w - 1] == _PULLEY:
                return w, "two pulleys in a row"
            if not entries[w][3] > 0.0:
                return w, f"a pulley of divisor {entries[w][3]:g}"
        if kind == _SITE and all(kinds[v] == _PULLEY for v in (w - 1, w + 1) if 0 <= v < n):
            return w, "a site whose neighbours are pulleys only"
    return None


def _poses(model: Model) -> np.ndarray:
    """The bodies' poses (p, q) at the model's pose: ``body_q``, an articulation's links by forward kinematics of
    ``joint_q``."""
    state = model.state()
    if model.articulation_count:
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    return state.body_q.numpy().astype(np.float64)


def _world_point(poses: np.ndarray, body: int, point: np.ndarray) -> np.ndarray:
    """A point of a body's frame in the world (a point of the world for body -1)."""
    if body < 0:
        return point
    return poses[body, :3] + _quat_to_matrix_np(poses[body, 3:7]) @ point


def check(model: Model, options: FeatureOptions, at: str) -> None:
    """``NotImplementedError`` (``"tendons"``) for Newton's own tendons and for MuJoCo's spatial tendons SolverIPC cannot
    hold: a path entry outside the model's, a path MuJoCo's compiler refuses, a shape that is no site where a site is
    named or a site where a shape is, a wrap around a shape other than a sphere or a cylinder, and a side site inside
    the shape it chooses the side of at the model's pose (MuJoCo's inside wrap)."""
    if at != "tendons":
        return
    if model.tendon_count:
        raise NotImplementedError(
            f"SolverIPC: the model has {model.tendon_count} tendons of Newton's own (ModelBuilder.add_tendon); remove "
            "them. SolverIPC simulates MuJoCo's fixed and spatial tendons, and not Newton's tendons routed over "
            "anchors, pinholes and rollers."
        )
    spatial = _spatial(model)
    if not spatial:
        return
    shape_type, shape_body = model.shape_type.numpy(), model.shape_body.numpy()
    site = (model.shape_flags.numpy() & int(ShapeFlags.SITE)) != 0
    xform = model.shape_transform.numpy().astype(np.float64)
    radius = model.shape_scale.numpy().astype(np.float64)[:, 0]
    poses = None
    for t, entries in spatial:
        if entries is None:
            n = model.custom_frequency_counts.get("mujoco:tendon_wrap", 0)
            raise NotImplementedError(
                f"SolverIPC: MuJoCo's spatial tendon {t} names path entries outside the model's {n}; give it its own "
                "entries (mujoco:tendon_wrap_adr and tendon_wrap_num)."
            )
        if len(entries) < 2:
            raise NotImplementedError(
                f"SolverIPC: MuJoCo's spatial tendon {t} has a path of {len(entries)} entries; give it two at least, as "
                "MuJoCo requires."
            )
        fault = _path_fault(entries)
        if fault is not None:
            w, what = fault
            raise NotImplementedError(
                f"SolverIPC: MuJoCo's spatial tendon {t} has {what} at path entry {w}, which MuJoCo's compiler "
                f"refuses; {_RULES}."
            )
        for w, (kind, shape, side, _) in enumerate(entries):
            if kind == _PULLEY:
                continue
            wants_site = kind == _SITE
            if not (0 <= shape < model.shape_count and site[shape] == wants_site):
                raise NotImplementedError(
                    f"SolverIPC: MuJoCo's spatial tendon {t} names shape {shape} at path entry {w}, which is no "
                    f"{'site' if wants_site else 'shape other than a site'} of the model; name one."
                )
            if kind == _GEOM and not (side == -1 or (0 <= side < model.shape_count and site[side])):
                raise NotImplementedError(
                    f"SolverIPC: MuJoCo's spatial tendon {t} has the side site {side} at path entry {w}, which is no "
                    "site of the model; name one, or -1 for none."
                )
            if kind == _GEOM and int(shape_type[shape]) not in _SHAPES:
                raise NotImplementedError(
                    f"SolverIPC: MuJoCo's spatial tendon {t} wraps shape {shape}, a {GeoType(int(shape_type[shape])).name}; "
                    "wrap it around a sphere or a cylinder. MuJoCo wraps tendons around spheres and cylinders only."
                )
            if kind == _GEOM and side >= 0:
                if poses is None:
                    poses = _poses(model)
                centre = _world_point(poses, int(shape_body[shape]), xform[shape, :3])
                where = _world_point(poses, int(shape_body[side]), xform[side, :3])
                if np.linalg.norm(where - centre) < radius[shape]:
                    raise NotImplementedError(
                        f"SolverIPC: MuJoCo's spatial tendon {t} has its side site {side} inside the shape {shape} it "
                        "wraps (an inside wrap); move the side site outside the shape. SolverIPC wraps tendons around "
                        "the outside of a shape only."
                    )


# ================================================================================================ the solve's tendons
@dataclass
class _Plan:
    """The spatial tendons of a solve (``solver._features.tendons.plan``, None for a model without any).

    ``rows``: the tendons' rows of ``model.mujoco.tendon_*``; ``paths``: their paths as ``SpatialTendons`` takes them;
    ``values``: the coefficients they are made with; ``measured``: per tendon, whether its spring length is its length at
    the model's pose (-1); ``solref``: per tendon, ``(T, zeta, d_max)`` of a limit given in MuJoCo's solref mode, else
    None; ``ke``, ``kd``: per tendon the stiffness [N/m] and damping [N s/m] of a limit given in force space (0 for
    none); ``limit_ke``: every limit's stiffness on the device, which the steps read (``prepare``) and a notification
    writes in place; ``tendons``: the ``SpatialTendons`` of the solve's engine; ``x0``: the scene's positions when the
    solve was built, the model's pose."""

    rows: np.ndarray
    paths: list
    values: dict
    measured: np.ndarray
    solref: list
    ke: np.ndarray
    kd: np.ndarray
    limit_ke: wp.array
    tendons: object = None
    x0: np.ndarray | None = None


def _make_plan(solver: SolverIPC) -> _Plan | None:
    """The paths and coefficients of the model's spatial tendons (``_Plan``), without what the model's pose gives (the
    spring lengths of -1, the limits' stiffness from solref; ``_measure``), or None for none."""
    model = solver.model
    spatial = [(t, entries) for t, entries in _spatial(model) if entries]
    if not spatial:
        return None
    shape_type, shape_body = model.shape_type.numpy(), model.shape_body.numpy()
    xform = model.shape_transform.numpy().astype(np.float64)
    radius = model.shape_scale.numpy().astype(np.float64)[:, 0]
    paths, rows = [], []
    for t, entries in spatial:
        path = []
        for kind, shape, side, divisor in entries:
            if kind == _PULLEY:
                path.append(("pulley", divisor))
                continue
            body, point = int(shape_body[shape]), tuple(xform[shape, :3])
            if kind == _SITE:
                path.append(("site", body, point))
                continue
            sd = None if side < 0 else (int(shape_body[side]), tuple(xform[side, :3]))
            if int(shape_type[shape]) == int(GeoType.SPHERE):
                path.append(("sphere", body, point, float(radius[shape]), sd))
            else:  # a cylinder's axis is its frame's z
                axis = tuple(_quat_to_matrix_np(xform[shape, 3:7])[:, 2])
                path.append(("cylinder", body, point, axis, float(radius[shape]), sd))
        if all(e[1] < 0 for e in path if e[0] != "pulley"):
            continue  # every point of the world: a constant length, no force (as in MuJoCo)
        paths.append(path)
        rows.append(t)
    if not paths:
        return None
    rows = np.asarray(rows, np.int64)
    mujoco = model.mujoco
    length = mujoco.tendon_springlength.numpy().astype(np.float64)[rows]
    lo, hi = length[:, 0].copy(), length[:, 1].copy()
    # SolverMuJoCo's reading of the spring length: -1 first, the length at the model's pose (_measure); a pair in
    # order, a band; else one length
    measured = lo == -1.0
    hi = np.where(measured, -1.0, np.where(hi < lo, lo, hi))
    limited, bounds = mujoco.tendon_limited.numpy()[rows], mujoco.tendon_range.numpy().astype(np.float64)[rows]
    on_limit = (limited == 1) | ((limited == 2) & (bounds[:, 0] < bounds[:, 1]))
    mode = mujoco.tendon_solref_limit_mode.numpy()[rows]
    ke, kd = (
        mujoco.tendon_limit_ke.numpy().astype(np.float64)[rows],
        mujoco.tendon_limit_kd.numpy().astype(np.float64)[rows],
    )
    solref = mujoco.tendon_solref_limit.numpy().astype(np.float64)[rows]
    solimp = mujoco.tendon_solimp_limit.numpy().astype(np.float64)[rows]
    refs, limit_ke, limit_kd = [], np.zeros(len(rows)), np.zeros(len(rows))
    for k in range(len(rows)):
        ref = None
        if on_limit[k] and int(mode[k]) == _SOLREF_FORCE_SPACE:
            limit_ke[k], limit_kd[k] = ke[k], kd[k]
        elif on_limit[k]:
            T, zeta = solref[k] if (solref[k] > 0.0).all() else _DEFAULT_SOLREF
            ref = (float(T), float(zeta), float(solimp[k][1]))
        refs.append(ref)
    values = {
        "spring": mujoco.tendon_stiffness.numpy().astype(np.float64)[rows],
        "spring_lower": np.minimum(lo, hi),
        "spring_upper": np.maximum(lo, hi),
        "damping": mujoco.tendon_damping.numpy().astype(np.float64)[rows],
        "armature": mujoco.tendon_armature.numpy().astype(np.float64)[rows],
        "friction": mujoco.tendon_frictionloss.numpy().astype(np.float64)[rows],
        "limit_lower": np.where(on_limit, bounds[:, 0], 0.0),
        "limit_upper": np.where(on_limit, bounds[:, 1], 0.0),
        "limit_damping": np.zeros(len(rows)),
    }
    return _Plan(rows=rows, paths=paths, values=values, measured=measured, solref=refs, ke=limit_ke, kd=limit_kd,
                 limit_ke=wp.zeros(len(rows), dtype=wp.float64, device=model.device))  # fmt: skip


def _vee(S: np.ndarray) -> np.ndarray:
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def _inverse_weights(solver: SolverIPC, bodies: list, gradients: list, x: np.ndarray) -> np.ndarray:
    """Every tendon's inverse weight ``J H^-1 J^T`` at the model's pose, MuJoCo's ``tendon_invweight0``: ``J`` its
    moment arms in the joint space of the articulations its bodies are links of, from its length's gradient on each
    body (``SpatialTendons.measure``: on ``p`` the force, on the rows of ``A`` the torque ``vee(G A^T - A G^T)`` about
    the body's origin) and the links' Jacobians (``newton.eval_jacobian``, of the centre of mass and the angular
    velocity), ``H`` the articulations' inertia (``newton.eval_mass_matrix``, armature included), summed over the
    articulations; 0 for a tendon over no articulation."""
    model = solver.model
    out = np.zeros(len(bodies))
    if model.articulation_count == 0:
        return out
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    J = newton.eval_jacobian(model, state).numpy().astype(np.float64)
    H = newton.eval_mass_matrix(model, state).numpy().astype(np.float64)
    child, articulation = model.joint_child.numpy(), model.joint_articulation.numpy()
    start, qd_start = model.articulation_start.numpy(), model.joint_qd_start.numpy()
    link = {int(child[j]): (int(articulation[j]), j - int(start[articulation[j]]))
            for j in range(model.joint_count) if articulation[j] >= 0}  # fmt: skip
    com = model.body_com.numpy().astype(np.float64)
    base = solver._scene.layout().body_base
    for k, (tendon_bodies, gradient) in enumerate(zip(bodies, gradients, strict=True)):
        rows = {}
        for b, g in zip(tendon_bodies, gradient, strict=True):
            if int(b) not in link:
                continue
            a, l = link[int(b)]
            n = int(qd_start[start[a + 1]]) - int(qd_start[start[a]])
            A = x[base[b] + 1 : base[b] + 4]
            torque = _vee(g[1:] @ A.T - A @ g[1:].T)
            r = -A @ com[b]  # the body's origin from its centre of mass
            row = g[0] @ J[a, 6 * l : 6 * l + 3, :n] + (torque + np.cross(r, g[0])) @ J[a, 6 * l + 3 : 6 * l + 6, :n]
            rows[a] = rows.get(a, 0.0) + row
        out[k] = sum(float(row @ np.linalg.solve(H[a, : len(row), : len(row)], row)) for a, row in rows.items())
    return out


def _measure(solver: SolverIPC, plan: _Plan) -> None:
    """What the model's pose gives the plan (``plan.x0``, with ``plan.tendons``): the spring lengths of -1 (MuJoCo's
    length at ``qpos0``) and the stiffness and damping of the limits given in solref mode, ``1 / (T^2 zeta^2 w)`` and
    ``2 / (T w)`` with ``w = invweight (1 - d_max)`` as Newton's MuJoCo solver reports them; the limits' damping as the
    tendons take it, with ``enable_joint_limit_damping``."""
    length, gradients = plan.tendons.measure(plan.x0)
    values = plan.values
    values["spring_lower"] = np.where(plan.measured, length, values["spring_lower"])
    values["spring_upper"] = np.where(plan.measured, length, values["spring_upper"])
    ke, kd = plan.ke.copy(), plan.kd.copy()
    if any(ref is not None for ref in plan.solref):
        weights = _inverse_weights(solver, plan.tendons.bodies, gradients, plan.x0)
        for k, ref in enumerate(plan.solref):
            if ref is None:
                continue
            T, zeta, d_max = ref
            w = weights[k] * (1.0 - d_max)
            ke[k], kd[k] = (1.0 / (T * T * zeta * zeta * w), 2.0 / (T * w)) if w > 0.0 else (0.0, 0.0)
    plan.limit_ke.assign(ke)
    values["limit_damping"] = kd if solver._joint_options.limit_damping else np.zeros(len(kd))


def _assign(plan: _Plan) -> None:
    """The plan's coefficients into its tendons' arrays, in place."""
    t, v = plan.tendons, plan.values
    t.spring.assign(v["spring"])
    t.spring_lower.assign(v["spring_lower"])
    t.spring_upper.assign(v["spring_upper"])
    t.damping.assign(v["damping"])
    t.armature.assign(v["armature"])
    t.friction.assign(v["friction"])
    t.limit_lower.assign(v["limit_lower"])
    t.limit_upper.assign(v["limit_upper"])
    t.limit_damping.assign(v["limit_damping"])


def _signature(model: Model) -> tuple:
    """What the tendons' paths are made of: their entries, and the bodies, transforms, sizes and types of the shapes they
    name. A change of it builds the solve again (``notify``)."""
    spatial = _spatial(model)
    if not spatial:
        return ()
    shapes = sorted({s for _, entries in spatial if entries for kind, shape, side, _ in entries if kind != _PULLEY
                     for s in (shape, side) if 0 <= s < model.shape_count})  # fmt: skip
    arrays = (model.shape_body, model.shape_transform, model.shape_scale, model.shape_type)
    return (
        tuple((t, None if e is None else tuple(e)) for t, e in spatial),
        *(tuple(a.numpy()[shapes].reshape(-1).tolist()) for a in arrays),
    )


@dataclass
class _State:
    """What the module keeps of a solve (``solver._features.tendons``): the plan of its spatial tendons (None for a
    model without any) and what their paths were made of (``_signature``)."""

    plan: _Plan | None
    signature: tuple


def _plan(solver: SolverIPC) -> _Plan | None:
    """The plan of the solve's spatial tendons, None for a solve without any or not built."""
    state = solver._features.tendons
    return None if state is None else state.plan


def build(solver: SolverIPC, scene) -> None:
    """The plan of the model's spatial tendons (``_Plan``) when the solve is built."""
    solver._features.tendons = _State(_make_plan(solver), _signature(solver.model))


def engine_inputs(solver: SolverIPC, inputs: dict) -> None:
    """The plan's ``SpatialTendons`` for an engine made from the scene, appended to ``inputs``' element groups; the
    first time, what the model's pose (the scene's positions) gives them (``_measure``); for an engine made again (an
    eager step's or ``grow()``'s longer lists), the step's limit strength the old one held."""
    plan = _plan(solver)
    if plan is None:
        return
    from uipc_warp.ext.tendons import SpatialTendons  # noqa: PLC0415 (libuipc-warp is imported lazily)

    old, plan.tendons = plan.tendons, SpatialTendons(solver._scene.layout(), plan.paths, solver.model.device,
                                                     **plan.values)  # fmt: skip
    if plan.x0 is None:
        plan.x0 = np.asarray(inputs["x"], np.float64).reshape(-1, 3).copy()
        _measure(solver, plan)
        _assign(plan)
    inputs["elements"] += plan.tendons.groups
    if old is not None:  # the step's input (prepare), which the frame that the new engine redoes takes
        wp.copy(plan.tendons.limit_strength, old.limit_strength)


@wp.kernel
def _write_limits(
    limit_ke: wp.array[wp.float64],
    lower: wp.array[wp.float64],
    upper: wp.array[wp.float64],
    dt: wp.float64,
    strength: wp.array[wp.float64],
):
    """The step's limit strength of tendon ``i``: its force-space stiffness as SolverIPC's cubic of a joint's limit,
    ``dt^2 ke w^3 / (3 * 1 mm)``, ``w`` the range's width (1 for a range of no width)."""
    i = wp.tid()
    ke = limit_ke[i]
    s = wp.float64(0.0)
    if ke > wp.float64(0.0):
        width = upper[i] - lower[i]
        w3 = wp.where(width > wp.float64(0.0), width * width * width, wp.float64(1.0))
        s = dt * dt * ke * w3 / (wp.float64(3.0) * _LIMIT_OVERSHOOT)
    strength[i] = s


def prepare(solver: SolverIPC, state_in: State, control: Control, dt: float) -> None:
    """The step's limit strengths (``_write_limits``), on the device."""
    plan = _plan(solver)
    if plan is None:
        return
    t = plan.tendons
    wp.launch(_write_limits, len(plan.rows), inputs=[plan.limit_ke, t.limit_lower, t.limit_upper, wp.float64(dt)],
              outputs=[t.limit_strength], device=solver.model.device)  # fmt: skip


def notify(solver: SolverIPC, flags: ModelFlags | int) -> bool:
    """Whether the solve must be built again: when the tendons' paths or the shapes they name have changed. New
    coefficients (``TENDON_PROPERTIES``) and new masses, which the limits' stiffness follows through the inverse weights
    (``BODY_INERTIAL_PROPERTIES``), are written into the tendons in place."""
    watched = ModelFlags.TENDON_PROPERTIES | ModelFlags.SHAPE_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES
    if not flags & watched:
        return False
    state = solver._features.tendons
    if state.signature != _signature(solver.model):
        return True
    plan = state.plan
    if plan is None or plan.tendons is None:
        return False
    now = _make_plan(solver)
    now.tendons, now.x0, now.limit_ke = plan.tendons, plan.x0, plan.limit_ke  # the arrays captured steps read
    _measure(solver, now)
    _assign(now)
    state.plan = now
    return False
