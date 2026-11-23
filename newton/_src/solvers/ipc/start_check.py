# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC's check of the start: surfaces that cross or start at or inside their contact thickness.

As libuipc's sanity check does at init, ``_check_start`` tests the state the first step takes, part pair by part pair
with libuipc-warp's ``SurfaceChecker``, and names the closest pair.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from newton import NewtonWarning

if TYPE_CHECKING:
    from .solver_ipc import SolverIPC


# the most edge-triangle pairs the start check tests: libuipc-warp's SurfaceChecker tests every pair, about 1e10
# pairs per second on a recent GPU, in one launch
_START_CHECK_PAIRS = 2**31 - 1


def _check_start(solver: SolverIPC, state: str | None = None) -> None:
    """Raise ``ValueError`` when the surfaces of different bodies cross each other at the start, or start at or
    inside their contact thickness.

    The barrier needs surfaces that start apart, by more than their contact thickness. A step fails on a contact
    pair at or inside its thickness (``THICKNESS_VIOLATED`` or ``INVALID_CCD_STEP``), which a captured step only
    reports, and surfaces that cross without such a pair (an edge through a triangle, away from its vertices) step
    on with the barrier keeping them crossed. As libuipc's sanity check does at init, this tests the start
    state, without the surfaces that never move (static shapes, kinematic particles and bodies), whose contact the
    solve skips, and without the pairs whose contact elements do not collide or whose subscenes do not interact (the
    two tables tested apart, as the engine tests them). The gap of a pair is its distance less its contact
    thickness, the sum of its two primitives' thicknesses, each that of its vertices (a free particle's radius, a
    shape's margin, the ``thickness`` of cloth and tetrahedra). Every contact point that moves is tested against
    every half plane. Triangle meshes and particles are tested part pair by part pair, a part being a contact body's
    primitives in one contact element and one subscene, with the particles on no cloth or tetrahedron taken together
    per element and subscene and also tested among themselves: for the pairs of parts whose subscenes interact (with
    several worlds no pair of two worlds is formed) and whose bounding boxes come within their thickness of each
    other, libuipc-warp's brute-force ``SurfaceChecker`` counts the edge-triangle crossings and finds the smallest
    gap over the point-triangle, point-edge, edge-edge and point-point pairs. A gap above 0, also inside ``d_hat``,
    is a valid start, unless it is within a few float32 steps of 0: Newton's positions are float32, and surfaces
    placed in touching contact come out that far apart. Not tested: the inside of a body (contact ignores crossings
    inside a rigid body made of overlapping shapes, and inside a cloth or a soft body the checker cannot tell the
    nearly coplanar triangles of a flat sheet from crossing ones). A pair of parts of more than
    ``_START_CHECK_PAIRS`` primitive pairs is skipped with a warning.

    The start is the state the first step takes (class docstring, **State**). At construction (``state`` None)
    the engine holds the model's state, and a pair that fails there raises, unless one of its bodies is one whose
    start the caller writes into the state (``_posed_body``: a kinematic body, and a link that the model leaves at
    the identity where forward kinematics puts it elsewhere). The failure of such a pair is kept
    (``_start_pending``), and the first step tests the pairs with such a body again: a first step that is not
    captured on the engine's positions it has just loaded from its ``state_in``, a captured one, which cannot read
    its ``state_in``, by raising the kept failure (``_start_in_capture``). With ``state``, the name of the state the
    engine holds in the message, only the pairs with such a body are tested, on the engine's positions as they
    stand, and a failure raises: the test holds for the state of any step loaded into the engine (it synchronizes).
    """
    eng = solver._engine
    if eng is None:
        return
    posed_only = state is not None
    layout = solver._scene.layout()
    still = np.zeros(layout.n, bool)  # vertices that the solve never moves
    still[: layout.n_fem] = solver._fixed_particle
    still[layout.obstacle_offsets[0] : layout.obstacle_offsets[-1]] = True
    offsets = layout.body_vertex_offsets
    held = solver._kinematic_body.copy()
    held[solver._aim_bodies] = False  # kinematic_motion="aim": these move in the step
    for b in np.flatnonzero(held):
        still[offsets[b] : offsets[b + 1]] = True
    posed_vertex = np.zeros(layout.n, bool)  # the surface vertices of the bodies whose start the caller writes
    for b in np.flatnonzero(solver._posed_body):
        posed_vertex[offsets[b] : offsets[b + 1]] = True
    x = eng.vm.x.numpy()
    thickness = solver._vertex_thickness
    points = np.asarray(eng.contact_points, np.int64)
    if len(points) == 0:
        return
    # the engine's contact bodies, with the particles on no surface (each a body of its own) taken as one. The
    # surface vertices are those of the contact triangles and edges: the engine's own list holds the particles too
    # where the scene gives it the contact surface (with springs or stitches)
    body = eng.body.numpy().astype(np.int64)
    tris = np.asarray(eng.surface["triangles"], np.int64).reshape(-1, 3)
    edges = np.asarray(eng.surface["edges"], np.int64).reshape(-1, 2)
    surface_vertices = np.unique(np.concatenate([tris.reshape(-1), edges.reshape(-1)]))
    lone = np.setdiff1d(points, surface_vertices)
    group = body.copy()
    lone_group = int(group.max()) + 1
    group[lone] = lone_group
    # parts: the groups split by contact element and subscene, compared only where both tables let them collide
    cid, sid, enabled, subscenes = solver._cids, solver._scids, solver._enabled, solver._subscene_enabled
    n_el, n_sub = len(enabled), len(subscenes)
    key = (group * n_el + cid) * n_sub + sid
    ids, point_group = np.unique(key[points], return_inverse=True)
    ids_group, ids_element, ids_subscene = ids // (n_el * n_sub), ids // n_sub % n_el, ids % n_sub
    posed = np.zeros(len(ids), bool)  # the parts of the bodies whose start the caller writes
    np.logical_or.at(posed, point_group, posed_vertex[points])
    if posed_only and not posed.any():
        return
    # Newton's positions are float32: surfaces placed in touching contact come out up to a few float32 steps
    # apart, which the barrier would take as a valid, nearly singular start. Such a gap counts as contact.
    slack = 4.0 * float(np.spacing(np.float32(np.abs(x[points]).max())))
    # (crossings, gap, distance, pair thickness, part, the half plane (as -1 - plane) or part, the pair's vertices,
    # whether a body of the pair is one whose start the caller writes)
    violations = []

    # contact points against half planes, each moved out by its shape's margin: the distance from the plane's own
    # surface and the margin's share of the pair's thickness (with state, the points of posed bodies alone)
    P, N, plane_shapes = solver._planes
    moves = ~still[points]
    if posed_only:
        moves &= posed[point_group]
    if moves.any() and len(P):
        moving = points[moves]
        margin = solver.model.shape_margin.numpy().astype(np.float64)[plane_shapes]
        d = x[moving] @ N.T - np.einsum("ij,ij->i", P, N) + margin
        t = thickness[moving][:, None] + margin
        gap = d - t
        meets = enabled[cid[moving][:, None], solver._plane_cids[None, :]]
        meets &= subscenes[sid[moving][:, None], solver._plane_scids[None, :]]
        owner = point_group[moves]
        for j in range(len(P)):
            hit = np.flatnonzero(meets[:, j] & (gap[:, j] <= slack))
            for g in np.unique(owner[hit]):
                k = hit[owner[hit] == g]
                k = k[np.argmin(gap[k, j])]
                violations.append((0, gap[k, j], d[k, j], t[k, j], g, -1 - j, (int(moving[k]),), posed[g]))

    # triangle meshes and particles, one pair of parts at a time
    lo = np.full((len(ids), 3), np.inf)
    hi = np.full((len(ids), 3), -np.inf)
    np.minimum.at(lo, point_group, x[points])
    np.maximum.at(hi, point_group, x[points])
    reach = np.zeros(len(ids))
    np.maximum.at(reach, point_group, thickness[points])
    all_still = np.ones(len(ids), bool)
    np.logical_and.at(all_still, point_group, still[points])
    # the primitives of each part, sorted by part: rows, whether they move, and where each part's rows start (a
    # primitive's vertices are in one contact body and one element)
    prims = {}
    kinds = (
        ("vertices", surface_vertices[:, None]),
        ("codim", lone[:, None]),
        ("triangles", tris),
        ("edges", edges),
    )
    for name, rows in kinds:
        part_of = np.searchsorted(ids, key[rows[:, 0]])
        order = np.argsort(part_of, kind="stable")
        prims[name] = (
            rows[order],
            ~still[rows[order]].all(axis=1),
            np.searchsorted(part_of[order], np.arange(len(ids) + 1)),
        )
    pairs = _near_parts(solver, lo, hi, reach, slack, ids_subscene)
    # the particles on no surface meet each other where their element and their subscene meet themselves
    lone_parts = np.flatnonzero(
        (ids_group == lone_group) & enabled[ids_element, ids_element] & subscenes[ids_subscene, ids_subscene]
    )
    pairs = np.concatenate([pairs, np.stack([lone_parts, lone_parts], axis=1)])
    # not tested: two parts that never move, the inside of a body, two parts whose elements do not collide or
    # whose subscenes do not interact (worlds that overlap in space); with state, pairs without a posed body
    a, b = pairs[:, 0], pairs[:, 1]
    keep = (
        ~(all_still[a] & all_still[b])
        & ~((a != b) & (ids_group[a] == ids_group[b]))
        & enabled[ids_element[a], ids_element[b]]
        & subscenes[ids_subscene[a], ids_subscene[b]]
    )
    if posed_only:
        keep &= posed[a] | posed[b]
    pairs = pairs[keep]
    device = solver.model.device
    arrays = (wp.array(body.astype(np.int32), device=device), wp.array(thickness, dtype=wp.float64, device=device))
    for a, b in pairs:
        result = _part_pair(solver, prims, arrays, a, b)
        if result is None:
            warnings.warn(
                f"SolverIPC does not check whether {_describe_group(solver, group, points, ids_group[a])} and "
                f"{_describe_group(solver, group, points, ids_group[b])} start in contact: their surfaces make more "
                f"primitive pairs than the check tests ({_START_CHECK_PAIRS}).",
                NewtonWarning,
                stacklevel=3,
            )
            continue
        found, gap = result
        if found or gap.gap <= slack:
            violations.append((found, gap.gap, gap.distance, gap.thickness, a, b, gap.vertices, posed[a] | posed[b]))
    if not posed_only:
        solver._start_pending = None
    if not violations:
        return
    lone_set = set(lone.tolist())

    def describe(part: int, vertices, other_than: int | None = None) -> tuple[str, int | None]:
        """A part's name, and its particle in the closest pair when it is the particles on no surface."""
        if ids_group[part] == lone_group:
            v = next((v for v in vertices if v in lone_set and v != other_than), None)
            if v is not None:
                return f"particle {v}", v
        return _describe_group(solver, group, points, ids_group[part]), None

    def failure(violations: list, state: str | None = None) -> str:
        """Why the start fails: the pair that crosses most, or else the closest pair, and how many more fail;
        ``state``: where (None: the model's state, said for crossings only, as the construction always said)."""
        found, _, distance, t, a, b, vertices, _ = min(violations, key=lambda v: (-v[0], v[1]))
        n = len(violations) - 1
        more = f" ({n} more pair{'s' if n > 1 else ''} of bodies {'do' if n > 1 else 'does'} too)" if n else ""
        one, particle = describe(a, vertices)
        other = (
            f"plane shape {plane_shapes[-1 - b]}" if b < 0 else describe(b, vertices, particle if a == b else None)[0]
        )
        if found:
            where = state or "the model's state"
            return (
                f"SolverIPC: the surfaces of {one} and {other} intersect in {where}: {found} edges of one cross "
                f"triangles of the other{more}. The barrier needs surfaces that start apart, by more than their "
                "contact thickness; a step would keep them crossed."
            )
        need = f" by more than {t - distance:.6g} m" if t - distance > 0.0 else ""
        fix = f"lift {one} off it{need}" if b < 0 else f"move them apart{need}, for example by lifting the upper one"
        rounding = ", up to the rounding of float32 positions" if distance > t else ""
        at = f" in {state}" if state else ""
        return (
            f"SolverIPC: {one} and {other} start {distance:.6g} m apart, at or inside their contact thickness "
            f"({t:.6g} m{rounding}){more}{at}. The barrier needs a gap larger than the thickness at the start: "
            f"{fix}."
        )

    if posed_only:
        raise ValueError(failure(violations, state))
    now = [v for v in violations if not v[-1]]
    if now:
        raise ValueError(failure(now))
    # only pairs with a body whose start the caller writes fail, in the model's state: the first step tests them
    # on its state_in, or raises this when it is captured
    solver._start_pending = failure(violations, "the model's state")


def _near_parts(solver: SolverIPC, lo: np.ndarray, hi: np.ndarray, reach: np.ndarray, slack: float, sub: np.ndarray):
    """The pairs ``(a, b)``, ``a < b``, of the parts ``_near_boxes`` finds, but where the subscene table has the
    form of separate worlds (several worlds): there no pair of two worlds is formed, which with worlds at one
    place would be every pair of their parts. The parts of the subscenes that meet every subscene (the global
    entities) are tested against every part, those of the others in one band per subscene (``_near_boxes``'s
    ``band``). Other tables keep every near pair, which the caller tests against the table."""
    table = solver._subscene_enabled
    meets_all = table.all(axis=1)
    diagonal = np.diagonal(table)
    separate = np.array_equal(
        table, meets_all[:, None] | meets_all[None, :] | (np.eye(len(table), dtype=bool) & diagonal[:, None])
    )
    if len(table) == 1 or not separate:
        return _near_boxes(lo, hi, reach, slack)
    wide = np.flatnonzero(meets_all[sub])
    local = np.flatnonzero(~meets_all[sub])
    out = [local[_near_boxes(lo[local], hi[local], reach[local], slack, band=sub[local])]]
    chunk = max(1, 1_000_000 // max(len(lo), 1))  # the wide parts' tests against every part, a block at a time
    for k in range(0, len(wide), chunk):
        a = wide[k : k + chunk]
        apart = np.maximum(lo[None, :, :] - hi[a][:, None, :], lo[a][:, None, :] - hi[None, :, :]).max(axis=2)
        i, b = np.nonzero(apart <= reach[a][:, None] + reach[None, :] + slack)
        b_wide = meets_all[sub[b]]
        keep = (a[i] != b) & (~b_wide | (a[i] < b))  # a pair of two wide parts once
        out.append(np.stack([np.minimum(a[i], b), np.maximum(a[i], b)], axis=1)[keep])
    return np.concatenate(out).reshape(-1, 2)


def _near_boxes(
    lo: np.ndarray, hi: np.ndarray, reach: np.ndarray, slack: float, band: np.ndarray | None = None
) -> np.ndarray:
    """The pairs ``(a, b)``, ``a < b``, of boxes ``[lo, hi]`` that come within ``reach[a] + reach[b] + slack`` of
    each other, by sweep and prune along the axis on which the boxes spread most. With ``band`` (an integer per
    box) only boxes of one band pair: the sweep runs over the bands laid end to end along that axis, each farther
    from the next than any box reaches."""
    if len(lo) == 0:
        return np.zeros((0, 2), np.int64)
    axis = int(np.argmax(np.ptp(lo + hi, axis=0)))
    key_lo, key_hi = lo[:, axis], hi[:, axis]
    pad = 0.0
    if band is not None:  # the gap between two bands exceeds the reach of any box by 1 (m)
        width = float(key_hi.max() - key_lo.min()) + 2.0 * float(reach.max()) + slack + 1.0
        shift = (band - band.min()).astype(np.float64) * width
        key_lo, key_hi = key_lo + shift, key_hi + shift
        pad = 4.0 * float(np.spacing(np.abs(key_hi).max()))  # the rounding of the shifted ends
    order = np.argsort(key_lo, kind="stable")
    start = key_lo[order]
    end = np.searchsorted(start, key_hi[order] + reach[order] + reach.max() + slack + pad, side="right")
    count = np.maximum(end - np.arange(len(order)) - 1, 0)
    first = np.repeat(np.arange(len(order)), count)
    second = first + 1 + np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    a, b = order[first], order[second]
    apart = np.maximum(lo[b] - hi[a], lo[a] - hi[b]).max(axis=1, initial=-np.inf)
    near = apart <= reach[a] + reach[b] + slack
    a, b = a[near], b[near]
    return np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1)


def _part_pair(solver: SolverIPC, prims: dict, arrays: tuple, a: int, b: int):
    """The edge-triangle crossings between parts ``a`` and ``b`` (indices into ``prims``, which may be one part)
    and their primitive pair of the smallest gap (``uipc_warp.validation.Gap``), without the pairs whose vertices
    never move: libuipc-warp's ``SurfaceChecker`` on the primitives of ``a`` that move against all of ``b``, and
    on those of ``a`` that never move against those of ``b`` that move. ``arrays``: the engine's contact body of
    every vertex (the checker compares no pair within one) and the vertices' thickness, on the device. None when
    the parts make more than ``_START_CHECK_PAIRS`` primitive pairs."""
    from uipc_warp.validation import Gap, SurfaceChecker  # noqa: PLC0415

    def rows(name: str, g: int, moves: bool | None) -> np.ndarray:
        all_rows, moving, starts = prims[name]
        r, m = all_rows[starts[g] : starts[g + 1]], moving[starts[g] : starts[g + 1]]
        return (r if moves is None else r[m == moves]).astype(np.int32)

    halves = [
        ({n: rows(n, a, mine) for n in prims}, {n: rows(n, b, theirs) for n in prims})
        for mine, theirs in ((True, None), (False, True))
    ]
    size = 0
    for one, other in halves:
        v1, c1, t1, e1 = (len(one[n]) for n in ("vertices", "codim", "triangles", "edges"))
        v2, c2, t2, e2 = (len(other[n]) for n in ("vertices", "codim", "triangles", "edges"))
        # point-triangle, point-edge, edge-edge and point-point pairs, and the crossings, edges against triangles
        size += (v1 + c1) * t2 + (v2 + c2) * t1 + c1 * e2 + c2 * e1 + e1 * e2 + c1 * c2 + e1 * t2 + e2 * t1
    if size > _START_CHECK_PAIRS:
        return None
    x, device = solver._engine.vm.x, solver.model.device
    chunk = 16384  # primitives of a kind on one side per launch: every launch stays below 2**31 threads

    body, thickness = arrays

    def checker(part: dict, k: int) -> SurfaceChecker:
        surface = {n: part[n][k : k + chunk] for n in prims}
        return SurfaceChecker(surface, body, device, thickness=thickness)

    found, best = 0, Gap(np.inf, np.inf, 0.0, "", ())
    for one, other in halves:
        for e, f in ((one["edges"], other["triangles"]), (other["edges"], one["triangles"])):
            for i in range(0, len(e), chunk):
                for j in range(0, len(f), chunk):
                    surface = {"edges": e[i : i + chunk], "triangles": f[j : j + chunk], "vertices": np.zeros(0)}
                    found += SurfaceChecker(surface, body, device, thickness=thickness).intersections(x)
        pieces = [max(len(part[n]) for n in prims) for part in (one, other)]
        for i in range(0, pieces[0], chunk):
            for j in range(0, pieces[1], chunk):
                gap = checker(one, i).min_gap(x, checker(other, j))
                if gap.gap < best.gap:
                    best = gap
    return found, best


def _describe_vertex(solver: SolverIPC, v: int) -> str:
    """The Newton particle, body or shape that engine vertex ``v`` belongs to."""
    layout = solver._scene.layout()
    if v < layout.n_fem:
        return f"particle {v}"
    if v < layout.obstacle_offsets[-1]:
        k = int(np.searchsorted(layout.obstacle_offsets, v, side="right")) - 1
        return f"static shape {solver._obstacle_shapes[k]}"
    b = int(np.searchsorted(layout.body_vertex_offsets, v, side="right")) - 1
    label = solver.model.body_label[b] if b < len(solver.model.body_label) else ""
    return f"body {b}" + (f" ({label!r})" if label and label != f"body_{b}" else "")


def _describe_group(solver: SolverIPC, group: np.ndarray, points: np.ndarray, g: int) -> str:
    """The Newton body, static shape or particles of contact body ``g``."""
    first = int(points[group[points] == g].min())
    n_fem = solver._scene.layout().n_fem
    if first >= n_fem:
        return _describe_vertex(solver, first)
    members = np.flatnonzero(group[:n_fem] == g)  # with the particles inside a soft body
    first, last = int(members.min()), int(members.max())
    if len(members) == 1:
        return f"particle {first}"
    if last - first + 1 == len(members):
        return f"particles {first} to {last}"
    return f"{len(members)} particles from particle {first}"
