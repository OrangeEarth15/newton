# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A Newton model laid out as libuipc-warp's ``ArrayScene``.

What the solver simulates (its particles, elements, kinematic bodies and a coupled solver's proxies) and the worlds
as libuipc subscenes; particles and tetrahedra, cloth with Newton's rest shapes, springs, and the stitches of
particle-body attachments and springs of rest length 0; shapes triangulated into meshes, rigid bodies as affine bodies
with their dyadic masses and orthogonality stiffness, static planes as half planes; and the default ``d_hat``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from newton import Axis, BodyFlags, GeoType, Mesh, Model, NewtonWarning, ParticleFlags, ShapeFlags, eval_fk

from .kernels import _body_stiffness, _distance_stiffness, _stitch_stiffness

if TYPE_CHECKING:
    from .contacts import _ContactElements
    from .solver_ipc import SolverIPC


# libuipc's barrier activation distance (contact/d_hat) [m]. The default d_hat stays below the shares of the smallest
# colliding shape's half-size and of the median edge of the cloth and tetrahedral meshes at which the barrier spans the
# feature: a contact list overflows from about half the radius of a small sphere, a mesh stretches from one edge length
_D_HAT = 0.01
_D_HAT_SHAPE_SHARE = 0.25
_D_HAT_MESH_SHARE = 0.5
# libuipc's orthogonality stiffness of an affine body (AffineBodyConstitution's kappa, 100 MPa) [Pa]
_BODY_KAPPA = 1.0e8
# the turn [rad] of a body in one step, about the axis of its largest principal moment of inertia, that the first Newton
# iteration takes in full under the default body_kappa: libuipc's stiffness is lowered where the orthogonality term of
# a step's incremental potential would exceed 1 / (4 _BODY_TURN^2) times that moment (docs/solvers/ipc.rst >
# "Rigid bodies")
_BODY_TURN = 0.1


def _colliding_shapes(model: Model) -> np.ndarray:
    """Which shapes the solver collides: a shape that collides with shapes, and, in a model with particles, a shape
    that collides with particles. In a model without particles a shape that collides with particles only collides
    with nothing."""
    flags = model.shape_flags.numpy()
    mask = int(ShapeFlags.COLLIDE_SHAPES) | (int(ShapeFlags.COLLIDE_PARTICLES) if model.particle_count else 0)
    return (flags & mask) != 0


def _default_d_hat(model: Model) -> float:
    """libuipc's d_hat, or less when the model has smaller colliding features [m]: a quarter of the smallest half-size of
    a colliding shape other than a plane (a sphere's or capsule's radius, the smaller of a cylinder's or cone's radius
    and half height, a box's or ellipsoid's smallest half extent, half of a mesh's thinnest extent that is not zero),
    and half the median rest length of the edges of the cloth triangles and tetrahedra the solver simulates
    (``_own_elements``)."""
    size = np.inf
    if model.shape_count:
        types, scale = model.shape_type.numpy(), model.shape_scale.numpy().astype(np.float64)
        for s in np.flatnonzero(_colliding_shapes(model)):
            t, sc = int(types[s]), scale[s]
            if t in (GeoType.BOX, GeoType.ELLIPSOID):
                half = sc.min()
            elif t in (GeoType.SPHERE, GeoType.CAPSULE):
                half = sc[0]
            elif t in (GeoType.CYLINDER, GeoType.CONE):
                half = min(sc[0], sc[1])
            elif t in (GeoType.MESH, GeoType.CONVEX_MESH) and model.shape_source[s] is not None:
                extent = np.ptp(np.asarray(model.shape_source[s].vertices, np.float64).reshape(-1, 3) * sc, axis=0)
                extent = extent[extent > 0.0]
                half = 0.5 * extent.min() if len(extent) else np.inf
            else:
                continue
            if half > 0.0:
                size = min(size, float(half))
    d_hat = min(_D_HAT, _D_HAT_SHAPE_SHARE * size)
    edges = []
    own_tris, own_tets, _ = _own_elements(model)
    if own_tris.any():
        t = model.tri_indices.numpy().reshape(-1, 3)[own_tris]
        edges.append(np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]))
    if own_tets.any():
        t = model.tet_indices.numpy().reshape(-1, 4)[own_tets]
        edges.append(np.concatenate([t[:, [a, b]] for a in range(3) for b in range(a + 1, 4)]))
    if edges:
        e = np.unique(np.sort(np.concatenate(edges), axis=1), axis=0)
        x = model.particle_q.numpy().astype(np.float64)
        length = np.linalg.norm(x[e[:, 0]] - x[e[:, 1]], axis=1)
        length = length[length > 0.0]
        if len(length):
            d_hat = min(d_hat, _D_HAT_MESH_SHARE * float(np.median(length)))
    return d_hat


def _check_springs(model: Model) -> None:
    """Springs that libuipc represents: no damping (libuipc has no damping element), a rest length that is not
    negative (above 0 libuipc's Hookean spring, 0 its soft vertex stitch) and a stiffness that is not negative. The
    springs of a coupled solver's other entries are theirs (``_own_elements``)."""
    if model.spring_count == 0:
        return
    own = _own_elements(model)[2]
    for name, array, bad, remedy in (
        ("spring_damping", model.spring_damping, lambda v: v != 0.0, "set it to 0. libuipc's springs have no damping"),
        ("spring_rest_length", model.spring_rest_length, lambda v: ~(v >= 0.0), "make it 0 or positive"),
        ("spring_stiffness", model.spring_stiffness, lambda v: ~(v >= 0.0), "make it 0 or positive"),
    ):
        values = array.numpy()
        found = np.flatnonzero(bad(values) & own)
        if len(found):
            k = int(found[0])
            raise NotImplementedError(f"SolverIPC: spring {k} has {name} {float(values[k])!r}; {remedy}.")
    ends = model.spring_indices.numpy().reshape(-1, 2)
    same = np.flatnonzero((ends[:, 0] == ends[:, 1]) & own)
    if len(same):
        k = int(same[0])
        raise NotImplementedError(
            f"SolverIPC: spring {k} joins particle {int(ends[k, 0])} to itself; remove it, or join two particles."
        )


def _check_attachments(model: Model) -> None:
    """Particle-body attachments that libuipc's soft vertex stitch represents: no damping (libuipc's stitch has none,
    and an implicit damping cannot be folded into it the way the joint drives fold theirs, since it has no rest
    offset to carry it) and a stiffness that is not negative, finite or infinite (hard)."""
    if model.attachment_particle_body_count == 0:
        return
    for name, array, bad, remedy in (
        ("damping", model.attachment_particle_body_damping, lambda v: v != 0.0, "set it to 0. libuipc's stitch has no "
         "damping"),
        ("stiffness", model.attachment_particle_body_stiffness, lambda v: ~(v >= 0.0), "make it 0 or positive, or "
         "infinite for a hard attachment"),
    ):  # fmt: skip
        values = array.numpy()
        found = np.flatnonzero(bad(values))
        if len(found):
            k = int(found[0])
            raise NotImplementedError(
                f"SolverIPC: particle-body attachment {k} has {name} {float(values[k])!r}; {remedy}."
            )


def _kinematic_bodies(model: Model) -> np.ndarray:
    """The bodies the solver does not let forces move: :attr:`~newton.BodyFlags.KINEMATIC` ones, those of mass 0, and
    those of inverse mass 0. A body of positive mass has an inverse mass of 0 only in a coupled solver's view, which
    zeroes the inverse mass of the bodies another entry owns and keeps their mass (:ref:`ipc-coupling`)."""
    if model.body_count == 0:
        return np.zeros(0, bool)
    kinematic = ((model.body_flags.numpy() & int(BodyFlags.KINEMATIC)) != 0) | (model.body_mass.numpy() <= 0.0)
    return kinematic | (model.body_inv_mass.numpy() <= 0.0)


def _own_elements(model: Model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Which triangles, tetrahedra and springs the solver simulates, as masks over ``tri_indices``, ``tet_indices`` and
    ``spring_indices``: every one but those whose particles are all disabled, of mass 0 and not
    :attr:`~newton.ParticleFlags.ACTIVE`. A coupled solver's view disables the particles another entry owns, which
    simulates their cloth, soft bodies and springs (:ref:`ipc-coupling`); in a model of one solver such a triangle or
    tetrahedron holds nothing that moves, and Newton's own contacts skip its faces."""
    own = tuple(np.ones(n, bool) for n in (model.tri_count, model.tet_count, model.spring_count))
    if model.particle_count == 0 or not (model.tri_count or model.tet_count or model.spring_count):
        return own
    flags = model.particle_flags.numpy()
    disabled = (model.particle_mass.numpy() <= 0.0) & ((flags & int(ParticleFlags.ACTIVE)) == 0)
    if not disabled.any():
        return own
    indices = ((model.tri_indices, 3), (model.tet_indices, 4), (model.spring_indices, 2))
    return tuple(
        ~disabled[rows.numpy().reshape(-1, k)].all(axis=1) if len(mask) else mask
        for mask, (rows, k) in zip(own, indices, strict=True)
    )


def _proxy_bodies(model: Model) -> np.ndarray:
    """The bodies a coupled solver's view marks as proxies (:attr:`~newton.BodyFlags.PROXY`): another entry's bodies
    that this entry moves as its own, with the view's masses (:ref:`ipc-coupling`)."""
    if model.body_count == 0:
        return np.zeros(0, bool)
    return (model.body_flags.numpy() & int(BodyFlags.PROXY)) != 0


def _proxy_particles(model: Model) -> np.ndarray:
    """The particles a coupled solver's view marks as proxies (:attr:`~newton.ParticleFlags.PROXY`)."""
    if model.particle_count == 0:
        return np.zeros(0, bool)
    return (model.particle_flags.numpy() & int(ParticleFlags.PROXY)) != 0


def _local_worlds(model: Model) -> np.ndarray:
    """The local worlds (index >= 0) that hold particles, bodies or colliding shapes, in increasing order."""
    worlds = []
    if model.particle_count:
        worlds.append(model.particle_world.numpy())
    if model.body_count:
        worlds.append(model.body_world.numpy())
    if model.shape_count:
        worlds.append(model.shape_world.numpy()[_colliding_shapes(model)])
    local = np.unique(np.concatenate(worlds)) if worlds else np.zeros(0, np.int32)
    return local[local >= 0].astype(np.int64)


def _element_across_worlds(model: Model, free: np.ndarray) -> str | None:
    """The first triangle, tetrahedron or spring the solver simulates whose free particles (``free``) lie in two local
    worlds, named with the two worlds, or None. With independent worlds no element joins two worlds, which
    libuipc-warp refuses in its own terms; Newton's builder lets a spring, a triangle or a tetrahedron join any
    particles. A bending edge joins two worlds only where one of its triangles does. A global particle is in no world
    (a free global particle is refused before)."""
    if model.particle_count == 0:
        return None
    world = model.particle_world.numpy().astype(np.int64)
    own_tris, own_tets, own_springs = _own_elements(model)
    for name, indices, k, own in (
        ("triangle", model.tri_indices, 3, own_tris),
        ("tetrahedron", model.tet_indices, 4, own_tets),
        ("spring", model.spring_indices, 2, own_springs),
    ):
        if indices is None or len(indices) == 0:
            continue
        ids = np.flatnonzero(own)
        rows = indices.numpy().reshape(-1, k).astype(np.int64)[ids]
        index = np.maximum(rows, 0)
        mine = (rows >= 0) & free[index] & (world[index] >= 0)
        lo = np.where(mine, world[index], np.iinfo(np.int64).max).min(axis=1)
        hi = np.where(mine, world[index], -1).max(axis=1)
        joined = np.flatnonzero(mine.any(axis=1) & (lo != hi))
        if len(joined):
            i = int(joined[0])
            more = f" ({len(joined) - 1} more {name}s do too)" if len(joined) > 1 else ""
            return f"{name} {int(ids[i])} joins free particles of the worlds {int(lo[i])} and {int(hi[i])}{more}"
    return None


@dataclass
class _Worlds:
    """The subscenes of a model with content in more than one local world (several worlds): subscene 0 holds the
    global entities (world -1), subscene k + 1 the k-th local world that holds content."""

    worlds: np.ndarray  # (L,): the world of subscene k + 1
    lookup: np.ndarray  # (world_count + 1,): the subscene of world w at index w, of the global world -1 at the last

    @classmethod
    def of(cls, model: Model) -> _Worlds | None:
        """The subscenes of ``model``, or None when it has content in at most one local world."""
        local = _local_worlds(model)
        if len(local) < 2:
            return None
        lookup = np.zeros(model.world_count + 1, np.int64)
        lookup[local] = 1 + np.arange(len(local))
        return cls(local, lookup)

    @property
    def count(self) -> int:
        """The subscenes, the global one included."""
        return len(self.worlds) + 1

    def subscene(self, world: np.ndarray) -> np.ndarray:
        """The subscene of every entity of ``world`` (-1 for a global one)."""
        return self.lookup[np.asarray(world, np.int64)]


def _quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate the rows of ``v`` by the unit quaternion ``q = (x, y, z, w)``."""
    u = np.asarray(q[:3], np.float64)
    w = float(q[3])
    return v + 2.0 * w * np.cross(u, v) + 2.0 * np.cross(u, np.cross(u, v))


def _quat_to_matrix_np(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64) / np.linalg.norm(q)
    return _quat_rotate_np(q, np.eye(3)).T


def _weld(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge coincident vertices (the seams and poles of generated meshes) and drop degenerate triangles."""
    v = np.asarray(v, np.float64).reshape(-1, 3)
    f = np.asarray(f, np.int64).reshape(-1, 3)
    if len(v) == 0 or len(f) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    tol = 1.0e-6 * max(float(np.ptp(v, axis=0).max()), 1.0e-12)
    _, first, inv = np.unique(np.round(v / tol).astype(np.int64), axis=0, return_index=True, return_inverse=True)
    f = inv.reshape(-1)[f]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 2] != f[:, 0])]
    _, keep = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
    f = f[np.sort(keep)]
    used, f = np.unique(f, return_inverse=True)
    return v[first][used], f.reshape(-1, 3)


def _shape_mesh(model: Model, s: int, segments: int) -> tuple[np.ndarray, np.ndarray]:
    """Closed triangle mesh of shape ``s`` in its local frame, scaled."""
    t = int(model.shape_type.numpy()[s])
    sc = model.shape_scale.numpy()[s].astype(np.float64)
    opts = {"compute_normals": False, "compute_uvs": False, "compute_inertia": False}
    if t == GeoType.BOX:
        mesh = Mesh.create_box(sc[0], sc[1], sc[2], duplicate_vertices=False, **opts)
    elif t == GeoType.SPHERE:
        mesh = Mesh.create_sphere(sc[0], num_latitudes=max(segments // 2, 2), num_longitudes=segments, **opts)
    elif t == GeoType.ELLIPSOID:
        mesh = Mesh.create_ellipsoid(
            sc[0], sc[1], sc[2], num_latitudes=max(segments // 2, 2), num_longitudes=segments, **opts
        )
    elif t == GeoType.CAPSULE:
        mesh = Mesh.create_capsule(sc[0], sc[1], up_axis=Axis.Z, segments=segments, **opts)
    elif t == GeoType.CYLINDER:
        mesh = Mesh.create_cylinder(sc[0], sc[1], up_axis=Axis.Z, segments=segments, **opts)
    elif t == GeoType.CONE:
        mesh = Mesh.create_cone(sc[0], sc[1], up_axis=Axis.Z, segments=segments, **opts)
    elif t in (GeoType.MESH, GeoType.CONVEX_MESH):
        src = model.shape_source[s]
        return _weld(np.asarray(src.vertices, np.float64) * sc, np.asarray(src.indices))
    elif t == GeoType.HFIELD:
        return _heightfield_mesh(model, s)
    else:
        raise NotImplementedError(
            f"SolverIPC: shape {s} is a {GeoType(t).name} shape; give it another type, or clear its collide flags. "
            "SolverIPC triangulates every colliding shape and has no triangulation of this type."
        )
    return _weld(mesh.vertices, mesh.indices)


def _heightfield_mesh(model: Model, s: int) -> tuple[np.ndarray, np.ndarray]:
    """The collision triangles of height field shape ``s`` in its frame, as Newton's narrow phase makes them
    (``get_triangle_shape_from_heightfield``): sample ``(r, c)`` at ``x = -hx + c dx``, ``y = -hy + r dy``,
    ``z = min_z + h (max_z - min_z)``, every cell ``(r, c)`` split into ``(p00, p10, p11)`` and ``(p00, p11, p01)``.
    The field is the shape's :class:`~newton.Heightfield` (:attr:`~newton.Model.shape_source`), its extents scaled by
    :attr:`~newton.Model.shape_scale` and rounded to float32 with its heights, as ``ModelBuilder.finalize`` stores them
    for Newton's collision. An open surface; none for a field of fewer than two rows or columns or without a source,
    which Newton's collision ignores too."""
    field = model.shape_source[s]
    if field is None or field.nrow < 2 or field.ncol < 2:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    nrow, ncol = int(field.nrow), int(field.ncol)
    sx, sy, sz = (float(k) for k in model.shape_scale.numpy()[s])
    hx, hy, min_z, max_z = (
        float(np.float32(k)) for k in (abs(field.hx * sx), abs(field.hy * sy), field.min_z * sz, field.max_z * sz)
    )
    h = np.asarray(field.data, np.float32).astype(np.float64).reshape(nrow, ncol)
    y, x = np.meshgrid(np.linspace(-hy, hy, nrow), np.linspace(-hx, hx, ncol), indexing="ij")
    v = np.stack([x, y, min_z + h * (max_z - min_z)], axis=-1).reshape(-1, 3)
    r, c = np.meshgrid(np.arange(nrow - 1), np.arange(ncol - 1), indexing="ij")
    p00, p10 = r * ncol + c, r * ncol + c + 1
    p01, p11 = p00 + ncol, p10 + ncol
    f = np.stack([np.stack([p00, p10, p11], -1), np.stack([p00, p11, p01], -1)], axis=2).reshape(-1, 3)
    return v, f.astype(np.int64)


def _winding_signs(tris: np.ndarray, rows: np.ndarray, names: np.ndarray) -> np.ndarray:
    """For bending edge rows ``(o0, o1, v1, v2)``: 1 where the triangle ``(o0, v1, v2)`` of ``tris`` runs the edge
    ``v1 -> v2``, -1 where it runs ``v2 -> v1``.

    Newton's rest angle of a row is the signed angle between the normals of ``(o0, v1, v2)`` and ``(o1, v2, v1)`` about
    ``v2 - v1`` (``edge_rest_angle``). Where ``o0``'s triangle runs ``v1 -> v2``, as in every row of ``MeshAdjacency``,
    these are the normals by the triangles' winding and the angle is the one libuipc takes; where it runs
    ``v2 -> v1``, both normals are reversed and the angle is its negative. Raises ``ValueError``, naming the edge by
    its row in ``names``, for a row whose triangles are not among ``tris`` or run its edge the same way, which has no
    angle by the winding.
    """
    record = [("a", np.int64), ("b", np.int64), ("c", np.int64)]

    def keys(t):
        return np.ascontiguousarray(np.sort(np.asarray(t, np.int64), axis=1)).view(record).reshape(-1)

    table = keys(tris)
    order = np.argsort(table, kind="stable")
    v1, v2 = rows[:, 2], rows[:, 3]

    def runs_forward(opposite):
        """Whether the triangle of ``opposite`` and the edge runs ``v1 -> v2``, and whether there is one."""
        wanted = keys(np.stack([opposite, v1, v2], axis=1))
        at = np.minimum(np.searchsorted(table[order], wanted), len(order) - 1)
        found = table[order[at]] == wanted
        t = tris[order[at]]
        return np.any((t == v1[:, None]) & (np.roll(t, -1, axis=1) == v2[:, None]), axis=1), found

    forward0, found0 = runs_forward(rows[:, 0])
    forward1, found1 = runs_forward(rows[:, 1])
    for bad, why in (
        (~(found0 & found1), "names opposite particles whose triangles with the edge are not both cloth triangles"),
        (forward0 == forward1, "has two triangles that run the edge in the same direction, so the edge has no rest "
         "angle by their winding; wind the cloth's triangles consistently"),
    ):  # fmt: skip
        k = np.flatnonzero(bad)
        if len(k):
            r = rows[k[0]]
            raise ValueError(
                f"SolverIPC: bending edge {int(names[k[0]])} (particles {int(r[2])} and {int(r[3])}, opposite "
                f"{int(r[0])} and {int(r[1])}) {why}."
            )
    return np.where(forward0, 1.0, -1.0)


# Copies of Newton's rest-shape helpers (newton/_src/geometry/rest_shape.py), which have no public name. Newton's
# builders compute tri_poses and edge_rest_angle with them; _rest_edges and _add_cloth compute them again to tell
# whether a cloth was built at rest. test_rest_shape_copies_match_newtons_builders fails when they no longer reproduce
# what the builders store.


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors along the last axis, leaving zero vectors unchanged."""
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    return vectors / lengths


def _triangle_rest_geometry(qp: np.ndarray, rp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return triangle rest bases and areas from two edges, preserving input precision."""
    normal = _normalize_vectors(np.cross(qp, rp))
    e1 = _normalize_vectors(qp)
    e2 = _normalize_vectors(np.cross(normal, e1))
    frame = np.stack((e1, e2), axis=-1)
    edges = np.stack((qp, rp), axis=-1)
    basis = np.swapaxes(frame, -2, -1) @ edges
    return basis, np.linalg.det(basis) / 2.0


def _dihedral_angles(
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    x4: np.ndarray,
    x3_other: np.ndarray,
    x4_other: np.ndarray,
) -> np.ndarray:
    """Return signed dihedrals for triangles (x1, x3, x4) and (x2, x4_other, x3_other).

    The second triangle can use separate copies of the shared edge for disjoint rest
    topology. Welded geometry passes x3 and x4 for both copies.
    """
    n1 = _normalize_vectors(np.cross(x3 - x1, x4 - x1))
    n2 = _normalize_vectors(np.cross(x4_other - x2, x3_other - x2))
    edge = _normalize_vectors(x4 - x3)
    cos_theta = np.clip((n1 * n2).sum(axis=-1), -1.0, 1.0)
    sin_theta = (np.cross(n1, n2) * edge).sum(axis=-1)
    return np.arctan2(sin_theta, cos_theta)


def _rest_edges(x: np.ndarray, tris: np.ndarray, poses: np.ndarray) -> np.ndarray | None:
    """``ArrayScene.add_shell``'s ``rest_edges`` of the cloth triangles ``tris`` from Newton's ``tri_poses`` ``poses``.

    Newton's builders compute a triangle's rest edges in float64 from ``particle_q`` (``_triangle_rest_geometry``, in
    the edge frame ``add_shell`` takes) and store their inverse in float32. A triangle whose ``poses`` are the float32
    rounding of the inverse of its rest edges at ``x`` (``particle_q`` in float64) takes those rest edges, any other
    the inverse of its ``poses``. None when every triangle has the rest shape of ``x``: the engine then takes the rest
    shapes from the rest positions, as libuipc takes them from its rest geometry.
    """
    p = x[tris[:, 0]]
    edges, area = _triangle_rest_geometry(x[tris[:, 1]] - p, x[tris[:, 2]] - p)
    start = area > 0.0
    inverse = np.zeros_like(edges)
    inverse[start] = np.linalg.inv(edges[start])
    start &= np.all(inverse.astype(np.float32) == poses, axis=(1, 2))
    if start.all():
        return None
    rest = np.linalg.inv(poses.astype(np.float64))
    rest[start] = edges[start]
    return rest


def _closed_volume(v: np.ndarray, f: np.ndarray) -> float:
    """Enclosed volume of a closed triangle mesh (divergence theorem), either orientation."""
    if len(f) == 0:
        return 0.0
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


def _check_independent_content(solver: SolverIPC) -> None:
    """With independent worlds: ``NotImplementedError`` for an unknown of the global world -1, which every world
    would push and so join (a particle with mass that is ACTIVE, a dynamic body, a coupled solver's proxy, or with
    ``kinematic_motion="aim"`` a kinematic body with colliding shapes), as Newton's implicit MPM refuses global
    colliders backed by dynamic bodies with ``separate_worlds``, and for a triangle, tetrahedron or spring that
    joins two worlds. Then ``_status_entries``: the worlds that hold unknowns, in increasing order, which are
    libuipc-warp's independent worlds in their order (its worlds are the subscenes from 1 on that hold unknowns,
    in subscene order, and subscene k + 1 is the k-th local world with content, ``_Worlds``)."""
    model = solver.model
    world = model.particle_world.numpy() if model.particle_count else np.zeros(0, np.int64)
    free = ~solver._fixed_particle
    particle_sub = solver._worlds.subscene(world)
    moving = ~solver._kinematic_body
    moving[solver._aim_bodies] = True
    body_sub = solver._body_row_subscene
    instead = "or pass coupled_worlds=True to solve the worlds as one, in which global entities may move"
    lost = np.flatnonzero(free & (particle_sub == 0))
    if len(lost):
        more = f" ({len(lost) - 1} more global particles are too)" if len(lost) > 1 else ""
        raise NotImplementedError(
            f"SolverIPC: particle {int(lost[0])} of the global world -1 is free (it has mass and is ACTIVE){more}: "
            "with independent worlds it would be an unknown that every world pushes, which would join the worlds. "
            f"Give it a world or make it kinematic (mass 0), {instead}."
        )
    lost = np.flatnonzero(moving & (body_sub == 0))
    if len(lost):
        b = int(lost[0])
        label = model.body_label[b] if b < len(model.body_label) else ""
        name = f"body {b}" + (f" ({label!r})" if label and label != f"body_{b}" else "")
        more = f" ({len(lost) - 1} more global bodies are too)" if len(lost) > 1 else ""
        if solver._kinematic_body[b]:
            what, fix = (
                "a kinematic body with colliding shapes, which kinematic_motion='aim' moves toward its aim as an "
                "unknown of the step",
                "Use kinematic_motion='jump', under which it is a fixed body that acts in every world, give it a world",
            )
        else:
            what, fix = "a dynamic body", "Give it a world or make it kinematic"
        raise NotImplementedError(
            f"SolverIPC: {name} of the global world -1 is {what}{more}: with independent worlds it would be an "
            f"unknown that every world pushes, which would join the worlds. {fix}, {instead}."
        )
    across = _element_across_worlds(model, free)
    if across is not None:
        raise NotImplementedError(
            f"SolverIPC: {across}: with independent worlds no element joins two worlds. Keep it inside one world, "
            f"{instead}."
        )
    subscenes = np.unique(np.concatenate([particle_sub[free], body_sub[moving]]).astype(np.int64))
    solver._status_entries = solver._worlds.worlds[subscenes - 1].astype(np.int32)


def _subscene_of_shape(solver: SolverIPC, s: int) -> int:
    """The subscene of shape ``s``'s vertices: its world's (several worlds), else the one subscene, 0."""
    return 0 if solver._worlds is None else int(solver._worlds.subscene(solver.model.shape_world.numpy()[s]))


def _subscene_of_body(solver: SolverIPC, b: int) -> int:
    """The subscene of body ``b``, for the anchors of a body without shapes: its world's (several worlds),
    else the one subscene, 0."""
    return 0 if solver._worlds is None else int(solver._worlds.subscene(solver.model.body_world.numpy()[b]))


def _world_gravity(solver: SolverIPC) -> np.ndarray:
    """``Model.gravity`` in float64: one row per world, the global world's last, so that the world of an entity
    (-1 for a global one) indexes its row."""
    return solver.model.gravity.numpy().reshape(-1, 3).astype(np.float64)


def _vertex_gravity(solver: SolverIPC) -> np.ndarray:
    """The gravity of every engine vertex, as the scene lays it out, from the gravity the solver took last
    (``_gravity``): a particle's world's on the particle, a body's world's on its translation, none elsewhere
    (``SimEngine.set_gravity``)."""
    layout = solver._scene.layout()
    g = solver._gravity
    out = np.zeros((layout.n, 3))
    if solver.model.particle_count:
        out[: layout.n_fem] = g[solver.model.particle_world.numpy()]
    if solver.model.body_count:
        out[layout.body_base[: solver.model.body_count]] = g[solver.model.body_world.numpy()]
        out[layout.body_base[solver._aim_bodies]] = 0.0  # gravity does not move a kinematic body toward its aim
    return out


def _find_body_stiffness(solver: SolverIPC) -> None:
    """With the default ``body_kappa``: the engine's orthogonality potential (libuipc-warp's ``orthogonality``),
    whose stiffness array a kernel writes before every step, the body of each of its rows, and the largest
    principal moment of inertia over the volume of that body (0 for a kinematic body and the world's). None
    without bodies or with a given ``body_kappa``, which the engine is built with."""
    solver._ortho = solver._ortho_inertia = None
    if solver._body_kappa is not None or solver.model.body_count == 0:
        return
    solver._ortho = solver._uipc_warp.arrays.orthogonality(solver._engine)
    solver._ortho_body = solver._ortho.bodies
    solver._ortho_inertia = wp.array(_ortho_ratios(solver), dtype=wp.float64, device=solver.model.device)


def _ortho_ratios(solver: SolverIPC) -> np.ndarray:
    """For every row of the orthogonality group, its body's largest principal moment of inertia about the
    centre of mass over its volume, from the dyadic mass the engine holds; 0 for a kinematic body and the
    world's."""
    mass, first, second = solver._dyadic
    nb = solver.model.body_count
    ratio = np.zeros(len(solver._ortho_body))
    for i, b in enumerate(solver._ortho_body):
        if b >= nb or solver._kinematic_body[b] or mass[b] <= 0.0:
            continue
        c = first[b] / mass[b]
        spread = np.linalg.eigvalsh(second[b] - mass[b] * np.outer(c, c))
        ratio[i] = (spread.sum() - spread.min()) / solver._body_volume[b]
    return ratio


def _write_body_stiffness(solver: SolverIPC, dt: float) -> None:
    """The orthogonality stiffness of every body for a step of ``dt`` (default ``body_kappa``, :ref:`ipc-rigid-bodies`):
    libuipc's, lowered to 1 / (4 _BODY_TURN^2) times the largest principal moment of inertia over ``V dt^2``, on the
    device, so that a captured step takes the stiffness of its own ``dt``."""
    if solver._ortho is None:
        return
    wp.launch(
        _body_stiffness,
        len(solver._ortho_body),
        inputs=[solver._ortho_inertia, wp.float64(_BODY_KAPPA), wp.float64(1.0 / (4.0 * _BODY_TURN**2 * dt * dt))],
        outputs=[solver._ortho.kappa],
        device=solver.model.device,
    )


def _add_particles_and_tets(solver: SolverIPC, scene, elements: _ContactElements) -> np.ndarray:
    """Every particle as a finite-element vertex (engine index = particle index), and the tetrahedra."""
    model = solver.model
    n = model.particle_count
    solver._fixed_particle = np.zeros(n, bool)
    if n == 0:
        return np.zeros((0, 4), np.int64)
    mass = model.particle_mass.numpy().astype(np.float64)
    flags = model.particle_flags.numpy()
    solver._fixed_particle = (mass <= 0.0) | ((flags & int(ParticleFlags.ACTIVE)) == 0)
    # a free particle is a sphere of its radius; the vertices of cloth and tetrahedra take the solver's thickness
    radius = model.particle_radius.numpy().astype(np.float64)
    scene.add_vertices(
        model.particle_q.numpy().astype(np.float64),
        model.particle_qd.numpy().astype(np.float64),
        mass=mass,
        fixed=solver._fixed_particle,
        thickness=np.where(elements.on_mesh, solver._thickness, radius),
        contact_element=elements.particle,
        subscene_element=0 if solver._worlds is None else solver._worlds.subscene(model.particle_world.numpy()),
        gravity=solver._gravity[model.particle_world.numpy()],
    )
    own = _own_elements(model)[1]  # in a coupled solver's view, without another entry's tetrahedra
    if not own.any():
        return np.zeros((0, 4), np.int64)
    tets = model.tet_indices.numpy().reshape(-1, 4).astype(np.int64)[own]
    Dm_inv = model.tet_poses.numpy().astype(np.float64).reshape(-1, 3, 3)[own]
    volume = 1.0 / (6.0 * np.abs(np.linalg.det(Dm_inv)))
    materials = model.tet_materials.numpy().astype(np.float64).reshape(-1, 3)[own]
    # k_mu and k_lambda are Lame parameters; libuipc's StableNeoHookean.apply_to stores mu and lambda + mu, the
    # SNK1 parameters whose linearization is Hooke's law with those Lame parameters (stable_neo_hookean.cpp)
    k_mu, k_lambda = materials[:, 0], materials[:, 1]
    scene.add_tetrahedra(tets, Dm_inv, volume, k_mu, k_lambda + k_mu)
    return tets


def _add_cloth(solver: SolverIPC, scene, tets: np.ndarray) -> None:
    """Triangles not on a tetrahedral body: Baraff-Witkin membrane and discrete-shell hinges; in a coupled solver's
    view, not another entry's (``_own_elements``)."""
    model = solver.model
    if model.tri_count == 0:
        return
    tris = model.tri_indices.numpy().reshape(-1, 3).astype(np.int64)
    on_tet = np.zeros(model.particle_count, bool)
    on_tet[tets.ravel()] = True
    cloth = ~on_tet[tris].all(axis=1) & _own_elements(model)[0]
    if not cloth.any():
        return
    tris = tris[cloth]
    mat = model.tri_materials.numpy().astype(np.float64).reshape(-1, 5)[cloth]
    k_mu, k_lambda = mat[:, 0], mat[:, 1]
    stretch = k_mu + 0.5 * k_lambda
    shear = 0.5 * k_mu
    # Newton's tri_poses are the inverses of the rest edges in each triangle's edge frame, the frame add_shell
    # takes them in, computed by Newton's builders in float64 from particle_q and stored in float32: a triangle
    # whose tri_poses are the rounding of the rest edges of particle_q takes those in float64 (_rest_edges)
    x = model.particle_q.numpy().astype(np.float64)
    rest_edges = _rest_edges(x, tris, model.tri_poses.numpy().reshape(-1, 2, 2)[cloth])

    edges, kappa, angles = np.zeros((0, 2), np.int64), np.zeros(0), None
    if model.edge_count:
        e = model.edge_indices.numpy().reshape(-1, 4).astype(np.int64)
        ke = model.edge_bending_properties.numpy().astype(np.float64).reshape(-1, 2)[:, 0]
        # hinges: both opposite vertices present, the edge on the cloth
        n = model.particle_count
        cloth_keys = np.unique(
            np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1) @ [n, 1]
        )
        keys = np.sort(e[:, 2:4], axis=1) @ [n, 1]
        sel = (e[:, 0] >= 0) & (e[:, 1] >= 0) & np.isin(keys, cloth_keys)
        e, ke = e[sel], ke[sel]
        # Newton's rest angle of a row by the winding of its triangles (add_shell's convention)
        signs = _winding_signs(tris, e, np.flatnonzero(sel))
        x0, x1, x2, x3 = x[e[:, 0]], x[e[:, 1]], x[e[:, 2]], x[e[:, 3]]
        # Newton's builders measure a row's rest angle at particle_q in float64 and store it in float32. When every
        # edge_rest_angle is that measure, the rest angles are those of the start, and the engine takes them from
        # the rest positions, as libuipc takes them from its rest geometry, with the dihedral angle its bending
        # energy evaluates: libuipc's arccosine of the normals' cosine reads up to 1.5e-8 rad on a flat hinge,
        # which Newton measures as 0, so that a rest angle of 0 would push a flat cloth out of its plane
        rest = model.edge_rest_angle.numpy()[sel]
        at_start = _dihedral_angles(x0, x1, x2, x3, x2, x3).astype(np.float32) == rest
        angles = None if at_start.all() else signs * rest
        L0 = np.linalg.norm(x3 - x2, axis=1)
        area = 0.5 * (
            np.linalg.norm(np.cross(x2 - x0, x3 - x0), axis=1) + np.linalg.norm(np.cross(x2 - x1, x3 - x1), axis=1)
        )
        h_bar = area / (3.0 * np.maximum(L0, 1.0e-30))
        edges, kappa = e[:, 2:4], 0.5 * ke * h_bar
    scene.add_shell(
        tris,
        stretch,
        stretch,
        shear,
        solver._cloth_strain_rate,
        bending=kappa,
        bending_edges=edges,
        rest_edges=rest_edges,
        rest_angles=angles,
    )


def _add_springs(solver: SolverIPC, scene) -> None:
    """Newton's springs of positive rest length as libuipc's Hookean springs: ``0.5 ke (l - L0)^2`` each
    (``spring_kappa`` with a radius of 1 m, which only scales the energy, since a spring is no contact
    primitive). Those of rest length 0 become stitches (``_add_stitches``)."""
    model = solver.model
    solver._zero_springs = np.zeros(0, np.int64)
    if model.spring_count == 0:
        return
    edges = model.spring_indices.numpy().reshape(-1, 2).astype(np.int64)
    rest = model.spring_rest_length.numpy().astype(np.float64)
    ke = model.spring_stiffness.numpy().astype(np.float64)
    own = _own_elements(model)[2]  # in a coupled solver's view, not another entry's
    solver._zero_springs = np.flatnonzero((rest == 0.0) & own)
    hookean = (rest > 0.0) & own
    if not hookean.any():
        return
    kappa = solver._uipc_warp.arrays.spring_kappa(ke[hookean], rest[hookean], 1.0)
    scene.add_springs(edges[hookean], kappa, rest_lengths=rest[hookean], radius=1.0)


def _plan_attachments(solver: SolverIPC) -> None:
    """The anchors of every body that attachments hold (:ref:`ipc-attachments`): the distinct
    ``attachment_particle_body_body_point`` of its attachments in the order they first appear, and for every attachment
    its anchor among its body's."""
    model = solver.model
    count = model.attachment_particle_body_count
    solver._body_anchors = {}
    solver._attachment_anchor = np.zeros(count, np.int64)
    solver._attachment_inputs = ()
    if count == 0:
        return
    body = model.attachment_particle_body_body.numpy().astype(np.int64)
    particle = model.attachment_particle_body_particle.numpy().astype(np.int64)
    point = model.attachment_particle_body_body_point.numpy()
    solver._attachment_inputs = (body, particle, point.copy())
    for b in np.unique(body):
        rows = np.flatnonzero(body == b)
        points, first, inverse = np.unique(point[rows], axis=0, return_index=True, return_inverse=True)
        order = np.argsort(first)
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        solver._body_anchors[int(b)] = points[order].astype(np.float64)
        solver._attachment_anchor[rows] = rank[inverse.reshape(-1)]


def _attachment_penalties(solver: SolverIPC, mass: np.ndarray) -> np.ndarray:
    """The penalty of every attachment if it is hard (infinite stiffness): a ball joint's, ``joint_strength_ratio``
    times the dynamic masses it joins (a particle that is not fixed, a dynamic body), in the units of the
    incremental potential, so that its stiffness is that over ``dt^2``."""
    model = solver.model
    count = model.attachment_particle_body_count
    if count == 0:
        return np.zeros(1)
    body = model.attachment_particle_body_body.numpy().astype(np.int64)
    particle = model.attachment_particle_body_particle.numpy().astype(np.int64)
    particle_mass = np.where(solver._fixed_particle, 0.0, model.particle_mass.numpy().astype(np.float64))
    body_mass = np.where(solver._kinematic_body, 0.0, mass)
    return solver._joint_strength_ratio * (particle_mass[particle] + body_mass[body])


def _add_stitches(solver: SolverIPC, scene) -> None:
    """Newton's particle-body attachments and springs of rest length 0 as libuipc's soft vertex stitches of rest length
    0 (:ref:`ipc-attachments`): an attachment between its particle and its body's anchor, a spring between its two
    particles. Keeps the attachment of every stitch, or -1 less its spring, for the stiffness written at every step."""
    model = solver.model
    pairs, source = [], []
    if model.attachment_particle_body_count:
        body, particle, _ = solver._attachment_inputs
        anchor = scene.layout().body_anchor_offsets[body] + solver._attachment_anchor
        pairs.append(np.stack([particle, anchor], axis=1))
        source.append(np.arange(len(body)))
    if len(solver._zero_springs):
        pairs.append(model.spring_indices.numpy().reshape(-1, 2).astype(np.int64)[solver._zero_springs])
        source.append(-1 - solver._zero_springs)
    solver._stitch_source = np.concatenate(source).astype(np.int32) if source else np.zeros(0, np.int32)
    if pairs:
        scene.add_stitches(np.concatenate(pairs), _stitch_kappa(solver, solver._config.dt), 0.0)
    # the distance joints of one length: a stitch of that rest length between the anchors of their two sides,
    # after the other stitches (docs/solvers/ipc.rst > "Distance joints")
    solver._distance_first = len(solver._stitch_source)
    if len(solver._distance_joints):
        offsets = scene.layout().body_anchor_offsets
        a = solver._distance_anchor
        ends = np.stack([offsets[a[:, 0, 0]] + a[:, 0, 1], offsets[a[:, 1, 0]] + a[:, 1, 1]], axis=1)
        dt = solver._config.dt
        model = solver.model
        first = model.joint_qd_start.numpy()[solver._distance_joints]
        length = model.joint_limit_lower.numpy()[first].astype(np.float64)
        kappa = solver._distance_hard.numpy() / (dt * dt)
        kappa[~model.joint_enabled.numpy()[solver._distance_joints]] = 0.0
        solver._distance_first = scene.add_stitches(ends, kappa, length)


def _stitch_kappa(solver: SolverIPC, dt: float) -> np.ndarray:
    """The stiffness of every stitch for a step of ``dt``, as ``_stitch_stiffness`` writes it, on the host."""
    model, source = solver.model, solver._stitch_source
    kappa = np.zeros(len(source))
    attached = source >= 0
    if attached.any():
        k = model.attachment_particle_body_stiffness.numpy().astype(np.float64)
        hard = np.isinf(k)
        k[hard] = solver._attachment_hard.numpy()[hard] / (dt * dt)
        k[~model.attachment_particle_body_enabled.numpy()] = 0.0
        kappa[attached] = k[source[attached]]
    if (~attached).any():
        kappa[~attached] = model.spring_stiffness.numpy().astype(np.float64)[-1 - source[~attached]]
    return kappa


def _write_stitch_stiffness(solver: SolverIPC, dt: float) -> None:
    """The stiffness of every stitch for a step of ``dt`` (``_stitch_stiffness``), on the device, so that a
    captured step takes the attachments' stiffness and enabled flags of its replay, and the hard ones' penalty
    for its own ``dt``; and of the distance joints' stitches (``_distance_stiffness``)."""
    model = solver.model
    for kappa, rest_length, source in solver._distance_rows:
        wp.launch(
            _distance_stiffness,
            source.shape[0],
            inputs=[
                source,
                solver._distance_joints_device,
                model.joint_qd_start,
                model.joint_enabled,
                model.joint_limit_lower,
                solver._distance_hard,
                wp.float64(1.0 / (dt * dt)),
            ],
            outputs=[kappa, rest_length],
            device=model.device,
        )
    if not solver._stitch_rows:
        return
    model = solver.model
    attached = model.attachment_particle_body_count > 0
    springs = model.spring_count > 0
    for kappa, source in solver._stitch_rows:
        wp.launch(
            _stitch_stiffness,
            source.shape[0],
            inputs=[
                source,
                model.attachment_particle_body_stiffness if attached else None,
                model.attachment_particle_body_enabled if attached else None,
                solver._attachment_hard,
                model.spring_stiffness if springs else None,
                wp.float64(1.0 / (dt * dt)),
            ],
            outputs=[kappa],
            device=model.device,
        )


def _body_mass_moments(solver: SolverIPC, stacklevel: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The dyadic mass ``(m, sum m x_bar, sum m x_bar x_bar^T)`` of every body from Newton's mass properties,
    and which bodies are kinematic (``solver._kinematic_body``)."""
    model = solver.model
    nb = model.body_count
    solver._kinematic_body = np.zeros(nb, bool)
    if nb == 0:
        return np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3, 3))
    body_mass = model.body_mass.numpy().astype(np.float64)
    body_com = model.body_com.numpy().astype(np.float64)
    body_inertia = model.body_inertia.numpy().astype(np.float64)
    solver._kinematic_body = _kinematic_bodies(model)
    mass, first, second = np.zeros(nb), np.zeros((nb, 3)), np.zeros((nb, 3, 3))
    for b in range(nb):
        inertia = 0.5 * (body_inertia[b] + body_inertia[b].T)
        if not solver._kinematic_body[b] and np.linalg.eigvalsh(inertia).min() <= 0.0:
            raise ValueError(f"SolverIPC needs a positive-definite inertia for the dynamic body {b}.")
        m, mx, S = solver._uipc_warp.arrays.dyadic_mass(body_mass[b], body_com[b], inertia)
        # the second moment about the centre of mass must be PSD (the inertia's triangle inequality)
        cov = S - m * np.outer(body_com[b], body_com[b])
        lam, vec = np.linalg.eigh(cov)
        if lam.min() < 0.0:
            warnings.warn(
                f"SolverIPC: the inertia of body {b} violates the triangle inequality; clamping its mass spread.",
                NewtonWarning,
                stacklevel=stacklevel,
            )
            S = (vec * np.maximum(lam, 0.0)) @ vec.T + m * np.outer(body_com[b], body_com[b])
        mass[b], first[b], second[b] = m, mx, S
    return mass, first, second


def _spreads(dyadic: tuple) -> np.ndarray:
    """The second moment of mass about the centre of mass, in the body frame, of every body of ``dyadic``:
    ``sum m x_bar x_bar^T - (sum m x_bar)(sum m x_bar)^T / m``, zero for a body without mass."""
    mass, first, second = dyadic
    out = np.zeros((len(mass), 3, 3))
    heavy = mass > 0.0
    out[heavy] = second[heavy] - np.einsum("bi,bj->bij", first[heavy], first[heavy]) / mass[heavy, None, None]
    return out


def _mass_inverse(solver: SolverIPC, mass: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """The inverse of the 4 x 4 block ``[[m, (sum m x_bar)^T], [sum m x_bar, sum m x_bar x_bar^T]]`` of every
    dynamic body's dyadic mass, which maps a force on ``(p_k, row k of A)`` to its acceleration."""
    dynamic = np.flatnonzero(~solver._kinematic_body)
    M = np.zeros((len(dynamic), 4, 4))
    M[:, 0, 0] = mass[dynamic]
    M[:, 0, 1:] = first[dynamic]
    M[:, 1:, 0] = first[dynamic]
    M[:, 1:, 1:] = second[dynamic]
    return np.linalg.inv(M) if len(dynamic) else np.zeros((0, 4, 4))


def _scene_inputs_now(solver: SolverIPC) -> tuple:
    """What the scene was laid out from besides gravity and the bodies' mass properties, which change in place:
    the triangulated shapes, the friction coefficients of the contact models, the contact thicknesses, the
    particles' masses and the springs."""
    model = solver.model
    materials = (float(model.soft_contact_mu), float(model.particle_mu))
    arrays = [model.particle_radius, model.particle_mass] if model.particle_count else []
    if model.spring_count:
        arrays += [model.spring_rest_length, model.spring_stiffness, model.spring_damping]
    if model.shape_count:
        arrays += [model.shape_transform, model.shape_scale, model.shape_material_mu, model.shape_margin]
    return (*materials, *(a.numpy().copy() for a in arrays))


def _element_inputs_now(solver: SolverIPC) -> tuple:
    """What the solve's elements were built from that Newton has no flag of its own for: the materials and rest
    shapes of the tetrahedra, triangles and bending edges, and the particles' flags (which particles are fixed and
    which elements the solver simulates). ``notify_model_changed`` compares them at every notification."""
    model = solver.model
    arrays = (
        model.tet_materials,
        model.tet_poses,
        model.tri_materials,
        model.tri_poses,
        model.edge_bending_properties,
        model.edge_rest_angle,
        model.particle_flags,
    )
    return tuple(None if a is None else a.numpy().copy() for a in arrays)


def _add_shapes_and_bodies(
    solver: SolverIPC, scene, elements: _ContactElements
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Static non-plane shapes as fixed triangle meshes; every body as an affine body. Returns the bodies'
    dyadic masses."""
    model = solver.model
    meshes: dict[int, list] = {b: [] for b in range(model.body_count)}
    solver._obstacle_shapes = []  # the shape of every obstacle, in the order they are added
    # the subscene of every body's rows: that of its surface vertices, as the engine takes it (one subscene)
    solver._body_row_subscene = np.zeros(model.body_count, np.int64)
    if model.shape_count:
        shape_type = model.shape_type.numpy()
        shape_body = model.shape_body.numpy()
        colliding = _colliding_shapes(model)
        shape_xform = model.shape_transform.numpy().astype(np.float64)
        margin = model.shape_margin.numpy().astype(np.float64)
        for s in range(model.shape_count):
            if not colliding[s]:
                continue
            b = int(shape_body[s])
            if shape_type[s] == GeoType.PLANE:
                if b != -1:
                    raise NotImplementedError(
                        f"SolverIPC: the plane shape {s} is on body {b}; put planes on the world (body -1). libuipc's "
                        "half planes do not move."
                    )
                continue
            v, f = _shape_mesh(model, s, solver._shape_segments)
            if len(f) == 0:
                continue  # a height field without a cell
            v = _quat_rotate_np(shape_xform[s, 3:7], v) + shape_xform[s, :3]
            if b == -1:
                scene.add_obstacle(
                    v,
                    f,
                    thickness=margin[s],
                    contact_element=elements.shape[s],
                    subscene_element=_subscene_of_shape(solver, s),
                )
                solver._obstacle_shapes.append(s)
            else:
                meshes[b].append((v, f, s))
    # the shape of every surface vertex of a body, in the order of its parts (the contact rows of CONTACT_F)
    solver._body_vertex_shape = [
        np.concatenate([np.full(len(v), s, np.int32) for v, _, s in meshes[b]]) if meshes[b] else np.zeros(0, np.int32)
        for b in range(model.body_count)
    ]

    mass, first, second = _body_mass_moments(solver, stacklevel=5)
    # kinematic_motion="aim": the kinematic bodies with colliding shapes, which move toward their aims
    solver._aim_bodies = np.zeros(0, np.int64)
    solver._aim_mesh_moments = {}
    # the bodies whose start the caller writes into the state, which the first step checks (_check_start)
    solver._posed_body = np.zeros(model.body_count, bool)
    if model.body_count == 0:
        return mass, first, second
    if solver._aims:
        solver._aim_bodies = np.flatnonzero(solver._kinematic_body & np.array([bool(meshes[b]) for b in meshes]))
        for b in solver._aim_bodies:
            if mass[b] <= 0.0:  # the mass of its collision meshes at 1000 kg/m^3
                solver._aim_mesh_moments[int(b)] = _mesh_moments(solver, int(b), meshes[b])
        mass, first, second = _with_aim_moments(solver, (mass, first, second))
    body_q, body_qd, placed = _start_bodies(solver)
    # a kinematic body, whose pose the caller writes at every step, and a link the model leaves at the identity,
    # which the caller places by writing forward kinematics into the state (docs/solvers/ipc.rst > "State")
    solver._posed_body = solver._kinematic_body | placed
    body_com = model.body_com.numpy().astype(np.float64)
    body_mass = model.body_mass.numpy().astype(np.float64)
    body_gravity = solver._gravity[model.body_world.numpy()]
    solver._body_volume = np.zeros(model.body_count)
    for b in range(model.body_count):
        parts = meshes[b]
        offsets = np.cumsum([0] + [len(v) for v, _, _ in parts])
        xbar = np.concatenate([v for v, _, _ in parts]) if parts else np.zeros((0, 3))
        tris = (
            np.concatenate([f + o for (_, f, _), o in zip(parts, offsets[:-1], strict=True)])
            if parts
            else np.zeros((0, 3), np.int64)
        )
        # every vertex in the element and the subscene of its shape, with the shape's margin as its thickness
        element = np.concatenate([np.full(len(v), elements.shape[s]) for v, _, s in parts]) if parts else 0
        subscene = (
            np.concatenate([np.full(len(v), _subscene_of_shape(solver, s)) for v, _, s in parts])
            if parts
            else _subscene_of_body(solver, b)
        )
        solver._body_row_subscene[b] = np.atleast_1d(subscene)[0]
        thickness = np.concatenate([np.full(len(v), margin[s]) for v, _, s in parts]) if parts else 0.0
        volume = sum(_closed_volume(v, f) for v, f, _ in parts)
        if volume <= 0.0:
            # no closed collision mesh: the volume of water of that mass
            volume = max(body_mass[b], 1.0e-12) / 1000.0
        solver._body_volume[b] = volume
        R = _quat_to_matrix_np(body_q[b, 3:7])
        w = body_qd[b, 3:6]
        W = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
        scene.add_affine_body(
            xbar,
            tris,
            mass=mass[b],
            first_moment=first[b],
            second_moment=second[b],
            volume=volume,
            kappa=_BODY_KAPPA if solver._body_kappa is None else solver._body_kappa,
            p=body_q[b, :3],
            A=R,
            p_dot=body_qd[b, :3] - np.cross(w, R @ body_com[b]),
            A_dot=W @ R,
            fixed=bool(solver._kinematic_body[b]) and b not in solver._aim_bodies,
            thickness=thickness,
            contact_element=element,
            subscene_element=subscene,
            # a kinematic body moving toward its aim falls with nothing: gravity does not move it
            gravity=np.zeros(3) if b in solver._aim_bodies else body_gravity[b],
            anchors=solver._body_anchors.get(b),
        )
    return mass, first, second


def _mesh_moments(solver: SolverIPC, b: int, parts: list) -> tuple[float, np.ndarray, np.ndarray]:
    """The dyadic mass of body ``b``'s collision meshes ``parts`` at 1000 kg/m^3, for a kinematic body of mass 0
    that moves toward its aim (``kinematic_motion="aim"``); ``NotImplementedError`` when they enclose no volume."""
    m, mx, S = 0.0, np.zeros(3), np.zeros((3, 3))
    for v, f, _ in parts:
        pm, pmx, pS, vol = solver._uipc_warp.arrays.mesh_dyadic_mass(v, f, 1000.0)
        sign = -1.0 if vol < 0.0 else 1.0  # a mesh wound inward
        m, mx, S = m + sign * pm, mx + sign * pmx, S + sign * pS
    if not m > 0.0:
        raise NotImplementedError(
            f"SolverIPC: kinematic_motion='aim' moves the kinematic body {b} with its mass; it has none, and its "
            "collision shapes enclose no volume to give it one at 1000 kg/m^3."
        )
    return m, mx, S


def _with_aim_moments(solver: SolverIPC, dyadic: tuple) -> tuple:
    """``dyadic`` with the mass moments of the kinematic bodies that move toward their aims (``kinematic_motion=
    "aim"``): Newton's for a body of positive mass, which needs a positive-definite inertia, else those of its
    collision meshes (``_mesh_moments``)."""
    if not len(solver._aim_bodies):
        return dyadic
    mass, first, second = (a.copy() for a in dyadic)
    inertia = solver.model.body_inertia.numpy().astype(np.float64)
    for b in solver._aim_bodies:
        if int(b) in solver._aim_mesh_moments:
            mass[b], first[b], second[b] = solver._aim_mesh_moments[int(b)]
        elif np.linalg.eigvalsh(0.5 * (inertia[b] + inertia[b].T)).min() <= 0.0:
            raise ValueError(
                f"SolverIPC needs a positive-definite inertia for the kinematic body {b}, which "
                "kinematic_motion='aim' moves with its mass."
            )
    return mass, first, second


def _start_bodies(solver: SolverIPC) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The poses and twists the bodies are built at and the start check tests (:ref:`ipc-state`): ``model.body_q`` and
    ``model.body_qd``, but for a body whose pose there is the identity, where ``ModelBuilder.add_link`` and ``add_body``
    put a body given no transform: it starts where ``newton.eval_fk`` puts it from ``model.joint_q`` and
    ``model.joint_qd``, the pose with which a maximal-coordinate solver starts once the caller has written forward
    kinematics into the state. Also the bodies that forward kinematics put elsewhere than the identity: the links whose
    start the caller places."""
    model = solver.model
    body_q = model.body_q.numpy().astype(np.float64)
    body_qd = model.body_qd.numpy().astype(np.float64)
    identity = np.array([0, 0, 0, 0, 0, 0, 1], np.float32)
    unplaced = np.all(model.body_q.numpy() == identity, axis=1)
    if not unplaced.any() or model.articulation_count == 0:
        return body_q, body_qd, np.zeros(model.body_count, bool)
    fk = SimpleNamespace(body_q=wp.clone(model.body_q), body_qd=wp.clone(model.body_qd))
    eval_fk(model, model.joint_q, model.joint_qd, fk)
    fk_q = fk.body_q.numpy()
    body_q[unplaced] = fk_q.astype(np.float64)[unplaced]
    body_qd[unplaced] = fk.body_qd.numpy().astype(np.float64)[unplaced]
    return body_q, body_qd, unplaced & ~np.all(fk_q == identity, axis=1)


def _engine_dyadic(solver: SolverIPC, dyadic: tuple) -> tuple:
    """The bodies' dyadic masses as the engine holds them: the world's (zero) after Newton's bodies."""
    if solver._world_body < 0:
        return dyadic
    mass, first, second = dyadic
    return np.append(mass, 0.0), np.vstack([first, np.zeros((1, 3))]), np.concatenate([second, np.zeros((1, 3, 3))])


def _collect_planes(solver: SolverIPC) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """The points, normals and shapes of the static planes that collide, each point moved out by the shape's
    margin."""
    model = solver.model
    if model.shape_count == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), []
    shape_type = model.shape_type.numpy()
    shape_body = model.shape_body.numpy()
    colliding = _colliding_shapes(model)
    shape_xform = model.shape_transform.numpy()
    shape_scale = model.shape_scale.numpy()
    margin = model.shape_margin.numpy().astype(np.float64)
    P, N, shapes = [], [], []
    for s in range(model.shape_count):
        if shape_type[s] != int(GeoType.PLANE) or shape_body[s] != -1:
            continue
        if not colliding[s]:
            continue
        if shape_scale[s][0] > 0.0 or shape_scale[s][1] > 0.0:
            warnings.warn(
                f"SolverIPC treats the finite plane shape {s} as an infinite half space.",
                NewtonWarning,
                stacklevel=4,
            )
        xf = shape_xform[s].astype(np.float64)
        n = _quat_rotate_np(xf[3:7], np.array([0.0, 0.0, 1.0]))
        # a half plane has no thickness: its margin moves its surface out along the normal
        P.append(xf[:3] + margin[s] * n)
        N.append(n)
        shapes.append(s)
    return np.asarray(P).reshape(-1, 3), np.asarray(N).reshape(-1, 3), shapes
