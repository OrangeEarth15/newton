# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC's contact: which pairs collide, and the contact rows it writes.

libuipc's contact tabular from Newton's collision groups, worlds, collide flags, filter pairs and friction
coefficients (``_contact_elements``), checked against ``Model.shape_contact_pairs``; what AL-IPC contact and mesh
surface velocities refuse; and the rows of ``CONTACT_F``, one per contact term of a step with a force, from
libuipc-warp's contact records (``_write_contacts``), which a coupled solver's proxy feedback reads too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from newton import GeoType, Mesh, Model, ModelBuilder, ParticleFlags, ShapeFlags, State
from newton.solvers import SolverBase

from .kernels import _contact_rows
from .scene import _colliding_shapes, _kinematic_bodies, _own_elements, _proxy_bodies, _proxy_particles

if TYPE_CHECKING:
    from .scene import _Worlds
    from .solver_ipc import SolverIPC


# threads per block of the contact rows of CONTACT_F, whose grid strides over the records of the step's contact terms
_CONTACT_ROW_BLOCK = 256


def _compatible_groups(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Newton's collision-group rule for two shapes that take part in shape contact (``shape_contact_pairs.py``): a
    positive group meets itself and every negative group, a negative group every group but itself. A shape of group 0
    takes part in no shape pair."""
    return np.where(a > 0, (a == b) | (b < 0), a != b) & (a != 0) & (b != 0)


def _compatible_worlds(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """A global entity (world -1) meets every world, a local one its own world only."""
    return (a == b) | (a == -1) | (b == -1)


@dataclass
class _ContactElements:
    """libuipc contact elements for a model and the contact model of every pair of them (:ref:`ipc-contact`). Element 0
    is libuipc's default element; it holds no vertex."""

    shape: np.ndarray  # (shape_count,): the element of every shape the solver triangulates, -1 for the others
    particle: np.ndarray  # (particle_count,): the element of every particle
    on_mesh: np.ndarray  # (particle_count,): the particles of cloth and tetrahedra
    enabled: np.ndarray  # (count, count): whether the two elements collide
    mu: np.ndarray  # (count, count): their friction coefficient

    @property
    def count(self) -> int:
        return len(self.enabled)


def _filter_classes(world: np.ndarray, filters: np.ndarray, ns: int) -> np.ndarray:
    """A key for every shape in a filter pair (-1 for the other shapes) that the copies of the shape in replicated
    worlds share: its offset in its world (a global shape: ``-1 - index``) and the offsets of its filter partners,
    which are in its world or global, as Newton compares replicated worlds (``shape_contact_pairs.py``, its
    normalized filters). Two shapes get one key only where their filter partners correspond."""
    local = world >= 0
    start = np.full(max(int(world.max(initial=-1)) + 1, 1), ns, np.int64)
    np.minimum.at(start, world[local], np.flatnonzero(local))
    offset = np.where(local, np.arange(ns) - start[np.maximum(world, 0)], -1 - np.arange(ns))
    partners: dict[int, list] = {}
    for a, b in filters.tolist():
        partners.setdefault(a, []).append(int(offset[b]))
        partners.setdefault(b, []).append(int(offset[a]))
    keys: dict[tuple, int] = {}
    own = np.full(ns, -1, np.int64)
    for shape in sorted(partners):
        own[shape] = keys.setdefault((int(offset[shape]), *sorted(partners[shape])), len(keys))
    return own


def _attachment_keys(model: Model, by_world: bool) -> tuple[np.ndarray, np.ndarray, dict[int, tuple]]:
    """For the particle-body attachments (:ref:`ipc-attachments`): which bodies particles are attached to, a key for
    every body, and for every attached particle the sorted keys of its bodies. A key is the body's index, or with
    ``by_world`` (several worlds, which subscenes keep apart) its offset in its world (a global body: ``-1 - index``),
    so that the copies of a body in replicated worlds share it, as ``_filter_classes`` keys shapes."""
    attached = np.zeros(model.body_count, bool)
    body_key = np.zeros(model.body_count, np.int64)
    if model.attachment_particle_body_count == 0:
        return attached, body_key, {}
    bodies = model.attachment_particle_body_body.numpy().astype(np.int64)
    particles = model.attachment_particle_body_particle.numpy().astype(np.int64)
    key = np.arange(model.body_count, dtype=np.int64)
    if by_world:
        world = model.body_world.numpy().astype(np.int64)
        local = world >= 0
        start = np.full(max(int(world.max(initial=-1)) + 1, 1), model.body_count, np.int64)
        np.minimum.at(start, world[local], np.flatnonzero(local))
        key = np.where(local, key - start[np.maximum(world, 0)], -1 - key)
    attached[bodies] = True
    body_key[bodies] = key[bodies]
    of_particle: dict[int, set] = {}
    for p, b in zip(particles.tolist(), bodies.tolist(), strict=True):
        of_particle.setdefault(p, set()).add(int(key[b]))
    return attached, body_key, {p: tuple(sorted(k)) for p, k in of_particle.items()}


def _contact_elements(model: Model, worlds: _Worlds | None = None, aims: bool = False) -> _ContactElements:
    """The contact elements of a model's shapes and particles and the models of their pairs, checked against
    :attr:`Model.shape_contact_pairs`.

    Shapes that Newton's rules cannot tell apart share an element: their signature is the collision group, the world,
    the two collide flags and the friction coefficient. A shape in a filter pair that would otherwise collide gets an
    element of its own, since the pair excludes two shapes and not two classes of them. The particles take three
    elements: those in no triangle and no tetrahedron, active (``P_free``) or not (``P_off``), and the vertices of cloth
    and tetrahedra (``P_surf``). With ``worlds`` (several worlds) the subscenes keep the worlds apart, so the world
    leaves the signature, and a shape in a filter pair shares its element with its copies in replicated worlds
    (``_filter_classes``). With ``aims`` (``kinematic_motion="aim"``) a shape's motion joins the signature: on a dynamic
    body, on a kinematic body (``_kinematic_bodies``), or static; the pairs of two shapes that are each kinematic or
    static are disabled, and ``Model.shape_contact_pairs`` less those pairs is the set compared. In a coupled solver's
    view with proxies (:ref:`ipc-coupling`) a shape's part joins the signature (``_proxy_roles``): the pairs of a shape
    on a proxy body with another such shape or with a static shape or one on a body of inverse mass 0 are disabled, and
    so are those of proxy particles, which take elements of their own (``Q_free``, ``Q_surf``), with those shapes and
    among themselves; the set compared leaves those shape pairs out too. A particle attached to a body
    (:ref:`ipc-attachments`) does not collide with that body: the shapes of a body that particles are attached to take
    the body's key in their signature (``_attachment_keys``), and the attached particles take elements of their own, one
    for each element they would have and set of bodies they are attached to, which collide with what that element
    collides with but those bodies' shapes. libuipc lets a primitive collide only where every pair of its vertices'
    elements does, so neither do the edges and triangles that hold such a particle. Raises ``NotImplementedError`` when the shape
    pairs the elements enable on different bodies are not exactly that set (a pair array assigned to the model; a
    coupled solver's view assigns it this set). Only the triangles and tetrahedra the solver simulates
    (``_own_elements``) make vertices of cloth and tetrahedra. The engine tests the elements' table and the subscenes'
    table apart, so their sizes are not multiplied; it refuses what it cannot hold itself, naming the sizes.
    """
    ns, n_particles = model.shape_count, model.particle_count
    shape_element = np.full(ns, -1, np.int64)
    classes = 0
    body_attached, body_key, attached_to = _attachment_keys(model, worlds is not None)
    proxy_particle = _proxy_particles(model)
    proxies = bool(proxy_particle.any() or _proxy_bodies(model).any())
    if ns:
        flags = model.shape_flags.numpy()
        group = model.shape_collision_group.numpy().astype(np.int64)
        world = model.shape_world.numpy().astype(np.int64)
        body = model.shape_body.numpy().astype(np.int64)
        shape_mu = model.shape_material_mu.numpy().astype(np.float32)
        hits_shapes = (flags & int(ShapeFlags.COLLIDE_SHAPES)) != 0
        hits_particles = (flags & int(ShapeFlags.COLLIDE_PARTICLES)) != 0
        takes_part = hits_shapes & (group != 0)
        # the filter pairs that remove a pair Newton's other rules let collide (_ShapeContactPairs._active_filters)
        filters = np.asarray(model.shape_collision_filter_pairs_array(), np.int64).reshape(-1, 2)
        fa, fb = filters[:, 0], filters[:, 1]
        effective = (
            takes_part[fa]
            & takes_part[fb]
            & (body[fa] != body[fb])
            & _compatible_worlds(world[fa], world[fb])
            & _compatible_groups(group[fa], group[fb])
        )
        filters = filters[effective]
        if worlds is None:
            own = np.full(ns, -1, np.int64)
            own[filters.reshape(-1)] = filters.reshape(-1)
        else:
            own = _filter_classes(world, filters, ns)
        colliding = np.flatnonzero(_colliding_shapes(model))
        # with subscenes the worlds are kept apart by them: the world does not split the classes
        class_world = world if worlds is None else np.zeros_like(world)
        columns = [group, class_world, hits_shapes, hits_particles, shape_mu.view(np.int32).astype(np.int64), own]
        if aims:  # the shape's motion: 0 on a dynamic body, 1 on a kinematic one, 2 static
            motion = np.where(body < 0, 2, 0)
            motion[body >= 0] = _kinematic_bodies(model)[body[body >= 0]]
            columns.append(motion)
        if proxies:  # the shape's part in a coupled solver's proxy contact
            role = _proxy_roles(model, body)
            columns.append(role)
        if attached_to:  # whether particles are attached to the shape's body, and the body's key
            shape_attached = np.where(body >= 0, body_attached[np.maximum(body, 0)], False)
            shape_key = np.where(shape_attached, body_key[np.maximum(body, 0)], 0)
            columns += [shape_attached.astype(np.int64), shape_key]
        signature = np.stack(columns, axis=1)[colliding]
        _, first, inverse = np.unique(signature, axis=0, return_index=True, return_inverse=True)
        order = np.argsort(first)  # classes numbered in the order of their first shapes
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        shape_class = rank[inverse.reshape(-1)]
        shape_element[colliding] = 1 + shape_class
        classes = len(first)
        rep = colliding[first[order]]  # one shape of each class
    count = 1 + classes + (3 if n_particles else 0) + (2 if proxy_particle.any() else 0)
    enabled = np.zeros((count, count), bool)
    mu = np.zeros((count, count))
    shapes = slice(1, 1 + classes)
    if classes:
        g, w, m = group[rep], class_world[rep], shape_mu[rep].astype(np.float64)
        part = takes_part[rep]
        pairs = (
            part[:, None]
            & part[None, :]
            & _compatible_groups(g[:, None], g[None, :])
            & _compatible_worlds(w[:, None], w[None, :])
        )
        cls = shape_element - 1
        pairs[cls[filters[:, 0]], cls[filters[:, 1]]] = False
        pairs[cls[filters[:, 1]], cls[filters[:, 0]]] = False
        immovable = None
        if aims:  # two shapes that each move only as the caller prescribes, or not at all, do not collide
            moves = motion[rep] == 0
            pairs &= moves[:, None] | moves[None, :]
            immovable = motion != 0
        if proxies:  # a proxy meets neither another proxy nor what is static or of inverse mass 0
            proxy, still = role[rep] == 1, role[rep] == 2
            pairs &= ~(proxy[:, None] & (proxy | still)[None, :]) & ~(still[:, None] & proxy[None, :])
        enabled[shapes, shapes] = pairs
        mu[shapes, shapes] = 0.5 * (m[:, None] + m[None, :])
        _check_shape_pairs(
            model,
            colliding,
            shape_class,
            body[colliding],
            pairs,
            worlds is not None,
            immovable,
            role if proxies else None,
        )
    particle_element = np.zeros(n_particles, np.int64)
    on_mesh = np.zeros(n_particles, bool)
    if n_particles:
        p_free, p_surf, p_off = 1 + classes, 2 + classes, 3 + classes
        own_tris, own_tets, _ = _own_elements(model)
        for cells, own, k in ((model.tri_indices, own_tris, 3), (model.tet_indices, own_tets, 4)):
            if own.any():
                on_mesh[cells.numpy().reshape(-1, k)[own].reshape(-1)] = True
        active = (model.particle_flags.numpy() & int(ParticleFlags.ACTIVE)) != 0
        particle_element = np.where(on_mesh, p_surf, np.where(active, p_free, p_off))
        if proxy_particle.any():  # a coupled solver's proxy particles, by kind
            q_free, q_surf = 4 + classes, 5 + classes
            particle_element = np.where(
                proxy_particle & (particle_element == p_surf),
                q_surf,
                np.where(proxy_particle & (particle_element == p_free), q_free, particle_element),
            )
        # particles meet the shapes that collide with particles, of their own world or global (with subscenes, the
        # subscenes keep the worlds apart)
        for p in (p_free, p_surf):
            if classes:
                enabled[p, shapes] = enabled[shapes, p] = hits_particles[rep]
                mu[p, shapes] = mu[shapes, p] = 0.5 * (model.soft_contact_mu + m)
        # particles are spheres to each other where Newton's particle solvers collide them, while the model has a
        # particle grid (finalize makes one for particles of positive radius; None switches particle contact off);
        # finite elements always collide with themselves and with each other
        spheres = n_particles > 1 and model.particle_grid is not None
        enabled[p_free, p_free] = enabled[p_free, p_surf] = enabled[p_surf, p_free] = spheres
        enabled[p_surf, p_surf] = True
        mu[p_free : p_surf + 1, p_free : p_surf + 1] = model.particle_mu
        if proxy_particle.any():
            # a proxy particle meets the shapes a particle of its kind meets but those a proxy does not, and the
            # entry's own particles as a particle of its kind does; no other proxy
            for q, p in ((q_free, p_free), (q_surf, p_surf)):
                if classes:
                    enabled[q, shapes] = enabled[shapes, q] = hits_particles[rep] & (role[rep] == 0)
                    mu[q, shapes] = mu[shapes, q] = 0.5 * (model.soft_contact_mu + m)
                for other in (p_free, p_surf):
                    enabled[q, other] = enabled[other, q] = enabled[p, other]
                    mu[q, other] = mu[other, q] = model.particle_mu
        if attached_to:
            # an attached particle: an element for each element it would have and set of bodies it is attached to,
            # which collides as that element does but with those bodies' shapes (a particle that collides with nothing
            # keeps its element)
            kinds: dict[tuple, int] = {}
            base_of = []
            for particle, keys in sorted(attached_to.items()):
                base = int(particle_element[particle])
                if not enabled[base].any():
                    continue
                if (base, keys) not in kinds:
                    kinds[(base, keys)] = count + len(base_of)
                    base_of.append(base)
                particle_element[particle] = kinds[(base, keys)]
            if base_of:
                of = np.concatenate([np.arange(count), base_of])
                enabled, mu = enabled[np.ix_(of, of)], mu[np.ix_(of, of)]
                for (_, keys), element in kinds.items():
                    if classes:
                        held = 1 + np.flatnonzero(shape_attached[rep] & np.isin(shape_key[rep], keys))
                        enabled[element, held] = enabled[held, element] = False
    return _ContactElements(shape_element, particle_element, on_mesh, enabled, mu)


def _proxy_roles(model: Model, body: np.ndarray) -> np.ndarray:
    """The part of every shape in a coupled solver's proxy contact (:ref:`ipc-coupling`), from its body ``body``: 1 on a
    proxy body, 2 static or on a body of inverse mass 0, 0 otherwise. A proxy meets none of the shapes of parts 1 and 2,
    as the default ``CouplingInterface.coupling_prepare_proxy_contacts`` drops Newton's contacts of a proxy with another
    proxy, a static shape or a body of inverse mass 0."""
    role = np.where(body < 0, 2, 0)
    on_body = body >= 0
    if on_body.any():
        b = body[on_body]
        inverse_mass = model.body_inv_mass.numpy()[b]
        role[on_body] = np.where(_proxy_bodies(model)[b], 1, np.where(inverse_mass <= 0.0, 2, 0))
    return role


def _enabled_pair_count(shape_class: np.ndarray, shape_body: np.ndarray, enabled: np.ndarray) -> int:
    """The pairs of shapes on different bodies (static shapes on body -1) whose classes ``enabled`` pairs, counted
    from the class sizes, less the pairs of shapes on one body."""
    sizes = np.bincount(shape_class, minlength=len(enabled)).astype(np.int64)
    upper = np.triu(enabled, 1)
    count = int((upper * np.outer(sizes, sizes)).sum() + (np.diag(enabled) * sizes * (sizes - 1) // 2).sum())
    entries, n = np.unique(np.stack([shape_body, shape_class], axis=1), axis=0, return_counts=True)
    count -= int((np.diag(enabled)[entries[:, 1]] * n * (n - 1) // 2).sum())
    starts = np.flatnonzero(np.r_[True, entries[1:, 0] != entries[:-1, 0]])
    for s, e in zip(starts, np.r_[starts[1:], len(entries)], strict=True):
        if e - s > 1:
            c, k = entries[s:e, 1], n[s:e]
            count -= int((upper[np.ix_(c, c)] * np.outer(k, k)).sum())
    return count


def _check_shape_pairs(
    model: Model,
    shapes: np.ndarray,
    shape_class: np.ndarray,
    shape_body: np.ndarray,
    enabled: np.ndarray,
    by_world: bool = False,
    immovable: np.ndarray | None = None,
    proxy_role: np.ndarray | None = None,
) -> None:
    """Raise unless the shape classes and their ``enabled`` pairs give exactly ``Model.shape_contact_pairs``: every
    listed pair enabled, and as many enabled pairs of shapes on different bodies (static shapes on body -1) as listed.
    ``shapes`` are the shapes the solver triangulates, ``shape_class`` and ``shape_body`` theirs. ``by_world``: the
    worlds are subscenes, and only shapes of one world or a global one pair; the count is taken per world, the global
    shapes among themselves and every world's shapes among themselves and with the global ones. ``immovable``: per
    shape, whether it is kinematic or static; the listed pairs of two such shapes are left out of the comparison.
    ``proxy_role``: per shape, its part in a coupled solver's proxy contact (``_proxy_roles``); the listed pairs of a
    proxy shape with another or with a shape of part 2 are left out too."""
    cls = np.full(model.shape_count, -1, np.int64)
    cls[shapes] = shape_class
    listed = model.shape_contact_pairs
    listed = np.zeros((0, 2), np.int64) if listed is None else listed.numpy().reshape(-1, 2).astype(np.int64)
    expected = model.shape_contact_pair_count
    dropped, of = np.zeros(len(listed), bool), []
    if immovable is not None:
        dropped |= immovable[listed[:, 0]] & immovable[listed[:, 1]]
        of.append("of two kinematic or static shapes")
    if proxy_role is not None:
        r0, r1 = proxy_role[listed[:, 0]], proxy_role[listed[:, 1]]
        dropped |= ((r0 == 1) & (r1 > 0)) | ((r1 == 1) & (r0 > 0))
        of.append("of a proxy with another proxy or a static shape")
    if of:
        listed = listed[~dropped]
        expected = len(listed)
    a, b = cls[listed[:, 0]], cls[listed[:, 1]]
    missing = (a < 0) | (b < 0) | ~enabled[np.maximum(a, 0), np.maximum(b, 0)]
    if not by_world:
        count = _enabled_pair_count(shape_class, shape_body, enabled)
    else:
        all_worlds = model.shape_world.numpy().astype(np.int64)
        missing |= ~_compatible_worlds(all_worlds[listed[:, 0]], all_worlds[listed[:, 1]])
        world = all_worlds[shapes]
        glob = world == -1
        count = global_count = _enabled_pair_count(shape_class[glob], shape_body[glob], enabled)
        for w in np.unique(world[~glob]):
            mine = glob | (world == w)
            count += _enabled_pair_count(shape_class[mine], shape_body[mine], enabled) - global_count
    missing = np.flatnonzero(missing)
    if len(missing) or count != expected:
        detail = (
            f"shapes {int(listed[missing[0], 0])} and {int(listed[missing[0], 1])} are listed but not in it"
            if len(missing)
            else f"they give {count} pairs, Model.shape_contact_pairs lists {expected}"
            + (f" that are not {' or '.join(of)}" if of else "")
        )
        raise NotImplementedError(
            "SolverIPC collides the shapes that Newton's collision groups, worlds, collide flags and filter pairs "
            f"let collide, which must be Model.shape_contact_pairs: {detail}. A pair array assigned to the model is "
            "supported only when it is that set, as a coupled solver's view assigns it."
        )


def _check_surface_velocity(model: Model) -> None:
    """Refuse a collision mesh with a surface velocity: Newton's contacts sample it (a conveyor belt), and the
    solver, which finds its own contacts, would ignore it."""
    if model.shape_count == 0:
        return
    meshes = np.flatnonzero(_colliding_shapes(model) & (model.shape_type.numpy() == int(GeoType.MESH)))
    for s in meshes:
        source = model.shape_source[s]
        if isinstance(source, Mesh) and source.enable_surface_velocity:
            raise NotImplementedError(
                f"SolverIPC: shape {int(s)} is a mesh with enable_surface_velocity=True; turn the surface velocity "
                "off. SolverIPC finds its own contacts, which do not apply surface velocities."
            )


def _unread_materials(model: Model) -> list[str]:
    """The contact coefficients SolverIPC does not read, where the model sets them to other than Newton's defaults
    (``ModelBuilder.ShapeConfig``'s): a colliding shape's restitution, a static friction coefficient other than its
    ``shape_material_mu``, and torsional and rolling friction, and the particles' ``soft_contact_restitution``. Each is
    named with the first shape that sets it."""
    out = []
    if model.shape_count:
        colliding = np.flatnonzero(_colliding_shapes(model))
        default = ModelBuilder.ShapeConfig()
        mu = model.shape_material_mu.numpy()
        for name, default_value in (
            ("shape_material_restitution", default.restitution),
            ("shape_material_mu_static", None),  # the dynamic coefficient where none is set
            ("shape_material_mu_torsional", default.mu_torsional),
            ("shape_material_mu_rolling", default.mu_rolling),
        ):
            array = getattr(model, name)
            if array is None:
                continue
            values = array.numpy()
            expected = mu if default_value is None else np.full(len(values), default_value, np.float32)
            other = colliding[values[colliding] != expected[colliding]]
            if len(other):
                s = int(other[0])
                out.append(f"{name} {float(values[s]):g} of shape {s}")
    if model.particle_count and model.soft_contact_restitution != 0.0:
        out.append(f"soft_contact_restitution {float(model.soft_contact_restitution):g}")
    return out


def _check_al_ipc(solver: SolverIPC, elements: _ContactElements) -> None:
    """``NotImplementedError`` for a kinematic body that ``kinematic_motion="aim"`` moves toward its aims, which AL-IPC
    does not offer (its frame tests animated aims with a check of its own, libuipc's ``animation_reach_target``, which
    this path does not run), and for contact that libuipc's AL-IPC does not constrain. Its active set holds the
    vertex / half-plane, point / triangle and edge / edge pairs of surfaces, so a particle on no cloth or
    tetrahedron that collides has no constraint; and the penalty of a pair is the smallest of its vertices'
    mass-scaled penalties, so a pair with a vertex without mass has none (a static mesh shape, a particle of no
    mass, a body of no mass), where libuipc's own obstacles are bodies or vertices with mass."""
    if len(solver._aim_bodies):
        raise NotImplementedError(
            "SolverIPC: kinematic_motion='aim' would move the kinematic body "
            f"{int(solver._aim_bodies[0])} toward its aims; leave kinematic_motion at its default (None) or pass "
            "'jump', under which contact_constitution='al-ipc' sweeps the moves of kinematic bodies. AL-IPC tests "
            "animated aims with a check of its own, which SolverIPC does not run."
        )
    model = solver.model
    colliding = elements.enabled[elements.particle].any(axis=1) if model.particle_count else np.zeros(0, bool)
    massless = []
    if np.any(colliding & ~elements.on_mesh):
        massless.append("particles on no cloth or tetrahedron, which are points of no surface")
    if np.any(colliding & elements.on_mesh & (model.particle_mass.numpy() <= 0.0)):
        massless.append("cloth or tetrahedron particles without mass")
    if solver._obstacle_shapes:
        massless.append(f"the static shapes {sorted(solver._obstacle_shapes)}, which have no mass")
    if model.shape_count and model.body_count:
        shape_body = model.shape_body.numpy()[_colliding_shapes(model)]
        bodies = np.unique(shape_body[shape_body >= 0])
        light = bodies[solver._dyadic[0][bodies] <= 0.0]
        if len(light):
            massless.append(f"the bodies {light.tolist()} without mass")
    if massless:
        raise NotImplementedError(
            "SolverIPC: contact_constitution='al-ipc' constrains contact between surfaces of vertices with mass "
            "(libuipc's AL-IPC scales the penalty of a pair by the smallest mass of its vertices); this model "
            "collides " + "; ".join(massless) + ". Static planes are taken."
        )


def _make_contact_export(solver: SolverIPC) -> None:
    """libuipc-warp's ``ContactForces`` of the engine, at the request and whenever the engine is built again, outside
    any capture: once a container requested ``CONTACT_F`` (:ref:`ipc-contact-output`), with what the contact rows read
    besides it, for a model whose contact holds a shape (one without has no rows); and for a coupled solver's proxies,
    whose feedback is the force of the contact terms on them (:ref:`ipc-coupling`). None otherwise."""
    solver._contact_export = None
    solver._export_step = -1  # the step whose records the export holds
    if solver._engine is None:
        return
    rows = solver._contact_output and len(solver._contact_shapes) > 0
    if not (rows or solver._proxies):
        return
    device = solver.model.device
    solver._contact_export = solver._uipc_warp.arrays.ContactForces(solver._engine)
    records = solver._contact_export.capacity
    blocks = max(1, min(-(-records // _CONTACT_ROW_BLOCK), 4 * device.sm_count))
    solver._contact_threads = blocks * _CONTACT_ROW_BLOCK
    if not rows:
        return
    solver._contact_vertex_shape = wp.array(solver._vertex_shape, dtype=wp.int32, device=device)
    # the half planes' shapes, points and normals, which only the rows of a vertex / half-plane term read
    P, N, shapes = solver._planes
    solver._contact_plane_shape = solver._contact_plane_P = solver._contact_plane_N = None
    if len(shapes):
        solver._contact_plane_shape = wp.array(np.asarray(shapes, np.int32), dtype=wp.int32, device=device)
        solver._contact_plane_P = wp.array(np.asarray(P, np.float64).reshape(-1, 3), dtype=wp.vec3d, device=device)
        solver._contact_plane_N = wp.array(np.asarray(N, np.float64).reshape(-1, 3), dtype=wp.vec3d, device=device)


def _write_contacts(solver: SolverIPC, observables: SolverBase.Observables | None, state_out: State | None) -> None:
    """With ``CONTACT_F`` requested: the solver's contact terms of the step into the step's ``Contacts``, one row per
    term with a force, and their forces into ``contact_f`` (:ref:`ipc-contact-output`), on the device. The counts and
    the forces are zeroed first; a step that failed has no rows."""
    if observables is None or not observables.is_requested(SolverBase.ObservableKind.CONTACT_F):
        return
    contacts, contact_f = observables.contacts, observables.contact_f
    contacts.rigid_contact_count.zero_()
    contacts.soft_contact_count.zero_()
    contact_f.zero_()
    if solver._contact_export is None or state_out is None or len(solver._contact_shapes) == 0:
        return
    export = _contact_records(solver)
    model = solver.model
    # only the rows of a shape on a body read the bodies' poses and centres of mass
    body_q = state_out.body_q if model.body_count and state_out.body_q is not None else None
    body_com = model.body_com if model.body_count else None
    wp.launch(
        _contact_rows,
        solver._contact_threads,
        inputs=[
            export.vertices,
            export.forces,
            export.counts,
            export.segment_offsets,
            solver._contact_threads,
            solver._engine.vm.x,
            model.particle_count,
            solver._contact_vertex_shape,
            solver._contact_plane_shape,
            solver._contact_plane_P,
            solver._contact_plane_N,
            model.shape_body,
            model.shape_margin,
            body_q,
            body_com,
            contacts.rigid_contact_max,
            contacts.soft_contact_max,
        ],
        outputs=[
            contacts.rigid_contact_count,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            contacts.rigid_contact_point0,
            contacts.rigid_contact_point1,
            contacts.rigid_contact_normal,
            contacts.rigid_contact_margin0,
            contacts.rigid_contact_margin1,
            contacts.rigid_contact_offset0,
            contacts.rigid_contact_offset1,
            contacts.rigid_contact_surface_velocity,
            contacts.soft_contact_count,
            contacts.soft_contact_shape,
            contacts.soft_contact_indices,
            contacts.soft_contact_barycentric,
            contacts.soft_contact_particle,
            contacts.soft_contact_body_pos,
            contacts.soft_contact_body_vel,
            contacts.soft_contact_normal,
            contacts.soft_contact_rigid_indices,
            contact_f,
        ],
        block_dim=_CONTACT_ROW_BLOCK,
        device=model.device,
    )


def _contact_records(solver: SolverIPC):
    """libuipc-warp's ``ContactForces`` with the records of the last step: its ``update()`` enqueued once per step,
    for the contact rows and the proxies' feedback alike."""
    if solver._export_step != solver._steps:
        solver._contact_export.update()
        solver._export_step = solver._steps
    return solver._contact_export
