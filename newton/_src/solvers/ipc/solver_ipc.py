# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Incremental Potential Contact (IPC) solver backed by libuipc-warp."""

from __future__ import annotations

import warnings

import numpy as np
import warp as wp

from ...core.types import Axis, override
from ...geometry import GeoType, Mesh, ParticleFlags, ShapeFlags
from ...sim import BodyFlags, Contacts, Control, JointType, Model, ModelFlags, State
from ..solver import SolverBase

_COLLIDE = int(ShapeFlags.COLLIDE_SHAPES) | int(ShapeFlags.COLLIDE_PARTICLES)


@wp.kernel
def _load_particles(
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    x: wp.array[wp.vec3d],
    v: wp.array[wp.vec3d],
):
    i = wp.tid()
    p = q[i]
    u = qd[i]
    x[i] = wp.vec3d(wp.float64(p[0]), wp.float64(p[1]), wp.float64(p[2]))
    v[i] = wp.vec3d(wp.float64(u[0]), wp.float64(u[1]), wp.float64(u[2]))


@wp.kernel
def _load_kinematic_particles(
    q: wp.array[wp.vec3],
    fixed: wp.array[wp.int32],
    x: wp.array[wp.vec3d],
):
    i = wp.tid()
    if fixed[i] != 0:
        p = q[i]
        x[i] = wp.vec3d(wp.float64(p[0]), wp.float64(p[1]), wp.float64(p[2]))


@wp.kernel
def _store_particles(
    x: wp.array[wp.vec3d],
    v: wp.array[wp.vec3d],
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
):
    i = wp.tid()
    p = x[i]
    u = v[i]
    q[i] = wp.vec3(float(p[0]), float(p[1]), float(p[2]))
    qd[i] = wp.vec3(float(u[0]), float(u[1]), float(u[2]))


@wp.func
def _to64(a: wp.vec3) -> wp.vec3d:
    return wp.vec3d(wp.float64(a[0]), wp.float64(a[1]), wp.float64(a[2]))


@wp.func
def _to32(a: wp.vec3d) -> wp.vec3:
    return wp.vec3(float(a[0]), float(a[1]), float(a[2]))


@wp.func
def _rotation64(q: wp.quat) -> wp.mat33d:
    qd = wp.quatd(wp.float64(q[0]), wp.float64(q[1]), wp.float64(q[2]), wp.float64(q[3]))
    return wp.quat_to_matrix(wp.normalize(qd))


@wp.func
def _nearest_rotation(A: wp.mat33d) -> wp.mat33d:
    """Rotation factor R of the polar decomposition A = R S (S symmetric positive semi-definite)."""
    U = wp.mat33d()
    s = wp.vec3d()
    V = wp.mat33d()
    wp.svd3(A, U, s, V)
    R = U * wp.transpose(V)
    if wp.determinant(R) < wp.float64(0.0):
        # a reflection: flip the direction of the smallest singular value
        k = int(2)
        if s[0] <= s[1] and s[0] <= s[2]:
            k = 0
        elif s[1] <= s[2]:
            k = 1
        D = wp.identity(n=3, dtype=wp.float64)
        D[k, k] = wp.float64(-1.0)
        R = U * D * wp.transpose(V)
    return R


@wp.kernel
def _load_bodies(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    base: wp.array[wp.int32],
    kinematic: wp.array[wp.int32],
    kinematic_only: int,
    x: wp.array[wp.vec3d],
    v: wp.array[wp.vec3d],
):
    """Affine-body state q = (p, rows of A) and its rate from Newton's body pose and COM twist."""
    b = wp.tid()
    if kinematic_only != 0 and kinematic[b] == 0:
        return
    tf = body_q[b]
    R = _rotation64(wp.transform_get_rotation(tf))
    o = base[b]
    x[o] = _to64(wp.transform_get_translation(tf))
    for i in range(3):
        x[o + 1 + i] = wp.vec3d(R[i, 0], R[i, 1], R[i, 2])
    if kinematic_only == 0:
        twist = body_qd[b]
        w = _to64(wp.spatial_bottom(twist))
        # Newton's linear velocity is the COM's; p is the body origin's position
        v[o] = _to64(wp.spatial_top(twist)) - wp.cross(w, R * _to64(body_com[b]))
        A_dot = wp.skew(w) * R
        for i in range(3):
            v[o + 1 + i] = wp.vec3d(A_dot[i, 0], A_dot[i, 1], A_dot[i, 2])


@wp.kernel
def _store_bodies(
    x: wp.array[wp.vec3d],
    v: wp.array[wp.vec3d],
    base: wp.array[wp.int32],
    body_com: wp.array[wp.vec3],
    kinematic: wp.array[wp.int32],
    body_q_in: wp.array[wp.transform],
    body_qd_in: wp.array[wp.spatial_vector],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Newton's body pose and COM twist from the affine state: A projected to its nearest rotation."""
    b = wp.tid()
    if kinematic[b] != 0:
        body_q[b] = body_q_in[b]
        body_qd[b] = body_qd_in[b]
        return
    o = base[b]
    A = wp.matrix_from_rows(x[o + 1], x[o + 2], x[o + 3])
    A_dot = wp.matrix_from_rows(v[o + 1], v[o + 2], v[o + 3])
    R = _nearest_rotation(A)
    c = _to64(body_com[b])
    # keep the centre of mass where the affine map puts it
    x_com = A * c + x[o]
    W = A_dot * wp.inverse(A)
    half = wp.float64(0.5)
    w = wp.vec3d(half * (W[2, 1] - W[1, 2]), half * (W[0, 2] - W[2, 0]), half * (W[1, 0] - W[0, 1]))
    q = wp.quat_from_matrix(R)
    body_q[b] = wp.transform(_to32(x_com - R * c), wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
    body_qd[b] = wp.spatial_vector(_to32(v[o] + A_dot * c), _to32(w))


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
    else:
        raise NotImplementedError(f"SolverIPC does not support shape {s} of type {GeoType(t).name}.")
    return _weld(mesh.vertices, mesh.indices)


def _closed_volume(v: np.ndarray, f: np.ndarray) -> float:
    """Enclosed volume of a closed triangle mesh (divergence theorem), either orientation."""
    if len(f) == 0:
        return 0.0
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


class SolverIPC(SolverBase):
    """Incremental Potential Contact solver for soft bodies, cloth and rigid bodies.

    Wraps the `libuipc-warp <https://github.com/OrangeEarth15/libuipc-warp>`_ engine,
    a Warp implementation of libuipc, through its array interface (no libuipc
    build is needed). Each :meth:`step` is one implicit-Euler (BDF1) time step:
    a Newton iteration with a backtracking line search, a block-Jacobi PCG
    linear solve, barrier contact and continuous collision detection. The
    Newton loop, the PCG loop and the line search run on the device under
    ``wp.capture_while``, so a step never synchronizes with the host and can be
    captured in a CUDA graph.

    Supported model content:

    - particles and tetrahedra. Elasticity is the SNK1 stable Neo-Hookean
      model with Lamé parameters ``tet_materials[:, 0]`` (``k_mu``) and
      ``tet_materials[:, 1]`` (``k_lambda``). ``k_damp`` is ignored.
    - cloth: triangles of :attr:`~newton.Model.tri_indices` that are not on a
      tetrahedral body become libuipc's strain-limiting Baraff-Witkin membrane,
      and bending edges of :attr:`~newton.Model.edge_indices` become
      discrete-shell hinges (see the mapping below).
    - rigid bodies with box, sphere, capsule, cylinder, cone, ellipsoid or mesh
      collision shapes become affine bodies (see below).
    - contact against static plane shapes, each treated as an infinite half
      space, and against other static shapes, which become fixed triangle
      meshes.
    - contact between all of these: point-triangle and edge-edge barriers with
      continuous collision detection. Each connected group of tetrahedra or
      cloth triangles is one body and collides with itself (as finite elements
      do in libuipc), so ``d_hat`` must stay below the spacing of
      non-adjacent surface primitives at rest. A rigid body does not collide
      with itself. Collision groups and filter pairs are not read: every shape
      with :attr:`~newton.ShapeFlags.COLLIDE_SHAPES` or
      :attr:`~newton.ShapeFlags.COLLIDE_PARTICLES` collides with everything.
    - kinematic particles (zero mass or not :attr:`~newton.ParticleFlags.ACTIVE`)
      and kinematic bodies (:attr:`~newton.BodyFlags.KINEMATIC` or zero mass).
      They follow the poses written to ``state_in`` every step; the jump is not
      swept by collision detection.
    - one gravity vector, taken from world 0.
    - lagged friction on every contact, with one coefficient ``friction_mu``.

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
    they are. The damping coefficients ``tri_kd`` and ``edge_kd``, drag, lift
    and ``tri_activations`` are ignored: implicit Euler supplies the only
    damping. Rest shapes (the membrane's rest frames, rest lengths and rest
    angles) are taken from :attr:`~newton.Model.particle_q`, as libuipc takes them
    from the rest geometry; Newton's ``tri_poses`` and ``edge_rest_angle`` are not
    read, so a cloth built in a deformed state rests in that state.

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
    ``body_kappa`` and the volume :math:`V` of the body's collision meshes
    keeps :math:`A` within a small strain of a rotation; this is libuipc's
    near-rigid body, not an exactly rigid one. Gravity acts on :math:`p`; with
    the full dyadic mass this is the exact gravity force on the affine body.
    The body's collision shapes are triangulated (curved shapes with
    ``shape_segments`` segments around) and act as its contact surface.

    After each step the solver writes ``body_q`` and ``body_qd``. :math:`A` is
    projected onto its nearest rotation :math:`R` by the polar decomposition
    :math:`A = R S` (from the SVD :math:`A = U \\Sigma V^T`, :math:`R = U V^T`),
    since a quaternion can only hold a rotation; the residual stretch
    :math:`S - 1` is of the order of the orthogonality strain. The body origin
    is placed so that the centre of mass stays where the affine map puts it,
    :math:`x_{com} = A c + p`; the linear velocity is that point's,
    :math:`\\dot p + \\dot A c`, and the angular velocity is the axial vector of
    the skew part of :math:`\\dot A A^{-1}`. The solver keeps the affine state
    itself between steps, so the projection does not feed back.

    Not supported: joints (models may only contain the free joints that
    :meth:`~newton.ModelBuilder.add_body` creates; ``joint_q`` is not written),
    external forces in ``particle_f`` / ``body_f`` and control inputs,
    separate worlds (everything shares one contact space), height-field and SDF
    shapes and planes attached to bodies. The ``contacts`` argument of
    :meth:`step` is not used: the solver detects its own contacts.

    The solver keeps positions and velocities in float64. The first step
    after construction or :meth:`reset` reads ``state_in``. Later steps
    continue from the internal state, apart from kinematic particles and
    bodies, and write float32 results to ``state_out``.

    This solver requires the optional ``libuipc-warp`` package (``import uipc_warp``).

    Args:
        model: The model to simulate.
        kappa: Barrier stiffness [Pa], libuipc's contact resistance.
        d_hat: Barrier activation distance [m].
        thickness: Contact thickness of every particle [m]; rigid-body and
            static-shape vertices have none, as in libuipc.
        newton_max_iter: Maximum Newton iterations per step.
        newton_velocity_tol: Newton convergence tolerance on ``max |dx| / dt`` [m/s].
        newton_transrate_tol: Newton convergence tolerance on ``max |dA| / dt`` of
            rigid bodies [1/s].
        linear_tol_rate: Relative PCG tolerance on ``r^T M^-1 r``.
        line_search_max_iter: Maximum backtracking trials per Newton iteration.
        semi_implicit: Enable libuipc's semi-implicit early exit after
            ``semi_implicit_k_min`` iterations of full feasible steps.
        semi_implicit_k_min: Newton iterations before the early exit applies.
        friction_mu: Friction coefficient of every contact; 0 disables friction.
        eps_velocity: Friction smoothing velocity [m/s]; below it static friction becomes a stiff spring.
        cloth_strain_rate: Strain-limiting coefficient :math:`\\eta` of the cloth
            membrane (libuipc's default 100); 0 makes the membrane quadratic.
        body_kappa: Orthogonality stiffness :math:`\\kappa` of rigid bodies [Pa]
            (libuipc's affine-body stiffness).
        shape_segments: Segments around a sphere, capsule, cylinder, cone or
            ellipsoid when it is triangulated.

    Note:
        A step ends once the Newton update moves no vertex by more than
        ``newton_velocity_tol * dt``. When a body should gain less velocity
        per step than ``newton_velocity_tol`` (slow sliding against friction,
        for example), lower the tolerance, or the body creeps at about
        ``eps_velocity`` instead.

        With a stiff barrier (large ``kappa``) a resting contact settles just
        inside ``d_hat``, so objects rest up to ``d_hat`` apart; lower ``d_hat``
        for tighter contact.
    """

    def __init__(
        self,
        model: Model,
        *,
        kappa: float = 1.0e9,
        d_hat: float = 0.01,
        thickness: float = 0.0,
        newton_max_iter: int = 1024,
        newton_velocity_tol: float = 0.05,
        newton_transrate_tol: float = 0.1,
        linear_tol_rate: float = 1.0e-3,
        line_search_max_iter: int = 8,
        semi_implicit: bool = True,
        semi_implicit_k_min: int = 6,
        friction_mu: float = 0.5,
        eps_velocity: float = 0.01,
        cloth_strain_rate: float = 100.0,
        body_kappa: float = 1.0e8,
        shape_segments: int = 24,
    ):
        super().__init__(model)
        try:
            import uipc_warp  # noqa: PLC0415
            import uipc_warp.arrays  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "SolverIPC requires the libuipc-warp package: pip install git+https://github.com/OrangeEarth15/libuipc-warp"
            ) from e
        self._uipc_warp = uipc_warp
        self._config = uipc_warp.SimConfig(
            kappa=kappa,
            d_hat=d_hat,
            newton_max_iter=newton_max_iter,
            newton_velocity_tol=newton_velocity_tol,
            newton_transrate_tol=newton_transrate_tol,
            linear_tol_rate=linear_tol_rate,
            line_search_max_iter=line_search_max_iter,
            semi_implicit=semi_implicit,
            semi_implicit_k_min=semi_implicit_k_min,
            friction_mu=friction_mu,
            eps_velocity=eps_velocity,
        )
        self._thickness = float(thickness)
        self._cloth_strain_rate = float(cloth_strain_rate)
        self._body_kappa = float(body_kappa)
        self._shape_segments = int(shape_segments)
        self._engine = None
        self._needs_load = True
        self._build_engine()

    # ------------------------------------------------------------------ setup
    def _build_engine(self) -> None:
        model = self.model
        self._engine = None
        if model.particle_count == 0 and model.body_count == 0:
            return
        if model.joint_count:
            jt = model.joint_type.numpy()
            if np.any(jt != int(JointType.FREE)):
                raise NotImplementedError(
                    "SolverIPC does not simulate joints; only the free joints of stand-alone bodies are allowed."
                )
        gravity = model.gravity.numpy().reshape(-1, 3).astype(np.float64)
        if len(gravity) > 1 and not np.allclose(gravity, gravity[0]):
            raise NotImplementedError("SolverIPC supports a single gravity vector for all worlds.")
        self._config.gravity = tuple(gravity[0])

        scene = self._uipc_warp.arrays.ArrayScene()
        tets = self._add_particles_and_tets(scene)
        self._add_cloth(scene, tets)
        self._add_shapes_and_bodies(scene)
        for P, N in zip(*self._collect_planes(), strict=True):
            scene.add_half_plane(P, N)

        layout = scene.layout()
        self._body_base = None
        if model.body_count:
            self._body_base = wp.array(layout.body_base.astype(np.int32), dtype=wp.int32, device=model.device)
            self._body_kinematic = wp.array(self._kinematic_body.astype(np.int32), dtype=wp.int32, device=model.device)
        self._engine = scene.build(self._config, device=model.device)

    def _add_particles_and_tets(self, scene) -> np.ndarray:
        """Every particle as a finite-element vertex (engine index = particle index), and the tetrahedra."""
        model = self.model
        n = model.particle_count
        if n == 0:
            return np.zeros((0, 4), np.int64)
        mass = model.particle_mass.numpy().astype(np.float64)
        flags = model.particle_flags.numpy()
        scene.add_vertices(
            model.particle_q.numpy().astype(np.float64),
            model.particle_qd.numpy().astype(np.float64),
            mass,
            fixed=(mass <= 0.0) | ((flags & int(ParticleFlags.ACTIVE)) == 0),
            thickness=self._thickness,
        )
        if model.tet_count == 0:
            return np.zeros((0, 4), np.int64)
        tets = model.tet_indices.numpy().reshape(-1, 4).astype(np.int64)
        Dm_inv = model.tet_poses.numpy().astype(np.float64).reshape(-1, 3, 3)
        volume = 1.0 / (6.0 * np.abs(np.linalg.det(Dm_inv)))
        materials = model.tet_materials.numpy().astype(np.float64).reshape(-1, 3)
        scene.add_tetrahedra(tets, Dm_inv, volume, materials[:, 0], materials[:, 1])
        return tets

    def _add_cloth(self, scene, tets: np.ndarray) -> None:
        """Triangles not on a tetrahedral body: Baraff-Witkin membrane and discrete-shell hinges."""
        model = self.model
        if model.tri_count == 0:
            return
        tris = model.tri_indices.numpy().reshape(-1, 3).astype(np.int64)
        on_tet = np.zeros(model.particle_count, bool)
        on_tet[tets.ravel()] = True
        cloth = ~on_tet[tris].all(axis=1)
        if not cloth.any():
            return
        tris = tris[cloth]
        mat = model.tri_materials.numpy().astype(np.float64).reshape(-1, 5)[cloth]
        k_mu, k_lambda = mat[:, 0], mat[:, 1]
        stretch = k_mu + 0.5 * k_lambda
        shear = 0.5 * k_mu

        edges, kappa = np.zeros((0, 2), np.int64), np.zeros(0)
        if model.edge_count:
            x = model.particle_q.numpy().astype(np.float64)
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
            x0, x1, x2, x3 = x[e[:, 0]], x[e[:, 1]], x[e[:, 2]], x[e[:, 3]]
            L0 = np.linalg.norm(x3 - x2, axis=1)
            area = 0.5 * (
                np.linalg.norm(np.cross(x2 - x0, x3 - x0), axis=1) + np.linalg.norm(np.cross(x2 - x1, x3 - x1), axis=1)
            )
            h_bar = area / (3.0 * np.maximum(L0, 1.0e-30))
            edges, kappa = e[:, 2:4], 0.5 * ke * h_bar
        scene.add_shell(tris, stretch, stretch, shear, self._cloth_strain_rate, bending=kappa, bending_edges=edges)

    def _add_shapes_and_bodies(self, scene) -> None:
        """Static non-plane shapes as fixed triangle meshes; every body as an affine body."""
        model = self.model
        meshes: dict[int, list] = {b: [] for b in range(model.body_count)}
        if model.shape_count:
            shape_type = model.shape_type.numpy()
            shape_body = model.shape_body.numpy()
            shape_flags = model.shape_flags.numpy()
            shape_xform = model.shape_transform.numpy().astype(np.float64)
            for s in range(model.shape_count):
                if not shape_flags[s] & _COLLIDE:
                    continue
                b = int(shape_body[s])
                if shape_type[s] == GeoType.PLANE:
                    if b != -1:
                        raise NotImplementedError(f"SolverIPC does not support the plane shape {s} on a body.")
                    continue
                v, f = _shape_mesh(model, s, self._shape_segments)
                v = _quat_rotate_np(shape_xform[s, 3:7], v) + shape_xform[s, :3]
                if b == -1:
                    scene.add_obstacle(v, f)
                else:
                    meshes[b].append((v, f))

        self._kinematic_body = np.zeros(model.body_count, bool)
        if model.body_count == 0:
            return
        body_q = model.body_q.numpy().astype(np.float64)
        body_qd = model.body_qd.numpy().astype(np.float64)
        body_mass = model.body_mass.numpy().astype(np.float64)
        body_com = model.body_com.numpy().astype(np.float64)
        body_inertia = model.body_inertia.numpy().astype(np.float64)
        body_flags = model.body_flags.numpy()
        for b in range(model.body_count):
            parts = meshes[b]
            offsets = np.cumsum([0] + [len(v) for v, _ in parts])
            xbar = np.concatenate([v for v, _ in parts]) if parts else np.zeros((0, 3))
            tris = (
                np.concatenate([f + o for (_, f), o in zip(parts, offsets[:-1], strict=True)])
                if parts
                else np.zeros((0, 3), np.int64)
            )
            volume = sum(_closed_volume(v, f) for v, f in parts)
            m = body_mass[b]
            kinematic = bool(body_flags[b] & int(BodyFlags.KINEMATIC)) or m <= 0.0
            self._kinematic_body[b] = kinematic
            inertia = 0.5 * (body_inertia[b] + body_inertia[b].T)
            if not kinematic and np.linalg.eigvalsh(inertia).min() <= 0.0:
                raise ValueError(f"SolverIPC needs a positive-definite inertia for the dynamic body {b}.")
            mass, first, second = self._uipc_warp.arrays.dyadic_mass(m, body_com[b], inertia)
            # the second moment about the centre of mass must be PSD (the inertia's triangle inequality)
            cov = second - mass * np.outer(body_com[b], body_com[b])
            lam, vec = np.linalg.eigh(cov)
            if lam.min() < 0.0:
                warnings.warn(
                    f"SolverIPC: the inertia of body {b} violates the triangle inequality; clamping its mass spread.",
                    stacklevel=3,
                )
                second = (vec * np.maximum(lam, 0.0)) @ vec.T + mass * np.outer(body_com[b], body_com[b])
            if volume <= 0.0:
                volume = max(m, 1.0e-12) / 1000.0  # no closed collision mesh: the volume of water of that mass
            R = _quat_to_matrix_np(body_q[b, 3:7])
            w = body_qd[b, 3:6]
            W = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
            scene.add_affine_body(
                xbar,
                tris,
                mass=mass,
                first_moment=first,
                second_moment=second,
                volume=volume,
                kappa=self._body_kappa,
                p=body_q[b, :3],
                A=R,
                p_dot=body_qd[b, :3] - np.cross(w, R @ body_com[b]),
                A_dot=W @ R,
                fixed=kinematic,
            )

    def _collect_planes(self) -> tuple[np.ndarray, np.ndarray]:
        model = self.model
        if model.shape_count == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))
        shape_type = model.shape_type.numpy()
        shape_body = model.shape_body.numpy()
        shape_flags = model.shape_flags.numpy()
        shape_xform = model.shape_transform.numpy()
        shape_scale = model.shape_scale.numpy()
        P, N = [], []
        for s in range(model.shape_count):
            if shape_type[s] != int(GeoType.PLANE) or shape_body[s] != -1:
                continue
            if not shape_flags[s] & _COLLIDE:
                continue
            if shape_scale[s][0] > 0.0 or shape_scale[s][1] > 0.0:
                warnings.warn(
                    f"SolverIPC treats the finite plane shape {s} as an infinite half space.",
                    stacklevel=3,
                )
            xf = shape_xform[s].astype(np.float64)
            P.append(xf[:3])
            N.append(_quat_rotate_np(xf[3:7], np.array([0.0, 0.0, 1.0])))
        return np.asarray(P).reshape(-1, 3), np.asarray(N).reshape(-1, 3)

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
        eng = self._engine
        if eng is None:
            return
        model = self.model
        device = model.device
        n = model.particle_count
        nb = model.body_count
        vm = eng.vm
        load = self._needs_load
        if n:
            if load:
                wp.launch(
                    _load_particles,
                    n,
                    inputs=[state_in.particle_q, state_in.particle_qd],
                    outputs=[vm.x, vm.v],
                    device=device,
                )
            else:
                wp.launch(
                    _load_kinematic_particles,
                    n,
                    inputs=[state_in.particle_q, vm.fixed],
                    outputs=[vm.x],
                    device=device,
                )
        if nb and (load or self._kinematic_body.any()):
            wp.launch(
                _load_bodies,
                nb,
                inputs=[
                    state_in.body_q,
                    state_in.body_qd,
                    model.body_com,
                    self._body_base,
                    self._body_kinematic,
                    0 if load else 1,
                ],
                outputs=[vm.x, vm.v],
                device=device,
            )
            self._uipc_warp.arrays.update_body_vertices(eng)  # surface vertices x = A x_bar + p
        self._needs_load = False
        eng.advance(dt)
        if n:
            wp.launch(
                _store_particles,
                n,
                inputs=[vm.x, vm.v],
                outputs=[state_out.particle_q, state_out.particle_qd],
                device=device,
            )
        if nb:
            wp.launch(
                _store_bodies,
                nb,
                inputs=[
                    vm.x,
                    vm.v,
                    self._body_base,
                    model.body_com,
                    self._body_kinematic,
                    state_in.body_q,
                    state_in.body_qd,
                ],
                outputs=[state_out.body_q, state_out.body_qd],
                device=device,
            )

    @override
    def reset(self, state: State, world_mask=None, flags=None) -> None:
        """Read positions and velocities from the next ``state_in`` again."""
        self._needs_load = True

    @override
    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        if flags & (ModelFlags.SHAPE_PROPERTIES | ModelFlags.MODEL_PROPERTIES | ModelFlags.BODY_INERTIAL_PROPERTIES):
            self._build_engine()
            self._needs_load = True

    def frame_stats(self) -> dict:
        """Solver counters of the last step: Newton/PCG iterations, line-search trials, convergence.

        Synchronizes with the device.
        """
        return {} if self._engine is None else self._engine.frame_stats()
