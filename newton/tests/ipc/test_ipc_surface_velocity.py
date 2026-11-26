# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC's surface velocity: collision meshes with ``Mesh.enable_surface_velocity``, whose Warp mesh's per-vertex
``velocities`` move the material of their surface, as Newton's contacts take them: scaled by the shape's scale, turned
into the world by the shape's pose at the step's start, the part along each contact's tangent plane.

* A belt: a box set on a static mesh whose velocities are (1, 0, 0) slides at ``mu g`` until it rides at 1 m/s, under
  implicit Euler and BDF2, as a box on a kinematic platform that moves at 1 m/s does; without the property the box
  stays at rest. A vertex the solver welded from several mesh vertices takes the velocity of the one the first
  triangle at its place uses.
* The world velocity: the shape's rotation and scale (a belt turned 90 degrees about z and stretched twice along its
  own x carries the box at 2 m/s along y), and the body's pose (a belt on a kinematic body turned 90 degrees).
* The normal part: Newton's own test with a clean start, two spheres on two belts, one of which moves along its
  normal, take the same steps; a tread's normal part on the ground plane leaves its body as without it.
* A tread on a free body drives the body over the ground plane.
* Captured steps equal eager ones while the velocities are written between steps; steps redone on grown engines
  keep the belt; independent worlds carry their boxes at their own belts' speeds, as alone; AL-IPC refuses a mesh with
  the property, naming it, and a libuipc-warp without the extension is named.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.ipc._fixtures import _TIGHT, _TINY_LISTS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, patch_sys_module, requires

DT = 0.01
G = 9.81
MU = 0.8
D_HAT = 1.0e-3
HALF = 0.1  # the boxes' half extent [m]
GAP = 5.0e-4  # how far above its support a box starts [m], inside d_hat
STEPS = 40


def _plane_mesh(enable=True):
    """A square of 4 m by 4 m in the plane z = 0, two triangles facing +z, with the surface-velocity property."""
    vertices = np.array([[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [2.0, 2.0, 0.0], [-2.0, 2.0, 0.0]], np.float32)
    return newton.Mesh(vertices, np.array([0, 1, 2, 0, 2, 3], np.int32), compute_inertia=False,
                       enable_surface_velocity=enable)  # fmt: skip


def _box(builder, center, material):
    """A free box of ``HALF`` resting ``GAP`` above the plane z = ``center[2]``; returns its body."""
    x, y, z = center
    body = builder.add_body(xform=wp.transform(wp.vec3(x, y, z + HALF + GAP), wp.quat_identity()))
    builder.add_shape_box(body, hx=HALF, hy=HALF, hz=HALF, cfg=material)
    return body


def _belt(device, velocity=(1.0, 0.0, 0.0), enable=True, xform=None, scale=None, kinematic_turn=None, at=(-1.0, 0.5)):
    """A box at ``at`` on a static belt mesh (``_plane_mesh``) whose Warp mesh's velocities are ``velocity``, both of
    friction ``MU``. ``xform`` and ``scale`` place the belt's shape; ``kinematic_turn`` puts the belt on a kinematic body
    turned by that angle about z instead. The box starts and stays away from the belt's diagonal, where edge-edge pairs
    would join its point-triangle pairs. Returns the model, the mesh and the box."""
    material = newton.ModelBuilder.ShapeConfig(mu=MU)
    builder = newton.ModelBuilder()
    mesh = _plane_mesh(enable)
    if kinematic_turn is None:
        builder.add_shape_mesh(body=-1, mesh=mesh, cfg=material, xform=xform, scale=scale)
    else:
        q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(kinematic_turn))
        holder = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), q), is_kinematic=True)
        builder.add_shape_mesh(body=holder, mesh=mesh, cfg=material)
    box = _box(builder, (*at, 0.0), material)
    model = builder.finalize(device=device)
    mesh.mesh.velocities.fill_(wp.vec3(*velocity))
    return model, mesh, box


def _steps(model, solver, steps, before=None, capture=False):
    """The states' ``body_q`` and ``body_qd`` (float64 copies) after every step of ``DT``; ``before(k)`` runs before step
    ``k``. ``capture``: the first step eager, then a graph of two steps replayed (an odd count of steps; ``before`` runs
    before every pair, with the pair's first step)."""
    s0, s1 = model.state(), model.state()
    q, qd = [], []

    def record(state):
        q.append(state.body_q.numpy().astype(np.float64))
        qd.append(state.body_qd.numpy().astype(np.float64))

    if capture:
        assert steps % 2 == 1, steps
        if before is not None:
            before(0)
        solver.step(s0, s1, None, None, DT)
        s0, s1 = s1, s0
        record(s0)
        with wp.ScopedCapture(device=model.device) as cap:
            solver.step(s0, s1, None, None, DT)
            solver.step(s1, s0, None, None, DT)
        for k in range(1, steps, 2):  # the steps k and k + 1
            if before is not None:
                before(k)
            wp.capture_launch(cap.graph)
            record(s1)
            record(s0)
    else:
        for k in range(steps):
            if before is not None:
                before(k)
            solver.step(s0, s1, None, None, DT)
            s0, s1 = s1, s0
            record(s0)
    return np.array(q), np.array(qd)


def _solver(model, **options):
    return newton.solvers.SolverIPC(model, d_hat=D_HAT, **{**_TIGHT, **options})


def _sliding_acceleration(v, toward):
    """The mean acceleration along ``toward`` over the steps that slide (speed between 5% and 90% of the target's)."""
    speed = v @ np.asarray(toward, np.float64) / np.linalg.norm(toward)
    target = np.linalg.norm(toward)
    sliding = np.flatnonzero((speed > 0.05 * target) & (speed < 0.9 * target))
    return float(np.mean(np.diff(speed)[sliding[:-1]]) / DT), len(sliding)


def test_a_belt_carries_a_box(test, device):
    """The box on a belt running at 1 m/s along x slides at ``mu g`` (2%) and then rides at 1 m/s (to 5e-4 m/s),
    under implicit Euler and under BDF2, as on a platform that moves at 1 m/s (``kinematic_motion="aim"``: within
    ``eps_velocity``). Without the property the mesh's velocities are not read (no extension) and the box stays at
    rest."""
    for integrator in ("bdf1", "bdf2"):
        with test.subTest(integrator=integrator):
            model, _, box = _belt(device)
            solver = _solver(model, integrator=integrator)
            _, qd = _steps(model, solver, STEPS)
            v = qd[:, box, :3]
            if integrator == "bdf1":
                belt_v = v
            a, sliding = _sliding_acceleration(v, (1.0, 0.0, 0.0))
            test.assertGreaterEqual(sliding, 8)
            test.assertAlmostEqual(a / (MU * G), 1.0, delta=0.02)
            # BDF2 damps the ringing of the catch-up less: 1e-4 m/s is left after 35 steps
            np.testing.assert_allclose(v[-5:], np.tile([1.0, 0.0, 0.0], (5, 1)), rtol=0.0, atol=5e-4)
            test.assertIsNotNone(solver._features.surface_velocity)
    # the same box on a kinematic platform that moves at 1 m/s
    material = newton.ModelBuilder.ShapeConfig(mu=MU)
    builder = newton.ModelBuilder()
    platform = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, -0.02), wp.quat_identity()), is_kinematic=True)
    builder.add_shape_box(platform, hx=2.0, hy=2.0, hz=0.02, cfg=material)
    carried = _box(builder, (-1.0, 0.5, 0.0), material)
    platform_model = builder.finalize(device=device)
    platform_solver = _solver(platform_model, kinematic_motion="aim")

    def move(k):
        state = platform_states[0]
        q, qd = state.body_q.numpy(), state.body_qd.numpy()
        q[platform, :3] = (DT * k, 0.0, -0.02)
        qd[platform] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        state.body_q.assign(q)
        state.body_qd.assign(qd)

    platform_states = [platform_model.state(), platform_model.state()]
    w = []
    for k in range(STEPS):
        move(k)
        platform_solver.step(platform_states[0], platform_states[1], None, None, DT)
        platform_states.reverse()
        w.append(platform_states[0].body_qd.numpy()[carried, :3].astype(np.float64))
    np.testing.assert_allclose(belt_v[-1], w[-1], rtol=0.0, atol=0.01)
    # without the property: no extension, the box at rest
    model, _, box = _belt(device, enable=False)
    solver = _solver(model)
    _, qd = _steps(model, solver, 10)
    test.assertIsNone(solver._features.surface_velocity)
    test.assertLess(np.abs(qd[:, box, :2]).max(), 1e-6)


def test_a_welded_vertex_takes_its_first_mesh_vertex(test, device):
    """A belt mesh whose six vertices list the square's corners out of order, two of them twice, the copies used by
    the second triangle only: the solver welds the copies, and every vertex of its triangulation takes the velocity of
    the vertex that the first triangle at its place uses, through that triangle corner (``wp.mesh_get_velocity`` takes
    a corner of the index buffer). The copies get a velocity of -7 m/s that a vertex taking them would carry the box
    with; the box rides at 1 m/s."""
    vertices = np.array([[2, 2, 0], [-2, -2, 0], [2, -2, 0], [-2, -2, 0], [-2, 2, 0], [2, 2, 0]], np.float32)
    indices = np.array([1, 2, 0, 3, 5, 4], np.int32)
    mesh = newton.Mesh(vertices, indices, compute_inertia=False, enable_surface_velocity=True)
    material = newton.ModelBuilder.ShapeConfig(mu=MU)
    builder = newton.ModelBuilder()
    shape = builder.add_shape_mesh(body=-1, mesh=mesh, cfg=material)
    box = _box(builder, (-1.0, 0.5, 0.0), material)
    model = builder.finalize(device=device)
    v = np.tile(np.float32([1.0, 0.0, 0.0]), (6, 1))
    v[[3, 5]] = (-7.0, 0.0, 0.0)  # the later copies of (-2, -2) and (2, 2)
    mesh.mesh.velocities.assign(v)
    solver = _solver(model)
    _, qd = _steps(model, solver, STEPS)
    rows = np.flatnonzero(solver._vertex_shape == shape)
    corners = solver._features.surface_velocity.corner.numpy()[
        np.argsort(solver._features.surface_velocity.rows.numpy())
    ]
    np.testing.assert_array_equal(corners, [0, 5, 1, 2])  # the welded corners (-2,-2), (-2,2), (2,-2), (2,2)
    np.testing.assert_array_equal(solver._engine.positions()[rows], vertices[indices[corners]])
    np.testing.assert_allclose(qd[-1, box, :3], (1.0, 0.0, 0.0), rtol=0.0, atol=1e-4)


def test_the_shape_pose_turns_and_scales_the_velocity(test, device):
    """The mesh's velocity (1, 0, 0) in its own coordinates: on a belt shape turned 90 degrees about z and stretched
    twice along its own x, the box rides at (0, 2, 0); on a belt that a kinematic body turns by 90 degrees, at (0, 1, 0)
    (float32, after the box has caught up)."""
    turn = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(np.pi / 2)))
    for label, kwargs, expected, steps in (
        ("shape", {"xform": turn, "scale": (2.0, 1.0, 1.0), "at": (-1.0, -0.5)}, (0.0, 2.0, 0.0), 50),
        ("body", {"kinematic_turn": np.pi / 2, "at": (-1.0, -0.5)}, (0.0, 1.0, 0.0), STEPS),
    ):
        with test.subTest(turned=label):
            model, _, box = _belt(device, **kwargs)
            _, qd = _steps(model, _solver(model), steps)
            np.testing.assert_allclose(qd[-1, box, :3], expected, rtol=0.0, atol=1e-4)


def test_newtons_normal_velocity_test_with_a_clean_start(test, device):
    """Newton's ``test_normal_surface_velocity_does_not_affect_contact``, with the spheres resting ``GAP`` above their
    belts instead of 1 cm inside them, which SolverIPC refuses: the belt whose velocities are (0, 0, 2), along its
    normal, takes its sphere through the steps of the one whose velocities are zero (float32)."""
    material = newton.ModelBuilder.ShapeConfig(mu=MU)
    builder = newton.ModelBuilder()
    meshes, bodies = [], []
    for x in (-1.0, 1.0):
        mesh = _plane_mesh()
        mesh_xform = wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity())
        builder.add_shape_mesh(body=-1, mesh=mesh, cfg=material, xform=mesh_xform, scale=(0.25, 0.25, 1.0))
        body = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.2 + GAP), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.2, cfg=material)
        meshes.append(mesh)
        bodies.append(body)
    model = builder.finalize(device=device)
    meshes[0].mesh.velocities.zero_()
    meshes[1].mesh.velocities.fill_(wp.vec3(0.0, 0.0, 2.0))
    q, qd = _steps(model, _solver(model), 10)
    np.testing.assert_allclose(qd[:, bodies[1]], qd[:, bodies[0]], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(q[:, bodies[1], 2:], q[:, bodies[0], 2:], rtol=0.0, atol=1e-6)


def _tread(device, velocity):
    """A free box of ``HALF`` whose shape is a mesh with the property, resting ``GAP`` above the ground plane, both of
    friction ``MU``; its Warp mesh's four bottom vertices have the velocity ``velocity`` (in the box's coordinates), the
    others none. Returns the model and the box."""
    material = newton.ModelBuilder.ShapeConfig(mu=MU)
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=material)
    mesh = newton.Mesh.create_box(HALF, HALF, HALF, duplicate_vertices=False, compute_normals=False,
                                  compute_uvs=False)  # fmt: skip
    mesh.enable_surface_velocity = True
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, HALF + GAP), wp.quat_identity()))
    builder.add_shape_mesh(body, mesh=mesh, cfg=material)
    model = builder.finalize(device=device)
    v = np.zeros((len(mesh.vertices), 3), np.float32)
    v[np.asarray(mesh.vertices)[:, 2] < 0.0] = velocity
    mesh.mesh.velocities.assign(v)
    return model, body


def test_a_tread_drives_its_body(test, device):
    """A box whose bottom vertices move backwards at 1 m/s over the ground plane slides forwards at ``mu g`` (2%) and
    then rides at 1 m/s. With a part along the ground's normal, (-1, 0, -0.5), it rides at 1 m/s too, and its steps
    differ by less than 2e-3 m/s: the box tilts a little while it accelerates, and the pose that turns the mesh's
    velocity into the world turns that much of the normal part into the plane, as in Newton's pipeline. Kept, the
    normal part would lift the half plane's friction base out of the barrier's range and the box would not move."""
    runs = {}
    for velocity in ((-1.0, 0.0, 0.0), (-1.0, 0.0, -0.5)):
        model, body = _tread(device, velocity)
        runs[velocity] = _steps(model, _solver(model), STEPS)[1][:, body, :3]
    v = runs[(-1.0, 0.0, 0.0)]
    a, sliding = _sliding_acceleration(v, (1.0, 0.0, 0.0))
    test.assertGreaterEqual(sliding, 8)
    test.assertAlmostEqual(a / (MU * G), 1.0, delta=0.02)
    np.testing.assert_allclose(v[-5:], np.tile([1.0, 0.0, 0.0], (5, 1)), rtol=0.0, atol=1e-4)
    np.testing.assert_allclose(runs[(-1.0, 0.0, -0.5)], v, rtol=0.0, atol=2e-3)
    np.testing.assert_allclose(runs[(-1.0, 0.0, -0.5)][-5:], v[-5:], rtol=0.0, atol=1e-4)


def test_captured_steps_equal_eager_ones(test, device):
    """The belt's velocities written in place between steps, 0.5 m/s for eleven steps and 1 m/s after: the captured
    steps give the eager steps' states (float32), and the box follows the change."""

    def runs(capture):
        model, mesh, box = _belt(device)

        def before(k):
            mesh.mesh.velocities.fill_(wp.vec3(0.5 if k < 11 else 1.0, 0.0, 0.0))

        return _steps(model, _solver(model), 31, before=before, capture=capture), box

    (q_e, qd_e), box = runs(False)
    (q_c, qd_c), _ = runs(True)
    np.testing.assert_allclose(q_c, q_e, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(qd_c, qd_e, rtol=0.0, atol=1e-5)
    test.assertAlmostEqual(float(qd_e[10, box, 0]), 0.5, delta=1e-3)
    test.assertAlmostEqual(float(qd_e[-1, box, 0]), 1.0, delta=1e-3)


def test_steps_redone_on_grown_engines_keep_the_belt(test, device):
    """An engine whose contact lists hold one pair: the eager steps overflow and are solved again on engines with
    longer lists, each given the solve's ``SurfaceVelocity`` again. The box rides as on an engine with room
    (float32)."""
    runs = []
    for tiny in (False, True):
        model, _, box = _belt(device)
        solver = _solver(model)
        if tiny:
            solver._make_engine(_TINY_LISTS)
        first = solver._engine
        runs.append(_steps(model, solver, 20)[1][:, box])
        if tiny:
            test.assertIsNot(solver._engine, first)  # grown
    np.testing.assert_allclose(runs[1], runs[0], rtol=0.0, atol=1e-5)
    test.assertGreater(float(runs[0][-1, 0]), 0.9)


def test_independent_worlds_ride_their_own_belts(test, device):
    """Two worlds, each a box on a belt of its own, at 1 and 0.5 m/s: every box rides at its belt's speed, and its
    steps are those of its world alone (float32)."""

    def world(builder, x):
        material = newton.ModelBuilder.ShapeConfig(mu=MU)
        mesh = _plane_mesh()
        builder.add_shape_mesh(body=-1, mesh=mesh, cfg=material,
                               xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()))  # fmt: skip
        return mesh, _box(builder, (x - 1.0, 0.5, 0.0), material)

    speeds = (1.0, 0.5)
    builder = newton.ModelBuilder()
    parts = []
    for k in range(2):
        builder.begin_world()
        parts.append(world(builder, 5.0 * k))
        builder.end_world()
    model = builder.finalize(device=device)
    for (mesh, _), speed in zip(parts, speeds, strict=True):
        mesh.mesh.velocities.fill_(wp.vec3(speed, 0.0, 0.0))
    solver = _solver(model)
    test.assertTrue(solver._independent)
    q, qd = _steps(model, solver, STEPS)
    for k, ((_, box), speed) in enumerate(zip(parts, speeds, strict=True)):
        alone_builder = newton.ModelBuilder()
        mesh, alone_box = world(alone_builder, 5.0 * k)
        alone = alone_builder.finalize(device=device)
        mesh.mesh.velocities.fill_(wp.vec3(speed, 0.0, 0.0))
        q_a, qd_a = _steps(alone, _solver(alone), STEPS)
        np.testing.assert_allclose(qd[:, box], qd_a[:, alone_box], rtol=0.0, atol=1e-5, err_msg=f"world {k}")
        np.testing.assert_allclose(q[:, box], q_a[:, alone_box], rtol=0.0, atol=1e-6, err_msg=f"world {k}")
        test.assertAlmostEqual(float(qd[-1, box, 0]), speed, delta=1e-4)


def test_al_ipc_refuses_a_mesh_with_the_property(test, device):
    """Under ``contact_constitution="al-ipc"`` a tread on a body is refused, naming its shape: libuipc-warp's AL-IPC
    friction has no surface velocity."""
    model, _ = _tread(device, (-1.0, 0.0, 0.0))
    shape = int(np.flatnonzero(model.shape_type.numpy() == int(newton.GeoType.MESH))[0])
    with test.assertRaises(NotImplementedError) as raised:
        newton.solvers.SolverIPC(model, contact_constitution="al-ipc")
    test.assertEqual(
        str(raised.exception),
        f"SolverIPC: shape {shape} is a mesh with enable_surface_velocity=True, which contact_constitution='al-ipc' "
        "does not take: libuipc-warp's AL-IPC friction has no surface velocity. Use contact_constitution='ipc', or "
        "turn the surface velocity off.",
    )


def test_a_libuipc_warp_without_the_extension_is_named(test, device):
    """With a libuipc-warp that has no ``uipc_warp.ext.surface_velocity`` (one before it), a mesh with the property
    raises ``ImportError`` naming the module when the solver is built, where the solver would otherwise fail at its
    first engine."""
    model, _ = _tread(device, (-1.0, 0.0, 0.0))
    with patch_sys_module("uipc_warp.ext.surface_velocity", None), test.assertRaises(ImportError) as raised:
        newton.solvers.SolverIPC(model)
    test.assertIn("needs libuipc-warp's uipc_warp.ext.surface_velocity", str(raised.exception))


@requires("uipc_warp")
class TestSolverIPCSurfaceVelocity(unittest.TestCase):
    """Collision meshes whose surface moves: conveyor belts and treads."""


devices = get_cuda_test_devices()
for name, func in (
    ("test_a_belt_carries_a_box", test_a_belt_carries_a_box),
    ("test_a_welded_vertex_takes_its_first_mesh_vertex", test_a_welded_vertex_takes_its_first_mesh_vertex),
    ("test_the_shape_pose_turns_and_scales_the_velocity", test_the_shape_pose_turns_and_scales_the_velocity),
    ("test_newtons_normal_velocity_test_with_a_clean_start", test_newtons_normal_velocity_test_with_a_clean_start),
    ("test_a_tread_drives_its_body", test_a_tread_drives_its_body),
    ("test_captured_steps_equal_eager_ones", test_captured_steps_equal_eager_ones),
    ("test_steps_redone_on_grown_engines_keep_the_belt", test_steps_redone_on_grown_engines_keep_the_belt),
    ("test_independent_worlds_ride_their_own_belts", test_independent_worlds_ride_their_own_belts),
    ("test_al_ipc_refuses_a_mesh_with_the_property", test_al_ipc_refuses_a_mesh_with_the_property),
    ("test_a_libuipc_warp_without_the_extension_is_named", test_a_libuipc_warp_without_the_extension_is_named),
):
    add_function_test(TestSolverIPCSurfaceVelocity, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
