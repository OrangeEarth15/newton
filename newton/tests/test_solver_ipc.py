# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

UIPC_WARP_AVAILABLE = importlib.util.find_spec("uipc_warp") is not None

# SolverIPC's default barrier activation distance; a resting contact settles just inside it
D_HAT = 0.01


def _soft_cube(device, height=0.05, ground=True, fix_left=False):
    builder = newton.ModelBuilder()  # z-up, gravity -9.81
    builder.add_soft_grid(
        pos=wp.vec3(-0.1, -0.1, height),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=4,
        dim_y=4,
        dim_z=4,
        cell_x=0.05,
        cell_y=0.05,
        cell_z=0.05,
        density=1000.0,
        k_mu=3.4e4,
        k_lambda=1.6e5,
        k_damp=0.0,
        fix_left=fix_left,
    )
    if ground:
        builder.add_ground_plane()
    return builder.finalize(device=device)


def _run(model, solver, frames, dt=0.01, capture=False, field="particle_q"):
    """Trajectory of ``field`` (``"particle_q"``, ``"body_q"`` or a tuple of both) over ``frames`` steps."""
    fields = (field,) if isinstance(field, str) else tuple(field)
    s0 = model.state()
    s1 = model.state()
    traj = {f: [] for f in fields}

    def record(state):
        for f in fields:
            traj[f].append(getattr(state, f).numpy())

    if capture:
        # one eager step loads the state and compiles the kernels, then capture a two-step swap
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        record(s0)
        with wp.ScopedCapture(device=model.device) as cap:
            solver.step(s0, s1, None, None, dt)
            solver.step(s1, s0, None, None, dt)
        for _ in range((frames - 1) // 2):
            wp.capture_launch(cap.graph)
            record(s0)
    else:
        for _ in range(frames):
            solver.step(s0, s1, None, None, dt)
            s0, s1 = s1, s0
            record(s0)
    out = tuple(np.array(traj[f]) for f in fields)
    return out[0] if isinstance(field, str) else out


def test_drop_on_ground(test, device):
    model = _soft_cube(device)
    solver = newton.solvers.SolverIPC(model)
    traj = _run(model, solver, 80)
    test.assertTrue(np.isfinite(traj).all())
    min_z = traj[:, :, 2].min(axis=1)
    test.assertGreater(min_z.min(), 0.0)  # the barrier keeps every vertex above the plane
    test.assertLess(min_z[-1], 0.01)  # and it settles inside the activation distance
    test.assertLess(np.abs(traj[-1] - traj[-2]).max(), 1e-5)
    stats = solver.frame_stats()
    test.assertTrue(stats["converged"])


def test_free_fall_matches_implicit_euler(test, device):
    model = _soft_cube(device, height=1.0, ground=False)
    solver = newton.solvers.SolverIPC(model, newton_velocity_tol=1e-5, linear_tol_rate=1e-12)
    x0 = model.particle_q.numpy().astype(np.float64)
    traj = _run(model, solver, 10)
    k = np.arange(1, 11)
    expected = x0[None, :, 2] - 9.81 * 0.01**2 * (k * (k + 1) / 2.0)[:, None]
    np.testing.assert_allclose(traj[:, :, 2], expected, atol=1e-5)  # float32 output


def test_kinematic_particles_stay(test, device):
    model = _soft_cube(device, height=0.5, ground=False, fix_left=True)
    fixed = model.particle_mass.numpy() == 0.0
    test.assertTrue(fixed.any())
    solver = newton.solvers.SolverIPC(model)
    x0 = model.particle_q.numpy()
    traj = _run(model, solver, 20)
    np.testing.assert_array_equal(traj[-1][fixed], x0[fixed])
    test.assertLess(traj[-1][~fixed, 2].min(), x0[~fixed, 2].min())


def test_soft_bodies_stack_without_intersection(test, device):
    from uipc_warp.validation import SurfaceChecker  # noqa: PLC0415

    builder = newton.ModelBuilder()
    for z, n, cell in ((0.0005, 4, 0.05), (0.26, 2, 0.05)):
        builder.add_soft_grid(
            pos=wp.vec3(-0.5 * n * cell, -0.5 * n * cell, z),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=n,
            dim_y=n,
            dim_z=n,
            cell_x=cell,
            cell_y=cell,
            cell_z=cell,
            density=1000.0,
            k_mu=3.4e4,
            k_lambda=1.6e5,
            k_damp=0.0,
        )
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model)
    checker = SurfaceChecker.from_engine(solver._engine)
    s0, s1 = model.state(), model.state()
    hits = 0
    dmin = np.inf
    for _ in range(60):
        solver.step(s0, s1, None, None, 0.01)
        s0, s1 = s1, s0
        hits += checker.intersections(solver._engine.vm.x)
        dmin = min(dmin, checker.min_distance(solver._engine.vm.x))
    test.assertEqual(hits, 0)
    test.assertGreater(dmin, 0.0)
    z = s0.particle_q.numpy()[:, 2]
    n_lower = 5 * 5 * 5
    test.assertGreater(z[n_lower:].min(), 0.15)  # the upper box rests on the lower one
    test.assertLess(z[n_lower:].min(), 0.21 + 0.01)


def test_friction_on_tilted_gravity(test, device):
    """A box on the ground sticks below the friction angle and slides with g (sin - mu cos) above it."""
    g = 9.81
    for tan_theta, sticks in ((0.3, True), (0.8, False)):
        th = np.arctan(tan_theta)
        builder = newton.ModelBuilder()
        builder.add_soft_grid(
            pos=wp.vec3(-0.05, -0.05, 0.005),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=2,
            dim_y=2,
            dim_z=2,
            cell_x=0.05,
            cell_y=0.05,
            cell_z=0.05,
            density=1000.0,
            k_mu=3.8e5,
            k_lambda=5.8e5,
            k_damp=0.0,
        )
        builder.add_ground_plane()
        model = builder.finalize(device=device)
        model.set_gravity((g * np.sin(th), 0.0, -g * np.cos(th)))
        solver = newton.solvers.SolverIPC(model, friction_mu=0.5, newton_velocity_tol=1e-4, semi_implicit=False)
        s0, s1 = model.state(), model.state()
        cx = []
        for _ in range(60):
            solver.step(s0, s1, None, None, 0.01)
            s0, s1 = s1, s0
            cx.append(s0.particle_q.numpy()[:, 0].astype(np.float64).mean())
        a = np.diff(np.array(cx[30:]), 2).mean() / 0.01**2
        if sticks:
            test.assertLess(abs(cx[-1] - cx[-2]) / 0.01, 0.01)
        else:
            test.assertAlmostEqual(a / (g * (np.sin(th) - 0.5 * np.cos(th))), 1.0, delta=0.01)


def test_graph_capture_matches_eager(test, device):
    # The solver adds with float atomics (SpMV, triplet reduction), as libuipc does, so converged
    # tightly the captured and eager runs agree to rounding rather than bit for bit.
    tight = {"newton_velocity_tol": 1e-7, "linear_tol_rate": 1e-12, "semi_implicit": False}
    model = _soft_cube(device)
    eager = _run(model, newton.solvers.SolverIPC(model, **tight), 21)
    model = _soft_cube(device)
    graph = _run(model, newton.solvers.SolverIPC(model, **tight), 21, capture=True)
    np.testing.assert_allclose(graph, eager[::2], rtol=0.0, atol=1e-6)


def _cloth(builder, n, cell, z, center=(0.0, 0.0), mass=2.0e-4, tri_k=1.0e2, edge_ke=1.0e-2):
    """A horizontal n x n cloth grid centred over ``center`` at height ``z``, without damping."""
    builder.add_cloth_grid(
        pos=wp.vec3(center[0] - 0.5 * n * cell, center[1] - 0.5 * n * cell, z),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=n,
        dim_y=n,
        cell_x=cell,
        cell_y=cell,
        mass=mass,
        tri_ke=tri_k,
        tri_ka=tri_k,
        tri_kd=0.0,
        edge_ke=edge_ke,
        edge_kd=0.0,
    )


def _box_body(builder, center, half, rot=None):
    """A rigid box body (density 1000 kg/m^3) centred at ``center``."""
    b = builder.add_body(xform=wp.transform(wp.vec3(*center), rot or wp.quat_identity()))
    builder.add_shape_box(b, hx=half[0], hy=half[1], hz=half[2])
    return b


def _run_checked(model, solver, frames, dt=0.01):
    """Steps with an edge-triangle intersection count and the minimum distance between bodies after each."""
    from uipc_warp.validation import SurfaceChecker  # noqa: PLC0415

    checker = SurfaceChecker.from_engine(solver._engine)
    s0, s1 = model.state(), model.state()
    hits, dmin = 0, np.inf
    for _ in range(frames):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        hits += checker.intersections(solver._engine.vm.x)
        dmin = min(dmin, checker.min_distance(solver._engine.vm.x))
    return s0, hits, dmin


def test_cloth_drapes_over_sphere(test, device):
    builder = newton.ModelBuilder()
    _cloth(builder, 20, 0.03, 0.40)
    center, radius = np.array([0.0, 0.0, 0.2]), 0.15
    builder.add_shape_sphere(-1, xform=wp.transform(wp.vec3(*center), wp.quat_identity()), radius=radius)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model)
    state, hits, dmin = _run_checked(model, solver, 80)
    test.assertEqual(hits, 0)
    test.assertGreater(dmin, 0.0)
    q = state.particle_q.numpy()
    test.assertTrue(np.isfinite(q).all())
    # outside the sphere, up to the sag of its 24-segment triangulation
    test.assertGreater(np.linalg.norm(q - center, axis=1).min(), radius - 0.005)
    mid = np.argmin(np.linalg.norm(model.particle_q.numpy()[:, :2], axis=1))
    test.assertGreater(q[mid, 2], center[2] + radius)  # the middle rests on the top of the sphere
    test.assertLess(q[mid, 2], center[2] + radius + D_HAT + 1e-3)
    test.assertLess(q[:, 2].min(), center[2])  # and the sheet hangs down around it
    test.assertFalse(solver.frame_stats()["contact_overflow"])


def test_cloth_self_contact_folded_sheet(test, device):
    """A sheet folded over itself: the top layer falls onto the bottom one and never passes through it."""
    nx, ny, ds, r, z0 = 28, 10, 0.025, 0.03, 0.02
    builder = newton.ModelBuilder()
    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=nx,
        dim_y=ny,
        cell_x=ds,
        cell_y=ds,
        mass=1.0e-3,
        tri_ke=1.0e3,
        tri_ka=1.0e3,
        tri_kd=0.0,
        edge_ke=1.0e-3,
        edge_kd=0.0,
    )
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model)  # flat rest shape

    # fold the sheet along x without stretching it: bottom layer, a half circle of radius r, top layer
    flat = model.particle_q.numpy().astype(np.float64)
    s = flat[:, 0]
    a = 0.5 * (nx * ds - np.pi * r)
    theta = np.clip((s - a) / r, 0.0, np.pi)
    x = np.where(s <= a, s, np.where(s >= a + np.pi * r, 2.0 * a + np.pi * r - s, a + r * np.sin(theta)))
    z = z0 + r - r * np.cos(theta)
    folded = np.stack([x, flat[:, 1], z], axis=1)
    from uipc_warp.validation import SurfaceChecker  # noqa: PLC0415

    checker = SurfaceChecker.from_engine(solver._engine)
    s0, s1 = model.state(), model.state()
    s0.particle_q.assign(folded.astype(np.float32))
    hits, pairs = 0, 0
    for _ in range(60):
        solver.step(s0, s1, None, None, 0.01)
        s0, s1 = s1, s0
        hits += checker.intersections(solver._engine.vm.x)
        pairs = max(pairs, solver.frame_stats()["contact_candidates"])
    test.assertEqual(hits, 0)
    test.assertGreater(pairs, 0)  # the only surface is the sheet's: these are self-contact pairs
    # particles are numbered row by row along x; column i of the top layer lies over column nx - i
    q = s0.particle_q.numpy().reshape(ny + 1, nx + 1, 3)
    outer = np.flatnonzero(s.reshape(ny + 1, nx + 1)[0] > a + np.pi * r + 0.5 * a)  # outer half of the top layer
    gap = q[:, outer, 2] - q[:, nx - outer, 2]
    test.assertGreater(gap.min(), 0.0)  # still on top
    test.assertLess(gap.max(), D_HAT + 1e-3)  # and resting on the bottom layer (the fold started 2 r apart)


def test_rigid_box_free_fall_matches_implicit_euler(test, device):
    builder = newton.ModelBuilder()
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5)
    b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), rot))
    # off-centre shape: the centre of mass is not the body origin
    builder.add_shape_box(b, xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()), hx=0.1, hy=0.05, hz=0.08)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, newton_velocity_tol=1e-6, linear_tol_rate=1e-12, semi_implicit=False)
    v0 = np.array([0.3, -0.2, 0.0])
    s0, s1 = model.state(), model.state()
    s0.body_qd.assign(np.array([[*v0, 0.0, 0.0, 0.0]], np.float32))
    q0 = model.body_q.numpy()[0].astype(np.float64)
    dt, g = 0.01, 9.81
    for k in range(1, 11):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        q, qd = s0.body_q.numpy()[0], s0.body_qd.numpy()[0]
        expected = q0[:3] + v0 * k * dt + np.array([0.0, 0.0, -g * dt * dt * k * (k + 1) / 2.0])
        np.testing.assert_allclose(q[:3], expected, atol=1e-5)  # float32 output
        test.assertAlmostEqual(abs(float(np.dot(q[3:], q0[3:]))), 1.0, delta=1e-6)  # no rotation
        np.testing.assert_allclose(qd[:3], v0 + np.array([0.0, 0.0, -g * k * dt]), atol=1e-5)
        np.testing.assert_allclose(qd[3:], 0.0, atol=1e-5)


def test_rigid_stack_rests_without_intersection(test, device):
    builder = newton.ModelBuilder()
    half, gap = 0.1, 0.004
    for i in range(3):
        _box_body(builder, (0.01 * i, 0.0, half + gap + i * (2.0 * half + gap)), (half, half, half))
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model)
    state, hits, dmin = _run_checked(model, solver, 100)
    test.assertEqual(hits, 0)
    test.assertGreater(dmin, 0.0)
    q, qd = state.body_q.numpy(), state.body_qd.numpy()
    test.assertTrue(np.isfinite(q).all())
    z = q[:, 2]
    test.assertGreater(z[0], half)  # the bottom box stands on the ground
    test.assertLess(z[0], half + D_HAT + 1e-3)
    np.testing.assert_array_less(2.0 * half, np.diff(z))  # each box on the one below, without overlap
    np.testing.assert_array_less(np.diff(z), 2.0 * half + D_HAT + 1e-3)
    np.testing.assert_allclose(np.abs(q[:, 6]), 1.0, atol=1e-4)  # upright
    test.assertLess(np.abs(qd).max(), 0.05)  # at rest


def test_cloth_on_rigid_box(test, device):
    builder = newton.ModelBuilder()
    _cloth(builder, 12, 0.025, 0.25, mass=1.0e-3, tri_k=1.0e3, edge_ke=1.0e-3)
    _box_body(builder, (0.0, 0.0, 0.1005), (0.1, 0.1, 0.1))
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model)
    state, hits, dmin = _run_checked(model, solver, 60)
    test.assertEqual(hits, 0)
    test.assertGreater(dmin, 0.0)
    body = state.body_q.numpy()[0]
    np.testing.assert_allclose(body[:2], 0.0, atol=1e-3)  # the box stays where it stood
    test.assertGreater(body[2], 0.1)  # on the ground
    test.assertLess(body[2], 0.1 + D_HAT + 1e-3)
    test.assertAlmostEqual(abs(float(body[6])), 1.0, delta=1e-4)
    top = body[2] + 0.1
    q = state.particle_q.numpy()
    mid = np.argmin(np.linalg.norm(model.particle_q.numpy()[:, :2], axis=1))
    test.assertGreater(q[mid, 2], top)  # the middle rests on the box top
    test.assertLess(q[mid, 2], top + D_HAT + 1e-3)
    test.assertLess(q[:, 2].min(), top - 0.02)  # the edges hang over the sides


def test_kinematic_body_lifts_cloth(test, device):
    """A kinematic box follows the poses written to ``state_in`` and carries the cloth lying on it."""
    from uipc_warp.validation import SurfaceChecker  # noqa: PLC0415

    builder = newton.ModelBuilder()
    _cloth(builder, 6, 0.025, 0.2095, mass=1.0e-3, tri_k=1.0e3, edge_ke=1.0e-3)
    b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()), is_kinematic=True)
    builder.add_shape_box(b, hx=0.1, hy=0.1, hz=0.1)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model)
    checker = SurfaceChecker.from_engine(solver._engine)
    s0, s1 = model.state(), model.state()
    hits = 0
    for k in range(1, 41):
        q = s0.body_q.numpy()
        q[0, 2] = 0.1 + 0.002 * k  # lift 2 mm per step
        s0.body_q.assign(q)
        solver.step(s0, s1, None, None, 0.01)
        s0, s1 = s1, s0
        hits += checker.intersections(solver._engine.vm.x)
    test.assertEqual(hits, 0)
    np.testing.assert_allclose(s0.body_q.numpy()[0, :3], [0.0, 0.0, 0.18], atol=1e-6)
    top = 0.28
    z = s0.particle_q.numpy()[:, 2]
    test.assertGreater(z.min(), top)  # carried up with the box
    test.assertLess(z.max(), top + D_HAT + 1e-3)


def test_graph_capture_matches_eager_cloth_and_rigid(test, device):
    tight = {"newton_velocity_tol": 1e-6, "linear_tol_rate": 1e-12, "semi_implicit": False}

    def scene():
        builder = newton.ModelBuilder()
        _cloth(builder, 8, 0.025, 0.24, mass=1.0e-3, tri_k=1.0e3, edge_ke=1.0e-3)
        _box_body(builder, (0.0, 0.0, 0.1005), (0.1, 0.1, 0.1))
        builder.add_ground_plane()
        return builder.finalize(device=device)

    model = scene()
    eager = _run(model, newton.solvers.SolverIPC(model, **tight), 21, field=("particle_q", "body_q"))
    model = scene()
    graph = _run(model, newton.solvers.SolverIPC(model, **tight), 21, capture=True, field=("particle_q", "body_q"))
    np.testing.assert_allclose(graph[0], eager[0][::2], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(graph[1], eager[1][::2], rtol=0.0, atol=1e-6)


@unittest.skipUnless(UIPC_WARP_AVAILABLE, "Requires libuipc-warp (uipc_warp)")
class TestSolverIPC(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
add_function_test(TestSolverIPC, "test_drop_on_ground", test_drop_on_ground, devices=devices)
add_function_test(
    TestSolverIPC, "test_free_fall_matches_implicit_euler", test_free_fall_matches_implicit_euler, devices=devices
)
add_function_test(TestSolverIPC, "test_kinematic_particles_stay", test_kinematic_particles_stay, devices=devices)
add_function_test(
    TestSolverIPC,
    "test_soft_bodies_stack_without_intersection",
    test_soft_bodies_stack_without_intersection,
    devices=devices,
)
add_function_test(TestSolverIPC, "test_friction_on_tilted_gravity", test_friction_on_tilted_gravity, devices=devices)
add_function_test(TestSolverIPC, "test_graph_capture_matches_eager", test_graph_capture_matches_eager, devices=devices)
add_function_test(TestSolverIPC, "test_cloth_drapes_over_sphere", test_cloth_drapes_over_sphere, devices=devices)
add_function_test(
    TestSolverIPC, "test_cloth_self_contact_folded_sheet", test_cloth_self_contact_folded_sheet, devices=devices
)
add_function_test(
    TestSolverIPC,
    "test_rigid_box_free_fall_matches_implicit_euler",
    test_rigid_box_free_fall_matches_implicit_euler,
    devices=devices,
)
add_function_test(
    TestSolverIPC,
    "test_rigid_stack_rests_without_intersection",
    test_rigid_stack_rests_without_intersection,
    devices=devices,
)
add_function_test(TestSolverIPC, "test_cloth_on_rigid_box", test_cloth_on_rigid_box, devices=devices)
add_function_test(TestSolverIPC, "test_kinematic_body_lifts_cloth", test_kinematic_body_lifts_cloth, devices=devices)
add_function_test(
    TestSolverIPC,
    "test_graph_capture_matches_eager_cloth_and_rigid",
    test_graph_capture_matches_eager_cloth_and_rigid,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
