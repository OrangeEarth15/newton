# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

UIPC_WARP_AVAILABLE = importlib.util.find_spec("uipc_warp") is not None


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


def _run(model, solver, frames, dt=0.01, capture=False):
    s0 = model.state()
    s1 = model.state()
    traj = []
    if capture:
        # one eager step loads the state and compiles the kernels, then capture a two-step swap
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        traj.append(s0.particle_q.numpy())
        with wp.ScopedCapture(device=model.device) as cap:
            solver.step(s0, s1, None, None, dt)
            solver.step(s1, s0, None, None, dt)
        for _ in range((frames - 1) // 2):
            wp.capture_launch(cap.graph)
            traj.append(s0.particle_q.numpy())
        return np.array(traj)
    for _ in range(frames):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        traj.append(s0.particle_q.numpy())
    return np.array(traj)


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
    model = _soft_cube(device)
    eager = _run(model, newton.solvers.SolverIPC(model), 21)
    model = _soft_cube(device)
    graph = _run(model, newton.solvers.SolverIPC(model), 21, capture=True)
    np.testing.assert_array_equal(graph, eager[::2])


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


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
