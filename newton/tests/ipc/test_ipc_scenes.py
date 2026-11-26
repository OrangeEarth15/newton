# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scenes of every kind of content SolverIPC takes: soft bodies, cloth, rigid bodies and their contact."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.ipc._fixtures import D_HAT, _box_body, _cloth, _run, _soft_cube
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires


def test_drop_on_ground(test, device):
    """Drop a soft cube onto the ground and rest it inside d_hat with every vertex above the plane.

    The cube is dropped from 5 cm onto the ground plane. The barrier keeps every vertex above the plane, the cube comes
    to rest inside the activation distance d_hat, and the last step converges.
    """
    model = _soft_cube(device)
    solver = newton.solvers.SolverIPC(model)
    traj = _run(model, solver, 80)
    test.assertTrue(np.isfinite(traj).all())
    min_z = traj[:, :, 2].min(axis=1)
    test.assertGreater(min_z.min(), 0.0)  # the barrier keeps every vertex above the plane
    test.assertLess(min_z[-1], 0.01)  # and it settles inside the activation distance
    test.assertLess(np.abs(traj[-1] - traj[-2]).max(), 1e-5)
    stats = solver.frame_stats()
    test.assertEqual(set(stats), {"newton_iterations", "pcg_iterations", "line_search_trials", "converged"})
    test.assertTrue(stats["converged"])


def test_free_fall_matches_implicit_euler(test, device):
    """Match implicit Euler's heights for a soft cube in free fall, to 1e-5 m over ten steps.

    A soft cube falling freely, without contact, follows implicit Euler's heights ``z_0 - g dt^2 k (k + 1) / 2`` to 1e-5
    m (the float32 output) over ten steps.
    """
    model = _soft_cube(device, height=1.0, ground=False)
    solver = newton.solvers.SolverIPC(model, newton_velocity_tol=1e-5, linear_tol_rate=1e-12)
    x0 = model.particle_q.numpy().astype(np.float64)
    traj = _run(model, solver, 10)
    k = np.arange(1, 11)
    expected = x0[None, :, 2] - 9.81 * 0.01**2 * (k * (k + 1) / 2.0)[:, None]
    np.testing.assert_allclose(traj[:, :, 2], expected, atol=1e-5)  # float32 output


def test_kinematic_particles_stay(test, device):
    """Keep particles of zero mass where they are, bit for bit, while the rest of the cube falls.

    The particles of zero mass are a soft cube's fixed left face.
    """
    model = _soft_cube(device, height=0.5, ground=False, fix_left=True)
    fixed = model.particle_mass.numpy() == 0.0
    test.assertTrue(fixed.any())
    solver = newton.solvers.SolverIPC(model)
    x0 = model.particle_q.numpy()
    traj = _run(model, solver, 20)
    np.testing.assert_array_equal(traj[-1][fixed], x0[fixed])
    test.assertLess(traj[-1][~fixed, 2].min(), x0[~fixed, 2].min())


def test_soft_bodies_stack_without_intersection(test, device):
    """Stack a small soft cube on a larger one without intersection.

    A small soft cube falls onto a larger one on the ground: in no step does an edge cross a triangle or two surfaces
    touch, and the small cube comes to rest on top of the large one.
    """
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
    """Stick a box below the friction angle and slide it with g (sin - mu cos) above it.

    The box rests on the ground under tilted gravity; mu is the mean of the ground's 0.5 and the particles'
    soft_contact_mu, 0.5.
    """
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
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
        model = builder.finalize(device=device)
        model.soft_contact_mu = 0.5
        model.set_gravity((g * np.sin(th), 0.0, -g * np.cos(th)))
        solver = newton.solvers.SolverIPC(model, newton_velocity_tol=1e-4, semi_implicit=False)
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
    """Match eager steps with a soft cube's steps replayed from a CUDA graph, to 1e-6 m.

    Steps of a soft cube dropping onto the ground, captured in a CUDA graph and replayed, follow eager steps to 1e-6 m
    over 21 steps. The solver adds with float atomics (SpMV, triplet reduction), as libuipc does, so converged tightly
    the captured and eager runs agree to rounding rather than bit for bit.
    """
    tight = {"newton_velocity_tol": 1e-7, "linear_tol_rate": 1e-12, "semi_implicit": False}
    model = _soft_cube(device)
    eager = _run(model, newton.solvers.SolverIPC(model, **tight), 21)
    model = _soft_cube(device)
    graph = _run(model, newton.solvers.SolverIPC(model, **tight), 21, capture=True)
    np.testing.assert_allclose(graph, eager[::2], rtol=0.0, atol=1e-6)


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
    """Drape a cloth over a fixed sphere without intersection or contact list overflow.

    A cloth grid falls onto a fixed sphere above the ground: in no step does an edge cross a triangle or two surfaces
    touch, the cloth stays outside the sphere up to the sag of its triangulation, its middle rests on the sphere's top,
    within d_hat + 1 mm, while the sheet hangs down around it, and no contact list overflows.
    """
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
    test.assertFalse(solver._engine.frame_stats()["contact_overflow"])


def test_cloth_self_contact_folded_sheet(test, device):
    """Drop the top layer of a sheet folded over itself onto the bottom one without passing through it."""
    nx, ny, ds, r, z0 = 28, 10, 0.025, 0.03, 0.02
    builder = newton.ModelBuilder()
    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, z0),  # the flat rest shape, apart from the ground as the solver requires
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
    # the sheet's friction with the ground 0.5 (the mean of this and soft_contact_mu): at the 0.75 of Newton's default
    # materials a fold of the top layer stays up 1 cm above the bottom one in about one run in ten
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
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
        pairs = max(pairs, solver._engine.frame_stats()["contact_candidates"])
    test.assertEqual(hits, 0)
    test.assertGreater(pairs, 0)  # the only surface is the sheet's: these are self-contact pairs
    # particles are numbered row by row along x; column i of the top layer lies over column nx - i
    q = s0.particle_q.numpy().reshape(ny + 1, nx + 1, 3)
    outer = np.flatnonzero(s.reshape(ny + 1, nx + 1)[0] > a + np.pi * r + 0.5 * a)  # outer half of the top layer
    gap = q[:, outer, 2] - q[:, nx - outer, 2]
    test.assertGreater(gap.min(), 0.0)  # still on top
    test.assertLess(gap.max(), D_HAT + 1e-3)  # and resting on the bottom layer (the fold started 2 r apart)


def test_rigid_box_free_fall_matches_implicit_euler(test, device):
    """Match implicit Euler's flight for a thrown rigid box whose centre of mass is off its origin.

    The box's shape, and with it the centre of mass, is off the body origin; the box is turned about z and thrown
    sideways. Over ten steps it moves by ``v_0 k dt - g dt^2 k (k + 1) / 2`` and its velocity is ``v_0 - g k dt``, to
    1e-5 (the float32 output), and it does not turn.
    """
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


def test_rigid_spin_slows_as_implicit_euler_turns_an_affine_body(test, device):
    """Slow a freely spinning box by about (omega dt)^2 per step under body_rotation="linear".

    A box spins freely about its axis of symmetry, without gravity, in libuipc's step (body_rotation="linear"). Within a
    step the affine body moves along a straight line in its 12 coordinates and back onto the rotations, so its angular
    velocity drops by about (omega dt)^2 per step, as a rotation projected back after every step gives it: the
    increments alpha_1 = atan(omega dt) and alpha_(n+1) = atan(sin alpha_n / (2 - cos alpha_n)), the angular velocity
    sin(alpha_n) / dt.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    b = _box_body(builder, (0.0, 0.0, 1.0), (0.1, 0.1, 0.05))
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(
        model, newton_velocity_tol=1e-6, linear_tol_rate=1e-12, semi_implicit=False, body_rotation="linear"
    )
    omega, dt, steps = 1.0, 0.01, 8
    s0, s1 = model.state(), model.state()
    s0.body_qd.assign(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, omega]], np.float32))
    alpha, expected, spin = np.arctan(omega * dt), [], []
    for _ in range(steps):
        expected.append(np.sin(alpha) / dt)
        alpha = np.arctan(np.sin(alpha) / (2.0 - np.cos(alpha)))
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        spin.append(float(s0.body_qd.numpy()[b, 5]))
    np.testing.assert_allclose(omega - np.array(spin), omega - np.array(expected), rtol=0.02)
    test.assertAlmostEqual((spin[-2] - spin[-1]) / spin[-2], (omega * dt) ** 2, delta=0.05 * (omega * dt) ** 2)


def test_rigid_stack_rests_without_intersection(test, device):
    """Rest three stacked rigid boxes upright on each other without intersection.

    Three rigid boxes stacked with 4 mm gaps fall onto the ground and onto each other: in no step does an edge cross a
    triangle or two surfaces touch, and after 100 steps every box rests upright on the one below, the bottom one on the
    ground, within d_hat + 1 mm.
    """
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
    """Rest a cloth on a rigid box standing on the ground without intersection.

    A cloth falls onto a rigid box standing on the ground: in no step does an edge cross a triangle or two surfaces
    touch, the box stays upright where it stood on the ground, and the cloth's middle rests on its top, within d_hat + 1
    mm, while the edges hang over the sides.
    """
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
    """Carry the cloth lying on a kinematic box that jumps to the poses written to ``state_in``.

    The box moves with ``kinematic_motion="jump"`` and follows those poses.
    """
    from uipc_warp.validation import SurfaceChecker  # noqa: PLC0415

    builder = newton.ModelBuilder()
    _cloth(builder, 6, 0.025, 0.2095, mass=1.0e-3, tri_k=1.0e3, edge_ke=1.0e-3)
    b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()), is_kinematic=True)
    # friction 0.5 (the mean of this and soft_contact_mu): the solve does not sweep the box's 2 mm jumps, and the
    # cloth, which moves up 2 mm against the box in every step, crumples at the 0.75 of Newton's default materials
    builder.add_shape_box(b, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, kinematic_motion="jump")
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
    """Match eager steps with CUDA graph replays of a cloth falling onto a rigid box, to 1e-6.

    As test_graph_capture_matches_eager, with a cloth falling onto a rigid box: the particles and the body of steps
    replayed from a CUDA graph follow eager steps to 1e-6 over 21 steps.
    """
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


def test_body_stiffness_takes_a_turn_in_one_iteration(test, device):
    """Take a turn of up to 0.1 rad in the first Newton iteration with the default body stiffness.

    By default a body's orthogonality stiffness is libuipc's 1e8 Pa, lowered in a step of length dt to 25 I_max / (V
    dt^2), I_max the body's largest principal moment of inertia, so that the first Newton iteration takes a turn of up
    to 0.1 rad about that moment's axis in full. A 1 m cube that a torque turns by 0.06 rad in one 0.1 s step (Newton's
    body-force test) and a 10 cm cube spinning at 0.6 rad/s over six steps of 1/60 s end where the converged solve does,
    as a rotation projected back after every step ends. At libuipc's stiffness, given as body_kappa, the line search
    cuts the turn short and the next directions pass the displacement test long before it is done (31% and 33% of the
    angular velocity). Eager and captured steps of two sizes take the stiffness of their own dt; a given body_kappa is
    taken at every step.
    """

    def cube(half, torque=0.0, spin=0.0, **kwargs):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi * 0.5)
        _box_body(builder, (1.0, 2.0, 3.0), (half, half, half), rot)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverIPC(model, **kwargs)
        s0, s1 = model.state(), model.state()
        s0.body_qd.assign(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, spin]], np.float32))
        s0.body_f.assign(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, torque]], np.float32))
        return model, solver, s0, s1

    def spin_after(half, dt, steps, torque=0.0, spin=0.0, **kwargs):
        model, solver, s0, s1 = cube(half, torque, spin, **kwargs)
        for _ in range(steps):
            solver.step(s0, s1, None, None, dt)
            s0, s1 = s1, s0
            s0.body_f.assign(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, torque]], np.float32))
        return float(s0.body_qd.numpy()[0, 5]), model, solver

    # one 0.1 s step of 1000 N m on a 1 m cube: a rigid body reaches 0.6 rad/s, an affine one projected back onto the
    # rotations sin(atan(0.06)) / 0.1
    omega, model, solver = spin_after(0.5, 0.1, 1, torque=1000.0)
    inertia = float(model.body_inertia.numpy()[0][2, 2])
    turn = 1000.0 / inertia * 0.1 * 0.1
    test.assertAlmostEqual(turn, 0.06, places=5)
    test.assertAlmostEqual(omega, np.sin(np.arctan(turn)) / 0.1, delta=1e-3)
    given, _, _ = spin_after(0.5, 0.1, 1, torque=1000.0, body_kappa=1.0e8)
    test.assertLess(given, 0.5 * omega)
    # a 10 cm cube at 0.6 rad/s over six steps of 1/60 s, which lose about (omega dt)^2 = 1e-4 each
    omega, _, _ = spin_after(0.05, 1.0 / 60.0, 6, spin=0.6)
    alpha = np.arctan(0.6 / 60.0)
    for _ in range(5):
        alpha = np.arctan(np.sin(alpha) / (2.0 - np.cos(alpha)))
    test.assertAlmostEqual(omega, np.sin(alpha) * 60.0, delta=5e-4)
    given, _, _ = spin_after(0.05, 1.0 / 60.0, 6, spin=0.6, body_kappa=1.0e8)
    test.assertLess(given, 0.5 * omega)

    # the stiffness of each step: eager steps of 1/600 s, and a captured step of 0.1 s replayed between them
    model, solver, s0, s1 = cube(0.5)
    i_max = float(np.linalg.eigvalsh(model.body_inertia.numpy()[0].astype(np.float64)).max())

    def stiffness():
        return float(solver._ortho.kappa.numpy()[list(solver._ortho_body).index(0)])

    def expected(dt):
        return min(1.0e8, 25.0 * i_max / (1.0 * dt * dt))  # a 1 m cube: V = 1 m^3

    solver.step(s0, s1, None, None, 0.1)
    with wp.ScopedCapture(device=device) as capture:
        solver.step(s1, s0, None, None, 0.1)
    for dt in (1.0 / 600.0, 0.1, 1.0 / 600.0):
        if dt == 0.1:
            wp.capture_launch(capture.graph)
        else:
            solver.step(s0, s1, None, None, dt)
        np.testing.assert_allclose(stiffness(), expected(dt), rtol=1e-6)
    test.assertEqual(expected(1.0 / 600.0), 1.0e8)
    test.assertLess(expected(0.1), 1.0e6)
    _, _, solver = spin_after(0.5, 0.1, 1, body_kappa=1.0e7)
    np.testing.assert_array_equal(solver._uipc_warp.arrays.orthogonality(solver._engine).kappa.numpy(), 1.0e7)
    test.assertIsNone(solver._ortho)
    for kappa in (0.0, -1.0e8, float("nan"), float("inf")):
        with test.assertRaisesRegex(ValueError, "body_kappa must be a finite positive stiffness"):
            newton.solvers.SolverIPC(model, body_kappa=kappa)


def test_rigid_body_rotation(test, device):
    """Keep a rigid body's angular momentum in its predicted flight under body_rotation="rigid".

    With body_rotation="rigid" (libuipc-warp's affine_rotation="rigid") a rigid body's prediction is a rigid body's
    flight that keeps its angular momentum, and its twist holds the angular velocity that carries that momentum.

    At converged Newton tolerances a cube spinning about any axis keeps its angular velocity over 200 steps of 0.01 s,
    where libuipc's straight-line step (body_rotation="linear") keeps atan2(sin t, 2 - cos t) of a turn t per step, and
    a box tumbling about no principal axis keeps the angular momentum Newton computes from its state. At the solver's
    default tolerances the Newton loop's displacement test leaves a little of each turn behind, which the twist reports:
    the cube loses less than a tenth of what libuipc's step loses. Newton's cases at the default tolerances: a torque
    and a force on a cube whose centre of mass is off its frame (the combined force-and-torque test) give tau / I and f
    / m per second, also captured; a 0.5 x 1 x 2 m box that a torque starts turning in one 0.1 s step reports the rigid
    0.24 rad/s (the skew part of A_dot A^-1 counts the shear of its soft default stiffness); the undamped hinge of
    test_joint_damping keeps its spin.
    """
    tight = {"newton_velocity_tol": 1e-9, "newton_transrate_tol": 1e-9, "linear_tol_rate": 1e-14, "semi_implicit": False,
             "newton_max_iter": 256}  # fmt: skip

    def body(builder, half, rot=None, com=None):
        b = _box_body(builder, (0.0, 0.0, 1.0), half, rot)
        if com is not None:
            builder.body_com[b] = wp.vec3(*com)
        return b

    def run(model, steps, dt, w=None, wrench=None, captured=False, **kwargs):
        solver = newton.solvers.SolverIPC(model, **kwargs)
        s0, s1 = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        if w is not None:
            s0.body_qd.assign(np.array([[0.0, 0.0, 0.0, *w]], np.float32))
        if wrench is not None:
            s0.body_f.assign(np.array([wrench], np.float32))
            s1.body_f.assign(np.array([wrench], np.float32))
        if captured:  # one eager step, then pairs of steps captured once and replayed
            solver.step(s0, s1, None, None, dt)
            with wp.ScopedCapture(device=device) as capture:
                solver.step(s1, s0, None, None, dt)
                solver.step(s0, s1, None, None, dt)
            for _ in range((steps - 1) // 2):
                wp.capture_launch(capture.graph)
            return s1
        for _ in range(steps):
            solver.step(s0, s1, None, None, dt)
            s0, s1 = s1, s0
        return s0

    # a 20 cm cube spinning at 2.24 rad/s about (0.3, -0.4, 1): 200 steps of 0.01 s
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body(builder, (0.1, 0.1, 0.1))
    model = builder.finalize(device=device)
    w = np.array([0.6, -0.8, 2.0], np.float32)

    def kept(**kwargs):
        return float(np.linalg.norm(run(model, 200, 0.01, w=w, **kwargs).body_qd.numpy()[0, 3:]) / np.linalg.norm(w))

    s = run(model, 200, 0.01, w=w, body_rotation="rigid", **tight)
    np.testing.assert_allclose(s.body_qd.numpy()[0, 3:], w, rtol=0, atol=1e-6 * np.linalg.norm(w))
    t = np.arctan(np.linalg.norm(w) * 0.01)
    for _ in range(199):
        t = np.arctan2(np.sin(t), 2.0 - np.cos(t))
    linear = kept(body_rotation="linear", **tight)
    test.assertAlmostEqual(linear, np.sin(t) / 0.01 / np.linalg.norm(w), delta=1e-4)
    test.assertLess(1.0 - kept(body_rotation="rigid"), 0.1 * (1.0 - kept(body_rotation="linear")))

    # Newton's combined force-and-torque test: 10 N and 10 N m for ten steps of 0.01 s, the centre of mass off the frame
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body(
        builder, (0.1, 0.1, 0.1), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi * 0.5), com=(0.05, -0.03, 0.02)
    )
    model = builder.finalize(device=device)
    inertia = model.body_inertia.numpy()[0].astype(np.float64)
    mass = float(model.body_mass.numpy()[0])
    wrench = [10.0, 0.0, 0.0, 0.0, 0.0, 10.0]
    eager = run(model, 11, 0.01, wrench=wrench, body_rotation="rigid").body_qd.numpy()[0]
    captured = run(model, 11, 0.01, wrench=wrench, captured=True, body_rotation="rigid").body_qd.numpy()[0]
    np.testing.assert_allclose(captured, eager, rtol=0, atol=1e-5)
    test.assertAlmostEqual(eager[5], 10.0 / inertia[2, 2] * 0.11, delta=2e-3 * 10.0 / inertia[2, 2] * 0.11)
    test.assertAlmostEqual(eager[0], 10.0 / mass * 0.11, delta=1e-5)
    np.testing.assert_allclose(eager[[1, 2, 3, 4]], 0.0, atol=1e-4)
    linear = run(model, 10, 0.01, wrench=wrench, body_rotation="linear").body_qd.numpy()[0]
    test.assertLess(linear[5], 0.95 * 10.0 / inertia[2, 2] * 0.1)

    # a 0.3 x 0.2 x 0.1 m box tumbling at 5 rad/s about (1, 2, 3) for 300 steps of 1/60 s
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body(builder, (0.15, 0.1, 0.05))
    model = builder.finalize(device=device)
    inertia = model.body_inertia.numpy()[0].astype(np.float64)
    w = (5.0 * np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)).astype(np.float32)

    def momentum(state):
        q = state.body_q.numpy()[0].astype(np.float64)
        R = np.array(wp.quat_to_matrix(wp.quat(*q[3:7])), np.float64).reshape(3, 3)
        return R @ inertia @ R.T @ state.body_qd.numpy()[0, 3:].astype(np.float64)

    L0 = inertia @ w.astype(np.float64)
    s = run(model, 300, 1.0 / 60.0, w=w, body_rotation="rigid", **tight)
    np.testing.assert_allclose(momentum(s), L0, rtol=0, atol=1e-5 * np.linalg.norm(L0))
    s = run(model, 300, 1.0 / 60.0, w=w, body_rotation="linear", **tight)
    test.assertLess(np.linalg.norm(momentum(s)), 0.9 * np.linalg.norm(L0))

    # Newton's D6 angular case on a free joint: 1000 N m about x on a 0.5 x 1 x 2 m box for one step of 0.1 s
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body(builder, (0.25, 0.5, 1.0))
    model = builder.finalize(device=device)
    s = run(model, 1, 0.1, wrench=[0.0, 0.0, 0.0, 1000.0, 0.0, 0.0], body_rotation="rigid")
    test.assertAlmostEqual(s.body_qd.numpy()[0, 3], 0.24, delta=2e-5)

    # test_joint_damping's undamped hinge: a body of unit mass and inertia on a revolute joint about z at 1 rad/s
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Y)
    link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
    joint = builder.add_joint_revolute(parent=-1, child=link, axis=newton.Axis.Z, limit_lower=-1.0e6, limit_upper=1.0e6)
    builder.add_articulation([joint])
    builder.joint_qd[0] = 1.0
    model = builder.finalize(device=device)
    test.assertAlmostEqual(float(run(model, 8, 0.01, body_rotation="rigid").joint_qd.numpy()[0]), 1.0, delta=1e-5)
    test.assertLess(float(run(model, 8, 0.01, body_rotation="linear").joint_qd.numpy()[0]), 1.0 - 5.0e-4)

    with test.assertRaisesRegex(ValueError, "body_rotation must be one of"):
        newton.solvers.SolverIPC(model, body_rotation="exact")


def test_body_rotation_defaults_to_rigid(test, device):
    """Default body_rotation to "rigid" under implicit Euler and to "linear" under BDF2.

    "linear" is libuipc's step; BDF2 does not take the rigid flight. By default a cube keeps its spin over 50 steps of
    0.01 s at converged tolerances, as with body_rotation="rigid", where libuipc's step keeps 0.976 of it; under BDF2
    the default steps as body_rotation="linear" does; and body_rotation="rigid" with BDF2 is refused, naming what to
    pass instead.
    """
    tight = {"newton_velocity_tol": 1e-9, "newton_transrate_tol": 1e-9, "linear_tol_rate": 1e-14, "semi_implicit": False,
             "newton_max_iter": 256}  # fmt: skip
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    _box_body(builder, (0.0, 0.0, 1.0), (0.1, 0.1, 0.1))
    model = builder.finalize(device=device)
    w = np.array([0.6, -0.8, 2.0], np.float32)
    atol = 1e-6 * np.linalg.norm(w)

    def spin(steps, **kwargs):
        solver = newton.solvers.SolverIPC(model, **kwargs)
        s0, s1 = model.state(), model.state()
        s0.body_qd.assign(np.array([[0.0, 0.0, 0.0, *w]], np.float32))
        for _ in range(steps):
            solver.step(s0, s1, None, None, 0.01)
            s0, s1 = s1, s0
        return s0.body_qd.numpy()[0, 3:]

    default = spin(50, **tight)
    np.testing.assert_allclose(default, w, rtol=0, atol=atol)
    np.testing.assert_allclose(default, spin(50, body_rotation="rigid", **tight), rtol=0, atol=atol)

    bdf2 = spin(10, integrator="bdf2")
    np.testing.assert_allclose(bdf2, spin(10, integrator="bdf2", body_rotation="linear"), rtol=0, atol=atol)

    with test.assertRaisesRegex(
        ValueError, r"body_rotation='rigid' integrates the rigid bodies with implicit Euler.* leave body_rotation unset"
    ):
        newton.solvers.SolverIPC(model, body_rotation="rigid", integrator="bdf2")


@requires("uipc_warp")
class TestSolverIPC(unittest.TestCase):
    """Scenes of every kind of content SolverIPC takes."""


devices = get_cuda_test_devices()
add_function_test(TestSolverIPC, "test_drop_on_ground", test_drop_on_ground, devices=devices)
add_function_test(TestSolverIPC, "test_rigid_body_rotation", test_rigid_body_rotation, devices=devices)
add_function_test(
    TestSolverIPC, "test_body_rotation_defaults_to_rigid", test_body_rotation_defaults_to_rigid, devices=devices
)
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
    "test_rigid_spin_slows_as_implicit_euler_turns_an_affine_body",
    test_rigid_spin_slows_as_implicit_euler_turns_an_affine_body,
    devices=devices,
)
add_function_test(
    TestSolverIPC,
    "test_body_stiffness_takes_a_turn_in_one_iteration",
    test_body_stiffness_takes_a_turn_in_one_iteration,
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
