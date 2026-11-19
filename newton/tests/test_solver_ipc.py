# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires

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


# ---------------------------------------------------------------------------- inputs the solver rejects


def _rejects(test, builder, pattern, device):
    """``SolverIPC`` raises ``NotImplementedError`` matching ``pattern`` for the model of ``builder``."""
    model = builder.finalize(device=device)
    with test.assertRaisesRegex(NotImplementedError, pattern):
        newton.solvers.SolverIPC(model)


def _two_particles():
    builder = newton.ModelBuilder()
    builder.add_particle(wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.0), 1.0)
    builder.add_particle(wp.vec3(0.1, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.0), 1.0)
    return builder


def test_rejects_springs(test, device):
    """Reject a model with springs, which the solver would not simulate."""
    builder = _two_particles()
    builder.add_spring(0, 1, 1.0e3, 0.0, 0.0)
    _rejects(test, builder, "springs", device)


def test_rejects_attachments(test, device):
    """Reject particle-body attachments."""
    builder = _two_particles()
    b = _box_body(builder, (0.0, 0.0, 2.0), (0.1, 0.1, 0.1))
    builder.add_attachment_particle_body(0, b)
    _rejects(test, builder, "attachments", device)


def _hinged_pair(builder):
    """Two boxes hinged to the world, in one articulation: the joints of a mimic relation."""
    joints = []
    for i in range(2):
        b = builder.add_link(xform=wp.transform(wp.vec3(0.5 * i, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_box(b, hx=0.1, hy=0.1, hz=0.1)
        joints.append(builder.add_joint_revolute(-1, b, axis=newton.Axis.Y))
    builder.add_articulation(joints)
    return joints


def test_rejects_mimic_constraints(test, device):
    """Reject mimic relations, before the revolute joints they need."""
    builder = newton.ModelBuilder()
    j0, j1 = _hinged_pair(builder)
    builder.set_joint_mimic(j1, j0)
    _rejects(test, builder, "mimic", device)


def test_rejects_equality_constraints(test, device):
    """Reject an enabled MuJoCo equality row instead of letting the body fall through it."""
    builder = newton.ModelBuilder()
    body = _box_body(builder, (0.0, 0.0, 1.0), (0.1, 0.1, 0.1))
    builder.add_custom_values(
        **{
            "mujoco:equality_constraint_type": 0,
            "mujoco:equality_constraint_objtype": 1,
            "mujoco:equality_constraint_body1": body,
            "mujoco:equality_constraint_body2": -1,
            "mujoco:equality_constraint_anchor": wp.vec3(0.0, 0.0, 1.0),
            "mujoco:equality_constraint_enabled": True,
        }
    )
    _rejects(test, builder, "equality", device)


def test_rejects_tendons(test, device):
    """Reject a tendon between two bodies."""
    builder = newton.ModelBuilder()
    a = _box_body(builder, (0.0, 0.0, 1.0), (0.1, 0.1, 0.1))
    b = _box_body(builder, (0.5, 0.0, 1.0), (0.1, 0.1, 0.1))
    builder.add_tendon([newton.TendonGuide(body=a), newton.TendonGuide(body=b)])
    _rejects(test, builder, "tendons", device)


def test_rejects_muscles(test, device):
    """Reject a legacy muscle between two bodies."""
    builder = newton.ModelBuilder()
    a = _box_body(builder, (0.0, 0.0, 1.0), (0.1, 0.1, 0.1))
    b = _box_body(builder, (0.5, 0.0, 1.0), (0.1, 0.1, 0.1))
    with test.assertWarns(newton.NewtonDeprecationWarning):
        builder.add_muscle([a, b], [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)], 1.0, 0.5, 0.1, 1.0, 0.0)
    _rejects(test, builder, "muscles", device)


def test_rejects_activations(test, device):
    """Reject non-zero triangle and tetrahedron activations, which the solver does not read."""
    for field in ("tri_activations", "tet_activations"):
        with test.subTest(field=field):
            builder = newton.ModelBuilder()
            if field == "tri_activations":
                _cloth(builder, 2, 0.05, 1.0)
            else:
                builder.add_soft_grid(
                    pos=wp.vec3(0.0, 0.0, 1.0),
                    rot=wp.quat_identity(),
                    vel=wp.vec3(0.0, 0.0, 0.0),
                    dim_x=1,
                    dim_y=1,
                    dim_z=1,
                    cell_x=0.1,
                    cell_y=0.1,
                    cell_z=0.1,
                    density=1000.0,
                    k_mu=1.0e4,
                    k_lambda=1.0e4,
                    k_damp=0.0,
                )
            getattr(builder, field)[0] = 0.5
            _rejects(test, builder, "activations", device)


def test_rejects_joints_and_drives(test, device):
    """Reject joints other than free joints, and drive targets on free joints."""
    builder = newton.ModelBuilder()
    _hinged_pair(builder)
    _rejects(test, builder, "joints", device)
    builder = newton.ModelBuilder()
    _box_body(builder, (0.0, 0.0, 1.0), (0.1, 0.1, 0.1))
    builder.joint_target_mode[0] = int(newton.JointTargetMode.POSITION)
    _rejects(test, builder, "POSITION", device)


def test_rejects_gradients(test, device):
    """Reject a model finalized with gradients, as SolverSteel does."""
    builder = _two_particles()
    model = builder.finalize(device=device, requires_grad=True)
    with test.assertRaisesRegex(NotImplementedError, "gradients"):
        newton.solvers.SolverIPC(model)


def test_rejects_worlds(test, device):
    """Reject content in two worlds: they would collide with each other and share one Newton solve."""
    builder = newton.ModelBuilder()
    for _ in range(2):
        builder.begin_world()
        _box_body(builder, (0.0, 0.0, 1.0), (0.1, 0.1, 0.1))
        builder.end_world()
    _rejects(test, builder, "one world", device)


def test_rejects_contact_filters(test, device):
    """Reject collision groups, filter pairs and shapes that collide with shapes only: every shape collides."""
    for case in ("group", "filter pair", "shapes only"):
        with test.subTest(case=case):
            builder = newton.ModelBuilder()
            a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
            b = builder.add_body(xform=wp.transform(wp.vec3(0.5, 0.0, 1.0), wp.quat_identity()))
            cfg = newton.ModelBuilder.ShapeConfig()
            if case == "group":
                cfg.collision_group = 2
            sa = builder.add_shape_box(a, hx=0.1, hy=0.1, hz=0.1)
            sb = builder.add_shape_box(b, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
            if case == "filter pair":
                builder.add_shape_collision_filter_pair(sa, sb)
            pattern = "filter contacts"
            if case == "shapes only":
                cfg.has_particle_collision = False
                builder.add_shape_box(
                    b,
                    xform=wp.transform(wp.vec3(0.0, 0.0, 0.2), wp.quat_identity()),
                    hx=0.05,
                    hy=0.05,
                    hz=0.05,
                    cfg=cfg,
                )
                builder.add_particle(wp.vec3(0.0, 0.0, 2.0), wp.vec3(0.0, 0.0, 0.0), 1.0)
                pattern = "COLLIDE_PARTICLES"
            _rejects(test, builder, pattern, device)


def test_particle_only_shapes_without_particles(test, device):
    """In a model without particles a shape that collides with particles only collides with nothing: two such boxes
    start inside each other and fall freely through each other and through the ground. With a particle in the
    model the same shapes filter contacts, which the solver refuses."""
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=False)  # keeps COLLIDE_PARTICLES
    for _ in range(2):
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.15), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model)
    s0, s1 = model.state(), model.state()
    dt, steps = 0.01, 20
    for _ in range(steps):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
    # implicit Euler from rest: z_n = z_0 - g dt^2 n (n + 1) / 2, here 5.6 cm below the ground
    expected = 0.15 - 9.81 * dt * dt * steps * (steps + 1) / 2
    np.testing.assert_allclose(s0.body_q.numpy()[:, 2], expected, atol=1.0e-4)

    builder.add_particle(wp.vec3(0.0, 0.0, 2.0), wp.vec3(0.0, 0.0, 0.0), 1.0)
    _rejects(test, builder, "filter contacts", device)


def _belt(device, enable_surface_velocity):
    """A box above a static two-triangle mesh, a conveyor belt when the mesh has a surface velocity."""
    vertices = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]], np.float32)
    belt = newton.Mesh(
        vertices,
        np.array([0, 1, 2, 0, 2, 3], np.int32),
        compute_inertia=False,
        enable_surface_velocity=enable_surface_velocity,
    )
    builder = newton.ModelBuilder()
    builder.add_shape_mesh(body=-1, mesh=belt)
    _box_body(builder, (0.0, 0.0, 0.205), (0.2, 0.2, 0.2))
    return builder.finalize(device=device)


def test_rejects_surface_velocity(test, device):
    """Reject a collision mesh with a surface velocity, which the solver's own contacts would ignore."""
    with test.assertRaisesRegex(NotImplementedError, "surface velocit"):
        newton.solvers.SolverIPC(_belt(device, True))
    newton.solvers.SolverIPC(_belt(device, False))


def _open_plane(builder, z=0.0):
    """A static square of two triangles, 2 m wide, at height ``z``: an open mesh."""
    vertices = np.array([[-1.0, -1.0, z], [1.0, -1.0, z], [1.0, 1.0, z], [-1.0, 1.0, z]], np.float32)
    plane = newton.Mesh(vertices, np.array([0, 1, 2, 0, 2, 3], np.int32), compute_inertia=False)
    builder.add_shape_mesh(body=-1, mesh=plane)


def test_rejects_intersecting_start(test, device):
    """Reject the surfaces of different bodies that cross at the start without a pair inside the contact thickness,
    which a step would not notice; accept the crossings that contact ignores, and flat sheets."""
    from uipc_warp.validation import SurfaceChecker  # noqa: PLC0415

    with test.subTest(case="box through an open plane"):
        builder = newton.ModelBuilder()
        _open_plane(builder)
        _box_body(builder, (0.0, 0.0, 0.19), (0.2, 0.2, 0.2))  # 1 cm through the plane
        with test.assertRaisesRegex(ValueError, "intersect"):
            newton.solvers.SolverIPC(builder.finalize(device=device))
    with test.subTest(case="cloth through a static box"):
        builder = newton.ModelBuilder()
        builder.add_shape_box(
            -1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()), hx=0.1, hy=0.1, hz=0.1
        )
        _cloth(builder, 4, 0.1, 0.5, center=(0.033, 0.021))  # the box's vertical edges pierce its triangles
        with test.assertRaisesRegex(ValueError, "intersect"):
            newton.solvers.SolverIPC(builder.finalize(device=device))
    with test.subTest(case="crossings that contact ignores"):
        builder = newton.ModelBuilder()
        _open_plane(builder)
        # two overlapping shapes of one body, above the plane
        body = _box_body(builder, (0.0, 0.0, 0.5), (0.2, 0.2, 0.2))
        builder.add_shape_box(
            body, xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()), hx=0.2, hy=0.05, hz=0.05
        )
        # a kinematic body through the static plane
        kinematic = builder.add_body(xform=wp.transform(wp.vec3(0.6, 0.6, 0.0), wp.quat_identity()), is_kinematic=True)
        builder.add_shape_box(kinematic, hx=0.1, hy=0.1, hz=0.1)
        solver = newton.solvers.SolverIPC(builder.finalize(device=device))
        # the surfaces do cross: the check let these crossings through
        test.assertGreater(SurfaceChecker.from_engine(solver._engine).intersections(solver._engine.vm.x), 0)
    with test.subTest(case="a flat sheet off the axes"):
        # a 10 x 10 cloth in the plane y = x + 50, 100 m wide: its float32 vertices are not exactly coplanar, and
        # the brute-force checker counts crossings between its nearly coplanar triangles, which are not there
        t = np.linspace(-50.0, 50.0, 10)
        x, z = np.meshgrid(t, t, indexing="ij")
        vertices = np.stack([x, x + 50.0, z], axis=-1).reshape(-1, 3)
        quads = [(10 * i + j, 10 * i + j + 10, 10 * i + j + 11, 10 * i + j + 1) for i in range(9) for j in range(9)]
        builder = newton.ModelBuilder()
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vertices=[wp.vec3(*v) for v in vertices],
            indices=[i for a, b, c, d in quads for i in (a, b, c, a, c, d)],
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.1,
            tri_ke=1.0e4,
            tri_ka=1.0e4,
            tri_kd=0.0,
        )
        solver = newton.solvers.SolverIPC(builder.finalize(device=device))
        test.assertGreater(SurfaceChecker.from_engine(solver._engine).intersections(solver._engine.vm.x), 0)


def test_rejects_start_in_contact(test, device):
    """Reject the surfaces of different bodies that start at or inside their contact thickness, naming the closest
    pair, which every step would fail on; accept a gap above the thickness, also inside d_hat, and contact that the
    solve skips (surfaces that never move, the inside of a body)."""

    def build(*parts, **kwargs):
        builder = newton.ModelBuilder()
        for part in parts:
            part(builder)
        return newton.solvers.SolverIPC(builder.finalize(device=device), **kwargs)

    def ground(builder):
        builder.add_ground_plane()

    def static_box(builder):
        # all coordinates of these boxes are exact in float32, so that touching faces are 0 m apart
        builder.add_shape_box(
            -1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.125), wp.quat_identity()), hx=0.125, hy=0.125, hz=0.125
        )

    def box(z, kinematic=False, half=0.125):
        def add(builder):
            b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()), is_kinematic=kinematic)
            builder.add_shape_box(b, hx=half, hy=half, hz=half)

        return add

    def cloth(z, center=(0.0, 0.0)):
        return lambda builder: _cloth(builder, 4, 0.05, z, center=center)

    for case, parts, kwargs, pattern in (
        ("on the ground", (ground, box(0.125)), {}, r"body 0 and plane shape 0 start 0 m apart"),
        ("into the ground", (ground, box(0.12)), {}, r"body 0 and plane shape 0 start -0.005 m apart"),
        ("on a static box", (static_box, box(0.375)), {}, r"static shape 0 and body 0 start 0 m apart"),
        ("on another body", (box(0.125), box(0.375)), {}, r"body 0 and body 1 start 0 m apart"),
        # 1 cm cubes stacked in touching contact: in float32 these faces are 9.3e-10 m apart
        ("apart by rounding", (box(0.015, half=0.005), box(0.025, half=0.005)), {}, r"rounding of float32 positions"),
        ("cloth inside its thickness", (static_box, cloth(0.251)), {"thickness": 0.002}, r"thickness \(0.002 m\)"),
    ):
        with test.subTest(case=case), test.assertRaisesRegex(ValueError, pattern):
            build(*parts, **kwargs)
    # a gap above the thickness, here inside d_hat (1 mm, 1 um); a kinematic body on a static box; touching shapes of
    # one body
    build(ground, static_box, box(0.376), cloth(0.003, center=(1.0, 0.0)), thickness=0.002)
    build(box(0.125), box(0.375 + 1.0e-6))
    build(static_box, box(0.375, kinematic=True))

    def two_shapes(builder):
        b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.2), wp.quat_identity()))
        builder.add_shape_box(b, hx=0.125, hy=0.125, hz=0.125)
        builder.add_shape_box(
            b, xform=wp.transform(wp.vec3(0.0, 0.0, 0.25), wp.quat_identity()), hx=0.125, hy=0.125, hz=0.125
        )

    build(ground, two_shapes)


# ---------------------------------------------------------------------------- forces


def test_external_forces_match_closed_form(test, device):
    """Accelerate a particle with particle_f and two free boxes with body_f and with a free joint's joint_f.

    Without gravity, a constant force F and torque tau give implicit Euler's x_k = x_0 + a dt^2 k (k + 1) / 2 for
    the centre of mass (a = F / m) and the same law for the angle about a principal axis (alpha = tau / I_zz), from
    rest. The boxes' shapes sit off their body origins, so the force also exercises the centre-of-mass offset.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    particle = builder.add_particle(wp.vec3(0.0, 1.0, 1.0), wp.vec3(0.0, 0.0, 0.0), 0.5)
    bodies = []
    for y in (0.0, -1.0):
        b = builder.add_body(xform=wp.transform(wp.vec3(0.0, y, 1.0), wp.quat_identity()))
        builder.add_shape_box(
            b, xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()), hx=0.1, hy=0.05, hz=0.08
        )
        bodies.append(b)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, newton_velocity_tol=1e-6, linear_tol_rate=1e-12, semi_implicit=False)
    m = model.body_mass.numpy()[bodies[0]]
    I_zz = model.body_inertia.numpy()[bodies[0]][2, 2]
    com = model.body_com.numpy()[bodies[0]].astype(np.float64)
    force, torque = np.array([m, 0.5 * m, 0.0]), np.array([0.0, 0.0, I_zz])  # 1 and 0.5 m/s^2, 1 rad/s^2
    f_particle = np.array([0.0, 0.0, 1.0])  # 2 m/s^2
    s0, s1 = model.state(), model.state()
    control = model.control()
    wrench = np.concatenate([force, torque]).astype(np.float32)
    joint_f = control.joint_f.numpy()
    joint_f[model.joint_qd_start.numpy()[1] : model.joint_qd_start.numpy()[1] + 6] = wrench
    control.joint_f.assign(joint_f)
    observables = solver.observables({solver.ObservableKind.STATUS, solver.ObservableKind.SOLVER_RESIDUAL})
    x0 = model.particle_q.numpy()[particle].astype(np.float64)
    q0 = model.body_q.numpy().astype(np.float64)
    dt = 0.01
    for k in range(1, 11):
        for s in (s0, s1):
            body_f = np.zeros((model.body_count, 6), np.float32)
            body_f[bodies[0]] = wrench
            s.body_f.assign(body_f)
            s.particle_f.assign(np.array([f_particle], np.float32))
        solver.step(s0, s1, control, None, dt, observables=observables)
        s0, s1 = s1, s0
        test.assertEqual(observables.status.numpy().tolist(), [0, 0])  # the global entry, and world 0 unused
        residual = float(observables.solver_residual.numpy()[-1])
        test.assertGreater(residual, 0.0)
        test.assertLessEqual(residual, 1.0)
        ramp = dt * dt * k * (k + 1) / 2.0
        np.testing.assert_allclose(s0.particle_q.numpy()[particle], x0 + 2.0 * f_particle * ramp, atol=1e-6)
        np.testing.assert_allclose(s0.particle_qd.numpy()[particle], 2.0 * f_particle * k * dt, atol=1e-6)
        q, qd = s0.body_q.numpy(), s0.body_qd.numpy()
        for b in bodies:
            angle = 2.0 * np.arctan2(q[b, 5], q[b, 6])
            R = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
            x_com = q[b, :3] + R @ com
            np.testing.assert_allclose(x_com, q0[b, :3] + com + force / m * ramp, atol=1e-6)
            np.testing.assert_allclose(qd[b, :3], force / m * k * dt, atol=1e-6)
            # the affine body is near-rigid: its rotation is within 1e-5 of the rigid one (measured)
            test.assertAlmostEqual(angle, ramp, delta=1e-4 * ramp)
            np.testing.assert_allclose(qd[b, 3:], [0.0, 0.0, k * dt], atol=1e-4 * k * dt)


# ---------------------------------------------------------------------------- status of a step


def _boxes_apart(device):
    """A box 1 mm above a static box, and the pose ``body_q`` that rests it exactly on the static box: their faces
    then touch at z = 0.25 (all coordinates exact)."""
    builder = newton.ModelBuilder()
    builder.add_shape_box(
        -1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.125), wp.quat_identity()), hx=0.125, hy=0.125, hz=0.125
    )
    _box_body(builder, (0.0, 0.0, 0.376), (0.125, 0.125, 0.125))
    model = builder.finalize(device=device)
    touching = model.body_q.numpy()
    touching[0, 2] = 0.375
    return model, touching


def test_contact_at_start_fails(test, device):
    """A step that starts with surfaces in touching contact fails: eagerly it raises, captured it sets the bit.

    The solver refuses such a start at construction, so the step takes it from a state written between steps.
    """
    model, touching = _boxes_apart(device)
    solver = newton.solvers.SolverIPC(model)
    s0, s1 = model.state(), model.state()
    s0.body_q.assign(touching)
    with test.assertRaisesRegex(RuntimeError, "THICKNESS_VIOLATED"):
        solver.step(s0, s1, None, None, 0.01)

    model, touching = _boxes_apart(device)
    solver = newton.solvers.SolverIPC(model)
    observables = solver.observables({solver.ObservableKind.STATUS, solver.ObservableKind.SOLVER_RESIDUAL})
    s0, s1 = model.state(), model.state()
    with wp.ScopedCapture(device=device) as capture:
        solver.step(s0, s1, None, None, 0.01, observables=observables)
    s0.body_q.assign(touching)
    wp.capture_launch(capture.graph)
    status = int(observables.status.numpy()[-1])
    test.assertTrue(status & newton.solvers.SolverIPC.StatusBits.THICKNESS_VIOLATED)
    test.assertTrue(status & newton.solvers.SolverIPC.StatusBits.FAILED)
    test.assertEqual(float(observables.solver_residual.numpy()[-1]), np.inf)
    # the caller of a captured step learns of the failure from check_status
    with test.assertRaisesRegex(RuntimeError, r"failed on THICKNESS_VIOLATED \(residual inf"):
        solver.check_status(observables)


def test_check_status(test, device):
    """check_status returns False after a clean step, warns when the Newton loop reached its iteration limit, and
    refuses a container without the status."""
    import warnings  # noqa: PLC0415

    model, _ = _boxes_apart(device)
    observables = {}
    for max_iter in (1024, 1):
        solver = newton.solvers.SolverIPC(model, newton_max_iter=max_iter)
        observables[max_iter] = solver.observables(
            {solver.ObservableKind.STATUS, solver.ObservableKind.SOLVER_RESIDUAL}
        )
        s0, s1 = model.state(), model.state()
        solver.step(s0, s1, None, None, 0.01, observables=observables[max_iter])  # the box lands on the static box
        if max_iter == 1:
            with test.assertWarnsRegex(newton.NewtonWarning, "newton_max_iter without converging \\(residual"):
                test.assertFalse(solver.check_status(observables[max_iter]))
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                test.assertFalse(solver.check_status(observables[max_iter]))
    with test.assertRaisesRegex(ValueError, "STATUS"):
        solver.check_status(solver.observables({solver.ObservableKind.SOLVER_RESIDUAL}))


def test_default_d_hat_follows_small_features(test, device):
    """Without a d_hat argument the solver lowers d_hat to a quarter of the smallest colliding feature.

    The tip of Newton's balance_bird example, a sphere of 8 mm radius 1 mm above the 6 mm top of a static cone,
    overflowed the contact lists of captured steps with libuipc's 0.01 m; with the default 2 mm it does not. A cloth of
    5 mm cells gets 2.5 mm and rests unstretched; with 0.01 m the barrier between its non-adjacent primitives
    stretches it. An explicit d_hat is kept, and a model of large features keeps 0.01 m.
    """

    def tip():
        builder = newton.ModelBuilder()
        pedestal = newton.Mesh.create_cylinder(
            0.08, 0.06, up_axis=newton.Axis.Z, segments=32, top_radius=0.006, compute_inertia=False
        )
        builder.add_shape_mesh(-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.06), wp.quat_identity()), mesh=pedestal)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.129), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.008)
        return builder.finalize(device=device)

    test.assertEqual(newton.solvers.SolverIPC(tip(), d_hat=0.005)._config.d_hat, 0.005)
    model = tip()
    solver = newton.solvers.SolverIPC(model)
    test.assertAlmostEqual(solver._config.d_hat, 0.002)
    observables = solver.observables({solver.ObservableKind.STATUS})
    s0, s1 = model.state(), model.state()
    with wp.ScopedCapture(device=device) as capture:
        solver.step(s0, s1, None, None, 1.0 / 600.0, observables=observables)
        solver.step(s1, s0, None, None, 1.0 / 600.0, observables=observables)
    for _ in range(5):
        wp.capture_launch(capture.graph)
        test.assertFalse(solver.check_status(observables))

    # a 10 cm cloth of 5 mm cells, 2 mm above the ground: its mean edge length after 30 steps over the rest length
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    _cloth(builder, 20, 0.005, 0.002, mass=7.5e-6, tri_k=1.0e2, edge_ke=1.0e-4)
    model = builder.finalize(device=device)
    tris = model.tri_indices.numpy().reshape(-1, 3)
    edges = np.unique(np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1), axis=0)

    def mean_edge(x):
        return np.linalg.norm(x[edges[:, 0]] - x[edges[:, 1]], axis=1).mean()

    rest = mean_edge(model.particle_q.numpy())
    for kwargs, expected_d_hat in (({}, 0.0025), ({"d_hat": 0.01}, 0.01)):
        solver = newton.solvers.SolverIPC(model, **kwargs)
        test.assertAlmostEqual(solver._config.d_hat, expected_d_hat)
        s0, s1 = model.state(), model.state()
        for _ in range(30):
            solver.step(s0, s1, None, None, 1.0 / 600.0)
            s0, s1 = s1, s0
        if kwargs:
            test.assertGreater(mean_edge(s0.particle_q.numpy()) / rest, 1.05)
        else:
            test.assertLess(mean_edge(s0.particle_q.numpy()) / rest, 1.001)

    builder = newton.ModelBuilder()
    _box_body(builder, (0.0, 0.0, 1.0), (0.05, 0.5, 0.5))
    test.assertEqual(newton.solvers.SolverIPC(builder.finalize(device=device))._config.d_hat, 0.01)


# small active contact lists, so that the contact of two soft cubes overflows them
_TINY_LISTS = {"active": dict.fromkeys(("PP", "PE", "PT", "EE"), 1)}


def _stacking_cubes(device):
    """A soft cube that falls 4 cm onto another resting on the ground."""
    builder = newton.ModelBuilder()
    for z in (0.0005, 0.14):
        builder.add_soft_grid(
            pos=wp.vec3(-0.05, -0.05, z),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=2,
            dim_y=2,
            dim_z=2,
            cell_x=0.05,
            cell_y=0.05,
            cell_z=0.05,
            density=1000.0,
            k_mu=3.4e4,
            k_lambda=1.6e5,
            k_damp=0.0,
        )
    builder.add_ground_plane()
    return builder.finalize(device=device)


def test_overflow_grows(test, device):
    """Contact lists that overflow grow: at once in an eager step, after grow() when the step is captured.

    Either way the cubes come to rest without an intersection, and a captured step sets the overflow bit.
    """
    from uipc_warp.validation import SurfaceChecker  # noqa: PLC0415

    # converged tightly, so that the two runs below agree to rounding
    tight = {"newton_velocity_tol": 1e-6, "linear_tol_rate": 1e-12, "semi_implicit": False}
    model = _stacking_cubes(device)
    solver = newton.solvers.SolverIPC(model, **tight)
    solver._make_engine(_TINY_LISTS)
    checker = SurfaceChecker.from_engine(solver._engine)
    s0, s1 = model.state(), model.state()
    hits = 0
    for _ in range(30):
        solver.step(s0, s1, None, None, 0.01)
        s0, s1 = s1, s0
        hits += checker.intersections(solver._engine.vm.x)
    test.assertEqual(hits, 0)
    test.assertGreater(solver._engine.simplex.capacities()["active"]["PT"], 1)
    eager = s0.particle_q.numpy()

    model = _stacking_cubes(device)
    solver = newton.solvers.SolverIPC(model, **tight)
    solver._make_engine(_TINY_LISTS)
    observables = solver.observables({solver.ObservableKind.STATUS})
    states = [model.state(), model.state()]

    def capture():
        graphs = []
        for a in range(2):
            with wp.ScopedCapture(device=device) as cap:
                solver.step(states[a], states[1 - a], None, None, 0.01, observables=observables)
            graphs.append(cap.graph)
        return graphs

    graphs = capture()
    overflows, hits, a = 0, 0, 0
    for _ in range(30):
        wp.capture_launch(graphs[a])
        if overflows == 0 and int(observables.status.numpy()[-1]) & newton.solvers.SolverIPC.StatusBits.OVERFLOW:
            with test.assertRaisesRegex(RuntimeError, "OVERFLOW"):  # without grow=True an overflow is a failure
                solver.check_status(observables)
        if solver.check_status(observables, grow=True):
            # the step ran without some pairs and the lists were grown: step again from the same input state
            overflows += 1
            graphs = capture()
            wp.capture_launch(graphs[a])
            test.assertFalse(solver.check_status(observables))
        hits += checker.intersections(solver._engine.vm.x)
        a = 1 - a
    test.assertGreater(overflows, 0)
    test.assertEqual(hits, 0)
    # the captured steps, grown and repeated, take the eager steps' path
    np.testing.assert_allclose(states[a].particle_q.numpy(), eager, rtol=0.0, atol=1e-6)


# ---------------------------------------------------------------------------- state and step


def test_teleport_equals_fresh_solver(test, device):
    """A state written between graph replays continues as a solver started from it does."""
    tight = {"newton_velocity_tol": 1e-6, "linear_tol_rate": 1e-12, "semi_implicit": False}

    def scene():
        builder = newton.ModelBuilder()
        builder.add_soft_grid(
            pos=wp.vec3(-0.25, -0.05, 0.002),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=2,
            dim_y=2,
            dim_z=2,
            cell_x=0.05,
            cell_y=0.05,
            cell_z=0.05,
            density=1000.0,
            k_mu=3.4e4,
            k_lambda=1.6e5,
            k_damp=0.0,
        )
        _box_body(builder, (0.2, 0.0, 0.1005), (0.1, 0.1, 0.1))
        builder.add_ground_plane()
        return builder.finalize(device=device)

    model = scene()
    solver = newton.solvers.SolverIPC(model, **tight)
    s0, s1 = model.state(), model.state()
    solver.step(s0, s1, None, None, 0.01)
    s0, s1 = s1, s0
    with wp.ScopedCapture(device=device) as capture:
        solver.step(s0, s1, None, None, 0.01)
        solver.step(s1, s0, None, None, 0.01)
    for _ in range(3):
        wp.capture_launch(capture.graph)
    # move both bodies up and give them a velocity: the soft cube falls back onto the ground, the box spins
    q, qd = s0.particle_q.numpy(), s0.particle_qd.numpy()
    q[:, 2] += 0.03
    qd[:] = (0.2, 0.0, -0.1)
    bq, bqd = s0.body_q.numpy(), s0.body_qd.numpy()
    bq[0, 2] += 0.02
    bqd[0] = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    for array, value in ((s0.particle_q, q), (s0.particle_qd, qd), (s0.body_q, bq), (s0.body_qd, bqd)):
        array.assign(value)
    teleported = []
    for _ in range(8):
        wp.capture_launch(capture.graph)
        teleported.append((s0.particle_q.numpy(), s0.body_q.numpy()))

    model = scene()
    fresh = newton.solvers.SolverIPC(model, **tight)
    f0, f1 = model.state(), model.state()
    for array, value in ((f0.particle_q, q), (f0.particle_qd, qd), (f0.body_q, bq), (f0.body_qd, bqd)):
        array.assign(value)
    for k in range(16):
        fresh.step(f0, f1, None, None, 0.01)
        f0, f1 = f1, f0
        if k % 2:
            np.testing.assert_allclose(teleported[k // 2][0], f0.particle_q.numpy(), rtol=0.0, atol=1e-6)
            np.testing.assert_allclose(teleported[k // 2][1], f0.body_q.numpy(), rtol=0.0, atol=1e-6)


def test_reset_inside_captured_graph(test, device):
    """A reset captured before every step discards the friction of the last step's contacts.

    A cube sliding at 1 m/s on the ground then keeps its speed, as a solver started anew every step would. A mask
    that selects no world with content leaves the steps as they are without a reset: friction slows the cube.
    """
    g, mu, v0, dt = 9.81, 0.5, 1.0, 0.01
    runs = {}
    for case in ("all", "global world", "world 0 only", "none"):
        builder = newton.ModelBuilder()
        builder.add_soft_grid(
            pos=wp.vec3(-0.05, -0.05, 0.004),
            rot=wp.quat_identity(),
            vel=wp.vec3(v0, 0.0, 0.0),
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
        # entries [world 0, global world -1]: without begin_world everything is global
        mask = {
            "all": None,
            "global world": wp.array([False, True], dtype=wp.bool, device=device),
            "world 0 only": wp.array([True, False], dtype=wp.bool, device=device),
        }.get(case)
        solver = newton.solvers.SolverIPC(model, friction_mu=mu, newton_velocity_tol=1e-4, semi_implicit=False)
        s0, s1 = model.state(), model.state()
        for _ in range(4):  # in contact, with friction pairs
            solver.step(s0, s1, None, None, dt)
            s0, s1 = s1, s0
        with wp.ScopedCapture(device=device) as capture:
            for a, b in ((s0, s1), (s1, s0)):
                if case != "none":
                    solver.reset(a, world_mask=mask)
                solver.step(a, b, None, None, dt)
        m = model.particle_mass.numpy()
        v_start = m @ s0.particle_qd.numpy()[:, 0] / m.sum()  # the centre of mass's
        for _ in range(5):
            wp.capture_launch(capture.graph)
        runs[case] = (m @ s0.particle_qd.numpy()[:, 0] / m.sum() - v_start, s0.particle_q.numpy())
    for case in ("all", "global world"):
        test.assertLess(abs(runs[case][0]), 1e-3 * mu * g * 10 * dt, case)
    test.assertLess(runs["none"][0], -0.5 * mu * g * 10 * dt)
    np.testing.assert_allclose(runs["world 0 only"][1], runs["none"][1], rtol=0.0, atol=1e-6)


def test_notify_updates_gravity_and_inertia_in_place(test, device):
    """Gravity, mass and inertia changed after a capture take effect in its replays.

    A box under a constant force and torque, and a particle: their accelerations follow the new gravity and the
    box's doubled mass and inertia without a new capture.
    """
    builder = newton.ModelBuilder()
    builder.add_particle(wp.vec3(0.0, 1.0, 1.0), wp.vec3(0.0, 0.0, 0.0), 0.5)
    b = _box_body(builder, (0.0, 0.0, 1.0), (0.1, 0.05, 0.08))
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, newton_velocity_tol=1e-6, linear_tol_rate=1e-12, semi_implicit=False)
    m = float(model.body_mass.numpy()[b])
    I_zz = float(model.body_inertia.numpy()[b][2, 2])
    wrench = np.array([[2.0 * m, 0.0, 0.0, 0.0, 0.0, I_zz]], np.float32)  # 2 m/s^2, 1 rad/s^2
    s0, s1 = model.state(), model.state()
    for s in (s0, s1):
        s.body_f.assign(wrench)
    solver.step(s0, s1, None, None, 0.01)
    s0, s1 = s1, s0
    with wp.ScopedCapture(device=device) as capture:
        solver.step(s0, s1, None, None, 0.01)
        solver.step(s1, s0, None, None, 0.01)

    def accelerations():
        """(particle, body linear, body angular) accelerations of a replay from its velocities."""
        before = (s0.particle_qd.numpy()[0], s0.body_qd.numpy()[b])
        wp.capture_launch(capture.graph)
        after = (s0.particle_qd.numpy()[0], s0.body_qd.numpy()[b])
        return (
            (after[0] - before[0]) / 0.02,
            (after[1][:3] - before[1][:3]) / 0.02,
            (after[1][3:] - before[1][3:]) / 0.02,
        )

    # float32 velocities over two steps resolve accelerations to about 1e-6; the near-rigid rotation is within
    # 1e-5 of the rigid one (measured)
    g = np.array([0.0, 0.0, -9.81])
    a_p, a_b, alpha = accelerations()
    np.testing.assert_allclose(a_p, g, atol=1e-5)
    np.testing.assert_allclose(a_b, g + np.array([2.0, 0.0, 0.0]), atol=1e-5)
    np.testing.assert_allclose(alpha, [0.0, 0.0, 1.0], atol=1e-4)

    model.set_gravity((1.0, 0.0, -3.0))
    for name in ("body_mass", "body_inertia"):
        getattr(model, name).assign(2.0 * getattr(model, name).numpy())
    for name in ("body_inv_mass", "body_inv_inertia"):
        getattr(model, name).assign(0.5 * getattr(model, name).numpy())
    solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES | newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    g = np.array([1.0, 0.0, -3.0])
    a_p, a_b, alpha = accelerations()
    np.testing.assert_allclose(a_p, g, atol=1e-5)
    np.testing.assert_allclose(a_b, g + np.array([1.0, 0.0, 0.0]), atol=1e-5)
    np.testing.assert_allclose(alpha, [0.0, 0.0, 0.5], atol=1e-4)


@requires("uipc_warp")
class TestSolverIPC(unittest.TestCase):
    pass


@requires("uipc_warp")
class TestSolverIPCInputs(unittest.TestCase):
    """What the solver takes from a model, a state and a control, and what it rejects."""

    def test_cpu_device(self):
        """Reject a model on the CPU: the engine runs on CUDA devices only."""
        model = _two_particles().finalize(device="cpu")
        with self.assertRaisesRegex(NotImplementedError, "CUDA"):
            newton.solvers.SolverIPC(model)

    def test_version_mismatches(self):
        """Name libuipc-warp releases outside the supported range and a Warp outside libuipc-warp's requirement."""
        from newton._src.solvers.ipc.solver_ipc import _version_mismatches  # noqa: PLC0415

        self.assertEqual(_version_mismatches("0.4.0.dev0", "1.19.0", ">=1.19,<1.20"), [])
        self.assertEqual(_version_mismatches("0.4.2", "1.19.3", None), [])
        for uipc_warp, warp in (("0.3.0", "1.19.0"), ("0.5.0", "1.19.0"), ("0.4.0", "1.20.0"), ("0.4.0", "1.18.1")):
            self.assertEqual(len(_version_mismatches(uipc_warp, warp, ">=1.19,<1.20")), 1, (uipc_warp, warp))


@requires("uipc_warp")
class TestSolverIPCStep(unittest.TestCase):
    """How a step ends and what carries from one step to the next: status, overflow, state, reset, model changes."""


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


for name, func in (
    ("test_rejects_springs", test_rejects_springs),
    ("test_rejects_attachments", test_rejects_attachments),
    ("test_rejects_mimic_constraints", test_rejects_mimic_constraints),
    ("test_rejects_equality_constraints", test_rejects_equality_constraints),
    ("test_rejects_tendons", test_rejects_tendons),
    ("test_rejects_muscles", test_rejects_muscles),
    ("test_rejects_activations", test_rejects_activations),
    ("test_rejects_joints_and_drives", test_rejects_joints_and_drives),
    ("test_rejects_gradients", test_rejects_gradients),
    ("test_rejects_worlds", test_rejects_worlds),
    ("test_rejects_contact_filters", test_rejects_contact_filters),
    ("test_particle_only_shapes_without_particles", test_particle_only_shapes_without_particles),
    ("test_rejects_surface_velocity", test_rejects_surface_velocity),
    ("test_rejects_intersecting_start", test_rejects_intersecting_start),
    ("test_rejects_start_in_contact", test_rejects_start_in_contact),
    ("test_external_forces_match_closed_form", test_external_forces_match_closed_form),
):
    add_function_test(TestSolverIPCInputs, name, func, devices=devices)
for name, func in (
    ("test_contact_at_start_fails", test_contact_at_start_fails),
    ("test_check_status", test_check_status),
    ("test_default_d_hat_follows_small_features", test_default_d_hat_follows_small_features),
    ("test_overflow_grows", test_overflow_grows),
    ("test_teleport_equals_fresh_solver", test_teleport_equals_fresh_solver),
    ("test_reset_inside_captured_graph", test_reset_inside_captured_graph),
    ("test_notify_updates_gravity_and_inertia_in_place", test_notify_updates_gravity_and_inertia_in_place),
):
    add_function_test(TestSolverIPCStep, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
