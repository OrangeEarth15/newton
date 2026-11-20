# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import ast
import importlib
import importlib.util
import pathlib
import re
import types
import unittest
import warnings
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.utils.heightfield import HeightfieldData, get_triangle_shape_from_heightfield
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
    """A soft cube dropped from 5 cm onto the ground plane: the barrier keeps every vertex above the plane, the cube
    comes to rest inside the activation distance d_hat, and the last step converges."""
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
    """A soft cube falling freely, without contact, follows implicit Euler's heights ``z_0 - g dt^2 k (k + 1) / 2`` to
    1e-5 m (the float32 output) over ten steps."""
    model = _soft_cube(device, height=1.0, ground=False)
    solver = newton.solvers.SolverIPC(model, newton_velocity_tol=1e-5, linear_tol_rate=1e-12)
    x0 = model.particle_q.numpy().astype(np.float64)
    traj = _run(model, solver, 10)
    k = np.arange(1, 11)
    expected = x0[None, :, 2] - 9.81 * 0.01**2 * (k * (k + 1) / 2.0)[:, None]
    np.testing.assert_allclose(traj[:, :, 2], expected, atol=1e-5)  # float32 output


def test_kinematic_particles_stay(test, device):
    """The particles of zero mass, a soft cube's fixed left face, stay where they are, bit for bit, while the rest of
    the cube falls."""
    model = _soft_cube(device, height=0.5, ground=False, fix_left=True)
    fixed = model.particle_mass.numpy() == 0.0
    test.assertTrue(fixed.any())
    solver = newton.solvers.SolverIPC(model)
    x0 = model.particle_q.numpy()
    traj = _run(model, solver, 20)
    np.testing.assert_array_equal(traj[-1][fixed], x0[fixed])
    test.assertLess(traj[-1][~fixed, 2].min(), x0[~fixed, 2].min())


def test_soft_bodies_stack_without_intersection(test, device):
    """A small soft cube falls onto a larger one on the ground: in no step does an edge cross a triangle or two
    surfaces touch, and the small cube comes to rest on top of the large one."""
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
    """A box on the ground sticks below the friction angle and slides with g (sin - mu cos) above it; mu is the mean
    of the ground's 0.5 and the particles' soft_contact_mu, 0.5."""
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
    """Steps of a soft cube dropping onto the ground, captured in a CUDA graph and replayed, follow eager steps to
    1e-6 m over 21 steps. The solver adds with float atomics (SpMV, triplet reduction), as libuipc does, so converged
    tightly the captured and eager runs agree to rounding rather than bit for bit."""
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
    """A cloth grid falls onto a fixed sphere above the ground: in no step does an edge cross a triangle or two
    surfaces touch, the cloth stays outside the sphere up to the sag of its triangulation, its middle rests on the
    sphere's top, within d_hat + 1 mm, while the sheet hangs down around it, and no contact list overflows."""
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
    """A rigid box whose shape, and with it the centre of mass, is off the body origin, turned about z and thrown
    sideways, falls as implicit Euler does: over ten steps it moves by ``v_0 k dt - g dt^2 k (k + 1) / 2`` and its
    velocity is ``v_0 - g k dt``, to 1e-5 (the float32 output), and it does not turn."""
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
    """A box spinning freely about its axis of symmetry, without gravity: within a step the affine body moves along a
    straight line in its 12 coordinates and back onto the rotations, so its angular velocity drops by about
    (omega dt)^2 per step, as a rotation projected back after every step gives it: the increments
    alpha_1 = atan(omega dt) and alpha_(n+1) = atan(sin alpha_n / (2 - cos alpha_n)), the angular velocity
    sin(alpha_n) / dt."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    b = _box_body(builder, (0.0, 0.0, 1.0), (0.1, 0.1, 0.05))
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, newton_velocity_tol=1e-6, linear_tol_rate=1e-12, semi_implicit=False)
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
    """Three rigid boxes stacked with 4 mm gaps fall onto the ground and onto each other: in no step does an edge cross
    a triangle or two surfaces touch, and after 100 steps every box rests upright on the one below, the bottom one on
    the ground, within d_hat + 1 mm."""
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
    """A cloth falls onto a rigid box standing on the ground: in no step does an edge cross a triangle or two surfaces
    touch, the box stays upright where it stood on the ground, and the cloth's middle rests on its top, within
    d_hat + 1 mm, while the edges hang over the sides."""
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
    # friction 0.5 (the mean of this and soft_contact_mu): the solve does not sweep the box's 2 mm jumps, and the
    # cloth, which moves up 2 mm against the box in every step, crumples at the 0.75 of Newton's default materials
    builder.add_shape_box(b, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
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
    """As test_graph_capture_matches_eager, with a cloth falling onto a rigid box: the particles and the body of steps
    replayed from a CUDA graph follow eager steps to 1e-6 over 21 steps."""
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


def test_rejects_springs_without_a_libuipc_form(test, device):
    """Reject a damped spring (libuipc has no damping element), a spring of negative rest length or of negative
    stiffness and one that joins a particle to itself."""
    for case, pattern in (
        ("damped", "damping is 0.5"),
        ("negative rest length", "rest length is -0.1"),
        ("negative", "stiffness"),
        ("itself", "joins particle 0 to itself"),
    ):
        with test.subTest(case=case):
            builder = _two_particles()
            builder.add_spring(0, 0 if case == "itself" else 1, -1.0e3 if case == "negative" else 1.0e3,
                               0.5 if case == "damped" else 0.0, 0.0)  # fmt: skip
            if case == "negative rest length":
                builder.spring_rest_length[0] = -0.1
            _rejects(test, builder, pattern, device)


def test_rejects_attachment_damping(test, device):
    """Reject a particle-body attachment with damping, naming it: libuipc's stitch has no damping."""
    builder = _two_particles()
    b = _box_body(builder, (0.0, 0.0, 2.0), (0.1, 0.1, 0.1))
    builder.add_attachment_particle_body(0, b, body_point=(0.0, 0.0, -0.5))
    builder.add_attachment_particle_body(1, b, body_point=(0.1, 0.0, -0.5), damping=2.0)
    _rejects(test, builder, "particle-body attachment 1, whose damping is 2.0", device)


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


def test_rejects_unsupported_joints(test, device):
    """Reject the joints SolverIPC does not simulate, naming what libuipc lacks: D6, distance and rod joints, drives,
    passive springs and limits on ball and free joints, joint armature and friction, and an effort limit on a driven
    joint."""

    def hinged(add, **dof):
        builder = newton.ModelBuilder()
        link = builder.add_link(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
        builder.add_articulation([add(builder, link)])
        for name, value in dof.items():  # on the first degree of freedom
            getattr(builder, f"joint_{name}")[0] = value
        return builder

    def revolute(builder, link):
        return builder.add_joint_revolute(-1, link, axis=newton.Axis.Y)

    def ball(builder, link):
        return builder.add_joint_ball(-1, link, parent_xform=wp.transform((0.0, 0.0, 1.2), wp.quat_identity()))

    def d6(builder, link):
        return builder.add_joint_d6(-1, link, angular_axes=[newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X)])

    def distance(builder, link):
        return builder.add_joint_distance(-1, link)

    def rod(builder, link):
        return builder.add_joint_rod(-1, link)

    free = newton.ModelBuilder()
    _box_body(free, (0.0, 0.0, 1.0), (0.1, 0.1, 0.1))
    free.joint_target_mode[0] = int(newton.JointTargetMode.POSITION)
    for name, builder, pattern in (
        ("d6", hinged(d6), "D6 joints \\(joint 0\\): libuipc has no D6 joint"),
        ("distance", hinged(distance), "DISTANCE joints \\(joint 0\\): libuipc has no distance joint"),
        ("rod", hinged(rod), "ROD joints \\(joint 0\\): libuipc has no constitution of stretch, shear, bend"),
        ("free drive", free, "FREE joint 0 has the target mode POSITION"),
        ("ball drive", hinged(ball, target_ke=10.0), "its ball joint has none .* BALL joint 0 has target_ke 10.0"),
        ("ball limit", hinged(ball, limit_lower=-0.5), "limits revolute and prismatic joints only"),
        ("armature", hinged(revolute, armature=0.01), "joint armature: libuipc's joints have no inertia"),
        ("friction", hinged(revolute, friction=0.5), "joint friction: libuipc's joints have none"),
        ("effort limit", hinged(revolute, target_ke=100.0, effort_limit=5.0), "joint_effort_limit 5.0"),
    ):
        with test.subTest(name):
            _rejects(test, builder, pattern, device)


def test_rejects_gradients(test, device):
    """Reject a model finalized with gradients, as SolverSteel does."""
    builder = _two_particles()
    model = builder.finalize(device=device, requires_grad=True)
    with test.assertRaisesRegex(NotImplementedError, "gradients"):
        newton.solvers.SolverIPC(model)


def test_rejects_worlds(test, device):
    """Reject content in two worlds unless coupled_worlds: they would share one Newton solve."""
    builder = newton.ModelBuilder()
    for _ in range(2):
        builder.begin_world()
        _box_body(builder, (0.0, 0.0, 1.0), (0.1, 0.1, 0.1))
        builder.end_world()
    _rejects(
        test,
        builder,
        r"one world: .* worlds \[0, 1\], which would share one Newton solve\. With coupled_worlds",
        device,
    )


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

    def particle(z, radius, x=0.0):
        return lambda builder: builder.add_particle(wp.vec3(x, 0.0, z), wp.vec3(0.0, 0.0, 0.0), 1.0e-3, radius=radius)

    def margin_box(z, margin):
        def add(builder):
            b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()))
            builder.add_shape_box(b, hx=0.125, hy=0.125, hz=0.125, cfg=newton.ModelBuilder.ShapeConfig(margin=margin))

        return add

    def thick_ground(builder):
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(margin=0.01))

    def spring(builder):
        builder.add_spring(0, 1, 10.0, 0.0, 0.0)

    for case, parts, kwargs, pattern in (
        ("on the ground", (ground, box(0.125)), {}, r"body 0 and plane shape 0 start 0 m apart"),
        ("into the ground", (ground, box(0.12)), {}, r"body 0 and plane shape 0 start -0.005 m apart"),
        ("on a static box", (static_box, box(0.375)), {}, r"static shape 0 and body 0 start 0 m apart"),
        ("on another body", (box(0.125), box(0.375)), {}, r"body 0 and body 1 start 0 m apart"),
        # 1 cm cubes stacked in touching contact: in float32 these faces are 9.3e-10 m apart
        ("apart by rounding", (box(0.015, half=0.005), box(0.025, half=0.005)), {}, r"rounding of float32 positions"),
        ("cloth inside its thickness", (static_box, cloth(0.251)), {"thickness": 0.002}, r"thickness \(0.002 m\)"),
        # a free particle's thickness is its radius, a shape's its margin; a pair's is the sum of its two sides'
        (
            "particle inside its radius",
            (ground, particle(0.099, 0.1)),
            {},
            r"particle 0 and plane shape 0 start 0.099 m apart, at or inside their contact thickness \(0.1 m\)",
        ),
        (
            "two particles inside their radii",
            (particle(1.0, 0.1), particle(1.0, 0.1, x=0.15)),
            {},
            r"particle 0 and particle 1 start 0.15 m apart, at or inside their contact thickness \(0.2 m\)",
        ),
        # with a spring the scene gives the engine its contact surface, whose vertices are the particles too
        (
            "two particles inside their radii, joined by a spring",
            (particle(1.0, 0.1), particle(1.0, 0.1, x=0.15), spring),
            {},
            r"particle 0 and particle 1 start 0.15 m apart, at or inside their contact thickness \(0.2 m\)",
        ),
        (
            "box inside its margin",
            (ground, margin_box(0.13, 0.01)),
            {},
            r"body 0 and plane shape 0 start 0.005 m apart",
        ),
        ("box inside the ground's margin", (thick_ground, margin_box(0.13, 0.0)), {}, r"thickness \(0.01 m\)"),
        ("both margins", (thick_ground, margin_box(0.14, 0.01)), {}, r"start 0.015 m apart, .* thickness \(0.02 m\)"),
    ):
        with test.subTest(case=case), test.assertRaisesRegex(ValueError, pattern):
            build(*parts, **kwargs)
    # a gap above the thickness, here inside d_hat (1 mm, 1 um); a kinematic body on a static box; touching shapes of
    # one body
    build(ground, static_box, box(0.376), cloth(0.003, center=(1.0, 0.0)), thickness=0.002)
    build(box(0.125), box(0.375 + 1.0e-6))
    build(static_box, box(0.375, kinematic=True))
    # each pair's own thickness: a small particle 1 cm above a static box beside a large particle far away; a free
    # particle and a box above their radius and margins; a cloth 3 mm above the ground whatever its radius (0.1)
    build(static_box, particle(0.26, 0.001), particle(0.5, 0.1, x=3.0))
    build(ground, particle(0.101, 0.1, x=1.0), margin_box(0.136, 0.01))
    build(thick_ground, margin_box(0.146, 0.01))

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
    # libuipc's body stiffness: the closed forms are rigid ones, and with the default stiffness a box of unequal sides
    # that a torque starts turning runs 7e-4 of its turn ahead of a rigid one after a step (class docstring, **Rigid
    # bodies**)
    solver = newton.solvers.SolverIPC(
        model, newton_velocity_tol=1e-6, linear_tol_rate=1e-12, semi_implicit=False, body_kappa=1.0e8
    )
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


# ---------------------------------------------------------------------------- contact elements and materials

# converged tightly: free fall and sliding follow their closed forms to rounding
_TIGHT = {"newton_velocity_tol": 1e-7, "linear_tol_rate": 1e-12, "semi_implicit": False}


def test_free_joint_force_pushes_the_parent(test, device):
    """joint_f on the free joint of a body whose parent is another body, without gravity: the child is pushed by the
    force, as implicit Euler from rest gives it, and the parent by the reaction, so that the pair's momentum stays 0
    (as SolverXPBD applies it)."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Y)
    parent = builder.add_link(xform=wp.transform((1.0, 0.0, 0.0), wp.quat_identity()))
    builder.add_shape_box(parent, hx=0.1, hy=0.1, hz=0.1)
    child = builder.add_link(xform=wp.transform((1.5, 0.0, 0.0), wp.quat_identity()))
    builder.add_shape_box(child, hx=0.1, hy=0.1, hz=0.1)
    builder.add_articulation([builder.add_joint_free(parent), builder.add_joint_free(child, parent=parent)])
    model = builder.finalize(device=device)
    control = model.control()
    f = np.zeros(model.joint_dof_count, np.float32)
    f[6:9] = (0.0, 2.0, 0.0)  # the force of the child's free joint, at its centre of mass
    control.joint_f.assign(f)
    h, steps = 0.01, 20
    qd = []
    s0, s1 = model.state(), model.state()
    solver = newton.solvers.SolverIPC(model, newton_velocity_tol=1e-6, linear_tol_rate=1e-12, semi_implicit=False)
    for _ in range(steps):
        solver.step(s0, s1, control, None, h)
        s0, s1 = s1, s0
        qd.append(s0.body_qd.numpy().copy())
    qd = np.array(qd)
    mass = model.body_mass.numpy()
    momentum = mass[parent] * qd[:, parent, 0:3] + mass[child] * qd[:, child, 0:3]
    np.testing.assert_allclose(momentum, 0.0, atol=1e-6)
    n = np.arange(1, steps + 1)
    np.testing.assert_allclose(qd[:, child, 1], 2.0 * n * h / mass[child], rtol=1e-5)


def _engine_z(solver, body):
    """The float64 height of a body's origin in the solver (its translation pseudo-vertex)."""
    return float(solver._engine.positions()[solver._scene.layout().body_base[body], 2])


def _free_fall(model, body, k, dt=0.01, g=9.81):
    """Implicit Euler's height of a body's origin after k steps of free fall from rest, from the model's (float32)
    pose. Under ``_TIGHT`` a step ends within 1e-9 m of its converged position, and the differences carry over to the
    later steps through the velocity: 30 steps stay within 1e-6 m, where a contact would move a body by millimetres."""
    return float(model.body_q.numpy()[body, 2]) - g * dt * dt * k * (k + 1) / 2.0


def test_friction_of_each_pair(test, device):
    """Two boxes of friction 0.1 and 0.3 and a particle (soft_contact_mu 0.2) slide over a ground of friction 0.5
    under gravity tilted to tan 0.8: each accelerates with g_t - mu g_n, mu the arithmetic mean of the pair's two
    coefficients (0.3, 0.4 and 0.35), and a box of friction 1.3 (pair 0.9) sticks."""
    g, tan_theta, dt = 9.81, 0.8, 0.01
    th = np.arctan(tan_theta)
    g_t, g_n = g * np.sin(th), g * np.cos(th)
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    boxes = []
    for k, mu in enumerate((0.1, 0.3, 1.3)):  # side by side across the slope
        b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.5 * k, 0.052), wp.quat_identity()))
        builder.add_shape_box(b, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(mu=mu))
        boxes.append(b)
    particle = builder.add_particle(wp.vec3(0.0, 2.0, 0.004), wp.vec3(0.0, 0.0, 0.0), 1.0e-3, radius=0.002)
    model = builder.finalize(device=device)
    model.soft_contact_mu = 0.2
    model.set_gravity((g_t, 0.0, -g_n))
    solver = newton.solvers.SolverIPC(model, d_hat=1.0e-3, **_TIGHT)
    s0, s1 = model.state(), model.state()
    v_box, v_particle = [], []
    for _ in range(80):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        v_box.append(s0.body_qd.numpy()[boxes, 0].astype(np.float64))
        v_particle.append(float(s0.particle_qd.numpy()[particle, 0]))
    # the steps after the bodies land, from their velocities (float32: about 1e-7 m/s)
    a_box = (v_box[79] - v_box[19]) / (60 * dt)
    a_particle = (v_particle[79] - v_particle[19]) / (60 * dt)
    np.testing.assert_allclose(a_box[:2], [g_t - 0.3 * g_n, g_t - 0.4 * g_n], rtol=1e-4)
    test.assertAlmostEqual(a_particle / (g_t - 0.35 * g_n), 1.0, delta=1e-5)
    test.assertLess(abs(v_box[79][2]), 0.01)  # the friction's smoothing velocity, eps_velocity


def test_rest_heights_follow_radius_and_margin(test, device):
    """Contact thickness: a free particle of radius r comes to rest between r and r + d_hat above the ground, and one
    of radius 0.02 on a kinematic particle of radius 0.01 between 0.03 and 0.03 + d_hat above it; a box of margin
    0.002 rests between the sum of its margin and the ground's (0.003) and that plus d_hat above the ground, and as
    far above a static box of margin 0.004 beside it; a cloth of particle radius 0.2 rests within d_hat of the
    ground, its thickness (the solver's, 0) being independent of the radius."""
    d_hat, dt = 1.0e-3, 0.01
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(margin=0.003))
    builder.add_shape_box(-1, xform=wp.transform(wp.vec3(1.0, 0.0, 0.05), wp.quat_identity()), hx=0.1, hy=0.1, hz=0.05,
                          cfg=newton.ModelBuilder.ShapeConfig(margin=0.004))  # fmt: skip
    on_ground = builder.add_particle(wp.vec3(0.0, 0.0, 0.015), wp.vec3(0.0, 0.0, 0.0), 1.0e-3, radius=0.01)
    below = builder.add_particle(wp.vec3(0.0, 0.5, 0.2), wp.vec3(0.0, 0.0, 0.0), 0.0, radius=0.01)  # kinematic
    above = builder.add_particle(wp.vec3(0.0, 0.5, 0.235), wp.vec3(0.0, 0.0, 0.0), 1.0e-3, radius=0.02)
    margin = newton.ModelBuilder.ShapeConfig(margin=0.002)
    boxes = []
    for x, z in ((0.5, 0.06), (1.0, 0.16)):  # 1 cm above the surfaces of the ground and the static box
        boxes.append(builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, z), wp.quat_identity())))
        builder.add_shape_box(boxes[-1], hx=0.05, hy=0.05, hz=0.05, cfg=margin)
    n = builder.particle_count
    builder.add_cloth_grid(
        pos=wp.vec3(-1.2, -0.1, 0.005),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=4,
        dim_y=4,
        cell_x=0.05,
        cell_y=0.05,
        mass=1.0e-3,
        tri_ke=1.0e3,
        tri_ka=1.0e3,
        tri_kd=0.0,
        particle_radius=0.2,
    )
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, d_hat=d_hat)
    s0, s1 = model.state(), model.state()
    for _ in range(100):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
    q = s0.particle_q.numpy().astype(np.float64)
    z = s0.body_q.numpy()[:, 2].astype(np.float64) - 0.05  # the boxes' bottoms
    for name, value, low in (
        ("particle on the ground", q[on_ground, 2], 0.01 + 0.003),
        ("particle on a particle", np.linalg.norm(q[above] - q[below]), 0.03),
        ("box on the ground", z[0], 0.002 + 0.003),
        ("box on a box", z[1] - 0.1, 0.002 + 0.004),
        ("cloth on the ground", q[n:, 2].min(), 0.003),
    ):
        with test.subTest(name=name):
            test.assertGreater(value, low)
            test.assertLess(value, low + d_hat)


def _implicit_euler_oscillator(u0, omega, dt, steps, w0=0.0):
    """u_1 .. u_steps of u'' = -omega^2 u from (u0, w0) under implicit Euler: u_{n+1} (1 + omega^2 dt^2) = u_n + dt w_n,
    w_{n+1} = (u_{n+1} - u_n) / dt."""
    u, w, out = u0, w0, []
    for _ in range(steps):
        u_next = (u + dt * w) / (1.0 + (omega * dt) ** 2)
        u, w = u_next, (u_next - u) / dt
        out.append(u)
    return np.array(out)


def test_springs_oscillate_as_implicit_euler(test, device):
    """Without gravity, two free particles of mass m joined by a spring of stiffness k along x, started 1 cm short of
    its rest length, and a free particle joined to a kinematic one, 1 cm beyond it: the stretch of each follows the
    implicit-Euler oscillator of omega^2 = 2 k / m and k / m frame by frame, and the free pair's centre stays put."""
    m, k, L0, dt, steps = 0.01, 4.0, 0.1, 0.01, 60  # a period and a half of the pair: 2 pi / omega = 0.22 s
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for x in (0.0, L0 - 0.01):
        builder.add_particle(wp.vec3(x, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.0), m, radius=0.001)
    builder.add_particle(wp.vec3(0.0, 1.0, 1.0), wp.vec3(0.0, 0.0, 0.0), 0.0, radius=0.001)  # kinematic
    builder.add_particle(wp.vec3(L0 + 0.01, 1.0, 1.0), wp.vec3(0.0, 0.0, 0.0), m, radius=0.001)
    for spring, i in enumerate((0, 2)):
        builder.add_spring(i, i + 1, k, 0.0, 0.0)
        builder.spring_rest_length[spring] = L0  # add_spring took the distance at the start
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, **_TIGHT)
    rest = model.spring_rest_length.numpy().astype(np.float64)  # float32, as the positions
    x0 = solver._engine.positions()[:4, 0]
    stretch = {first: x0[first + 1] - x0[first] - rest[spring] for spring, first in enumerate((0, 2))}
    expected = {
        0: _implicit_euler_oscillator(stretch[0], np.sqrt(2.0 * k / m), dt, steps),
        2: _implicit_euler_oscillator(stretch[2], np.sqrt(k / m), dt, steps),
    }
    s0, s1 = model.state(), model.state()
    measured = {first: [] for first in expected}
    for _ in range(steps):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        x = solver._engine.positions()[:4, 0]  # float64
        for spring, first in enumerate(expected):
            measured[first].append(x[first + 1] - x[first] - rest[spring])
        test.assertAlmostEqual(x[0] + x[1], x0[0] + x0[1], delta=1e-12)
    # to the Newton tolerance of _TIGHT, 1e-9 m per step
    for first, u in expected.items():
        np.testing.assert_allclose(measured[first], u, rtol=0.0, atol=1e-9)


def _box_on_ramp(builder, center, angle, mu, half=0.05, gap=0.002):
    """A box of friction ``mu`` resting ``gap`` above the top of a ramp tilted by ``angle`` about x whose top centre is
    ``center``; returns the body."""
    q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(angle))
    up = np.array(wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0)))
    b = builder.add_body(xform=wp.transform(wp.vec3(*(np.asarray(center) + (half + gap) * up)), q))
    builder.add_shape_box(b, hx=half, hy=half, hz=half, cfg=newton.ModelBuilder.ShapeConfig(mu=mu))
    return b


def test_ramp_sticks_and_slips(test, device):
    """Boxes on a static ramp of 20 degrees (friction 0.3): one of friction 0.2 (pair 0.25, below tan 20 = 0.364)
    slides down with g (sin - 0.25 cos), one of friction 0.6 (pair 0.45) sticks.

    The sliding box moves away from the diagonal that splits the ramp's top face (along x = y in the ramp's frame).
    A box that crosses it meets edge-edge pairs besides its point-triangle pairs there, which change its acceleration
    over a few steps (by 2% to 4% over ten steps at this ``d_hat``).
    """
    g, angle, dt = 9.81, np.radians(20.0), 0.01
    q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(angle))
    up = np.array(wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0)))
    down = -np.array(wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0)))  # down the slope
    builder = newton.ModelBuilder()
    ramp_center = np.array([0.0, 0.0, 0.5])
    builder.add_shape_box(
        -1,
        xform=wp.transform(wp.vec3(*ramp_center), q),
        hx=0.5,
        hy=0.5,
        hz=0.05,
        cfg=newton.ModelBuilder.ShapeConfig(mu=0.3),
    )
    top = ramp_center + 0.05 * up
    slides = _box_on_ramp(builder, top + np.array([0.2, 0.0, 0.0]), angle, 0.2)
    sticks = _box_on_ramp(builder, top + np.array([-0.2, 0.0, 0.0]), angle, 0.6)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, d_hat=1.0e-3, **_TIGHT)
    s0, s1 = model.state(), model.state()
    v = []
    for _ in range(60):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        v.append(s0.body_qd.numpy()[[slides, sticks], :3].astype(np.float64) @ down)
    a = (v[59][0] - v[19][0]) / (40 * dt)
    test.assertAlmostEqual(a / (g * (np.sin(angle) - 0.25 * np.cos(angle))), 1.0, delta=2e-4)
    test.assertLess(abs(v[59][1]), 0.01)


def test_excluded_pairs_never_collide(test, device):
    """A box over a static table falls through it as in free fall, without a contact candidate in any step, where
    Newton's rules exclude the pair: collision groups 2 and 3, one negative group (-1 and -1), or a filter pair. In
    groups 2 and 2, or -1 and -2, the box comes to rest on the table."""
    dt, half, table_top = 0.01, 0.05, 0.1
    z0 = table_top + half + 0.002
    for box_group, table_group, filtered, falls in (
        (2, 3, False, True),
        (-1, -1, False, True),
        (1, 1, True, True),
        (2, 2, False, False),
        (-1, -2, False, False),
    ):
        with test.subTest(box_group=box_group, table_group=table_group, filtered=filtered):
            builder = newton.ModelBuilder()
            table = builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 * table_top), wp.quat_identity()),
                hx=0.3,
                hy=0.3,
                hz=0.5 * table_top,
                cfg=newton.ModelBuilder.ShapeConfig(collision_group=table_group),
            )
            body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, z0), wp.quat_identity()))
            box = builder.add_shape_box(
                body, hx=half, hy=half, hz=half, cfg=newton.ModelBuilder.ShapeConfig(collision_group=box_group)
            )
            if filtered:
                builder.add_shape_collision_filter_pair(table, box)
            model = builder.finalize(device=device)
            test.assertEqual(model.shape_contact_pair_count, 0 if falls else 1)
            solver = newton.solvers.SolverIPC(model, d_hat=1.0e-3, **_TIGHT)
            s0, s1 = model.state(), model.state()
            candidates = 0
            for k in range(1, 31):
                solver.step(s0, s1, None, None, dt)
                s0, s1 = s1, s0
                candidates += solver.frame_stats()["contact_candidates"]
                if falls:
                    test.assertAlmostEqual(_engine_z(solver, body), _free_fall(model, body, k), delta=1e-6)
            if falls:
                test.assertEqual(candidates, 0)
            else:
                test.assertGreater(candidates, 0)
                z = float(s0.body_q.numpy()[body, 2])
                test.assertGreater(z, table_top + half)
                test.assertLess(z, table_top + half + 1.0e-3)


def test_particle_only_shapes(test, device):
    """A shape that collides with particles only (``has_shape_collision=False``, which keeps COLLIDE_PARTICLES):
    a static box of that kind holds a particle on its top and lets a box fall through it as in free fall. Two such
    boxes on bodies, which start inside each other, fall freely through each other and through the ground, whether
    the model has particles or not (without particles they are not even triangulated)."""
    dt = 0.01
    particle_only = newton.ModelBuilder.ShapeConfig(has_shape_collision=False)
    builder = newton.ModelBuilder()
    builder.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()), hx=0.5, hy=0.5, hz=0.05,
                          cfg=particle_only)  # fmt: skip
    particle = builder.add_particle(wp.vec3(-0.25, 0.0, 0.104), wp.vec3(0.0, 0.0, 0.0), 1.0e-3, radius=0.002)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.25, 0.0, 0.152), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, d_hat=1.0e-3, **_TIGHT)
    s0, s1 = model.state(), model.state()
    for k in range(1, 31):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        test.assertAlmostEqual(_engine_z(solver, body), _free_fall(model, body, k), delta=1e-6)
    z = float(s0.particle_q.numpy()[particle, 2])
    test.assertGreater(z, 0.1)
    test.assertLess(z, 0.1 + 1.0e-3 + 0.002)

    for with_particle in (False, True):
        with test.subTest(with_particle=with_particle):
            builder = newton.ModelBuilder()
            bodies = []
            for _ in range(2):
                bodies.append(builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.15), wp.quat_identity())))
                builder.add_shape_box(bodies[-1], hx=0.1, hy=0.1, hz=0.1, cfg=particle_only)
            builder.add_ground_plane()
            if with_particle:
                builder.add_particle(wp.vec3(1.0, 0.0, 2.0), wp.vec3(0.0, 0.0, 0.0), 1.0, radius=0.01)
            model = builder.finalize(device=device)
            solver = newton.solvers.SolverIPC(model, **_TIGHT)
            s0, s1 = model.state(), model.state()
            for k in range(1, 21):
                solver.step(s0, s1, None, None, dt)
                s0, s1 = s1, s0
                for b in bodies:  # 5.6 cm below the ground at the end
                    test.assertAlmostEqual(_engine_z(solver, b), _free_fall(model, b, k), delta=1e-6)


def test_inactive_particles_do_not_collide(test, device):
    """A box falls onto a particle: through it as in free fall when the particle is not ACTIVE (Newton gives an
    inactive particle no contact), onto it when the particle is active and kinematic (zero mass)."""
    dt, z0 = 0.01, 0.054
    for active in (False, True):
        with test.subTest(active=active):
            builder = newton.ModelBuilder()
            flags = int(newton.ParticleFlags.ACTIVE) if active else 0
            builder.add_particle(wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0), 0.0, radius=0.001, flags=flags)
            body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, z0), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
            model = builder.finalize(device=device)
            solver = newton.solvers.SolverIPC(model, d_hat=1.0e-3, **_TIGHT)
            s0, s1 = model.state(), model.state()
            for k in range(1, 21):
                solver.step(s0, s1, None, None, dt)
                s0, s1 = s1, s0
                if not active:
                    test.assertAlmostEqual(_engine_z(solver, body), _free_fall(model, body, k), delta=1e-6)
            if active:
                test.assertGreater(float(s0.body_q.numpy()[body, 2]), 0.05)  # held above the particle


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
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=mu))
        model = builder.finalize(device=device)
        model.soft_contact_mu = mu  # the pair's friction is the mean of the two
        # entries [world 0, global world -1]: without begin_world everything is global
        mask = {
            "all": None,
            "global world": wp.array([False, True], dtype=wp.bool, device=device),
            "world 0 only": wp.array([True, False], dtype=wp.bool, device=device),
        }.get(case)
        solver = newton.solvers.SolverIPC(model, newton_velocity_tol=1e-4, semi_implicit=False)
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


def test_reset_checks_world_mask_as_newton_does(test, device):
    """reset checks world_mask with SolverIPC's copy of Newton's check, which has no public name: for a model of one
    world and one of three coupled worlds, it accepts the masks that SolverBase.reset accepts and raises the same
    error for the others (another length, dtype, rank or device, a NumPy array, a list)."""

    def outcome(solver, state, mask):
        try:
            solver.reset(state, world_mask=mask)
        except (TypeError, ValueError) as error:
            return type(error).__name__, str(error)
        return "accepted"

    for copies in (1, 3):
        with test.subTest(copies=copies):
            model = _worlds(device, copies)
            n = model.world_count + 1
            masks = [
                None,
                wp.array([True] * n, dtype=wp.bool, device=device),
                wp.zeros(n, dtype=wp.bool, device=device),
                wp.ones(n - 1, dtype=wp.bool, device=device),
                wp.ones(n + 1, dtype=wp.bool, device=device),
                wp.ones(n, dtype=wp.int32, device=device),
                wp.ones(n, dtype=wp.uint8, device=device),
                wp.ones((n, 1), dtype=wp.bool, device=device),
                wp.ones(n, dtype=wp.bool, device="cpu"),
                np.ones(n, dtype=bool),
                [True] * n,
            ]
            state = model.state()
            expected = [outcome(newton.solvers.SolverBase(model), state, m) for m in masks]
            test.assertEqual(expected.count("accepted"), 3)
            solver = newton.solvers.SolverIPC(model, coupled_worlds=copies > 1)
            test.assertEqual([outcome(solver, state, m) for m in masks], expected)


def _hanging_cloth(device):
    """A 32 x 32 cloth of 1 cm cells hanging from its top edge, without contact."""
    builder = newton.ModelBuilder()
    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, 1.0), rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * np.pi),
        vel=wp.vec3(0.0, 0.0, 0.0), dim_x=32, dim_y=32, cell_x=0.01, cell_y=0.01, mass=1.0e-4,
        tri_ke=1.0e4, tri_ka=1.0e4, tri_kd=0.0, edge_ke=1.0e-3, edge_kd=0.0, fix_top=True,
    )  # fmt: skip
    return builder.finalize(device=device)


def test_graph_outdated_after_long_solves(test, device):
    """graph_outdated() takes the iterations of the captured steps replayed since it was last called. A cloth hanging
    from its top edge, solved to 1e-12, takes about 270 PCG iterations per Newton iteration: the first call after
    replays returns True, the steps captured from then on re-read the loop's test every 32 iterations instead of 8,
    and they end where the steps of the old graph end, to 1e-6. A soft cube falling freely, solved to 1e-3, takes
    fewer than 128, and the loop stays."""
    for long in (True, False):
        with test.subTest(long_solves=long):
            if long:
                model = _hanging_cloth(device)
            else:
                model = _soft_cube(device, height=0.5, ground=False)
            outdated = long
            args = {"linear_tol_rate": 1e-12 if long else 1e-3, "newton_velocity_tol": 1e-4}
            s0, s1 = model.state(), model.state()
            newton.solvers.SolverIPC(model, **args).step(s0, s1, None, None, 0.01)  # loads the kernels
            solver = newton.solvers.SolverIPC(model, **args)
            test.assertFalse(solver.graph_outdated())  # nothing captured yet
            pcg = solver._engine.solver.pcg
            s0, s1 = model.state(), model.state()
            with wp.ScopedCapture(device=device) as capture:
                solver.step(s0, s1, None, None, 0.01)
                solver.step(s1, s0, None, None, 0.01)
            wp.capture_launch(capture.graph)
            after = (s0.particle_q.numpy(), s0.particle_qd.numpy())
            wp.capture_launch(capture.graph)
            test.assertEqual(solver.graph_outdated(), outdated)
            test.assertEqual(pcg.iterations_per_check, 32 if outdated else 8)
            test.assertFalse(solver.graph_outdated())  # the counts were taken
            if outdated:
                old = s0.particle_q.numpy()
                s0.particle_q.assign(after[0])
                s0.particle_qd.assign(after[1])
                with wp.ScopedCapture(device=device) as capture:
                    solver.step(s0, s1, None, None, 0.01)
                    solver.step(s1, s0, None, None, 0.01)
                wp.capture_launch(capture.graph)
                np.testing.assert_allclose(s0.particle_q.numpy(), old, rtol=0.0, atol=1e-6)


def test_graph_outdated_takes_the_ladder_guesses(test, device):
    """graph_outdated() also answers for the size ladders, whose guess a captured step holds (libuipc-warp's
    SimEngine.graph_outdated and prepare_capture). With libuipc-warp's RECAPTURE_NODES at 8, replays of the hanging
    cloth, whose matrix conversions take no contact block, ask for the guess of that size: the first call returns
    True and the engine takes the guess, a second call without new steps returns False, and so does a call after
    replays of the steps captured on the guess. The PCG loop's own choice is off (``check_bodies = None``), so that
    only the ladders ask. Without prepare_capture the second call would ask again."""
    from uipc_warp.backend.engine import sim_engine  # noqa: PLC0415

    model = _hanging_cloth(device)
    s0, s1 = model.state(), model.state()
    newton.solvers.SolverIPC(model).step(s0, s1, None, None, 0.01)  # loads the kernels

    def replays(solver):
        s0, s1 = model.state(), model.state()
        with wp.ScopedCapture(device=device) as capture:
            solver.step(s0, s1, None, None, 0.01)
            solver.step(s1, s0, None, None, 0.01)
        for _ in range(3):
            wp.capture_launch(capture.graph)

    def built():
        solver = newton.solvers.SolverIPC(model)
        solver._engine.solver.pcg.check_bodies = None  # the loop body stays: only the ladders ask for captures
        return solver

    with mock.patch.object(sim_engine, "RECAPTURE_NODES", 8):
        solver = built()
        ladder = solver._engine.solver.converter.ladder
        test.assertGreaterEqual(ladder.depth, 2)
        replays(solver)
        test.assertIsNone(ladder.guess)
        test.assertTrue(solver.graph_outdated())
        test.assertEqual(ladder.sizes[ladder.guess], 0)
        test.assertFalse(solver.graph_outdated())  # a second check without new steps
        replays(solver)
        test.assertFalse(solver.graph_outdated())  # the steps captured on the guess
        # the negative control: an engine that does not take the guesses keeps asking
        solver = built()
        with mock.patch.object(type(solver._engine), "prepare_capture", lambda engine: None):
            replays(solver)
            test.assertTrue(solver.graph_outdated())
            test.assertTrue(solver.graph_outdated())


def test_default_control_is_built_once(test, device):
    """step(control=None) builds the model's Control once, not at every step, and again when the model holds other
    arrays for the inputs the solver reads (here joint_f, which then drives a hinge)."""
    model, state, _ = _hinged(device)
    solver = newton.solvers.SolverIPC(model, **_CONVERGED)
    built = []
    control = model.control

    def counted(*args, **kwargs):
        built.append(1)
        return control(*args, **kwargs)

    model.control = counted
    try:
        s0, s1 = state, model.state()
        for _ in range(3):
            solver.step(s0, s1, None, None, 0.01)
            s0, s1 = s1, s0
        test.assertEqual(len(built), 1)
        test.assertAlmostEqual(float(s0.joint_q.numpy()[0]), 0.0, delta=1e-7)
        model.joint_f = wp.array([2.0], dtype=wp.float32, device=device)
        solver.step(s0, s1, None, None, 0.01)
        test.assertEqual(len(built), 2)
        test.assertGreater(float(s1.joint_q.numpy()[0]), 0.0)
    finally:
        del model.control


def test_notify_updates_gravity_and_inertia_in_place(test, device):
    """Gravity, mass and inertia changed after a capture take effect in its replays.

    A box under a constant force and torque, and a particle: their accelerations follow the new gravity and the
    box's doubled mass and inertia without a new capture.
    """
    builder = newton.ModelBuilder()
    builder.add_particle(wp.vec3(0.0, 1.0, 1.0), wp.vec3(0.0, 0.0, 0.0), 0.5)
    b = _box_body(builder, (0.0, 0.0, 1.0), (0.1, 0.05, 0.08))
    model = builder.finalize(device=device)
    # libuipc's body stiffness, for the rigid closed form (see test_external_forces_match_closed_form)
    solver = newton.solvers.SolverIPC(
        model, newton_velocity_tol=1e-6, linear_tol_rate=1e-12, semi_implicit=False, body_kappa=1.0e8
    )
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


@wp.kernel
def _newton_heightfield_triangles(
    fields: wp.array[HeightfieldData],
    elevations: wp.array[wp.float32],
    xform: wp.transform,
    out: wp.array2d[wp.vec3],
):
    """The world vertices of triangle ``tid`` of a height field, as Newton's narrow phase takes them."""
    t = wp.tid()
    shape, v0 = get_triangle_shape_from_heightfield(fields[0], elevations, xform, t)
    q = wp.transform_get_rotation(xform)
    out[t, 0] = v0
    out[t, 1] = v0 + wp.quat_rotate(q, shape.scale)
    out[t, 2] = v0 + wp.quat_rotate(q, shape.auxiliary)


def test_heightfield_triangles_and_rest(test, device):
    """A height field is the triangle mesh Newton's narrow phase collides with: its triangles in the solver are those
    of get_triangle_shape_from_heightfield, in the field's scaled and moved frame. Spheres dropped on its two
    plateaus (heights 0 and 0.2 of a scaled field, moved up by 0.1) rest there, within d_hat of touching."""
    d_hat, radius = 1.0e-3, 0.08
    data = np.zeros((9, 9), np.float32)
    data[:, 4] = 0.5
    data[:, 5:] = 1.0
    field = newton.Heightfield(data=data, nrow=9, ncol=9, hx=0.5, hy=0.5, min_z=0.0, max_z=0.1)
    xform = wp.transform(wp.vec3(0.3, 0.0, 0.1), wp.quat_identity())
    builder = newton.ModelBuilder()
    builder.add_shape_heightfield(xform=xform, heightfield=field, scale=(2.0, 2.0, 2.0))
    # the plateaus: x below -0.25 and above 0.25 in the field's frame, at 0 and 0.2 there
    spots = [(-0.6, -0.5, 0.0), (-0.6, 0.5, 0.0), (0.6, 0.0, 0.2)]
    bodies = []
    for x, y, top in spots:
        bodies.append(
            builder.add_body(xform=wp.transform(wp.vec3(0.3 + x, y, 0.1 + top + radius + 0.005), wp.quat_identity()))
        )
        builder.add_shape_sphere(bodies[-1], radius=radius)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, d_hat=d_hat)

    expected = wp.empty((2 * 8 * 8, 3), dtype=wp.vec3, device=device)
    wp.launch(
        _newton_heightfield_triangles,
        len(expected),
        inputs=[model.heightfield_data, model.heightfield_elevations, xform],
        outputs=[expected],
        device=device,
    )
    layout = solver._scene.layout()
    first, end = layout.obstacle_offsets[0], layout.obstacle_offsets[1]
    tris = np.asarray(solver._engine.surface["triangles"])
    tris = tris[(tris >= first).all(axis=1) & (tris < end).all(axis=1)]
    x = solver._engine.positions()

    def rows(t):  # the triangles' vertices in order, sorted by triangle
        t = np.asarray(t, np.float64).reshape(-1, 9)
        return t[np.lexsort(t.T[::-1])]

    np.testing.assert_allclose(rows(x[tris]), rows(expected.numpy()), rtol=0.0, atol=1e-6)

    s0, s1 = model.state(), model.state()
    for _ in range(60):
        solver.step(s0, s1, None, None, 0.01)
        s0, s1 = s1, s0
    q = s0.body_q.numpy()
    for b, (x, y, top) in zip(bodies, spots, strict=True):
        np.testing.assert_allclose(q[b, :2], [0.3 + x, y], atol=1e-3)
        test.assertGreater(q[b, 2], 0.1 + top + radius)
        test.assertLess(q[b, 2], 0.1 + top + radius + d_hat)


def _sheet(builder, x, tris, rows=None, rest=None, mass=0.01, tri_k=1.0e3, edge_ke=30.0):
    """Particles at ``x`` with the triangles ``tris`` and, given, the bending edge ``rows`` (o0, o1, v1, v2) with their
    rest angles (None: Newton measures them at ``x``), without damping."""
    n = len(x)
    builder.add_particles([wp.vec3(*p) for p in x], [wp.vec3(0.0, 0.0, 0.0)] * n, [mass] * n, radius=[0.001] * n)
    t = np.asarray(tris)
    builder.add_triangles(
        t[:, 0], t[:, 1], t[:, 2], tri_ke=[tri_k] * len(t), tri_ka=[tri_k] * len(t), tri_kd=[0.0] * len(t)
    )
    if rows is not None:
        r = np.asarray(rows)
        builder.add_edges(
            r[:, 0], r[:, 1], r[:, 2], r[:, 3], rest=rest, edge_ke=[edge_ke] * len(r), edge_kd=[0.0] * len(r)
        )


def _newton_angles(model, x):
    """Newton's angle of every bending edge row at the positions ``x``, as it measures rest angles (SolverIPC's copy of
    Newton's helper, which test_rest_shape_copies_match_newtons_builders holds to the builders)."""
    from newton._src.solvers.ipc.solver_ipc import _dihedral_angles  # noqa: PLC0415

    e = model.edge_indices.numpy().reshape(-1, 4)
    x = np.asarray(x, np.float64)
    return _dihedral_angles(x[e[:, 0]], x[e[:, 1]], x[e[:, 2]], x[e[:, 3]], x[e[:, 2]], x[e[:, 3]])


def test_tet_lame_parameters_reach_the_engine_as_libuipc_stores_them(test, device):
    """k_mu and k_lambda are Lame parameters. libuipc's StableNeoHookean.apply_to stores mu and lambda + mu, the SNK1
    parameters that linearize to Hooke's law with those Lame parameters; the engine gets the same pair, so a block
    with k_mu = k_lambda has Poisson's ratio 0.25, not 0."""
    from uipc_warp.backend.finite_element.constitutions.stable_neo_hookean_3d import (  # noqa: PLC0415
        StableNeoHookean3D,
    )

    builder = newton.ModelBuilder()
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, 0.5),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=2,
        dim_y=2,
        dim_z=2,
        cell_x=0.1,
        cell_y=0.1,
        cell_z=0.1,
        density=1000.0,
        k_mu=2.0e5,
        k_lambda=5.0e5,
        k_damp=0.0,
    )
    solver = newton.solvers.SolverIPC(builder.finalize(device=device))
    elements = solver._scene.engine_inputs(solver._config, solver.model.device)["elements"]
    snk = next(g for g in elements if isinstance(g, StableNeoHookean3D))
    np.testing.assert_allclose(snk.mu.numpy(), 2.0e5)
    np.testing.assert_allclose(snk.lam.numpy(), 7.0e5)


def _rest_angles(solver):
    """The rest angles of the solver's hinges, in its own (winding) convention."""
    from uipc_warp.backend.finite_element.constitutions.discrete_shell_bending import (  # noqa: PLC0415
        DiscreteShellBending,
    )

    elements = solver._scene.engine_inputs(solver._config, solver.model.device)["elements"]
    return next(g for g in elements if isinstance(g, DiscreteShellBending)).params[3].numpy()


def test_rest_shapes_from_tri_poses_and_edge_rest_angle(test, device):
    """Newton's rest shapes, without gravity. A flat hinge whose edge_rest_angle is 0.6 folds until Newton's angle of
    its row is 0.6, and the same with the row given in another order (o0's triangle running v2 -> v1) and the rest
    angle that order measures (-0.6). A triangle whose tri_poses are scaled by 1 / 1.2 grows until its edges are 1.2
    times as long. A curved sheet gets the same rest angles from its builder's rows as from the rows given by hand in
    the other order. A hinge whose triangles run their edge the same way is refused."""
    dt, flat = 0.01, np.array([[0.5, -1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
    tris = [[0, 1, 2], [3, 2, 1]]  # (0, 1, 2) runs the edge 1 -> 2, (3, 2, 1) runs it 2 -> 1
    folded = {}
    for row, rest in (((0, 3, 1, 2), 0.6), ((3, 0, 1, 2), -0.6), ((0, 3, 2, 1), -0.6)):
        with test.subTest(row=row):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            _sheet(builder, flat, tris, [row], [rest])
            model = builder.finalize(device=device)
            solver = newton.solvers.SolverIPC(model, **_TIGHT)
            s0, s1 = model.state(), model.state()
            for _ in range(200):
                solver.step(s0, s1, None, None, dt)
                s0, s1 = s1, s0
            test.assertAlmostEqual(float(_newton_angles(model, s0.particle_q.numpy())[0]), rest, delta=1e-4)
            folded[row] = s0.particle_q.numpy()
    for x in folded.values():
        np.testing.assert_allclose(x, folded[(0, 3, 1, 2)], atol=1e-5)

    x = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.03, 0.08, 0.0]])
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    _sheet(builder, x, [[0, 1, 2]])
    model = builder.finalize(device=device)
    model.tri_poses.assign(model.tri_poses.numpy() / 1.2)
    solver = newton.solvers.SolverIPC(model, cloth_strain_rate=0.0, **_TIGHT)
    s0, s1 = model.state(), model.state()
    for _ in range(200):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
    y = s0.particle_q.numpy().astype(np.float64)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        test.assertAlmostEqual(np.linalg.norm(y[b] - y[a]) / np.linalg.norm(x[b] - x[a]), 1.2, delta=1.2e-5)

    # a 3 x 3 sheet bent around a cylinder; the same rows by hand with their opposite vertices swapped
    u, v = np.meshgrid(np.linspace(0.0, 0.3, 4), np.linspace(0.0, 0.3, 4), indexing="ij")
    curved = np.stack([0.5 * np.sin(u / 0.5), v, 0.5 * (1.0 - np.cos(u / 0.5))], axis=-1).reshape(-1, 3)
    quads = [(4 * i + j, 4 * i + j + 4, 4 * i + j + 5, 4 * i + j + 1) for i in range(3) for j in range(3)]
    sheet = [t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))]
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=[wp.vec3(*p) for p in curved],
        indices=[i for t in sheet for i in t],
        density=0.1,
        tri_ke=1.0e3,
        tri_ka=1.0e3,
        tri_kd=0.0,
        edge_ke=1.0,
        edge_kd=0.0,
    )
    given = builder.finalize(device=device)
    rows = given.edge_indices.numpy().reshape(-1, 4)
    rows = rows[(rows[:, 0] >= 0) & (rows[:, 1] >= 0)][:, [1, 0, 2, 3]]
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    _sheet(builder, curved, sheet, rows, edge_ke=1.0)
    by_hand = builder.finalize(device=device)
    angles = _rest_angles(newton.solvers.SolverIPC(given))
    test.assertGreater(np.abs(angles).max(), 0.1)  # the edges along the cylinder's axis are folded (0.2 rad)
    np.testing.assert_allclose(_rest_angles(newton.solvers.SolverIPC(by_hand)), angles, rtol=0.0, atol=1e-6)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    _sheet(builder, flat, [[0, 1, 2], [3, 1, 2]], [(0, 3, 1, 2)], [0.0])
    with test.assertRaisesRegex(ValueError, r"bending edge 0 \(particles 1 and 2.*the same direction"):
        newton.solvers.SolverIPC(builder.finalize(device=device))


def _flat_sheet(device, rot=None):
    """A 6 x 6 cloth grid of 3 cm cells built at rest at 0.9 m: flat, and Newton measures its rest angles as 0."""
    builder = newton.ModelBuilder()
    builder.add_cloth_grid(
        pos=wp.vec3(-0.1, -0.1, 0.9),
        rot=wp.quat_identity() if rot is None else rot,
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=6,
        dim_y=6,
        cell_x=0.03,
        cell_y=0.03,
        mass=0.01,
        tri_ke=1.0e3,
        tri_ka=1.0e3,
        tri_kd=0.0,
        edge_ke=5.0e-3,
        edge_kd=0.0,
    )
    return builder.finalize(device=device)


def test_cloth_at_rest_starts_without_bending_force(test, device):
    """A cloth built at rest starts at its rest angles as the engine measures them, so that no hinge pushes it out of
    its plane. Newton measures the angle of every hinge of this flat sheet as 0; libuipc's dihedral angle, the
    arccosine of the normals' cosine, reads 1.5e-8 rad on some of them, and it is the angle the engine's bending
    energy takes. The engine's rest angles are its own of the start positions (as libuipc takes them from its rest
    geometry), and its bending gradient at the start is 0. A rest angle other than the start's is Newton's
    (test_rest_shapes_from_tri_poses_and_edge_rest_angle)."""
    from uipc_warp.backend.finite_element.constitutions.discrete_shell_bending import (  # noqa: PLC0415
        DiscreteShellBending,
        hinge_rest_shape,
    )
    from uipc_warp.backend.types import mat33d, vec3d  # noqa: PLC0415

    model = _flat_sheet(device)
    test.assertFalse(np.any(model.edge_rest_angle.numpy()))
    engine = newton.solvers.SolverIPC(model)._engine
    bending = next(g for g in engine.elements if isinstance(g, DiscreteShellBending))
    stencils = bending.stencils.numpy()
    measured = hinge_rest_shape(model.particle_q.numpy().astype(np.float64), stencils)[2]
    test.assertGreater(np.count_nonzero(measured), 0)  # libuipc's angle of some flat hinges is not 0
    np.testing.assert_array_equal(bending.params[3].numpy(), measured)
    n = len(stencils)
    grad = wp.zeros(4 * n, dtype=vec3d, device=device)
    hess = wp.zeros(10 * n, dtype=mat33d, device=device)
    bending.gradient_hessian(engine.vm.x, engine.state.fs, grad, hess)
    test.assertFalse(np.any(grad.numpy()))


def test_cloth_at_rest_takes_its_rest_triangles_from_particle_q(test, device):
    """Newton's builders compute a cloth triangle's rest shape in float64 from particle_q and keep its inverse in float32
    (tri_poses). For a cloth built at rest the engine's membrane takes the rest shapes of particle_q, as libuipc
    computes them from its rest geometry: its inverse rest matrices and rest areas are libuipc's of the start, bit for
    bit, while those of the float32 tri_poses are off by their rounding, a strain the sheet would start with. A
    triangle whose tri_poses were changed keeps them; the others then take the rest shapes of particle_q to
    round-off."""
    from uipc_warp.backend.finite_element.constitutions.strain_limiting_baraff_witkin_shell_2d import (  # noqa: PLC0415
        StrainLimitingBaraffWitkinShell2D,
        slbws_rest_matrices,
        slbws_rest_shape,
    )

    def membrane(model):
        engine = newton.solvers.SolverIPC(model)._engine
        return next(g for g in engine.elements if isinstance(g, StrainLimitingBaraffWitkinShell2D))

    model = _flat_sheet(device, rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.3))
    x = model.particle_q.numpy().astype(np.float64)
    tris = model.tri_indices.numpy().reshape(-1, 3)
    IB, area = slbws_rest_shape(x, tris)
    shell = membrane(model)
    np.testing.assert_array_equal(shell.IB.numpy(), IB)
    np.testing.assert_array_equal(shell.area.numpy(), area)
    poses = model.tri_poses.numpy()
    rounded = slbws_rest_matrices(x, tris, np.linalg.inv(poses.astype(np.float64)))
    entries = IB != 0.0
    test.assertGreater(np.abs(rounded[entries] / IB[entries] - 1.0).max(), 1e-8)  # the float32 tri_poses are not it

    poses[0] /= 1.2
    model.tri_poses.assign(poses)
    shell = membrane(model)
    stretched = slbws_rest_matrices(x[tris[0]], np.array([[0, 1, 2]]), np.linalg.inv(poses[:1].astype(np.float64)))
    np.testing.assert_allclose(shell.IB.numpy()[0], stretched[0], rtol=0.0, atol=1e-12 * np.abs(IB).max())
    np.testing.assert_allclose(shell.IB.numpy()[1:], IB[1:], rtol=0.0, atol=1e-12 * np.abs(IB).max())


def test_gravity_per_world(test, device):
    """A global particle and body fall with the global world's gravity (the last entry of Model.gravity) and a
    particle and body of world 0 with world 0's, as implicit Euler from rest. New gravities of both worlds, set with
    set_gravity and notify_model_changed(MODEL_PROPERTIES) after a capture, take effect in its next replays."""
    dt = 0.01
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -2.0))  # the global world's
    bodies, particles = [], []
    for local in (False, True):
        if local:
            builder.begin_world(gravity=(0.0, 0.0, -5.0))
        x = 1.0 if local else 0.0
        particles.append(builder.add_particle(wp.vec3(x, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.0), 1.0, radius=0.01))
        bodies.append(_box_body(builder, (x + 0.5, 0.0, 1.0), (0.05, 0.05, 0.05)))
        if local:
            builder.end_world()
    model = builder.finalize(device=device)
    np.testing.assert_array_equal(model.particle_world.numpy(), [-1, 0])
    np.testing.assert_array_equal(model.body_world.numpy(), [-1, 0])
    solver = newton.solvers.SolverIPC(model, **_TIGHT)
    rows = [*particles, *solver._scene.layout().body_base[bodies]]  # the engine rows of their positions
    x = np.concatenate([model.particle_q.numpy()[particles], model.body_q.numpy()[bodies, :3]]).astype(np.float64)
    v = np.zeros_like(x)

    def check(gravity, steps):
        """Implicit Euler from the last state with the gravity of each entity's world, against the solver's."""
        nonlocal x, v
        g = np.array([gravity[-1], gravity[0], gravity[-1], gravity[0]])
        for _ in range(steps):
            v = v + g * dt
            x = x + v * dt
        np.testing.assert_allclose(solver._engine.positions()[rows], x, rtol=0.0, atol=1e-6)

    s0, s1 = model.state(), model.state()
    solver.step(s0, s1, None, None, dt)
    check({-1: (0.0, 0.0, -2.0), 0: (0.0, 0.0, -5.0)}, 1)
    with wp.ScopedCapture(device=device) as capture:
        solver.step(s1, s0, None, None, dt)
        solver.step(s0, s1, None, None, dt)
    for _ in range(2):
        wp.capture_launch(capture.graph)
    check({-1: (0.0, 0.0, -2.0), 0: (0.0, 0.0, -5.0)}, 4)
    model.set_gravity((1.0, 0.0, -3.0), world=-1)
    model.set_gravity((0.0, 2.0, -1.0), world=0)
    solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
    for _ in range(2):
        wp.capture_launch(capture.graph)
    check({-1: (1.0, 0.0, -3.0), 0: (0.0, 2.0, -1.0)}, 4)


# ---------------------------------------------------------------------------- worlds


def _world_content():
    """One world: a soft cube above the global static box, a rigid box and a free particle above the ground."""
    world = newton.ModelBuilder()
    world.add_soft_grid(
        pos=wp.vec3(0.3, -0.05, 0.25),
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
    _box_body(world, (-0.3, 0.0, 0.15), (0.05, 0.05, 0.05))
    world.add_particle(wp.vec3(0.0, 0.3, 0.1), wp.vec3(0.0, 0.0, 0.0), 0.01, radius=0.01)
    return world


def _worlds(device, copies, content=_world_content):
    """``copies`` worlds of ``content``, all at the origin as ``replicate`` places them, over a global ground plane and
    a global static box whose top is at z = 0.1."""
    scene = newton.ModelBuilder()
    scene.add_ground_plane()
    scene.add_shape_box(-1, xform=wp.transform(wp.vec3(0.35, 0.0, 0.05), wp.quat_identity()), hx=0.15, hy=0.15, hz=0.05)
    scene.replicate(content(), copies)
    return scene.finalize(device=device)


def _world_rows(model, copies):
    """The particle and body indices of every world, in the order of one world's."""
    pw, bw = model.particle_world.numpy(), model.body_world.numpy()
    return [(np.flatnonzero(pw == k), np.flatnonzero(bw == k)) for k in range(copies)]


def _pair_subscenes(solver):
    """The subscenes of the vertices of every stored, candidate and active simplex pair of the solver's last step, one
    array per list, padded with -1 to four vertices."""
    f = solver._engine.simplex
    sub = solver._scids  # the subscene of every vertex
    sv, cv = f.info.surf_verts.numpy(), f.info.codim_verts.numpy()
    ed, tr = f.info.edges.numpy().reshape(-1, 2), f.info.tris.numpy().reshape(-1, 3)
    out = []
    for lists in (f.stored_lists(), f.candidate_lists()):
        p = {k: v.reshape(-1, 2) for k, v in lists.items()}
        out += [
            np.stack([sv[p["PP"][:, 0]], cv[p["PP"][:, 1]]], axis=1),
            np.concatenate([cv[p["PE"][:, :1]], ed[p["PE"][:, 1]]], axis=1),
            np.concatenate([ed[p["EE"][:, 0]], ed[p["EE"][:, 1]]], axis=1),
            np.concatenate([sv[p["PT"][:, :1]], tr[p["PT"][:, 1]]], axis=1),
        ]
    for pairs in f.active_lists().values():
        a = np.asarray(pairs)
        out.append(a.reshape(len(a), -1) if a.size else np.zeros((0, 2), np.int64))
    return [sub[v] for v in out]


def _pairs_across_worlds(sub):
    """Of pairs given by the subscenes of their vertices: how many join two worlds, and how many join a world with
    the global entities (subscene 0)."""
    local = np.where(sub > 0, sub, 0)
    high = local.max(axis=1)
    low = np.where(sub > 0, sub, high[:, None]).min(axis=1)
    return int(np.sum(low != high)), int(np.sum((sub == 0).any(axis=1) & (high > 0)))


def test_coupled_worlds_follow_the_single_world(test, device):
    """Four copies of a world at the origin, each a world of its own, with coupled_worlds: no stored, candidate or
    active pair joins two worlds at any step, every world meets the global static box, and every world follows the run
    of the one world alone, with the same Newton iterations at every step (the copies' shares of every sum of the solve
    are equal) and positions within the rounding of those sums."""
    copies, steps, dt = 4, 60, 0.01
    runs = []
    for n in (1, copies):
        model = _worlds(device, n)
        solver = newton.solvers.SolverIPC(model, coupled_worlds=True)
        test.assertEqual(solver._worlds is None, n == 1)
        s0, s1 = model.state(), model.state()
        rows = _world_rows(model, n)
        q, body_q, newton_iterations, global_pairs = [], [], [], 0
        for _ in range(steps):
            solver.step(s0, s1, None, None, dt)
            s0, s1 = s1, s0
            newton_iterations.append(solver.frame_stats()["newton_iterations"])
            pq, bq = s0.particle_q.numpy(), s0.body_q.numpy()
            q.append([pq[p] for p, _ in rows])
            body_q.append([bq[b] for _, b in rows])
            if n > 1:
                for sub in _pair_subscenes(solver):
                    across, with_global = _pairs_across_worlds(sub)
                    test.assertEqual(across, 0)
                    global_pairs += with_global
        runs.append((np.array(q), np.array(body_q), newton_iterations))
        if n > 1:
            test.assertGreater(global_pairs, 0)
    (q1, b1, n1), (qn, bn, nn) = runs
    test.assertEqual(n1, nn)
    # measured: 1e-12 m and 0 (the float32 output hides the rounding of the shared sums)
    test.assertLess(np.abs(qn - q1).max(), 1e-6)
    test.assertLess(np.abs(bn[..., :3] - b1[..., :3]).max(), 1e-6)
    # every world rests: the soft cube on the global box (top at 0.1), the rigid box and the particle on the ground
    for k in range(copies):
        test.assertGreater(qn[-1, k, :-1, 2].min(), 0.1)
        test.assertLess(qn[-1, k, :-1, 2].min(), 0.1 + D_HAT)
        test.assertGreater(bn[-1, k, 0, 2], 0.05)
        test.assertLess(bn[-1, k, 0, 2], 0.05 + D_HAT)
        test.assertGreater(qn[-1, k, -1, 2], 0.01)


def test_coupled_worlds_are_built_as_one_world(test, device):
    """A model with content in one local world is built as without coupled_worlds: no subscenes, the same contact
    elements and ids, and the same steps."""
    model = _worlds(device, 1)
    trajectories = []
    for coupled in (False, True):
        solver = newton.solvers.SolverIPC(model, coupled_worlds=coupled)
        test.assertIsNone(solver._worlds)
        trajectories.append((solver._cids.copy(), solver._enabled.copy(), _run(model, solver, 5)))
    (c0, e0, t0), (c1, e1, t1) = trajectories
    np.testing.assert_array_equal(c0, c1)
    np.testing.assert_array_equal(e0, e1)
    np.testing.assert_allclose(t0, t1, rtol=0.0, atol=1e-7)


def test_coupled_worlds_share_filter_classes(test, device):
    """A box filtered against a second box that starts inside it, replicated over six worlds: the worlds take the
    contact elements of one world, the pairs the elements enable are Model.shape_contact_pairs, and in every world the
    smaller box falls through the larger and both come to rest on the ground."""

    def content():
        world = newton.ModelBuilder()
        _box_body(world, (0.0, 0.0, 0.3), (0.05, 0.05, 0.05))
        _box_body(world, (0.0, 0.0, 0.31), (0.03, 0.03, 0.03))
        world.add_shape_collision_filter_pair(0, 1)
        return world

    elements = []
    for copies in (1, 6):
        model = _worlds(device, copies, content)
        solver = newton.solvers.SolverIPC(model, coupled_worlds=True)
        elements.append(solver._scene._elements)
        if copies > 1:
            s0, s1 = model.state(), model.state()
            for _ in range(60):
                solver.step(s0, s1, None, None, 0.01)
                s0, s1 = s1, s0
            z = s0.body_q.numpy()[:, 2].reshape(copies, 2)
            np.testing.assert_array_less(0.05, z[:, 0])
            np.testing.assert_array_less(z[:, 0], 0.05 + D_HAT)
            np.testing.assert_array_less(0.03, z[:, 1])
            np.testing.assert_array_less(z[:, 1], 0.03 + D_HAT)
    test.assertEqual(elements[0], elements[1])


def test_coupled_worlds_status_in_every_world(test, device):
    """The status and the residual of the one solve reach every world entry, also from a captured step, and a step that
    fails in one world is reported in every world."""
    copies = 3
    model = _worlds(device, copies)
    solver = newton.solvers.SolverIPC(model, coupled_worlds=True)
    observables = solver.observables({solver.ObservableKind.STATUS, solver.ObservableKind.SOLVER_RESIDUAL})
    s0, s1 = model.state(), model.state()
    solver.step(s0, s1, None, None, 0.01, observables=observables)
    with wp.ScopedCapture(device=device) as capture:
        solver.step(s1, s0, None, None, 0.01, observables=observables)
    wp.capture_launch(capture.graph)
    status, residual = observables.status.numpy(), observables.solver_residual.numpy()
    test.assertEqual(status.tolist(), [0] * (copies + 1))
    test.assertTrue(np.all(residual[:copies] == residual[0]) and 0.0 < residual[0] <= 1.0)
    test.assertEqual(residual[copies], 0.0)  # the global entry: no global particle or body
    test.assertFalse(solver.check_status(observables))
    # the particle of world 1 written below the ground into the captured step's input: it fails, in every world
    q = s1.particle_q.numpy()
    q[_world_rows(model, copies)[1][0][-1], 2] = -0.01
    s1.particle_q.assign(q)
    wp.capture_launch(capture.graph)
    bits = observables.status.numpy()
    test.assertTrue(np.all(bits[:copies] == bits[0]) and bits[0] & int(solver.StatusBits.FAILED))
    with test.assertRaises(RuntimeError):
        solver.check_status(observables)


def test_coupled_worlds_fall_with_their_gravity(test, device):
    """A particle in each of three worlds of different gravity falls as implicit Euler with its world's gravity."""
    dt, steps = 0.01, 10
    gravities = ((0.0, 0.0, -9.81), (0.0, 0.0, -2.0), (1.0, 0.0, -5.0))
    builder = newton.ModelBuilder()
    for g in gravities:
        builder.begin_world(gravity=g)
        builder.add_particle(wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.0), 1.0, radius=0.01)
        builder.end_world()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, coupled_worlds=True, **_TIGHT)
    traj = _run(model, solver, steps, dt=dt)
    k = np.arange(1, steps + 1)[:, None, None]
    expected = np.array([[0.0, 0.0, 1.0]]) + np.array(gravities)[None] * dt * dt * (k * (k + 1) / 2.0)
    np.testing.assert_allclose(traj, expected, rtol=0.0, atol=1e-5)


def test_coupled_worlds_beyond_the_old_element_limit(test, device):
    """1100 worlds at one place, each a box and a particle dropped onto the global ground: five contact elements times
    1101 subscenes, beyond the 2048 that the engine took before it tested the two tables apart. The solver builds, its
    start check keeps the two tables (no table of their product), and in 20 steps every box and particle lands on the
    ground in its own world: no stored, candidate or active pair joins two worlds."""
    worlds = 1100
    world = newton.ModelBuilder()
    _box_body(world, (0.0, 0.0, 0.07), (0.05, 0.05, 0.05))
    world.add_particle(wp.vec3(0.2, 0.0, 0.03), wp.vec3(0.0, 0.0, 0.0), 0.01, radius=0.01)
    scene = newton.ModelBuilder()
    scene.add_ground_plane()
    scene.replicate(world, worlds)
    model = scene.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, coupled_worlds=True)
    n_el, n_sub = len(solver._enabled), len(solver._subscene_enabled)
    test.assertEqual(n_sub, worlds + 1)
    test.assertGreater(n_el * n_sub, 2048)
    test.assertEqual(solver._enabled.shape, (n_el, n_el))
    s0, s1 = model.state(), model.state()
    for _ in range(20):
        solver.step(s0, s1, None, None, 0.01)
        s0, s1 = s1, s0
        for sub in _pair_subscenes(solver):
            test.assertEqual(_pairs_across_worlds(sub)[0], 0)
    z_box, z_particle = s0.body_q.numpy()[:, 2], s0.particle_q.numpy()[:, 2]
    test.assertGreater(z_box.min(), 0.05)
    test.assertLess(z_box.max(), 0.05 + D_HAT)
    test.assertGreater(z_particle.min(), 0.01)
    test.assertLess(z_particle.max(), 0.01 + D_HAT)


def test_start_check_of_many_worlds(test, device):
    """5000 worlds of a particle at one place over the global ground: the start check keeps the contact elements' table
    and the subscenes' apart (their product would hold 6.3e8 entries) and forms no pair of two worlds, whose particles
    all overlap; the model is accepted. With world 3172's particle placed through the ground it is refused, naming
    that particle."""
    worlds = 5000
    for inside in (False, True):
        world = newton.ModelBuilder()
        world.add_particle(wp.vec3(0.0, 0.0, 0.05), wp.vec3(0.0, 0.0, 0.0), 0.01, radius=0.01)
        scene = newton.ModelBuilder()
        scene.add_ground_plane()
        scene.replicate(world, worlds)
        model = scene.finalize(device=device)
        if inside:
            q = model.particle_q.numpy()
            q[3172, 2] = 0.005
            model.particle_q.assign(q)
            with test.assertRaisesRegex(ValueError, r"particle 3172 and plane shape 0 start"):
                newton.solvers.SolverIPC(model, coupled_worlds=True)
            continue
        solver = newton.solvers.SolverIPC(model, coupled_worlds=True)
        n_el = len(solver._enabled)
        test.assertEqual(solver._enabled.shape, (n_el, n_el))
        test.assertEqual(solver._subscene_enabled.shape, (worlds + 1, worlds + 1))
        test.assertGreater(n_el * (worlds + 1), 2048)


def test_near_parts_forms_no_pair_of_two_worlds(test, device):
    """The sweep of the start check against a brute-force test of every pair of boxes, on random boxes in 40 worlds
    at one place and a few global ones, with subscene tables of the form of separate worlds (some worlds meeting
    themselves, some not, the global subscene meeting all) and one of another form: every near pair the table lets
    meet, each once, and with the form of separate worlds no pair of two worlds (the caller tests the pairs within a
    world against the table)."""
    del device
    rng = np.random.default_rng(7)
    solver = newton.solvers.SolverIPC
    for trial in range(6):
        n_sub = 41
        table = np.eye(n_sub, dtype=bool) & (rng.uniform(size=n_sub) < 0.8)[:, None]
        table[0, :] = table[:, 0] = True
        if trial == 5:  # two worlds that meet: not the form of separate worlds, every near pair kept
            table[3, 4] = table[4, 3] = True
        n = 600
        sub = np.where(rng.uniform(size=n) < 0.05, 0, rng.integers(1, n_sub, n))
        centre = rng.uniform(-0.2, 0.2, (n, 3))
        half = rng.uniform(0.0, 0.03, (n, 3))
        lo, hi = centre - half, centre + half
        reach = rng.uniform(0.0, 0.01, n)
        slack = 1e-6
        owner = types.SimpleNamespace(_subscene_enabled=table, _near_boxes=solver._near_boxes)
        got = solver._near_parts(owner, lo, hi, reach, slack, sub)
        a, b = np.triu_indices(n, 1)
        apart = np.maximum(lo[b] - hi[a], lo[a] - hi[b]).max(axis=1)
        near = apart <= reach[a] + reach[b] + slack
        brute = {(int(i), int(j)) for i, j in zip(a[near], b[near], strict=True)}
        meets = {(i, j) for i, j in brute if table[sub[i], sub[j]]}
        found = {(int(i), int(j)) for i, j in got}
        test.assertEqual(len(found), len(got))  # no pair twice
        test.assertTrue(meets <= found, trial)
        test.assertTrue(found <= brute, trial)
        if trial < 5:  # no pair of two worlds is formed: those the table refuses are pairs within a world
            test.assertTrue(all(sub[i] == sub[j] for i, j in found - meets), trial)
            test.assertTrue(all(sub[i] == 0 or sub[j] == 0 or sub[i] == sub[j] for i, j in found), trial)


# ---------------------------------------------------------------------------- joints

# a converged Newton loop and PCG, so that a joint follows a closed form to rounding
_CONVERGED = {
    "newton_velocity_tol": 1e-6,
    "newton_transrate_tol": 1e-6,
    "linear_tol_rate": 1e-12,
    "semi_implicit": False,
}


def _attached_world():
    """One world: a box 30 cm up with a particle attached 10 cm under its centre, over the global ground."""
    world = newton.ModelBuilder()
    box = _box_body(world, (0.0, 0.0, 0.3), (0.05, 0.05, 0.05))
    world.add_particle(wp.vec3(0.0, 0.0, 0.2), wp.vec3(0.0, 0.0, 0.0), 0.05, radius=0.005)
    world.add_attachment_particle_body(0, box, body_point=(0.0, 0.0, -0.1), stiffness=500.0)
    return world


def test_coupled_worlds_with_attachments(test, device):
    """Three copies of a world whose particle is attached to a falling box, each a world of its own at the origin with
    coupled_worlds: every world follows the run of the one world alone, with the same Newton iterations at every step,
    and the anchors are in their worlds' subscenes."""
    runs = []
    for n in (1, 3):
        scene = newton.ModelBuilder()
        scene.add_ground_plane()
        scene.replicate(_attached_world(), n)
        model = scene.finalize(device=device)
        solver = newton.solvers.SolverIPC(model, coupled_worlds=True)
        if n > 1:
            layout = solver._scene.layout()
            anchors = np.arange(layout.body_anchor_offsets[0], layout.body_anchor_offsets[-1])
            np.testing.assert_array_equal(solver._scids[anchors], [1, 2, 3])
        rows = _world_rows(model, n)
        s0, s1 = model.state(), model.state()
        q, iterations = [], []
        for _ in range(40):
            solver.step(s0, s1, None, None, 0.01)
            s0, s1 = s1, s0
            iterations.append(solver.frame_stats()["newton_iterations"])
            pq, bq = s0.particle_q.numpy(), s0.body_q.numpy()
            q.append([np.concatenate([pq[p].reshape(-1), bq[b][:, :3].reshape(-1)]) for p, b in rows])
        runs.append((np.array(q), iterations))
    (q1, n1), (qn, nn) = runs
    test.assertEqual(n1, nn)
    test.assertLess(np.abs(qn - q1).max(), 1e-6)
    test.assertLess(q1[-1, 0, 2], 0.1)  # it fell, the particle under the box


def _pushed_world():
    """One world: a box on the global ground and a kinematic box 2 cm behind it."""
    world = newton.ModelBuilder()
    _box_body(world, (0.0, 0.0, 0.052), (0.05, 0.05, 0.05))
    pusher = world.add_body(xform=wp.transform(wp.vec3(-0.12, 0.0, 0.052), wp.quat_identity()), is_kinematic=True)
    world.add_shape_box(pusher, hx=0.05, hy=0.05, hz=0.05)
    return world


def test_coupled_worlds_with_kinematic_aims(test, device):
    """Three copies of a world whose kinematic box pushes a box across the ground with kinematic_motion="aim", each a
    world of its own at the origin with coupled_worlds, every pusher moved by state_in: every world follows the run of
    the one world alone, with the same Newton iterations at every step, and the boxes are pushed. The solves are
    converged, so that the worlds' runs differ by the rounding of the shared sums only: at the default tolerances
    the rounding moves where the linear and Newton solves stop, by up to 2e-6 m in 20 steps."""
    runs = []
    for n in (1, 3):
        scene = newton.ModelBuilder()
        scene.add_ground_plane()
        scene.replicate(_pushed_world(), n)
        model = scene.finalize(device=device)
        solver = newton.solvers.SolverIPC(model, coupled_worlds=True, kinematic_motion="aim", **_CONVERGED)
        rows = _world_rows(model, n)
        s0, s1 = model.state(), model.state()
        q, iterations = [], []
        for k in range(20):
            body_q, body_qd = s0.body_q.numpy(), s0.body_qd.numpy()
            for _, b in rows:
                body_q[b[1], 0] = -0.12 + 0.01 * k
                body_qd[b[1]] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            s0.body_q.assign(body_q)
            s0.body_qd.assign(body_qd)
            solver.step(s0, s1, None, None, 0.01)
            s0, s1 = s1, s0
            iterations.append(solver.frame_stats()["newton_iterations"])
            q.append([s0.body_q.numpy()[b][:, :3].reshape(-1) for _, b in rows])
        runs.append((np.array(q), iterations))
    (q1, n1), (qn, nn) = runs
    test.assertEqual(n1, nn)
    test.assertLess(np.abs(qn - q1).max(), 1e-6)
    test.assertGreater(q1[-1, 0, 0], 0.05)  # the box was pushed


def _reset_world():
    """One world for the resets of one world: a box sliding at 0.2 m/s on the global static box of ``_worlds``, a
    particle sliding at 0.3 m/s on the ground, and a sphere turned about a vertical hinge by a drive, a passive spring
    and passive damping."""
    world = newton.ModelBuilder()
    box = _box_body(world, (0.35, 0.0, 0.1505), (0.05, 0.05, 0.05))
    world.body_qd[box] = (0.2, 0.0, 0.0, 0.0, 0.0, 0.0)
    world.add_particle(wp.vec3(-0.3, 0.0, 0.0101), wp.vec3(0.3, 0.0, 0.0), 0.01, radius=0.01)
    link = world.add_link(xform=wp.transform(wp.vec3(-0.1, 0.4, 0.5), wp.quat_identity()))
    world.add_shape_sphere(link, radius=0.05)
    joint = world.add_joint_revolute(
        -1,
        link,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(0.0, 0.4, 0.5), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()),
        target_ke=40.0,
        target_kd=2.0,
        stiffness=10.0,
        damping=1.0,
        target_q=0.5,
        target_qd=0.1,
        rest_q=-0.2,
    )
    world.add_articulation([joint])
    return world


def _row_worlds(solver, model):
    """The world of every row of the solver's engine: a particle's, a body's for its pseudo-vertices and surface
    vertices, -1 for the static shapes and the fixed body the world-attached joints hang from."""
    layout = solver._scene.layout()
    world = np.full(layout.n, -1, np.int64)
    world[: model.particle_count] = model.particle_world.numpy()
    body_world, offsets = model.body_world.numpy(), layout.body_vertex_offsets
    for b in range(model.body_count):
        world[layout.body_base[b] : layout.body_base[b] + 4] = body_world[b]
        world[offsets[b] : offsets[b + 1]] = body_world[b]
    return world


def _solver_snapshot(solver):
    """What a step changes: the engine's state (its dump) and the solver's own device state."""
    arrays = [solver._body_q_written, solver._body_qd_written, solver._body_hold]
    for history in solver._joint_history.values():
        arrays += list(history)
    return solver._engine.state_dict(), [(a, a.numpy().copy()) for a in arrays]


def _restore(solver, snapshot):
    """The solver at ``snapshot``: the engine recovered from its dump (the friction candidates of the next frame from
    one DCD pass at its state, the same in every recovery) and the solver's device state."""
    state, arrays = snapshot
    solver._engine.load_state_dict(state)
    for a, value in arrays:
        a.assign(value)


_FRICTION_LISTS = ("PP", "PE", "PT", "EE")  # the engine's simplex friction lists, in its friction_counts order
_PAIR_WIDTH = {"PP": 2, "PE": 3, "PT": 4, "EE": 4, "PH": 2}  # the entries of a friction pair (PH: vertex, plane)


def _frame_start(solver):
    """What the engine's last frame started with, which its solve leaves: the friction candidates (per list, vertex
    tuples; PH: vertex and plane) and the lagged records of the simplex ones (their bits), the prediction and the
    kinetic masses of every row and body, the rows the frame restarted (BDF2), and the revolute drives' terms."""
    eng = solver._engine
    s, h, c = eng.simplex, eng.half_plane, eng.contact
    counts = s.friction_counts.numpy()
    lists = {name: getattr(s, f"friction_{name}s").numpy()[: counts[k]] for k, name in enumerate(_FRICTION_LISTS)}
    lists["PH"] = h.friction_PHs.numpy()[: int(h.friction_count.numpy()[0])]
    cPT, cEE, cPE, _, _ = c.fric.capacities  # the lagged records: PT, EE, PE, PP at fixed offsets
    lagged = c.lagged.numpy().view(np.int64)
    base = {"PT": 0, "EE": cPT, "PE": cPT + cEE, "PP": cPT + cEE + cPE}
    rows = eng.subscene_restarts
    group = solver._joint_groups.revolute
    return {
        "lists": lists,
        "records": {name: lagged[base[name] : base[name] + len(lists[name])] for name in _FRICTION_LISTS},
        "x_tilde": eng.vm.x_tilde.numpy().copy(),
        "mass": eng.vm.mass.numpy().copy(),
        "body_mass": np.concatenate([eng.abd.mass.numpy()[:, None], eng.abd.mx.numpy()], axis=1),
        "restarted": None if rows is None else rows.rows.frame_requests(),
        "drive": None if group is None else np.stack([group.drv_aim.numpy(), group.drv_ratio.numpy()], axis=1),
    }


def _check_friction_kept(test, got, full, world, flagged):
    """``got``'s friction candidates are ``full``'s without those with a vertex in the worlds ``flagged``, each with
    the same lagged record, bit for bit. The order is not compared: every case recovers one engine, whose stored pairs
    carry over from the case before, and the engine's own tests compare the order on fresh engines."""
    for name, listed in full["lists"].items():
        pairs = np.asarray(listed).reshape(-1, _PAIR_WIDTH[name])
        vertices = pairs[:, :1] if name == "PH" else pairs  # a PH pair: its vertex, then its plane
        out = np.isin(world[vertices], flagged).any(axis=1)
        kept = pairs[~out]
        got_pairs = np.asarray(got["lists"][name]).reshape(-1, _PAIR_WIDTH[name])
        order, got_order = np.lexsort(kept.T[::-1]), np.lexsort(got_pairs.T[::-1])
        np.testing.assert_array_equal(got_pairs[got_order], kept[order], name)
        if name != "PH":
            np.testing.assert_array_equal(got["records"][name][got_order], full["records"][name][~out][order], name)


def _branch_steps(solver, model, snapshot, state, cases, dt=0.01, control=None, pusher=None, after_reset=None):
    """From ``snapshot``, one step for every case: ``"none"``, ``"all"`` (``reset`` without a mask), a world index
    (``reset`` of that world) or ``("state", k)`` (a new velocity for world ``k``'s particles); ``pusher`` moves the
    state's kinematic bodies before the step, and ``after_reset(case)`` looks at the solver between the reset and the
    step. Returns the frame start of every case."""
    out = {}
    for case in cases:
        _restore(solver, snapshot)
        state_in = model.state()
        state_in.assign(state)
        if pusher is not None:
            pusher(state_in)
        if case == "all":
            solver.reset(state_in)
        elif isinstance(case, int):
            mask = np.zeros(model.world_count + 1, bool)
            mask[case] = True
            solver.reset(state_in, world_mask=wp.array(mask, dtype=wp.bool, device=model.device))
        elif isinstance(case, tuple):
            qd = state_in.particle_qd.numpy()
            qd[model.particle_world.numpy() == case[1]] += (0.0, 0.05, 0.0)
            state_in.particle_qd.assign(qd)
        if after_reset is not None:
            after_reset(case)
        solver.step(state_in, model.state(), control, None, dt)
        out[case] = _frame_start(solver)
    return out


def test_reset_of_one_world_keeps_the_others(test, device):
    """Three worlds at one place with coupled_worlds, each a box sliding on the global static box, a particle sliding on
    the ground and a sphere driven about a hinge, under implicit Euler and BDF2. From one state of the solver, a step
    after no reset, after a reset of world 1, after a reset of every world, and after a new velocity written for world
    1's particle. The reset of world 1 and the new state record the friction candidates of the step without a reset but
    those with a vertex in world 1, each with its lagged record, bit for bit; the reset of every world records none.
    Under BDF2 the frame restarts world 1's rows and no other, and world 1's rows, bodies and drive take the prediction,
    the masses and the terms of the reset of every world, the other worlds' those of no reset, bit for bit; under
    implicit Euler there is no history, and the resets leave them as they are. The branches begin at the first step with
    friction candidates in every world."""
    for integrator in ("bdf1", "bdf2"):
        with test.subTest(integrator=integrator):
            model = _worlds(device, 3, _reset_world)
            solver = newton.solvers.SolverIPC(model, coupled_worlds=True, integrator=integrator)
            control = model.control()
            s0, s1 = model.state(), model.state()
            world = _row_worlds(solver, model)
            for _ in range(100):  # until the boxes and particles rest on the static box and the ground in every world
                solver.step(s0, s1, control, None, 0.01)
                s0, s1 = s1, s0
                pt = np.asarray(solver._engine.simplex.active_lists()["PT"]).reshape(-1, 4)
                ph = np.asarray(solver._engine.half_plane.active_list()).reshape(-1, 2)
                if all(np.isin(k, world[pt]) and np.isin(k, world[ph[:, 0]]) for k in range(3)):
                    break
            else:
                test.fail("no step with friction candidates in every world")
            one = world == 1
            body_one = np.append(model.body_world.numpy() == 1, False)  # the fixed body of the hinges is global
            joints = solver._joint_rows["revolute"].numpy()
            drive_one = model.body_world.numpy()[model.joint_child.numpy()[joints]] == 1
            frames = _branch_steps(
                solver, model, _solver_snapshot(solver), s0, ("none", 1, "all", ("state", 1)), control=control
            )
            none, w1, every, new_state = frames["none"], frames[1], frames["all"], frames[("state", 1)]
            for k in range(3):  # every world has friction candidates on the static box and the ground
                for name in ("PT", "PH"):
                    pairs = np.asarray(none["lists"][name]).reshape(-1, _PAIR_WIDTH[name])
                    vertices = pairs[:, :1] if name == "PH" else pairs
                    test.assertTrue(np.any(np.isin(world[vertices], [k]).any(axis=1)), (k, name))
            _check_friction_kept(test, w1, none, world, [1])
            _check_friction_kept(test, new_state, none, world, [1])
            test.assertEqual(sum(len(p) for p in every["lists"].values()), 0)
            for key in ("x_tilde", "mass"):
                np.testing.assert_array_equal(w1[key][~one], none[key][~one], key)
                np.testing.assert_array_equal(w1[key][one], every[key][one], key)
                np.testing.assert_array_equal(new_state[key][~one], none[key][~one], key)
            np.testing.assert_array_equal(w1["body_mass"][~body_one], none["body_mass"][~body_one])
            np.testing.assert_array_equal(w1["body_mass"][body_one], every["body_mass"][body_one])
            np.testing.assert_array_equal(w1["drive"][~drive_one], none["drive"][~drive_one])
            np.testing.assert_array_equal(w1["drive"][drive_one], every["drive"][drive_one])
            if integrator == "bdf1":
                test.assertIsNone(w1["restarted"])
                for key in ("x_tilde", "mass", "drive"):  # no history: the resets change no prediction or term
                    np.testing.assert_array_equal(w1[key], none[key], key)
                continue
            np.testing.assert_array_equal(w1["restarted"], one.astype(np.int32))
            np.testing.assert_array_equal(new_state["restarted"], one.astype(np.int32))
            test.assertFalse(np.any(none["restarted"]))
            test.assertFalse(np.all(w1["x_tilde"][one] == none["x_tilde"][one]))  # world 1 took an implicit Euler step
            test.assertFalse(np.all(w1["drive"][drive_one] == none["drive"][drive_one]))


def test_reset_of_one_world_keeps_the_other_worlds_aims(test, device):
    """Three copies of the pushed box's world with kinematic_motion="aim" under BDF2, whose aim kernel writes the
    history of every pusher at every step; each pusher moved by state_in. From one state, a step after no reset, after
    a reset of world 1 and after a reset of every world: the reset of world 1 releases world 1's pusher alone (it takes
    state_in), and the frame restarts world 1's rows alone: world 1's pusher and box take the prediction and the masses
    of the reset of every world, the other worlds' pushers and boxes those of no reset, bit for bit; the aims are the
    same in every case."""
    scene = newton.ModelBuilder()
    scene.add_ground_plane()
    scene.replicate(_pushed_world(), 3)
    model = scene.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, coupled_worlds=True, kinematic_motion="aim", integrator="bdf2")
    rows = _world_rows(model, 3)
    pushers = [b[1] for _, b in rows]

    def push(state, k):
        body_q, body_qd = state.body_q.numpy(), state.body_qd.numpy()
        body_q[pushers, 0] = -0.12 + 0.01 * k
        body_qd[pushers] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        state.body_q.assign(body_q)
        state.body_qd.assign(body_qd)

    s0, s1 = model.state(), model.state()
    for k in range(6):
        push(s0, k)
        solver.step(s0, s1, None, None, 0.01)
        s0, s1 = s1, s0
    world = _row_worlds(solver, model)
    one = world == 1
    body_one = model.body_world.numpy() == 1
    holds = {}

    def hold(case):
        holds[case] = solver._body_hold.numpy()[pushers]

    frames = _branch_steps(
        solver, model, _solver_snapshot(solver), s0, ("none", 1, "all"), pusher=lambda s: push(s, 6), after_reset=hold
    )
    np.testing.assert_array_equal(holds["none"], [1, 1, 1])
    np.testing.assert_array_equal(holds[1], [1, 0, 1])
    np.testing.assert_array_equal(holds["all"], [0, 0, 0])
    none, w1, every = frames["none"], frames[1], frames["all"]
    np.testing.assert_array_equal(w1["restarted"], one.astype(np.int32))
    for key in ("x_tilde", "mass"):
        np.testing.assert_array_equal(w1[key][~one], none[key][~one], key)
        np.testing.assert_array_equal(w1[key][one], every[key][one], key)
    np.testing.assert_array_equal(w1["body_mass"][~body_one], none["body_mass"][~body_one])
    np.testing.assert_array_equal(w1["body_mass"][body_one], every["body_mass"][body_one])
    test.assertFalse(np.all(w1["body_mass"][body_one] == none["body_mass"][body_one]))  # unscaled where restarted


def _hinged(device, kind="revolute", gravity=0.0, axis=newton.Axis.Z, offset=(0.0, 0.0, 0.0), radius=0.1, **dof):
    """A sphere hanging from the world on a revolute or prismatic joint (y up); ``offset`` is the joint's origin in
    the sphere's frame, and ``dof`` the joint's arguments. Returns the model, a state at its joint coordinates and
    the joint."""
    builder = newton.ModelBuilder(gravity=(0.0, -gravity, 0.0), up_axis=newton.Axis.Y)
    link = builder.add_link()
    builder.add_shape_sphere(link, radius=radius)
    add = builder.add_joint_revolute if kind == "revolute" else builder.add_joint_prismatic
    j = add(-1, link, axis=axis, child_xform=wp.transform(offset, wp.quat_identity()), **dof)
    builder.add_articulation([j])
    model = builder.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, state, j


def _steps(model, solver, state, steps, dt, control=None, record=lambda s: s.joint_q.numpy().copy()):
    """``steps`` steps from ``state``; what ``record`` takes of the state after each."""
    s0, s1 = state, model.state()
    out = []
    for _ in range(steps):
        solver.step(s0, s1, control, None, dt)
        s0, s1 = s1, s0
        out.append(record(s0))
    return np.array(out), s0


def _drive_recurrence(
    inertia, h, steps, q0=0.0, qd0=0.0, ke=0.0, kd=0.0, ks=0.0, c=0.0, q_target=0.0, qd_target=0.0, q_rest=0.0, f=0.0
):
    """q_1 .. q_steps of implicit Euler for inertia q'' = ke (q* - q) + kd (qd* - qd) - ks (q - q_rest) - c qd + f."""
    q, qd, out = q0, qd0, []
    k = inertia / h**2 + ke + ks + (kd + c) / h
    for _ in range(steps):
        q_next = (
            inertia * (q + h * qd) / h**2 + ke * q_target + ks * q_rest + kd * (q / h + qd_target) + c * q / h + f
        ) / k
        q, qd = q_next, (q_next - q) / h
        out.append(q)
    return np.array(out)


def test_unplaced_links_start_at_forward_kinematics(test, device):
    """A double pendulum built as Newton's basic_pendulum example builds it: links added without a pose, so that the
    model holds both at the origin, through the ground plane, and the forward kinematics written into the state only
    after the solver is built. The solver is built at, and checks, the poses newton.eval_fk computes from the model's
    joint coordinates, and steps captured from that state run as those of a solver built from a model that holds the
    poses. A link that the joint coordinates put into the ground is refused at construction, by name."""

    def pendulum(height=5.0, angle=0.0, posed=False):
        builder = newton.ModelBuilder()
        links = [builder.add_link(label=f"link_{k}") for k in range(2)]
        for link in links:
            builder.add_shape_box(link, hx=1.0, hy=0.1, hz=0.1)
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.pi * 0.5)
        end = wp.transform(wp.vec3(-1.0, 0.0, 0.0), wp.quat_identity())
        joints = [
            builder.add_joint_revolute(-1, links[0], axis=wp.vec3(0.0, 1.0, 0.0),
                                       parent_xform=wp.transform(wp.vec3(0.0, 0.0, height), rot), child_xform=end),
            builder.add_joint_revolute(links[0], links[1], axis=wp.vec3(0.0, 1.0, 0.0),
                                       parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
                                       child_xform=end),
        ]  # fmt: skip
        builder.add_articulation(joints)
        builder.add_ground_plane()
        builder.joint_q[0] = angle
        model = builder.finalize(device=device)
        if posed:
            newton.eval_fk(model, model.joint_q, model.joint_qd, model)
        return model

    model = pendulum()
    np.testing.assert_array_equal(model.body_q.numpy(), [[0, 0, 0, 0, 0, 0, 1]] * 2)
    solver = newton.solvers.SolverIPC(model)  # refused before: the links crossed the ground at the origin
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    base = solver._scene.layout().body_base[:2]
    np.testing.assert_allclose(solver._engine.positions()[base], state.body_q.numpy()[:, :3], atol=1e-6)
    test.assertAlmostEqual(float(state.body_q.numpy()[1, 2]), 5.0, places=5)  # the links lie level at the pivot

    def captured(model, solver, state, frames=20):
        s0, s1 = state, model.state()
        solver.step(s0, s1, None, None, 0.001)
        with wp.ScopedCapture(device=device) as capture:
            solver.step(s1, s0, None, None, 0.001)
            solver.step(s0, s1, None, None, 0.001)
        for _ in range(frames):
            wp.capture_launch(capture.graph)
        return s1.body_q.numpy()

    swung = captured(model, solver, state)
    posed = pendulum(posed=True)
    state = posed.state()
    reference = captured(posed, newton.solvers.SolverIPC(posed), state)
    test.assertLess(float(swung[1, 2]), 5.0 - 1e-3)  # the pendulum swings down
    np.testing.assert_allclose(swung, reference, atol=1e-6)
    # turned down from a pivot 3 m up, the outer link reaches 1 m into the ground
    with test.assertRaisesRegex(ValueError, "body 1 \\('link_1'\\) and plane shape 2 start"):
        newton.solvers.SolverIPC(pendulum(height=3.0, angle=0.5 * np.pi))


def test_pendulum_is_implicit_euler(test, device):
    """A pendulum of 2 cm per metre swings as implicit Euler integrates the linearized pendulum, step by step to
    1e-3 of its amplitude, with the period of the closed form 2 pi sqrt(I / (m g L)) to 0.1%."""
    L, theta0, g, h = 1.0, 0.02, 10.0, 1e-3
    model, state, _ = _hinged(device, gravity=g, offset=(0.0, L, 0.0), radius=0.05)
    q = model.joint_q.numpy()
    q[0] = theta0
    state.joint_q.assign(q)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    solver = newton.solvers.SolverIPC(model, **_CONVERGED)
    m = float(model.body_mass.numpy()[0])
    inertia = float(model.body_inertia.numpy()[0][2, 2]) + m * L * L
    period = 2.0 * np.pi * np.sqrt(inertia / (m * g * L))
    steps = int(1.25 * period / h)
    angles, _ = _steps(model, solver, state, steps, h)
    expected = _drive_recurrence(inertia, h, steps, q0=theta0, ke=m * g * L)
    np.testing.assert_allclose(angles[:, 0], expected, rtol=0.0, atol=1e-3 * theta0)
    # the first zero crossing upwards after the start: three quarters of a period
    up = np.flatnonzero((angles[:-1, 0] < 0.0) & (angles[1:, 0] >= 0.0))[0]
    test.assertAlmostEqual(h * (up + 1) / (0.75 * period), 1.0, delta=1e-3)


def test_joint_forces_match_closed_form(test, device):
    """joint_f from rest without gravity: a revolute and a prismatic joint follow q_n = a h^2 n (n + 1) / 2 of
    implicit Euler, a = tau / I or F / m, and a ball joint whose parent frame is turned spins its child about the
    torque turned with it."""
    # small enough that the bodies turn slowly: an affine body turning under implicit Euler moves along a straight
    # line in its 12 coordinates and back onto the rotations, which loses about (omega dt)^2 of its angular velocity
    # per step
    h, steps = 0.01, 20
    n = np.arange(1, steps + 1)
    for kind, f in (("revolute", 0.005), ("prismatic", 2.0)):
        with test.subTest(kind):
            model, state, _ = _hinged(device, kind, axis=newton.Axis.X)
            control = model.control()
            control.joint_f.assign(np.array([f], np.float32))
            q, _ = _steps(model, newton.solvers.SolverIPC(model, **_CONVERGED), state, steps, h, control)
            m = float(model.body_mass.numpy()[0])
            a = f / (float(model.body_inertia.numpy()[0][0, 0]) if kind == "revolute" else m)
            np.testing.assert_allclose(q[:, 0], a * h * h * n * (n + 1) / 2.0, rtol=1e-5, atol=0.0)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Y)
    ball = builder.add_link()
    builder.add_shape_sphere(ball, radius=0.1)
    turn = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * np.pi)
    builder.add_articulation([builder.add_joint_ball(-1, ball, parent_xform=wp.transform((0.0, 0.0, 0.0), turn))])
    model = builder.finalize(device=device)
    control = model.control()
    # the torque about z of the turned parent frame: about -y in the world
    control.joint_f.assign(np.array([0.0, 0.0, 0.003], np.float32))
    qd, _ = _steps(model, newton.solvers.SolverIPC(model, **_CONVERGED), model.state(), steps, h, control,
                   lambda s: s.body_qd.numpy().copy())  # fmt: skip
    inertia = float(model.body_inertia.numpy()[ball][0, 0])
    np.testing.assert_allclose(qd[:, ball, 3:6], np.outer(-0.003 * n * h / inertia, [0.0, 1.0, 0.0]), atol=1e-6)


def test_fixed_and_ball_joints_hold(test, device):
    """A box welded to a swinging pendulum keeps its pose relative to it, and a bob on a ball joint keeps its anchor
    on the pendulum's, within the joints' penalty (a load F stretches a joint by about F dt^2 / K, a few 1e-5 m here),
    while the pendulum swings through 1.2 rad."""
    builder = newton.ModelBuilder(gravity=(0.0, -10.0, 0.0), up_axis=newton.Axis.Y)
    arm = builder.add_link(xform=wp.transform((0.0, -0.5, 0.0), wp.quat_identity()))
    builder.add_shape_box(arm, hx=0.05, hy=0.5, hz=0.05)
    weld = builder.add_link(xform=wp.transform((0.15, -0.9, 0.0), wp.quat_identity()))
    builder.add_shape_box(weld, hx=0.1, hy=0.05, hz=0.05)
    bob = builder.add_link(xform=wp.transform((0.0, -1.3, 0.0), wp.quat_identity()))
    builder.add_shape_sphere(bob, radius=0.1)
    joints = [
        builder.add_joint_revolute(
            -1, arm, axis=newton.Axis.Z, child_xform=wp.transform((0.0, 0.5, 0.0), wp.quat_identity())
        ),
        builder.add_joint_fixed(arm, weld, parent_xform=wp.transform((0.15, -0.4, 0.0), wp.quat_identity())),
        builder.add_joint_ball(
            arm, bob, parent_xform=wp.transform((0.0, -0.6, 0.0), wp.quat_identity()),
            child_xform=wp.transform((0.0, 0.2, 0.0), wp.quat_identity()),
        ),
    ]  # fmt: skip
    builder.add_articulation(joints)
    model = builder.finalize(device=device)
    q = model.joint_q.numpy()
    q[0] = 1.2
    state = model.state()
    state.joint_q.assign(q)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    # libuipc's body stiffness: with the default one the arm's own stretch under the swing takes the ball joint's
    # anchors 1.03e-4 m apart, past the joints' penalty (class docstring, **Rigid bodies**)
    body_q, _ = _steps(
        model,
        newton.solvers.SolverIPC(model, body_kappa=1.0e8),
        state,
        100,
        0.01,
        None,
        lambda s: s.body_q.numpy().copy(),
    )

    def transforms(rows):
        return [wp.transform(*row[:3], *row[3:]) for row in rows]

    swing = np.arctan2(body_q[:, arm, 0], -body_q[:, arm, 1])
    test.assertGreater(swing.max() - swing.min(), 1.0)
    for frame in body_q:
        X_arm, X_weld, X_bob = transforms(frame[[arm, weld, bob]])
        relative = wp.transform_inverse(X_arm) * X_weld
        np.testing.assert_allclose(np.array(wp.transform_get_translation(relative)), [0.15, -0.4, 0.0], atol=1e-4)
        turn = wp.transform_get_rotation(relative)  # its angle is twice the length of its vector part
        test.assertLess(2.0 * float(wp.length(wp.vec3(turn[0], turn[1], turn[2]))), 1e-4)
        anchor_arm = wp.transform_point(X_arm, wp.vec3(0.0, -0.6, 0.0))
        anchor_bob = wp.transform_point(X_bob, wp.vec3(0.0, 0.2, 0.0))
        test.assertLess(float(wp.length(anchor_arm - anchor_bob)), 1e-4)


def test_joint_penalty_follows_the_masses(test, device):
    """A joint's penalty is joint_strength_ratio times the mass of the dynamic bodies it joins, a kinematic body
    counting 0, for every kind of joint, and it follows the masses that notify_model_changed gives the bodies (libuipc's
    revolute and prismatic joints keep the masses they were built with and scale their ratio by them)."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Y)
    arm = builder.add_link(xform=wp.transform((0.0, -0.5, 0.0), wp.quat_identity()))
    builder.add_shape_box(arm, hx=0.05, hy=0.5, hz=0.05)
    bob = builder.add_link(xform=wp.transform((0.0, -1.3, 0.0), wp.quat_identity()))
    builder.add_shape_sphere(bob, radius=0.2)
    builder.add_articulation(
        [
            builder.add_joint_revolute(-1, arm, child_xform=wp.transform((0.0, 0.5, 0.0), wp.quat_identity())),
            builder.add_joint_ball(
                arm, bob, parent_xform=wp.transform((0.0, -0.5, 0.0), wp.quat_identity()),
                child_xform=wp.transform((0.0, 0.3, 0.0), wp.quat_identity()),
            ),
        ]
    )  # fmt: skip
    base = builder.add_link(xform=wp.transform((2.0, 0.0, 0.0), wp.quat_identity()), is_kinematic=True)
    builder.add_shape_box(base, hx=0.2, hy=0.1, hz=0.2)
    slider = builder.add_link(xform=wp.transform((2.0, -0.5, 0.0), wp.quat_identity()))
    builder.add_shape_box(slider, hx=0.1, hy=0.1, hz=0.1)
    builder.add_articulation(
        [
            builder.add_joint_fixed(-1, base, parent_xform=wp.transform((2.0, 0.0, 0.0), wp.quat_identity())),
            builder.add_joint_prismatic(
                base, slider, axis=newton.Axis.Y, parent_xform=wp.transform((0.0, -0.5, 0.0), wp.quat_identity())
            ),
        ]
    )
    model = builder.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    solver = newton.solvers.SolverIPC(model)

    def penalties():
        solver.step(state, model.state(), None, None, 0.01)  # writes the joints' terms
        groups = solver._joint_groups
        return [
            float(groups.revolute.ratio.numpy()[0] * groups.revolute.mass_sum.numpy()[0]),
            float(groups.spherical.kappa.numpy()[0]),
            float(groups.fixed.kappa.numpy()[0]),
            float(groups.prismatic.ratio.numpy()[0] * groups.prismatic.mass_sum.numpy()[0]),
        ]

    for scale in (1.0, 2.0):
        with test.subTest(scale=scale):
            if scale != 1.0:
                for name in ("body_mass", "body_inertia"):
                    getattr(model, name).assign(scale * getattr(model, name).numpy())
                for name in ("body_inv_mass", "body_inv_inertia"):
                    getattr(model, name).assign(getattr(model, name).numpy() / scale)
                solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
            m = model.body_mass.numpy().astype(np.float64)
            expected = [100.0 * m[arm], 100.0 * (m[arm] + m[bob]), 0.0, 100.0 * m[slider]]
            np.testing.assert_allclose(penalties(), expected, rtol=1e-12)


def test_joint_coordinates_are_written(test, device):
    """After every step joint_q and joint_qd hold the coordinates eval_ik computes from the bodies' state, for a
    revolute, a prismatic, a ball and a free joint, but the revolute angle, which goes on past pi; an angle written into
    state_in picks the turn the joint goes on from."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Y)
    links = []
    for x in (0.0, 1.0, 2.0):
        links.append(builder.add_link(xform=wp.transform((x, 0.0, 0.0), wp.quat_identity())))
        builder.add_shape_box(links[-1], hx=0.1, hy=0.1, hz=0.1)
    joints = [
        builder.add_joint_revolute(-1, links[0], axis=newton.Axis.Z),
        builder.add_joint_prismatic(
            -1, links[1], axis=newton.Axis.X, parent_xform=wp.transform((1.0, 0.0, 0.0), wp.quat_identity())
        ),
        builder.add_joint_ball(
            links[1], links[2], parent_xform=wp.transform((0.5, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform((-0.5, 0.0, 0.0), wp.quat_identity()),
        ),
    ]  # fmt: skip
    builder.add_articulation(joints)
    _box_body(builder, (0.0, 2.0, 0.0), (0.1, 0.1, 0.1))  # a free body
    model = builder.finalize(device=device)
    qd = model.joint_qd.numpy()
    qd[:] = (10.0, 0.5, 1.0, -2.0, 0.5, 0.3, -0.2, 0.1, 0.4, 0.2, 0.7)  # 10 rad/s about the hinge
    state = model.state()
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    solver = newton.solvers.SolverIPC(model, **_CONVERGED)
    q_ik, qd_ik = wp.zeros_like(state.joint_q), wp.zeros_like(state.joint_qd)
    s0, s1 = state, model.state()
    angles = []
    for _ in range(50):
        solver.step(s0, s1, None, None, 0.01)
        s0, s1 = s1, s0
        angles.append(float(s0.joint_q.numpy()[0]))
        newton.eval_ik(model, s0, q_ik, qd_ik)
        q, q_eval = s0.joint_q.numpy(), q_ik.numpy()
        turns = np.round((q[0] - q_eval[0]) / (2.0 * np.pi))
        np.testing.assert_allclose(q[0] - 2.0 * np.pi * turns, q_eval[0], atol=1e-5)
        np.testing.assert_allclose(q[1:], q_eval[1:], atol=1e-5)
        np.testing.assert_allclose(s0.joint_qd.numpy(), qd_ik.numpy(), atol=1e-4)
    test.assertGreater(float(s0.joint_q.numpy()[0]), 4.0, "the hinge turns past pi")
    # a turn more, written into state_in with the bodies' state unchanged
    angle = float(s0.joint_q.numpy()[0])
    q = s0.joint_q.numpy()
    q[0] += 2.0 * np.pi
    s0.joint_q.assign(q)
    solver.step(s0, s1, None, None, 0.01)
    test.assertAlmostEqual(float(s1.joint_q.numpy()[0]) - angle, 2.0 * np.pi + angles[-1] - angles[-2], delta=2e-3)


def test_grown_lists_keep_joint_angles(test, device):
    """A wheel spins on its hinge while a box drops onto the ground; when it lands, past half a turn of the wheel, the
    contact lists overflow, and the eager step grows them and solves again from the step's start: the wheel's angle
    goes on as in a solver with room, and so does every body."""

    def model():
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, -9.81, 0.0))
        wheel = builder.add_link(xform=wp.transform((0.0, 1.0, 0.0), wp.quat_identity()))
        builder.add_shape_box(wheel, hx=0.2, hy=0.2, hz=0.02)
        builder.add_articulation(
            [
                builder.add_joint_revolute(
                    -1, wheel, axis=newton.Axis.Z, parent_xform=wp.transform((0.0, 1.0, 0.0), wp.quat_identity())
                )
            ]
        )
        _box_body(builder, (1.0, 0.6, 0.0), (0.05, 0.05, 0.05))
        builder.add_ground_plane()
        m = builder.finalize(device=device)
        qd = m.joint_qd.numpy()
        qd[0] = 15.0
        m.joint_qd.assign(qd)
        return m

    results = []
    for tiny in (False, True):
        m = model()
        state = m.state()
        newton.eval_fk(m, m.joint_q, m.joint_qd, state)
        solver = newton.solvers.SolverIPC(m, **_CONVERGED)
        if tiny:
            solver._make_engine(_TINY_LISTS)
        q, last = _steps(m, solver, state, 45, 0.01)
        results.append((q[:, 0], last.body_q.numpy()))
    (roomy, roomy_body_q), (grown, grown_body_q) = results
    test.assertGreater(roomy[-1], np.pi)
    np.testing.assert_allclose(grown, roomy, atol=1e-5)
    np.testing.assert_allclose(grown_body_q, roomy_body_q, atol=1e-5)


def test_drives_are_implicit_euler(test, device):
    """A revolute and a prismatic joint driven from rest by target gains, a passive spring and passive damping,
    towards a moving target, follow implicit Euler's recurrence of I q'' = ke (q* - q) + kd (qd* - qd) - ks (q - q_rest)
    - c qd step by step to 1e-4 of their motion (SolverIPC folds the four terms into libuipc's driving term of every
    step). The hinge turns a large sphere through small angles: an affine body turned under implicit Euler moves
    along a straight line in its 12 coordinates and back onto the rotations, which adds a term of order
    (omega dt)^3 to every step, 1e-4 of the motion at 0.5 rad and 1e-6 at 0.05 rad here."""
    for kind, length, scale in (("revolute", 0.1, 1.0), ("prismatic", 1.0, 10.0)):
        with test.subTest(kind):
            gains = {"target_ke": 40.0, "target_kd": 2.0, "stiffness": 10.0, "damping": 1.0}
            gains = {name: scale * value for name, value in gains.items()}
            targets = {"target_q": 0.5 * length, "target_qd": 0.1 * length, "rest_q": -0.2 * length}
            model, state, _ = _hinged(device, kind, axis=newton.Axis.X, radius=0.5, **targets, **gains)
            solver = newton.solvers.SolverIPC(model, **_CONVERGED)
            m = float(model.body_mass.numpy()[0])
            inertia = float(model.body_inertia.numpy()[0][0, 0]) if kind == "revolute" else m
            h, steps = 0.01, 100
            q, _ = _steps(model, solver, state, steps, h, model.control())
            expected = _drive_recurrence(
                inertia, h, steps, ke=gains["target_ke"], kd=gains["target_kd"], ks=gains["stiffness"],
                c=gains["damping"], q_target=targets["target_q"], qd_target=targets["target_qd"],
                q_rest=targets["rest_q"],
            )  # fmt: skip
            test.assertGreater(np.abs(expected).max(), 0.1 * length)
            np.testing.assert_allclose(q[:, 0], expected, rtol=0.0, atol=1e-4 * np.abs(expected).max())


def test_joint_limits_hold(test, device):
    """A revolute joint turned by a torque into its upper limit, and a slider pulled down by gravity onto its lower
    limit, come to rest past the limit by sqrt(F delta_r / limit_ke), where libuipc's cubic limit balances the load
    (SolverIPC sets its strength so that it pushes back with limit_ke delta_r at the overshoot delta_r = 1e-3); four
    times the stiffness halves the overshoot. Without the limit they run on."""
    tau, upper = 4.0, 0.3
    for limit_ke in (1.0e4, 4.0e4):
        with test.subTest(limit_ke=limit_ke):
            model, state, _ = _hinged(device, limit_lower=-1.0, limit_upper=upper, limit_ke=limit_ke, damping=2.0)
            control = model.control()
            control.joint_f.assign(np.array([tau], np.float32))
            q, _ = _steps(model, newton.solvers.SolverIPC(model, **_CONVERGED), state, 300, 0.01, control)
            test.assertLess(q[:, 0].max(), upper + 0.01)
            test.assertAlmostEqual(float(q[-1, 0]) - upper, np.sqrt(tau * 1e-3 / limit_ke), delta=1e-6)

            g, lower = 10.0, -0.1
            model, state, _ = _hinged(
                device, "prismatic", gravity=g, axis=newton.Axis.Y, limit_lower=lower, limit_upper=1.0,
                limit_ke=limit_ke, damping=20.0,
            )  # fmt: skip
            m = float(model.body_mass.numpy()[0])
            q, _ = _steps(model, newton.solvers.SolverIPC(model, **_CONVERGED), state, 300, 0.01)
            test.assertAlmostEqual(float(q[-1, 0]) - lower, -np.sqrt(m * g * 1e-3 / limit_ke), delta=1e-6)
    model, state, _ = _hinged(device, limit_lower=-1.0, limit_upper=upper, limit_ke=0.0, damping=2.0)
    control = model.control()
    control.joint_f.assign(np.array([tau], np.float32))
    q, _ = _steps(model, newton.solvers.SolverIPC(model), state, 100, 0.01, control)
    test.assertGreater(q[-1, 0], 1.0, "a limit of stiffness 0 holds nothing")


def test_drive_targets_change_inside_a_captured_graph(test, device):
    """Targets written into control.joint_target_q between replays of a captured step drive a revolute joint as they
    do in eager steps, to 1e-6: the drive is computed on the device from the control at every step."""

    def run(capture):
        model, state, _ = _hinged(device, axis=newton.Axis.X, target_ke=50.0, target_kd=5.0)
        solver = newton.solvers.SolverIPC(model, **_CONVERGED)
        control = model.control()
        s0, s1 = state, model.state()
        out = []
        solver.step(s0, s1, control, None, 0.01)  # loads the kernels
        s0, s1 = s1, s0
        graph = None
        if capture:
            with wp.ScopedCapture(device=device) as cap:
                solver.step(s0, s1, control, None, 0.01)
                solver.step(s1, s0, control, None, 0.01)
            graph = cap.graph
        for k in range(20):
            control.joint_target_q.assign(np.array([0.4 * np.sin(0.5 * k)], np.float32))
            if capture:
                wp.capture_launch(graph)
            else:
                for _ in range(2):
                    solver.step(s0, s1, control, None, 0.01)
                    s0, s1 = s1, s0
            out.append(s0.joint_q.numpy()[0])
        return np.array(out)

    eager, captured = run(False), run(True)
    test.assertGreater(np.abs(eager).max(), 0.1)
    np.testing.assert_allclose(captured, eager, atol=1e-6)


def _libuipc_corridor(model, dt, d_hat):
    """libuipc's contact stiffness range of a model of particles over a plane through the origin, from the particles'
    positions and masses: every particle and the plane's point are libuipc's vertices, and the plane has no mass."""
    from uipc_warp.backend.contact_system.contact_stiffness import scene_diagonal  # noqa: PLC0415
    from uipc_warp.backend.contact_system.global_contact_manager import kappa_corridor  # noqa: PLC0415

    points = np.vstack([model.particle_q.numpy().astype(np.float64), np.zeros((1, 3))])
    mean_mass = model.particle_mass.numpy().astype(np.float64).sum() / len(points)
    return kappa_corridor(scene_diagonal(points), mean_mass, d_hat, dt, 1.0e-16)


def test_body_stiffness_takes_a_turn_in_one_iteration(test, device):
    """By default a body's orthogonality stiffness is libuipc's 1e8 Pa, lowered in a step of length dt to
    25 I_max / (V dt^2), I_max the body's largest principal moment of inertia, so that the first Newton iteration takes
    a turn of up to 0.1 rad about that moment's axis in full: a 1 m cube that a torque turns by 0.06 rad in one 0.1 s
    step (Newton's body-force test) and a 10 cm cube spinning at 0.6 rad/s over six steps of 1/60 s end where the
    converged solve does, as a rotation projected back after every step ends. At libuipc's stiffness, given as
    body_kappa, the line search cuts the turn short and the next directions pass the displacement test long before it
    is done (31% and 33% of the angular velocity). Eager and captured steps of two sizes take the stiffness of their own
    dt; a given body_kappa is taken at every step."""

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
    (ortho,) = (e for e in solver._engine.elements if isinstance(e, solver._ortho_potential))
    np.testing.assert_array_equal(ortho.kappa.numpy(), 1.0e7)
    test.assertIsNone(solver._ortho)
    for kappa in (0.0, -1.0e8, float("nan"), float("inf")):
        with test.assertRaisesRegex(ValueError, "body_kappa must be a finite positive stiffness"):
            newton.solvers.SolverIPC(model, body_kappa=kappa)


def test_kappa_is_taken_as_given(test, device):
    """By default every pair takes kappa as given, at every step size, also outside libuipc's range, without a
    warning; kappa=None, libuipc's default, needs adaptive_kappa."""
    model = _soft_cube(device, height=0.05)
    lo = _libuipc_corridor(model, 0.01, D_HAT)[0]
    for kappa in (1.0e9, 0.1 * lo):
        solver = newton.solvers.SolverIPC(model, d_hat=D_HAT, kappa=kappa)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _run(model, solver, 3)
        for dt in (0.01, 1.0 / 600.0):
            test.assertEqual(solver.contact_stiffness(dt), kappa)
        np.testing.assert_array_equal(solver._engine.contact.table.kappa.numpy(), kappa)
    with test.assertRaisesRegex(ValueError, "kappa=None, libuipc's default stiffness, needs adaptive_kappa=True"):
        newton.solvers.SolverIPC(model, kappa=None)


def test_adaptive_kappa_is_libuipcs(test, device):
    """adaptive_kappa=True: kappa=None takes the lower bound of libuipc's range for the model and the step, which
    scales with 1/dt^2; a kappa inside the range is used as given and one outside it is clamped, with one
    NewtonWarning that names both values, at the first step that clamps it."""
    model = _soft_cube(device, height=0.05)
    adaptive = {"d_hat": D_HAT, "adaptive_kappa": True}
    for dt in (0.01, 1.0 / 600.0):
        lo, hi = _libuipc_corridor(model, dt, D_HAT)
        default = newton.solvers.SolverIPC(model, kappa=None, **adaptive).contact_stiffness(dt)
        test.assertAlmostEqual(default / lo, 1.0, places=12)
    lo, hi = _libuipc_corridor(model, 0.01, D_HAT)
    inside = newton.solvers.SolverIPC(model, kappa=3.0 * lo, **adaptive)
    test.assertEqual(inside.contact_stiffness(0.01), 3.0 * lo)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _run(model, inside, 2)
    above = newton.solvers.SolverIPC(model, kappa=1.0e3 * hi, **adaptive)
    test.assertAlmostEqual(above.contact_stiffness(0.01) / hi, 1.0, places=12)
    s0, s1 = model.state(), model.state()
    given, used = re.escape(f"{1.0e3 * hi:g}"), re.escape(f"{hi:.6g}")
    with test.assertWarnsRegex(newton.NewtonWarning, rf"kappa={given} is outside .* the steps use {used} Pa"):
        above.step(s0, s1, None, None, 0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        above.step(s1, s0, None, None, 0.01)
        above.step(s0, s1, None, None, 0.005)  # clamped again, at another value: not repeated
    for kappa in (0.0, -1.0e9, float("nan"), float("inf")):
        with test.assertRaisesRegex(ValueError, "kappa must be a finite positive stiffness"):
            newton.solvers.SolverIPC(model, kappa=kappa, **adaptive)


def test_adaptive_kappa_follows_dt_in_captured_steps(test, device):
    """adaptive_kappa=True: every pair takes the stiffness of its step's dt: steps captured at two step sizes, replayed
    in turn and between eager steps of a third, each run with their own, as eager steps of those sizes do (to rounding:
    the solver adds with float atomics)."""
    tight = {"newton_velocity_tol": 1e-7, "linear_tol_rate": 1e-12, "semi_implicit": False}
    model = _soft_cube(device, height=0.012)
    captured, eager = (
        newton.solvers.SolverIPC(model, d_hat=D_HAT, kappa=None, adaptive_kappa=True, **tight) for _ in range(2)
    )
    states = {solver: (model.state(), model.state()) for solver in (captured, eager)}

    def table(solver):
        return solver._engine.contact.table.kappa.numpy()

    graphs = {}
    for dt in (0.01, 0.005):
        s0, s1 = model.state(), model.state()
        with wp.ScopedCapture(device=device) as cap:
            captured.step(s0, s1, None, None, dt)
        graphs[dt] = (cap.graph, s0, s1)
    for k, dt in enumerate((0.01, 0.005, 0.02, 0.01, 0.02, 0.005, 0.01)):
        for solver in (captured, eager):
            a, b = states[solver]
            if solver is captured and dt in graphs:
                graph, g0, g1 = graphs[dt]
                g0.assign(a)
                wp.capture_launch(graph)
                b.assign(g1)
            else:
                solver.step(a, b, None, None, dt)
            states[solver] = (b, a)
        np.testing.assert_array_equal(table(captured), table(eager), err_msg=f"step {k}, dt {dt}")
        test.assertEqual(float(table(eager).max()), eager.contact_stiffness(dt))
        test.assertEqual(float(table(eager).min()), eager.contact_stiffness(dt))
        np.testing.assert_allclose(
            states[captured][0].particle_q.numpy(), states[eager][0].particle_q.numpy(), rtol=0.0, atol=1e-6
        )
    test.assertLess(states[eager][0].particle_q.numpy()[:, 2].min(), D_HAT)  # the cube reached the ground


# ---------------------------------------------------------------------------- libuipc's options


def _bdf2_from_rest(n, h, a):
    """``(x_n - x_0, v_n)`` of BDF2 from rest under a constant acceleration ``a``, its history started by an implicit
    Euler step: ``v_n = n h a`` exactly and ``x_n - x_0 = a h^2 (n^2 / 2 + 3/4 (1 - 3^-n))``."""
    n = np.asarray(n, np.float64)
    return a * h * h * (n * n / 2.0 + 0.75 * (1.0 - 3.0**-n)), n * h * a


def _drive_recurrence_bdf2(
    inertia, h, steps, q0=0.0, qd0=0.0, ke=0.0, kd=0.0, ks=0.0, c=0.0, q_target=0.0, qd_target=0.0, q_rest=0.0, f=0.0
):
    """q_1 .. q_steps of BDF2 for inertia q'' = ke (q* - q) + kd (qd* - qd) - ks (q - q_rest) - c qd + f after an
    implicit Euler step (``_drive_recurrence``): ``I (q' - q~) = 4/9 h^2 tau(q', qd')`` with
    ``q~ = 4/3 q - 1/3 q_prev + 2/9 h (4 qd - qd_prev)`` and ``qd' = (3 q' - 4 q + q_prev) / (2 h)``."""
    q1 = _drive_recurrence(inertia, h, 1, q0, qd0, ke, kd, ks, c, q_target, qd_target, q_rest, f)[0]
    q_prev, qd_prev, q, qd = q0, qd0, q1, (q1 - q0) / h
    out = [q1]
    for _ in range(steps - 1):
        predicted = 4.0 / 3.0 * q - q_prev / 3.0 + 2.0 / 9.0 * h * (4.0 * qd - qd_prev)
        a, b = 1.5 / h, (4.0 * q - q_prev) / (2.0 * h)  # qd' = a q' - b
        load = ke * q_target + kd * qd_target + ks * q_rest + f + (kd + c) * b
        w = 4.0 / 9.0 * h * h
        q_next = (inertia * predicted + w * load) / (inertia + w * (ke + ks + (kd + c) * a))
        q_prev, qd_prev, q, qd = q, qd, q_next, a * q_next - b
        out.append(q)
    return np.array(out)


def test_bdf2_falls_from_an_implicit_euler_step(test, device):
    """With integrator="bdf2" (libuipc's integrator/type; BDF1 by default, other values refused) a soft cube and a
    rigid box fall from rest, eagerly and captured, as BDF2 does after a first implicit Euler step: ``v_n = -g n h`` and
    ``z_n - z_0 = -g h^2 (n^2 / 2 + 3/4 (1 - 3^-n))``, to the float32 output. (A history started at rest, libuipc's,
    would give the first step two thirds of the impulse.)"""
    builder = newton.ModelBuilder()
    builder.add_soft_grid(
        pos=wp.vec3(-0.1, -0.1, 1.0), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0), dim_x=2, dim_y=2,
        dim_z=2, cell_x=0.05, cell_y=0.05, cell_z=0.05, density=1000.0, k_mu=3.4e4, k_lambda=1.6e5, k_damp=0.0,
    )  # fmt: skip
    _box_body(builder, (0.5, 0.0, 1.0), (0.1, 0.1, 0.1))
    model = builder.finalize(device=device)
    h, g = 0.01, 9.81
    z_particles, z_body = model.particle_q.numpy()[:, 2].astype(np.float64), float(model.body_q.numpy()[0, 2])

    with test.assertRaisesRegex(ValueError, "integrator"):
        newton.solvers.SolverIPC(model, integrator="bdf3")
    test.assertIsNone(newton.solvers.SolverIPC(model)._engine.bdf1_frame)  # BDF1 by default

    def check(state, n):
        dz, v = _bdf2_from_rest(n, h, -g)
        np.testing.assert_allclose(state.particle_q.numpy()[:, 2], z_particles + dz, rtol=0.0, atol=2e-6)
        np.testing.assert_allclose(state.particle_qd.numpy()[:, 2], v, rtol=0.0, atol=1e-5)
        test.assertAlmostEqual(float(state.body_q.numpy()[0, 2]), z_body + dz, delta=2e-6)
        test.assertAlmostEqual(float(state.body_qd.numpy()[0, 2]), v, delta=1e-5)

    for capture in (False, True):
        with test.subTest(capture=capture):
            solver = newton.solvers.SolverIPC(model, integrator="bdf2", **_CONVERGED)
            test.assertEqual(solver._engine.options.bdf2_start, "bdf1")
            s0, s1 = model.state(), model.state()
            solver.step(s0, s1, None, None, h)
            s0, s1 = s1, s0
            check(s0, 1)
            if capture:  # steps 2 and 3 in one graph, replayed
                with wp.ScopedCapture(device=device) as cap:
                    solver.step(s0, s1, None, None, h)
                    solver.step(s1, s0, None, None, h)
                for n in (3, 5, 7):
                    wp.capture_launch(cap.graph)
                    check(s0, n)
            else:
                for n in range(2, 8):
                    solver.step(s0, s1, None, None, h)
                    s0, s1 = s1, s0
                    check(s0, n)


def test_bdf2_turns_a_body_by_the_impulse_of_its_first_step(test, device):
    """A torque from rest turns a free box in one step of 0.1 s to tau dt / I with integrator="bdf2" as with implicit
    Euler, to 1%: the first step starts the history. Newton's body-force test turns a cube so; a BDF2 history started
    at rest gives two thirds of it."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    b = _box_body(builder, (0.0, 0.0, 1.0), (0.5, 0.5, 0.5))
    model = builder.finalize(device=device)
    I_zz = float(model.body_inertia.numpy()[b][2, 2])
    dt, alpha = 0.1, 6.0  # 0.6 rad/s after the step
    for integrator in ("bdf1", "bdf2"):
        with test.subTest(integrator=integrator):
            solver = newton.solvers.SolverIPC(model, integrator=integrator, **_CONVERGED)
            s0, s1 = model.state(), model.state()
            s0.body_f.assign(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, alpha * I_zz]], np.float32))
            solver.step(s0, s1, None, None, dt)
            test.assertAlmostEqual(float(s1.body_qd.numpy()[b, 5]), alpha * dt, delta=0.01 * alpha * dt)


def test_bdf2_is_second_order_on_a_spring(test, device):
    """Two particles on a spring, started 1 cm beyond its rest length, oscillate as u'' = -omega^2 u. At t = 0.4 s
    (four radians) halving dt divides the error of integrator="bdf2" by about four, second order, and that of implicit
    Euler by about two; BDF2's error is the smaller."""
    m, k, L0 = 1.0, 50.0, 0.1
    omega, T = np.sqrt(2.0 * k / m), 0.4

    def error(integrator, h):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for x in (0.0, L0 + 0.01):
            builder.add_particle(wp.vec3(x, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.0), m, radius=0.001)
        builder.add_spring(0, 1, k, 0.0, 0.0)
        builder.spring_rest_length[0] = L0
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverIPC(model, integrator=integrator, **_TIGHT)
        rest = float(model.spring_rest_length.numpy()[0])
        x = solver._engine.positions()[:2, 0]
        u0 = x[1] - x[0] - rest
        s0, s1 = model.state(), model.state()
        for _ in range(round(T / h)):
            solver.step(s0, s1, None, None, h)
            s0, s1 = s1, s0
        x = solver._engine.positions()[:2, 0]
        return abs(x[1] - x[0] - rest - u0 * np.cos(omega * T))

    e1 = [error("bdf1", h) for h in (0.01, 0.005)]
    e2 = [error("bdf2", h) for h in (0.01, 0.005)]
    test.assertGreater(e1[0] / e1[1], 1.7)
    test.assertLess(e1[0] / e1[1], 2.3)
    test.assertGreater(e2[0] / e2[1], 3.4)
    test.assertLess(e2[0], 0.5 * e1[0])


def test_bdf2_history_starts_anew(test, device):
    """With integrator="bdf2" a state written between graph replays, and a reset, continue as a solver started from
    that state does, to 1e-6: the step from it is an implicit Euler step whose start becomes the history, where a BDF2
    step from the history the solver holds would move the bodies elsewhere."""
    tight = {"newton_velocity_tol": 1e-6, "linear_tol_rate": 1e-12, "semi_implicit": False}

    def scene():
        builder = newton.ModelBuilder()
        builder.add_soft_grid(
            pos=wp.vec3(-0.25, -0.05, 0.002), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0), dim_x=2, dim_y=2,
            dim_z=2, cell_x=0.05, cell_y=0.05, cell_z=0.05, density=1000.0, k_mu=3.4e4, k_lambda=1.6e5, k_damp=0.0,
        )  # fmt: skip
        _box_body(builder, (0.2, 0.0, 0.1005), (0.1, 0.1, 0.1))
        builder.add_ground_plane()
        return builder.finalize(device=device)

    for case in ("new state", "reset"):
        with test.subTest(case=case):
            model = scene()
            solver = newton.solvers.SolverIPC(model, integrator="bdf2", **tight)
            s0, s1 = model.state(), model.state()
            solver.step(s0, s1, None, None, 0.01)
            s0, s1 = s1, s0
            with wp.ScopedCapture(device=device) as capture:
                solver.step(s0, s1, None, None, 0.01)
                solver.step(s1, s0, None, None, 0.01)
            for _ in range(3):
                wp.capture_launch(capture.graph)
            q, qd = s0.particle_q.numpy(), s0.particle_qd.numpy()
            bq, bqd = s0.body_q.numpy(), s0.body_qd.numpy()
            if case == "new state":
                q[:, 2] += 0.03
                qd[:] = (0.2, 0.0, -0.1)
                bq[0, 2] += 0.02
                bqd[0] = (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
                for array, value in ((s0.particle_q, q), (s0.particle_qd, qd), (s0.body_q, bq), (s0.body_qd, bqd)):
                    array.assign(value)
            else:
                solver.reset(s0)
            continued = []
            for _ in range(4):
                wp.capture_launch(capture.graph)
                continued.append((s0.particle_q.numpy(), s0.body_q.numpy()))
            model = scene()
            fresh = newton.solvers.SolverIPC(model, integrator="bdf2", **tight)
            f0, f1 = model.state(), model.state()
            for array, value in ((f0.particle_q, q), (f0.particle_qd, qd), (f0.body_q, bq), (f0.body_qd, bqd)):
                array.assign(value)
            for step in range(8):
                fresh.step(f0, f1, None, None, 0.01)
                f0, f1 = f1, f0
                if step % 2:
                    np.testing.assert_allclose(continued[step // 2][0], f0.particle_q.numpy(), rtol=0.0, atol=1e-6)
                    np.testing.assert_allclose(continued[step // 2][1], f0.body_q.numpy(), rtol=0.0, atol=1e-6)


def test_bdf2_drives(test, device):
    """With integrator="bdf2" a revolute and a prismatic joint driven from rest by target gains, a passive spring and
    passive damping follow BDF2's recurrence of I q'' = ke (q* - q) + kd (qd* - qd) - ks (q - q_rest) - c qd after an
    implicit Euler step, step by step to 1e-4 of their motion; implicit Euler's recurrence is farther off."""
    for kind, length, scale in (("revolute", 0.1, 1.0), ("prismatic", 1.0, 10.0)):
        with test.subTest(kind):
            gains = {"target_ke": 40.0, "target_kd": 2.0, "stiffness": 10.0, "damping": 1.0}
            gains = {name: scale * value for name, value in gains.items()}
            targets = {"target_q": 0.5 * length, "target_qd": 0.1 * length, "rest_q": -0.2 * length}
            model, state, _ = _hinged(device, kind, axis=newton.Axis.X, radius=0.5, **targets, **gains)
            solver = newton.solvers.SolverIPC(model, integrator="bdf2", **_CONVERGED)
            m = float(model.body_mass.numpy()[0])
            inertia = float(model.body_inertia.numpy()[0][0, 0]) if kind == "revolute" else m
            h, steps = 0.01, 100
            q, _ = _steps(model, solver, state, steps, h, model.control())
            args = {
                "ke": gains["target_ke"], "kd": gains["target_kd"], "ks": gains["stiffness"], "c": gains["damping"],
                "q_target": targets["target_q"], "qd_target": targets["target_qd"], "q_rest": targets["rest_q"],
            }  # fmt: skip
            expected = _drive_recurrence_bdf2(inertia, h, steps, **args)
            scale_q = np.abs(expected).max()
            test.assertGreater(scale_q, 0.1 * length)
            np.testing.assert_allclose(q[:, 0], expected, rtol=0.0, atol=1e-4 * scale_q)
            euler = _drive_recurrence(inertia, h, steps, **args)
            test.assertGreater(np.abs(euler - expected).max(), 1e-3 * scale_q)


def test_al_ipc_rests_at_d_hat(test, device):
    """With contact_constitution="al-ipc" a box dropped onto the ground comes to rest with its gap at d_hat, the
    distance AL-IPC's constraint keeps, to 1%, and at rest; the barrier keeps it inside d_hat. (libuipc's AL-IPC does
    not keep a stack of two such boxes: on its own CUDA backend, as here, the upper one slides off within two
    seconds.) A free particle and a static box, which AL-IPC would not constrain, are refused."""
    d_hat = 0.005
    for constitution in ("ipc", "al-ipc"):
        with test.subTest(contact_constitution=constitution):
            builder = newton.ModelBuilder()
            _box_body(builder, (0.0, 0.0, 0.06), (0.05, 0.05, 0.05))
            builder.add_ground_plane()
            model = builder.finalize(device=device)
            solver = newton.solvers.SolverIPC(model, d_hat=d_hat, contact_constitution=constitution)
            s0, s1 = model.state(), model.state()
            for _ in range(200):
                solver.step(s0, s1, None, None, 0.01)
                s0, s1 = s1, s0
            gap = float(s0.body_q.numpy()[0, 2]) - 0.05
            test.assertLess(np.abs(s0.body_qd.numpy()).max(), 1e-3)
            if constitution == "al-ipc":
                test.assertAlmostEqual(gap, d_hat, delta=0.01 * d_hat)
            else:
                test.assertGreater(gap, 0.0)
                test.assertLess(gap, 0.99 * d_hat)
    # a particle on no cloth or tetrahedron takes part in no AL-IPC constraint, and a static shape has no mass for
    # its penalty: refused
    builder = newton.ModelBuilder()
    builder.add_particle(wp.vec3(0.0, 0.0, 0.02), wp.vec3(0.0, 0.0, 0.0), 0.01, radius=0.01)
    builder.add_ground_plane()
    with test.assertRaisesRegex(NotImplementedError, "particles on no cloth"):
        newton.solvers.SolverIPC(builder.finalize(device=device), contact_constitution="al-ipc")
    builder = newton.ModelBuilder()
    _box_body(builder, (0.0, 0.0, 0.17), (0.05, 0.05, 0.05))
    builder.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()), hx=0.1, hy=0.1, hz=0.05)
    with test.assertRaisesRegex(NotImplementedError, "static shapes"):
        newton.solvers.SolverIPC(builder.finalize(device=device), contact_constitution="al-ipc")


def test_al_ipc_status_and_new_masses(test, device):
    """contact_constitution="al-ipc" (libuipc's contact/constitution; other values refused) builds libuipc-warp's
    AL-IPC engine and refuses adaptive_kappa, a barrier stiffness, and the residual observable, which AL-IPC has not; a
    captured step writes STATUS, which check_status reads, and new body masses build the solve again (the AL-IPC
    penalty follows the masses it was built with), after which a captured step uses them."""
    builder = newton.ModelBuilder()
    _box_body(builder, (0.0, 0.0, 0.06), (0.05, 0.05, 0.05))
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    with test.assertRaisesRegex(ValueError, "contact_constitution"):
        newton.solvers.SolverIPC(model, contact_constitution="barrier")
    with test.assertRaisesRegex(ValueError, "adaptive_kappa"):
        newton.solvers.SolverIPC(model, contact_constitution="al-ipc", adaptive_kappa=True)
    solver = newton.solvers.SolverIPC(model, d_hat=0.005, contact_constitution="al-ipc")
    test.assertEqual(type(solver._engine).__name__, "ALIPCSimEngine")
    kind = solver.ObservableKind
    test.assertEqual(solver.supported_observables, {kind.STATUS})
    with test.assertRaisesRegex(ValueError, "solver_residual"):
        solver.observables({kind.STATUS, kind.SOLVER_RESIDUAL})
    observables = solver.observables({solver.ObservableKind.STATUS})
    s0, s1 = model.state(), model.state()
    solver.step(s0, s1, None, None, 0.01, observables=observables)
    with wp.ScopedCapture(device=device) as capture:
        solver.step(s1, s0, None, None, 0.01, observables=observables)
        solver.step(s0, s1, None, None, 0.01, observables=observables)
    wp.capture_launch(capture.graph)
    test.assertFalse(solver.check_status(observables))
    engine = solver._engine
    model.body_mass.assign(2.0 * model.body_mass.numpy())
    model.body_inv_mass.assign(0.5 * model.body_inv_mass.numpy())
    model.body_inertia.assign(2.0 * model.body_inertia.numpy())
    model.body_inv_inertia.assign(0.5 * model.body_inv_inertia.numpy())
    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    test.assertIsNot(solver._engine, engine)
    test.assertAlmostEqual(float(solver._engine.abd.mass.numpy()[0]), float(model.body_mass.numpy()[0]), places=4)
    with wp.ScopedCapture(device=device) as capture:
        solver.step(s1, s0, None, None, 0.01, observables=observables)
    wp.capture_launch(capture.graph)
    test.assertFalse(solver.check_status(observables))


def test_mas_matches_the_block_diagonal_preconditioner(test, device):
    """With fem_preconditioner="mas" (libuipc's linear_system/fem_preconditioner; other values refused) a stiff beam
    hanging from one end, and a soft cube and a cloth falling onto the ground, end where they end under the block
    diagonal preconditioner, to the tolerance of the solves, in at most 85% of its PCG iterations."""
    out = {}
    for preconditioner in ("diag", "mas"):
        builder = newton.ModelBuilder()
        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.5, 0.5), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0), dim_x=16, dim_y=2,
            dim_z=2, cell_x=0.02, cell_y=0.02, cell_z=0.02, density=1000.0, k_mu=1.0e6, k_lambda=4.0e6, k_damp=0.0,
            fix_left=True,
        )  # fmt: skip
        builder.add_soft_grid(
            pos=wp.vec3(-0.25, -0.1, 0.02), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0), dim_x=4, dim_y=4,
            dim_z=4, cell_x=0.05, cell_y=0.05, cell_z=0.05, density=1000.0, k_mu=3.4e5, k_lambda=1.6e6, k_damp=0.0,
        )  # fmt: skip
        _cloth(builder, 16, 0.02, 0.05, center=(0.3, -0.2))
        builder.add_ground_plane()
        model = builder.finalize(device=device)
        if preconditioner == "mas":
            with test.assertRaisesRegex(ValueError, "fem_preconditioner"):
                newton.solvers.SolverIPC(model, fem_preconditioner="jacobi")
        solver = newton.solvers.SolverIPC(model, fem_preconditioner=preconditioner, **_TIGHT)
        s0, s1 = model.state(), model.state()
        pcg = 0
        for _ in range(15):
            solver.step(s0, s1, None, None, 0.01)
            s0, s1 = s1, s0
            pcg += solver.frame_stats()["pcg_iterations"]
        out[preconditioner] = (s0.particle_q.numpy(), pcg)
    np.testing.assert_allclose(out["mas"][0], out["diag"][0], rtol=0.0, atol=1e-5)
    test.assertLess(out["mas"][1], 0.85 * out["diag"][1])


@requires("uipc_warp")
def _attached_pair(device, stiffness=2.0e4, damping=0.0):
    """VBD's test_body_particle_attachment_is_native_vbd without its damping: a free body of 1 kg (inertia 0.1, no
    shape) and a particle of 1 kg at its origin moving at 1 m/s along x, joined by an attachment at that origin,
    without gravity."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body(mass=1.0, inertia=wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1), lock_inertia=True)
    particle = builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.0), vel=wp.vec3(1.0, 0.0, 0.0), mass=1.0)
    builder.add_attachment_particle_body(particle, body, stiffness=stiffness, damping=damping)
    return builder.finalize(device=device), body, particle


def test_attachment_moves_a_free_body(test, device):
    """A particle attached to a free body that it starts moving: the pair moves together, its momentum stays the
    particle's 1 kg m/s to rounding, and the stretch follows the implicit-Euler recurrence of the two-mass oscillator,
    omega^2 = k (1 / m_p + 1 / m_b), from no stretch at a relative speed of 1 m/s, step by step to the Newton
    tolerance. The body only translates: the anchor is at its centre of mass."""
    model, body, particle = _attached_pair(device)
    solver = newton.solvers.SolverIPC(model, **_TIGHT)
    base = solver._scene.layout().body_base[body]
    dt, steps, k = 2.0e-3, 20, 2.0e4
    expected = _implicit_euler_oscillator(0.0, np.sqrt(2.0 * k), dt, steps, w0=1.0)
    s0, s1 = model.state(), model.state()
    stretch = []
    for _ in range(steps):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        x, v = solver._engine.positions(), solver._engine.velocities()
        stretch.append(x[particle, 0] - x[base, 0])
        np.testing.assert_allclose(v[particle] + v[base], [1.0, 0.0, 0.0], rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(x[base + 1 : base + 4], np.eye(3), rtol=0.0, atol=1e-12)  # no turn
    np.testing.assert_allclose(stretch, expected, rtol=0.0, atol=1e-9)
    anchor = s0.body_q.numpy()[body, :3]
    test.assertLess(np.linalg.norm(s0.particle_q.numpy()[particle] - anchor), 2.0e-2)  # VBD's bound
    test.assertGreater(anchor[0], 1.0e-3)


def _hung_cloth(device, height=0.3, stiffness=1.0e4, shared=False):
    """VBD's _simulate_cloth_hung_by_attachments: a 4 x 4 cloth hung by its corners 0 and 4 from a body of mass 0
    (static) at ``height``, above the ground. Each corner is attached at its own point of the body (VBD's test attaches
    both to the body's origin, which pulls corner 4 the 8 cm onto corner 0); ``shared``: a third attachment of corner
    1 at corner 0's point."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, height), wp.quat_identity()), mass=0.0)
    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, height), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0), dim_x=4, dim_y=4,
        cell_x=0.02, cell_y=0.02, mass=1.0e-3, tri_ke=1.0e3, tri_ka=1.0e3, tri_kd=0.0, particle_radius=2.0e-3,
    )  # fmt: skip
    corners = [0, 4] + ([1] if shared else [])
    points = [(0.0, 0.0, 0.0), (0.08, 0.0, 0.0), (0.0, 0.0, 0.0)]
    for particle, point in zip(corners, points, strict=False):
        builder.add_attachment_particle_body(particle, body, body_point=point, stiffness=stiffness)
    return builder.finalize(device=device), body, corners


def test_cloth_hangs_from_a_static_body(test, device):
    """A cloth hung by two corners from a static body: the corners stay within 1e-4 m of their points on the body
    (their share of the cloth's weight over the stiffness is 1.2e-5 m), the rest of the cloth hangs below them, clear
    of the ground, and no step fails. The anchors are no contact primitives; attachments at one point of one body
    share an anchor."""
    model, body, corners = _hung_cloth(device)
    solver = newton.solvers.SolverIPC(model)
    layout = solver._scene.layout()
    anchors = layout.body_anchor_offsets
    test.assertEqual(anchors[body + 1] - anchors[body], 2)
    test.assertFalse(set(range(anchors[0], anchors[-1])) & set(solver._engine.contact_points.tolist()))
    s0, s1 = model.state(), model.state()
    for _ in range(50):
        solver.step(s0, s1, None, None, 1.0e-3)
        s0, s1 = s1, s0
    q = s0.particle_q.numpy()
    np.testing.assert_allclose(q[corners], [[0.0, 0.0, 0.3], [0.08, 0.0, 0.3]], rtol=0.0, atol=1.0e-4)
    test.assertLess(q[:, 2].min(), 0.3 - 0.01)  # the far edge fell
    test.assertGreater(q[:, 2].min(), 0.2)
    shared_model, shared_body, _ = _hung_cloth(device, shared=True)
    shared = newton.solvers.SolverIPC(shared_model)
    offsets = shared._scene.layout().body_anchor_offsets
    test.assertEqual(offsets[shared_body + 1] - offsets[shared_body], 2)  # three attachments, two points


def test_attachment_enabled_inside_a_captured_graph(test, device):
    """A particle attached to a static body under gravity, ``attachment_particle_body_enabled`` switched off for ten
    steps and on again between the replays of a captured step: the steps follow eager ones given the same flags to
    1e-6 m, the particle falls while the attachment is off and is pulled back once it is on."""

    def run(capture):
        builder = newton.ModelBuilder()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), mass=0.0)
        builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.9), vel=wp.vec3(0.0, 0.0, 0.0), mass=0.01, radius=0.001)
        builder.add_attachment_particle_body(0, body, body_point=(0.0, 0.0, -0.1), stiffness=50.0)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverIPC(model, **_TIGHT)
        s0, s1 = model.state(), model.state()
        solver.step(s0, s1, None, None, 0.01)  # loads the kernels
        s0, s1 = s1, s0
        graph = None
        if capture:
            with wp.ScopedCapture(device=device) as cap:
                solver.step(s0, s1, None, None, 0.01)
                solver.step(s1, s0, None, None, 0.01)
            graph = cap.graph
        z = []
        for k in range(15):
            model.attachment_particle_body_enabled.assign(np.array([not 3 <= k < 8]))
            if capture:
                wp.capture_launch(graph)
            else:
                for _ in range(2):
                    solver.step(s0, s1, None, None, 0.01)
                    s0, s1 = s1, s0
            z.append(float(s0.particle_q.numpy()[0, 2]))
        return np.array(z)

    eager, captured = run(False), run(True)
    np.testing.assert_allclose(captured, eager, rtol=0.0, atol=1e-6)
    test.assertLess(eager[7], eager[2] - 0.05)  # ten steps of implicit Euler's free fall: g dt^2 55 = 0.054 m
    test.assertGreater(eager[8:].max(), eager[7] + 0.04)


def test_hard_attachment_holds_as_a_ball_joint(test, device):
    """A hard attachment (infinite stiffness) of a particle of mass m hanging from a static body takes the penalty of
    a ball joint, joint_strength_ratio gamma times the dynamic mass over dt^2: under gravity it rests g dt^2 / gamma
    below its point (9.8e-6 m at dt 0.01 s), to 1%, and twice as far with half the ratio."""
    for gamma in (100.0, 50.0):
        builder = newton.ModelBuilder()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), mass=0.0)
        builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.9), vel=wp.vec3(0.0, 0.0, 0.0), mass=0.02, radius=0.001)
        builder.add_attachment_particle_body(0, body, body_point=(0.0, 0.0, -0.1), stiffness=float("inf"))
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverIPC(model, joint_strength_ratio=gamma, **_TIGHT)
        s0, s1 = model.state(), model.state()
        for _ in range(20):
            solver.step(s0, s1, None, None, 0.01)
            s0, s1 = s1, s0
        sag = 0.9 - solver._engine.positions()[0, 2]
        test.assertAlmostEqual(sag / (9.81 * 0.01**2 / gamma), 1.0, delta=0.01)


def test_zero_length_springs_oscillate_as_implicit_euler(test, device):
    """Springs of rest length 0, libuipc's soft vertex stitch: without gravity two free particles of mass m 1 cm
    apart, and a free particle 1 cm from a kinematic one, follow the implicit-Euler oscillator of omega^2 = 2 k / m and
    k / m step by step to the Newton tolerance, and the free pair's centre stays put. The particles have no radius,
    so that they do not collide with each other (a spring changes no contact pair)."""
    m, k, dt, steps = 0.01, 4.0, 0.01, 60
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for x, y, mass in ((0.0, 0.0, m), (0.01, 0.0, m), (0.0, 1.0, 0.0), (0.01, 1.0, m)):
        builder.add_particle(wp.vec3(x, y, 1.0), wp.vec3(0.0, 0.0, 0.0), mass, radius=0.0)
    for i in (0, 2):
        builder.add_spring(i, i + 1, k, 0.0, 0.0)
        builder.spring_rest_length[-1] = 0.0
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, **_TIGHT)
    test.assertEqual([g.group.K for g in solver._uipc_warp.arrays.stitch_groups(solver._engine)], [2])
    x0 = solver._engine.positions()[:4, 0]
    expected = {
        0: _implicit_euler_oscillator(x0[1] - x0[0], np.sqrt(2.0 * k / m), dt, steps),
        2: _implicit_euler_oscillator(x0[3] - x0[2], np.sqrt(k / m), dt, steps),
    }
    s0, s1 = model.state(), model.state()
    measured = {first: [] for first in expected}
    for _ in range(steps):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        x = solver._engine.positions()[:4, 0]
        for first in expected:
            measured[first].append(x[first + 1] - x[first])
        test.assertAlmostEqual(x[0] + x[1], x0[0] + x0[1], delta=1e-12)
    for first, u in expected.items():
        np.testing.assert_allclose(measured[first], u, rtol=0.0, atol=1e-9)


def test_attachment_changes(test, device):
    """notify_model_changed(CONSTRAINT_PROPERTIES): a new stiffness is taken at the next step without building the
    solve again, damping is refused, and a new body point builds the solve again; a particle attached on its body's
    surface starts in contact with it, which the start check refuses."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    b = _box_body(builder, (0.0, 0.0, 1.0), (0.1, 0.1, 0.1))
    builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.8), vel=wp.vec3(0.0, 0.0, 0.0), mass=0.01, radius=0.001)
    builder.add_attachment_particle_body(0, b, body_point=(0.0, 0.0, -0.18), stiffness=10.0)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model)
    engine = solver._engine
    model.attachment_particle_body_stiffness.assign(np.array([20.0], np.float32))
    solver.notify_model_changed(newton.ModelFlags.CONSTRAINT_PROPERTIES)
    test.assertIs(solver._engine, engine)
    s0, s1 = model.state(), model.state()
    solver.step(s0, s1, None, None, 0.01)
    [rows] = solver._uipc_warp.arrays.stitch_groups(solver._engine)
    np.testing.assert_array_equal(rows.kappa.numpy(), [20.0])
    model.attachment_particle_body_damping.assign(np.array([1.0], np.float32))
    with test.assertRaisesRegex(NotImplementedError, "attachment 0, whose damping is 1.0"):
        solver.notify_model_changed(newton.ModelFlags.CONSTRAINT_PROPERTIES)
    model.attachment_particle_body_damping.assign(np.array([0.0], np.float32))
    model.attachment_particle_body_body_point.assign(np.array([[0.0, 0.0, -0.19]], np.float32))
    solver.notify_model_changed(newton.ModelFlags.CONSTRAINT_PROPERTIES)
    test.assertIsNot(solver._engine, engine)
    touching = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    b = _box_body(touching, (0.0, 0.0, 1.0), (0.1, 0.1, 0.1))
    touching.add_particle(pos=wp.vec3(0.0, 0.0, 0.9), vel=wp.vec3(0.0, 0.0, 0.0), mass=0.01, radius=0.001)
    touching.add_attachment_particle_body(0, b, body_point=(0.0, 0.0, -0.1))
    with test.assertRaisesRegex(ValueError, "body 0 and particle 0 start 2.2.*e-08 m apart"):
        newton.solvers.SolverIPC(touching.finalize(device=device))


def test_no_stitches_without_attachments(test, device):
    """A model without attachments and springs of rest length 0 gives the engine no stitch and no anchor, and a model
    without springs no contact surface of its own: the inputs of a model that uses neither are those of before."""
    model = _soft_cube(device, height=0.3)
    builder = newton.ModelBuilder()
    _box_body(builder, (0.0, 0.0, 0.5), (0.1, 0.1, 0.1))
    builder.add_ground_plane()
    for m in (model, builder.finalize(device=device)):
        solver = newton.solvers.SolverIPC(m)
        test.assertEqual(solver._uipc_warp.arrays.stitch_groups(solver._engine), [])
        layout = solver._scene.layout()
        test.assertEqual(layout.body_anchor_offsets[-1], layout.n)
        test.assertEqual(layout.body_anchor_offsets[0], layout.n)
        test.assertIsNone(solver._scene.engine_inputs(solver._config, "cpu")["surface"])


def _pusher(device, kinematic_motion, d_hat=0.005, steps=12, capture=False, capacity=None):
    """A kinematic box that moves 3 d_hat along x in every step of 0.01 s toward a box of 1 kg resting on the ground
    2 cm ahead of it, its pose and twist written into state_in before every step as Newton's kinematic bodies take
    them; ``capacity``: the engine's list capacities (None: its defaults). Returns the pusher's and the box's x after
    every step, the edge-triangle crossings of every step, the status bits of every step, the step that raised, if one
    did, and the solver."""
    from uipc_warp.validation import SurfaceChecker  # noqa: PLC0415

    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    box = _box_body(builder, (0.0, 0.0, 0.052), (0.05, 0.05, 0.05))
    pusher = builder.add_body(xform=wp.transform(wp.vec3(-0.12, 0.0, 0.052), wp.quat_identity()), is_kinematic=True)
    builder.add_shape_box(pusher, hx=0.05, hy=0.05, hz=0.05)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, d_hat=d_hat, kinematic_motion=kinematic_motion)
    if capacity is not None:
        solver._make_engine(capacity)
    observables = solver.observables({solver.ObservableKind.STATUS})
    checker = SurfaceChecker.from_engine(solver._engine)
    dt, speed = 0.01, 3.0 * d_hat / 0.01
    s0, s1 = model.state(), model.state()
    xs, crossings, bits, raised = [], [], [], None

    def prescribe(state, k):
        q, qd = state.body_q.numpy(), state.body_qd.numpy()
        q[pusher, :3] = (-0.12 + speed * k * dt, 0.0, 0.052)
        qd[pusher] = (speed, 0.0, 0.0, 0.0, 0.0, 0.0)
        state.body_q.assign(q)
        state.body_qd.assign(qd)

    graph = None
    for k in range(steps):
        prescribe(s0, k)
        try:
            if not capture or k == 0:  # the first step loads the kernels
                solver.step(s0, s1, None, None, dt, observables=observables)
            else:
                if graph is None:
                    with wp.ScopedCapture(device=device) as cap:
                        solver.step(s0, s1, None, None, dt, observables=observables)
                    graph = cap.graph
                wp.capture_launch(graph)
        except RuntimeError:
            raised = k
            break
        s0.assign(s1)  # the next step starts where this one ended
        xs.append(s0.body_q.numpy()[[pusher, box], 0].copy())
        crossings.append(checker.intersections(solver._engine.vm.x))
        bits.append(int(np.bitwise_or.reduce(observables.status.numpy())))
    return np.array(xs).reshape(-1, 2), crossings, bits, raised, solver


def test_kinematic_pusher_is_swept(test, device):
    """The acceptance of kinematic aims: a kinematic box that moves three times d_hat in every step drives a box across
    the ground. With kinematic_motion="aim" no edge crosses a triangle in any step, no status bit is set, the box
    stays ahead of the pusher, a face apart at most, and is carried 10 cm. With the default jump, the negative
    control, the pusher lands inside the box: a step fails, or edges cross, or the box ends behind it."""
    xs, crossings, bits, raised, _ = _pusher(device, "aim")
    test.assertIsNone(raised)
    test.assertEqual(sum(crossings), 0)
    test.assertEqual(bits, [0] * len(bits))
    gap = xs[:, 1] - xs[:, 0] - 0.1  # the faces' distance
    test.assertGreater(gap.min(), 0.0)
    test.assertGreater(xs[-1, 1], 0.1)
    np.testing.assert_allclose(xs[-1, 0], -0.12 + 12 * 0.015, atol=1e-3)  # the pusher kept to its path
    xs, crossings, bits, raised, _ = _pusher(device, "jump")
    failed = (
        raised is not None
        or sum(crossings) > 0
        or any(b & int(newton.solvers.SolverIPC.StatusBits.FAILED) for b in bits)
    )
    test.assertTrue(failed or (xs[:, 1] - xs[:, 0] - 0.1).min() <= 0.0, (raised, crossings, bits))


def test_kinematic_aims_captured_equal_eager(test, device):
    """The pusher's steps captured in a graph, its aims written by a kernel inside it from the state_in written before
    every replay, end where eager steps end, to 1e-6 m (the float32 output)."""
    eager, _, _, _, _ = _pusher(device, "aim", steps=8)
    captured, crossings, bits, raised, _ = _pusher(device, "aim", steps=8, capture=True)
    test.assertIsNone(raised)
    test.assertEqual(sum(crossings), 0)
    test.assertEqual(bits, [0] * len(bits))
    np.testing.assert_allclose(captured, eager, rtol=0.0, atol=1e-6)


def test_grown_lists_keep_kinematic_aims(test, device):
    """The pusher's steps with active contact lists of one pair: its first contact overflows them, and the eager step
    grows them and solves again from the step's start with the step's aims. The pusher and the box go on as in a
    solver with room, to 1e-5 m, with no crossing and no status bit."""
    roomy, _, _, _, _ = _pusher(device, "aim", steps=8)
    grown, crossings, bits, raised, solver = _pusher(device, "aim", steps=8, capacity=_TINY_LISTS)
    test.assertIsNone(raised)
    test.assertGreater(max(solver._engine.simplex.capacities()["active"].values()), 1)
    test.assertEqual(sum(crossings), 0)
    test.assertEqual(bits, [0] * len(bits))
    np.testing.assert_allclose(grown, roomy, rtol=0.0, atol=1e-5)


def test_kinematic_aims_under_bdf2(test, device):
    """With integrator="bdf2" a kinematic body moved by kinematic_motion="aim" along a path that accelerates and
    turns faster at every step (its pose and twist written into state_in before every step) ends every step on its
    aim, to 1e-9: the history of a restart that the solver writes for its rows puts BDF2's prediction on the aim, as
    implicit Euler's is."""
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()), is_kinematic=True)
    builder.add_shape_box(body, hx=0.1, hy=0.05, hz=0.02)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, kinematic_motion="aim", integrator="bdf2", **_TIGHT)
    base = solver._scene.layout().body_base[body]
    dt, a, alpha = 0.01, 2.0, 30.0
    s0, s1 = model.state(), model.state()
    for k in range(12):
        t = k * dt
        q, qd = s0.body_q.numpy(), s0.body_qd.numpy()
        q[body, :3] = (0.5 * a * t * t, 0.0, 0.5)
        q[body, 3:] = (0.0, 0.0, np.sin(0.25 * alpha * t * t), np.cos(0.25 * alpha * t * t))  # a turn about z
        qd[body] = (a * t, 0.0, 0.0, 0.0, 0.0, alpha * t)
        s0.body_q.assign(q)
        s0.body_qd.assign(qd)
        solver.step(s0, s1, None, None, dt)
        reached = solver._engine.positions()[base : base + 4]
        np.testing.assert_allclose(reached, solver._aim_group.q_aim.numpy()[0:4], rtol=0.0, atol=1e-9)
        s0, s1 = s1, s0


def _platform(device, kinematic_motion, steps, speed=0.5, dt=0.01, load_density=1000.0, platform_density=1000.0):
    """A kinematic platform 4 cm thick, sliding along x at ``speed`` (its pose and twist written into state_in before
    every step), with a box resting on it 2 mm up, under gravity. Returns the solver, the model, the platform, the box
    and the final state."""
    builder = newton.ModelBuilder()
    platform = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), is_kinematic=True)
    builder.add_shape_box(
        platform, hx=0.3, hy=0.3, hz=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=platform_density)
    )
    box = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.072), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=load_density))
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, kinematic_motion=kinematic_motion, **_TIGHT)
    s0, s1 = model.state(), model.state()
    for k in range(steps):
        q, qd = s0.body_q.numpy(), s0.body_qd.numpy()
        q[platform, :3] = (speed * k * dt, 0.0, 0.0)
        qd[platform] = (speed, 0.0, 0.0, 0.0, 0.0, 0.0)
        s0.body_q.assign(q)
        s0.body_qd.assign(qd)
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
    return solver, model, platform, box, s0


def test_kinematic_platform_carries_a_box(test, device):
    """A platform sliding at 0.5 m/s under a resting box carries it by friction with kinematic_motion="aim": after
    40 steps the box moves within eps_velocity of the platform's speed. With the default jump friction never sees the
    platform's motion, and the box stays behind (below 0.05 m/s)."""
    for mode, check in (("aim", lambda v: abs(v - 0.5) < 0.01), ("jump", lambda v: abs(v) < 0.05)):
        _, _, _, box, state = _platform(device, mode, 40)
        v = float(state.body_qd.numpy()[box, 0])
        test.assertTrue(check(v), f"{mode}: the box moves at {v} m/s")


def test_kinematic_body_lags_under_load(test, device):
    """A kinematic platform held still with kinematic_motion="aim" carries a box of weight F: it rests
    F dt^2 / ((1 + eta) m) below its aim, eta = kinematic_strength_ratio, m its mass, to 5%."""
    dt = 1.0 / 60.0
    solver, model, platform, box, _ = _platform(device, "aim", 60, speed=0.0, dt=dt, platform_density=100.0)
    m = float(solver._dyadic[0][platform])
    F = float(model.body_mass.numpy()[box]) * 9.81
    base = solver._scene.layout().body_base[platform]
    sag = -float(solver._engine.positions()[base, 2])
    test.assertAlmostEqual(sag / (F * dt * dt / (101.0 * m)), 1.0, delta=0.05)


def test_kinematic_root_tracks_its_angle(test, device):
    """A kinematic link on a revolute joint to the world, driven through eval_fk at 2 rad/s, with
    kinematic_motion="aim": its pose ends every step at the joint's angle to 1e-5 rad, and joint_q reports it."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    link = builder.add_link(is_kinematic=True)
    builder.add_shape_box(
        link, xform=wp.transform(wp.vec3(0.3, 0.0, 0.0), wp.quat_identity()), hx=0.2, hy=0.05, hz=0.05
    )
    joint = builder.add_joint_revolute(-1, link, axis=newton.Axis.Z)
    builder.add_articulation([joint])
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, kinematic_motion="aim", **_TIGHT)
    s0, s1 = model.state(), model.state()
    dt, omega = 0.01, 2.0
    for k in range(30):
        s0.joint_q.assign(np.array([omega * k * dt], np.float32))
        s0.joint_qd.assign(np.array([omega], np.float32))
        newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0, body_flag_filter=newton.BodyFlags.KINEMATIC)
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        q = s0.body_q.numpy()[link, 3:7]
        angle = 2.0 * np.arctan2(q[2], q[3])
        test.assertAlmostEqual(angle, omega * (k + 1) * dt, delta=1e-5)
        test.assertAlmostEqual(float(s0.joint_q.numpy()[0]), omega * (k + 1) * dt, delta=1e-5)


def test_reset_teleports_a_kinematic_body(test, device):
    """With kinematic_motion="aim" a kinematic body moved by state_in sweeps its way there and pushes a free box in its
    path; after reset() that selects its world it takes the new pose without sweeping, and the box stays put."""

    def run(reset):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        kin = builder.add_body(xform=wp.transform(wp.vec3(-0.3, 0.0, 0.0), wp.quat_identity()), is_kinematic=True)
        builder.add_shape_box(kin, hx=0.05, hy=0.05, hz=0.05)
        box = _box_body(builder, (0.0, 0.0, 0.0), (0.05, 0.05, 0.05))
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverIPC(model, kinematic_motion="aim")
        s0, s1 = model.state(), model.state()
        solver.step(s0, s1, None, None, 0.01)
        s0, s1 = s1, s0
        q = s0.body_q.numpy()
        q[kin, 0] = 0.3  # beyond the box
        s0.body_q.assign(q)
        if reset:
            solver.reset(s0, wp.array([True, True], dtype=wp.bool, device=device))
        solver.step(s0, s1, None, None, 0.01)
        return s1.body_q.numpy()[[kin, box], 0]

    swept, teleported = run(False), run(True)
    test.assertLess(swept[0], 0.3)  # blocked by the box, which it pushes
    test.assertGreater(swept[1], 0.001)
    test.assertAlmostEqual(float(teleported[0]), 0.3, delta=1e-6)
    test.assertEqual(float(teleported[1]), 0.0)


def test_jump_mode_builds_the_engine_of_before(test, device):
    """The default kinematic_motion="jump" builds a kinematic body as before: fixed, without a soft transform
    constraint, with the contact elements of before, its state_out a copy of state_in; "aim" builds it free, with the
    constraint and the kinematic and static shapes in elements of their own. Invalid options and AL-IPC with aims are
    refused, and so is a kinematic body of mass 0 whose shapes enclose no volume."""
    from newton._src.solvers.ipc.solver_ipc import _contact_elements  # noqa: PLC0415

    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    _box_body(builder, (0.0, 0.0, 0.2), (0.05, 0.05, 0.05))
    kin = builder.add_body(xform=wp.transform(wp.vec3(0.5, 0.0, 0.2), wp.quat_identity()), is_kinematic=True)
    builder.add_shape_box(kin, hx=0.05, hy=0.05, hz=0.05)
    model = builder.finalize(device=device)
    jump, aim = (newton.solvers.SolverIPC(model, kinematic_motion=m) for m in ("jump", "aim"))
    groups = jump._uipc_warp.arrays.constraint_groups
    test.assertIsNone(groups(jump._engine).transform)
    np.testing.assert_array_equal(groups(aim._engine).bodies, [kin])
    for solver, fixed in ((jump, 1), (aim, 0)):
        base = solver._scene.layout().body_base[kin]
        test.assertEqual(int(solver._engine.vm.fixed.numpy()[base]), fixed)
    # jump: the elements of before, one class for the three shapes; aim: a class per motion (dynamic, kinematic,
    # static), the kinematic and the static one apart
    before, moving = _contact_elements(model), _contact_elements(model, aims=True)
    np.testing.assert_array_equal(jump._enabled[1:, 1:], before.enabled[1:, 1:])  # element 0 holds no vertex
    np.testing.assert_array_equal(aim._enabled[1:, 1:], moving.enabled[1:, 1:])
    test.assertEqual(len(before.enabled), 2)
    shape_body = model.shape_body.numpy()
    ground, box_shape, kin_shape = (int(np.flatnonzero(shape_body == b)[0]) for b in (-1, 0, kin))
    e = moving.shape
    test.assertFalse(moving.enabled[e[kin_shape], e[ground]])
    test.assertTrue(moving.enabled[e[kin_shape], e[box_shape]] and moving.enabled[e[box_shape], e[ground]])
    s0, s1 = model.state(), model.state()
    jump.step(s0, s1, None, None, 0.01)
    np.testing.assert_array_equal(s1.body_q.numpy()[kin], s0.body_q.numpy()[kin])
    with test.assertRaisesRegex(ValueError, "kinematic_motion must be one of 'jump', 'aim'"):
        newton.solvers.SolverIPC(model, kinematic_motion="slide")
    with test.assertRaisesRegex(ValueError, "kinematic_strength_ratio"):
        newton.solvers.SolverIPC(model, kinematic_motion="aim", kinematic_strength_ratio=-1.0)
    with test.assertRaisesRegex(NotImplementedError, "al-ipc"):
        newton.solvers.SolverIPC(model, kinematic_motion="aim", contact_constitution="al-ipc")
    sheet = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    b = sheet.add_body(mass=0.0, is_kinematic=True)
    v = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], np.float32)
    sheet.add_shape_mesh(
        b, mesh=newton.Mesh(v, np.array([0, 1, 2], np.int32)), cfg=newton.ModelBuilder.ShapeConfig(density=0.0)
    )
    with test.assertRaisesRegex(NotImplementedError, "enclose no volume"):
        newton.solvers.SolverIPC(sheet.finalize(device=device), kinematic_motion="aim")


# ---------------------------------------------------------------------------- contact output (CONTACT_F)
_CONTACT_F = newton.solvers.SolverBase.ObservableKind.CONTACT_F


def _stacking(device, density=1000.0, lower_z=0.8, upper=True):
    """Newton's stacking scene of test_sensor_contact.py: a static base box with its top at z = 0.25, a 0.3 x 0.3 x
    0.5 m box "a" 30 cm above it and a 0.2 x 0.2 x 0.1 m box "b" 5 cm above that."""
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.density = density
    builder.add_shape_box(body=-1, hx=1.0, hy=1.0, hz=0.25, label="base")
    a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, lower_z), wp.quat_identity()), label="a")
    builder.add_shape_box(a, hx=0.15, hy=0.15, hz=0.25)
    if upper:
        b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, lower_z + 0.35), wp.quat_identity()), label="b")
        builder.add_shape_box(b, hx=0.1, hy=0.1, hz=0.05)
    return builder.finalize(device=device)


def _native_contacts(solver, model, rigid=None, soft=None):
    """The step's Contacts of a native contact backend, sized as the solver's export asks unless given."""
    pipeline = newton.CollisionPipeline(
        model,
        rigid_contact_max=solver.get_max_contact_count() if rigid is None else rigid,
        soft_contact_max=solver.get_max_soft_contact_count() if soft is None else soft,
    )
    return pipeline.contacts()


def _contact_steps(model, solver, contacts, observables, steps, dt, capture=False, state=None):
    """``steps`` steps (an even number when captured) from ``state`` (the model's), the observables written by each;
    returns the last output state."""
    s0 = model.state() if state is None else state
    s1 = model.state()
    if not capture:
        for _ in range(steps):
            solver.step(s0, s1, None, contacts, dt, observables=observables)
            s0, s1 = s1, s0
        return s0
    solver.step(s0, s1, None, contacts, dt, observables=observables)  # loads the state and the modules
    s0, s1 = s1, s0
    solver.step(s0, s1, None, contacts, dt, observables=observables)
    s0, s1 = s1, s0
    with wp.ScopedCapture(device=model.device) as capture_:
        solver.step(s0, s1, None, contacts, dt, observables=observables)
        solver.step(s1, s0, None, contacts, dt, observables=observables)
    for _ in range((steps - 2) // 2):
        wp.capture_launch(capture_.graph)
        solver.check_status(observables)
    return s0


def _transform_points(X, points):
    """``points`` (n, 3) of a body frame in the world, at the pose ``X`` (p, quaternion xyzw)."""
    p, q = np.asarray(X[:3], np.float64), np.asarray(X[3:], np.float64)
    v = np.asarray(points, np.float64).reshape(-1, 3)
    t = 2.0 * np.cross(q[:3], v)
    return v + q[3] * t + np.cross(q[:3], t) + p


def _rigid_rows(contacts, observables):
    """The written rigid rows: (shape0, shape1, contact_f rows)."""
    n = min(int(contacts.rigid_contact_count.numpy()[0]), contacts.rigid_contact_max)
    return (
        contacts.rigid_contact_shape0.numpy()[:n],
        contacts.rigid_contact_shape1.numpy()[:n],
        observables.contact_f.numpy()[:n],
    )


def test_contact_output_stacking_reports_weights(test, device):
    """Newton's stacking test with SolverIPC as a native contact backend, its steps captured: after 2 s at 1/240 s
    SensorContact, reading the rows the steps write, reports each box's weight within 1% (a: 45 kg, b: 4 kg, as the
    base holds a and b, and a holds b), the force-weighted contact positions sit at the interfaces within 5 cm (IPC
    rests up to d_hat apart: a on the base at z = 0.25, b on a at z = 0.75), and b never touches the base."""
    from newton.sensors import SensorContact  # noqa: PLC0415

    model = _stacking(device)
    solver = newton.solvers.SolverIPC(model)
    sensor = SensorContact(model, request_contact_attributes=False, sensing_bodies=["a", "b"], counterpart_shapes="*")
    contacts = _native_contacts(solver, model)
    kinds = set(sensor.solver_observable_kinds) | {solver.ObservableKind.STATUS}
    observables = solver.observables(kinds, contacts=contacts)
    test.assertIs(observables.contacts, contacts)
    state = _contact_steps(model, solver, contacts, observables, 470, 1.0 / 240.0, capture=True)
    s1 = model.state()
    total = np.zeros((2, 3))
    for _ in range(10):
        solver.step(state, s1, None, contacts, 1.0 / 240.0, observables=observables)
        state, s1 = s1, state
        sensor.update(state, contacts, observables=observables)
        total += sensor.total_force.numpy()
    total /= 10
    g = 9.81
    test.assertAlmostEqual(total[0, 2], 45.0 * g, delta=45.0 * g * 0.01)
    test.assertAlmostEqual(total[1, 2], 4.0 * g, delta=4.0 * g * 0.01)
    shape_base, shape_a, shape_b = 0, 1, 2
    row_a, row_b = sensor.sensing_indices.index(0), sensor.sensing_indices.index(1)
    positions = sensor.position_matrix.numpy()
    np.testing.assert_allclose(
        positions[row_a, sensor.counterpart_indices[row_a].index(shape_base)], [0, 0, 0.25], atol=0.05
    )
    np.testing.assert_allclose(
        positions[row_a, sensor.counterpart_indices[row_a].index(shape_b)], [0, 0, 0.75], atol=0.05
    )
    np.testing.assert_allclose(
        positions[row_b, sensor.counterpart_indices[row_b].index(shape_a)], [0, 0, 0.75], atol=0.05
    )
    np.testing.assert_array_equal(positions[row_b, sensor.counterpart_indices[row_b].index(shape_base)], 0.0)
    shape0, shape1, _ = _rigid_rows(contacts, observables)
    pairs = {tuple(sorted(p)) for p in zip(shape0.tolist(), shape1.tolist(), strict=True)}
    test.assertEqual(pairs, {(shape_base, shape_a), (shape_a, shape_b)})


def test_contact_output_friction_at_rest(test, device):
    """Newton's stacking-friction test: a 4.5 kg box at rest on the base reports its weight within 2% and a friction
    force within 2% of it, its barrier rows pushing along their normals and its friction rows holding no more than
    the slip of a body at rest asks."""
    from newton.sensors import SensorContact  # noqa: PLC0415

    model = _stacking(device, density=100.0, upper=False)
    solver = newton.solvers.SolverIPC(model)
    sensor = SensorContact(model, request_contact_attributes=False, sensing_bodies=["a"])
    contacts = _native_contacts(solver, model)
    observables = solver.observables(sensor.solver_observable_kinds, contacts=contacts)
    state = _contact_steps(model, solver, contacts, observables, 470, 1.0 / 240.0)
    s1 = model.state()
    total, friction = np.zeros(3), np.zeros(3)
    for _ in range(10):
        solver.step(state, s1, None, contacts, 1.0 / 240.0, observables=observables)
        state, s1 = s1, state
        sensor.update(state, contacts, observables=observables)
        total += sensor.total_force.numpy()[0]
        friction += sensor.total_force_friction.numpy()[0]
    weight = 4.5 * 9.81
    test.assertAlmostEqual(total[2] / 10, weight, delta=weight * 0.02)
    np.testing.assert_allclose(friction / 10, 0.0, atol=weight * 0.02)


def test_contact_output_soft_rows(test, device):
    """A cloth draped over a static box and a free particle on the ground: the soft rows' forces on the shapes add up
    to the cloth's and the particle's weights within 1%; every row names a shape, its record is a particle, an edge or
    a face of the cloth (all three occur) with weights that add up to 1, its point lies on the shape's surface, its
    normal points from that point toward the record's point, and its torque is that of its force at its point about
    the world origin (static shapes); contact of the cloth with itself gives no row, and there are no rigid rows."""
    builder = newton.ModelBuilder()
    _cloth(builder, 12, 0.02, 0.2050, mass=1.0e-3, tri_k=1.0e3, edge_ke=1.0e-3)  # 2 cm wider than the box each way
    builder.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()), hx=0.1, hy=0.1, hz=0.1)
    builder.add_particle(wp.vec3(0.5, 0.0, 0.0101), wp.vec3(0.0, 0.0, 0.0), 0.01, radius=0.01)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model)
    test.assertEqual(solver.get_max_soft_contact_count(), solver.get_max_contact_count())
    contacts = _native_contacts(solver, model)
    observables = solver.observables({_CONTACT_F}, contacts=contacts)
    state = _contact_steps(model, solver, contacts, observables, 200, 0.01)  # the flaps settle after 1.5 s
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
    n = int(contacts.soft_contact_count.numpy()[0])
    test.assertGreater(n, 0)
    test.assertLessEqual(n, contacts.soft_contact_max)
    shape = contacts.soft_contact_shape.numpy()[:n]
    indices = contacts.soft_contact_indices.numpy()[:n]
    weights = contacts.soft_contact_barycentric.numpy()[:n]
    particle = contacts.soft_contact_particle.numpy()[:n]
    pos = contacts.soft_contact_body_pos.numpy()[:n]
    normal = contacts.soft_contact_normal.numpy()[:n]
    wrench = observables.contact_f.numpy()[contacts.rigid_contact_max : contacts.rigid_contact_max + n]
    box, ground = 0, 1
    cloth = np.arange(model.particle_count - 1)
    free = model.particle_count - 1
    test.assertTrue(np.isin(shape, [box, ground]).all())
    kinds = (indices >= 0).sum(axis=1)
    test.assertTrue(np.isin(kinds, [1, 2, 3]).all())
    test.assertTrue({1, 2, 3}.issubset(set(kinds.tolist())))  # particle, edge and face records
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(particle, np.where(kinds == 1, indices[:, 0], -1))
    on_box = shape == box
    test.assertTrue(np.isin(indices[on_box][indices[on_box] >= 0], cloth).all())
    np.testing.assert_array_equal(indices[~on_box], [[free, -1, -1]] * int((~on_box).sum()))
    # on the surface of the box [-0.1, 0.1]^2 x [0, 0.2] (a static shape's frame is the world's), and on the ground
    surface = np.abs(pos[on_box] - [0.0, 0.0, 0.1]).max(axis=1)
    np.testing.assert_allclose(surface, 0.1, atol=1e-5)
    np.testing.assert_allclose(pos[~on_box, 2], 0.0, atol=1e-6)
    q = state.particle_q.numpy()
    soft_point = np.einsum("ij,ijk->ik", weights, q[np.maximum(indices, 0)] * (indices >= 0)[:, :, None])
    test.assertTrue((np.einsum("ij,ij->i", normal, soft_point - pos) > 0.0).all())
    np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1.0, atol=1e-5)
    cloth_weight = model.particle_mass.numpy()[cloth].sum() * 9.81
    np.testing.assert_allclose(wrench[on_box, :3].sum(axis=0), [0.0, 0.0, -cloth_weight], atol=0.01 * cloth_weight)
    np.testing.assert_allclose(wrench[~on_box, :3].sum(axis=0), [0.0, 0.0, -0.01 * 9.81], atol=0.01 * 0.01 * 9.81)
    # the torque about the world origin, of the forces on the box's vertices: that of the force at the point where the
    # force splits by the closest point's weights (a particle or face record: a triangle's or a single vertex's)
    split = kinds != 2
    torque = np.cross(pos[split], wrench[split, :3])
    np.testing.assert_allclose(wrench[split, 3:], torque, rtol=0, atol=1e-5 * max(np.abs(torque).max(), 1e-3))


def test_contact_output_rigid_rows_balance_a_body(test, device):
    """A box held on a static ramp by friction, its steps converged tightly (at the default newton_velocity_tol the
    box creeps down at 1.2 mm/s, and its forces balance only to that tolerance, 0.12 N of its 9.81 N): its rows'
    forces add up to its weight and their torques about its centre of mass to zero, each row holding the force on its
    shape 0 with its torque about that body's centre of mass at the end of the step (the world origin for the static
    ramp); the points lie on the box's bottom face in its frame of state_out; the margins are the shapes'
    shape_margin, the offsets the normal times them, and every normal is the ramp's."""
    builder = newton.ModelBuilder()
    angle = 0.2
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle)
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.8, margin=0.002)
    builder.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), rot), hx=0.5, hy=0.5, hz=0.05, cfg=cfg)
    up = np.array(wp.quat_rotate(rot, wp.vec3(0.0, 0.0, 1.0)))
    body = builder.add_body(xform=wp.transform(wp.vec3(*(up * (0.05 + 0.05 + 0.0045))), rot))
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
    model = builder.finalize(device=device)
    tight = {"newton_velocity_tol": 1e-4, "linear_tol_rate": 1e-10, "semi_implicit": False}
    solver = newton.solvers.SolverIPC(model, d_hat=0.005, **tight)
    contacts = _native_contacts(solver, model)
    observables = solver.observables({_CONTACT_F}, contacts=contacts)
    state = _contact_steps(model, solver, contacts, observables, 60, 0.01)
    shape0, _, f = _rigid_rows(contacts, observables)
    test.assertGreater(len(shape0), 0)
    sign = np.where(shape0 == 1, 1.0, -1.0)  # the box's shape is 1: a row with it as shape 0 holds the force on it
    on_box = sign[:, None] * f[:, :3]
    np.testing.assert_allclose(on_box.sum(axis=0), [0.0, 0.0, model.body_mass.numpy()[0] * 9.81], rtol=0, atol=1e-3)
    # torques about the box's centre: a row of the box as shape 0 holds it; a row of the ramp as shape 0 holds the
    # torque about the world origin, moved here to the box's centre
    X = state.body_q.numpy()[0]
    com = _transform_points(X, model.body_com.numpy()[0])[0]
    n = len(shape0)
    point1 = _transform_points(X, contacts.rigid_contact_point1.numpy()[:n])
    # a row of the box as shape 0 holds the torque about its centre; a row of the ramp as shape 0 the force on the
    # ramp, whose reaction acts on the box at the box's point
    torque = np.where(shape0[:, None] == 1, f[:, 3:], np.cross(point1 - com, -f[:, :3]))
    test.assertLess(np.linalg.norm(torque.sum(axis=0)), 1e-3 * model.body_mass.numpy()[0] * 9.81 * 0.05)
    local = np.where(
        shape0[:, None] == 1, contacts.rigid_contact_point0.numpy()[:n], contacts.rigid_contact_point1.numpy()[:n]
    )
    np.testing.assert_allclose(local[:, 2], -0.05, atol=1e-4)  # on the box's bottom face, in its body frame
    test.assertTrue((np.abs(local[:, :2]) <= 0.05 + 1e-4).all())
    margin0, margin1 = contacts.rigid_contact_margin0.numpy()[:n], contacts.rigid_contact_margin1.numpy()[:n]
    np.testing.assert_allclose(margin0, 0.002, rtol=1e-6)
    np.testing.assert_allclose(margin1, 0.002, rtol=1e-6)
    normal = contacts.rigid_contact_normal.numpy()[:n]
    offset0 = contacts.rigid_contact_offset0.numpy()[:n]
    ramp_rows = shape0 == 0
    np.testing.assert_allclose(offset0[ramp_rows], 0.002 * normal[ramp_rows], atol=1e-7)  # the static ramp's frame
    np.testing.assert_allclose(np.abs(normal @ up), 1.0, atol=1e-3)  # every row's normal is the ramp's


def test_contact_output_only_enabled_pairs(test, device):
    """Rows only between pairs that collide: a box made a filter pair with the box under it falls through it to the
    ground, and no row joins the two; the other pairs give rows."""
    builder = newton.ModelBuilder()
    ground = builder.add_ground_plane()
    lower = builder.shape_count
    _box_body(builder, (0.0, 0.0, 0.051), (0.05, 0.05, 0.05))
    upper = builder.shape_count
    _box_body(builder, (0.02, 0.0, 0.16), (0.03, 0.03, 0.03))
    builder.add_shape_collision_filter_pair(lower, upper)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model)
    contacts = _native_contacts(solver, model)
    observables = solver.observables({_CONTACT_F}, contacts=contacts)
    pairs = set()
    s0, s1 = model.state(), model.state()
    for _ in range(40):
        solver.step(s0, s1, None, contacts, 0.01, observables=observables)
        s0, s1 = s1, s0
        shape0, shape1, _ = _rigid_rows(contacts, observables)
        pairs |= {tuple(sorted(p)) for p in zip(shape0.tolist(), shape1.tolist(), strict=True)}
    test.assertNotIn((lower, upper), pairs)
    test.assertEqual(pairs, {tuple(sorted((lower, ground))), tuple(sorted((upper, ground)))})


def test_contact_output_capacity(test, device):
    """A Contacts smaller than the solver's worst case is taken: the counts are the true numbers, the rows past the
    capacity are dropped and the rows within it hold the same terms the full buffer's rows do."""
    model = _stacking(device, upper=False, lower_z=0.502)
    solver = newton.solvers.SolverIPC(model)
    full = _native_contacts(solver, model)
    observables = solver.observables({_CONTACT_F}, contacts=full)
    state = _contact_steps(model, solver, full, observables, 20, 0.01)
    rows = int(full.rigid_contact_count.numpy()[0])
    test.assertGreater(rows, 3)
    small = newton.Contacts(3, 0, device=device)
    s1 = model.state()
    solver.step(state, s1, None, small, 0.01, observables=observables)
    test.assertIs(observables.contacts, small)
    test.assertEqual(int(small.rigid_contact_count.numpy()[0]), rows)
    test.assertTrue(np.isin(small.rigid_contact_shape0.numpy(), [0, 1]).all())
    test.assertTrue((np.linalg.norm(observables.contact_f.numpy()[:3, :3], axis=1) > 0.0).all())
    np.testing.assert_array_equal(observables.contact_f.numpy()[3:], 0.0)
    # no soft rows can come: the soft count stays 0 with a buffer that holds none
    test.assertEqual(int(small.soft_contact_count.numpy()[0]), 0)
    with test.assertRaisesRegex(ValueError, "Pass the step's Contacts"):
        solver.step(s1, state, None, None, 0.01, observables=observables)


def test_contact_output_not_requested(test, device):
    """Without CONTACT_F nothing of the output exists or runs: no records are allocated, a step launches none of its
    kernels, and the Contacts passed to step is not touched; requesting it allocates the records once."""
    model = _stacking(device, upper=False, lower_z=0.502)
    solver = newton.solvers.SolverIPC(model)
    contacts = newton.Contacts(16, 4, device=device)
    contacts.rigid_contact_count.fill_(7)
    contacts.soft_contact_count.fill_(5)
    observables = solver.observables({solver.ObservableKind.STATUS})
    launched = []
    launch = wp.launch

    def recording(kernel, *args, **kwargs):
        launched.append(getattr(kernel, "key", str(kernel)))
        return launch(kernel, *args, **kwargs)

    s0, s1 = model.state(), model.state()
    with mock.patch.object(wp, "launch", recording):
        for _ in range(3):
            solver.step(s0, s1, None, contacts, 0.01, observables=observables)
            s0, s1 = s1, s0
    test.assertIsNone(solver._contact_export)
    test.assertGreater(len(launched), 0)
    test.assertFalse([k for k in launched if "contact_rows" in k or "contact_force_records" in k])
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 7)
    test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 5)
    observables = solver.observables({_CONTACT_F}, contacts=contacts)
    export = solver._contact_export
    test.assertIsNotNone(export)
    solver.observables({_CONTACT_F}, contacts=contacts)
    test.assertIs(solver._contact_export, export)
    _contact_steps(model, solver, contacts, observables, 20, 0.01, state=s0)  # the box bounces off the base first
    test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 0)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_contact_output_captured_equals_eager(test, device):
    """Captured steps write the rows eager steps write: the same pairs with the same forces summed per pair, to the
    rounding of the solve, over 40 steps of the stacking scene."""

    def run(capture):
        model = _stacking(device)
        solver = newton.solvers.SolverIPC(model, newton_velocity_tol=1e-6, linear_tol_rate=1e-12, semi_implicit=False)
        contacts = _native_contacts(solver, model)
        observables = solver.observables({_CONTACT_F, solver.ObservableKind.STATUS}, contacts=contacts)
        _contact_steps(model, solver, contacts, observables, 120, 1.0 / 60.0, capture=capture)
        shape0, shape1, f = _rigid_rows(contacts, observables)
        totals = {}
        for a, b, w in zip(shape0.tolist(), shape1.tolist(), f, strict=True):
            key, sign = ((a, b), 1.0) if a < b else ((b, a), -1.0)
            totals[key] = totals.get(key, 0.0) + sign * w[:3]
        return totals

    eager, captured = run(False), run(True)
    test.assertEqual(set(eager), set(captured))
    for key in eager:
        np.testing.assert_allclose(captured[key], eager[key], rtol=0, atol=1e-3 * np.abs(eager[key]).max())


def test_contact_output_failed_step_has_no_rows(test, device):
    """A step that fails writes no rows: an eager step from a state through the ground raises and leaves zero counts
    and forces; the next step from a valid state writes rows again."""
    model, touching = _boxes_apart(device)
    solver = newton.solvers.SolverIPC(model)
    contacts = _native_contacts(solver, model)
    observables = solver.observables({_CONTACT_F}, contacts=contacts)
    s0, s1 = model.state(), model.state()
    solver.step(s0, s1, None, contacts, 0.01, observables=observables)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
    s0.body_q.assign(touching)
    with test.assertRaises(RuntimeError):
        solver.step(s0, s1, None, contacts, 0.01, observables=observables)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
    np.testing.assert_array_equal(observables.contact_f.numpy(), 0.0)
    s0 = model.state()
    solver.step(s0, s1, None, contacts, 0.01, observables=observables)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_contact_output_after_grow(test, device):
    """Contact lists that overflow a captured step and grow with grow(): the export follows the grown solve, whose
    get_max_contact_count() is larger, and the steps captured again write their rows."""
    model = _stacking_cubes(device)
    solver = newton.solvers.SolverIPC(model)
    solver._make_engine(_TINY_LISTS)
    contacts = newton.Contacts(4096, 4096, device=device)
    observables = solver.observables({_CONTACT_F, solver.ObservableKind.STATUS}, contacts=contacts)
    before = solver.get_max_soft_contact_count()
    s0, s1 = model.state(), model.state()

    def capture():
        with wp.ScopedCapture(device=device) as cap:
            solver.step(s0, s1, None, contacts, 0.01, observables=observables)
        return cap.graph

    graph, grown = capture(), 0
    for _ in range(30):
        wp.capture_launch(graph)
        if solver.check_status(observables, grow=True):
            grown += 1
            graph = capture()
            wp.capture_launch(graph)
            test.assertFalse(solver.check_status(observables))
        wp.copy(s0.particle_q, s1.particle_q)
        wp.copy(s0.particle_qd, s1.particle_qd)
    test.assertGreater(grown, 0)
    test.assertGreater(solver.get_max_soft_contact_count(), before)
    test.assertEqual(solver._contact_export.capacity, solver.get_max_soft_contact_count())
    # the cubes' rows on the ground; the cubes on each other give none (between particle features)
    rows = int(contacts.soft_contact_count.numpy()[0])
    test.assertGreater(rows, 0)
    test.assertTrue((contacts.soft_contact_shape.numpy()[:rows] == 0).all())


def test_contact_output_refusals(test, device):
    """CONTACT_F is not offered with AL-IPC; a model with fewer than two colliding shapes has no rigid rows and one
    without particles no soft rows."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    _box_body(builder, (0.0, 0.0, 0.06), (0.05, 0.05, 0.05))
    al = newton.solvers.SolverIPC(builder.finalize(device=device), contact_constitution="al-ipc")
    test.assertNotIn(_CONTACT_F, al.supported_observables)
    with test.assertRaisesRegex(ValueError, "does not support solver observable"):
        al.observables({_CONTACT_F}, contacts=newton.Contacts(4, 0, device=device))
    solver = newton.solvers.SolverIPC(_stacking(device, upper=False, lower_z=0.502))
    test.assertIn(_CONTACT_F, solver.supported_observables)
    test.assertGreater(solver.get_max_contact_count(), 0)
    test.assertEqual(solver.get_max_soft_contact_count(), 0)
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.add_particle(wp.vec3(0.0, 0.0, 0.02), wp.vec3(0.0, 0.0, 0.0), 0.01, radius=0.01)
    lone = newton.solvers.SolverIPC(builder.finalize(device=device))
    test.assertEqual(lone.get_max_contact_count(), 0)
    test.assertGreater(lone.get_max_soft_contact_count(), 0)


def test_contact_output_coupled_worlds(test, device):
    """With coupled_worlds the rows of every world share the buffer: in four copies of a world of a box on a global
    ground, SensorContact reports every world's box at its weight within 1%. The worlds are one solve, so a step that
    fails in one world (its box written through the ground) fails in every world and writes no row of any."""
    from newton.sensors import SensorContact  # noqa: PLC0415

    def content():
        world = newton.ModelBuilder()
        _box_body(world, (0.0, 0.0, 0.0505), (0.05, 0.05, 0.05))
        return world

    scene = newton.ModelBuilder()
    scene.add_ground_plane()
    scene.replicate(content(), 4)
    model = scene.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, coupled_worlds=True)
    sensor = SensorContact(model, request_contact_attributes=False, sensing_bodies="*")
    contacts = _native_contacts(solver, model)
    observables = solver.observables(sensor.solver_observable_kinds, contacts=contacts)
    state = _contact_steps(model, solver, contacts, observables, 60, 0.01)
    sensor.update(state, contacts, observables=observables)
    weight = model.body_mass.numpy() * 9.81
    np.testing.assert_allclose(sensor.total_force.numpy()[:, 2], weight, rtol=0.01)
    rows = int(contacts.rigid_contact_count.numpy()[0])
    worlds = model.shape_world.numpy()
    shapes = np.concatenate(
        [contacts.rigid_contact_shape0.numpy()[:rows], contacts.rigid_contact_shape1.numpy()[:rows]]
    )
    test.assertEqual(set(worlds[shapes].tolist()) - {-1}, {0, 1, 2, 3})
    q = state.body_q.numpy()
    q[0, 2] = 0.04  # world 0's box 1 cm through the ground
    state.body_q.assign(q)
    with test.assertRaises(RuntimeError):
        solver.step(state, model.state(), None, contacts, 0.01, observables=observables)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
    np.testing.assert_array_equal(observables.contact_f.numpy(), 0.0)


# ---------------------------------------------------------------------------- joints on request (enable_*)


def _group_arrays(solver):
    """Every device array of every joint group of ``solver``'s engine, by kind and name, on the host."""
    groups = solver._uipc_warp.arrays.joint_groups(solver._engine)
    out = {}
    for kind in ("revolute", "prismatic", "fixed", "spherical"):
        group = getattr(groups, kind)
        if group is not None:
            for name, value in vars(group).items():
                if isinstance(value, wp.array):
                    out[kind, name] = value.numpy().copy()
    return out


def _d6_twins(device, d6):
    """A chain of every joint SolverIPC maps a D6 joint onto, y up with gravity: an arm on a driven, limited revolute
    joint from the world, a box welded to it, a bob on a ball joint from it, and a slider on a driven, limited
    prismatic joint from the world. With ``d6`` every joint is the D6 joint of the same axes (one angular, none, three
    angular, one linear)."""
    D = newton.ModelBuilder.JointDofConfig
    builder = newton.ModelBuilder(gravity=(0.0, -10.0, 0.0), up_axis=newton.Axis.Y)
    arm = builder.add_link(xform=wp.transform((0.0, -0.5, 0.0), wp.quat_identity()))
    builder.add_shape_box(arm, hx=0.05, hy=0.5, hz=0.05)
    weld = builder.add_link(xform=wp.transform((0.15, -0.9, 0.0), wp.quat_identity()))
    builder.add_shape_box(weld, hx=0.1, hy=0.05, hz=0.05)
    bob = builder.add_link(xform=wp.transform((0.0, -1.3, 0.0), wp.quat_identity()))
    builder.add_shape_sphere(bob, radius=0.1)
    slider = builder.add_link(xform=wp.transform((1.0, 0.0, 0.0), wp.quat_identity()))
    builder.add_shape_box(slider, hx=0.1, hy=0.1, hz=0.1)
    drive = {"target_ke": 20.0, "target_kd": 1.0, "limit_lower": -0.6, "limit_upper": 0.6, "limit_ke": 1.0e4}
    hinge = {"child_xform": wp.transform((0.0, 0.5, 0.0), wp.quat_identity())}
    weld_at = {"parent_xform": wp.transform((0.15, -0.4, 0.0), wp.quat_identity())}
    ball_at = {
        "parent_xform": wp.transform((0.0, -0.6, 0.0), wp.quat_identity()),
        "child_xform": wp.transform((0.0, 0.2, 0.0), wp.quat_identity()),
    }
    slide_at = {"parent_xform": wp.transform((1.0, 0.0, 0.0), wp.quat_identity())}
    if d6:
        joints = [
            builder.add_joint_d6(-1, arm, angular_axes=[D(axis=newton.Axis.Z, **drive)], **hinge),
            builder.add_joint_d6(arm, weld, **weld_at),
            builder.add_joint_d6(arm, bob, angular_axes=[D(axis=a) for a in "XYZ"], **ball_at),
            builder.add_joint_d6(-1, slider, linear_axes=[D(axis=newton.Axis.Y, **drive)], **slide_at),
        ]
    else:
        joints = [
            builder.add_joint_revolute(-1, arm, axis=newton.Axis.Z, **drive, **hinge),
            builder.add_joint_fixed(arm, weld, **weld_at),
            builder.add_joint_ball(arm, bob, **ball_at),
            builder.add_joint_prismatic(-1, slider, axis=newton.Axis.Y, **drive, **slide_at),
        ]
    builder.add_articulation(joints)
    model = builder.finalize(device=device)
    state = model.state()
    q = model.joint_q.numpy()
    q[0] = 0.5  # the arm's angle, the first coordinate either way
    state.joint_q.assign(q)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, state


def test_d6_joints_take_libuipcs_joints(test, device):
    """With enable_d6_joints=True a D6 joint of one angular axis is libuipc's revolute joint about it, of one linear
    axis its prismatic joint along it, of no axis its fixed joint and of three orthonormal angular axes its ball joint,
    with their drives, limits and joint_f: a chain of the four built with D6 joints gives the engine the joint groups of
    the same chain built with Newton's revolute, fixed, ball and prismatic joints, array by array, bit for bit, and
    steps as it does, the revolute and prismatic coordinates included. Without the option the D6 joints are refused."""
    with test.assertRaisesRegex(NotImplementedError, "D6 joints \\(joint 0\\).*enable_d6_joints=True"):
        newton.solvers.SolverIPC(_d6_twins(device, True)[0])
    runs = []
    for d6 in (False, True):
        model, state = _d6_twins(device, d6)
        solver = newton.solvers.SolverIPC(model, enable_d6_joints=d6)
        groups = _group_arrays(solver)
        control = model.control()
        f = np.zeros(model.joint_dof_count, np.float32)
        f[0], f[-1] = 0.3, -2.0  # the arm's torque and the slider's force, the first and the last DOF either way
        control.joint_f.assign(f)
        control.joint_target_q.assign(np.where(np.arange(model.joint_coord_count) == 0, -0.2, 0.0).astype(np.float32))
        frames, _ = _steps(
            model, solver, state, 60, 0.01, control,
            lambda s: np.concatenate([s.body_q.numpy().ravel(), s.joint_q.numpy()[[0, -1]]]),
        )  # fmt: skip
        runs.append((groups, frames))
    (native_groups, native), (d6_groups, d6) = runs
    test.assertEqual(sorted(native_groups), sorted(d6_groups))
    for key, value in native_groups.items():
        np.testing.assert_array_equal(d6_groups[key], value, err_msg=str(key))
    test.assertGreater(native[:, -2].max() - native[:, -2].min(), 0.3)  # the arm swings
    np.testing.assert_allclose(d6, native, rtol=0.0, atol=1e-6)


def _d6_body(device, lin, axes, q, qd, gravity=0.0, offset=0.2):
    """A body of mass 1 and inertia diag(0.1, 0.2, 0.15) on a D6 joint from the world, posed by eval_fk at the
    coordinates ``q`` and rates ``qd`` (z up), with three linear axes when ``lin``, a sphere of radius 0.05 for its
    collision shape and its joint frame ``offset`` along its z axis. Returns the model, the posed state and the
    joint."""
    D = newton.ModelBuilder.JointDofConfig
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -gravity))
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.diag([0.1, 0.2, 0.15])))
    builder.add_shape_sphere(body, radius=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    linear = [D.create_unlimited(a) for a in "XYZ"] if lin else []
    j = builder.add_joint_d6(
        -1, body, linear_axes=linear, angular_axes=[D.create_unlimited(a) for a in axes],
        parent_xform=wp.transform((0.0, 0.0, 1.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.3)),
        child_xform=wp.transform((0.0, 0.0, offset), wp.quat_identity()),
    )  # fmt: skip
    builder.add_articulation([j])
    model = builder.finalize(device=device)
    state = model.state()
    state.joint_q.assign(np.array(q, np.float32))
    state.joint_qd.assign(np.array(qd, np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, state, j


def test_d6_coordinates_are_newtons(test, device):
    """A body on a D6 joint of three angular axes, a ball joint for libuipc (in Newton's (X, Z, Y) order of the
    nv_humanoid hips, a left-handed triple, and in (X, Y, Z)), and one of three linear and three angular axes, no joint
    for libuipc, swings and tumbles under gravity from a turned pose: after every step joint_q and joint_qd hold the
    coordinates newton.eval_ik computes from the bodies' state."""
    for lin, axes in ((False, "XZY"), (False, "XYZ"), (True, "XZY")):
        with test.subTest(lin=lin, axes=axes):
            q = ([0.1, -0.2, 0.05] if lin else []) + [0.5, -0.4, 0.7]
            qd = ([0.3, 0.0, -0.1] if lin else []) + [0.9, -0.6, 0.3]
            model, state, _ = _d6_body(device, lin, axes, q, qd, gravity=10.0)
            solver = newton.solvers.SolverIPC(model, enable_d6_joints=True)
            s0, s1 = state, model.state()
            moved = 0.0
            for _ in range(40):
                solver.step(s0, s1, None, None, 0.01)
                s0, s1 = s1, s0
                written_q, written_qd = s0.joint_q.numpy().copy(), s0.joint_qd.numpy().copy()
                newton.eval_ik(model, s0, s0.joint_q, s0.joint_qd)
                np.testing.assert_allclose(written_q, s0.joint_q.numpy(), rtol=0.0, atol=1e-5)
                np.testing.assert_allclose(written_qd, s0.joint_qd.numpy(), rtol=0.0, atol=1e-4)
                moved = max(moved, float(np.abs(written_q - q).max()))
            test.assertGreater(moved, 0.1)


def test_d6_joint_f_is_featherstones(test, device):
    """joint_f on a D6 joint is Featherstone's generalized force: on the third coordinate of three angular axes in the
    (X, Z, Y) order, from the pose (0.5, -0.4, 0.7), the joint rates after ten steps of 1 ms are SolverFeatherstone's
    to 2e-3 (Newton's own test of this), and so are those of three linear and three angular axes pushed along one
    linear and turned about one angular coordinate, whose joint frame is 0.2 m off the centre of mass; with the joint
    frame at the centre of mass, from the zero pose, the body moves along and turns about the axes of the forces as
    implicit Euler integrates F = m a and tau = I alpha, and the other coordinates stay at rest."""
    cases = (
        (False, "XZY", [0.5, -0.4, 0.7], [0.0, 0.0, 1.0]),
        (True, "XYZ", [0.1, 0.0, -0.1, 0.3, -0.2, 0.4], [0.0, 2.0, 0.0, 0.0, 0.0, 0.05]),
    )
    for lin, axes, q0, f in cases:
        with test.subTest(lin=lin):
            rates = []
            for make in (
                lambda m: newton.solvers.SolverFeatherstone(m, angular_damping=0.0),
                lambda m: newton.solvers.SolverIPC(m, enable_d6_joints=True),
            ):
                model, state, _ = _d6_body(device, lin, axes, q0, [0.0] * len(q0))
                control = model.control()
                control.joint_f.assign(np.array(f, np.float32))
                solver, s1 = make(model), model.state()
                for _ in range(10):
                    solver.step(state, s1, control, None, 1.0e-3)
                    state, s1 = s1, state
                newton.eval_ik(model, state, state.joint_q, state.joint_qd)
                rates.append(state.joint_qd.numpy().copy())
            test.assertGreater(float(np.abs(rates[0]).max()), 0.02)
            np.testing.assert_allclose(rates[1], rates[0], rtol=0.0, atol=2.0e-3)

    h, steps = 0.01, 10
    model, state, _ = _d6_body(device, True, "XYZ", [0.0] * 6, [0.0] * 6, offset=0.0)
    control = model.control()
    control.joint_f.assign(np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.3], np.float32))
    solver = newton.solvers.SolverIPC(model, enable_d6_joints=True, **_CONVERGED)
    q, _ = _steps(model, solver, state, steps, h, control)
    n = np.arange(1, steps + 1)
    # implicit Euler from rest under a constant acceleration a: q_n = a h^2 n (n + 1) / 2; the angle to the float32
    # rounding of the quaternion it is decomposed from
    np.testing.assert_allclose(q[:, 1], 2.0 * h * h * n * (n + 1) / 2.0, rtol=1e-3, atol=0.0)
    np.testing.assert_allclose(q[:, 5], 0.3 / 0.15 * h * h * n * (n + 1) / 2.0, rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(q[:, [0, 2, 3, 4]], 0.0, rtol=0.0, atol=1e-6)


def _distance_pendulum(device, length, theta, lower, upper, gravity=10.0):
    """A sphere of radius 0.05 hanging from the world on a distance joint at its centre, ``length`` below the
    joint's origin turned by ``theta`` (y up), the joint's bounds ``lower`` and ``upper``."""
    builder = newton.ModelBuilder(gravity=(0.0, -gravity, 0.0), up_axis=newton.Axis.Y)
    x = (length * np.sin(theta), 1.0 - length * np.cos(theta), 0.0)
    bob = builder.add_link(xform=wp.transform(x, wp.quat_identity()))
    builder.add_shape_sphere(bob, radius=0.05)
    j = builder.add_joint_distance(
        -1, bob, parent_xform=wp.transform((0.0, 1.0, 0.0), wp.quat_identity()), min_distance=lower,
        max_distance=upper,
    )  # fmt: skip
    builder.add_articulation([j])
    return builder.finalize(device=device)


def test_distance_joint_swings_as_a_pendulum(test, device):
    """With enable_distance_joints=True a distance joint of one length L is libuipc's soft vertex stitch of rest length
    L between its joint frames' origins: a sphere hanging 1 m from the world on it at its centre swings 0.02 rad as
    implicit Euler integrates the linearized point-mass pendulum, step by step to 1e-3 of its amplitude, and stays at
    L within the ball joint's penalty. Without the option the joint is refused, and a distance joint with a range or a
    single bound is refused with it."""
    L, theta0, g, h = 1.0, 0.02, 10.0, 1.0e-3
    model = _distance_pendulum(device, L, theta0, L, L, gravity=g)
    with test.assertRaisesRegex(NotImplementedError, "DISTANCE joints \\(joint 0\\).*enable_distance_joints=True"):
        newton.solvers.SolverIPC(model)
    solver = newton.solvers.SolverIPC(model, enable_distance_joints=True, **_CONVERGED)
    test.assertEqual(len(solver._distance_rows), 1)
    m = float(model.body_mass.numpy()[0])
    steps = 700  # beyond half a period, 2 pi sqrt(L / g) = 1.99 s
    frames, _ = _steps(model, solver, model.state(), steps, h, None, lambda s: s.body_q.numpy()[0, :3].copy())
    offset = frames - np.array([0.0, 1.0, 0.0])
    angle = np.arctan2(offset[:, 0], -offset[:, 1])
    expected = _drive_recurrence(m * L * L, h, steps, q0=theta0, ke=m * g * L)
    test.assertLess(expected.min(), -0.5 * theta0)
    np.testing.assert_allclose(angle, expected, rtol=0.0, atol=1e-3 * theta0)
    # the penalty K = joint_strength_ratio m holds the weight m g at a stretch of g h^2 / ratio, 1e-7 m here
    np.testing.assert_allclose(np.linalg.norm(offset, axis=1), L, rtol=0.0, atol=1e-6)
    for lower, upper in ((0.5, 1.0), (-1.0, 1.0), (0.5, -1.0)):
        with test.subTest(bounds=(lower, upper)):
            with test.assertRaisesRegex(NotImplementedError, "a range or a single bound is an inequality"):
                newton.solvers.SolverIPC(_distance_pendulum(device, L, 0.0, lower, upper), enable_distance_joints=True)


def test_distance_joint_between_bodies(test, device):
    """A distance joint of 0.5 m between two free spheres thrown apart sideways at 1 m/s, without gravity: they turn
    about each other at 0.5 m, within the penalty, the pair keeps its momentum to 1e-6 and its joint coordinates are a
    free joint's (eval_ik's). A distance joint with both bounds negative holds nothing: the spheres fly apart as two
    free bodies, and a disabled one too."""

    def pair(lower, upper, enabled=True):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        a = builder.add_link(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_sphere(a, radius=0.05)
        b = builder.add_link(xform=wp.transform((0.5, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_sphere(b, radius=0.05)
        joints = [
            builder.add_joint_free(a),
            builder.add_joint_distance(a, b, min_distance=lower, max_distance=upper, enabled=enabled),
        ]
        builder.add_articulation(joints)
        model = builder.finalize(device=device)
        state = model.state()
        qd = state.body_qd.numpy()
        qd[0, 1], qd[1, 1] = -1.0, 1.0
        state.body_qd.assign(qd)
        return model, state

    h, steps = 0.005, 200
    model, state = pair(0.5, 0.5)
    solver = newton.solvers.SolverIPC(model, enable_distance_joints=True, **_CONVERGED)
    s0, s1 = state, model.state()
    m = model.body_mass.numpy().astype(np.float64)
    for _ in range(steps):
        solver.step(s0, s1, None, None, h)
        s0, s1 = s1, s0
        x = s0.body_q.numpy()[:, :3].astype(np.float64)
        test.assertAlmostEqual(float(np.linalg.norm(x[1] - x[0])), 0.5, delta=1e-4)
        momentum = (m[:, None] * s0.body_qd.numpy()[:, :3]).sum(axis=0)
        np.testing.assert_allclose(momentum, 0.0, rtol=0.0, atol=1e-6 * m.sum())
    written_q, written_qd = s0.joint_q.numpy().copy(), s0.joint_qd.numpy().copy()
    newton.eval_ik(model, s0, s0.joint_q, s0.joint_qd)
    np.testing.assert_allclose(written_q, s0.joint_q.numpy(), rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(written_qd, s0.joint_qd.numpy(), rtol=0.0, atol=1e-4)
    turned = np.arctan2(x[1, 1] - x[0, 1], x[1, 0] - x[0, 0])
    test.assertGreater(abs(turned), 1.0)  # 4 rad/s for a second
    for lower, upper, enabled in ((-1.0, -1.0, True), (0.5, 0.5, False)):
        with test.subTest(bounds=(lower, upper), enabled=enabled):
            model, state = pair(lower, upper, enabled)
            solver = newton.solvers.SolverIPC(model, enable_distance_joints=True)
            _, s0 = _steps(model, solver, state, 20, h)
            x = s0.body_q.numpy()[:, :3]
            np.testing.assert_allclose(x[:, 1], [-20 * h, 20 * h], rtol=0.0, atol=1e-5)


def _limit_recurrence(m, h, steps, g, lower, limit_ke, damping=0.0, at_start=True):
    """q_1 .. q_steps of implicit Euler for a slider of mass m falling from rest at 0 under gravity g onto its lower
    limit, held by libuipc's cubic limit as SolverIPC sets it (the force limit_ke (lower - q)^2 / delta_r below the
    limit, delta_r = 1e-3) and damped by ``damping`` (q' - q) / h in the steps that start below the limit
    (``at_start``) or end below it: one scalar Newton solve per step."""
    q, v, out = 0.0, 0.0, []
    for _ in range(steps):
        x = q + h * v
        for _ in range(100):
            below = x < lower
            active = (q < lower) if at_start else below
            d = damping if active else 0.0
            f_lim = limit_ke * (lower - x) ** 2 / 1.0e-3 if below else 0.0
            df_lim = -2.0 * limit_ke * (lower - x) / 1.0e-3 if below else 0.0
            r = m * (x - q - h * v) / (h * h) + m * g - f_lim + d * (x - q) / h
            step = r / (m / (h * h) - df_lim + d / h)
            x -= step
            if abs(step) < 1e-15:
                break
        q, v = x, (x - q) / h
        out.append(q)
    return np.array(out)


def test_limit_damping_is_implicit_euler(test, device):
    """With enable_joint_limit_damping=True a slider that falls under gravity onto its lower limit takes the damping
    -joint_limit_kd qd of Featherstone and SemiImplicit in the steps that start past the limit, at the step's end
    velocity: it follows the implicit-Euler recurrence of the cubic limit with that damping step by step to 1e-4 m,
    and stays at the limit where the undamped slider (the option off, kd ignored as before) bounces 5 cm off it,
    which follows the recurrence without damping. A recurrence that switches the damping on in the steps that end
    past the limit is farther off."""
    g, lower, limit_ke, kd, h, steps = 10.0, -0.1, 1.0e3, 200.0, 2.0e-3, 300
    runs = {}
    for option in (False, True):
        model, state, _ = _hinged(
            device, "prismatic", gravity=g, axis=newton.Axis.Y, limit_lower=lower, limit_upper=1.0,
            limit_ke=limit_ke, limit_kd=kd,
        )  # fmt: skip
        solver = newton.solvers.SolverIPC(model, enable_joint_limit_damping=option, **_CONVERGED)
        runs[option] = _steps(model, solver, state, steps, h)[0][:, 0]
    m = float(model.body_mass.numpy()[0])
    damped = _limit_recurrence(m, h, steps, g, lower, limit_ke, kd)
    undamped = _limit_recurrence(m, h, steps, g, lower, limit_ke)
    np.testing.assert_allclose(runs[True], damped, rtol=0.0, atol=1e-4)
    np.testing.assert_allclose(runs[False], undamped, rtol=0.0, atol=1e-4)
    hit = int(np.argmax(damped < lower))
    test.assertGreater(runs[False][hit:].max() - lower, 0.02)  # the undamped slider bounces
    test.assertLess(runs[True][hit:].max() - lower, 0.001)  # the damped one does not
    at_end = _limit_recurrence(m, h, steps, g, lower, limit_ke, kd, at_start=False)
    test.assertGreater(np.abs(at_end - damped).max(), 5e-4)


def test_joint_options_change_nothing_unused(test, device):
    """The enable_* options leave a model that does not use them as it is: a revolute and a prismatic joint driven
    within their limits, with joint_limit_kd that never acts, step bit for bit as without the options."""
    runs = []
    for option in (False, True):
        steps = []
        for kind, length in (("revolute", 0.1), ("prismatic", 1.0)):
            model, state, _ = _hinged(
                device, kind, axis=newton.Axis.X, radius=0.5, target_ke=40.0, target_kd=2.0, stiffness=10.0,
                damping=1.0, target_q=0.5 * length, limit_lower=-length, limit_upper=length, limit_ke=1.0e4,
                limit_kd=10.0,
            )  # fmt: skip
            solver = newton.solvers.SolverIPC(model, enable_joint_limit_damping=option)
            test.assertEqual(solver._joint_options.drive_terms, option)
            steps.append(_steps(model, solver, state, 50, 0.01, None, lambda s: s.body_q.numpy().copy())[0])
        runs.append(steps)
    for before, after in zip(*runs, strict=True):
        test.assertGreater(np.abs(before[-1] - before[0]).max(), 1e-3)
        np.testing.assert_array_equal(after, before)


def test_rejects_d6_joints_libuipc_lacks(test, device):
    """With enable_d6_joints=True the D6 joints that lock what no libuipc joint locks are refused, naming what they
    lock: two angular axes, three linear axes, one linear and one angular; and three angular axes that are not
    orthonormal, and drives and limits on the D6 joints that become a ball joint or no joint."""
    D = newton.ModelBuilder.JointDofConfig

    def model(linear=(), angular=(), **dof):
        builder = newton.ModelBuilder()
        link = builder.add_link(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
        lin = [D(axis=a, **dof) for a in linear]
        ang = [D(axis=a, **dof) for a in angular]
        builder.add_articulation([builder.add_joint_d6(-1, link, linear_axes=lin, angular_axes=ang)])
        return builder.finalize(device=device)

    skew = (1.0, 1.0, 0.0)
    for name, m, pattern in (
        ("universal", model(angular="XY"), "0 linear and 2 angular axes lock 3 translations and 1 rotations"),
        ("translations", model(linear="XYZ"), "3 linear and 0 angular axes lock 0 translations and 3 rotations"),
        ("cylindrical", model(linear="X", angular="X"), "1 linear and 1 angular axes lock 2 translations"),
        ("skew axes", model(angular=("X", skew, "Z")), "its angular axes are not orthogonal"),
        ("ball drive", model(angular="XYZ", target_ke=10.0), "D6 joint 0 has the target mode POSITION"),
        ("free limit", model(linear="XYZ", angular="XYZ", limit_lower=-0.1), "limits revolute and prismatic"),
    ):
        with test.subTest(name):
            with test.assertRaisesRegex(NotImplementedError, pattern):
                newton.solvers.SolverIPC(m, enable_d6_joints=True)


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

    def test_imports_only_public_newton_names(self):
        """SolverIPC's modules import Newton through its public modules only: no import of theirs resolves under
        newton._src outside SolverIPC's own package."""
        module = importlib.import_module(newton.solvers.SolverIPC.__module__)
        package = module.__package__
        private = []
        for path in sorted(pathlib.Path(module.__file__).parent.glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [importlib.util.resolve_name("." * node.level + (node.module or ""), package)]
                else:
                    continue
                for name in names:
                    own = name == package or name.startswith(package + ".")
                    if (name == "newton._src" or name.startswith("newton._src.")) and not own:
                        private.append(f"{path.name}:{node.lineno} {name}")
        self.assertEqual(private, [])

    def test_rest_shape_copies_match_newtons_builders(self):
        """SolverIPC's copies of Newton's rest-shape helpers, which have no public name, reproduce bit for bit the
        tri_poses, tri_areas and edge_rest_angle that ModelBuilder stores, computed from particle_q as the solver
        computes them to tell a cloth built at rest: random sheets from add_cloth_grid, from add_cloth_mesh (bent)
        and from add_triangles with add_edges (random particles, the hinge rows in both orders)."""
        from newton._src.solvers.ipc.solver_ipc import _dihedral_angles, _triangle_rest_geometry  # noqa: PLC0415

        rng = np.random.default_rng(3)

        def rotation():
            axis = rng.normal(size=3)
            return wp.quat_from_axis_angle(wp.vec3(*(axis / np.linalg.norm(axis))), float(rng.uniform(0.0, 6.0)))

        bent = 0.0
        for trial in range(12):
            builder = newton.ModelBuilder()
            kind = ("grid", "mesh", "triangles")[trial % 3]
            if kind == "grid":
                builder.add_cloth_grid(
                    pos=wp.vec3(*rng.uniform(-1.0, 1.0, 3)),
                    rot=rotation(),
                    vel=wp.vec3(0.0, 0.0, 0.0),
                    dim_x=int(rng.integers(2, 8)),
                    dim_y=int(rng.integers(2, 8)),
                    cell_x=float(rng.uniform(0.005, 0.2)),
                    cell_y=float(rng.uniform(0.005, 0.2)),
                    mass=0.1,
                )
            elif kind == "mesh":
                nx, ny = (int(k) for k in rng.integers(3, 7, size=2))
                u, v = np.meshgrid(np.linspace(0.0, 0.4, nx), np.linspace(0.0, 0.3, ny), indexing="ij")
                vertices = np.stack([u, v, 0.05 * rng.normal(size=u.shape)], axis=-1).reshape(-1, 3)
                corners = [(ny * i + j, ny * (i + 1) + j) for i in range(nx - 1) for j in range(ny - 1)]
                builder.add_cloth_mesh(
                    pos=wp.vec3(*rng.uniform(-1.0, 1.0, 3)),
                    rot=rotation(),
                    scale=float(rng.uniform(0.5, 2.0)),
                    vel=wp.vec3(0.0, 0.0, 0.0),
                    vertices=[wp.vec3(*p) for p in vertices],
                    indices=[k for a, b in corners for k in (a, b, b + 1, a, b + 1, a + 1)],
                    density=0.1,
                )
            else:
                # a fan of triangles about particle 0, at float32 positions, its hinges given in both orders
                n = int(rng.integers(4, 9))
                x = rng.uniform(-1.0, 1.0, (n + 2, 3)).astype(np.float32).astype(np.float64)
                builder.add_particles(x.tolist(), [(0.0, 0.0, 0.0)] * (n + 2), [0.1] * (n + 2))
                tris = np.array([(0, k, k + 1) for k in range(1, n + 1)])
                builder.add_triangles(tris[:, 0], tris[:, 1], tris[:, 2])
                rows = np.array([(k - 1, k + 1, 0, k) if k % 2 else (k + 1, k - 1, k, 0) for k in range(2, n + 1)])
                builder.add_edges(rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3])
            model = builder.finalize(device="cpu")
            x = model.particle_q.numpy().astype(np.float64)
            t = model.tri_indices.numpy().reshape(-1, 3)
            edges, area = _triangle_rest_geometry(x[t[:, 1]] - x[t[:, 0]], x[t[:, 2]] - x[t[:, 0]])
            np.testing.assert_array_equal(np.linalg.inv(edges).astype(np.float32), model.tri_poses.numpy(), kind)
            np.testing.assert_array_equal(area.astype(np.float32), model.tri_areas.numpy(), kind)
            rows = model.edge_indices.numpy().reshape(-1, 4)
            hinge = (rows[:, 0] >= 0) & (rows[:, 1] >= 0)
            e = rows[hinge]
            angles = _dihedral_angles(x[e[:, 0]], x[e[:, 1]], x[e[:, 2]], x[e[:, 3]], x[e[:, 2]], x[e[:, 3]])
            np.testing.assert_array_equal(angles.astype(np.float32), model.edge_rest_angle.numpy()[hinge], kind)
            if kind != "grid":
                bent = max(bent, float(np.abs(angles).max()))
        self.assertGreater(bent, 1.0)  # folded hinges too, not only flat ones

    def test_import_names_missing_calls(self):
        """Refuse a libuipc-warp that lacks a call the solver makes, naming it."""
        import uipc_warp.arrays  # noqa: PLC0415

        imported, contact_model = newton.solvers.SolverIPC._uipc_warp, uipc_warp.arrays.ArrayScene.contact_model
        try:
            newton.solvers.SolverIPC._uipc_warp = None
            del uipc_warp.arrays.ArrayScene.contact_model
            with self.assertRaisesRegex(ImportError, r"lacks ArrayScene\.contact_model"):
                newton.solvers.SolverIPC._import_uipc_warp()
        finally:
            uipc_warp.arrays.ArrayScene.contact_model = contact_model
            newton.solvers.SolverIPC._uipc_warp = imported

    def test_contact_elements_match_shape_contact_pairs(self):
        """On random models (collision groups -2 to 3, worlds -1 and 0, collide flags, friction coefficients, filter
        pairs, shapes on one body), the shape pairs that SolverIPC's contact elements enable on different bodies are
        exactly Model.shape_contact_pairs, with the mean of the two coefficients; particles meet the shapes with
        COLLIDE_PARTICLES."""
        from newton._src.solvers.ipc.solver_ipc import _colliding_shapes, _contact_elements  # noqa: PLC0415

        rng = np.random.default_rng(7)
        for trial in range(40):
            builder = newton.ModelBuilder()

            def shape(body, builder=builder):
                cfg = newton.ModelBuilder.ShapeConfig(
                    collision_group=int(rng.choice([-2, -1, 0, 1, 2, 3])),
                    has_shape_collision=bool(rng.random() < 0.8),
                    has_particle_collision=bool(rng.random() < 0.8),
                    mu=float(rng.choice([0.2, 0.5])),
                )
                xform = wp.transform(wp.vec3(*rng.uniform(-1.0, 1.0, 3)), wp.quat_identity())
                return builder.add_shape_box(body, xform=xform, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)

            for local in (False, True):
                if local:
                    builder.begin_world()
                for _ in range(int(rng.integers(0, 3))):
                    shape(-1)
                for _ in range(int(rng.integers(1, 4))):
                    body = builder.add_body(xform=wp.transform(wp.vec3(*rng.uniform(-1.0, 1.0, 3)), wp.quat_identity()))
                    for _ in range(int(rng.integers(1, 3))):
                        shape(body)
                if local:
                    builder.end_world()
            for _ in range(int(rng.integers(0, 4))):
                a, b = rng.choice(builder.shape_count, 2, replace=False)
                builder.add_shape_collision_filter_pair(int(a), int(b))
            if trial % 2:
                builder.add_particle(wp.vec3(0.0, 0.0, 3.0), wp.vec3(0.0, 0.0, 0.0), 1.0)
            model = builder.finalize(device="cpu")
            elements = _contact_elements(model)
            body = model.shape_body.numpy()
            mu = model.shape_material_mu.numpy().astype(np.float64)
            shapes = np.flatnonzero(_colliding_shapes(model))
            enabled = set()
            for i in shapes:
                for j in shapes[shapes > i]:
                    e, f = elements.shape[i], elements.shape[j]
                    if body[i] != body[j] and elements.enabled[e, f]:
                        enabled.add((int(i), int(j)))
                        self.assertAlmostEqual(elements.mu[e, f], 0.5 * (mu[i] + mu[j]), places=7)
            listed = model.shape_contact_pairs
            listed = set() if listed is None else {tuple(sorted(map(int, p))) for p in listed.numpy()}
            self.assertEqual(enabled, listed, f"trial {trial}")
            if model.particle_count:
                flags = model.shape_flags.numpy()
                p = elements.particle[0]
                for i in shapes:
                    self.assertEqual(
                        bool(elements.enabled[p, elements.shape[i]]),
                        bool(flags[i] & int(newton.ShapeFlags.COLLIDE_PARTICLES)),
                    )

    def test_contact_elements_with_kinematic_aims(self):
        """With kinematic_motion="aim" (``aims``), on random models of static shapes, dynamic and kinematic bodies: the
        pairs the contact elements enable on different bodies are Model.shape_contact_pairs less the pairs of two
        shapes that are each static or kinematic, and without the option the elements of before."""
        from newton._src.solvers.ipc.solver_ipc import _colliding_shapes, _contact_elements  # noqa: PLC0415

        rng = np.random.default_rng(11)
        for trial in range(30):
            builder = newton.ModelBuilder()
            for _ in range(int(rng.integers(0, 3))):
                xform = wp.transform(wp.vec3(*rng.uniform(-1.0, 1.0, 3)), wp.quat_identity())
                builder.add_shape_box(-1, xform=xform, hx=0.1, hy=0.1, hz=0.1)
            for _ in range(int(rng.integers(1, 5))):
                kinematic = bool(rng.random() < 0.5)
                xform = wp.transform(wp.vec3(*rng.uniform(-1.0, 1.0, 3)), wp.quat_identity())
                body = builder.add_body(xform=xform, is_kinematic=kinematic)
                cfg = newton.ModelBuilder.ShapeConfig(collision_group=int(rng.choice([-1, 1, 2])))
                builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
            model = builder.finalize(device="cpu")
            elements = _contact_elements(model, aims=True)
            body = model.shape_body.numpy()
            immovable = (body < 0) | np.isin(
                body, np.flatnonzero(model.body_flags.numpy() & int(newton.BodyFlags.KINEMATIC))
            )
            shapes = np.flatnonzero(_colliding_shapes(model))
            enabled = set()
            for i in shapes:
                for j in shapes[shapes > i]:
                    if body[i] != body[j] and elements.enabled[elements.shape[i], elements.shape[j]]:
                        enabled.add((int(i), int(j)))
            listed = model.shape_contact_pairs
            listed = set() if listed is None else {tuple(sorted(map(int, p))) for p in listed.numpy()}
            self.assertEqual(
                enabled, {p for p in listed if not (immovable[p[0]] and immovable[p[1]])}, f"trial {trial}"
            )

    def test_contact_elements_of_coupled_worlds(self):
        """On random models of global shapes and two to four local worlds, some replicated with their filter pairs and
        some drawn on their own, the shape pairs that the contact elements enable on different bodies, in the same
        world or with a global shape (the subscenes), are exactly Model.shape_contact_pairs; replicated worlds add no
        contact element."""
        from newton._src.solvers.ipc.solver_ipc import _colliding_shapes, _contact_elements, _Worlds  # noqa: PLC0415

        rng = np.random.default_rng(11)

        def content(builder):
            """Bodies of random shapes and filter pairs among them."""
            first = builder.shape_count
            for _ in range(int(rng.integers(1, 4))):
                body = builder.add_body(xform=wp.transform(wp.vec3(*rng.uniform(-1.0, 1.0, 3)), wp.quat_identity()))
                for _ in range(int(rng.integers(1, 3))):
                    cfg = newton.ModelBuilder.ShapeConfig(
                        collision_group=int(rng.choice([-2, -1, 0, 1, 2])),
                        has_shape_collision=bool(rng.random() < 0.85),
                        mu=float(rng.choice([0.2, 0.5])),
                    )
                    xform = wp.transform(wp.vec3(*rng.uniform(-0.2, 0.2, 3)), wp.quat_identity())
                    builder.add_shape_box(body, xform=xform, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
            for _ in range(int(rng.integers(0, 3))):
                a, b = rng.choice(np.arange(first, builder.shape_count), 2)
                if a != b:
                    builder.add_shape_collision_filter_pair(int(a), int(b))

        for trial in range(30):
            builder = newton.ModelBuilder()
            for _ in range(int(rng.integers(1, 3))):
                cfg = newton.ModelBuilder.ShapeConfig(collision_group=int(rng.choice([-1, 1, 2])))
                g = builder.add_shape_box(
                    -1,
                    xform=wp.transform(wp.vec3(*rng.uniform(-1, 1, 3)), wp.quat_identity()),
                    hx=0.3,
                    hy=0.3,
                    hz=0.05,
                    cfg=cfg,
                )
            template = newton.ModelBuilder()
            content(template)
            copies = int(rng.integers(2, 4))
            builder.replicate(template, copies)
            for _ in range(int(rng.integers(0, 2))):
                builder.begin_world()
                content(builder)
                builder.end_world()
            if trial % 3 == 0:  # the last shape of every copy filtered against a global one
                shape_world = np.asarray(builder.shape_world)
                for k in range(copies):
                    builder.add_shape_collision_filter_pair(g, int(np.flatnonzero(shape_world == k)[-1]))
            model = builder.finalize(device="cpu")
            worlds = _Worlds.of(model)
            self.assertIsNotNone(worlds)
            elements = _contact_elements(model, worlds)
            body, world = model.shape_body.numpy(), model.shape_world.numpy()
            shapes = np.flatnonzero(_colliding_shapes(model))
            enabled = set()
            for i in shapes:
                for j in shapes[shapes > i]:
                    e, f = elements.shape[i], elements.shape[j]
                    together = world[i] == world[j] or world[i] == -1 or world[j] == -1
                    if body[i] != body[j] and together and elements.enabled[e, f]:
                        enabled.add((int(i), int(j)))
            listed = model.shape_contact_pairs
            listed = set() if listed is None else {tuple(sorted(map(int, p))) for p in listed.numpy()}
            self.assertEqual(enabled, listed, f"trial {trial}")
            replicated = world[shapes] < copies  # the replicated worlds come first
            local = world[shapes] >= 0
            used = np.unique(elements.shape[shapes[local & replicated]])
            first = np.unique(elements.shape[shapes[world[shapes] == 0]])
            self.assertEqual(used.tolist(), first.tolist(), f"trial {trial}: the copies share the elements of world 0")


@requires("uipc_warp")
class TestSolverIPCContact(unittest.TestCase):
    """Which pairs collide, with which friction and at which distance."""


@requires("uipc_warp")
class TestSolverIPCStep(unittest.TestCase):
    """How a step ends and what carries from one step to the next: status, overflow, state, reset, model changes."""


@requires("uipc_warp")
class TestSolverIPCWorlds(unittest.TestCase):
    """Several worlds with coupled_worlds: kept apart in contact, solved together."""


@requires("uipc_warp")
class TestSolverIPCJoints(unittest.TestCase):
    """Revolute, prismatic, fixed and ball joints: how they hold, forces, coordinates, drives and limits, and where links
    built without a pose start."""


@requires("uipc_warp")
class TestSolverIPCOptions(unittest.TestCase):
    """libuipc's options: BDF2, AL-IPC contact and the MAS preconditioner."""


@requires("uipc_warp")
class TestSolverIPCAttachments(unittest.TestCase):
    """Particle-body attachments and springs of rest length 0 as libuipc's soft vertex stitch."""


@requires("uipc_warp")
class TestSolverIPCJointOptions(unittest.TestCase):
    """Joints on request (enable_d6_joints and the other enable_* options): Newton's joints that map onto libuipc's
    own constitutions beyond the revolute, prismatic, fixed, ball and free joints."""


@requires("uipc_warp")
class TestSolverIPCKinematicAims(unittest.TestCase):
    """kinematic_motion="aim": kinematic bodies moved inside the step toward their aims by libuipc's soft transform
    constraint, swept by the CCD."""


@requires("uipc_warp")
class TestSolverIPCContactOutput(unittest.TestCase):
    """CONTACT_F: the solver's own contact terms written into the step's Contacts, one row per term with a force, and
    their forces at the end of the step."""


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


for name, func in (
    ("test_rejects_springs_without_a_libuipc_form", test_rejects_springs_without_a_libuipc_form),
    ("test_rejects_attachment_damping", test_rejects_attachment_damping),
    ("test_rejects_mimic_constraints", test_rejects_mimic_constraints),
    ("test_rejects_equality_constraints", test_rejects_equality_constraints),
    ("test_rejects_tendons", test_rejects_tendons),
    ("test_rejects_muscles", test_rejects_muscles),
    ("test_rejects_activations", test_rejects_activations),
    ("test_rejects_unsupported_joints", test_rejects_unsupported_joints),
    ("test_rejects_gradients", test_rejects_gradients),
    ("test_rejects_worlds", test_rejects_worlds),
    ("test_rejects_surface_velocity", test_rejects_surface_velocity),
    ("test_rejects_intersecting_start", test_rejects_intersecting_start),
    ("test_rejects_start_in_contact", test_rejects_start_in_contact),
    ("test_external_forces_match_closed_form", test_external_forces_match_closed_form),
    ("test_free_joint_force_pushes_the_parent", test_free_joint_force_pushes_the_parent),
):
    add_function_test(TestSolverIPCInputs, name, func, devices=devices)
for name, func in (
    ("test_friction_of_each_pair", test_friction_of_each_pair),
    ("test_ramp_sticks_and_slips", test_ramp_sticks_and_slips),
    ("test_excluded_pairs_never_collide", test_excluded_pairs_never_collide),
    ("test_particle_only_shapes", test_particle_only_shapes),
    ("test_inactive_particles_do_not_collide", test_inactive_particles_do_not_collide),
    ("test_rest_heights_follow_radius_and_margin", test_rest_heights_follow_radius_and_margin),
    ("test_springs_oscillate_as_implicit_euler", test_springs_oscillate_as_implicit_euler),
    ("test_heightfield_triangles_and_rest", test_heightfield_triangles_and_rest),
    ("test_kappa_is_taken_as_given", test_kappa_is_taken_as_given),
    ("test_adaptive_kappa_is_libuipcs", test_adaptive_kappa_is_libuipcs),
    ("test_adaptive_kappa_follows_dt_in_captured_steps", test_adaptive_kappa_follows_dt_in_captured_steps),
    ("test_rest_shapes_from_tri_poses_and_edge_rest_angle", test_rest_shapes_from_tri_poses_and_edge_rest_angle),
    ("test_cloth_at_rest_starts_without_bending_force", test_cloth_at_rest_starts_without_bending_force),
    (
        "test_cloth_at_rest_takes_its_rest_triangles_from_particle_q",
        test_cloth_at_rest_takes_its_rest_triangles_from_particle_q,
    ),
    (
        "test_tet_lame_parameters_reach_the_engine_as_libuipc_stores_them",
        test_tet_lame_parameters_reach_the_engine_as_libuipc_stores_them,
    ),
):
    add_function_test(TestSolverIPCContact, name, func, devices=devices)
for name, func in (
    ("test_contact_at_start_fails", test_contact_at_start_fails),
    ("test_check_status", test_check_status),
    ("test_default_d_hat_follows_small_features", test_default_d_hat_follows_small_features),
    ("test_overflow_grows", test_overflow_grows),
    ("test_teleport_equals_fresh_solver", test_teleport_equals_fresh_solver),
    ("test_reset_inside_captured_graph", test_reset_inside_captured_graph),
    ("test_reset_checks_world_mask_as_newton_does", test_reset_checks_world_mask_as_newton_does),
    ("test_notify_updates_gravity_and_inertia_in_place", test_notify_updates_gravity_and_inertia_in_place),
    ("test_gravity_per_world", test_gravity_per_world),
    ("test_graph_outdated_after_long_solves", test_graph_outdated_after_long_solves),
    ("test_graph_outdated_takes_the_ladder_guesses", test_graph_outdated_takes_the_ladder_guesses),
    ("test_default_control_is_built_once", test_default_control_is_built_once),
):
    add_function_test(TestSolverIPCStep, name, func, devices=devices)
for name, func in (
    ("test_coupled_worlds_follow_the_single_world", test_coupled_worlds_follow_the_single_world),
    ("test_coupled_worlds_are_built_as_one_world", test_coupled_worlds_are_built_as_one_world),
    ("test_coupled_worlds_share_filter_classes", test_coupled_worlds_share_filter_classes),
    ("test_coupled_worlds_status_in_every_world", test_coupled_worlds_status_in_every_world),
    ("test_coupled_worlds_fall_with_their_gravity", test_coupled_worlds_fall_with_their_gravity),
    ("test_coupled_worlds_beyond_the_old_element_limit", test_coupled_worlds_beyond_the_old_element_limit),
    ("test_start_check_of_many_worlds", test_start_check_of_many_worlds),
    ("test_near_parts_forms_no_pair_of_two_worlds", test_near_parts_forms_no_pair_of_two_worlds),
    ("test_coupled_worlds_with_attachments", test_coupled_worlds_with_attachments),
    ("test_coupled_worlds_with_kinematic_aims", test_coupled_worlds_with_kinematic_aims),
    ("test_reset_of_one_world_keeps_the_others", test_reset_of_one_world_keeps_the_others),
    ("test_reset_of_one_world_keeps_the_other_worlds_aims", test_reset_of_one_world_keeps_the_other_worlds_aims),
):
    add_function_test(TestSolverIPCWorlds, name, func, devices=devices)

for name, func in (
    ("test_pendulum_is_implicit_euler", test_pendulum_is_implicit_euler),
    ("test_joint_forces_match_closed_form", test_joint_forces_match_closed_form),
    ("test_fixed_and_ball_joints_hold", test_fixed_and_ball_joints_hold),
    ("test_joint_penalty_follows_the_masses", test_joint_penalty_follows_the_masses),
    ("test_joint_coordinates_are_written", test_joint_coordinates_are_written),
    ("test_grown_lists_keep_joint_angles", test_grown_lists_keep_joint_angles),
    ("test_drives_are_implicit_euler", test_drives_are_implicit_euler),
    ("test_joint_limits_hold", test_joint_limits_hold),
    ("test_drive_targets_change_inside_a_captured_graph", test_drive_targets_change_inside_a_captured_graph),
    ("test_unplaced_links_start_at_forward_kinematics", test_unplaced_links_start_at_forward_kinematics),
):
    add_function_test(TestSolverIPCJoints, name, func, devices=devices)
for name, func in (
    ("test_bdf2_falls_from_an_implicit_euler_step", test_bdf2_falls_from_an_implicit_euler_step),
    (
        "test_bdf2_turns_a_body_by_the_impulse_of_its_first_step",
        test_bdf2_turns_a_body_by_the_impulse_of_its_first_step,
    ),
    ("test_bdf2_is_second_order_on_a_spring", test_bdf2_is_second_order_on_a_spring),
    ("test_bdf2_history_starts_anew", test_bdf2_history_starts_anew),
    ("test_bdf2_drives", test_bdf2_drives),
    ("test_al_ipc_rests_at_d_hat", test_al_ipc_rests_at_d_hat),
    ("test_al_ipc_status_and_new_masses", test_al_ipc_status_and_new_masses),
    ("test_mas_matches_the_block_diagonal_preconditioner", test_mas_matches_the_block_diagonal_preconditioner),
):
    add_function_test(TestSolverIPCOptions, name, func, devices=devices)

for name, func in (
    ("test_attachment_moves_a_free_body", test_attachment_moves_a_free_body),
    ("test_cloth_hangs_from_a_static_body", test_cloth_hangs_from_a_static_body),
    ("test_attachment_enabled_inside_a_captured_graph", test_attachment_enabled_inside_a_captured_graph),
    ("test_hard_attachment_holds_as_a_ball_joint", test_hard_attachment_holds_as_a_ball_joint),
    ("test_zero_length_springs_oscillate_as_implicit_euler", test_zero_length_springs_oscillate_as_implicit_euler),
    ("test_attachment_changes", test_attachment_changes),
    ("test_no_stitches_without_attachments", test_no_stitches_without_attachments),
):
    add_function_test(TestSolverIPCAttachments, name, func, devices=devices)
for name, func in (
    ("test_kinematic_pusher_is_swept", test_kinematic_pusher_is_swept),
    ("test_kinematic_aims_captured_equal_eager", test_kinematic_aims_captured_equal_eager),
    ("test_grown_lists_keep_kinematic_aims", test_grown_lists_keep_kinematic_aims),
    ("test_kinematic_aims_under_bdf2", test_kinematic_aims_under_bdf2),
    ("test_kinematic_platform_carries_a_box", test_kinematic_platform_carries_a_box),
    ("test_kinematic_body_lags_under_load", test_kinematic_body_lags_under_load),
    ("test_kinematic_root_tracks_its_angle", test_kinematic_root_tracks_its_angle),
    ("test_reset_teleports_a_kinematic_body", test_reset_teleports_a_kinematic_body),
    ("test_jump_mode_builds_the_engine_of_before", test_jump_mode_builds_the_engine_of_before),
):
    add_function_test(TestSolverIPCKinematicAims, name, func, devices=devices)

for name, func in (
    ("test_contact_output_stacking_reports_weights", test_contact_output_stacking_reports_weights),
    ("test_contact_output_friction_at_rest", test_contact_output_friction_at_rest),
    ("test_contact_output_soft_rows", test_contact_output_soft_rows),
    ("test_contact_output_rigid_rows_balance_a_body", test_contact_output_rigid_rows_balance_a_body),
    ("test_contact_output_only_enabled_pairs", test_contact_output_only_enabled_pairs),
    ("test_contact_output_capacity", test_contact_output_capacity),
    ("test_contact_output_not_requested", test_contact_output_not_requested),
    ("test_contact_output_captured_equals_eager", test_contact_output_captured_equals_eager),
    ("test_contact_output_failed_step_has_no_rows", test_contact_output_failed_step_has_no_rows),
    ("test_contact_output_after_grow", test_contact_output_after_grow),
    ("test_contact_output_refusals", test_contact_output_refusals),
    ("test_contact_output_coupled_worlds", test_contact_output_coupled_worlds),
):
    add_function_test(TestSolverIPCContactOutput, name, func, devices=devices)
for name, func in (
    ("test_d6_joints_take_libuipcs_joints", test_d6_joints_take_libuipcs_joints),
    ("test_d6_coordinates_are_newtons", test_d6_coordinates_are_newtons),
    ("test_d6_joint_f_is_featherstones", test_d6_joint_f_is_featherstones),
    ("test_rejects_d6_joints_libuipc_lacks", test_rejects_d6_joints_libuipc_lacks),
    ("test_distance_joint_swings_as_a_pendulum", test_distance_joint_swings_as_a_pendulum),
    ("test_distance_joint_between_bodies", test_distance_joint_between_bodies),
    ("test_limit_damping_is_implicit_euler", test_limit_damping_is_implicit_euler),
    ("test_joint_options_change_nothing_unused", test_joint_options_change_nothing_unused),
):
    add_function_test(TestSolverIPCJointOptions, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
