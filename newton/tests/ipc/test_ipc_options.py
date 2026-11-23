# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""libuipc's options in SolverIPC: BDF2, AL-IPC contact and the MAS preconditioner."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.ipc._fixtures import (
    _CONVERGED,
    _TIGHT,
    _box_body,
    _cloth,
    _drive_recurrence,
    _drive_recurrence_bdf2,
    _hinged,
    _scene_calls,
    _steps,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires


def _bdf2_from_rest(n, h, a):
    """``(x_n - x_0, v_n)`` of BDF2 from rest under a constant acceleration ``a``, its history started by an implicit
    Euler step: ``v_n = n h a`` exactly and ``x_n - x_0 = a h^2 (n^2 / 2 + 3/4 (1 - 3^-n))``."""
    n = np.asarray(n, np.float64)
    return a * h * h * (n * n / 2.0 + 0.75 * (1.0 - 3.0**-n)), n * h * a


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


def test_bdf2_history_restarts_where_gravity_changes(test, device):
    """With integrator="bdf2" a change of gravity starts the history anew, at the next step and also between replays of
    a captured graph: a particle and a rigid box that fall from rest under one gravity and then under another follow
    v = v_k + n h g2 after the change, to the float32 output, where BDF2 within its history would lag h (g2 - g1) / 2
    behind for good. With two independent worlds a change in world 1 restarts world 1 alone: world 0 steps bit for
    bit as in a run without the change. A gravity notified unchanged restarts nothing: the steps are bit for bit those
    of a run without the notification."""
    tight = {"newton_velocity_tol": 1e-6, "newton_transrate_tol": 1e-6, "linear_tol_rate": 1e-12}
    tight["semi_implicit"] = False
    h = 0.01
    g1, g2 = np.array([0.0, 0.0, -9.81]), np.array([1.0, 0.0, 3.0])

    def run(model, solver, capture, change):
        """Five steps, ``change()``, four steps; the states after the last step and after steps 7 and 9."""
        s0, s1 = model.state(), model.state()
        solver.step(s0, s1, None, None, h)
        s0, s1 = s1, s0
        if capture:
            with wp.ScopedCapture(device=device) as cap:
                solver.step(s0, s1, None, None, h)
                solver.step(s1, s0, None, None, h)
        out = []
        for k in range(4):  # two steps each
            if k == 2:
                change()
            if capture:
                wp.capture_launch(cap.graph)
            else:
                for _ in range(2):
                    solver.step(s0, s1, None, None, h)
                    s0, s1 = s1, s0
            bodies = (s0.body_q.numpy(), s0.body_qd.numpy()) if model.body_count else (None, None)
            out.append((s0.particle_q.numpy(), s0.particle_qd.numpy(), *bodies))
        return out

    builder = newton.ModelBuilder(gravity=tuple(g1))
    builder.add_particle(wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.0), 0.5)
    b = _box_body(builder, (1.0, 0.0, 1.0), (0.1, 0.1, 0.1))
    for capture in (False, True):
        with test.subTest(capture=capture):
            model = builder.finalize(device=device)
            solver = newton.solvers.SolverIPC(model, integrator="bdf2", **tight)

            def change(model=model, solver=solver):
                model.set_gravity(tuple(g2))
                solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)

            states = run(model, solver, capture, change)
            for (_, qd, _, body_qd), n in zip(states[2:], (2, 4), strict=True):
                v = 5.0 * h * g1 + n * h * g2
                np.testing.assert_allclose(qd[0], v, rtol=0.0, atol=1e-5)
                np.testing.assert_allclose(body_qd[b, :3], v, rtol=0.0, atol=1e-5)

    builder = newton.ModelBuilder(gravity=tuple(g1))
    for w in range(2):
        builder.begin_world(gravity=tuple(g1))
        builder.add_particle(wp.vec3(float(w), 0.0, 1.0), wp.vec3(0.0, 0.0, 0.0), 0.5)
        builder.end_world()
    runs = {}
    for case in ("none", "world 1", "unchanged"):
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverIPC(model, integrator="bdf2", **tight)
        test.assertTrue(solver._independent)
        gravity = {"none": None, "world 1": g2, "unchanged": g1}[case]

        def change(model=model, solver=solver, gravity=gravity):
            if gravity is not None:
                model.set_gravity(tuple(gravity), world=1)
                solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)

        runs[case] = run(model, solver, True, change)
    for k in range(4):
        for i in (0, 1):
            np.testing.assert_array_equal(runs["unchanged"][k][i], runs["none"][k][i])
            np.testing.assert_array_equal(runs["world 1"][k][i][0], runs["none"][k][i][0])
    for (_, qd, _, _), n in zip(runs["world 1"][2:], (2, 4), strict=True):
        np.testing.assert_allclose(qd[1], 5.0 * h * g1 + n * h * g2, rtol=0.0, atol=1e-5)


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
    distance AL-IPC's constraint keeps, to 1%, and at rest; the barrier keeps it inside d_hat. A static box, whose
    vertices have no mass for AL-IPC's penalty, is refused."""
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
    # a static shape has no mass for AL-IPC's penalty: refused
    builder = newton.ModelBuilder()
    _box_body(builder, (0.0, 0.0, 0.17), (0.05, 0.05, 0.05))
    builder.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()), hx=0.1, hy=0.1, hz=0.05)
    with test.assertRaisesRegex(NotImplementedError, "static shapes"):
        newton.solvers.SolverIPC(builder.finalize(device=device), contact_constitution="al-ipc")


def test_al_ipc_holds_free_particles(test, device):
    """With contact_constitution="al-ipc" free particles collide. Two particles of radius 1 cm closing at 2 m/s without
    gravity are held d_hat beyond the sum of their radii, and a particle dropped onto the ground comes to rest d_hat
    above its radius, every step without a status bit: libuipc-warp's AL-IPC holds the pairs of two points and of a
    point and an edge. (libuipc's AL-IPC has no such pairs: the two particles stop where they touch, in a step that
    runs to its Newton iteration limit, and the solver refused models whose free particles collide.)"""

    def run(model, d_hat, steps):
        solver = newton.solvers.SolverIPC(model, d_hat=d_hat, contact_constitution="al-ipc")
        observables = solver.observables({solver.ObservableKind.STATUS})
        s0, s1 = model.state(), model.state()
        for _ in range(steps):
            solver.step(s0, s1, None, None, 0.01, observables=observables)
            test.assertFalse(solver.check_status(observables))
            s0, s1 = s1, s0
        test.assertLess(np.abs(s0.particle_qd.numpy()).max(), 1e-3)  # at rest
        return s0.particle_q.numpy()[:, 2]

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_particle(wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, 1.0), mass=1.0, radius=0.01)
    builder.add_particle(wp.vec3(0.0, 0.0, 1.1), wp.vec3(0.0, 0.0, -1.0), mass=1.0, radius=0.01)
    z = run(builder.finalize(device=device), 0.005, 20)
    test.assertAlmostEqual(z[1] - z[0], 0.02 + 0.005, delta=0.01 * 0.005)
    builder = newton.ModelBuilder()
    builder.add_particle(wp.vec3(0.0, 0.0, 0.02), wp.vec3(0.0, 0.0, 0.0), 0.01, radius=0.01)
    builder.add_ground_plane()
    z = run(builder.finalize(device=device), 0.001, 100)
    test.assertAlmostEqual(z[0], 0.01 + 0.001, delta=0.01 * 0.001)


def test_al_ipc_keeps_a_stack(test, device):
    """Under contact_constitution="al-ipc" SolverIPC takes the AL-IPC paper's decay factor 0.9 of an opened pair's
    weight, where libuipc's default 0.3 drops the pair within three outer updates, so that a contact that opens for a
    moment joins again at the surface and is pushed out to d_hat again: a stack of two 0.1 m boxes released 10 mm above
    the ground and above each other, d_hat 5 mm, holds through 300 steps of 0.01 s, the upper box over the lower one at
    every step, and is at rest over the last 100 steps. (Under 0.3 the stack keeps bouncing, faster than 6 m/s, and the
    upper box falls off in some runs.)"""
    d_hat = 0.005
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    _box_body(builder, (0.0, 0.0, 0.06), (0.05, 0.05, 0.05))
    _box_body(builder, (0.0, 0.0, 0.17), (0.05, 0.05, 0.05))
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, d_hat=d_hat, contact_constitution="al-ipc")
    s0, s1 = model.state(), model.state()
    for k in range(300):
        solver.step(s0, s1, None, None, 0.01)
        s0, s1 = s1, s0
        q = s0.body_q.numpy()
        test.assertLess(np.abs(q[1, :2] - q[0, :2]).max(), 0.05, f"step {k}: the upper box is off the lower one")
        test.assertGreater(q[1, 2] - q[0, 2], 0.1, f"step {k}: the upper box is below the lower one's top")
        if k >= 200:
            test.assertLess(np.abs(s0.body_qd.numpy()).max(), 1e-3, f"step {k}: the stack moves")
    test.assertEqual(solver._config.al_decay_factor, 0.9)


def test_al_ipc_sweeps_kinematic_bodies(test, device):
    """Under contact_constitution="al-ipc" a kinematic body moved between steps is swept from where the last step left
    it to its state_in pose (AL-IPC's moving boundaries; libuipc's frame takes the written pose, and nothing the body
    moves into collides with it): a box moved at 1.2 m/s pushes a free sphere it reaches along, the two apart after
    every step. The first step, and the step after reset(), place it at its pose instead: put across the sphere there,
    it does not touch it, where the same move swept pushes the sphere away."""
    dt, speed = 1.0 / 240.0, 1.2

    def scene():
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        kinematic = builder.add_body(
            xform=wp.transform(wp.vec3(-0.3, 0.0, 0.0), wp.quat_identity()), mass=1.0, is_kinematic=True
        )
        builder.add_shape_box(kinematic, hx=0.1, hy=0.1, hz=0.1)
        probe = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), mass=1.0)
        builder.add_shape_sphere(probe, radius=0.05)
        model = builder.finalize(device=device)
        return model, newton.solvers.SolverIPC(model, contact_constitution="al-ipc"), kinematic, probe

    def place(state, body, x, vx=0.0):
        q, qd = state.body_q.numpy(), state.body_qd.numpy()
        q[body, 0], qd[body, :3] = x, (vx, 0.0, 0.0)
        state.body_q.assign(q)
        state.body_qd.assign(qd)

    model, solver, kinematic, probe = scene()
    s0, s1 = model.state(), model.state()
    gaps = []
    for k in range(60):
        place(s0, kinematic, -0.3 + speed * dt * (k + 1), speed)
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        q = s0.body_q.numpy()
        gaps.append(q[probe, 0] - 0.05 - (q[kinematic, 0] + 0.1))
    test.assertGreater(min(gaps), -1e-6)
    test.assertGreater(float(s0.body_q.numpy()[probe, 0]), 0.1)
    # pushed along; with libuipc's decay factor 0.3 the push-out of a pair that joins again drove the sphere faster
    # than the box, with SolverIPC's 0.9 it keeps up with it
    test.assertGreater(float(s0.body_qd.numpy()[probe, 0]), 0.5 * speed)
    # placements: the first step, and the step after reset(); a move without one is swept
    model, solver, kinematic, probe = scene()
    s0, s1 = model.state(), model.state()
    start = s0.body_q.numpy()[probe].copy()
    for x, reset, swept in ((0.3, False, False), (-0.3, True, False), (0.3, False, True)):
        if reset:
            solver.reset(s0)
        place(s0, kinematic, x)
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        moved = float(np.linalg.norm(s0.body_q.numpy()[probe, :3] - start[:3]))
        if swept:
            test.assertGreater(moved, 0.1)
        else:
            test.assertLess(moved, 1e-6)


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
    test.assertEqual(solver.supported_observables, {kind.STATUS, kind.BODY_PARENT_F, kind.BODY_QDD})
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
    with _scene_calls(solver._uipc_warp.arrays.ArrayScene.add_affine_body) as bodies:
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    test.assertIsNot(solver._engine, engine)
    test.assertAlmostEqual(float(bodies[0]["mass"]), float(model.body_mass.numpy()[0]), places=4)
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
class TestSolverIPCOptions(unittest.TestCase):
    """libuipc's options: BDF2, AL-IPC contact and the MAS preconditioner."""


devices = get_cuda_test_devices()
for name, func in (
    ("test_bdf2_falls_from_an_implicit_euler_step", test_bdf2_falls_from_an_implicit_euler_step),
    (
        "test_bdf2_turns_a_body_by_the_impulse_of_its_first_step",
        test_bdf2_turns_a_body_by_the_impulse_of_its_first_step,
    ),
    ("test_bdf2_is_second_order_on_a_spring", test_bdf2_is_second_order_on_a_spring),
    ("test_bdf2_history_starts_anew", test_bdf2_history_starts_anew),
    ("test_bdf2_history_restarts_where_gravity_changes", test_bdf2_history_restarts_where_gravity_changes),
    ("test_bdf2_drives", test_bdf2_drives),
    ("test_al_ipc_rests_at_d_hat", test_al_ipc_rests_at_d_hat),
    ("test_al_ipc_holds_free_particles", test_al_ipc_holds_free_particles),
    ("test_al_ipc_keeps_a_stack", test_al_ipc_keeps_a_stack),
    ("test_al_ipc_sweeps_kinematic_bodies", test_al_ipc_sweeps_kinematic_bodies),
    ("test_al_ipc_status_and_new_masses", test_al_ipc_status_and_new_masses),
    ("test_mas_matches_the_block_diagonal_preconditioner", test_mas_matches_the_block_diagonal_preconditioner),
):
    add_function_test(TestSolverIPCOptions, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
