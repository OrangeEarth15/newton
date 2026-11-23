# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""How a SolverIPC step ends and what carries from one step to the next."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton.tests.ipc._fixtures import (
    _CONVERGED,
    _TIGHT,
    _TINY_LISTS,
    _active_slots,
    _box_body,
    _boxes_apart,
    _cloth,
    _hinged,
    _run,
    _soft_cube,
    _stacking_cubes,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires


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
    test.assertGreater(_active_slots(solver)[0], 1)  # the point-triangle list grew from _TINY_LISTS's one pair
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


def test_grown_lists_keep_a_changed_gravity(test, device):
    """A gravity changed through notify_model_changed holds after contact lists that overflow have grown, in an eager
    step and through grow(): the cubes fall and land as those of a model built with that gravity, whose lists grow the
    same way. (The grown solve used to take the gravity the model was laid out with.)"""
    tight = {"newton_velocity_tol": 1e-6, "linear_tol_rate": 1e-12, "semi_implicit": False}
    g = (0.0, 0.0, -2.0 * 9.81)

    def run(changed: bool, capture: bool):
        model = _stacking_cubes(device)
        if not changed:
            model.set_gravity(g)
        solver = newton.solvers.SolverIPC(model, **tight)
        solver._make_engine(_TINY_LISTS)
        if changed:
            model.set_gravity(g)
            solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
        observables = solver.observables({solver.ObservableKind.STATUS})
        s0, s1 = model.state(), model.state()
        for _ in range(20):
            if capture:
                with wp.ScopedCapture(device=device) as cap:
                    solver.step(s0, s1, None, None, 0.01, observables=observables)
                wp.capture_launch(cap.graph)
                if solver.check_status(observables, grow=True):  # grown: the same step again
                    with wp.ScopedCapture(device=device) as cap:
                        solver.step(s0, s1, None, None, 0.01, observables=observables)
                    wp.capture_launch(cap.graph)
            else:
                solver.step(s0, s1, None, None, 0.01)
            s0, s1 = s1, s0
        test.assertGreater(_active_slots(solver)[0], 1)  # the point-triangle list grew from _TINY_LISTS's one pair
        return s0.particle_q.numpy(), s0.particle_qd.numpy()

    for capture in (False, True):
        with test.subTest(capture=capture):
            q, qd = run(True, capture)
            q_ref, qd_ref = run(False, capture)
            np.testing.assert_allclose(q, q_ref, rtol=0.0, atol=1e-6)
            np.testing.assert_allclose(qd, qd_ref, rtol=0.0, atol=1e-4)


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
    """graph_outdated() gives the engine the Newton and PCG iterations of the captured steps replayed since it was last
    called (libuipc-warp's SimEngine.graph_outdated, whose choice of the PCG loop its own tests hold). A cloth hanging
    from its top edge, solved to 1e-12, takes about 270 PCG iterations per Newton iteration: the first call after
    replays returns True, a second call False, and the steps captured from then on, with the other loop, end where
    the steps of the old graph end, to 1e-6. A soft cube falling freely, solved to 1e-3, takes fewer than 128, and the
    loop stays."""
    import uipc_warp  # noqa: PLC0415

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
            s0, s1 = model.state(), model.state()
            with wp.ScopedCapture(device=device) as capture:
                solver.step(s0, s1, None, None, 0.01)
                solver.step(s1, s0, None, None, 0.01)
            wp.capture_launch(capture.graph)
            after = (s0.particle_q.numpy(), s0.particle_qd.numpy())
            wp.capture_launch(capture.graph)
            outdated_of = uipc_warp.SimEngine.graph_outdated
            with mock.patch.object(
                uipc_warp.SimEngine, "graph_outdated", autospec=True, side_effect=outdated_of
            ) as asked:
                test.assertEqual(solver.graph_outdated(), outdated)
                test.assertFalse(solver.graph_outdated())  # the counts were taken
            newton_iterations, pcg_iterations = asked.call_args_list[0].kwargs["iterations"]
            test.assertGreater(newton_iterations, 0)  # the four replayed steps' sums
            test.assertEqual(pcg_iterations / newton_iterations > 128, long)
            test.assertEqual(tuple(asked.call_args_list[1].kwargs["iterations"]), (0, 0))
            if outdated:
                old = s0.particle_q.numpy()
                s0.particle_q.assign(after[0])
                s0.particle_qd.assign(after[1])
                with wp.ScopedCapture(device=device) as capture:
                    solver.step(s0, s1, None, None, 0.01)
                    solver.step(s1, s0, None, None, 0.01)
                wp.capture_launch(capture.graph)
                np.testing.assert_allclose(s0.particle_q.numpy(), old, rtol=0.0, atol=1e-6)


def test_graph_outdated_takes_the_engines_guesses(test, device):
    """When the engine says that a step captured from now on would hold better guesses of its size ladders or another
    PCG loop (libuipc-warp's SimEngine.graph_outdated, whose ladders and loop its own tests hold), graph_outdated()
    gives it the iterations of the replayed steps and returns True, the engine takes the guesses for the next capture
    (prepare_capture, as libuipc-warp's FrameRunner does before it captures) and the solver drops its own graph of a
    step, which the next eager step captures anew; when the engine says no, it returns False and asks for nothing. The
    engine's answers are given here."""
    import uipc_warp  # noqa: PLC0415

    model = _hanging_cloth(device)
    solver = newton.solvers.SolverIPC(model)
    answers = []
    with (
        mock.patch.object(
            uipc_warp.SimEngine,
            "graph_outdated",
            autospec=True,
            side_effect=lambda engine, iterations=None: answers.pop(0) if answers else False,
        ) as asked,
        mock.patch.object(uipc_warp.SimEngine, "prepare_capture", autospec=True) as prepared,
    ):
        s0, s1 = model.state(), model.state()
        solver.step(s0, s1, None, None, 0.01)  # an eager step: captured as the solver's own graph, which stays
        graph = solver._frame_graph
        test.assertIsNotNone(graph)
        with wp.ScopedCapture(device=device) as capture:
            solver.step(s1, s0, None, None, 0.01)
        wp.capture_launch(capture.graph)
        answers[:] = [True, False]
        test.assertTrue(solver.graph_outdated())
        newton_iterations, pcg_iterations = asked.call_args.kwargs["iterations"]  # the replayed step's
        test.assertGreater(newton_iterations, 0)
        test.assertGreater(pcg_iterations, 0)
        test.assertEqual(prepared.call_count, 1)
        test.assertIsNone(solver._frame_graph)
        test.assertFalse(solver.graph_outdated())
        test.assertEqual(prepared.call_count, 1)
        solver.step(s0, s1, None, None, 0.01)
        test.assertIsNotNone(solver._frame_graph)
        test.assertIsNot(solver._frame_graph, graph)


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


def test_notify_new_masses_without_joints(test, device):
    """A model of bodies without joints (links that no joint holds) takes new masses in place: a box built with
    add_link, given twice its mass and inertia, falls as before (gravity does not depend on the mass) and the
    engine holds the new dyadic mass."""
    builder = newton.ModelBuilder()
    b = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
    builder.add_shape_box(b, hx=0.1, hy=0.1, hz=0.1)
    model = builder.finalize(device=device)
    test.assertEqual(model.joint_count, 0)
    solver = newton.solvers.SolverIPC(model, newton_velocity_tol=1e-5, linear_tol_rate=1e-12)
    mass = float(model.body_mass.numpy()[b])
    model.body_mass.assign(2.0 * model.body_mass.numpy())
    model.body_inertia.assign(2.0 * model.body_inertia.numpy())
    model.body_inv_mass.assign(0.5 * model.body_inv_mass.numpy())
    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    test.assertAlmostEqual(float(solver._dyadic[0][b]), 2.0 * mass, places=5)
    traj = _run(model, solver, 10, field="body_q")
    k = np.arange(1, 11)
    np.testing.assert_allclose(traj[:, b, 2], 1.0 - 9.81 * 0.01**2 * k * (k + 1) / 2.0, atol=1e-5)


def test_notify_new_particle_masses(test, device):
    """New particle masses reported with MODEL_PROPERTIES, as a coupled solver gives its proxy particles theirs, build
    the solve again with them: a particle pushed up by particle_f = 10 N, its mass doubled from 1 kg to 2 kg, accelerates
    at 10 / 2 - 9.81 m/s^2 in implicit Euler's steps (with 1 kg it would rise), and the same notification without a
    new mass keeps the solve."""
    builder = newton.ModelBuilder()
    builder.add_particle(wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.0), mass=1.0, radius=0.01)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, newton_velocity_tol=1e-5, linear_tol_rate=1e-12)
    engine = solver._engine
    solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
    test.assertIs(solver._engine, engine)  # nothing new: the same solve
    model.particle_mass.assign(np.array([2.0], np.float32))
    model.particle_inv_mass.assign(np.array([0.5], np.float32))
    solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
    test.assertIsNot(solver._engine, engine)
    s0, s1 = model.state(), model.state()
    z = []
    for _ in range(10):
        s0.particle_f.assign(np.array([[0.0, 0.0, 10.0]], np.float32))
        solver.step(s0, s1, None, None, 0.01)
        s0, s1 = s1, s0
        z.append(float(s0.particle_q.numpy()[0, 2]))
    k = np.arange(1, 11)
    np.testing.assert_allclose(z, 1.0 + (10.0 / 2.0 - 9.81) * 0.01**2 * k * (k + 1) / 2.0, atol=1e-5)


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


@requires("uipc_warp")
class TestSolverIPCStep(unittest.TestCase):
    """How a step ends and what carries from one step to the next: status, overflow, state, reset, model changes."""


devices = get_cuda_test_devices()
for name, func in (
    ("test_contact_at_start_fails", test_contact_at_start_fails),
    ("test_check_status", test_check_status),
    ("test_default_d_hat_follows_small_features", test_default_d_hat_follows_small_features),
    ("test_overflow_grows", test_overflow_grows),
    ("test_grown_lists_keep_a_changed_gravity", test_grown_lists_keep_a_changed_gravity),
    ("test_teleport_equals_fresh_solver", test_teleport_equals_fresh_solver),
    ("test_reset_inside_captured_graph", test_reset_inside_captured_graph),
    ("test_notify_updates_gravity_and_inertia_in_place", test_notify_updates_gravity_and_inertia_in_place),
    ("test_notify_new_masses_without_joints", test_notify_new_masses_without_joints),
    ("test_notify_new_particle_masses", test_notify_new_particle_masses),
    ("test_gravity_per_world", test_gravity_per_world),
    ("test_graph_outdated_after_long_solves", test_graph_outdated_after_long_solves),
    ("test_graph_outdated_takes_the_engines_guesses", test_graph_outdated_takes_the_engines_guesses),
    ("test_default_control_is_built_once", test_default_control_is_built_once),
):
    add_function_test(TestSolverIPCStep, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
