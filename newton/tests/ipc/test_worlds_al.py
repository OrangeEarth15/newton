# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC's AL-IPC contact with several worlds: independent worlds each take the AL-IPC step of the same world alone,
and coupled worlds are one AL-IPC solve.

``contact_constitution="al-ipc"`` is libuipc's augmented-Lagrangian contact. With several worlds, libuipc-warp runs the
augmented-Lagrangian loop of every world apart: its own Newton iterations, line searches and outer updates, its own
active contact pairs and its own status (libuipc-warp's ``tests/test_worlds_al.py`` holds every world to its engine
alone, with failures, quotas and restarts). Here, through Newton's model: every world takes the Newton iterations of the
same world built alone at every step, eager and captured, and its positions and status are those of the run alone.
With ``coupled_worlds=True`` the worlds are one solve, and the solver discards friction per world, as with independent
worlds (libuipc-warp's ``tests/test_subscene_friction.py``): identical worlds take the step of one world alone."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.ipc._fixtures import _box_body, _world_rows
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires


def _al_world():
    """One world: a soft cube of 2 x 2 x 2 cells, a rigid box turned about two axes and thrown down at 4 m/s, which
    strikes the ground with a corner (steps of more Newton iterations than ``semi_implicit_k_min``), and a free
    particle."""
    world = newton.ModelBuilder()
    world.add_soft_grid(
        pos=wp.vec3(0.3, -0.05, 0.08), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0), dim_x=2, dim_y=2,
        dim_z=2, cell_x=0.05, cell_y=0.05, cell_z=0.05, density=1000.0, k_mu=3.4e4, k_lambda=1.6e5, k_damp=0.0,
    )  # fmt: skip
    turn = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 1.0, 0.0)), 0.4)
    box = _box_body(world, (-0.3, 0.0, 0.15), (0.05, 0.05, 0.05), rot=turn)
    world.body_qd[box] = wp.spatial_vector(0.0, 0.0, -4.0, 0.0, 0.0, 0.0)  # (linear, angular)
    world.add_particle(wp.vec3(0.0, 0.3, 0.05), wp.vec3(0.0, 0.0, -1.0), 0.01, radius=0.01)
    return world


def _al_worlds(device, shifts):
    """One world of ``_al_world`` per entry of ``shifts``, raised by it, over the global ground plane: worlds that
    differ. ``_al_worlds(device, [shift])`` is one of them alone."""
    scene = newton.ModelBuilder()
    scene.add_ground_plane()
    for shift in shifts:
        scene.add_world(_al_world(), xform=wp.transform(wp.vec3(0.0, 0.0, shift), wp.quat_identity()))
    return scene.finalize(device=device)


def _record(model, solver, steps, dt=0.01, capture=False, resets=None):
    """``steps`` steps from the model's state, eager, or captured two at a time after one eager step. After each step
    (each replay's second step when captured): every local world's particle and body positions, its Newton iterations
    (libuipc-warp's ``frame_status(engine).worlds``, or the one solve's) and the STATUS entries of every world and the
    global one. ``resets``: eager only, ``{step: world_mask}``, a ``reset`` with that mask before that step."""
    import uipc_warp.arrays  # noqa: PLC0415

    observables = solver.observables({solver.ObservableKind.STATUS})
    rows = _world_rows(model, model.world_count)
    out = {"q": [], "iterations": [], "status": []}

    def record(state):
        pq, bq = state.particle_q.numpy(), state.body_q.numpy()
        out["q"].append([np.concatenate([pq[p].reshape(-1), bq[b][:, :3].reshape(-1)]) for p, b in rows])
        status = uipc_warp.arrays.frame_status(solver._engine)
        out["iterations"].append([world.newton_iterations for world in status.worlds] or [status.newton_iterations])
        out["status"].append(observables.status.numpy().copy())

    s0, s1 = model.state(), model.state()
    if capture:
        solver.step(s0, s1, None, None, dt, observables=observables)
        s0, s1 = s1, s0
        record(s0)
        with wp.ScopedCapture(device=model.device) as cap:
            solver.step(s0, s1, None, None, dt, observables=observables)
            solver.step(s1, s0, None, None, dt, observables=observables)
        for _ in range((steps - 1) // 2):
            wp.capture_launch(cap.graph)
            record(s0)
    else:
        for i in range(steps):
            if resets and i in resets:
                solver.reset(s0, resets[i])
            solver.step(s0, s1, None, None, dt, observables=observables)
            s0, s1 = s1, s0
            record(s0)
    return {key: np.array(value) for key, value in out.items()}


def test_independent_worlds_under_al_ipc_step_as_alone(test, device):
    """Three worlds of a soft cube, a turned rigid box and a particle (``_al_world``) dropped from heights 3 cm apart
    onto the global ground, independent worlds under AL-IPC: in every step each world takes the Newton iterations of the
    same world built alone under AL-IPC, its positions stay within 1e-6 m of the run alone and its STATUS entry is the
    run alone's; the global entry holds 0. The same from captured steps, replayed two at a time. The worlds differ:
    their Newton iterations part in the step where a box strikes the ground."""
    shifts, steps = (0.0, 0.03, 0.06), 31
    for capture in (False, True):
        with test.subTest(captured=capture):
            model = _al_worlds(device, shifts)
            solver = newton.solvers.SolverIPC(model, contact_constitution="al-ipc")
            test.assertTrue(solver._independent)
            run = _record(model, solver, steps, capture=capture)
            np.testing.assert_array_equal(run["status"][:, -1], 0)  # the global world: no unknowns, no step
            if not capture:  # every step recorded (a replay records every second one)
                test.assertTrue(np.any(run["iterations"].min(axis=1) != run["iterations"].max(axis=1)))
            for k, shift in enumerate(shifts):
                one = _al_worlds(device, [shift])
                alone = _record(
                    one, newton.solvers.SolverIPC(one, contact_constitution="al-ipc"), steps, capture=capture
                )
                np.testing.assert_array_equal(run["iterations"][:, k], alone["iterations"][:, 0], f"world {k}")
                worst = np.abs(run["q"][:, k] - alone["q"][:, 0]).max()
                test.assertLess(worst, 1.0e-6, f"world {k}: {worst:.3g} m")
                np.testing.assert_array_equal(run["status"][:, k], alone["status"][:, 0], f"world {k}")


def test_coupled_worlds_under_al_ipc_step_as_one_world(test, device):
    """With coupled_worlds=True the worlds are one AL-IPC solve, and the solver discards friction per world: at every
    step for the worlds whose state it did not write, and at a reset by mask (libuipc-warp's ``discard_friction(where=)``
    with one flag per subscene, which its AL-IPC pipeline refused unless the worlds were independent: the first step
    raised).
    Two copies of ``_al_world`` at one height take the Newton iterations of the world built alone at every step, stay
    within 1e-6 m of its run and take its STATUS entry, eager and captured; eager across a reset of both worlds by mask
    after 10 steps, as the world alone across its reset. A reset of one of two worlds by mask steps on, every STATUS
    entry 0."""
    steps = 21
    both = wp.array([True, True, False], dtype=wp.bool, device=device)
    for capture in (False, True):
        with test.subTest(captured=capture):
            model = _al_worlds(device, (0.0, 0.0))
            solver = newton.solvers.SolverIPC(model, coupled_worlds=True, contact_constitution="al-ipc")
            test.assertFalse(solver._independent)
            run = _record(model, solver, steps, capture=capture, resets=None if capture else {10: both})
            one = _al_worlds(device, [0.0])
            solo = newton.solvers.SolverIPC(one, contact_constitution="al-ipc")
            alone = _record(one, solo, steps, capture=capture, resets=None if capture else {10: None})
            np.testing.assert_array_equal(run["iterations"][:, 0], alone["iterations"][:, 0])
            for k in range(2):
                worst = np.abs(run["q"][:, k] - alone["q"][:, 0]).max()
                test.assertLess(worst, 1.0e-6, f"world {k}: {worst:.3g} m")
                np.testing.assert_array_equal(run["status"][:, k], alone["status"][:, 0], f"world {k}")
    model = _al_worlds(device, (0.0, 0.03))
    solver = newton.solvers.SolverIPC(model, coupled_worlds=True, contact_constitution="al-ipc")
    one_world = wp.array([False, True, False], dtype=wp.bool, device=device)
    run = _record(model, solver, 11, resets={5: one_world})
    np.testing.assert_array_equal(run["status"], 0)


@requires("uipc_warp")
class TestSolverIPCWorldsALIPC(unittest.TestCase):
    """AL-IPC with several worlds: independent worlds each as the same world alone, coupled worlds as one solve."""


devices = get_cuda_test_devices()
add_function_test(
    TestSolverIPCWorldsALIPC,
    "test_independent_worlds_under_al_ipc_step_as_alone",
    test_independent_worlds_under_al_ipc_step_as_alone,
    devices=devices,
)
add_function_test(
    TestSolverIPCWorldsALIPC,
    "test_coupled_worlds_under_al_ipc_step_as_one_world",
    test_coupled_worlds_under_al_ipc_step_as_one_world,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
