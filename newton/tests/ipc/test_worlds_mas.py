# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC's MAS preconditioner with independent worlds: every world is preconditioned as the same world alone.

``fem_preconditioner="mas"`` builds libuipc-warp's MAS on the particles of cloth, soft bodies and springs. With
several worlds, libuipc-warp builds MAS's levels once per world, from that world's own particles, so that a world's
preconditioner, and with it its PCG iterations, do not depend on the other worlds (libuipc-warp's
``tests/test_worlds_mas.py`` holds every world's PCG iterations to those of the world alone). Here, through Newton's
model: every world takes the Newton iterations of the same world built alone at every step, eager and captured, and
its positions, status and residual are those of the run alone."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.ipc._fixtures import _box_body, _cloth, _world_rows, _worlds_at
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires


def _mas_world():
    """One world whose particles MAS takes: a soft cube of 3 x 3 x 3 cells above the global static box, a 10 x 10 cloth
    above the ground beside it, a rigid box and a free particle, which MAS leaves to its 3 x 3 block."""
    world = newton.ModelBuilder()
    world.add_soft_grid(
        pos=wp.vec3(0.3, -0.075, 0.25), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0), dim_x=3, dim_y=3,
        dim_z=3, cell_x=0.05, cell_y=0.05, cell_z=0.05, density=1000.0, k_mu=3.4e4, k_lambda=1.6e5, k_damp=0.0,
    )  # fmt: skip
    _cloth(world, 10, 0.03, 0.08, center=(-0.6, 0.0))
    _box_body(world, (-0.3, 0.3, 0.15), (0.05, 0.05, 0.05))
    world.add_particle(wp.vec3(0.0, 0.3, 0.1), wp.vec3(0.0, 0.0, 0.0), 0.01, radius=0.01)
    return world


def _record(model, solver, steps, dt=0.01, capture=False):
    """``steps`` steps from the model's state, eager, or captured two at a time after one eager step. After each step
    (each replay's second step when captured): every local world's particle and body positions, its Newton iterations
    (libuipc-warp's ``frame_status(engine).worlds``), and the STATUS and SOLVER_RESIDUAL entries of every world and the
    global one."""
    import uipc_warp.arrays  # noqa: PLC0415

    observables = solver.observables({solver.ObservableKind.STATUS, solver.ObservableKind.SOLVER_RESIDUAL})
    rows = _world_rows(model, model.world_count)
    out = {"q": [], "iterations": [], "status": [], "residual": []}

    def record(state):
        pq, bq = state.particle_q.numpy(), state.body_q.numpy()
        out["q"].append([np.concatenate([pq[p].reshape(-1), bq[b][:, :3].reshape(-1)]) for p, b in rows])
        status = uipc_warp.arrays.frame_status(solver._engine)
        out["iterations"].append([world.newton_iterations for world in status.worlds] or [status.newton_iterations])
        out["status"].append(observables.status.numpy().copy())
        out["residual"].append(observables.solver_residual.numpy().copy())

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
        for _ in range(steps):
            solver.step(s0, s1, None, None, dt, observables=observables)
            s0, s1 = s1, s0
            record(s0)
    return {key: np.array(value) for key, value in out.items()}


def test_independent_worlds_with_mas_step_as_alone(test, device):
    """Three worlds of a soft cube, a cloth, a rigid box and a particle (``_mas_world``) dropped from heights 3 cm apart
    onto the global static box and ground, independent worlds under MAS: in every step each world takes the Newton
    iterations of the same world built alone under MAS, its positions stay within 1e-6 m of the run alone, its STATUS
    entry is the run alone's and its SOLVER_RESIDUAL entry the run alone's to 1e-4 (MAS adds its coarse residuals in
    float32 with atomics, in another order in the scene than alone, which moves the last Newton step's displacement:
    3.6e-6 relative measured); the global entry holds 0. The same from captured steps, replayed two at a time. The
    worlds differ: their Newton iterations part in some steps."""
    shifts, steps = (0.0, 0.03, 0.06), 31
    for capture in (False, True):
        with test.subTest(captured=capture):
            model = _worlds_at(device, shifts, _mas_world)
            solver = newton.solvers.SolverIPC(model, fem_preconditioner="mas")
            test.assertTrue(solver._independent)
            run = _record(model, solver, steps, capture=capture)
            np.testing.assert_array_equal(run["status"][:, -1], 0)  # the global world: no unknowns, no step
            test.assertTrue(np.any(run["iterations"].min(axis=1) != run["iterations"].max(axis=1)))
            for k, shift in enumerate(shifts):
                one = _worlds_at(device, [shift], _mas_world)
                alone = _record(one, newton.solvers.SolverIPC(one, fem_preconditioner="mas"), steps, capture=capture)
                np.testing.assert_array_equal(run["iterations"][:, k], alone["iterations"][:, 0], f"world {k}")
                worst = np.abs(run["q"][:, k] - alone["q"][:, 0]).max()
                test.assertLess(worst, 1.0e-6, f"world {k}: {worst:.3g} m")
                np.testing.assert_array_equal(run["status"][:, k], alone["status"][:, 0], f"world {k}")
                np.testing.assert_allclose(
                    run["residual"][:, k], alone["residual"][:, 0], rtol=1.0e-4, atol=0.0, err_msg=f"world {k}"
                )


@requires("uipc_warp")
class TestSolverIPCWorldsMAS(unittest.TestCase):
    """MAS with independent worlds, each world as the same world alone."""


devices = get_cuda_test_devices()
add_function_test(
    TestSolverIPCWorldsMAS,
    "test_independent_worlds_with_mas_step_as_alone",
    test_independent_worlds_with_mas_step_as_alone,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
