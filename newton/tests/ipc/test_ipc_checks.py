# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""What SolverIPC checks and assumes of bodies, and how it reports the ends of a step: an inertia that is not
positive definite is refused, one that violates the triangle inequality is clamped with a warning, a body without a
closed collision mesh takes the volume of water of its mass, and the status bits TRAVERSAL_EXHAUSTED and
LINE_SEARCH_LIMIT fail a step or warn as documented."""

import dataclasses
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton.tests.ipc._fixtures import _box_body, _worlds_at
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires

_TIGHT = {"newton_velocity_tol": 1.0e-6, "linear_tol_rate": 1.0e-12, "semi_implicit": False}


def _bodies(device, count, kinematic=(), gravity=-9.81):
    """``count`` bodies of 1 kg and inertia 1e-3 kg m^2 a metre apart, 1 m up, each with a 10 cm box that adds no
    mass; the bodies of ``kinematic`` KINEMATIC."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
    for b in range(count):
        builder.add_body(
            xform=wp.transform(wp.vec3(float(b), 0.0, 1.0), wp.quat_identity()), mass=1.0,
            inertia=wp.mat33(np.eye(3) * 1.0e-3), is_kinematic=b in kinematic,
        )  # fmt: skip
        builder.add_shape_box(b, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    return builder.finalize(device=device)


def _set_inertia(model, body, moments):
    """The inertia of ``body`` in place, diagonal: Newton's builder would have corrected one that is not valid."""
    inertia = model.body_inertia.numpy()
    inertia[body] = np.diag(moments)
    model.body_inertia.assign(inertia)


def test_inertia_checks(test, device):
    """A dynamic body needs a positive-definite inertia: one with a zero principal moment, given in place, is refused
    by name when the solver is built, and when notify_model_changed(BODY_INERTIAL_PROPERTIES) reports it. A kinematic
    body may have none (forces do not move it), but not under kinematic_motion="aim", which moves it with its mass."""
    flat = (1.0e-3, 1.0e-3, 0.0)
    model = _bodies(device, 2, kinematic=(1,))
    _set_inertia(model, 1, flat)
    solver = newton.solvers.SolverIPC(model)
    with test.assertRaisesRegex(ValueError, "needs a positive-definite inertia for the kinematic body 1"):
        newton.solvers.SolverIPC(model, kinematic_motion="aim")
    _set_inertia(model, 0, flat)
    with test.assertRaisesRegex(ValueError, "needs a positive-definite inertia for the dynamic body 0"):
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    with test.assertRaisesRegex(ValueError, "needs a positive-definite inertia for the dynamic body 0"):
        newton.solvers.SolverIPC(model)


# A bug this test found (package N-B, reported, not fixed here): the clamp leaves the second moment with a zero
# eigenvalue, and building the solver then raises numpy's LinAlgError ("Singular matrix") where it inverts the body's
# 4 x 4 mass block (scene._mass_inverse). The test fails until that is fixed; then this mark goes.
@unittest.expectedFailure
def test_inertia_that_violates_the_triangle_inequality_is_clamped(test, device):
    """An inertia diag(1, 1, 3) g m^2, whose largest moment exceeds the sum of the others, has no mass distribution:
    its second moment about the centre of mass, tr(I)/2 - I = diag(1.5, 1.5, -0.5), is not positive semi-definite.
    The solver warns and clamps the negative part to 0, which makes the inertia diag(1.5, 1.5, 3): over ten steps of
    0.01 s without gravity a torque of 1e-4 N m about x spins the body at tau t / 1.5 g m^2, and one about z another
    body at tau t / 3 g m^2, as implicit Euler has a rigid body spin up."""
    dt, tau = 0.01, 1.0e-4
    model = _bodies(device, 2, gravity=0.0)
    for b in (0, 1):
        _set_inertia(model, b, (1.0e-3, 1.0e-3, 3.0e-3))
    with test.assertWarnsRegex(
        newton.NewtonWarning, "the inertia of body 0 violates the triangle inequality; clamping"
    ):
        solver = newton.solvers.SolverIPC(model, **_TIGHT)
    s0, s1 = model.state(), model.state()
    wrench = np.zeros((2, 6), np.float32)
    wrench[0, 3], wrench[1, 5] = tau, tau
    for s in (s0, s1):
        s.body_f.assign(wrench)
    k = np.arange(1, 11)
    spin = []
    for _ in range(10):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        spin.append(s0.body_qd.numpy()[:, 3:].astype(np.float64))
    spin = np.array(spin)
    np.testing.assert_allclose(spin[:, 0, 0], tau * dt * k / 1.5e-3, rtol=1.0e-3)
    np.testing.assert_allclose(spin[:, 1, 2], tau * dt * k / 3.0e-3, rtol=1.0e-3)
    np.testing.assert_allclose(spin[:, 0, 1:], 0.0, atol=1.0e-6)
    np.testing.assert_allclose(spin[:, 1, :2], 0.0, atol=1.0e-6)


def test_body_without_a_closed_mesh_takes_the_volume_of_water(test, device):
    """libuipc's affine body needs a volume, the weight of its orthogonality potential, which SolverIPC takes from the
    body's closed collision meshes. A body of 2 kg whose only collision mesh is an open sheet (two triangles) takes the
    volume of 2 kg of water, 2e-3 m^3: its orthogonality stiffness for a step of dt is libuipc's 1e8 Pa lowered to its
    largest moment of inertia over V (4 * 0.1^2 dt^2), with that V, where a 20 cm box of the same mass and inertia
    takes its own 8e-3 m^3. Both fall as implicit Euler has a free body fall."""
    import uipc_warp.arrays  # noqa: PLC0415

    dt, mass, inertia = 0.01, 2.0, (2.0e-2, 3.0e-2, 4.0e-2)
    builder = newton.ModelBuilder()
    sheet = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), mass=mass,
                             inertia=wp.mat33(np.diag(inertia)))  # fmt: skip
    v = np.array([[-0.1, -0.1, 0.0], [0.1, -0.1, 0.0], [0.1, 0.1, 0.0], [-0.1, 0.1, 0.0]])
    builder.add_shape_mesh(sheet, mesh=newton.Mesh(v, np.array([0, 1, 2, 0, 2, 3]), compute_inertia=False),
                           cfg=newton.ModelBuilder.ShapeConfig(density=0.0))  # fmt: skip
    box = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 1.0), wp.quat_identity()), mass=mass,
                           inertia=wp.mat33(np.diag(inertia)))  # fmt: skip
    builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, **_TIGHT)
    s0, s1 = model.state(), model.state()
    z = []
    for _ in range(5):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        z.append(s0.body_q.numpy()[:, 2].astype(np.float64))
    k = np.arange(1, 6)
    np.testing.assert_allclose(np.array(z), np.outer(1.0 - 9.81 * dt * dt * k * (k + 1) / 2.0, [1.0, 1.0]), atol=1e-6)
    ortho = uipc_warp.arrays.orthogonality(solver._engine)
    kappa = dict(zip(ortho.bodies.tolist(), ortho.kappa.numpy().tolist(), strict=True))
    for b, volume in ((sheet, mass / 1000.0), (box, 0.2**3)):
        expected = min(1.0e8, max(inertia) / volume / (4.0 * 0.1**2 * dt * dt))
        test.assertAlmostEqual(kappa[b] / expected, 1.0, places=6)


def test_line_search_limit_warns(test, device):
    """A line search that runs out of its trials does not fail the step. With line_search_max_iter=1, a plank of 80 x 20
    x 4 cm spun at 30 rad/s in libuipc's step (body_rotation="linear", libuipc's body stiffness 1e8 Pa), whose
    quartic orthogonality potential the first trial of a Newton step overshoots, sets LINE_SEARCH_LIMIT in every step,
    and check_status warns of it, naming line_search_max_iter, and returns False; with the default eight trials no
    step sets it."""

    def spin(trials):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        plank = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_box(plank, hx=0.4, hy=0.1, hz=0.02)
        model = builder.finalize(device=device)
        model.body_qd.assign(np.array([[0.0, 0.0, 0.0, 9.0, 15.0, 30.0]], np.float32))
        solver = newton.solvers.SolverIPC(model, body_rotation="linear", body_kappa=1.0e8, line_search_max_iter=trials)
        observables = solver.observables({solver.ObservableKind.STATUS})
        s0, s1 = model.state(), model.state()
        bits = []
        for _ in range(5):
            solver.step(s0, s1, None, None, 0.01, observables=observables)
            s0, s1 = s1, s0
            bits.append(int(observables.status.numpy().max()))
            if bits[-1] & solver.StatusBits.LINE_SEARCH_LIMIT:
                with test.assertWarnsRegex(newton.NewtonWarning, "a line search ran out of its line_search_max_iter"):
                    test.assertFalse(solver.check_status(observables))
        return bits

    bits = newton.solvers.SolverIPC.StatusBits
    limited = spin(1)
    test.assertTrue(all(b & bits.LINE_SEARCH_LIMIT and not b & bits.FAILED for b in limited), limited)
    test.assertFalse(any(b & bits.LINE_SEARCH_LIMIT for b in spin(8)))


def test_traversal_exhausted_fails_the_step(test, device):
    """An exhausted BVH traversal leaves contact candidates out: the step fails. A captured step sets
    TRAVERSAL_EXHAUSTED in its status, and check_status raises naming it and the BVH's node budget. An eager step whose
    frame ends so (libuipc-warp's frame_status reporting it) raises the same, leaves state_out unwritten, and the next
    step takes state_in, as a new solver would. With independent worlds the traversal is shared: the whole step fails,
    not the worlds one by one."""
    import uipc_warp.arrays  # noqa: PLC0415

    dt = 0.01
    real = uipc_warp.arrays.frame_status

    def exhausted(engine):
        return dataclasses.replace(real(engine), traversal_exhausted=True)

    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    _box_body(builder, (0.0, 0.0, 0.07), (0.05, 0.05, 0.05))
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, **_TIGHT)
    observables = solver.observables({solver.ObservableKind.STATUS})
    s0, s1 = model.state(), model.state()
    solver.step(s0, s1, None, None, dt, observables=observables)
    observables.status.assign(observables.status.numpy() | int(solver.StatusBits.TRAVERSAL_EXHAUSTED))
    with test.assertRaisesRegex(
        RuntimeError, "a step failed on TRAVERSAL_EXHAUSTED.*A BVH traversal ran out of its node budget"
    ):
        solver.check_status(observables)

    sentinel = np.full((1, 7), 7.0, np.float32)
    s0, s1 = s1, s0
    s1.body_q.assign(sentinel)
    with mock.patch.object(uipc_warp.arrays, "frame_status", exhausted):
        with test.assertRaisesRegex(RuntimeError, "SolverIPC: a BVH traversal ran out of its node budget"):
            solver.step(s0, s1, None, None, dt)
    np.testing.assert_array_equal(s1.body_q.numpy(), sentinel)
    solver.step(s0, s1, None, None, dt)
    fresh = newton.solvers.SolverIPC(model, **_TIGHT)
    s2 = model.state()
    fresh.step(s0, s2, None, None, dt)
    np.testing.assert_allclose(s1.body_q.numpy(), s2.body_q.numpy(), rtol=0.0, atol=1.0e-7)

    model = _worlds_at(device, [0.0, 0.1])
    solver = newton.solvers.SolverIPC(model)
    test.assertTrue(solver._independent)
    s0, s1 = model.state(), model.state()
    s1.body_q.assign(np.full((model.body_count, 7), 7.0, np.float32))
    with mock.patch.object(uipc_warp.arrays, "frame_status", exhausted):
        with test.assertRaisesRegex(RuntimeError, "^SolverIPC: a BVH traversal ran out of its node budget"):
            solver.step(s0, s1, None, None, dt)
    np.testing.assert_array_equal(s1.body_q.numpy(), 7.0)


devices = get_cuda_test_devices()


@requires("uipc_warp")
class TestSolverIPCChecks(unittest.TestCase):
    pass


for name, func in (
    ("test_inertia_checks", test_inertia_checks),
    (
        "test_inertia_that_violates_the_triangle_inequality_is_clamped",
        test_inertia_that_violates_the_triangle_inequality_is_clamped,
    ),
    (
        "test_body_without_a_closed_mesh_takes_the_volume_of_water",
        test_body_without_a_closed_mesh_takes_the_volume_of_water,
    ),
    ("test_line_search_limit_warns", test_line_search_limit_warns),
    ("test_traversal_exhausted_fails_the_step", test_traversal_exhausted_fails_the_step),
):
    add_function_test(TestSolverIPCChecks, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
