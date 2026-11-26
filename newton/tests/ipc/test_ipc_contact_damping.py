# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC's contact damping (``contact_damping``): a viscous force against the motion of surfaces in contact along
their normal, through libuipc-warp's ``ContactDamping``.

* The values it takes: ``None`` gives the engines no extension; a number gives every pair that coefficient;
  ``"material"`` gives every engine vertex and plane the row of its coefficient (its shape's ``shape_material_kd``, a
  particle's ``soft_contact_kd``) in a table of means, so that a pair of two shapes takes the mean of their
  coefficients, a particle and a shape the mean of ``soft_contact_kd`` and the shape's, and two particles
  ``soft_contact_kd``; other values raise ``ValueError``; AL-IPC refuses it; a libuipc-warp without the extension is
  named.
* A rebound: a box dropped flat at 1 m/s onto the ground plane under BDF2 rebounds as the one-dimensional model of a
  mass on four of libuipc's barrier pairs with the damping says (its rebound over the undamped one to 1e-4), and
  ``"material"`` with the box's and the ground's coefficients takes the steps of their mean given as a number, bit for
  bit.
* A box at rest on the ground rests at the height of the undamped one; captured steps equal eager ones; independent
  worlds of different coefficients step as alone; ``CONTACT_F``'s rows carry the damping, so that they sum to the
  box's momentum change; ``notify_model_changed`` builds the solve again for new coefficients of ``"material"``.
"""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.ipc import features
from newton.tests.ipc._fixtures import _TIGHT, _native_contacts, _rigid_rows
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, patch_sys_module, requires

_CONTACT_F = newton.solvers.SolverBase.ObservableKind.CONTACT_F
D_HAT = 1.0e-3
HALF = 0.1  # the box's half extent [m]; of density 1000, 8 kg
MASS = 8.0
KAPPA = 1.0e9  # SolverIPC's default contact stiffness


def _drop(device, kd_box=30.0, kd_ground=50.0, v0=1.0, dt=1.0 / 600.0, gravity=0.0, mu=0.0, worlds=None):
    """A box falling flat at ``v0`` onto the ground plane (z up, ``gravity`` along z) from ``d_hat + 2 v0 dt + 1 um``
    above it, of the contact damping coefficients ``kd_box`` and ``kd_ground`` and the friction ``mu``. With ``worlds`` (a list of the box's
    coefficients) one such box per world, the ground shared. Returns the model."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(kd=kd_ground, mu=mu))
    gap = D_HAT + 2.0 * v0 * dt + 1.0e-6
    for k, kd in enumerate([kd_box] if worlds is None else worlds):
        if worlds is not None:
            builder.begin_world()
        body = builder.add_body(xform=wp.transform(wp.vec3(3.0 * k, 0.0, HALF + gap), wp.quat_identity()))
        builder.add_shape_box(body, hx=HALF, hy=HALF, hz=HALF, cfg=newton.ModelBuilder.ShapeConfig(kd=kd, mu=mu))
        builder.body_qd[body] = wp.spatial_vector(0.0, 0.0, -v0, 0.0, 0.0, 0.0)
        if worlds is not None:
            builder.end_world()
    return builder.finalize(device=device)


def _solver(model, integrator="bdf2", **options):
    return newton.solvers.SolverIPC(model, d_hat=D_HAT, integrator=integrator, **{**_TIGHT, **options})


def _steps(model, solver, steps, dt, capture=False):
    """The states' ``body_q`` and ``body_qd`` after every step (float32), eager, or with ``capture`` the first step
    eager and then a captured pair of steps replayed (an odd count of steps)."""
    s0, s1 = model.state(), model.state()
    q, qd = [], []

    def record(state):
        q.append(state.body_q.numpy().copy())
        qd.append(state.body_qd.numpy().copy())

    if capture:
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        record(s0)
        with wp.ScopedCapture(device=model.device) as cap:
            solver.step(s0, s1, None, None, dt)
            solver.step(s1, s0, None, None, dt)
        for _ in range(1, steps, 2):
            wp.capture_launch(cap.graph)
            record(s1)
            record(s0)
    else:
        for _ in range(steps):
            solver.step(s0, s1, None, None, dt)
            s0, s1 = s1, s0
            record(s0)
    return np.array(q), np.array(qd)


def _rebound(model, solver, v0, dt):
    """The box's rebound speed over ``v0``: its vertical velocity at the first step that ends above ``d_hat`` after one
    that ended inside it."""
    q, qd = _steps(model, solver, 40, dt)
    gap = q[:, 0, 2] - HALF
    inside = np.flatnonzero(gap < D_HAT)
    after = np.flatnonzero((gap > D_HAT) & (np.arange(len(gap)) > inside[0]))
    return float(qd[after[0], 0, 2]) / v0


def _barrier(g, d_hat, kappa):
    """libuipc's barrier ``kappa (D - D_hat)^2 ln(D / D_hat)^2`` at the gap ``g`` (``D = g^2``) and its first two
    derivatives in ``g``; zero at and past ``d_hat``."""
    if g >= d_hat:
        return 0.0, 0.0, 0.0
    D, x1 = g * g, d_hat * d_hat
    t = D - x1
    lg = math.log(D / x1)
    B = kappa * t * t * lg * lg
    dB = 2.0 * kappa * t * lg * lg + 2.0 * kappa * lg * t * t / D
    q = 2.0 * kappa * t * t / (D * D)
    ddB = 8.0 * kappa * lg * t / D + 2.0 * kappa * lg * lg - lg * q + q
    return B, 2.0 * g * dB, 4.0 * D * ddB + 2.0 * dB


def _model_rebound(v0, gap, m, dt, d_hat, kappa, c, pairs):
    """The one-dimensional model of the damping's design under BDF2: a mass ``m`` falling at ``v0`` from ``gap`` onto a
    plane through ``pairs`` equal barrier pairs, each damped with ``c`` in a step that starts inside the barrier's
    range, the history started one step back at the start's velocity, every step minimized by Newton's method with a
    backtracking line search to the last bit. Returns the rebound speed over ``v0``."""
    g, v = gap, -v0
    gm, vm = g - dt * v, v
    touched = False
    for _ in range(10000):
        gt = 4.0 / 3.0 * g - 1.0 / 3.0 * gm + 2.0 / 9.0 * dt * (4.0 * v - vm)
        k = pairs * c * dt if 0.0 < g < d_hat else 0.0

        def energy(y):
            barrier = dt * dt * pairs * _barrier(y, d_hat, kappa)[0]
            return 0.5 * 2.25 * m * (y - gt) ** 2 + barrier + 0.5 * k * (y - g) ** 2  # noqa: B023

        y = g
        for _ in range(500):
            _, b1, b2 = _barrier(y, d_hat, kappa)
            gradient = 2.25 * m * (y - gt) + dt * dt * pairs * b1 + k * (y - g)
            step = -gradient / (2.25 * m + dt * dt * pairs * max(b2, 0.0) + k)
            a = 1.0 if y + step > 0.0 else 0.9 * y / (-step)
            e0 = energy(y)
            while energy(y + a * step) > e0 and a > 1e-14:
                a *= 0.5
            y_new = y + a * step
            converged = abs(y_new - y) < 1e-16
            y = y_new
            if converged:
                break
        v_new = (3.0 * y - 4.0 * g + gm) / (2.0 * dt)
        gm, vm = g, v
        g, v = y, v_new
        touched = touched or g < d_hat
        if touched and g > d_hat and v > 0.0:
            return v / v0
    raise AssertionError("the model's mass did not rebound")


def test_the_values_it_takes(test, device):
    """Take ``None``, a number or ``"material"`` for ``contact_damping`` and raise ``ValueError`` otherwise.

    ``None`` makes no damping and gives an engine no extension; a number gives every pair that coefficient (a table of
    one entry); ``"material"`` a table of the means of the model's coefficients, every engine vertex and plane the row
    of its own. Any other value raises ``ValueError`` naming the choices.
    """
    model = _drop(device)
    solver = _solver(model)
    test.assertIsNone(solver._features.contact_damping)
    inputs = {"extensions": []}
    features.contact_damping.engine_inputs(solver, inputs)
    test.assertEqual(inputs["extensions"], [])
    solver = _solver(model, contact_damping=40.0)
    np.testing.assert_array_equal(solver._features.contact_damping.extension.table.numpy(), [[40.0]])
    solver = _solver(model, contact_damping=np.float32(7.5))
    np.testing.assert_array_equal(solver._features.contact_damping.extension.table.numpy(), [[7.5]])
    solver = _solver(model, contact_damping="material")
    extension = solver._features.contact_damping.extension
    np.testing.assert_array_equal(extension.table.numpy(), [[30.0, 40.0], [40.0, 50.0]])
    test.assertEqual(extension.plane_elements.numpy().tolist(), [1])  # the ground's 50 N s/m
    rows = extension.elements.numpy()
    box_rows = rows[solver._vertex_shape == 1]
    test.assertGreater(len(box_rows), 0)
    test.assertTrue(np.all(box_rows == 0))  # the box's 30 N s/m
    for value in (-1.0, float("nan"), float("inf"), "materials", True, [1.0]):
        with test.subTest(value=value), test.assertRaises(ValueError) as raised:
            _solver(model, contact_damping=value)
        test.assertEqual(
            str(raised.exception),
            "SolverIPC: contact_damping must be None (no damping), 'material' (Newton's contact damping coefficients) "
            f"or a finite coefficient not below 0 [N s/m], got {value!r}.",
        )


def test_material_takes_vbds_means(test, device):
    """Give each pair the mean of its two damping coefficients under ``"material"``.

    A pair of two shapes takes the mean of their ``shape_material_kd``, a particle and a shape the mean of
    ``soft_contact_kd`` and the shape's, two particles ``soft_contact_kd``, a vertex and the ground plane the mean with
    the ground's: the entries of the table at the rows of the engine vertices.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(kd=50.0))
    for k, kd in enumerate((30.0, 70.0)):
        body = builder.add_body(xform=wp.transform(wp.vec3(1.0 * k, 0.0, 0.5), wp.quat_identity()))
        builder.add_shape_box(body, hx=HALF, hy=HALF, hz=HALF, cfg=newton.ModelBuilder.ShapeConfig(kd=kd))
    builder.add_particle(wp.vec3(0.0, 2.0, 0.5), wp.vec3(0.0, 0.0, 0.0), 0.1, radius=0.01)
    model = builder.finalize(device=device)
    model.soft_contact_kd = 10.0
    solver = _solver(model, contact_damping="material")
    extension = solver._features.contact_damping.extension
    table, rows = extension.table.numpy(), extension.elements.numpy()
    np.testing.assert_array_equal(table, table.T)
    shapes = solver._vertex_shape
    plane = int(extension.plane_elements.numpy()[0])
    a, b, p = rows[shapes == 1][0], rows[shapes == 2][0], rows[0]  # box 30, box 70, the particle (engine row 0)
    test.assertEqual(table[a, b], 50.0)
    test.assertEqual(table[p, a], 20.0)
    test.assertEqual(table[p, b], 40.0)
    test.assertEqual(table[p, p], 10.0)
    test.assertEqual(table[a, plane], 40.0)
    test.assertEqual(table[p, plane], 30.0)


def test_a_rebound_as_the_model(test, device):
    """Rebound a damped box dropped onto the ground as the one-dimensional model predicts.

    A box dropped flat at 1 m/s onto the ground plane under BDF2 in steps of 1/600 s rebounds at more than half its
    speed. With 40 N s/m on each of its four corner pairs its rebound over the undamped one is the model's to 1e-3 (the
    model's mass is rigid, the affine body deforms a little in the impact; 10 N s/m more per pair lowers the ratio by
    9e-3), and ``"material"`` with the box's 30 N s/m and the ground's 50 takes the same steps bit for bit.
    """
    dt, v0 = 1.0 / 600.0, 1.0
    model = _drop(device)
    e0 = _rebound(model, _solver(model), v0, dt)
    test.assertGreater(e0, 0.5)
    q_number, qd_number = _steps(model, _solver(model, contact_damping=40.0), 12, dt)
    q_material, qd_material = _steps(model, _solver(model, contact_damping="material"), 12, dt)
    np.testing.assert_array_equal(q_material, q_number)
    np.testing.assert_array_equal(qd_material, qd_number)
    e = _rebound(model, _solver(model, contact_damping=40.0), v0, dt)
    gap = D_HAT + 2.0 * v0 * dt + 1.0e-6
    m0 = _model_rebound(v0, gap, MASS, dt, D_HAT, KAPPA, 0.0, 4)
    m40 = _model_rebound(v0, gap, MASS, dt, D_HAT, KAPPA, 40.0, 4)
    test.assertLess(e, e0)
    test.assertAlmostEqual(e / e0, m40 / m0, delta=1e-3)


def test_a_box_at_rest_rests_where_it_rests_without(test, device):
    """Rest a box on the ground with contact damping at the height of the undamped box.

    The box is set on the ground under gravity, implicit Euler in steps of 10 ms. With Newton's default coefficients
    (``"material"``, 100 N s/m on every shape) it settles at the height of the undamped box, to 1e-6 m, and at rest.
    """
    heights = []
    for damping in (None, "material"):
        model = _drop(device, kd_box=100.0, kd_ground=100.0, v0=0.0, dt=0.01, gravity=-9.81, mu=0.5)
        q, qd = _steps(model, _solver(model, integrator="bdf1", contact_damping=damping), 100, 0.01)
        test.assertLess(np.abs(qd[-1]).max(), 1e-5)
        heights.append(float(q[-1, 0, 2]))
    test.assertAlmostEqual(heights[1], heights[0], delta=1e-6)


def test_captured_steps_equal_eager_ones(test, device):
    """Reach the same states in captured and in eager steps of the drop with ``"material"``."""
    runs = []
    for capture in (False, True):
        model = _drop(device)
        runs.append(_steps(model, _solver(model, contact_damping="material"), 13, 1.0 / 600.0, capture=capture))
    np.testing.assert_array_equal(runs[1][0], runs[0][0])
    np.testing.assert_array_equal(runs[1][1], runs[0][1])


def test_independent_worlds_as_alone(test, device):
    """Step each independent world of a damped box as that world alone.

    Two worlds, a box of 30 N s/m and one of 90 N s/m, are dropped on the shared ground: each takes the steps of its
    world alone (float32), and the more damped box rebounds slower.
    """
    dt = 1.0 / 600.0
    model = _drop(device, worlds=[30.0, 90.0])
    solver = _solver(model, contact_damping="material")
    test.assertTrue(solver._independent)
    q, qd = _steps(model, solver, 12, dt)
    for k, kd in enumerate((30.0, 90.0)):
        alone = _drop(device, kd_box=kd)
        q_a, qd_a = _steps(alone, _solver(alone, contact_damping="material"), 12, dt)
        np.testing.assert_allclose(q[:, k, 2], q_a[:, 0, 2], rtol=0.0, atol=1e-6, err_msg=f"world {k}")
        np.testing.assert_allclose(qd[:, k], qd_a[:, 0], rtol=0.0, atol=1e-5, err_msg=f"world {k}")
    test.assertLess(float(qd[-1, 1, 2]), float(qd[-1, 0, 2]))


def test_contact_rows_carry_the_damping(test, device):
    """Carry the damping in ``CONTACT_F``'s rows so that they sum to the box's momentum change and weight.

    With friction off and 200 N s/m, at every step of the landing the rows' forces on the box sum to its momentum change
    and its weight, ``m (v - v_prev) / dt - m g``, the damping in the rows of the friction pairs.
    """
    dt = 0.002
    model = _drop(device, kd_box=200.0, kd_ground=200.0, v0=0.5, dt=dt, gravity=-9.81)
    solver = _solver(model, integrator="bdf1", contact_damping="material")
    contacts = _native_contacts(solver, model)
    observables = solver.observables({_CONTACT_F}, contacts=contacts)
    s0, s1 = model.state(), model.state()
    damped = 0
    for step in range(10):
        v_prev = s0.body_qd.numpy()[0, :3].astype(np.float64)
        solver.step(s0, s1, None, contacts, dt, observables=observables)
        s0, s1 = s1, s0
        v = s0.body_qd.numpy()[0, :3].astype(np.float64)
        shape0, _, f = _rigid_rows(contacts, observables)
        if len(shape0) == 0:
            continue
        on_box = np.where(shape0[:, None] == 1, 1.0, -1.0) * f[:, :3]
        balance = MASS * (v - v_prev) / dt - MASS * np.array([0.0, 0.0, -9.81])
        np.testing.assert_allclose(on_box.sum(axis=0), balance, rtol=0.0, atol=0.05, err_msg=f"step {step}")
        damped += int(len(shape0) > 4)  # barrier rows and the damping's rows of the same corner pairs
    test.assertGreater(damped, 1)


def test_notify_builds_the_solve_again_for_new_coefficients(test, device):
    """Build the solve again when ``SHAPE_PROPERTIES`` notifies new coefficients under ``"material"``.

    With ``"material"``, a new ``shape_material_kd`` notified with ``SHAPE_PROPERTIES`` builds the solve again with the
    new means. The same coefficients, or other flags, keep the solve; a number reads no coefficient.
    """
    model = _drop(device)
    solver = _solver(model, contact_damping="material")
    engine = solver._engine
    solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
    test.assertIs(solver._engine, engine)
    kd = model.shape_material_kd.numpy()
    kd[1] = 70.0
    model.shape_material_kd.assign(kd)
    solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
    test.assertIs(solver._engine, engine)
    solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
    test.assertIsNot(solver._engine, engine)
    np.testing.assert_array_equal(
        solver._features.contact_damping.extension.table.numpy(), [[50.0, 60.0], [60.0, 70.0]]
    )
    number = _solver(model, contact_damping=5.0)
    engine = number._engine
    kd[1] = 20.0
    model.shape_material_kd.assign(kd)
    number.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
    test.assertIs(number._engine, engine)


def test_al_ipc_refuses_it(test, device):
    """Refuse any contact damping but None under ``contact_constitution="al-ipc"``.

    Any value but None raises ``NotImplementedError``: AL-IPC's contact terms are an active set with multipliers, which
    the damping does not act on.
    """
    model = _drop(device)
    with test.assertRaises(NotImplementedError) as raised:
        newton.solvers.SolverIPC(model, contact_constitution="al-ipc", contact_damping="material")
    test.assertEqual(
        str(raised.exception),
        "SolverIPC: contact_damping='material' is not offered with contact_constitution='al-ipc': AL-IPC's contact "
        "terms are an active set with multipliers, which libuipc-warp's contact damping does not act on. Use "
        "contact_constitution='ipc', or leave contact_damping at None.",
    )


def test_a_libuipc_warp_without_the_extension_is_named(test, device):
    """Raise ``ImportError`` naming ``uipc_warp.ext.contact_damping`` when libuipc-warp lacks it.

    With a libuipc-warp that has no ``uipc_warp.ext.contact_damping`` (one before it), a contact damping raises
    ``ImportError`` naming the module when the solver is built.
    """
    model = _drop(device)
    with patch_sys_module("uipc_warp.ext.contact_damping", None), test.assertRaises(ImportError) as raised:
        newton.solvers.SolverIPC(model, contact_damping=1.0)
    test.assertIn("needs libuipc-warp's uipc_warp.ext.contact_damping", str(raised.exception))


@requires("uipc_warp")
class TestSolverIPCContactDamping(unittest.TestCase):
    """A damping of the normal motion of surfaces in contact."""


devices = get_cuda_test_devices()
for name, func in (
    ("test_the_values_it_takes", test_the_values_it_takes),
    ("test_material_takes_vbds_means", test_material_takes_vbds_means),
    ("test_a_rebound_as_the_model", test_a_rebound_as_the_model),
    ("test_a_box_at_rest_rests_where_it_rests_without", test_a_box_at_rest_rests_where_it_rests_without),
    ("test_captured_steps_equal_eager_ones", test_captured_steps_equal_eager_ones),
    ("test_independent_worlds_as_alone", test_independent_worlds_as_alone),
    ("test_contact_rows_carry_the_damping", test_contact_rows_carry_the_damping),
    (
        "test_notify_builds_the_solve_again_for_new_coefficients",
        test_notify_builds_the_solve_again_for_new_coefficients,
    ),
    ("test_al_ipc_refuses_it", test_al_ipc_refuses_it),
    ("test_a_libuipc_warp_without_the_extension_is_named", test_a_libuipc_warp_without_the_extension_is_named),
):
    add_function_test(TestSolverIPCContactDamping, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
