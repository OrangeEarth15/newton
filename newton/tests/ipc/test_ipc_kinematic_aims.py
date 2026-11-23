# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC's kinematic_motion="aim": kinematic bodies moved inside the step toward their aims."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.ipc._fixtures import _TIGHT, _TINY_LISTS, _active_slots, _box_body
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires


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
    test.assertGreater(_active_slots(solver).max(), 1)
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
    from newton._src.solvers.ipc.contacts import _contact_elements  # noqa: PLC0415

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


@requires("uipc_warp")
class TestSolverIPCKinematicAims(unittest.TestCase):
    """kinematic_motion="aim": kinematic bodies moved inside the step toward their aims by libuipc's soft transform
    constraint, swept by the CCD."""


devices = get_cuda_test_devices()
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


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
