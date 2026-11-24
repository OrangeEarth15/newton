# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC's kinematic_motion="aim": kinematic bodies moved inside the step toward their aims."""

import unittest
import warnings
from unittest import mock

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
    40 steps the box moves within eps_velocity of the platform's speed. With the jump friction never sees the
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
    """kinematic_motion="jump" builds a kinematic body as before: fixed, without a soft transform constraint, with the
    contact elements of before, its state_out a copy of state_in; "aim", which the default picks for this model,
    builds it free, with the constraint and the kinematic and static shapes in elements of their own. Invalid options
    and AL-IPC with an explicit "aim" are refused; AL-IPC with the default takes the jump."""
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
    with test.assertRaisesRegex(NotImplementedError, "kinematic_motion='aim' would move the kinematic body"):
        newton.solvers.SolverIPC(model, contact_constitution="al-ipc", kinematic_motion="aim")
    test.assertEqual(newton.solvers.SolverIPC(model, contact_constitution="al-ipc").kinematic_motion, "jump")
    test.assertEqual(newton.solvers.SolverIPC(model).kinematic_motion, "aim")


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


def _anchored_pendulum(device, sheet=False):
    """basic_joints' revolute pendulum: an anchor link of density 0 (mass 0, so kinematic) fixed to the world at
    z = 1, and a box of 1000 kg/m^3 hanging from it on a revolute joint about x, released 0.6 rad off the vertical;
    with ``sheet`` also a kinematic body of mass 0 whose only shape is a single triangle, which encloses no volume.
    Returns the model and the pendulum's body."""
    builder = newton.ModelBuilder()
    static = newton.ModelBuilder.ShapeConfig(density=0.0)
    anchor = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
    builder.add_shape_box(anchor, hx=0.05, hy=0.05, hz=0.05, cfg=static)
    bob = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.7), wp.quat_identity()))
    builder.add_shape_box(bob, hx=0.03, hy=0.03, hz=0.2)
    fixed = builder.add_joint_fixed(-1, anchor, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
    hinge = builder.add_joint_revolute(
        anchor,
        bob,
        axis=wp.vec3(1.0, 0.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.25), wp.quat_identity()),
    )
    builder.add_articulation([fixed, hinge])
    builder.joint_q[-1] = 0.6
    if sheet:
        b = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()), mass=0.0)
        v = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], np.float32)
        builder.add_shape_mesh(b, mesh=newton.Mesh(v, np.array([0, 1, 2], np.int32)), cfg=static)
    return builder.finalize(device=device), bob


def test_massless_kinematic_bodies_keep_the_jump(test, device):
    """A kinematic body of mass 0 keeps the jump under kinematic_motion="aim": there is no mass for the soft transform
    constraint to move, and a soft anchor lets the joints it holds sag. basic_joints' revolute pendulum, whose anchor
    is a link of density 0: under "aim" the anchor is fixed, as under "jump", and the pendulum swings in its hinge's
    plane and as it swings under "jump" over 60 steps, to 1e-9 m. A kinematic body of mass 0 whose shapes enclose no
    volume (a single triangle) is taken too, without a RuntimeWarning: the solver computes no mass from its shapes."""
    model, bob = _anchored_pendulum(device)
    paths = {}
    for mode in ("jump", "aim"):
        solver = newton.solvers.SolverIPC(model, kinematic_motion=mode, **_TIGHT)
        test.assertEqual(len(solver._aim_bodies), 0)
        layout = solver._scene.layout()
        test.assertTrue(np.all(solver._engine.vm.fixed.numpy()[layout.body_base[0] : layout.body_base[0] + 4] != 0))
        s0, s1 = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        path = []
        for _ in range(60):
            solver.step(s0, s1, None, None, 1.0 / 240.0)
            s0, s1 = s1, s0
            path.append(s0.body_q.numpy()[bob, :3].copy())
        paths[mode] = np.array(path)
    test.assertLess(np.abs(paths["aim"][:, 0]).max(), 1e-5)  # in the hinge's plane, basic_joints' check
    test.assertGreater(np.ptp(paths["aim"][:, 1]), 0.05)  # it swings
    np.testing.assert_allclose(paths["aim"], paths["jump"], rtol=0.0, atol=1e-9)
    sheet_model, _ = _anchored_pendulum(device, sheet=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = newton.solvers.SolverIPC(sheet_model, kinematic_motion="aim")
    test.assertEqual([str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)], [])
    test.assertEqual(len(solver._aim_bodies), 0)


def test_rekinematic_body_drops_its_twist(test, device):
    """Newton's test_kinematic_runtime_toggle under kinematic_motion="aim": a kinematic sphere, made dynamic and pushed
    by 10 N for 60 steps of 1/240 s, then made kinematic again (notify_model_changed(BODY_PROPERTIES)). Its state_in
    still holds the twist it gained as a dynamic body, which the solver wrote; the body does not take it, and stays
    where it is over the next 60 steps, to 1e-6 m (it moved on at that twist, 0.12 m). A caller who writes another
    twist at the second toggle moves the body at it from the first step on."""

    def run(prescribed):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_body(mass=1.0, is_kinematic=True)
        builder.add_shape_sphere(body, radius=0.1)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverIPC(model, kinematic_motion="aim")
        s0, s1 = model.state(), model.state()

        def phase(steps):
            nonlocal s0, s1
            for _ in range(steps):
                s0.clear_forces()
                s0.body_f.assign(np.array([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0]], np.float32))
                solver.step(s0, s1, None, None, 1.0 / 240.0)
                s0, s1 = s1, s0

        def toggle(flag):
            flags = model.body_flags.numpy()
            flags[body] = int(flag)
            model.body_flags.assign(flags)
            solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)

        phase(60)
        test.assertLess(np.linalg.norm(s0.body_q.numpy()[body, :3]), 1e-6)  # kinematic: forces do not move it
        toggle(newton.BodyFlags.DYNAMIC)
        phase(60)
        start = s0.body_q.numpy()[body, :3].copy()
        test.assertGreater(start[0], 0.05)
        test.assertGreater(float(s0.body_qd.numpy()[body, 0]), 0.4)  # the twist it gained
        toggle(newton.BodyFlags.KINEMATIC)
        if prescribed:
            qd = s0.body_qd.numpy()
            qd[body] = (0.2, 0.0, 0.0, 0.0, 0.0, 0.0)
            s0.body_qd.assign(qd)
        phase(60)
        return s0.body_q.numpy()[body, :3] - start

    np.testing.assert_allclose(run(False), 0.0, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(run(True), [0.2 * 60 / 240.0, 0.0, 0.0], rtol=0.0, atol=1e-4)


def test_al_ipc_takes_aims_without_aiming_bodies(test, device):
    """contact_constitution="al-ipc" refuses kinematic_motion="aim" only for a model with a body that would move toward
    its aims: two boxes dropped onto the ground, without a kinematic body, step under "aim" as under "jump" (to 1e-9 m
    over 30 steps); a kinematic box of mass 1 kg with a collision shape is refused, naming it."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    _box_body(builder, (0.0, 0.0, 0.06), (0.05, 0.05, 0.05))
    _box_body(builder, (0.0, 0.0, 0.17), (0.05, 0.05, 0.05))
    model = builder.finalize(device=device)
    paths = {}
    for mode in ("jump", "aim"):
        solver = newton.solvers.SolverIPC(model, d_hat=0.005, contact_constitution="al-ipc", kinematic_motion=mode)
        s0, s1 = model.state(), model.state()
        path = []
        for _ in range(30):
            solver.step(s0, s1, None, None, 0.01)
            s0, s1 = s1, s0
            path.append(s0.body_q.numpy()[:, :3].copy())
        paths[mode] = np.array(path)
    np.testing.assert_allclose(paths["aim"], paths["jump"], rtol=0.0, atol=1e-9)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    kinematic = builder.add_body(mass=1.0, is_kinematic=True)
    builder.add_shape_box(kinematic, hx=0.1, hy=0.1, hz=0.1)
    with test.assertRaisesRegex(NotImplementedError, f"move the kinematic body {kinematic} toward its aims"):
        newton.solvers.SolverIPC(builder.finalize(device=device), contact_constitution="al-ipc", kinematic_motion="aim")


def _lift(device, mode, steps=90, capture=False):
    """R10's lift: a 0.2 m box of 8 kg resting 2 mm above a kinematic platform (1 x 1 x 0.1 m, friction 0.5) that after
    60 steps of 1/120 s rises at 0.5 m/s, its pose and twist written into state_in before every step. Returns the gap
    between the box's bottom and the platform's top at the end of every step, and the RuntimeError a step raised (an
    eager step, or ``check_status`` after a captured one), or None."""
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.mu = 0.5
    platform = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()), is_kinematic=True)
    builder.add_shape_box(platform, hx=0.5, hy=0.5, hz=0.05)
    box = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.652), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, kinematic_motion=mode)
    observables = solver.observables({solver.ObservableKind.STATUS})
    dt, s0, s1 = 1.0 / 120.0, model.state(), model.state()
    gaps, graph = [], None
    for k in range(steps):
        rising = k >= 60
        q, qd = s0.body_q.numpy(), s0.body_qd.numpy()
        q[platform, 2] = 0.5 + 0.5 * dt * max(k - 59, 0)
        qd[platform] = (0.0, 0.0, 0.5 if rising else 0.0, 0.0, 0.0, 0.0)
        s0.body_q.assign(q)
        s0.body_qd.assign(qd)
        try:
            if not capture or k == 0:
                solver.step(s0, s1, None, None, dt, observables=observables)
            else:
                if graph is None:
                    with wp.ScopedCapture(device=device) as cap:
                        solver.step(s0, s1, None, None, dt, observables=observables)
                    graph = cap.graph
                wp.capture_launch(graph)
                solver.check_status(observables)
        except RuntimeError as e:
            return np.array(gaps), e
        s0.assign(s1)
        q = s0.body_q.numpy()
        gaps.append(float(q[box, 2] - 0.1 - (q[platform, 2] + 0.05)))
    return np.array(gaps), None


def test_jump_through_a_surface_fails_the_step(test, device):
    """R10-1: a kinematic body's jump is swept at the step's start. In R10's lift, where the platform's jumps of
    4.2 mm outrun the barrier and, unswept, took the box through the platform at the eighth jump with no status bit,
    the step whose jump meets the box fails with INVALID_CCD_STEP, named as the kinematic sweep: an eager step raises,
    a captured one sets the bit and check_status raises; no step before it ends with the box inside the platform. With
    kinematic_motion="aim" the platform's motion is part of the step, and the box rides on it through every step."""
    for capture in (False, True):
        with test.subTest(capture=capture):
            gaps, raised = _lift(device, "jump", capture=capture)
            test.assertIsNotNone(raised, "a jump through the box was taken")
            test.assertIn("INVALID_CCD_STEP", str(raised))
            test.assertIn("the kinematic sweep", str(raised))
            test.assertGreaterEqual(len(gaps), 60)  # while the platform rests, nothing fails
            test.assertGreater(gaps.min(), 0.0)
    gaps, raised = _lift(device, "aim")
    test.assertIsNone(raised)
    test.assertGreater(gaps.min(), 0.0)
    test.assertLess(gaps[-1], 0.011)  # on the platform, within d_hat


def test_import_requires_the_kinematic_sweep(test, device):
    """The sweep of the jumps is libuipc-warp's (``EngineOptions(sweep_kinematic=)``), a capability that a development
    build of the release SolverIPC asks for may lack: the import refuses a build without it, naming it."""
    import uipc_warp  # noqa: PLC0415

    imported = newton.solvers.SolverIPC._uipc_warp
    without = tuple(name for name in uipc_warp.EngineOptions.names() if name != "sweep_kinematic")
    try:
        newton.solvers.SolverIPC._uipc_warp = None
        with mock.patch.object(uipc_warp.EngineOptions, "names", return_value=without):
            with test.assertRaisesRegex(ImportError, r"lacks EngineOptions\(sweep_kinematic=\)"):
                newton.solvers.SolverIPC._import_uipc_warp()
    finally:
        newton.solvers.SolverIPC._uipc_warp = imported


def test_model_without_contact_asks_for_no_sweep(test, device):
    """The barrier's sweep of kinematic moves needs a contact surface to sweep against, and SolverIPC asks libuipc-warp
    for it only where the engine has one: a hinged box whose shape collides with nothing builds and steps without a
    warning, where libuipc-warp warned that the option had no effect; the same box whose shape collides builds and steps
    without one too, with the sweep that test_jump_through_a_surface_fails_the_step holds to its work."""
    for collide in (False, True):
        builder = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=collide, has_particle_collision=collide)
        pose = wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity())
        link = builder.add_link(xform=pose)
        builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
        builder.add_articulation([builder.add_joint_revolute(-1, link, parent_xform=pose)])
        model = builder.finalize(device=device)
        with test.subTest(collide=collide):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                solver = newton.solvers.SolverIPC(model)
                s0, s1 = model.state(), model.state()
                solver.step(s0, s1, None, None, 0.01)
            test.assertEqual([f"{w.category.__name__}: {w.message}" for w in caught], [])


def test_refused_first_step_takes_state_in_again(test, device):
    """With kinematic_motion="aim" a first step that the start check refuses, its state_in putting a kinematic box
    through a body, leaves the next step the first: the step again from the model's state, where the box is clear,
    takes the box's pose from its state_in and runs. The refused step held the box at the pose it had loaded, and the
    step again failed on it."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    _box_body(builder, (0.0, 0.0, 1.0), (0.1, 0.1, 0.1))
    pusher = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 1.0), wp.quat_identity()), is_kinematic=True)
    builder.add_shape_box(pusher, hx=0.1, hy=0.1, hz=0.1)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, kinematic_motion="aim")
    np.testing.assert_array_equal(solver._aim_bodies, [pusher])
    through = model.state()
    q = through.body_q.numpy()
    q[pusher, 0] = 0.05
    through.body_q.assign(q)
    with test.assertRaisesRegex(ValueError, r"body 0 and body 1\b.*intersect in the state of the first step"):
        solver.step(through, model.state(), None, None, 0.01)
    s1 = model.state()
    solver.step(model.state(), s1, None, None, 0.01)
    test.assertEqual(solver._engine.frames, 1)
    np.testing.assert_allclose(s1.body_q.numpy()[pusher, :3], [1.0, 0.0, 1.0], rtol=0.0, atol=1e-6)


def test_aims_pass_the_twist_through(test, device):
    """With kinematic_motion="aim" state_out holds the pose a kinematic box reached and the twist of state_in, as
    Newton's solvers pass a kinematic body's twist through. A caller who writes only body_q, 2 mm further at every
    step, gets the box at that pose (to 1e-6 m) with the twist 0 it never wrote; one who writes a twist of 0.2 m/s
    once gets motion at that twist. The box's twist in state_out was the one it reached (0.2 m/s), which the next
    step took as the caller's: the poses written alone ran a step ahead and swung about their path."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    box = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), is_kinematic=True)
    builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05)
    model = builder.finalize(device=device)
    dt = 0.01
    for written in ("poses", "twist"):
        solver = newton.solvers.SolverIPC(model, kinematic_motion="aim")
        np.testing.assert_array_equal(solver._aim_bodies, [box])
        s0, s1 = model.state(), model.state()
        twist = np.zeros(6, np.float32)
        if written == "twist":
            twist[0] = 0.2
            qd = s0.body_qd.numpy()
            qd[box] = twist
            s0.body_qd.assign(qd)
        for k in range(1, 11):
            if written == "poses":
                q = s0.body_q.numpy()
                q[box, 0] = 0.002 * k
                s0.body_q.assign(q)
            solver.step(s0, s1, None, None, dt)
            s0, s1 = s1, s0
            with test.subTest(written=written, step=k):
                np.testing.assert_allclose(s0.body_q.numpy()[box, :3], [0.002 * k, 0.0, 1.0], rtol=0.0, atol=1e-6)
                np.testing.assert_array_equal(s0.body_qd.numpy()[box], twist)


def _kinematic_box(builder, x):
    """A kinematic box of 1 kg, inertia 1e-3 kg m^2 about every axis, with a colliding shape of 0.1 m at ``x``."""
    body = builder.add_body(
        xform=wp.transform(wp.vec3(*x), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3) * 1.0e-3),
        is_kinematic=True,
    )
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    return body


def _fallback_models(device):
    """The three models for which kinematic_motion=None takes the jump, with the explicit "aim" refusal of each: a
    kinematic box under contact_constitution="al-ipc"; a kinematic box of the global world beside two worlds with a
    box each (independent worlds); a kinematic box whose inertia has a zero principal moment."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    _box_body(builder, (0.0, 0.0, 0.0), (0.05, 0.05, 0.05))
    _kinematic_box(builder, (0.5, 0.0, 0.0))
    al = (builder.finalize(device=device), {"contact_constitution": "al-ipc"}, NotImplementedError, "toward its aims")

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for w in range(2):
        builder.begin_world()
        _box_body(builder, (0.0, float(w), 0.0), (0.05, 0.05, 0.05))
        builder.end_world()
    _kinematic_box(builder, (0.5, 3.0, 0.0))  # after the worlds: the global world
    worlds = (builder.finalize(device=device), {}, NotImplementedError, "global world -1 is a kinematic body")

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    _box_body(builder, (0.0, 0.0, 0.0), (0.05, 0.05, 0.05))
    flat = _kinematic_box(builder, (0.5, 0.0, 0.0))
    model = builder.finalize(device=device)
    inertia = model.body_inertia.numpy()
    inertia[flat] = np.diag([1.0e-3, 1.0e-3, 0.0])
    model.body_inertia.assign(inertia)
    singular = (model, {}, ValueError, "positive-definite inertia for the kinematic body")
    return {"al-ipc": al, "global body": worlds, "singular inertia": singular}


def test_default_picks_the_mode_per_solver(test, device):
    """kinematic_motion=None picks "aim" on the barrier pipeline when every kinematic body that would aim can, and
    "jump" otherwise: under AL-IPC, for such a body of the global world with independent worlds, and for one with a
    singular inertia. The fallback warns of nothing and logs its choice at the debug level; solver.kinematic_motion
    gives the mode (read-only). An explicit "aim" is refused in each fallback case; an explicit "jump" stays the jump,
    and so does a model that would aim. A kinematic_strength_ratio other than the default is refused where the
    default takes the jump."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    _box_body(builder, (0.0, 0.0, 0.0), (0.05, 0.05, 0.05))
    kinematic = _kinematic_box(builder, (0.5, 0.0, 0.0))
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model)
    test.assertEqual(solver.kinematic_motion, "aim")
    np.testing.assert_array_equal(solver._aim_bodies, [kinematic])
    with test.assertRaises(AttributeError):
        solver.kinematic_motion = "jump"
    jump = newton.solvers.SolverIPC(model, kinematic_motion="jump")
    test.assertEqual(jump.kinematic_motion, "jump")
    test.assertEqual(len(jump._aim_bodies), 0)
    for name, (model, options, error, pattern) in _fallback_models(device).items():
        with test.subTest(name):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with test.assertLogs("newton._src.solvers.ipc.solver_ipc", "DEBUG") as logs:
                    solver = newton.solvers.SolverIPC(model, **options)
            test.assertEqual(solver.kinematic_motion, "jump")
            test.assertEqual(len(solver._aim_bodies), 0)
            test.assertEqual(len(logs.records), 1)
            test.assertIn("kinematic_motion=None takes 'jump'", logs.output[0])
            s0, s1 = model.state(), model.state()
            solver.step(s0, s1, None, None, 0.01)
            with test.assertRaisesRegex(error, pattern):
                newton.solvers.SolverIPC(model, kinematic_motion="aim", **options)
            with test.assertRaisesRegex(ValueError, "the default kinematic_motion=None does not take here"):
                newton.solvers.SolverIPC(model, kinematic_strength_ratio=50.0, **options)


@requires("uipc_warp")
class TestSolverIPCKinematicMotion(unittest.TestCase):
    """Which kinematic bodies move toward their aims, the twist of a body that becomes kinematic, and the sweep of the
    jumps."""


for name, func in (
    ("test_massless_kinematic_bodies_keep_the_jump", test_massless_kinematic_bodies_keep_the_jump),
    ("test_rekinematic_body_drops_its_twist", test_rekinematic_body_drops_its_twist),
    ("test_al_ipc_takes_aims_without_aiming_bodies", test_al_ipc_takes_aims_without_aiming_bodies),
    ("test_jump_through_a_surface_fails_the_step", test_jump_through_a_surface_fails_the_step),
    ("test_import_requires_the_kinematic_sweep", test_import_requires_the_kinematic_sweep),
    ("test_model_without_contact_asks_for_no_sweep", test_model_without_contact_asks_for_no_sweep),
    ("test_refused_first_step_takes_state_in_again", test_refused_first_step_takes_state_in_again),
    ("test_aims_pass_the_twist_through", test_aims_pass_the_twist_through),
    ("test_default_picks_the_mode_per_solver", test_default_picks_the_mode_per_solver),
):
    add_function_test(TestSolverIPCKinematicMotion, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
