# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC's universal joints (D6 joints of two angular axes) and distance joints with a range.

A universal joint is libuipc-warp's ``UniversalJoints``: the anchors held, the rotation about ``axis_0 x axis_1``
locked, and per angle the terms of its degree of freedom. The tests hold it to closed forms and to Newton's own
coordinates: two shafts joined by it follow the Cardan formula; ``joint_q`` is ``eval_ik``'s angles, multi-turn; a
drive reaches its targets and takes an angle through three turns; a limit rests a load at the cubic's overshoot,
friction holds a smaller torque, ``joint_f`` accelerates at ``tau / I``; ``BODY_PARENT_F`` is the momentum balance;
captured steps equal eager ones. A distance joint with a range is libuipc-warp's ``RangeStitches``: free inside,
resting past the bound at the cubic's overshoot ``sqrt(F delta_r / limit_ke)``, with one bound too; a change of
its bounds builds the solve again where the stitch changes kind. And the refusals that remain."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.ipc._fixtures import _TIGHT_ROTATIONS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires

D = newton.ModelBuilder.JointDofConfig
_KIND = newton.solvers.SolverBase.ObservableKind
_G = 9.81
_DELTA_R = 1.0e-3  # SolverIPC's overshoot of a limit at which it pushes back with joint_limit_ke times it


def _quat(axis, angle):
    return wp.quat_from_axis_angle(wp.vec3(*axis), float(angle))


def _box_link(builder, half=(0.1, 0.1, 0.1), xform=None):
    link = builder.add_link(xform=xform or wp.transform_identity())
    builder.add_shape_box(link, hx=half[0], hy=half[1], hz=half[2])
    return link


def _pendulum(device, gravity=_G, second=None, **dof):
    """A 0.2 m box (8 kg) 0.5 m below a universal joint at the world's origin, z up: first axis x, second y, the
    joint's first degree of freedom given ``dof`` and its second ``second`` (``dof`` where None)."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -gravity), up_axis=newton.Axis.Z)
    link = _box_link(builder, xform=wp.transform(wp.vec3(0.0, 0.0, -0.5), wp.quat_identity()))
    j = builder.add_joint_d6(
        -1, link, child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        angular_axes=[D(axis=newton.Axis.X, **dof), D(axis=newton.Axis.Y, **(dof if second is None else second))],
    )  # fmt: skip
    builder.add_articulation([j])
    return builder.finalize(device=device)


def _state(model, q=None, qd=None):
    """A state of ``model`` at the joint coordinates ``q`` and rates ``qd`` (the model's where None)."""
    s = model.state()
    if q is not None:
        s.joint_q.assign(np.asarray(q, np.float32))
    if qd is not None:
        s.joint_qd.assign(np.asarray(qd, np.float32))
    newton.eval_fk(model, s.joint_q, s.joint_qd, s)
    return s


def _run(model, solver, steps, dt, state=None, control=None, observables=None, each=None):
    """``steps`` steps from ``state``; ``each(k, state_out)`` after every step. Returns the last state."""
    s0 = state if state is not None else _state(model)
    s1 = model.state()
    for k in range(steps):
        solver.step(s0, s1, control, None, dt, observables=observables)
        if each is not None:
            each(k, s1)
        s0, s1 = s1, s0
    return s0


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def test_joint_q_is_eval_iks(test, device):
    """A box swinging on a universal joint about both axes: after every step ``joint_q`` is the angles ``eval_ik``
    computes from the bodies' poses, to the joint's violation of its lock, and ``joint_qd`` is ``eval_ik``'s
    rates."""
    model = _pendulum(device)
    solver = newton.solvers.SolverIPC(model)
    q_ik, qd_ik = wp.zeros_like(model.joint_q), wp.zeros_like(model.joint_qd)
    seen = []

    def check(k, s):
        newton.eval_ik(model, s, q_ik, qd_ik)
        q, ik = s.joint_q.numpy(), q_ik.numpy()
        np.testing.assert_allclose(_wrap(q - ik), 0.0, atol=2e-4, err_msg=f"step {k}")
        np.testing.assert_allclose(s.joint_qd.numpy(), qd_ik.numpy(), atol=1e-4, err_msg=f"step {k}")
        seen.append(q.copy())

    _run(model, solver, 80, 0.01, state=_state(model, [0.6, -0.3], [1.0, 2.0]), each=check)
    test.assertGreater(np.ptp(np.array(seen), axis=0).min(), 0.5)  # both angles swung


def _cardan(device, beta):
    """Two shafts (0.5 m x 4 cm x 4 cm) on revolute joints to the world, the input along x, the output along
    ``(cos beta, sin beta, 0)``, both through the origin, joined there by a universal joint whose first axis is z on
    the input and whose second is ``z x`` the output's axis on the output; no gravity. The input's joint has a velocity
    drive. Returns the model and the two revolute joints."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    bend = _quat((0.0, 0.0, 1.0), beta)
    half = (0.25, 0.02, 0.02)
    shaft_in = _box_link(builder, half, wp.transform(wp.vec3(-0.3, 0.0, 0.0), wp.quat_identity()))
    s2 = np.array([np.cos(beta), np.sin(beta), 0.0])
    shaft_out = _box_link(builder, half, wp.transform(wp.vec3(*(0.3 * s2)), bend))
    j_in = builder.add_joint_revolute(
        -1, shaft_in, child_xform=wp.transform(wp.vec3(0.3, 0.0, 0.0), wp.quat_identity()), axis=newton.Axis.X,
        target_kd=1.0,
    )  # fmt: skip
    j_out = builder.add_joint_revolute(
        -1, shaft_out, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), bend),
        child_xform=wp.transform(wp.vec3(-0.3, 0.0, 0.0), wp.quat_identity()), axis=newton.Axis.X,
    )  # fmt: skip
    builder.add_articulation([j_in])
    builder.add_articulation([j_out])
    # the joint frame on the input turned by beta about z: its z is the input's z, its y z x the output's axis
    builder.add_joint_d6(
        shaft_in, shaft_out, parent_xform=wp.transform(wp.vec3(0.3, 0.0, 0.0), bend),
        child_xform=wp.transform(wp.vec3(-0.3, 0.0, 0.0), wp.quat_identity()),
        angular_axes=[D(axis=newton.Axis.Z), D(axis=newton.Axis.Y)],
    )  # fmt: skip
    return builder.finalize(device=device), (j_in, j_out)


def test_cardan_formula(test, device):
    """The input shaft driven at 2 rad/s turns the output through a universal joint bent by 30 degrees: over a turn
    and a half the output's angle is ``atan2(cos beta sin phi_1, cos phi_1)``, its speed ``cos beta / (1 - sin^2
    beta sin^2 phi_1)`` times the input's."""
    beta, omega, dt = np.radians(30.0), 2.0, 0.01
    model, (j_in, j_out) = _cardan(device, beta)
    solver = newton.solvers.SolverIPC(model, **_TIGHT_ROTATIONS)
    control = model.control()
    target_qd = control.joint_target_qd.numpy()
    target_qd[model.joint_qd_start.numpy()[j_in]] = omega
    control.joint_target_qd.assign(target_qd)
    starts = model.joint_q_start.numpy()
    phi = [np.zeros(2)]
    _run(
        model, solver, int(3.0 * np.pi / (omega * dt)), dt, state=model.state(), control=control,
        each=lambda k, s: phi.append(s.joint_q.numpy()[[starts[j_in], starts[j_out]]].astype(np.float64)),
    )  # fmt: skip
    phi = np.array(phi)
    phi1, phi2 = phi[:, 0], phi[:, 1]
    test.assertGreater(phi1[-1], 2.9 * np.pi)
    expected = np.unwrap(np.arctan2(np.cos(beta) * np.sin(phi1), np.cos(phi1)))
    np.testing.assert_allclose(phi2, expected, atol=2e-3)
    mid = 0.5 * (phi1[1:] + phi1[:-1])
    ratio = np.diff(phi2) / np.diff(phi1)
    np.testing.assert_allclose(
        ratio[10:], (np.cos(beta) / (1.0 - np.sin(beta) ** 2 * np.sin(mid) ** 2))[10:], atol=2e-2
    )


def test_drive_targets_and_turns(test, device):
    """A drive (``target_ke``, ``target_kd``) brings both angles to their targets; then the first angle's target
    ramps through three turns and ``joint_q`` follows it past 2 pi, as a revolute joint's angle does."""
    model = _pendulum(device, gravity=0.0, target_ke=50.0, target_kd=20.0)
    solver = newton.solvers.SolverIPC(model)
    control = model.control()
    control.joint_target_q.assign(np.array([0.4, -0.3], np.float32))
    state = _run(model, solver, 250, 0.01, control=control)
    np.testing.assert_allclose(state.joint_q.numpy(), [0.4, -0.3], atol=2e-3)
    turns = 300

    def ramp(k, s):
        control.joint_target_q.assign(np.array([0.4 + 6.0 * np.pi * min(k + 2, turns) / turns, -0.3], np.float32))

    control.joint_target_q.assign(np.array([0.4 + 6.0 * np.pi / turns, -0.3], np.float32))
    state = _run(model, solver, turns + 200, 0.01, state=state, control=control, each=ramp)
    np.testing.assert_allclose(state.joint_q.numpy(), [0.4 + 6.0 * np.pi, -0.3], atol=5e-3)


def test_limit_friction_and_joint_f(test, device):
    """Per angle: gravity rests a pendulum on a limit of its first angle past the bound by the cubic's
    ``sqrt(tau delta_r / ke)``; friction above the load's torque holds the pendulum where it started (it creeps at most
    ``eps_velocity`` times the ratio of the two); ``joint_f`` spins a box about the first axis at ``tau / I`` from the
    fourth step on (the first steps carry the joint's first pull on the box's stretch)."""
    dt = 0.01
    with test.subTest("limit"):
        # released at 0.2 rad about x, the lower bound of a limit [0.2, 0.5] of limit_ke 1e3 N m/rad: gravity pulls the
        # box toward 0, and it rests below the bound by the overshoot of its weight's torque on its 0.5 m arm
        model = _pendulum(device, second={}, limit_lower=0.2, limit_upper=0.5, limit_ke=1.0e3, limit_kd=20.0)
        solver = newton.solvers.SolverIPC(model, **_TIGHT_ROTATIONS)
        state = _run(model, solver, 300, dt, state=_state(model, [0.2, 0.0]))
        q = state.joint_q.numpy().astype(np.float64)
        tau = 8.0 * _G * 0.5 * np.sin(q[0])
        test.assertAlmostEqual(q[0], 0.2 - np.sqrt(tau * _DELTA_R / 1.0e3), delta=1e-4)
        test.assertAlmostEqual(q[1], 0.0, delta=1e-5)
    with test.subTest("friction"):
        model = _pendulum(device, friction=10.0)  # the load at 0.2 rad: 8 g 0.5 sin(0.2) = 7.8 N m
        solver = newton.solvers.SolverIPC(model, **_TIGHT_ROTATIONS)
        state = _run(model, solver, 100, dt, state=_state(model, [0.2, 0.0]))
        creep = 0.01 * 7.8 / (2.0 * 10.0) * 1.0  # eps_velocity tau / (2 F) for 1 s
        test.assertAlmostEqual(float(state.joint_q.numpy()[0]), 0.2, delta=2.0 * creep)
    with test.subTest("joint_f"):
        model = _pendulum(device, gravity=0.0)
        solver = newton.solvers.SolverIPC(model, **_TIGHT_ROTATIONS)
        control = model.control()
        control.joint_f.assign(np.array([0.5, 0.0], np.float32))
        inertia = 8.0 * (0.2**2 + 0.2**2) / 12.0 + 8.0 * 0.5**2  # about x through the joint
        rates = []
        _run(model, solver, 10, dt, control=control, each=lambda k, s: rates.append(float(s.joint_qd.numpy()[0])))
        # from the fourth step on: the first three carry the joint's first pull on the box's stretch
        np.testing.assert_allclose(np.diff(rates)[3:] / dt, 0.5 / inertia, rtol=1e-4)


def test_parent_force(test, device):
    """``BODY_PARENT_F`` of a box on a universal joint with a drive is the box's momentum balance at every step."""
    model = _pendulum(device, target_ke=20.0, target_kd=1.0, friction=0.5)
    solver = newton.solvers.SolverIPC(model, **_TIGHT_ROTATIONS)
    observables = solver.observables(kinds={_KIND.BODY_PARENT_F})
    weight = 8.0 * _G
    s0, s1 = _state(model, [0.5, -0.2], [0.5, 0.5]), model.state()
    for _ in range(10):
        q0, qd0 = s0.body_q.numpy().astype(np.float64), s0.body_qd.numpy().astype(np.float64)
        solver.step(s0, s1, None, None, 0.005, observables=observables)
        qd1 = s1.body_qd.numpy().astype(np.float64)
        force = 8.0 * (qd1[0, :3] - qd0[0, :3]) / 0.005 - 8.0 * np.array([0.0, 0.0, -_G])
        np.testing.assert_allclose(observables.body_parent_f.numpy()[0, :3], force, atol=1e-4 * weight)
        test.assertGreater(np.abs(q0).max(), 0.0)
        s0, s1 = s1, s0


def test_captured_equal_eager(test, device):
    """Steps of a universal joint with every term on, captured in a graph, give the eager steps' states."""
    dof = {"target_ke": 20.0, "target_kd": 1.0, "friction": 0.2, "armature": 0.01, "damping": 0.1}
    dof.update(limit_lower=-0.5, limit_upper=0.5, limit_ke=100.0)
    eager_model, model = _pendulum(device, **dof), _pendulum(device, **dof)
    eager = newton.solvers.SolverIPC(eager_model)
    end = _run(eager_model, eager, 12, 0.01, state=_state(eager_model, [0.3, 0.2], [1.0, -1.0]))
    solver = newton.solvers.SolverIPC(model)
    s0, s1 = _state(model, [0.3, 0.2], [1.0, -1.0]), model.state()
    solver.step(s0, s1, None, None, 0.01)  # loads the state, compiles the kernels
    with wp.ScopedCapture(device) as capture:
        solver.step(s1, s0, None, None, 0.01)
        solver.step(s0, s1, None, None, 0.01)
    for _ in range(5):
        wp.capture_launch(capture.graph)
    solver.step(s1, s0, None, None, 0.01)
    np.testing.assert_allclose(s0.body_q.numpy(), end.body_q.numpy(), atol=1e-6)
    np.testing.assert_allclose(s0.joint_q.numpy(), end.joint_q.numpy(), atol=1e-6)


def _hanging(device, lower, upper, start=0.45, gravity=_G, **dof):
    """A 0.2 m box (8 kg) hanging ``start`` below a point of the world on a distance joint of the bounds
    ``[lower, upper]`` between the point and the box's centre; z up. Its DOF 0 takes ``dof``."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -gravity), up_axis=newton.Axis.Z)
    link = _box_link(builder, xform=wp.transform(wp.vec3(0.0, 0.0, -start), wp.quat_identity()))
    j = builder.add_joint_distance(-1, link, min_distance=lower, max_distance=upper)
    for name, value in dof.items():
        getattr(builder, f"joint_{name}")[builder.joint_qd_start[j]] = value
    builder.add_articulation([j])
    return builder.finalize(device=device)


def test_range_free_inside_and_stopped(test, device):
    """A box on a distance joint [0.4, 0.6] falls freely while its distance is inside and rests past 0.6 m by the
    cubic's overshoot ``sqrt(m g delta_r / limit_ke)``; with only a lower bound it falls freely past 0.6 m; with only
    an upper bound it rests as with both. ``BODY_PARENT_F`` holds its weight at rest."""
    dt = 0.01
    for lower, upper, rests in ((0.4, 0.6, True), (0.4, -1.0, False), (-1.0, 0.6, True)):
        with test.subTest(lower=lower, upper=upper):
            model = _hanging(device, lower, upper)
            solver = newton.solvers.SolverIPC(model, **_TIGHT_ROTATIONS)
            observables = solver.observables(kinds={_KIND.BODY_PARENT_F})
            heights = []

            def record(k, s, heights=heights):
                heights.append(float(s.body_q.numpy()[0, 2]))

            _run(model, solver, 300, dt, state=model.state(), observables=observables, each=record)
            free = [-0.45 - _G * dt * dt * n * (n + 1) / 2.0 for n in range(1, 8)]  # implicit Euler, 7 steps
            np.testing.assert_allclose(heights[:7], free, atol=1e-5)
            if rests:
                overshoot = np.sqrt(8.0 * _G * _DELTA_R / 1.0e4)  # Newton's default limit_ke
                test.assertAlmostEqual(-heights[-1], 0.6 + overshoot, delta=1e-5)
                np.testing.assert_allclose(observables.body_parent_f.numpy()[0, :3], [0.0, 0.0, 8.0 * _G], atol=1e-3)
            else:
                test.assertLess(heights[-1], -10.0)


def test_range_lower_bound(test, device):
    """A box thrown at 2 m/s toward the point from 0.5 m on a distance joint [0.4, 0.6] without gravity turns back
    at the lower bound and leaves again."""
    model = _hanging(device, 0.4, 0.6, start=0.5, gravity=0.0, limit_kd=0.0)
    solver = newton.solvers.SolverIPC(model, **_TIGHT_ROTATIONS)
    state = model.state()
    qd = state.body_qd.numpy()
    qd[0, 2] = 2.0
    state.body_qd.assign(qd)
    nearest = []
    # 0.15 s: back past the lower bound, before the upper one
    end = _run(model, solver, 15, 0.01, state=state, each=lambda k, s: nearest.append(-float(s.body_q.numpy()[0, 2])))
    test.assertLess(min(nearest), 0.4)
    test.assertGreater(min(nearest), 0.37)
    test.assertLess(float(end.body_qd.numpy()[0, 2]), 0.0)


def test_range_bounds_notified(test, device):
    """A change of a distance joint's bounds that changes its kind builds the solve again (a range becomes a stitch
    of one length, a length becomes a range, both bounds negative hold nothing); one within a range does not."""
    model = _hanging(device, 0.4, 0.6)
    solver = newton.solvers.SolverIPC(model)
    s = _run(model, solver, 2, 0.01, state=model.state())
    first = model.joint_qd_start.numpy()[0]

    def bounds(lower, upper):
        lo, hi = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
        lo[first], hi[first] = lower, upper
        model.joint_limit_lower.assign(lo)
        model.joint_limit_upper.assign(hi)
        engine = solver._engine
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        return solver._engine is not engine

    for lower, upper, rebuilt in ((0.3, 0.7, False), (0.5, 0.5, True), (0.4, 0.6, True), (-1.0, -1.0, True)):
        with test.subTest(lower=lower, upper=upper):
            test.assertEqual(bounds(lower, upper), rebuilt)
            s = _run(model, solver, 2, 0.01, state=s)
    test.assertIsNone(solver._features.universal.ranges)


def test_refusals(test, device):
    """A D6 joint of two angular axes that are not orthonormal, D6 joints of other axes than a universal joint's that
    lock what no joint of SolverIPC locks, and a distance joint whose lower bound is above its upper are refused."""

    def refused(add):
        builder = newton.ModelBuilder()
        link = _box_link(builder)
        builder.add_articulation([add(builder, link)])
        with test.assertRaises(NotImplementedError) as caught:
            newton.solvers.SolverIPC(builder.finalize(device=device))
        return str(caught.exception)

    message = refused(
        lambda b, link: b.add_joint_d6(-1, link, angular_axes=[D(axis=(1.0, 0.0, 0.0)), D(axis=(1.0, 1.0, 0.0))])
    )
    test.assertEqual(
        message,
        "SolverIPC: the angular axes of the D6 joint 0 are not orthonormal; make them orthonormal. Newton's "
        "coordinates of a universal joint assume they are.",
    )
    message = refused(lambda b, link: b.add_joint_d6(-1, link, linear_axes=[D(axis=newton.Axis.X)],
                                                     angular_axes=[D(axis=newton.Axis.X)]))  # fmt: skip
    test.assertEqual(
        message,
        "SolverIPC: the D6 joint 0 has 1 linear and 1 angular axes; give it no axis, one axis, two or three angular "
        "axes, or three linear and three angular axes. Its axes lock 2 translations and 2 rotations, which no joint of "
        "SolverIPC locks.",
    )
    message = refused(lambda b, link: b.add_joint_distance(-1, link, min_distance=0.6, max_distance=0.4))
    test.assertEqual(
        message,
        "SolverIPC: the DISTANCE joint 0 bounds its distance to [0.6000000238418579, 0.4000000059604645], the lower "
        "bound above the upper; make the lower bound at most the upper, or negative for none.",
    )


@requires("uipc_warp")
class TestSolverIPCUniversalRange(unittest.TestCase):
    """Universal joints and distance joints with a range."""


devices = get_cuda_test_devices()
for name, func in (
    ("test_joint_q_is_eval_iks", test_joint_q_is_eval_iks),
    ("test_cardan_formula", test_cardan_formula),
    ("test_drive_targets_and_turns", test_drive_targets_and_turns),
    ("test_limit_friction_and_joint_f", test_limit_friction_and_joint_f),
    ("test_parent_force", test_parent_force),
    ("test_captured_equal_eager", test_captured_equal_eager),
    ("test_range_free_inside_and_stopped", test_range_free_inside_and_stopped),
    ("test_range_lower_bound", test_range_lower_bound),
    ("test_range_bounds_notified", test_range_bounds_notified),
    ("test_refusals", test_refusals),
):
    add_function_test(TestSolverIPCUniversalRange, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
