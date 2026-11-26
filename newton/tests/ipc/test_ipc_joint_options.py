# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC's joints beyond libuipc's own: D6 and distance joints, armature, limit damping and effort limits."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.ipc._fixtures import (
    _CONVERGED,
    _box_body,
    _drive_recurrence,
    _drive_recurrence_bdf2,
    _hinged,
    _scene_calls,
    _steps,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires


def _joint_inputs(model, **options):
    """SolverIPC built for ``model``, and what it gives the engine of its joints: the arguments of its calls of
    libuipc-warp's ``ArrayScene`` joint methods, by kind."""
    import uipc_warp.arrays  # noqa: PLC0415

    scene = uipc_warp.arrays.ArrayScene
    with (
        _scene_calls(scene.add_revolute_joints) as revolute,
        _scene_calls(scene.add_prismatic_joints) as prismatic,
        _scene_calls(scene.add_fixed_joints) as fixed,
        _scene_calls(scene.add_spherical_joints) as spherical,
    ):
        solver = newton.solvers.SolverIPC(model, **options)
    return solver, {"revolute": revolute, "prismatic": prismatic, "fixed": fixed, "spherical": spherical}


def _assert_same_calls(test, got, want):
    """The same calls, kind by kind, each with the same arguments, bit for bit."""
    test.assertEqual({k: len(v) for k, v in got.items()}, {k: len(v) for k, v in want.items()})
    for kind, calls in want.items():
        for call, other in zip(calls, got[kind], strict=True):
            test.assertEqual(sorted(other), sorted(call), kind)
            for name, value in call.items():
                if name != "self":
                    np.testing.assert_array_equal(np.asarray(other[name]), np.asarray(value), err_msg=f"{kind}.{name}")


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
    """A D6 joint of one angular axis is libuipc's revolute joint about it, of one linear axis its prismatic joint along
    it, of no axis its fixed joint and of three orthonormal angular axes its ball joint, with their drives, limits and
    joint_f: a chain of the four built with D6 joints gives the engine the joints of the same chain built with Newton's
    revolute, fixed, ball and prismatic joints (libuipc-warp's ArrayScene calls, argument by argument, bit for bit), and
    steps as it does, the revolute and prismatic coordinates included."""
    runs = []
    for d6 in (False, True):
        model, state = _d6_twins(device, d6)
        solver, groups = _joint_inputs(model)
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
    _assert_same_calls(test, d6_groups, native_groups)
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
            solver = newton.solvers.SolverIPC(model)
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
                newton.solvers.SolverIPC,
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
    solver = newton.solvers.SolverIPC(model, **_CONVERGED)
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
    """A distance joint of one length L is libuipc's soft vertex stitch of rest length L between its joint frames'
    origins: a sphere hanging 1 m from the world on it at its centre swings 0.02 rad as implicit Euler integrates the
    linearized point-mass pendulum, step by step to 1e-3 of its amplitude, and stays at L within the ball joint's
    penalty. A distance joint with a range or a single bound is no stitch: the universal module takes it
    (test_universal_range), and refuses one whose lower bound is above its upper."""
    L, theta0, g, h = 1.0, 0.02, 10.0, 1.0e-3
    model = _distance_pendulum(device, L, theta0, L, L, gravity=g)
    solver = newton.solvers.SolverIPC(model, **_CONVERGED)
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
            solver = newton.solvers.SolverIPC(_distance_pendulum(device, L, 0.0, lower, upper))
            test.assertEqual(len(solver._distance_rows), 0)
            np.testing.assert_array_equal(solver._features.universal.ranges.joints, [0])
    with test.assertRaisesRegex(NotImplementedError, "the lower bound above the upper"):
        newton.solvers.SolverIPC(_distance_pendulum(device, L, 0.0, 1.0, 0.5))


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
    solver = newton.solvers.SolverIPC(model, **_CONVERGED)
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
            solver = newton.solvers.SolverIPC(model)
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
    """By default (enable_joint_limit_damping=True) a slider that falls under gravity onto its lower limit takes the
    damping -joint_limit_kd qd of Featherstone and SemiImplicit in the steps that start past the limit, at the step's
    end velocity: it follows the implicit-Euler recurrence of the cubic limit with that damping step by step to 1e-4
    m, and stays at the limit where the undamped slider (enable_joint_limit_damping=False, kd not read) bounces 5 cm
    off it, which follows the recurrence without damping. A recurrence that switches the damping on in the steps that
    end past the limit is farther off."""
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


def test_armature_is_implicit_euler(test, device):
    """A revolute and a prismatic joint driven from rest by target gains, a passive spring, passive damping and joint_f
    take their joint_armature as the inertia of their coordinate: they follow the recurrence of
    (I + a) q'' = ke (q* - q) + kd (qd* - qd) - ks (q - q_rest) - c qd + f step by step to 1e-4 of their motion, under
    implicit Euler and, from an implicit Euler step, under BDF2, where the recurrence of I alone is farther off. A free
    joint's armature is refused (a ball joint's is test_ball's)."""
    for integrator, recurrence in (("bdf1", _drive_recurrence), ("bdf2", _drive_recurrence_bdf2)):
        for kind, length, scale, armature in (("revolute", 0.1, 1.0, 20.0), ("prismatic", 1.0, 10.0, 200.0)):
            with test.subTest(integrator=integrator, kind=kind):
                gains = {"target_ke": 40.0, "target_kd": 2.0, "stiffness": 10.0, "damping": 1.0}
                gains = {name: scale * value for name, value in gains.items()}
                targets = {"target_q": 0.5 * length, "target_qd": 0.1 * length, "rest_q": -0.2 * length}
                model, state, _ = _hinged(
                    device, kind, axis=newton.Axis.X, radius=0.5, armature=armature, **targets, **gains
                )
                solver = newton.solvers.SolverIPC(model, integrator=integrator, **_CONVERGED)
                control = model.control()
                control.joint_f.assign(np.array([0.2 * scale * length], np.float32))
                m = float(model.body_mass.numpy()[0])
                inertia = float(model.body_inertia.numpy()[0][0, 0]) if kind == "revolute" else m
                h, steps = 0.01, 100
                q, _ = _steps(model, solver, state, steps, h, control)
                args = {
                    "ke": gains["target_ke"], "kd": gains["target_kd"], "ks": gains["stiffness"],
                    "c": gains["damping"], "q_target": targets["target_q"], "qd_target": targets["target_qd"],
                    "q_rest": targets["rest_q"], "f": 0.2 * scale * length,
                }  # fmt: skip
                expected = recurrence(inertia + armature, h, steps, **args)
                motion = np.abs(expected).max()
                test.assertGreater(motion, 0.1 * length)
                np.testing.assert_allclose(q[:, 0], expected, rtol=0.0, atol=1e-4 * motion)
                test.assertGreater(np.abs(recurrence(inertia, h, steps, **args) - expected).max(), 1e-2 * motion)
    builder = newton.ModelBuilder()
    _box_body(builder, (0.0, 0.0, 1.0), (0.1, 0.1, 0.1))
    builder.joint_armature[3] = 0.01  # the free joint's first angular degree of freedom
    with test.assertRaisesRegex(NotImplementedError, "armature of revolute, prismatic, ball and universal joints only"):
        newton.solvers.SolverIPC(builder.finalize(device=device))


def _bounded_drive_recurrence(inertia, h, steps, ke, kd, q_target, bound, forces=None):
    """q_1 .. q_steps of implicit Euler for inertia q'' = clamp(ke (q* - q) - kd qd) from rest, the drive's force
    bounded by ``bound`` as decided at the start of every step (SolverIPC's enable_joint_effort_limit): a step that
    starts with |ke (q* - q_n) - kd qd_n| > bound takes the constant force of that sign, another the drive. With
    ``bound`` None the drive is not bounded. ``forces`` collects the drive's force at every step's start."""
    q, qd, out = 0.0, 0.0, []
    for _ in range(steps):
        tau = ke * (q_target - q) - kd * qd
        if forces is not None:
            forces.append(tau)
        if bound is not None and abs(tau) > bound:
            q_next = q + h * qd + h * h * np.sign(tau) * bound / inertia
        else:
            q_next = (inertia * (q + h * qd) / h**2 + ke * q_target + kd * q / h) / (inertia / h**2 + ke + kd / h)
        q, qd = q_next, (q_next - q) / h
        out.append(q)
    return np.array(out)


def test_effort_limit_bounds_the_drive(test, device):
    """By default (enable_joint_effort_limit=True) a hinge driven through a step of its target by gains whose force
    exceeds its joint_effort_limit takes the bound as a constant torque in the steps that start above it, and the drive
    in the others: it follows that recurrence step by step to 1e-4 of its motion (through the bound, the drive, and the
    bound again as it brakes), where the unbounded drive's recurrence is far off; under BDF2 its saturated steps from
    rest follow BDF2's recurrence of the constant torque. With enable_joint_effort_limit=False the limit is refused,
    and a negative one always."""
    ke, kd, q_target, bound, h, steps = 400.0, 100.0, 0.05, 2.0, 0.01, 300
    model, state, _ = _hinged(
        device, axis=newton.Axis.X, radius=0.5, target_ke=ke, target_kd=kd, target_q=q_target, effort_limit=bound
    )
    with test.assertRaisesRegex(NotImplementedError, "joint_effort_limit 2.0 and a drive; raise it to 1e\\+06"):
        newton.solvers.SolverIPC(model, enable_joint_effort_limit=False)
    inertia = float(model.body_inertia.numpy()[0][0, 0])
    forces = []
    expected = _bounded_drive_recurrence(inertia, h, steps, ke, kd, q_target, bound, forces)
    saturated = np.abs(np.array(forces)) > bound
    test.assertTrue(saturated[0] and not saturated.all())  # the bound, the drive, and the bound again
    test.assertGreater(np.count_nonzero(np.diff(saturated.astype(int))), 1)
    solver = newton.solvers.SolverIPC(model, **_CONVERGED)
    q, _ = _steps(model, solver, state, steps, h, model.control())
    motion = np.abs(expected).max()
    np.testing.assert_allclose(q[:, 0], expected, rtol=0.0, atol=1e-4 * motion)
    unbounded = _bounded_drive_recurrence(inertia, h, steps, ke, kd, q_target, None)
    test.assertGreater(np.abs(unbounded - expected).max(), 0.1 * motion)

    model, state, _ = _hinged(
        device, axis=newton.Axis.X, radius=0.5, target_ke=ke, target_kd=kd, target_q=q_target, effort_limit=bound
    )
    solver = newton.solvers.SolverIPC(model, integrator="bdf2", **_CONVERGED)
    first = int(np.argmin(saturated))  # the steps from rest that start above the bound
    q, _ = _steps(model, solver, state, first, h, model.control())
    torque = _drive_recurrence_bdf2(inertia, h, first, f=bound)
    np.testing.assert_allclose(q[:, 0], torque, rtol=0.0, atol=1e-4 * np.abs(torque).max())

    model, _, _ = _hinged(device, axis=newton.Axis.X, target_ke=ke, effort_limit=-1.0)
    with test.assertRaisesRegex(NotImplementedError, "joint_effort_limit -1.0 and a drive; make it 0 or positive"):
        newton.solvers.SolverIPC(model)


def test_joint_options_change_nothing_unused(test, device):
    """The terms beyond libuipc's driving term leave a joint that does not use them as it is: a revolute and a
    prismatic joint driven within their limits, with joint_limit_kd that never acts, no armature and Newton's default
    effort limit, step bit for bit with enable_joint_limit_damping and enable_joint_effort_limit on (the default) and
    off."""
    runs = []
    for option in (False, True):
        steps = []
        for kind, length in (("revolute", 0.1), ("prismatic", 1.0)):
            model, state, _ = _hinged(
                device, kind, axis=newton.Axis.X, radius=0.5, target_ke=40.0, target_kd=2.0, stiffness=10.0,
                damping=1.0, target_q=0.5 * length, limit_lower=-length, limit_upper=length, limit_ke=1.0e4,
                limit_kd=10.0,
            )  # fmt: skip
            solver = newton.solvers.SolverIPC(
                model, enable_joint_limit_damping=option, enable_joint_effort_limit=option
            )
            test.assertEqual(
                (solver._joint_options.limit_damping, solver._joint_options.effort_limit), (option, option)
            )
            steps.append(_steps(model, solver, state, 50, 0.01, None, lambda s: s.body_q.numpy().copy())[0])
        runs.append(steps)
    for before, after in zip(*runs, strict=True):
        test.assertGreater(np.abs(before[-1] - before[0]).max(), 1e-3)
        np.testing.assert_array_equal(after, before)


def test_rejects_d6_joints_libuipc_lacks(test, device):
    """The D6 joints that lock what no joint of SolverIPC locks are refused, naming what they lock: three linear axes,
    one linear and one angular; and two or three angular axes that are not orthonormal, and drives and limits on the D6
    joints that become a ball joint or no joint. (Two orthonormal angular axes are a universal joint, which the
    universal module takes: test_universal_range.)"""
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
        ("skew universal", model(angular=("X", skew)), "angular axes of the D6 joint 0 are not orthonormal"),
        ("translations", model(linear="XYZ"), "has 3 linear and 0 angular axes.*lock 0 translations and 3 rotations"),
        ("cylindrical", model(linear="X", angular="X"), "has 1 linear and 1 angular axes.*lock 2 translations"),
        ("skew axes", model(angular=("X", skew, "Z")), "angular axes of the D6 joint 0 are not orthonormal"),
        ("ball drive", model(angular="XYZ", target_ke=10.0), "D6 joint 0 has the target mode POSITION"),
        ("free limit", model(linear="XYZ", angular="XYZ", limit_lower=-0.1), "limits revolute and prismatic"),
    ):
        with test.subTest(name):
            with test.assertRaisesRegex(NotImplementedError, pattern):
                newton.solvers.SolverIPC(m)


@requires("uipc_warp")
class TestSolverIPCJointOptions(unittest.TestCase):
    """Newton's joints and joint terms beyond libuipc's revolute, prismatic, fixed, ball and free joints: D6 and
    distance joints that map onto libuipc's own constitutions, armature, limit damping and effort limits."""


devices = get_cuda_test_devices()
for name, func in (
    ("test_d6_joints_take_libuipcs_joints", test_d6_joints_take_libuipcs_joints),
    ("test_d6_coordinates_are_newtons", test_d6_coordinates_are_newtons),
    ("test_d6_joint_f_is_featherstones", test_d6_joint_f_is_featherstones),
    ("test_rejects_d6_joints_libuipc_lacks", test_rejects_d6_joints_libuipc_lacks),
    ("test_distance_joint_swings_as_a_pendulum", test_distance_joint_swings_as_a_pendulum),
    ("test_distance_joint_between_bodies", test_distance_joint_between_bodies),
    ("test_limit_damping_is_implicit_euler", test_limit_damping_is_implicit_euler),
    ("test_armature_is_implicit_euler", test_armature_is_implicit_euler),
    ("test_effort_limit_bounds_the_drive", test_effort_limit_bounds_the_drive),
    ("test_joint_options_change_nothing_unused", test_joint_options_change_nothing_unused),
):
    add_function_test(TestSolverIPCJointOptions, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
