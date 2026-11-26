# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC's revolute, prismatic, fixed and ball joints, joints switched by ``joint_enabled`` between replays of a
captured step, and a loop of joints: a four-bar linkage closed by a joint outside the articulation."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.ipc._fixtures import _CONVERGED, _TIGHT, _TINY_LISTS, _box_body, _drive_recurrence, _hinged, _steps
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires


def test_unplaced_links_start_at_forward_kinematics(test, device):
    """Build the solve at the forward-kinematics poses of links added without a pose.

    A double pendulum is built as Newton's basic_pendulum example builds it: links added without a pose, so that the
    model holds both at the origin, through the ground plane, and the forward kinematics written into the state only
    after the solver is built. The solver is built at the poses newton.eval_fk computes from the model's joint
    coordinates, and steps captured from that state run as those of a solver built from a model that holds the poses. A
    link that the joint coordinates put into the ground fails the first step from that state before it runs, by name:
    an eager step raises, and a captured one reports it through check_status.
    """

    def pendulum(height=5.0, angle=0.0, posed=False):
        builder = newton.ModelBuilder()
        links = [builder.add_link(label=f"link_{k}") for k in range(2)]
        for link in links:
            builder.add_shape_box(link, hx=1.0, hy=0.1, hz=0.1)
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -wp.pi * 0.5)
        end = wp.transform(wp.vec3(-1.0, 0.0, 0.0), wp.quat_identity())
        joints = [
            builder.add_joint_revolute(-1, links[0], axis=wp.vec3(0.0, 1.0, 0.0),
                                       parent_xform=wp.transform(wp.vec3(0.0, 0.0, height), rot), child_xform=end),
            builder.add_joint_revolute(links[0], links[1], axis=wp.vec3(0.0, 1.0, 0.0),
                                       parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
                                       child_xform=end),
        ]  # fmt: skip
        builder.add_articulation(joints)
        builder.add_ground_plane()
        builder.joint_q[0] = angle
        model = builder.finalize(device=device)
        if posed:
            newton.eval_fk(model, model.joint_q, model.joint_qd, model)
        return model

    model = pendulum()
    np.testing.assert_array_equal(model.body_q.numpy(), [[0, 0, 0, 0, 0, 0, 1]] * 2)
    solver = newton.solvers.SolverIPC(model)  # refused before: the links crossed the ground at the origin
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    base = solver._scene.layout().body_base[:2]
    np.testing.assert_allclose(solver._engine.positions()[base], state.body_q.numpy()[:, :3], atol=1e-6)
    test.assertAlmostEqual(float(state.body_q.numpy()[1, 2]), 5.0, places=5)  # the links lie level at the pivot

    def captured(model, solver, state, frames=20):
        s0, s1 = state, model.state()
        solver.step(s0, s1, None, None, 0.001)
        with wp.ScopedCapture(device=device) as capture:
            solver.step(s1, s0, None, None, 0.001)
            solver.step(s0, s1, None, None, 0.001)
        for _ in range(frames):
            wp.capture_launch(capture.graph)
        return s1.body_q.numpy()

    swung = captured(model, solver, state)
    posed = pendulum(posed=True)
    state = posed.state()
    reference = captured(posed, newton.solvers.SolverIPC(posed), state)
    test.assertLess(float(swung[1, 2]), 5.0 - 1e-3)  # the pendulum swings down
    np.testing.assert_allclose(swung, reference, atol=1e-6)
    # turned down from a pivot 3 m up, the outer link reaches 1 m into the ground: the caller places such links, so the
    # first step's device check refuses the start it takes from them, by name, captured or not
    low = pendulum(height=3.0, angle=0.5 * np.pi)
    state = low.state()
    newton.eval_fk(low, low.joint_q, low.joint_qd, state)
    pattern = r"before it ran: the state the step took from state_in puts body 1 \('link_1'\) and plane shape 2 "
    with test.assertRaisesRegex(RuntimeError, "failed on THICKNESS_VIOLATED " + pattern):
        newton.solvers.SolverIPC(low).step(state, low.state(), None, None, 0.001)
    solver = newton.solvers.SolverIPC(low)
    observables = solver.observables({solver.ObservableKind.STATUS})
    with wp.ScopedCapture(device=device) as capture:
        solver.step(state, low.state(), None, None, 0.001, observables=observables)
    wp.capture_launch(capture.graph)
    with test.assertRaisesRegex(RuntimeError, "The last step failed " + pattern):
        solver.check_status(observables)


def test_pendulum_is_implicit_euler(test, device):
    """Swing a pendulum as implicit Euler integrates the linearized pendulum.

    A pendulum of 2 cm per metre follows that recurrence step by step to 1e-3 of its amplitude, with the period of the
    closed form 2 pi sqrt(I / (m g L)) to 0.1%.
    """
    L, theta0, g, h = 1.0, 0.02, 10.0, 1e-3
    model, state, _ = _hinged(device, gravity=g, offset=(0.0, L, 0.0), radius=0.05)
    q = model.joint_q.numpy()
    q[0] = theta0
    state.joint_q.assign(q)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    solver = newton.solvers.SolverIPC(model, **_CONVERGED)
    m = float(model.body_mass.numpy()[0])
    inertia = float(model.body_inertia.numpy()[0][2, 2]) + m * L * L
    period = 2.0 * np.pi * np.sqrt(inertia / (m * g * L))
    steps = int(1.25 * period / h)
    angles, _ = _steps(model, solver, state, steps, h)
    expected = _drive_recurrence(inertia, h, steps, q0=theta0, ke=m * g * L)
    np.testing.assert_allclose(angles[:, 0], expected, rtol=0.0, atol=1e-3 * theta0)
    # the first zero crossing upwards after the start: three quarters of a period
    up = np.flatnonzero((angles[:-1, 0] < 0.0) & (angles[1:, 0] >= 0.0))[0]
    test.assertAlmostEqual(h * (up + 1) / (0.75 * period), 1.0, delta=1e-3)


def test_joint_forces_match_closed_form(test, device):
    """Move revolute, prismatic and ball joints under joint_f as implicit Euler's closed forms say.

    Each joint_f acts from rest without gravity: a revolute and a prismatic joint follow q_n = a h^2 n (n + 1) / 2 of
    implicit Euler, a = tau / I or F / m, and a ball joint whose parent frame is turned spins its child about the torque
    turned with it.
    """
    # small enough that the bodies turn slowly: an affine body turning under implicit Euler moves along a straight
    # line in its 12 coordinates and back onto the rotations, which loses about (omega dt)^2 of its angular velocity
    # per step
    h, steps = 0.01, 20
    n = np.arange(1, steps + 1)
    for kind, f in (("revolute", 0.005), ("prismatic", 2.0)):
        with test.subTest(kind):
            model, state, _ = _hinged(device, kind, axis=newton.Axis.X)
            control = model.control()
            control.joint_f.assign(np.array([f], np.float32))
            q, _ = _steps(model, newton.solvers.SolverIPC(model, **_CONVERGED), state, steps, h, control)
            m = float(model.body_mass.numpy()[0])
            a = f / (float(model.body_inertia.numpy()[0][0, 0]) if kind == "revolute" else m)
            np.testing.assert_allclose(q[:, 0], a * h * h * n * (n + 1) / 2.0, rtol=1e-5, atol=0.0)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Y)
    ball = builder.add_link()
    builder.add_shape_sphere(ball, radius=0.1)
    turn = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * np.pi)
    builder.add_articulation([builder.add_joint_ball(-1, ball, parent_xform=wp.transform((0.0, 0.0, 0.0), turn))])
    model = builder.finalize(device=device)
    control = model.control()
    # the torque about z of the turned parent frame: about -y in the world
    control.joint_f.assign(np.array([0.0, 0.0, 0.003], np.float32))
    qd, _ = _steps(model, newton.solvers.SolverIPC(model, **_CONVERGED), model.state(), steps, h, control,
                   lambda s: s.body_qd.numpy().copy())  # fmt: skip
    inertia = float(model.body_inertia.numpy()[ball][0, 0])
    np.testing.assert_allclose(qd[:, ball, 3:6], np.outer(-0.003 * n * h / inertia, [0.0, 1.0, 0.0]), atol=1e-6)


def test_fixed_and_ball_joints_hold(test, device):
    """Hold a welded box and a ball-jointed bob on a swinging pendulum within the joints' penalty.

    A box welded to a swinging pendulum keeps its pose relative to it, and a bob on a ball joint keeps its anchor on the
    pendulum's, within the joints' penalty (a load F stretches a joint by about F dt^2 / K, a few 1e-5 m here), while
    the pendulum swings through 1.2 rad.
    """
    builder = newton.ModelBuilder(gravity=(0.0, -10.0, 0.0), up_axis=newton.Axis.Y)
    arm = builder.add_link(xform=wp.transform((0.0, -0.5, 0.0), wp.quat_identity()))
    builder.add_shape_box(arm, hx=0.05, hy=0.5, hz=0.05)
    weld = builder.add_link(xform=wp.transform((0.15, -0.9, 0.0), wp.quat_identity()))
    builder.add_shape_box(weld, hx=0.1, hy=0.05, hz=0.05)
    bob = builder.add_link(xform=wp.transform((0.0, -1.3, 0.0), wp.quat_identity()))
    builder.add_shape_sphere(bob, radius=0.1)
    joints = [
        builder.add_joint_revolute(
            -1, arm, axis=newton.Axis.Z, child_xform=wp.transform((0.0, 0.5, 0.0), wp.quat_identity())
        ),
        builder.add_joint_fixed(arm, weld, parent_xform=wp.transform((0.15, -0.4, 0.0), wp.quat_identity())),
        builder.add_joint_ball(
            arm, bob, parent_xform=wp.transform((0.0, -0.6, 0.0), wp.quat_identity()),
            child_xform=wp.transform((0.0, 0.2, 0.0), wp.quat_identity()),
        ),
    ]  # fmt: skip
    builder.add_articulation(joints)
    model = builder.finalize(device=device)
    q = model.joint_q.numpy()
    q[0] = 1.2
    state = model.state()
    state.joint_q.assign(q)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    # libuipc's body stiffness: with the default one the arm's own stretch under the swing takes the ball joint's
    # anchors 1.03e-4 m apart, past the joints' penalty (docs/solvers/ipc.rst > "Rigid bodies")
    body_q, _ = _steps(
        model,
        newton.solvers.SolverIPC(model, body_kappa=1.0e8),
        state,
        100,
        0.01,
        None,
        lambda s: s.body_q.numpy().copy(),
    )

    def transforms(rows):
        return [wp.transform(*row[:3], *row[3:]) for row in rows]

    swing = np.arctan2(body_q[:, arm, 0], -body_q[:, arm, 1])
    test.assertGreater(swing.max() - swing.min(), 1.0)
    for frame in body_q:
        X_arm, X_weld, X_bob = transforms(frame[[arm, weld, bob]])
        relative = wp.transform_inverse(X_arm) * X_weld
        np.testing.assert_allclose(np.array(wp.transform_get_translation(relative)), [0.15, -0.4, 0.0], atol=1e-4)
        turn = wp.transform_get_rotation(relative)  # its angle is twice the length of its vector part
        test.assertLess(2.0 * float(wp.length(wp.vec3(turn[0], turn[1], turn[2]))), 1e-4)
        anchor_arm = wp.transform_point(X_arm, wp.vec3(0.0, -0.6, 0.0))
        anchor_bob = wp.transform_point(X_bob, wp.vec3(0.0, 0.2, 0.0))
        test.assertLess(float(wp.length(anchor_arm - anchor_bob)), 1e-4)


def test_joint_penalty_follows_the_masses(test, device):
    """Scale every joint's penalty with the masses of the dynamic bodies it joins, also after new masses.

    A joint's penalty is joint_strength_ratio times the mass of the dynamic bodies it joins, a kinematic body counting
    0, for every kind of joint, and it follows the masses that notify_model_changed gives the bodies (libuipc's revolute
    and prismatic joints keep the masses they were built with and scale their ratio by them).
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Y)
    arm = builder.add_link(xform=wp.transform((0.0, -0.5, 0.0), wp.quat_identity()))
    builder.add_shape_box(arm, hx=0.05, hy=0.5, hz=0.05)
    bob = builder.add_link(xform=wp.transform((0.0, -1.3, 0.0), wp.quat_identity()))
    builder.add_shape_sphere(bob, radius=0.2)
    builder.add_articulation(
        [
            builder.add_joint_revolute(-1, arm, child_xform=wp.transform((0.0, 0.5, 0.0), wp.quat_identity())),
            builder.add_joint_ball(
                arm, bob, parent_xform=wp.transform((0.0, -0.5, 0.0), wp.quat_identity()),
                child_xform=wp.transform((0.0, 0.3, 0.0), wp.quat_identity()),
            ),
        ]
    )  # fmt: skip
    base = builder.add_link(xform=wp.transform((2.0, 0.0, 0.0), wp.quat_identity()), is_kinematic=True)
    builder.add_shape_box(base, hx=0.2, hy=0.1, hz=0.2)
    slider = builder.add_link(xform=wp.transform((2.0, -0.5, 0.0), wp.quat_identity()))
    builder.add_shape_box(slider, hx=0.1, hy=0.1, hz=0.1)
    builder.add_articulation(
        [
            builder.add_joint_fixed(-1, base, parent_xform=wp.transform((2.0, 0.0, 0.0), wp.quat_identity())),
            builder.add_joint_prismatic(
                base, slider, axis=newton.Axis.Y, parent_xform=wp.transform((0.0, -0.5, 0.0), wp.quat_identity())
            ),
        ]
    )
    model = builder.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    solver = newton.solvers.SolverIPC(model)

    def penalties():
        solver.step(state, model.state(), None, None, 0.01)  # writes the joints' terms
        groups = solver._joint_groups
        return [
            float(groups.revolute.ratio.numpy()[0] * groups.revolute.mass_sum.numpy()[0]),
            float(groups.spherical.kappa.numpy()[0]),
            float(groups.fixed.kappa.numpy()[0]),
            float(groups.prismatic.ratio.numpy()[0] * groups.prismatic.mass_sum.numpy()[0]),
        ]

    for scale in (1.0, 2.0):
        with test.subTest(scale=scale):
            if scale != 1.0:
                for name in ("body_mass", "body_inertia"):
                    getattr(model, name).assign(scale * getattr(model, name).numpy())
                for name in ("body_inv_mass", "body_inv_inertia"):
                    getattr(model, name).assign(getattr(model, name).numpy() / scale)
                solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
            m = model.body_mass.numpy().astype(np.float64)
            expected = [100.0 * m[arm], 100.0 * (m[arm] + m[bob]), 0.0, 100.0 * m[slider]]
            np.testing.assert_allclose(penalties(), expected, rtol=1e-12)


def test_joint_coordinates_are_written(test, device):
    """Write joint_q and joint_qd after every step as eval_ik computes them from the bodies' state.

    This holds for a revolute, a prismatic, a ball and a free joint, except the revolute angle, which goes on past pi;
    an angle written into state_in picks the turn the joint goes on from.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Y)
    links = []
    for x in (0.0, 1.0, 2.0):
        links.append(builder.add_link(xform=wp.transform((x, 0.0, 0.0), wp.quat_identity())))
        builder.add_shape_box(links[-1], hx=0.1, hy=0.1, hz=0.1)
    joints = [
        builder.add_joint_revolute(-1, links[0], axis=newton.Axis.Z),
        builder.add_joint_prismatic(
            -1, links[1], axis=newton.Axis.X, parent_xform=wp.transform((1.0, 0.0, 0.0), wp.quat_identity())
        ),
        builder.add_joint_ball(
            links[1], links[2], parent_xform=wp.transform((0.5, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform((-0.5, 0.0, 0.0), wp.quat_identity()),
        ),
    ]  # fmt: skip
    builder.add_articulation(joints)
    _box_body(builder, (0.0, 2.0, 0.0), (0.1, 0.1, 0.1))  # a free body
    model = builder.finalize(device=device)
    qd = model.joint_qd.numpy()
    qd[:] = (10.0, 0.5, 1.0, -2.0, 0.5, 0.3, -0.2, 0.1, 0.4, 0.2, 0.7)  # 10 rad/s about the hinge
    state = model.state()
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    solver = newton.solvers.SolverIPC(model, **_CONVERGED)
    q_ik, qd_ik = wp.zeros_like(state.joint_q), wp.zeros_like(state.joint_qd)
    s0, s1 = state, model.state()
    angles = []
    for _ in range(50):
        solver.step(s0, s1, None, None, 0.01)
        s0, s1 = s1, s0
        angles.append(float(s0.joint_q.numpy()[0]))
        newton.eval_ik(model, s0, q_ik, qd_ik)
        q, q_eval = s0.joint_q.numpy(), q_ik.numpy()
        turns = np.round((q[0] - q_eval[0]) / (2.0 * np.pi))
        np.testing.assert_allclose(q[0] - 2.0 * np.pi * turns, q_eval[0], atol=1e-5)
        np.testing.assert_allclose(q[1:], q_eval[1:], atol=1e-5)
        np.testing.assert_allclose(s0.joint_qd.numpy(), qd_ik.numpy(), atol=1e-4)
    test.assertGreater(float(s0.joint_q.numpy()[0]), 4.0, "the hinge turns past pi")
    # a turn more, written into state_in with the bodies' state unchanged
    angle = float(s0.joint_q.numpy()[0])
    q = s0.joint_q.numpy()
    q[0] += 2.0 * np.pi
    s0.joint_q.assign(q)
    solver.step(s0, s1, None, None, 0.01)
    test.assertAlmostEqual(float(s1.joint_q.numpy()[0]) - angle, 2.0 * np.pi + angles[-1] - angles[-2], delta=2e-3)


def test_grown_lists_keep_joint_angles(test, device):
    """Keep a spinning wheel's angle when an eager step grows overflowing contact lists.

    A wheel spins on its hinge while a box drops onto the ground; when it lands, past half a turn of the wheel, the
    contact lists overflow, and the eager step grows them and solves again from the step's start: the wheel's angle goes
    on as in a solver with room, and so does every body.
    """

    def model():
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, -9.81, 0.0))
        wheel = builder.add_link(xform=wp.transform((0.0, 1.0, 0.0), wp.quat_identity()))
        builder.add_shape_box(wheel, hx=0.2, hy=0.2, hz=0.02)
        builder.add_articulation(
            [
                builder.add_joint_revolute(
                    -1, wheel, axis=newton.Axis.Z, parent_xform=wp.transform((0.0, 1.0, 0.0), wp.quat_identity())
                )
            ]
        )
        _box_body(builder, (1.0, 0.6, 0.0), (0.05, 0.05, 0.05))
        builder.add_ground_plane()
        m = builder.finalize(device=device)
        qd = m.joint_qd.numpy()
        qd[0] = 15.0
        m.joint_qd.assign(qd)
        return m

    results = []
    for tiny in (False, True):
        m = model()
        state = m.state()
        newton.eval_fk(m, m.joint_q, m.joint_qd, state)
        solver = newton.solvers.SolverIPC(m, **_CONVERGED)
        if tiny:
            solver._make_engine(_TINY_LISTS)
        q, last = _steps(m, solver, state, 45, 0.01)
        results.append((q[:, 0], last.body_q.numpy()))
    (roomy, roomy_body_q), (grown, grown_body_q) = results
    test.assertGreater(roomy[-1], np.pi)
    np.testing.assert_allclose(grown, roomy, atol=1e-5)
    np.testing.assert_allclose(grown_body_q, roomy_body_q, atol=1e-5)


def test_drives_are_implicit_euler(test, device):
    """Follow implicit Euler's recurrence for driven, sprung and damped revolute and prismatic joints.

    A revolute and a prismatic joint driven from rest by target gains, a passive spring and passive damping, towards a
    moving target, follow implicit Euler's recurrence of I q'' = ke (q* - q) + kd (qd* - qd) - ks (q - q_rest) - c qd
    step by step to 1e-4 of their motion (SolverIPC folds the four terms into libuipc's driving term of every step). The
    hinge turns a large sphere through small angles: an affine body turned under implicit Euler moves along a straight
    line in its 12 coordinates and back onto the rotations, which adds a term of order (omega dt)^3 to every step, 1e-4
    of the motion at 0.5 rad and 1e-6 at 0.05 rad here.
    """
    for kind, length, scale in (("revolute", 0.1, 1.0), ("prismatic", 1.0, 10.0)):
        with test.subTest(kind):
            gains = {"target_ke": 40.0, "target_kd": 2.0, "stiffness": 10.0, "damping": 1.0}
            gains = {name: scale * value for name, value in gains.items()}
            targets = {"target_q": 0.5 * length, "target_qd": 0.1 * length, "rest_q": -0.2 * length}
            model, state, _ = _hinged(device, kind, axis=newton.Axis.X, radius=0.5, **targets, **gains)
            solver = newton.solvers.SolverIPC(model, **_CONVERGED)
            m = float(model.body_mass.numpy()[0])
            inertia = float(model.body_inertia.numpy()[0][0, 0]) if kind == "revolute" else m
            h, steps = 0.01, 100
            q, _ = _steps(model, solver, state, steps, h, model.control())
            expected = _drive_recurrence(
                inertia, h, steps, ke=gains["target_ke"], kd=gains["target_kd"], ks=gains["stiffness"],
                c=gains["damping"], q_target=targets["target_q"], qd_target=targets["target_qd"],
                q_rest=targets["rest_q"],
            )  # fmt: skip
            test.assertGreater(np.abs(expected).max(), 0.1 * length)
            np.testing.assert_allclose(q[:, 0], expected, rtol=0.0, atol=1e-4 * np.abs(expected).max())


def test_joint_limits_hold(test, device):
    """Hold a joint pushed into its limit at the overshoot where libuipc's cubic limit balances the load.

    A revolute joint turned by a torque into its upper limit, and a slider pulled down by gravity onto its lower limit,
    come to rest past the limit by sqrt(F delta_r / limit_ke), where libuipc's cubic limit balances the load (SolverIPC
    sets its strength so that it pushes back with limit_ke delta_r at the overshoot delta_r = 1e-3); four times the
    stiffness halves the overshoot. Without the limit they run on.
    """
    tau, upper = 4.0, 0.3
    for limit_ke in (1.0e4, 4.0e4):
        with test.subTest(limit_ke=limit_ke):
            model, state, _ = _hinged(device, limit_lower=-1.0, limit_upper=upper, limit_ke=limit_ke, damping=2.0)
            control = model.control()
            control.joint_f.assign(np.array([tau], np.float32))
            q, _ = _steps(model, newton.solvers.SolverIPC(model, **_CONVERGED), state, 300, 0.01, control)
            test.assertLess(q[:, 0].max(), upper + 0.01)
            test.assertAlmostEqual(float(q[-1, 0]) - upper, np.sqrt(tau * 1e-3 / limit_ke), delta=1e-6)

            g, lower = 10.0, -0.1
            model, state, _ = _hinged(
                device, "prismatic", gravity=g, axis=newton.Axis.Y, limit_lower=lower, limit_upper=1.0,
                limit_ke=limit_ke, damping=20.0,
            )  # fmt: skip
            m = float(model.body_mass.numpy()[0])
            q, _ = _steps(model, newton.solvers.SolverIPC(model, **_CONVERGED), state, 300, 0.01)
            test.assertAlmostEqual(float(q[-1, 0]) - lower, -np.sqrt(m * g * 1e-3 / limit_ke), delta=1e-6)
    model, state, _ = _hinged(device, limit_lower=-1.0, limit_upper=upper, limit_ke=0.0, damping=2.0)
    control = model.control()
    control.joint_f.assign(np.array([tau], np.float32))
    q, _ = _steps(model, newton.solvers.SolverIPC(model), state, 100, 0.01, control)
    test.assertGreater(q[-1, 0], 1.0, "a limit of stiffness 0 holds nothing")


def test_drive_targets_change_inside_a_captured_graph(test, device):
    """Drive a revolute joint by targets written between replays of a captured step as eager steps do.

    Targets written into control.joint_target_q between replays drive the joint as they do in eager steps, to 1e-6: the
    drive is computed on the device from the control at every step.
    """

    def run(capture):
        model, state, _ = _hinged(device, axis=newton.Axis.X, target_ke=50.0, target_kd=5.0)
        solver = newton.solvers.SolverIPC(model, **_CONVERGED)
        control = model.control()
        s0, s1 = state, model.state()
        out = []
        solver.step(s0, s1, control, None, 0.01)  # loads the kernels
        s0, s1 = s1, s0
        graph = None
        if capture:
            with wp.ScopedCapture(device=device) as cap:
                solver.step(s0, s1, control, None, 0.01)
                solver.step(s1, s0, control, None, 0.01)
            graph = cap.graph
        for k in range(20):
            control.joint_target_q.assign(np.array([0.4 * np.sin(0.5 * k)], np.float32))
            if capture:
                wp.capture_launch(graph)
            else:
                for _ in range(2):
                    solver.step(s0, s1, control, None, 0.01)
                    s0, s1 = s1, s0
            out.append(s0.joint_q.numpy()[0])
        return np.array(out)

    eager, captured = run(False), run(True)
    test.assertGreater(np.abs(eager).max(), 0.1)
    np.testing.assert_allclose(captured, eager, atol=1e-6)


_G = 9.81


def _angle_about_y(q):
    """The angle of a rotation about +y from its quaternion (x, y, z, w)."""
    return 2.0 * np.arctan2(q[..., 1], q[..., 3])


def _jointed_spheres(device):
    """Four spheres of radius 5 cm, 1 m above the ground plane's level, each held at its centre by a joint to the
    world: a revolute joint about z, a prismatic joint along x, a fixed joint and a ball joint, in that order."""
    builder = newton.ModelBuilder()
    joints = []
    for k, add in enumerate(
        (builder.add_joint_revolute, builder.add_joint_prismatic, builder.add_joint_fixed, builder.add_joint_ball)
    ):
        link = builder.add_link(xform=wp.transform(wp.vec3(float(k), 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_sphere(link, radius=0.05)
        axis = {0: {"axis": wp.vec3(0.0, 0.0, 1.0)}, 1: {"axis": wp.vec3(1.0, 0.0, 0.0)}}.get(k, {})
        joints.append(add(-1, link, parent_xform=wp.transform(wp.vec3(float(k), 0.0, 1.0), wp.quat_identity()),
                          **axis))  # fmt: skip
        builder.add_articulation([joints[-1]])
    return builder.finalize(device=device)


def test_joint_enabled_is_read_at_every_replay(test, device):
    """Read joint_enabled of revolute, prismatic, fixed and ball joints at every step, also in a captured one.

    Four spheres hang at their centres from joints to the world, the revolute and ball joints turned by a joint_f torque
    of 5e-3 N m about z and the prismatic one pushed along x by 0.5 N. Over six replays of a captured pair of steps
    every sphere stays put but for its joint's motion, and the driven ones speed up as implicit Euler has them
    (tau dt / I and f dt / m per step). Clearing joint_enabled of the revolute and fixed joints in place between replays
    drops those two spheres into implicit Euler's free fall from where they are, the revolute one spinning on at its
    speed (its joint_f no longer acts), while the prismatic and ball spheres stay held and driven; clearing the other
    two then drops them likewise, the prismatic one sliding on at its speed.
    """
    dt = 0.01
    model = _jointed_spheres(device)
    solver = newton.solvers.SolverIPC(model, **_TIGHT)
    mass = model.body_mass.numpy().astype(np.float64)
    inertia = model.body_inertia.numpy().astype(np.float64)[:, 2, 2]
    tau, force = 5.0e-3, 0.5
    control = model.control()
    joint_f = np.zeros(model.joint_dof_count, np.float32)
    qd_start = model.joint_qd_start.numpy()
    joint_f[qd_start[0]] = tau  # the revolute joint's torque
    joint_f[qd_start[1]] = force  # the prismatic joint's force
    joint_f[qd_start[3] + 2] = tau  # the ball joint's torque about z of its parent frame
    control.joint_f.assign(joint_f)
    observables = [solver.observables({solver.ObservableKind.STATUS}) for _ in range(2)]
    s0, s1 = model.state(), model.state()
    solver.step(s0, s1, control, None, dt, observables=observables[1])
    s0, s1 = s1, s0
    with wp.ScopedCapture(device=device) as capture:
        solver.step(s0, s1, control, None, dt, observables=observables[0])
        solver.step(s1, s0, control, None, dt, observables=observables[1])

    def replays(count):
        """The bodies' positions and twists after each of ``count`` replays (two steps each)."""
        q, qd = [], []
        for _ in range(count):
            wp.capture_launch(capture.graph)
            for o in observables:
                test.assertFalse(solver.check_status(o))
            q.append(s0.body_q.numpy().astype(np.float64))
            qd.append(s0.body_qd.numpy().astype(np.float64))
        return np.array(q), np.array(qd)

    def held(q, bodies):
        """At the joint's origin; the prismatic joint's sphere anywhere along x."""
        for b in bodies:
            axes = [1, 2] if b == 1 else [0, 1, 2]
            np.testing.assert_allclose(q[:, b, axes], np.broadcast_to(np.array([float(b), 0.0, 1.0])[axes],
                                       (len(q), len(axes))), rtol=0.0, atol=1.0e-4)  # fmt: skip

    def falls(q, qd, start, start_qd, bodies):
        """Implicit Euler's free fall from ``start`` with ``start_qd``, two steps per replay."""
        k = 2.0 * np.arange(1, len(q) + 1)
        for b in bodies:
            z = start[b, 2] + k * dt * start_qd[b, 2] - _G * dt * dt * k * (k + 1.0) / 2.0
            np.testing.assert_allclose(q[:, b, 2], z, rtol=0.0, atol=1.0e-6)
            np.testing.assert_allclose(qd[:, b, :3], np.broadcast_to(start_qd[b, :3], (len(q), 3))
                                       + np.outer(k, [0.0, 0.0, -_G * dt]), rtol=0.0, atol=1.0e-5)  # fmt: skip

    def driven(qd, steps, bodies):
        """The speed of joint_f's torque or force from rest, ``steps`` steps in after each replay."""
        for b, column, rate in ((0, 5, tau / inertia[0]), (1, 0, force / mass[1]), (3, 5, tau / inertia[3])):
            if b in bodies:
                np.testing.assert_allclose(qd[:, b, column], rate * dt * steps, rtol=2.0e-3, atol=0.0)

    enabled = np.ones(model.joint_count, bool)
    q, qd = replays(6)
    held(q, range(4))
    driven(qd, 1.0 + 2.0 * np.arange(1, 7), (0, 1, 3))
    for off, on, steps in (((0, 2), (1, 3), 13.0), ((1, 3), (), 25.0)):
        start, start_qd = q[-1], qd[-1]
        enabled[list(off)] = False
        model.joint_enabled.assign(enabled)
        q, qd = replays(6)
        falls(q, qd, start, start_qd, off)
        for b, column in ((0, 5), (1, 0), (3, 5)):  # the speed of a joint that let go stays: its joint_f is gone
            if b in off:
                np.testing.assert_allclose(qd[:, b, column], start_qd[b, column], rtol=0.0, atol=1.0e-4)
        held(q, on)
        driven(qd, steps + 2.0 * np.arange(1, 7), on)


def _bar(builder, link, center, a, b, y):
    """A bar of square section 1 cm from ``a`` to ``b`` (points of the x-z plane) on ``link`` at ``center``, set
    ``y`` off the plane so that bars do not meet."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    length = float(np.linalg.norm(b - a))
    angle = float(np.arctan2(b[2] - a[2], b[0] - a[0]))  # from +x toward +z: a turn of -angle about +y
    mid = 0.5 * (a + b) - center + np.array([0.0, y, 0.0])
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -angle)
    builder.add_shape_box(link, xform=wp.transform(wp.vec3(*mid), rot), hx=0.5 * length, hy=0.005, hz=0.005)


# a four-bar linkage in the x-z plane: ground A-D 0.4 m, crank A-B 0.2 m, coupler B-C 0.5 m, rocker D-C 0.4 m
_A, _D = np.array([0.0, 0.0, 1.0]), np.array([0.4, 0.0, 1.0])
_CRANK, _COUPLER, _ROCKER, _GROUND = 0.2, 0.5, 0.4, 0.4


def _rocker_angle(theta2):
    """The rocker's angle (from +x toward +z) of the open four-bar at the crank's angle ``theta2``: Freudenstein's
    equation solved for it."""
    a, b, c, d = _CRANK, _COUPLER, _ROCKER, _GROUND
    P = 2.0 * c * (a * np.cos(theta2) - d)
    Q = 2.0 * a * c * np.sin(theta2)
    R = a * a + c * c + d * d - b * b - 2.0 * a * d * np.cos(theta2)
    return np.arctan2(Q, P) - np.arccos(R / np.hypot(P, Q))


def _four_bar(device, placed):
    """The four-bar linkage, crank up (90 degrees): world -> crank -> coupler -> rocker as an articulation of revolute
    joints about y, and the revolute joint rocker -> world at D outside it, the loop joint. The links' frames have the
    world's orientation in this pose, so that every joint coordinate is 0 here. ``placed``: the links are built at
    this pose; else without a pose, at the origin, and placed by forward kinematics."""
    theta2 = 0.5 * np.pi
    theta4 = _rocker_angle(theta2)
    B = _A + _CRANK * np.array([np.cos(theta2), 0.0, np.sin(theta2)])
    C = _D + _ROCKER * np.array([np.cos(theta4), 0.0, np.sin(theta4)])
    builder = newton.ModelBuilder()
    links, centers = [], []
    for a, b, y in ((_A, B, -0.03), (B, C, 0.0), (_D, C, 0.03)):
        center = 0.5 * (a + b)
        xform = wp.transform(wp.vec3(*center), wp.quat_identity()) if placed else None
        links.append(builder.add_link(xform=xform))
        centers.append(center)
        _bar(builder, links[-1], center, a, b, y)

    def frame(p):
        return wp.transform(wp.vec3(*p), wp.quat_identity())

    y_axis = wp.vec3(0.0, 1.0, 0.0)
    joints = [
        builder.add_joint_revolute(-1, links[0], axis=y_axis, parent_xform=frame(_A), child_xform=frame(_A - centers[0])),
        builder.add_joint_revolute(links[0], links[1], axis=y_axis, parent_xform=frame(B - centers[0]),
                                   child_xform=frame(B - centers[1])),
        builder.add_joint_revolute(links[1], links[2], axis=y_axis, parent_xform=frame(C - centers[1]),
                                   child_xform=frame(C - centers[2])),
    ]  # fmt: skip
    builder.add_articulation(joints)
    loop = builder.add_joint_revolute(
        -1, links[2], axis=y_axis, parent_xform=frame(_D), child_xform=frame(_D - centers[2])
    )
    model = builder.finalize(device=device)
    return model, links, centers, loop, theta2, theta4


def test_four_bar_linkage_closes_its_loop(test, device):
    """Close a four-bar linkage's loop with a revolute joint outside the articulation.

    A four-bar linkage whose loop is closed by a revolute joint outside the articulation (joint_articulation -1) falls
    from crank-up under gravity as a mechanism of one degree of freedom: over 40 steps of 0.01 s the crank turns by more
    than 0.3 rad, the loop joint holds the rocker's end at its world pivot within 0.5 mm, and the rocker's angle is the
    closed form of Freudenstein's equation at the crank's angle within 2e-3 rad, both angles from the links' rotations.
    joint_q of the crank's joint and of the loop joint are the links' rotations from the start, as newton.eval_ik has
    them. Built without poses, the links sit at the model's origin, and the solve is built at the poses newton.eval_fk
    computes from the joint coordinates, the loop joint placing no body; the steps from the state eval_fk writes are
    those of the linkage built at its pose.
    """
    model, links, centers, loop, theta2_0, theta4_0 = _four_bar(device, placed=True)
    test.assertEqual(int(model.joint_articulation.numpy()[loop]), -1)
    solver = newton.solvers.SolverIPC(model, **_TIGHT)
    s0, s1 = model.state(), model.state()
    body_q, joint_q = [], []
    for _ in range(40):
        solver.step(s0, s1, None, None, 0.01)
        s0, s1 = s1, s0
        body_q.append(s0.body_q.numpy().astype(np.float64))
        joint_q.append(s0.joint_q.numpy().astype(np.float64))
    body_q, joint_q = np.array(body_q), np.array(joint_q)
    turn_crank = _angle_about_y(body_q[:, links[0], 3:])
    turn_rocker = _angle_about_y(body_q[:, links[2], 3:])
    test.assertGreater(np.abs(turn_crank).max(), 0.3)
    # a turn of phi about +y takes an angle measured from +x toward +z to theta - phi
    np.testing.assert_allclose(turn_rocker, theta4_0 - _rocker_angle(theta2_0 - turn_crank), rtol=0.0, atol=2.0e-3)
    # the loop joint's child point at D, where its parent (the world) holds it
    offset = _D - centers[2]
    for q in body_q[:, links[2]]:
        x, w = q[3:6], q[6]
        t = 2.0 * np.cross(x, offset)
        end = q[:3] + offset + w * t + np.cross(x, t)
        np.testing.assert_allclose(end, _D, rtol=0.0, atol=5.0e-4)
    q_start = model.joint_q_start.numpy()
    np.testing.assert_allclose(joint_q[:, q_start[0]], turn_crank, rtol=0.0, atol=1.0e-5)
    np.testing.assert_allclose(joint_q[:, q_start[loop]], turn_rocker, rtol=0.0, atol=1.0e-5)

    unplaced, *_ = _four_bar(device, placed=False)
    np.testing.assert_array_equal(unplaced.body_q.numpy(), [[0, 0, 0, 0, 0, 0, 1]] * 3)
    solver = newton.solvers.SolverIPC(unplaced, **_TIGHT)
    # the solve is built where forward kinematics puts the links, not at the model's origin
    base = solver._scene.layout().body_base[:3]
    np.testing.assert_allclose(solver._engine.positions()[base], np.array(centers), rtol=0.0, atol=1.0e-6)
    s0, s1 = unplaced.state(), unplaced.state()
    newton.eval_fk(unplaced, unplaced.joint_q, unplaced.joint_qd, s0)
    np.testing.assert_allclose(s0.body_q.numpy()[:, :3], np.array(centers), rtol=0.0, atol=1.0e-6)
    for k in range(40):
        solver.step(s0, s1, None, None, 0.01)
        s0, s1 = s1, s0
        np.testing.assert_allclose(s0.body_q.numpy(), body_q[k], rtol=0.0, atol=1.0e-6)


@requires("uipc_warp")
class TestSolverIPCJoints(unittest.TestCase):
    """Revolute, prismatic, fixed and ball joints: how they hold, forces, coordinates, drives and limits, where links
    built without a pose start, joints switched on and off between replays, and a closed loop of joints."""


devices = get_cuda_test_devices()
for name, func in (
    ("test_pendulum_is_implicit_euler", test_pendulum_is_implicit_euler),
    ("test_joint_forces_match_closed_form", test_joint_forces_match_closed_form),
    ("test_fixed_and_ball_joints_hold", test_fixed_and_ball_joints_hold),
    ("test_joint_penalty_follows_the_masses", test_joint_penalty_follows_the_masses),
    ("test_joint_coordinates_are_written", test_joint_coordinates_are_written),
    ("test_grown_lists_keep_joint_angles", test_grown_lists_keep_joint_angles),
    ("test_drives_are_implicit_euler", test_drives_are_implicit_euler),
    ("test_joint_limits_hold", test_joint_limits_hold),
    ("test_drive_targets_change_inside_a_captured_graph", test_drive_targets_change_inside_a_captured_graph),
    ("test_unplaced_links_start_at_forward_kinematics", test_unplaced_links_start_at_forward_kinematics),
    ("test_joint_enabled_is_read_at_every_replay", test_joint_enabled_is_read_at_every_replay),
    ("test_four_bar_linkage_closes_its_loop", test_four_bar_linkage_closes_its_loop),
):
    add_function_test(TestSolverIPCJoints, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
