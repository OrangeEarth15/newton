# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC's ball joints with drives, springs, damping, armature and friction: libuipc-warp's ball terms
(``features/ball.py``).

* Newton's own tests of ball-joint drives, which it runs for SolverMuJoCo, run with SolverIPC: position targets about
  x, y and z and velocity targets, also with a turned child frame; and SolverSteel's ball-joint damping test against
  SolverFeatherstone.
* Closed forms on a sphere that turns about the z axis of its ball joint, held at its centre: a drive, a spring
  toward a rest, damping and an armature follow implicit Euler's recurrence of the angle as a revolute joint's do, and
  BDF2's under ``integrator="bdf2"``; a drive at its effort limit is the constant torque of the limit; friction holds a
  smaller ``joint_f`` at libuipc's creep rate and lets a larger one turn the joint at ``(tau - F) / I``, also at the
  Newton tolerances a model with joint friction takes by default, about a coordinate axis and a diagonal.
* ``BODY_PARENT_F`` of a box that a ball joint's drive holds out against gravity is its momentum balance at every
  step, the drive's torque included (``features.ball.parent_terms``).
* The solve: captured steps equal eager ones; a step solved again on lists grown after an overflow takes the step's
  terms; gains given to a ball joint after the solver was built take effect after ``notify_model_changed``; a disabled
  joint takes no terms; two independent worlds step as each alone.
* What stays refused: a ball joint's limits, which Newton does not define, the terms of a D6 joint that becomes a
  ball joint, and a free joint's armature and drives.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.ipc._fixtures import (
    _TIGHT_ROTATIONS,
    _TINY_LISTS,
    _assert_refused,
    _box_body,
    _drive_recurrence,
    _drive_recurrence_bdf2,
    _model_of,
    _with_d6,
    _with_free,
)
from newton.tests.ipc.test_ipc_observables import _momentum_balance
from newton.tests.ipc.test_ipc_observables import _steps as _observed_steps
from newton.tests.test_joint_controllers import test_ball_controller, test_ball_controller_rotated_anchor
from newton.tests.test_joint_damping import _simulate_ball_joint_damping
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires

_KIND = newton.solvers.SolverBase.ObservableKind
# ten times tighter than _TIGHT_ROTATIONS (1e-7 m/s and 1e-7 1/s): the recurrences of the angle are compared at 1e-6 of
# the motion and the friction's creep at 1e-5
_TIGHTER = {**_TIGHT_ROTATIONS, "newton_velocity_tol": 1.0e-7, "newton_transrate_tol": 1.0e-7}
_H = 0.01


def _quat_z(angle: float) -> list:
    """The quaternion (x, y, z, w) of a turn by ``angle`` about z."""
    return [0.0, 0.0, float(np.sin(0.5 * angle)), float(np.cos(0.5 * angle))]


def _sphere(device, omega=0.0, gravity=0.0, frame=None, **ball):
    """A sphere of radius 0.1 m (4.19 kg) whose ball joint holds it at its centre, 1 m above the world's origin, turning
    at ``omega`` about z of the joint frame, which ``frame`` (a quaternion, x y z w) turns in the world; ``ball``:
    ``add_joint_ball``'s arguments. Returns the model, a state at its joint coordinates and the sphere's moment of
    inertia."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -gravity), up_axis=newton.Axis.Z)
    link = builder.add_link()
    builder.add_shape_sphere(link, radius=0.1)
    turn = wp.quat_identity() if frame is None else wp.quat(*frame)
    joint = builder.add_joint_ball(-1, link, parent_xform=wp.transform((0.0, 0.0, 1.0), turn), **ball)
    builder.add_articulation([joint])
    builder.joint_qd[builder.joint_qd_start[joint] + 2] = omega
    model = builder.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    return model, state, float(model.body_inertia.numpy()[0][2, 2])


def _angles(model, solver, state, steps, control=None, h=_H):
    """The joint's turn about z after each of ``steps`` steps, from its quaternion, unwrapped."""
    s0, s1 = state, model.state()
    out = []
    for _ in range(steps):
        solver.step(s0, s1, control, None, h)
        s0, s1 = s1, s0
        q = s0.joint_q.numpy()[:4]
        out.append(2.0 * np.arctan2(q[2], q[3]))
    return np.unwrap(np.array(out))


def _set(model, **arrays):
    """Write the per-DOF or per-coordinate model arrays ``joint_<name>`` (values for all, or one per entry)."""
    for name, value in arrays.items():
        a = getattr(model, f"joint_{name}")
        a.assign(np.broadcast_to(np.asarray(value, np.float32), a.shape).copy())


def test_newton_ball_drive_tests(test, device):
    """Newton's ball-joint controller tests of SolverMuJoCo (test_joint_controllers), with SolverIPC: a quarter turn
    about x, y and z as a position target, a quarter turn per second about each as a velocity target, and both with a
    child frame turned 30 degrees about y."""
    quarter = np.pi / 2.0
    for axis, k in (("x", 0), ("y", 1), ("z", 2)):
        target = [0.0, 0.0, 0.0]
        target[k] = quarter
        quat = [0.0, 0.0, 0.0, 0.7071068]
        quat[k] = 0.7071068
        with test.subTest(position=axis):
            test_ball_controller(
                test, device, newton.solvers.SolverIPC, target, [0.0] * 3, quat, [0.0] * 3, 2000.0, 500.0
            )
        with test.subTest(velocity=axis):
            test_ball_controller(test, device, newton.solvers.SolverIPC, [0.0] * 3, target, None, target, 0.0, 500.0)
    turned = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 6.0)
    with test.subTest(position="x, turned child frame"):
        test_ball_controller_rotated_anchor(
            test, device, newton.solvers.SolverIPC, turned, [quarter, 0.0, 0.0], [0.7071068, 0.0, 0.0, 0.7071068],
            [0.0] * 3, 2000.0, 500.0,
        )  # fmt: skip
    for axis, k in (("x", 0), ("y", 1), ("z", 2)):
        velocity = [0.0, 0.0, 0.0]
        velocity[k] = quarter
        with test.subTest(velocity=f"{axis}, turned child frame"):
            test_ball_controller_rotated_anchor(
                test, device, newton.solvers.SolverIPC, turned, [0.0] * 3, None, velocity, 0.0, 500.0
            )


def test_ball_damping_against_featherstone(test, device):
    """SolverSteel's ball-joint damping test with SolverIPC: without damping the joint keeps its speed, with
    ``joint_damping`` 3 it slows by more than 15%, and its final speed is SolverFeatherstone's within 2%, about every
    axis and with a turned parent frame."""
    turned = wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.3))

    def featherstone(model):
        return newton.solvers.SolverFeatherstone(model, angular_damping=0.0)

    for initial_qd, parent_xform in (
        ((1.0, 0.0, 0.0), None),
        ((0.0, 1.0, 0.0), None),
        ((0.0, 0.0, 1.0), None),
        ((0.5, -0.25, 1.0), None),
        ((0.5, -0.25, 1.0), turned),
    ):
        with test.subTest(initial_qd=initial_qd, turned=parent_xform is not None):
            kwargs = {"initial_qd": initial_qd, "parent_xform": parent_xform}
            free_initial, free_final = _simulate_ball_joint_damping(
                device, newton.solvers.SolverIPC, damping=0.0, **kwargs
            )
            damped_initial, damped_final = _simulate_ball_joint_damping(device, newton.solvers.SolverIPC, damping=3.0, **kwargs)  # fmt: skip
            _, reference = _simulate_ball_joint_damping(device, featherstone, damping=3.0, **kwargs)
            np.testing.assert_allclose(free_final, free_initial, atol=1.0e-5, rtol=1.0e-5)
            test.assertLess(damped_final, damped_initial * 0.85)
            np.testing.assert_allclose(damped_final, reference, rtol=0.02)


def test_terms_are_implicit_euler(test, device):
    """A sphere turning about z on its ball joint: a drive toward a target with a velocity target, a spring toward a
    rest, damping and an armature follow implicit Euler's recurrence of the angle with the inertia ``I + a``, as a
    revolute joint's terms do, to 2e-5 of the motion over 150 steps, from 2 rad/s toward 0.6 rad; and BDF2's (started by
    an implicit Euler step) under ``integrator="bdf2"``, 30 times slower and smaller, since BDF2 predicts a turning
    body along a straight line, which loses about ``(omega dt)^2`` of its spin per step. (Under implicit Euler the solver
    turns a body as a rigid one, whose kinetic term of a turn ``delta`` off its flight is ``I (1 - cos(delta))``.) The
    armature changes the motion by more than 1% of it."""
    gains = {"target_ke": 2.0, "target_kd": 0.05, "stiffness": 1.0, "damping": 0.03}
    for integrator, recurrence, scale in (
        ("bdf1", _drive_recurrence, 1.0),
        ("bdf2", _drive_recurrence_bdf2, 1.0 / 30.0),
    ):
        for armature in (0.0, 0.02):
            with test.subTest(integrator=integrator, armature=armature):
                model, state, inertia = _sphere(device, omega=2.0 * scale, armature=armature)
                _set(model, **gains)
                model.joint_rest_q.assign(np.array(_quat_z(0.2 * scale), np.float32))
                control = model.control()
                control.joint_target_q.assign(np.array(_quat_z(0.6 * scale), np.float32))
                control.joint_target_qd.assign(np.array([0.0, 0.0, 0.3 * scale], np.float32))
                solver = newton.solvers.SolverIPC(model, integrator=integrator, **_TIGHTER)
                got = _angles(model, solver, state, 150, control)
                args = {
                    "ke": 2.0, "kd": 0.05, "ks": 1.0, "c": 0.03, "q_target": 0.6 * scale, "qd_target": 0.3 * scale,
                    "q_rest": 0.2 * scale, "qd0": 2.0 * scale,
                }  # fmt: skip
                want = recurrence(inertia + armature, _H, 150, **args)
                motion = np.abs(want).max()
                np.testing.assert_allclose(got, want, rtol=0.0, atol=2e-5 * motion)
                if armature:
                    test.assertGreater(np.abs(recurrence(inertia, _H, 150, **args) - want).max(), 1e-2 * motion)


def test_effort_limit(test, device):
    """A drive of 100 N m/rad toward 3 rad about z with an effort limit of 0.05 N m: the joint turns under the constant
    torque 0.05 N m from the first step, as implicit Euler's recurrence of it says; with
    ``enable_joint_effort_limit=False`` the solver refuses the limit, as for a revolute joint."""
    model, state, inertia = _sphere(device)
    _set(model, target_ke=100.0, effort_limit=0.05)
    control = model.control()
    control.joint_target_q.assign(np.array(_quat_z(3.0), np.float32))
    got = _angles(model, newton.solvers.SolverIPC(model, **_TIGHTER), state, 60, control)
    want = _drive_recurrence(inertia, _H, 60, f=0.05)
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-6 * np.abs(want).max())
    with test.assertRaisesRegex(NotImplementedError, "joint_effort_limit 0.05.* and a drive"):
        newton.solvers.SolverIPC(model, enable_joint_effort_limit=False)


def test_friction(test, device):
    """``joint_friction`` 0.1 N m on the ball joint against a ``joint_f`` about z: 0.05 N m does not turn the joint but
    for libuipc's creep below ``eps_velocity``, ``eps_v (1 - sqrt(1 - tau / F))`` rad/s, to 1e-5 of it; 0.2 N m turns
    it faster by ``h (tau - F) / I`` at every step once it slides, to 2e-5 (the float32 ``joint_qd``)."""
    for tau in (0.05, 0.2):
        with test.subTest(joint_f=tau):
            model, state, inertia = _sphere(device, friction=0.1)
            control = model.control()
            control.joint_f.assign(np.array([0.0, 0.0, tau], np.float32))
            solver = newton.solvers.SolverIPC(model, **_TIGHTER)
            s0, s1 = state, model.state()
            omega = []
            for _ in range(40):
                solver.step(s0, s1, control, None, _H)
                s0, s1 = s1, s0
                omega.append(float(s0.joint_qd.numpy()[2]))
            omega = np.array(omega)
            eps_v = float(solver._config.eps_velocity)
            if tau < 0.1:
                np.testing.assert_allclose(omega[-5:], eps_v * (1.0 - np.sqrt(1.0 - tau / 0.1)), rtol=1e-5)
            else:
                np.testing.assert_allclose(np.diff(omega)[10:], _H * (tau - 0.1) / inertia, rtol=2e-5)


def test_friction_slides_at_the_default_tolerances(test, device):
    """``joint_friction`` 0.1 N m and a ``joint_f`` of 1.5 times it about z of the joint frame, along a coordinate axis
    of the world and along its diagonal: with the Newton tolerances a model with joint friction takes by default, 0.4
    ``eps_velocity`` (``features.joint_terms``), the joint slides, turning faster by ``h (tau - F) / I`` at every step
    to 1%. At SolverIPC's defaults for a model without friction the loop would end inside the friction's smoothing band
    and the joint would creep."""
    d = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    axis = np.cross([0.0, 0.0, 1.0], d)
    half = 0.5 * np.arccos(d[2])
    diagonal = [*(np.sin(half) * axis / np.linalg.norm(axis)), np.cos(half)]
    for name, frame in (("a coordinate axis", None), ("a diagonal", diagonal)):
        with test.subTest(name):
            model, state, inertia = _sphere(device, frame=frame, friction=0.1)
            control = model.control()
            control.joint_f.assign(np.array([0.0, 0.0, 0.15], np.float32))
            solver = newton.solvers.SolverIPC(model)
            s0, s1 = state, model.state()
            omega = []
            for _ in range(12):
                solver.step(s0, s1, control, None, _H)
                s0, s1 = s1, s0
                omega.append(float(s0.joint_qd.numpy()[2]))
            np.testing.assert_allclose(np.diff(omega)[1:], _H * (0.15 - 0.1) / inertia, rtol=1e-2)
            eps_v = float(solver._config.eps_velocity)
            tolerances = (solver._config.newton_velocity_tol, solver._config.newton_transrate_tol)
            test.assertEqual(tolerances, (0.4 * eps_v, 0.4 * eps_v))


def _held_out(device):
    """A 0.2 m box of 8 kg, z up under gravity, whose ball joint at the world's origin holds its centre 0.5 m out along x;
    its drive (5000 N m/rad, 200 N m s/rad toward the identity) holds it out."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis=newton.Axis.Z)
    link = builder.add_link()
    builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
    joint = builder.add_joint_ball(
        -1, link, child_xform=wp.transform((-0.5, 0.0, 0.0), wp.quat_identity()), damping=0.0
    )
    builder.add_articulation([joint])
    builder.joint_target_ke[0:3] = [5000.0] * 3
    builder.joint_target_kd[0:3] = [200.0] * 3
    return builder.finalize(device=device)


def test_parent_force_holds_the_drive_torque(test, device):
    """BODY_PARENT_F of the box a ball joint's drive holds out: at every step of 20 its momentum balance, the force
    within 1e-6 of the weight and the torque within 1e-5 N m. About the box's centre the anchor's force alone turns it
    by 0.5 m times its weight, 39 N m, which the drive's torque, part of the joint's wrench, holds."""
    model = _held_out(device)
    solver = newton.solvers.SolverIPC(model, **_TIGHTER)
    observables = solver.observables(kinds={_KIND.BODY_PARENT_F})
    steps, _ = _observed_steps(model, solver, 20, 0.005, observables, control=model.control())
    weight = float(model.body_mass.numpy()[0]) * 9.81
    for q0, qd0, q1, qd1, parent_f, _ in steps:
        balance = _momentum_balance(model, q0, qd0, q1, qd1, 0.005, gravity=(0.0, 0.0, -9.81))
        np.testing.assert_allclose(parent_f[:, :3], balance[:, :3], rtol=0.0, atol=1e-6 * weight)
        np.testing.assert_allclose(parent_f[:, 3:], balance[:, 3:], rtol=0.0, atol=1e-5)


def test_captured_steps_equal_eager_steps(test, device):
    """Steps of a sphere with every term, captured in a graph with targets written between replays, as eager steps,
    to 1e-6."""

    def run(capture):
        model, state, _ = _sphere(device, omega=1.0, armature=0.01, friction=0.02)
        _set(model, target_ke=2.0, target_kd=0.1, stiffness=0.5, damping=0.02, effort_limit=0.4)
        solver = newton.solvers.SolverIPC(model, **_TIGHTER)
        control = model.control()
        s0, s1 = state, model.state()
        solver.step(s0, s1, control, None, _H)  # loads the kernels
        s0, s1 = s1, s0
        graph = None
        if capture:
            with wp.ScopedCapture(device=device) as cap:
                solver.step(s0, s1, control, None, _H)
                solver.step(s1, s0, control, None, _H)
            graph = cap.graph
        out = []
        for k in range(15):
            control.joint_target_q.assign(np.array(_quat_z(0.5 * np.sin(0.4 * k)), np.float32))
            if capture:
                wp.capture_launch(graph)
            else:
                for _ in range(2):
                    solver.step(s0, s1, control, None, _H)
                    s0, s1 = s1, s0
            out.append(s0.body_q.numpy()[0])
        return np.array(out)

    eager, captured = run(False), run(True)
    np.testing.assert_allclose(captured, eager, rtol=0.0, atol=1e-6)


def test_grown_lists_keep_the_step_terms(test, device):
    """A sphere turns on a driven ball joint while a box drops onto a static box; when it lands the contact lists, made
    short, overflow, and the eager step grows them and solves the step again with the step's ball terms: the sphere's
    joint goes on as in a solver with room, to 1e-6 rad, and so does every body."""

    def run(tiny):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81), up_axis=newton.Axis.Z)
        builder.add_shape_box(-1, xform=wp.transform((1.0, 0.0, 0.0), wp.quat_identity()), hx=0.3, hy=0.3, hz=0.1)
        link = builder.add_link()
        builder.add_shape_sphere(link, radius=0.1)
        joint = builder.add_joint_ball(-1, link, parent_xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
        builder.add_articulation([joint])
        builder.joint_target_ke[0:3] = [2.0] * 3
        builder.joint_target_kd[0:3] = [0.05] * 3
        _box_body(builder, (1.0, 0.0, 0.3), (0.05, 0.05, 0.05))
        model = builder.finalize(device=device)
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        solver = newton.solvers.SolverIPC(model, **_TIGHTER)
        regrown = []
        regrow = solver._regrow
        solver._regrow = lambda: (regrown.append(1), regrow())
        if tiny:
            solver._make_engine(_TINY_LISTS)
        control = model.control()
        target = control.joint_target_q.numpy()
        target[0:4] = _quat_z(1.5)
        control.joint_target_q.assign(target)
        s0, s1 = state, model.state()
        out = []
        for _ in range(40):
            solver.step(s0, s1, control, None, _H)
            s0, s1 = s1, s0
            out.append(s0.body_q.numpy())
        return np.array(out), len(regrown)

    (roomy, roomy_regrown), (grown, grown_regrown) = run(False), run(True)
    test.assertLess(roomy[-1, 1, 2], 0.2)  # the box has landed on the static box (top at 0.1 m)
    test.assertEqual(roomy_regrown, 0)
    test.assertGreater(grown_regrown, 0)
    np.testing.assert_allclose(grown, roomy, rtol=0.0, atol=1e-6)


def test_gains_given_after_the_solver_was_built(test, device):
    """A ball joint without terms when the solver was built gets its drive after ``notify_model_changed`` with
    ``JOINT_DOF_PROPERTIES``, which builds the solve again: the joint then reaches its target, a quarter turn about z,
    to 1e-4 rad."""
    model, state, _ = _sphere(device)
    solver = newton.solvers.SolverIPC(model, **_TIGHTER)
    control = model.control()
    control.joint_target_q.assign(np.array(_quat_z(np.pi / 2.0), np.float32))
    test.assertAlmostEqual(_angles(model, solver, state, 5, control)[-1], 0.0, delta=1e-9)
    _set(model, target_ke=5.0, target_kd=0.5)
    solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
    state2 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state2)
    test.assertAlmostEqual(_angles(model, solver, state2, 400, control)[-1], np.pi / 2.0, delta=1e-4)


def test_a_disabled_joint_has_no_terms(test, device):
    """A ball joint with a drive toward a quarter turn and ``joint_enabled`` False: neither its anchor nor its terms act,
    so the sphere stays as it was; enabled again, the drive turns it toward the target."""
    model, state, _ = _sphere(device)
    _set(model, target_ke=5.0, target_kd=0.5)
    model.joint_enabled.assign(np.array([False]))
    control = model.control()
    control.joint_target_q.assign(np.array(_quat_z(np.pi / 2.0), np.float32))
    solver = newton.solvers.SolverIPC(model, **_TIGHTER)
    start = state.body_q.numpy().copy()
    s0, s1 = state, model.state()
    for _ in range(10):
        solver.step(s0, s1, control, None, _H)
        s0, s1 = s1, s0
    np.testing.assert_allclose(s0.body_q.numpy(), start, rtol=0.0, atol=1e-7)
    model.joint_enabled.assign(np.array([True]))
    test.assertGreater(_angles(model, solver, s0, 20, control)[-1], 0.1)


def test_independent_worlds(test, device):
    """Two worlds of a sphere on a driven ball joint, the second's drive twice as stiff, solved as independent worlds:
    each world's joint angles as those of the same model alone, to 1e-9 rad."""

    def model_of(stiffness: list):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
        for k, ke in enumerate(stiffness):
            builder.begin_world()
            link = builder.add_link()
            builder.add_shape_sphere(link, radius=0.1)
            joint = builder.add_joint_ball(
                -1, link, parent_xform=wp.transform((3.0 * k, 0.0, 1.0), wp.quat_identity()), damping=0.05
            )
            builder.add_articulation([joint])
            builder.joint_target_ke[-3:] = [ke] * 3
            builder.end_world()
        model = builder.finalize(device=device)
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        return model, state

    def angles(model, state, worlds):
        control = model.control()
        target = np.tile(np.array(_quat_z(1.0), np.float32), worlds)
        control.joint_target_q.assign(target)
        solver = newton.solvers.SolverIPC(model, **_TIGHTER)
        s0, s1 = state, model.state()
        out = []
        for _ in range(30):
            solver.step(s0, s1, control, None, _H)
            s0, s1 = s1, s0
            q = s0.joint_q.numpy().reshape(worlds, 4)
            out.append(2.0 * np.arctan2(q[:, 2], q[:, 3]))
        return np.array(out)

    together = angles(*model_of([2.0, 4.0]), 2)
    for k, ke in enumerate((2.0, 4.0)):
        alone = angles(*model_of([ke]), 1)
        np.testing.assert_allclose(together[:, k], alone[:, 0], rtol=0.0, atol=1e-9)


def test_refusals(test, device):
    """A ball joint's limits are refused (Newton defines none); the terms of a D6 joint of three angular axes, which
    becomes a ball joint, are refused (its coordinates are angles about its axes); a free joint's armature and drives
    stay refused."""
    D = newton.ModelBuilder.JointDofConfig

    def one(add, **dof):
        builder = newton.ModelBuilder()
        link = builder.add_link(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
        builder.add_articulation([add(builder, link)])
        for name, value in dof.items():
            getattr(builder, f"joint_{name}")[0] = value
        return builder.finalize(device=device)

    def ball(builder, link):
        return builder.add_joint_ball(-1, link, parent_xform=wp.transform((0.0, 0.0, 1.2), wp.quat_identity()))

    def d6(builder, link):
        return builder.add_joint_d6(-1, link, angular_axes=[D(axis=a) for a in "XYZ"])

    for name, model, message in (
        (
            "ball limit",
            one(ball, limit_lower=-0.5, limit_upper=0.5, limit_ke=10.0),
            "SolverIPC: DOF 0 of the BALL joint 0 has the limits [-0.5, 0.5] with limit_ke 10.0; set limit_ke to 0, or "
            "the bounds to -MAXVAL and MAXVAL. Newton defines no limits of a ball joint: add_joint_ball takes none, its "
            "importers drop them and its solvers do not read them.",
        ),
        ("D6 friction", one(d6, friction=0.25), "D6 joint 0 has joint_friction 0.25; set it to 0. SolverIPC takes"),
        ("D6 drive", one(d6, target_ke=5.0), "D6 joint 0 has target_ke 5.0, target_kd 0.0, stiffness 0.0 and damping"),
        ("D6 armature", one(d6, armature=0.5), "D6 joint 0 has joint_armature 0.5; set it to 0. SolverIPC takes"),
    ):
        with test.subTest(name):
            with test.assertRaises(NotImplementedError) as caught:
                newton.solvers.SolverIPC(model)
            test.assertIn(message, str(caught.exception))
    drive = (
        "set the gains, stiffness and damping to 0 and the target mode to NONE, and apply forces through "
        "Control.joint_f, which acts on every joint. SolverIPC drives, springs and damps revolute, prismatic, ball and "
        "universal joints only"
    )
    d6_ball = (
        "SolverIPC takes the drives, springs, damping, armature and friction of a BALL joint, on the rotation vector of "
        "its rotation, and those of no D6 joint of three angular axes, whose coordinates are angles about its axes"
    )
    for name, content, message in (
        (
            "free armature",
            _with_free(5, armature=0.5),
            "SolverIPC: DOF 5 of the FREE joint 0 has joint_armature 0.5; set it to 0. SolverIPC takes the armature of "
            "revolute, prismatic, ball and universal joints only; that of a free joint, a mass without weight and an "
            "inertia, has no term.",
        ),
        (
            "free drive",
            _with_free(0, target_ke=10.0),
            "SolverIPC: DOF 0 of the FREE joint 0 has target_ke 10.0, target_kd 0.0, stiffness 0.0 and damping 0.0; "
            f"{drive}.",
        ),
        (
            "free target mode",
            _with_free(2, target_mode=int(newton.JointTargetMode.POSITION)),
            f"SolverIPC: DOF 2 of the FREE joint 0 has the target mode POSITION; {drive}.",
        ),
        (
            "D6 damping",
            _with_d6("", "XYZ", 1, damping=2.0),
            "SolverIPC: DOF 1 of the D6 joint 0 has target_ke 0.0, target_kd 0.0, stiffness 0.0 and damping 2.0; set "
            f"the gains, stiffness and damping to 0 and the target mode to NONE. {d6_ball}.",
        ),
        (
            "D6 limits",
            _with_d6("", "XYZ", 0, limit_lower=-0.5, limit_upper=0.5, limit_ke=100.0),
            "SolverIPC: DOF 0 of the D6 joint 0 has the limits [-0.5, 0.5] with limit_ke 100.0; set limit_ke to 0, or "
            "the bounds to -MAXVAL and MAXVAL. libuipc limits revolute and prismatic joints only.",
        ),
    ):
        with test.subTest(name):
            _assert_refused(test, _model_of(device, content), message)


@requires("uipc_warp")
class TestSolverIPCBall(unittest.TestCase):
    """Ball joints with drives, springs, damping, armature and friction."""


devices = get_cuda_test_devices()
for name, func in (
    ("test_newton_ball_drive_tests", test_newton_ball_drive_tests),
    ("test_ball_damping_against_featherstone", test_ball_damping_against_featherstone),
    ("test_terms_are_implicit_euler", test_terms_are_implicit_euler),
    ("test_effort_limit", test_effort_limit),
    ("test_friction", test_friction),
    ("test_friction_slides_at_the_default_tolerances", test_friction_slides_at_the_default_tolerances),
    ("test_parent_force_holds_the_drive_torque", test_parent_force_holds_the_drive_torque),
    ("test_captured_steps_equal_eager_steps", test_captured_steps_equal_eager_steps),
    ("test_grown_lists_keep_the_step_terms", test_grown_lists_keep_the_step_terms),
    ("test_gains_given_after_the_solver_was_built", test_gains_given_after_the_solver_was_built),
    ("test_a_disabled_joint_has_no_terms", test_a_disabled_joint_has_no_terms),
    ("test_independent_worlds", test_independent_worlds),
    ("test_refusals", test_refusals),
):
    add_function_test(TestSolverIPCBall, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
