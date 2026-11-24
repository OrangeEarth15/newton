# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC's terms on joint coordinates (``features/joint_terms.py``): mimic joints, MuJoCo's fixed tendons and the
friction of revolute and prismatic joints, as libuipc-warp's joint coordinate terms.

* Mimic joints: Newton's scalar case of ``test_solver_mimic`` (a prismatic joint following a revolute one), the
  deprecated ``constraint_mimic_*`` rows with their enabled flags and ``joint_enabled``, the penalty's give under a
  load against ``tau dt^2 / K``, and the relation's coefficients changed in place by ``notify_model_changed``.
* Joint friction: a torque or force below the friction creeps at the smoothed friction's steady rate, one above it
  accelerates at ``(tau - F) / I``, also at the default tolerances, which a model with friction on any joint takes at
  0.4 ``eps_velocity`` (a tolerance given above it is kept and warned of); ``BODY_PARENT_F`` holds the friction of the
  joint, and an engine grown again keeps the step's friction.
* MuJoCo's JOINT equality rows: held where they repeat the mimic row the importer made of them, refused otherwise.
* Fixed tendons: the scene of Newton's ``test_fixed_tendon`` (two sliders, a tendon of stiffness 2 and damping 1
  toward 0) against implicit Euler's recurrence, in one world and in two, and a tendon's limit at its closed-form
  overshoot with the stiffness MuJoCo's solref gives.
* What stays refused, by name: Newton's own tendons, MuJoCo's spatial tendons and tendon actuators, a mimic of joints
  of several coordinates, a tendon's limit margin, and the friction of free joints.
"""

import math
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.solvers.ipc.features import joint_terms
from newton.tests.ipc._fixtures import _TIGHT_ROTATIONS, _TINY_LISTS, _box_body
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires

# _TIGHT_ROTATIONS lies far below the friction's smoothing band (eps_velocity dt): the steps converge to their minimum
_EPS_V = 0.01  # SolverIPC's eps_velocity [m/s, rad/s]
_KIND = newton.solvers.SolverBase.ObservableKind


def _steps(model, solver, steps, dt, control=None, observables=None):
    """``joint_q`` after every step (and the observables' BODY_PARENT_F when requested)."""
    s0, s1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    q, parent = [], []
    for _ in range(steps):
        solver.step(s0, s1, control, None, dt, observables=observables)
        s0, s1 = s1, s0
        q.append(s0.joint_q.numpy().astype(np.float64))
        if observables is not None:
            parent.append(observables.body_parent_f.numpy().astype(np.float64))
    return np.array(q), np.array(parent)


def _hinged_box(device, kind="revolute", friction=0.0, half=0.1, axis=None):
    """A box of ``half`` half extents on a revolute (about z) or prismatic (along x) joint to the world, through its
    centre, or along ``axis``, without gravity; the model and the box's inertia about the axis (mass for a prismatic
    joint)."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    body = builder.add_link()
    builder.add_shape_box(body, hx=half, hy=half, hz=half)
    if axis is None:
        axis = (0.0, 0.0, 1.0) if kind == "revolute" else (1.0, 0.0, 0.0)
    add = builder.add_joint_revolute if kind == "revolute" else builder.add_joint_prismatic
    builder.add_articulation([add(-1, body, axis=wp.vec3(*axis), friction=friction)])
    model = builder.finalize(device=device)
    n = np.asarray(axis, np.float64)
    if kind == "revolute":
        inertia = float(n @ model.body_inertia.numpy()[0].astype(np.float64) @ n)
    else:
        inertia = float(model.body_mass.numpy()[0])
    return model, inertia


def test_mimic_scalar_case(test, device):
    """Newton's scalar mimic case of test_solver_mimic: a prismatic follower of a revolute reference with offset 0.1 and
    multiplier -1.5, which starts 1.2 off the relation; after a second of steps of 1/60 s the relation holds within
    Newton's 2e-3, here within 1e-6, and the reference has taken its share of the correction."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    reference_body, follower_body = builder.add_link(), builder.add_link()
    for body in (reference_body, follower_body):
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    reference = builder.add_joint_revolute(-1, reference_body, axis=newton.Axis.Z)
    # the follower's joint 1 m off the reference's (Newton's case puts both at the origin, where SolverIPC's boxes would
    # start inside each other's contact thickness)
    beside = wp.transform(wp.vec3(0.0, 1.0, 0.0), wp.quat_identity())
    follower = builder.add_joint_prismatic(-1, follower_body, parent_xform=beside, axis=newton.Axis.X)
    builder.add_articulation([reference, follower])
    builder.set_joint_mimic(follower, reference, coeffs=(0.1, -1.5))
    model = builder.finalize(device=device)
    model.joint_q.assign(np.array([0.35, 0.8], np.float32))
    q, _ = _steps(model, newton.solvers.SolverIPC(model), 60, 1.0 / 60.0)
    test.assertLess(abs(q[-1, 1] - (0.1 - 1.5 * q[-1, 0])), 1e-6)
    test.assertGreater(abs(q[-1, 0] - 0.35), 1e-3)


def _two_hinges(device, **mimic):
    """Two boxes on revolute joints about z, each through its centre, the second a mimic of the first by
    ``set_joint_mimic`` or a deprecated mimic row (``row=True``); torques act through ``joint_f``."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    bodies = []
    for x in (-0.5, 0.5):
        b = builder.add_link(xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_box(b, hx=0.1, hy=0.1, hz=0.1)
        bodies.append(b)
    joints = [
        builder.add_joint_revolute(-1, b, parent_xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
                                   axis=newton.Axis.Z)
        for b, x in zip(bodies, (-0.5, 0.5), strict=True)
    ]  # fmt: skip
    builder.add_articulation(joints)
    coeffs = (mimic.get("offset", 0.0), mimic.get("multiplier", 1.0))
    if mimic.get("row"):
        with warnings.catch_warnings():  # the deprecated rows are what this case holds
            warnings.simplefilter("ignore", DeprecationWarning)
            builder.add_constraint_mimic(joints[1], joints[0], *coeffs)
    else:
        builder.set_joint_mimic(joints[1], joints[0], coeffs=coeffs)
    return builder.finalize(device=device)


def test_mimic_penalty_and_enabled_flags(test, device):
    """A follower held to -1.5 times its reference under the torques 2 N m on it and 3 N m on the reference (whose
    resultant along the relation is 0): off by the penalty's ``tau dt^2 / K``, ``K = joint_strength_ratio`` times the
    two boxes' masses, by set_joint_mimic and by a deprecated row. A row disabled, or a joint disabled, lets the
    follower turn alone."""
    dt = 0.01
    for row in (False, True):
        with test.subTest(row=row):
            model = _two_hinges(device, multiplier=-1.5, row=row)
            control = model.control()
            control.joint_f.assign(np.array([3.0, 2.0], np.float32))
            solver = newton.solvers.SolverIPC(model, **_TIGHT_ROTATIONS)
            q, _ = _steps(model, solver, 30, dt, control)
            K = solver._joint_strength_ratio * float(model.body_mass.numpy().sum())
            test.assertAlmostEqual((q[-1, 1] + 1.5 * q[-1, 0]) / (2.0 * dt * dt / K), 1.0, delta=1e-5)
            if row:  # a disabled row: the follower turns alone
                model.constraint_mimic_enabled.assign(np.array([False]))
                solver.notify_model_changed(newton.ModelFlags.CONSTRAINT_PROPERTIES)
                q2, _ = _steps(model, solver, 10, dt, control)
                test.assertGreater(abs(q2[-1, 1] + 1.5 * q2[-1, 0]), 0.1)
            else:  # a disabled joint: the relation is off at the next step
                model.joint_enabled.assign(np.array([True, False]))
                _steps(model, solver, 1, dt, control)
                test.assertEqual(solver._joint_terms.terms.on.numpy().tolist(), [0])


def test_mimic_coefficients_change_in_place(test, device):
    """New coefficients of a mimic relation (CONSTRAINT_PROPERTIES) take effect at the next steps without building the
    solve again, also in a graph captured before."""
    model = _two_hinges(device, multiplier=0.5)
    control = model.control()
    control.joint_f.assign(np.array([1.0, 0.0], np.float32))
    solver = newton.solvers.SolverIPC(model, **_TIGHT_ROTATIONS)
    s0, s1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    solver.step(s0, s1, control, None, 0.01)
    solver.step(s1, s0, control, None, 0.01)
    engine = solver._engine
    with wp.ScopedCapture(device) as capture:  # two steps, from s0 to s1 and back
        solver.step(s0, s1, control, None, 0.01)
        solver.step(s1, s0, control, None, 0.01)
    for _ in range(10):
        wp.capture_launch(capture.graph)
    q = s0.joint_q.numpy()
    test.assertAlmostEqual(q[1], 0.5 * q[0], delta=1e-5)
    coeffs = model.joint_mimic_coeffs.numpy()
    coeffs[1] = (0.2, -2.0)
    model.joint_mimic_coeffs.assign(coeffs)
    solver.notify_model_changed(newton.ModelFlags.CONSTRAINT_PROPERTIES)
    test.assertIs(solver._engine, engine)
    for _ in range(20):
        wp.capture_launch(capture.graph)
    q = s0.joint_q.numpy()
    test.assertAlmostEqual(q[1], 0.2 - 2.0 * q[0], delta=1e-5)


# the friction of the hinged box (N m) and of the sliding one (N), in proportion to their inertia (0.0533 kg m^2 and
# 8 kg) within a factor of two: under 1.5 times its friction either leaves the smoothing band in its first step
_FRICTION = {"revolute": 2.0, "prismatic": 200.0}


def test_joint_friction(test, device):
    """A box on a revolute or prismatic joint with a friction ``F`` (``_FRICTION``): under a joint_f of 0.25 and 0.95
    times it the joint settles to the smoothed friction's steady creep, ``eps_v (1 - sqrt(1 - f / F))``; under 1.5
    times it the joint accelerates at ``(f - F) / I`` from the first step (to 1e-3 over three steps for the hinge, whose
    affine box loses a little speed in implicit Euler's rotation)."""
    dt = 0.01
    for kind, F in _FRICTION.items():
        for load in (0.25, 0.95, 1.5):
            with test.subTest(kind=kind, load=load):
                model, inertia = _hinged_box(device, kind, friction=F)
                control = model.control()
                control.joint_f.assign(np.array([load * F], np.float32))
                q, _ = _steps(model, newton.solvers.SolverIPC(model, **_TIGHT_ROTATIONS), 30, dt, control)
                v = np.diff(np.concatenate([[0.0], q[:, 0]])) / dt
                if load < 1.0:
                    np.testing.assert_allclose(v[-10:], _EPS_V * (1.0 - math.sqrt(1.0 - load)), rtol=1e-4)
                else:
                    np.testing.assert_allclose(np.diff(v[:4]) / dt, (load - 1.0) * F / inertia, rtol=1e-3)


def test_friction_takes_tight_default_tolerances(test, device):
    """A model with joint friction takes Newton tolerances of 0.4 eps_velocity where they are left at their default,
    silently, and its hinge and its slider under 1.5 times their friction then slide at ``(f - F) / I`` to 1% (at the
    defaults they would creep), with the axis along a coordinate axis and along the diagonal, where Newton's test, which
    takes a step's largest component, sees the least of it; tolerances given are kept, with a NewtonWarning where they
    are above that bound (and the joint creeps); a model without friction keeps the defaults, and so does a model whose
    friction a notification has removed. The rule reads the ``joint_friction`` of every degree of freedom, whatever its
    joint (a ball joint's here), and a fixed tendon's friction loss."""
    eps = _EPS_V
    for kind, F in _FRICTION.items():
        for axis in (None, (1.0 / math.sqrt(3.0),) * 3):
            with test.subTest(kind=kind, axis=axis):
                model, inertia = _hinged_box(device, kind, friction=F, axis=axis)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", newton.NewtonWarning)
                    solver = newton.solvers.SolverIPC(model)
                tolerances = (solver._config.newton_velocity_tol, solver._config.newton_transrate_tol)
                test.assertEqual(tolerances, (0.4 * eps, 0.4 * eps))
                control = model.control()
                control.joint_f.assign(np.array([1.5 * F], np.float32))
                q, _ = _steps(model, solver, 7, 0.01, control)
                v = np.diff(np.concatenate([[0.0], q[:, 0]])) / 0.01
                np.testing.assert_allclose(np.diff(v)[1:6] / 0.01, 0.5 * F / inertia, rtol=1e-2)

    model, _ = _hinged_box(device, friction=2.0)
    with test.assertWarnsRegex(newton.NewtonWarning, "creeps instead of sliding"):
        solver = newton.solvers.SolverIPC(model, newton_velocity_tol=0.02, newton_transrate_tol=0.04)
    test.assertEqual((solver._config.newton_velocity_tol, solver._config.newton_transrate_tol), (0.02, 0.04))
    control = model.control()
    control.joint_f.assign(np.array([3.0], np.float32))
    q, _ = _steps(model, solver, 20, 0.01, control)
    test.assertLess(abs(q[-1, 0] - q[-2, 0]) / 0.01, 0.05)  # sliding, it would turn at 3.75 rad/s after 20 steps
    with warnings.catch_warnings():
        warnings.simplefilter("error", newton.NewtonWarning)
        solver = newton.solvers.SolverIPC(model, newton_velocity_tol=1e-4, newton_transrate_tol=1e-4)
    test.assertEqual((solver._config.newton_velocity_tol, solver._config.newton_transrate_tol), (1e-4, 1e-4))

    model, _ = _hinged_box(device)
    solver = newton.solvers.SolverIPC(model)
    test.assertEqual((solver._config.newton_velocity_tol, solver._config.newton_transrate_tol), (0.05, 0.1))
    model, _ = _hinged_box(device, friction=2.0)
    solver = newton.solvers.SolverIPC(model)
    model.joint_friction.assign(np.zeros(1, np.float32))
    solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
    test.assertEqual((solver._config.newton_velocity_tol, solver._config.newton_transrate_tol), (0.05, 0.1))

    builder = newton.ModelBuilder()
    body = builder.add_link()
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_articulation([builder.add_joint_ball(-1, body)])
    builder.joint_friction[1] = 0.5
    test.assertTrue(joint_terms._has_friction(builder.finalize(device=device)))
    test.assertTrue(joint_terms._has_friction(_sliders(device, tendon='frictionloss="0.5"')))
    test.assertFalse(joint_terms._has_friction(_sliders(device)))


def test_tolerances_given_at_their_default_values_are_kept(test, device):
    """A model with joint friction keeps a Newton tolerance the caller gives, also one equal to SolverIPC's default
    (0.05 m/s and 0.1 1/s), with a NewtonWarning as for any tolerance above 0.4 eps_velocity, and lowers only the one
    left unset to 0.4 eps_velocity."""
    model, _ = _hinged_box(device, friction=2.0)
    bound = 0.4 * _EPS_V
    for given, expected in (
        ({"newton_velocity_tol": 0.05, "newton_transrate_tol": 0.1}, (0.05, 0.1)),
        ({"newton_velocity_tol": 0.05}, (0.05, bound)),
        ({"newton_transrate_tol": 0.1}, (bound, 0.1)),
    ):
        with test.subTest(**given):
            with test.assertWarnsRegex(newton.NewtonWarning, "creeps instead of sliding"):
                solver = newton.solvers.SolverIPC(model, **given)
            tolerances = (solver._config.newton_velocity_tol, solver._config.newton_transrate_tol)
            test.assertEqual(tolerances, expected)


_EQUALITY = """
<mujoco>
  <worldbody>
    <body name="a" pos="0 0 1"><joint name="ja" type="hinge" axis="0 0 1"/><geom type="box" size="0.1 0.1 0.1"/></body>
    <body name="b" pos="1 0 1"><joint name="jb" type="hinge" axis="0 0 1"/><geom type="box" size="0.1 0.1 0.1"/></body>
  </worldbody>
  <equality><joint joint1="jb" joint2="ja" polycoef="{polycoef}"/></equality>
</mujoco>
"""


def _joint_equality(device, polycoef):
    """Two hinges and MuJoCo's JOINT equality of the second on the first, imported as a mimic row and the equality
    row linked to it."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    builder.add_mjcf(_EQUALITY.replace("{polycoef}", polycoef))
    return builder.finalize(device=device)


def test_joint_equality_rows_repeating_a_mimic(test, device):
    """A JOINT equality row that repeats the mimic row the importer made of it (the same two joints, the same linear
    coefficients) is held by the mimic relation, which holds; a row with a term of higher order, one whose mimic row
    has other coefficients and one whose mimic row is disabled stay refused as rows that no joint holds."""
    model = _joint_equality(device, "0.1 -1.5 0 0 0")
    test.assertEqual((model.constraint_mimic_count, model.mujoco.equality_constraint_count), (1, 1))
    control = model.control()
    control.joint_f.assign(np.array([0.5, 0.0], np.float32))
    q, _ = _steps(model, newton.solvers.SolverIPC(model, **_TIGHT_ROTATIONS), 20, 0.01, control)
    test.assertGreater(abs(q[-1, 0]), 0.05)
    test.assertAlmostEqual(q[-1, 1], 0.1 - 1.5 * q[-1, 0], delta=1e-5)

    with test.assertWarnsRegex(newton.NewtonWarning, "higher-order polycoef terms"):
        model = _joint_equality(device, "0.1 -1.5 0.2 0 0")
    test.assertIn("1 enabled equality rows that no loop joint holds", _refusal(test, model))
    model = _joint_equality(device, "0.1 -1.5 0 0 0")
    model.constraint_mimic_coef1.assign(np.array([-1.4], np.float32))
    test.assertIn("1 enabled equality rows that no loop joint holds", _refusal(test, model))
    model = _joint_equality(device, "0.1 -1.5 0 0 0")
    model.constraint_mimic_enabled.assign(np.array([False]))
    test.assertIn("1 enabled equality rows that no loop joint holds", _refusal(test, model))


def test_parent_force_holds_the_friction(test, device):
    """BODY_PARENT_F of a box whose hinge's friction holds it against a joint_f below the friction: the joint's wrench
    on the box is the torque of joint_f and of the friction, which cancel once the box creeps steadily."""
    model, _ = _hinged_box(device, friction=2.0)
    control = model.control()
    control.joint_f.assign(np.array([1.0], np.float32))
    solver = newton.solvers.SolverIPC(model, **_TIGHT_ROTATIONS)
    observables = solver.observables(kinds={_KIND.BODY_PARENT_F})
    _, parent = _steps(model, solver, 30, 0.01, control, observables)
    test.assertLess(abs(parent[-1, 0, 5]), 1e-3)  # without the friction's records: 1 N m


def test_grown_lists_keep_the_terms_inputs(test, device):
    """A wheel spins down on a hinge with a friction of 0.5 N m while a box drops onto the ground; when it lands the
    contact lists overflow, and the eager step grows them and solves the step again on a new engine, whose terms take
    the step's friction and flags from the old: the wheel's angle goes on as in a solver with room."""

    def model():
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, -9.81, 0.0))
        wheel = builder.add_link(xform=wp.transform((0.0, 1.0, 0.0), wp.quat_identity()))
        builder.add_shape_box(wheel, hx=0.2, hy=0.2, hz=0.02)
        joint = builder.add_joint_revolute(
            -1, wheel, axis=newton.Axis.Z, parent_xform=wp.transform((0.0, 1.0, 0.0), wp.quat_identity()), friction=0.5
        )
        builder.add_articulation([joint])
        _box_body(builder, (1.0, 0.6, 0.0), (0.05, 0.05, 0.05))
        builder.add_ground_plane()
        m = builder.finalize(device=device)
        m.joint_qd.assign(np.concatenate([[15.0], m.joint_qd.numpy()[1:]]).astype(np.float32))
        return m

    results = []
    for tiny in (False, True):
        m = model()
        solver = newton.solvers.SolverIPC(m, **_TIGHT_ROTATIONS)
        if tiny:
            solver._make_engine(_TINY_LISTS)
        q, _ = _steps(m, solver, 45, 0.01)
        results.append(q[:, 0])
    roomy, grown = results
    test.assertGreater(roomy[-1], np.pi)
    test.assertLess(roomy[-1], 15.0 * 0.45 - 0.1)  # the friction slowed the wheel
    np.testing.assert_allclose(grown, roomy, atol=1e-5)


_SLIDERS = """<?xml version="1.0" ?>
<mujoco model="two_prismatic_links">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="root" pos="0 0 0">
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="link1" pos="0.0 -0.5 0">
        <joint name="joint1" type="slide" axis="1 0 0"/>
        <geom type="box" size="0.05 0.05 0.05"/>
      </body>
      <body name="link2" pos="0.0 -0.7 0">
        <joint name="joint2" type="slide" axis="1 0 0"/>
        <geom type="box" size="0.05 0.05 0.05"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="coupling_tendon" {tendon}>
      <joint joint="joint1" coef="1"/>
      <joint joint="joint2" coef="1"/>
    </fixed>
  </tendon>
</mujoco>
"""


def _sliders(device, worlds=1, tendon='stiffness="2" damping="1" springlength="0.0"'):
    """Newton's fixed-tendon scene of test_fixed_tendon: two sliders on a root (its box fixed to the world) coupled by a
    fixed tendon, in ``worlds`` worlds."""
    single = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    single.add_mjcf(_SLIDERS.replace("{tendon}", tendon), ignore_inertial_definitions=True)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for _ in range(worlds):
        builder.add_world(single)
    return builder.finalize(device=device)


def test_unread_tendon_friction_warns(test, device):
    """A fixed tendon with a friction loss whose solreffriction or solimpfriction differs from Newton's default gets a
    NewtonWarning naming the value and the tendon, once, when the solver is built, as the contact coefficients it does
    not read do; Newton's defaults, and a tendon without a friction loss, get none."""
    for tendon, named in (
        ('frictionloss="0.5" solreffriction="0.05 1"', "the solreffriction (0.05, 1) of MuJoCo's fixed tendon 0"),
        ('frictionloss="0.5" solimpfriction="0.8 0.9 0.001 0.5 2"', "the solimpfriction (0.8, 0.9, 0.001, 0.5, 2)"),
    ):
        with test.subTest(tendon=tendon):
            model = _sliders(device, tendon=tendon)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                newton.solvers.SolverIPC(model)
            messages = [str(w.message) for w in caught if issubclass(w.category, newton.NewtonWarning)]
            test.assertEqual(len(messages), 1, messages)
            test.assertIn(f"SolverIPC does not read {named}", messages[0])
    for tendon in ('frictionloss="0.5"', 'solreffriction="0.05 1" solimpfriction="0.8 0.9 0.001 0.5 2"'):
        with test.subTest(tendon=tendon):
            model = _sliders(device, tendon=tendon)
            with warnings.catch_warnings():
                warnings.simplefilter("error", newton.NewtonWarning)
                newton.solvers.SolverIPC(model)


def _recurrence(q0, m, k, c, h, steps):
    """Implicit Euler's recurrence of two coordinates of masses ``m`` at rest at ``q0`` under ``-k L - c L'``, ``L =
    q_1 + q_2``."""
    q, v, e = np.array(q0, float), np.zeros(2), np.ones(2)
    out = []
    for _ in range(steps):
        A = np.diag(m) + h * h * (k + c / h) * np.outer(e, e)
        new = np.linalg.solve(A, np.asarray(m) * (q + h * v) + h * c * q.sum() * e)
        v, q = (new - q) / h, new
        out.append(q.copy())
    return np.array(out)


def test_fixed_tendon_scene(test, device):
    """The scene of Newton's test_fixed_tendon: the tendon (stiffness 2, damping 1, spring length 0) pulls the sliders
    from (0.5, 0) along implicit Euler's recurrence of the coupled oscillator, 200 steps of 0.02 s, every step to
    1e-4 m (the root, welded to the world by a fixed joint, is held by that joint's penalty and gives a little; the
    terms alone hold the recurrence to 1e-6 m in libuipc-warp's tests). Newton's test asks SolverMuJoCo for the length
    0 after the 200 steps; implicit Euler's oscillator is at 0.0098 then. With two worlds, the second world's joints
    follow the first's."""
    for worlds in (1, 2):
        with test.subTest(worlds=worlds):
            model = _sliders(device, worlds)
            q = model.joint_q.numpy()
            q[0::2] = 0.5  # the first slider of every world (the root is welded to the world: no coordinates)
            model.joint_q.assign(q)
            solver = newton.solvers.SolverIPC(model, **_TIGHT_ROTATIONS)
            got, _ = _steps(model, solver, 200, 0.02)
            mass = model.body_mass.numpy().astype(np.float64)
            ref = _recurrence((0.5, 0.0), mass[1:3], 2.0, 1.0, 0.02, 200)
            np.testing.assert_allclose(got[:, 0:2], ref, rtol=0, atol=1e-4)
            if worlds == 2:
                np.testing.assert_allclose(got[:, 2:4], got[:, 0:2], rtol=0, atol=1e-9)


def test_tendon_limit(test, device):
    """A tendon limited to [-0.1, 0.1] (MuJoCo's solreflimit 0.02 1 and solimplimit 0.9 0.95 0.001) holds a joint_f of
    1 N on each slider at the overshoot of SolverIPC's cubic limit, ``sqrt(f delta_r / ke)`` (``delta_r`` 1 mm), with
    MuJoCo's stiffness ``ke = 1 / (T^2 zeta^2 w)``, ``w = (1 / m_1 + 1 / m_2) (1 - 0.95)``, the tendon's inverse weight
    over the width of its impedance."""
    model = _sliders(device, tendon='range="-0.1 0.1" solreflimit="0.02 1" solimplimit="0.9 0.95 0.001"')
    control = model.control()
    control.joint_f.assign(np.array([1.0, 1.0], np.float32))
    q, _ = _steps(model, newton.solvers.SolverIPC(model, **_TIGHT_ROTATIONS), 300, 0.01, control)
    mass = model.body_mass.numpy().astype(np.float64)
    ke = 1.0 / (0.02**2 * (1.0 / mass[1] + 1.0 / mass[2]) * (1.0 - 0.95))
    L = q[-1, 0] + q[-1, 1]
    test.assertAlmostEqual((L - 0.1) / math.sqrt(1.0 * 1e-3 / ke), 1.0, delta=2e-3)


def _refusal(test, model, **options):
    with test.assertRaises(NotImplementedError) as caught:
        newton.solvers.SolverIPC(model, **options)
    return str(caught.exception)


def test_refusals(test, device):
    """What SolverIPC still refuses of joint coordinate terms, each with its message."""
    builder = newton.ModelBuilder()
    a, b = builder.add_link(), builder.add_link()
    for body in (a, b):
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_articulation([builder.add_joint_revolute(-1, a), builder.add_joint_revolute(a, b)])
    builder.add_tendon([newton.TendonGuide(body=a), newton.TendonGuide(body=b)])
    test.assertIn("tendons of Newton's own (ModelBuilder.add_tendon)", _refusal(test, builder.finalize(device=device)))

    model = _sliders(device, tendon='stiffness="2" margin="0.01" range="-1 1"')
    test.assertIn("has a limit margin of 0.01", _refusal(test, model))

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    reference, follower = builder.add_link(), builder.add_link()
    axes = [newton.ModelBuilder.JointDofConfig.create_unlimited(a) for a in (newton.Axis.X, newton.Axis.Y)]
    joints = [builder.add_joint_d6(-1, body, linear_axes=axes) for body in (reference, follower)]
    builder.add_articulation(joints)
    builder.set_joint_mimic(joints[1], joints[0], coeffs=(0.1, -1.5))
    message = _refusal(test, builder.finalize(device=device))
    test.assertIn("the follower of the mimic joint 1 is the D6 joint 1", message)

    builder = newton.ModelBuilder()
    body = builder.add_body()
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.joint_friction[builder.joint_qd_start[-1] + 3] = 0.5
    test.assertIn("DOF 3 of the FREE joint 0 has joint_friction 0.5", _refusal(test, builder.finalize(device=device)))


@requires("uipc_warp")
class TestSolverIPCJointTerms(unittest.TestCase):
    """Mimic joints, MuJoCo's fixed tendons and joint friction: closed forms, recurrences and refusals."""


devices = get_cuda_test_devices()
for name, func in (
    ("test_mimic_scalar_case", test_mimic_scalar_case),
    ("test_mimic_penalty_and_enabled_flags", test_mimic_penalty_and_enabled_flags),
    ("test_mimic_coefficients_change_in_place", test_mimic_coefficients_change_in_place),
    ("test_joint_friction", test_joint_friction),
    ("test_friction_takes_tight_default_tolerances", test_friction_takes_tight_default_tolerances),
    ("test_tolerances_given_at_their_default_values_are_kept", test_tolerances_given_at_their_default_values_are_kept),
    ("test_unread_tendon_friction_warns", test_unread_tendon_friction_warns),
    ("test_joint_equality_rows_repeating_a_mimic", test_joint_equality_rows_repeating_a_mimic),
    ("test_parent_force_holds_the_friction", test_parent_force_holds_the_friction),
    ("test_grown_lists_keep_the_terms_inputs", test_grown_lists_keep_the_terms_inputs),
    ("test_fixed_tendon_scene", test_fixed_tendon_scene),
    ("test_tendon_limit", test_tendon_limit),
    ("test_refusals", test_refusals),
):
    add_function_test(TestSolverIPCJointTerms, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
