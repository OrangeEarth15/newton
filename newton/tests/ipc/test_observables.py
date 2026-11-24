# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC's BODY_PARENT_F, the wrench a body's parent joint passes to it, and BODY_QDD, the bodies' accelerations.

A joint of SolverIPC is a penalty on bodies that stretch a little, so the wrench it passes is the force of its terms at
the end of the step, which the step's momentum balance gives independently: a body held by its parent joint alone has
``m (v1 - v0) / dt - m g`` for the force and the change of its angular momentum about its centre of mass over ``dt``
for the torque. That is not the weight in the first steps of a pendulum (Newton's test_parent_force expects a rigid
joint's): the records follow the momentum balance at every step and reach the weight once the body settles."""

import builtins
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.ipc.features import observables as observables_module
from newton.tests.ipc._fixtures import _TIGHT_ROTATIONS, _box_body
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires

_KIND = newton.solvers.SolverBase.ObservableKind
_G = 9.81


def _add_pendulum(builder, axis=(0.0, 1.0, 0.0), child_offset=(0.0, 0.0, 1.0), at=(0.0, 0.0, 0.0)):
    """Newton's test_parent_force pendulum into ``builder``: a 0.2 m box (8 kg) 1 m below a revolute joint at ``at`` in
    the world."""
    link = builder.add_link()
    builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
    joint = builder.add_joint_revolute(
        -1,
        link,
        parent_xform=wp.transform(wp.vec3(*at), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(*child_offset), wp.quat_identity()),
        axis=wp.vec3(*axis),
    )
    builder.add_articulation([joint])


def _pendulum(device, ground=False, **pendulum):
    """A model of ``_add_pendulum``'s pendulum, z up, with a ground plane at z = 0 (``ground``)."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -_G), up_axis=newton.Axis.Z)
    if ground:
        builder.add_ground_plane()
    _add_pendulum(builder, **pendulum)
    return builder.finalize(device=device)


def _rotation(q):
    """The rotation matrix of a quaternion (x, y, z, w)."""
    x, y, z, w = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _momentum_balance(model, q0, qd0, q1, qd1, dt, gravity=(0.0, 0.0, -_G)):
    """(n, 6): per body the force and the torque about its centre of mass that change its momentum over the step as it
    changed, gravity aside: ``m (v1 - v0) / dt - m g`` and ``(L1 - L0) / dt``, ``L = R I R^T w`` (the angular velocity
    of SolverIPC's rigid rotation carries the angular momentum)."""
    mass, inertia = model.body_mass.numpy().astype(np.float64), model.body_inertia.numpy().astype(np.float64)
    out = np.zeros((model.body_count, 6))
    for b in range(model.body_count):
        out[b, :3] = mass[b] * (qd1[b, :3] - qd0[b, :3]) / dt - mass[b] * np.asarray(gravity)
        R0, R1 = _rotation(q0[b, 3:]), _rotation(q1[b, 3:])
        L0 = R0 @ inertia[b] @ R0.T @ qd0[b, 3:].astype(np.float64)
        L1 = R1 @ inertia[b] @ R1.T @ qd1[b, 3:].astype(np.float64)
        out[b, 3:] = (L1 - L0) / dt
    return out


def _steps(model, solver, steps, dt, observables, control=None, state=None, body_f=None):
    """``steps`` steps from ``state`` (the model's at its joint coordinates), with the wrenches ``body_f`` in every
    step's ``state_in``; per step the states it went from and to and the observables it wrote: ``(q0, qd0, q1, qd1,
    body_parent_f, body_qdd)``, None for a kind not requested."""
    s0 = state
    if s0 is None:
        s0 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    s1 = model.state()
    out = []
    for _ in range(steps):
        if body_f is not None:
            s0.body_f.assign(body_f)
        q0, qd0 = s0.body_q.numpy().copy(), s0.body_qd.numpy().copy()
        solver.step(s0, s1, control, None, dt, observables=observables)
        parent_f = observables.body_parent_f.numpy().copy() if observables.body_parent_f is not None else None
        qdd = observables.body_qdd.numpy().copy() if observables.body_qdd is not None else None
        out.append((q0, qd0, s1.body_q.numpy().copy(), s1.body_qd.numpy().copy(), parent_f, qdd))
        s0, s1 = s1, s0
    return out, s0


def test_parent_force_is_the_momentum_balance(test, device):
    """Newton's test_parent_force pendulum, released from rest in steps of 5 ms solved to tight tolerances: at every step
    BODY_PARENT_F is the box's momentum balance, the force within 1e-6 of the weight and the torque within 1e-6 N m
    (the default tolerances leave up to 5e-5 N m of it in the torque), and BODY_QDD the
    finite difference of the twists bit for bit. The joint's vertical force over the weight is not 1 in the first
    step (0.57 with SolverIPC's default body stiffness, 0.94 with ``body_kappa=1e8``: the joint pulls the box's affine
    stretch on a 1 m lever) and with ``body_kappa=1e8`` it is within 1e-4 of 1 from the sixth step on."""
    for kappa, first in ((None, 0.569), (1e8, 0.943)):
        with test.subTest(body_kappa=kappa):
            model = _pendulum(device)
            solver = newton.solvers.SolverIPC(model, body_kappa=kappa, **_TIGHT_ROTATIONS)
            observables = solver.observables(kinds={_KIND.BODY_PARENT_F, _KIND.BODY_QDD})
            steps, _ = _steps(model, solver, 12, 0.005, observables)
            weight = float(model.body_mass.numpy()[0]) * _G
            lift = []
            for q0, qd0, q1, qd1, parent_f, qdd in steps:
                balance = _momentum_balance(model, q0, qd0, q1, qd1, 0.005)
                np.testing.assert_allclose(parent_f[:, :3], balance[:, :3], rtol=0.0, atol=1e-6 * weight)
                np.testing.assert_allclose(parent_f[:, 3:], balance[:, 3:], rtol=0.0, atol=1e-6)
                np.testing.assert_array_equal(qdd, (qd1 - qd0) / np.float32(0.005))
                lift.append(parent_f[0, 2] / weight)
            test.assertAlmostEqual(lift[0], first, delta=2e-3)
            if kappa is not None:
                np.testing.assert_allclose(lift[5:], 1.0, atol=1e-4)


def _every_joint(device):
    """One 0.2 m box per joint kind, each held by its joint from the world alone (z up), with a joint_f on each: a
    revolute joint about y and a prismatic one along x, 0.5 m above the box; a fixed and a ball joint there; a free
    joint; a distance joint of 0.5 m; a D6 joint of three angular axes (a ball joint) and one of one (a revolute
    joint). Every box sits 6 cm off its body's origin, so that a body's centre of mass is not its origin. The model's
    body poses are where every joint holds its body (``eval_fk`` from the model's ``joint_q`` would put the distance
    joint's anchors together). Returns the model and the joint_f of the control."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -_G), up_axis=newton.Axis.Z)
    D = newton.ModelBuilder.JointDofConfig
    above = wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity())
    joints = []

    def link(x):
        body = builder.add_link(xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_box(
            body, xform=wp.transform(wp.vec3(0.05, 0.03, 0.0), wp.quat_identity()), hx=0.1, hy=0.1, hz=0.1
        )
        return body

    def at(x, z=0.5):
        return wp.transform(wp.vec3(x, 0.0, z), wp.quat_identity())

    joints.append(builder.add_joint_revolute(-1, link(0.0), parent_xform=at(0.0), child_xform=above, axis=(0, 1, 0)))
    joints.append(builder.add_joint_prismatic(-1, link(2.0), parent_xform=at(2.0), child_xform=above, axis=(1, 0, 0)))
    joints.append(builder.add_joint_fixed(-1, link(4.0), parent_xform=at(4.0), child_xform=above))
    joints.append(builder.add_joint_ball(-1, link(6.0), parent_xform=at(6.0), child_xform=above))
    joints.append(builder.add_joint_free(link(8.0)))
    joints.append(
        builder.add_joint_distance(
            -1, link(10.0), parent_xform=at(10.0, 0.6), child_xform=at(0.0, 0.1), min_distance=0.5, max_distance=0.5
        )
    )
    axes = [D(axis=(1, 0, 0)), D(axis=(0, 1, 0)), D(axis=(0, 0, 1))]
    joints.append(builder.add_joint_d6(-1, link(12.0), parent_xform=at(12.0), child_xform=above, angular_axes=axes))
    joints.append(
        builder.add_joint_d6(-1, link(14.0), parent_xform=at(14.0), child_xform=above, angular_axes=[D(axis=(0, 1, 0))])
    )
    for j in joints:
        builder.add_articulation([j])
    model = builder.finalize(device=device)
    joint_f = np.zeros(model.joint_dof_count, np.float32)
    start = model.joint_qd_start.numpy()
    for j, values in (
        (0, [0.4]),
        (1, [5.0]),
        (3, [0.3, -0.2, 0.5]),
        (4, [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]),
        (5, [0.5, 0.0, 0.0, 0.0, 0.05, 0.0]),
        (6, [0.2, 0.1, -0.3]),
        (7, [0.25]),
    ):
        joint_f[start[j] : start[j] + len(values)] = values
    return model, joint_f


def test_parent_force_of_every_joint_kind(test, device):
    """A box per joint kind, each held by its joint alone, with a joint_f on each: at every step of 20, each box's
    BODY_PARENT_F is its momentum balance, the force within 1e-5 of the weight and the torque within 1e-4 N m (the
    float32 rounding of the twists). It holds the joint's penalty, drive and limit, a revolute or prismatic joint's
    joint_f (libuipc's joint torque or force), and the joint_f that the solver applies as a wrench to the child of a
    free, ball or distance joint or a D6 joint of three angular axes: a free joint passes its joint_f alone."""
    model, joint_f = _every_joint(device)
    solver = newton.solvers.SolverIPC(model, **_TIGHT_ROTATIONS)
    observables = solver.observables(kinds={_KIND.BODY_PARENT_F})
    control = model.control()
    control.joint_f.assign(joint_f)
    steps, _ = _steps(model, solver, 20, 0.01, observables, control, state=model.state())
    weight = float(model.body_mass.numpy().max()) * _G
    for q0, qd0, q1, qd1, parent_f, _ in steps:
        balance = _momentum_balance(model, q0, qd0, q1, qd1, 0.01)
        np.testing.assert_allclose(parent_f[:, :3], balance[:, :3], rtol=0.0, atol=1e-5 * weight)
        np.testing.assert_allclose(parent_f[:, 3:], balance[:, 3:], rtol=0.0, atol=1e-4)
    np.testing.assert_array_equal(steps[-1][4][4], joint_f[model.joint_qd_start.numpy()[4] :][:6])
    test.assertGreater(np.abs(steps[-1][4][:, 3:]).max(), 0.1)  # the torques are not all zero


def test_parent_force_of_a_chain_at_rest(test, device):
    """Newton's test_apply_body_f chain, two links on revolute joints about y, a force along y (10 N) or a torque about x
    (5 N m) on the lower link, which the joints hold, with ``body_kappa=1e8``: once the chain has settled, the upper
    link's parent joint carries both links' weight and the force or the torque, as Newton's test expects of a rigid
    joint, within its tolerances: 1e-4 of the weight and 0.01 N m."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -_G), up_axis=newton.Axis.Z)
    links = []
    for _ in range(2):
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
        links.append(link)
    j0 = builder.add_joint_revolute(
        -1, links[0], child_xform=wp.transform(wp.vec3(0, 0, 1), wp.quat_identity()), axis=wp.vec3(0, 1, 0)
    )
    j1 = builder.add_joint_revolute(
        links[0],
        links[1],
        parent_xform=wp.transform(wp.vec3(0, 0, -1), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0, 0, 1), wp.quat_identity()),
        axis=wp.vec3(0, 1, 0),
    )
    builder.add_articulation([j0, j1])
    model = builder.finalize(device=device)
    weight = float(model.body_mass.numpy().sum()) * _G
    for name, wrench, expected in (
        ("force", (0.0, 10.0, 0.0, 0.0, 0.0, 0.0), (0.0, -10.0, weight, -20.0, 0.0, 0.0)),
        ("torque", (0.0, 0.0, 0.0, 5.0, 0.0, 0.0), (0.0, 0.0, weight, -5.0, 0.0, 0.0)),
    ):
        with test.subTest(name):
            solver = newton.solvers.SolverIPC(model, body_kappa=1e8)
            observables = solver.observables(kinds={_KIND.BODY_PARENT_F})
            body_f = np.zeros((2, 6), np.float32)
            body_f[1] = wrench
            steps, _ = _steps(model, solver, 200, 5e-3, observables, body_f=body_f)
            parent_f = steps[-1][4][0]
            np.testing.assert_allclose(parent_f[:3], expected[:3], rtol=0.0, atol=1e-4 * weight)
            np.testing.assert_allclose(parent_f[3:], expected[3:], rtol=0.0, atol=1e-2)


def test_parent_force_of_a_spinning_pendulum(test, device):
    """Newton's centrifugal case: a box on a 1 m arm about a vertical revolute joint, spinning at 5 rad/s. At every step
    BODY_PARENT_F is the momentum balance, and once the arm has taken up its stretch its radial force is the centripetal
    force ``m w^2 r`` toward the axis within 1%, at the angular rate of the step."""
    model = _pendulum(device, axis=(0.0, 0.0, 1.0), child_offset=(-1.0, 0.0, 0.0))
    solver = newton.solvers.SolverIPC(model, body_kappa=1e8)
    observables = solver.observables(kinds={_KIND.BODY_PARENT_F})
    state = model.state()
    state.joint_qd.assign([5.0])
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    steps, _ = _steps(model, solver, 60, 5e-3, observables, state=state)
    mass = float(model.body_mass.numpy()[0])
    for q0, qd0, q1, qd1, parent_f, _ in steps:
        balance = _momentum_balance(model, q0, qd0, q1, qd1, 5e-3)
        np.testing.assert_allclose(parent_f[:, :3], balance[:, :3], rtol=0.0, atol=1e-5 * mass * 25.0)
    _, _, q1, qd1, parent_f, _ = steps[-1]
    radial = -q1[0, :2] / np.linalg.norm(q1[0, :2])  # toward the axis
    w = float(qd1[0, 5])
    r = float(np.linalg.norm(q1[0, :2]))
    test.assertAlmostEqual(float(parent_f[0, :2] @ radial) / (mass * w * w * r), 1.0, delta=0.01)
    test.assertAlmostEqual(float(parent_f[0, 2]) / (mass * _G), 1.0, delta=0.01)


def test_body_observables_captured_equal_eager(test, device):
    """Steps captured in a graph write BODY_PARENT_F and BODY_QDD as eager steps write them, also where state_in is
    state_out (BODY_QDD keeps the step's starting twists in its own array until the step's end)."""
    results = []
    for capture in (False, True):
        model, joint_f = _every_joint(device)
        solver = newton.solvers.SolverIPC(model)
        observables = solver.observables(kinds={_KIND.BODY_PARENT_F, _KIND.BODY_QDD})
        control = model.control()
        control.joint_f.assign(joint_f)
        state = model.state()
        solver.step(state, state, control, None, 0.01, observables=observables)  # loads the state and the modules
        qd0 = state.body_qd.numpy().copy()
        if capture:
            with wp.ScopedCapture(device=device) as captured:
                solver.step(state, state, control, None, 0.01, observables=observables)
            wp.capture_launch(captured.graph)
        else:
            solver.step(state, state, control, None, 0.01, observables=observables)
        qdd = observables.body_qdd.numpy()
        np.testing.assert_array_equal(qdd, (state.body_qd.numpy() - qd0) / np.float32(0.01))
        results.append((observables.body_parent_f.numpy(), qdd))
    (pf_eager, qdd_eager), (pf_captured, qdd_captured) = results
    np.testing.assert_allclose(pf_captured, pf_eager, rtol=0.0, atol=1e-4)
    np.testing.assert_allclose(qdd_captured, qdd_eager, rtol=0.0, atol=1e-4)


def test_body_observables_only_where_requested(test, device):
    """Without a request the solver allocates nothing for the kinds, gives the engines it builds no extension and
    writes neither array: a container that requests none of them, or a selection without them, leaves their arrays as
    they were. BODY_QDD alone allocates no records of the joints' forces and gives no extension either."""
    model = _pendulum(device)
    solver = newton.solvers.SolverIPC(model)
    status = solver.observables(kinds={solver.ObservableKind.STATUS})
    s0, s1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    solver.step(s0, s1, None, None, 5e-3, observables=status)
    test.assertIsNone(getattr(solver, "_body_observables", None))

    def extensions():
        """What the observables module adds to the inputs of an engine the solver builds."""
        inputs = {"elements": [], "extensions": []}
        observables_module.engine_inputs(solver, inputs)
        return inputs["extensions"]

    test.assertEqual(extensions(), [])
    observables = solver.observables(kinds={_KIND.BODY_QDD})
    test.assertIsNone(solver._body_observables.parents)
    test.assertEqual(extensions(), [])
    observables = solver.observables(kinds={_KIND.BODY_PARENT_F, _KIND.BODY_QDD})
    test.assertIsNotNone(solver._body_observables.parents)
    test.assertEqual([type(e).__name__ for e in extensions()], ["_Records"])
    observables.body_parent_f.fill_(wp.spatial_vector(-1.0))
    observables.body_qdd.fill_(wp.spatial_vector(-1.0))
    solver.step(s1, s0, None, None, 5e-3, observables=observables.select(set()))
    np.testing.assert_array_equal(observables.body_parent_f.numpy(), -1.0)
    np.testing.assert_array_equal(observables.body_qdd.numpy(), -1.0)
    solver.step(s0, s1, None, None, 5e-3, observables=observables.select({_KIND.BODY_QDD}))
    np.testing.assert_array_equal(observables.body_parent_f.numpy(), -1.0)
    test.assertTrue(np.all(observables.body_qdd.numpy() != -1.0))
    solver.step(s1, s0, None, None, 5e-3, observables=observables)
    test.assertTrue(np.all(np.isfinite(observables.body_parent_f.numpy())))


def test_parent_force_after_the_solve_is_built_again(test, device):
    """A joint frame changed and notified builds the solve again: the records are made anew for the new engine's
    joint groups, and BODY_PARENT_F is the momentum balance of the new solve's steps."""
    model = _pendulum(device)
    solver = newton.solvers.SolverIPC(model)
    observables = solver.observables(kinds={_KIND.BODY_PARENT_F})
    _, state = _steps(model, solver, 3, 5e-3, observables)
    engine, parents = solver._engine, solver._body_observables.parents
    X_c = model.joint_X_c.numpy()
    X_c[0, 2] = 0.9  # the box 0.9 m below the joint
    model.joint_X_c.assign(X_c)
    solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
    test.assertIsNot(solver._engine, engine)
    rebuilt = solver._body_observables.parents
    test.assertIsNot(rebuilt, parents)
    test.assertIs(rebuilt.forces.groups[0], solver._uipc_warp.arrays.joint_groups(solver._engine).revolute)
    steps, _ = _steps(model, solver, 5, 5e-3, observables, state=state)
    weight = float(model.body_mass.numpy()[0]) * _G
    for q0, qd0, q1, qd1, parent_f, _ in steps:
        balance = _momentum_balance(model, q0, qd0, q1, qd1, 5e-3)
        np.testing.assert_allclose(parent_f[:, :3], balance[:, :3], rtol=1e-6, atol=1e-6 * weight)


def test_body_observables_of_independent_worlds(test, device):
    """Three independent worlds of the pendulum, hung 2 m above a ground plane, write what the pendulum alone writes,
    within the rounding of the worlds' sums. In a step in which world 1 fails alone (its box given a velocity that is not
    a number), BODY_PARENT_F and BODY_QDD are zero in world 1 and the other worlds' are those of a clean solver's same
    step."""
    content = newton.ModelBuilder(gravity=(0.0, 0.0, -_G), up_axis=newton.Axis.Z)
    _add_pendulum(content, at=(0.0, 0.0, 2.0))
    scene = newton.ModelBuilder(gravity=(0.0, 0.0, -_G), up_axis=newton.Axis.Z)
    scene.add_ground_plane()
    scene.replicate(content, 3)
    model = scene.finalize(device=device)
    alone = _pendulum(device, ground=True, at=(0.0, 0.0, 2.0))
    kinds = {_KIND.BODY_PARENT_F, _KIND.BODY_QDD}
    one = newton.solvers.SolverIPC(alone)
    steps_alone, _ = _steps(alone, one, 4, 5e-3, one.observables(kinds=kinds))
    runs = []
    for _ in range(2):
        solver = newton.solvers.SolverIPC(model)
        observables = solver.observables(kinds=kinds)
        steps, state = _steps(model, solver, 4, 5e-3, observables)
        runs.append((solver, observables, steps, state))
    for k in range(4):
        for w in range(3):
            np.testing.assert_allclose(runs[0][2][k][4][w], steps_alone[k][4][0], rtol=0.0, atol=1e-4)
            np.testing.assert_allclose(runs[0][2][k][5][w], steps_alone[k][5][0], rtol=0.0, atol=1e-3)
    (clean, clean_observables, _, c0), (faulty, observables, _, f0) = runs
    qd = f0.body_qd.numpy()
    qd[1, 0] = np.nan  # world 1's box with a velocity that is not a number
    f0.body_qd.assign(qd)
    c1, f1 = model.state(), model.state()
    clean.step(c0, c1, None, None, 5e-3, observables=clean_observables)
    with test.assertRaisesRegex(RuntimeError, "failed in 1 of its 3 worlds, world 1 on NONFINITE"):
        faulty.step(f0, f1, None, None, 5e-3, observables=observables)
    for array, want in (
        (observables.body_parent_f.numpy(), clean_observables.body_parent_f.numpy()),
        (observables.body_qdd.numpy(), clean_observables.body_qdd.numpy()),
    ):
        np.testing.assert_array_equal(array[1], 0.0)
        np.testing.assert_allclose(array[[0, 2]], want[[0, 2]], rtol=0.0, atol=1e-4)


def test_body_qdd_of_a_free_body(test, device):
    """A free box under a constant force and a sphere under a constant torque, no gravity, steps solved to tight
    tolerances: BODY_QDD is ``F / m`` and ``I^-1 tau`` (the rigid rotation integrates the sphere's spin exactly), to the
    float32 rounding of the twists. Also under AL-IPC, which offers BODY_QDD and BODY_PARENT_F."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Z)
    _box_body(builder, (0.0, 0.0, 1.0), (0.1, 0.1, 0.1))
    ball = builder.add_body(xform=wp.transform(wp.vec3(2.0, 0.0, 1.0), wp.quat_identity()))
    builder.add_shape_sphere(ball, radius=0.1)
    model = builder.finalize(device=device)
    wrench = np.array([[3.0, -2.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.02, -0.01, 0.03]], np.float32)
    mass, inertia = model.body_mass.numpy(), model.body_inertia.numpy()
    expected = np.zeros((2, 6))
    expected[0, :3] = wrench[0, :3] / mass[0]
    expected[1, 3:] = np.linalg.solve(inertia[1].astype(np.float64), wrench[1, 3:].astype(np.float64))
    for name, options in (("ipc", _TIGHT_ROTATIONS), ("al-ipc", {"contact_constitution": "al-ipc"})):
        with test.subTest(name):
            solver = newton.solvers.SolverIPC(model, **options)
            test.assertTrue({_KIND.BODY_QDD, _KIND.BODY_PARENT_F} <= solver.supported_observables)
            observables = solver.observables(kinds={_KIND.BODY_QDD, _KIND.BODY_PARENT_F})
            steps, _ = _steps(model, solver, 10, 0.01, observables, state=model.state(), body_f=wrench)
            for _, qd0, _, qd1, parent_f, qdd in steps:
                np.testing.assert_allclose(qdd, expected, rtol=1e-4, atol=1e-5)
                np.testing.assert_array_equal(qdd, (qd1 - qd0) / np.float32(0.01))
                np.testing.assert_array_equal(parent_f, 0.0)  # free joints without joint_f pass nothing


def test_imu_on_a_box_at_rest(test, device):
    """SensorIMU on a box resting on the ground reads the ground's reaction to gravity, (0, 0, g) in its frame, within
    0.01 m/s^2 once the box has settled, and no rotation."""
    from newton.sensors import SensorIMU  # noqa: PLC0415

    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    body = _box_body(builder, (0.0, 0.0, 0.11), (0.1, 0.1, 0.1))
    builder.add_site(body, label="imu")
    model = builder.finalize(device=device)
    imu = SensorIMU(model, sites="imu")
    solver = newton.solvers.SolverIPC(model)
    observables = solver.observables(kinds=imu.solver_observable_kinds)
    _, state = _steps(model, solver, 150, 0.01, observables, state=model.state())
    imu.update(state, observables=observables)
    np.testing.assert_allclose(imu.accelerometer.numpy()[0], [0.0, 0.0, _G], rtol=0.0, atol=0.01)
    np.testing.assert_allclose(imu.gyroscope.numpy()[0], 0.0, atol=1e-3)


def test_parent_force_needs_element_forces(test, device):
    """A libuipc-warp without ``uipc_warp.ext.element_forces`` takes BODY_QDD and refuses BODY_PARENT_F with an
    ImportError that names the module."""
    model = _pendulum(device)
    solver = newton.solvers.SolverIPC(model)
    solver.observables(kinds={_KIND.BODY_QDD})
    real = builtins.__import__

    def without(name, *args, **kwargs):
        if name.startswith("uipc_warp.ext.element_forces"):
            raise ImportError(name)
        return real(name, *args, **kwargs)

    with mock.patch.object(builtins, "__import__", side_effect=without):
        with test.assertRaisesRegex(ImportError, "uipc_warp.ext.element_forces"):
            solver.observables(kinds={_KIND.BODY_PARENT_F})
    test.assertIsNone(solver._body_observables.parents)


@requires("uipc_warp")
class TestSolverIPCObservables(unittest.TestCase):
    """BODY_PARENT_F and BODY_QDD: the wrench of every body's parent joint and the bodies' accelerations."""


devices = get_cuda_test_devices()
for name, func in (
    ("test_parent_force_is_the_momentum_balance", test_parent_force_is_the_momentum_balance),
    ("test_parent_force_of_every_joint_kind", test_parent_force_of_every_joint_kind),
    ("test_parent_force_of_a_chain_at_rest", test_parent_force_of_a_chain_at_rest),
    ("test_parent_force_of_a_spinning_pendulum", test_parent_force_of_a_spinning_pendulum),
    ("test_body_observables_captured_equal_eager", test_body_observables_captured_equal_eager),
    ("test_body_observables_only_where_requested", test_body_observables_only_where_requested),
    ("test_parent_force_after_the_solve_is_built_again", test_parent_force_after_the_solve_is_built_again),
    ("test_body_observables_of_independent_worlds", test_body_observables_of_independent_worlds),
    ("test_body_qdd_of_a_free_body", test_body_qdd_of_a_free_body),
    ("test_imu_on_a_box_at_rest", test_imu_on_a_box_at_rest),
    ("test_parent_force_needs_element_forces", test_parent_force_needs_element_forces),
):
    add_function_test(TestSolverIPCObservables, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
