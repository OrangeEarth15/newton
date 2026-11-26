# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC's rod joints: libuipc-warp's rod joints for Newton's rods and cables (``features/rods.py``).

* SolverKamino's expectations of a rod joint, from ``newton/tests/kamino/test_kamino_solver_kamino_rods.py``, with
  SolverIPC: the stretch restores and its damping slows, the stretch acts at the follower's anchor (a body anchored at
  its centre does not turn), a bend and a twist restore, a pre-curved rest stays at rest, the rest follows
  ``joint_rest_q`` (its rotation and its translation), a binary rod balances its wrenches, the twist's damping across
  both branches of the angle, a twist step as the linearly implicit reference, no twist terms leave the spin alone.
* ``joint_f`` of a rod joint is a world wrench at the child's centre of mass with its reaction on the parent, as
  SolverVBD applies it (Newton's ``_rod_vbd_joint_f_applies_six_dof_wrench_impl``); a rod joint's ``joint_q`` and
  ``joint_qd`` after steps are ``eval_ik``'s.
* Against SolverVBD on the same models: the static shapes of three cables of bend stiffness 50, 500 and 5000 N m/rad
  under gravity (Newton's bend-stiffness sweep) agree to 1e-3 of the length with the bodies at libuipc's stiffness, both
  minimizing the same elastic energy;
  a twist with Dahl hysteresis under a cycle of torque follows SolverVBD's and keeps the same residual twist.
* ``BODY_PARENT_F`` of a hanging cable is each segment's momentum balance (``features.rods.parent_terms``).
* The solve: captured steps equal eager ones; a step solved again on lists grown after an overflow takes the rods and
  their hysteresis; gains and rests written after the solver was built take effect at the next step; a disabled rod
  joint acts on nothing; two independent worlds step as each alone; a reset starts the hysteresis anew.
* The refusals: joint friction, armature, ``joint_stiffness``, ``joint_damping``, a target mode, negative gains,
  finite limits and an effort limit below 1e6 on a rod's degrees of freedom; a rod joint of another layout than
  ``add_joint_rod``'s; SolverVBD's Dahl attributes other than 0 on another joint, or negative on a rod joint.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.ipc._fixtures import _TIGHT_ROTATIONS, _TINY_LISTS
from newton.tests.ipc.test_ipc_observables import _momentum_balance
from newton.tests.ipc.test_ipc_observables import _steps as _observed_steps
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires

_KIND = newton.solvers.SolverBase.ObservableKind
_H = 0.01  # SolverKamino's rod tests' step [s]
_UNIT = wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
# the Kamino cases' tolerances: a step ends once no affine body moves by more than 1e-7 m/s or turns by more than 1e-7
# 1/s, and its angular velocity is a rigid body's (body_rotation="rigid"), as Kamino's bodies are
_RIGID = {**_TIGHT_ROTATIONS, "newton_velocity_tol": 1.0e-7, "newton_transrate_tol": 1.0e-7, "body_rotation": "rigid"}


# ================================================================================================ SolverKamino's cases
def _rod(device, binary=False, child_xform=None, **rod):
    """SolverKamino's rod model: a unit-inertia body of 1 kg on a rod joint to the world (``binary``: to another such
    body), no gravity; ``rod``: ``add_joint_rod``'s coefficients, 0 where not given. Returns (model, parent, child)."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    parent = builder.add_link(mass=1.0, inertia=_UNIT, lock_inertia=True) if binary else -1
    xform = wp.transform_identity() if child_xform is None else child_xform
    child = builder.add_link(xform=xform, mass=1.0, inertia=_UNIT, lock_inertia=True)
    names = ("stretch", "shear", "bend", "twist")
    coefficients = {f"{n}_{k}": rod.get(f"{n}_{k}", 0.0) for n in names for k in ("stiffness", "damping")}
    joint = builder.add_joint_rod(parent, child, **coefficients)
    builder.add_articulation([joint])
    return builder.finalize(device=device), parent, child


def _step(model, body_q=None, body_qd=None, control=None, h=_H) -> np.ndarray:
    """The bodies' velocities after one step of SolverIPC from the model's state with ``body_q`` and ``body_qd``."""
    solver = newton.solvers.SolverIPC(model, **_RIGID)
    s0, s1 = model.state(), model.state()
    if body_q is not None:
        s0.body_q.assign(np.asarray(body_q, np.float32))
    if body_qd is not None:
        s0.body_qd.assign(np.asarray(body_qd, np.float32))
    solver.step(s0, s1, control or model.control(), None, h)
    return s1.body_qd.numpy()


def _pose(p=(0.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0), angle=0.0) -> list:
    return [*p, *np.asarray(wp.quat_from_axis_angle(wp.vec3(*axis), angle))]


def test_kamino_expectations(test, device):
    """Meet SolverKamino's rod-test expectations in one step of 10 ms each.

    The expectations are those of ``test_kamino_solver_kamino_rods.py``.
    """
    with test.subTest("the stretch restores, its damping slows"):
        model, _, child = _rod(device, stretch_stiffness=100.0)
        test.assertLess(_step(model, body_q=[_pose((0.0, 0.0, 0.1))])[child, 2], 0.0)
        speeds = []
        for damping in (0.0, 10.0):
            model, _, child = _rod(device, stretch_damping=damping)
            speeds.append(_step(model, body_qd=[[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])[child, 2])
        test.assertLess(speeds[1], speeds[0])
    with test.subTest("the stretch acts at the follower's anchor"):
        model, _, child = _rod(device, stretch_stiffness=100.0)
        v = _step(model, body_q=[_pose((0.1, 0.0, 0.1))])
        test.assertLess(v[child, 2], 0.0)
        np.testing.assert_allclose(v[child, 3:], 0.0, atol=1e-6)
    for name, axis, coefficients, k in (
        ("bend", (1.0, 0.0, 0.0), {"bend_stiffness": 20.0}, 3),
        ("twist", (0.0, 0.0, 1.0), {"twist_stiffness": 20.0}, 5),
    ):
        with test.subTest(f"a {name} restores"):
            model, _, child = _rod(device, **coefficients)
            v = _step(model, body_q=[_pose(axis=axis, angle=0.1)])
            test.assertLess(v[child, k], 0.0)
            test.assertTrue(np.isfinite(v).all())
    with test.subTest("a pre-curved rest stays at rest"):
        curved = wp.transform(wp.vec3(0.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.2))
        model, _, child = _rod(device, bend_stiffness=20.0, twist_stiffness=10.0, child_xform=curved)
        np.testing.assert_allclose(_step(model)[child], 0.0, atol=1e-6)
    with test.subTest("the rest follows joint_rest_q's rotation"):
        model, _, child = _rod(device, twist_stiffness=20.0)
        rest = model.joint_rest_q.numpy()
        rest[3:7] = np.asarray(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.1))
        model.joint_rest_q.assign(rest)
        test.assertGreater(_step(model)[child, 5], 0.0)
    with test.subTest("and its translation"):
        model, _, child = _rod(device, stretch_stiffness=100.0)
        rest = model.joint_rest_q.numpy()
        rest[0:3] = [0.0, 0.0, 0.1]
        model.joint_rest_q.assign(rest)
        v = _step(model)
        test.assertGreater(v[child, 2], 0.0)
        np.testing.assert_allclose(v[child, :2], 0.0, atol=1e-6)
    with test.subTest("a binary rod balances its wrenches"):
        model, parent, child = _rod(device, binary=True, shear_stiffness=50.0)
        v = _step(model, body_q=[_pose(), _pose((0.1, 0.0, 0.0))])
        test.assertGreater(v[parent, 0], 0.0)
        test.assertLess(v[child, 0], 0.0)
        np.testing.assert_allclose(v[parent, :3] + v[child, :3], 0.0, atol=1e-6)
    with test.subTest("the twist's damping across both branches of the angle"):
        # SolverIPC's body is affine: turning 0.05 rad in the step, it takes the damping on the chord of its turn, which
        # is 0.3% from Kamino's rigid body's (a first-order difference of implicit Euler, (omega dt)^2); so 1% here
        for strain, speed in ((3.1, 10.0), (-3.1, -10.0), (0.5, 10.0)):
            model, _, child = _rod(device, twist_damping=100.0)
            v = _step(model, body_q=[_pose(axis=(0.0, 0.0, 1.0), angle=strain)], body_qd=[[0, 0, 0, 0, 0, speed]])
            test.assertAlmostEqual(v[child, 5], speed / (1.0 + _H * 100.0), delta=0.01 * abs(speed) / 2.0)
    with test.subTest("a twist step as the linearly implicit reference"):
        rest = wp.transform(wp.vec3(0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -0.35))
        model, _, child = _rod(device, twist_stiffness=37.0, twist_damping=13.0, child_xform=rest)
        v = _step(model, body_q=[_pose(axis=(0.0, 0.0, 1.0), angle=-0.35 + 0.8)], body_qd=[[0, 0, 0, 0, 0, 7.0]])
        want = (7.0 - _H * 37.0 * 0.8) / (1.0 + _H * 13.0 + _H * _H * 37.0)
        test.assertAlmostEqual(v[child, 5], want, delta=0.01 * want)
    with test.subTest("no twist terms leave the spin alone"):
        model, _, child = _rod(device)
        v = _step(model, body_q=[_pose(axis=(0.0, 0.0, 1.0), angle=3.1)], body_qd=[[0, 0, 0, 0, 0, 10.0]])
        np.testing.assert_allclose(v[child], [0.0, 0.0, 0.0, 0.0, 0.0, 10.0], rtol=0.0, atol=1e-6)


def test_joint_f_is_a_world_wrench(test, device):
    """Apply a rod joint's ``joint_f`` as a world wrench at the child's centre of mass, as SolverVBD does.

    The wrench (12 N along x, 6 N m about y) acts at the child's centre of mass, the parent taking its reaction there:
    the step's velocities match to 1e-5, and the bodies have no angular momentum about the world's origin
    (``newton/tests/test_cable.py``, ``_rod_vbd_joint_f_applies_six_dof_wrench_impl``).
    """
    mass, force, torque, dt = 2.0, 12.0, 6.0, 1.0 / 60.0
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    parent = builder.add_link(
        xform=wp.transform((0.0, 0.0, -1.0), wp.quat_identity()), com=(0.0, 0.0, -0.5), mass=mass, inertia=_UNIT
    )
    child = builder.add_link(com=(0.0, 0.0, 0.5), mass=mass, inertia=_UNIT)
    rod = builder.add_joint_rod(parent, child, stretch_stiffness=0.0, bend_stiffness=0.0)
    builder.add_articulation([rod])
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, **_RIGID)
    control = model.control()
    joint_f = np.zeros(model.joint_dof_count, np.float32)
    start = int(model.joint_qd_start.numpy()[rod])
    joint_f[start] = force
    joint_f[start + 4] = torque
    control.joint_f.assign(joint_f)
    s0, s1 = model.state(), model.state()
    solver.step(s0, s1, control, None, dt)
    qd = s1.body_qd.numpy()
    np.testing.assert_allclose(qd[child], [force * dt / mass, 0, 0, 0, torque * dt, 0], atol=1e-5)
    np.testing.assert_allclose(qd[parent], [-force * dt / mass, 0, 0, 0, -(torque + 2 * force) * dt, 0], atol=1e-5)
    q = s1.body_q.numpy()
    com = model.body_com.numpy()
    L = np.zeros(3)
    for b in (parent, child):
        c = np.asarray(wp.transform_point(wp.transform(*q[b]), wp.vec3(*com[b])))
        L += qd[b, 3:] + mass * np.cross(c, qd[b, :3])
    np.testing.assert_allclose(L, 0.0, atol=1e-5)


# ================================================================================================ cables
def _cable(builder, start, bend, n=10, length=0.2, radius=0.05, **rod):
    """A cable of ``n`` segments of ``length`` along x from ``start`` (``add_rod``), its first segment kinematic: its
    bodies."""
    straight = newton.Rod.create_straight(
        start=wp.vec3(*start), direction=wp.vec3(1.0, 0.0, 0.0), length=n * length, segment_count=n, radius=radius
    )
    bodies, _ = builder.add_rod(rod=straight, bend_stiffness=bend, **rod)
    builder.body_flags[bodies[0]] = int(newton.BodyFlags.KINEMATIC)
    return bodies


def _sweep(device, vbd: bool):
    """Newton's bend-stiffness sweep (``test_cable.py``, ``_cable_bend_stiffness_impl``): three cables of 10 segments
    of 0.2 m, bend stiffness 50, 500 and 5000 N m/rad, 5 m apart, each held by its first segment, under gravity; their
    bend damping in N m s/rad equal to the stiffness (Newton's case takes ten times it, under which they take tens of
    seconds to come to rest)."""
    builder = newton.ModelBuilder()
    if vbd:
        newton.solvers.SolverVBD.register_custom_attributes(builder)
    cables = [_cable(builder, (0.0, y, 3.0), k, bend_damping=k) for k, y in ((5e1, -5.0), (5e2, 0.0), (5e3, 5.0))]
    builder.color()
    return builder.finalize(device=device), cables


def _settle(model, solver, steps, dt, contacts=None):
    s0, s1 = model.state(), model.state()
    control = model.control()
    for _ in range(steps):
        s0.clear_forces()
        solver.step(s0, s1, control, contacts, dt)
        s0, s1 = s1, s0
    return s0


def test_static_shapes_against_vbd(test, device):
    """Rest the bend-stiffness sweep's cables in the shapes SolverVBD rests them in.

    After 15 s under gravity every segment's centre lies where SolverVBD rests it, within 1e-3 of the cable's length
    (2 m), and the softest cable's tip is more than 0.5 m below its root. Both minimize the same elastic energy at rest:
    the stretch and shear springs and the bend of the discrete elastic rod, below Newton's cap on the bend
    (168.6 degrees per joint). The bodies take libuipc's stiffness, 1e8 Pa: the default, lowered for a step of 1/60 s,
    lets each segment shear, which relieves the bend of the joint it is the parent of, and the stiffest cable's tip then
    rests 0.16 m lower.
    """
    model, cables = _sweep(device, vbd=True)
    vbd = _settle(model, newton.solvers.SolverVBD(model, iterations=40), 4500, 1.0 / 300.0).body_q.numpy()
    model, cables = _sweep(device, vbd=False)
    # kinematic_motion="jump" places the kinematic segments where their state is, as SolverVBD does; the default
    # moves a kinematic body with mass by a soft aim, which a cable's weight would pull down
    solver = newton.solvers.SolverIPC(model, kinematic_motion="jump", body_kappa=1.0e8)
    ipc = _settle(model, solver, 900, 1.0 / 60.0).body_q.numpy()
    for k, bodies in enumerate(cables):
        with test.subTest(cable=k):
            np.testing.assert_allclose(ipc[bodies, :3], vbd[bodies, :3], rtol=0.0, atol=2e-3)
    sag = float(ipc[cables[0][0], 2] - ipc[cables[0][-1], 2])
    test.assertGreater(sag, 0.5)


def _twister(device, vbd: bool, dahl=(0.5, 0.05)):
    """A body of 0.01 kg m^2 about z on a rod joint to the world at its centre: twist stiffness 1 N m/rad, damping 0.05
    N m s/rad, Dahl hysteresis ``(eps_max, tau)``; the anchor and the bend held stiffly."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverVBD.register_custom_attributes(builder)
    body = builder.add_link(
        mass=1.0, inertia=wp.mat33f(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01), lock_inertia=True
    )
    builder.add_shape_sphere(body, radius=0.05)
    joint = builder.add_joint_rod(
        -1, body, stretch_stiffness=1e4, bend_stiffness=50.0, twist_stiffness=1.0, twist_damping=0.05
    )
    builder.add_articulation([joint])
    if vbd:
        builder.color()
    model = builder.finalize(device=device)
    model.vbd.dahl_eps_max.assign(np.array([dahl[0]], np.float32))
    model.vbd.dahl_tau.assign(np.array([dahl[1]], np.float32))
    return model, joint


def _torque_cycle(model, joint, solver, dt=0.002):
    """The twist about z under ``joint_f``'s torque about z: up to 1 N m over 1 s, held 0.5 s, down to 0 over 1 s, held
    1.5 s; per step."""
    control = model.control()
    joint_f = np.zeros(model.joint_dof_count, np.float32)
    z = int(model.joint_qd_start.numpy()[joint]) + 5
    s0, s1 = model.state(), model.state()
    out = []
    for k in range(int(4.0 / dt)):
        t = k * dt
        joint_f[z] = min(t, 1.0) if t < 1.5 else max(0.0, 1.0 - (t - 1.5))
        control.joint_f.assign(joint_f)
        s0.clear_forces()
        solver.step(s0, s1, control, None, dt)
        s0, s1 = s1, s0
        q = s0.body_q.numpy()[0, 3:]
        out.append(2.0 * np.arctan2(q[2], q[3]))
    return np.array(out)


def test_dahl_hysteresis_against_vbd(test, device):
    """Follow SolverVBD's twist under Dahl hysteresis and keep its residual twist.

    A twist with Dahl hysteresis (``eps_max`` 0.5, ``tau`` 0.05) under a torque that rises to 1 N m and falls back to 0
    follows SolverVBD's to 2% of its largest, and after the torque is gone keeps the residual twist SolverVBD keeps
    (2%); without hysteresis less than 5% of it remains.
    """
    model, joint = _twister(device, vbd=True)
    vbd = _torque_cycle(model, joint, newton.solvers.SolverVBD(model, iterations=20))
    model, joint = _twister(device, vbd=False)
    ipc = _torque_cycle(model, joint, newton.solvers.SolverIPC(model, **_TIGHT_ROTATIONS))
    np.testing.assert_allclose(ipc, vbd, rtol=0.0, atol=0.02 * np.abs(vbd).max())
    test.assertGreater(abs(vbd[-1]), 0.05)
    test.assertAlmostEqual(ipc[-1], vbd[-1], delta=0.02 * abs(vbd[-1]))
    model, joint = _twister(device, vbd=False, dahl=(0.0, 0.0))
    plain = _torque_cycle(model, joint, newton.solvers.SolverIPC(model, **_TIGHT_ROTATIONS))
    test.assertLess(abs(plain[-1]), 0.05 * abs(vbd[-1]))  # what is left of its ringing


def test_reset_restarts_the_hysteresis(test, device):
    """Start a rod joint's Dahl stress anew at a reset.

    After the torque cycle the joint keeps a Dahl stress; ``reset`` starts it anew: one step from the same state with
    the stress (the residual twist held) and after a reset (the joint released to its elastic rest).
    """
    model, joint = _twister(device, vbd=False)
    solver = newton.solvers.SolverIPC(model, **_TIGHT_ROTATIONS)
    twist = _torque_cycle(model, joint, solver)
    state = model.state()
    q = state.body_q.numpy()
    q[0, 3:] = [0.0, 0.0, np.sin(0.5 * twist[-1]), np.cos(0.5 * twist[-1])]
    state.body_q.assign(q)
    out = model.state()
    solver.step(state, out, model.control(), None, 0.002)
    held = out.body_qd.numpy()[0, 5]
    solver.reset(state)
    solver.step(state, out, model.control(), None, 0.002)
    released = out.body_qd.numpy()[0, 5]
    test.assertLess(abs(held), 1e-2)
    test.assertLess(released * twist[-1], 0.0)  # the elastic twist turns it back
    test.assertGreater(abs(released), 10.0 * abs(held))


def _hanging(device, n=4, **rod):
    """A cable of ``n`` segments of 0.2 m hanging along -z from a kinematic first segment, gravity -9.81."""
    builder = newton.ModelBuilder()
    points = [wp.vec3(0.0, 0.0, 3.0 - 0.2 * k) for k in range(n + 1)]
    bodies, joints = builder.add_rod(
        rod=newton.Rod(points=points, radius=0.02), bend_stiffness=rod.pop("bend", 5.0), bend_damping=0.1, **rod
    )
    builder.body_flags[bodies[0]] = int(newton.BodyFlags.KINEMATIC)
    return builder.finalize(device=device), bodies, joints


def test_parent_force_is_the_momentum_balance(test, device):
    """Write each cable segment's BODY_PARENT_F as the momentum balance of the segments it holds up.

    A hanging cable of four segments is released from rest with a sideways push: at every step of 20, each dynamic
    segment's ``BODY_PARENT_F`` is the momentum balance of the segments it holds up, itself and those below it, the
    force within 1e-5 of a segment's weight and the torque about its centre of mass within 1e-5 N m (its rod joint's
    terms, ``features.rods.parent_terms``; the last segment's is its own balance).
    """
    model, bodies, _ = _hanging(device)
    solver = newton.solvers.SolverIPC(model, **_TIGHT_ROTATIONS)
    observables = solver.observables(kinds={_KIND.BODY_PARENT_F})
    state = model.state()
    qd = state.body_qd.numpy()
    qd[bodies[-1], 0] = 0.5
    state.body_qd.assign(qd)
    steps, _ = _observed_steps(model, solver, 20, 0.005, observables, state=state)
    weight = float(model.body_mass.numpy()[bodies[1]]) * 9.81
    com = model.body_com.numpy().astype(np.float64)
    for q0, qd0, q1, qd1, parent_f, _ in steps:
        balance = _momentum_balance(model, q0, qd0, q1, qd1, 0.005, gravity=(0.0, 0.0, -9.81))
        centre = [np.asarray(wp.transform_point(wp.transform(*q1[b]), wp.vec3(*com[b])), np.float64) for b in bodies]
        held = np.zeros(6)  # the wrench the segments below take, about the centre of mass of the one above them
        for k in range(len(bodies) - 1, 0, -1):
            b = bodies[k]
            held = balance[b] + held
            np.testing.assert_allclose(parent_f[b, :3], held[:3], rtol=0.0, atol=1e-5 * weight)
            np.testing.assert_allclose(parent_f[b, 3:], held[3:], rtol=0.0, atol=1e-5)
            held = held.copy()
            held[3:] += np.cross(centre[k] - centre[k - 1], held[:3])


def test_joint_coordinates_are_eval_iks(test, device):
    """Write every rod joint's ``joint_q`` and ``joint_qd`` as ``newton.eval_ik`` computes them.

    After 30 steps of a swinging, twisting cable, every rod joint's ``joint_q`` and ``joint_qd`` are what
    ``newton.eval_ik`` computes from the bodies' state (a free joint's: the child's anchor frame in the parent's).
    """
    model, bodies, joints = _hanging(device, twist_stiffness=0.5)
    solver = newton.solvers.SolverIPC(model)
    s0, s1 = model.state(), model.state()
    qd = s0.body_qd.numpy()
    qd[bodies[-1]] = [0.5, 0.0, 0.0, 0.0, 0.0, 3.0]
    s0.body_qd.assign(qd)
    for _ in range(30):
        solver.step(s0, s1, None, None, 0.01)
        s0, s1 = s1, s0
    q, qd = wp.zeros_like(model.joint_q), wp.zeros_like(model.joint_qd)
    newton.eval_ik(model, s0, q, qd)
    starts, dofs = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
    for j in joints:
        got, want = s0.joint_q.numpy()[starts[j] : starts[j] + 7], q.numpy()[starts[j] : starts[j] + 7]
        if np.dot(got[3:], want[3:]) < 0.0:
            want[3:] *= -1.0
        np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-5)
        np.testing.assert_allclose(
            s0.joint_qd.numpy()[dofs[j] : dofs[j] + 6], qd.numpy()[dofs[j] : dofs[j] + 6], rtol=0.0, atol=1e-4
        )


def test_captured_steps_equal_eager_steps(test, device):
    """Step a cable in a captured graph as eager steps do, with its bend stiffness written between replays.

    The cable swings and twists, and the captured steps equal the eager ones to 1e-6.
    """

    def run(capture):
        model, _, joints = _hanging(device, twist_stiffness=0.5)
        solver = newton.solvers.SolverIPC(model, **_TIGHT_ROTATIONS)
        s0, s1 = model.state(), model.state()
        qd = s0.body_qd.numpy()
        qd[-1] = [0.5, 0.0, 0.0, 0.0, 0.0, 2.0]
        s0.body_qd.assign(qd)
        solver.step(s0, s1, None, None, _H)  # loads the kernels
        s0, s1 = s1, s0
        if capture:
            with wp.ScopedCapture(device=device) as cap:
                solver.step(s0, s1, None, None, _H)
                solver.step(s1, s0, None, None, _H)
        ke = model.joint_target_ke.numpy()
        out = []
        for k in range(10):
            for j in joints:
                d = model.joint_qd_start.numpy()[j]
                ke[d + 3 : d + 5] = 5.0 + 2.0 * np.sin(0.7 * k)
            model.joint_target_ke.assign(ke)
            if capture:
                wp.capture_launch(cap.graph)
            else:
                for _ in range(2):
                    solver.step(s0, s1, None, None, _H)
                    s0, s1 = s1, s0
            out.append(s0.body_q.numpy().copy())
        return np.array(out)

    np.testing.assert_allclose(run(True), run(False), rtol=0.0, atol=1e-6)


def test_grown_lists_keep_the_rods(test, device):
    """Keep the rods and their hysteresis when an eager step grows overflowing contact lists.

    A cable with Dahl hysteresis falls onto a box; when it lands the contact lists, made short, overflow, and the eager
    step grows them and solves the step again with the rods and their hysteresis of that step's start: every body as in
    a solver with room, to 1e-6.
    """

    def run(tiny):
        builder = newton.ModelBuilder()
        newton.solvers.SolverVBD.register_custom_attributes(builder)
        builder.add_shape_box(-1, xform=wp.transform((0.5, 0.0, 0.5), wp.quat_identity()), hx=1.0, hy=0.5, hz=0.1)
        straight = newton.Rod.create_straight(
            start=wp.vec3(0.0, 0.0, 0.7), direction=wp.vec3(1.0, 0.0, 0.0), length=1.0, segment_count=5, radius=0.02
        )
        builder.add_rod(rod=straight, bend_stiffness=2.0, bend_damping=0.1)
        model = builder.finalize(device=device)
        rods = model.joint_type.numpy() == int(newton.JointType.ROD)  # the first segment hangs on a free joint
        model.vbd.dahl_eps_max.assign(np.where(rods, 0.3, 0.0).astype(np.float32))
        model.vbd.dahl_tau.assign(np.where(rods, 0.05, 0.0).astype(np.float32))
        solver = newton.solvers.SolverIPC(model, **_TIGHT_ROTATIONS)
        regrown = []
        regrow = solver._regrow
        solver._regrow = lambda: (regrown.append(1), regrow())
        if tiny:
            solver._make_engine(_TINY_LISTS)
        s0, s1 = model.state(), model.state()
        out = []
        for _ in range(40):
            solver.step(s0, s1, None, None, _H)
            s0, s1 = s1, s0
            out.append(s0.body_q.numpy().copy())
        return np.array(out), len(regrown), solver._features.rods.terms.bend_twist.sigma.numpy()

    (roomy, roomy_regrown, roomy_sigma), (grown, grown_regrown, grown_sigma) = run(False), run(True)
    test.assertEqual(roomy_regrown, 0)
    test.assertGreater(grown_regrown, 0)
    test.assertGreater(np.abs(roomy_sigma).max(), 0.0)
    np.testing.assert_allclose(grown, roomy, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(grown_sigma, roomy_sigma, rtol=0.0, atol=1e-6)


def test_gains_and_rests_at_every_step(test, device):
    """Read a rod joint's gains and rest at every step, with or without ``notify_model_changed``.

    A rod joint's gains and rest written into the model after the solver was built act from the next step, with or
    without ``notify_model_changed`` (they are read at every step, as a joint's drives are): a twist stiffness given
    later turns the body, and a rest twist given later turns it the other way.
    """
    model, _, child = _rod(device)
    solver = newton.solvers.SolverIPC(model, **_RIGID)
    s0, s1 = model.state(), model.state()
    q = s0.body_q.numpy()
    q[child] = _pose(axis=(0.0, 0.0, 1.0), angle=0.2)
    s0.body_q.assign(q)
    solver.step(s0, s1, None, None, _H)
    test.assertAlmostEqual(float(s1.body_qd.numpy()[child, 5]), 0.0, delta=1e-6)
    ke = model.joint_target_ke.numpy()
    ke[5] = 20.0
    model.joint_target_ke.assign(ke)
    solver.step(s0, s1, None, None, _H)
    test.assertLess(float(s1.body_qd.numpy()[child, 5]), -1e-3)
    rest = model.joint_rest_q.numpy()
    rest[3:7] = np.asarray(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5))
    model.joint_rest_q.assign(rest)
    solver.notify_model_changed(newton.ModelFlags.JOINT_REFERENCE_POSE_PROPERTIES)
    solver.step(s0, s1, None, None, _H)
    test.assertGreater(float(s1.body_qd.numpy()[child, 5]), 1e-3)


def test_a_disabled_joint_acts_on_nothing(test, device):
    """Let a disabled rod joint's body fall freely, and hold it again once the joint is enabled.

    A rod joint disabled (``joint_enabled`` False) leaves its body to fall freely; enabled again, it holds it.
    """
    model, bodies, joints = _hanging(device, n=2)
    enabled = model.joint_enabled.numpy()
    enabled[joints] = False
    model.joint_enabled.assign(enabled)
    solver = newton.solvers.SolverIPC(model, **_TIGHT_ROTATIONS)
    state = _settle(model, solver, 10, _H)
    np.testing.assert_allclose(state.body_qd.numpy()[bodies[1], 2], -9.81 * 10 * _H, rtol=1e-4)
    enabled[joints] = True
    model.joint_enabled.assign(enabled)
    state = _settle(model, solver, 10, _H)
    test.assertLess(abs(state.body_qd.numpy()[bodies[1], 2]), 0.1)


def test_independent_worlds(test, device):
    """Step two independent worlds of hanging cables as each world alone.

    Two worlds of a hanging cable, the second's bend twice as stiff, are solved as independent worlds: each world's
    bodies are those of the same model alone, to 1e-9 m.
    """

    def model_of(bends, first=0):
        builder = newton.ModelBuilder()
        for k, bend in enumerate(bends, start=first):
            builder.begin_world()
            points = [wp.vec3(2.0 * k, 0.1 * j, 3.0 - 0.2 * j) for j in range(4)]
            bodies, _ = builder.add_rod(
                rod=newton.Rod(points=points, radius=0.02), bend_stiffness=bend, twist_stiffness=1.0
            )
            builder.body_flags[bodies[0]] = int(newton.BodyFlags.KINEMATIC)
            builder.end_world()
        return builder.finalize(device=device)

    def run(model):
        return _settle(model, newton.solvers.SolverIPC(model, **_TIGHT_ROTATIONS), 20, _H).body_q.numpy()

    together = run(model_of([2.0, 4.0]))
    for k, bend in enumerate((2.0, 4.0)):
        alone = run(model_of([bend], first=k))
        np.testing.assert_allclose(together[3 * k : 3 * k + 3, :3], alone[:, :3], rtol=0.0, atol=1e-9)


def test_refusals(test, device):
    """Refuse what a rod joint cannot take, each with its message.

    Among them are an effort limit below Newton's default on a rod's degree of freedom, and SolverVBD's Dahl attributes
    other than 0 on a joint that is not a rod joint or negative on one, which would otherwise take no effect.
    """

    def model_of(dof=None, mode=None, layout=False):
        builder = newton.ModelBuilder()
        link = builder.add_link(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
        joint = builder.add_joint_rod(-1, link, bend_stiffness=1.0)
        builder.add_articulation([joint])
        s = builder.joint_qd_start[joint]
        for name, (k, value) in (dof or {}).items():
            getattr(builder, f"joint_{name}")[s + k] = value
        if mode is not None:
            builder.joint_target_mode[s + 3] = int(mode)
        if layout:
            builder.joint_axis[s], builder.joint_axis[s + 1] = builder.joint_axis[s + 1], builder.joint_axis[s]
        return builder.finalize(device=device)

    def with_dahl(on, eps_max=0.0, tau=0.0):
        """A hinge to the world and a rod joint below it, SolverVBD's Dahl attributes 0 but on the joint ``on``."""
        builder = newton.ModelBuilder()
        newton.solvers.SolverVBD.register_custom_attributes(builder)
        top = builder.add_link(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_box(top, hx=0.1, hy=0.1, hz=0.1)
        hinge = builder.add_joint_revolute(-1, top, axis=(1.0, 0.0, 0.0))
        bottom = builder.add_link(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
        builder.add_shape_box(bottom, hx=0.1, hy=0.1, hz=0.1)
        rod = builder.add_joint_rod(top, bottom, bend_stiffness=1.0)
        builder.add_articulation([hinge, rod])
        model = builder.finalize(device=device)
        j = {"hinge": hinge, "rod": rod}[on]
        for array, value in ((model.vbd.dahl_eps_max, eps_max), (model.vbd.dahl_tau, tau)):
            values = np.zeros(model.joint_count, np.float32)
            values[j] = value
            array.assign(values)
        return model

    takes = "SolverIPC's rod joints take the rod's material response only, as SolverKamino's do"
    remedy = "A rod's stiffness and damping are its joint_target_ke and joint_target_kd, its rest joint_rest_q"
    for name, model, message in (
        ("friction", model_of({"friction": (3, 0.5)}), f"DOF 3 of the ROD joint 0 has joint_friction 0.5; set it to 0. {takes}."),
        ("armature", model_of({"armature": (5, 0.1)}), "DOF 5 of the ROD joint 0 has joint_armature 0.10000000149011612;"),
        ("stiffness", model_of({"stiffness": (4, 2.0)}), f"DOF 4 of the ROD joint 0 has joint_stiffness 2.0; set it to 0. {remedy}."),
        ("damping", model_of({"damping": (2, 1.0)}), "DOF 2 of the ROD joint 0 has joint_damping 1.0; set it to 0."),
        ("target mode", model_of(mode=newton.JointTargetMode.POSITION), "DOF 3 of the ROD joint 0 has the target mode POSITION; set it to NONE. A rod joint has no drive."),
        ("negative gain", model_of({"target_kd": (1, -1.0)}), "DOF 1 of the ROD joint 0 has joint_target_kd -1.0; make it 0 or positive."),
        ("limit", model_of({"limit_lower": (3, -0.5)}), f"DOF 3 of the ROD joint 0 has the limits [-0.5, {newton.MAXVAL!r}] with limit_ke 10000.0; set limit_ke to 0, or the bounds to -MAXVAL and MAXVAL. {takes}."),
        ("effort limit", model_of({"effort_limit": (2, 100.0)}), "DOF 2 of the ROD joint 0 has joint_effort_limit 100.0; raise it to 1e+06, Newton's default. A rod's stiffness and damping are its material, which no drive's effort limit bounds."),
        ("Dahl on another joint", with_dahl("hinge", eps_max=0.5), "the REVOLUTE joint 0 has model.vbd.dahl_eps_max 0.5; set it to 0. Dahl hysteresis acts on rod joints only, as in SolverVBD."),
        ("Dahl's tau on another joint", with_dahl("hinge", tau=0.25), "the REVOLUTE joint 0 has model.vbd.dahl_tau 0.25; set it to 0."),
        ("a negative Dahl attribute", with_dahl("rod", tau=-0.5), "the ROD joint 1 has model.vbd.dahl_tau -0.5; make it 0 or positive."),
        ("layout", model_of(layout=True), "the ROD joint 0 does not have three linear and three angular axes X, Y and Z in order"),
    ):  # fmt: skip
        with test.subTest(name):
            with test.assertRaises(NotImplementedError) as caught:
                newton.solvers.SolverIPC(model)
            test.assertIn(message, str(caught.exception))


@requires("uipc_warp")
class TestSolverIPCRods(unittest.TestCase):
    """Rod joints of Newton's rods and cables."""


devices = get_cuda_test_devices()
for name, func in (
    ("test_kamino_expectations", test_kamino_expectations),
    ("test_joint_f_is_a_world_wrench", test_joint_f_is_a_world_wrench),
    ("test_static_shapes_against_vbd", test_static_shapes_against_vbd),
    ("test_dahl_hysteresis_against_vbd", test_dahl_hysteresis_against_vbd),
    ("test_reset_restarts_the_hysteresis", test_reset_restarts_the_hysteresis),
    ("test_parent_force_is_the_momentum_balance", test_parent_force_is_the_momentum_balance),
    ("test_joint_coordinates_are_eval_iks", test_joint_coordinates_are_eval_iks),
    ("test_captured_steps_equal_eager_steps", test_captured_steps_equal_eager_steps),
    ("test_grown_lists_keep_the_rods", test_grown_lists_keep_the_rods),
    ("test_gains_and_rests_at_every_step", test_gains_and_rests_at_every_step),
    ("test_a_disabled_joint_acts_on_nothing", test_a_disabled_joint_acts_on_nothing),
    ("test_independent_worlds", test_independent_worlds),
    ("test_refusals", test_refusals),
):
    add_function_test(TestSolverIPCRods, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
