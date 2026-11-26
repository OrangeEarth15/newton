# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC on a robot arm imported from MJCF, with the joint inputs that imported robots carry.

The arm stands on a base that the importer welds to the world: a shoulder and an elbow hinge, each driven by a MuJoCo
position actuator (``kp``, ``kv``) and carrying an armature, damping, a friction loss and a force range, and a hand on
a ball joint with an armature, damping and a friction loss. Newton's MJCF importer turns these into
``joint_target_ke`` and ``joint_target_kd`` (target mode POSITION), ``joint_armature``, ``joint_damping``,
``joint_friction`` and ``joint_effort_limit``, which SolverIPC takes: the drives as libuipc's driving terms bounded by
the force range, the friction and the ball joint's terms as libuipc-warp's joint terms.

* Driven toward a pose under gravity, the arm comes to rest where every hinge is in balance: the drive's torque and
  the gravity torque of the links beyond the hinge, about its axis and from their resting poses, cancel to within the
  hinge's friction; the hand hangs with the gravity torque about its ball joint within the ball's friction about each
  of the joint's axes.
* With the shoulder's force range below the gravity torque, the shoulder's drive pushes with the force range alone and
  the arm sags until the gravity torque about the shoulder is the force range, within the friction.
* Every step ends without a status bit.
"""

import unittest

import numpy as np

import newton
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

_G = 9.81
_DT = 0.005
_TARGETS = {"shoulder": -0.2, "elbow": 0.3}  # [rad]: the shoulder lifts the arm, the elbow bends it down
# What the balance may miss beyond the friction [N m]: it is taken from Newton's state, the rigid parts of the affine
# bodies, which the joints' loads strain a little, and a step ends once a Newton update turns no link by more than
# 0.4 eps_velocity dt (the tolerance a model with joint friction takes), 2e-5 rad, against which the shoulder's drive
# pushes with 200 * 2e-5 = 4e-3 N m
_SLACK = 0.01

_ARM = """<mujoco model="arm">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <body name="base" pos="0 0 1">
      <geom type="box" size="0.06 0.06 0.04" density="1000"/>
      <body name="upper">
        <joint name="shoulder" type="hinge" axis="0 1 0" armature="0.01" damping="0.5" frictionloss="0.05"
               actuatorfrcrange="-{shoulder_force} {shoulder_force}"/>
        <geom type="capsule" fromto="0.1 0 0 0.35 0 0" size="0.03" density="500"/>
        <body name="fore" pos="0.4 0 0">
          <joint name="elbow" type="hinge" axis="0 1 0" armature="0.01" damping="0.3" frictionloss="0.05"
                 actuatorfrcrange="-30 30"/>
          <geom type="capsule" fromto="0.05 0 0 0.3 0 0" size="0.03" density="500"/>
          <body name="hand" pos="0.35 0 0">
            <joint name="wrist" type="ball" armature="0.001" damping="0.02" frictionloss="0.005"/>
            <geom type="capsule" fromto="0.04 0 0 0.12 0 0" size="0.025" density="500"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position joint="shoulder" kp="200" kv="20"/>
    <position joint="elbow" kp="100" kv="10"/>
  </actuator>
</mujoco>"""


def _arm(device, shoulder_force=60.0):
    """The arm imported from MJCF, with the shoulder's force range ``shoulder_force`` [N m]; the model and the index of
    each joint by its name."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -_G), up_axis=newton.Axis.Z)
    builder.add_mjcf(_ARM.format(shoulder_force=shoulder_force))
    model = builder.finalize(device=device)
    joints = {label.rsplit("/", 1)[-1]: j for j, label in enumerate(builder.joint_label)}
    return model, joints


def _rotate(q, v):
    """The vector ``v`` turned by the unit quaternion ``q`` (x, y, z, w)."""
    u, w = np.asarray(q[:3], np.float64), float(q[3])
    return v + 2.0 * w * np.cross(u, v) + 2.0 * np.cross(u, np.cross(u, v))


def _rest(model, joints, seconds):
    """Drive the arm toward ``_TARGETS`` for ``seconds`` with SolverIPC at its defaults; the final state and the OR of
    every step's status bits."""
    solver = newton.solvers.SolverIPC(model)
    observables = solver.observables({solver.ObservableKind.STATUS})
    control = model.control()
    target = control.joint_target_q.numpy()
    q_start = model.joint_q_start.numpy()
    for name, value in _TARGETS.items():
        target[q_start[joints[name]]] = value
    control.joint_target_q.assign(target)
    s0, s1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    bits = 0
    for _ in range(round(seconds / _DT)):
        solver.step(s0, s1, control, None, _DT, observables=observables)
        bits |= int(observables.status.numpy().ravel()[0])
        s0, s1 = s1, s0
    return s0, bits


def _gravity_torque(model, state, joint, bodies):
    """The gravity torque [N m] about the anchor of ``joint`` of the bodies ``bodies``, from their poses in ``state``,
    and the world frame of the joint on its parent (its anchor, and its rotation as a quaternion)."""
    body_q = state.body_q.numpy().astype(np.float64)
    parent = int(model.joint_parent.numpy()[joint])
    x_p = model.joint_X_p.numpy()[joint].astype(np.float64)
    pose = body_q[parent]
    anchor = pose[:3] + _rotate(pose[3:], x_p[:3])
    # the joint frame's rotation in the world: the parent's rotation, then X_p's (quaternion product, x y z w)
    a, b = pose[3:], x_p[3:]
    frame = np.array(
        [
            a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
            a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
            a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
            a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
        ]
    )
    com = model.body_com.numpy().astype(np.float64)
    mass = model.body_mass.numpy().astype(np.float64)
    torque = np.zeros(3)
    for body in bodies:
        c = body_q[body, :3] + _rotate(body_q[body, 3:], com[body])
        torque += np.cross(c - anchor, mass[body] * np.array([0.0, 0.0, -_G]))
    return torque, frame


def _hinge_balance(model, state, joints, name, bodies):
    """The torque left over at the hinge ``name`` [N m]: its drive's torque, bounded by its force range, plus the
    gravity torque of ``bodies`` about its axis. Returns it, the drive's unbounded torque and the hinge's friction."""
    j = joints[name]
    dof = int(model.joint_qd_start.numpy()[j])
    q = float(state.joint_q.numpy()[int(model.joint_q_start.numpy()[j])])
    qd = float(state.joint_qd.numpy()[dof])
    ke, kd = float(model.joint_target_ke.numpy()[dof]), float(model.joint_target_kd.numpy()[dof])
    effort = float(model.joint_effort_limit.numpy()[dof])
    drive = ke * (_TARGETS[name] - q) - kd * qd
    torque, frame = _gravity_torque(model, state, j, bodies)
    axis = _rotate(frame, model.joint_axis.numpy()[dof].astype(np.float64))
    return float(np.clip(drive, -effort, effort) + torque @ axis), drive, float(model.joint_friction.numpy()[dof])


def test_imported_inputs(test, device):
    """The importer gives the arm the inputs the other tests hold SolverIPC to: the hinges' drives (target mode
    POSITION), armature, damping, friction and force ranges, and the ball joint's armature, damping and friction on
    each of its three degrees of freedom; SolverIPC takes the model."""
    model, joints = _arm(device)
    dof = model.joint_qd_start.numpy()
    shoulder, elbow, wrist = (int(dof[joints[n]]) for n in ("shoulder", "elbow", "wrist"))
    np.testing.assert_allclose(model.joint_target_ke.numpy()[[shoulder, elbow]], [200.0, 100.0])
    np.testing.assert_allclose(model.joint_target_kd.numpy()[[shoulder, elbow]], [20.0, 10.0])
    np.testing.assert_array_equal(
        model.joint_target_mode.numpy()[[shoulder, elbow]], int(newton.JointTargetMode.POSITION)
    )
    np.testing.assert_allclose(model.joint_effort_limit.numpy()[[shoulder, elbow]], [60.0, 30.0])
    np.testing.assert_allclose(model.joint_friction.numpy()[[shoulder, elbow]], [0.05, 0.05])
    np.testing.assert_allclose(model.joint_armature.numpy()[[shoulder, elbow]], [0.01, 0.01])
    np.testing.assert_allclose(model.joint_damping.numpy()[[shoulder, elbow]], [0.5, 0.3])
    np.testing.assert_allclose(model.joint_friction.numpy()[wrist : wrist + 3], 0.005)
    np.testing.assert_allclose(model.joint_armature.numpy()[wrist : wrist + 3], 0.001)
    np.testing.assert_allclose(model.joint_damping.numpy()[wrist : wrist + 3], 0.02)
    newton.solvers.SolverIPC(model)


def test_arm_rests_in_balance(test, device):
    """Driven toward its targets under gravity for 3 s, the arm rests where each hinge's drive torque and the gravity
    torque of the links beyond it cancel to within the hinge's friction, and the hand hangs with the gravity torque about
    its ball joint within the ball's friction about each of the joint's axes, creeping within that band slower than the
    friction's smoothing velocity. Every step ends without a status bit."""
    model, joints = _arm(device)
    state, bits = _rest(model, joints, seconds=3.0)
    test.assertEqual(bits, 0)
    qd, dof = state.joint_qd.numpy(), model.joint_qd_start.numpy()
    test.assertLess(float(np.abs(qd[[dof[joints["shoulder"]], dof[joints["elbow"]]]]).max()), 1.0e-3)
    # within its friction the hand creeps, slower than the friction's smoothing velocity (eps_velocity, 0.01 rad/s)
    test.assertLess(float(np.abs(qd[dof[joints["wrist"]] : dof[joints["wrist"]] + 3]).max()), 0.01)
    upper, fore, hand = (int(model.joint_child.numpy()[joints[n]]) for n in ("shoulder", "elbow", "wrist"))
    for name, bodies in (("shoulder", (upper, fore, hand)), ("elbow", (fore, hand))):
        with test.subTest(hinge=name):
            left, drive, friction = _hinge_balance(model, state, joints, name, bodies)
            test.assertGreater(abs(drive), 0.5)  # the drive holds a load
            test.assertLess(abs(left), friction + _SLACK)
    torque, frame = _gravity_torque(model, state, joints["wrist"], (hand,))
    dof = int(model.joint_qd_start.numpy()[joints["wrist"]])
    for k in range(3):
        with test.subTest(ball_axis=k):
            axis = _rotate(frame, model.joint_axis.numpy()[dof + k].astype(np.float64))
            test.assertLess(abs(float(torque @ axis)), float(model.joint_friction.numpy()[dof + k]) + _SLACK)
    # the hand hangs: its centre of mass, 0.08 m from the wrist, lies more than 0.07 m below it (within 30 degrees)
    body_q = state.body_q.numpy().astype(np.float64)
    wrist = body_q[fore, :3] + _rotate(
        body_q[fore, 3:], model.joint_X_p.numpy()[joints["wrist"], :3].astype(np.float64)
    )
    centre = body_q[hand, :3] + _rotate(body_q[hand, 3:], model.body_com.numpy()[hand].astype(np.float64))
    test.assertGreater(wrist[2] - centre[2], 0.07)


def test_force_range_bounds_the_shoulder(test, device):
    """With the shoulder's force range 2 N m, half the gravity torque of the level arm, the shoulder's drive pushes with
    2 N m alone and the arm sags, after 6 s at rest, until the gravity torque about the shoulder is 2 N m, within the
    friction; the elbow still rests in balance."""
    model, joints = _arm(device, shoulder_force=2.0)
    state, bits = _rest(model, joints, seconds=6.0)  # the saturated drive does not damp: the arm swings down longer
    test.assertEqual(bits, 0)
    qd, dof = state.joint_qd.numpy(), model.joint_qd_start.numpy()
    test.assertLess(float(np.abs(qd[[dof[joints["shoulder"]], dof[joints["elbow"]]]]).max()), 1.0e-3)
    upper, fore, hand = (int(model.joint_child.numpy()[joints[n]]) for n in ("shoulder", "elbow", "wrist"))
    left, drive, friction = _hinge_balance(model, state, joints, "shoulder", (upper, fore, hand))
    test.assertGreater(abs(drive), 2.0 * 5.0)  # the drive would push five times its range
    test.assertLess(abs(left), friction + _SLACK)
    q = float(state.joint_q.numpy()[int(model.joint_q_start.numpy()[joints["shoulder"]])])
    test.assertGreater(q, 0.5)  # sagged by more than half a radian below level
    left, _, friction = _hinge_balance(model, state, joints, "elbow", (fore, hand))
    test.assertLess(abs(left), friction + _SLACK)


class TestSolverIPCImportedArm(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
for _name, _func in (
    ("test_imported_inputs", test_imported_inputs),
    ("test_arm_rests_in_balance", test_arm_rests_in_balance),
    ("test_force_range_bounds_the_shoulder", test_force_range_bounds_the_shoulder),
):
    add_function_test(TestSolverIPCImportedArm, _name, _func, devices=devices, requires=("uipc_warp",))


if __name__ == "__main__":
    unittest.main(verbosity=2)
