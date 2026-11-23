# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC's joints switched by ``joint_enabled`` between replays of a captured step."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires

_TIGHT = {"newton_velocity_tol": 1.0e-6, "linear_tol_rate": 1.0e-12, "semi_implicit": False}
_G = 9.81


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
    """joint_enabled of revolute, prismatic, fixed and ball joints is read at every step, also by a captured one.

    Four spheres hang at their centres from joints to the world, the revolute and ball joints turned by a joint_f
    torque of 5e-3 N m about z and the prismatic one pushed along x by 0.5 N. Over six replays of a captured pair of
    steps every sphere stays put but for its joint's motion, and the driven ones speed up as implicit Euler has them
    (tau dt / I and f dt / m per step). Clearing joint_enabled of the revolute and fixed joints in place between
    replays drops those two spheres into implicit Euler's free fall from where they are, the revolute one spinning on
    at its speed (its joint_f no longer acts), while the prismatic and ball spheres stay held and driven; clearing the
    other two then drops them likewise, the prismatic one sliding on at its speed."""
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


devices = get_cuda_test_devices()


@requires("uipc_warp")
class TestSolverIPCJointCases(unittest.TestCase):
    pass


for name, func in (("test_joint_enabled_is_read_at_every_replay", test_joint_enabled_is_read_at_every_replay),):
    add_function_test(TestSolverIPCJointCases, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
