# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC's joints switched by ``joint_enabled`` between replays of a captured step, and a loop of joints: a
four-bar linkage closed by a joint outside the articulation."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.ipc._fixtures import _TIGHT
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires

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
    """A four-bar linkage whose loop is closed by a revolute joint outside the articulation (joint_articulation -1)
    falls from crank-up under gravity as a mechanism of one degree of freedom: over 40 steps of 0.01 s the crank turns
    by more than 0.3 rad, the loop joint holds the rocker's end at its world pivot within 0.5 mm, and the rocker's
    angle is the closed form of Freudenstein's equation at the crank's angle within 2e-3 rad, both angles from the
    links' rotations. joint_q of the crank's joint and of the loop joint are the links' rotations from the start, as
    newton.eval_ik has them. Built without poses, the links sit at the model's origin, and the solve is built at the
    poses newton.eval_fk computes from the joint coordinates, the loop joint placing no body; the steps from the state
    eval_fk writes are those of the linkage built at its pose."""
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


devices = get_cuda_test_devices()


@requires("uipc_warp")
class TestSolverIPCJointCases(unittest.TestCase):
    pass


for name, func in (
    ("test_joint_enabled_is_read_at_every_replay", test_joint_enabled_is_read_at_every_replay),
    ("test_four_bar_linkage_closes_its_loop", test_four_bar_linkage_closes_its_loop),
):
    add_function_test(TestSolverIPCJointCases, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
