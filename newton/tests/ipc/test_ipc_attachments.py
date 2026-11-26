# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC's particle-body attachments and springs of rest length 0."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.ipc._fixtures import _TIGHT, _box_body, _implicit_euler_oscillator, _scene_calls, _soft_cube
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires


def _attached_pair(device, stiffness=2.0e4, damping=0.0):
    """VBD's test_body_particle_attachment_is_native_vbd without its damping: a free body of 1 kg (inertia 0.1, no
    shape) and a particle of 1 kg at its origin moving at 1 m/s along x, joined by an attachment at that origin,
    without gravity."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body(mass=1.0, inertia=wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1), lock_inertia=True)
    particle = builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.0), vel=wp.vec3(1.0, 0.0, 0.0), mass=1.0)
    builder.add_attachment_particle_body(particle, body, stiffness=stiffness, damping=damping)
    return builder.finalize(device=device), body, particle


def test_attachment_moves_a_free_body(test, device):
    """Move a free body and an attached particle together as an implicit-Euler two-mass oscillator.

    The particle starts the free body moving: the pair moves together, its momentum stays the particle's 1 kg m/s to
    rounding, and the stretch follows the implicit-Euler recurrence of the two-mass oscillator, omega^2 = k (1 / m_p + 1
    / m_b), from no stretch at a relative speed of 1 m/s, step by step to the Newton tolerance. The body only
    translates: the anchor is at its centre of mass.
    """
    model, body, particle = _attached_pair(device)
    solver = newton.solvers.SolverIPC(model, **_TIGHT)
    base = solver._scene.layout().body_base[body]
    dt, steps, k = 2.0e-3, 20, 2.0e4
    expected = _implicit_euler_oscillator(0.0, np.sqrt(2.0 * k), dt, steps, w0=1.0)
    s0, s1 = model.state(), model.state()
    stretch = []
    for _ in range(steps):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        x, v = solver._engine.positions(), solver._engine.velocities()
        stretch.append(x[particle, 0] - x[base, 0])
        np.testing.assert_allclose(v[particle] + v[base], [1.0, 0.0, 0.0], rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(x[base + 1 : base + 4], np.eye(3), rtol=0.0, atol=1e-12)  # no turn
    np.testing.assert_allclose(stretch, expected, rtol=0.0, atol=1e-9)
    anchor = s0.body_q.numpy()[body, :3]
    test.assertLess(np.linalg.norm(s0.particle_q.numpy()[particle] - anchor), 2.0e-2)  # VBD's bound
    test.assertGreater(anchor[0], 1.0e-3)


def _hung_cloth(device, height=0.3, stiffness=1.0e4, shared=False):
    """VBD's _simulate_cloth_hung_by_attachments: a 4 x 4 cloth hung by its corners 0 and 4 from a body of mass 0
    (static) at ``height``, above the ground. Each corner is attached at its own point of the body (VBD's test attaches
    both to the body's origin, which pulls corner 4 the 8 cm onto corner 0); ``shared``: a third attachment of corner
    1 at corner 0's point."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, height), wp.quat_identity()), mass=0.0)
    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, height), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0), dim_x=4, dim_y=4,
        cell_x=0.02, cell_y=0.02, mass=1.0e-3, tri_ke=1.0e3, tri_ka=1.0e3, tri_kd=0.0, particle_radius=2.0e-3,
    )  # fmt: skip
    corners = [0, 4] + ([1] if shared else [])
    points = [(0.0, 0.0, 0.0), (0.08, 0.0, 0.0), (0.0, 0.0, 0.0)]
    for particle, point in zip(corners, points, strict=False):
        builder.add_attachment_particle_body(particle, body, body_point=point, stiffness=stiffness)
    return builder.finalize(device=device), body, corners


def test_cloth_hangs_from_a_static_body(test, device):
    """Hang a cloth by two corners from a static body with the corners held at their points.

    The corners stay within 1e-4 m of their points on the body (their share of the cloth's weight over the stiffness is
    1.2e-5 m), the rest of the cloth hangs below them, clear of the ground, and no step fails. The anchors are no
    contact primitives; attachments at one point of one body share an anchor.
    """
    model, body, corners = _hung_cloth(device)
    solver = newton.solvers.SolverIPC(model)
    layout = solver._scene.layout()
    anchors = layout.body_anchor_offsets
    test.assertEqual(anchors[body + 1] - anchors[body], 2)
    test.assertFalse(set(range(anchors[0], anchors[-1])) & set(solver._engine.contact_points.tolist()))
    s0, s1 = model.state(), model.state()
    for _ in range(50):
        solver.step(s0, s1, None, None, 1.0e-3)
        s0, s1 = s1, s0
    q = s0.particle_q.numpy()
    np.testing.assert_allclose(q[corners], [[0.0, 0.0, 0.3], [0.08, 0.0, 0.3]], rtol=0.0, atol=1.0e-4)
    test.assertLess(q[:, 2].min(), 0.3 - 0.01)  # the far edge fell
    test.assertGreater(q[:, 2].min(), 0.2)
    shared_model, shared_body, _ = _hung_cloth(device, shared=True)
    shared = newton.solvers.SolverIPC(shared_model)
    offsets = shared._scene.layout().body_anchor_offsets
    test.assertEqual(offsets[shared_body + 1] - offsets[shared_body], 2)  # three attachments, two points


def test_attachment_enabled_inside_a_captured_graph(test, device):
    """Follow ``attachment_particle_body_enabled`` switched between the replays of a captured step.

    A particle is attached to a static body under gravity, and ``attachment_particle_body_enabled`` is switched off for
    ten steps and on again between the replays of a captured step. The steps follow eager ones given the same flags to
    1e-6 m; the particle falls while the attachment is off and is pulled back once it is on.
    """

    def run(capture):
        builder = newton.ModelBuilder()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), mass=0.0)
        builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.9), vel=wp.vec3(0.0, 0.0, 0.0), mass=0.01, radius=0.001)
        builder.add_attachment_particle_body(0, body, body_point=(0.0, 0.0, -0.1), stiffness=50.0)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverIPC(model, **_TIGHT)
        s0, s1 = model.state(), model.state()
        solver.step(s0, s1, None, None, 0.01)  # loads the kernels
        s0, s1 = s1, s0
        graph = None
        if capture:
            with wp.ScopedCapture(device=device) as cap:
                solver.step(s0, s1, None, None, 0.01)
                solver.step(s1, s0, None, None, 0.01)
            graph = cap.graph
        z = []
        for k in range(15):
            model.attachment_particle_body_enabled.assign(np.array([not 3 <= k < 8]))
            if capture:
                wp.capture_launch(graph)
            else:
                for _ in range(2):
                    solver.step(s0, s1, None, None, 0.01)
                    s0, s1 = s1, s0
            z.append(float(s0.particle_q.numpy()[0, 2]))
        return np.array(z)

    eager, captured = run(False), run(True)
    np.testing.assert_allclose(captured, eager, rtol=0.0, atol=1e-6)
    test.assertLess(eager[7], eager[2] - 0.05)  # ten steps of implicit Euler's free fall: g dt^2 55 = 0.054 m
    test.assertGreater(eager[8:].max(), eager[7] + 0.04)


def test_hard_attachment_holds_as_a_ball_joint(test, device):
    """Hold a hard attachment with the penalty of a ball joint.

    A particle of mass m hangs from a static body by an attachment of infinite stiffness. Its penalty is that of a ball
    joint, joint_strength_ratio gamma times the dynamic mass over dt^2: under gravity the particle rests g dt^2 / gamma
    below its point (9.8e-6 m at dt 0.01 s), to 1%, and twice as far with half the ratio.
    """
    for gamma in (100.0, 50.0):
        builder = newton.ModelBuilder()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), mass=0.0)
        builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.9), vel=wp.vec3(0.0, 0.0, 0.0), mass=0.02, radius=0.001)
        builder.add_attachment_particle_body(0, body, body_point=(0.0, 0.0, -0.1), stiffness=float("inf"))
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverIPC(model, joint_strength_ratio=gamma, **_TIGHT)
        s0, s1 = model.state(), model.state()
        for _ in range(20):
            solver.step(s0, s1, None, None, 0.01)
            s0, s1 = s1, s0
        sag = 0.9 - solver._engine.positions()[0, 2]
        test.assertAlmostEqual(sag / (9.81 * 0.01**2 / gamma), 1.0, delta=0.01)


def test_zero_length_springs_oscillate_as_implicit_euler(test, device):
    """Oscillate particles joined by springs of rest length 0 as the implicit-Euler oscillator.

    Springs of rest length 0 are libuipc's soft vertex stitch. Without gravity, two free particles of mass m 1 cm apart,
    and a free particle 1 cm from a kinematic one, follow the implicit-Euler oscillator of omega^2 = 2 k / m and k / m
    step by step to the Newton tolerance, and the free pair's centre stays put. The particles have no radius, so that
    they do not collide with each other (a spring changes no contact pair).
    """
    m, k, dt, steps = 0.01, 4.0, 0.01, 60
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for x, y, mass in ((0.0, 0.0, m), (0.01, 0.0, m), (0.0, 1.0, 0.0), (0.01, 1.0, m)):
        builder.add_particle(wp.vec3(x, y, 1.0), wp.vec3(0.0, 0.0, 0.0), mass, radius=0.0)
    for i in (0, 2):
        builder.add_spring(i, i + 1, k, 0.0, 0.0)
        builder.spring_rest_length[-1] = 0.0
    model = builder.finalize(device=device)
    import uipc_warp.arrays  # noqa: PLC0415

    with _scene_calls(uipc_warp.arrays.ArrayScene.add_stitches) as stitches:
        solver = newton.solvers.SolverIPC(model, **_TIGHT)
    [given] = stitches  # one stitch between particles 0 and 1 and one between 2 and 3: one group, of two particles
    pairs = np.sort(np.asarray(given["vertices"]).reshape(-1, 2), axis=1)
    np.testing.assert_array_equal(pairs[np.lexsort(pairs.T[::-1])], [[0, 1], [2, 3]])
    test.assertEqual(len(solver._uipc_warp.arrays.stitch_groups(solver._engine)), 1)
    x0 = solver._engine.positions()[:4, 0]
    expected = {
        0: _implicit_euler_oscillator(x0[1] - x0[0], np.sqrt(2.0 * k / m), dt, steps),
        2: _implicit_euler_oscillator(x0[3] - x0[2], np.sqrt(k / m), dt, steps),
    }
    s0, s1 = model.state(), model.state()
    measured = {first: [] for first in expected}
    for _ in range(steps):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        x = solver._engine.positions()[:4, 0]
        for first in expected:
            measured[first].append(x[first + 1] - x[first])
        test.assertAlmostEqual(x[0] + x[1], x0[0] + x0[1], delta=1e-12)
    for first, u in expected.items():
        np.testing.assert_allclose(measured[first], u, rtol=0.0, atol=1e-9)


def test_attachment_changes(test, device):
    """Take attachment changes notified with CONSTRAINT_PROPERTIES, building the solve again only where needed.

    After notify_model_changed(CONSTRAINT_PROPERTIES), a new stiffness is taken at the next step without building the
    solve again; a damping where the solve has none builds it again, with the damping of the attachment's stitch, and a
    damping changed after that is taken at the next step without building it again; a new body point builds the solve
    again.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    b = _box_body(builder, (0.0, 0.0, 1.0), (0.1, 0.1, 0.1))
    builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.8), vel=wp.vec3(0.0, 0.0, 0.0), mass=0.01, radius=0.001)
    builder.add_attachment_particle_body(0, b, body_point=(0.0, 0.0, -0.18), stiffness=10.0)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model)
    engine = solver._engine
    model.attachment_particle_body_stiffness.assign(np.array([20.0], np.float32))
    solver.notify_model_changed(newton.ModelFlags.CONSTRAINT_PROPERTIES)
    test.assertIs(solver._engine, engine)
    s0, s1 = model.state(), model.state()
    solver.step(s0, s1, None, None, 0.01)
    [rows] = solver._uipc_warp.arrays.stitch_groups(solver._engine)
    np.testing.assert_array_equal(rows.kappa.numpy(), [20.0])
    model.attachment_particle_body_damping.assign(np.array([1.0], np.float32))
    solver.notify_model_changed(newton.ModelFlags.CONSTRAINT_PROPERTIES)
    test.assertIsNot(solver._engine, engine)
    engine = solver._engine
    model.attachment_particle_body_damping.assign(np.array([2.0], np.float32))
    solver.notify_model_changed(newton.ModelFlags.CONSTRAINT_PROPERTIES)
    test.assertIs(solver._engine, engine)
    solver.step(s0, s1, None, None, 0.01)
    np.testing.assert_array_equal(solver._features.damping.stitches.group.c.numpy(), [2.0])
    model.attachment_particle_body_body_point.assign(np.array([[0.0, 0.0, -0.19]], np.float32))
    solver.notify_model_changed(newton.ModelFlags.CONSTRAINT_PROPERTIES)
    test.assertIsNot(solver._engine, engine)


def test_no_stitches_without_attachments(test, device):
    """Give the engine no stitch, anchor or contact surface of its own for a model that needs none.

    A model without attachments and springs of rest length 0 gives the engine no stitch and no anchor, and a model
    without springs no contact surface of its own.
    """
    model = _soft_cube(device, height=0.3)
    builder = newton.ModelBuilder()
    _box_body(builder, (0.0, 0.0, 0.5), (0.1, 0.1, 0.1))
    builder.add_ground_plane()
    for m in (model, builder.finalize(device=device)):
        solver = newton.solvers.SolverIPC(m)
        test.assertEqual(solver._uipc_warp.arrays.stitch_groups(solver._engine), [])
        layout = solver._scene.layout()
        test.assertEqual(layout.body_anchor_offsets[-1], layout.n)
        test.assertEqual(layout.body_anchor_offsets[0], layout.n)
        test.assertIsNone(solver._scene.engine_inputs(solver._config, "cpu")["surface"])


@requires("uipc_warp")
class TestSolverIPCAttachments(unittest.TestCase):
    """Particle-body attachments and springs of rest length 0 as libuipc's soft vertex stitch."""


devices = get_cuda_test_devices()
for name, func in (
    ("test_attachment_moves_a_free_body", test_attachment_moves_a_free_body),
    ("test_cloth_hangs_from_a_static_body", test_cloth_hangs_from_a_static_body),
    ("test_attachment_enabled_inside_a_captured_graph", test_attachment_enabled_inside_a_captured_graph),
    ("test_hard_attachment_holds_as_a_ball_joint", test_hard_attachment_holds_as_a_ball_joint),
    ("test_zero_length_springs_oscillate_as_implicit_euler", test_zero_length_springs_oscillate_as_implicit_euler),
    ("test_attachment_changes", test_attachment_changes),
    ("test_no_stitches_without_attachments", test_no_stitches_without_attachments),
):
    add_function_test(TestSolverIPCAttachments, name, func, devices=devices)


def _particles_on_a_box(device, worlds=1):
    """A 0.1 m box of 1 kg 1 mm above the ground, particle 0 attached at the centre of its top face (stiffness 1e4) and
    particle 1, free, 5 mm above that face 3 cm from the centre, both of radius 1 mm and 10 g; ``worlds`` copies of it,
    each in a world of its own. Returns the model."""
    builder = newton.ModelBuilder()
    box = _box_body(builder, (0.0, 0.0, 0.051), (0.05, 0.05, 0.05))
    builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.101), vel=wp.vec3(0.0, 0.0, 0.0), mass=0.01, radius=1.0e-3)
    builder.add_particle(pos=wp.vec3(0.03, 0.0, 0.107), vel=wp.vec3(0.0, 0.0, 0.0), mass=0.01, radius=1.0e-3)
    builder.add_attachment_particle_body(0, box, body_point=(0.0, 0.0, 0.05), stiffness=1.0e4)
    if worlds == 1:
        builder.add_ground_plane()
        return builder.finalize(device=device)
    scene = newton.ModelBuilder()
    scene.add_ground_plane()
    scene.replicate(builder, worlds)
    return scene.finalize(device=device)


def test_attached_particle_skips_its_body(test, device):
    """Keep a particle attached to a body from colliding with that body.

    A particle attached at a point of the body's surface starts in contact with it, which the barrier refuses. Particle
    0, attached at the centre of a box's top face, takes a contact element that collides with what a free particle's
    does but with that box; the free particle 1 still collides with the box and lands on it, and particle 0 holds its
    point through 100 steps of 0.01 s (within 1e-4 m; its weight over the stiffness is 1e-5 m) with no failed step. In
    two replicated worlds the attached particles share one element, which the free ones do not.
    """
    model = _particles_on_a_box(device)
    solver = newton.solvers.SolverIPC(model)
    e = solver._enabled
    box_shape = int(model.body_shapes[0][0])
    attached, free = (int(solver._cids[k]) for k in (0, 1))
    box = int(solver._cids[solver._vertex_shape == box_shape][0])
    test.assertNotEqual(attached, free)
    test.assertFalse(e[attached, box])
    test.assertTrue(e[free, box] and e[attached, free] and e[attached, attached])
    test.assertTrue(e[attached, solver._plane_cids[0]] and e[free, solver._plane_cids[0]])
    s0, s1 = model.state(), model.state()
    for _ in range(100):
        solver.step(s0, s1, None, None, 0.01)
        s0, s1 = s1, s0
    q, top = s0.particle_q.numpy(), s0.body_q.numpy()[0, :3] + (0.0, 0.0, 0.05)
    np.testing.assert_allclose(q[0], top, rtol=0.0, atol=1.0e-4)
    test.assertGreater(q[1, 2] - top[2], 1.0e-3)  # on the box: its radius, and the barrier's gap
    test.assertLess(q[1, 2] - top[2], 1.0e-3 + 0.01)
    replicated = newton.solvers.SolverIPC(_particles_on_a_box(device, worlds=2))
    test.assertEqual(int(replicated._cids[0]), int(replicated._cids[2]))
    test.assertNotEqual(int(replicated._cids[0]), int(replicated._cids[1]))


@requires("uipc_warp")
class TestSolverIPCAttachedContact(unittest.TestCase):
    """A particle attached to a body does not collide with that body."""


for name, func in (("test_attached_particle_skips_its_body", test_attached_particle_skips_its_body),):
    add_function_test(TestSolverIPCAttachedContact, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
