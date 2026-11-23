# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC's CONTACT_F: its own contact terms written into the step's Contacts."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton.tests.ipc._fixtures import (
    _TINY_LISTS,
    _box_body,
    _boxes_apart,
    _cloth,
    _native_contacts,
    _rigid_rows,
    _stacking_cubes,
    _transform_points,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires

_CONTACT_F = newton.solvers.SolverBase.ObservableKind.CONTACT_F


def _stacking(device, density=1000.0, lower_z=0.8, upper=True):
    """Newton's stacking scene of test_sensor_contact.py: a static base box with its top at z = 0.25, a 0.3 x 0.3 x
    0.5 m box "a" 30 cm above it and a 0.2 x 0.2 x 0.1 m box "b" 5 cm above that."""
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.density = density
    builder.add_shape_box(body=-1, hx=1.0, hy=1.0, hz=0.25, label="base")
    a = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, lower_z), wp.quat_identity()), label="a")
    builder.add_shape_box(a, hx=0.15, hy=0.15, hz=0.25)
    if upper:
        b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, lower_z + 0.35), wp.quat_identity()), label="b")
        builder.add_shape_box(b, hx=0.1, hy=0.1, hz=0.05)
    return builder.finalize(device=device)


def _contact_steps(model, solver, contacts, observables, steps, dt, capture=False, state=None):
    """``steps`` steps (an even number when captured) from ``state`` (the model's), the observables written by each;
    returns the last output state."""
    s0 = model.state() if state is None else state
    s1 = model.state()
    if not capture:
        for _ in range(steps):
            solver.step(s0, s1, None, contacts, dt, observables=observables)
            s0, s1 = s1, s0
        return s0
    solver.step(s0, s1, None, contacts, dt, observables=observables)  # loads the state and the modules
    s0, s1 = s1, s0
    solver.step(s0, s1, None, contacts, dt, observables=observables)
    s0, s1 = s1, s0
    with wp.ScopedCapture(device=model.device) as capture_:
        solver.step(s0, s1, None, contacts, dt, observables=observables)
        solver.step(s1, s0, None, contacts, dt, observables=observables)
    for _ in range((steps - 2) // 2):
        wp.capture_launch(capture_.graph)
        solver.check_status(observables)
    return s0


def test_contact_output_stacking_reports_weights(test, device):
    """Newton's stacking test with SolverIPC as a native contact backend, its steps captured: after 2 s at 1/240 s
    SensorContact, reading the rows the steps write, reports each box's weight within 1% (a: 45 kg, b: 4 kg, as the
    base holds a and b, and a holds b), the force-weighted contact positions sit at the interfaces within 5 cm (IPC
    rests up to d_hat apart: a on the base at z = 0.25, b on a at z = 0.75), and b never touches the base."""
    from newton.sensors import SensorContact  # noqa: PLC0415

    model = _stacking(device)
    solver = newton.solvers.SolverIPC(model)
    sensor = SensorContact(model, request_contact_attributes=False, sensing_bodies=["a", "b"], counterpart_shapes="*")
    contacts = _native_contacts(solver, model)
    kinds = set(sensor.solver_observable_kinds) | {solver.ObservableKind.STATUS}
    observables = solver.observables(kinds, contacts=contacts)
    test.assertIs(observables.contacts, contacts)
    state = _contact_steps(model, solver, contacts, observables, 470, 1.0 / 240.0, capture=True)
    s1 = model.state()
    total = np.zeros((2, 3))
    for _ in range(10):
        solver.step(state, s1, None, contacts, 1.0 / 240.0, observables=observables)
        state, s1 = s1, state
        sensor.update(state, contacts, observables=observables)
        total += sensor.total_force.numpy()
    total /= 10
    g = 9.81
    test.assertAlmostEqual(total[0, 2], 45.0 * g, delta=45.0 * g * 0.01)
    test.assertAlmostEqual(total[1, 2], 4.0 * g, delta=4.0 * g * 0.01)
    shape_base, shape_a, shape_b = 0, 1, 2
    row_a, row_b = sensor.sensing_indices.index(0), sensor.sensing_indices.index(1)
    positions = sensor.position_matrix.numpy()
    np.testing.assert_allclose(
        positions[row_a, sensor.counterpart_indices[row_a].index(shape_base)], [0, 0, 0.25], atol=0.05
    )
    np.testing.assert_allclose(
        positions[row_a, sensor.counterpart_indices[row_a].index(shape_b)], [0, 0, 0.75], atol=0.05
    )
    np.testing.assert_allclose(
        positions[row_b, sensor.counterpart_indices[row_b].index(shape_a)], [0, 0, 0.75], atol=0.05
    )
    np.testing.assert_array_equal(positions[row_b, sensor.counterpart_indices[row_b].index(shape_base)], 0.0)
    shape0, shape1, _ = _rigid_rows(contacts, observables)
    pairs = {tuple(sorted(p)) for p in zip(shape0.tolist(), shape1.tolist(), strict=True)}
    test.assertEqual(pairs, {(shape_base, shape_a), (shape_a, shape_b)})


def test_contact_output_friction_at_rest(test, device):
    """Newton's stacking-friction test: a 4.5 kg box at rest on the base reports its weight within 2% and a friction
    force within 2% of it, its barrier rows pushing along their normals and its friction rows holding no more than
    the slip of a body at rest asks."""
    from newton.sensors import SensorContact  # noqa: PLC0415

    model = _stacking(device, density=100.0, upper=False)
    solver = newton.solvers.SolverIPC(model)
    sensor = SensorContact(model, request_contact_attributes=False, sensing_bodies=["a"])
    contacts = _native_contacts(solver, model)
    observables = solver.observables(sensor.solver_observable_kinds, contacts=contacts)
    state = _contact_steps(model, solver, contacts, observables, 470, 1.0 / 240.0)
    s1 = model.state()
    total, friction = np.zeros(3), np.zeros(3)
    for _ in range(10):
        solver.step(state, s1, None, contacts, 1.0 / 240.0, observables=observables)
        state, s1 = s1, state
        sensor.update(state, contacts, observables=observables)
        total += sensor.total_force.numpy()[0]
        friction += sensor.total_force_friction.numpy()[0]
    weight = 4.5 * 9.81
    test.assertAlmostEqual(total[2] / 10, weight, delta=weight * 0.02)
    np.testing.assert_allclose(friction / 10, 0.0, atol=weight * 0.02)


def test_contact_output_soft_rows(test, device):
    """A cloth draped over a static box and a free particle on the ground: the soft rows' forces on the shapes add up
    to the cloth's and the particle's weights within 1%; every row names a shape, its record is a particle, an edge or
    a face of the cloth (all three occur) with weights that add up to 1, its point lies on the shape's surface, its
    normal points from that point toward the record's point, and its torque is that of its force at its point about
    the world origin (static shapes); contact of the cloth with itself gives no row, and there are no rigid rows."""
    builder = newton.ModelBuilder()
    _cloth(builder, 12, 0.02, 0.2050, mass=1.0e-3, tri_k=1.0e3, edge_ke=1.0e-3)  # 2 cm wider than the box each way
    builder.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()), hx=0.1, hy=0.1, hz=0.1)
    builder.add_particle(wp.vec3(0.5, 0.0, 0.0101), wp.vec3(0.0, 0.0, 0.0), 0.01, radius=0.01)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model)
    test.assertEqual(solver.get_max_soft_contact_count(), solver.get_max_contact_count())
    contacts = _native_contacts(solver, model)
    observables = solver.observables({_CONTACT_F}, contacts=contacts)
    state = _contact_steps(model, solver, contacts, observables, 200, 0.01)  # the flaps settle after 1.5 s
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
    n = int(contacts.soft_contact_count.numpy()[0])
    test.assertGreater(n, 0)
    test.assertLessEqual(n, contacts.soft_contact_max)
    shape = contacts.soft_contact_shape.numpy()[:n]
    indices = contacts.soft_contact_indices.numpy()[:n]
    weights = contacts.soft_contact_barycentric.numpy()[:n]
    particle = contacts.soft_contact_particle.numpy()[:n]
    pos = contacts.soft_contact_body_pos.numpy()[:n]
    normal = contacts.soft_contact_normal.numpy()[:n]
    wrench = observables.contact_f.numpy()[contacts.rigid_contact_max : contacts.rigid_contact_max + n]
    box, ground = 0, 1
    cloth = np.arange(model.particle_count - 1)
    free = model.particle_count - 1
    test.assertTrue(np.isin(shape, [box, ground]).all())
    kinds = (indices >= 0).sum(axis=1)
    test.assertTrue(np.isin(kinds, [1, 2, 3]).all())
    test.assertTrue({1, 2, 3}.issubset(set(kinds.tolist())))  # particle, edge and face records
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(particle, np.where(kinds == 1, indices[:, 0], -1))
    on_box = shape == box
    test.assertTrue(np.isin(indices[on_box][indices[on_box] >= 0], cloth).all())
    np.testing.assert_array_equal(indices[~on_box], [[free, -1, -1]] * int((~on_box).sum()))
    # on the surface of the box [-0.1, 0.1]^2 x [0, 0.2] (a static shape's frame is the world's), and on the ground
    surface = np.abs(pos[on_box] - [0.0, 0.0, 0.1]).max(axis=1)
    np.testing.assert_allclose(surface, 0.1, atol=1e-5)
    np.testing.assert_allclose(pos[~on_box, 2], 0.0, atol=1e-6)
    q = state.particle_q.numpy()
    soft_point = np.einsum("ij,ijk->ik", weights, q[np.maximum(indices, 0)] * (indices >= 0)[:, :, None])
    test.assertTrue((np.einsum("ij,ij->i", normal, soft_point - pos) > 0.0).all())
    np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1.0, atol=1e-5)
    cloth_weight = model.particle_mass.numpy()[cloth].sum() * 9.81
    np.testing.assert_allclose(wrench[on_box, :3].sum(axis=0), [0.0, 0.0, -cloth_weight], atol=0.01 * cloth_weight)
    np.testing.assert_allclose(wrench[~on_box, :3].sum(axis=0), [0.0, 0.0, -0.01 * 9.81], atol=0.01 * 0.01 * 9.81)
    # the torque about the world origin, of the forces on the box's vertices: that of the force at the point where the
    # force splits by the closest point's weights (a particle or face record: a triangle's or a single vertex's)
    split = kinds != 2
    torque = np.cross(pos[split], wrench[split, :3])
    np.testing.assert_allclose(wrench[split, 3:], torque, rtol=0, atol=1e-5 * max(np.abs(torque).max(), 1e-3))


def test_contact_output_rigid_rows_balance_a_body(test, device):
    """A box held on a static ramp by friction, its steps converged tightly (at the default newton_velocity_tol the
    box creeps down at 1.2 mm/s, and its forces balance only to that tolerance, 0.12 N of its 9.81 N): its rows'
    forces add up to its weight and their torques about its centre of mass to zero, each row holding the force on its
    shape 0 with its torque about that body's centre of mass at the end of the step (the world origin for the static
    ramp); the points lie on the box's bottom face in its frame of state_out; the margins are the shapes'
    shape_margin, the offsets the normal times them, and every normal is the ramp's."""
    builder = newton.ModelBuilder()
    angle = 0.2
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle)
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.8, margin=0.002)
    builder.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), rot), hx=0.5, hy=0.5, hz=0.05, cfg=cfg)
    up = np.array(wp.quat_rotate(rot, wp.vec3(0.0, 0.0, 1.0)))
    body = builder.add_body(xform=wp.transform(wp.vec3(*(up * (0.05 + 0.05 + 0.0045))), rot))
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
    model = builder.finalize(device=device)
    tight = {"newton_velocity_tol": 1e-4, "linear_tol_rate": 1e-10, "semi_implicit": False}
    solver = newton.solvers.SolverIPC(model, d_hat=0.005, **tight)
    contacts = _native_contacts(solver, model)
    observables = solver.observables({_CONTACT_F}, contacts=contacts)
    state = _contact_steps(model, solver, contacts, observables, 60, 0.01)
    shape0, _, f = _rigid_rows(contacts, observables)
    test.assertGreater(len(shape0), 0)
    sign = np.where(shape0 == 1, 1.0, -1.0)  # the box's shape is 1: a row with it as shape 0 holds the force on it
    on_box = sign[:, None] * f[:, :3]
    np.testing.assert_allclose(on_box.sum(axis=0), [0.0, 0.0, model.body_mass.numpy()[0] * 9.81], rtol=0, atol=1e-3)
    # torques about the box's centre: a row of the box as shape 0 holds it; a row of the ramp as shape 0 holds the
    # torque about the world origin, moved here to the box's centre
    X = state.body_q.numpy()[0]
    com = _transform_points(X, model.body_com.numpy()[0])[0]
    n = len(shape0)
    point1 = _transform_points(X, contacts.rigid_contact_point1.numpy()[:n])
    # a row of the box as shape 0 holds the torque about its centre; a row of the ramp as shape 0 the force on the
    # ramp, whose reaction acts on the box at the box's point
    torque = np.where(shape0[:, None] == 1, f[:, 3:], np.cross(point1 - com, -f[:, :3]))
    test.assertLess(np.linalg.norm(torque.sum(axis=0)), 1e-3 * model.body_mass.numpy()[0] * 9.81 * 0.05)
    local = np.where(
        shape0[:, None] == 1, contacts.rigid_contact_point0.numpy()[:n], contacts.rigid_contact_point1.numpy()[:n]
    )
    np.testing.assert_allclose(local[:, 2], -0.05, atol=1e-4)  # on the box's bottom face, in its body frame
    test.assertTrue((np.abs(local[:, :2]) <= 0.05 + 1e-4).all())
    margin0, margin1 = contacts.rigid_contact_margin0.numpy()[:n], contacts.rigid_contact_margin1.numpy()[:n]
    np.testing.assert_allclose(margin0, 0.002, rtol=1e-6)
    np.testing.assert_allclose(margin1, 0.002, rtol=1e-6)
    normal = contacts.rigid_contact_normal.numpy()[:n]
    offset0 = contacts.rigid_contact_offset0.numpy()[:n]
    ramp_rows = shape0 == 0
    np.testing.assert_allclose(offset0[ramp_rows], 0.002 * normal[ramp_rows], atol=1e-7)  # the static ramp's frame
    np.testing.assert_allclose(np.abs(normal @ up), 1.0, atol=1e-3)  # every row's normal is the ramp's


def test_contact_output_only_enabled_pairs(test, device):
    """Rows only between pairs that collide: a box made a filter pair with the box under it falls through it to the
    ground, and no row joins the two; the other pairs give rows."""
    builder = newton.ModelBuilder()
    ground = builder.add_ground_plane()
    lower = builder.shape_count
    _box_body(builder, (0.0, 0.0, 0.051), (0.05, 0.05, 0.05))
    upper = builder.shape_count
    _box_body(builder, (0.02, 0.0, 0.16), (0.03, 0.03, 0.03))
    builder.add_shape_collision_filter_pair(lower, upper)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model)
    contacts = _native_contacts(solver, model)
    observables = solver.observables({_CONTACT_F}, contacts=contacts)
    pairs = set()
    s0, s1 = model.state(), model.state()
    for _ in range(40):
        solver.step(s0, s1, None, contacts, 0.01, observables=observables)
        s0, s1 = s1, s0
        shape0, shape1, _ = _rigid_rows(contacts, observables)
        pairs |= {tuple(sorted(p)) for p in zip(shape0.tolist(), shape1.tolist(), strict=True)}
    test.assertNotIn((lower, upper), pairs)
    test.assertEqual(pairs, {tuple(sorted((lower, ground))), tuple(sorted((upper, ground)))})


def test_contact_output_capacity(test, device):
    """A Contacts smaller than the solver's worst case is taken: the counts are the true numbers, the rows past the
    capacity are dropped and the rows within it hold the same terms the full buffer's rows do."""
    model = _stacking(device, upper=False, lower_z=0.502)
    solver = newton.solvers.SolverIPC(model)
    full = _native_contacts(solver, model)
    observables = solver.observables({_CONTACT_F}, contacts=full)
    state = _contact_steps(model, solver, full, observables, 20, 0.01)
    rows = int(full.rigid_contact_count.numpy()[0])
    test.assertGreater(rows, 3)
    small = newton.Contacts(3, 0, device=device)
    s1 = model.state()
    solver.step(state, s1, None, small, 0.01, observables=observables)
    test.assertIs(observables.contacts, small)
    test.assertEqual(int(small.rigid_contact_count.numpy()[0]), rows)
    test.assertTrue(np.isin(small.rigid_contact_shape0.numpy(), [0, 1]).all())
    test.assertTrue((np.linalg.norm(observables.contact_f.numpy()[:3, :3], axis=1) > 0.0).all())
    np.testing.assert_array_equal(observables.contact_f.numpy()[3:], 0.0)
    # no soft rows can come: the soft count stays 0 with a buffer that holds none
    test.assertEqual(int(small.soft_contact_count.numpy()[0]), 0)
    with test.assertRaisesRegex(ValueError, "Pass the step's Contacts"):
        solver.step(s1, state, None, None, 0.01, observables=observables)


def test_contact_output_not_requested(test, device):
    """Without CONTACT_F nothing of the output exists or runs: no records are allocated, a step launches none of its
    kernels, and the Contacts passed to step is not touched; requesting it allocates the records once."""
    model = _stacking(device, upper=False, lower_z=0.502)
    solver = newton.solvers.SolverIPC(model)
    contacts = newton.Contacts(16, 4, device=device)
    contacts.rigid_contact_count.fill_(7)
    contacts.soft_contact_count.fill_(5)
    observables = solver.observables({solver.ObservableKind.STATUS})
    launched = []
    launch = wp.launch

    def recording(kernel, *args, **kwargs):
        launched.append(getattr(kernel, "key", str(kernel)))
        return launch(kernel, *args, **kwargs)

    s0, s1 = model.state(), model.state()
    with mock.patch.object(wp, "launch", recording):
        for _ in range(3):
            solver.step(s0, s1, None, contacts, 0.01, observables=observables)
            s0, s1 = s1, s0
    test.assertIsNone(solver._contact_export)
    test.assertGreater(len(launched), 0)
    test.assertFalse([k for k in launched if "contact_rows" in k or "contact_force_records" in k])
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 7)
    test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 5)
    observables = solver.observables({_CONTACT_F}, contacts=contacts)
    export = solver._contact_export
    test.assertIsNotNone(export)
    solver.observables({_CONTACT_F}, contacts=contacts)
    test.assertIs(solver._contact_export, export)
    _contact_steps(model, solver, contacts, observables, 20, 0.01, state=s0)  # the box bounces off the base first
    test.assertEqual(int(contacts.soft_contact_count.numpy()[0]), 0)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_contact_output_captured_equals_eager(test, device):
    """Captured steps write the rows eager steps write: the same pairs with the same forces summed per pair, to the
    rounding of the solve, over 40 steps of the stacking scene."""

    def run(capture):
        model = _stacking(device)
        solver = newton.solvers.SolverIPC(model, newton_velocity_tol=1e-6, linear_tol_rate=1e-12, semi_implicit=False)
        contacts = _native_contacts(solver, model)
        observables = solver.observables({_CONTACT_F, solver.ObservableKind.STATUS}, contacts=contacts)
        _contact_steps(model, solver, contacts, observables, 120, 1.0 / 60.0, capture=capture)
        shape0, shape1, f = _rigid_rows(contacts, observables)
        totals = {}
        for a, b, w in zip(shape0.tolist(), shape1.tolist(), f, strict=True):
            key, sign = ((a, b), 1.0) if a < b else ((b, a), -1.0)
            totals[key] = totals.get(key, 0.0) + sign * w[:3]
        return totals

    eager, captured = run(False), run(True)
    test.assertEqual(set(eager), set(captured))
    for key in eager:
        np.testing.assert_allclose(captured[key], eager[key], rtol=0, atol=1e-3 * np.abs(eager[key]).max())


def test_contact_output_failed_step_has_no_rows(test, device):
    """A step that fails writes no rows: an eager step from a state through the ground raises and leaves zero counts
    and forces; the next step from a valid state writes rows again."""
    model, touching = _boxes_apart(device)
    solver = newton.solvers.SolverIPC(model)
    contacts = _native_contacts(solver, model)
    observables = solver.observables({_CONTACT_F}, contacts=contacts)
    s0, s1 = model.state(), model.state()
    solver.step(s0, s1, None, contacts, 0.01, observables=observables)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
    s0.body_q.assign(touching)
    with test.assertRaises(RuntimeError):
        solver.step(s0, s1, None, contacts, 0.01, observables=observables)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
    np.testing.assert_array_equal(observables.contact_f.numpy(), 0.0)
    s0 = model.state()
    solver.step(s0, s1, None, contacts, 0.01, observables=observables)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


def test_contact_output_after_grow(test, device):
    """Contact lists that overflow a captured step and grow with grow(): the export follows the grown solve, whose
    get_max_contact_count() is larger, and the steps captured again write their rows."""
    model = _stacking_cubes(device)
    solver = newton.solvers.SolverIPC(model)
    solver._make_engine(_TINY_LISTS)
    contacts = newton.Contacts(4096, 4096, device=device)
    observables = solver.observables({_CONTACT_F, solver.ObservableKind.STATUS}, contacts=contacts)
    before = solver.get_max_soft_contact_count()
    s0, s1 = model.state(), model.state()

    def capture():
        with wp.ScopedCapture(device=device) as cap:
            solver.step(s0, s1, None, contacts, 0.01, observables=observables)
        return cap.graph

    graph, grown = capture(), 0
    for _ in range(30):
        wp.capture_launch(graph)
        if solver.check_status(observables, grow=True):
            grown += 1
            graph = capture()
            wp.capture_launch(graph)
            test.assertFalse(solver.check_status(observables))
        wp.copy(s0.particle_q, s1.particle_q)
        wp.copy(s0.particle_qd, s1.particle_qd)
    test.assertGreater(grown, 0)
    test.assertGreater(solver.get_max_soft_contact_count(), before)
    test.assertEqual(solver._contact_export.capacity, solver.get_max_soft_contact_count())
    # the cubes' rows on the ground; the cubes on each other give none (between particle features)
    rows = int(contacts.soft_contact_count.numpy()[0])
    test.assertGreater(rows, 0)
    test.assertTrue((contacts.soft_contact_shape.numpy()[:rows] == 0).all())


def test_contact_output_refusals(test, device):
    """CONTACT_F is not offered with AL-IPC; a model with fewer than two colliding shapes has no rigid rows and one
    without particles no soft rows."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    _box_body(builder, (0.0, 0.0, 0.06), (0.05, 0.05, 0.05))
    al = newton.solvers.SolverIPC(builder.finalize(device=device), contact_constitution="al-ipc")
    test.assertNotIn(_CONTACT_F, al.supported_observables)
    with test.assertRaisesRegex(ValueError, "does not support solver observable"):
        al.observables({_CONTACT_F}, contacts=newton.Contacts(4, 0, device=device))
    solver = newton.solvers.SolverIPC(_stacking(device, upper=False, lower_z=0.502))
    test.assertIn(_CONTACT_F, solver.supported_observables)
    test.assertGreater(solver.get_max_contact_count(), 0)
    test.assertEqual(solver.get_max_soft_contact_count(), 0)
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.add_particle(wp.vec3(0.0, 0.0, 0.02), wp.vec3(0.0, 0.0, 0.0), 0.01, radius=0.01)
    lone = newton.solvers.SolverIPC(builder.finalize(device=device))
    test.assertEqual(lone.get_max_contact_count(), 0)
    test.assertGreater(lone.get_max_soft_contact_count(), 0)


def test_contact_output_coupled_worlds(test, device):
    """With coupled_worlds the rows of every world share the buffer: in four copies of a world of a box on a global
    ground, SensorContact reports every world's box at its weight within 1%. The worlds are one solve, so a step that
    fails in one world (its box written through the ground) fails in every world and writes no row of any."""
    from newton.sensors import SensorContact  # noqa: PLC0415

    def content():
        world = newton.ModelBuilder()
        _box_body(world, (0.0, 0.0, 0.0505), (0.05, 0.05, 0.05))
        return world

    scene = newton.ModelBuilder()
    scene.add_ground_plane()
    scene.replicate(content(), 4)
    model = scene.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, coupled_worlds=True)
    sensor = SensorContact(model, request_contact_attributes=False, sensing_bodies="*")
    contacts = _native_contacts(solver, model)
    observables = solver.observables(sensor.solver_observable_kinds, contacts=contacts)
    state = _contact_steps(model, solver, contacts, observables, 60, 0.01)
    sensor.update(state, contacts, observables=observables)
    weight = model.body_mass.numpy() * 9.81
    np.testing.assert_allclose(sensor.total_force.numpy()[:, 2], weight, rtol=0.01)
    rows = int(contacts.rigid_contact_count.numpy()[0])
    worlds = model.shape_world.numpy()
    shapes = np.concatenate(
        [contacts.rigid_contact_shape0.numpy()[:rows], contacts.rigid_contact_shape1.numpy()[:rows]]
    )
    test.assertEqual(set(worlds[shapes].tolist()) - {-1}, {0, 1, 2, 3})
    q = state.body_q.numpy()
    q[0, 2] = 0.04  # world 0's box 1 cm through the ground
    state.body_q.assign(q)
    with test.assertRaises(RuntimeError):
        solver.step(state, model.state(), None, contacts, 0.01, observables=observables)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
    np.testing.assert_array_equal(observables.contact_f.numpy(), 0.0)


def test_contact_output_independent_worlds(test, device):
    """With independent worlds the rows of every world share the buffer: four worlds of a box resting on the global
    ground report every world's box at its weight within 1% through SensorContact, and every world's shapes have rows.
    A step in which world 0's box is written 1 cm through the ground fails in world 0 alone and writes no row of world
    0 (the other worlds' rows: ``test_contact_output_when_a_world_fails``)."""
    from newton.sensors import SensorContact  # noqa: PLC0415

    def content():
        world = newton.ModelBuilder()
        _box_body(world, (0.0, 0.0, 0.0505), (0.05, 0.05, 0.05))
        return world

    scene = newton.ModelBuilder()
    scene.add_ground_plane()
    scene.replicate(content(), 4)
    model = scene.finalize(device=device)
    solver = newton.solvers.SolverIPC(model)
    test.assertTrue(solver._independent)
    sensor = SensorContact(model, request_contact_attributes=False, sensing_bodies="*")
    contacts = _native_contacts(solver, model)
    kinds = sensor.solver_observable_kinds | {solver.ObservableKind.STATUS}
    observables = solver.observables(kinds, contacts=contacts)
    state = _contact_steps(model, solver, contacts, observables, 60, 0.01)
    sensor.update(state, contacts, observables=observables)
    weight = model.body_mass.numpy() * 9.81
    np.testing.assert_allclose(sensor.total_force.numpy()[:, 2], weight, rtol=0.01)
    worlds = model.shape_world.numpy()

    def row_worlds():
        rows = int(contacts.rigid_contact_count.numpy()[0])
        shapes = np.concatenate(
            [contacts.rigid_contact_shape0.numpy()[:rows], contacts.rigid_contact_shape1.numpy()[:rows]]
        )
        return set(worlds[shapes].tolist()) - {-1}

    test.assertEqual(row_worlds(), {0, 1, 2, 3})
    q = state.body_q.numpy()
    q[0, 2] = 0.04  # world 0's box 1 cm through the ground
    state.body_q.assign(q)
    with test.assertRaisesRegex(RuntimeError, r"failed in 1 of its 4 worlds, world 0 on"):
        solver.step(state, model.state(), None, contacts, 0.01, observables=observables)
    words = observables.status.numpy()
    test.assertEqual([bool(w & int(solver.StatusBits.FAILED)) for w in words], [True, False, False, False, False])
    test.assertNotIn(0, row_worlds())


def test_contact_output_when_a_world_fails(test, device):
    """Four independent worlds of a box resting on the global ground, two solvers in lockstep, one of them clean. A
    step in which world 0's box is written 1 cm through the ground fails in world 0 alone and writes the rows of worlds
    1 to 3 as the clean solver's same step writes them (libuipc-warp's records give the terms of a world that failed
    no force): the same shape pairs, their points within 1e-6 m and their forces and torques within 1e-4, each of those
    boxes reporting its weight within 1% through SensorContact, and no row of world 0, whose box reports no force."""
    from newton.sensors import SensorContact  # noqa: PLC0415

    def content():
        world = newton.ModelBuilder()
        _box_body(world, (0.0, 0.0, 0.0505), (0.05, 0.05, 0.05))
        return world

    scene = newton.ModelBuilder()
    scene.add_ground_plane()
    scene.replicate(content(), 4)
    model = scene.finalize(device=device)
    worlds = model.shape_world.numpy()
    runs = []
    for _ in range(2):
        solver = newton.solvers.SolverIPC(model)
        sensor = SensorContact(model, request_contact_attributes=False, sensing_bodies="*")
        contacts = _native_contacts(solver, model)
        observables = solver.observables(sensor.solver_observable_kinds, contacts=contacts)
        state = _contact_steps(model, solver, contacts, observables, 60, 0.01)
        runs.append((solver, sensor, contacts, observables, state))
    (clean, _, clean_contacts, clean_observables, c0), (faulty, sensor, contacts, observables, f0) = runs
    q = f0.body_q.numpy()
    q[0, 2] = 0.04  # world 0's box 1 cm through the ground
    f0.body_q.assign(q)
    c1, f1 = model.state(), model.state()
    clean.step(c0, c1, None, clean_contacts, 0.01, observables=clean_observables)
    with test.assertRaisesRegex(RuntimeError, r"failed in 1 of its 4 worlds, world 0 on"):
        faulty.step(f0, f1, None, contacts, 0.01, observables=observables)

    def rows(contacts, observables):
        """The rigid rows (shape 0, shape 1, point 0, contact_f) in one order, and the world of each."""
        n = min(int(contacts.rigid_contact_count.numpy()[0]), contacts.rigid_contact_max)
        shape0, shape1 = contacts.rigid_contact_shape0.numpy()[:n], contacts.rigid_contact_shape1.numpy()[:n]
        point0, force = contacts.rigid_contact_point0.numpy()[:n], observables.contact_f.numpy()[:n]
        order = np.lexsort((point0[:, 2], point0[:, 1], point0[:, 0], shape1, shape0))
        world = np.maximum(worlds[shape0], worlds[shape1])[order]
        return shape0[order], shape1[order], point0[order], force[order], world

    got, want = rows(contacts, observables), rows(clean_contacts, clean_observables)
    test.assertEqual(set(got[4].tolist()), {1, 2, 3})
    test.assertEqual(set(want[4].tolist()), {0, 1, 2, 3})
    kept = want[4] != 0
    np.testing.assert_array_equal(got[0], want[0][kept])
    np.testing.assert_array_equal(got[1], want[1][kept])
    np.testing.assert_allclose(got[2], want[2][kept], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(got[3], want[3][kept], rtol=0.0, atol=1e-4)
    sensor.update(f1, contacts, observables=observables)
    total = sensor.total_force.numpy()[:, 2]
    weight = model.body_mass.numpy() * 9.81
    test.assertEqual(float(total[0]), 0.0)
    np.testing.assert_allclose(total[1:], weight[1:], rtol=0.01)


@requires("uipc_warp")
class TestSolverIPCContactOutput(unittest.TestCase):
    """CONTACT_F: the solver's own contact terms written into the step's Contacts, one row per term with a force, and
    their forces at the end of the step."""


devices = get_cuda_test_devices()
for name, func in (
    ("test_contact_output_stacking_reports_weights", test_contact_output_stacking_reports_weights),
    ("test_contact_output_friction_at_rest", test_contact_output_friction_at_rest),
    ("test_contact_output_soft_rows", test_contact_output_soft_rows),
    ("test_contact_output_rigid_rows_balance_a_body", test_contact_output_rigid_rows_balance_a_body),
    ("test_contact_output_only_enabled_pairs", test_contact_output_only_enabled_pairs),
    ("test_contact_output_capacity", test_contact_output_capacity),
    ("test_contact_output_not_requested", test_contact_output_not_requested),
    ("test_contact_output_captured_equals_eager", test_contact_output_captured_equals_eager),
    ("test_contact_output_failed_step_has_no_rows", test_contact_output_failed_step_has_no_rows),
    ("test_contact_output_after_grow", test_contact_output_after_grow),
    ("test_contact_output_refusals", test_contact_output_refusals),
    ("test_contact_output_coupled_worlds", test_contact_output_coupled_worlds),
    ("test_contact_output_independent_worlds", test_contact_output_independent_worlds),
    ("test_contact_output_when_a_world_fails", test_contact_output_when_a_world_fails),
):
    add_function_test(TestSolverIPCContactOutput, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
