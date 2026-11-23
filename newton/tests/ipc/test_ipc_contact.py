# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC's contact: which pairs collide, with which friction and at which distance."""

import re
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.utils.heightfield import HeightfieldData, get_triangle_shape_from_heightfield
from newton.tests.ipc._fixtures import _TIGHT, D_HAT, _implicit_euler_oscillator, _run, _scene_calls, _soft_cube
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires


def _engine_z(solver, body):
    """The float64 height of a body's origin in the solver (its translation pseudo-vertex)."""
    return float(solver._engine.positions()[solver._scene.layout().body_base[body], 2])


def _free_fall(model, body, k, dt=0.01, g=9.81):
    """Implicit Euler's height of a body's origin after k steps of free fall from rest, from the model's (float32)
    pose. Under ``_TIGHT`` a step ends within 1e-9 m of its converged position, and the differences carry over to the
    later steps through the velocity: 30 steps stay within 1e-6 m, where a contact would move a body by millimetres."""
    return float(model.body_q.numpy()[body, 2]) - g * dt * dt * k * (k + 1) / 2.0


def test_friction_of_each_pair(test, device):
    """Two boxes of friction 0.1 and 0.3 and a particle (soft_contact_mu 0.2) slide over a ground of friction 0.5
    under gravity tilted to tan 0.8: each accelerates with g_t - mu g_n, mu the arithmetic mean of the pair's two
    coefficients (0.3, 0.4 and 0.35), and a box of friction 1.3 (pair 0.9) sticks."""
    g, tan_theta, dt = 9.81, 0.8, 0.01
    th = np.arctan(tan_theta)
    g_t, g_n = g * np.sin(th), g * np.cos(th)
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    boxes = []
    for k, mu in enumerate((0.1, 0.3, 1.3)):  # side by side across the slope
        b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.5 * k, 0.052), wp.quat_identity()))
        builder.add_shape_box(b, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(mu=mu))
        boxes.append(b)
    particle = builder.add_particle(wp.vec3(0.0, 2.0, 0.004), wp.vec3(0.0, 0.0, 0.0), 1.0e-3, radius=0.002)
    model = builder.finalize(device=device)
    model.soft_contact_mu = 0.2
    model.set_gravity((g_t, 0.0, -g_n))
    solver = newton.solvers.SolverIPC(model, d_hat=1.0e-3, **_TIGHT)
    s0, s1 = model.state(), model.state()
    v_box, v_particle = [], []
    for _ in range(80):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        v_box.append(s0.body_qd.numpy()[boxes, 0].astype(np.float64))
        v_particle.append(float(s0.particle_qd.numpy()[particle, 0]))
    # the steps after the bodies land, from their velocities (float32: about 1e-7 m/s)
    a_box = (v_box[79] - v_box[19]) / (60 * dt)
    a_particle = (v_particle[79] - v_particle[19]) / (60 * dt)
    np.testing.assert_allclose(a_box[:2], [g_t - 0.3 * g_n, g_t - 0.4 * g_n], rtol=1e-4)
    test.assertAlmostEqual(a_particle / (g_t - 0.35 * g_n), 1.0, delta=1e-5)
    test.assertLess(abs(v_box[79][2]), 0.01)  # the friction's smoothing velocity, eps_velocity


def test_rest_heights_follow_radius_and_margin(test, device):
    """Contact thickness: a free particle of radius r comes to rest between r and r + d_hat above the ground, and one
    of radius 0.02 on a kinematic particle of radius 0.01 between 0.03 and 0.03 + d_hat above it; a box of margin
    0.002 rests between the sum of its margin and the ground's (0.003) and that plus d_hat above the ground, and as
    far above a static box of margin 0.004 beside it; a cloth of particle radius 0.2 rests within d_hat of the
    ground, its thickness (the solver's, 0) being independent of the radius."""
    d_hat, dt = 1.0e-3, 0.01
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(margin=0.003))
    builder.add_shape_box(-1, xform=wp.transform(wp.vec3(1.0, 0.0, 0.05), wp.quat_identity()), hx=0.1, hy=0.1, hz=0.05,
                          cfg=newton.ModelBuilder.ShapeConfig(margin=0.004))  # fmt: skip
    on_ground = builder.add_particle(wp.vec3(0.0, 0.0, 0.015), wp.vec3(0.0, 0.0, 0.0), 1.0e-3, radius=0.01)
    below = builder.add_particle(wp.vec3(0.0, 0.5, 0.2), wp.vec3(0.0, 0.0, 0.0), 0.0, radius=0.01)  # kinematic
    above = builder.add_particle(wp.vec3(0.0, 0.5, 0.235), wp.vec3(0.0, 0.0, 0.0), 1.0e-3, radius=0.02)
    margin = newton.ModelBuilder.ShapeConfig(margin=0.002)
    boxes = []
    for x, z in ((0.5, 0.06), (1.0, 0.16)):  # 1 cm above the surfaces of the ground and the static box
        boxes.append(builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, z), wp.quat_identity())))
        builder.add_shape_box(boxes[-1], hx=0.05, hy=0.05, hz=0.05, cfg=margin)
    n = builder.particle_count
    builder.add_cloth_grid(
        pos=wp.vec3(-1.2, -0.1, 0.005),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=4,
        dim_y=4,
        cell_x=0.05,
        cell_y=0.05,
        mass=1.0e-3,
        tri_ke=1.0e3,
        tri_ka=1.0e3,
        tri_kd=0.0,
        particle_radius=0.2,
    )
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, d_hat=d_hat)
    s0, s1 = model.state(), model.state()
    for _ in range(100):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
    q = s0.particle_q.numpy().astype(np.float64)
    z = s0.body_q.numpy()[:, 2].astype(np.float64) - 0.05  # the boxes' bottoms
    for name, value, low in (
        ("particle on the ground", q[on_ground, 2], 0.01 + 0.003),
        ("particle on a particle", np.linalg.norm(q[above] - q[below]), 0.03),
        ("box on the ground", z[0], 0.002 + 0.003),
        ("box on a box", z[1] - 0.1, 0.002 + 0.004),
        ("cloth on the ground", q[n:, 2].min(), 0.003),
    ):
        with test.subTest(name=name):
            test.assertGreater(value, low)
            test.assertLess(value, low + d_hat)


def test_springs_oscillate_as_implicit_euler(test, device):
    """Without gravity, two free particles of mass m joined by a spring of stiffness k along x, started 1 cm short of
    its rest length, and a free particle joined to a kinematic one, 1 cm beyond it: the stretch of each follows the
    implicit-Euler oscillator of omega^2 = 2 k / m and k / m frame by frame, and the free pair's centre stays put."""
    m, k, L0, dt, steps = 0.01, 4.0, 0.1, 0.01, 60  # a period and a half of the pair: 2 pi / omega = 0.22 s
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for x in (0.0, L0 - 0.01):
        builder.add_particle(wp.vec3(x, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.0), m, radius=0.001)
    builder.add_particle(wp.vec3(0.0, 1.0, 1.0), wp.vec3(0.0, 0.0, 0.0), 0.0, radius=0.001)  # kinematic
    builder.add_particle(wp.vec3(L0 + 0.01, 1.0, 1.0), wp.vec3(0.0, 0.0, 0.0), m, radius=0.001)
    for spring, i in enumerate((0, 2)):
        builder.add_spring(i, i + 1, k, 0.0, 0.0)
        builder.spring_rest_length[spring] = L0  # add_spring took the distance at the start
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, **_TIGHT)
    rest = model.spring_rest_length.numpy().astype(np.float64)  # float32, as the positions
    x0 = solver._engine.positions()[:4, 0]
    stretch = {first: x0[first + 1] - x0[first] - rest[spring] for spring, first in enumerate((0, 2))}
    expected = {
        0: _implicit_euler_oscillator(stretch[0], np.sqrt(2.0 * k / m), dt, steps),
        2: _implicit_euler_oscillator(stretch[2], np.sqrt(k / m), dt, steps),
    }
    s0, s1 = model.state(), model.state()
    measured = {first: [] for first in expected}
    for _ in range(steps):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        x = solver._engine.positions()[:4, 0]  # float64
        for spring, first in enumerate(expected):
            measured[first].append(x[first + 1] - x[first] - rest[spring])
        test.assertAlmostEqual(x[0] + x[1], x0[0] + x0[1], delta=1e-12)
    # to the Newton tolerance of _TIGHT, 1e-9 m per step
    for first, u in expected.items():
        np.testing.assert_allclose(measured[first], u, rtol=0.0, atol=1e-9)


def _box_on_ramp(builder, center, angle, mu, half=0.05, gap=0.002):
    """A box of friction ``mu`` resting ``gap`` above the top of a ramp tilted by ``angle`` about x whose top centre is
    ``center``; returns the body."""
    q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(angle))
    up = np.array(wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0)))
    b = builder.add_body(xform=wp.transform(wp.vec3(*(np.asarray(center) + (half + gap) * up)), q))
    builder.add_shape_box(b, hx=half, hy=half, hz=half, cfg=newton.ModelBuilder.ShapeConfig(mu=mu))
    return b


def test_ramp_sticks_and_slips(test, device):
    """Boxes on a static ramp of 20 degrees (friction 0.3): one of friction 0.2 (pair 0.25, below tan 20 = 0.364)
    slides down with g (sin - 0.25 cos), one of friction 0.6 (pair 0.45) sticks.

    The sliding box moves away from the diagonal that splits the ramp's top face (along x = y in the ramp's frame).
    A box that crosses it meets edge-edge pairs besides its point-triangle pairs there, which change its acceleration
    over a few steps (by 2% to 4% over ten steps at this ``d_hat``).
    """
    g, angle, dt = 9.81, np.radians(20.0), 0.01
    q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(angle))
    up = np.array(wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0)))
    down = -np.array(wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0)))  # down the slope
    builder = newton.ModelBuilder()
    ramp_center = np.array([0.0, 0.0, 0.5])
    builder.add_shape_box(
        -1,
        xform=wp.transform(wp.vec3(*ramp_center), q),
        hx=0.5,
        hy=0.5,
        hz=0.05,
        cfg=newton.ModelBuilder.ShapeConfig(mu=0.3),
    )
    top = ramp_center + 0.05 * up
    slides = _box_on_ramp(builder, top + np.array([0.2, 0.0, 0.0]), angle, 0.2)
    sticks = _box_on_ramp(builder, top + np.array([-0.2, 0.0, 0.0]), angle, 0.6)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, d_hat=1.0e-3, **_TIGHT)
    s0, s1 = model.state(), model.state()
    v = []
    for _ in range(60):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        v.append(s0.body_qd.numpy()[[slides, sticks], :3].astype(np.float64) @ down)
    a = (v[59][0] - v[19][0]) / (40 * dt)
    test.assertAlmostEqual(a / (g * (np.sin(angle) - 0.25 * np.cos(angle))), 1.0, delta=2e-4)
    test.assertLess(abs(v[59][1]), 0.01)


def test_excluded_pairs_never_collide(test, device):
    """A box over a static table falls through it as in free fall, without a contact candidate in any step, where
    Newton's rules exclude the pair: collision groups 2 and 3, one negative group (-1 and -1), or a filter pair. In
    groups 2 and 2, or -1 and -2, the box comes to rest on the table."""
    dt, half, table_top = 0.01, 0.05, 0.1
    z0 = table_top + half + 0.002
    for box_group, table_group, filtered, falls in (
        (2, 3, False, True),
        (-1, -1, False, True),
        (1, 1, True, True),
        (2, 2, False, False),
        (-1, -2, False, False),
    ):
        with test.subTest(box_group=box_group, table_group=table_group, filtered=filtered):
            builder = newton.ModelBuilder()
            table = builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.5 * table_top), wp.quat_identity()),
                hx=0.3,
                hy=0.3,
                hz=0.5 * table_top,
                cfg=newton.ModelBuilder.ShapeConfig(collision_group=table_group),
            )
            body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, z0), wp.quat_identity()))
            box = builder.add_shape_box(
                body, hx=half, hy=half, hz=half, cfg=newton.ModelBuilder.ShapeConfig(collision_group=box_group)
            )
            if filtered:
                builder.add_shape_collision_filter_pair(table, box)
            model = builder.finalize(device=device)
            test.assertEqual(model.shape_contact_pair_count, 0 if falls else 1)
            solver = newton.solvers.SolverIPC(model, d_hat=1.0e-3, **_TIGHT)
            s0, s1 = model.state(), model.state()
            candidates = 0
            for k in range(1, 31):
                solver.step(s0, s1, None, None, dt)
                s0, s1 = s1, s0
                candidates += solver.frame_stats()["contact_candidates"]
                if falls:
                    test.assertAlmostEqual(_engine_z(solver, body), _free_fall(model, body, k), delta=1e-6)
            if falls:
                test.assertEqual(candidates, 0)
            else:
                test.assertGreater(candidates, 0)
                z = float(s0.body_q.numpy()[body, 2])
                test.assertGreater(z, table_top + half)
                test.assertLess(z, table_top + half + 1.0e-3)


def test_particle_only_shapes(test, device):
    """A shape that collides with particles only (``has_shape_collision=False``, which keeps COLLIDE_PARTICLES):
    a static box of that kind holds a particle on its top and lets a box fall through it as in free fall. Two such
    boxes on bodies, which start inside each other, fall freely through each other and through the ground, whether
    the model has particles or not (without particles they are not even triangulated)."""
    dt = 0.01
    particle_only = newton.ModelBuilder.ShapeConfig(has_shape_collision=False)
    builder = newton.ModelBuilder()
    builder.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()), hx=0.5, hy=0.5, hz=0.05,
                          cfg=particle_only)  # fmt: skip
    particle = builder.add_particle(wp.vec3(-0.25, 0.0, 0.104), wp.vec3(0.0, 0.0, 0.0), 1.0e-3, radius=0.002)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.25, 0.0, 0.152), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, d_hat=1.0e-3, **_TIGHT)
    s0, s1 = model.state(), model.state()
    for k in range(1, 31):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        test.assertAlmostEqual(_engine_z(solver, body), _free_fall(model, body, k), delta=1e-6)
    z = float(s0.particle_q.numpy()[particle, 2])
    test.assertGreater(z, 0.1)
    test.assertLess(z, 0.1 + 1.0e-3 + 0.002)

    for with_particle in (False, True):
        with test.subTest(with_particle=with_particle):
            builder = newton.ModelBuilder()
            bodies = []
            for _ in range(2):
                bodies.append(builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.15), wp.quat_identity())))
                builder.add_shape_box(bodies[-1], hx=0.1, hy=0.1, hz=0.1, cfg=particle_only)
            builder.add_ground_plane()
            if with_particle:
                builder.add_particle(wp.vec3(1.0, 0.0, 2.0), wp.vec3(0.0, 0.0, 0.0), 1.0, radius=0.01)
            model = builder.finalize(device=device)
            solver = newton.solvers.SolverIPC(model, **_TIGHT)
            s0, s1 = model.state(), model.state()
            for k in range(1, 21):
                solver.step(s0, s1, None, None, dt)
                s0, s1 = s1, s0
                for b in bodies:  # 5.6 cm below the ground at the end
                    test.assertAlmostEqual(_engine_z(solver, b), _free_fall(model, b, k), delta=1e-6)


def test_inactive_particles_do_not_collide(test, device):
    """A box falls onto a particle: through it as in free fall when the particle is not ACTIVE (Newton gives an
    inactive particle no contact), onto it when the particle is active and kinematic (zero mass)."""
    dt, z0 = 0.01, 0.054
    for active in (False, True):
        with test.subTest(active=active):
            builder = newton.ModelBuilder()
            flags = int(newton.ParticleFlags.ACTIVE) if active else 0
            builder.add_particle(wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0), 0.0, radius=0.001, flags=flags)
            body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, z0), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
            model = builder.finalize(device=device)
            solver = newton.solvers.SolverIPC(model, d_hat=1.0e-3, **_TIGHT)
            s0, s1 = model.state(), model.state()
            for k in range(1, 21):
                solver.step(s0, s1, None, None, dt)
                s0, s1 = s1, s0
                if not active:
                    test.assertAlmostEqual(_engine_z(solver, body), _free_fall(model, body, k), delta=1e-6)
            if active:
                test.assertGreater(float(s0.body_q.numpy()[body, 2]), 0.05)  # held above the particle


def test_particle_contact_follows_the_particle_grid(test, device):
    """Two free particles of radius 2 cm closing at 2 m/s without gravity stop apart, as Newton's particle solvers
    collide particles while the model has its particle grid; with Model.particle_grid set to None, Newton's switch of
    particle contact, they pass through each other, given so at construction or after notify_model_changed with
    MODEL_PROPERTIES."""

    def two_particles():
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_particle(wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, 1.0), mass=1.0, radius=0.02)
        builder.add_particle(wp.vec3(0.0, 0.0, 1.1), wp.vec3(0.0, 0.0, -1.0), mass=1.0, radius=0.02)
        return builder.finalize(device=device)

    def heights(model, solver):
        return _run(model, solver, 20)[-1, :, 2]  # after 0.2 s

    model = two_particles()
    test.assertIsNotNone(model.particle_grid)
    z = heights(model, newton.solvers.SolverIPC(model, d_hat=0.005))
    test.assertGreater(z[1] - z[0], 0.04)  # held apart by their radii
    model.particle_grid = None
    np.testing.assert_allclose(heights(model, newton.solvers.SolverIPC(model, d_hat=0.005)), [1.2, 0.9], atol=1e-6)
    model = two_particles()
    solver = newton.solvers.SolverIPC(model, d_hat=0.005)
    model.particle_grid = None
    solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
    np.testing.assert_allclose(heights(model, solver), [1.2, 0.9], atol=1e-6)


def test_warns_of_unread_contact_coefficients(test, device):
    """The solver warns once, when it is built, of the contact coefficients it does not read that a colliding shape or
    the particles set to other than Newton's defaults: restitution, a static friction other than the dynamic one, and
    torsional and rolling friction, each named with its first shape. Newton's defaults, which MJCF's default friction
    "1 0.005 0.0001" holds, draw no warning."""

    def model(cfg=None, mjcf=False, particle_restitution=0.0):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        if mjcf:
            builder.add_mjcf(
                '<mujoco><worldbody><body pos="0 0 1"><freejoint/><geom type="box" size="0.1 0.1 0.1"/></body>'
                "</worldbody></mujoco>"
            )
        else:
            body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
        builder.add_particle(wp.vec3(1.0, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.0), 1.0, radius=0.01)
        model = builder.finalize(device=device)
        model.soft_contact_restitution = particle_restitution
        return model

    for kwargs in ({}, {"mjcf": True}):
        m = model(**kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            newton.solvers.SolverIPC(m)
    Shape = newton.ModelBuilder.ShapeConfig
    for kwargs, named in (
        ({"cfg": Shape(restitution=0.5)}, "shape_material_restitution 0.5 of shape 1"),
        ({"cfg": Shape(mu=0.5, mu_static=0.8)}, "shape_material_mu_static 0.8 of shape 1"),
        ({"cfg": Shape(mu_torsional=0.01, mu_rolling=0.0)}, "shape_material_mu_torsional 0.01 of shape 1, "
         "shape_material_mu_rolling 0 of shape 1"),
        ({"particle_restitution": 0.3}, "soft_contact_restitution 0.3"),
    ):  # fmt: skip
        with test.subTest(named):
            m = model(**kwargs)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                newton.solvers.SolverIPC(m)
            messages = [str(w.message) for w in caught if issubclass(w.category, newton.NewtonWarning)]
            test.assertEqual(len(messages), 1, messages)
            test.assertIn(f"SolverIPC does not read {named}: its contact has no restitution", messages[0])


@wp.kernel
def _newton_heightfield_triangles(
    fields: wp.array[HeightfieldData],
    elevations: wp.array[wp.float32],
    xform: wp.transform,
    out: wp.array2d[wp.vec3],
):
    """The world vertices of triangle ``tid`` of a height field, as Newton's narrow phase takes them."""
    t = wp.tid()
    shape, v0 = get_triangle_shape_from_heightfield(fields[0], elevations, xform, t)
    q = wp.transform_get_rotation(xform)
    out[t, 0] = v0
    out[t, 1] = v0 + wp.quat_rotate(q, shape.scale)
    out[t, 2] = v0 + wp.quat_rotate(q, shape.auxiliary)


def test_heightfield_triangles_and_rest(test, device):
    """A height field is the triangle mesh Newton's narrow phase collides with: its triangles in the solver are those
    of get_triangle_shape_from_heightfield, in the field's scaled and moved frame. Spheres dropped on its two
    plateaus (heights 0 and 0.2 of a scaled field, moved up by 0.1) rest there, within d_hat of touching."""
    d_hat, radius = 1.0e-3, 0.08
    data = np.zeros((9, 9), np.float32)
    data[:, 4] = 0.5
    data[:, 5:] = 1.0
    field = newton.Heightfield(data=data, nrow=9, ncol=9, hx=0.5, hy=0.5, min_z=0.0, max_z=0.1)
    xform = wp.transform(wp.vec3(0.3, 0.0, 0.1), wp.quat_identity())
    builder = newton.ModelBuilder()
    builder.add_shape_heightfield(xform=xform, heightfield=field, scale=(2.0, 2.0, 2.0))
    # the plateaus: x below -0.25 and above 0.25 in the field's frame, at 0 and 0.2 there
    spots = [(-0.6, -0.5, 0.0), (-0.6, 0.5, 0.0), (0.6, 0.0, 0.2)]
    bodies = []
    for x, y, top in spots:
        bodies.append(
            builder.add_body(xform=wp.transform(wp.vec3(0.3 + x, y, 0.1 + top + radius + 0.005), wp.quat_identity()))
        )
        builder.add_shape_sphere(bodies[-1], radius=radius)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, d_hat=d_hat)

    expected = wp.empty((2 * 8 * 8, 3), dtype=wp.vec3, device=device)
    wp.launch(
        _newton_heightfield_triangles,
        len(expected),
        inputs=[model.heightfield_data, model.heightfield_elevations, xform],
        outputs=[expected],
        device=device,
    )
    layout = solver._scene.layout()
    first, end = layout.obstacle_offsets[0], layout.obstacle_offsets[1]
    tris = np.asarray(solver._engine.surface["triangles"])
    tris = tris[(tris >= first).all(axis=1) & (tris < end).all(axis=1)]
    x = solver._engine.positions()

    def rows(t):  # the triangles' vertices in order, sorted by triangle
        t = np.asarray(t, np.float64).reshape(-1, 9)
        return t[np.lexsort(t.T[::-1])]

    np.testing.assert_allclose(rows(x[tris]), rows(expected.numpy()), rtol=0.0, atol=1e-6)

    s0, s1 = model.state(), model.state()
    for _ in range(60):
        solver.step(s0, s1, None, None, 0.01)
        s0, s1 = s1, s0
    q = s0.body_q.numpy()
    for b, (x, y, top) in zip(bodies, spots, strict=True):
        np.testing.assert_allclose(q[b, :2], [0.3 + x, y], atol=1e-3)
        test.assertGreater(q[b, 2], 0.1 + top + radius)
        test.assertLess(q[b, 2], 0.1 + top + radius + d_hat)


def _sheet(builder, x, tris, rows=None, rest=None, mass=0.01, tri_k=1.0e3, edge_ke=30.0):
    """Particles at ``x`` with the triangles ``tris`` and, given, the bending edge ``rows`` (o0, o1, v1, v2) with their
    rest angles (None: Newton measures them at ``x``), without damping."""
    n = len(x)
    builder.add_particles([wp.vec3(*p) for p in x], [wp.vec3(0.0, 0.0, 0.0)] * n, [mass] * n, radius=[0.001] * n)
    t = np.asarray(tris)
    builder.add_triangles(
        t[:, 0], t[:, 1], t[:, 2], tri_ke=[tri_k] * len(t), tri_ka=[tri_k] * len(t), tri_kd=[0.0] * len(t)
    )
    if rows is not None:
        r = np.asarray(rows)
        builder.add_edges(
            r[:, 0], r[:, 1], r[:, 2], r[:, 3], rest=rest, edge_ke=[edge_ke] * len(r), edge_kd=[0.0] * len(r)
        )


def _newton_angles(model, x):
    """Newton's angle of every bending edge row at the positions ``x``, as it measures rest angles (SolverIPC's copy of
    Newton's helper, which test_rest_shape_copies_match_newtons_builders holds to the builders)."""
    from newton._src.solvers.ipc.scene import _dihedral_angles  # noqa: PLC0415

    e = model.edge_indices.numpy().reshape(-1, 4)
    x = np.asarray(x, np.float64)
    return _dihedral_angles(x[e[:, 0]], x[e[:, 1]], x[e[:, 2]], x[e[:, 3]], x[e[:, 2]], x[e[:, 3]])


def test_tet_lame_parameters_reach_the_engine_as_libuipc_stores_them(test, device):
    """k_mu and k_lambda are Lame parameters. libuipc's StableNeoHookean.apply_to stores mu and lambda + mu, the SNK1
    parameters that linearize to Hooke's law with those Lame parameters; the engine gets the same pair, so a block
    with k_mu = k_lambda has Poisson's ratio 0.25, not 0."""
    import uipc_warp.arrays  # noqa: PLC0415

    builder = newton.ModelBuilder()
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, 0.5),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=2,
        dim_y=2,
        dim_z=2,
        cell_x=0.1,
        cell_y=0.1,
        cell_z=0.1,
        density=1000.0,
        k_mu=2.0e5,
        k_lambda=5.0e5,
        k_damp=0.0,
    )
    with _scene_calls(uipc_warp.arrays.ArrayScene.add_tetrahedra) as calls:
        newton.solvers.SolverIPC(builder.finalize(device=device))
    [tetrahedra] = calls
    np.testing.assert_allclose(tetrahedra["mu"], 2.0e5)
    np.testing.assert_allclose(tetrahedra["lam"], 7.0e5)


def _shell_inputs(model, **options):
    """The cloth SolverIPC gives the engine for ``model``: the arguments of its one call of libuipc-warp's
    ``ArrayScene.add_shell``."""
    import uipc_warp.arrays  # noqa: PLC0415

    with _scene_calls(uipc_warp.arrays.ArrayScene.add_shell) as calls:
        newton.solvers.SolverIPC(model, **options)
    [shell] = calls
    return shell


def test_rest_shapes_from_tri_poses_and_edge_rest_angle(test, device):
    """Newton's rest shapes, without gravity. A flat hinge whose edge_rest_angle is 0.6 folds until Newton's angle of
    its row is 0.6, and the same with the row given in another order (o0's triangle running v2 -> v1) and the rest
    angle that order measures (-0.6). A triangle whose tri_poses are scaled by 1 / 1.2 grows until its edges are 1.2
    times as long. A curved sheet gets the same rest angles from its builder's rows as from the rows given by hand in
    the other order. A hinge whose triangles run their edge the same way is refused."""
    dt, flat = 0.01, np.array([[0.5, -1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
    tris = [[0, 1, 2], [3, 2, 1]]  # (0, 1, 2) runs the edge 1 -> 2, (3, 2, 1) runs it 2 -> 1
    folded = {}
    for row, rest in (((0, 3, 1, 2), 0.6), ((3, 0, 1, 2), -0.6), ((0, 3, 2, 1), -0.6)):
        with test.subTest(row=row):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            _sheet(builder, flat, tris, [row], [rest])
            model = builder.finalize(device=device)
            solver = newton.solvers.SolverIPC(model, **_TIGHT)
            s0, s1 = model.state(), model.state()
            for _ in range(200):
                solver.step(s0, s1, None, None, dt)
                s0, s1 = s1, s0
            test.assertAlmostEqual(float(_newton_angles(model, s0.particle_q.numpy())[0]), rest, delta=1e-4)
            folded[row] = s0.particle_q.numpy()
    for x in folded.values():
        np.testing.assert_allclose(x, folded[(0, 3, 1, 2)], atol=1e-5)

    x = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.03, 0.08, 0.0]])
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    _sheet(builder, x, [[0, 1, 2]])
    model = builder.finalize(device=device)
    model.tri_poses.assign(model.tri_poses.numpy() / 1.2)
    solver = newton.solvers.SolverIPC(model, cloth_strain_rate=0.0, **_TIGHT)
    s0, s1 = model.state(), model.state()
    for _ in range(200):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
    y = s0.particle_q.numpy().astype(np.float64)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        test.assertAlmostEqual(np.linalg.norm(y[b] - y[a]) / np.linalg.norm(x[b] - x[a]), 1.2, delta=1.2e-5)

    # a 3 x 3 sheet bent around a cylinder; the same rows by hand with their opposite vertices swapped
    u, v = np.meshgrid(np.linspace(0.0, 0.3, 4), np.linspace(0.0, 0.3, 4), indexing="ij")
    curved = np.stack([0.5 * np.sin(u / 0.5), v, 0.5 * (1.0 - np.cos(u / 0.5))], axis=-1).reshape(-1, 3)
    quads = [(4 * i + j, 4 * i + j + 4, 4 * i + j + 5, 4 * i + j + 1) for i in range(3) for j in range(3)]
    sheet = [t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))]
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=[wp.vec3(*p) for p in curved],
        indices=[i for t in sheet for i in t],
        density=0.1,
        tri_ke=1.0e3,
        tri_ka=1.0e3,
        tri_kd=0.0,
        edge_ke=1.0,
        edge_kd=0.0,
    )
    given = builder.finalize(device=device)
    rows = given.edge_indices.numpy().reshape(-1, 4)
    rows = rows[(rows[:, 0] >= 0) & (rows[:, 1] >= 0)][:, [1, 0, 2, 3]]
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    _sheet(builder, curved, sheet, rows, edge_ke=1.0)
    by_hand = builder.finalize(device=device)
    from newton._src.solvers.ipc.scene import _winding_signs  # noqa: PLC0415

    hinges = []
    for model in (given, by_hand):
        # the rest angles the solver gives the engine, hinge by hinge (by the edge's two particles); where it gives
        # none (a cloth whose edge_rest_angle is the start's measure, by_hand's), the engine takes the rest positions'
        # angles, which libuipc-warp's tests hold to libuipc's: here Newton's rest angles of those rows by the
        # triangles' winding, as the solver gives them otherwise
        shell = _shell_inputs(model)
        edges = np.asarray(shell["bending_edges"])
        angles = shell["rest_angles"]
        if angles is None:
            e = model.edge_indices.numpy().reshape(-1, 4)
            hinge = (e[:, 0] >= 0) & (e[:, 1] >= 0)
            e, tris = e[hinge], model.tri_indices.numpy().reshape(-1, 3)
            np.testing.assert_array_equal(edges, e[:, 2:4])
            angles = _winding_signs(tris, e, np.arange(len(e))) * model.edge_rest_angle.numpy()[hinge]
        edges = np.sort(edges, axis=1)
        order = np.lexsort(edges.T[::-1])
        hinges.append((edges[order], np.asarray(angles)[order]))
    (edges, angles), (edges_by_hand, angles_by_hand) = hinges
    np.testing.assert_array_equal(edges_by_hand, edges)
    test.assertEqual(len(edges), len(rows))
    test.assertGreater(np.abs(angles).max(), 0.1)  # the edges along the cylinder's axis are folded (0.2 rad)
    np.testing.assert_allclose(angles_by_hand, angles, rtol=0.0, atol=1e-6)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    _sheet(builder, flat, [[0, 1, 2], [3, 1, 2]], [(0, 3, 1, 2)], [0.0])
    with test.assertRaisesRegex(ValueError, r"bending edge 0 \(particles 1 and 2.*the same direction"):
        newton.solvers.SolverIPC(builder.finalize(device=device))


def _flat_sheet(device, rot=None):
    """A 6 x 6 cloth grid of 3 cm cells built at rest at 0.9 m: flat, and Newton measures its rest angles as 0."""
    builder = newton.ModelBuilder()
    builder.add_cloth_grid(
        pos=wp.vec3(-0.1, -0.1, 0.9),
        rot=wp.quat_identity() if rot is None else rot,
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=6,
        dim_y=6,
        cell_x=0.03,
        cell_y=0.03,
        mass=0.01,
        tri_ke=1.0e3,
        tri_ka=1.0e3,
        tri_kd=0.0,
        edge_ke=5.0e-3,
        edge_kd=0.0,
    )
    return builder.finalize(device=device)


def test_cloth_at_rest_starts_without_bending_force(test, device):
    """A cloth built at rest starts at its rest angles as the engine measures them, so that no hinge pushes it out of
    its plane. Newton measures the angle of every hinge of this flat sheet as 0; libuipc's dihedral angle, the
    arccosine of the normals' cosine, reads 1.5e-8 rad on some of them, and it is the angle the engine's bending
    energy takes. The solver gives the engine no rest angles, so that the engine takes its own of the start positions
    (as libuipc takes them from its rest geometry; libuipc-warp's tests/test_arrays_rest_shapes.py holds them to
    libuipc's bit for bit, with no bending force at the start). A rest angle other than the start's is Newton's
    (test_rest_shapes_from_tri_poses_and_edge_rest_angle)."""
    model = _flat_sheet(device)
    test.assertFalse(np.any(model.edge_rest_angle.numpy()))
    shell = _shell_inputs(model)
    test.assertIsNone(shell["rest_angles"])
    test.assertGreater(len(shell["bending_edges"]), 0)


def test_cloth_at_rest_takes_its_rest_triangles_from_particle_q(test, device):
    """Newton's builders compute a cloth triangle's rest shape in float64 from particle_q and keep its inverse in float32
    (tri_poses). For a cloth built at rest the solver gives the engine no rest edges, so that the engine takes the rest
    shapes of particle_q, as libuipc computes them from its rest geometry (libuipc-warp's
    tests/test_arrays_rest_shapes.py holds them to libuipc's, bit for bit), not those of the float32 tri_poses, which
    are off by their rounding, a strain the sheet would start with. A triangle whose tri_poses were changed keeps them;
    the others then take the rest edges of particle_q in float64."""
    from newton._src.solvers.ipc.scene import _triangle_rest_geometry  # noqa: PLC0415

    model = _flat_sheet(device, rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.3))
    x = model.particle_q.numpy().astype(np.float64)
    tris = model.tri_indices.numpy().reshape(-1, 3)
    test.assertIsNone(_shell_inputs(model)["rest_edges"])
    edges, _ = _triangle_rest_geometry(x[tris[:, 1]] - x[tris[:, 0]], x[tris[:, 2]] - x[tris[:, 0]])
    poses = model.tri_poses.numpy()
    rounded = np.linalg.inv(poses.astype(np.float64))
    entries = edges != 0.0
    test.assertGreater(np.abs(rounded[entries] / edges[entries] - 1.0).max(), 1e-8)  # the float32 tri_poses are not it

    poses[0] /= 1.2
    model.tri_poses.assign(poses)
    rest = _shell_inputs(model)["rest_edges"]
    np.testing.assert_array_equal(rest[0], np.linalg.inv(poses[0].astype(np.float64)))
    np.testing.assert_array_equal(rest[1:], edges[1:])


def _libuipc_corridor(model, dt, d_hat):
    """libuipc's contact stiffness range of a model of particles over a plane through the origin, from the particles'
    positions and masses: libuipc-warp's stiffness policy (libuipc's, which its own tests hold to libuipc) of a scene
    laid out here, apart from the solver's, of the particles alone over that plane, in which every particle and the
    plane's point are libuipc's vertices and the plane has no mass."""
    import uipc_warp  # noqa: PLC0415
    import uipc_warp.arrays  # noqa: PLC0415

    scene = uipc_warp.arrays.ArrayScene(kappa_corridor=True)
    scene.add_vertices(model.particle_q.numpy(), mass=model.particle_mass.numpy())
    scene.add_half_plane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    engine = scene.build(uipc_warp.SimConfig(dt=dt, d_hat=d_hat, kappa=None), device=model.device)
    _, _, lo, hi, _ = engine.contact_stiffness.resolve(dt)
    return lo, hi


def test_kappa_is_taken_as_given(test, device):
    """By default every pair takes kappa as given, at every step size, also outside libuipc's range, without a
    warning; kappa=None, libuipc's default, needs adaptive_kappa."""
    model = _soft_cube(device, height=0.05)
    lo = _libuipc_corridor(model, 0.01, D_HAT)[0]
    for kappa in (1.0e9, 0.1 * lo):
        solver = newton.solvers.SolverIPC(model, d_hat=D_HAT, kappa=kappa)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _run(model, solver, 3)
        for dt in (0.01, 1.0 / 600.0):
            test.assertEqual(solver.contact_stiffness(dt), kappa)
        tabular = solver._scene.engine_inputs(solver._config, "cpu")["contact_tabular"]  # what the engine is given
        np.testing.assert_array_equal(solver._config.kappa if tabular is None else tabular["kappa"], kappa)
    with test.assertRaisesRegex(ValueError, "kappa=None, libuipc's default stiffness, needs adaptive_kappa=True"):
        newton.solvers.SolverIPC(model, kappa=None)


def test_adaptive_kappa_is_libuipcs(test, device):
    """adaptive_kappa=True: kappa=None takes the lower bound of libuipc's range for the model and the step, which
    scales with 1/dt^2; a kappa inside the range is used as given and one outside it is clamped, with one
    NewtonWarning that names both values, at the first step that clamps it."""
    model = _soft_cube(device, height=0.05)
    adaptive = {"d_hat": D_HAT, "adaptive_kappa": True}
    for dt in (0.01, 1.0 / 600.0):
        lo, hi = _libuipc_corridor(model, dt, D_HAT)
        default = newton.solvers.SolverIPC(model, kappa=None, **adaptive).contact_stiffness(dt)
        test.assertAlmostEqual(default / lo, 1.0, places=12)
    lo, hi = _libuipc_corridor(model, 0.01, D_HAT)
    inside = newton.solvers.SolverIPC(model, kappa=3.0 * lo, **adaptive)
    test.assertEqual(inside.contact_stiffness(0.01), 3.0 * lo)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _run(model, inside, 2)
    above = newton.solvers.SolverIPC(model, kappa=1.0e3 * hi, **adaptive)
    test.assertAlmostEqual(above.contact_stiffness(0.01) / hi, 1.0, places=12)
    s0, s1 = model.state(), model.state()
    given, used = re.escape(f"{1.0e3 * hi:g}"), re.escape(f"{hi:.6g}")
    with test.assertWarnsRegex(newton.NewtonWarning, rf"kappa={given} is outside .* the steps use {used} Pa"):
        above.step(s0, s1, None, None, 0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        above.step(s1, s0, None, None, 0.01)
        above.step(s0, s1, None, None, 0.005)  # clamped again, at another value: not repeated
    for kappa in (0.0, -1.0e9, float("nan"), float("inf")):
        with test.assertRaisesRegex(ValueError, "kappa must be a finite positive stiffness"):
            newton.solvers.SolverIPC(model, kappa=kappa, **adaptive)


def test_adaptive_kappa_follows_dt_in_captured_steps(test, device):
    """adaptive_kappa=True: every pair takes the stiffness of its step's dt: steps captured at two step sizes, replayed
    in turn and between eager steps of a third, each run with their own, as eager steps of those sizes do (to rounding:
    the solver adds with float atomics). That a graph captured at a dt writes the engine's table of that dt is
    libuipc-warp's (its tests/test_arrays_contact_stiffness.py, test_graphs_of_different_dt)."""
    tight = {"newton_velocity_tol": 1e-7, "linear_tol_rate": 1e-12, "semi_implicit": False}
    model = _soft_cube(device, height=0.012)
    captured, eager = (
        newton.solvers.SolverIPC(model, d_hat=D_HAT, kappa=None, adaptive_kappa=True, **tight) for _ in range(2)
    )
    states = {solver: (model.state(), model.state()) for solver in (captured, eager)}
    graphs = {}
    for dt in (0.01, 0.005):
        s0, s1 = model.state(), model.state()
        with wp.ScopedCapture(device=device) as cap:
            captured.step(s0, s1, None, None, dt)
        graphs[dt] = (cap.graph, s0, s1)
    for k, dt in enumerate((0.01, 0.005, 0.02, 0.01, 0.02, 0.005, 0.01)):
        for solver in (captured, eager):
            a, b = states[solver]
            if solver is captured and dt in graphs:
                graph, g0, g1 = graphs[dt]
                g0.assign(a)
                wp.capture_launch(graph)
                b.assign(g1)
            else:
                solver.step(a, b, None, None, dt)
            states[solver] = (b, a)
        np.testing.assert_allclose(
            states[captured][0].particle_q.numpy(),
            states[eager][0].particle_q.numpy(),
            rtol=0.0,
            atol=1e-6,
            err_msg=f"step {k}, dt {dt}",
        )
    test.assertLess(states[eager][0].particle_q.numpy()[:, 2].min(), D_HAT)  # the cube reached the ground


@requires("uipc_warp")
class TestSolverIPCContact(unittest.TestCase):
    """Which pairs collide, with which friction and at which distance."""


devices = get_cuda_test_devices()
for name, func in (
    ("test_friction_of_each_pair", test_friction_of_each_pair),
    ("test_ramp_sticks_and_slips", test_ramp_sticks_and_slips),
    ("test_excluded_pairs_never_collide", test_excluded_pairs_never_collide),
    ("test_particle_only_shapes", test_particle_only_shapes),
    ("test_inactive_particles_do_not_collide", test_inactive_particles_do_not_collide),
    ("test_warns_of_unread_contact_coefficients", test_warns_of_unread_contact_coefficients),
    ("test_particle_contact_follows_the_particle_grid", test_particle_contact_follows_the_particle_grid),
    ("test_rest_heights_follow_radius_and_margin", test_rest_heights_follow_radius_and_margin),
    ("test_springs_oscillate_as_implicit_euler", test_springs_oscillate_as_implicit_euler),
    ("test_heightfield_triangles_and_rest", test_heightfield_triangles_and_rest),
    ("test_kappa_is_taken_as_given", test_kappa_is_taken_as_given),
    ("test_adaptive_kappa_is_libuipcs", test_adaptive_kappa_is_libuipcs),
    ("test_adaptive_kappa_follows_dt_in_captured_steps", test_adaptive_kappa_follows_dt_in_captured_steps),
    ("test_rest_shapes_from_tri_poses_and_edge_rest_angle", test_rest_shapes_from_tri_poses_and_edge_rest_angle),
    ("test_cloth_at_rest_starts_without_bending_force", test_cloth_at_rest_starts_without_bending_force),
    (
        "test_cloth_at_rest_takes_its_rest_triangles_from_particle_q",
        test_cloth_at_rest_takes_its_rest_triangles_from_particle_q,
    ),
    (
        "test_tet_lame_parameters_reach_the_engine_as_libuipc_stores_them",
        test_tet_lame_parameters_reach_the_engine_as_libuipc_stores_them,
    ),
):
    add_function_test(TestSolverIPCContact, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
