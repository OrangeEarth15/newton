# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC's damping of springs, particle-body attachments, tetrahedra, triangles and bending edges, as Newton's VBD
solver defines each coefficient (libuipc-warp's ``uipc_warp.ext.damping``), and its drag and lift, as SemiImplicit
computes them (``features/damping.py``).

Checked: two particles on a damped spring, a zero-length spring and an attachment to a static body against the
recurrences of implicit Euler (and BDF2) with the model's coefficients; what the engine takes of a model with every kind
(the terms, their inputs and coefficients, the hinges in libuipc's order, the elements left out where VBD evaluates no
elasticity) and of one without damping (no group); coefficients read at every step, also between the replays of a
captured step; the refusals; drag and lift against SemiImplicit's at the same state.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.semi_implicit.kernels_particle import eval_triangle
from newton.tests.ipc._fixtures import _TIGHT, _assert_refused, _model_of, _with_attachment
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires

_DT = 0.01


def _recurrence(u0, v0, mu, k, c, h, steps, integrator):
    """``u_1 .. u_steps`` of ``mu u'' = -k u - c u'`` as SolverIPC steps it: implicit Euler, or BDF2 (masses scaled by
    9/4, the damping's base ``(4 u_n - u_n-1) / 3``) after a first step of implicit Euler, whose start becomes the
    history (libuipc-warp's ``bdf2_start="bdf1"``, which SolverIPC takes)."""
    u, v = np.asarray(u0, np.float64), np.asarray(v0, np.float64)
    out = []
    for step in range(steps):
        if integrator == "bdf1" or step == 0:
            um, vm = u, v
            un = (mu * (u + h * v) + h * c * u) / (mu + h * h * k + h * c)
            vn = (un - u) / h
        else:
            tilde = 4.0 / 3.0 * u - 1.0 / 3.0 * um + 2.0 / 9.0 * h * (4.0 * v - vm)
            base, alpha = (4.0 * u - um) / 3.0, 1.5 / h
            un = (2.25 * mu * tilde + h * h * c * alpha * base) / (2.25 * mu + h * h * k + h * h * c * alpha)
            vn = (3.0 * un - 4.0 * u + um) / (2.0 * h)
            um, vm = u, v
        u, v = un, vn
        out.append(u.copy())
    return np.array(out)


def _steps(model, solver, steps, field="particle_q"):
    s0, s1 = model.state(), model.state()
    out = []
    for _ in range(steps):
        solver.step(s0, s1, None, None, _DT)
        s0, s1 = s1, s0
        out.append(getattr(s0, field).numpy().astype(np.float64))
    return np.array(out)


def test_spring_damping_as_implicit_euler(test, device):
    """Match a damped spring's length to the implicit Euler and BDF2 recurrence of a damped oscillator.

    Two particles of 0.1 kg on a spring of 50 N/m and rest length 10 cm with ``spring_damping`` 0.8 N s/m are let go 2
    cm stretched and moving apart at 0.3 m/s, without gravity. The spring's length follows the recurrence of ``(m / 2)
    u'' = -k u - c u'`` under implicit Euler and BDF2, step by step to the Newton tolerance.
    """
    for integrator in ("bdf1", "bdf2"):
        with test.subTest(integrator=integrator):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            a = builder.add_particle(wp.vec3(-0.05, 0.0, 0.0), wp.vec3(-0.15, 0.0, 0.0), 0.1, radius=0.001)
            b = builder.add_particle(wp.vec3(0.05, 0.0, 0.0), wp.vec3(0.15, 0.0, 0.0), 0.1, radius=0.001)
            builder.add_spring(a, b, 50.0, 0.8, 0)
            model = builder.finalize(device=device)
            model.particle_q.assign(np.array([[-0.06, 0.0, 0.0], [0.06, 0.0, 0.0]], np.float32))
            solver = newton.solvers.SolverIPC(model, integrator=integrator, **_TIGHT)
            test.assertIsNotNone(solver._features.damping.springs)
            q = _steps(model, solver, 30)
            expected = 0.1 + _recurrence(0.02, 0.3, 0.05, 50.0, 0.8, _DT, 30, integrator)
            np.testing.assert_allclose(q[:, 1, 0] - q[:, 0, 0], expected, rtol=0.0, atol=2e-7)


def test_stitch_damping_as_implicit_euler(test, device):
    """Match damped attachments and zero-length springs to the implicit Euler recurrence of a damped oscillator.

    A particle of 10 g on an attachment of 4 N/m with damping 0.05 N s/m to a point of a static body, and two particles
    of 10 g on a spring of rest length 0 with the same stiffness and damping, start from offsets and velocities in every
    direction, without gravity. The offset follows ``m u'' = -k u - c u'`` (the spring's pair with ``m / 2``) under
    implicit Euler, step by step to the Newton tolerance. The particle of a disabled attachment flies free. The
    particles do not collide with each other (``particle_grid`` None): the zero-length spring brings its two together.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), mass=0.0)
    u0, v0 = np.array([0.01, -0.02, 0.005]), np.array([0.05, 0.1, -0.2])
    builder.add_particle(wp.vec3(*u0), wp.vec3(*v0), 0.01, radius=0.001)
    builder.add_attachment_particle_body(0, body, body_point=(0.0, 0.0, -1.0), stiffness=4.0, damping=0.05)
    builder.add_particle(wp.vec3(*(u0 + np.array([1.0, 0.0, 0.0]))), wp.vec3(*v0), 0.01, radius=0.001)
    builder.add_particle(wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0), 0.01, radius=0.001)
    builder.add_spring(1, 2, 4.0, 0.05, 0)
    builder.add_particle(wp.vec3(2.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.1), 0.01, radius=0.001)
    builder.add_attachment_particle_body(3, body, body_point=(2.0, 0.0, -1.0), stiffness=4.0, damping=0.05,
                                         enabled=False)  # fmt: skip
    model = builder.finalize(device=device)
    model.spring_rest_length.assign(np.zeros(1, np.float32))
    model.particle_grid = None
    solver = newton.solvers.SolverIPC(model, **_TIGHT)
    test.assertIsNotNone(solver._features.damping.stitches)
    q = _steps(model, solver, 30)
    np.testing.assert_allclose(q[:, 0], _recurrence(u0, v0, 0.01, 4.0, 0.05, _DT, 30, "bdf1"), atol=2e-7)
    relative = _recurrence(u0, v0, 0.005, 4.0, 0.05, _DT, 30, "bdf1")
    np.testing.assert_allclose(q[:, 1] - q[:, 2], relative, atol=2e-7)
    free = np.array([[2.0, 0.0, 0.1 * _DT * (k + 1)] for k in range(30)])
    np.testing.assert_allclose(q[:, 3], free, atol=1e-6)


def _every_kind(device, tri_kd=0.5, edge_kd=1e-3, k_damp=2.0, spring_kd=0.3):
    """A cloth of 3 x 3 particles (8 triangles, interior bending edges), a cube of tetrahedra and two particles on a
    spring; the second triangle without elasticity (tri_ke = tri_ka = 0) and the first bending edge without stiffness,
    so VBD evaluates no damping there."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, 1.0), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0), dim_x=2, dim_y=2, cell_x=0.05,
        cell_y=0.05, mass=1.0e-3, tri_ke=100.0, tri_ka=100.0, tri_kd=tri_kd, edge_ke=1e-3, edge_kd=edge_kd,
        particle_radius=1.0e-3,
    )  # fmt: skip
    builder.add_soft_grid(
        pos=wp.vec3(1.0, 0.0, 1.0), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0), dim_x=1, dim_y=1, dim_z=1,
        cell_x=0.05, cell_y=0.05, cell_z=0.05, density=1000.0, k_mu=1.0e4, k_lambda=1.0e4, k_damp=k_damp,
    )  # fmt: skip
    a = builder.add_particle(wp.vec3(2.0, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.0), 0.01, radius=0.001)
    b = builder.add_particle(wp.vec3(2.1, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.0), 0.01, radius=0.001)
    builder.add_spring(a, b, 5.0, spring_kd, 0)
    model = builder.finalize(device=device)
    materials = model.tri_materials.numpy()
    materials[1, :2] = 0.0
    model.tri_materials.assign(materials)
    properties = model.edge_bending_properties.numpy()
    properties[0, 0] = 0.0
    model.edge_bending_properties.assign(properties)
    return model


def test_what_the_engine_takes(test, device):
    """Give the engine one damping group per element kind, with its coefficients and rest data.

    Of a model with every kind, the solver gives its engine the springs of positive rest length with ``spring_damping``;
    the tetrahedra with ``tet_poses``, their rest volumes and ``k_damp``; the cloth triangles that have elasticity, with
    ``tri_poses``, ``tri_areas`` and ``tri_kd``; the cloth hinges with stiffness, as libuipc's ``(o0, v1, v2, o1)`` of
    Newton's ``(o0, o1, v1, v2)``, with ``edge_rest_length`` and ``edge_kd``. A model without damping gives it none, and
    a default cloth (``tri_kd`` 10) its triangles' damping.
    """
    from uipc_warp.ext import damping as ext  # noqa: PLC0415

    model = _every_kind(device)
    solver = newton.solvers.SolverIPC(model)
    plan = solver._features.damping
    groups = [plan.springs.group, plan.tets.group, plan.tris.group, plan.hinges.group]
    kinds = [ext.SpringDamping, ext.StrainRateDamping, ext.StrainRateDamping, ext.HingeDamping]
    test.assertEqual([type(g) for g in groups], kinds)
    test.assertIsNone(plan.stitches)
    np.testing.assert_array_equal(plan.springs.group.edges.numpy(), model.spring_indices.numpy().reshape(-1, 2))
    np.testing.assert_allclose(plan.springs.group.c.numpy(), [0.3], rtol=1e-7)
    tets = plan.tets.group
    np.testing.assert_array_equal(plan.tets.rows, np.arange(model.tet_count))
    np.testing.assert_array_equal(tets.elements.numpy(), model.tet_indices.numpy().reshape(-1, 4))
    np.testing.assert_allclose(tets.Dm_inv.numpy(), model.tet_poses.numpy(), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(tets.measure.numpy(), 1.0 / (6.0 * np.abs(np.linalg.det(model.tet_poses.numpy()))))
    np.testing.assert_allclose(tets.k.numpy(), 2.0)
    tris = plan.tris.group
    np.testing.assert_array_equal(plan.tris.rows, [0, *range(2, 8)])  # the second triangle has no elasticity
    rows = plan.tris.rows
    np.testing.assert_array_equal(tris.elements.numpy(), model.tri_indices.numpy().reshape(-1, 3)[rows])
    np.testing.assert_array_equal(tris.Dm_inv.numpy(), model.tri_poses.numpy()[rows])
    np.testing.assert_array_equal(tris.measure.numpy(), model.tri_areas.numpy()[rows])
    np.testing.assert_allclose(tris.k.numpy(), 0.5)
    edges = model.edge_indices.numpy().reshape(-1, 4)
    cloth = (edges[:, 2] < 9) & (edges[:, 3] < 9)  # the cloth's 9 particles come first
    interior = np.flatnonzero((edges[:, 0] >= 0) & (edges[:, 1] >= 0) & cloth)
    np.testing.assert_array_equal(plan.hinges.rows, interior[interior != 0])  # edge 0 has no stiffness
    rows = plan.hinges.rows
    np.testing.assert_array_equal(plan.hinges.group.hinges.numpy(), edges[rows][:, [0, 2, 3, 1]])
    np.testing.assert_array_equal(plan.hinges.group.rest_length.numpy(), model.edge_rest_length.numpy()[rows])
    np.testing.assert_allclose(plan.hinges.group.k.numpy(), 1e-3, rtol=1e-6)
    plain = newton.solvers.SolverIPC(_every_kind(device, tri_kd=0.0, edge_kd=0.0, k_damp=0.0, spring_kd=0.0))
    terms = plain._features.damping
    test.assertEqual([terms.springs, terms.stitches, terms.tets, terms.tris, terms.hinges], [None] * 5)
    builder = newton.ModelBuilder()
    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, 1.0), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0), dim_x=2, dim_y=2, cell_x=0.05,
        cell_y=0.05, mass=1.0e-3,
    )  # fmt: skip
    default = newton.solvers.SolverIPC(builder.finalize(device=device))
    np.testing.assert_allclose(default._features.damping.tris.group.k.numpy(), 10.0)  # Newton's default tri_kd


def test_coefficients_read_at_every_step(test, device):
    """Read the triangles' damping coefficients from the model at every replay of a captured step.

    Two steps are captured once, and the triangles' ``tri_kd`` is written into the model between the replays (0 for the
    replays 3 to 5). A cloth of 4 x 4 squares held at one edge and swinging under gravity follows eager steps given the
    same coefficients, to 1e-9 m, and swings otherwise while its damping is off (by more than 1e-7 m).
    """

    def run(capture, zero=range(3, 6)):
        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 1.0), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0), dim_x=4, dim_y=4,
            cell_x=0.05, cell_y=0.05, mass=1.0e-3, tri_ke=100.0, tri_ka=100.0, tri_kd=0.05, edge_ke=1e-3,
            fix_left=True, particle_radius=1.0e-3,
        )  # fmt: skip
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverIPC(model, **_TIGHT)
        test.assertIsNotNone(solver._features.damping.tris)
        s0, s1 = model.state(), model.state()
        solver.step(s0, s1, None, None, _DT)  # loads the kernels
        s0, s1 = s1, s0
        graph = None
        if capture:
            with wp.ScopedCapture(device=device) as cap:
                solver.step(s0, s1, None, None, _DT)
                solver.step(s1, s0, None, None, _DT)
            graph = cap.graph
        given = model.tri_materials.numpy()
        out = []
        for k in range(8):
            materials = given.copy()
            if k in zero:
                materials[:, 2] = 0.0
            model.tri_materials.assign(materials)
            if capture:
                wp.capture_launch(graph)
            else:
                for _ in range(2):
                    solver.step(s0, s1, None, None, _DT)
                    s0, s1 = s1, s0
            out.append(s0.particle_q.numpy().astype(np.float64))
        return np.array(out)

    eager, captured = run(False), run(True)
    np.testing.assert_allclose(captured, eager, rtol=0.0, atol=1e-9)
    damped = run(False, zero=())
    np.testing.assert_allclose(eager[:3], damped[:3], rtol=0.0, atol=1e-9)
    # the damping's effect on the swing over those steps, 2.7e-6 m measured: far above the replays' 1e-9
    test.assertGreater(np.abs(eager[3:] - damped[3:]).max(), 1e-7)


def test_refusals(test, device):
    """Raise NotImplementedError at construction for damping, drag and lift values that SolverIPC refuses.

    A damping, drag or lift that is not finite, and a negative damping or drag, raise NotImplementedError when the
    solver is built, naming the item; so does a damped bending edge whose ``edge_rest_length`` is not positive.
    """

    def refused(edit, message):
        model = _every_kind(device)
        edit(model)
        with test.assertRaises(NotImplementedError) as caught:
            newton.solvers.SolverIPC(model)
        test.assertEqual(str(caught.exception), message)

    def column(name, row, col, value):
        def edit(model):
            a = getattr(model, name).numpy()
            a[row, col] = value
            getattr(model, name).assign(a)

        return edit

    def entry(name, row, value):
        def edit(model):
            a = getattr(model, name).numpy()
            a[row] = value
            getattr(model, name).assign(a)

        return edit

    refused(entry("spring_damping", 0, -1.0), "SolverIPC: spring 0 has spring_damping -1.0; make it 0 or positive.")
    refused(column("tet_materials", 2, 2, np.nan), "SolverIPC: tetrahedron 2 has the damping k_damp nan; make it 0 or "
            "positive.")  # fmt: skip
    refused(column("tri_materials", 3, 2, -0.5), "SolverIPC: triangle 3 has the damping tri_kd -0.5; make it 0 or "
            "positive.")  # fmt: skip
    refused(column("tri_materials", 0, 3, -2.0), "SolverIPC: triangle 0 has the drag tri_drag -2.0; make it 0 or "
            "positive.")  # fmt: skip
    refused(column("tri_materials", 0, 4, np.inf), "SolverIPC: triangle 0 has the lift tri_lift inf; make it finite.")
    edges = _every_kind(device).edge_indices.numpy().reshape(-1, 4)
    hinge = int(np.flatnonzero((edges[:, 0] >= 0) & (edges[:, 1] >= 0))[1])
    refused(column("edge_bending_properties", hinge, 1, -1.0),
            f"SolverIPC: bending edge {hinge} has the damping edge_kd -1.0; make it 0 or positive.")  # fmt: skip
    refused(entry("edge_rest_length", hinge, 0.0), f"SolverIPC: bending edge {hinge} has the damping edge_kd "
            "0.0010000000474974513 and the edge_rest_length 0.0; make the rest length positive, which the damping of "
            "a hinge scales with.")  # fmt: skip
    _assert_refused(
        test,
        _model_of(device, _with_attachment(), _with_attachment(damping=-2.0)),
        "SolverIPC: particle-body attachment 1 has damping -2.0; make it 0 or positive.",
    )


def test_drag_and_lift_as_semi_implicit(test, device):
    """Apply triangle drag and lift as SemiImplicit's external accelerations at the step's start.

    A free triangle of three particles with ``tri_drag`` 0.8 and ``tri_lift`` 0.6 moves at three velocities: the
    external accelerations a step adds to its particles are SemiImplicit's ``-(f_drag + f_lift) / m`` at the step's
    start, its ``eval_triangle`` with no elasticity. A fixed particle takes none.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    x = [(0.0, 0.0, 1.0), (0.1, 0.0, 1.0), (0.0, 0.1, 1.05)]
    v = [(0.3, -0.2, -1.0), (0.1, 0.0, -1.2), (-0.2, 0.4, -0.8)]
    for p, u, m in zip(x, v, (0.01, 0.02, 0.0), strict=True):
        builder.add_particle(wp.vec3(*p), wp.vec3(*u), m, radius=0.001)
    builder.add_triangle(0, 1, 2, tri_ke=0.0, tri_ka=0.0, tri_kd=0.0, tri_drag=0.8, tri_lift=0.6)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model)
    s0, s1 = model.state(), model.state()
    solver.step(s0, s1, None, None, _DT)
    acc = solver._forces.a.numpy()
    f = wp.zeros(3, dtype=wp.vec3, device=device)
    materials = wp.array(np.array([[0.0, 0.0, 0.0, 0.8, 0.6]], np.float32), dtype=float, device=device)
    activation = wp.zeros(1, dtype=float, device=device)
    indices = wp.array(np.array([[0, 1, 2]], np.int32), dtype=int, device=device)
    wp.launch(eval_triangle, 1, [s0.particle_q, s0.particle_qd, indices, model.tri_poses, activation, materials], [f],
              device=device)  # fmt: skip
    f = f.numpy().astype(np.float64)
    test.assertEqual(len(acc), 2)  # the fixed particle has no row
    np.testing.assert_allclose(acc[0], f[0] / 0.01, rtol=1e-6)
    np.testing.assert_allclose(acc[1], f[1] / 0.02, rtol=1e-6)
    test.assertGreater(np.abs(f[:2]).max(), 1e-4)


@requires("uipc_warp")
class TestSolverIPCDamping(unittest.TestCase):
    """The damping of springs, attachments and elements, and drag and lift."""


devices = get_cuda_test_devices()
for name, func in (
    ("test_spring_damping_as_implicit_euler", test_spring_damping_as_implicit_euler),
    ("test_stitch_damping_as_implicit_euler", test_stitch_damping_as_implicit_euler),
    ("test_what_the_engine_takes", test_what_the_engine_takes),
    ("test_coefficients_read_at_every_step", test_coefficients_read_at_every_step),
    ("test_refusals", test_refusals),
    ("test_drag_and_lift_as_semi_implicit", test_drag_and_lift_as_semi_implicit),
):
    add_function_test(TestSolverIPCDamping, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
