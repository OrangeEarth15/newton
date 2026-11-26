# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC's triangulation of Newton's collision shapes: capsules, cylinders, cones, ellipsoids, convex hulls and
scaled meshes rest on the ground at their triangulated extents, along local Z, with ``shape_segments`` segments around;
a plane on a body is refused."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.ipc._fixtures import _TIGHT
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires

_D_HAT = 1.0e-3
# _TIGHT with ten times its velocity tolerance (1e-5 m/s: an update of at most 1e-7 m at the steps of 0.01 s), far below
# the band of d_hat (1 mm) in which the shapes come to rest
_SETTLED = {**_TIGHT, "newton_velocity_tol": 1.0e-5}


def _rotation(axis, angle):
    """The rotation matrix of ``angle`` about ``axis``."""
    axis = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * K @ K


def _quat(R):
    """The quaternion (x, y, z, w) of the rotation matrix ``R``."""
    w = 0.5 * np.sqrt(max(1.0 + R[0, 0] + R[1, 1] + R[2, 2], 0.0))
    return np.array(
        [(R[2, 1] - R[1, 2]) / (4.0 * w), (R[0, 2] - R[2, 0]) / (4.0 * w), (R[1, 0] - R[0, 1]) / (4.0 * w), w]
    )


def _pyramid():
    """A square pyramid, base 0.1 m wide on z = 0 and apex 0.08 m up, as a closed triangle mesh."""
    v = np.array([[-0.05, -0.05, 0.0], [0.05, -0.05, 0.0], [0.05, 0.05, 0.0], [-0.05, 0.05, 0.0], [0.0, 0.0, 0.08]])
    f = np.array([[0, 2, 1], [0, 3, 2], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    return newton.Mesh(v, f.reshape(-1))


def _mesh(kind, segments):
    """The triangle mesh of a shape as the documentation says SolverIPC makes it: Newton's mesh of the type along
    local Z with ``segments`` segments around, at the shape's size."""
    opts = {"compute_normals": False, "compute_uvs": False, "compute_inertia": False}
    name, size = kind
    if name == "capsule":
        return newton.Mesh.create_capsule(*size, up_axis=newton.Axis.Z, segments=segments, **opts).vertices
    if name == "cylinder":
        return newton.Mesh.create_cylinder(*size, up_axis=newton.Axis.Z, segments=segments, **opts).vertices
    if name == "cone":
        return newton.Mesh.create_cone(*size, up_axis=newton.Axis.Z, segments=segments, **opts).vertices
    if name == "ellipsoid":
        return newton.Mesh.create_ellipsoid(
            *size, num_latitudes=segments // 2, num_longitudes=segments, **opts
        ).vertices
    if name == "convex hull":
        return _pyramid().vertices * np.array(size)
    return newton.Mesh.create_box(0.1, 0.1, 0.1, duplicate_vertices=False, **opts).vertices * np.array(size)


# shape, size, pose of the body (axis, angle): lying shapes are turned about their own axis so that a face of the
# triangulation is down, for 8 segments as for the default 24 where that is checked
_SHAPES = (
    (("capsule", (0.05, 0.08)), ((1.0, 0.0, 0.0), 0.0)),
    (("capsule", (0.05, 0.08)), "lying"),
    (("cylinder", (0.1, 0.15)), ((1.0, 0.0, 0.0), 0.0)),
    (("cylinder", (0.1, 0.15)), "lying"),
    (("cone", (0.08, 0.1)), ((1.0, 0.0, 0.0), 0.0)),
    (("ellipsoid", (0.1, 0.08, 0.05)), ((1.0, 0.0, 0.0), 0.0)),
    (("ellipsoid", (0.05, 0.1, 0.08)), ((0.0, 1.0, 0.0), 0.5 * np.pi)),  # local x vertical, its shortest axis
    (("convex hull", (1.0, 1.5, 0.75)), ((1.0, 0.0, 0.0), 0.0)),
    (("mesh", (1.5, 1.0, 0.5)), ((0.0, 0.0, 1.0), 0.3)),
)


def _pose(kind, pose):
    """The rotation of a body: about an axis, or for a lying capsule or cylinder along x, turned about its axis so
    that a face of the triangulation is down: a capsule by pi / 24 (24 segments), a cylinder by pi / 8, where 8 and
    24 segments both have a face down."""
    if pose != "lying":
        return _rotation(*pose)
    turn = np.pi / 24.0 if kind[0] == "capsule" else np.pi / 8.0
    return _rotation((1.0, 0.0, 0.0), turn) @ _rotation((0.0, 1.0, 0.0), 0.5 * np.pi)


def _scene(device, gap):
    """Every shape of ``_SHAPES`` on a body of its own, ``gap`` above the ground plane with its default triangulation,
    a metre apart."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    rotations = []
    for k, (kind, pose) in enumerate(_SHAPES):
        R = _pose(kind, pose)
        rotations.append(R)
        z = gap - (_mesh(kind, 24) @ R.T)[:, 2].min()
        body = builder.add_body(xform=wp.transform(wp.vec3(float(k), 0.0, float(z)), wp.quat(*_quat(R))))
        name, size = kind
        add = {
            "capsule": lambda b, s: builder.add_shape_capsule(b, radius=s[0], half_height=s[1]),
            "cylinder": lambda b, s: builder.add_shape_cylinder(b, radius=s[0], half_height=s[1]),
            "cone": lambda b, s: builder.add_shape_cone(b, radius=s[0], half_height=s[1]),
            "ellipsoid": lambda b, s: builder.add_shape_ellipsoid(b, rx=s[0], ry=s[1], rz=s[2]),
            "convex hull": lambda b, s: builder.add_shape_convex_hull(b, mesh=_pyramid(), scale=s),
            "mesh": lambda b, s: builder.add_shape_mesh(
                b, mesh=newton.Mesh.create_box(0.1, 0.1, 0.1, duplicate_vertices=False), scale=s
            ),
        }[name]
        add(body, size)
    return builder.finalize(device=device), rotations


def _rest_gaps(model, solver, segments, steps=25):
    """How far above the ground each body's triangulated shape (``_mesh`` with ``segments``) rests after ``steps``
    steps of 0.01 s, at the body's pose then."""
    s0, s1 = model.state(), model.state()
    for _ in range(steps):
        solver.step(s0, s1, None, None, 0.01)
        s0, s1 = s1, s0
    body_q = s0.body_q.numpy().astype(np.float64)
    gaps = []
    for k, (kind, _) in enumerate(_SHAPES):
        q = body_q[k]
        x, w = q[3:6], q[6]
        v = _mesh(kind, segments)
        t = 2.0 * np.cross(x, v)
        gaps.append(float((v + w * t + np.cross(x, t))[:, 2].min() + q[2]))
    return np.array(gaps), body_q


def test_primitives_rest_at_their_triangulated_extents(test, device):
    """Rest each primitive shape on the ground at the extent of its triangulation.

    Capsules upright and lying, cylinders upright and lying, an upright cone (base at -half_height), ellipsoids with
    local z and local x vertical, a convex hull scaled (1, 1.5, 0.75) and a box mesh scaled (1.5, 1, 0.5) each start 0.5
    mm above the ground with its triangulation along local Z of 24 segments around. Each comes to rest with the lowest
    point of that triangulation, at the body's pose, inside the barrier's reach of the ground (between 0 and d_hat), its
    pose unturned. A shape triangulated along another axis, unscaled or with other segments would rest millimetres off.
    With shape_segments=8 the lying cylinder rests on a face of 8 segments, r cos(pi/8) under its axis instead of r
    cos(pi/24).
    """
    model, rotations = _scene(device, 0.5 * _D_HAT)
    test.assertTrue(np.all(model.body_mass.numpy() > 0.0))  # every body falls onto the ground
    solver = newton.solvers.SolverIPC(model, d_hat=_D_HAT, **_SETTLED)
    gaps, body_q = _rest_gaps(model, solver, 24)
    for k, ((name, _), _) in enumerate(_SHAPES):
        with test.subTest(shape=name, body=k):
            test.assertGreater(gaps[k], 0.0)
            test.assertLess(gaps[k], _D_HAT)
            np.testing.assert_allclose(np.abs(body_q[k, 3:] @ _quat(rotations[k])), 1.0, rtol=0.0, atol=1.0e-5)
    cylinder = 3
    radius = _SHAPES[cylinder][0][1][0]
    test.assertAlmostEqual(body_q[cylinder, 2] - gaps[cylinder], radius * np.cos(np.pi / 24.0), delta=1.0e-5)
    model, _ = _scene(device, 0.5 * _D_HAT)
    coarse = newton.solvers.SolverIPC(model, d_hat=_D_HAT, shape_segments=8, **_SETTLED)
    gaps8, body_q8 = _rest_gaps(model, coarse, 8)
    test.assertGreater(gaps8[cylinder], 0.0)
    test.assertLess(gaps8[cylinder], _D_HAT)
    test.assertAlmostEqual(body_q8[cylinder, 2] - gaps8[cylinder], radius * np.cos(np.pi / 8.0), delta=1.0e-5)
    test.assertGreater(
        body_q[cylinder, 2] - body_q8[cylinder, 2], 0.9 * radius * (np.cos(np.pi / 24) - np.cos(np.pi / 8))
    )


def test_rejects_a_plane_on_a_body(test, device):
    """Refuse a plane shape attached to a body by name when the solver is built.

    libuipc's half planes do not move. A plane on the world is taken.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    model = builder.finalize(device=device)
    newton.solvers.SolverIPC(model)
    builder.add_shape_plane(body=body, xform=wp.transform(wp.vec3(0.0, 0.0, -0.1), wp.quat_identity()))
    model = builder.finalize(device=device)
    with test.assertRaisesRegex(NotImplementedError, "the plane shape 2 is on body 0; put planes on the world"):
        newton.solvers.SolverIPC(model)


devices = get_cuda_test_devices()


@requires("uipc_warp")
class TestSolverIPCShapes(unittest.TestCase):
    pass


for name, func in (
    ("test_primitives_rest_at_their_triangulated_extents", test_primitives_rest_at_their_triangulated_extents),
    ("test_rejects_a_plane_on_a_body", test_rejects_a_plane_on_a_body),
):
    add_function_test(TestSolverIPCShapes, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
