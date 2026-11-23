# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""What SolverIPC takes from a model, a state and a control, and what it rejects."""

import ast
import importlib
import importlib.util
import pathlib
import unittest
import warnings
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton.tests.ipc._fixtures import _box_body, _cloth
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires


def _rejects(test, builder, pattern, device):
    """``SolverIPC`` raises ``NotImplementedError`` matching ``pattern`` for the model of ``builder``."""
    model = builder.finalize(device=device)
    with test.assertRaisesRegex(NotImplementedError, pattern):
        newton.solvers.SolverIPC(model)


def _two_particles():
    builder = newton.ModelBuilder()
    builder.add_particle(wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.0), 1.0)
    builder.add_particle(wp.vec3(0.1, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.0), 1.0)
    return builder


def test_rejects_springs_without_a_libuipc_form(test, device):
    """Reject a damped spring (libuipc has no damping element), a spring of negative rest length or of negative
    stiffness and one that joins a particle to itself."""
    for case, pattern in (
        ("damped", "spring 0 has spring_damping 0.5; set it to 0"),
        ("negative rest length", "spring_rest_length -0.1[0-9]*; make it 0 or positive"),
        ("negative", "spring_stiffness -1000.0; make it 0 or positive"),
        ("itself", "joins particle 0 to itself"),
    ):
        with test.subTest(case=case):
            builder = _two_particles()
            builder.add_spring(0, 0 if case == "itself" else 1, -1.0e3 if case == "negative" else 1.0e3,
                               0.5 if case == "damped" else 0.0, 0.0)  # fmt: skip
            if case == "negative rest length":
                builder.spring_rest_length[0] = -0.1
            _rejects(test, builder, pattern, device)


def test_rejects_attachment_damping(test, device):
    """Reject a particle-body attachment with damping, naming it: libuipc's stitch has no damping."""
    builder = _two_particles()
    b = _box_body(builder, (0.0, 0.0, 2.0), (0.1, 0.1, 0.1))
    builder.add_attachment_particle_body(0, b, body_point=(0.0, 0.0, -0.5))
    builder.add_attachment_particle_body(1, b, body_point=(0.1, 0.0, -0.5), damping=2.0)
    _rejects(test, builder, "particle-body attachment 1 has damping 2.0; set it to 0", device)


def _hinged_pair(builder):
    """Two boxes hinged to the world, in one articulation: the joints of a mimic relation."""
    joints = []
    for i in range(2):
        b = builder.add_link(xform=wp.transform(wp.vec3(0.5 * i, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_box(b, hx=0.1, hy=0.1, hz=0.1)
        joints.append(builder.add_joint_revolute(-1, b, axis=newton.Axis.Y))
    builder.add_articulation(joints)
    return joints


def test_rejects_mimic_constraints(test, device):
    """Reject mimic relations, before the revolute joints they need."""
    builder = newton.ModelBuilder()
    j0, j1 = _hinged_pair(builder)
    builder.set_joint_mimic(j1, j0)
    _rejects(test, builder, "mimic", device)


def _equality_row(enabled):
    """A box body and a MuJoCo CONNECT row that holds it to the world, linked to no loop joint."""
    builder = newton.ModelBuilder()
    body = _box_body(builder, (0.0, 0.0, 1.0), (0.1, 0.1, 0.1))
    builder.add_custom_values(
        **{
            "mujoco:equality_constraint_type": 0,
            "mujoco:equality_constraint_objtype": 1,
            "mujoco:equality_constraint_body1": body,
            "mujoco:equality_constraint_body2": -1,
            "mujoco:equality_constraint_anchor": wp.vec3(0.0, 0.0, 1.0),
            "mujoco:equality_constraint_enabled": enabled,
        }
    )
    return builder


def test_rejects_equality_constraints(test, device):
    """Reject an enabled MuJoCo equality row that no loop joint holds instead of letting the body fall through it;
    a row disabled when the solver is built and enabled later is refused when notify_model_changed reports
    CONSTRAINT_PROPERTIES, and so is a mimic relation set later."""
    _rejects(test, _equality_row(True), "1 enabled equality rows that no loop joint holds \\(rows \\[0\\]\\)", device)
    model = _equality_row(False).finalize(device=device)
    solver = newton.solvers.SolverIPC(model)
    model.mujoco.equality_constraint_enabled.assign(np.array([True]))
    with test.assertRaisesRegex(NotImplementedError, "1 enabled equality rows that no loop joint holds"):
        solver.notify_model_changed(newton.ModelFlags.CONSTRAINT_PROPERTIES)

    builder = newton.ModelBuilder()
    _hinged_pair(builder)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model)
    model.joint_mimic_joint.assign(np.array([-1, 0], np.int32))
    with test.assertRaisesRegex(NotImplementedError, "1 mimic constraints; remove them"):
        solver.notify_model_changed(newton.ModelFlags.CONSTRAINT_PROPERTIES)


# Newton's importers turn the weld into a FIXED joint between the boxes and the connect into a BALL joint from the world
# to box a, both loop joints, and keep the two rows linked to them
_LOOP_MJCF = """
<mujoco>
  <worldbody>
    <body name="a" pos="0 0 1.0">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" density="1000"/>
    </body>
    <body name="b" pos="0.3 0 1.0">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" density="1000"/>
    </body>
  </worldbody>
  <equality>
    <weld body1="a" body2="b"/>
    <connect body1="a" anchor="0 0 0"/>
  </equality>
</mujoco>
"""


def _rotation(q):
    """The rotation matrix of the xyzw quaternion ``q``."""
    x, y, z, w = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_equality_rows_held_by_loop_joints(test, device):
    """The MJCF equalities that Newton's importers turn into loop joints are taken, by SolverSteel's rule: a connect of
    a free box to the world at its centre and a weld of a second free box to it become a BALL and a FIXED joint in no
    articulation, their rows stay enabled and linked to them, and over 30 steps of 0.01 s the first box stays at its
    anchor and the second keeps its pose in the first's frame, within the joints' penalties, while the pair swings
    down under gravity. The model imported without the conversion keeps rows linked to no joint, which are refused,
    and a row linked to a joint of another kind is refused too."""

    def imported(convert):
        builder = newton.ModelBuilder()
        if convert:
            with test.assertWarnsRegex(newton.NewtonWarning, "another joint already connects these bodies"):
                builder.add_mjcf(_LOOP_MJCF)
        else:
            builder.add_mjcf(_LOOP_MJCF, convert_mjc_equality_constraints=False)
        return builder.finalize(device=device)

    model = imported(True)
    rows = model.mujoco
    kinds = [newton.JointType(int(t)) for t in model.joint_type.numpy()]
    targets = rows.equality_constraint_target.numpy()
    test.assertEqual(rows.equality_constraint_enabled.numpy().tolist(), [True, True])
    test.assertEqual(sorted(kinds[t] for t in targets), [newton.JointType.BALL, newton.JointType.FIXED])
    test.assertTrue(np.all(model.joint_articulation.numpy()[targets] < 0))
    solver = newton.solvers.SolverIPC(model)
    s0, s1 = model.state(), model.state()
    for _ in range(30):
        solver.step(s0, s1, None, None, 0.01)
        s0, s1 = s1, s0
    q = s0.body_q.numpy().astype(np.float64)
    a, b = q[0], q[1]
    np.testing.assert_allclose(a[:3], (0.0, 0.0, 1.0), rtol=0.0, atol=1e-4)  # the connect
    R_a, R_b = _rotation(a[3:]), _rotation(b[3:])
    np.testing.assert_allclose(R_a.T @ (b[:3] - a[:3]), (0.3, 0.0, 0.0), rtol=0.0, atol=1e-3)  # the weld
    np.testing.assert_allclose(R_a.T @ R_b, np.eye(3), rtol=0.0, atol=1e-3)
    test.assertGreater(1.0 - b[2], 0.05)  # the pair swings down about a's centre

    with test.assertRaisesRegex(
        NotImplementedError, "2 enabled equality rows that no loop joint holds \\(rows \\[0, 1\\]\\)"
    ):
        newton.solvers.SolverIPC(imported(False))
    swapped = imported(True)
    swapped.mujoco.equality_constraint_target.assign(targets[::-1].copy())  # each row on the other's joint
    with test.assertRaisesRegex(NotImplementedError, "2 enabled equality rows that no loop joint holds"):
        newton.solvers.SolverIPC(swapped)


def test_rejects_tendons(test, device):
    """Reject a tendon between two bodies."""
    builder = newton.ModelBuilder()
    a = _box_body(builder, (0.0, 0.0, 1.0), (0.1, 0.1, 0.1))
    b = _box_body(builder, (0.5, 0.0, 1.0), (0.1, 0.1, 0.1))
    builder.add_tendon([newton.TendonGuide(body=a), newton.TendonGuide(body=b)])
    _rejects(test, builder, "tendons", device)


def test_rejects_muscles(test, device):
    """Reject a legacy muscle between two bodies."""
    builder = newton.ModelBuilder()
    a = _box_body(builder, (0.0, 0.0, 1.0), (0.1, 0.1, 0.1))
    b = _box_body(builder, (0.5, 0.0, 1.0), (0.1, 0.1, 0.1))
    with test.assertWarns(newton.NewtonDeprecationWarning):
        builder.add_muscle([a, b], [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)], 1.0, 0.5, 0.1, 1.0, 0.0)
    _rejects(test, builder, "muscles", device)


def test_rejects_activations(test, device):
    """Reject non-zero triangle and tetrahedron activations, which the solver does not read."""
    for field in ("tri_activations", "tet_activations"):
        with test.subTest(field=field):
            builder = newton.ModelBuilder()
            if field == "tri_activations":
                _cloth(builder, 2, 0.05, 1.0)
            else:
                builder.add_soft_grid(
                    pos=wp.vec3(0.0, 0.0, 1.0),
                    rot=wp.quat_identity(),
                    vel=wp.vec3(0.0, 0.0, 0.0),
                    dim_x=1,
                    dim_y=1,
                    dim_z=1,
                    cell_x=0.1,
                    cell_y=0.1,
                    cell_z=0.1,
                    density=1000.0,
                    k_mu=1.0e4,
                    k_lambda=1.0e4,
                    k_damp=0.0,
                )
            getattr(builder, field)[0] = 0.5
            _rejects(test, builder, "activations", device)


def test_control_activations_refused_at_eager_steps(test, device):
    """A step that is not captured refuses a Control with a triangle or a tetrahedron activation other than 0, which
    the solver does not read, naming the array, and steps again once it is 0; given no control it checks the model's
    own, which change in place after construction. A captured step does not check them."""
    builder = newton.ModelBuilder()
    _cloth(builder, 2, 0.05, 1.0)
    builder.add_soft_grid(
        pos=wp.vec3(1.0, 0.0, 1.0), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0), dim_x=1, dim_y=1, dim_z=1,
        cell_x=0.1, cell_y=0.1, cell_z=0.1, density=1000.0, k_mu=1.0e4, k_lambda=1.0e4, k_damp=0.0,
    )  # fmt: skip
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model)
    s0, s1 = model.state(), model.state()
    control = model.control()
    solver.step(s0, s1, control, None, 0.01)
    for name in ("tri_activations", "tet_activations"):
        with test.subTest(name):
            values = getattr(control, name).numpy()
            values[-1] = 0.25
            getattr(control, name).assign(values)
            with test.assertRaisesRegex(NotImplementedError, f"the step's control has non-zero {name}; set them to 0"):
                solver.step(s1, s0, control, None, 0.01)
            values[-1] = 0.0
            getattr(control, name).assign(values)
            solver.step(s1, s0, control, None, 0.01)
    activations = model.tet_activations.numpy()
    activations[0] = 1.0
    model.tet_activations.assign(activations)
    with test.assertRaisesRegex(NotImplementedError, "non-zero tet_activations"):
        solver.step(s0, s1, None, None, 0.01)
    with wp.ScopedCapture(device=device) as capture:
        solver.step(s0, s1, None, None, 0.01)
    wp.capture_launch(capture.graph)


def test_rejects_unsupported_joints(test, device):
    """Reject the joints SolverIPC does not simulate, naming what libuipc lacks: rod joints, drives, passive springs,
    limits and armature on ball and free joints, joint friction, and a negative effort limit on a driven joint."""

    def hinged(add, **dof):
        builder = newton.ModelBuilder()
        link = builder.add_link(xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
        builder.add_articulation([add(builder, link)])
        for name, value in dof.items():  # on the first degree of freedom
            getattr(builder, f"joint_{name}")[0] = value
        return builder

    def revolute(builder, link):
        return builder.add_joint_revolute(-1, link, axis=newton.Axis.Y)

    def ball(builder, link):
        return builder.add_joint_ball(-1, link, parent_xform=wp.transform((0.0, 0.0, 1.2), wp.quat_identity()))

    def rod(builder, link):
        return builder.add_joint_rod(-1, link)

    free = newton.ModelBuilder()
    _box_body(free, (0.0, 0.0, 1.0), (0.1, 0.1, 0.1))
    free.joint_target_mode[0] = int(newton.JointTargetMode.POSITION)
    for name, builder, pattern in (
        ("rod", hinged(rod), "joint 0 is a ROD joint; use SolverVBD or SolverKamino for rods"),
        ("free drive", free, "FREE joint 0 has the target mode POSITION"),
        ("ball drive", hinged(ball, target_ke=10.0), "BALL joint 0 has target_ke 10.0.*; set the gains"),
        ("ball limit", hinged(ball, limit_lower=-0.5), "limits revolute and prismatic joints only"),
        ("ball armature", hinged(ball, armature=0.01), "armature of revolute and prismatic joints only"),
        ("friction", hinged(revolute, friction=0.5), "joint_friction 0.5; set it to 0. libuipc's joints have no"),
        ("effort limit", hinged(revolute, target_ke=100.0, effort_limit=-5.0), "joint_effort_limit -5.0"),
    ):
        with test.subTest(name):
            _rejects(test, builder, pattern, device)


def test_rejects_gradients(test, device):
    """Reject a model finalized with gradients, as SolverSteel does."""
    builder = _two_particles()
    model = builder.finalize(device=device, requires_grad=True)
    with test.assertRaisesRegex(NotImplementedError, "gradients"):
        newton.solvers.SolverIPC(model)


def test_warns_of_velocity_limits(test, device):
    """SolverIPC does not limit joint velocities, so it warns of a joint velocity limit below Newton's default, once
    per solver: when it is built, or at the first notification of the joints' DOFs that brings one, naming how many
    DOFs have one and the first of them. Newton's default draws no warning."""
    builder = newton.ModelBuilder()
    for x, add in ((0.0, builder.add_joint_revolute), (1.0, builder.add_joint_ball)):  # DOF 0; DOFs 1 to 3
        link = builder.add_link(xform=wp.transform((x, 0.0, 1.0), wp.quat_identity()))
        builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
        builder.add_articulation([add(-1, link, parent_xform=wp.transform((x, 0.0, 1.0), wp.quat_identity()))])
    model = builder.finalize(device=device)
    default = model.joint_velocity_limit.numpy()
    flags = newton.ModelFlags.JOINT_DOF_PROPERTIES

    def limit(dofs, value):
        limits = default.copy()
        limits[dofs] = value
        model.joint_velocity_limit.assign(limits)

    def warned(call):
        """What ``call()`` returns, and the messages of the NewtonWarnings it gives."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = call()
        return result, [str(w.message) for w in caught if issubclass(w.category, newton.NewtonWarning)]

    def message(dofs, first):
        return (
            f"SolverIPC does not limit joint velocities: {dofs} a joint_velocity_limit below Newton's default 1e+06, "
            f"the first DOF {first}."
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        first = newton.solvers.SolverIPC(model)
        first.notify_model_changed(flags)
    limit([1, 2, 3], 3.0)
    second, at_build = warned(lambda: newton.solvers.SolverIPC(model))
    test.assertEqual(at_build, [message("3 joint DOFs have", "1 of the BALL joint 1 (3.0)")])
    test.assertEqual(warned(lambda: second.notify_model_changed(flags))[1], [])
    limit([0], 2.5)  # the first solver, built at the default, warns at its next notification and only then
    for messages in ([message("1 joint DOF has", "0 of the REVOLUTE joint 0 (2.5)")], []):
        test.assertEqual(warned(lambda: first.notify_model_changed(flags))[1], messages)


def _belt(device, enable_surface_velocity):
    """A box above a static two-triangle mesh, a conveyor belt when the mesh has a surface velocity."""
    vertices = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]], np.float32)
    belt = newton.Mesh(
        vertices,
        np.array([0, 1, 2, 0, 2, 3], np.int32),
        compute_inertia=False,
        enable_surface_velocity=enable_surface_velocity,
    )
    builder = newton.ModelBuilder()
    builder.add_shape_mesh(body=-1, mesh=belt)
    _box_body(builder, (0.0, 0.0, 0.205), (0.2, 0.2, 0.2))
    return builder.finalize(device=device)


def test_rejects_surface_velocity(test, device):
    """Reject a collision mesh with a surface velocity, which the solver's own contacts would ignore."""
    with test.assertRaisesRegex(NotImplementedError, "surface velocit"):
        newton.solvers.SolverIPC(_belt(device, True))
    newton.solvers.SolverIPC(_belt(device, False))


def _open_plane(builder, z=0.0):
    """A static square of two triangles, 2 m wide, at height ``z``: an open mesh."""
    vertices = np.array([[-1.0, -1.0, z], [1.0, -1.0, z], [1.0, 1.0, z], [-1.0, 1.0, z]], np.float32)
    plane = newton.Mesh(vertices, np.array([0, 1, 2, 0, 2, 3], np.int32), compute_inertia=False)
    builder.add_shape_mesh(body=-1, mesh=plane)


def test_rejects_intersecting_start(test, device):
    """Reject the surfaces of different bodies that cross at the start without a pair inside the contact thickness,
    which a step would not notice; accept the crossings that contact ignores, and flat sheets."""
    from uipc_warp.validation import SurfaceChecker  # noqa: PLC0415

    with test.subTest(case="box through an open plane"):
        builder = newton.ModelBuilder()
        _open_plane(builder)
        _box_body(builder, (0.0, 0.0, 0.19), (0.2, 0.2, 0.2))  # 1 cm through the plane
        with test.assertRaisesRegex(ValueError, "intersect"):
            newton.solvers.SolverIPC(builder.finalize(device=device))
    with test.subTest(case="cloth through a static box"):
        builder = newton.ModelBuilder()
        builder.add_shape_box(
            -1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()), hx=0.1, hy=0.1, hz=0.1
        )
        _cloth(builder, 4, 0.1, 0.5, center=(0.033, 0.021))  # the box's vertical edges pierce its triangles
        with test.assertRaisesRegex(ValueError, "intersect"):
            newton.solvers.SolverIPC(builder.finalize(device=device))
    with test.subTest(case="crossings that contact ignores"):
        builder = newton.ModelBuilder()
        _open_plane(builder)
        # two overlapping shapes of one body, above the plane
        body = _box_body(builder, (0.0, 0.0, 0.5), (0.2, 0.2, 0.2))
        builder.add_shape_box(
            body, xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()), hx=0.2, hy=0.05, hz=0.05
        )
        # a kinematic body through the static plane
        kinematic = builder.add_body(xform=wp.transform(wp.vec3(0.6, 0.6, 0.0), wp.quat_identity()), is_kinematic=True)
        builder.add_shape_box(kinematic, hx=0.1, hy=0.1, hz=0.1)
        solver = newton.solvers.SolverIPC(builder.finalize(device=device))
        # the surfaces do cross: the check let these crossings through
        test.assertGreater(SurfaceChecker.from_engine(solver._engine).intersections(solver._engine.vm.x), 0)
    with test.subTest(case="a flat sheet off the axes"):
        # a 10 x 10 cloth in the plane y = x + 50, 100 m wide: its float32 vertices are not exactly coplanar, and
        # the brute-force checker counts crossings between its nearly coplanar triangles, which are not there
        t = np.linspace(-50.0, 50.0, 10)
        x, z = np.meshgrid(t, t, indexing="ij")
        vertices = np.stack([x, x + 50.0, z], axis=-1).reshape(-1, 3)
        quads = [(10 * i + j, 10 * i + j + 10, 10 * i + j + 11, 10 * i + j + 1) for i in range(9) for j in range(9)]
        builder = newton.ModelBuilder()
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vertices=[wp.vec3(*v) for v in vertices],
            indices=[i for a, b, c, d in quads for i in (a, b, c, a, c, d)],
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.1,
            tri_ke=1.0e4,
            tri_ka=1.0e4,
            tri_kd=0.0,
        )
        solver = newton.solvers.SolverIPC(builder.finalize(device=device))
        test.assertGreater(SurfaceChecker.from_engine(solver._engine).intersections(solver._engine.vm.x), 0)


def test_rejects_start_in_contact(test, device):
    """Reject the surfaces of different bodies that start at or inside their contact thickness, naming the closest
    pair, which every step would fail on; accept a gap above the thickness, also inside d_hat, and contact that the
    solve skips (surfaces that never move, the inside of a body)."""

    def build(*parts, **kwargs):
        builder = newton.ModelBuilder()
        for part in parts:
            part(builder)
        return newton.solvers.SolverIPC(builder.finalize(device=device), **kwargs)

    def ground(builder):
        builder.add_ground_plane()

    def static_box(builder):
        # all coordinates of these boxes are exact in float32, so that touching faces are 0 m apart
        builder.add_shape_box(
            -1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.125), wp.quat_identity()), hx=0.125, hy=0.125, hz=0.125
        )

    def box(z, kinematic=False, half=0.125):
        def add(builder):
            b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()), is_kinematic=kinematic)
            builder.add_shape_box(b, hx=half, hy=half, hz=half)

        return add

    def cloth(z, center=(0.0, 0.0)):
        return lambda builder: _cloth(builder, 4, 0.05, z, center=center)

    def particle(z, radius, x=0.0):
        return lambda builder: builder.add_particle(wp.vec3(x, 0.0, z), wp.vec3(0.0, 0.0, 0.0), 1.0e-3, radius=radius)

    def margin_box(z, margin):
        def add(builder):
            b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()))
            builder.add_shape_box(b, hx=0.125, hy=0.125, hz=0.125, cfg=newton.ModelBuilder.ShapeConfig(margin=margin))

        return add

    def thick_ground(builder):
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(margin=0.01))

    def spring(builder):
        builder.add_spring(0, 1, 10.0, 0.0, 0.0)

    for case, parts, kwargs, pattern in (
        ("on the ground", (ground, box(0.125)), {}, r"body 0 and plane shape 0 start 0 m apart"),
        ("into the ground", (ground, box(0.12)), {}, r"body 0 and plane shape 0 start -0.005 m apart"),
        ("on a static box", (static_box, box(0.375)), {}, r"static shape 0 and body 0 start 0 m apart"),
        ("on another body", (box(0.125), box(0.375)), {}, r"body 0 and body 1 start 0 m apart"),
        # 1 cm cubes stacked in touching contact: in float32 these faces are 9.3e-10 m apart
        ("apart by rounding", (box(0.015, half=0.005), box(0.025, half=0.005)), {}, r"rounding of float32 positions"),
        ("cloth inside its thickness", (static_box, cloth(0.251)), {"thickness": 0.002}, r"thickness \(0.002 m\)"),
        # a free particle's thickness is its radius, a shape's its margin; a pair's is the sum of its two sides'
        (
            "particle inside its radius",
            (ground, particle(0.099, 0.1)),
            {},
            r"particle 0 and plane shape 0 start 0.099 m apart, at or inside their contact thickness \(0.1 m\)",
        ),
        (
            "two particles inside their radii",
            (particle(1.0, 0.1), particle(1.0, 0.1, x=0.15)),
            {},
            r"particle 0 and particle 1 start 0.15 m apart, at or inside their contact thickness \(0.2 m\)",
        ),
        # with a spring the scene gives the engine its contact surface, whose vertices are the particles too
        (
            "two particles inside their radii, joined by a spring",
            (particle(1.0, 0.1), particle(1.0, 0.1, x=0.15), spring),
            {},
            r"particle 0 and particle 1 start 0.15 m apart, at or inside their contact thickness \(0.2 m\)",
        ),
        (
            "box inside its margin",
            (ground, margin_box(0.13, 0.01)),
            {},
            r"body 0 and plane shape 0 start 0.005 m apart",
        ),
        ("box inside the ground's margin", (thick_ground, margin_box(0.13, 0.0)), {}, r"thickness \(0.01 m\)"),
        ("both margins", (thick_ground, margin_box(0.14, 0.01)), {}, r"start 0.015 m apart, .* thickness \(0.02 m\)"),
    ):
        with test.subTest(case=case), test.assertRaisesRegex(ValueError, pattern):
            build(*parts, **kwargs)
    # a gap above the thickness, here inside d_hat (1 mm, 1 um); a kinematic body on a static box; touching shapes of
    # one body
    build(ground, static_box, box(0.376), cloth(0.003, center=(1.0, 0.0)), thickness=0.002)
    build(box(0.125), box(0.375 + 1.0e-6))
    build(static_box, box(0.375, kinematic=True))
    # each pair's own thickness: a small particle 1 cm above a static box beside a large particle far away; a free
    # particle and a box above their radius and margins; a cloth 3 mm above the ground whatever its radius (0.1)
    build(static_box, particle(0.26, 0.001), particle(0.5, 0.1, x=3.0))
    build(ground, particle(0.101, 0.1, x=1.0), margin_box(0.136, 0.01))
    build(thick_ground, margin_box(0.146, 0.01))

    def two_shapes(builder):
        b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.2), wp.quat_identity()))
        builder.add_shape_box(b, hx=0.125, hy=0.125, hz=0.125)
        builder.add_shape_box(
            b, xform=wp.transform(wp.vec3(0.0, 0.0, 0.25), wp.quat_identity()), hx=0.125, hy=0.125, hz=0.125
        )

    build(ground, two_shapes)


def _kinematic_root_pendulum(device):
    """Newton's test_kinematic_links scene of a kinematic revolute root, without gravity: a kinematic link that turns
    about y with a box 0.6 m out along x, a pendulum link hinged to it 0.45 m out with a sphere, and a free probe sphere
    at x = 0.72 m. The links are added without a pose: the model holds them at the identity, where the probe crosses
    the root's box (and, built where forward kinematics puts it, the pendulum's sphere)."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    root = builder.add_link(mass=1.0, is_kinematic=True, label="kinematic_root")
    pendulum = builder.add_link(mass=1.0, label="pendulum")
    joints = [
        builder.add_joint_revolute(parent=-1, child=root, axis=newton.Axis.Y),
        builder.add_joint_revolute(
            parent=root,
            child=pendulum,
            axis=newton.Axis.Y,
            parent_xform=wp.transform(wp.vec3(0.45, 0.0, 0.0), wp.quat_identity()),
        ),
    ]
    builder.add_articulation(joints)
    builder.add_shape_box(
        root, xform=wp.transform(wp.vec3(0.6, 0.0, 0.0), wp.quat_identity()), hx=0.15, hy=0.08, hz=0.08
    )
    builder.add_shape_sphere(pendulum, xform=wp.transform(wp.vec3(0.3, 0.0, 0.0), wp.quat_identity()), radius=0.1)
    probe = builder.add_body(xform=wp.transform(wp.vec3(0.72, 0.0, 0.0), wp.quat_identity()), mass=0.6, label="probe")
    builder.add_shape_sphere(probe, radius=0.1)
    return builder.finalize(device=device), root, probe


def test_start_of_posed_bodies_is_checked_on_the_first_step(test, device):
    """The start of a pair with a kinematic body, whose pose the caller writes, is checked on the first step's
    state_in, not on the model's state. Newton's kinematic-root pendulum, whose probe sphere crosses the root's box in
    the model and is 0.43 m clear of it once the caller turns the root to -1 rad for the first step, runs, and the
    turning box pushes the probe without crossing it. A kinematic box that the first step's state_in puts through a
    body is refused by that step, by name, before it runs. One that crosses the body in the model is refused also
    when the first step is captured, which cannot read its state_in: the capture raises before anything is captured,
    and a first step outside capture from a clear state then runs, and the steps captured after it."""

    def clearance(q):
        """The probe sphere's distance from the root's box less its radius, from the bodies' poses ``q``."""
        root_pose = wp.transform(wp.vec3(*q[root, :3]), wp.quat(*q[root, 3:]))
        local = np.array(wp.transform_point(wp.transform_inverse(root_pose), wp.vec3(*q[probe, :3])))
        d = np.abs(local - (0.6, 0.0, 0.0)) - (0.15, 0.08, 0.08)
        return float(np.linalg.norm(np.maximum(d, 0.0)) + min(d.max(), 0.0)) - 0.1

    model, root, probe = _kinematic_root_pendulum(device)
    test.assertLess(clearance(model.body_q.numpy()), -0.1)
    solver = newton.solvers.SolverIPC(model)  # refused before: the probe crosses the root's box in the model
    dt, omega = 1.0 / 240.0, 2.0
    s0, s1 = model.state(), model.state()
    start = s0.body_q.numpy()[probe, :3].copy()
    speed, closest = 0.0, np.inf
    for k in range(120):  # the test's prescribed motion: the root turns from -1 rad at 2 rad/s
        q, qd = s0.body_q.numpy(), s0.body_qd.numpy()
        q[root] = [0.0, 0.0, 0.0, *wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -1.0 + omega * k * dt)]
        qd[root] = [0.0, 0.0, 0.0, 0.0, omega, 0.0]
        s0.body_q.assign(q)
        s0.body_qd.assign(qd)
        if k == 0:
            test.assertGreater(clearance(q), 0.4)
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        speed = max(speed, float(np.linalg.norm(s0.body_qd.numpy()[probe, :3])))
        closest = min(closest, clearance(s0.body_q.numpy()))
    test.assertGreater(speed, 2e-2)
    test.assertGreater(float(np.linalg.norm(s0.body_q.numpy()[probe, :3] - start)), 1e-2)
    # the box reaches the probe and stays out of it, up to the sagitta of the triangulated sphere (8.6e-4 m)
    test.assertLess(closest, 0.01)
    test.assertGreater(closest, -1e-3)

    def boxes(x):
        """A box body at the origin and a kinematic box ('pusher') of its size at ``x`` along x."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        _box_body(builder, (0.0, 0.0, 1.0), (0.1, 0.1, 0.1))
        kinematic = builder.add_body(
            xform=wp.transform(wp.vec3(x, 0.0, 1.0), wp.quat_identity()), is_kinematic=True, label="pusher"
        )
        builder.add_shape_box(kinematic, hx=0.1, hy=0.1, hz=0.1)
        return builder.finalize(device=device), kinematic

    def placed(model, kinematic, x):
        state = model.state()
        q = state.body_q.numpy()
        q[kinematic, 0] = x
        state.body_q.assign(q)
        return state

    # clear in the model, through the body in the first step's state_in
    model, kinematic = boxes(1.0)
    solver = newton.solvers.SolverIPC(model)
    with test.assertRaisesRegex(ValueError, r"body 0 and body 1 \('pusher'\) intersect in the state of the first step"):
        solver.step(placed(model, kinematic, 0.05), model.state(), None, None, 0.01)
    test.assertEqual(solver._engine.frames, 0)
    solver.step(model.state(), model.state(), None, None, 0.01)  # the first step again, from the model's clear state
    test.assertEqual(solver._engine.frames, 1)
    # through the body in the model: not refused at construction, since the caller writes the pusher's pose
    model, kinematic = boxes(0.05)
    solver = newton.solvers.SolverIPC(model)
    s0, s1 = placed(model, kinematic, 1.0), model.state()
    with test.assertRaisesRegex(ValueError, r"intersect in the model's state.*The first step is captured"):
        with wp.ScopedCapture(device=device):
            solver.step(s0, s1, None, None, 0.01)
    test.assertEqual(solver._engine.frames, 0)
    with test.assertRaisesRegex(ValueError, r"intersect in the state of the first step"):
        solver.step(model.state(), s1, None, None, 0.01)
    solver.step(s0, s1, None, None, 0.01)
    with wp.ScopedCapture(device=device) as capture:
        solver.step(s1, s0, None, None, 0.01)
    wp.capture_launch(capture.graph)
    test.assertEqual(solver._engine.frames, 2)


def test_external_forces_match_closed_form(test, device):
    """Accelerate a particle with particle_f and two free boxes with body_f and with a free joint's joint_f.

    Without gravity, a constant force F and torque tau give implicit Euler's x_k = x_0 + a dt^2 k (k + 1) / 2 for
    the centre of mass (a = F / m) and the same law for the angle about a principal axis (alpha = tau / I_zz), from
    rest. The boxes' shapes sit off their body origins, so the force also exercises the centre-of-mass offset.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    particle = builder.add_particle(wp.vec3(0.0, 1.0, 1.0), wp.vec3(0.0, 0.0, 0.0), 0.5)
    bodies = []
    for y in (0.0, -1.0):
        b = builder.add_body(xform=wp.transform(wp.vec3(0.0, y, 1.0), wp.quat_identity()))
        builder.add_shape_box(
            b, xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()), hx=0.1, hy=0.05, hz=0.08
        )
        bodies.append(b)
    model = builder.finalize(device=device)
    # libuipc's body stiffness: the closed forms are rigid ones, and with the default stiffness a box of unequal sides
    # that a torque starts turning runs 7e-4 of its turn ahead of a rigid one after a step (docs/solvers/ipc.rst > "Rigid # bodies")
    solver = newton.solvers.SolverIPC(
        model, newton_velocity_tol=1e-6, linear_tol_rate=1e-12, semi_implicit=False, body_kappa=1.0e8
    )
    m = model.body_mass.numpy()[bodies[0]]
    I_zz = model.body_inertia.numpy()[bodies[0]][2, 2]
    com = model.body_com.numpy()[bodies[0]].astype(np.float64)
    force, torque = np.array([m, 0.5 * m, 0.0]), np.array([0.0, 0.0, I_zz])  # 1 and 0.5 m/s^2, 1 rad/s^2
    f_particle = np.array([0.0, 0.0, 1.0])  # 2 m/s^2
    s0, s1 = model.state(), model.state()
    control = model.control()
    wrench = np.concatenate([force, torque]).astype(np.float32)
    joint_f = control.joint_f.numpy()
    joint_f[model.joint_qd_start.numpy()[1] : model.joint_qd_start.numpy()[1] + 6] = wrench
    control.joint_f.assign(joint_f)
    observables = solver.observables({solver.ObservableKind.STATUS, solver.ObservableKind.SOLVER_RESIDUAL})
    x0 = model.particle_q.numpy()[particle].astype(np.float64)
    q0 = model.body_q.numpy().astype(np.float64)
    dt = 0.01
    for k in range(1, 11):
        for s in (s0, s1):
            body_f = np.zeros((model.body_count, 6), np.float32)
            body_f[bodies[0]] = wrench
            s.body_f.assign(body_f)
            s.particle_f.assign(np.array([f_particle], np.float32))
        solver.step(s0, s1, control, None, dt, observables=observables)
        s0, s1 = s1, s0
        test.assertEqual(observables.status.numpy().tolist(), [0, 0])  # the global entry, and world 0 unused
        residual = float(observables.solver_residual.numpy()[-1])
        test.assertGreater(residual, 0.0)
        test.assertLessEqual(residual, 1.0)
        ramp = dt * dt * k * (k + 1) / 2.0
        np.testing.assert_allclose(s0.particle_q.numpy()[particle], x0 + 2.0 * f_particle * ramp, atol=1e-6)
        np.testing.assert_allclose(s0.particle_qd.numpy()[particle], 2.0 * f_particle * k * dt, atol=1e-6)
        q, qd = s0.body_q.numpy(), s0.body_qd.numpy()
        for b in bodies:
            angle = 2.0 * np.arctan2(q[b, 5], q[b, 6])
            R = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
            x_com = q[b, :3] + R @ com
            np.testing.assert_allclose(x_com, q0[b, :3] + com + force / m * ramp, atol=1e-6)
            np.testing.assert_allclose(qd[b, :3], force / m * k * dt, atol=1e-6)
            # the affine body is near-rigid: its rotation is within 1e-5 of the rigid one (measured)
            test.assertAlmostEqual(angle, ramp, delta=1e-4 * ramp)
            np.testing.assert_allclose(qd[b, 3:], [0.0, 0.0, k * dt], atol=1e-4 * k * dt)


def test_free_joint_force_pushes_the_parent(test, device):
    """joint_f on the free joint of a body whose parent is another body, without gravity: the child is pushed by the
    force, as implicit Euler from rest gives it, and the parent by the reaction, so that the pair's momentum stays 0
    (as SolverXPBD applies it)."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0), up_axis=newton.Axis.Y)
    parent = builder.add_link(xform=wp.transform((1.0, 0.0, 0.0), wp.quat_identity()))
    builder.add_shape_box(parent, hx=0.1, hy=0.1, hz=0.1)
    child = builder.add_link(xform=wp.transform((1.5, 0.0, 0.0), wp.quat_identity()))
    builder.add_shape_box(child, hx=0.1, hy=0.1, hz=0.1)
    builder.add_articulation([builder.add_joint_free(parent), builder.add_joint_free(child, parent=parent)])
    model = builder.finalize(device=device)
    control = model.control()
    f = np.zeros(model.joint_dof_count, np.float32)
    f[6:9] = (0.0, 2.0, 0.0)  # the force of the child's free joint, at its centre of mass
    control.joint_f.assign(f)
    h, steps = 0.01, 20
    qd = []
    s0, s1 = model.state(), model.state()
    solver = newton.solvers.SolverIPC(model, newton_velocity_tol=1e-6, linear_tol_rate=1e-12, semi_implicit=False)
    for _ in range(steps):
        solver.step(s0, s1, control, None, h)
        s0, s1 = s1, s0
        qd.append(s0.body_qd.numpy().copy())
    qd = np.array(qd)
    mass = model.body_mass.numpy()
    momentum = mass[parent] * qd[:, parent, 0:3] + mass[child] * qd[:, child, 0:3]
    np.testing.assert_allclose(momentum, 0.0, atol=1e-6)
    n = np.arange(1, steps + 1)
    np.testing.assert_allclose(qd[:, child, 1], 2.0 * n * h / mass[child], rtol=1e-5)


@requires("uipc_warp")
class TestSolverIPCInputs(unittest.TestCase):
    """What the solver takes from a model, a state and a control, and what it rejects."""

    def test_cpu_device(self):
        """Reject a model on the CPU: the engine runs on CUDA devices only."""
        model = _two_particles().finalize(device="cpu")
        with self.assertRaisesRegex(NotImplementedError, "CUDA"):
            newton.solvers.SolverIPC(model)

    def test_velocity_limit_bound_is_newtons_default(self):
        """The velocity limit below which SolverIPC warns of one, which it does not read, is Newton's default."""
        from newton._src.solvers.ipc.joints import _VELOCITY_LIMIT  # noqa: PLC0415

        self.assertEqual(_VELOCITY_LIMIT, newton.ModelBuilder.JointDofConfig().velocity_limit)

    def test_equality_constants_are_newtons(self):
        """SolverIPC's copies of MuJoCo's CONNECT and WELD equality types and of Newton's target kind of a row turned
        into a loop joint, which have no public name, are Newton's."""
        from newton._src.solvers.ipc import solver_ipc  # noqa: PLC0415
        from newton._src.solvers.mujoco.enums import EqType  # noqa: PLC0415
        from newton._src.solvers.mujoco.equality import MjcEqualityTargetKind  # noqa: PLC0415

        self.assertEqual(
            (solver_ipc._EQ_CONNECT, solver_ipc._EQ_WELD, solver_ipc._EQ_TARGET_JOINT),
            (int(EqType.CONNECT), int(EqType.WELD), int(MjcEqualityTargetKind.JOINT)),
        )

    def test_imports_only_public_newton_names(self):
        """SolverIPC's modules import Newton through its public modules only: no import of theirs resolves under
        newton._src outside SolverIPC's own package."""
        module = importlib.import_module(newton.solvers.SolverIPC.__module__)
        package = module.__package__
        private = []
        for path in sorted(pathlib.Path(module.__file__).parent.glob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [importlib.util.resolve_name("." * node.level + (node.module or ""), package)]
                else:
                    continue
                for name in names:
                    own = name == package or name.startswith(package + ".")
                    if (name == "newton._src" or name.startswith("newton._src.")) and not own:
                        private.append(f"{path.name}:{node.lineno} {name}")
        self.assertEqual(private, [])

    def test_rest_shape_copies_match_newtons_builders(self):
        """SolverIPC's copies of Newton's rest-shape helpers, which have no public name, reproduce bit for bit the
        tri_poses, tri_areas and edge_rest_angle that ModelBuilder stores, computed from particle_q as the solver
        computes them to tell a cloth built at rest: random sheets from add_cloth_grid, from add_cloth_mesh (bent)
        and from add_triangles with add_edges (random particles, the hinge rows in both orders)."""
        from newton._src.solvers.ipc.scene import _dihedral_angles, _triangle_rest_geometry  # noqa: PLC0415

        rng = np.random.default_rng(3)

        def rotation():
            axis = rng.normal(size=3)
            return wp.quat_from_axis_angle(wp.vec3(*(axis / np.linalg.norm(axis))), float(rng.uniform(0.0, 6.0)))

        bent = 0.0
        for trial in range(12):
            builder = newton.ModelBuilder()
            kind = ("grid", "mesh", "triangles")[trial % 3]
            if kind == "grid":
                builder.add_cloth_grid(
                    pos=wp.vec3(*rng.uniform(-1.0, 1.0, 3)),
                    rot=rotation(),
                    vel=wp.vec3(0.0, 0.0, 0.0),
                    dim_x=int(rng.integers(2, 8)),
                    dim_y=int(rng.integers(2, 8)),
                    cell_x=float(rng.uniform(0.005, 0.2)),
                    cell_y=float(rng.uniform(0.005, 0.2)),
                    mass=0.1,
                )
            elif kind == "mesh":
                nx, ny = (int(k) for k in rng.integers(3, 7, size=2))
                u, v = np.meshgrid(np.linspace(0.0, 0.4, nx), np.linspace(0.0, 0.3, ny), indexing="ij")
                vertices = np.stack([u, v, 0.05 * rng.normal(size=u.shape)], axis=-1).reshape(-1, 3)
                corners = [(ny * i + j, ny * (i + 1) + j) for i in range(nx - 1) for j in range(ny - 1)]
                builder.add_cloth_mesh(
                    pos=wp.vec3(*rng.uniform(-1.0, 1.0, 3)),
                    rot=rotation(),
                    scale=float(rng.uniform(0.5, 2.0)),
                    vel=wp.vec3(0.0, 0.0, 0.0),
                    vertices=[wp.vec3(*p) for p in vertices],
                    indices=[k for a, b in corners for k in (a, b, b + 1, a, b + 1, a + 1)],
                    density=0.1,
                )
            else:
                # a fan of triangles about particle 0, at float32 positions, its hinges given in both orders
                n = int(rng.integers(4, 9))
                x = rng.uniform(-1.0, 1.0, (n + 2, 3)).astype(np.float32).astype(np.float64)
                builder.add_particles(x.tolist(), [(0.0, 0.0, 0.0)] * (n + 2), [0.1] * (n + 2))
                tris = np.array([(0, k, k + 1) for k in range(1, n + 1)])
                builder.add_triangles(tris[:, 0], tris[:, 1], tris[:, 2])
                rows = np.array([(k - 1, k + 1, 0, k) if k % 2 else (k + 1, k - 1, k, 0) for k in range(2, n + 1)])
                builder.add_edges(rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3])
            model = builder.finalize(device="cpu")
            x = model.particle_q.numpy().astype(np.float64)
            t = model.tri_indices.numpy().reshape(-1, 3)
            edges, area = _triangle_rest_geometry(x[t[:, 1]] - x[t[:, 0]], x[t[:, 2]] - x[t[:, 0]])
            np.testing.assert_array_equal(np.linalg.inv(edges).astype(np.float32), model.tri_poses.numpy(), kind)
            np.testing.assert_array_equal(area.astype(np.float32), model.tri_areas.numpy(), kind)
            rows = model.edge_indices.numpy().reshape(-1, 4)
            hinge = (rows[:, 0] >= 0) & (rows[:, 1] >= 0)
            e = rows[hinge]
            angles = _dihedral_angles(x[e[:, 0]], x[e[:, 1]], x[e[:, 2]], x[e[:, 3]], x[e[:, 2]], x[e[:, 3]])
            np.testing.assert_array_equal(angles.astype(np.float32), model.edge_rest_angle.numpy()[hinge], kind)
            if kind != "grid":
                bent = max(bent, float(np.abs(angles).max()))
        self.assertGreater(bent, 1.0)  # folded hinges too, not only flat ones

    def test_import_checks_the_version_and_the_capabilities(self):
        """Refuse a libuipc-warp older than the release SolverIPC is written for, naming it, and one without a
        capability SolverIPC calls (the rigid flight, AL-IPC's sweep of kinematic moves, the frame restart), naming
        what it lacks; take a development build of the release. Every refusal says where to install it from."""
        import uipc_warp  # noqa: PLC0415

        imported = newton.solvers.SolverIPC._uipc_warp

        def refused(pattern):
            newton.solvers.SolverIPC._uipc_warp = None
            with self.assertRaisesRegex(ImportError, pattern + r".*pip install git\+https://github\.com/OrangeEarth15"):
                newton.solvers.SolverIPC._import_uipc_warp()

        names = uipc_warp.EngineOptions.names()
        try:
            for version in ("0.4.4.dev0", "0.4.4", "0.5.0", "1.0"):
                with mock.patch.object(uipc_warp, "__version__", version):
                    newton.solvers.SolverIPC._uipc_warp = None
                    self.assertIs(newton.solvers.SolverIPC._import_uipc_warp(), uipc_warp)
            for version in ("0.4.3", "0.3.0"):
                with mock.patch.object(uipc_warp, "__version__", version):
                    refused(f"requires libuipc-warp 0\\.4\\.4 or newer, and the installed one is {version}")
            for option in ("affine_rotation", "al_sweep_kinematic"):
                without = tuple(name for name in names if name != option)
                with mock.patch.object(uipc_warp.EngineOptions, "names", return_value=without):
                    refused(f"lacks EngineOptions\\({option}=\\)")

            class WithoutRestart:  # an engine of a build without the frame restart
                place_kinematic = uipc_warp.SimEngine.place_kinematic

            with mock.patch.object(uipc_warp, "SimEngine", WithoutRestart):
                refused(r"lacks the frame restart \(SimEngine\.enable_frame_restart\)")
        finally:
            newton.solvers.SolverIPC._uipc_warp = imported

    def test_contact_elements_match_shape_contact_pairs(self):
        """On random models (collision groups -2 to 3, worlds -1 and 0, collide flags, friction coefficients, filter
        pairs, shapes on one body), the shape pairs that SolverIPC's contact elements enable on different bodies are
        exactly Model.shape_contact_pairs, with the mean of the two coefficients; particles meet the shapes with
        COLLIDE_PARTICLES."""
        from newton._src.solvers.ipc.contacts import _contact_elements  # noqa: PLC0415
        from newton._src.solvers.ipc.scene import _colliding_shapes  # noqa: PLC0415

        rng = np.random.default_rng(7)
        for trial in range(40):
            builder = newton.ModelBuilder()

            def shape(body, builder=builder):
                cfg = newton.ModelBuilder.ShapeConfig(
                    collision_group=int(rng.choice([-2, -1, 0, 1, 2, 3])),
                    has_shape_collision=bool(rng.random() < 0.8),
                    has_particle_collision=bool(rng.random() < 0.8),
                    mu=float(rng.choice([0.2, 0.5])),
                )
                xform = wp.transform(wp.vec3(*rng.uniform(-1.0, 1.0, 3)), wp.quat_identity())
                return builder.add_shape_box(body, xform=xform, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)

            for local in (False, True):
                if local:
                    builder.begin_world()
                for _ in range(int(rng.integers(0, 3))):
                    shape(-1)
                for _ in range(int(rng.integers(1, 4))):
                    body = builder.add_body(xform=wp.transform(wp.vec3(*rng.uniform(-1.0, 1.0, 3)), wp.quat_identity()))
                    for _ in range(int(rng.integers(1, 3))):
                        shape(body)
                if local:
                    builder.end_world()
            for _ in range(int(rng.integers(0, 4))):
                a, b = rng.choice(builder.shape_count, 2, replace=False)
                builder.add_shape_collision_filter_pair(int(a), int(b))
            if trial % 2:
                builder.add_particle(wp.vec3(0.0, 0.0, 3.0), wp.vec3(0.0, 0.0, 0.0), 1.0)
            model = builder.finalize(device="cpu")
            elements = _contact_elements(model)
            body = model.shape_body.numpy()
            mu = model.shape_material_mu.numpy().astype(np.float64)
            shapes = np.flatnonzero(_colliding_shapes(model))
            enabled = set()
            for i in shapes:
                for j in shapes[shapes > i]:
                    e, f = elements.shape[i], elements.shape[j]
                    if body[i] != body[j] and elements.enabled[e, f]:
                        enabled.add((int(i), int(j)))
                        self.assertAlmostEqual(elements.mu[e, f], 0.5 * (mu[i] + mu[j]), places=7)
            listed = model.shape_contact_pairs
            listed = set() if listed is None else {tuple(sorted(map(int, p))) for p in listed.numpy()}
            self.assertEqual(enabled, listed, f"trial {trial}")
            if model.particle_count:
                flags = model.shape_flags.numpy()
                p = elements.particle[0]
                for i in shapes:
                    self.assertEqual(
                        bool(elements.enabled[p, elements.shape[i]]),
                        bool(flags[i] & int(newton.ShapeFlags.COLLIDE_PARTICLES)),
                    )

    def test_contact_elements_with_kinematic_aims(self):
        """With kinematic_motion="aim" (``aims``), on random models of static shapes, dynamic and kinematic bodies: the
        pairs the contact elements enable on different bodies are Model.shape_contact_pairs less the pairs of two
        shapes that are each static or kinematic, and without the option the elements of before."""
        from newton._src.solvers.ipc.contacts import _contact_elements  # noqa: PLC0415
        from newton._src.solvers.ipc.scene import _colliding_shapes  # noqa: PLC0415

        rng = np.random.default_rng(11)
        for trial in range(30):
            builder = newton.ModelBuilder()
            for _ in range(int(rng.integers(0, 3))):
                xform = wp.transform(wp.vec3(*rng.uniform(-1.0, 1.0, 3)), wp.quat_identity())
                builder.add_shape_box(-1, xform=xform, hx=0.1, hy=0.1, hz=0.1)
            for _ in range(int(rng.integers(1, 5))):
                kinematic = bool(rng.random() < 0.5)
                xform = wp.transform(wp.vec3(*rng.uniform(-1.0, 1.0, 3)), wp.quat_identity())
                body = builder.add_body(xform=xform, is_kinematic=kinematic)
                cfg = newton.ModelBuilder.ShapeConfig(collision_group=int(rng.choice([-1, 1, 2])))
                builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
            model = builder.finalize(device="cpu")
            elements = _contact_elements(model, aims=True)
            body = model.shape_body.numpy()
            immovable = (body < 0) | np.isin(
                body, np.flatnonzero(model.body_flags.numpy() & int(newton.BodyFlags.KINEMATIC))
            )
            shapes = np.flatnonzero(_colliding_shapes(model))
            enabled = set()
            for i in shapes:
                for j in shapes[shapes > i]:
                    if body[i] != body[j] and elements.enabled[elements.shape[i], elements.shape[j]]:
                        enabled.add((int(i), int(j)))
            listed = model.shape_contact_pairs
            listed = set() if listed is None else {tuple(sorted(map(int, p))) for p in listed.numpy()}
            self.assertEqual(
                enabled, {p for p in listed if not (immovable[p[0]] and immovable[p[1]])}, f"trial {trial}"
            )

    def test_contact_elements_of_coupled_worlds(self):
        """On random models of global shapes and two to four local worlds, some replicated with their filter pairs and
        some drawn on their own, the shape pairs that the contact elements enable on different bodies, in the same
        world or with a global shape (the subscenes), are exactly Model.shape_contact_pairs; replicated worlds add no
        contact element."""
        from newton._src.solvers.ipc.contacts import _contact_elements  # noqa: PLC0415
        from newton._src.solvers.ipc.scene import _colliding_shapes, _Worlds  # noqa: PLC0415

        rng = np.random.default_rng(11)

        def content(builder):
            """Bodies of random shapes and filter pairs among them."""
            first = builder.shape_count
            for _ in range(int(rng.integers(1, 4))):
                body = builder.add_body(xform=wp.transform(wp.vec3(*rng.uniform(-1.0, 1.0, 3)), wp.quat_identity()))
                for _ in range(int(rng.integers(1, 3))):
                    cfg = newton.ModelBuilder.ShapeConfig(
                        collision_group=int(rng.choice([-2, -1, 0, 1, 2])),
                        has_shape_collision=bool(rng.random() < 0.85),
                        mu=float(rng.choice([0.2, 0.5])),
                    )
                    xform = wp.transform(wp.vec3(*rng.uniform(-0.2, 0.2, 3)), wp.quat_identity())
                    builder.add_shape_box(body, xform=xform, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
            for _ in range(int(rng.integers(0, 3))):
                a, b = rng.choice(np.arange(first, builder.shape_count), 2)
                if a != b:
                    builder.add_shape_collision_filter_pair(int(a), int(b))

        for trial in range(30):
            builder = newton.ModelBuilder()
            for _ in range(int(rng.integers(1, 3))):
                cfg = newton.ModelBuilder.ShapeConfig(collision_group=int(rng.choice([-1, 1, 2])))
                g = builder.add_shape_box(
                    -1,
                    xform=wp.transform(wp.vec3(*rng.uniform(-1, 1, 3)), wp.quat_identity()),
                    hx=0.3,
                    hy=0.3,
                    hz=0.05,
                    cfg=cfg,
                )
            template = newton.ModelBuilder()
            content(template)
            copies = int(rng.integers(2, 4))
            builder.replicate(template, copies)
            for _ in range(int(rng.integers(0, 2))):
                builder.begin_world()
                content(builder)
                builder.end_world()
            if trial % 3 == 0:  # the last shape of every copy filtered against a global one
                shape_world = np.asarray(builder.shape_world)
                for k in range(copies):
                    builder.add_shape_collision_filter_pair(g, int(np.flatnonzero(shape_world == k)[-1]))
            model = builder.finalize(device="cpu")
            worlds = _Worlds.of(model)
            self.assertIsNotNone(worlds)
            elements = _contact_elements(model, worlds)
            body, world = model.shape_body.numpy(), model.shape_world.numpy()
            shapes = np.flatnonzero(_colliding_shapes(model))
            enabled = set()
            for i in shapes:
                for j in shapes[shapes > i]:
                    e, f = elements.shape[i], elements.shape[j]
                    together = world[i] == world[j] or world[i] == -1 or world[j] == -1
                    if body[i] != body[j] and together and elements.enabled[e, f]:
                        enabled.add((int(i), int(j)))
            listed = model.shape_contact_pairs
            listed = set() if listed is None else {tuple(sorted(map(int, p))) for p in listed.numpy()}
            self.assertEqual(enabled, listed, f"trial {trial}")
            replicated = world[shapes] < copies  # the replicated worlds come first
            local = world[shapes] >= 0
            used = np.unique(elements.shape[shapes[local & replicated]])
            first = np.unique(elements.shape[shapes[world[shapes] == 0]])
            self.assertEqual(used.tolist(), first.tolist(), f"trial {trial}: the copies share the elements of world 0")

    def test_contact_elements_of_proxies(self):
        """In a view with proxies, as a coupled solver gives an entry, on random models of static shapes, dynamic and
        kinematic bodies, free particles and a cloth: the pairs the contact elements enable on different bodies are
        Model.shape_contact_pairs less those of a proxy shape with another proxy shape, a static shape or a shape on a
        body of inverse mass 0 (another entry's); a proxy particle meets the shapes a particle of its kind meets but
        those, the entry's own particles as a particle of its kind does, and no other proxy particle."""
        from newton._src.solvers.ipc.contacts import _contact_elements  # noqa: PLC0415
        from newton._src.solvers.ipc.scene import _colliding_shapes  # noqa: PLC0415
        from newton.solvers.experimental.coupled import ModelView  # noqa: PLC0415

        rng = np.random.default_rng(13)
        for trial in range(30):
            builder = newton.ModelBuilder()
            for _ in range(int(rng.integers(1, 3))):
                xform = wp.transform(wp.vec3(*rng.uniform(-1.0, 1.0, 3)), wp.quat_identity())
                builder.add_shape_box(-1, xform=xform, hx=0.1, hy=0.1, hz=0.1)
            bodies = []
            for _ in range(int(rng.integers(3, 7))):
                xform = wp.transform(wp.vec3(*rng.uniform(-1.0, 1.0, 3)), wp.quat_identity())
                body = builder.add_body(xform=xform, is_kinematic=bool(rng.random() < 0.2))
                cfg = newton.ModelBuilder.ShapeConfig(collision_group=int(rng.choice([-1, 1, 2])))
                builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
                bodies.append(body)
            for _ in range(3):
                builder.add_particle(wp.vec3(*rng.uniform(-1.0, 1.0, 3)), wp.vec3(0.0, 0.0, 0.0), 0.1, radius=0.02)
            _cloth(builder, 2, 0.1, 2.0)
            model = builder.finalize(device="cpu")
            view = ModelView(model, "entry")
            shuffled = rng.permutation(bodies)
            proxies, others = shuffled[: int(rng.integers(1, 3))], shuffled[3:4]
            view.mark_proxy_bodies(wp.array(proxies, dtype=int, device="cpu"))
            if len(others):
                view.disable_body_dynamics(wp.array(others, dtype=int, device="cpu"))
            proxy_particles = rng.choice(model.particle_count, 3, replace=False)
            view.mark_proxy_particles(wp.array(proxy_particles, dtype=int, device="cpu"))
            elements = _contact_elements(view)
            body = model.shape_body.numpy()
            role = np.where(body < 0, 2, 0)
            role[np.isin(body, others)] = 2
            role[np.isin(body, proxies)] = 1
            shapes = np.flatnonzero(_colliding_shapes(model))
            enabled = set()
            for i in shapes:
                for j in shapes[shapes > i]:
                    if body[i] != body[j] and elements.enabled[elements.shape[i], elements.shape[j]]:
                        enabled.add((int(i), int(j)))
            listed = {tuple(sorted(map(int, p))) for p in model.shape_contact_pairs.numpy()}
            kept = {p for p in listed if not (1 in (role[p[0]], role[p[1]]) and min(role[p[0]], role[p[1]]) >= 1)}
            self.assertEqual(enabled, kept, f"trial {trial}")
            proxy = np.zeros(model.particle_count, bool)
            proxy[proxy_particles] = True
            own = elements.particle[~proxy]
            for p in proxy_particles:
                e = elements.particle[p]
                kind = 2 if elements.on_mesh[p] else 1  # the element of a particle of its kind: Q_surf or Q_free
                same = np.flatnonzero((elements.on_mesh == elements.on_mesh[p]) & ~proxy)
                self.assertEqual(e, elements.enabled.shape[0] - 3 + kind, f"trial {trial}")
                for i in shapes:
                    meets = (
                        bool(model.shape_flags.numpy()[i] & int(newton.ShapeFlags.COLLIDE_PARTICLES)) and role[i] == 0
                    )
                    self.assertEqual(bool(elements.enabled[e, elements.shape[i]]), meets, f"trial {trial}")
                for q in own:  # the entry's own particles as a particle of its kind meets them
                    if len(same):
                        self.assertEqual(elements.enabled[e, q], elements.enabled[elements.particle[same[0]], q])
                self.assertFalse(elements.enabled[e, elements.particle[proxy]].any(), f"trial {trial}")


devices = get_cuda_test_devices()
for name, func in (
    ("test_rejects_springs_without_a_libuipc_form", test_rejects_springs_without_a_libuipc_form),
    ("test_rejects_attachment_damping", test_rejects_attachment_damping),
    ("test_rejects_mimic_constraints", test_rejects_mimic_constraints),
    ("test_rejects_equality_constraints", test_rejects_equality_constraints),
    ("test_equality_rows_held_by_loop_joints", test_equality_rows_held_by_loop_joints),
    ("test_rejects_tendons", test_rejects_tendons),
    ("test_rejects_muscles", test_rejects_muscles),
    ("test_rejects_activations", test_rejects_activations),
    ("test_control_activations_refused_at_eager_steps", test_control_activations_refused_at_eager_steps),
    ("test_rejects_unsupported_joints", test_rejects_unsupported_joints),
    ("test_rejects_gradients", test_rejects_gradients),
    ("test_warns_of_velocity_limits", test_warns_of_velocity_limits),
    ("test_rejects_surface_velocity", test_rejects_surface_velocity),
    ("test_rejects_intersecting_start", test_rejects_intersecting_start),
    ("test_rejects_start_in_contact", test_rejects_start_in_contact),
    (
        "test_start_of_posed_bodies_is_checked_on_the_first_step",
        test_start_of_posed_bodies_is_checked_on_the_first_step,
    ),
    ("test_external_forces_match_closed_form", test_external_forces_match_closed_form),
    ("test_free_joint_force_pushes_the_parent", test_free_joint_force_pushes_the_parent),
):
    add_function_test(TestSolverIPCInputs, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
