# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The paths of SolverIPC's ``notify_model_changed`` that build the solve again, and those that check the model
again: after the change the next steps equal those of a new solver on the changed model, from the same state, and
leave those of the unchanged one (or the refusal is raised)."""

import re
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.ipc._fixtures import _box_body, _model_of, _with_attachment, _with_revolute, _with_spring
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires

# converged steps, so that two solves of one step agree to rounding
_OPTIONS = {"newton_velocity_tol": 1.0e-5, "linear_tol_rate": 1.0e-12, "semi_implicit": False}
_DT = 0.01


def _clone(model, state):
    """A new state of ``model`` holding the particles', bodies' and joints' state of ``state``."""
    out = model.state()
    for name in ("particle_q", "particle_qd", "body_q", "body_qd", "joint_q", "joint_qd"):
        a = getattr(state, name, None)
        if a is not None and a.shape[0]:
            getattr(out, name).assign(a)
    return out


def _run(model, solver, state, steps):
    """``steps`` steps from a copy of ``state``: the particles' and bodies' positions after each (one array per step,
    particles then bodies, positions and quaternions), and the last state."""
    s0, s1 = _clone(model, state), model.state()
    out = []
    for _ in range(steps):
        solver.step(s0, s1, None, None, _DT)
        s0, s1 = s1, s0
        out.append(np.concatenate([a.numpy().reshape(-1) for a in (s0.particle_q, s0.body_q) if a is not None]))
    return np.array(out, np.float64), s0


def _check_rebuilds(test, build, change, flags, before=2, after=13, atol=1.0e-6, moved=1.0e-4, options=_OPTIONS):
    """For each ``change`` (name: function of the model, changing it in place): a solver steps ``before`` steps, the
    model changes, ``notify_model_changed(flags[name])`` builds the solve again, and the solver's next ``after`` steps
    equal those of a new solver on the changed model from the same state, within ``atol``, where they leave the
    steps of the unchanged model by more than ``moved``."""
    model = build()
    unchanged, _ = _run(model, newton.solvers.SolverIPC(model, **options), model.state(), before + after)
    for name, apply in change.items():
        with test.subTest(change=name):
            model = build()
            solver = newton.solvers.SolverIPC(model, **options)
            head, state = _run(model, solver, model.state(), before)
            np.testing.assert_allclose(head, unchanged[:before], rtol=0.0, atol=atol)
            engine = solver._engine
            apply(model)
            solver.notify_model_changed(flags[name])
            test.assertIsNot(solver._engine, engine)
            notified, _ = _run(model, solver, state, after)
            fresh, _ = _run(model, newton.solvers.SolverIPC(model, **options), state, after)
            np.testing.assert_allclose(notified, fresh, rtol=0.0, atol=atol)
            test.assertGreater(np.abs(fresh - unchanged[before:]).max(), moved)


def _set(array_name, index, value, column=None):
    """A change of the model: ``getattr(model, array_name)[index] = value`` (of ``column``), in place."""

    def apply(model):
        array = getattr(model, array_name)
        values = array.numpy().copy()
        if column is None:
            values[index] = value
        else:
            values[index, column] = value
        array.assign(values)

    return apply


def _mu(shape, value):
    """A new friction coefficient of ``shape``, static and dynamic alike, as Newton's ShapeConfig sets them."""

    def apply(model):
        for name in ("shape_material_mu", "shape_material_mu_static"):
            _set(name, shape, value)(model)

    return apply


def _sliding_box(device):
    """A box (shape 1) 2 cm above a static box (shape 0, its top at z = 0.1), moving along x at 0.5 m/s: it lands
    after about six steps of 0.01 s and slides to rest by friction. It is off the static box's centre and turned by
    0.1 rad about z, so that none of its corners lies on the diagonal of the face it lands on, where rounding would
    decide between point-triangle and point-edge terms (see test_coupled_worlds_with_attachments)."""
    builder = newton.ModelBuilder()
    builder.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()), hx=0.3, hy=0.3, hz=0.05)
    _box_body(
        builder, (0.013, 0.031, 0.17), (0.05, 0.05, 0.05), rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.1)
    )
    model = builder.finalize(device=device)
    model.body_qd.assign(np.array([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0]], np.float32))
    return model


def test_notify_rebuilds_for_new_shapes(test, device):
    """Rebuild the solve when SHAPE_PROPERTIES or SHAPE_TYPE_AND_SOURCE report changed shapes.

    SHAPE_PROPERTIES with a new shape transform (the static box 5 mm lower), scale (the sliding box 5 mm thinner),
    friction coefficient (the sliding box's 0.1, the pair's mean 0.55 instead of 1) or margin (the static box's 3 mm),
    and SHAPE_TYPE_AND_SOURCE with a new type (the sliding box a sphere of its half width, which rolls), build the solve
    again after two steps: the box lands and slides as a new solver on the changed model has it. SHAPE_PROPERTIES with
    nothing changed keeps the solve.
    """
    model = _sliding_box(device)
    solver = newton.solvers.SolverIPC(model, **_OPTIONS)
    engine = solver._engine
    solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
    test.assertIs(solver._engine, engine)

    def sphere(model):
        _set("shape_type", 1, int(newton.GeoType.SPHERE))(model)

    shape, source = newton.ModelFlags.SHAPE_PROPERTIES, newton.ModelFlags.SHAPE_TYPE_AND_SOURCE
    _check_rebuilds(
        test,
        lambda: _sliding_box(device),
        {
            "shape_transform": _set("shape_transform", 0, 0.045, column=2),
            "shape_scale": _set("shape_scale", 1, 0.045, column=2),
            "shape_material_mu": _mu(1, 0.1),
            "shape_margin": _set("shape_margin", 0, 0.003),
            "shape_type": sphere,
        },
        {"shape_transform": shape, "shape_scale": shape, "shape_material_mu": shape, "shape_margin": shape,
         "shape_type": source},
    )  # fmt: skip


def _particle_scene(device):
    """Free particles of radius 1 cm, 2 cm above what they land on, moving along x at 0.5 m/s: one over the ground,
    one over a kinematic particle of radius 0.2 m (particle contact, friction particle_mu); and two particles 10 cm
    apart on a spring, closing at 0.2 m/s."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.add_particle(wp.vec3(0.0, 0.0, 0.03), wp.vec3(0.5, 0.0, 0.0), 0.01, radius=0.01)
    builder.add_particle(wp.vec3(1.0, 0.0, 0.5), wp.vec3(0.0, 0.0, 0.0), 0.0, radius=0.2)
    builder.add_particle(wp.vec3(1.0, 0.0, 0.73), wp.vec3(0.5, 0.0, 0.0), 0.01, radius=0.01)
    a = builder.add_particle(wp.vec3(2.0, 0.0, 1.0), wp.vec3(0.1, 0.0, 0.0), 0.01, radius=0.01)
    b = builder.add_particle(wp.vec3(2.1, 0.0, 1.0), wp.vec3(-0.1, 0.0, 0.0), 0.01, radius=0.01)
    builder.add_spring(a, b, 1.0, 0.0, 0)
    model = builder.finalize(device=device)
    model.soft_contact_mu = 0.5
    model.particle_mu = 0.5
    return model


def test_notify_rebuilds_for_new_particles_and_springs(test, device):
    """Rebuild the solve when MODEL_PROPERTIES reports new contact, particle or spring coefficients.

    MODEL_PROPERTIES with a new soft_contact_mu (0.1), particle_mu (0.1), particle radius (1.5 cm), spring rest length
    (12 cm), spring stiffness (4 N/m) or spring damping (0.5 N s/m) builds the solve again after two steps: the
    particles land, slide and oscillate as a new solver on the changed model has them. A spring given a negative damping
    in place is refused at the notification, as at construction.
    """

    def setter(name, value):
        def apply(model):
            setattr(model, name, value)

        return apply

    flags = dict.fromkeys(
        (
            "soft_contact_mu",
            "particle_mu",
            "particle_radius",
            "spring_rest_length",
            "spring_stiffness",
            "spring_damping",
        ),
        newton.ModelFlags.MODEL_PROPERTIES,
    )
    _check_rebuilds(
        test,
        lambda: _particle_scene(device),
        {
            "soft_contact_mu": setter("soft_contact_mu", 0.1),
            "particle_mu": setter("particle_mu", 0.1),
            "particle_radius": _set("particle_radius", 0, 0.015),
            "spring_rest_length": _set("spring_rest_length", 0, 0.12),
            "spring_stiffness": _set("spring_stiffness", 0, 4.0),
            "spring_damping": _set("spring_damping", 0, 0.5),
        },
        flags,
    )
    model = _particle_scene(device)
    solver = newton.solvers.SolverIPC(model, **_OPTIONS)
    _set("spring_damping", 0, -0.5)(model)
    with test.assertRaisesRegex(NotImplementedError, "spring 0 has spring_damping -0.5; make it 0 or positive"):
        solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)


def _pendulum(device):
    """A sphere of radius 5 cm hanging 0.3 m from a revolute joint about y at (0, 0, 1), released level, and a
    sphere on a distance joint 0.3 m from the world at (1, 0, 1) whose distance is not bounded (bounds -1): it falls."""
    builder = newton.ModelBuilder()
    link = builder.add_link(xform=wp.transform(wp.vec3(0.3, 0.0, 1.0), wp.quat_identity()))
    builder.add_shape_sphere(link, radius=0.05)
    hinge = builder.add_joint_revolute(
        -1, link, axis=wp.vec3(0.0, 1.0, 0.0), parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.3, 0.0, 0.0), wp.quat_identity()),
    )  # fmt: skip
    builder.add_articulation([hinge])
    bob = builder.add_link(xform=wp.transform(wp.vec3(1.3, 0.0, 1.0), wp.quat_identity()))
    builder.add_shape_sphere(bob, radius=0.05)
    rope = builder.add_joint_distance(
        -1, bob, parent_xform=wp.transform(wp.vec3(1.0, 0.0, 1.0), wp.quat_identity()), min_distance=-1.0,
        max_distance=-1.0,
    )  # fmt: skip
    builder.add_articulation([rope])
    return builder.finalize(device=device)


def test_notify_rebuilds_for_new_joints(test, device):
    """Rebuild the solve when a joint's frame or a distance joint's bounds change.

    JOINT_PROPERTIES with a new joint frame (the hinge's pivot 2 mm higher) builds the solve again after two steps, and
    so does a JOINT_DOF_PROPERTIES that gives the free-falling distance joint one length, 0.3 m (its two bounds equal):
    from then on it holds its sphere as a pendulum. In both the next steps equal those of a new solver on the changed
    model. JOINT_PROPERTIES with nothing changed keeps the solve.
    """
    model = _pendulum(device)
    solver = newton.solvers.SolverIPC(model, **_OPTIONS)
    engine = solver._engine
    solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES | newton.ModelFlags.JOINT_DOF_PROPERTIES)
    test.assertIs(solver._engine, engine)

    def pivot(model):
        X_p = model.joint_X_p.numpy().copy()
        X_p[0, 2] += 0.002
        model.joint_X_p.assign(X_p)

    def length(model):
        dof = int(model.joint_qd_start.numpy()[1])
        for name in ("joint_limit_lower", "joint_limit_upper"):
            _set(name, dof, 0.3)(model)

    _check_rebuilds(
        test,
        lambda: _pendulum(device),
        {"joint_X_p": pivot, "distance": length},
        {"joint_X_p": newton.ModelFlags.JOINT_PROPERTIES, "distance": newton.ModelFlags.JOINT_DOF_PROPERTIES},
    )


def test_notify_checks_the_joints_again(test, device):
    """Check the joints again against SolverIPC's refusals at each notification of their degrees of freedom.

    Each of the four flags of the joints' degrees of freedom (JOINT_DOF_PROPERTIES, JOINT_DOF_FORCE_PROPERTIES,
    JOINT_DOF_INERTIAL_PROPERTIES, JOINT_REFERENCE_POSE_PROPERTIES) checks the joints again against what SolverIPC
    rejects: a negative armature given to the hinge in place is refused by name, and once it is 0 again the same
    notification is taken and keeps the solve. The other flags do not check the joints.
    """
    model = _pendulum(device)
    solver = newton.solvers.SolverIPC(model, **_OPTIONS)
    engine = solver._engine
    armature = _set("joint_armature", 0, -0.5)
    for flag in (
        newton.ModelFlags.JOINT_DOF_PROPERTIES,
        newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES,
        newton.ModelFlags.JOINT_DOF_INERTIAL_PROPERTIES,
        newton.ModelFlags.JOINT_REFERENCE_POSE_PROPERTIES,
    ):
        with test.subTest(flag=flag.name):
            armature(model)
            with test.assertRaisesRegex(NotImplementedError, "DOF 0 of the REVOLUTE joint 0 has joint_armature -0.5"):
                solver.notify_model_changed(flag)
            _set("joint_armature", 0, 0.0)(model)
            solver.notify_model_changed(flag)
            test.assertIs(solver._engine, engine)
    armature(model)
    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES | newton.ModelFlags.SHAPE_PROPERTIES)


def test_notify_checks_constraints_attachments_and_springs_again(test, device):
    """Refuse at notify_model_changed the constraint, attachment and spring values that SolverIPC refuses when built.

    CONSTRAINT_PROPERTIES checks the mimic constraints, then the attachments' damping before their stiffness;
    JOINT_DOF_PROPERTIES checks the joints' degrees of freedom; MODEL_PROPERTIES with a new spring damping builds the
    solve again, which refuses a negative one. Each refusal names the item, and a notification after the values are
    valid again is taken."""
    flags = newton.ModelFlags
    model = _model_of(
        device, _with_revolute(), _with_revolute(), _with_attachment(), _with_attachment(), _with_spring()
    )
    solver = newton.solvers.SolverIPC(model)

    def refused(flag, message):
        with test.assertRaisesRegex(NotImplementedError, f"^{re.escape(message)}$"):
            solver.notify_model_changed(flag)

    mimic = model.joint_mimic_joint.numpy()  # the two hinges, and the free joints of the attachments' bodies
    mimic[2] = 0
    model.joint_mimic_joint.assign(mimic)
    refused(
        flags.CONSTRAINT_PROPERTIES,
        "SolverIPC: the follower of the mimic joint 2 is the FREE joint 2, which has more or other coordinates than "
        "one angle or one distance; mimic revolute and prismatic joints (or D6 joints of one axis). SolverIPC holds a "
        "mimic relation of one coordinate.",
    )
    mimic[2] = -1
    model.joint_mimic_joint.assign(mimic)
    model.attachment_particle_body_stiffness.assign(np.array([-1.0, 1.0e4], np.float32))
    model.attachment_particle_body_damping.assign(np.array([0.0, -2.0], np.float32))
    refused(
        flags.CONSTRAINT_PROPERTIES, "SolverIPC: particle-body attachment 1 has damping -2.0; make it 0 or positive."
    )
    model.attachment_particle_body_damping.assign(np.array([0.0, 0.0], np.float32))
    refused(
        flags.CONSTRAINT_PROPERTIES,
        "SolverIPC: particle-body attachment 0 has stiffness -1.0; make it 0 or positive, or infinite for a hard "
        "attachment.",
    )
    model.attachment_particle_body_stiffness.assign(np.array([1.0e4, 1.0e4], np.float32))
    solver.notify_model_changed(flags.CONSTRAINT_PROPERTIES)
    friction = model.joint_friction.numpy()
    friction[5] = 0.5  # the fourth degree of freedom of the free joint 2
    model.joint_friction.assign(friction)
    refused(
        flags.JOINT_DOF_PROPERTIES,
        "SolverIPC: DOF 5 of the FREE joint 2 has joint_friction 0.5; set it to 0. SolverIPC holds the friction of "
        "revolute, prismatic, ball and universal joints only, as a force on one coordinate or a torque about each axis "
        "of a ball joint or each angle of a universal joint.",
    )
    model.joint_friction.assign(np.zeros_like(friction))
    model.spring_damping.assign(np.array([-0.5], np.float32))
    refused(flags.MODEL_PROPERTIES, "SolverIPC: spring 0 has spring_damping -0.5; make it 0 or positive.")


def _falling_box(device):
    """A free box 0.3 m above the ground, falling."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    _box_body(builder, (0.0, 0.0, 0.35), (0.05, 0.05, 0.05))
    return builder.finalize(device=device)


def test_notify_rebuilds_for_another_kinematic_set(test, device):
    """Rebuild the solve when BODY_PROPERTIES makes a body kinematic or dynamic in place.

    BODY_PROPERTIES with new body flags builds the solve again: a falling box made KINEMATIC in place after three steps
    keeps the pose that state_in gives it (kinematic_motion="jump"), and made DYNAMIC again three steps later falls on.
    Each time the next steps equal those of a new solver on the changed model from the same state.
    """
    model = _falling_box(device)
    options = {**_OPTIONS, "kinematic_motion": "jump"}
    solver = newton.solvers.SolverIPC(model, **options)
    state = model.state()
    path, state = _run(model, solver, state, 3)
    for flags, steps in ((newton.BodyFlags.KINEMATIC, 3), (newton.BodyFlags.DYNAMIC, 4)):
        with test.subTest(flags=flags.name):
            engine = solver._engine
            model.body_flags.assign(np.array([int(flags)], np.int32))
            solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)
            test.assertIsNot(solver._engine, engine)
            notified, after = _run(model, solver, state, steps)
            fresh, _ = _run(model, newton.solvers.SolverIPC(model, **options), state, steps)
            np.testing.assert_allclose(notified, fresh, rtol=0.0, atol=1.0e-6)
            z = notified[:, 2]
            if flags == newton.BodyFlags.KINEMATIC:  # held where state_in puts it
                np.testing.assert_allclose(z, np.full(steps, path[-1][2]), rtol=0.0, atol=1.0e-7)
            else:  # falling on, faster at every step
                test.assertTrue(np.all(np.diff(np.concatenate([[path[-1][2]], z]), 2) < 0.0))
            path, state = notified, after


def _lift(device):
    """A kinematic sphere of 1 kg rising at 0.2 m/s, and a free sphere of 1 kg hanging 0.3 m under it on a ball joint,
    whose weight loads the kinematic one."""
    builder = newton.ModelBuilder()
    lift = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), is_kinematic=True, mass=1.0,
        inertia=wp.mat33(np.eye(3) * 1.0e-2),
    )  # fmt: skip
    builder.add_shape_sphere(lift, radius=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    load = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.7), wp.quat_identity()), mass=1.0,
                            inertia=wp.mat33(np.eye(3) * 1.0e-3))  # fmt: skip
    builder.add_shape_sphere(load, radius=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    joints = [
        builder.add_joint_free(lift),
        builder.add_joint_ball(lift, load, parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.3), wp.quat_identity())),
    ]
    builder.add_articulation(joints)
    model = builder.finalize(device=device)
    model.body_qd.assign(np.array([[0.0, 0.0, 0.2, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], np.float32))
    return model


def test_notify_rebuilds_for_new_masses_of_aim_bodies(test, device):
    """Rebuild the solve for a new mass of an aimed kinematic body, and take other new masses in place.

    With kinematic_motion="aim" a kinematic body moves toward its aim with the mass the solve was built with: the rising
    sphere made ten times lighter in place (BODY_INERTIAL_PROPERTIES) lags its aim further under the sphere hanging from
    it, and the solve is built again, the next steps equal to those of a new solver on the changed model. A new mass of
    the hanging sphere alone is taken in place, without building the solve again.
    """
    options = {**_OPTIONS, "kinematic_motion": "aim"}

    def lighter(model):
        for name, factor in (("body_mass", 0.1), ("body_inertia", 0.1), ("body_inv_mass", 10.0),
                             ("body_inv_inertia", 10.0)):  # fmt: skip
            values = getattr(model, name).numpy().copy()
            values[0] *= factor
            getattr(model, name).assign(values)

    _check_rebuilds(
        test,
        lambda: _lift(device),
        {"lift mass": lighter},
        {"lift mass": newton.ModelFlags.BODY_INERTIAL_PROPERTIES},
        options=options,
    )
    model = _lift(device)
    solver = newton.solvers.SolverIPC(model, **options)
    engine = solver._engine
    for name, factor in (("body_mass", 2.0), ("body_inertia", 2.0), ("body_inv_mass", 0.5), ("body_inv_inertia", 0.5)):
        values = getattr(model, name).numpy().copy()
        values[1] *= factor
        getattr(model, name).assign(values)
    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    test.assertIs(solver._engine, engine)
    test.assertAlmostEqual(float(solver._dyadic[0][1]), float(model.body_mass.numpy()[1]), places=4)


devices = get_cuda_test_devices()


@requires("uipc_warp")
class TestSolverIPCNotify(unittest.TestCase):
    pass


for name, func in (
    ("test_notify_rebuilds_for_new_shapes", test_notify_rebuilds_for_new_shapes),
    ("test_notify_rebuilds_for_new_particles_and_springs", test_notify_rebuilds_for_new_particles_and_springs),
    ("test_notify_rebuilds_for_new_joints", test_notify_rebuilds_for_new_joints),
    ("test_notify_checks_the_joints_again", test_notify_checks_the_joints_again),
    (
        "test_notify_checks_constraints_attachments_and_springs_again",
        test_notify_checks_constraints_attachments_and_springs_again,
    ),
    ("test_notify_rebuilds_for_another_kinematic_set", test_notify_rebuilds_for_another_kinematic_set),
    ("test_notify_rebuilds_for_new_masses_of_aim_bodies", test_notify_rebuilds_for_new_masses_of_aim_bodies),
):
    add_function_test(TestSolverIPCNotify, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
