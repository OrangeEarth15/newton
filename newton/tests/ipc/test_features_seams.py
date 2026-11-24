# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The points at which SolverIPC meets the features libuipc has no term for (``newton._src.solvers.ipc.features``):
the refusals the feature modules hold, with their messages and in their order, the joints the universal module is
offered, the modules' hooks at build, step and notify, the supported observables and their allocation, the step's
observables a module reads, and the ``contact_damping`` option."""

import contextlib
import types
import unittest
import warnings
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.ipc import features, solver_ipc
from newton._src.solvers.ipc.joints import _map_joints
from newton.tests.ipc._fixtures import _TINY_LISTS, _box_body, _scene_calls, _stacking_cubes
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires

D = newton.ModelBuilder.JointDofConfig
_ARMATURE = (
    "SolverIPC takes the armature of revolute, prismatic, ball and universal joints only; that of a free joint, a "
    "mass without weight and an inertia, has no term"
)
_DRIVE = (
    "set the gains, stiffness and damping to 0 and the target mode to NONE, and apply forces through Control.joint_f, "
    "which acts on every joint. SolverIPC drives, springs and damps revolute, prismatic, ball and universal joints "
    "only"
)
_D6_BALL = (
    "SolverIPC takes the drives, springs, damping, armature and friction of a BALL joint, on the rotation vector of its "
    "rotation, and those of no D6 joint of three angular axes, whose coordinates are angles about its axes"
)


def _link(builder, x):
    link = builder.add_link(xform=wp.transform((x, 0.0, 1.0), wp.quat_identity()))
    builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
    return link


def _joint(add, offset=0, **dof):
    """Content: a link and the joint ``add(builder, link, x)`` from the world, in an articulation of its own; ``dof``
    sets the joint's arrays (``joint_<name>``) at its DOF ``offset``."""

    def content(builder, x):
        j = add(builder, _link(builder, x), x)
        builder.add_articulation([j])
        for name, value in dof.items():
            getattr(builder, f"joint_{name}")[builder.joint_qd_start[j] + offset] = value

    return content


def _revolute(**dof):
    return _joint(lambda b, link, x: b.add_joint_revolute(-1, link, axis=newton.Axis.Y), **dof)


def _ball(offset=0, **dof):
    def add(b, link, x):
        return b.add_joint_ball(-1, link, parent_xform=wp.transform((x, 0.0, 1.2), wp.quat_identity()))

    return _joint(add, offset, **dof)


def _d6(linear, angular, offset=0, **dof):
    def add(b, link, x):
        return b.add_joint_d6(
            -1, link, linear_axes=[D(axis=a) for a in linear], angular_axes=[D(axis=a) for a in angular]
        )

    return _joint(add, offset, **dof)


def _distance(lower, upper):
    def add(b, link, x):
        return b.add_joint_distance(
            -1, link, parent_xform=wp.transform((x, 0.0, 1.5), wp.quat_identity()), min_distance=lower,
            max_distance=upper,
        )  # fmt: skip

    return _joint(add)


def _rod():
    """A rod joint whose first two axes ``add_joint_rod`` made are swapped afterwards: a layout SolverIPC refuses."""

    def add(b, link, x):
        j = b.add_joint_rod(-1, link)
        s = b.joint_qd_start[j]
        b.joint_axis[s], b.joint_axis[s + 1] = b.joint_axis[s + 1], b.joint_axis[s]
        return j

    return _joint(add)


def _free(offset=0, **dof):
    """Content: a free body, whose free joint ``add_body`` adds; ``dof`` sets the joint's arrays at its DOF
    ``offset``."""

    def content(builder, x):
        _box_body(builder, (x, 0.0, 2.0), (0.1, 0.1, 0.1))
        for name, value in dof.items():
            getattr(builder, f"joint_{name}")[builder.joint_qd_start[-1] + offset] = value

    return content


def _tendon(builder, x):
    a, b = _link(builder, x), _link(builder, x + 0.3)
    builder.add_tendon([newton.TendonGuide(body=a), newton.TendonGuide(body=b)])


def _muscle(builder, x):
    a, b = _link(builder, x), _link(builder, x + 0.3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", newton.NewtonDeprecationWarning)  # legacy muscles are deprecated
        builder.add_muscle([a, b], [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)], 1.0, 0.5, 0.1, 1.0, 0.0)


def _vector_mimic(builder, x):
    """A mimic of two D6 joints of two linear axes each, which SolverIPC refuses (one coordinate per relation)."""
    a, b = _link(builder, x), _link(builder, x + 0.3)
    axes = [D(axis=newton.Axis.X), D(axis=newton.Axis.Y)]
    joints = [builder.add_joint_d6(-1, link, linear_axes=axes) for link in (a, b)]
    builder.add_articulation(joints)
    builder.set_joint_mimic(joints[1], joints[0])


def _equality(builder, x):
    """A body and a MuJoCo CONNECT row that holds it to the world, linked to no loop joint."""
    body = _box_body(builder, (x, 0.0, 1.0), (0.1, 0.1, 0.1))
    builder.add_custom_values(
        **{
            "mujoco:equality_constraint_type": 0,
            "mujoco:equality_constraint_objtype": 1,
            "mujoco:equality_constraint_body1": body,
            "mujoco:equality_constraint_body2": -1,
            "mujoco:equality_constraint_anchor": wp.vec3(0.0, 0.0, 1.0),
            "mujoco:equality_constraint_enabled": True,
        }
    )


def _spring(damping=0.0, rest=None):
    """Content: two particles 0.5 m apart (of Newton's default radius, 0.1 m) and a spring between them."""

    def content(builder, x):
        first = len(builder.particle_q)
        for k in range(2):
            builder.add_particle(wp.vec3(x + 0.5 * k, 3.0, 1.0), wp.vec3(0.0, 0.0, 0.0), 1.0)
        builder.add_spring(first, first + 1, 1.0e3, damping, 0.0)
        if rest is not None:
            builder.spring_rest_length[-1] = rest

    return content


def _attachment(damping=0.0, stiffness=None):
    """Content: a particle attached to a body below it."""

    def content(builder, x):
        builder.add_particle(wp.vec3(x, 3.0, 1.0), wp.vec3(0.0, 0.0, 0.0), 1.0)
        body = _box_body(builder, (x, 3.0, 1.5), (0.1, 0.1, 0.1))
        builder.add_attachment_particle_body(
            len(builder.particle_q) - 1, body, body_point=(0.0, 0.0, -0.5), damping=max(damping, 0.0)
        )
        if stiffness is not None:  # which the builder refuses: set afterwards
            builder.attachment_particle_body_stiffness[-1] = stiffness
        if damping < 0.0:  # likewise
            builder.attachment_particle_body_damping[-1] = damping

    return content


def _model(device, *contents):
    """A model of ``contents`` (functions of the builder and an x position), in this order."""
    builder = newton.ModelBuilder()
    for k, content in enumerate(contents):
        content(builder, 2.0 * k)
    return builder.finalize(device=device)


def _refusal(test, model, **options):
    """The message of the ``NotImplementedError`` SolverIPC raises for ``model``."""
    with test.assertRaises(NotImplementedError) as caught:
        newton.solvers.SolverIPC(model, **options)
    return str(caught.exception)


# (name, contents, the message), every refusal a feature module holds, alone in its model
_MOVED = (
    (
        "tendons (tendons)",
        (_tendon,),
        "SolverIPC: the model has 1 tendons of Newton's own (ModelBuilder.add_tendon); remove them. SolverIPC simulates "
        "MuJoCo's fixed and spatial tendons, and not Newton's tendons routed over anchors, pinholes and rollers.",
    ),
    (
        "a mimic of joints of several coordinates (joint_terms)",
        (_vector_mimic,),
        "SolverIPC: the follower of the mimic joint 1 is the D6 joint 1, which has more or other coordinates than one "
        "angle or one distance; mimic revolute and prismatic joints (or D6 joints of one axis). SolverIPC holds a mimic "
        "relation of one coordinate.",
    ),
    (
        "friction of a free joint (joint_terms)",
        (_free(3, friction=0.75),),
        "SolverIPC: DOF 3 of the FREE joint 0 has joint_friction 0.75; set it to 0. SolverIPC holds the friction of "
        "revolute, prismatic, ball and universal joints only, as a force on one coordinate or a torque about each axis "
        "of a ball joint or each angle of a universal joint.",
    ),
    (
        "a universal joint of axes that are not orthonormal (universal)",
        (_d6("", ("X", (1.0, 1.0, 0.0))),),
        "SolverIPC: the angular axes of the D6 joint 0 are not orthonormal; make them orthonormal. Newton's "
        "coordinates of a universal joint assume they are.",
    ),
    (
        "a cylindrical D6 joint (universal)",
        (_d6("X", "X"),),
        "SolverIPC: the D6 joint 0 has 1 linear and 1 angular axes; give it no axis, one axis, two or three angular "
        "axes, or three linear and three angular axes. Its axes lock 2 translations and 2 rotations, which no joint of "
        "SolverIPC locks.",
    ),
    (
        "a distance whose lower bound is above its upper (universal)",
        (_distance(0.6, 0.4),),
        "SolverIPC: the DISTANCE joint 0 bounds its distance to [0.6000000238418579, 0.4000000059604645], the lower "
        "bound above the upper; make the lower bound at most the upper, or negative for none.",
    ),
    (
        "friction of a D6 joint that becomes a ball joint (ball)",
        (_d6("", "XYZ", 1, friction=0.25),),
        f"SolverIPC: DOF 1 of the D6 joint 0 has joint_friction 0.25; set it to 0. {_D6_BALL}.",
    ),
    (
        "armature of a D6 joint that becomes a ball joint (ball)",
        (_d6("", "XYZ", 2, armature=0.5),),
        f"SolverIPC: DOF 2 of the D6 joint 0 has joint_armature 0.5; set it to 0. {_D6_BALL}.",
    ),
    (
        "armature of a free joint (ball)",
        (_free(5, armature=0.5),),
        f"SolverIPC: DOF 5 of the FREE joint 0 has joint_armature 0.5; set it to 0. {_ARMATURE}.",
    ),
    (
        "a drive of a free joint (ball)",
        (_free(0, target_ke=10.0),),
        "SolverIPC: DOF 0 of the FREE joint 0 has target_ke 10.0, target_kd 0.0, stiffness 0.0 and damping 0.0; "
        f"{_DRIVE}.",
    ),
    (
        "damping of a D6 joint that becomes a ball joint (ball)",
        (_d6("", "XYZ", 1, damping=2.0),),
        "SolverIPC: DOF 1 of the D6 joint 0 has target_ke 0.0, target_kd 0.0, stiffness 0.0 and damping 2.0; set the "
        f"gains, stiffness and damping to 0 and the target mode to NONE. {_D6_BALL}.",
    ),
    (
        "a target mode of a free joint (ball)",
        (_free(2, target_mode=int(newton.JointTargetMode.POSITION)),),
        f"SolverIPC: DOF 2 of the FREE joint 0 has the target mode POSITION; {_DRIVE}.",
    ),
    (
        "limits of a ball joint (ball)",
        (_ball(0, limit_lower=-0.5, limit_upper=0.5, limit_ke=100.0),),
        "SolverIPC: DOF 0 of the BALL joint 0 has the limits [-0.5, 0.5] with limit_ke 100.0; set limit_ke to 0, or "
        "the bounds to -MAXVAL and MAXVAL. Newton defines no limits of a ball joint: add_joint_ball takes none, its "
        "importers drop them and its solvers do not read them.",
    ),
    (
        "limits of a D6 joint that becomes a ball joint (ball)",
        (_d6("", "XYZ", 0, limit_lower=-0.5, limit_upper=0.5, limit_ke=100.0),),
        "SolverIPC: DOF 0 of the D6 joint 0 has the limits [-0.5, 0.5] with limit_ke 100.0; set limit_ke to 0, or "
        "the bounds to -MAXVAL and MAXVAL. libuipc limits revolute and prismatic joints only.",
    ),
    (
        "a negative damping of a spring (damping)",
        (_spring(damping=-0.5),),
        "SolverIPC: spring 0 has spring_damping -0.5; make it 0 or positive.",
    ),
    (
        "a negative damping of a particle-body attachment (damping)",
        (_attachment(), _attachment(damping=-2.0)),
        "SolverIPC: particle-body attachment 1 has damping -2.0; make it 0 or positive.",
    ),
)


def test_moved_refusals_keep_their_messages(test, device):
    """Every refusal a feature module holds raises NotImplementedError with the message it had before the modules
    held it, when the solver is built: Newton's tendons (tendons; its message now names the spatial tendons SolverIPC
    takes), mimics of joints of several coordinates and the friction of free joints (joint_terms), D6 joints that lock what no joint of SolverIPC locks, universal joints of axes that are
    not orthonormal and distance joints whose lower bound is above the upper (universal), the armature, drives and
    target mode of free joints, the friction, armature, damping and limits of D6 joints that become ball joints and
    the limits of ball joints (ball), and springs and attachments with a negative damping (damping)."""
    for name, contents, message in _MOVED:
        with test.subTest(name):
            test.assertEqual(_refusal(test, _model(device, *contents)), message)


# (name, contents, options, the first message): models that trip two refusals, of which the one that fired before the
# feature modules held them still fires first
_ORDER = (
    ("tendons before muscles", (_muscle, _tendon), {}, "the model has 1 tendons of Newton's own"),
    ("muscles before mimic constraints", (_vector_mimic, _muscle), {}, "the model has 1 muscles;"),
    ("mimic constraints before equality rows", (_equality, _vector_mimic), {}, "the follower of the mimic joint 2"),
    (
        "the lowest DOF of a category, a D6 joint's that becomes a ball joint",
        (_d6("", "XYZ", 1, friction=0.25), _free(0, friction=0.5)),
        {},
        "DOF 1 of the D6 joint 0 has joint_friction 0.25;",
    ),
    (
        "the lowest DOF of a category, a free joint's",
        (_free(0, friction=0.5), _d6("", "XYZ", 1, friction=0.25)),
        {},
        "DOF 0 of the FREE joint 0 has joint_friction 0.5;",
    ),
    (
        "friction before armature",
        (_d6("", "XYZ", 0, armature=0.5), _free(3, friction=0.5)),
        {},
        "DOF 6 of the FREE joint 1 has joint_friction 0.5;",
    ),
    (
        "a negative armature before the drives of free joints",
        (_free(0, target_ke=10.0), _revolute(armature=-0.5)),
        {},
        "DOF 6 of the REVOLUTE joint 1 has joint_armature -0.5; make it 0 or positive.",
    ),
    (
        "armature before limits",
        (_ball(0, limit_lower=-0.5, limit_upper=0.5, limit_ke=100.0), _free(0, armature=0.5)),
        {},
        "DOF 3 of the FREE joint 1 has joint_armature 0.5; set it to 0.",
    ),
    (
        "the drives of free joints before the effort limit",
        (_revolute(target_ke=100.0, effort_limit=-5.0), _free(2, target_ke=10.0)),
        {},
        "DOF 3 of the FREE joint 1 has target_ke 10.0,",
    ),
    (
        "the axes of a D6 joint before a later cylindrical one",
        (_d6("", ("X", (1.0, 1.0, 0.0), "Z")), _d6("X", "X")),
        {},
        "the angular axes of the D6 joint 0 are not orthonormal;",
    ),
    (
        "a cylindrical D6 joint before the axes of a later D6 joint",
        (_d6("X", "X"), _d6("", ("X", (1.0, 1.0, 0.0), "Z"))),
        {},
        "the D6 joint 0 has 1 linear and 1 angular axes;",
    ),
    (
        "D6 joints before distance joints",
        (_distance(0.6, 0.4), _d6("X", "X")),
        {},
        "the D6 joint 1 has 1 linear and 1 angular axes;",
    ),
    ("rod joints before D6 joints", (_d6("X", "X"), _rod()), {}, "the ROD joint 1 does not have three linear"),
    (
        "joints before springs",
        (_spring(damping=0.5), _free(3, friction=0.5)),
        {},
        "DOF 3 of the FREE joint 0 has joint_friction 0.5;",
    ),
    (
        "the damping of springs before their rest lengths",
        (_spring(rest=-0.1), _spring(damping=-0.5)),
        {},
        "spring 1 has spring_damping -0.5;",
    ),
    (
        "springs before attachments",
        (_attachment(damping=2.0), _spring(rest=-0.1)),
        {},
        "spring 0 has spring_rest_length -0.10000000149011612;",
    ),
    (
        "the damping of attachments before their stiffness",
        (_attachment(stiffness=-1.0), _attachment(damping=-2.0)),
        {},
        "particle-body attachment 1 has damping -2.0;",
    ),
    ("the model before contact_damping", (_rod(),), {"contact_damping": -1.0}, "the ROD joint 0 does not have"),
)


def test_refusals_keep_their_order(test, device):
    """A model that trips two refusals, one of them a feature module's, is refused with the message that came first
    before the modules held them: the modules' checks run at the points of the checks they replace, a module's
    refusal of the joints' degrees of freedom names the lowest degree of freedom of its category whichever module
    holds it, and the categories keep their order among the joints' own checks (friction, a negative armature, the
    armature, drives and limits of free joints and of D6 joints that become ball joints, the effort limit)."""
    for name, contents, options, start in _ORDER:
        with test.subTest(name):
            message = _refusal(test, _model(device, *contents), **options)
            test.assertTrue(message.startswith(f"SolverIPC: {start}"), message)
    # a contact_damping the module refuses is refused among the checks of the options, after the choices
    model = _model(device, _revolute())
    with test.assertRaisesRegex(ValueError, "integrator must be one of"):
        newton.solvers.SolverIPC(model, integrator="bdf3", contact_damping=-1.0)


def test_refusals_at_notify(test, device):
    """notify_model_changed checks again where it did before: CONSTRAINT_PROPERTIES the mimic constraints and the
    attachments' damping (before their stiffness), the flags of the joints' degrees of freedom the joints, and a new
    spring damping (MODEL_PROPERTIES) builds the solve again, which refuses a negative one. Each refusal has its
    message."""
    flags = newton.ModelFlags
    model = _model(device, _revolute(), _revolute(), _attachment(), _attachment(), _spring())
    solver = newton.solvers.SolverIPC(model)

    def refused(flag, message):
        with test.assertRaises(NotImplementedError) as caught:
            solver.notify_model_changed(flag)
        test.assertEqual(str(caught.exception), message)

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
        flags.CONSTRAINT_PROPERTIES,
        "SolverIPC: particle-body attachment 1 has damping -2.0; make it 0 or positive.",
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
    refused(
        flags.MODEL_PROPERTIES,
        "SolverIPC: spring 0 has spring_damping -0.5; make it 0 or positive.",
    )


def test_taken_joints_are_left_to_their_module(test, device):
    """A joint the universal module takes when it is offered (take_d6, take_distance return) is the module's: a D6
    joint of no libuipc joint maps onto no libuipc joint (kind D6) and is stored by nobody (FIXED), the solver lays
    out the libuipc joints of the others only and leaves the taken joint's joint_q as it was, and the joints' checks
    give the module its degrees of freedom (kind D6) without refusing its drive; a distance joint with a range is no
    stitch. The universal module stores its joints' joint_q itself (test_universal_range), which is left out here."""
    import uipc_warp.arrays  # noqa: PLC0415

    model = _model(device, _revolute(target_ke=10.0), _d6("", "XY", 0, target_ke=10.0), _distance(0.4, 0.6))
    offered, kinds = [], {}

    def record(category, dofs):
        kinds[category] = dofs.kind.copy()

    with (
        mock.patch.object(features.universal, "take_d6", side_effect=lambda *args: offered.append(args[1:])),
        mock.patch.object(features.universal, "take_distance", side_effect=lambda *args: offered.append(args[1:2])),
        mock.patch.object(features.universal, "joint_refusal", side_effect=record),
        mock.patch.object(features.universal, "finish"),
    ):
        mapping = _map_joints(model)
        test.assertEqual(offered, [(1, 0, 2), (2,)])
        np.testing.assert_array_equal(
            mapping.kind, [int(newton.JointType.REVOLUTE), int(newton.JointType.D6), int(newton.JointType.FREE)]
        )
        np.testing.assert_array_equal(
            mapping.store, [int(newton.JointType.REVOLUTE), int(newton.JointType.FIXED), int(newton.JointType.FREE)]
        )
        test.assertEqual((len(mapping.d6), len(mapping.distance)), (0, 0))
        scene = uipc_warp.arrays.ArrayScene
        with (
            _scene_calls(scene.add_revolute_joints) as revolute,
            _scene_calls(scene.add_spherical_joints) as spherical,
            _scene_calls(scene.add_fixed_joints) as fixed,
            _scene_calls(scene.add_stitches) as stitches,
        ):
            solver = newton.solvers.SolverIPC(model)
        test.assertEqual(sorted(kinds), ["armature", "drive", "friction", "limit"])
        np.testing.assert_array_equal(
            kinds["drive"][:3], [int(newton.JointType.REVOLUTE), int(newton.JointType.D6), int(newton.JointType.D6)]
        )
        test.assertEqual((len(spherical), len(fixed), len(stitches)), (0, 0, 0))
        test.assertEqual(len(revolute), 1)
        np.testing.assert_array_equal(np.asarray(revolute[0]["bodies"])[:, 1], [0])
        s0, s1 = model.state(), model.state()
        q = s1.joint_q.numpy()
        q[1:3] = 7.0
        s1.joint_q.assign(q)
        solver.step(s0, s1, None, None, 0.01)
        np.testing.assert_array_equal(s1.joint_q.numpy()[1:3], [7.0, 7.0])


# the hooks every feature module has besides its checks, and the modules' names in the order the solver calls them
_HOOKS = ("build", "engine_inputs", "prepare", "finish", "notify")
_NAMES = tuple(feature.__name__.rsplit(".", 1)[1] for feature in features.FEATURES)


def _every(hook):
    """``hook`` of every module, in the order the solver calls them."""
    return [f"{name}.{hook}" for name in _NAMES]


@contextlib.contextmanager
def _recorded(solver=None, returns=None):
    """Record, in the order they happen, the calls of every feature module's hooks (``"damping.build"``, ...), each
    calling the module's own hook, and of the solver's points around them: its last write of a step's inputs
    (``"inputs"``), the eager frame (``"_advance_eager"``), the engine's frame (``"advance"``), state_out
    (``"_store"``) and the observables (``"_write_observables"``). Yields ``calls``, a list of ``(name, args, engine,
    scene)``: whether the solver had an engine at the call, and its scene then. ``returns``: what a hook returns
    instead of the module's answer, by name."""
    calls = []

    def recorder(name, function):
        def call(*args, **kwargs):
            owner = args[0] if name.split(".")[0] in _NAMES else solver
            calls.append((name, args, getattr(owner, "_engine", None) is not None, getattr(owner, "_scene", None)))
            result = function(*args, **kwargs)
            return (returns or {}).get(name, result)

        return call

    with contextlib.ExitStack() as stack:
        for name, module in zip(_NAMES, features.FEATURES, strict=True):
            for hook in _HOOKS:
                stack.enter_context(mock.patch.object(module, hook, recorder(f"{name}.{hook}", getattr(module, hook))))
        stack.enter_context(
            mock.patch.object(solver_ipc, "_write_body_stiffness", recorder("inputs", solver_ipc._write_body_stiffness))
        )
        if solver is not None:
            for method in ("_advance_eager", "_store", "_write_observables"):
                stack.enter_context(mock.patch.object(solver, method, recorder(method, getattr(solver, method))))
            stack.enter_context(
                mock.patch.object(solver._engine, "advance", recorder("advance", solver._engine.advance))
            )
        yield calls


def _names(calls):
    return [call[0] for call in calls]


def test_hooks_are_called_at_their_points(test, device):
    """The solver calls every feature module's hooks at their points, the modules in the order of FEATURES: build when
    the solve is built, with the scene the solver laid out, before the scene is the solve's and the engine is made;
    engine_inputs for every engine made from it, with the solver's external accelerations among the element groups,
    before the engine is made (again when an eager step enlarges the contact lists); prepare in every step after the
    solver's last input and before the frame, and finish after state_out and the observables, in an eager step and in
    a captured one (at its capture: a replay calls nothing), and in a step that fails in some worlds alone, not in one
    that fails; notify in notify_model_changed, where a module's True builds the solve again."""
    import uipc_warp.arrays  # noqa: PLC0415

    model = _falling_box(device)
    with _recorded() as calls:
        solver = newton.solvers.SolverIPC(model)
    test.assertEqual(_names(calls), _every("build") + _every("engine_inputs"))
    for name, args, engine, scene in calls:
        test.assertIs(args[0], solver)
        test.assertFalse(engine, name)  # before the engine is made
        if name.endswith(".build"):
            test.assertIsInstance(args[1], uipc_warp.arrays.ArrayScene)
            test.assertIs(args[1], solver._scene)  # the solve's scene, which it was not yet
            test.assertIsNone(scene)
        else:
            test.assertIs(scene, solver._scene)
            test.assertIs(args[1]["elements"][-1], solver._forces)  # the solver's own groups come first
    test.assertIsNotNone(solver._engine)

    observables = solver.observables({solver.ObservableKind.STATUS})
    s0, s1 = model.state(), model.state()
    control = model.control()
    with _recorded(solver) as calls:
        solver.step(s0, s1, control, None, 0.01, observables=observables)
        eager = len(calls)
        with wp.ScopedCapture(device=device) as capture:
            solver.step(s1, s0, None, None, 0.01, observables=observables)
        captured = len(calls)
        wp.capture_launch(capture.graph)
    # the first eager frame of an engine captures the engine's frame into the solver's own graph
    frame = ["_advance_eager", "advance", "_store", "_write_observables"]
    test.assertEqual(_names(calls[:eager]), ["inputs", *_every("prepare"), *frame, *_every("finish")])
    frame = ["advance", "_store", "_write_observables"]
    test.assertEqual(_names(calls[eager:captured]), ["inputs", *_every("prepare"), *frame, *_every("finish")])
    test.assertEqual(len(calls), captured)  # the replay calls nothing
    for name, args, _, _ in calls[:eager]:
        if name.endswith(".prepare"):
            test.assertEqual(args[1:], (s0, control, 0.01))
        if name.endswith(".finish"):
            test.assertEqual(args[1:], (s1, observables))

    # a step that fails in some independent worlds alone writes state_out and calls finish; one that fails does not
    failed = types.SimpleNamespace(failed=True, overflowed=False, traversal_exhausted=False)
    for alone in (True, False):
        with test.subTest(alone=alone):
            with (
                _recorded(solver) as calls,
                mock.patch.object(solver, "_advance_eager", return_value=failed),
                mock.patch.object(solver, "_failed_alone", return_value=alone),
                mock.patch.object(solver, "_failed_worlds", return_value={0: 1}),
                mock.patch.object(solver, "_world_failure", return_value="failed alone"),
                mock.patch.object(solver, "_failure", return_value="failed"),
                test.assertRaisesRegex(RuntimeError, "failed alone" if alone else "failed"),
            ):
                solver.step(s0, s1, None, None, 0.01, observables=observables)
            ends = ["_store", "_write_observables", *_every("finish")] if alone else ["_write_observables"]
            test.assertEqual(_names(calls)[-len(ends) :], ends)

    # notify: every module, after which a True builds the solve again
    engine = solver._engine
    with _recorded(solver) as calls:
        solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
    test.assertEqual(_names(calls), _every("notify"))
    test.assertIs(solver._engine, engine)
    with _recorded(solver, returns={"contact_damping.notify": True}) as calls:
        solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
    test.assertEqual(_names(calls), _every("notify") + _every("build") + _every("engine_inputs"))
    test.assertIsNot(solver._engine, engine)
    for name, args, _, _ in calls[: len(_NAMES)]:
        test.assertEqual(args[1], newton.ModelFlags.JOINT_PROPERTIES, name)

    # an eager step whose contact lists overflow makes an engine with longer lists: engine_inputs again, in the step
    model = _stacking_cubes(device)
    solver = newton.solvers.SolverIPC(model)
    solver._make_engine(_TINY_LISTS)
    s0, s1 = model.state(), model.state()
    regrown = []
    for _ in range(30):
        with _recorded(solver) as calls:
            solver.step(s0, s1, None, None, 0.01)
        s0, s1 = s1, s0
        names = _names(calls)
        if "damping.engine_inputs" in names:
            regrown.append(names)
    test.assertGreater(len(regrown), 0)
    for names in regrown:  # each longer engine: the modules' inputs, then the solver's inputs of the step again
        between = names[names.index("_advance_eager") + 1 : names.index("_store")]
        if between[:1] == ["advance"]:  # the engine before captured its frame again (_engine_outdated)
            between = between[1:]
        regrow = [*_every("engine_inputs"), "inputs"]
        test.assertGreater(len(between), 0)
        test.assertEqual(between, regrow * (len(between) // len(regrow)))


def test_supported_observables_come_from_the_observables_module(test, device):
    """The solver's supported observables are the observables module's: SUPPORTED_OBSERVABLES holds STATUS,
    SOLVER_RESIDUAL, CONTACT_F, BODY_PARENT_F and BODY_QDD, a solver supports them, and under AL-IPC STATUS,
    BODY_PARENT_F and BODY_QDD only; a kind the module adds is taken by observables(), which refuses it, as every kind
    outside the set, without it."""
    kind = newton.solvers.SolverIPC.ObservableKind
    body = {kind.BODY_PARENT_F, kind.BODY_QDD}
    test.assertEqual(
        newton.solvers.SolverIPC.SUPPORTED_OBSERVABLES, {kind.STATUS, kind.SOLVER_RESIDUAL, kind.CONTACT_F, *body}
    )
    model = _falling_box(device)
    solver = newton.solvers.SolverIPC(model)
    test.assertEqual(solver.supported_observables, newton.solvers.SolverIPC.SUPPORTED_OBSERVABLES)
    al = newton.solvers.SolverIPC(model, contact_constitution="al-ipc")
    test.assertEqual(al.supported_observables, {kind.STATUS, *body})
    with test.assertRaisesRegex(ValueError, "SolverIPC does not support solver observable\\(s\\): 'solver_residual'"):
        al.observables({kind.SOLVER_RESIDUAL})
    more = al.supported_observables | {kind.SOLVER_RESIDUAL}
    with mock.patch.object(features.observables, "supported", return_value=more) as supported:
        test.assertEqual(al.supported_observables, more)
        test.assertIsNotNone(al.observables({kind.SOLVER_RESIDUAL}).solver_residual)
    supported.assert_called_with(al)


def test_observables_module_allocates_for_every_container(test, device):
    """SolverIPC.observables() calls the observables module's allocate once for every container it allocates, with the
    solver and the container, after Newton's allocation (the arrays of the kinds the container requests exist then)
    and after the solver's own (the contact records of CONTACT_F), also for a container that requests nothing."""
    model = _falling_box(device)
    solver = newton.solvers.SolverIPC(model)
    kind = solver.ObservableKind
    calls = []

    def allocate(owner, observables):
        allocated = (observables.status is not None, observables.contact_f is not None)
        calls.append((owner, observables, allocated, solver._contact_export is not None))

    contacts = newton.Contacts(16, 4, device=device)
    with mock.patch.object(features.observables, "allocate", side_effect=allocate):
        status = solver.observables({kind.STATUS})
        test.assertEqual(calls, [(solver, status, (True, False), False)])
        forces = solver.observables({kind.STATUS, kind.CONTACT_F}, contacts=contacts)
        nothing = solver.observables(set())
    test.assertEqual(
        calls,
        [
            (solver, status, (True, False), False),
            (solver, forces, (True, True), True),
            (solver, nothing, (False, False), True),
        ],
    )


def test_prepare_reads_the_steps_observables(test, device):
    """When the feature modules' prepare runs, solver._observables is the observables container the step was given,
    or None for a step given none: in eager steps and in captured ones (at their capture), whatever the step before
    was given; and None before the first step."""
    model = _falling_box(device)
    solver = newton.solvers.SolverIPC(model)
    test.assertIsNone(solver._observables)
    first = solver.observables({solver.ObservableKind.STATUS})
    second = solver.observables({solver.ObservableKind.STATUS})
    seen = []
    s0, s1 = model.state(), model.state()
    with mock.patch.object(
        features.observables, "prepare", side_effect=lambda owner, *args: seen.append(owner._observables)
    ):
        for given in (first, None, second, first):
            solver.step(s0, s1, None, None, 0.01, observables=given)
            s0, s1 = s1, s0
        for given in (None, second):
            with wp.ScopedCapture(device=device) as capture:
                solver.step(s0, s1, None, None, 0.01, observables=given)
            wp.capture_launch(capture.graph)
            s0, s1 = s1, s0
    test.assertEqual(
        [id(o) if o is not None else None for o in seen], [id(first), None, id(second), id(first), None, id(second)]
    )


def _falling_box(device):
    """A box 0.4 m above the ground."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    _box_body(builder, (0.0, 0.0, 0.5), (0.1, 0.1, 0.1))
    return builder.finalize(device=device)


devices = get_cuda_test_devices()


@requires("uipc_warp")
class TestSolverIPCFeatureSeams(unittest.TestCase):
    """The points at which SolverIPC meets the features libuipc has no term for."""


for name, func in (
    ("test_moved_refusals_keep_their_messages", test_moved_refusals_keep_their_messages),
    ("test_refusals_keep_their_order", test_refusals_keep_their_order),
    ("test_refusals_at_notify", test_refusals_at_notify),
    ("test_taken_joints_are_left_to_their_module", test_taken_joints_are_left_to_their_module),
    ("test_hooks_are_called_at_their_points", test_hooks_are_called_at_their_points),
    (
        "test_supported_observables_come_from_the_observables_module",
        test_supported_observables_come_from_the_observables_module,
    ),
    ("test_observables_module_allocates_for_every_container", test_observables_module_allocates_for_every_container),
    ("test_prepare_reads_the_steps_observables", test_prepare_reads_the_steps_observables),
):
    add_function_test(TestSolverIPCFeatureSeams, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
