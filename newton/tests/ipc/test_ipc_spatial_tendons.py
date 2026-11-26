# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC's MuJoCo spatial tendons (``features/tendons.py``), as libuipc-warp's spatial tendons.

* Closed forms on models of the MJCF importer: a box hangs from a tendon of a spring over a fixed cylinder, which the
  path wraps on its side site's side, where the spring holds its weight; over a pulley of divisor 2 the branch after
  it pulls with half the tension.
* The coefficients as SolverMuJoCo reads them: the spring length -1 as the length at the model's pose, a pair as a
  band, the damping, armature and friction loss, the limit's range, its stiffness in force space and from solref
  through the inverse weight ``J H^-1 J^T`` (the moment arm r of a hinged arm: ``r^2 / I``), the limit's damping.
* ``notify_model_changed``: new coefficients written in place; a moved site builds the solve again.
* Independent worlds step as each world alone; a step solved again on an engine with longer contact lists (a box
  landing on a static box) keeps the step's limit strength; a spatial tendon's friction loss lowers the default Newton tolerances as a joint's friction
  does; a tendon of the world alone gets no term.
* The refusals, each with its message: Newton's own tendons, a wrap around a box, a path MuJoCo's compiler refuses, a
  side site inside its shape; an actuator on a spatial tendon and a limit margin stay refused (``joint_terms``).
"""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.ipc import solver_ipc
from newton.solvers import mujoco
from newton.tests.ipc._fixtures import _TINY_LISTS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires

_TIGHT = {"newton_velocity_tol": 1e-6, "newton_transrate_tol": 1e-6, "semi_implicit": False, "linear_tol_rate": 1e-10}

# a box of 0.1 m (1 kg) on a free joint whose top hangs from a spatial tendon: from a point of the world 0.4 m left of a
# fixed cylinder of radius 0.05 m (axis y, at height 1), over its top (a side site above it), down to the box's top
_HANGING = """<?xml version="1.0" ?>
<mujoco model="hanging">
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <site name="anchor" pos="-0.4 0 1"/>
    <site name="above" pos="0 0 1.2"/>
    <geom name="wheel" type="cylinder" size="0.05 0.02" pos="0 0 1" zaxis="0 1 0" contype="0" conaffinity="0"/>
    <body name="box" pos="0.05 0 0.75">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" density="1000"/>
      <site name="top" pos="0 0 0.05"/>
    </body>
  </worldbody>
  <tendon>
    <spatial name="rope" {attributes}>
      <site site="anchor"/>
      <geom geom="wheel" sidesite="above"/>
      <site site="top"/>
    </spatial>
  </tendon>
</mujoco>
"""

# a hinged arm (about y at the origin) whose site at x = r, 0.3 m long, is pulled up by a tendon to a point of the
# world straight above it: the tendon's moment arm in joint space is r
_ARM = """<?xml version="1.0" ?>
<mujoco model="arm">
  <option gravity="0 0 0"/>
  <worldbody>
    <site name="up" pos="0.3 0 1"/>
    <body name="arm">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="box" size="0.15 0.02 0.02" pos="0.15 0 0" density="1000"/>
      <site name="end" pos="0.3 0 0"/>
    </body>
  </worldbody>
  <tendon>
    <spatial name="t" {attributes}>
      <site site="up"/>
      <site site="end"/>
    </spatial>
  </tendon>
</mujoco>
"""


def _model(device, mjcf, worlds=1, **attributes):
    """The model of ``mjcf`` with the tendon's ``attributes`` (MJCF names), in ``worlds`` worlds."""
    text = mjcf.format(attributes=" ".join(f'{k}="{v}"' for k, v in attributes.items()))
    one = newton.ModelBuilder()
    one.add_mjcf(text, parse_sites=True)
    if worlds == 1:
        return one.finalize(device=device)
    builder = newton.ModelBuilder()
    for _ in range(worlds):
        builder.add_world(one)
    return builder.finalize(device=device)


def _run(model, solver, steps, dt):
    """The state after ``steps`` steps from the model's pose."""
    s0, s1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    for _ in range(steps):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
    return s0


def _site(state, body, point):
    """A body point in the world."""
    q = state.body_q.numpy().astype(np.float64)[body]
    x, y, z, w = q[3:7]
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # fmt: skip
    return q[:3] + R @ np.asarray(point, np.float64)


def _over_the_wheel():
    """The fixed part of the hanging tendon: the tangent from the anchor to the wheel's top and the arc over it to its
    rightmost point, where the hanging branch leaves it straight down."""
    d, R = 0.4, 0.05
    return math.sqrt(d * d - R * R) + R * (math.pi - math.acos(R / d))


def test_a_box_hangs_over_a_cylinder(test, device):
    """Settle a box hanging from a spring tendon wrapped over a cylinder where the spring holds its weight.

    The box hangs from the tendon (500 N/m, spring length 0.6 m, critically damped) over the wheel, which the path wraps
    over its top on the side site's side. It settles where the spring holds its weight: the tendon's length is ``0.6 + m
    g / k`` (the fixed part over the wheel and the hanging branch), and the box's top is straight under the wheel's
    rightmost point.
    """
    k, m, g = 500.0, 1.0, 9.81
    model = _model(device, _HANGING, stiffness=k, springlength=0.6, damping=2.0 * math.sqrt(k * m))
    test.assertAlmostEqual(float(model.body_mass.numpy()[0]), m, delta=1e-5)
    solver = newton.solvers.SolverIPC(model, **_TIGHT)
    state = _run(model, solver, 200, 0.005)
    top = _site(state, 0, (0.0, 0.0, 0.05))
    length = _over_the_wheel() + (1.0 - top[2])
    # to 1e-5 m: the affine body gives under the 9.8 N at its top by about 2 micrometres (SolverIPC's default body
    # stiffness at this step)
    test.assertAlmostEqual(length, 0.6 + m * g / k, delta=1e-5)
    test.assertAlmostEqual(top[0], 0.05, delta=2e-6)


def test_a_pulley_halves_its_branch(test, device):
    """Halve the length of a tendon's branch after a pulley of divisor 2.

    A tendon runs from a point of the world down to the box's top, then a pulley of divisor 2 and a branch from the
    box's top up to another point of the world. The box's top hangs where ``k (L - 0.6)`` times 1.5 holds its weight,
    ``L`` the first branch's length plus half the second's.
    """
    mjcf = """<?xml version="1.0" ?>
<mujoco model="pulley">
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <site name="a" pos="0 0 1"/>
    <site name="b" pos="0 0 1.2"/>
    <body name="box" pos="0 0 0.5">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" density="1000"/>
      <site name="top" pos="0 0 0.05"/>
    </body>
  </worldbody>
  <tendon>
    <spatial name="t" {attributes}>
      <site site="a"/>
      <site site="top"/>
      <pulley divisor="2"/>
      <site site="top"/>
      <site site="b"/>
    </spatial>
  </tendon>
</mujoco>
"""
    k, m, g = 500.0, 1.0, 9.81
    # the box's height changes L 1.5 times as fast: k and the damping act on it 2.25 times as strongly
    model = _model(device, mjcf, stiffness=k, springlength=0.6, damping=2.0 * math.sqrt(k * m) / 1.5)
    solver = newton.solvers.SolverIPC(model, **_TIGHT)
    state = _run(model, solver, 200, 0.005)
    top = _site(state, 0, (0.0, 0.0, 0.05))
    length = (1.0 - top[2]) + 0.5 * (1.2 - top[2])
    # to 1e-5 m: the affine body gives under the 9.8 N at its top by about 2 micrometres (SolverIPC's default body
    # stiffness at this step), which L takes 1.5 times
    test.assertAlmostEqual(length, 0.6 + m * g / (1.5 * k), delta=1e-5)


def test_coefficients(test, device):
    """Read a spatial tendon's coefficients as SolverMuJoCo reads them.

    Stiffness, damping, armature and friction loss are taken as they are; a spring length pair as a band, and -1 as the
    length at the model's pose. The limit takes its range, and its stiffness in force space, or from solref ``(T,
    zeta)`` and solimp's ``d_max`` as ``1 / (T^2 zeta^2 w)``, ``w = (r^2 / I) (1 - d_max)``, with the arm's inverse
    weight ``r^2 / I`` (r = 0.3 m, I its inertia about the hinge). Its damping is ``2 / (T w)``, which the tendon takes
    past a bound with ``enable_joint_limit_damping``.
    """
    attributes = {"stiffness": 3.0, "damping": 0.5, "armature": 0.2, "frictionloss": 0.1, "springlength": "0.8 0.9",
                  "limited": "true", "range": "0.5 1.5", "solreflimit": "0.05 0.8", "solimplimit": "0.9 0.95 0.001"}  # fmt: skip
    model = _model(device, _ARM, **attributes)
    solver = newton.solvers.SolverIPC(model)
    plan = solver._features.tendons.plan
    tendons = plan.tendons
    for name, array, value in (
        ("spring", tendons.spring, 3.0), ("spring_lower", tendons.spring_lower, 0.8),
        ("spring_upper", tendons.spring_upper, 0.9), ("damping", tendons.damping, 0.5),
        ("armature", tendons.armature, 0.2), ("friction", tendons.friction, 0.1),
        ("limit_lower", tendons.limit_lower, 0.5), ("limit_upper", tendons.limit_upper, 1.5),
    ):  # fmt: skip
        test.assertAlmostEqual(float(array.numpy()[0]), value, delta=1e-6, msg=name)
    m = float(model.body_mass.numpy()[0])
    inertia = m * (0.3 * 0.3 + 0.04 * 0.04) / 12.0 + m * 0.15 * 0.15  # the box about the hinge
    w = 0.3 * 0.3 / inertia * (1.0 - 0.95)
    test.assertAlmostEqual(float(plan.limit_ke.numpy()[0]) * (0.05 * 0.05 * 0.8 * 0.8 * w), 1.0, delta=1e-5)
    test.assertAlmostEqual(float(tendons.limit_damping.numpy()[0]) * (0.05 * w) / 2.0, 1.0, delta=1e-5)
    solver.step(model.state(), model.state(), None, None, 0.01)
    width = 1.0
    test.assertAlmostEqual(
        float(tendons.limit_strength.numpy()[0]),
        0.01 * 0.01 * float(plan.limit_ke.numpy()[0]) * width**3 / (3.0 * 1e-3),
        delta=1e-9,
    )
    # without the limit's damping; in force space; the spring length -1: the length at the model's pose, 1 m
    solver = newton.solvers.SolverIPC(model, enable_joint_limit_damping=False)
    test.assertEqual(float(solver._features.tendons.plan.tendons.limit_damping.numpy()[0]), 0.0)
    model = _model(device, _ARM, stiffness=3.0, springlength=-1, limited="true", range="0.5 1.5")
    model.mujoco.tendon_solref_limit_mode.fill_(0)  # force space
    model.mujoco.tendon_limit_ke.fill_(250.0)
    model.mujoco.tendon_limit_kd.fill_(4.0)
    solver = newton.solvers.SolverIPC(model)
    tendons = solver._features.tendons.plan.tendons
    test.assertAlmostEqual(float(solver._features.tendons.plan.limit_ke.numpy()[0]), 250.0, delta=1e-12)
    test.assertAlmostEqual(float(tendons.limit_damping.numpy()[0]), 4.0, delta=1e-12)
    test.assertAlmostEqual(float(tendons.spring_lower.numpy()[0]), 1.0, delta=1e-12)
    test.assertAlmostEqual(float(tendons.spring_upper.numpy()[0]), 1.0, delta=1e-12)


def test_notify(test, device):
    """Write new tendon coefficients in place and rebuild the solve for a moved site.

    New coefficients (``TENDON_PROPERTIES``) are written into the tendons in place, with the limit's stiffness that new
    masses give (``BODY_INERTIAL_PROPERTIES``); a site moved (``SHAPE_PROPERTIES``) builds the solve again.
    """
    model = _model(device, _ARM, stiffness=3.0, limited="true", range="0.5 1.5")
    solver = newton.solvers.SolverIPC(model)
    tendons, engine = solver._features.tendons.plan.tendons, solver._engine
    ke = float(solver._features.tendons.plan.limit_ke.numpy()[0])
    model.mujoco.tendon_stiffness.fill_(7.0)
    model.mujoco.tendon_damping.fill_(0.25)
    solver.notify_model_changed(newton.ModelFlags.TENDON_PROPERTIES)
    test.assertIs(solver._engine, engine)
    test.assertIs(solver._features.tendons.plan.tendons, tendons)
    test.assertEqual(float(tendons.spring.numpy()[0]), 7.0)
    test.assertEqual(float(tendons.damping.numpy()[0]), 0.25)
    model.body_mass.assign(2.0 * model.body_mass.numpy())
    model.body_inertia.assign(2.0 * model.body_inertia.numpy())
    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    test.assertIs(solver._engine, engine)
    test.assertAlmostEqual(float(solver._features.tendons.plan.limit_ke.numpy()[0]) / ke, 2.0, delta=1e-5)
    xform = model.shape_transform.numpy()
    sites = np.flatnonzero(model.shape_flags.numpy() & int(newton.ShapeFlags.SITE))
    xform[sites[-1], 0] += 0.01
    model.shape_transform.assign(xform)
    solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
    test.assertIsNot(solver._engine, engine)


def test_worlds(test, device):
    """Settle each of three independent worlds of the hanging box as the model of one world does."""
    attributes = {"stiffness": 500.0, "springlength": 0.6, "damping": 44.7}
    one = _model(device, _HANGING, **attributes)
    three = _model(device, _HANGING, 3, **attributes)
    alone = _run(one, newton.solvers.SolverIPC(one, **_TIGHT), 40, 0.005).body_q.numpy()
    worlds = _run(three, newton.solvers.SolverIPC(three, **_TIGHT), 40, 0.005).body_q.numpy()
    for k in range(3):
        np.testing.assert_allclose(worlds[k], alone[0], rtol=0, atol=1e-6)


def test_friction_lowers_the_tolerances(test, device):
    """Lower the default Newton tolerances for a spatial tendon's friction loss as for joint friction.

    A spatial tendon's friction loss gives the model the Newton tolerances of a model with joint friction, 0.4
    ``eps_velocity``, where they are left at their default.
    """
    solver = newton.solvers.SolverIPC(_model(device, _ARM, stiffness=3.0, frictionloss=0.5))
    bound = solver_ipc._FRICTION_TOLERANCE * solver._config.eps_velocity
    test.assertEqual(solver._config.newton_velocity_tol, bound)
    test.assertEqual(solver._config.newton_transrate_tol, bound)


def test_a_tendon_of_the_world_alone(test, device):
    """Add no term for a spatial tendon whose sites are all of the world.

    Such a tendon has a constant length and no force, as in MuJoCo.
    """
    builder = newton.ModelBuilder()
    body = builder.add_body()
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    a = builder.add_site(-1, xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
    b = builder.add_site(-1, xform=wp.transform((0.5, 0.0, 1.0), wp.quat_identity()))
    mujoco.add_tendon_spatial(builder, [mujoco.TendonWrapSite(s) for s in (a, b)], stiffness=10.0)
    solver = newton.solvers.SolverIPC(builder.finalize(device=device))
    test.assertIsNone(solver._features.tendons.plan)


def test_grown_lists_keep_the_limit(test, device):
    """Keep a tendon limit's strength when an eager step grows overflowing contact lists.

    A hinged arm falls until a tendon's limit holds it, while a box drops onto a static box. When the box lands the
    contact lists, made short, overflow, and the eager step grows them and solves the step again on a new engine, whose
    tendons take the step's limit strength from the old: the arm's angle goes on as in a solver with room.
    """
    mjcf = """<?xml version="1.0" ?>
<mujoco model="grown">
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="table" type="box" size="0.3 0.3 0.1" pos="1 0 0"/>
    <site name="up" pos="0.3 0 1.2"/>
    <body name="arm" pos="0 0 1">
      <joint type="hinge" axis="0 1 0"/>
      <geom type="box" size="0.15 0.02 0.02" pos="0.15 0 0" density="1000" contype="0" conaffinity="0"/>
      <site name="end" pos="0.3 0 0"/>
    </body>
    <body name="drop" pos="1 0 0.3"><freejoint/><geom type="box" size="0.05 0.05 0.05" density="1000"/></body>
  </worldbody>
  <tendon><spatial limited="true" range="0 0.25"><site site="up"/><site site="end"/></spatial></tendon>
</mujoco>
"""

    def run(tiny):
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf, parse_sites=True)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverIPC(model, **_TIGHT)
        regrown = []
        regrow = solver._regrow
        solver._regrow = lambda: (regrown.append(1), regrow())
        if tiny:
            solver._make_engine(_TINY_LISTS)
        s0, s1 = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        q = []
        for _ in range(40):
            solver.step(s0, s1, None, None, 0.01)
            s0, s1 = s1, s0
            q.append(float(s0.joint_q.numpy()[0]))
        return np.array(q), len(regrown)

    (roomy, roomy_regrown), (grown, grown_regrown) = run(False), run(True)
    test.assertGreater(roomy[-1], 0.1, "the arm has fallen onto the limit (0.167 rad)")
    test.assertEqual(roomy_regrown, 0)
    test.assertGreater(grown_regrown, 0)
    np.testing.assert_allclose(grown, roomy, rtol=0, atol=1e-5)


def _refusal(test, model, **options):
    with test.assertRaises(NotImplementedError) as caught:
        newton.solvers.SolverIPC(model, **options)
    return str(caught.exception)


def test_refusals(test, device):
    """Refuse the tendons that SolverIPC does not simulate, each with its message.

    The refused cases are Newton's own tendons (``ModelBuilder.add_tendon``), a wrap around a box, a side site inside
    the shape it wraps, paths that MuJoCo's compiler refuses, a path of one entry, and, through ``joint_terms``, an
    actuator that drives a tendon and a limit margin.
    """
    builder = newton.ModelBuilder()
    a, b = builder.add_link(), builder.add_link()
    for body in (a, b):
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_articulation([builder.add_joint_revolute(-1, a), builder.add_joint_revolute(a, b)])
    builder.add_tendon([newton.TendonGuide(body=a), newton.TendonGuide(body=b)])
    test.assertEqual(
        _refusal(test, builder.finalize(device=device)),
        "SolverIPC: the model has 1 tendons of Newton's own (ModelBuilder.add_tendon); remove them. SolverIPC simulates "
        "MuJoCo's fixed and spatial tendons, and not Newton's tendons routed over anchors, pinholes and rollers.",
    )

    def wrapped(geom: str, path: str | None = None, side: str = "") -> newton.Model:
        mjcf = f"""<?xml version="1.0" ?>
<mujoco><worldbody>
  <site name="s0" pos="-0.3 0 1"/>
  <body name="b" pos="0 0 1"><freejoint/>{geom}<site name="s1" pos="0.3 0 0"/><site name="in" pos="0.01 0 0"/></body>
</worldbody><tendon><spatial stiffness="1">{path or '<site site="s0"/><geom geom="g" ' + side + '/><site site="s1"/>'}
</spatial></tendon></mujoco>"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf, parse_sites=True)
        return builder.finalize(device=device)

    box = '<geom name="g" type="box" size="0.05 0.05 0.05" contype="0" conaffinity="0"/>'
    sphere = '<geom name="g" type="sphere" size="0.05" contype="0" conaffinity="0"/>'
    message = _refusal(test, wrapped(box))
    test.assertRegex(message, "SolverIPC: MuJoCo's spatial tendon 0 wraps shape [0-9]+, a BOX; wrap it around a sphere")
    message = _refusal(test, wrapped(sphere, side='sidesite="in"'))
    test.assertRegex(message, "spatial tendon 0 has its side site [0-9]+ inside the shape [0-9]+ it wraps \\(an inside")
    for path, what in (
        ('<geom geom="g"/><site site="s0"/><site site="s1"/>', "a geom that is not between two sites at path entry 0"),
        ('<site site="s0"/><site site="s1"/><pulley divisor="2"/>', "a pulley at the end at path entry 2"),
        ('<site site="s0"/><site site="s1"/><pulley divisor="2"/><pulley divisor="2"/><site site="s1"/>'
         '<site site="s0"/>', "two pulleys in a row at path entry 3"),
        ('<site site="s0"/><site site="s1"/><pulley divisor="0"/><site site="s1"/><site site="s0"/>',
         "a pulley of divisor 0 at path entry 2"),
        ('<site site="s0"/><pulley divisor="2"/><site site="s1"/><site site="s0"/>',
         "a site whose neighbours are pulleys only at path entry 0"),
    ):  # fmt: skip
        with test.subTest(what):
            test.assertIn(f"SolverIPC: MuJoCo's spatial tendon 0 has {what}, which MuJoCo's compiler refuses;",
                          _refusal(test, wrapped(sphere, path)))  # fmt: skip
    test.assertIn("has a path of 1 entries; give it two at least", _refusal(test, wrapped(sphere, '<site site="s0"/>')))
    # what joint_terms refuses of every MuJoCo tendon: an actuator that drives it, a limit margin
    actuated = _ARM.replace("</tendon>", '</tendon><actuator><position tendon="t" kp="10"/></actuator>')
    test.assertIn("MuJoCo actuators that drive a tendon", _refusal(test, _model(device, actuated)))
    test.assertIn("has a limit margin of 0.01", _refusal(test, _model(device, _ARM, margin=0.01, range="0 1")))


@requires("uipc_warp")
class TestSolverIPCSpatialTendons(unittest.TestCase):
    """MuJoCo's spatial tendons: closed forms, coefficients, notifications, worlds and refusals."""


devices = get_cuda_test_devices()
for name, func in (
    ("test_a_box_hangs_over_a_cylinder", test_a_box_hangs_over_a_cylinder),
    ("test_a_pulley_halves_its_branch", test_a_pulley_halves_its_branch),
    ("test_coefficients", test_coefficients),
    ("test_notify", test_notify),
    ("test_worlds", test_worlds),
    ("test_grown_lists_keep_the_limit", test_grown_lists_keep_the_limit),
    ("test_friction_lowers_the_tolerances", test_friction_lowers_the_tolerances),
    ("test_a_tendon_of_the_world_alone", test_a_tendon_of_the_world_alone),
    ("test_refusals", test_refusals),
):
    add_function_test(TestSolverIPCSpatialTendons, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
