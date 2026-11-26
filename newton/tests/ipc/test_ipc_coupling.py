# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC as an entry of Newton's coupled solvers."""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.tests.ipc._fixtures import (
    _CONVERGED,
    _TIGHT,
    D_HAT,
    _box_body,
    _cloth,
    _frame_start,
    _hinged,
    _native_contacts,
    _restore,
    _rigid_rows,
    _solver_snapshot,
    _transform_points,
    _world_rows,
    _worlds_at,
    _write_particle,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires


def _joints_of(model, bodies):
    """The joints whose child is one of ``bodies`` (the free joints add_body gives them)."""
    return [int(j) for j in np.flatnonzero(np.isin(model.joint_child.numpy(), bodies))]


def _coupled_content(device, others=True):
    """SolverIPC's content of the coupled-solver tests, a box standing on the ground and a cloth falling onto it, with
    ``others`` between and after XPBD's sphere 2 m away and VBD's cloth 2 m away on the other side. Returns the model,
    the box, the sphere, the IPC cloth's particles and the VBD cloth's."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    sphere = None
    if others:
        sphere = builder.add_body(xform=wp.transform(wp.vec3(2.0, 0.0, 0.5), wp.quat_identity()), label="sphere")
        builder.add_shape_sphere(sphere, radius=0.1)
    box = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1005), wp.quat_identity()), label="box")
    builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
    _cloth(builder, 12, 0.025, 0.25, mass=1.0e-3, tri_k=1.0e3, edge_ke=1.0e-3)
    ipc_cloth = list(range(builder.particle_count))
    vbd_cloth = []
    if others:
        _cloth(builder, 8, 0.025, 0.3, center=(-2.0, 0.0), mass=1.0e-3, tri_k=1.0e3, edge_ke=1.0e-3)
        vbd_cloth = list(range(len(ipc_cloth), builder.particle_count))
        builder.color()
    return builder.finalize(device=device), box, sphere, ipc_cloth, vbd_cloth


def _full_layout_coupler():
    """``SolverCoupled`` in the layout it falls back to when it cannot compact an entry's view: views that hold every
    body, joint and particle of the model, the other entries' disabled."""
    from newton.solvers.experimental.coupled import SolverCoupled  # noqa: PLC0415

    class FullLayout(SolverCoupled):
        def _compact_entry_view_if_needed(self, view, cfg, *args):
            return None

    return FullLayout


def test_coupled_entry_steps_as_alone(test, device):
    """Step a SolverIPC entry of SolverCoupled as SolverIPC alone steps the same content.

    SolverIPC, as an entry of SolverCoupled, owns a cloth that falls onto a box on the ground, next to XPBD owning a
    sphere and VBD a second cloth. It steps as SolverIPC alone on the cloth, the box and the ground, in the compact
    views SolverCoupled builds and in the full layout it falls back to: over 40 steps at tight tolerances the particles'
    and the box's float64 states stay within 1e-7 m of the run alone. (Two runs of SolverIPC alone part by up to 2e-9 m
    over these steps, from the order of float atomics in the contact assembly, so no bound of 1e-10 holds.) The other
    entries' content is no part of SolverIPC's solve: VBD's cloth adds no triangle, its particles are fixed and in an
    element that meets nothing, and in the full layout XPBD's sphere is a kinematic body without surface.
    """
    from newton.solvers.experimental.coupled import SolverCoupled  # noqa: PLC0415

    alone, box0, _, cloth0, _ = _coupled_content(device, others=False)
    for layout, coupler in (("compact", SolverCoupled), ("full", _full_layout_coupler())):
        with test.subTest(layout=layout):
            model, box, sphere, ipc_cloth, vbd_cloth = _coupled_content(device)
            entries = [
                SolverCoupled.Entry(
                    "ipc",
                    lambda view: newton.solvers.SolverIPC(view, **_TIGHT),
                    bodies=[box],
                    joints=_joints_of(model, [box]),
                    particles=ipc_cloth,
                ),
                SolverCoupled.Entry(
                    "xpbd", newton.solvers.SolverXPBD, bodies=[sphere], joints=_joints_of(model, [sphere])
                ),
                SolverCoupled.Entry("vbd", newton.solvers.SolverVBD, particles=vbd_cloth),
            ]
            coupled = coupler(model, entries)
            ipc = coupled.solver("ipc")
            solo = newton.solvers.SolverIPC(alone, **_TIGHT)
            view = coupled.view("ipc")
            local = list(view.body_label).index("box")
            test.assertEqual(local, 0 if layout == "compact" else box)
            base, base0 = ipc._scene.layout().body_base[local], solo._scene.layout().body_base[box0]
            n = len(cloth0)
            s0, s1, t0, t1 = model.state(), model.state(), alone.state(), alone.state()
            worst = 0.0
            for _ in range(40):
                coupled.step(s0, s1, None, None, 0.01)
                s0, s1 = s1, s0
                solo.step(t0, t1, None, None, 0.01)
                t0, t1 = t1, t0
                x, x0 = ipc._engine.vm.x.numpy(), solo._engine.vm.x.numpy()
                worst = max(
                    worst, np.abs(x[:n] - x0[:n]).max(), np.abs(x[base : base + 4] - x0[base0 : base0 + 4]).max()
                )
            test.assertLess(worst, 1.0e-7, f"{layout}: {worst:.3g} m")
            np.testing.assert_allclose(s0.particle_q.numpy()[ipc_cloth], t0.particle_q.numpy()[cloth0], atol=1e-6)
            test.assertGreater(s0.particle_q.numpy()[vbd_cloth, 2].max(), -1.0)  # VBD's cloth stepped on its own
            # the other entries' content: no triangle or contact point of VBD's cloth, a sphere without surface
            tris, tris0 = (np.asarray(e._engine.surface["triangles"]).reshape(-1, 3) for e in (ipc, solo))
            test.assertEqual(len(tris), len(tris0))
            test.assertFalse(np.isin(tris, vbd_cloth).any())
            test.assertTrue(np.all(ipc._fixed_particle[vbd_cloth]))
            # the elements of the contact points and half planes, which VBD's cloth meets none of
            points = np.asarray(ipc._engine.contact_points, np.int64)
            held = np.unique(np.concatenate([ipc._cids[points], ipc._plane_cids]))
            test.assertFalse(ipc._enabled[np.ix_(np.unique(ipc._cids[vbd_cloth]), held)].any())
            if layout == "full":
                offsets = ipc._scene.layout().body_vertex_offsets
                test.assertTrue(ipc._kinematic_body[sphere])
                test.assertEqual(offsets[sphere + 1] - offsets[sphere], 0)
            ipc.check_status()


def _proxy_scene(device, gap=0.012):
    """The proxy scene of the coupling tests: a static support (top at z = 0.1), a 30 cm cloth ``gap`` above it, and a
    1 kg box ``gap`` above the cloth. Returns the model and the box."""
    builder = newton.ModelBuilder()
    builder.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()), hx=0.1, hy=0.1, hz=0.05)
    _cloth(builder, 12, 0.025, 0.1 + gap, mass=1.0e-3, tri_k=1.0e3, edge_ke=1.0e-3)
    box = _box_body(builder, (0.0, 0.0, 0.1 + 2.0 * gap + 0.05), (0.05, 0.05, 0.05))
    return builder.finalize(device=device), box


def _proxy_coupler(model, box, mode, **ipc_options):
    """XPBD owning the box as the source of a body proxy, SolverIPC owning the cloth as its destination, one
    iteration of ``mode`` ("lagged" or "staggered")."""
    from newton.solvers.experimental.coupled import SolverCoupledProxy  # noqa: PLC0415

    return SolverCoupledProxy(
        model,
        [
            SolverCoupledProxy.Entry("xpbd", newton.solvers.SolverXPBD, bodies=[box], joints=_joints_of(model, [box])),
            SolverCoupledProxy.Entry(
                "ipc",
                lambda view: newton.solvers.SolverIPC(view, **ipc_options),
                particles=list(range(model.particle_count)),
            ),
        ],
        SolverCoupledProxy.Config(
            proxies=[SolverCoupledProxy.Proxy(source="xpbd", destination="ipc", bodies=[box], mode=mode)]
        ),
    )


def test_proxy_box_rests_on_cloth(test, device):
    """Rest a proxy box on SolverIPC's cloth with staggered transfer and keep it on the cloth with lagged transfer.

    SolverCoupledProxy runs XPBD owning a 1 kg box and SolverIPC a cloth on a static support, the box SolverIPC's proxy,
    at 1/240 s with converged steps. With the staggered transfer, which syncs the source's pose at the end of its step,
    the box comes to rest on the cloth and the harvested feedback carries its weight within 1%, with no torque. With the
    lagged transfer, which syncs the pose at its start, the box keeps bouncing on the cloth: the feedback, one step
    late, of a contact this stiff does not settle; it stays on the cloth, and the feedback averages out to its weight.
    In every step of both, no edge of SolverIPC's engine crosses a triangle, no step fails, and the proxy's synced state
    keeps the friction of its contacts (no friction discard requested); its contact with the static support is disabled.
    """
    from uipc_warp.validation import SurfaceChecker  # noqa: PLC0415

    dt = 1.0 / 240.0
    for mode, steps in (("staggered", 240), ("lagged", 400)):
        with test.subTest(mode=mode):
            model, box = _proxy_scene(device)
            coupled = _proxy_coupler(model, box, mode, **_CONVERGED)
            ipc = coupled.solver("ipc")
            test.assertTrue(ipc._proxy_body.all())
            test.assertFalse(ipc._kinematic_body.any())
            # the proxy's shape does not meet the static support, and meets the cloth
            support, shape = 0, int(model.body_shapes[box][0])
            e = ipc._enabled
            test.assertFalse(e[ipc._cids[ipc._vertex_shape == shape][0], ipc._cids[ipc._vertex_shape == support][0]])
            test.assertTrue(e[ipc._cids[ipc._vertex_shape == shape][0], ipc._cids[0]])
            checker = SurfaceChecker.from_engine(ipc._engine)
            feedback = coupled.entry_state("xpbd", "input")
            weight = float(model.body_mass.numpy()[box]) * 9.81
            s0, s1 = model.state(), model.state()
            fz, hits, z = [], 0, []
            for _ in range(steps):
                coupled.step(s0, s1, None, None, dt)
                s0, s1 = s1, s0
                hits += checker.intersections(ipc._engine.vm.x)
                test.assertEqual(int(ipc._changed.numpy()[0]), 0)  # the proxy's synced state discards no friction
                fz.append(coupled._proxy_mappings[0].coupling_forces.numpy()[box])
                z.append(float(s0.body_q.numpy()[box, 2]))
            ipc.check_status()
            test.assertEqual(hits, 0)
            fz, z = np.array(fz), np.array(z)
            top = 0.1 + 0.05  # the box's centre resting on the support, without the cloth
            test.assertGreater(z.min(), top)
            test.assertLess(z[-100:].max(), top + 0.03)  # on the cloth, not thrown off
            if mode == "staggered":
                np.testing.assert_allclose(fz[-60:, 2], weight, rtol=0.01)  # at rest: the box's weight
                test.assertLess(np.abs(fz[-60:, :2]).max(), 0.01 * weight)
                test.assertLess(np.abs(fz[-60:, 3:]).max(), 0.01 * weight * 0.05)
                test.assertLess(np.abs(s0.body_qd.numpy()[box]).max(), 1e-3)
            else:  # over the last 300 steps the feedback's mean holds the box up
                test.assertAlmostEqual(float(fz[-300:, 2].mean()) / weight, 1.0, delta=0.02)
            test.assertTrue(np.isfinite(feedback.body_f.numpy()).all())


def _repeating_particles(device):
    """Four particles of 1 g with friction on the ground plane: one sliding, two meeting head on, one landing. Every
    particle's contact terms add into its gradient one at a time, so SolverIPC repeats these steps bit for bit."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for p, v in (((0.0, 0.0, 0.0105), (1.0, 0.0, 0.0)), ((0.3, 0.0, 0.0105), (0.5, 0.2, 0.0)),
                 ((0.4, 0.0, 0.0105), (-0.5, 0.0, 0.0)), ((0.8, 0.0, 0.05), (0.0, 0.0, 0.0))):  # fmt: skip
        builder.add_particle(wp.vec3(*p), wp.vec3(*v), 1.0e-3, radius=0.01)
    return builder.finalize(device=device)


def _engine_state(solver):
    """Where the last step of ``solver`` ended, as host arrays: libuipc-warp's float64 positions and velocities of every
    vertex (``engine.vm.x``, ``engine.vm.v``)."""
    e = solver._engine
    return {"x": e.vm.x.numpy().copy(), "v": e.vm.v.numpy().copy()}


def _two_passes(model, solver, steps, dt=0.01, capture=False, inputs=None, start=None):
    """``steps`` coupled steps of two passes each, with the coupler's protocol by hand: a write of the input state
    (``coupling_notify_input_state_update``), the first pass, a restart's notification, the second pass, both from the
    same ``state_in``; ``inputs(k, pass)`` may write the input state of a pass first. The solver is an entry whose
    coupler restarts (``coupling_notify_iteration_restarts(True)``). Captured: the two passes and the copy of the result
    into the input in one graph, replayed; ``start``: the state of the first step (the model's by default). Returns the
    (first, second) ``_engine_state`` of every step (of the first pass, captured, its positions) and the state."""
    solver.coupling_notify_iteration_restarts(True)
    s0, s1 = model.state(), model.state()
    if start is not None:
        s0.assign(start)
    flags = newton.StateFlags.ALL
    out, first = [], [None]

    def passes(k):
        solver.coupling_notify_input_state_update(s0, flags, dt=dt)
        if inputs is not None:
            inputs(k, 0, s0)
        solver.step(s0, s1, None, None, dt)
        if not capture:
            first[0] = _engine_state(solver)
        else:
            wp.copy(kept_x, solver._engine.vm.x)
        solver.coupling_notify_input_state_update(s0, flags, iteration_restart=True, dt=dt)
        if inputs is not None:
            inputs(k, 1, s0)
        solver.step(s0, s1, None, None, dt)
        s0.assign(s1)

    if capture:
        kept_x = wp.empty_like(solver._engine.vm.x)
        with wp.ScopedCapture(device=model.device) as cap:
            passes(0)
    for k in range(steps):
        if capture:
            wp.capture_launch(cap.graph)
            second = _engine_state(solver)
            out.append(({"x": kept_x.numpy().copy()}, second))
        else:
            passes(k)
            out.append((first[0], _engine_state(solver)))
    return out, s0


def test_restarted_passes_repeat(test, device):
    """Repeat the first pass of a step bit for bit in a pass after a restart given the same input.

    The coupler's protocol runs by hand (``_two_passes``): a pass after a restart starts where the first pass of the
    step started, from the float64 state with the friction candidates of the step's start, so that a pass with the
    first's input repeats it. On particles that repeat bit for bit, every second pass is its first pass bit for bit,
    eager and captured, and the run is SolverIPC's run without restarts, bit for bit; and so on a driven pendulum under
    ``integrator="bdf2"``, whose drive reads the joint coordinate of the step before, which the solver carries from step
    to step and restores with the step's start. Without restarts (the negative control) a second step from the same
    state_in starts from state_in's float32 rounding: the passes differ.
    """
    pendulum, pendulum_start, _ = _hinged(device, gravity=9.81, offset=(0.0, 0.3, 0.0), radius=0.05, target_ke=5.0,
                                          target_kd=0.1)  # fmt: skip
    pendulum_start.joint_q.assign(np.array([0.5], np.float32))  # swinging from 0.5 rad
    newton.eval_fk(pendulum, pendulum_start.joint_q, pendulum_start.joint_qd, pendulum_start)
    cases = (
        ("particles", _repeating_particles(device), None, {}),
        ("pendulum, bdf2", pendulum, pendulum_start, {"integrator": "bdf2"}),
    )
    for name, model, start, options in cases:
        alone = newton.solvers.SolverIPC(model, **options)
        a0, a1 = model.state(), model.state()
        if start is not None:
            a0.assign(start)
        plain = []
        for _ in range(30):
            alone.step(a0, a1, None, None, 0.01)
            a0, a1 = a1, a0
            plain.append(_engine_state(alone))
        for capture in (False, True):
            with test.subTest(name, captured=capture):
                solver = newton.solvers.SolverIPC(model, **options)
                passes, _ = _two_passes(model, solver, 30, capture=capture, start=start)
                for k, ((first, second), p) in enumerate(zip(passes, plain, strict=True)):
                    for key, a in first.items():
                        test.assertEqual(a.tobytes(), second[key].tobytes(), f"step {k}: {key} of the two passes")
                    for key, a in p.items():
                        test.assertEqual(a.tobytes(), second[key].tobytes(), f"step {k}: {key} against the plain run")
                solver.check_status()
    model = _repeating_particles(device)
    # the negative control: two steps from one state_in without restarts
    solver = newton.solvers.SolverIPC(model)
    s0, s1 = model.state(), model.state()
    moved = 0
    for _ in range(30):
        solver.step(s0, s1, None, None, 0.01)
        once = _engine_state(solver)
        solver.step(s0, s1, None, None, 0.01)
        moved += once["x"].tobytes() != _engine_state(solver)["x"].tobytes()
        s0.assign(s1)
    test.assertGreater(moved, 20)


def test_restart_restores_every_world(test, device):
    """Restore every world, the failed one too, to the step's start when a pass is restarted.

    Three independent worlds of a soft cube, a box and a particle (``_world_content``) run the coupler's protocol by
    hand. After ten steps the first pass of a step takes world 1's particle written through the ground, which fails
    world 1 alone (put back at its start, its friction and history requested anew for the next step); the second pass,
    restarted with the particle written back, starts every world from the step's start again, the failed world and its
    requests too, and takes the step of a solver that never failed, in every world within 1e-6 m. In every other step
    the two passes end within the rounding of the soft cube's sums, 1e-9 m.
    """
    shifts = (0.0, 0.03, 0.06)
    model = _worlds_at(device, shifts)
    rows = _world_rows(model, len(shifts))
    particle = int(rows[1][0][-1])
    heights = {}

    def inputs(k, attempt, state):
        if k == 10 and attempt == 0:
            heights[k] = float(state.particle_q.numpy()[particle, 2])
            _write_particle(state, particle, z=-0.01)
        elif k == 10:
            _write_particle(state, particle, z=heights[k])

    solver = newton.solvers.SolverIPC(model)
    clean = newton.solvers.SolverIPC(model)
    c0, c1 = model.state(), model.state()
    reference = []
    for _ in range(12):
        clean.step(c0, c1, None, None, 0.01)
        c0, c1 = c1, c0
        reference.append(c0.particle_q.numpy().copy())
    failed = []
    real = solver.step

    def step(*args, **kw):
        try:
            real(*args, **kw)
        except RuntimeError as e:  # the eager first pass of step 10 fails in world 1 alone
            failed.append(str(e))

    solver.step = step
    passes, state = _two_passes(model, solver, 12, inputs=inputs)
    test.assertEqual(len(failed), 1)
    test.assertRegex(failed[0], r"failed in 1 of its 3 worlds, world 1")
    for k, (first, second) in enumerate(passes):
        if k != 10:
            np.testing.assert_allclose(second["x"], first["x"], rtol=0.0, atol=1e-9, err_msg=f"step {k}")
    np.testing.assert_allclose(state.particle_q.numpy(), reference[-1], rtol=0.0, atol=1e-6)
    solver.check_status()  # the failed pass was replaced by its restart


def test_restart_replaces_the_status_of_a_pass(test, device):
    """Keep the status bits of the last pass of a step, not those of a failed pass before the restart.

    In captured passes of one step, a first pass whose input gives a particle a velocity that is not a number fails (a
    NONFINITE bit, as the same pass alone reports), and its restart from the step's start with a clean input does not;
    the status bits an entry keeps are the last pass's, so check_status() reports nothing.
    """
    model = _repeating_particles(device)
    solver = newton.solvers.SolverIPC(model)
    solver.coupling_notify_iteration_restarts(True)
    flags = newton.StateFlags.PARTICLE_Q | newton.StateFlags.PARTICLE_QD
    s0, s1, bad = model.state(), model.state(), model.state()
    solver.step(s0, s1, None, None, 0.01)  # loads the state and the modules
    s0.assign(s1)
    bad.assign(s0)
    _write_particle(bad, 0, qd=(np.nan, 0.0, 0.0))
    with wp.ScopedCapture(device=device) as cap:
        solver.coupling_notify_input_state_update(bad, flags, dt=0.01)
        solver.step(bad, s1, None, None, 0.01)
        solver.coupling_notify_input_state_update(s0, flags, iteration_restart=True, dt=0.01)
        solver.step(s0, s1, None, None, 0.01)
    wp.capture_launch(cap.graph)
    test.assertFalse(solver.check_status())
    # the same first pass alone is a failure the kept bits report
    with wp.ScopedCapture(device=device) as cap:
        solver.coupling_notify_input_state_update(bad, flags, dt=0.01)
        solver.step(bad, s1, None, None, 0.01)
    wp.capture_launch(cap.graph)
    with test.assertRaisesRegex(RuntimeError, "NONFINITE"):
        solver.check_status()


def test_admm_with_two_iterations(test, device):
    """Step SolverIPC's entry twice per step under ``SolverCoupledADMM`` as SolverIPC alone, bit for bit.

    ``SolverCoupledADMM`` runs two iterations, SolverIPC owning four particles with friction on the ground and XPBD a
    box falling 3 m away: every step solves SolverIPC's entry twice, the second from the step's start restored on the
    device. With nothing coupling the entries both passes have the same input, and the second ends where the first did,
    bit for bit, and the entry steps as SolverIPC alone, bit for bit, over 30 steps; the box falls on its own. A
    restart's notification from a coupler that did not announce restarts is refused, and a reset's is not.
    """
    from newton.solvers.experimental.coupled import SolverCoupledADMM  # noqa: PLC0415

    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for p, v in (((0.0, 0.0, 0.0105), (1.0, 0.0, 0.0)), ((0.3, 0.0, 0.0105), (0.5, 0.2, 0.0)),
                 ((0.4, 0.0, 0.0105), (-0.5, 0.0, 0.0)), ((0.8, 0.0, 0.05), (0.0, 0.0, 0.0))):  # fmt: skip
        builder.add_particle(wp.vec3(*p), wp.vec3(*v), 1.0e-3, radius=0.01)
    box = _box_body(builder, (3.0, 0.0, 0.5), (0.05, 0.05, 0.05))
    model = builder.finalize(device=device)
    coupled = SolverCoupledADMM(
        model,
        [
            SolverCoupledADMM.Entry("ipc", newton.solvers.SolverIPC, particles=list(range(model.particle_count))),
            SolverCoupledADMM.Entry("xpbd", newton.solvers.SolverXPBD, bodies=[box], joints=_joints_of(model, [box])),
        ],
        SolverCoupledADMM.Config(iterations=2),
    )
    ipc = coupled.solver("ipc")
    starts, real = [], ipc.step

    def step(*args, **kw):
        real(*args, **kw)
        starts.append(_engine_state(ipc))

    ipc.step = step
    alone_model = _repeating_particles(device)
    alone = newton.solvers.SolverIPC(alone_model)
    s0, s1, a0, a1 = model.state(), model.state(), alone_model.state(), alone_model.state()
    z0 = float(s0.body_q.numpy()[box, 2])
    for k in range(30):
        starts.clear()
        coupled.step(s0, s1, None, None, 0.01)
        s0, s1 = s1, s0
        alone.step(a0, a1, None, None, 0.01)
        a0, a1 = a1, a0
        test.assertEqual(len(starts), 2)
        for name in ("x", "v"):
            test.assertEqual(starts[0][name].tobytes(), starts[1][name].tobytes(), f"step {k}: {name}")
        x, xa = ipc._engine.vm.x.numpy(), alone._engine.vm.x.numpy()
        test.assertEqual(x.tobytes(), xa.tobytes(), f"step {k}")
    ipc.check_status()
    test.assertLess(float(s0.body_q.numpy()[box, 2]), z0 - 0.3)
    # a coupler that did not announce restarts
    solver = newton.solvers.SolverIPC(alone_model)
    state = alone_model.state()
    solver.coupling_notify_iteration_restarts(False)
    solver.coupling_notify_input_state_update(state, newton.StateFlags.ALL, iteration_restart=True, dt=0.0)
    with test.assertRaisesRegex(NotImplementedError, "did not announce restarts"):
        solver.coupling_notify_input_state_update(state, newton.StateFlags.ALL, iteration_restart=True, dt=0.01)


def test_admm_leaves_attachments_to_the_entry(test, device):
    """Leave a particle-body attachment whose particle and body one SolverIPC entry owns to that entry.

    SolverCoupledADMM leaves such an attachment to the entry as it leaves them to SolverVBD
    (CouplingInterface.coupling_supports_particle_body_attachments). A particle of 1 kg moving at 1 m/s on an attachment
    to a free body of 1 kg at its origin, without gravity, steps as SolverIPC alone steps it, finite (no warning that
    the attachment applies no force) and hard (which the coupler refused), and pulls the body along; the hard attachment
    holds the particle at the body's origin.
    """
    import math  # noqa: PLC0415

    from newton.solvers.experimental.coupled import SolverCoupledADMM  # noqa: PLC0415

    def pair(stiffness):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        inertia = wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1)
        body = builder.add_body(mass=1.0, inertia=inertia, lock_inertia=True)
        particle = builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.0), vel=wp.vec3(1.0, 0.0, 0.0), mass=1.0)
        builder.add_attachment_particle_body(particle, body, stiffness=stiffness)
        return builder.finalize(device=device)

    def run(model, solver):
        s0, s1 = model.state(), model.state()
        for _ in range(20):
            solver.step(s0, s1, None, None, 0.01)
            s0, s1 = s1, s0
        return s0.body_q.numpy()[0, :3].astype(np.float64), s0.particle_q.numpy()[0].astype(np.float64)

    for stiffness in (2.0e4, math.inf):
        with test.subTest(stiffness=stiffness):
            model = pair(stiffness)
            entry = SolverCoupledADMM.Entry(
                "ipc", newton.solvers.SolverIPC, bodies=[0], particles=[0], joints=_joints_of(model, [0])
            )
            with test.assertNoLogs("newton._src.solvers.coupled.solver_coupled_admm", level="WARNING"):
                coupled = SolverCoupledADMM(model, [entry], SolverCoupledADMM.Config(iterations=1))
            body, particle = run(model, coupled)
            alone = pair(stiffness)
            body_alone, particle_alone = run(alone, newton.solvers.SolverIPC(alone))
            np.testing.assert_allclose(body, body_alone, rtol=0.0, atol=1e-9)
            np.testing.assert_allclose(particle, particle_alone, rtol=0.0, atol=1e-9)
            test.assertGreater(body[0], 0.05)  # the attachment pulls the body along
            if math.isinf(stiffness):
                test.assertLess(np.linalg.norm(particle - body), 1e-3)


def test_proxy_passes_converge(test, device):
    """Converge the passes of ``SolverCoupledProxy`` so that a proxy box rests on cloth with its weight as feedback.

    The scene is XPBD's 1 kg box on SolverIPC's cloth over a static support (``_proxy_scene``), at 1/240 s with
    converged steps and several passes per step. Every pass of SolverIPC's entry starts from the step's start, and the
    coupler iterates the step's feedback: the source steps again with the feedback of the pass before, so the feedback
    is no longer one step late.

    Lagged transfer, four passes: the passes converge (the last two differ by less than 0.1% of the weight in each of
    the last 60 steps) and the box comes to rest with its weight as feedback within 0.1%, where one pass keeps it
    bouncing (``test_proxy_box_rests_on_cloth``). Staggered transfer: the feedback of a pass undoes the motion that made
    it, so the passes alternate unless relaxed; with a relaxation of 0.5, two passes rest the box with its weight within
    0.1%. No crossing and no failure in any step.
    """
    from uipc_warp.validation import SurfaceChecker  # noqa: PLC0415

    from newton.solvers.experimental.coupled import SolverCoupledProxy  # noqa: PLC0415

    dt = 1.0 / 240.0
    for mode, iterations, relaxation in (("lagged", 4, 1.0), ("staggered", 2, 0.5)):
        with test.subTest(mode=mode):
            model, box = _proxy_scene(device)
            coupled = SolverCoupledProxy(
                model,
                [
                    SolverCoupledProxy.Entry(
                        "xpbd", newton.solvers.SolverXPBD, bodies=[box], joints=_joints_of(model, [box])
                    ),
                    SolverCoupledProxy.Entry(
                        "ipc",
                        lambda view: newton.solvers.SolverIPC(view, **_CONVERGED),
                        particles=list(range(model.particle_count)),
                    ),
                ],
                SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source="xpbd", destination="ipc", bodies=[box], mode=mode, proxy_relaxation=relaxation
                        )
                    ],
                    iterations=iterations,
                ),
            )
            ipc = coupled.solver("ipc")
            checker = SurfaceChecker.from_engine(ipc._engine)
            feedback = coupled._proxy_mappings[0].coupling_forces
            passes = []

            def step_proxy(*args, real=coupled._step_proxy, passes=passes, feedback=feedback, box=box, **kw):
                real(*args, **kw)
                passes.append(float(feedback.numpy()[box, 2]))

            coupled._step_proxy = step_proxy
            weight = float(model.body_mass.numpy()[box]) * 9.81
            s0, s1 = model.state(), model.state()
            fz, z, hits = [], [], 0
            for _ in range(240):
                passes.clear()
                coupled.step(s0, s1, None, None, dt)
                s0, s1 = s1, s0
                test.assertEqual(len(passes), iterations)
                fz.append(list(passes))
                z.append(float(s0.body_q.numpy()[box, 2]))
                hits += checker.intersections(ipc._engine.vm.x)
            ipc.check_status()
            test.assertEqual(hits, 0)
            fz = np.array(fz) / weight
            np.testing.assert_allclose(fz[-60:, -1], 1.0, rtol=0.0, atol=1e-3)
            test.assertLess(np.abs(fz[-60:, -1] - fz[-60:, -2]).max(), 1e-3)
            test.assertLess(np.ptp(z[-60:]), 1e-5)  # at rest
            test.assertGreater(min(z), 0.15)  # on the cloth, above the support


def _coupled_boxes(device):
    """``_boxes_apart`` with XPBD's sphere 2 m away: SolverCoupled with SolverIPC owning the box above the static box,
    and the pose that rests it exactly on the static box (touching contact)."""
    from newton.solvers.experimental.coupled import SolverCoupled  # noqa: PLC0415

    builder = newton.ModelBuilder()
    builder.add_shape_box(
        -1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.125), wp.quat_identity()), hx=0.125, hy=0.125, hz=0.125
    )
    box = _box_body(builder, (0.0, 0.0, 0.376), (0.125, 0.125, 0.125))
    sphere = builder.add_body(xform=wp.transform(wp.vec3(2.0, 0.0, 0.5), wp.quat_identity()))
    builder.add_shape_sphere(sphere, radius=0.1)
    model = builder.finalize(device=device)
    coupled = SolverCoupled(
        model,
        [
            SolverCoupled.Entry("ipc", newton.solvers.SolverIPC, bodies=[box], joints=_joints_of(model, [box])),
            SolverCoupled.Entry("xpbd", newton.solvers.SolverXPBD, bodies=[sphere], joints=_joints_of(model, [sphere])),
        ],
    )
    touching = model.body_q.numpy()
    touching[box, 2] = 0.375
    return model, coupled, box, touching


def test_coupled_status_of_captured_steps(test, device):
    """Keep the status bits of captured coupled steps for check_status() without a container.

    SolverCoupled steps SolverIPC without observables, so SolverIPC keeps the status bits of its steps for
    check_status() without a container. A captured coupled step replayed from a clean state reports nothing; replayed
    from a state that rests the box in touching contact it reports THICKNESS_VIOLATED and raises, and the read clears
    the bits. A solver that is no entry of a coupled solver refuses check_status() without a container.
    """
    model, coupled, box, touching = _coupled_boxes(device)
    ipc = coupled.solver("ipc")
    s0, s1 = model.state(), model.state()
    coupled.step(s0, s1, None, None, 0.01)  # loads the state and the modules
    s0.assign(s1)
    with wp.ScopedCapture(device=device) as capture:
        coupled.step(s0, s1, None, None, 0.01)
    wp.capture_launch(capture.graph)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        test.assertFalse(ipc.check_status())
    q = s0.body_q.numpy()
    q[box] = touching[box]
    s0.body_q.assign(q)
    wp.capture_launch(capture.graph)
    with test.assertRaisesRegex(RuntimeError, "failed on THICKNESS_VIOLATED"):
        ipc.check_status()
    test.assertFalse(ipc.check_status())  # cleared by the read
    alone = newton.solvers.SolverIPC(model)
    with test.assertRaisesRegex(ValueError, "or none for an entry of a coupled solver"):
        alone.check_status()


def test_coupled_reset_is_the_solvers_reset(test, device):
    """Reset SolverIPC through SolverCoupled as SolverIPC's own reset with the coupler's world mask.

    The reset's own notification does nothing. From one state of a box sliding on a static box, the next step keeps its
    friction candidates after a reset whose mask leaves out the global entities (SolverIPC's content is global), and
    records none after a reset that selects them and after a reset without a mask.
    """
    model, coupled, box, _ = _coupled_boxes(device)
    ipc = coupled.solver("ipc")
    s0, s1 = model.state(), model.state()
    qd = s0.body_qd.numpy()
    qd[box, 0] = 0.2  # sliding along x
    s0.body_qd.assign(qd)
    for _ in range(30):  # lands on the static box and slides
        coupled.step(s0, s1, None, None, 0.01)
        s0, s1 = s1, s0
    snapshot = _solver_snapshot(ipc)
    export = ipc._uipc_warp.arrays.ContactForces(ipc._engine)  # every case recovers this engine
    counts = {}
    for case, mask in (("none", None), ("world 0", [True, False]), ("global", [False, True]), ("all", "all")):
        _restore(ipc, snapshot)
        state = model.state()
        state.assign(s0)
        if mask == "all":
            coupled.reset(state)
        elif mask is not None:
            coupled.reset(state, world_mask=wp.array(mask, dtype=wp.bool, device=device))
        coupled.step(state, s1, None, None, 0.01)
        counts[case] = sum(len(p) for p in _frame_start(ipc, export)["lists"].values())
    test.assertGreater(counts["none"], 0)
    test.assertEqual(counts["world 0"], counts["none"])
    test.assertEqual(counts["global"], 0)
    test.assertEqual(counts["all"], 0)


def test_coupled_entry_of_independent_worlds(test, device):
    """Solve the worlds of a SolverCoupled entry's view as independent worlds and keep each world's status bits.

    SolverIPC is an entry of SolverCoupled on a model of two worlds, each a box above the global static box owned by
    SolverIPC (world 0's falling, world 1's 1 mm above it) and an XPBD sphere 2 m away: the entry's view holds the two
    worlds and solves them as independent worlds.

    It keeps the status bits of every world. A captured coupled step replayed from a clean state reports nothing;
    replayed from a state that rests world 1's box in touching contact it reports THICKNESS_VIOLATED in world 1 alone,
    check_status() without a container names world 1 and the read clears the bits, and world 0's box takes its step.
    """
    from newton.solvers.experimental.coupled import SolverCoupled  # noqa: PLC0415

    builder = newton.ModelBuilder()
    builder.add_shape_box(
        -1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.125), wp.quat_identity()), hx=0.125, hy=0.125, hz=0.125
    )
    boxes, spheres = [], []
    for height in (0.6, 0.376):  # world 0's box falls for 20 steps, world 1's lands at once
        builder.begin_world()
        boxes.append(_box_body(builder, (0.0, 0.0, height), (0.125, 0.125, 0.125)))
        sphere = builder.add_body(xform=wp.transform(wp.vec3(2.0, 0.0, 0.5), wp.quat_identity()))
        builder.add_shape_sphere(sphere, radius=0.1)
        spheres.append(sphere)
        builder.end_world()
    model = builder.finalize(device=device)
    coupled = SolverCoupled(
        model,
        [
            SolverCoupled.Entry("ipc", newton.solvers.SolverIPC, bodies=boxes, joints=_joints_of(model, boxes)),
            SolverCoupled.Entry("xpbd", newton.solvers.SolverXPBD, bodies=spheres, joints=_joints_of(model, spheres)),
        ],
    )
    ipc = coupled.solver("ipc")
    test.assertTrue(ipc._independent)
    test.assertEqual(ipc._sticky_status.shape[0], ipc.model.world_count + 1)
    s0, s1 = model.state(), model.state()
    coupled.step(s0, s1, None, None, 0.01)  # loads the state and the modules
    s0.assign(s1)
    with wp.ScopedCapture(device=device) as capture:
        coupled.step(s0, s1, None, None, 0.01)
    wp.capture_launch(capture.graph)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        test.assertFalse(ipc.check_status())
    s0.assign(s1)
    q = s0.body_q.numpy()
    q[boxes[1]] = (0.0, 0.0, 0.375, 0.0, 0.0, 0.0, 1.0)  # world 1's box resting exactly on the static box: touching
    s0.body_q.assign(q)
    wp.capture_launch(capture.graph)
    with test.assertRaisesRegex(RuntimeError, r"failed in world 1 on THICKNESS_VIOLATED; the result of those worlds"):
        ipc.check_status()
    test.assertFalse(ipc.check_status())  # cleared by the read
    after = s1.body_q.numpy()
    np.testing.assert_allclose(after[boxes[1]], q[boxes[1]], rtol=0.0, atol=1e-7)  # world 1 back at its start
    test.assertLess(after[boxes[0], 2], q[boxes[0], 2] - 1e-3)  # world 0's box fell on


def test_proxies_refuse_bdf2_and_al_ipc(test, device):
    """Refuse integrator="bdf2" and contact_constitution="al-ipc" in an entry with proxies.

    integrator="bdf2" is refused because the coupler writes a proxy's state at every step, and
    contact_constitution="al-ipc" because it has no contact forces to harvest the feedback from; without proxies an
    entry takes both. SolverIPC reads no Contacts: it takes full-surface soft contacts and returns proxy contacts
    unchanged.
    """
    from newton.solvers.experimental.coupled import SolverCoupled  # noqa: PLC0415

    model, box = _proxy_scene(device)
    for options, pattern in (({"integrator": "bdf2"}, "bdf2"), ({"contact_constitution": "al-ipc"}, "al-ipc")):
        with test.subTest(**options):
            with test.assertRaisesRegex(NotImplementedError, pattern):
                _proxy_coupler(model, box, "lagged", **options)
    coupled = SolverCoupled(
        model,
        [
            SolverCoupled.Entry("xpbd", newton.solvers.SolverXPBD, bodies=[box], joints=_joints_of(model, [box])),
            SolverCoupled.Entry(
                "ipc",
                lambda view: newton.solvers.SolverIPC(view, integrator="bdf2"),
                particles=list(range(model.particle_count)),
            ),
        ],
    )
    ipc = coupled.solver("ipc")
    test.assertTrue(ipc.coupling_supports_full_surface_soft_contacts())
    contacts = newton.CollisionPipeline(model).contacts()
    test.assertIs(ipc.coupling_prepare_proxy_contacts(model.state(), contacts), contacts)


def test_proxy_particles_rest_on_a_body(test, device):
    """Rest proxy particles on SolverIPC's box, each harvesting its weight as its feedback.

    SolverCoupledProxy runs XPBD owning four particles and SolverIPC a box standing on the ground, the particles
    SolverIPC's proxies, staggered at 1/240 s with converged steps: the particles come to rest on the box, each
    harvesting its weight as its feedback within 1%, the box at rest on the ground. A proxy particle meets the box and
    neither the ground, a static shape, nor another proxy particle.
    """
    from newton.solvers.experimental.coupled import SolverCoupledProxy  # noqa: PLC0415

    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    box = _box_body(builder, (0.0, 0.0, 0.1005), (0.1, 0.1, 0.1))
    for x, y in ((-0.05, -0.05), (0.05, -0.05), (-0.05, 0.05), (0.05, 0.05)):
        builder.add_particle(wp.vec3(x, y, 0.2355), wp.vec3(0.0, 0.0, 0.0), mass=0.01, radius=0.01)
    model = builder.finalize(device=device)
    particles = list(range(model.particle_count))
    coupled = SolverCoupledProxy(
        model,
        [
            SolverCoupledProxy.Entry("xpbd", newton.solvers.SolverXPBD, particles=particles),
            SolverCoupledProxy.Entry(
                "ipc",
                lambda view: newton.solvers.SolverIPC(view, **_CONVERGED),
                bodies=[box],
                joints=_joints_of(model, [box]),
            ),
        ],
        SolverCoupledProxy.Config(
            proxies=[SolverCoupledProxy.Proxy(source="xpbd", destination="ipc", particles=particles, mode="staggered")]
        ),
    )
    ipc = coupled.solver("ipc")
    test.assertTrue(ipc._proxy_particle.all())
    element = ipc._cids[particles]
    test.assertEqual(len(set(element.tolist())), 1)
    box_element = ipc._cids[ipc._vertex_shape == int(model.body_shapes[box][0])][0]
    ground = ipc._plane_cids[0]
    test.assertTrue(ipc._enabled[element[0], box_element])
    test.assertFalse(ipc._enabled[element[0], ground])
    test.assertFalse(ipc._enabled[element[0], element[0]])
    s0, s1 = model.state(), model.state()
    for _ in range(240):
        coupled.step(s0, s1, None, None, 1.0 / 240.0)
        s0, s1 = s1, s0
    ipc.check_status()
    feedback = coupled._proxy_particle_mappings[0].coupling_forces.numpy()[particles]
    np.testing.assert_allclose(feedback[:, 2], 0.01 * 9.81, rtol=0.01)
    test.assertLess(np.abs(feedback[:, :2]).max(), 1e-3 * 0.0981)
    z = s0.particle_q.numpy()[particles, 2]
    top = float(s0.body_q.numpy()[box, 2]) + 0.1
    test.assertTrue(np.all(z > top + 0.01))  # resting on the box, a radius above its top
    test.assertTrue(np.all(z < top + 0.01 + D_HAT))
    test.assertLess(np.abs(s0.particle_qd.numpy()[particles]).max(), 1e-3)


def test_proxy_feedback_is_the_contact_force(test, device):
    """Harvest as a proxy's feedback the force of the step's contact terms on it at the end of the step.

    A view (a coupled solver's view, built by hand) holds a box resting on a larger box on the ground and a particle
    resting on the larger box, the box and the particle SolverIPC's proxies. After 120 steps at tight tolerances,
    coupling_harvest_proxy_wrenches gives the proxy box the forces of libuipc-warp's records on its vertices, summed
    with their torques about its centre of mass (to float32 rounding), the net force of the step's CONTACT_F rows on it,
    and its weight within 1e-3 with no torque; coupling_harvest_proxy_particle_forces gives the particle the records'
    forces on it, minus the soft rows' force on the larger box by it, and its weight. Both zero their outputs first and
    write no other index.
    """
    from newton.solvers.experimental.coupled import ModelView  # noqa: PLC0415

    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    _box_body(builder, (0.0, 0.0, 0.1005), (0.2, 0.2, 0.1))  # the base, which the proxies rest on
    top = _box_body(builder, (0.05, 0.0, 0.2535), (0.05, 0.05, 0.05))
    builder.add_particle(wp.vec3(-0.1, 0.0, 0.2135), wp.vec3(0.0, 0.0, 0.0), mass=0.02, radius=0.01)
    model = builder.finalize(device=device)
    view = ModelView(model, "ipc")
    view.mark_proxy_bodies(wp.array([top], dtype=int, device=device))
    view.mark_proxy_particles(wp.array([0], dtype=int, device=device))
    solver = newton.solvers.SolverIPC(view, **_CONVERGED)
    contacts = _native_contacts(solver, model)
    observables = solver.observables({solver.ObservableKind.CONTACT_F, solver.ObservableKind.STATUS}, contacts=contacts)
    dt = 1.0 / 120.0
    s0, s1 = view.state(), view.state()
    for _ in range(120):
        solver.step(s0, s1, None, contacts, dt, observables=observables)
        s0, s1 = s1, s0
    solver.check_status(observables)
    # the harvest of the last step, which wrote s0
    proxy_of_body = wp.array([-1, 7], dtype=int, device=device)  # the proxy box at the coupler's index 7
    out = wp.full(9, wp.spatial_vector(1.0, 1.0, 1.0, 1.0, 1.0, 1.0), dtype=wp.spatial_vector, device=device)
    solver.coupling_harvest_proxy_wrenches(
        proxy_of_body, out, body_qd_before=s0.body_qd, state=s1, state_out=s0, contacts=contacts, dt=dt
    )
    wrench = out.numpy()
    test.assertTrue(np.all(np.delete(wrench, 7, axis=0) == 0.0))
    # the records on the proxy box's vertices
    layout = solver._scene.layout()
    first, last = layout.body_vertex_offsets[top], layout.body_vertex_offsets[top + 1]
    x = solver._engine.vm.x.numpy()
    com = _transform_points(s0.body_q.numpy()[top], model.body_com.numpy()[top : top + 1])[0]
    force, torque = np.zeros(3), np.zeros(3)
    particle_force = np.zeros(3)
    for name, (vertices, forces) in solver._contact_export.numpy().items():
        count = 1 if name.startswith("PH") else (4 if name[:2] in ("PT", "EE") else 3 if name[:2] == "PE" else 2)
        for k in range(count):
            v, f = vertices[:, k], forces[:, k].astype(np.float64)
            on = (v >= first) & (v < last)
            force += f[on].sum(axis=0)
            torque += np.cross(x[v[on]] - com, f[on]).sum(axis=0)
            particle_force += f[v == 0].sum(axis=0)
    np.testing.assert_allclose(wrench[7, :3], force, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(wrench[7, 3:], torque, atol=1e-5)
    # the CONTACT_F rows on the proxy box: rows where its shape is shape 0 push it, rows where it is shape 1 the reverse
    shape = int(model.body_shapes[top][0])
    s0_, s1_, rows = _rigid_rows(contacts, observables)
    net = rows[s0_ == shape, :3].sum(axis=0) - rows[s1_ == shape, :3].sum(axis=0)
    np.testing.assert_allclose(wrench[7, :3], net, rtol=1e-4, atol=1e-4)
    weight = float(model.body_mass.numpy()[top]) * 9.81
    np.testing.assert_allclose(wrench[7, :3], [0.0, 0.0, weight], atol=1e-3 * weight)
    test.assertLess(np.abs(wrench[7, 3:]).max(), 1e-3 * weight * 0.05)
    # the proxy particle: the records' force on it, minus the soft rows' force on the base by it
    proxy_of_particle = wp.array([3], dtype=int, device=device)
    out = wp.full(5, wp.vec3(1.0, 1.0, 1.0), dtype=wp.vec3, device=device)
    solver.coupling_harvest_proxy_particle_forces(
        proxy_of_particle, out, particle_qd_before=s0.particle_qd, state=s1, state_out=s0, contacts=contacts, dt=dt
    )
    got = out.numpy()
    test.assertTrue(np.all(np.delete(got, 3, axis=0) == 0.0))
    np.testing.assert_allclose(got[3], particle_force, rtol=1e-5, atol=1e-6)
    n = min(int(contacts.soft_contact_count.numpy()[0]), contacts.soft_contact_max)
    mine = contacts.soft_contact_particle.numpy()[:n] == 0
    by_rows = -observables.contact_f.numpy()[contacts.rigid_contact_max : contacts.rigid_contact_max + n][mine, :3].sum(
        axis=0
    )
    np.testing.assert_allclose(got[3], by_rows, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(got[3], [0.0, 0.0, 0.02 * 9.81], atol=1e-3 * 0.02 * 9.81)


def test_feedback_of_captured_steps(test, device):
    """Harvest the proxies' feedback of the last replay when an entry's step is captured alone.

    A coupled solver may capture an entry's step alone and harvest the proxies' feedback outside the graph, after every
    replay. The harvest reads the records of the last replay: the solver counts its steps on the device and brings the
    records up to date there. A proxy box resting on a box is pressed down by a body force 5 N larger at each of four
    steps: the wrenches harvested after each of four replays of a captured step are those harvested after each of four
    eager steps, from the same rest, to 1e-3 of the largest force, while each step's force differs from that of the step
    before by more than ten times that.
    """
    from newton.solvers.experimental.coupled import ModelView  # noqa: PLC0415

    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    _box_body(builder, (0.0, 0.0, 0.1005), (0.2, 0.2, 0.1))
    top = _box_body(builder, (0.0, 0.0, 0.2535), (0.05, 0.05, 0.05))
    model = builder.finalize(device=device)
    view = ModelView(model, "ipc")
    view.mark_proxy_bodies(wp.array([top], dtype=int, device=device))
    dt = 1.0 / 120.0
    proxy_of_body = wp.array([-1, 0], dtype=int, device=device)

    def at_rest():
        solver = newton.solvers.SolverIPC(view, **_CONVERGED)
        s0, s1 = view.state(), view.state()
        for _ in range(120):
            solver.step(s0, s1, None, None, dt)
            s0, s1 = s1, s0
        return solver, s0, s1

    def press(state, k):
        force = np.zeros((model.body_count, 6), np.float32)
        force[top, 2] = -5.0 * k
        state.body_f.assign(force)

    def harvest(solver, state, state_out):
        out = wp.zeros(1, dtype=wp.spatial_vector, device=device)
        solver.coupling_harvest_proxy_wrenches(
            proxy_of_body, out, body_qd_before=state.body_qd, state=state, state_out=state_out, contacts=None, dt=dt
        )
        return out.numpy()[0].astype(np.float64)

    solver, s0, s1 = at_rest()
    eager = []
    for k in range(1, 5):
        press(s0, k)
        solver.step(s0, s1, None, None, dt)
        eager.append(harvest(solver, s0, s1))
        s0, s1 = s1, s0
    eager = np.array(eager)
    tolerance = 1e-3 * np.linalg.norm(eager[:, :3], axis=1).max()
    change = np.abs(np.diff(eager[:, :3], axis=0)).max(axis=1)
    test.assertGreater(change.min(), 10.0 * tolerance, f"the forces of the eager steps: {eager[:, :3].tolist()}")

    solver, s0, s1 = at_rest()
    with wp.ScopedCapture(device=device) as capture:
        solver.step(s0, s1, None, None, dt)
    replayed = []
    for k in range(1, 5):
        press(s0, k)
        wp.capture_launch(capture.graph)
        replayed.append(harvest(solver, s0, s1))
        wp.copy(s0.body_q, s1.body_q)
        wp.copy(s0.body_qd, s1.body_qd)
    np.testing.assert_allclose(np.array(replayed), eager, rtol=0.0, atol=tolerance)


@requires("uipc_warp")
class TestSolverIPCCoupling(unittest.TestCase):
    """SolverIPC as an entry of Newton's coupled solvers: its view of the model, proxies and their feedback, the status
    of its steps and the couplers it refuses."""


devices = get_cuda_test_devices()
for name, func in (
    ("test_coupled_entry_steps_as_alone", test_coupled_entry_steps_as_alone),
    ("test_proxy_box_rests_on_cloth", test_proxy_box_rests_on_cloth),
    ("test_proxy_particles_rest_on_a_body", test_proxy_particles_rest_on_a_body),
    ("test_restarted_passes_repeat", test_restarted_passes_repeat),
    ("test_restart_restores_every_world", test_restart_restores_every_world),
    ("test_restart_replaces_the_status_of_a_pass", test_restart_replaces_the_status_of_a_pass),
    ("test_admm_with_two_iterations", test_admm_with_two_iterations),
    ("test_admm_leaves_attachments_to_the_entry", test_admm_leaves_attachments_to_the_entry),
    ("test_proxy_passes_converge", test_proxy_passes_converge),
    ("test_coupled_status_of_captured_steps", test_coupled_status_of_captured_steps),
    ("test_coupled_reset_is_the_solvers_reset", test_coupled_reset_is_the_solvers_reset),
    ("test_proxies_refuse_bdf2_and_al_ipc", test_proxies_refuse_bdf2_and_al_ipc),
    ("test_proxy_feedback_is_the_contact_force", test_proxy_feedback_is_the_contact_force),
    ("test_feedback_of_captured_steps", test_feedback_of_captured_steps),
    ("test_coupled_entry_of_independent_worlds", test_coupled_entry_of_independent_worlds),
):
    add_function_test(TestSolverIPCCoupling, name, func, devices=devices)


def _thrown_proxy(device, proxy=True):
    """A coupled solver's view built by hand: a 1 kg box (0.1 m), SolverIPC's proxy unless ``proxy`` is False, 2 mm
    above a 32 kg box resting on the ground. Returns the view and the box."""
    from newton.solvers.experimental.coupled import ModelView  # noqa: PLC0415

    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    _box_body(builder, (0.0, 0.0, 0.1005), (0.2, 0.2, 0.1))
    top = _box_body(builder, (0.02, 0.01, 0.2525), (0.05, 0.05, 0.05))
    view = ModelView(builder.finalize(device=device), "ipc")
    if proxy:
        view.mark_proxy_bodies(wp.array([top], dtype=int, device=device))
    return view, top


def test_proxy_entry_takes_converged_steps(test, device):
    """Take converged steps by default in an entry with proxies, so that the harvested feedback is the converged one.

    An entry with proxies takes converged steps unless told otherwise (newton_velocity_tol 5e-4 m/s,
    newton_transrate_tol 1e-3 1/s, semi_implicit False), so that a proxy's feedback, the force of the step's contact
    terms at its end, is the converged one. A proxy box is thrown at 0.5 m/s onto a box, in steps of 1/60 s. From every
    state of a converged run (newton_velocity_tol 1e-6) one step with the defaults gives the feedback of one converged
    step within 1% of the impact's peak (2.5 times the converged force, the step of impact, under the defaults of an
    entry without proxies). Without proxies the defaults are libuipc's, and given values are taken.
    """
    view, top = _thrown_proxy(device)
    dt, fields = 1.0 / 60.0, ("body_q", "body_qd")
    reference = newton.solvers.SolverIPC(view, **_TIGHT)
    s0, s1 = view.state(), view.state()
    qd = s0.body_qd.numpy()
    qd[top, 2] = -0.5
    s0.body_qd.assign(qd)
    states = []
    for _ in range(10):
        states.append([getattr(s0, f).numpy().copy() for f in fields])
        reference.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
    proxy_of = wp.array([-1, 0], dtype=int, device=device)

    def feedback(solver):
        out, forces = wp.zeros(1, dtype=wp.spatial_vector, device=device), []
        a, b = view.state(), view.state()
        for snapshot in states:
            for f, values in zip(fields, snapshot, strict=True):
                getattr(a, f).assign(values)
            solver.step(a, b, None, None, dt)
            solver.coupling_harvest_proxy_wrenches(
                proxy_of, out, body_qd_before=a.body_qd, state=a, state_out=b, contacts=None, dt=dt
            )
            forces.append(out.numpy()[0, :3].astype(np.float64))
        return np.array(forces)

    converged = feedback(newton.solvers.SolverIPC(view, **_TIGHT))
    peak = np.linalg.norm(converged, axis=1).max()
    test.assertGreater(peak, 5.0 * 9.81)  # an impact, several times the box's weight
    solver = newton.solvers.SolverIPC(view)
    np.testing.assert_allclose(feedback(solver), converged, rtol=0.0, atol=0.01 * peak)
    config = solver._config
    test.assertEqual(
        (config.newton_velocity_tol, config.newton_transrate_tol, config.semi_implicit), (5e-4, 1e-3, False)
    )
    given = newton.solvers.SolverIPC(view, newton_velocity_tol=0.01, newton_transrate_tol=0.02, semi_implicit=True)
    test.assertEqual(
        (given._config.newton_velocity_tol, given._config.newton_transrate_tol, given._config.semi_implicit),
        (0.01, 0.02, True),
    )
    plain = newton.solvers.SolverIPC(_thrown_proxy(device, proxy=False)[0])._config
    test.assertEqual((plain.newton_velocity_tol, plain.newton_transrate_tol, plain.semi_implicit), (0.05, 0.1, True))


def test_late_proxy_fails_the_step(test, device):
    """Sweep a proxy's new pose from where the last step left it and fail the step when it moves into a box.

    A proxy's new pose, written by the coupler before every step, is swept as a kinematic body's jump: the proxy box 2
    mm above a box (``_thrown_proxy``), written 0.5 mm lower, is taken without a friction discard; written 3 cm lower,
    into the box, it fails the step with INVALID_CCD_STEP, named as the kinematic sweep (an eager step raises), where
    the barrier would have pushed it further into the box.
    """
    view, top = _thrown_proxy(device)
    solver = newton.solvers.SolverIPC(view)
    s0, s1 = view.state(), view.state()
    dt = 1.0 / 240.0
    solver.step(s0, s1, None, None, dt)  # the first step places the proxy
    s0.assign(s1)
    for drop, fails in ((5e-4, False), (0.03, True)):
        q = s0.body_q.numpy()
        q[top, 2] -= drop
        s0.body_q.assign(q)
        if fails:
            with test.assertRaisesRegex(RuntimeError, "INVALID_CCD_STEP.*the kinematic sweep"):
                solver.step(s0, s1, None, None, dt)
        else:
            solver.step(s0, s1, None, None, dt)
            test.assertEqual(int(solver._changed.numpy()[0]), 0)  # a proxy's write keeps the friction
            s0.assign(s1)


@requires("uipc_warp")
class TestSolverIPCProxyTolerances(unittest.TestCase):
    """An entry with proxies: the Newton tolerances, at which a proxy's feedback is as accurate as the step's
    convergence, and the sweep of a proxy's written pose."""


for name, func in (
    ("test_proxy_entry_takes_converged_steps", test_proxy_entry_takes_converged_steps),
    ("test_late_proxy_fails_the_step", test_late_proxy_fails_the_step),
):
    add_function_test(TestSolverIPCProxyTolerances, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
