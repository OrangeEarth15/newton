# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC with several worlds: independent by default, or solved together with coupled_worlds."""

import types
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.tests.ipc._fixtures import (
    _CONVERGED,
    _PAIR_WIDTH,
    _TIGHT,
    D_HAT,
    _box_body,
    _frame_start,
    _restore,
    _run,
    _solver_snapshot,
    _world_rows,
    _worlds,
    _worlds_at,
    _write_particle,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires


def _term_subscenes(solver, export):
    """The subscenes of the vertices of every contact term of the solver's last step: the barrier terms of its final
    active set and its friction terms, from libuipc-warp's ``ContactForces`` ``export`` of the solver's engine, updated
    here; a vertex / half-plane term gives its vertex's and its plane's. One array per segment. The broad phase's
    stored and candidate pairs, which no public name holds, libuipc-warp's own tests hold to the subscenes
    (tests/test_arrays_subscenes.py)."""
    export.update()
    sub = solver._scids  # the subscene of every vertex
    out = []
    for name, (vertices, _) in export.numpy().items():
        kind = name.split("+")[0]
        v = vertices[:, : _PAIR_WIDTH[kind]].astype(np.int64)
        if kind == "PH":
            out.append(np.stack([sub[v[:, 0]], solver._plane_scids[v[:, 1]]], axis=1))
        else:
            out.append(sub[v])
    return out


def _pairs_across_worlds(sub):
    """Of pairs given by the subscenes of their vertices: how many join two worlds, and how many join a world with
    the global entities (subscene 0)."""
    local = np.where(sub > 0, sub, 0)
    high = local.max(axis=1)
    low = np.where(sub > 0, sub, high[:, None]).min(axis=1)
    return int(np.sum(low != high)), int(np.sum((sub == 0).any(axis=1) & (high > 0)))


def test_coupled_worlds_follow_the_single_world(test, device):
    """Four copies of a world at the origin, each a world of its own, with coupled_worlds: no contact term joins two
    worlds at any step, every world meets the global static box, and every world follows the run of the one world
    alone, with the same Newton iterations at every step (the copies' shares of every sum of the solve are equal) and
    positions within the rounding of those sums."""
    import uipc_warp.arrays  # noqa: PLC0415

    copies, steps, dt = 4, 60, 0.01
    runs = []
    for n in (1, copies):
        model = _worlds(device, n)
        solver = newton.solvers.SolverIPC(model, coupled_worlds=True)
        test.assertEqual(solver._worlds is None, n == 1)
        engine = solver._engine
        export = uipc_warp.arrays.ContactForces(engine)  # the step's contact terms (_term_subscenes)
        s0, s1 = model.state(), model.state()
        rows = _world_rows(model, n)
        q, body_q, newton_iterations, global_pairs = [], [], [], 0
        for _ in range(steps):
            solver.step(s0, s1, None, None, dt)
            s0, s1 = s1, s0
            newton_iterations.append(solver.frame_stats()["newton_iterations"])
            pq, bq = s0.particle_q.numpy(), s0.body_q.numpy()
            q.append([pq[p] for p, _ in rows])
            body_q.append([bq[b] for _, b in rows])
            test.assertIs(solver._engine, engine)  # the export's engine: no list grew
            if n > 1:
                for sub in _term_subscenes(solver, export):
                    across, with_global = _pairs_across_worlds(sub)
                    test.assertEqual(across, 0)
                    global_pairs += with_global
        runs.append((np.array(q), np.array(body_q), newton_iterations))
        if n > 1:
            test.assertGreater(global_pairs, 0)
    (q1, b1, n1), (qn, bn, nn) = runs
    test.assertEqual(n1, nn)
    # measured: 1e-12 m and 0 (the float32 output hides the rounding of the shared sums)
    test.assertLess(np.abs(qn - q1).max(), 1e-6)
    test.assertLess(np.abs(bn[..., :3] - b1[..., :3]).max(), 1e-6)
    # every world rests: the soft cube on the global box (top at 0.1), the rigid box and the particle on the ground
    for k in range(copies):
        test.assertGreater(qn[-1, k, :-1, 2].min(), 0.1)
        test.assertLess(qn[-1, k, :-1, 2].min(), 0.1 + D_HAT)
        test.assertGreater(bn[-1, k, 0, 2], 0.05)
        test.assertLess(bn[-1, k, 0, 2], 0.05 + D_HAT)
        test.assertGreater(qn[-1, k, -1, 2], 0.01)


def test_coupled_worlds_are_built_as_one_world(test, device):
    """A model with content in one local world is built as without coupled_worlds: no subscenes, the same contact
    elements and ids, and the same steps."""
    model = _worlds(device, 1)
    trajectories = []
    for coupled in (False, True):
        solver = newton.solvers.SolverIPC(model, coupled_worlds=coupled)
        test.assertIsNone(solver._worlds)
        test.assertFalse(solver._independent)
        trajectories.append((solver._cids.copy(), solver._enabled.copy(), _run(model, solver, 5)))
    (c0, e0, t0), (c1, e1, t1) = trajectories
    np.testing.assert_array_equal(c0, c1)
    np.testing.assert_array_equal(e0, e1)
    np.testing.assert_allclose(t0, t1, rtol=0.0, atol=1e-7)


def test_coupled_worlds_share_filter_classes(test, device):
    """A box filtered against a second box that starts inside it, replicated over six worlds: the worlds take the
    contact elements of one world, the pairs the elements enable are Model.shape_contact_pairs, and in every world the
    smaller box falls through the larger and both come to rest on the ground."""

    def content():
        world = newton.ModelBuilder()
        _box_body(world, (0.0, 0.0, 0.3), (0.05, 0.05, 0.05))
        _box_body(world, (0.0, 0.0, 0.31), (0.03, 0.03, 0.03))
        world.add_shape_collision_filter_pair(0, 1)
        return world

    elements = []
    for copies in (1, 6):
        model = _worlds(device, copies, content)
        solver = newton.solvers.SolverIPC(model, coupled_worlds=True)
        elements.append(len(solver._enabled))  # the contact elements the solver makes
        if copies > 1:
            s0, s1 = model.state(), model.state()
            for _ in range(60):
                solver.step(s0, s1, None, None, 0.01)
                s0, s1 = s1, s0
            z = s0.body_q.numpy()[:, 2].reshape(copies, 2)
            np.testing.assert_array_less(0.05, z[:, 0])
            np.testing.assert_array_less(z[:, 0], 0.05 + D_HAT)
            np.testing.assert_array_less(0.03, z[:, 1])
            np.testing.assert_array_less(z[:, 1], 0.03 + D_HAT)
    test.assertEqual(elements[0], elements[1])


def test_coupled_worlds_status_in_every_world(test, device):
    """The status and the residual of the one solve reach every world entry, also from a captured step, and a step that
    fails in one world is reported in every world."""
    copies = 3
    model = _worlds(device, copies)
    solver = newton.solvers.SolverIPC(model, coupled_worlds=True)
    observables = solver.observables({solver.ObservableKind.STATUS, solver.ObservableKind.SOLVER_RESIDUAL})
    s0, s1 = model.state(), model.state()
    solver.step(s0, s1, None, None, 0.01, observables=observables)
    with wp.ScopedCapture(device=device) as capture:
        solver.step(s1, s0, None, None, 0.01, observables=observables)
    wp.capture_launch(capture.graph)
    status, residual = observables.status.numpy(), observables.solver_residual.numpy()
    test.assertEqual(status.tolist(), [0] * (copies + 1))
    test.assertTrue(np.all(residual[:copies] == residual[0]) and 0.0 < residual[0] <= 1.0)
    test.assertEqual(residual[copies], 0.0)  # the global entry: no global particle or body
    test.assertFalse(solver.check_status(observables))
    # the particle of world 1 written below the ground into the captured step's input: it fails, in every world
    q = s1.particle_q.numpy()
    q[_world_rows(model, copies)[1][0][-1], 2] = -0.01
    s1.particle_q.assign(q)
    wp.capture_launch(capture.graph)
    bits = observables.status.numpy()
    test.assertTrue(np.all(bits[:copies] == bits[0]) and bits[0] & int(solver.StatusBits.FAILED))
    with test.assertRaises(RuntimeError):
        solver.check_status(observables)


def test_coupled_worlds_fall_with_their_gravity(test, device):
    """A particle in each of three worlds of different gravity falls as implicit Euler with its world's gravity."""
    dt, steps = 0.01, 10
    gravities = ((0.0, 0.0, -9.81), (0.0, 0.0, -2.0), (1.0, 0.0, -5.0))
    builder = newton.ModelBuilder()
    for g in gravities:
        builder.begin_world(gravity=g)
        builder.add_particle(wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0, 0.0, 0.0), 1.0, radius=0.01)
        builder.end_world()
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, coupled_worlds=True, **_TIGHT)
    traj = _run(model, solver, steps, dt=dt)
    k = np.arange(1, steps + 1)[:, None, None]
    expected = np.array([[0.0, 0.0, 1.0]]) + np.array(gravities)[None] * dt * dt * (k * (k + 1) / 2.0)
    np.testing.assert_allclose(traj, expected, rtol=0.0, atol=1e-5)


def test_coupled_worlds_beyond_the_old_element_limit(test, device):
    """1100 worlds at one place, each a box and a particle dropped onto the global ground: five contact elements times
    1101 subscenes, beyond the 2048 that the engine took before it tested the two tables apart. The solver builds, its
    start check keeps the two tables (no table of their product), and in 20 steps every box and particle lands on the
    ground in its own world: no contact term joins two worlds."""
    import uipc_warp.arrays  # noqa: PLC0415

    worlds = 1100
    world = newton.ModelBuilder()
    _box_body(world, (0.0, 0.0, 0.07), (0.05, 0.05, 0.05))
    world.add_particle(wp.vec3(0.2, 0.0, 0.03), wp.vec3(0.0, 0.0, 0.0), 0.01, radius=0.01)
    scene = newton.ModelBuilder()
    scene.add_ground_plane()
    scene.replicate(world, worlds)
    model = scene.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, coupled_worlds=True)
    n_el, n_sub = len(solver._enabled), len(solver._subscene_enabled)
    test.assertEqual(n_sub, worlds + 1)
    test.assertGreater(n_el * n_sub, 2048)
    test.assertEqual(solver._enabled.shape, (n_el, n_el))
    engine = solver._engine
    export = uipc_warp.arrays.ContactForces(engine)
    s0, s1 = model.state(), model.state()
    for _ in range(20):
        solver.step(s0, s1, None, None, 0.01)
        s0, s1 = s1, s0
        test.assertIs(solver._engine, engine)
        for sub in _term_subscenes(solver, export):
            test.assertEqual(_pairs_across_worlds(sub)[0], 0)
    z_box, z_particle = s0.body_q.numpy()[:, 2], s0.particle_q.numpy()[:, 2]
    test.assertGreater(z_box.min(), 0.05)
    test.assertLess(z_box.max(), 0.05 + D_HAT)
    test.assertGreater(z_particle.min(), 0.01)
    test.assertLess(z_particle.max(), 0.01 + D_HAT)


def test_start_check_of_many_worlds(test, device):
    """5000 worlds of a particle at one place over the global ground: the start check keeps the contact elements' table
    and the subscenes' apart (their product would hold 6.3e8 entries) and forms no pair of two worlds, whose particles
    all overlap; the model is accepted. With world 3172's particle placed through the ground it is refused, naming
    that particle."""
    worlds = 5000
    for inside in (False, True):
        world = newton.ModelBuilder()
        world.add_particle(wp.vec3(0.0, 0.0, 0.05), wp.vec3(0.0, 0.0, 0.0), 0.01, radius=0.01)
        scene = newton.ModelBuilder()
        scene.add_ground_plane()
        scene.replicate(world, worlds)
        model = scene.finalize(device=device)
        if inside:
            q = model.particle_q.numpy()
            q[3172, 2] = 0.005
            model.particle_q.assign(q)
            with test.assertRaisesRegex(ValueError, r"particle 3172 and plane shape 0 start"):
                newton.solvers.SolverIPC(model, coupled_worlds=True)
            continue
        solver = newton.solvers.SolverIPC(model, coupled_worlds=True)
        n_el = len(solver._enabled)
        test.assertEqual(solver._enabled.shape, (n_el, n_el))
        test.assertEqual(solver._subscene_enabled.shape, (worlds + 1, worlds + 1))
        test.assertGreater(n_el * (worlds + 1), 2048)


def test_near_parts_forms_no_pair_of_two_worlds(test, device):
    """The sweep of the start check against a brute-force test of every pair of boxes, on random boxes in 40 worlds
    at one place and a few global ones, with subscene tables of the form of separate worlds (some worlds meeting
    themselves, some not, the global subscene meeting all) and one of another form: every near pair the table lets
    meet, each once, and with the form of separate worlds no pair of two worlds (the caller tests the pairs within a
    world against the table)."""
    from newton._src.solvers.ipc.start_check import _near_parts  # noqa: PLC0415

    del device
    rng = np.random.default_rng(7)
    for trial in range(6):
        n_sub = 41
        table = np.eye(n_sub, dtype=bool) & (rng.uniform(size=n_sub) < 0.8)[:, None]
        table[0, :] = table[:, 0] = True
        if trial == 5:  # two worlds that meet: not the form of separate worlds, every near pair kept
            table[3, 4] = table[4, 3] = True
        n = 600
        sub = np.where(rng.uniform(size=n) < 0.05, 0, rng.integers(1, n_sub, n))
        centre = rng.uniform(-0.2, 0.2, (n, 3))
        half = rng.uniform(0.0, 0.03, (n, 3))
        lo, hi = centre - half, centre + half
        reach = rng.uniform(0.0, 0.01, n)
        slack = 1e-6
        owner = types.SimpleNamespace(_subscene_enabled=table)
        got = _near_parts(owner, lo, hi, reach, slack, sub)
        a, b = np.triu_indices(n, 1)
        apart = np.maximum(lo[b] - hi[a], lo[a] - hi[b]).max(axis=1)
        near = apart <= reach[a] + reach[b] + slack
        brute = {(int(i), int(j)) for i, j in zip(a[near], b[near], strict=True)}
        meets = {(i, j) for i, j in brute if table[sub[i], sub[j]]}
        found = {(int(i), int(j)) for i, j in got}
        test.assertEqual(len(found), len(got))  # no pair twice
        test.assertTrue(meets <= found, trial)
        test.assertTrue(found <= brute, trial)
        if trial < 5:  # no pair of two worlds is formed: those the table refuses are pairs within a world
            test.assertTrue(all(sub[i] == sub[j] for i, j in found - meets), trial)
            test.assertTrue(all(sub[i] == 0 or sub[j] == 0 or sub[i] == sub[j] for i, j in found), trial)


def _attached_world():
    """One world: a box 30 cm up with a particle attached 10 cm under its centre, over the global ground."""
    world = newton.ModelBuilder()
    box = _box_body(world, (0.0, 0.0, 0.3), (0.05, 0.05, 0.05))
    world.add_particle(wp.vec3(0.0, 0.0, 0.2), wp.vec3(0.0, 0.0, 0.0), 0.05, radius=0.005)
    world.add_attachment_particle_body(0, box, body_point=(0.0, 0.0, -0.1), stiffness=500.0)
    return world


def test_coupled_worlds_with_attachments(test, device):
    """Three copies of a world whose particle is attached to a falling box, each a world of its own at the origin with
    coupled_worlds: every world follows the run of the one world alone, with the same Newton iterations at every step,
    and the anchors are in their worlds' subscenes."""
    runs = []
    for n in (1, 3):
        scene = newton.ModelBuilder()
        scene.add_ground_plane()
        scene.replicate(_attached_world(), n)
        model = scene.finalize(device=device)
        solver = newton.solvers.SolverIPC(model, coupled_worlds=True)
        if n > 1:
            layout = solver._scene.layout()
            anchors = np.arange(layout.body_anchor_offsets[0], layout.body_anchor_offsets[-1])
            np.testing.assert_array_equal(solver._scids[anchors], [1, 2, 3])
        rows = _world_rows(model, n)
        s0, s1 = model.state(), model.state()
        q, iterations = [], []
        for _ in range(40):
            solver.step(s0, s1, None, None, 0.01)
            s0, s1 = s1, s0
            iterations.append(solver.frame_stats()["newton_iterations"])
            pq, bq = s0.particle_q.numpy(), s0.body_q.numpy()
            q.append([np.concatenate([pq[p].reshape(-1), bq[b][:, :3].reshape(-1)]) for p, b in rows])
        runs.append((np.array(q), iterations))
    (q1, n1), (qn, nn) = runs
    test.assertEqual(n1, nn)
    test.assertLess(np.abs(qn - q1).max(), 1e-6)
    test.assertLess(q1[-1, 0, 2], 0.1)  # it fell, the particle under the box


def _pushed_world():
    """One world: a box on the global ground and a kinematic box 2 cm behind it."""
    world = newton.ModelBuilder()
    _box_body(world, (0.0, 0.0, 0.052), (0.05, 0.05, 0.05))
    pusher = world.add_body(xform=wp.transform(wp.vec3(-0.12, 0.0, 0.052), wp.quat_identity()), is_kinematic=True)
    world.add_shape_box(pusher, hx=0.05, hy=0.05, hz=0.05)
    return world


def test_coupled_worlds_with_kinematic_aims(test, device):
    """Three copies of a world whose kinematic box pushes a box across the ground with kinematic_motion="aim", each a
    world of its own at the origin with coupled_worlds, every pusher moved by state_in: every world follows the run of
    the one world alone, with the same Newton iterations at every step, and the boxes are pushed. The solves are
    converged, so that the worlds' runs differ by the rounding of the shared sums only: at the default tolerances
    the rounding moves where the linear and Newton solves stop, by up to 2e-6 m in 20 steps."""
    runs = []
    for n in (1, 3):
        scene = newton.ModelBuilder()
        scene.add_ground_plane()
        scene.replicate(_pushed_world(), n)
        model = scene.finalize(device=device)
        solver = newton.solvers.SolverIPC(model, coupled_worlds=True, kinematic_motion="aim", **_CONVERGED)
        rows = _world_rows(model, n)
        s0, s1 = model.state(), model.state()
        q, iterations = [], []
        for k in range(20):
            body_q, body_qd = s0.body_q.numpy(), s0.body_qd.numpy()
            for _, b in rows:
                body_q[b[1], 0] = -0.12 + 0.01 * k
                body_qd[b[1]] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            s0.body_q.assign(body_q)
            s0.body_qd.assign(body_qd)
            solver.step(s0, s1, None, None, 0.01)
            s0, s1 = s1, s0
            iterations.append(solver.frame_stats()["newton_iterations"])
            q.append([s0.body_q.numpy()[b][:, :3].reshape(-1) for _, b in rows])
        runs.append((np.array(q), iterations))
    (q1, n1), (qn, nn) = runs
    test.assertEqual(n1, nn)
    test.assertLess(np.abs(qn - q1).max(), 1e-6)
    test.assertGreater(q1[-1, 0, 0], 0.05)  # the box was pushed


def _reset_world():
    """One world for the resets of one world: a box sliding at 0.2 m/s on the global static box of ``_worlds``, a
    particle sliding at 0.3 m/s on the ground, and a sphere turned about a vertical hinge by a drive, a passive spring
    and passive damping."""
    world = newton.ModelBuilder()
    box = _box_body(world, (0.35, 0.0, 0.1505), (0.05, 0.05, 0.05))
    world.body_qd[box] = (0.2, 0.0, 0.0, 0.0, 0.0, 0.0)
    world.add_particle(wp.vec3(-0.3, 0.0, 0.0101), wp.vec3(0.3, 0.0, 0.0), 0.01, radius=0.01)
    link = world.add_link(xform=wp.transform(wp.vec3(-0.1, 0.4, 0.5), wp.quat_identity()))
    world.add_shape_sphere(link, radius=0.05)
    joint = world.add_joint_revolute(
        -1,
        link,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(0.0, 0.4, 0.5), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()),
        target_ke=40.0,
        target_kd=2.0,
        stiffness=10.0,
        damping=1.0,
        target_q=0.5,
        target_qd=0.1,
        rest_q=-0.2,
    )
    world.add_articulation([joint])
    return world


def _row_worlds(solver, model):
    """The world of every row of the solver's engine: a particle's, a body's for its pseudo-vertices and surface
    vertices, -1 for the static shapes and the fixed body the world-attached joints hang from."""
    layout = solver._scene.layout()
    world = np.full(layout.n, -1, np.int64)
    world[: model.particle_count] = model.particle_world.numpy()
    body_world, offsets = model.body_world.numpy(), layout.body_vertex_offsets
    for b in range(model.body_count):
        world[layout.body_base[b] : layout.body_base[b] + 4] = body_world[b]
        world[offsets[b] : offsets[b + 1]] = body_world[b]
    return world


def _check_friction_kept(test, got, full, world, flagged):
    """``got``'s friction candidates are ``full``'s without those with a vertex in the worlds ``flagged``. The order is
    not compared: every case recovers one engine, whose stored pairs carry over from the case before, and the engine's
    own tests compare the order on fresh engines."""
    for name, listed in full["lists"].items():
        pairs = np.asarray(listed).reshape(-1, _PAIR_WIDTH[name])
        vertices = pairs[:, :1] if name == "PH" else pairs  # a PH pair: its vertex, then its plane
        out = np.isin(world[vertices], flagged).any(axis=1)
        kept = pairs[~out]
        got_pairs = np.asarray(got["lists"][name]).reshape(-1, _PAIR_WIDTH[name])
        order, got_order = np.lexsort(kept.T[::-1]), np.lexsort(got_pairs.T[::-1])
        np.testing.assert_array_equal(got_pairs[got_order], kept[order], name)


def _branch_steps(solver, model, snapshot, state, cases, dt=0.01, control=None, pusher=None, after_reset=None):
    """From ``snapshot``, one step for every case: ``"none"``, ``"all"`` (``reset`` without a mask), a world index
    (``reset`` of that world) or ``("state", k)`` (a new velocity for world ``k``'s particles); ``pusher`` moves the
    state's kinematic bodies before the step, and ``after_reset(case)`` looks at the solver between the reset and the
    step. Returns the frame start of every case."""
    import uipc_warp.arrays  # noqa: PLC0415

    export = uipc_warp.arrays.ContactForces(solver._engine)  # every case recovers this engine
    out = {}
    for case in cases:
        _restore(solver, snapshot)
        state_in = model.state()
        state_in.assign(state)
        if pusher is not None:
            pusher(state_in)
        if case == "all":
            solver.reset(state_in)
        elif isinstance(case, int):
            mask = np.zeros(model.world_count + 1, bool)
            mask[case] = True
            solver.reset(state_in, world_mask=wp.array(mask, dtype=wp.bool, device=model.device))
        elif isinstance(case, tuple):
            qd = state_in.particle_qd.numpy()
            qd[model.particle_world.numpy() == case[1]] += (0.0, 0.05, 0.0)
            state_in.particle_qd.assign(qd)
        if after_reset is not None:
            after_reset(case)
        solver.step(state_in, model.state(), control, None, dt)
        out[case] = _frame_start(solver, export)
    return out


def test_reset_of_one_world_keeps_the_others(test, device):
    """Three worlds at one place, independent and with coupled_worlds, each a box sliding on the global static box, a
    particle sliding on the ground and a sphere driven about a hinge, under implicit Euler and BDF2. From one state of
    the solver, a step after no reset, after a reset of world 1, after a reset of every world, and after a new velocity
    written for world 1's particle. The reset of world 1 and the new state discard the friction of world 1's subscene
    alone, and the step has the friction candidates of the step without a reset but those with a vertex in world 1; the
    reset of every world discards all of them, and a step without a reset none. Under BDF2 the frame restarts world 1's
    rows and no other, and world 1's drive takes the terms of the reset of every world, the other worlds' those of no
    reset, bit for bit; under implicit Euler there is no history, and the resets leave the terms as they are. With
    independent worlds a reset of the mask's global entry alone is the step without a reset: the global entities are
    no unknowns. What the engine keeps of the others (their lagged friction, prediction and masses) libuipc-warp's own
    tests hold. The branches begin at the first step with contact in every world."""
    import uipc_warp.arrays  # noqa: PLC0415

    for coupled, integrator in ((True, "bdf1"), (True, "bdf2"), (False, "bdf1"), (False, "bdf2")):
        with test.subTest(coupled_worlds=coupled, integrator=integrator):
            model = _worlds(device, 3, _reset_world)
            solver = newton.solvers.SolverIPC(model, coupled_worlds=coupled, integrator=integrator)
            test.assertEqual(solver._independent, not coupled)
            control = model.control()
            s0, s1 = model.state(), model.state()
            world = _row_worlds(solver, model)
            export = uipc_warp.arrays.ContactForces(solver._engine)
            for _ in range(100):  # until the boxes and particles rest on the static box and the ground in every world
                solver.step(s0, s1, control, None, 0.01)
                s0, s1 = s1, s0
                export.update()
                terms = export.numpy()
                pt, ph = terms["PT+N"][0], terms["PH+N"][0]
                if all(np.isin(k, world[pt]) and np.isin(k, world[ph[:, 0]]) for k in range(3)):
                    break
            else:
                test.fail("no step with contact in every world")
            one = world == 1
            joints = solver._joint_rows["revolute"].numpy()
            drive_one = model.body_world.numpy()[model.joint_child.numpy()[joints]] == 1
            glob = model.world_count  # the mask's global entry
            cases = ("none", 1, "all", ("state", 1)) + (() if coupled else (glob,))
            frames = _branch_steps(solver, model, _solver_snapshot(solver), s0, cases, control=control)
            none, w1, every, new_state = frames["none"], frames[1], frames["all"], frames[("state", 1)]
            if not coupled:  # the global entry selects nothing
                test.assertFalse(frames[glob]["discard"] or np.any(frames[glob]["discarded"]))
                _check_friction_kept(test, frames[glob], none, world, [])
                np.testing.assert_array_equal(frames[glob]["drive"], none["drive"])
                if integrator == "bdf2":
                    test.assertFalse(np.any(frames[glob]["restarted"]))
            for k in range(3):  # every world has friction candidates on the static box and the ground
                for name in ("PT", "PH"):
                    pairs = np.asarray(none["lists"][name]).reshape(-1, _PAIR_WIDTH[name])
                    vertices = pairs[:, :1] if name == "PH" else pairs
                    test.assertTrue(np.any(np.isin(world[vertices], [k]).any(axis=1)), (k, name))
            subscene_one = np.zeros(len(none["discarded"]), np.int32)
            subscene_one[solver._scids[np.flatnonzero(one)[0]]] = 1
            for case in (w1, new_state):
                test.assertFalse(case["discard"])
                np.testing.assert_array_equal(case["discarded"], subscene_one)
            test.assertFalse(none["discard"] or np.any(none["discarded"]))
            test.assertTrue(every["discard"])
            _check_friction_kept(test, w1, none, world, [1])
            _check_friction_kept(test, new_state, none, world, [1])
            test.assertEqual(sum(len(p) for p in every["lists"].values()), 0)
            np.testing.assert_array_equal(w1["drive"][~drive_one], none["drive"][~drive_one])
            np.testing.assert_array_equal(w1["drive"][drive_one], every["drive"][drive_one])
            if integrator == "bdf1":
                test.assertIsNone(w1["restarted"])
                np.testing.assert_array_equal(w1["drive"], none["drive"])  # no history: the resets change no term
                continue
            np.testing.assert_array_equal(w1["restarted"], one.astype(np.int32))
            np.testing.assert_array_equal(new_state["restarted"], one.astype(np.int32))
            test.assertFalse(np.any(none["restarted"]))
            test.assertFalse(np.all(w1["drive"][drive_one] == none["drive"][drive_one]))


def test_reset_of_one_world_keeps_the_other_worlds_aims(test, device):
    """Three copies of the pushed box's world with kinematic_motion="aim" under BDF2, independent and with
    coupled_worlds, whose aim kernel writes the history of every pusher at every step; each pusher moved by state_in.
    From one state, a step after no reset, after a reset of world 1 and after a reset of every world: the reset of
    world 1 releases world 1's pusher alone (it takes state_in), and the frame restarts world 1's rows alone (the
    prediction and masses of a restarted row are those libuipc-warp's own tests hold)."""
    for coupled in (True, False):
        with test.subTest(coupled_worlds=coupled):
            _reset_one_world_with_aims(test, device, coupled)


def _reset_one_world_with_aims(test, device, coupled):
    """``test_reset_of_one_world_keeps_the_other_worlds_aims`` with or without coupled_worlds."""
    scene = newton.ModelBuilder()
    scene.add_ground_plane()
    scene.replicate(_pushed_world(), 3)
    model = scene.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, coupled_worlds=coupled, kinematic_motion="aim", integrator="bdf2")
    rows = _world_rows(model, 3)
    pushers = [b[1] for _, b in rows]

    def push(state, k):
        body_q, body_qd = state.body_q.numpy(), state.body_qd.numpy()
        body_q[pushers, 0] = -0.12 + 0.01 * k
        body_qd[pushers] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        state.body_q.assign(body_q)
        state.body_qd.assign(body_qd)

    s0, s1 = model.state(), model.state()
    for k in range(6):
        push(s0, k)
        solver.step(s0, s1, None, None, 0.01)
        s0, s1 = s1, s0
    world = _row_worlds(solver, model)
    one = world == 1
    holds = {}

    def hold(case):
        holds[case] = solver._body_hold.numpy()[pushers]

    frames = _branch_steps(
        solver, model, _solver_snapshot(solver), s0, ("none", 1, "all"), pusher=lambda s: push(s, 6), after_reset=hold
    )
    np.testing.assert_array_equal(holds["none"], [1, 1, 1])
    np.testing.assert_array_equal(holds[1], [1, 0, 1])
    np.testing.assert_array_equal(holds["all"], [0, 0, 0])
    none, w1 = frames["none"], frames[1]
    np.testing.assert_array_equal(w1["restarted"], one.astype(np.int32))
    test.assertFalse(np.any(none["restarted"]))


def _world_iterations(solver):
    """The Newton iterations of every independent world in the solver's last step, in the order of the worlds
    (libuipc-warp's ``frame_status(engine).worlds``, which its array-api.md documents), or the step's for an engine of
    one world."""
    import uipc_warp.arrays  # noqa: PLC0415

    status = uipc_warp.arrays.frame_status(solver._engine)
    return [world.newton_iterations for world in status.worlds] or [status.newton_iterations]


def _record_worlds(model, solver, steps, dt=0.01, capture=False):
    """``steps`` steps from the model's state, eager, or captured two at a time after one eager step that loads the
    state and the kernels. After each step (each replay's second step when captured): every local world's particle and
    body positions, its Newton iterations (``_world_iterations``), the STATUS and SOLVER_RESIDUAL entries of every
    world and the global one, and the Newton iterations of the step's loop (``frame_stats``)."""
    observables = solver.observables({solver.ObservableKind.STATUS, solver.ObservableKind.SOLVER_RESIDUAL})
    rows = _world_rows(model, model.world_count)
    out = {"q": [], "iterations": [], "loop": [], "status": [], "residual": []}

    def record(state):
        pq, bq = state.particle_q.numpy(), state.body_q.numpy()
        out["q"].append([np.concatenate([pq[p].reshape(-1), bq[b][:, :3].reshape(-1)]) for p, b in rows])
        out["iterations"].append(_world_iterations(solver))
        out["loop"].append(solver.frame_stats()["newton_iterations"])
        out["status"].append(observables.status.numpy().copy())
        out["residual"].append(observables.solver_residual.numpy().copy())

    s0, s1 = model.state(), model.state()
    if capture:
        solver.step(s0, s1, None, None, dt, observables=observables)
        s0, s1 = s1, s0
        record(s0)
        with wp.ScopedCapture(device=model.device) as cap:
            solver.step(s0, s1, None, None, dt, observables=observables)
            solver.step(s1, s0, None, None, dt, observables=observables)
        for _ in range((steps - 1) // 2):
            wp.capture_launch(cap.graph)
            record(s0)
    else:
        for _ in range(steps):
            solver.step(s0, s1, None, None, dt, observables=observables)
            s0, s1 = s1, s0
            record(s0)
    return {key: np.array(value) for key, value in out.items()}


def test_independent_worlds_step_as_alone(test, device):
    """Three worlds of a soft cube, a rigid box and a particle (``_world_content``) dropped from heights 3 cm apart
    onto the global static box and ground, independent worlds by default: in every step each world takes the Newton
    iterations of the same world built alone, the step's loop those of its slowest world, and its positions stay
    within 1e-6 m of the run alone (measured: 1e-12 m in float64, so the float32 outputs differ by at most an ulp);
    its STATUS entry is that of the run alone and its SOLVER_RESIDUAL entry that of the run alone to 1e-6, and the
    global entry holds 0. The same from captured steps, replayed two at a time. The worlds differ: their Newton
    iterations part in some steps."""
    shifts, steps = (0.0, 0.03, 0.06), 41
    for capture in (False, True):
        with test.subTest(captured=capture):
            model = _worlds_at(device, shifts)
            solver = newton.solvers.SolverIPC(model)
            test.assertTrue(solver._independent)
            run = _record_worlds(model, solver, steps, capture=capture)
            test.assertEqual(run["status"].shape[1], len(shifts) + 1)
            np.testing.assert_array_equal(run["status"][:, -1], 0)  # the global world: no unknowns, no step
            np.testing.assert_array_equal(run["residual"][:, -1], 0.0)
            np.testing.assert_array_equal(run["loop"], run["iterations"].max(axis=1))
            test.assertTrue(np.any(run["iterations"].min(axis=1) != run["iterations"].max(axis=1)))
            for k, shift in enumerate(shifts):
                one = _worlds_at(device, [shift])
                alone = _record_worlds(one, newton.solvers.SolverIPC(one), steps, capture=capture)
                np.testing.assert_array_equal(run["iterations"][:, k], alone["iterations"][:, 0], f"world {k}")
                worst = np.abs(run["q"][:, k] - alone["q"][:, 0]).max()
                test.assertLess(worst, 1.0e-6, f"world {k}: {worst:.3g} m")
                np.testing.assert_array_equal(run["status"][:, k], alone["status"][:, 0], f"world {k}")
                np.testing.assert_allclose(
                    run["residual"][:, k], alone["residual"][:, 0], rtol=1.0e-6, atol=0.0, err_msg=f"world {k}"
                )


def _weighted_world(density, size):
    """One world of ``_world_content``'s parts with their masses and sizes changed: a soft cube of cells ``size`` and
    a rigid box of half size ``size``, both of ``density``, and a free particle. Worlds of different densities and
    sizes have stiffness ranges of their own (libuipc's range follows the mean vertex mass and the extent)."""
    world = newton.ModelBuilder()
    world.add_soft_grid(
        pos=wp.vec3(0.3, -0.05, 0.25), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0), dim_x=2, dim_y=2, dim_z=2,
        cell_x=size, cell_y=size, cell_z=size, density=density, k_mu=3.4e4, k_lambda=1.6e5, k_damp=0.0,
    )  # fmt: skip
    b = world.add_body(xform=wp.transform(wp.vec3(-0.3, 0.0, 0.15), wp.quat_identity()))
    world.add_shape_box(b, hx=size, hy=size, hz=size, cfg=newton.ModelBuilder.ShapeConfig(density=density))
    world.add_particle(wp.vec3(0.0, 0.3, 0.1), wp.vec3(0.0, 0.0, 0.0), 0.01, radius=0.01)
    return world


def _worlds_of(device, contents):
    """One world of every builder of ``contents``, at the origin, over the global ground plane and static box of
    ``_worlds``; ``_worlds_of(device, [content])`` is that world alone."""
    scene = newton.ModelBuilder()
    scene.add_ground_plane()
    scene.add_shape_box(-1, xform=wp.transform(wp.vec3(0.35, 0.0, 0.05), wp.quat_identity()), hx=0.15, hy=0.15, hz=0.05)
    for content in contents:
        scene.add_world(content)
    return scene.finalize(device=device)


def test_independent_worlds_take_their_own_stiffness_range(test, device):
    """adaptive_kappa=True with independent worlds of different densities and sizes (``_weighted_world``): every world
    takes libuipc's stiffness range of the same world alone. ``contact_stiffness(dt)`` gives one value per world, that
    of the model of the world alone at two step sizes bit for bit: with kappa=None every world's own lower bound, which
    differ between the worlds, and with a kappa that lies above some worlds' lower bounds and below the others', the
    kappa in the first and their lower bounds in the others, the first step warning once and naming the worlds it
    clamps. In every step, eager and captured, each world takes the Newton iterations of its model alone and its
    positions stay within 1e-6 m of that run. d_hat is given: its default follows the smallest features of the whole
    model, which differ from those of a world alone."""
    specs = ((1000.0, 0.05), (5000.0, 0.04), (300.0, 0.07))
    model = _worlds_of(device, [_weighted_world(*spec) for spec in specs])
    singles = [_worlds_of(device, [_weighted_world(*spec)]) for spec in specs]
    adaptive = {"kappa": None, "adaptive_kappa": True, "d_hat": D_HAT}
    solver = newton.solvers.SolverIPC(model, **adaptive)
    test.assertTrue(solver._independent)
    for dt in (0.01, 0.005):
        kappa = solver.contact_stiffness(dt)
        test.assertEqual(kappa.shape, (len(specs),))
        test.assertEqual(len(set(kappa.tolist())), len(specs))
        for k, one in enumerate(singles):
            test.assertEqual(kappa[k], newton.solvers.SolverIPC(one, **adaptive).contact_stiffness(dt), f"world {k}")
    lows = solver.contact_stiffness(0.01)
    given = float(np.sqrt(lows.min() * lows.max()))
    clamped = newton.solvers.SolverIPC(model, kappa=given, adaptive_kappa=True, d_hat=D_HAT)
    used = clamped.contact_stiffness(0.01)
    np.testing.assert_array_equal(used, np.maximum(lows, given))
    for k, one in enumerate(singles):
        test.assertEqual(
            used[k],
            newton.solvers.SolverIPC(one, kappa=given, adaptive_kappa=True, d_hat=D_HAT).contact_stiffness(0.01),
        )
    s0, s1 = model.state(), model.state()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clamped.step(s0, s1, None, None, 0.01)
    messages = [str(w.message) for w in caught if issubclass(w.category, newton.NewtonWarning)]
    test.assertEqual(len(messages), 1)
    test.assertTrue(lows.min() < given < lows.max())
    for k in range(len(specs)):
        test.assertEqual(f"world {k} uses {lows[k]:.6g} Pa" in messages[0], bool(lows[k] > given), messages[0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        clamped.step(s1, s0, None, None, 0.005)  # clamped again, at other values: not repeated
    steps = 21
    for capture in (False, True):
        with test.subTest(captured=capture):
            run = _record_worlds(model, newton.solvers.SolverIPC(model, **adaptive), steps, capture=capture)
            test.assertTrue(np.any(run["iterations"].min(axis=1) != run["iterations"].max(axis=1)))
            for k, one in enumerate(singles):
                alone = _record_worlds(one, newton.solvers.SolverIPC(one, **adaptive), steps, capture=capture)
                np.testing.assert_array_equal(run["iterations"][:, k], alone["iterations"][:, 0], f"world {k}")
                worst = np.abs(run["q"][:, k] - alone["q"][:, 0]).max()
                test.assertLess(worst, 1.0e-6, f"world {k}: {worst:.3g} m")


def test_independent_world_fails_alone(test, device):
    """Three worlds of ``_world_content`` at heights 3 cm apart, two solvers stepping them in lockstep, one of them
    clean. After ten steps world 1's particle is written 1 cm through the ground, inside its contact thickness, into the
    other's input: its eager step raises naming world 1 alone and writes state_out, world 1 at the state it started
    from, bit for bit, and worlds 0 and 2 at the clean solver's step within 1e-6 m; STATUS has the failure in world 1's
    entry alone. With the particle written back the next step takes every world on, and a NaN velocity of world 1's
    particle fails world 1 alone the same way. Captured, the step that fails in world 1 sets world 1's entry alone,
    the other worlds take their step, and check_status names world 1."""
    failed = int(newton.solvers.SolverIPC.StatusBits.FAILED)
    shifts = (0.0, 0.03, 0.06)
    model = _worlds_at(device, shifts)
    rows = _world_rows(model, len(shifts))
    particle = int(rows[1][0][-1])  # world 1's free particle
    kinds = {newton.solvers.SolverIPC.ObservableKind.STATUS}
    clean, faulty = newton.solvers.SolverIPC(model), newton.solvers.SolverIPC(model)
    observables = faulty.observables(kinds)
    c0, f0 = model.state(), model.state()
    for _ in range(10):
        c1, f1 = model.state(), model.state()
        clean.step(c0, c1, None, None, 0.01)
        faulty.step(f0, f1, None, None, 0.01, observables=observables)
        c0, f0 = c1, f1
    height = float(f0.particle_q.numpy()[particle, 2])
    start = model.state()
    start.assign(f0)
    _write_particle(start, particle, z=-0.01)
    out, c1 = model.state(), model.state()
    clean.step(c0, c1, None, None, 0.01)
    with test.assertRaisesRegex(RuntimeError, r"failed in 1 of its 3 worlds, world 1 on [\w|]+\."):
        faulty.step(start, out, None, None, 0.01, observables=observables)
    words = observables.status.numpy()
    test.assertTrue(words[1] & failed)
    test.assertEqual([int(words[k] & failed) for k in (0, 2, 3)], [0, 0, 0])
    p1, b1 = rows[1]
    for name, index in (("particle_q", p1), ("particle_qd", p1), ("body_q", b1), ("body_qd", b1)):
        np.testing.assert_array_equal(getattr(out, name).numpy()[index], getattr(start, name).numpy()[index], name)
    for k in (0, 2):
        p, b = rows[k]
        np.testing.assert_allclose(out.particle_q.numpy()[p], c1.particle_q.numpy()[p], rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(out.body_q.numpy()[b], c1.body_q.numpy()[b], rtol=0.0, atol=1e-6)
        test.assertGreater(np.abs(out.particle_q.numpy()[p] - start.particle_q.numpy()[p]).max(), 1e-6)  # it stepped
    _write_particle(out, particle, z=height)  # written back above the ground: every world steps on
    again = model.state()
    faulty.step(out, again, None, None, 0.01, observables=observables)
    test.assertEqual(int(np.bitwise_or.reduce(observables.status.numpy()) & failed), 0)
    _write_particle(again, particle, qd=(np.nan, 0.0, 0.0))
    nan_out = model.state()
    with test.assertRaisesRegex(RuntimeError, r"failed in 1 of its 3 worlds, world 1 on NONFINITE"):
        faulty.step(again, nan_out, None, None, 0.01, observables=observables)
    words = observables.status.numpy()
    test.assertTrue(words[1] & failed)
    test.assertEqual([int(words[k] & failed) for k in (0, 2, 3)], [0, 0, 0])
    np.testing.assert_array_equal(nan_out.particle_q.numpy()[p1], again.particle_q.numpy()[p1])

    # captured: the failure in world 1's entry alone, check_status names world 1
    solver = newton.solvers.SolverIPC(model)
    observables = solver.observables(kinds)
    s0, s1 = model.state(), model.state()
    solver.step(s0, s1, None, None, 0.01, observables=observables)  # loads the state and the modules
    s0.assign(s1)
    with wp.ScopedCapture(device=device) as capture:
        solver.step(s0, s1, None, None, 0.01, observables=observables)
    wp.capture_launch(capture.graph)
    test.assertFalse(solver.check_status(observables))
    s0.assign(s1)
    _write_particle(s0, particle, z=-0.01)
    wp.capture_launch(capture.graph)
    words = observables.status.numpy()
    test.assertTrue(words[1] & failed)
    test.assertEqual([int(words[k] & failed) for k in (0, 2, 3)], [0, 0, 0])
    np.testing.assert_array_equal(s1.particle_q.numpy()[p1], s0.particle_q.numpy()[p1])
    for k in (0, 2):
        test.assertGreater(np.abs(s1.particle_q.numpy()[rows[k][0]] - s0.particle_q.numpy()[rows[k][0]]).max(), 1e-6)
    with test.assertRaisesRegex(
        RuntimeError, r"failed in world 1 on [\w|]+; the result of those worlds must not be used"
    ):
        solver.check_status(observables)


def test_independent_worlds_overflow_alone(test, device):
    """Three worlds of ``_world_content`` at heights 3 cm apart, world 1 with a quota of one pair of every active
    contact list (libuipc-warp's capacity per world, a count per world, 0 keeping a world's own; the worlds share the
    lists). Eager steps grow world 1's quota and solve the step again, every world then taking the Newton iterations of
    its run alone, with its positions within 1e-6 m. Captured beside a solver without the quota, a step in which world
    1 asks for more sets OVERFLOW in world 1's entry alone and leaves world 1 at the state it started from, bit for
    bit, while worlds 0 and 2 take the other solver's step within 1e-6 m; check_status(grow=True) grows the lists, and
    the step repeated from the same state reports nothing and takes every world to the other solver's step."""
    overflow = int(newton.solvers.SolverIPC.StatusBits.OVERFLOW)
    shifts, steps = (0.0, 0.03, 0.06), 30
    one_pair = {"active": {kind: [0, 1, 0] for kind in ("PP", "PE", "PT", "EE")}}
    model = _worlds_at(device, shifts)
    rows = _world_rows(model, len(shifts))
    solver = newton.solvers.SolverIPC(model)
    solver._make_engine(one_pair)
    first = solver._engine
    run = _record_worlds(model, solver, steps)
    test.assertIsNot(solver._engine, first)  # built again on world 1's grown quota
    np.testing.assert_array_equal(run["status"], 0)
    for k, shift in enumerate(shifts):
        one = _worlds_at(device, [shift])
        alone = _record_worlds(one, newton.solvers.SolverIPC(one), steps)
        np.testing.assert_array_equal(run["iterations"][:, k], alone["iterations"][:, 0], f"world {k}")
        test.assertLess(np.abs(run["q"][:, k] - alone["q"][:, 0]).max(), 1e-6, f"world {k}")

    def of_world(state, k, names=("particle_q", "particle_qd", "body_q", "body_qd")):
        p, b = rows[k]
        return [getattr(state, name).numpy()[p if name.startswith("particle") else b] for name in names]

    def apart(a, b, k):
        """The largest distance of world ``k``'s particle and body positions between states ``a`` and ``b``."""
        names = ("particle_q", "body_q")
        return max(np.abs(x - y).max() for x, y in zip(of_world(a, k, names), of_world(b, k, names), strict=True))

    def capture(solver, observables, s0, s1):
        with wp.ScopedCapture(device=device) as cap:
            solver.step(s0, s1, None, None, 0.01, observables=observables)
        return cap.graph

    kinds = {newton.solvers.SolverIPC.ObservableKind.STATUS}
    plain, solver = newton.solvers.SolverIPC(model), newton.solvers.SolverIPC(model)
    solver._make_engine(one_pair)
    plain_observables, observables = plain.observables(kinds), solver.observables(kinds)
    p0, p1, s0, s1 = (model.state() for _ in range(4))
    plain_graph, graph, grown = capture(plain, plain_observables, p0, p1), capture(solver, observables, s0, s1), 0
    for _ in range(steps):
        wp.capture_launch(plain_graph)
        wp.capture_launch(graph)
        test.assertFalse(np.any(plain_observables.status.numpy() & overflow))
        words = observables.status.numpy()
        if np.any(words & overflow):
            test.assertEqual([bool(w & overflow) for w in words], [False, True, False, False])
            for got, start in zip(of_world(s1, 1), of_world(s0, 1), strict=True):
                np.testing.assert_array_equal(got, start)  # world 1 at the state it started from
            for k in (0, 2):
                test.assertLess(apart(s1, p1, k), 1e-6, f"world {k}")
                test.assertGreater(apart(s1, s0, k), 1e-6, f"world {k}")  # it took its step
            test.assertTrue(solver.check_status(observables, grow=True))
            grown += 1
            graph = capture(solver, observables, s0, s1)  # the step again from the same input state, grown
            wp.capture_launch(graph)
            test.assertFalse(solver.check_status(observables))
        for k in range(len(shifts)):
            test.assertLess(apart(s1, p1, k), 1e-6, f"world {k}")
        s0.assign(s1)
        p0.assign(p1)
    test.assertGreater(grown, 0)


def _two_worlds(device, content, extra=None):
    """Two worlds of ``content(builder)`` over a global ground plane, with ``extra(builder)`` outside the worlds."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for _ in range(2):
        builder.begin_world()
        content(builder)
        builder.end_world()
    if extra is not None:
        extra(builder)
    return builder.finalize(device=device)


def test_rejects_what_independent_worlds_do_not_take(test, device):
    """Content in several worlds is independent worlds by default: a model of two worlds builds without
    coupled_worlds, and one of a single world builds as before. Independent worlds refuse, naming coupled_worlds=True as
    an alternative: AL-IPC contact and MAS, which libuipc-warp's independent worlds do not take yet; a free particle and
    a dynamic body of the global world, which every world would push; a global kinematic body with
    kinematic_motion="aim", an unknown too (with the jump it is fixed and builds); and a spring between two worlds. With
    coupled_worlds=True every one of them builds. adaptive_kappa=True builds with independent worlds, every world in a
    stiffness range of its own (test_independent_worlds_take_their_own_stiffness_range). Of two worlds, one of which
    holds a static shape alone, one world moves, and the options are taken as for one world."""

    def box(builder):
        _box_body(builder, (0.0, 0.0, 0.3), (0.05, 0.05, 0.05))

    def particle(builder):
        builder.add_particle(wp.vec3(0.3, 0.0, 0.3), wp.vec3(0.0, 0.0, 0.0), 0.01, radius=0.01)

    def soft(builder):
        builder.add_soft_grid(
            pos=wp.vec3(-0.05, -0.05, 0.2), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0), dim_x=2, dim_y=2,
            dim_z=2, cell_x=0.05, cell_y=0.05, cell_z=0.05, density=1000.0, k_mu=3.4e4, k_lambda=1.6e5, k_damp=0.0,
        )  # fmt: skip

    def global_box(builder):
        _box_body(builder, (1.0, 0.0, 0.3), (0.05, 0.05, 0.05))

    def global_pusher(builder):
        pusher = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.3), wp.quat_identity()), is_kinematic=True)
        builder.add_shape_box(pusher, hx=0.05, hy=0.05, hz=0.05)

    def spring(builder):
        builder.add_spring(0, 1, 100.0, 0.0, 0.0)  # world 0's particle to world 1's

    solver = newton.solvers.SolverIPC(_two_worlds(device, box))
    test.assertTrue(solver._independent)
    one = newton.solvers.SolverIPC(_worlds(device, 1))
    test.assertFalse(one._independent)
    test.assertIsNone(one._worlds)
    cases = (
        ("AL-IPC", box, None, {"contact_constitution": "al-ipc"}, r"contact_constitution='al-ipc' is not offered"),
        ("MAS", soft, None, {"fem_preconditioner": "mas"}, r"fem_preconditioner='mas' is not offered"),
        ("global particle", box, particle, {}, r"particle 0 of the global world -1 is free"),
        ("global body", box, global_box, {}, r"body 2 .*of the global world -1 is a dynamic body"),
        ("global aim", box, global_pusher, {"kinematic_motion": "aim"}, r"global world -1 is a kinematic body.*'jump'"),
        ("spring", particle, spring, {}, r"spring 0 joins free particles of the worlds 0 and 1"),
    )
    for name, content, extra, options, pattern in cases:
        with test.subTest(name):
            model = _two_worlds(device, content, extra)
            with test.assertRaisesRegex(NotImplementedError, pattern + r"(.|\n)*coupled_worlds=True"):
                newton.solvers.SolverIPC(model, **options)
            test.assertTrue(newton.solvers.SolverIPC(model, coupled_worlds=True, **options)._coupled_worlds)
    jump = newton.solvers.SolverIPC(_two_worlds(device, box, global_pusher), kinematic_motion="jump")
    test.assertTrue(jump._independent)
    adaptive = newton.solvers.SolverIPC(_two_worlds(device, box), adaptive_kappa=True, kappa=None)
    test.assertTrue(adaptive._independent)
    test.assertEqual(adaptive.contact_stiffness(0.01).shape, (2,))
    # one world that moves, the other a static shape alone: solved as one world, which takes the options
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.begin_world()
    box(builder)
    builder.end_world()
    builder.begin_world()
    builder.add_shape_box(-1, xform=wp.transform(wp.vec3(1.0, 0.0, 0.3), wp.quat_identity()), hx=0.05, hy=0.05, hz=0.05)
    builder.end_world()
    still = newton.solvers.SolverIPC(builder.finalize(device=device), adaptive_kappa=True, kappa=None)
    test.assertTrue(still._independent)
    test.assertEqual(still._status_entries.tolist(), [0])


def test_independent_worlds_with_attachments(test, device):
    """Three worlds of a particle attached to a box that falls onto the global ground (``_attached_world``), from
    heights 5 cm apart, independent: at every step each world takes the Newton iterations of the same world alone, its
    positions within 1e-6 m of the run alone, and the anchors are in their worlds' subscenes. The solves are
    converged: at the default tolerances the particle's landing, which the barrier makes a stiff system, carries the
    rounding of sums that the scene adds in another order than the world alone to where the PCG and Newton loops stop,
    and the worlds land on the same iterations up to 4e-5 m apart over 40 steps (runs alone repeat bit for bit)."""
    shifts, steps = (0.0, 0.05, 0.1), 40
    converged = {"newton_velocity_tol": 1e-6, "linear_tol_rate": 1e-12, "semi_implicit": False}

    def run(shifts):
        model = _worlds_at(device, shifts, _attached_world)
        solver = newton.solvers.SolverIPC(model, **converged)
        if len(shifts) > 1:
            layout = solver._scene.layout()
            anchors = np.arange(layout.body_anchor_offsets[0], layout.body_anchor_offsets[-1])
            np.testing.assert_array_equal(solver._scids[anchors], [1, 2, 3])
        return _record_worlds(model, solver, steps)

    worlds = run(shifts)
    for k, shift in enumerate(shifts):
        alone = run([shift])
        np.testing.assert_array_equal(worlds["iterations"][:, k], alone["iterations"][:, 0], f"world {k}")
        test.assertLess(np.abs(worlds["q"][:, k] - alone["q"][:, 0]).max(), 1e-6, f"world {k}")
    test.assertLess(worlds["q"][-1, 0, 2], 0.1)  # it fell, the particle under the box


def test_independent_worlds_with_kinematic_aims(test, device):
    """Three worlds of a box on the global ground and a kinematic box behind it (``_pushed_world``) with
    kinematic_motion="aim", the pushers moved by state_in at 1, 0.5 and 0 m/s: at every step each world takes the Newton
    iterations of the same world alone with its pusher's speed, its positions within 1e-6 m of the run alone, and the
    boxes are pushed as far as their pushers go. The solves are converged, as in the coupled worlds' test."""
    speeds, steps = (1.0, 0.5, 0.0), 20
    converged = {"newton_velocity_tol": 1e-6, "newton_transrate_tol": 1e-6, "linear_tol_rate": 1e-12}

    def run(speeds):
        scene = newton.ModelBuilder()
        scene.add_ground_plane()
        for _ in speeds:
            scene.add_world(_pushed_world())
        model = scene.finalize(device=device)
        solver = newton.solvers.SolverIPC(model, kinematic_motion="aim", semi_implicit=False, **converged)
        rows = _world_rows(model, len(speeds))
        s0, s1 = model.state(), model.state()
        q, iterations = [], []
        for k in range(steps):
            body_q, body_qd = s0.body_q.numpy(), s0.body_qd.numpy()
            for (_, b), v in zip(rows, speeds, strict=True):
                body_q[b[1], 0] = -0.12 + v * 0.01 * k
                body_qd[b[1]] = (v, 0.0, 0.0, 0.0, 0.0, 0.0)
            s0.body_q.assign(body_q)
            s0.body_qd.assign(body_qd)
            solver.step(s0, s1, None, None, 0.01)
            s0, s1 = s1, s0
            iterations.append(_world_iterations(solver))
            q.append([s0.body_q.numpy()[b][:, :3].reshape(-1) for _, b in rows])
        return np.array(q), np.array(iterations)

    q, iterations = run(speeds)
    for k, v in enumerate(speeds):
        q1, iterations1 = run([v])
        np.testing.assert_array_equal(iterations[:, k], iterations1[:, 0], f"world {k}")
        test.assertLess(np.abs(q[:, k] - q1[:, 0]).max(), 1e-6, f"world {k}")
    test.assertGreater(q[-1, 0, 0], 0.05)  # pushed at 1 m/s
    test.assertGreater(q[-1, 0, 0], q[-1, 1, 0] + 0.05)  # further than at 0.5 m/s
    test.assertAlmostEqual(q[-1, 2, 0], 0.0, delta=1e-3)  # not pushed


def test_many_independent_worlds(test, device):
    """1100 independent worlds at one place, each a box and a particle dropped onto the global ground from heights that
    differ from world to world: five contact elements times 1101 subscenes. In 20 steps every box and particle lands on
    the ground in its own world, no contact term joins two worlds, and every world's STATUS entry holds no failure."""
    import uipc_warp.arrays  # noqa: PLC0415

    worlds = 1100
    scene = newton.ModelBuilder()
    scene.add_ground_plane()
    for k in range(worlds):
        world = newton.ModelBuilder()
        _box_body(world, (0.0, 0.0, 0.07 + 1e-5 * k), (0.05, 0.05, 0.05))
        world.add_particle(wp.vec3(0.2, 0.0, 0.03 + 1e-5 * k), wp.vec3(0.0, 0.0, 0.0), 0.01, radius=0.01)
        scene.add_world(world)
    model = scene.finalize(device=device)
    solver = newton.solvers.SolverIPC(model)
    test.assertTrue(solver._independent)
    test.assertEqual(len(solver._status_entries), worlds)
    observables = solver.observables({solver.ObservableKind.STATUS})
    engine = solver._engine
    export = uipc_warp.arrays.ContactForces(engine)
    s0, s1 = model.state(), model.state()
    for _ in range(20):
        solver.step(s0, s1, None, None, 0.01, observables=observables)
        s0, s1 = s1, s0
        test.assertIs(solver._engine, engine)
        for sub in _term_subscenes(solver, export):
            test.assertEqual(_pairs_across_worlds(sub)[0], 0)
    test.assertEqual(int(np.bitwise_or.reduce(observables.status.numpy()) & int(solver.StatusBits.FAILED)), 0)
    z_box, z_particle = s0.body_q.numpy()[:, 2], s0.particle_q.numpy()[:, 2]
    test.assertGreater(z_box.min(), 0.05)
    test.assertLess(z_box.max(), 0.05 + D_HAT)
    test.assertGreater(z_particle.min(), 0.01)
    test.assertLess(z_particle.max(), 0.01 + D_HAT)


@requires("uipc_warp")
class TestSolverIPCWorlds(unittest.TestCase):
    """Several worlds, kept apart in contact: independent by default, each taking the step it would take alone, or
    solved together with coupled_worlds."""


devices = get_cuda_test_devices()
for name, func in (
    ("test_coupled_worlds_follow_the_single_world", test_coupled_worlds_follow_the_single_world),
    ("test_coupled_worlds_are_built_as_one_world", test_coupled_worlds_are_built_as_one_world),
    ("test_coupled_worlds_share_filter_classes", test_coupled_worlds_share_filter_classes),
    ("test_coupled_worlds_status_in_every_world", test_coupled_worlds_status_in_every_world),
    ("test_coupled_worlds_fall_with_their_gravity", test_coupled_worlds_fall_with_their_gravity),
    ("test_coupled_worlds_beyond_the_old_element_limit", test_coupled_worlds_beyond_the_old_element_limit),
    ("test_start_check_of_many_worlds", test_start_check_of_many_worlds),
    ("test_near_parts_forms_no_pair_of_two_worlds", test_near_parts_forms_no_pair_of_two_worlds),
    ("test_coupled_worlds_with_attachments", test_coupled_worlds_with_attachments),
    ("test_coupled_worlds_with_kinematic_aims", test_coupled_worlds_with_kinematic_aims),
    ("test_reset_of_one_world_keeps_the_others", test_reset_of_one_world_keeps_the_others),
    ("test_reset_of_one_world_keeps_the_other_worlds_aims", test_reset_of_one_world_keeps_the_other_worlds_aims),
    ("test_independent_worlds_step_as_alone", test_independent_worlds_step_as_alone),
    ("test_independent_worlds_take_their_own_stiffness_range", test_independent_worlds_take_their_own_stiffness_range),
    ("test_independent_world_fails_alone", test_independent_world_fails_alone),
    ("test_rejects_what_independent_worlds_do_not_take", test_rejects_what_independent_worlds_do_not_take),
    ("test_independent_worlds_with_attachments", test_independent_worlds_with_attachments),
    ("test_independent_worlds_with_kinematic_aims", test_independent_worlds_with_kinematic_aims),
    ("test_many_independent_worlds", test_many_independent_worlds),
    ("test_independent_worlds_overflow_alone", test_independent_worlds_overflow_alone),
):
    add_function_test(TestSolverIPCWorlds, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
