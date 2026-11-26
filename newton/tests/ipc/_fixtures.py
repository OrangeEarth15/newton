# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scenes, solver settings and helpers shared by the SolverIPC test modules."""

import contextlib
import inspect
import re
import warnings
from unittest import mock

import numpy as np
import warp as wp

import newton

# SolverIPC's default barrier activation distance; a resting contact settles just inside it
D_HAT = 0.01


@contextlib.contextmanager
def _scene_calls(method):
    """The calls of ``method``, a method of libuipc-warp's ``ArrayScene``, while the block runs: what SolverIPC gives
    the engine through the public array path, each call's arguments by name."""
    import uipc_warp.arrays  # noqa: PLC0415

    signature, calls = inspect.signature(method), []

    def record(*args, **kwargs):
        calls.append(signature.bind(*args, **kwargs).arguments)
        return method(*args, **kwargs)

    with mock.patch.object(uipc_warp.arrays.ArrayScene, method.__name__, autospec=True, side_effect=record):
        yield calls


def _active_slots(solver):
    """The slots of the engine's active contact lists PT, EE, PE and PP, which libuipc-warp's ``ContactForces`` counts
    (``segment_capacities``, its first four segments)."""
    import uipc_warp.arrays  # noqa: PLC0415

    return uipc_warp.arrays.ContactForces.segment_capacities(solver._engine)[:4]


def _soft_cube(device, height=0.05, ground=True, fix_left=False):
    builder = newton.ModelBuilder()  # z-up, gravity -9.81
    builder.add_soft_grid(
        pos=wp.vec3(-0.1, -0.1, height),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=4,
        dim_y=4,
        dim_z=4,
        cell_x=0.05,
        cell_y=0.05,
        cell_z=0.05,
        density=1000.0,
        k_mu=3.4e4,
        k_lambda=1.6e5,
        k_damp=0.0,
        fix_left=fix_left,
    )
    if ground:
        builder.add_ground_plane()
    return builder.finalize(device=device)


def _run(model, solver, frames, dt=0.01, capture=False, field="particle_q"):
    """Trajectory of ``field`` (``"particle_q"``, ``"body_q"`` or a tuple of both) over ``frames`` steps."""
    fields = (field,) if isinstance(field, str) else tuple(field)
    s0 = model.state()
    s1 = model.state()
    traj = {f: [] for f in fields}

    def record(state):
        for f in fields:
            traj[f].append(getattr(state, f).numpy())

    if capture:
        # one eager step loads the state and compiles the kernels, then capture a two-step swap
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        record(s0)
        with wp.ScopedCapture(device=model.device) as cap:
            solver.step(s0, s1, None, None, dt)
            solver.step(s1, s0, None, None, dt)
        for _ in range((frames - 1) // 2):
            wp.capture_launch(cap.graph)
            record(s0)
    else:
        for _ in range(frames):
            solver.step(s0, s1, None, None, dt)
            s0, s1 = s1, s0
            record(s0)
    out = tuple(np.array(traj[f]) for f in fields)
    return out[0] if isinstance(field, str) else out


def _cloth(builder, n, cell, z, center=(0.0, 0.0), mass=2.0e-4, tri_k=1.0e2, edge_ke=1.0e-2):
    """A horizontal n x n cloth grid centred over ``center`` at height ``z``, without damping."""
    builder.add_cloth_grid(
        pos=wp.vec3(center[0] - 0.5 * n * cell, center[1] - 0.5 * n * cell, z),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=n,
        dim_y=n,
        cell_x=cell,
        cell_y=cell,
        mass=mass,
        tri_ke=tri_k,
        tri_ka=tri_k,
        tri_kd=0.0,
        edge_ke=edge_ke,
        edge_kd=0.0,
    )


def _box_body(builder, center, half, rot=None):
    """A rigid box body (density 1000 kg/m^3) centred at ``center``."""
    b = builder.add_body(xform=wp.transform(wp.vec3(*center), rot or wp.quat_identity()))
    builder.add_shape_box(b, hx=half[0], hy=half[1], hz=half[2])
    return b


def _implicit_euler_oscillator(u0, omega, dt, steps, w0=0.0):
    """u_1 .. u_steps of u'' = -omega^2 u from (u0, w0) under implicit Euler: u_{n+1} (1 + omega^2 dt^2) = u_n + dt w_n,
    w_{n+1} = (u_{n+1} - u_n) / dt."""
    u, w, out = u0, w0, []
    for _ in range(steps):
        u_next = (u + dt * w) / (1.0 + (omega * dt) ** 2)
        u, w = u_next, (u_next - u) / dt
        out.append(u)
    return np.array(out)


def _boxes_apart(device):
    """A box 1 mm above a static box, and the pose ``body_q`` that rests it exactly on the static box: their faces
    then touch at z = 0.25 (all coordinates exact)."""
    builder = newton.ModelBuilder()
    builder.add_shape_box(
        -1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.125), wp.quat_identity()), hx=0.125, hy=0.125, hz=0.125
    )
    _box_body(builder, (0.0, 0.0, 0.376), (0.125, 0.125, 0.125))
    model = builder.finalize(device=device)
    touching = model.body_q.numpy()
    touching[0, 2] = 0.375
    return model, touching


# small active contact lists, so that the contact of two soft cubes overflows them
_TINY_LISTS = {"active": dict.fromkeys(("PP", "PE", "PT", "EE"), 1)}


def _stacking_cubes(device):
    """A soft cube that falls 4 cm onto another resting on the ground."""
    builder = newton.ModelBuilder()
    for z in (0.0005, 0.14):
        builder.add_soft_grid(
            pos=wp.vec3(-0.05, -0.05, z),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=2,
            dim_y=2,
            dim_z=2,
            cell_x=0.05,
            cell_y=0.05,
            cell_z=0.05,
            density=1000.0,
            k_mu=3.4e4,
            k_lambda=1.6e5,
            k_damp=0.0,
        )
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _world_content():
    """One world: a soft cube above the global static box, a rigid box and a free particle above the ground."""
    world = newton.ModelBuilder()
    world.add_soft_grid(
        pos=wp.vec3(0.3, -0.05, 0.25),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=2,
        dim_y=2,
        dim_z=2,
        cell_x=0.05,
        cell_y=0.05,
        cell_z=0.05,
        density=1000.0,
        k_mu=3.4e4,
        k_lambda=1.6e5,
        k_damp=0.0,
    )
    _box_body(world, (-0.3, 0.0, 0.15), (0.05, 0.05, 0.05))
    world.add_particle(wp.vec3(0.0, 0.3, 0.1), wp.vec3(0.0, 0.0, 0.0), 0.01, radius=0.01)
    return world


def _worlds(device, copies, content=_world_content):
    """``copies`` worlds of ``content``, all at the origin as ``replicate`` places them, over a global ground plane and
    a global static box whose top is at z = 0.1."""
    scene = newton.ModelBuilder()
    scene.add_ground_plane()
    scene.add_shape_box(-1, xform=wp.transform(wp.vec3(0.35, 0.0, 0.05), wp.quat_identity()), hx=0.15, hy=0.15, hz=0.05)
    scene.replicate(content(), copies)
    return scene.finalize(device=device)


def _world_rows(model, copies):
    """The particle and body indices of every world, in the order of one world's."""
    pw, bw = model.particle_world.numpy(), model.body_world.numpy()
    return [(np.flatnonzero(pw == k), np.flatnonzero(bw == k)) for k in range(copies)]


def _solver_snapshot(solver):
    """What a step changes: the engine's state (its dump) and the solver's own device state."""
    arrays = [solver._body_q_written, solver._body_qd_written, solver._body_hold]
    for history in solver._joint_history.values():
        arrays += list(history)
    return solver._engine.state_dict(), [(a, a.numpy().copy()) for a in arrays]


def _restore(solver, snapshot):
    """The solver at ``snapshot``: the engine recovered from its dump (the friction candidates of the next frame from
    one DCD pass at its state, the same in every recovery) and the solver's device state."""
    state, arrays = snapshot
    solver._engine.load_state_dict(state)
    for a, value in arrays:
        a.assign(value)


# the entries of a contact term of each kind, as libuipc-warp's ContactForces records them (PH: vertex, plane)
_PAIR_WIDTH = {"PT": 4, "EE": 4, "PE": 3, "PP": 2, "PH": 2}


def _frame_start(solver, export):
    """What the engine's last frame started with, through public names: its friction candidates (the vertex tuples of
    the friction terms of libuipc-warp's ``ContactForces`` ``export``, updated here; PH: vertex and plane), the friction
    it discarded and the rows whose BDF2 history it restarted (the engine's ``frame_start_state``), and the revolute
    drives' terms. What the engine keeps of a candidate (its lagged record) and the prediction and masses of a
    restarted row are libuipc-warp's own tests' (tests/test_subscene_friction.py, tests/test_subscene_restart.py)."""
    export.update()
    terms, start = export.numpy(), solver._engine.frame_start_state()
    group = solver._joint_groups.revolute
    return {
        "lists": {name: terms[f"{name}+F"][0][:, :width] for name, width in _PAIR_WIDTH.items()},
        "discard": start["discard_friction"],
        "discarded": start.get("discard_subscenes"),
        "restarted": start.get("restart_rows"),
        "drive": None if group is None else np.stack([group.drv_aim.numpy(), group.drv_ratio.numpy()], axis=1),
    }


def _worlds_at(device, shifts, content=_world_content):
    """One world of ``content`` per entry of ``shifts``, raised by it, over the global ground plane and static box of
    ``_worlds``: worlds that differ. ``_worlds_at(device, [shift])`` is one of them alone."""
    scene = newton.ModelBuilder()
    scene.add_ground_plane()
    scene.add_shape_box(-1, xform=wp.transform(wp.vec3(0.35, 0.0, 0.05), wp.quat_identity()), hx=0.15, hy=0.15, hz=0.05)
    for shift in shifts:
        scene.add_world(content(), xform=wp.transform(wp.vec3(0.0, 0.0, shift), wp.quat_identity()))
    return scene.finalize(device=device)


def _write_particle(state, particle, z=None, qd=None):
    """``state`` with particle ``particle`` at height ``z`` or with velocity ``qd``."""
    if z is not None:
        q = state.particle_q.numpy()
        q[particle, 2] = z
        state.particle_q.assign(q)
    if qd is not None:
        v = state.particle_qd.numpy()
        v[particle] = qd
        state.particle_qd.assign(v)


def _hinged(device, kind="revolute", gravity=0.0, axis=newton.Axis.Z, offset=(0.0, 0.0, 0.0), radius=0.1, **dof):
    """A sphere hanging from the world on a revolute or prismatic joint (y up); ``offset`` is the joint's origin in
    the sphere's frame, and ``dof`` the joint's arguments. Returns the model, a state at its joint coordinates and
    the joint."""
    builder = newton.ModelBuilder(gravity=(0.0, -gravity, 0.0), up_axis=newton.Axis.Y)
    link = builder.add_link()
    builder.add_shape_sphere(link, radius=radius)
    add = builder.add_joint_revolute if kind == "revolute" else builder.add_joint_prismatic
    j = add(-1, link, axis=axis, child_xform=wp.transform(offset, wp.quat_identity()), **dof)
    builder.add_articulation([j])
    model = builder.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, state, j


def _steps(model, solver, state, steps, dt, control=None, record=lambda s: s.joint_q.numpy().copy()):
    """``steps`` steps from ``state``; what ``record`` takes of the state after each."""
    s0, s1 = state, model.state()
    out = []
    for _ in range(steps):
        solver.step(s0, s1, control, None, dt)
        s0, s1 = s1, s0
        out.append(record(s0))
    return np.array(out), s0


def _drive_recurrence(
    inertia, h, steps, q0=0.0, qd0=0.0, ke=0.0, kd=0.0, ks=0.0, c=0.0, q_target=0.0, qd_target=0.0, q_rest=0.0, f=0.0
):
    """q_1 .. q_steps of implicit Euler for inertia q'' = ke (q* - q) + kd (qd* - qd) - ks (q - q_rest) - c qd + f."""
    q, qd, out = q0, qd0, []
    k = inertia / h**2 + ke + ks + (kd + c) / h
    for _ in range(steps):
        q_next = (
            inertia * (q + h * qd) / h**2 + ke * q_target + ks * q_rest + kd * (q / h + qd_target) + c * q / h + f
        ) / k
        q, qd = q_next, (q_next - q) / h
        out.append(q)
    return np.array(out)


def _drive_recurrence_bdf2(
    inertia, h, steps, q0=0.0, qd0=0.0, ke=0.0, kd=0.0, ks=0.0, c=0.0, q_target=0.0, qd_target=0.0, q_rest=0.0, f=0.0
):
    """q_1 .. q_steps of BDF2 for inertia q'' = ke (q* - q) + kd (qd* - qd) - ks (q - q_rest) - c qd + f after an
    implicit Euler step (``_drive_recurrence``): ``I (q' - q~) = 4/9 h^2 tau(q', qd')`` with
    ``q~ = 4/3 q - 1/3 q_prev + 2/9 h (4 qd - qd_prev)`` and ``qd' = (3 q' - 4 q + q_prev) / (2 h)``."""
    q1 = _drive_recurrence(inertia, h, 1, q0, qd0, ke, kd, ks, c, q_target, qd_target, q_rest, f)[0]
    q_prev, qd_prev, q, qd = q0, qd0, q1, (q1 - q0) / h
    out = [q1]
    for _ in range(steps - 1):
        predicted = 4.0 / 3.0 * q - q_prev / 3.0 + 2.0 / 9.0 * h * (4.0 * qd - qd_prev)
        a, b = 1.5 / h, (4.0 * q - q_prev) / (2.0 * h)  # qd' = a q' - b
        load = ke * q_target + kd * qd_target + ks * q_rest + f + (kd + c) * b
        w = 4.0 / 9.0 * h * h
        q_next = (inertia * predicted + w * load) / (inertia + w * (ke + ks + (kd + c) * a))
        q_prev, qd_prev, q, qd = q, qd, q_next, a * q_next - b
        out.append(q)
    return np.array(out)


def _native_contacts(solver, model, rigid=None, soft=None):
    """The step's Contacts of a native contact backend, sized as the solver's export asks unless given."""
    pipeline = newton.CollisionPipeline(
        model,
        rigid_contact_max=solver.get_max_contact_count() if rigid is None else rigid,
        soft_contact_max=solver.get_max_soft_contact_count() if soft is None else soft,
    )
    return pipeline.contacts()


def _transform_points(X, points):
    """``points`` (n, 3) of a body frame in the world, at the pose ``X`` (p, quaternion xyzw)."""
    p, q = np.asarray(X[:3], np.float64), np.asarray(X[3:], np.float64)
    v = np.asarray(points, np.float64).reshape(-1, 3)
    t = 2.0 * np.cross(q[:3], v)
    return v + q[3] * t + np.cross(q[:3], t) + p


def _rigid_rows(contacts, observables):
    """The written rigid rows: (shape0, shape1, contact_f rows)."""
    n = min(int(contacts.rigid_contact_count.numpy()[0]), contacts.rigid_contact_max)
    return (
        contacts.rigid_contact_shape0.numpy()[:n],
        contacts.rigid_contact_shape1.numpy()[:n],
        observables.contact_f.numpy()[:n],
    )


# Tight Newton tolerances, for comparing SolverIPC's steps with a closed form or with SolverIPC alone: a step ends once a
# Newton update moves no vertex by more than newton_velocity_tol * dt (1e-6 m/s: 1e-8 m at dt = 0.01 s), each linear
# solve reaches 1e-12 of its residual, and libuipc's semi-implicit early exit is off. The rotations of affine bodies
# keep the default newton_transrate_tol (0.1 1/s); _TIGHT_ROTATIONS bounds them too.
_TIGHT = {"newton_velocity_tol": 1.0e-6, "linear_tol_rate": 1.0e-12, "semi_implicit": False}

# _TIGHT with newton_transrate_tol 1e-6 1/s as well: a step also ends only once a Newton update changes no affine body's
# matrix by more than 1e-6 * dt, for the tests of turning bodies and of joint coordinates
_TIGHT_ROTATIONS = {**_TIGHT, "newton_transrate_tol": 1.0e-6}


# converged steps for proxies: a proxy's feedback is the force of the step's contact terms at its end, which the stiff
# barrier makes as accurate as the Newton loop's convergence
_CONVERGED = {"newton_velocity_tol": 1.0e-4, "linear_tol_rate": 1.0e-10, "semi_implicit": False}


# Content for the refusal tests: each ``_with_*`` returns a function of the builder and an x position that adds one
# item there; ``_model_of`` puts several side by side.
def _model_of(device, *contents):
    """A model of ``contents``, the k-th at x = 2k, in this order."""
    builder = newton.ModelBuilder()
    for k, content in enumerate(contents):
        content(builder, 2.0 * k)
    return builder.finalize(device=device)


def _link(builder, x):
    link = builder.add_link(xform=wp.transform((x, 0.0, 1.0), wp.quat_identity()))
    builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
    return link


def _with_joint(add, offset=0, **dof):
    """A link and the joint ``add(builder, link, x)`` from the world, in an articulation of its own; ``dof`` sets the
    joint's arrays (``joint_<name>``) at its DOF ``offset``."""

    def content(builder, x):
        j = add(builder, _link(builder, x), x)
        builder.add_articulation([j])
        for name, value in dof.items():
            getattr(builder, f"joint_{name}")[builder.joint_qd_start[j] + offset] = value

    return content


def _with_revolute(**dof):
    return _with_joint(lambda b, link, x: b.add_joint_revolute(-1, link, axis=newton.Axis.Y), **dof)


def _with_ball(offset=0, **dof):
    def add(b, link, x):
        return b.add_joint_ball(-1, link, parent_xform=wp.transform((x, 0.0, 1.2), wp.quat_identity()))

    return _with_joint(add, offset, **dof)


def _with_d6(linear, angular, offset=0, **dof):
    """A D6 joint with a linear axis per entry of ``linear`` and an angular axis per entry of ``angular``."""
    D = newton.ModelBuilder.JointDofConfig

    def add(b, link, x):
        return b.add_joint_d6(
            -1, link, linear_axes=[D(axis=a) for a in linear], angular_axes=[D(axis=a) for a in angular]
        )

    return _with_joint(add, offset, **dof)


def _with_distance(lower, upper):
    def add(b, link, x):
        return b.add_joint_distance(
            -1, link, parent_xform=wp.transform((x, 0.0, 1.5), wp.quat_identity()), min_distance=lower,
            max_distance=upper,
        )  # fmt: skip

    return _with_joint(add)


def _with_rod():
    """A rod joint whose first two axes ``add_joint_rod`` made are swapped afterwards: a layout SolverIPC refuses."""

    def add(b, link, x):
        j = b.add_joint_rod(-1, link)
        s = b.joint_qd_start[j]
        b.joint_axis[s], b.joint_axis[s + 1] = b.joint_axis[s + 1], b.joint_axis[s]
        return j

    return _with_joint(add)


def _with_free(offset=0, **dof):
    """A free body, whose free joint ``add_body`` adds; ``dof`` sets the joint's arrays at its DOF ``offset``."""

    def content(builder, x):
        _box_body(builder, (x, 0.0, 2.0), (0.1, 0.1, 0.1))
        for name, value in dof.items():
            getattr(builder, f"joint_{name}")[builder.joint_qd_start[-1] + offset] = value

    return content


def _with_tendon():
    """One of Newton's own tendons between two links."""

    def content(builder, x):
        a, b = _link(builder, x), _link(builder, x + 0.3)
        builder.add_tendon([newton.TendonGuide(body=a), newton.TendonGuide(body=b)])

    return content


def _with_muscle():
    """A legacy muscle between two links."""

    def content(builder, x):
        a, b = _link(builder, x), _link(builder, x + 0.3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", newton.NewtonDeprecationWarning)  # legacy muscles are deprecated
            builder.add_muscle([a, b], [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)], 1.0, 0.5, 0.1, 1.0, 0.0)

    return content


def _with_vector_mimic():
    """A mimic of two D6 joints of two linear axes each, which SolverIPC refuses (one coordinate per relation)."""

    def content(builder, x):
        a, b = _link(builder, x), _link(builder, x + 0.3)
        axes = [newton.ModelBuilder.JointDofConfig(axis=a) for a in (newton.Axis.X, newton.Axis.Y)]
        joints = [builder.add_joint_d6(-1, link, linear_axes=axes) for link in (a, b)]
        builder.add_articulation(joints)
        builder.set_joint_mimic(joints[1], joints[0])

    return content


def _with_equality():
    """A body and a MuJoCo CONNECT row that holds it to the world, linked to no loop joint."""

    def content(builder, x):
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

    return content


def _with_spring(damping=0.0, rest=None):
    """Two particles 0.5 m apart (of Newton's default radius, 0.1 m) and a spring between them."""

    def content(builder, x):
        first = len(builder.particle_q)
        for k in range(2):
            builder.add_particle(wp.vec3(x + 0.5 * k, 3.0, 1.0), wp.vec3(0.0, 0.0, 0.0), 1.0)
        builder.add_spring(first, first + 1, 1.0e3, damping, 0.0)
        if rest is not None:
            builder.spring_rest_length[-1] = rest

    return content


def _with_attachment(damping=0.0, stiffness=None):
    """A particle attached to a body above it."""

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


def _assert_refused(test, model, message, **options):
    """Assert that building ``SolverIPC(model, **options)`` raises ``NotImplementedError`` with exactly ``message``."""
    with test.assertRaisesRegex(NotImplementedError, f"^{re.escape(message)}$"):
        newton.solvers.SolverIPC(model, **options)
