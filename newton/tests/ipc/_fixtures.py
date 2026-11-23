# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scenes, solver settings and helpers shared by the SolverIPC test modules."""

import contextlib
import inspect
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


# tight tolerances of the comparisons with SolverIPC alone
_TIGHT = {"newton_velocity_tol": 1.0e-6, "linear_tol_rate": 1.0e-12, "semi_implicit": False}


# converged steps for proxies: a proxy's feedback is the force of the step's contact terms at its end, which the stiff
# barrier makes as accurate as the Newton loop's convergence
_CONVERGED = {"newton_velocity_tol": 1.0e-4, "linear_tol_rate": 1.0e-10, "semi_implicit": False}
