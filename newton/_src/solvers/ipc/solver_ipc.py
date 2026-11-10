# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Incremental Potential Contact (IPC) solver backed by libuipc-warp."""

from __future__ import annotations

import warnings

import numpy as np
import warp as wp

from ...core.types import override
from ...geometry import GeoType, ParticleFlags, ShapeFlags
from ...sim import Contacts, Control, Model, ModelFlags, State
from ..solver import SolverBase


@wp.kernel
def _load_particles(
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    x: wp.array[wp.vec3d],
    v: wp.array[wp.vec3d],
):
    i = wp.tid()
    p = q[i]
    u = qd[i]
    x[i] = wp.vec3d(wp.float64(p[0]), wp.float64(p[1]), wp.float64(p[2]))
    v[i] = wp.vec3d(wp.float64(u[0]), wp.float64(u[1]), wp.float64(u[2]))


@wp.kernel
def _load_kinematic_particles(
    q: wp.array[wp.vec3],
    fixed: wp.array[wp.int32],
    x: wp.array[wp.vec3d],
):
    i = wp.tid()
    if fixed[i] != 0:
        p = q[i]
        x[i] = wp.vec3d(wp.float64(p[0]), wp.float64(p[1]), wp.float64(p[2]))


@wp.kernel
def _store_particles(
    x: wp.array[wp.vec3d],
    v: wp.array[wp.vec3d],
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
):
    i = wp.tid()
    p = x[i]
    u = v[i]
    q[i] = wp.vec3(float(p[0]), float(p[1]), float(p[2]))
    qd[i] = wp.vec3(float(u[0]), float(u[1]), float(u[2]))


def _quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` by the unit quaternion ``q = (x, y, z, w)``."""
    u = q[:3]
    w = q[3]
    return v + 2.0 * w * np.cross(u, v) + 2.0 * np.cross(u, np.cross(u, v))


class SolverIPC(SolverBase):
    """Incremental Potential Contact solver for tetrahedral soft bodies.

    Wraps the `libuipc-warp <https://github.com/OrangeEarth15/libuipc-warp>`_ engine,
    a Warp implementation of libuipc. Each :meth:`step` is one implicit-Euler
    (BDF1) time step: a Newton iteration with a backtracking line search, a
    block-Jacobi PCG linear solve, barrier contact and continuous collision
    detection. The Newton loop, the PCG loop and the line search run on the
    device under ``wp.capture_while``, so a step never synchronizes with the
    host and can be captured in a CUDA graph.

    Supported model content:

    - particles and tetrahedra. Elasticity is the SNK1 stable Neo-Hookean
      model with Lamé parameters ``tet_materials[:, 0]`` (``k_mu``) and
      ``tet_materials[:, 1]`` (``k_lambda``). ``k_damp`` is ignored.
    - contact against static plane shapes (``shape_body == -1`` with
      :attr:`~newton.ShapeFlags.COLLIDE_PARTICLES`), each treated as an
      infinite half space.
    - contact between soft bodies: point-triangle and edge-edge barriers on
      the boundary of the tetrahedra, with continuous collision detection.
      Each connected group of tetrahedra is one body; a body does not
      collide with itself.
    - kinematic particles (zero mass or not :attr:`~newton.ParticleFlags.ACTIVE`).
      They follow the positions written to ``state_in`` every step.
    - one gravity vector, taken from world 0.

    Cloth, rigid bodies, self contact and friction are not handled yet.

    The solver keeps positions and velocities in float64. The first step
    after construction or :meth:`reset` reads ``state_in``. Later steps
    continue from the internal state, apart from kinematic particles, and
    write float32 results to ``state_out``.

    This solver requires the optional ``libuipc-warp`` package (``import uipc_warp``).

    Args:
        model: The model to simulate.
        kappa: Barrier stiffness [Pa], libuipc's contact resistance.
        d_hat: Barrier activation distance [m].
        thickness: Contact thickness of every particle [m].
        newton_max_iter: Maximum Newton iterations per step.
        newton_velocity_tol: Newton convergence tolerance on ``max |dx| / dt`` [m/s].
        linear_tol_rate: Relative PCG tolerance on ``r^T M^-1 r``.
        line_search_max_iter: Maximum backtracking trials per Newton iteration.
        semi_implicit: Enable libuipc's semi-implicit early exit after
            ``semi_implicit_k_min`` iterations of full feasible steps.
        semi_implicit_k_min: Newton iterations before the early exit applies.
    """

    def __init__(
        self,
        model: Model,
        *,
        kappa: float = 1.0e9,
        d_hat: float = 0.01,
        thickness: float = 0.0,
        newton_max_iter: int = 1024,
        newton_velocity_tol: float = 0.05,
        linear_tol_rate: float = 1.0e-3,
        line_search_max_iter: int = 8,
        semi_implicit: bool = True,
        semi_implicit_k_min: int = 6,
    ):
        super().__init__(model)
        try:
            import uipc_warp  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "SolverIPC requires the libuipc-warp package: pip install git+https://github.com/OrangeEarth15/libuipc-warp"
            ) from e
        self._uipc_warp = uipc_warp
        self._config = uipc_warp.SimConfig(
            kappa=kappa,
            d_hat=d_hat,
            newton_max_iter=newton_max_iter,
            newton_velocity_tol=newton_velocity_tol,
            linear_tol_rate=linear_tol_rate,
            line_search_max_iter=line_search_max_iter,
            semi_implicit=semi_implicit,
            semi_implicit_k_min=semi_implicit_k_min,
        )
        self._thickness = float(thickness)
        self._engine = None
        self._needs_load = True
        self._build_engine()

    # ------------------------------------------------------------------ setup
    def _build_engine(self) -> None:
        model = self.model
        if model.tet_count == 0 and model.tri_count > 0:
            raise NotImplementedError("SolverIPC does not support cloth (triangle-only models) yet.")
        n = model.particle_count
        if n == 0:
            self._engine = None
            return

        x = model.particle_q.numpy().astype(np.float64)
        v = model.particle_qd.numpy().astype(np.float64)
        mass = model.particle_mass.numpy().astype(np.float64)
        flags = model.particle_flags.numpy()
        fixed = ((mass <= 0.0) | ((flags & int(ParticleFlags.ACTIVE)) == 0)).astype(np.int32)

        m = model.tet_count
        if m > 0:
            tets = model.tet_indices.numpy().reshape(-1, 4)
            Dm_inv = model.tet_poses.numpy().astype(np.float64).reshape(-1, 3, 3)
            volume = 1.0 / (6.0 * np.abs(np.linalg.det(Dm_inv)))
            materials = model.tet_materials.numpy().astype(np.float64).reshape(-1, 3)
            mu, lam = materials[:, 0], materials[:, 1]
        else:
            tets = np.zeros((0, 4), dtype=np.int32)
            Dm_inv = np.zeros((0, 3, 3))
            volume = mu = lam = np.zeros(0)

        gravity = model.gravity.numpy().reshape(-1, 3).astype(np.float64)
        if len(gravity) > 1 and not np.allclose(gravity, gravity[0]):
            raise NotImplementedError("SolverIPC supports a single gravity vector for all worlds.")

        plane_P, plane_N = self._collect_planes()
        self._config.gravity = tuple(gravity[0])
        self._engine = self._uipc_warp.SimEngine(
            x=x,
            v=v,
            mass=mass,
            fixed=fixed,
            thickness=np.full(n, self._thickness),
            tets=tets,
            Dm_inv=Dm_inv,
            volume=volume,
            mu=mu,
            lam=lam,
            plane_P=plane_P,
            plane_N=plane_N,
            config=self._config,
            device=model.device,
        )

    def _collect_planes(self) -> tuple[np.ndarray, np.ndarray]:
        model = self.model
        if model.shape_count == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))
        shape_type = model.shape_type.numpy()
        shape_body = model.shape_body.numpy()
        shape_flags = model.shape_flags.numpy()
        shape_xform = model.shape_transform.numpy()
        shape_scale = model.shape_scale.numpy()
        P, N = [], []
        for s in range(model.shape_count):
            if shape_type[s] != int(GeoType.PLANE) or shape_body[s] != -1:
                continue
            if not shape_flags[s] & int(ShapeFlags.COLLIDE_PARTICLES):
                continue
            if shape_scale[s][0] > 0.0 or shape_scale[s][1] > 0.0:
                warnings.warn(
                    f"SolverIPC treats the finite plane shape {s} as an infinite half space.",
                    stacklevel=3,
                )
            xf = shape_xform[s].astype(np.float64)
            P.append(xf[:3])
            N.append(_quat_rotate_np(xf[3:7], np.array([0.0, 0.0, 1.0])))
        return np.asarray(P).reshape(-1, 3), np.asarray(N).reshape(-1, 3)

    # ------------------------------------------------------------------ SolverBase
    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
        *,
        observables: SolverBase.Observables | None = None,
    ) -> None:
        eng = self._engine
        if eng is None:
            return
        device = self.model.device
        n = self.model.particle_count
        vm = eng.vm
        if self._needs_load:
            wp.launch(
                _load_particles,
                n,
                inputs=[state_in.particle_q, state_in.particle_qd],
                outputs=[vm.x, vm.v],
                device=device,
            )
            self._needs_load = False
        else:
            wp.launch(
                _load_kinematic_particles,
                n,
                inputs=[state_in.particle_q, vm.fixed],
                outputs=[vm.x],
                device=device,
            )
        eng.advance(dt)
        wp.launch(
            _store_particles,
            n,
            inputs=[vm.x, vm.v],
            outputs=[state_out.particle_q, state_out.particle_qd],
            device=device,
        )

    @override
    def reset(self, state: State, world_mask=None, flags=None) -> None:
        """Read positions and velocities from the next ``state_in`` again."""
        self._needs_load = True

    @override
    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        if flags & (ModelFlags.SHAPE_PROPERTIES | ModelFlags.MODEL_PROPERTIES):
            self._build_engine()
            self._needs_load = True

    def frame_stats(self) -> dict:
        """Solver counters of the last step: Newton/PCG iterations, line-search trials, convergence.

        Synchronizes with the device.
        """
        return {} if self._engine is None else self._engine.frame_stats()
