# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC's elastic energies against Newton's: the cloth membrane's stretch and shear moduli and the hinges'
bending stiffness, measured by the oscillations of one free vertex."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.ipc._fixtures import _implicit_euler_oscillator
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires

_TIGHT = {"newton_velocity_tol": 1.0e-6, "linear_tol_rate": 1.0e-12, "semi_implicit": False}


def test_membrane_and_hinge_follow_newtons_moduli(test, device):
    """Newton's membrane, linear elasticity mu eps:eps + lambda / 2 (tr eps)^2 per rest area with mu = tri_ke and
    lambda = tri_ka, and its hinge energy 1/2 k_e L0 (theta - theta_bar)^2, as libuipc's Baraff-Witkin membrane and
    discrete-shell hinge with K_a = K_b = mu + lambda / 2, K_g = mu / 2 and kappa = k_e h_bar / 2.

    A right triangle of legs L = 0.1 m with two vertices held (mass 0) and the third free (8 g), released 1 mm off its
    rest place, oscillates as implicit Euler's oscillator of omega^2 = k / m: k = mu + lambda / 2 when the free
    vertex moves along its edge from the held corner (a stretch of a material axis, the other held fixed; the first
    axis runs along the first edge, the second at right angles to it) and k = mu / 2 when it moves across it (simple
    shear), for either free vertex. A hinge of two triangles whose free vertex, H = 5 cm from the hinge, is released
    1 mm out of the plane oscillates with k = k_e L0 / H^2. With mu = 10 N/m and lambda = 20 N/m, a mapping of K_a to mu
    alone or K_g to mu would change omega by 29% or 41%. cloth_strain_rate=0 keeps the membrane quadratic."""
    mu, lam, ke, L, H, dt, steps = 10.0, 20.0, 0.1, 0.1, 0.05, 0.01, 20
    m, delta = 0.008, 1.0e-3
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cases = []
    # free vertex 1 (end of the first edge) or 2 (end of the second), displaced along x or y
    for k, (free, axis, stiffness) in enumerate(
        ((1, 0, mu + 0.5 * lam), (1, 1, 0.5 * mu), (2, 1, mu + 0.5 * lam), (2, 0, 0.5 * mu))
    ):
        corners = ((float(k), 0.0, 1.0), (float(k) + L, 0.0, 1.0), (float(k), L, 1.0))
        ids = [builder.add_particle(wp.vec3(*c), wp.vec3(0.0), m if i == free else 0.0) for i, c in enumerate(corners)]
        builder.add_triangle(*ids, tri_ke=mu, tri_ka=lam, tri_kd=0.0)
        cases.append((ids[free], axis, stiffness))
    # a hinge: the held triangle (o0, v1, v2) and the free vertex o1 of (o1, v2, v1), the edge v1 v2 along x
    v1 = builder.add_particle(wp.vec3(5.0, 0.0, 1.0), wp.vec3(0.0), 0.0)
    v2 = builder.add_particle(wp.vec3(5.0 + L, 0.0, 1.0), wp.vec3(0.0), 0.0)
    o0 = builder.add_particle(wp.vec3(5.0 + 0.5 * L, -H, 1.0), wp.vec3(0.0), 0.0)
    o1 = builder.add_particle(wp.vec3(5.0 + 0.5 * L, H, 1.0), wp.vec3(0.0), m)
    builder.add_triangle(o0, v1, v2, tri_ke=1.0, tri_ka=1.0, tri_kd=0.0)
    builder.add_triangle(o1, v2, v1, tri_ke=1.0, tri_ka=1.0, tri_kd=0.0)
    builder.add_edge(o0, o1, v1, v2, edge_ke=ke, edge_kd=0.0)
    cases.append((o1, 2, ke * L / H**2))
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, cloth_strain_rate=0.0, **_TIGHT)
    s0, s1 = model.state(), model.state()
    q = s0.particle_q.numpy()
    for p, axis, _ in cases:
        q[p, axis] += delta
    s0.particle_q.assign(q)
    rest = model.particle_q.numpy().astype(np.float64)
    path = []
    for _ in range(steps):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        path.append(solver._engine.positions()[: model.particle_count] - rest)
    path = np.array(path)
    names = ("stretch of axis a", "shear from vertex 1", "stretch of axis b", "shear from vertex 2", "hinge")
    for name, (p, axis, stiffness) in zip(names, cases, strict=True):
        with test.subTest(name):
            expected = _implicit_euler_oscillator(delta, np.sqrt(stiffness / m), dt, steps)
            np.testing.assert_allclose(path[:, p, axis], expected, rtol=0.0, atol=2.0e-3 * delta)
            others = [a for a in range(3) if a != axis]
            test.assertLess(np.abs(path[:, p, others]).max(), 2.0e-2 * delta)  # it moves along its own axis


devices = get_cuda_test_devices()


@requires("uipc_warp")
class TestSolverIPCElasticity(unittest.TestCase):
    pass


for name, func in (("test_membrane_and_hinge_follow_newtons_moduli", test_membrane_and_hinge_follow_newtons_moduli),):
    add_function_test(TestSolverIPCElasticity, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
