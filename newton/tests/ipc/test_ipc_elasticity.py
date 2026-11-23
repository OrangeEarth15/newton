# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC's elastic energies against Newton's: the cloth membrane's stretch and shear moduli and the hinges'
bending stiffness, measured by the oscillations of one free vertex, and the Lame parameters of the tetrahedra,
measured by the lateral contraction of a block under uniaxial stress."""

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


def _boundary_faces(tets):
    """The faces of a tetrahedral mesh that only one tetrahedron has."""
    faces = np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]])
    keys, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    return keys[counts == 1]


def test_tetrahedra_contract_by_poissons_ratio(test, device):
    """k_mu and k_lambda are Lame parameters: a block of 4 x 2 x 2 cells stretched along x by a uniform traction on
    its end faces (the consistent loads of linear tetrahedra, a third of each boundary triangle's share on each of its
    corners) settles into homogeneous uniaxial stress, its axial strain sigma / E with E = mu (3 lambda + 2 mu) /
    (lambda + mu) and its lateral strains -nu times that with nu = lambda / (2 (lambda + mu)), each within 1%: with
    k_mu = k_lambda Poisson's ratio is 0.25, not 0 (libuipc's StableNeoHookean stores mu and lambda + mu, which
    linearize to Hooke's law with these Lame parameters). Gravity off, the block free, its strain 1e-3."""
    mu = lam = 1.0e4
    E, nu = mu * (3.0 * lam + 2.0 * mu) / (lam + mu), lam / (2.0 * (lam + mu))
    sigma = 1.0e-3 * E
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, 1.0), rot=wp.quat_identity(), vel=wp.vec3(0.0), dim_x=4, dim_y=2, dim_z=2, cell_x=0.05,
        cell_y=0.05, cell_z=0.05, density=10.0, k_mu=mu, k_lambda=lam, k_damp=0.0,
    )  # fmt: skip
    model = builder.finalize(device=device)
    rest = model.particle_q.numpy().astype(np.float64)
    force = np.zeros_like(rest)
    for face in _boundary_faces(model.tet_indices.numpy().reshape(-1, 4)):
        x = rest[face, 0]
        for end, sign in ((rest[:, 0].min(), -1.0), (rest[:, 0].max(), 1.0)):
            if np.allclose(x, end):
                a, b, c = rest[face]
                area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
                force[face, 0] += sign * sigma * area / 3.0
    np.testing.assert_allclose(force[force[:, 0] > 0.0, 0].sum(), sigma * 0.1 * 0.1, rtol=1.0e-6)  # float32 rest
    solver = newton.solvers.SolverIPC(model, **_TIGHT)
    s0, s1 = model.state(), model.state()
    for s in (s0, s1):
        s.particle_f.assign(force.astype(np.float32))
    for _ in range(30):
        solver.step(s0, s1, None, None, 0.01)
        s0, s1 = s1, s0
    x = solver._engine.positions()[: model.particle_count]
    # the affine map that takes the rest positions to the settled ones, and how far the block is from it
    design = np.hstack([rest, np.ones((len(rest), 1))])
    fit, *_ = np.linalg.lstsq(design, x, rcond=None)
    F = fit[:3].T
    test.assertLess(np.abs(design @ fit - x).max(), 1.0e-3 * 1.0e-3 * 0.2)  # homogeneous within 0.1% of the stretch
    strain = 0.5 * (F + F.T) - np.eye(3)
    np.testing.assert_allclose(strain[0, 0], sigma / E, rtol=1.0e-2)
    np.testing.assert_allclose([-strain[1, 1] / strain[0, 0], -strain[2, 2] / strain[0, 0]], [nu, nu], rtol=1.0e-2)
    test.assertLess(np.abs(strain - np.diag(np.diag(strain))).max(), 1.0e-3 * sigma / E)


devices = get_cuda_test_devices()


@requires("uipc_warp")
class TestSolverIPCElasticity(unittest.TestCase):
    pass


for name, func in (
    ("test_membrane_and_hinge_follow_newtons_moduli", test_membrane_and_hinge_follow_newtons_moduli),
    ("test_tetrahedra_contract_by_poissons_ratio", test_tetrahedra_contract_by_poissons_ratio),
):
    add_function_test(TestSolverIPCElasticity, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
