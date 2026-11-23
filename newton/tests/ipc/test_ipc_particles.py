# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SolverIPC's particles: friction between two particles is ``Model.particle_mu``, and a particle that is not
ACTIVE stays where ``state_in`` puts it whatever its mass."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires

_G = 9.81


def test_particle_friction_is_particle_mu(test, device):
    """A particle of radius 1 cm on top of a kinematic particle of radius 2 m slides under gravity tilted by
    atan(0.8) from the vertical, past the friction angle of particle_mu = 0.5 (Model.soft_contact_mu, 0.1, and the mean
    of the two are not taken). At the angle phi from the top of the big particle and the distance rho from its centre,
    its motion along the surface is that of a particle sliding on a sphere: rho phi'' = g sin(alpha + phi) - mu (g
    cos(alpha + phi) - rho phi'^2), from the second differences of 0.4 s of steps of 0.01 s after the first 0.1 s, to 2%
    of g sin(alpha) in every step and 1% on average (the friction lags by a step); with particle_mu = 0.1 it slides
    faster, as that law has it too. The mean of the coefficients, 0.3, would be 1.5 m/s^2 off."""
    R, r, dt, steps = 2.0, 0.01, 0.01, 50
    alpha = np.arctan(0.8)

    def slide(particle_mu):
        builder = newton.ModelBuilder()
        builder.add_particle(wp.vec3(0.0, 0.0, -R), wp.vec3(0.0), 0.0, radius=R)
        builder.add_particle(wp.vec3(0.0, 0.0, r + 5.0e-4), wp.vec3(0.0), 0.01, radius=r)
        model = builder.finalize(device=device)
        model.set_gravity((_G * np.sin(alpha), 0.0, -_G * np.cos(alpha)))
        model.particle_mu = particle_mu
        model.soft_contact_mu = 0.1
        solver = newton.solvers.SolverIPC(
            model, d_hat=1.0e-3, newton_velocity_tol=1.0e-5, linear_tol_rate=1.0e-12, semi_implicit=False
        )
        s0, s1 = model.state(), model.state()
        x = []
        for _ in range(steps):
            solver.step(s0, s1, None, None, dt)
            s0, s1 = s1, s0
            x.append(solver._engine.positions()[1] - [0.0, 0.0, -R])
        x = np.array(x)
        return np.arctan2(x[:, 0], x[:, 2]), np.linalg.norm(x, axis=1)

    for mu in (0.5, 0.1):
        with test.subTest(particle_mu=mu):
            phi, rho = slide(mu)
            k = np.arange(10, steps - 1)  # central differences at step k
            accel = rho[k] * (phi[k + 1] - 2.0 * phi[k] + phi[k - 1]) / dt**2
            rate = (phi[k + 1] - phi[k - 1]) / (2.0 * dt)
            law = _G * np.sin(alpha + phi[k]) - mu * (_G * np.cos(alpha + phi[k]) - rho[k] * rate**2)
            test.assertTrue(np.all(rate > 0.0))  # sliding from the start
            np.testing.assert_allclose(accel, law, rtol=0.0, atol=2.0e-2 * _G * np.sin(alpha))
            test.assertLess(abs(accel.mean() / law.mean() - 1.0), 1.0e-2)
            test.assertLess(rho.max() - rho.min(), 1.0e-3)  # on the big particle all along


def test_inactive_particle_with_mass_stays_put(test, device):
    """A particle of 1 kg that is not ACTIVE is kinematic, as one of mass 0: under gravity and a particle_f of 50 N it
    stays where state_in puts it, step after step, also when state_in moves it, while the same particle ACTIVE beside
    it falls as implicit Euler has it under gravity and that force."""
    dt = 0.01
    builder = newton.ModelBuilder()
    inactive = builder.add_particle(wp.vec3(0.0, 0.0, 1.0), wp.vec3(0.0), 1.0, radius=0.01, flags=0)
    active = builder.add_particle(wp.vec3(1.0, 0.0, 1.0), wp.vec3(0.0), 1.0, radius=0.01)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverIPC(model, newton_velocity_tol=1.0e-5, linear_tol_rate=1.0e-12)
    s0, s1 = model.state(), model.state()
    for s in (s0, s1):
        s.particle_f.assign(np.array([[0.0, 50.0, 0.0], [0.0, 50.0, 0.0]], np.float32))
    k = np.arange(1, 11)
    path = []
    for _ in range(10):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        path.append(s0.particle_q.numpy().astype(np.float64))
    path = np.array(path)
    np.testing.assert_array_equal(path[:, inactive], np.broadcast_to([0.0, 0.0, 1.0], (10, 3)))
    falls = 1.0 - _G * dt * dt * k * (k + 1) / 2.0
    pushed = 50.0 * dt * dt * k * (k + 1) / 2.0
    np.testing.assert_allclose(path[:, active], np.stack([np.ones(10), pushed, falls], axis=1), rtol=0.0, atol=1e-5)
    q = s0.particle_q.numpy()
    q[inactive] = (0.2, 0.3, 0.4)  # state_in moves it
    s0.particle_q.assign(q)
    for _ in range(3):
        solver.step(s0, s1, None, None, dt)
        s0, s1 = s1, s0
        np.testing.assert_array_equal(s0.particle_q.numpy()[inactive], np.array([0.2, 0.3, 0.4], np.float32))


devices = get_cuda_test_devices()


@requires("uipc_warp")
class TestSolverIPCParticles(unittest.TestCase):
    pass


for name, func in (
    ("test_particle_friction_is_particle_mu", test_particle_friction_is_particle_mu),
    ("test_inactive_particle_with_mass_stays_put", test_inactive_particle_with_mass_stays_put),
):
    add_function_test(TestSolverIPCParticles, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
