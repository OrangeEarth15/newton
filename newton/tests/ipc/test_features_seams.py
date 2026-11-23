# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The points at which SolverIPC meets the features libuipc has no term for: the ``contact_damping`` option."""

import unittest

import newton
from newton.tests.ipc._fixtures import _box_body
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, requires


def _falling_box(device):
    """A box 0.4 m above the ground."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    _box_body(builder, (0.0, 0.0, 0.5), (0.1, 0.1, 0.1))
    return builder.finalize(device=device)


def test_contact_damping_takes_none_only(test, device):
    """contact_damping takes None, its default, which adds no damping: the solver is built and steps; any other value,
    0 among them, raises NotImplementedError naming it, since the solver's contact has no damping term."""
    model = _falling_box(device)
    solver = newton.solvers.SolverIPC(model, contact_damping=None)
    s0, s1 = model.state(), model.state()
    solver.step(s0, s1, None, None, 0.01)
    test.assertLess(float(s1.body_q.numpy()[0, 2]), 0.5)
    for value in (1.0, 0.0, "material"):
        with test.subTest(value=value):
            with test.assertRaises(NotImplementedError) as caught:
                newton.solvers.SolverIPC(model, contact_damping=value)
            test.assertEqual(
                str(caught.exception),
                f"SolverIPC: contact_damping={value!r} is not offered yet: the solver's contact has no damping term. "
                "Leave contact_damping at None, its default.",
            )


devices = get_cuda_test_devices()


@requires("uipc_warp")
class TestSolverIPCFeatureSeams(unittest.TestCase):
    """The points at which SolverIPC meets the features libuipc has no term for."""


for name, func in (("test_contact_damping_takes_none_only", test_contact_damping_takes_none_only),):
    add_function_test(TestSolverIPCFeatureSeams, name, func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
