# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rotations by Newton's quaternions ``(x, y, z, w)`` on the host, in float64, for laying out the scene and the joints'
frames."""

from __future__ import annotations

import numpy as np


def _quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate the rows of ``v`` by the unit quaternion ``q = (x, y, z, w)``."""
    u = np.asarray(q[:3], np.float64)
    w = float(q[3])
    return v + 2.0 * w * np.cross(u, v) + 2.0 * np.cross(u, np.cross(u, v))


def _quat_to_matrix_np(q: np.ndarray) -> np.ndarray:
    """The rotation matrix of the quaternion ``q = (x, y, z, w)``, normalized first."""
    q = np.asarray(q, np.float64) / np.linalg.norm(q)
    return _quat_rotate_np(q, np.eye(3)).T
