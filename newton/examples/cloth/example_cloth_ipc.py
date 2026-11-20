# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example Cloth IPC

A sheet of cloth falls onto a pile of rigid boxes and drapes over them,
simulated with the Incremental Potential Contact solver
(:class:`newton.solvers.SolverIPC`). The cloth is libuipc's Baraff-Witkin
membrane with discrete-shell bending, the boxes are affine bodies, and every
contact (cloth on boxes, cloth on itself, boxes on boxes and on the ground)
is a barrier with continuous collision detection, so nothing interpenetrates.

The example captures its substeps in a CUDA graph. A captured step cannot
raise, so after every replay the example reads each substep's status with
:meth:`~newton.solvers.SolverIPC.check_status`: a failed step raises there,
and a contact-list overflow grows the lists and repeats the frame in a graph
captured anew. It also captures again when
:meth:`~newton.solvers.SolverIPC.graph_outdated` says that the substeps' PCG
loop would run faster with the other length.

Requires the optional ``libuipc-warp`` package.

Success criteria:
- The cloth falls, drapes over the boxes and folds where it meets the ground.
- The cloth never passes through a box, the ground or itself.
- The boxes stay stacked and upright when the cloth lands on them.

Runs:
- `python -m newton.examples cloth_ipc`
- `python -m newton.examples cloth_ipc --cloth-res 32`
"""

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        """Build boxes on the ground and a cloth sheet above them."""
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer

        builder = newton.ModelBuilder()
        # friction 0.4 on every contact: the solver takes the mean of the two materials of a pair
        builder.default_shape_cfg.mu = 0.4
        builder.add_ground_plane()

        # rigid boxes (density 1000 kg/m^3), each a few millimetres above what it rests on
        gap = 0.004
        boxes = [
            ((0.0, 0.0, 0.15 + gap), (0.25, 0.25, 0.15), 0.0),
            ((0.05, 0.02, 0.4 + 2 * gap), (0.1, 0.1, 0.1), 0.5),
            ((0.55, 0.1, 0.08 + gap), (0.12, 0.3, 0.08), 0.25),
            ((-0.5, -0.35, 0.2 + gap), (0.1, 0.1, 0.2), -0.3),
        ]
        for center, half, yaw in boxes:
            body = builder.add_body(
                xform=wp.transform(wp.vec3(*center), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw))
            )
            builder.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2])

        # a 1.44 m square sheet, about 0.3 kg/m^2
        res = args.cloth_res
        cell = 1.44 / res
        builder.add_cloth_grid(
            pos=wp.vec3(-0.72, -0.72, 0.9),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=res,
            dim_y=res,
            cell_x=cell,
            cell_y=cell,
            mass=0.3 * cell * cell,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=0.0,
            edge_ke=5.0e-3,
            edge_kd=0.0,
        )

        self.model = builder.finalize()
        self.model.soft_contact_mu = 0.4  # the cloth's side of a contact with a shape
        self.model.particle_mu = 0.4  # the cloth on itself
        # resting contacts settle just inside d_hat: 5 mm keeps the visible gaps small
        self.solver = newton.solvers.SolverIPC(self.model, d_hat=0.005)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        # the state a frame starts from, to repeat the frame after an overflow
        self.frame_start = self.model.state()
        # how each substep ended: a captured step reports a failure only here, one container per substep
        kinds = {
            newton.solvers.SolverIPC.ObservableKind.STATUS,
            newton.solvers.SolverIPC.ObservableKind.SOLVER_RESIDUAL,
        }
        self.observables = [self.solver.observables(kinds) for _ in range(self.sim_substeps)]

        self.viewer.set_model(self.model)
        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for observables in self.observables:
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt, observables=observables)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def check_substeps(self) -> bool:
        """Raise if a substep of the last replay failed; return True when a contact list overflowed and the lists
        were grown, so that the frame must be repeated."""
        grown = False
        for observables in self.observables:
            grown = self.solver.check_status(observables, grow=True) or grown
        return grown

    def step(self):
        if self.graph:
            self.frame_start.assign(self.state_0)
            wp.capture_launch(self.graph)
            # after an overflow, repeat the frame from its start in a new graph (the old one holds the old lists)
            while self.check_substeps():
                self.state_0.assign(self.frame_start)
                self.capture()
                wp.capture_launch(self.graph)
            if self.solver.graph_outdated():  # the next frames run a PCG loop of the other length
                self.capture()
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        newton.examples.test_particle_state(
            self.state_0,
            "particles are above the ground",
            lambda q, qd: q[2] > 0.0,
        )
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, wp.vec3(-1.5, -1.5, 0.0), wp.vec3(1.5, 1.5, 1.0)),
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "boxes stay near where they were placed",
            lambda q, qd: newton.math.vec_inside_limits(
                wp.transform_get_translation(q), wp.vec3(-0.8, -0.8, 0.05), wp.vec3(0.8, 0.8, 0.55)
            ),
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "boxes stay upright",
            lambda q, qd: wp.abs(wp.quat_rotate(wp.transform_get_rotation(q), wp.vec3(0.0, 0.0, 1.0))[2]) > 0.95,
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--cloth-res", type=int, default=48, help="Cloth cells along each side.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
