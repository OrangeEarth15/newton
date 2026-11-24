Add the experimental `newton.solvers.SolverIPC`, an Incremental Potential Contact solver that runs on the optional `libuipc-warp` package (0.5.0 or newer) and never lets a surface pass through another: each step is one implicit time step in which a barrier and continuous collision detection keep surfaces apart. The step runs on the device and can be captured in a CUDA graph. It simulates:

  - Particles, tetrahedral soft bodies (stable Neo-Hookean), cloth (Baraff-Witkin membrane with discrete-shell bending and self-contact), springs and particle-body attachments, with Newton's damping coefficients as `SolverVBD` defines them and the drag and lift of triangles as `SolverSemiImplicit` computes them.
  - Rigid bodies as near-rigid affine bodies with box, sphere, capsule, cylinder, cone, ellipsoid and mesh shapes, whose rotation is integrated as a rigid body's by default (`body_rotation`); static shapes, planes and height fields.
  - Contact with lagged friction between the pairs that Newton's collision groups, worlds, collide flags and filter pairs let collide, with the mean friction of the two materials, shape margins and particle radii as contact thickness, a default `d_hat` of every shape and particle, as libuipc's per-geometry `d_hat`, below the small features it can touch, optional contact damping (`contact_damping`) and the surface velocity of collision meshes.
  - Revolute, prismatic, fixed, ball, D6, universal and distance joints as penalty energies, with drives, springs, damping, limits, armature, effort limits, joint friction and `Control.joint_f`; mimic joints and MuJoCo's fixed tendons; `joint_q` and `joint_qd` written after every step.
  - Kinematic particles and bodies, which move toward the poses of `state_in` inside the step or jump to them after a swept check (`kinematic_motion`).
  - Several worlds, each solved as it would be alone, or as one solve with `coupled_worlds=True`.
  - libuipc's BDF2 (`integrator="bdf2"`), augmented-Lagrangian contact (`contact_constitution="al-ipc"`) and MAS preconditioner (`fem_preconditioner="mas"`) as options.
  - The observables `STATUS`, `SOLVER_RESIDUAL`, `CONTACT_F`, `BODY_PARENT_F` and `BODY_QDD`, with `check_status()`, `grow()` and `graph_outdated()` for captured steps.
  - An entry of Newton's coupled solvers, proxies included.

  It refuses at construction what it does not simulate, naming the item and what to change, and runs on CUDA devices only. The `cloth_ipc` example shows it, and eleven other examples run with it through `--solver ipc`.
