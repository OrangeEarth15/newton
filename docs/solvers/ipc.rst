.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

IPC
===

:class:`~newton.solvers.SolverIPC` simulates soft bodies, cloth, particles and
rigid bodies with Incremental Potential Contact (IPC): each step is an implicit
time step in which a barrier energy keeps surfaces apart, so that no surface
passes through another. The solver runs on
`libuipc-warp <https://github.com/OrangeEarth15/libuipc-warp>`_, a Warp
implementation of the libuipc simulator, through its array interface; no libuipc
build is needed.

.. experimental::

   :class:`~newton.solvers.SolverIPC` is experimental. Its options and its
   behavior may change without the normal deprecation period.

This page is the IPC backend guide: when to choose the solver, how to build and
step it, how it takes each part of a Newton model, and what it refuses. The API
reference of :class:`~newton.solvers.SolverIPC` documents every method and
argument.

When to choose SolverIPC
------------------------

Choose SolverIPC when contact must never let surfaces cross: cloth that folds
onto itself, stacks and piles, soft bodies pressed together, thin parts in tight
spaces. IPC guarantees this by its construction. A barrier energy grows without
bound as the distance between two surfaces falls toward zero, and continuous
collision detection (CCD) stops every update of the positions before the first
time of impact, so no update can carry a surface through another. The
method is that of Li et al., "Incremental Potential Contact: Intersection- and
Inversion-free, Large-Deformation Dynamics", ACM Transactions on Graphics 39(4),
2020, as libuipc implements it.

What it costs:

- A step is a full implicit solve: a Newton loop with a line search, and a
  preconditioned conjugate gradient (PCG) solve in every Newton iteration. It
  costs more than an explicit substep, and in exchange takes larger steps:
  Newton's ``cloth_ipc`` example takes two steps per 60 Hz frame.
- Surfaces in contact rest a small gap apart, up to the barrier distance
  ``d_hat``: 1 cm by default, less near small parts (see
  :ref:`ipc-d-hat`).
- Every pair of surfaces must start apart (see :ref:`ipc-start`).
- Rigid bodies are libuipc's affine bodies, held close to rigid by a stiff
  energy, and joints are penalty energies, not hard constraints (see
  :ref:`ipc-rigid-bodies` and :ref:`ipc-joints`).
- It runs on CUDA devices only, and computes no gradients.

The :ref:`Supported Features` matrix and the :ref:`Joint feature support` tables
compare it with Newton's other solvers.

.. _ipc-stepping:

Construction and stepping
-------------------------

SolverIPC needs the optional ``libuipc-warp`` package, 0.4.4 or newer; a
development build such as 0.4.4.dev0 counts as its release. Building the first
solver imports it, and raises :class:`ImportError` when it is missing, older,
or lacks one of the calls the solver makes; the message names what is missing
and where to install it from. Build the model on a CUDA device and step it as
any other solver:

.. code-block:: python

   import warp as wp

   import newton
   from newton.solvers import SolverIPC

   wp.config.log_level = wp.LOG_WARNING  # no Warp banner, no line per loaded kernel module

   builder = newton.ModelBuilder()  # z up, gravity 9.81 m/s^2 along -z
   builder.add_ground_plane()
   box = builder.add_body(xform=wp.transform((0.0, 0.0, 0.3), wp.quat_identity()))
   builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)  # a 20 cm cube, density 1000 kg/m^3
   model = builder.finalize(device="cuda:0")

   solver = SolverIPC(model)
   state_0, state_1 = model.state(), model.state()
   control = model.control()
   dt = 1.0 / 60.0  # one step of 1/60 s is one implicit solve
   for _ in range(60):  # one second
       solver.step(state_0, state_1, control, None, dt)
       state_0, state_1 = state_1, state_0
   z = float(state_0.body_q.numpy()[box, 2])  # the height of the box's centre
   print(f"gap under the box: {1e3 * (z - 0.1):.2f} mm")
   print("Newton iterations of the last step:", solver.frame_stats()["newton_iterations"])

It prints:

.. code-block:: text

   gap under the box: 9.31 mm
   Newton iterations of the last step: 1

The box rests just inside ``d_hat``, 1 cm for this model, where the barrier
holds its weight. The last digits can differ from run to run, because the solver
sums with floating-point atomics.

Each :meth:`~newton.solvers.SolverIPC.step` is one implicit time step, by
default of implicit Euler: a Newton loop with a backtracking line search, a
block-diagonal (block-Jacobi) preconditioned conjugate gradient solve in every
Newton iteration, barrier contact with lagged friction, and CCD. The loops run
on the device under ``wp.capture_while``, so a step never synchronizes with the
host when it is captured in a CUDA graph. A step that is not captured runs its
solve as a CUDA graph of its own, then reads how it ended (one
synchronization): when a contact list overflowed, it enlarges the lists and
solves the step again; when the step failed, it raises :class:`RuntimeError`.
A captured step can do neither, and reports through the ``STATUS`` observable
instead (see :ref:`ipc-status`).

The ``contacts`` argument of :meth:`~newton.solvers.SolverIPC.step` is not read:
the solver detects its own contacts. Pass ``None``, or with ``CONTACT_F``
requested, the buffer that receives the solver's contacts (see
:ref:`ipc-contact-output`).

.. _ipc-arguments:

Constructor arguments
~~~~~~~~~~~~~~~~~~~~~

The defaults are libuipc's where libuipc has one, and otherwise follow Newton's
other solvers.

.. list-table::
   :header-rows: 1
   :widths: 24 22 54

   * - argument
     - default
     - what it sets
   * - ``d_hat``
     - ``None``: 0.01 m, or less near small parts
     - The barrier distance [m] of every shape and particle: the barrier acts
       on pairs closer than this, and surfaces at rest stay up to this far
       apart (:ref:`ipc-d-hat`).
   * - ``kappa``
     - 1e9
     - The barrier stiffness [Pa] of every contact pair, taken as given
       (:ref:`ipc-contact`).
   * - ``adaptive_kappa``
     - ``False``
     - Keep the stiffness in libuipc's range for the model and the step size,
       as libuipc does (:ref:`ipc-contact`).
   * - ``thickness``
     - 0
     - The contact thickness [m] of the vertices of cloth and tetrahedra
       (:ref:`ipc-contact`).
   * - ``contact_damping``
     - ``None``
     - A viscous damping of the normal motion of surfaces in contact
       (:ref:`ipc-contact-damping`).
   * - ``eps_velocity``
     - 0.01
     - The friction's smoothing speed [m/s]: below it, static friction acts
       as a stiff spring (:ref:`ipc-contact`).
   * - ``newton_velocity_tol``
     - ``None``: 0.05
     - The Newton loop ends when an update moves no vertex faster than this
       [m/s] (:ref:`ipc-tolerances`).
   * - ``newton_transrate_tol``
     - ``None``: 0.1
     - The same test for the rotation part of a rigid body [1/s].
   * - ``newton_max_iter``
     - 1024
     - The most Newton iterations of a step.
   * - ``line_search_max_iter``
     - 8
     - The most backtracking trials of a line search.
   * - ``linear_tol_rate``
     - 1e-3
     - The PCG's tolerance, relative to its first residual.
   * - ``semi_implicit``, ``semi_implicit_k_min``
     - ``None`` (on), 6
     - libuipc's early exit: after ``semi_implicit_k_min`` Newton iterations
       the loop ends once CCD admits a full update (:ref:`ipc-tolerances`).
   * - ``integrator``
     - ``"bdf1"``
     - Implicit Euler, or ``"bdf2"`` (:ref:`ipc-bdf2`).
   * - ``contact_constitution``
     - ``"ipc"``
     - The barrier, or ``"al-ipc"``, libuipc's augmented-Lagrangian contact
       (:ref:`ipc-al-ipc`).
   * - ``fem_preconditioner``
     - ``"diag"``
     - The block-diagonal preconditioner, or ``"mas"`` for particles, cloth
       and tetrahedra (:ref:`ipc-mas`).
   * - ``body_rotation``
     - ``None``: ``"rigid"`` under ``"bdf1"``
     - How a rigid body's rotation is integrated (:ref:`ipc-rigid-bodies`).
   * - ``body_kappa``
     - ``None``: up to 1e8
     - The stiffness [Pa] that holds an affine body close to rigid
       (:ref:`ipc-rigid-bodies`).
   * - ``shape_segments``
     - 24
     - The segments around a sphere, capsule, cylinder, cone or ellipsoid
       when it is triangulated.
   * - ``cloth_strain_rate``
     - 100
     - The cloth membrane's strain limiting; 0 makes it quadratic
       (:ref:`ipc-cloth`).
   * - ``joint_strength_ratio``
     - 100
     - The stiffness of the joints' penalties, relative to the mass they join
       (:ref:`ipc-joints`).
   * - ``enable_joint_limit_damping``
     - ``True``
     - Damp a joint past its limit with ``joint_limit_kd`` (:ref:`ipc-joints`).
   * - ``enable_joint_effort_limit``
     - ``True``
     - Bound a drive's force by ``joint_effort_limit`` (:ref:`ipc-joints`).
   * - ``kinematic_motion``, ``kinematic_strength_ratio``
     - ``None``, 100
     - How a kinematic body follows ``state_in``, and the strength of its
       pull (:ref:`ipc-kinematic-bodies`).
   * - ``coupled_worlds``
     - ``False``
     - Solve several worlds as one, instead of each as it would be alone
       (:ref:`ipc-worlds`).

A value other than the default of an argument that the other options leave
unread raises :class:`ValueError`: ``kappa`` or ``adaptive_kappa`` with
``contact_constitution="al-ipc"``, which has no barrier, and
``kinematic_strength_ratio`` where the kinematic bodies jump. ``thickness``,
``cloth_strain_rate``, ``joint_strength_ratio`` and
``kinematic_strength_ratio`` must be finite and not negative, ``kappa`` and
``body_kappa`` finite and positive, and ``shape_segments`` a whole number of at
least 3.

.. _ipc-status:

Status of a step
~~~~~~~~~~~~~~~~

A step captured in a CUDA graph cannot raise and cannot enlarge the contact
lists: it writes how it ended into the observables passed to it. A caller that
captures :meth:`~newton.solvers.SolverIPC.step` therefore requests
``SolverIPC.ObservableKind.STATUS``, passes one container to each captured
step (a container holds the status of the last step that wrote it), and calls
:meth:`~newton.solvers.SolverIPC.check_status` with every container after each
replay:

.. code-block:: python

   import warp as wp

   import newton
   from newton.solvers import SolverIPC

   wp.config.log_level = wp.LOG_WARNING

   builder = newton.ModelBuilder()
   builder.add_ground_plane()
   box = builder.add_body(xform=wp.transform((0.0, 0.0, 0.3), wp.quat_identity()))
   builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
   model = builder.finalize(device="cuda:0")
   solver = SolverIPC(model)
   state_0, state_1 = model.state(), model.state()
   dt = 1.0 / 120.0

   # one container per captured step: a container holds the status of the last step that wrote it
   kinds = {SolverIPC.ObservableKind.STATUS}
   status = [solver.observables(kinds), solver.observables(kinds)]
   with wp.ScopedCapture() as capture:  # two steps, so that the states end where they started
       solver.step(state_0, state_1, None, None, dt, observables=status[0])
       solver.step(state_1, state_0, None, None, dt, observables=status[1])

   for _ in range(60):  # one second
       wp.capture_launch(capture.graph)
       for observables in status:
           solver.check_status(observables)  # raises RuntimeError if the step failed
   print(f"box centre after 1 s: {float(state_0.body_q.numpy()[box, 2]):.4f} m")

It prints ``box centre after 1 s: 0.1093 m``.

The status is a mask of :class:`~newton.solvers.SolverIPC.StatusBits`. A bit of
``StatusBits.FAILED`` means the step failed and its result must not be used:

- ``OVERFLOW``: a contact list was too short, and the step ran without the
  pairs that did not fit.
- ``NONFINITE_ENERGY``, ``NONFINITE_PCG_INIT``, ``NONFINITE_PCG_ITER``: the
  solve met a value that is not finite.
- ``THICKNESS_VIOLATED``: a contact pair started at or inside its contact
  thickness, or a state written since the last step put surfaces through each
  other or within their contact thickness (:ref:`ipc-start`).
- ``INVALID_CCD_STEP``: CCD admitted no update, or a move written since the
  last step met a surface (:ref:`ipc-kinematic-bodies`).
- ``TRAVERSAL_EXHAUSTED``: a search of the collision tree ran out of its node
  budget, so contact candidates are missing.

``NEWTON_LIMIT`` (the loop reached ``newton_max_iter``) and
``LINE_SEARCH_LIMIT`` (a line search ran out of trials) do not fail the step.
Under friction a line search can run out of trials in some runs and not in
others: lagged friction has no curvature beyond its smoothing band, so a Newton
update can slide a vertex past the turn of the friction energy, and the
backtracking cannot find a decrease. libuipc does the same on the same scenes;
the loop goes on from the last trial.
:meth:`~newton.solvers.SolverIPC.check_status` raises :class:`RuntimeError` on
a failed step and warns with :class:`~newton.NewtonWarning` on an iteration
limit. With ``grow=True`` it answers ``OVERFLOW`` instead: it enlarges the lists
(:meth:`~newton.solvers.SolverIPC.grow`) and returns ``True``. The caller then
restores the state the replayed steps started from, captures the graph again,
since the enlarged lists are new arrays, and repeats the steps. Newton's
``cloth_ipc`` example does all of this. A later libuipc-warp may set bits above
``StatusBits.KNOWN``: they do not fail the step, and ``check_status`` warns,
naming them.

Request ``ObservableKind.SOLVER_RESIDUAL`` as well for the measure the Newton
loop stops on: the largest move of the last Newton update over
``newton_velocity_tol * dt``, the rows of a rigid body's rotation scaled by
``newton_velocity_tol / newton_transrate_tol``. It is dimensionless, at most 1
when the loop ended on its tolerance, above 1 when the semi-implicit exit ended
it, and infinite after a failure. ``contact_constitution="al-ipc"`` has no such
measure and does not offer it.

Two choices of a step are fixed when it is captured: the PCG loop reads its
stopping test every 8 or 32 iterations, and work whose size only the device
knows starts from a guessed size. :meth:`~newton.solvers.SolverIPC.graph_outdated`,
called after replays, returns ``True`` when the replayed steps asked for other
choices; capturing again then runs faster. The old graphs stay valid and give
the same results.

.. _ipc-state:

State
~~~~~

The solver keeps positions and velocities in float64 and writes ``state_out``
in float32. Every step reads ``state_in`` on the device. Where ``state_in`` holds
what the solver wrote last (the float32 rounding of its state), the step goes on
from the float64 state; anything else is taken from ``state_in``, a body as the
rigid motion of its pose and twist. A new state for a particle or body that is
not kinematic discards the friction of that step's contacts, as libuipc does
after a state is written: friction acts again from the next step. With several
worlds only that world's friction is discarded.
:meth:`~newton.solvers.SolverIPC.reset` discards friction, the history of
``integrator="bdf2"`` and the Dahl hysteresis of rod joints (:ref:`ipc-rods`),
without a new state, also inside a captured graph.

The solve is built at the model's state, :attr:`~newton.Model.body_q` and
``body_qd``, with one exception. A body whose pose there is the identity, where
:meth:`~newton.ModelBuilder.add_link` and :meth:`~newton.ModelBuilder.add_body`
put a body given no transform, starts where :func:`newton.eval_fk` puts it from
:attr:`~newton.Model.joint_q` and ``joint_qd``. As for Newton's other
maximal-coordinate solvers, the caller writes those poses into the state before
the first step (``newton.eval_fk(model, model.joint_q, model.joint_qd, state)``).

.. _ipc-start:

The start
~~~~~~~~~

The barrier needs every pair of surfaces apart at the start: its energy is
infinite at contact. The solver raises :class:`ValueError` when the surfaces of
different bodies cross each other, or are at or inside their contact thickness
(up to the rounding of float32 positions), as libuipc's check rejects such a
scene. The message names the pair that crosses most, or the closest pair and the
distance to move it by. Give such a
model a gap; the solver does not move geometry for you, since no rule says which
part to move or where. A gap inside ``d_hat`` is valid: the barrier then starts
compressed and pushes the surfaces apart in the first steps.

The start is the state the first step takes. Where the model holds it, the
solver checks it when it is built. The pose of a kinematic body, and that of a
link the model leaves at the identity, are the caller's, written before the
first step: the pairs with such a body are checked by the first step, on its
``state_in``, before anything runs. A first step captured in a graph cannot read
its ``state_in`` on the host: it raises when such a pair fails in the model's
state, and otherwise its start is checked on the device, as every state written
between steps is (below).

The thickness of a pair is the sum of its two sides' (see :ref:`ipc-contact`).
Surfaces that never move may touch or cross, and pairs that do not collide may
overlap. The check tests every primitive of one body against every primitive of
another where their bounding boxes come that close, and skips, with a warning,
a pair of bodies with more than ``2**31 - 1`` primitive pairs.

**States written between steps.** A step takes from ``state_in`` what differs
from what the solver wrote last (:ref:`ipc-state`), and the kernels that load it
note every position they take. Where a step takes a new position for a dynamic
body or particle, as after a reset, it first checks its start, on the device and
inside a captured graph too: the check above, of the pairs with a vertex whose
position was written. A pair that crosses, or is at or inside its contact
thickness (up to the rounding of float32 positions), fails the step with
``THICKNESS_VIOLATED`` before it runs, and the bodies stay where they were
written. A captured step sets the bit; one that is not captured raises
:class:`RuntimeError` naming the pair, and with independent worlds only the
world of the pair fails. The check takes the written state as it is, with no
sweep from where the last step left it: a reset is a jump, not a motion, and a
body put back through the space another one fills is a valid reset. The moves
of kinematic bodies and particles and of a coupled solver's proxies are swept
instead (see :ref:`ipc-kinematic-bodies`), and checked as written only where a
step places them without a sweep: at the first step and at the steps after a
reset or a failed step. A step in which nothing new is written costs, besides
the comparison its loads make anyway, the reset of two small arrays of flags and
one conditional node: 12 µs on a captured step of 0.49 ms of 64 boxes resting on
the ground. A step that takes a written state searches the pairs near the
written vertices through bounding-volume trees refit to its start, those of the
written vertex's world only: one box written among the 64, or in one of 256
worlds of a box, adds 0.18 ms.

.. _ipc-tolerances:

Tolerances
~~~~~~~~~~

A step ends when the Newton update moves no vertex by more than
``newton_velocity_tol * dt``: the last update changes no velocity by more than
``newton_velocity_tol``, 0.05 m/s by default, as in libuipc. Where a body should
gain less velocity than that in a step, as in slow sliding against friction,
lower the tolerance; otherwise the body creeps at about ``eps_velocity``. The solver lowers its defaults itself in
two cases: an entry of a coupled solver with proxies takes 5e-4 m/s, 1e-3 1/s
and no semi-implicit exit (:ref:`ipc-coupling`), and a model with friction on its
joints or tendons takes 0.4 ``eps_velocity`` (:ref:`ipc-joint-terms`). The
semi-implicit exit, on by default as in libuipc, ends the loop after
``semi_implicit_k_min`` iterations once CCD admits a full update, without the
tolerance test.

.. _ipc-d-hat:

The barrier distance d_hat
~~~~~~~~~~~~~~~~~~~~~~~~~~

``d_hat`` [m] is the distance below which the barrier acts. With the default
stiffness a surface at rest settles just inside it: the box above rests 9.3 mm
over the ground with ``d_hat`` 1 cm, and in Newton's ``basic_shapes`` example a
sphere rests 5.8 mm, a capsule 6.4 mm and a cylinder 8.5 mm above their rest
heights. Pass a smaller ``d_hat`` for tighter contact; the solver gives a
``d_hat`` argument to every shape and particle. A smaller value makes the
barrier steeper, which can take more Newton iterations, and leaves less room
for a kinematic jump, which fails the step when it meets a surface (see
:ref:`ipc-kinematic-bodies`).

Without a ``d_hat`` argument every shape and particle gets a value of its own,
as a libuipc geometry gets its meta ``d_hat``. Two surfaces in contact act
below the mean of their two values, and a surface and a plane below the
surface's value. A part's value is libuipc's 0.01 m, lowered to the smallest
limit of the part itself and of every part it can touch:

- a colliding shape: a quarter of its half-size, which is a sphere's or
  capsule's radius, the smaller of a cylinder's or cone's radius and half
  height, a box's or ellipsoid's smallest half extent, or half of a mesh's
  thinnest extent;
- a free particle that collides: a quarter of its radius;
- the particles of cloth and tetrahedra: half the median rest length of the
  edges of the cloth and tetrahedra of their world.

A plane and a height field set no limit. Two parts can touch where Newton's
collision groups, collide flags, filter pairs and particle contact let them
collide, in one world or with a global part.

A ``d_hat`` that spans a small shape lets it rest up to its own size away, and
puts more contact pairs inside ``d_hat`` than the contact lists hold. One as long
as the edges of a cloth or soft body puts its non-adjacent primitives inside
``d_hat`` at rest, and the barrier stretches it. Since both values of a pair are
below both parts' limits, so is their mean: the rule keeps every pair below
both, and a part that can never touch a small one keeps its larger value. In a
model whose parts can all touch each other every part gets the same value. The
particles of one cloth or soft body take the smallest value among them, because
libuipc needs one value on the corners of a contact triangle.

With several worlds a world's parts get the values they would get with the
world alone. A global shape other than a plane gets the smallest value of the
worlds that can touch it, so the pairs of a world with such a shape can be
closer than with the world alone. The values are chosen whenever the solve is
built. The solver also keeps the smallest value of all parts as the value for
the whole model, libuipc's ``contact/d_hat``, from which libuipc's stiffness
range is computed (``adaptive_kappa``, :ref:`ipc-contact`); with independent
worlds that range takes the smallest part of every world, not only of its own.

.. _ipc-model-content:

Model content
-------------

The solver simulates:

- particles and tetrahedra, with libuipc's stable Neo-Hookean elasticity
  (:ref:`ipc-soft-bodies`);
- cloth: the triangles of :attr:`~newton.Model.tri_indices` that are on no
  tetrahedral body, as libuipc's Baraff-Witkin membrane, and the bending edges
  of :attr:`~newton.Model.edge_indices`, as discrete-shell hinges
  (:ref:`ipc-cloth`);
- springs between particles (:ref:`ipc-soft-bodies`) and particle-body
  attachments (:ref:`ipc-attachments`);
- Newton's damping of springs, attachments, tetrahedra, triangles and hinges,
  and the drag and lift of triangles (:ref:`ipc-damping`);
- rigid bodies with box, sphere, capsule, cylinder, cone, ellipsoid, mesh or
  convex mesh shapes, as affine bodies (:ref:`ipc-rigid-bodies`);
- static shapes as fixed triangle meshes, static planes as half spaces, and
  height fields as Newton's collision triangles, two per cell of the grid;
- contact between all of these, with friction, between the pairs that Newton's
  collision rules let collide (:ref:`ipc-contact`), and collision meshes whose
  surface moves along itself, such as a conveyor belt
  (:ref:`ipc-surface-velocity`);
- kinematic particles and bodies, which follow the poses of ``state_in``
  (:ref:`ipc-kinematic-bodies`);
- revolute, prismatic, fixed, ball, D6, universal and distance joints with
  drives, springs, damping, limits, armature, friction and
  :attr:`~newton.Control.joint_f`, mimic joints and MuJoCo's fixed tendons
  (:ref:`ipc-joints`);
- MuJoCo's spatial tendons, through sites, around spheres and cylinders and over
  pulleys (:ref:`ipc-spatial-tendons`);
- rods and cables: the stretch, shear, bend and twist of rod joints, with their
  damping and SolverVBD's Dahl hysteresis (:ref:`ipc-rods`);
- external forces: :attr:`~newton.State.particle_f`,
  :attr:`~newton.State.body_f` and :attr:`~newton.Control.joint_f`, each held
  constant over the step, as libuipc holds its external forces. A body wrench
  acts at the centre of mass;
- gravity per world: a particle or body falls with the
  :attr:`~newton.Model.gravity` of its world, a global one with the last entry;
- several worlds, each solved as it would be alone (:ref:`ipc-worlds`);
- as outputs: the status of every step (:ref:`ipc-status`), its contacts and
  their forces (:ref:`ipc-contact-output`), and the bodies' parent-joint
  wrenches and accelerations (:ref:`ipc-body-observables`).

It is an entry of Newton's coupled solvers (:ref:`ipc-coupling`). What it does
not simulate it refuses when it is built (:ref:`ipc-refused`), and
:ref:`ipc-not-read` lists the few inputs it takes without reading.

.. _ipc-soft-bodies:

Particles, soft bodies and springs
----------------------------------

A particle that is on no cloth triangle and no tetrahedron is a free particle:
a sphere of its :attr:`~newton.Model.particle_radius` in contact. A particle of
mass 0, or one that is not :attr:`~newton.ParticleFlags.ACTIVE`, is kinematic:
it follows ``state_in`` (:ref:`ipc-kinematic-bodies`). An inactive particle on
no cloth or tetrahedron collides with nothing, as in Newton.

A tetrahedron is libuipc's stable Neo-Hookean element. Its Lamé parameters are
``tet_materials[:, 0]`` (``k_mu``) and ``tet_materials[:, 1]`` (``k_lambda``);
libuipc stores the second as ``k_lambda + k_mu``, and the solver passes it so.
``tet_materials[:, 2]`` (``k_damp``) damps the rate of the strain
(:ref:`ipc-damping`). Each connected group of tetrahedra, like each connected
cloth, is one body that collides with itself, as finite elements do in
libuipc; ``d_hat`` must stay below the spacing of its non-adjacent surface
primitives at rest, as the default does.

A spring is libuipc's Hookean spring, with Newton's energy
:math:`\tfrac{1}{2} k_e (l - L_0)^2` of ``spring_stiffness`` and
``spring_rest_length``. A spring of rest length 0 is libuipc's soft vertex
stitch, which has the same energy and a force along the vector between its two
particles. A spring adds no contact primitive: the particles it joins collide
as other particles do. A negative rest length or stiffness, and a spring of a
particle to itself, are refused. ``spring_damping`` damps it
(:ref:`ipc-damping`).

.. _ipc-cloth:

Cloth
~~~~~

libuipc-warp's membrane energy per rest area is

.. math::

    \Psi = K_a \phi(|F a|) + K_b \phi(|F b|) + K_g ((F a) \cdot (F b))^2,
    \qquad \phi(s) = (s - 1)^2 + \eta \max(s - 1, 0)^3,

with the material axes ``a`` and ``b`` of each triangle's rest frame and
:math:`\eta` = ``cloth_strain_rate`` (libuipc's strain limiting; 0 makes the
membrane quadratic). Newton's ``tri_materials[:, 0]`` (``tri_ke``,
:math:`\mu`) and ``tri_materials[:, 1]`` (``tri_ka``, :math:`\lambda`) are
membrane Lamé moduli [N/m], and at small strain Newton's membrane is linear
elasticity, :math:`\mu\,\varepsilon:\varepsilon + \tfrac{\lambda}{2}(\mathrm{tr}\,\varepsilon)^2`.
The solver matches its energy for a stretch along either material axis with the
other held, and for simple shear:

- :math:`K_a = K_b = \mu + \lambda / 2` [N/m];
- :math:`K_g = \mu / 2` [N/m], since :math:`(F a) \cdot (F b) \approx 2 \varepsilon_{ab}`.

The Baraff-Witkin model has no Poisson coupling (no
:math:`\lambda\,\varepsilon_{aa}\varepsilon_{bb}` term), so an equal stretch in
both directions meets less resistance than in Newton's model.

Newton's hinge energy is :math:`\tfrac{1}{2} k_e L_0 (\theta - \bar\theta)^2`
with :math:`k_e` = ``edge_bending_properties[:, 0]``. libuipc's is
:math:`\kappa L_0 (\theta - \bar\theta)^2 / \bar h`, with
:math:`\bar h = (A_1 + A_2) / (3 L_0)` from the two rest triangle areas, so each
hinge takes :math:`\kappa = k_e \bar h / 2`. Particle masses are used as they
are.

The rest shapes are Newton's. A triangle's rest edges come from its
:attr:`~newton.Model.tri_poses`, laid into the plane of the triangle at
:attr:`~newton.Model.particle_q`. A hinge's rest angle is its
:attr:`~newton.Model.edge_rest_angle`: the signed angle between the normals of
``(o0, v1, v2)`` and ``(o1, v2, v1)`` for the row ``(o0, o1, v1, v2)``. That is
libuipc's angle where ``o0``'s triangle runs ``v1 -> v2``, as in the rows
Newton's cloth builders make; the solver negates it where a row given by hand
runs the edge the other way, and raises :class:`ValueError` for a row whose two
triangles run the edge the same way. The hinges keep the rest lengths and areas
of ``particle_q``, where Newton's VBD takes ``edge_rest_length``; for a rest
shape equal to the start the two agree.

A cloth built at rest starts without strain. Newton's builders store
``tri_poses`` and ``edge_rest_angle`` in float32, and libuipc's hinge angle, the
arccosine of the normals' cosine, reads up to 1.5e-8 rad on a hinge Newton
measures as flat. Where every triangle's ``tri_poses`` and every hinge's
``edge_rest_angle`` are the float32 roundings of what Newton's builders compute
at ``particle_q``, the solver takes the rest shapes from ``particle_q`` itself,
as libuipc takes them from its rest geometry.

The solve holds copies of the materials and rest shapes of the tetrahedra,
triangles and hinges (``tet_materials``, ``tri_materials``,
``edge_bending_properties``, ``tet_poses``, ``tri_poses``, ``edge_rest_angle``)
and of :attr:`~newton.Model.particle_flags`. Newton has no flag for a change of
them, so :meth:`~newton.solvers.SolverIPC.notify_model_changed` compares them at
every notification, whatever its flags, and builds the solve again when one
changed.

.. _ipc-damping:

Damping
~~~~~~~

Newton's damping coefficients act as Newton's VBD solver defines them: each
resists the rate of change of one quantity. libuipc has no damping term; each
is a dissipation potential of libuipc-warp's ``uipc_warp.ext.damping``, which
the implicit step solves with its own estimate of the rate at the end of the
step, so damping adds no stability limit to the step.

.. list-table::
   :header-rows: 1

   * - Newton's input
     - resists the rate of
     - applies to
   * - ``spring_damping`` [N s/m]
     - a spring's length; for a spring of rest length 0, the vector between
       its particles
     - every spring
   * - ``attachment_particle_body_damping`` [N s/m]
     - the vector between the particle and its point on the body
     - an enabled attachment
   * - ``tet_materials[:, 2]`` (``k_damp``) [Pa s]
     - a tetrahedron's :math:`F^T F`, over its rest volume
     - ``k_mu`` or ``k_lambda`` above 0
   * - ``tri_materials[:, 2]`` (``tri_kd``) [Pa s m]
     - a cloth triangle's :math:`F^T F`, over its ``tri_areas``
     - ``tri_ke`` or ``tri_ka`` above 0
   * - ``edge_bending_properties[:, 1]`` (``edge_kd``) [N s]
     - a hinge's dihedral angle, times its ``edge_rest_length``
     - a hinge of positive stiffness and rest length

None of these quantities changes when the particles move as one rigid body, so
the damping exerts no net force or moment and does not slow a rigid rotation.
SolverIPC differs from VBD in two places. VBD damps a spring of rest length 0
along its length, which has no direction where such a spring rests; SolverIPC
damps the vector between its particles, the damper that pairs with that spring's
force. VBD ignores the damping of a hard attachment, which it holds exactly;
SolverIPC holds a hard attachment by a penalty (:ref:`ipc-attachments`) and
damps the penalty's stretch.

Newton's builders give every cloth triangle ``tri_kd`` 10
(``ModelBuilder.default_tri_kd``), so every cloth is damped unless ``tri_kd`` is
set to 0. On a level cloth of 48 x 48 squares the damping of the triangles and
hinges makes a Newton iteration 29% slower, with the same iterations per step.

A kind of damping takes its terms when the solve is built, if one of its
coefficients is above 0 then. The coefficients are read on the device at every
step, so a captured step takes new values at its next replay.
:meth:`~newton.solvers.SolverIPC.notify_model_changed` builds the solve again for
a changed spring, element or hinge damping, and for an attachment damping where
the solve has none. A negative or non-finite damping, and a damped hinge without
a positive ``edge_rest_length``, are refused. A step with damping terms solves a
larger system, so its rounding differs from that of a step without them, even
where every coefficient is 0.

Drag and lift, ``tri_materials[:, 3]`` (``tri_drag``) and ``[:, 4]``
(``tri_lift``), are SemiImplicit's: explicit forces of every triangle, computed
from the positions and velocities of ``state_in`` and applied to its free
particles as external accelerations of the step. Being explicit, the drag is
stable while :math:`k_{drag} A |\bar v| \Delta t / m` stays below about 1. A
drag or lift that is not finite, or a negative drag, is refused.

.. _ipc-contact:

Contact
-------

Which pairs collide, and with which friction, is libuipc's contact table: every
vertex and half plane belongs to a contact element, and every pair of elements
has a friction coefficient and an on-off switch. Two shapes collide as
:attr:`~newton.Model.shape_contact_pairs` says: both collide with shapes, their
collision groups and worlds let them meet, and they are not a filter pair. Shapes
of one body never collide, and the solver checks at construction that its pairs
are exactly the model's. A particle collides with a shape that has
:attr:`~newton.ShapeFlags.COLLIDE_PARTICLES`. Two free particles, or a free
particle and a cloth or tetrahedron vertex, collide while the model has more
than one particle and a :attr:`~newton.Model.particle_grid`, as Newton's particle
solvers decide; setting the grid to ``None`` switches particle contact off.
Cloth and tetrahedra always collide with themselves and with each other.

The friction coefficient of a pair is the arithmetic mean of the two shapes'
:attr:`~newton.Model.shape_material_mu`, of :attr:`~newton.Model.soft_contact_mu`
and the shape's for a particle and a shape, and
:attr:`~newton.Model.particle_mu` for two particles, as in Newton's
semi-implicit, Featherstone and XPBD solvers. Friction is libuipc's lagged
friction: the normal force and the contact basis of a pair are taken at the
step's start, and the friction force is smoothed below the sliding speed
``eps_velocity``. Below that speed static friction acts as a stiff spring, so a
body held by friction creeps at a speed below ``eps_velocity``.

The contact thickness is the distance at which the barrier becomes infinite:

- a free particle is a sphere of its :attr:`~newton.Model.particle_radius`, so it
  rests its radius above a shape, and two free particles rest the sum of their
  radii apart;
- a shape's vertices take its :attr:`~newton.Model.shape_margin`, and a static
  plane's surface moves out by it, so that two shapes rest the sum of their
  margins apart;
- the vertices of cloth and tetrahedra take the ``thickness`` argument, not
  their radius. libuipc gives a vertex's thickness to every edge and triangle it
  is on, and finite elements collide with themselves, so a cloth vertex of
  radius r would hold the cloth's non-adjacent triangles 2r apart. Newton's
  radii are sphere radii, often larger than the spacing of a cloth.

The contact stiffness, the barrier's :math:`\kappa`, is ``kappa`` for every
pair, taken as given. libuipc does not take a stiffness as given: it keeps it in
a range :math:`[\kappa_{lo}, 100\,\kappa_{lo}]` with
:math:`\kappa_{lo} = c\,\bar m / \Delta t^2`, where :math:`\bar m` is the mean
mass of the vertices and :math:`c` a factor of the scene's diagonal and
``d_hat``. A scene that sets no contact model takes :math:`\kappa_{lo}`, and a
stiffness the user gives is clamped into the range. With
``adaptive_kappa=True`` the solver does the same: ``kappa=None`` takes
:math:`\kappa_{lo}`, a given ``kappa`` is clamped into the range, and the first
step that clamps it warns with :class:`~newton.NewtonWarning`. The range
follows every step's ``dt``, also in captured steps, and comes from the masses
when the solve is built: every particle, every vertex of the triangulated
surfaces of the bodies, and one vertex per plane, each body's mass counted once.
Newton's static shapes have no mass and count with 0, as an obstacle does on
libuipc-warp's array path, where libuipc counts the mass of every body, fixed or
not. :meth:`~newton.solvers.SolverIPC.contact_stiffness` gives the stiffness of a
step of any ``dt``. With independent worlds every world takes the range of the
same world alone (:ref:`ipc-worlds`).

A face that presses on a triangulated surface is a patch of contact terms, one
per vertex or edge in range. Their resultant need not pass through the centre of
mass, and the smoothed static friction lets the patch creep below
``eps_velocity``, so a box that strikes a convex hull can twist a little more
than in Newton's other solvers (5.1e-3 rad/s in Newton's
``test_box_convex_hull``, which allows 3e-3). This is no missing feature: the
patch resists twisting as a face should, and the excess is the creep of the
smoothed static friction, at a speed that scales with ``eps_velocity``.

.. _ipc-contact-damping:

Contact damping
~~~~~~~~~~~~~~~

libuipc's contact has no damping: the barrier gives back the energy an impact
stores in it, and what an impact loses is the integrator's numerical damping.
Under implicit Euler at the usual step sizes that is nearly all of it; under
``integrator="bdf2"``, or with small steps, a body rebounds with most of its
impact speed. ``contact_damping`` adds a viscous force against the normal motion
of two surfaces in contact:

- ``None`` (the default): no damping, as in libuipc.
- A number :math:`c` [N s/m]: every pair in contact takes the force
  :math:`-c\, v_n \mathbf n`, :math:`v_n` the pair's velocity along its normal
  :math:`\mathbf n` over the step.
- ``"material"``: Newton's contact damping coefficients, combined as
  :class:`~newton.solvers.SolverVBD` combines them: the mean of the two shapes'
  :attr:`~newton.Model.shape_material_kd`, the mean of
  :attr:`~newton.Model.soft_contact_kd` and the shape's for a particle and a
  shape, and ``soft_contact_kd`` for two particles. It is not the default,
  because Newton's default ``kd`` is 100 N s/m on every shape, which libuipc's
  contact does not have.

The damping acts on the pairs in contact at the step's start, with their
normals and closest points taken there, as friction takes them: a pair that
comes into contact during a step is damped from the next step on. It acts while a
pair closes and while it opens, and is zero at rest, so a body settles where it
settles without it. It is the gradient of a dissipation potential that the
step's energy holds beside the barrier, so the step stays a minimization.

It lowers a rebound, the more the slower the impact. An impact over in one or
two steps keeps :math:`1 / (1 + c\, \Delta t / m)` of its rebound under implicit
Euler, :math:`m` being the mass behind the pair. So the rebound depends on the
impact speed: this is no coefficient of restitution, and
:attr:`~newton.Model.shape_material_restitution` stays unread. Every contact term
is damped, as Newton's solvers damp every contact point: a box resting on a
plane through its four corners is damped with :math:`4c`.

``"material"`` reads the coefficients when the solve is built, and
:meth:`~newton.solvers.SolverIPC.notify_model_changed` with ``SHAPE_PROPERTIES``
or ``MODEL_PROPERTIES`` builds the solve again when they changed. A value other
than ``None``, ``"material"`` or a finite number not below 0 raises
:class:`ValueError`, and ``contact_constitution="al-ipc"`` refuses any value but
``None``. The damping's force is part of the friction rows of ``CONTACT_F``
(:ref:`ipc-contact-output`).

.. _ipc-surface-velocity:

Moving surfaces
~~~~~~~~~~~~~~~

A collision mesh with ``Mesh.enable_surface_velocity`` is a surface whose
material moves along it, such as a conveyor belt or a tank tread. The velocity
of the material at a vertex [m/s] is that vertex's entry of
``mesh.mesh.velocities``, the mesh's finalized Warp mesh, in the mesh's
coordinates; the caller may write it on the device before any step, also between
the replays of a captured graph. As Newton's collision pipeline does, the solver
scales it by the shape's scale and turns it into the world by the shape's pose
at the step's start, ``body_q * shape_transform``, so it turns with the body that
carries the mesh.

Friction then measures each contact's slip from where the material was a step
before: a box set on a static belt whose velocities are (1, 0, 0) slides at
:math:`\mu g` until it rides along at 1 m/s, and a tread on a body drives the
body. Only the part along a contact's tangent plane acts, as in Newton's other
solvers; to move a surface along its normal, move the mesh. The velocity enters
each contact through the closest point's weights, where Newton's contacts sample
it at their contact point; both interpolate it linearly over the triangles.
Coincident mesh vertices are welded into one vertex, which takes the velocity of
the vertex that the first triangle at that place uses.
``contact_constitution="al-ipc"`` refuses such a mesh.

.. _ipc-rigid-bodies:

Rigid bodies
------------

Each Newton body is a libuipc affine body: a body whose 12 degrees of freedom
are the translation :math:`p` and the matrix :math:`A` of an affine map, so that
a point of body-frame coordinates :math:`\bar x` sits at :math:`x = A \bar x + p`.
An orthogonality potential :math:`\kappa V \Delta t^2 \|A A^T - I\|_F^2`, with
:math:`V` the volume of the body's collision meshes, holds :math:`A` within a
small strain of a rotation: a near-rigid body, not an exactly rigid one. Its
collision shapes are triangulated, curved ones with ``shape_segments`` segments
around, and form its contact surface.

The kinetic term uses libuipc's dyadic mass, built from
:attr:`~newton.Model.body_mass` :math:`m`, :attr:`~newton.Model.body_com`
:math:`c` and :attr:`~newton.Model.body_inertia` :math:`I`:
:math:`\sum m \bar x = m c` and
:math:`\sum m \bar x \bar x^T = \tfrac{1}{2}\mathrm{tr}(I)\,1 - I + m c c^T`, so
that the affine kinetic energy equals the rigid one for every rigid motion. That
second moment is positive definite where the largest principal moment of
:math:`I` is below the sum of the other two. An inertia past that triangle
inequality takes the correction :meth:`~newton.ModelBuilder.finalize` makes by
default (:attr:`~newton.ModelBuilder.balance_inertia`): every principal moment
raised by the difference and 1e-6 kg m², with a :class:`~newton.NewtonWarning`.
A dynamic body's inertia must be positive definite. Gravity is the exact gravity
force on the affine body. A force :math:`f` at the centre of mass is the
generalized force :math:`(f, f c^T)` on :math:`(p, A)`, and a torque
:math:`\tau` is :math:`(0, \tfrac{1}{2} [\tau]_\times A^{-T})`.

**The body stiffness.** The stiffness is ``body_kappa`` at every step when it is
given. By default it is libuipc's 1e8 Pa, lowered per body and per step to
:math:`25\, I_{max} / (V \Delta t^2)` where that is lower, :math:`I_{max}` being
the body's largest principal moment of inertia. The reason: the first Newton
update of a step moves a turning body along a straight line in its 12
coordinates, which stretches it by about :math:`\theta^2 / 2` for a turn
:math:`\theta`. At libuipc's stiffness the line search cuts that update short
wherever the turn is large for the body, and the later updates, held back by the
stretch, pass the displacement test long before the turn is done: a 1 m cube that
a torque turns by 0.06 rad in a 0.1 s step reaches 31% of its angular velocity,
and a 10 cm cube spinning at 0.6 rad/s keeps 33% of it over six steps of
1/60 s. With the lowered stiffness the first update takes a turn of up to 0.1 rad
about the axis of the largest moment in full. The price is strain in proportion
to :math:`\Delta t^2`: the lowest of five stacked 10 cm cubes is compressed by
4e-5 at :math:`\Delta t` = 1/600 s and 4e-3 at 1/60 s (6e-6 at 1e8 Pa). A thin
body keeps its stiffness along its length and so turns less about its long axis
than about the others. Pass ``body_kappa=1e8`` for libuipc's stiffness at every
step.

**The rotation.** In libuipc's step (``body_rotation="linear"``) a turning body
moves along a straight line in its 12 coordinates within the step, and the
orthogonality potential pulls it back onto the rotations. Under implicit Euler a
free spin at :math:`\omega` then keeps :math:`\operatorname{atan2}(\sin t, 2 - \cos t)`
of a turn :math:`t = \omega \Delta t` per step: it loses about :math:`t^2` of its
angular velocity per step, 1e-4 at 1 rad/s and :math:`\Delta t` = 0.01 s, and a
body tumbling about no principal axis loses angular momentum and energy alike.
The loss belongs to the integrator, not to the bodies, and Newton's rigid-body
solvers and tests expect a free body to keep its spin. So by default under
implicit Euler (``body_rotation="rigid"``, libuipc-warp's
``affine_rotation="rigid"``) every free body is predicted by the flight of a
rigid body, which keeps its angular momentum :math:`L` about the centre of mass
and turns at the implicit midpoint of the free rigid body; after the step its
velocity carries that momentum plus the angular impulse of the step's contact and
joint forces. A free body then keeps its angular momentum and kinetic energy, a
spin about a principal axis is the exact motion, and a torque spins a body up by
:math:`\tau / I` per second. At the default tolerances the Newton loop leaves a
little of each turn: a 20 cm cube at 2.24 rad/s keeps 0.995 of its spin over 200
steps of 0.01 s, against 0.910 in libuipc's step. A pendulum on a hinge away from
its centre of mass keeps implicit Euler's damping of its swing. Fixed, kinematic
and massless bodies keep libuipc's prediction.

``integrator="bdf2"`` predicts from the step before as well, which the rigid
flight does not turn. Under BDF2 the default is therefore ``"linear"``, and
``body_rotation="rigid"`` with ``integrator="bdf2"`` raises :class:`ValueError`.
Pass ``body_rotation="linear"`` under implicit Euler to follow libuipc's
trajectories, as a comparison with libuipc needs.

**The output.** After each step the solver writes ``body_q`` and ``body_qd``.
:math:`A` is projected onto its nearest rotation :math:`R` by the polar
decomposition :math:`A = R S`, since a quaternion holds only a rotation; the
residual stretch :math:`S - 1` is of the order of the orthogonality strain. The
body origin is placed so that the centre of mass stays where the affine map puts
it, :math:`A c + p`, and the linear velocity is that point's,
:math:`\dot p + \dot A c`. The angular velocity is the one that carries the
body's angular momentum, :math:`(R I R^T)^{-1} L`, under
``body_rotation="rigid"``, and the axial vector of the skew part of
:math:`\dot A A^{-1}` in libuipc's step, which also counts the spin of the
body's strain.

.. _ipc-joints:

Joints
------

Revolute, prismatic, fixed and ball joints are libuipc's affine-body joints
between the affine bodies of parent and child; a parent of -1 is a fixed body
without mass at the world's origin. A revolute or prismatic joint holds two
points of its axis together, at ``joint_X_p`` and ``joint_X_c`` and one metre
along ``joint_axis`` from them; a prismatic joint also holds the relative
rotation, a fixed joint the anchors and the rotation, and a ball joint the
anchors. The zero of a joint's coordinate, and the rotation a joint holds, are
where the joint frames coincide, as in Newton, whatever the bodies' poses when
the solver is built.

**Joints are penalties.** libuipc's joint is the energy
:math:`\tfrac{K}{2}|C|^2` of the joint's violation :math:`C`, a term of the
step's incremental potential, not a hard constraint. :math:`K` is
``joint_strength_ratio`` (default 100, libuipc's) times the mass of the dynamic
bodies the joint joins; the world and kinematic bodies count 0. Against a
translation of the child, a revolute or prismatic joint, which holds two points,
is a spring of :math:`2K/\Delta t^2` [N/m], and a fixed or ball joint one of
:math:`K/\Delta t^2`. A load that would move the child by :math:`\delta` in one
step stretches the joint by about :math:`\delta / (2\gamma)` or
:math:`\delta / \gamma`, :math:`\gamma` = ``joint_strength_ratio``: gravity at
1/60 s stretches a revolute joint by 1.4e-5 m and a ball joint by 2.7e-5 m. A
larger ratio holds the joints closer and makes the linear system stiffer. Making
the joints hard would take a formulation that libuipc does not have and that
changes how the Newton loop converges, so the solver keeps libuipc's penalty, as
Newton's Featherstone and MuJoCo solvers keep their limits soft.

**Drives, springs and damping.** The drive, passive spring and damping of a
revolute or prismatic joint,
:math:`\tau = k_e (q^* - q) + k_d (\dot q^* - \dot q) - k_s (q - q_r) - c \dot q`
(``joint_target_ke``, ``joint_target_kd``, ``joint_stiffness``,
``joint_rest_q``, ``joint_damping``, with the targets
:attr:`~newton.Control.joint_target_q` and ``joint_target_qd``), are exact for
the step's integrator: taken at the end of the step with
:math:`\dot q = (q - q_n)/\Delta t` (BDF2:
:math:`(3 q - 4 q_n + q_{n-1}) / (2 \Delta t)`), the force is minus the gradient
of one quadratic potential in :math:`q`, which the solver writes into libuipc's
driving term at every step. The gains act whatever ``joint_target_mode`` says, as
in Newton's other maximal-coordinate solvers.

**Armature.** A revolute or prismatic joint's ``joint_armature`` :math:`a`,
which Featherstone and MuJoCo add to the joint's inertia, is the kinetic term of
its coordinate in the same potential,
:math:`\tfrac{a}{2 \Delta t^2} (q - \tilde q)^2` with
:math:`\tilde q = q_n + \Delta t \dot q_n`, :math:`\dot q_n` being the rate the
bodies' twists give at the step's start (under BDF2, 9/4 of it with BDF2's
prediction). A free joint's armature is refused: it has no such term.

**Effort limits.** libuipc's driving term is a quadratic potential and cannot
bound its force. A ``joint_effort_limit`` :math:`L` below Newton's default 1e6
therefore bounds the force of the gains as decided at the start of every step:
where :math:`|k_e (q^* - q_n) + k_d (\dot q^* - \dot q_n)| > L`, the step takes
the constant force :math:`\pm L` in place of the gains' part of the potential,
and the passive spring and damping remain. The bound holds at the start of every
step, where MuJoCo clamps its actuators' force. A step that starts below the
bound and ends above it, or the reverse, takes it one step late. For a hinge
driven through a step change of its target, the motion then differs from that
of a force clamped at the end of every step by 1.6e-4 of its size at steps of
10 ms, and by 4e-6 at 1 ms. A bound inside the step would need a drive potential
of another form, which libuipc does not have; at the step sizes of robot models
the late bound costs little. With ``enable_joint_effort_limit=False`` such a limit on a driven degree
of freedom is refused instead.

**Limits.** A limit is libuipc's cubic penalty outside
``[joint_limit_lower, joint_limit_upper]``, not Newton's linear spring: at an
overshoot :math:`\delta` it stores :math:`k_l \delta^3 / (3 \delta_r)` with
:math:`k_l` = ``joint_limit_ke`` and :math:`\delta_r` = 1e-3 rad or m, so that it
pushes back with :math:`k_l \delta_r` at :math:`\delta_r` and harder beyond. A
load :math:`F` rests :math:`\sqrt{F \delta_r / k_l}` past the limit. A
``joint_limit_ke`` of 0 holds nothing, and a bound at Newton's ``MAXVAL``
(1e10) is no bound. libuipc's limit has no damping. A joint whose coordinate is
past one of its limits at the start of a step takes Featherstone's and
SemiImplicit's damping :math:`-k_{ld} \dot q`, :math:`k_{ld}` =
``joint_limit_kd``, in the same quadratic potential as ``joint_damping``: exact
for the integrator, decided at the step's start, as those solvers decide it. A
joint that reaches its limit within a step is damped from the next step on.
Those solvers also switch the drive off past a limit; SolverIPC keeps it on, as
MuJoCo does, since robot models take MuJoCo's meaning. With
``enable_joint_limit_damping=False`` ``joint_limit_kd`` is not read, and a body
that falls onto a limit bounces.

**joint_f.** :attr:`~newton.Control.joint_f` of a revolute or prismatic joint is
libuipc's joint torque or force between the two bodies; of a ball joint a torque
in the parent's joint frame on the child, the parent taking the reaction; of a
free joint a wrench at the child's centre of mass in the world frame, the
parent, if it is a body, taking the reaction.

**What a step reads.** ``joint_enabled``, the gains, targets, springs, damping,
limits and forces are read on the device at every step: a captured step takes
new values at its next replay. A change of the joints' types, bodies, frames or
axes builds the solve again (``notify_model_changed`` with ``JOINT_PROPERTIES``).

**joint_q and joint_qd.** After every step they hold the coordinates
:func:`newton.eval_ik` computes from the bodies' state, with two exceptions: a
revolute joint's angle is libuipc's angle, which goes on past :math:`\pi` as in
Newton's generalized-coordinate solvers, and a prismatic joint's position is
libuipc's distance. A revolute joint's angle continues from the step before;
where ``state_in.joint_q`` is not what the solver wrote, it picks the turn of
the bodies' angle.

**Loop joints.** Newton's MJCF and USD importers turn a CONNECT or WELD equality
into a BALL or FIXED joint outside the articulations, a loop joint, and keep the
equality row, linked to the joint (``model.mujoco.equality_constraint_target_kind``
and ``equality_constraint_target``). SolverIPC simulates the loop joint as any
other joint and takes the row as held by it, by SolverSteel's rule: a CONNECT
row linked to a BALL joint, or a WELD row linked to a FIXED joint (to a BALL
joint when its ``torquescale`` is 0), that is in no articulation and joins the
row's two bodies, or its body and the world. Every other enabled row is refused,
when the solver is built and when
:meth:`~newton.solvers.SolverIPC.notify_model_changed` reports
``CONSTRAINT_PROPERTIES``. A row's ``equality_constraint_enabled`` does not
switch its loop joint; ``joint_enabled`` does.

.. _ipc-ball:

Ball joints
~~~~~~~~~~~

A ball joint holds its anchors and lets the bodies turn freely about them. Its
drives, springs, damping, armature and friction are libuipc-warp's ball terms
(``uipc_warp.ext.ball``), since libuipc has none. They act per degree of
freedom, an axis of the parent's joint frame, on the joint's rotation: the
rotation of the child's joint frame in the parent's, which ``joint_q`` holds as
a quaternion.

- The drive pulls the rotation vector of the joint's rotation toward that of
  :attr:`~newton.Control.joint_target_q` (the four entries at the joint's
  coordinates, a quaternion, converted as SolverMuJoCo converts it, to at most a
  half turn) with ``joint_target_ke``, and the joint's angular velocity toward
  ``joint_target_qd`` (in the parent's joint frame) with ``joint_target_kd``.
- ``joint_stiffness`` pulls the rotation from ``joint_rest_q``, a quaternion,
  back to zero, per axis of the child's joint frame.
- ``joint_damping`` damps the joint's angular velocity, and ``joint_armature``
  adds an inertia to its rotation.
- ``joint_friction`` bounds a dry friction torque about each axis, the smoothed
  friction of :ref:`ipc-joint-terms`.
- ``joint_effort_limit`` below 1e6 bounds the drive's torque about each axis,
  decided at the start of every step as for a revolute joint.

Each is a term of the step's energy, so the step is implicit in all of them. The
degrees of freedom are the axes of the parent's joint frame, as in ``joint_qd``;
MuJoCo's per-axis actuators act about the child body's axes instead. With equal
gains on the three axes, which ``add_joint_ball`` writes, that changes nothing:
the drive is MuJoCo's ball actuators' to first order in the rotation (MuJoCo's
torque is not the gradient of an energy), and the spring and damping are
MuJoCo's and Featherstone's. With unequal gains it does, where ``joint_X_c``
turns the child body's axes away from the joint frame's; such a drive differs
from MuJoCo's. ``joint_target_mode`` is not read: the gains act, as for revolute
joints.

The rotation vector flips where the rotation crosses a half turn from the
identity (the drive) or from the rest (the spring), as MuJoCo's does; the line
search shortens an update that would cross it. A ball joint whose terms are all
0 gets none, and one given a term after the solver was built gets its terms when
:meth:`~newton.solvers.SolverIPC.notify_model_changed` reports
``JOINT_DOF_PROPERTIES``, which builds the solve again. Newton defines no limits
of a ball joint (``add_joint_ball`` takes none, its importers drop them, and its
solvers do not read them), so a finite limit with a ``joint_limit_ke`` above 0
is refused.

.. _ipc-d6-joints:

D6 joints
~~~~~~~~~

libuipc has no D6 joint. A D6 joint whose free axes leave locked what one of
libuipc's joints locks is that joint, exactly:

- no axis: a fixed joint;
- one angular axis: a revolute joint about it, with its drive, spring, damping,
  limits, armature and ``joint_f``;
- one linear axis: a prismatic joint along it, likewise;
- three orthonormal angular axes: a ball joint, which takes no drive, spring,
  damping, armature, friction or limit, since its coordinates are angles about
  its axes, not the rotation vector the ball terms act on;
- three orthonormal linear and three orthonormal angular axes: no joint, as a
  free joint.

Two angular axes and no linear one are a universal joint
(:ref:`ipc-universal-joints`). ``joint_q`` and ``joint_qd`` hold Newton's
coordinates of the D6 joint: of three angular axes the angles
:func:`newton.eval_ik` takes, of linear axes the projections of the joint
frames' offset. ``joint_f`` of a D6 joint of three angular axes is
Featherstone's generalized force: a torque along the dual basis of the axes as
the coordinates carry them, and with linear axes a force along them on the
child's joint frame, the parent taking the reaction. Other D6 joints (linear and
angular axes together, but three and three) lock what no joint of SolverIPC
locks, and angular axes that are not orthonormal have no coordinates in Newton's
decomposition; both are refused.

.. _ipc-universal-joints:

Universal joints
~~~~~~~~~~~~~~~~

A D6 joint of two angular axes and no linear one is a universal (Cardan) joint:
the child turns about the parent's ``axis_0`` by the angle :math:`q_0`, then about
the carried ``axis_1`` by :math:`q_1`. libuipc's joints lock no rotation, two or
three; libuipc-warp's universal joint locks one. It holds the joint frames'
origins together, as a ball joint does, and keeps the child's ``axis_1``
perpendicular to the parent's ``axis_0`` with the energy
:math:`K (u_0 \cdot w_1)^2` of the two axes as the bodies carry them, :math:`K`
being a ball joint's. Against a turn about ``axis_0 x axis_1`` it is as stiff as a
fixed joint. Its angles are the angles :func:`newton.eval_ik` computes, also
where the penalty lets the lock give a little.

Each angle takes the terms of its degree of freedom as a revolute joint's angle
takes them: the drive, passive spring and damping, exact for the integrator; the
armature; the effort limit, decided at the start of every step; the limits, with
``joint_limit_kd``; ``joint_f``, a torque on the angle; and ``joint_friction``,
the smoothed friction of :ref:`ipc-joint-terms`. The angles are multi-turn, as a
revolute joint's: ``joint_q`` continues them past :math:`\pi`, and
``joint_qd`` holds :func:`newton.eval_ik`'s rates.

.. _ipc-distance-joints:

Distance joints
~~~~~~~~~~~~~~~

libuipc has no distance joint. The bounds of a distance joint are those of its
first degree of freedom, a negative bound being none.

- Equal bounds :math:`L \ge 0` (one length): the origins of the two joint
  frames are held :math:`L` apart by libuipc's soft vertex stitch of rest length
  :math:`L` between two anchors of the bodies (the world's fixed body for a
  parent of -1), with a ball joint's penalty, ``K/2 (|x_c - x_p| - L)^2``.
- A range ``[lower, upper]``, or one bound: the distance is free inside the
  range and pushed back outside it by the cubic limit of every SolverIPC joint,
  of stiffness ``joint_limit_ke`` at 1 mm past a bound, with ``joint_limit_kd``
  while the distance starts a step outside the range. Newton's default
  ``limit_ke`` 1e4 N/m lets a weight of 78 N rest 2.8 mm past a bound.
- Both bounds negative: nothing is held, as Featherstone and SemiImplicit take
  every distance joint.

A disabled distance joint holds nothing, and a lower bound above the upper is
refused. Every distance joint takes ``joint_q``, ``joint_qd`` and ``joint_f`` as
a free joint does; drives and friction on it are refused. A change of the bounds
that turns a range into one length or none, or back, builds the solve again; a
change within a range is read at the next step.

.. _ipc-rods:

Rods and cables
~~~~~~~~~~~~~~~

A rod or cable that :meth:`~newton.ModelBuilder.add_rod` builds is a chain of
capsule bodies joined by joints of type ROD
(:meth:`~newton.ModelBuilder.add_joint_rod`). libuipc's rods are finite
elements of points without a twist; SolverIPC gives every rod joint
libuipc-warp's rod joints (``uipc_warp.ext.rods``), which act between the two
bodies with Newton's strain, each component a spring toward its rest and a
damping of its rate:

- the stretch and shear, the child's anchor in the parent's anchor frame: the
  springs are ``joint_target_ke`` of the degrees of freedom ``shear_x``,
  ``shear_y`` and ``stretch``, their damping ``joint_target_kd``, and the
  translation of ``joint_rest_q`` is the rest (SolverKamino reads it too;
  ``add_rod`` writes 0, so a rod it builds behaves as in SolverVBD);
- the bend, the curvature binormal of the two segments' tangents in the parent's
  frame (``2 tan(theta / 2)`` for a bend by :math:`\theta`), and the twist,
  after the parent's frame is carried to the child's tangent, against the
  rotation of ``joint_rest_q``: the degrees of freedom ``bend_x``, ``bend_y``
  and ``twist``. A twist beyond a half turn at one joint wraps, as in SolverVBD
  and SolverKamino. X and Y may differ, as for a flat ribbon (SolverVBD refuses
  that).

Every term is part of the step's energy, so the step is implicit in all of
them, with the damping on the rate of the strain as SolverVBD's. Newton caps the
bend at 20 (168.6 degrees per joint), a guard of its own solvers past which its
energy is flat; SolverIPC does not, so a joint with bend stiffness keeps from
folding. Toward a fold the twist is undefined: past 168.6 degrees the twist's
terms and the bend's damping fade smoothly to 0, and a joint without bend
stiffness folds freely, as a chain of hinges does. The two differ only past
168.6 degrees at one joint.

With :meth:`~newton.solvers.SolverVBD.register_custom_attributes`,
``model.vbd.dahl_eps_max`` and ``dahl_tau`` both above 0 on a rod joint give its
bend and twist SolverVBD's Dahl hysteresis: a stress that lags the strain, so
that a rod bent and released keeps part of its bend. Within a step the stress is
linear in the strain, from its value and slope at the step's start; at its end
it follows the step's own change of strain, as in SolverVBD. A
:meth:`~newton.solvers.SolverIPC.reset` starts it anew, also per world. On other
joints the attributes are not read, as in SolverVBD.

``joint_q`` and ``joint_qd`` of a rod joint are a free joint's (the child's
anchor frame in the parent's), as :func:`newton.eval_ik` computes them, and
``joint_f`` is a world wrench at the child's centre of mass with its reaction on
the parent, as SolverVBD applies it. ``joint_enabled``, the gains, the rest and
the Dahl attributes are read at every step, also in graphs captured before. A
rod joint's terms are part of its child's ``BODY_PARENT_F``
(:ref:`ipc-body-observables`). The capsules are triangulated as every capsule
is, and the filters of ``add_rod`` (the joints' parents,
``junction_collision_filter``) leave their pairs out of contact.

The segments are affine bodies (:ref:`ipc-rigid-bodies`), and a joint measures
its bend and twist in its parent's frame, which a shear of the parent turns
against the parent's centreline. Under the default ``body_kappa``, lowered for
the step, a segment shears enough to relieve the joint it is the parent of: that
joint bends and twists about as if its stiffness were in series with
:math:`4 \kappa V = 100\, I_{max} / \Delta t^2`, about 2200 N m/rad for a
segment of 0.2 m, radius 0.05 m and 1.57 kg at :math:`\Delta t` = 1/60 s. A
joint to the world or to a kinematic body is not relieved. In Newton's
bend-stiffness sweep at that step (bend stiffness 50, 500 and 5000 N m/rad),
the tips rest 4 mm, 71 mm and 160 mm below SolverVBD's; with
``body_kappa=1e8``, libuipc's stiffness, within 1 mm. Pass a ``body_kappa`` for
which :math:`4 \kappa V` is far above the rods' bend and twist stiffness.

What it costs: the bend and twist of a joint take a projection of an 18 x 18
matrix in every Newton iteration, about 0.25 ms for one joint, 0.5 ms for 64 and
3.7 ms for 2000 on an RTX PRO 5000, where the rest of the frame of 2000 bodies
takes 3.5 ms per iteration.

.. _ipc-joint-terms:

Mimic joints, fixed tendons and joint friction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three inputs act on joint coordinates rather than on one joint's bodies: a mimic
relation, MuJoCo's fixed tendon and the friction of a joint. libuipc's joints
have none of them. The solver gives each to libuipc-warp as a potential of a
linear combination :math:`L = c_0 + \sum_t c_t q_t` of the coordinates of
revolute and prismatic joints (and of D6 joints of one axis), which are Newton's
``joint_q``.

- **A mimic relation**, :attr:`~newton.Model.joint_mimic_joint` with
  :attr:`~newton.Model.joint_mimic_coeffs`
  (:meth:`~newton.ModelBuilder.set_joint_mimic`), or a row of the deprecated
  ``constraint_mimic_*``, is the penalty :math:`\tfrac{K}{2} L^2` of
  :math:`L = q_f - c_1 q_r - c_0`, with a joint's :math:`K`. Like a joint it
  gives under a load: a torque :math:`\tau` on the follower moves it off the
  relation by :math:`\tau \Delta t^2 / K`. It acts while both joints, and its
  row, are enabled, and holds the multi-turn coordinates, so a gear follows its
  reference through any number of turns. A follower keeps its own drive, which
  adds to the relation, as in Newton's other solvers but Featherstone. MuJoCo's
  JOINT equality rows that the importers turned into mimic relations are held by
  them where a row repeats its relation: the same two joints and the same linear
  coefficients. Other JOINT rows are refused.
- **A fixed tendon of MuJoCo** (``model.mujoco.tendon_*``, which the MJCF and USD
  importers fill) is :math:`L = \sum_i c_i q_i` over its joints. Its
  ``stiffness`` is a spring toward its ``springlength``, with no force inside
  the length's range when it is one, as SolverMuJoCo reads it. Its ``damping``,
  ``armature`` and ``frictionloss`` act on :math:`L` as a joint's do on its
  coordinate. Its limit (``limited``, ``range``) is a joint's limit, with the
  stiffness ``tendon_limit_ke`` where it is given so, and otherwise the one
  Newton's MuJoCo solver reports for its ``solreflimit`` :math:`(T, \zeta)` and
  ``solimplimit``, :math:`1 / (T^2 \zeta^2 w)` with :math:`w` the tendon's
  inverse inertia at ``model.joint_q`` times one less the impedance's width;
  with ``enable_joint_limit_damping`` it is damped with :math:`2 / (T w)` past a
  bound. ``solreffriction`` and ``solimpfriction`` are not read: the friction is
  smoothed below ``eps_velocity`` instead. Where a tendon with a friction loss
  sets either to other than Newton's default, the solver warns once with
  :class:`~newton.NewtonWarning` when it is built.
- **The friction of a joint**, the ``joint_friction`` of a degree of freedom
  (MuJoCo's ``frictionloss``), is a force against the joint's motion of that
  magnitude: libuipc's smoothed friction on the step's increment of the
  coordinate. Below ``eps_velocity``, taken in rad/s for an angle, it is
  viscous, so a joint under a load below its friction creeps at about
  ``eps_velocity`` times the load over twice the friction. A ball joint takes it
  about each axis (:ref:`ipc-ball`), and a universal joint on each angle.

**Friction and the Newton tolerances.** The Newton loop ends when its update is
below ``newton_velocity_tol`` :math:`\Delta t` (for a rigid body's rotation,
``newton_transrate_tol`` :math:`\Delta t`). Within its smoothing band the
friction is stiff, :math:`2 F / (\epsilon_v \Delta t)` on the coordinate with
:math:`\epsilon_v` = ``eps_velocity``, so the Newton updates there are short. At
the default tolerances (0.05 m/s and 0.1 1/s, against ``eps_velocity`` 0.01) the
loop would end inside the band, and a joint loaded beyond its friction would
creep instead of sliding: a box on a hinge with a friction of 2 N m under 3 N m
would turn at 7.5e-3 rad/s instead of speeding up. A model with friction on any
joint (a ``joint_friction`` other than 0 on any degree of freedom, whatever the
joint's type) or on a tendon, fixed or spatial, therefore takes both tolerances
at 0.4 ``eps_velocity`` (4e-3 by default) where they are left at their defaults.
That is the largest at which a hinge or slider under 1.5 times its friction
slides at :math:`(f - F) / I` within 1% whatever its axis: 0.7 ``eps_velocity``
suffices along a coordinate axis, and a diagonal axis needs 0.4. A load nearer
the friction needs a lower tolerance: 1.2 times the friction needs 0.25
``eps_velocity``, 1.1 times needs 0.1, so no single value serves every load
above the friction. The cost is Newton iterations: a chain of hinged links with
friction falling next to a box resting on the ground takes 5.5 per step instead
of 4.0. The solver lowers the tolerances without a warning. A tolerance passed
to the constructor is kept, also when it equals the default, and one above 0.4
``eps_velocity`` gets a :class:`~newton.NewtonWarning`.

New coefficients of the mimic relations and tendons (``CONSTRAINT_PROPERTIES``,
``TENDON_PROPERTIES``), and the mimic penalties of new masses, take effect in
place at ``notify_model_changed``, also in graphs captured before; a change of
which joints they hold, or of which joints have friction, builds the solve
again. The terms' forces through each joint are part of its ``BODY_PARENT_F``
(:ref:`ipc-body-observables`).

Refused: actuators that drive a tendon and a tendon's limit ``margin`` other
than 0, of fixed and spatial tendons, a fixed tendon over a joint that is no
revolute or prismatic joint, a mimic relation of a joint of more than one
coordinate, and the friction of free and distance joints.
:ref:`ipc-spatial-tendons` refuses Newton's own tendons.

.. _ipc-spatial-tendons:

MuJoCo's spatial tendons
~~~~~~~~~~~~~~~~~~~~~~~~

A spatial tendon of MuJoCo is a cable whose length is that of a path: through
sites, around spheres and cylinders on its way, and over pulleys. It pulls on
every body it touches according to its length, as a robot hand's finger tendons
do. The solver takes every tendon of type 1 of ``model.mujoco``
(:func:`newton.solvers.mujoco.add_tendon_spatial`, and the MJCF and USD
importers) as libuipc-warp's spatial tendons, whose length and moment arms are
MuJoCo's.

- **The path** is MuJoCo's: a site of a body, or of the world for a site of
  body -1; a sphere or cylinder geom between two sites, which the path wraps
  around where the straight line between them would cut the geom's circle,
  around the shorter side, or with a ``sidesite`` on the side site's side, the
  long way round if need be; and a pulley, whose ``divisor`` divides the length
  of the branch after it.
- **The coefficients** are a fixed tendon's (:ref:`ipc-joint-terms`): the
  ``stiffness`` toward the ``springlength`` (the length at the model's pose for
  -1), the ``damping``, ``armature`` and ``frictionloss``, and the limit with
  its stiffness and damping. The inverse inertia :math:`w` that a limit's
  ``solreflimit`` needs is the tendon's :math:`J H^{-1} J^T` at the model's
  pose, MuJoCo's ``tendon_invweight0``, with :math:`J` its moment arms in joint
  space and :math:`H` the joint-space inertia. A tendon's friction loss lowers
  the default Newton tolerances as a joint's friction does.
- **The forces.** A tendon under tension pulls each site toward its neighbours
  along the path, and a geom's body at the points where the path touches the
  geom. Every force is an energy of the step, so the step is implicit in all of
  them. A tendon whose sites and geoms are all of the world has a constant
  length and no force, as in MuJoCo.

On the same models the lengths and moment arms agree with MuJoCo's to 2e-7 m,
and the inverse weights with MuJoCo's ``tendon_invweight0`` to 3e-7 of
themselves, over random poses of sphere and cylinder wraps with and without side
sites, pulleys and sites of the world; a tendon spring that holds a hinged arm
up against gravity rests it at MuJoCo's angle to 1e-6 rad. MuJoCo's limit and
friction loss are soft constraints of its own solver; here the limit is a
joint's cubic limit (:ref:`ipc-joints`) and the friction libuipc's smoothed
friction, as for a fixed tendon.

New coefficients (``TENDON_PROPERTIES``), and the limits' stiffness that new
masses give (``BODY_INERTIAL_PROPERTIES``), take effect in place at
``notify_model_changed``, also in graphs captured before; a change of a path,
or of a site or geom it names (``SHAPE_PROPERTIES``), builds the solve again.

Refused (:class:`NotImplementedError`): Newton's own tendons
(:meth:`~newton.ModelBuilder.add_tendon`, routed over anchors, pinholes and
rollers); a spatial tendon that wraps a geom other than a sphere or a cylinder,
whose path MuJoCo's compiler refuses (a geom not between two sites, a site whose
neighbours are pulleys only, two pulleys in a row, a pulley at the end or with a
divisor that is not positive), or whose side site lies inside the geom it wraps
at the model's pose: there MuJoCo wraps inside the geom, and SolverIPC wraps
around its outside only. A side site of another body that comes inside the geom
later keeps the outside wrap.

.. _ipc-refused-joints:

Refused joints
~~~~~~~~~~~~~~

Refused with :class:`NotImplementedError` when the solver is built; the message
names the joint and its degree of freedom, and says what to change:

- D6 joints other than those above, and D6 joints of three angular axes that
  are not orthonormal;
- a distance joint whose lower bound is above its upper;
- drives, springs, damping, a target mode other than ``NONE`` and limits on free
  joints, and on D6 joints that become ball or free joints;
- limits on ball joints, which Newton does not define;
- a negative ``joint_armature``, and one other than 0 on a free joint or a D6
  joint that becomes a ball joint;
- ``joint_friction`` other than 0 on free and distance joints and on D6 joints
  that become free or ball joints;
- a negative ``joint_effort_limit`` on a driven degree of freedom, and with
  ``enable_joint_effort_limit=False`` one below 1e6;
- on a rod joint's degrees of freedom, as SolverKamino refuses them: joint
  friction, armature, limits, a ``joint_stiffness`` or ``joint_damping`` other
  than 0, a target mode other than ``NONE``, and a negative ``joint_target_ke``
  or ``joint_target_kd``; and a rod joint not of the layout
  :meth:`~newton.ModelBuilder.add_joint_rod` makes (:ref:`ipc-rods`).

A ``joint_velocity_limit`` below Newton's default 1e6 is not enforced: the
solver warns once, with :class:`~newton.NewtonWarning`, and runs. Newton's URDF
importer writes every ``<limit velocity>`` into it, so refusing it would refuse
nearly every URDF robot.

.. _ipc-kinematic-bodies:

Kinematic bodies
----------------

A kinematic body (:attr:`~newton.BodyFlags.KINEMATIC`, or of mass 0) follows the
poses the caller writes into ``state_in``; forces and contacts do not change its
motion. The solver moves it in one of two ways, chosen by ``kinematic_motion``:

- ``"aim"``: the body moves inside the step toward the pose ``state_in``
  prescribes. CCD sweeps the motion, so surfaces stay apart, and friction acts
  on it: a platform sliding under a box carries the box.
- ``"jump"``: before the step the solver writes the pose of ``state_in`` into a
  fixed body. CCD sweeps the jump first, and a jump that meets a surface fails
  the step. Friction does not act on the jump.

The default, ``None``, picks ``"aim"`` on the barrier pipeline when every
kinematic body with mass and colliding shapes can aim, and ``"jump"``
otherwise: under ``contact_constitution="al-ipc"``, with independent worlds when
such a body belongs to the global world, and when such a body has a singular
inertia. It picks without a warning (the choice is logged at the debug level),
and :attr:`~newton.solvers.SolverIPC.kinematic_motion` gives the choice. The aim
is the default where it works because it is the motion IPC can guarantee:
surfaces stay apart and friction sees it. An explicit ``"aim"`` where a body
cannot aim is refused, naming the body.

**Aims.** A body that aims is an unknown of the step, pulled toward its aim by
libuipc's soft transform constraint with the strength ratio :math:`\eta` =
``kinematic_strength_ratio`` (default 100, the value libuipc's motors use). Its
aim is the pose of ``state_in`` advanced over the step by its twist
(``body_qd``). What the caller writes sets the aim: ``body_q`` and ``body_qd`` at
every step give the next pose of that motion, ``body_q`` alone the written pose,
and nothing lets the body move on at its twist. The pull is soft: a body that
contact blocks stops short of its aim, and under a load :math:`F` a body of mass
:math:`m` lags its aim by about :math:`F \Delta t^2 / ((1 + \eta) m)`, 2.8e-5 m
for 10 N on 1 kg at 1/60 s. ``state_out`` holds the pose the body reached and
the twist of ``state_in``, which the step passes through, as Newton's other
solvers do for kinematic bodies. Each step starts from the pose the body
reached, so the move from there to the aim is swept, even when the caller writes
a pose far away. The first step after the solver is built, and the first step
after a :meth:`~newton.solvers.SolverIPC.reset` that selects the body's world,
take the pose of ``state_in`` without a sweep: use them to place a body
elsewhere. A first step that the start check refuses does not count. A body that
``notify_model_changed`` makes kinematic does not move on at the twist the
solver wrote for it while it was dynamic: it stays where it is until the caller
writes a twist, as in Newton's other solvers.

**Jumps.** Kinematic particles always jump, because a soft aim would let a pinned
vertex sag. So do kinematic bodies without colliding shapes, and those of mass
0, such as a link of density 0 that anchors joints, since the aim has no mass to
pull. Before the step the solver sweeps every jump with CCD, from where the last
step left the body to its new pose, where the model has something to collide
with. A jump that meets a surface fails the step with ``INVALID_CCD_STEP`` before
the solve runs: an eager step raises, and a captured step sets the bit. Without
the sweep, a box on a platform that rose at 0.5 m/s, 120 steps per second, fell
through it. A jump that meets no surface is taken, but it is not part of the
step's motion: the barrier has one step to push away what the jump brought
close. The first step after the solver is built, the step after a ``reset`` that
selects the body's world and the step after a failed step take the new poses
without a sweep. Under AL-IPC the step's own collision-free updates carry the
jump and push what it meets (:ref:`ipc-al-ipc`).

Two shapes that are each kinematic or static never collide, as Newton's
collision pipeline can filter such pairs
(``include_static_kinematic_pairs=False``); the solver checks its pairs against
:attr:`~newton.Model.shape_contact_pairs` without them. Newton's test suite gives
the same results with either mode.

.. _ipc-attachments:

Attachments
-----------

A particle-body attachment
(:meth:`~newton.ModelBuilder.add_attachment_particle_body`) is libuipc's soft
vertex stitch of rest length 0 between the particle and an anchor of the body
at ``attachment_particle_body_body_point`` in Newton's body frame. The anchor
moves with the body and is no contact primitive, as the interior vertices of an
affine-body mesh are in libuipc; attachments at one point of one body share it.
Its energy in the step is :math:`\tfrac{1}{2} k \Delta t^2 |x_p - x_a|^2`, the
spring :math:`\tfrac{1}{2} k |x_p - x_a|^2` of implicit Euler with :math:`k` =
``attachment_particle_body_stiffness``: the particle and the body take equal and
opposite forces, the body with the torque of the anchor's offset. A disabled
attachment (``attachment_particle_body_enabled``) takes none. The stiffness and
the enabled flags are read at every step, so a captured step takes new values at
its next replay.

libuipc has no hard constraint, so a hard attachment (infinite stiffness) takes
the penalty of a ball joint: :math:`K/\Delta t^2` with :math:`K` =
``joint_strength_ratio`` times the masses of the particle, if it is not fixed,
and of the body, if it is dynamic. It holds a load :math:`F` within
:math:`F \Delta t^2 / K`. Its damping acts on the vector between particle and
anchor (:ref:`ipc-damping`).

An attached particle does not collide with the body it is attached to, whether
or not the attachment is enabled, and neither do the edges and triangles of
cloth or a soft body that hold it, since libuipc lets a primitive collide only
where each of its vertices may. So a particle may be attached at a point on the
body's surface, which the start check would otherwise refuse as a start in
contact. The particle collides with every other body and shape. Newton's other
solvers keep this contact, which does them no harm; in IPC an anchor on the
surface would be a pair at distance 0.

.. _ipc-worlds:

Worlds
------

Particles, bodies and colliding shapes in more than one local world are
simulated as independent worlds, Newton's meaning of worlds: every world takes
the step it would take alone, with its own Newton iterations, line searches,
convergence test, linear solves and status. libuipc solves the subscenes of a
scene together; the independent worlds are libuipc-warp's own. Every local world
is a libuipc subscene of its own and the global entities (world -1) one more,
which meets every world: no contact pair joins two worlds, also where the worlds
overlap in space, and a global ground or static shape acts in every world.

The worlds share one CUDA graph, one collision detection and one set of contact
lists. A step runs as many Newton iterations as its slowest world, and a world
that has converged waits where it is while the others iterate. A world takes the
Newton iterations of the same world alone and agrees with it to the rounding of
the solver's sums, which add in another order; where a step carries rounding
far, as a stiff landing does at the default tolerances, the two part as far as a
run alone parts from one started a rounding error away.

**Failures.** A world that fails (a non-finite value, a contact pair at or inside
its thickness, a time of impact that is not positive) fails alone: it is put
back at its state at the step's start, its friction discarded and its BDF2
history started anew, and the other worlds take their step. ``STATUS`` and
``SOLVER_RESIDUAL`` hold every world's own bits and residual at its entry; the
entries of the global world and of worlds that hold no unknowns stay 0. A step
that is not captured and fails in some worlds writes ``state_out`` (those worlds
at the step's start, the others at their step) and raises
:class:`RuntimeError` naming the worlds; ``check_status`` names them too. Every
world has a quota of the shared contact lists, sized as for that world alone: a
world whose contact asks for more fails alone in the same way, while the others
keep every pair. An eager step then grows that world's quota and solves the step
again; a captured step sets ``OVERFLOW`` in that world's entry alone, and
:meth:`~newton.solvers.SolverIPC.grow` grows its quota.

**Resets and options.** A reset with a ``world_mask``, or a new state written for
one world, discards that world's friction and starts its BDF2 history anew, and
leaves the other worlds as they were, bit for bit; the mask's global entry
selects nothing. :meth:`~newton.solvers.SolverIPC.frame_stats` gives the counts
of the step's loops: the Newton iterations of its slowest world. With
``adaptive_kappa=True`` every world takes libuipc's stiffness range of the same
world alone, from the mass and extent of its own vertices and the global ones
and from the model's ``d_hat`` (:ref:`ipc-d-hat`);
``contact_stiffness(dt)`` then gives one value per world. MAS preconditions every
world as the same world alone (:ref:`ipc-mas`), and AL-IPC runs every world's
augmented-Lagrangian loop as the same world alone (:ref:`ipc-al-ipc`). The
engine's Newton exit check is not used.

**What is refused.** Global entities must not move, because every world would
push them and so join the worlds. The solver refuses a global free particle (one
with mass that is :attr:`~newton.ParticleFlags.ACTIVE`), a global dynamic body, a
global proxy of a coupled solver, and a global kinematic body that an explicit
``kinematic_motion="aim"`` would move; the default makes such a body jump, as a
fixed body. Newton's implicit MPM solver refuses global colliders backed by
dynamic bodies with ``separate_worlds`` in the same way. A triangle,
tetrahedron or spring that joins two worlds is refused too. Global static
shapes and planes, kinematic particles and kinematic bodies that jump act in
every world. Each refusal names ``coupled_worlds=True``, which takes them.

**coupled_worlds=True** solves the worlds as one, with the same subscenes: one
step length (the smallest time of impact over all worlds, and a line search on
the sum of their energies), one convergence test, one PCG solve and one set of
contact lists. A world's result then depends on the other worlds through the
iteration counts and step lengths, and a failure or an overflow in one world
fails the step in every world. Copies of one world take the iterations of one
copy and follow its run alone to the rounding of the shared sums. Global
particles and bodies may move, as unknowns of the one solve that meet every
world. A reset of one world, or a new state written for it, still discards only
that world's friction and history; a reset of the global entities discards the
friction of every pair with a global entity. With ``adaptive_kappa`` the
stiffness range is that of the whole scene. Use it for a model whose global
entities move; otherwise the independent worlds give every world its own result.

A model with content in one local world is solved as one world either way. The
engine takes up to 32,768 subscenes, and its contact elements times the subscene
count, rounded up to a power of two, must stay below :math:`2^{31}`.

.. _ipc-bdf2:

BDF2
----

``integrator="bdf2"`` is libuipc's second-order backward differentiation
formula. The step predicts

.. math::

    \tilde x = \tfrac{4}{3} x_n - \tfrac{1}{3} x_{n-1}
        + \tfrac{2}{9} \Delta t (4 v_n - v_{n-1}) + \tfrac{4}{9} \Delta t^2 a,

weighs the inertia :math:`\tfrac{9}{4}` times as much against the potentials as
implicit Euler does, and ends at
:math:`v_{n+1} = (3 x_{n+1} - 4 x_n + x_{n-1}) / (2 \Delta t)`. It damps far less
than implicit Euler: Newton's energy-conservation test passes under it, where
implicit Euler loses 0.79% of the pendulum's energy against the test's 0.5%. The
price is that contact keeps more energy and bodies take longer to settle: a
cube dropped onto the ground in steps of 1/1800 s rebounds at 0.93 of its impact
speed under BDF2 and at 0.47 under implicit Euler, on libuipc as here.

The history :math:`(x_{n-1}, v_{n-1})` is the solver's, not the state's. A step
from the state the solver wrote last continues it; every other step is an
implicit Euler step whose start becomes the history: the first step, a step from
a state the solver did not write last, the step after
:meth:`~newton.solvers.SolverIPC.reset`, and a step after a change of gravity.
A step from a new state therefore applies the whole impulse of its forces, as in
Newton's other solvers, which are one-step methods. libuipc instead starts the
history from rest, without acceleration: a constant force from rest then gets
two thirds of its impulse in the first step, and the velocity lags half a step's
impulse behind for good. libuipc-warp's array path offers both starts, and the
solver takes the implicit Euler start.

A change of gravity (:meth:`~newton.solvers.SolverIPC.notify_model_changed` with
``MODEL_PROPERTIES``) starts the history anew in the worlds whose gravity
changed: the next step compares every world's gravity with the step before's, on
the device, also inside a captured graph. Without it the velocity would lag
:math:`\Delta t\,\Delta g / 2` behind for good. Gravity is a parameter of the
model that jumps between steps; the external forces and the drives are inputs of
every step, and BDF2 takes them as they come, as libuipc does: a force that
jumps within a history leaves the velocity half a step's impulse of the jump
behind. With several worlds a new state in one world, or its reset, makes the
step an implicit Euler step of that world alone. As in libuipc, the coefficients
assume one ``dt`` for consecutive steps.

Joint drives, springs and damping take BDF2's velocity at the end of the step
(:ref:`ipc-joints`). Rigid bodies take libuipc's straight-line step under BDF2
(:ref:`ipc-rigid-bodies`). Proxies of a coupled solver are refused with BDF2,
since the coupler's write of a proxy's state at every step would restart its
history every time.

.. _ipc-al-ipc:

AL-IPC contact
--------------

``contact_constitution="al-ipc"`` is libuipc's augmented-Lagrangian contact,
the method of Zheng, Luo and Li, "Robust and Efficient Penetration-Free
Elastodynamics without Barriers". Each Newton iteration linearizes the
constraint :math:`d \ge \hat d + \text{thickness}` of the active contact pairs,
with a penalty that follows the masses and multipliers that converge to the
contact forces, and a collision-free sweep moves the configuration toward the
solution. Contact rests at ``d_hat`` beyond the contact thickness, where the
barrier rests inside ``d_hat``: two free particles that meet are held ``d_hat``
beyond the sum of their radii.

**Masses.** A pair's penalty scales with the smallest mass of its vertices. A
vertex without mass never moves (the vertices of static mesh shapes, which have
no mass in Newton, of particles without mass and of bodies without mass), and it
adds no penalty of its own: its pairs take the penalty of their side that moves,
so a box resting on a static mesh takes the penalty of the box. A change of the
bodies' masses builds the solve again, since the penalty follows the masses the
solve was built with.

**Rebounds.** A pair joins the active set only when a sweep finds its two
primitives touching, and the step that solves its constraint ends with the gap at
``d_hat``: a body whose step began a gap :math:`g` from a surface leaves it at
about :math:`(\hat d - g) / \Delta t`, whatever its speed of impact. This is the
method's own, in the paper, the authors' code and libuipc alike, and it gains
energy where ``d_hat`` exceeds the distance a body travels in a step: a cube
dropped 0.4 m onto the ground in steps of 1/1800 s leaves at 4.4 times its impact
speed of 2.8 m/s with ``d_hat`` 1 cm, at 0.7 times with 1 mm, and in steps of
1/100 s at 0.33 times with 1 cm. Keep ``d_hat`` well below the distance a body
moves in a step.

**Resting contact.** libuipc's default decay of an opened pair's weight, 0.3,
keeps an opened pair in the active set for three outer updates and drops it at
the fourth, so a
resting contact that opens for a moment joins again at the surface and is pushed
out again: a stack of two 0.1 m boxes in steps of 1/100 s keeps rocking with
``d_hat`` 5 mm or 2 mm, in most runs until its upper box falls off, on libuipc as
here. SolverIPC takes the paper's decay factor 0.9 instead, which keeps a pair
for 43 updates: such a stack came to rest in eight runs of eight, where 0.3
brought it to rest in none. The libuipc-warp engine keeps libuipc's 0.3 for
libuipc's scenes.

**Kinematic bodies.** A kinematic body moved between steps is swept from where
the last step left it to its pose in ``state_in``, the method's moving
boundaries, which libuipc's own frame lacks: what it meets joins the active set
and is pushed, and one that contact holds back ends the step short of its pose
and goes on toward it in the next. The first step, the step after a ``reset``
that selects its world and the step after a failed one place it at its pose
without a sweep. Kinematic bodies do not aim under AL-IPC: its frame checks
animated aims with a test of its own that the solver does not run.

**Convergence.** The step ends on AL-IPC's own criterion, the paper's progress
weight, which has no Newton residual: ``SOLVER_RESIDUAL`` is not offered. Without
contact a step takes ``semi_implicit_k_min`` Newton iterations (the paper's
:math:`K_{min}`), and one with ``semi_implicit=False``, which makes AL-IPC's
steps the least converged: with libuipc's body stiffness a free cube spinning at
1.08 rad/s keeps 6% of its spin after 100 steps of 0.01 s. A larger
``semi_implicit_k_min`` converges the steps further.

**Worlds.** With independent worlds every world runs the augmented-Lagrangian
loop of the same world alone: its own Newton iterations, line searches and outer
updates, its own active pairs and quota of them. A world that fails is put back
with the active pairs it started the step with. A kinematic body of the global
world has one pose for every world, which no world's contact may hold back: it
jumps, swept before the step, and a world whose surfaces the move meets fails the
step with ``INVALID_CCD_STEP``. A kinematic body of a local world is carried by
that world's sweeps, as alone. This costs time: with 64 worlds a step takes 1.27
to 2.29 times as long as the same worlds solved as one AL-IPC solve, since every
step waits for its slowest world and every world's PCG solves to its own
tolerance.

**Not offered under AL-IPC**: ``kappa`` and ``adaptive_kappa`` (no barrier),
``contact_damping``, collision meshes with a surface velocity, the contact
output ``CONTACT_F``, and proxies of a coupled solver.

.. _ipc-mas:

MAS
---

``fem_preconditioner="mas"`` is libuipc's multilevel additive Schwarz
preconditioner on the vertices of particles, cloth and tetrahedra; bodies keep
their 12 x 12 blocks, and particles on no element their 3 x 3 blocks. It inverts
the system exactly on clusters of up to 16 vertices and corrects between them on
coarser levels made of neighbouring clusters. libuipc takes the clusters from
METIS; libuipc-warp's array path makes them itself, by recursive inertial
bisection of every connected cloth, soft body and spring network at its rest
positions, the same on every machine and without libuipc. The preconditioner is
libuipc's on another partition: the PCG iterations differ from libuipc's, and
the result is the same to the tolerance of the linear solve.

With independent worlds every world is preconditioned as the same world alone,
so it takes the PCG iterations of its run alone: libuipc-warp builds the levels
once per world, from that world's own particles.

.. _ipc-contact-output:

Contact output
--------------

Request ``SolverBase.ObservableKind.CONTACT_F`` with
:meth:`~newton.solvers.SolverIPC.observables`, passing the
:class:`~newton.Contacts` that sizes ``contact_f``, and pass the container and a
``Contacts`` to :meth:`~newton.solvers.SolverIPC.step`. The step writes its own
contacts into that buffer and their forces into ``contact_f``, as a native
collision backend does (SolverMuJoCo with its own collision detection). Do not
run ``CollisionPipeline.collide`` into the buffer: the step overwrites its rows.

**Rows.** Each contact term of the step with a force is one row: a
point-triangle, edge-edge, point-edge, point-point or vertex and half-plane pair
of the final active set (its barrier) or of the step's friction pairs (its
lagged friction). A pair in both gives two rows, one normal and one tangential.
A term between two shapes is a rigid row. A term between a particle feature (a
particle, or an edge or a face of cloth or tetrahedra) and a shape is a soft row,
with the particle, edge or face record of
:attr:`~newton.Contacts.soft_contact_indices` and the weights of its contact
point. So one shape pair in contact is several rows: two resting boxes give a row
for every term between their primitives within ``d_hat``. The rows carry the
distribution of the contact over the surface, which
:class:`~newton.sensors.SensorContact` and Newton's force-based conveyor read; a
consumer that sums the rows of a body or a shape pair, as ``SensorContact``
does, gets the totals. A consumer that expects one row per shape pair does not:
Newton's ``test_native_solvers_follow_step_contacts`` counts one row for a ball
on the ground, a property of its analytic sphere-plane contact, and SolverIPC's
triangulated ball gives several. Terms between particle features (cloth
self-contact, particles among themselves) are not exported, since Newton's
:class:`~newton.Contacts` has no row of that kind.

**Forces.** The forces are those at the end of the step: minus the gradient of
the term's energy in the step's incremental potential at the final positions,
over :math:`\Delta t^2`, which the step balanced to the Newton tolerance. A box
resting on the ground reports its weight. On a body of mass ``m`` whose step
ended on the displacement test, the forces can be off by up to about
``m * newton_velocity_tol / dt``: a box creeping down a ramp under friction
reports its weight within 1.2% at the default tolerances, and within 1e-4 with
``newton_velocity_tol=1e-4``, ``linear_tol_rate=1e-10`` and
``semi_implicit=False``. A rigid row's ``contact_f`` is the force on the body of
shape 0 by the body of shape 1, the sum of the forces on its vertices, with their
torque about that body's centre of mass at the end of the step (about the world
origin for a static shape), in the world frame. A soft row's is the force on the
shape's body by the particle feature, with its torque about that body's centre
of mass.

**Geometry.** The points are written in the body frames of ``state_out``, so pass
``state_out`` to a consumer such as :class:`~newton.sensors.SensorContact`. Each
side's point is the force-weighted mean of its vertices, the closest point of
its primitive on the triangulated surface. The normal points from shape 0 toward
shape 1, a soft row's from the shape toward the particle feature: against the
force on shape 0 for a barrier term, and from shape 0's point to shape 1's for a
friction term, whose force is tangential. The margins are the shapes'
``shape_margin``, and the offsets the normal times each margin toward the other
side, as Newton's collision pipeline writes them.
``rigid_contact_surface_velocity`` (when allocated) and ``soft_contact_body_vel``
are zero, and ``soft_contact_rigid_indices`` is ``-1``.

**Capacity.** The counts are true counts. Rows past the buffer's capacity are
dropped, as Newton's collision pipeline drops them, and a ``rigid_contact_count``
or ``soft_contact_count`` above the capacity tells it; a buffer of any size is
taken. :meth:`~newton.solvers.SolverIPC.get_max_contact_count` and
:meth:`~newton.solvers.SolverIPC.get_max_soft_contact_count` give capacities that
drop none: every slot of the solver's contact lists, a worst case several times
the usual count. The order of the rows changes from run to run (atomic counters);
their set and values do not. A step that failed writes no rows. With several
worlds the rows of every world share the buffer; with independent worlds a step
in which some worlds failed writes the rows of the others only. Not offered with
``contact_constitution="al-ipc"``. The output costs, only when requested, a pass
over the step's contact terms and one over its rows, and 64 bytes per slot of the
contact lists.

.. _ipc-body-observables:

Parent wrenches and accelerations
---------------------------------

Request ``SolverBase.ObservableKind.BODY_PARENT_F`` or ``BODY_QDD`` with
:meth:`~newton.solvers.SolverIPC.observables` and pass the container to
:meth:`~newton.solvers.SolverIPC.step`. The step writes, per body, the wrench
its parent joint passes to it into ``body_parent_f`` and its acceleration into
``body_qdd``, on the device, also during graph replay. Both are in the world
frame, as in Newton's other solvers: the force or the linear acceleration of the
centre of mass first, then the torque about the centre of mass or the angular
acceleration. :class:`~newton.sensors.SensorIMU` reads ``body_qdd``.

``body_parent_f`` is the force of the parent joint's terms at the end of the step,
with the ``joint_f`` the solver applies to the child. The terms are those of the
libuipc joint the joint maps onto (its penalty, drive and limit), of a universal
joint (its anchor, lock and the terms of its angles), the stitch of a distance
joint, and the ball terms, mimic relations, fixed tendons and friction through
the joint; a spatial tendon pulls on the bodies themselves, as a contact does.
A joint is a penalty (:ref:`ipc-joints`), so this is the force that holds the
body where the step moved it: the body's momentum balance, to the Newton
tolerance, not a rigid joint's force at every step. A 0.2 m box released 1 m
below a hinge pulls on it with 57% of its weight in the first step of 5 ms with
the default body stiffness, and with 94% with ``body_kappa=1e8``, within 1e-4 of
the weight from the sixth step on: in the first steps the hinge also pulls on
the box's affine stretch. Newton's ``test_parent_force``, which expects the
weight from the first step, fails with SolverIPC for this reason.

A body's parent joint is the joint of an articulation whose child it is, or else
the first joint whose child it is: a loop joint is no body's parent joint, a free
joint passes its ``joint_f`` only, and a body that is no joint's child gets zero.
The records of the joints' forces take about 2.8 KB per joint, allocated when a
container first requests ``BODY_PARENT_F``, and a step that requests it
evaluates the joints' forces once more.

``body_qdd`` is the finite difference of the twists over the step,
``(qd_out - qd_in) / dt``, as SolverKamino computes it: for the linear part the
velocity of the centre of mass, for the angular part the angular velocity, which
under ``body_rotation="rigid"`` is the one that carries the angular momentum.
``state_in`` may be ``state_out``.

With independent worlds a world whose step failed gets zero in both, and a step
that raises leaves no result in them. Both are offered under
``contact_constitution="al-ipc"`` too, where the joint's force balances the
step's motion as far as AL-IPC's step converges.

.. _ipc-coupling:

Coupling
--------

The solver is a :class:`~newton.solvers.experimental.coupled.CouplingInterface`:
an entry of Newton's coupled solvers (``SolverCoupled``, ``SolverCoupledProxy``,
``SolverCoupledADMM``), built on the entry's ``ModelView``. It reads a view as a
model, and three things that only a view holds take their meaning from the
coupler:

- a body of positive mass whose inverse mass is 0, which the view makes of a
  body another entry owns, is a kinematic body: a fixed affine body whose pose
  follows ``state_in``, and whose shapes, which the view hides, collide with
  nothing;
- a triangle, tetrahedron or spring whose particles are all disabled (mass 0
  and not :attr:`~newton.ParticleFlags.ACTIVE`), as the view disables another
  entry's particles, is that entry's: not simulated here, its particles fixed
  points that collide with nothing;
- the view's :attr:`~newton.Model.shape_contact_pairs`, which the coupler
  filters, is the set the solver's rules give (:ref:`ipc-contact`).

An entry therefore steps its own content as the solver alone does, to the
rounding of its sums. A particle-body attachment whose particle and body the
entry owns is in its view, and the solver applies it: ``SolverCoupledADMM``
leaves such an attachment to the entry, a hard one too, as it does for
SolverVBD, since the solver reports
``coupling_supports_particle_body_attachments()``.

**Proxies.** A proxy body or particle (:attr:`~newton.BodyFlags.PROXY`,
:attr:`~newton.ParticleFlags.PROXY`), another entry's object that this entry
moves with the view's scaled mass, is an unknown of the step. It does not collide
with another proxy, with a static shape or with a body of inverse mass 0, as the
interface's default ``coupling_prepare_proxy_contacts`` drops Newton's contacts
of those pairs. The coupler writes a proxy's pose and twist at every step. Unlike
any other new state, that write keeps the friction of the proxy's contacts, where
libuipc discards friction after any write of a state: it continues the same
motion, and a proxy would otherwise never have friction. The solver sweeps a
proxy's new pose from where the last step left it, as it sweeps a kinematic
jump, and a source solver that moves a proxy into this entry's content fails the
step with ``INVALID_CCD_STEP``, because IPC does not take a state in which
surfaces cross.

**Feedback.** A proxy's feedback is the force of the step's contact terms on it
at the end of the step, a wrench about its centre of mass for a body
(``coupling_harvest_proxy_wrenches``,
``coupling_harvest_proxy_particle_forces``), as SolverVBD harvests it, rather
than its change of momentum. The barrier's force rises steeply as surfaces
approach, so the feedback is only as accurate as the step's convergence. An entry
with proxies therefore takes converged steps by default: ``newton_velocity_tol``
5e-4 m/s, ``newton_transrate_tol`` 1e-3 1/s and ``semi_implicit=False``. These
keep an impact's feedback within 1% of the converged feedback, measured step by
step for a box and a particle hitting a box at 0.5 and 2 m/s with steps of 1/60
and 1/240 s, at about twice the Newton iterations; at the defaults without
proxies an impact's feedback was up to 27 times the converged one. Pass the three
arguments to choose other values. The harvest reads the contact records of the
last step. The solver counts its steps on the device and updates the records
once per step, so a harvest outside the graph that holds the step, after a
replay, reads the records of that replay.

**Lagged and staggered passes.** With ``SolverCoupledProxy``'s lagged transfer,
its default, a proxy resting on this entry's content does not come to rest with
one pass per step: the feedback of a contact this stiff reaches the source one
step late and keeps the proxy bouncing, though its average holds the proxy up.
Several lagged passes per step converge: four bring a box resting on a cloth to
rest with its weight as feedback within 0.1%. Use several lagged passes. Several
staggered passes (``mode="staggered"``) alternate, since each pass's feedback
undoes the motion that made it, unless the coupler relaxes them
(``proxy_relaxation=0.5`` settles them in one pass). How a coupler relaxes its
passes by default is the coupler's choice. Proxies are refused with
``integrator="bdf2"``, whose history the coupler's writes would leave behind, and
with ``contact_constitution="al-ipc"``, which neither sweeps a proxy's synced
pose into its active set nor exports its contact forces. The effective masses,
gravity and the rewinds of the proxies take the interface's defaults.

**Restarted passes.** A coupler that solves a step more than once (a
``SolverCoupledProxy`` or ``SolverCoupledADMM`` with more than one iteration)
starts every pass after the first from the step's start. The solver keeps that
start on the device: libuipc-warp's frame restart (the float64 positions and
velocities, the BDF2 history, the active contact set the step takes its friction
from, the joints' angles) and what the solver carries from step to step. The
step after a coupler's write of the input state copies the start, and the step
after a restart's notification copies it back, inside a captured graph too.
Every pass therefore starts from the same float64 state with the same friction,
and a pass with the inputs of the one before repeats it, bit for bit where a step
repeats itself. With independent worlds every world starts again, also one that
failed in the pass before. A solver whose coupler does not restart allocates none
of this.

**Status.** The coupler steps its entries without observables, so the solver as
an entry keeps the status bits of its steps, captured or not, with independent
worlds a word for every world. :meth:`~newton.solvers.SolverIPC.check_status`
without a container reports them, naming the worlds, and clears them; a restarted
pass replaces the bits of the pass before it, so a coupled step reports its last
pass. A reset through the coupler is :meth:`~newton.solvers.SolverIPC.reset`
with the coupler's world mask.

.. _ipc-refused:

Refused content
---------------

The solver refuses what it cannot simulate, when it is built, rather than ignore
it. :class:`NotImplementedError`, with a message that names the item and says
what to change:

- the CPU device: libuipc-warp runs on CUDA devices only;
- gradients (``model.requires_grad``): the solver computes none;
- body variants (:meth:`~newton.ModelBuilder.add_body_variants`): the solver
  triangulates the shapes when it builds the solve and finds its own contacts,
  so it cannot follow a variant row's shape;
- muscles, Newton's own tendons, and the tendons and mimic relations that
  :ref:`ipc-joint-terms` and :ref:`ipc-spatial-tendons` list;
- enabled MuJoCo equality rows that no loop joint or mimic relation holds
  (:ref:`ipc-joints`);
- triangle or tetrahedron activations other than 0, which the solver does not
  read; a step that is not captured also checks those of its
  :class:`~newton.Control` (a captured step cannot raise);
- the joints and joint properties :ref:`ipc-refused-joints` lists;
- springs of a negative rest length or stiffness or that join a particle to
  itself, and a negative particle-body attachment stiffness;
- a damping, drag or lift :ref:`ipc-damping` refuses;
- shape types without a triangulation, such as Gaussian shapes, and a plane
  on a body: libuipc's half planes do not move;
- with independent worlds, a moving global entity and an element between two
  worlds (:ref:`ipc-worlds`);
- a :attr:`~newton.Model.shape_contact_pairs` array assigned to the model,
  unless it is the set of pairs the solver's rules give, as a coupled solver's
  view assigns it;
- more contact elements times worlds than the engine takes (:ref:`ipc-worlds`);
- under AL-IPC, what :ref:`ipc-al-ipc` lists; with proxies, BDF2 and AL-IPC.

:class:`ValueError` for an argument out of range (:ref:`ipc-arguments`), for a
start in contact (:ref:`ipc-start`), for a hinge row whose triangles run its
edge the same way (:ref:`ipc-cloth`), and for a dynamic body whose inertia is
not positive definite. A finite plane shape is taken as an infinite half space,
with a :class:`~newton.NewtonWarning`.

.. _ipc-not-read:

Inputs not read
---------------

The solver reads every input of Newton's model, state and control that its
features use. It takes these without reading them:

- the penalty contact parameters of shapes and particles: ``ke``, ``kf``,
  ``ka``, ``particle_cohesion`` and ``particle_adhesion``, since ``kappa`` holds
  for every pair, and ``kd`` but with ``contact_damping="material"``;
- restitution (``shape_material_restitution``, ``soft_contact_restitution``),
  static friction (``shape_material_mu_static``), and torsional and rolling
  friction (``shape_material_mu_torsional``, ``shape_material_mu_rolling``): its
  contact has one friction coefficient and no restitution. The solver warns once,
  with :class:`~newton.NewtonWarning` when it is built, where a colliding shape or
  the particles set one of them to other than Newton's default; MJCF's default
  friction holds Newton's defaults and draws no warning;
- ``shape_gap``: the solver's ``d_hat`` is its detection distance;
- hydroelastic contact: the ``HYDROELASTIC`` shape flag and
  ``shape_material_kh``; such shapes take the barrier as every shape does;
- :attr:`~newton.Model.particle_radius` of cloth and tetrahedron vertices, whose
  thickness is the ``thickness`` argument (:ref:`ipc-contact`);
- ``particle_max_velocity``: the solver does not clamp particle velocities;
- ``spring_control``, as Newton's other solvers;
- ``joint_target_mode`` of revolute, prismatic, ball and universal joints, whose
  gains act whatever it says (on free joints and on D6 joints that become ball or
  free joints, a mode other than ``NONE`` is refused);
- ``joint_velocity_limit``, of which the solver warns once (:ref:`ipc-refused-joints`);
- ``joint_effort_limit`` of a rod joint's degrees of freedom: a rod's gains are
  its material, which no drive's bound limits (:ref:`ipc-rods`);
- ``joint_limit_kd`` with ``enable_joint_limit_damping=False``;
- ``Control.joint_act``;
- the tendons' ``solreffriction`` and ``solimpfriction``, fixed and spatial; the
  solver warns once of a fixed tendon's, as of the contact coefficients above,
  where a tendon with a friction loss sets one to other than Newton's default.

Examples
--------

Newton's ``cloth_ipc`` example drapes a cloth over a pile of boxes, captures its
substeps in a CUDA graph and checks every substep's status:

.. code-block:: console

   python -m newton.examples cloth_ipc

Fourteen other examples take ``--solver ipc``: ``basic_dzhanibekov``,
``basic_joints``, ``basic_mimic_joint``, ``basic_pendulum``, ``basic_shapes``,
``cable_initial_rest_pose``, ``cable_plectoneme``, ``cable_y_junction``,
``cloth_hanging``, ``balance_bird``, ``sensor_imu``,
``softbody_dropping_to_cloth``, ``softbody_gift`` and ``softbody_hanging``, for
example ``python -m newton.examples basic_pendulum --solver ipc``. Add
``--viewer null`` to run an example without a display.
