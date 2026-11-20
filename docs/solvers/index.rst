.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

Solvers
========

Solvers are used to integrate the dynamics of a Newton model.
The typical workflow is to construct a :class:`~newton.Model` and a :class:`~newton.State` object, then use a solver to advance the state forward in time
via the :meth:`~newton.solvers.SolverBase.step` method:

.. mermaid::
  :config: {"theme": "forest", "themeVariables": {"lineColor": "#76b900"}}

  flowchart LR
      subgraph Input["Input Data"]
          M[newton.Model]
          S[newton.State]
          C[newton.Control]
          K[newton.Contacts]
          DT[Time step dt]
      end

      STEP["solver.step()"]

      subgraph Output["Output Data"]
          SO["newton.State (updated)"]
      end

      %% Connections
      M --> STEP
      S --> STEP
      C --> STEP
      K --> STEP
      DT --> STEP
      STEP --> SO

See the :doc:`solver API reference </api/newton_solvers>` for constructor and
method details. For symptom-driven diagnosis and parameter tuning, start with
:ref:`Simulation Tuning`.

Choosing a Solver
-----------------

MuJoCo, Kamino, and VBD currently have dedicated backend guides:

- :doc:`MuJoCo <mujoco>` — generalized-coordinate rigid-body simulation and
  MuJoCo or MJCF workflows.
- :doc:`Kamino <kamino>` — constrained rigid mechanisms with kinematic loops
  and hard frictional contacts; experimental.
- :doc:`VBD <vbd>` — unified simulation of cloth, soft bodies, and rigid
  bodies in a single VBD-based solver; experimental.

The other solver backends are documented through their linked API references
and the comparison tables below.

For a rigid articulated robot, first decide which coordinate representation
fits the application. :class:`~newton.solvers.SolverMuJoCo`,
:class:`~newton.solvers.SolverFeatherstone`, and
:class:`~newton.solvers.SolverSteel` use generalized coordinates, while
:class:`~newton.solvers.SolverXPBD`,
:class:`~newton.solvers.SolverSemiImplicit`, and
:class:`~newton.solvers.SolverKamino` use maximal coordinates. For deformable
bodies, particles, or differentiable simulation, use the feature matrix below
to narrow the choice, then follow the linked API documentation.

.. _Supported Features:

Supported Features
------------------

.. list-table::
   :header-rows: 1
   :widths: auto
   :stub-columns: 0

   * - Solver
     - :abbr:`Integration (Available methods for integrating the dynamics)`
     - Rigid bodies
     - :ref:`Articulations <Articulations>`
     - Particles
     - Cloth
     - Soft bodies
     - Differentiable
     - Deterministic :sup:`3`
   * - :class:`~newton.solvers.SolverFeatherstone`
     - Semi-implicit
     - ✅
     - ✅ generalized coordinates
     - ✅
     - 🟨 no self-collision
     - ✅
     - 🟨 basic :sup:`2`
     - ✅
   * - :class:`~newton.solvers.SolverImplicitMPM`
     - Implicit
     - ❌
     - ❌
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
   * - :class:`~newton.solvers.SolverIPC`
     - Implicit (BDF1, BDF2)
     - 🟨 affine bodies :sup:`5`
     - 🟨 penalty joints :sup:`5`
     - ✅
     - ✅
     - 🟨 tetrahedra :sup:`5`
     - ❌
     - ❌
   * - :class:`~newton.solvers.SolverKamino`
     - Semi-implicit: Euler, Moreau-Jean
     - ✅ maximal coordinates
     - ✅ maximal coordinates
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
   * - :class:`~newton.solvers.SolverMuJoCo`
     - Explicit, Semi-implicit, Implicit-in-velocity
     - ✅ :sup:`1`
     - ✅ generalized coordinates
     - 🟨 element vertices only :sup:`4`
     - ✅ :sup:`4`
     - ✅ :sup:`4`
     - ❌
     - 🟨 CPU only
   * - :class:`~newton.solvers.SolverSemiImplicit`
     - Semi-implicit
     - ✅
     - ✅ maximal coordinates
     - ✅
     - 🟨 no self-collision
     - ✅
     - 🟨 basic :sup:`2`
     - ✅
   * - :class:`~newton.solvers.SolverSteel`
     - Semi-implicit, implicit drives
     - ✅
     - ✅ generalized coordinates
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅
   * - :class:`~newton.solvers.SolverStyle3D`
     - Implicit
     - ❌
     - ❌
     - ✅
     - ✅
     - ❌
     - ❌
     - ❌
   * - :class:`~newton.solvers.SolverVBD`
     - Implicit
     - ✅
     - 🟨 :ref:`limited joint support <Joint feature support>`
     - ✅
     - ✅
     - ✅
     - ❌
     - ✅
   * - :class:`~newton.solvers.SolverXPBD`
     - Implicit
     - ✅
     - ✅ maximal coordinates
     - ✅
     - 🟨 no self-collision
     - 🟨 experimental
     - ❌
     - ✅

| :sup:`1` Uses MuJoCo's built-in collision pipeline by default.
  Set ``use_mujoco_contacts=False`` to use Newton's collision pipeline.
| :sup:`2` ``basic`` means Newton includes several examples that use these solvers in diffsim workflows,
  see :ref:`Differentiability` for further details.
| :sup:`3` Run-to-run determinism with ``deterministic=wp.DeterministicMode.RUN_TO_RUN``
  (MuJoCo's CPU backend is deterministic regardless of that argument);
  see :doc:`/concepts/determinism` for limitations and performance trade-offs.
| :sup:`4` Simulated as MuJoCo flexes; particles that belong to no spring,
  triangle, or tetrahedron are not simulated. See :ref:`mujoco-flexes`.
| :sup:`5` Incremental Potential Contact: barrier contact, or on request libuipc's augmented-Lagrangian contact, with
  continuous collision detection between
  soft bodies, cloth (with self-collision), particles, near-rigid affine bodies, static shapes, planes and
  height fields, between the pairs that the collision groups, worlds and filter pairs let collide; springs and
  particle-body attachments without damping; revolute, prismatic, fixed and ball joints as penalty energies; one
  world besides the global
  one, or several with ``coupled_worlds=True``, which keeps them apart in contact but solves them together, so that
  a world's result depends on the others' iterations; on request its own contacts and their forces written into the
  step's ``Contacts`` (``CONTACT_F``); requires the optional ``libuipc-warp`` package.

.. experimental::
    :class:`~newton.solvers.SolverSteel`'s public API and behavior may change without prior notice.
    Its matrix-free solve runs on CPU and CUDA; contact torsion, contact compliance, joint friction, the non-default ``friction_mode`` values and sleeping with ``solve_mode="colored"`` require CUDA, and ``solve_mode="colored"`` runs the serial propagation sweep on CPU.

.. experimental::
    :class:`~newton.solvers.SolverKamino`'s public API and behavior may change without prior notice.

.. experimental::
    :class:`~newton.solvers.SolverVBD`'s public API and behavior may change without prior notice.

.. _Contact material support:

Contact Material Support
------------------------

:class:`~newton.ModelBuilder.ShapeConfig` and the matching :class:`~newton.Model`
shape material arrays store solver-neutral contact data. This section documents
which fields are currently used by Newton's built-in solvers. External solvers
may use different subsets or interpret these fields according to their own
formulation.

- ``mu``: :class:`~newton.solvers.SolverSteel` (arithmetic mean of the two
  shapes, scaled by ``contact_friction_scale``), :class:`~newton.solvers.SolverIPC`
  (arithmetic mean of the two shapes; ``Model.soft_contact_mu`` stands for a
  particle and ``Model.particle_mu`` holds between particles), :class:`~newton.solvers.SolverFeatherstone`,
  :class:`~newton.solvers.SolverSemiImplicit`,
  :class:`~newton.solvers.SolverXPBD`, :class:`~newton.solvers.SolverMuJoCo`,
  :class:`~newton.solvers.SolverVBD`, :class:`~newton.solvers.SolverKamino`,
  :class:`~newton.solvers.SolverStyle3D`, and
  :class:`~newton.solvers.SolverImplicitMPM`.
- ``ke`` / ``kd``: :class:`~newton.solvers.SolverFeatherstone`,
  :class:`~newton.solvers.SolverSemiImplicit`,
  :class:`~newton.solvers.SolverMuJoCo`, and
  :class:`~newton.solvers.SolverVBD`.
- ``kf`` / ``ka``: :class:`~newton.solvers.SolverFeatherstone` and
  :class:`~newton.solvers.SolverSemiImplicit`; ``kf`` is also used by
  :class:`~newton.solvers.SolverMuJoCo`
  (see :ref:`mujoco-contact-friction-solreffriction`).
- ``restitution``: :class:`~newton.solvers.SolverSteel` (arithmetic mean of
  the two shapes, for contacts closing faster than ``restitution_velocity_threshold``
  at the start of the step, unless ``enable_restitution=False``),
  :class:`~newton.solvers.SolverXPBD` when ``enable_restitution=True``, and
  :class:`~newton.solvers.SolverKamino`.
- ``mu_torsional`` / ``mu_rolling``: :class:`~newton.solvers.SolverXPBD` and
  :class:`~newton.solvers.SolverMuJoCo`.
  :class:`~newton.solvers.SolverVBD` also supports these properties for
  compliant-ALM rigid-rigid contacts.
- ``mu_static``: stored as :attr:`~newton.Model.shape_material_mu_static` and
  carried into :class:`~newton.solvers.SolverKamino`'s material tables, but no
  built-in solver distinguishes static from dynamic friction in its contact
  model yet; all of them use ``mu``.
- ``kh``: consumed by hydroelastic contact generation for Newton-generated
  contacts used by :class:`~newton.solvers.SolverFeatherstone`,
  :class:`~newton.solvers.SolverSemiImplicit`, and
  :class:`~newton.solvers.SolverMuJoCo` when ``use_mujoco_contacts=False``. See
  :ref:`Hydroelastic Contacts`.

:class:`~newton.solvers.SolverIPC` finds its own contacts and reads ``mu`` only:
its contact stiffness is the barrier's, the solver's ``kappa`` for every pair,
taken as given, or with ``adaptive_kappa`` kept in libuipc's scene-adaptive
range as libuipc keeps it.
It takes ``margin`` as the contact thickness of a shape's surface and
:attr:`~newton.Model.particle_radius` as that of a particle on no cloth or
tetrahedron, so that two shapes rest the sum of their margins apart; ``gap`` is
not read, the solver's ``d_hat`` being its detection distance.

.. _Joint feature support:

Joint Feature Support
---------------------

Not every solver supports every joint type or joint property.
The tables below document which joint features each solver handles.

:class:`~newton.solvers.SolverFeatherstone`, :class:`~newton.solvers.SolverSteel`, and
:class:`~newton.solvers.SolverMuJoCo` operate on :ref:`articulations <Articulations>` (generalized/reduced coordinates).
The maximal-coordinate solvers (:class:`~newton.solvers.SolverSemiImplicit`,
:class:`~newton.solvers.SolverXPBD`, and :class:`~newton.solvers.SolverKamino`)
enforce joints as pairwise body constraints but do not use the articulation kinematic-tree structure.
:class:`~newton.solvers.SolverVBD` supports a subset of joint types through maximal-coordinate
constraints, using a compliant augmented-Lagrangian (ALM) formulation.
:class:`~newton.solvers.SolverIPC` holds joints as penalty energies between affine bodies (libuipc's
affine-body joints), stiff in proportion to the mass of the dynamic bodies a joint joins, and writes
``joint_q`` and ``joint_qd`` after every step.
:class:`~newton.solvers.SolverStyle3D` and :class:`~newton.solvers.SolverImplicitMPM` do not support joints.

**Joint types**

.. list-table::
   :header-rows: 1
   :widths: auto
   :stub-columns: 1

   * - Joint type
     - :class:`~newton.solvers.SolverFeatherstone`
     - :class:`~newton.solvers.SolverSteel`
     - :class:`~newton.solvers.SolverSemiImplicit`
     - :class:`~newton.solvers.SolverXPBD`
     - :class:`~newton.solvers.SolverMuJoCo`
     - :class:`~newton.solvers.SolverVBD`
     - :class:`~newton.solvers.SolverKamino`
     - :class:`~newton.solvers.SolverIPC`
   * - PRISMATIC
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
   * - REVOLUTE
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
   * - BALL
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
   * - FIXED
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
   * - FREE
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
   * - DISTANCE
     - 🟨 :sup:`1`
     - 🟨 :sup:`1`
     - 🟨 :sup:`1`
     - |yes|
     - |no|
     - |no|
     - |no|
     - 🟨 :sup:`18`
   * - D6
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |no|
     - 🟨 :sup:`17`
   * - ROD
     - |no|
     - |no|
     - |no|
     - |no|
     - |no|
     - |yes|
     - |yes| :sup:`2`
     - |no|

| :sup:`1` DISTANCE joints are treated as FREE (no distance constraint enforcement).
| :sup:`2` See :ref:`kamino-rod-joints` for the supported rod joint properties.
| :sup:`17` With ``enable_d6_joints=True``, SolverIPC takes a D6 joint that locks what one of libuipc's joints
  locks: no axis (fixed), one angular axis (revolute), one linear axis (prismatic), three orthonormal angular axes
  (ball), or three orthonormal linear and three orthonormal angular axes (no joint). Other D6 joints raise
  ``NotImplementedError``.
| :sup:`18` With ``enable_distance_joints=True``, SolverIPC holds a distance joint of one length (equal bounds) as
  libuipc's soft vertex stitch of that rest length, and one with both bounds negative as a free joint; a range or a
  single bound raises ``NotImplementedError``.

**Joint properties**

.. list-table::
   :header-rows: 1
   :widths: auto
   :stub-columns: 1

   * - Property
     - :class:`~newton.solvers.SolverFeatherstone`
     - :class:`~newton.solvers.SolverSteel`
     - :class:`~newton.solvers.SolverSemiImplicit`
     - :class:`~newton.solvers.SolverXPBD`
     - :class:`~newton.solvers.SolverMuJoCo`
     - :class:`~newton.solvers.SolverVBD`
     - :class:`~newton.solvers.SolverKamino`
     - :class:`~newton.solvers.SolverIPC`
   * - :attr:`~newton.Model.joint_enabled`
     - |no|
     - |no| :sup:`8`
     - |yes|
     - |yes|
     - |no|
     - |yes|
     - |no|
     - |yes|
   * - :attr:`~newton.Model.joint_armature`
     - |yes|
     - |yes|
     - |no|
     - |no|
     - |yes|
     - |no|
     - |yes|
     - |no| :sup:`15`
   * - :attr:`~newton.Model.joint_stiffness` / :attr:`~newton.Model.joint_rest_q`
     - REVOLUTE, PRISMATIC, D6
     - REVOLUTE, PRISMATIC, D6
     - REVOLUTE, PRISMATIC, D6
     - |no|
     - Scalar joints and isotropic BALL springs
     - ROD ``joint_rest_q`` only
     - |no|
     - REVOLUTE, PRISMATIC
   * - :attr:`~newton.Model.joint_friction`
     - |no|
     - |yes| :sup:`9`
     - |no|
     - |no|
     - |yes|
     - |no|
     - |yes|
     - |no| :sup:`15`
   * - :attr:`~newton.Model.joint_limit_lower` / :attr:`~newton.Model.joint_limit_upper`
     - |yes|
     - |yes| :sup:`10`
     - |yes| :sup:`2`
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - REVOLUTE, PRISMATIC
   * - :attr:`~newton.Model.joint_limit_ke` / :attr:`~newton.Model.joint_limit_kd`
     - |yes|
     - |no|
     - |yes| :sup:`2`
     - |no|
     - |yes|
     - |yes| :sup:`7`
     - |no|
     - ``joint_limit_ke`` only :sup:`16`
   * - :attr:`~newton.Model.joint_effort_limit`
     - |no|
     - |yes| :sup:`11`
     - |no|
     - |no|
     - |yes|
     - |no|
     - |yes|
     - |no| :sup:`15`
   * - :attr:`~newton.Model.joint_velocity_limit`
     - |no|
     - |yes| :sup:`12`
     - |no|
     - |no|
     - |no|
     - |no|
     - |no|
     - |no|

| :sup:`2` Not enforced for BALL joints in SemiImplicit.
| :sup:`15` SolverIPC raises ``NotImplementedError`` instead of ignoring the property: for non-zero armature or
  friction, and for an effort limit below 1e6 on a degree of freedom with a drive.
| :sup:`16` SolverIPC enforces a limit as libuipc's cubic penalty, set to push back with ``joint_limit_ke`` times
  the overshoot at an overshoot of 1e-3 rad or m, more steeply beyond; ``joint_limit_kd`` is not read.

SemiImplicit angular springs use principal coordinates in [-pi, pi). Crossing
that boundary produces a force jump of ``2*pi*stiffness`` and can destabilize
motion near it. Use Featherstone or MuJoCo for boundary-crossing or multi-turn
motion; see :ref:`passive-joint-springs` for details.

**Actuation and control**

.. list-table::
   :header-rows: 1
   :widths: auto
   :stub-columns: 1

   * - Feature
     - :class:`~newton.solvers.SolverFeatherstone`
     - :class:`~newton.solvers.SolverSteel`
     - :class:`~newton.solvers.SolverSemiImplicit`
     - :class:`~newton.solvers.SolverXPBD`
     - :class:`~newton.solvers.SolverMuJoCo`
     - :class:`~newton.solvers.SolverVBD`
     - :class:`~newton.solvers.SolverKamino`
     - :class:`~newton.solvers.SolverIPC`
   * - :attr:`~newton.Model.joint_target_ke` / :attr:`~newton.Model.joint_target_kd`
     - |yes|
     - |yes|
     - |yes| :sup:`2`
     - |yes|
     - |yes|
     - |yes| :sup:`7`
     - |yes|
     - REVOLUTE, PRISMATIC
   * - :attr:`~newton.Model.joint_target_mode`
     - |no|
     - |no|
     - |no|
     - |no|
     - |yes|
     - |no|
     - |yes|
     - |no|
   * - :attr:`~newton.Control.joint_f` (feedforward forces)
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|
     - |yes|

**Constraints**

.. list-table::
   :header-rows: 1
   :widths: auto
   :stub-columns: 1

   * - Feature
     - :class:`~newton.solvers.SolverFeatherstone`
     - :class:`~newton.solvers.SolverSteel`
     - :class:`~newton.solvers.SolverSemiImplicit`
     - :class:`~newton.solvers.SolverXPBD`
     - :class:`~newton.solvers.SolverMuJoCo`
     - :class:`~newton.solvers.SolverVBD`
     - :class:`~newton.solvers.SolverKamino`
     - :class:`~newton.solvers.SolverIPC`
   * - Equality constraints (CONNECT, WELD, JOINT)
     - |no|
     - |no|
     - |no|
     - |no|
     - |yes|
     - |no|
     - |no|
     - |no|
   * - Mimic joints
     - |yes| :sup:`3`
     - |yes| :sup:`13`
     - |yes| :sup:`4`
     - |yes| :sup:`5`
     - |yes| :sup:`6`
     - |yes| :sup:`5`
     - |no|
     - |no|
   * - Particle-body attachments
     - |no|
     - |no|
     - |no|
     - |no|
     - |no|
     - |yes| :sup:`14`
     - |no|
     - |no|

| :sup:`3` Featherstone eliminates follower degrees of freedom from its reduced dynamics and transfers follower forces and inertia to the reference joint.
| :sup:`4` SemiImplicit enforces joint-owned mimic relationships with penalty springs configured by ``joint_mimic_ke`` and ``joint_mimic_kd``.
| :sup:`5` XPBD and VBD enforce joint-owned mimic relationships through coupled maximal-coordinate corrections. Both apply one mimic correction per solver iteration.
| :sup:`6` MuJoCo lowers each joint-owned relationship to joint equality constraints. Multi-axis D6 relationships produce one equality constraint per axis.
| :sup:`7` VBD interprets ``joint_target_kd`` and ``joint_limit_kd`` as absolute damping coefficients in physical units.
| :sup:`8` Steel rejects models with disabled articulation joints instead of ignoring them; a disabled loop-closing joint starts as a released closure.
| :sup:`9` Steel enforces dry joint friction of PRISMATIC, REVOLUTE, BALL, and D6 DOFs as one bounded constraint row per DOF when constructed with ``enable_joint_friction=True`` on CUDA (off by default, experimental); ``steel:joint_friction_load_coefficient`` adds the load-proportional joint friction of PhysX articulations before PhysX 5.6.
| :sup:`10` Steel enforces joint limits as hard unilateral constraint rows when constructed with ``enable_joint_limits=True`` (off by default).
| :sup:`11` Steel clamps the explicit joint drive force to the effort limit. With the default ``drive_mode="augmented"`` the implicit stiffness and damping response is unbounded, so under a large external load the drive reaction can exceed the limit; the PGS drive rows of ``drive_mode="rows"`` bound the complete reaction.
| :sup:`12` Steel enforces velocity limits of PRISMATIC, REVOLUTE, and D6 DOFs when constructed with ``enable_joint_velocity_limits=True``. With ``drive_mode="rows"``, driven DOFs are clamped at the end of every solver iteration instead of using velocity-limit rows (``fuse_joint_velocity_limits``).
| :sup:`13` Steel enforces each mimic relationship within one articulation as one bilateral constraint row per follower coordinate; mimics of BALL, FREE, and DISTANCE joints and mimics across articulations are rejected. It also enforces loop-closing BALL joints as point constraints, loop-closing REVOLUTE joints as point constraints plus two hinge-axis constraints, and loop-closing FIXED joints as point constraints plus three rotation constraints. The propagation contact response (``articulated_contact_response="propagation"``) rejects loop-closing joints and solves mimic rows iteratively.
| :sup:`14` See :ref:`Particle-body attachments` for authoring and semantics. Attachments spanning two
  solvers are coupled by :class:`~newton.solvers.experimental.coupled.SolverCoupledADMM` instead.

.. _Particle-body attachments:

Particle-Body Attachments
-------------------------

A particle-body attachment ties one cloth or solid particle to a point in a
rigid body's local frame. Author attachments with
:meth:`newton.ModelBuilder.add_attachment_particle_body`. They are stored on the
:class:`~newton.Model` and do not require a coupled solver.

:class:`~newton.solvers.SolverVBD` applies an attachment whenever it integrates
both endpoints. It solves every attachment as a compliant augmented-Lagrangian
(ALM) row, with the same formulation as the VBD joint rows: a reaction
multiplier carries the attachment force, and a numerical penalty chosen from the
endpoint masses and the particle's surrounding material helps the iterations
converge. The penalty is chosen like the joint limit rows, so a finite
attachment reaches its spring over several iterations. The penalty affects convergence only, not physical compliance.
A finite ``stiffness`` [N/m] converges to a spring with ``damping`` [N·s/m], so a
loaded attachment keeps a physical offset. Set ``stiffness=math.inf`` to request
a hard point constraint; damping is ignored for hard attachments. Multipliers are
retained across steps, scaled by ``rigid_avbd_gamma``. A finite iteration budget
can leave constraint error, especially for heavy free bodies at large time steps;
more iterations or smaller steps reduce it. Incompatible hard attachments cannot
all be satisfied. Separated initial anchors can cause large correction impulses.
The constraint is translational, similar to the positional part of a ball joint;
a single particle has no orientation, so there is no angular counterpart. Forces
are equal and opposite, including the torque about the body's center of mass, and
the path is independent of contact, so it stays active without penetration.

Both endpoints must be integrated by the same solver, so
:class:`~newton.solvers.SolverVBD` rejects attachments when it is constructed
with ``integrate_with_external_rigid_solver=True``.

:class:`~newton.solvers.SolverIPC` applies an attachment as libuipc's soft
vertex stitch: a spring of the attachment's ``stiffness`` between the particle
and the body point, a term of each step's implicit solve, so a finite
attachment is the spring of implicit Euler and its forces are equal and
opposite. A hard attachment takes the penalty of SolverIPC's ball joint, and an
attachment with damping is rejected, since libuipc's stitch has no damping.

When the endpoints of a finite-stiffness attachment belong to two different solvers in a coupled simulation,
:class:`~newton.solvers.experimental.coupled.SolverCoupledADMM` turns the same
model rows into cross-solver interface constraints; see :ref:`ADMM coupling`.
The two cases are mutually exclusive, so no attachment is applied twice, and a
row whose body or particle is owned by no solver applies no force at all. The
coupler also warns when both endpoints belong to one entry whose solver does not
support attachments. Hard attachments require both endpoints in one VBD entry;
the coupler raises an error for other ownership arrangements.

.. _External forces:

External Forces
---------------

:attr:`State.body_f <newton.State.body_f>` and
:attr:`State.particle_f <newton.State.particle_f>` hold external forces that the
caller applies. Every solver reads them from ``state_in`` and never writes them
in ``state_in`` or ``state_out``; contact, joint, and elastic forces are
accumulated in solver-owned buffers. Forces therefore persist until the caller
clears them, so a force written once can act on every substep of a frame, and
double-buffered states may share their force arrays:

.. code-block:: python

    state_1.body_f = state_0.body_f
    state_1.particle_f = state_0.particle_f

    state_0.clear_forces()
    apply_external_forces(state_0)  # written once per frame
    for _ in range(sim_substeps):
        solver.step(state_0, state_1, control, contacts, sim_dt)
        state_0, state_1 = state_1, state_0

Forces that are recomputed every substep, such as viewer picking forces, still
call :meth:`State.clear_forces() <newton.State.clear_forces>` before they are
reapplied on each substep.

.. _Differentiability:

Differentiability
-----------------

Differentiable simulation in Newton typically runs a forward rollout inside
``wp.Tape()``, computes a scalar loss from the simulated state, and then calls
``tape.backward(loss)`` to populate gradients on differentiable state,
control, or model arrays. In practice, this starts by calling
:meth:`~newton.ModelBuilder.finalize` with ``requires_grad=True``.

.. testcode::

    import warp as wp
    import newton

    @wp.kernel
    def loss_kernel(particle_q: wp.array[wp.vec3], target: wp.vec3, loss: wp.array[float]):
        delta = particle_q[0] - target
        loss[0] = wp.dot(delta, delta)

    builder = newton.ModelBuilder()
    builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.0), vel=wp.vec3(1.0, 0.0, 0.0), mass=1.0)

    model = builder.finalize(requires_grad=True)
    solver = newton.solvers.SolverSemiImplicit(model)

    state_in = model.state(requires_grad=True)
    state_out = model.state(requires_grad=True)
    control = model.control()
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    target = wp.vec3(0.25, 0.0, 0.0)

    tape = wp.Tape()
    with tape:
        state_in.clear_forces()
        solver.step(state_in, state_out, control, None, 1.0 / 60.0)
        wp.launch(
            loss_kernel,
            dim=1,
            inputs=[state_out.particle_q, target],
            outputs=[loss],
        )

    tape.backward(loss)
    initial_velocity_grad = state_in.particle_qd.grad.numpy()
    assert float(initial_velocity_grad[0, 0]) < 0.0

See the `DiffSim examples on GitHub`_ for the current reference workflows.

.. _DiffSim examples on GitHub: https://github.com/newton-physics/newton/tree/main/newton/examples/diffsim

.. |yes| unicode:: U+2705
.. |no| unicode:: U+274C
