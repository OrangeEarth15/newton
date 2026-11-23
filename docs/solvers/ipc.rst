.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

.. currentmodule:: newton

IPC
===

:class:`~newton.solvers.SolverIPC` is an Incremental Potential Contact solver
for soft bodies, cloth and rigid bodies. It wraps the
`libuipc-warp <https://github.com/OrangeEarth15/libuipc-warp>`_ engine, a Warp
implementation of libuipc, through its array interface (no libuipc build is
needed). Each :meth:`~newton.solvers.SolverIPC.step` is one implicit time step,
by default of implicit Euler (BDF1): a Newton iteration with a backtracking line
search, a block-Jacobi PCG linear solve, barrier contact and continuous
collision detection. ``integrator="bdf2"``, ``contact_constitution="al-ipc"``
and ``fem_preconditioner="mas"`` take libuipc's BDF2, its augmented-Lagrangian
contact and its MAS preconditioner instead (see :ref:`ipc-bdf2`,
:ref:`ipc-al-ipc` and :ref:`ipc-mas`). Under implicit Euler the rotation of a
rigid body is by default integrated as a rigid body's, where libuipc's step
moves the body along a straight line in its affine coordinates and damps its
spin (``body_rotation``, see :ref:`ipc-rigid-bodies`). The Newton loop, the PCG
loop and the line search run on the device under ``wp.capture_while``, so a step
never synchronizes with the host when it is captured in a CUDA graph. A step
that is not captured captures the solve itself and checks how it ended (see
:ref:`ipc-status`).

SolverIPC requires the optional ``libuipc-warp`` package, 0.4.4 or newer
(``import uipc_warp``): an older one, or a build without the capabilities the
solver calls (``EngineOptions(affine_rotation=)`` for ``body_rotation="rigid"``,
``EngineOptions(al_sweep_kinematic=)`` for AL-IPC, and the frame restart for an
entry of a coupler that restarts its steps), raises :class:`ImportError` naming
what it lacks.
This page describes how the solver takes Newton's model, state and control and
what it refuses; the API reference of :class:`~newton.solvers.SolverIPC` lists
the constructor's arguments.

.. _ipc-model-content:

Model content
-------------

The solver simulates:

- particles (a particle on no cloth or tetrahedron is a sphere of its radius,
  see :ref:`ipc-contact`) and tetrahedra. Elasticity is the SNK1 stable
  Neo-Hookean model with Lamé parameters ``tet_materials[:, 0]`` (``k_mu``) and
  ``tet_materials[:, 1]`` (``k_lambda``). ``k_damp`` is ignored.
- cloth: triangles of :attr:`~newton.Model.tri_indices` that are not on a
  tetrahedral body become libuipc's strain-limiting Baraff-Witkin membrane, and
  bending edges of :attr:`~newton.Model.edge_indices` become discrete-shell
  hinges (see :ref:`ipc-cloth`).
- springs between particles: libuipc's Hookean springs with Newton's energy
  :math:`\tfrac{1}{2} k_e (l - L_0)^2` (``spring_stiffness``,
  ``spring_rest_length``), and libuipc's soft vertex stitch for a rest length
  of 0. A spring adds no contact primitive: the particles it joins collide as
  other particles do.
- particle-body attachments: libuipc's soft vertex stitch between the particle
  and a point of the body (see :ref:`ipc-attachments`).
- rigid bodies with box, sphere, capsule, cylinder, cone, ellipsoid or mesh
  collision shapes become affine bodies (see :ref:`ipc-rigid-bodies`).
- contact against static plane shapes, each treated as an infinite half space,
  and against other static shapes, which become fixed triangle meshes; a height
  field with Newton's collision triangles, two per cell of its grid (an open
  surface, as Newton's).
- contact between all of these: point-triangle and edge-edge barriers with
  continuous collision detection, between the pairs that Newton's collision
  groups, worlds, collide flags and filter pairs let collide, with the friction
  of the two materials (see :ref:`ipc-contact`). Each connected group of
  tetrahedra or cloth triangles is one body and collides with itself (as finite
  elements do in libuipc), so ``d_hat`` must stay below the spacing of
  non-adjacent surface primitives at rest, as the default ``d_hat`` does (see
  :ref:`ipc-d-hat`). A rigid body does not collide with itself.
- kinematic particles (zero mass or not :attr:`~newton.ParticleFlags.ACTIVE`)
  and kinematic bodies (:attr:`~newton.BodyFlags.KINEMATIC`, zero mass or zero
  inverse mass, see :ref:`ipc-coupling`). A kinematic body with mass and
  colliding shapes moves inside the step toward the pose that ``state_in``
  prescribes, where the solver can aim it. Kinematic particles and the other
  kinematic bodies jump to their ``state_in`` pose before the step, and a jump
  that meets a surface fails the step (see :ref:`ipc-kinematic-bodies`). A particle that is not active and
  on no cloth or tetrahedron collides with nothing, as in Newton.
- revolute, prismatic, fixed and ball joints between bodies or to the world,
  with drives, passive springs and damping, limits, armature and
  :attr:`~newton.Control.joint_f` (see :ref:`ipc-joints`); ``joint_q`` and
  ``joint_qd`` are written at every step. D6 joints that lock what one of these
  locks, or nothing, and distance joints of one length or none.
- external forces: :attr:`~newton.State.particle_f`,
  :attr:`~newton.State.body_f` and :attr:`~newton.Control.joint_f`, held
  constant over the step as libuipc holds its external forces (a constant
  acceleration in the implicit-Euler prediction). A body wrench acts at the
  centre of mass; its torque does work on the rotations of the affine body and
  none on its stretch.
- gravity per world: a particle or body falls with the
  :attr:`~newton.Model.gravity` of its world, a global one with the global
  world's (the last entry).
- lagged friction on every contact.

.. _ipc-refused:

Refused content
---------------

Not supported, and rejected when the solver is constructed:
:class:`NotImplementedError` for springs and particle-body attachments with
damping (libuipc has no damping element), springs of a negative rest length or
stiffness or that join a particle to itself, mimic constraints, enabled MuJoCo
equality rows that no loop joint holds (see :ref:`ipc-joints`), tendons, muscles, body variants (:meth:`~newton.ModelBuilder.add_body_variants`:
the solver triangulates the shapes when it builds the solve and finds its own
contacts, so it cannot follow a variant row's shape), non-zero triangle or
tetrahedron activations, the joint types
and properties that :ref:`ipc-joints` rejects, gradients
(``model.requires_grad``), with independent worlds what :ref:`ipc-worlds`
refuses (an unknown of the global world, an element that joins two worlds, and
the options they do not take yet), a :attr:`~newton.Model.shape_contact_pairs`
array assigned to the model, more contact elements than the engine takes (see
:ref:`ipc-contact`), collision meshes with a surface velocity
(``Mesh.enable_surface_velocity``, which the solver's own contacts would
ignore), and a CPU device. Gaussian shapes and planes attached to bodies raise
when the shapes are triangulated. A step that is not captured raises
:class:`NotImplementedError` when its :class:`~newton.Control` holds a triangle
or tetrahedron activation other than 0 (a captured step cannot raise, and does
not check them). The ``contacts`` argument of
:meth:`~newton.solvers.SolverIPC.step` is not read: the solver detects its own
contacts, and writes them into it when ``CONTACT_F`` is requested (see
:ref:`ipc-contact-output`).

.. _ipc-not-read:

Inputs not read
---------------

Not read: the penalty contact parameters of shapes and particles (``ke``,
``kd``, ``kf``, ``ka``, ``particle_cohesion`` and ``particle_adhesion``:
``kappa`` holds for every pair), restitution (``shape_material_restitution``,
``soft_contact_restitution``), static friction (``shape_material_mu_static``),
torsional and rolling friction (``shape_material_mu_torsional``,
``shape_material_mu_rolling``), of which the solver warns once, with
:class:`~newton.NewtonWarning` when it is built, where a colliding shape or the
particles set one to other than Newton's default (its contact has one friction
coefficient and no restitution, and MJCF's default friction holds Newton's
default torsional and rolling coefficients), ``shape_gap`` (``d_hat`` is the
solver's detection distance), the :attr:`~newton.Model.particle_radius` of cloth and tetrahedron
vertices (see :ref:`ipc-contact`), damping coefficients, ``spring_control``,
drag, lift, ``edge_rest_length`` (see :ref:`ipc-cloth`), the activations of a
:class:`~newton.Control` (an eager step refuses those other than 0, see
:ref:`ipc-refused`), ``joint_target_mode``, ``joint_velocity_limit``, of
which the solver warns once, with :class:`~newton.NewtonWarning` when it is
built or notified of the joints' degrees of freedom, where one is below Newton's
default 1e6 (it does not limit joint velocities, and Newton's URDF importer
sets the limit from every ``<limit velocity>``), ``joint_limit_kd`` with
``enable_joint_limit_damping=False`` and ``joint_act``.

.. _ipc-cloth:

Cloth parameters
----------------

libuipc-warp's membrane energy per rest area is

.. math::

    \Psi = K_a \phi(|F a|) + K_b \phi(|F b|) + K_g ((F a) \cdot (F b))^2,
    \qquad \phi(s) = (s - 1)^2 + \eta \max(s - 1, 0)^3,

with the material axes ``a``, ``b`` of each triangle's rest frame. Newton
integrates its membrane energy over the rest area only, so
``tri_materials[:, 0]`` (``tri_ke`` = :math:`\mu`) and ``tri_materials[:, 1]``
(``tri_ka`` = :math:`\lambda`) act as membrane Lamé moduli [N/m]. At small
strain Newton's membrane is linear elasticity,
:math:`\mu\,\varepsilon:\varepsilon + \tfrac{\lambda}{2}(\mathrm{tr}\,\varepsilon)^2`,
and the mapping matches its energy for a stretch along either material axis with
the other held fixed and for simple shear:

- :math:`K_a = K_b = \mu + \lambda / 2` [N/m];
- :math:`K_g = \mu / 2` [N/m], since
  :math:`(F a) \cdot (F b) \approx 2 \varepsilon_{ab}`;
- :math:`\eta` = ``cloth_strain_rate`` (libuipc's strain limiting; 0 keeps the
  membrane quadratic). The Baraff-Witkin model has no Poisson coupling (no
  :math:`\lambda\,\varepsilon_{aa}\varepsilon_{bb}` term), so an equal stretch
  in both directions meets less resistance than in Newton's model.

Newton's hinge energy is :math:`\tfrac{1}{2} k_e L_0 (\theta - \bar\theta)^2`
(``edge_bending_properties[:, 0]`` = :math:`k_e`); libuipc's is
:math:`\kappa L_0 (\theta - \bar\theta)^2 / \bar h` with
:math:`\bar h = (A_1 + A_2) / (3 L_0)` from the two rest triangle areas, so each
hinge gets :math:`\kappa = k_e \bar h / 2`. Particle masses are used as they
are. The damping coefficients ``tri_kd`` and ``edge_kd``, drag and lift are
ignored: implicit Euler supplies the only damping.

Rest shapes are Newton's. A membrane triangle's rest edges are the inverse of
its :attr:`~newton.Model.tri_poses` (the edges in the triangle's frame whose
first axis runs along its first edge), laid into the plane of the triangle at
:attr:`~newton.Model.particle_q`, first edge along first edge, where it takes
libuipc's material axes. Newton's builders compute ``tri_poses`` in float64 from
``particle_q`` and store them in float32: a triangle whose ``tri_poses`` are
that rounding takes its rest edges at ``particle_q`` in float64, and when every
triangle does, as for a cloth built at rest, the engine takes them from
``particle_q`` as libuipc takes them from its rest geometry, so that the
membrane starts without strain. A hinge's rest angle is its
:attr:`~newton.Model.edge_rest_angle`, the signed angle between the normals of
``(o0, v1, v2)`` and ``(o1, v2, v1)`` for the row ``(o0, o1, v1, v2)``:
libuipc's angle by the winding of the two triangles where ``o0``'s triangle runs
``v1 -> v2``, as in the rows Newton's cloth builders make, and its negative
where a row given by hand has ``o0``'s triangle run ``v2 -> v1``. A row whose
two triangles run the edge the same way raises :class:`ValueError`. Where every
hinge's ``edge_rest_angle`` is the angle Newton's builders measure at
``particle_q``, as for a cloth built at rest, the engine takes the rest angles
from ``particle_q`` as libuipc takes them from its rest geometry, with the
dihedral angle its bending energy evaluates, so that every hinge starts at its
rest angle: libuipc's angle, the arccosine of the normals' cosine, reads up to
1.5e-8 rad on a flat hinge that Newton measures as 0. The hinges keep the rest
lengths and areas of ``particle_q`` (libuipc's :math:`L_0` and :math:`\bar h`)
where Newton's VBD takes ``edge_rest_length``; for a rest shape equal to the
start the two agree.

The solve holds copies of the materials and rest shapes of the tetrahedra,
triangles and hinges (``tet_materials``, ``tri_materials``,
``edge_bending_properties``, ``tet_poses``, ``tri_poses``, ``edge_rest_angle``)
and of :attr:`~newton.Model.particle_flags`. Newton has no flag of its own for
them, so :meth:`~newton.solvers.SolverIPC.notify_model_changed` compares them at
every notification, whatever its flags, and builds the solve again when one
changed in place.

.. _ipc-contact:

Contact
-------

Which pairs collide, and with which friction, is libuipc's contact tabular:
every vertex and half plane is in a contact element, and every pair of elements
has a friction coefficient and a switch. Shapes that Newton's rules cannot tell
apart share an element (the same collision group, world, collide flags and
friction coefficient; with several worlds the subscenes keep the worlds apart,
see :ref:`ipc-worlds`, and the world leaves the signature); a shape in a filter
pair gets one of its own, which with several worlds its copies in replicated
worlds share. Free particles (in no triangle and no tetrahedron) are in one
element, or in another one that collides with nothing when they are not
:attr:`~newton.ParticleFlags.ACTIVE`, and the vertices of cloth and tetrahedra
in a third. Two shapes collide as :attr:`~newton.Model.shape_contact_pairs`
says: both collide with shapes, their collision groups and worlds are compatible
and they are not a filter pair; shapes of one body never collide, and the solver
checks at construction that its pairs are exactly the model's. A particle
collides with a shape that has :attr:`~newton.ShapeFlags.COLLIDE_PARTICLES`, two
free particles or a free particle and a cloth or tetrahedron vertex collide while
the model has more than one particle and a :attr:`~newton.Model.particle_grid`
(as Newton's particle solvers decide: ``finalize`` makes the grid for particles
of positive radius, and setting it to ``None`` switches particle contact off;
``notify_model_changed`` with ``MODEL_PROPERTIES`` takes the change), and cloth
and tetrahedra always collide with themselves and with each other. The
friction of two shapes is the arithmetic mean of their
:attr:`~newton.Model.shape_material_mu`, of a particle and a shape the mean of
:attr:`~newton.Model.soft_contact_mu` and the shape's, and of two particles
:attr:`~newton.Model.particle_mu`, as in Newton's semi-implicit, Featherstone
and XPBD solvers. The engine tests the table of the elements and that of the
worlds (see :ref:`ipc-worlds`) apart, as libuipc does.

Contact thickness, libuipc's distance at which the barrier diverges: a free
particle is a sphere of its :attr:`~newton.Model.particle_radius`, so it rests
its radius above a shape and two free particles rest the sum of their radii
apart; a shape's vertices take its :attr:`~newton.Model.shape_margin` (a static
plane's surface is moved out by it), so that two shapes rest the sum of their
margins apart, as Newton's margins add up; and the vertices of cloth and
tetrahedra take the solver's ``thickness`` argument. libuipc gives a vertex's
thickness to every edge and triangle it is on, and finite elements collide with
themselves, so a cloth vertex of radius r would keep the cloth's non-adjacent
triangles and any other cloth 2r away: Newton's radii are sphere radii, often
larger than the spacing of a cloth, and Newton's cloth solvers do not use them
for cloth self-contact either.

Contact stiffness, the barrier's :math:`\kappa`, is ``kappa`` for every pair,
taken as given. libuipc does not take a stiffness as given: its contact manager
keeps it in a range :math:`[\kappa_{lo}, 100\,\kappa_{lo}]` with
:math:`\kappa_{lo} = c\,\bar m / \Delta t^2`, :math:`\bar m` the mean mass of
libuipc's vertices and :math:`c` a factor of the scene's diagonal and ``d_hat``
(Stiff-GIPC's estimate); a scene whose default contact model the user has not
set takes :math:`\kappa_{lo}`, and a stiffness the user gives is clamped into
the range. libuipc-warp's pyuipc front end does the same. With
``adaptive_kappa=True`` the solver does too: ``kappa=None`` takes
:math:`\kappa_{lo}`, a given ``kappa`` is clamped into the range and the first
step that clamps it warns with :class:`~newton.NewtonWarning`. The vertices are
then every particle, every vertex of the triangulated surfaces of bodies and
static shapes, and one per plane; each body's mass counts once, static shapes
and planes have none. The range follows the ``dt`` of each step, also of a
captured one, and is computed from the masses when the solve is built: a change
of mass in place (:meth:`~newton.solvers.SolverIPC.notify_model_changed`) keeps
it, as libuipc computes it once at init. Every contact pair takes one value,
which :meth:`~newton.solvers.SolverIPC.contact_stiffness` gives for a step of
any ``dt``; with independent worlds every world takes the range of the same
world alone, from its own vertices and the global ones, and every pair of a
world its value (see :ref:`ipc-worlds`).

.. _ipc-bdf2:

BDF2
----

``integrator="bdf2"`` is libuipc's BDF2 (its ``integrator/type``, taken as
libuipc and its pyuipc front end take it): the step predicts

.. math::

    \tilde x = \tfrac{4}{3} x_n - \tfrac{1}{3} x_{n-1}
        + \tfrac{2}{9} \Delta t (4 v_n - v_{n-1}) + \tfrac{4}{9} \Delta t^2 a,

weighs the inertia :math:`\tfrac{9}{4}` times as much against the potentials as
implicit Euler does, and ends at
:math:`v_{n+1} = (3 x_{n+1} - 4 x_n + x_{n-1}) / (2 \Delta t)`: second order,
with less numerical damping than implicit Euler. Contact so loses less energy
and bodies take longer to settle: a cube dropped onto the ground in steps of
1/1800 s rebounds at 0.93 of its impact speed, at 0.47 under implicit Euler, as
in libuipc. The history :math:`(x_{n-1}, v_{n-1})` is the solver's, not the
state's: a step from the state the solver wrote last continues it, and every
other step is an implicit Euler step whose start becomes the history: the first
step, a step from a state the solver did not write last (also one it wrote
earlier) and the step after :meth:`~newton.solvers.SolverIPC.reset`. A step from
a new state so applies the whole impulse of its forces, as in Newton's other
solvers. (libuipc starts the history from rest, without acceleration: a constant
force from rest gets two thirds of its impulse in the first step, and the
velocity lags half a step's impulse behind for good.) A change of gravity
(:meth:`~newton.solvers.SolverIPC.notify_model_changed` with
``MODEL_PROPERTIES``) starts the history anew too: the next step compares the
gravity of every world with the step before's, on the device, also inside a
captured graph, and is an implicit Euler step of the worlds whose gravity
changed, so that their velocity follows the new gravity at once; without it the
velocity would lag :math:`\Delta t \, \Delta g / 2` behind for good, as above.
Gravity is a parameter of the model, changed between steps by a jump that a
history taken under the old gravity cannot hold. The external forces
(``particle_f``, ``body_f``, ``joint_f``) and the drives are inputs of every
step that may change at every step, and BDF2 takes them as they come, as libuipc
does: a force that jumps within a history leaves the velocity half a step's
impulse of the jump behind. With several worlds a new state in one world, or its
reset, makes the step an implicit Euler step of that world's particles and
bodies, and the other worlds go on with BDF2. As in libuipc the coefficients
assume one ``dt`` for consecutive steps: a step of another ``dt`` is taken as it
is. The drives, passive springs and damping of joints take BDF2's velocity at
the end of the step (see :ref:`ipc-joints`); the default body stiffness follows
the rule of :ref:`ipc-rigid-bodies`. Rigid bodies take libuipc's straight-line
step under BDF2: ``body_rotation`` is ``"linear"`` there by default, and
``"rigid"``, whose flight replaces the implicit Euler prediction, is refused
(see :ref:`ipc-rigid-bodies`).

.. _ipc-al-ipc:

AL-IPC contact
--------------

``contact_constitution="al-ipc"`` is libuipc's augmented-Lagrangian contact (its
``contact/constitution``; the method of Zheng, Luo and Li, "Robust and Efficient
Penetration-Free Elastodynamics without Barriers"): each Newton iteration
linearizes the constraint :math:`d \ge \hat d + \text{thickness}` of the active
contact pairs, with a penalty that follows the masses and multipliers that
converge to the contact forces, and a collision-free sweep moves the
configuration toward the solution. Contact rests at ``d_hat`` beyond the contact
thickness, where the barrier rests inside ``d_hat``. Its constraints hold
between surfaces of vertices with mass, the penalty of a pair scaling with the
smallest mass of its vertices: particles on no cloth or tetrahedron that collide
are refused, and so are static mesh shapes, particles and bodies without mass
that collide (Newton's static shapes have no mass; static planes are taken). A
pair joins the active set only when a sweep finds its two primitives touching,
and the step that solves its constraint ends with the gap at ``d_hat``: a body
whose step began a gap :math:`g` from a surface leaves it at about
:math:`(\hat d - g) / \Delta t`, whatever its speed of impact. This is the
method's own (libuipc and the authors' code do the same), and it gains energy
where ``d_hat`` exceeds the distance a body travels in a step: a cube dropped
0.4 m onto the ground in steps of 1/1800 s leaves at 4.4 times its impact speed
of 2.8 m/s with ``d_hat`` 1 cm, at 0.7 times with 1 mm, and in steps of 1/100 s
at 0.34 times with 1 cm. Keep ``d_hat`` well below the distance a body moves in
a step. libuipc's default ``decay_factor`` 0.3 drops a pair from the active set
three outer updates after it opens, so a resting contact that opens for a moment
joins again at the surface and is pushed out again: a stack of two 0.1 m boxes
in steps of 1/100 s keeps rocking with ``d_hat`` 5 mm or 2 mm, in most runs
until its upper box falls off, on libuipc's own backend too. SolverIPC takes the
paper's decay factor 0.9 instead (libuipc-warp's ``al_decay_factor``), which
keeps a pair about 44 updates: such a stack comes to rest. A kinematic body moved between steps is swept
from where the last step left it to its pose in ``state_in`` (the method's
moving boundaries, which libuipc's frame lacks: there nothing a kinematic body
moves into collides with it): what it meets joins the active set and is pushed,
and one that contact holds back ends the step short of its pose and goes on
toward it in the next. The first step after the solver is built, the step after
a :meth:`~newton.solvers.SolverIPC.reset` that selects its world and the step
after a failed one place it at its pose instead, without a sweep. ``kappa`` is
not used: a ``kappa`` other than the default and ``adaptive_kappa`` are
refused. The step ends on AL-IPC's own
criterion, the paper's progress weight, which has no Newton residual:
``SOLVER_RESIDUAL`` is not offered, and without contact a step takes
``semi_implicit_k_min`` Newton iterations (the paper's ``K_min``), and one with
``semi_implicit=False``, which solves the barrier's steps to the tolerance but
makes AL-IPC's the least converged (with libuipc's body stiffness a free cube
spinning at 1.08 rad/s keeps 6% of its spin after 100 steps of 0.01 s). A larger
``semi_implicit_k_min`` converges AL-IPC's steps further. A change of the
bodies' masses (:meth:`~newton.solvers.SolverIPC.notify_model_changed`) builds
the solve again, since the penalty follows the masses the solve was built with.

.. _ipc-mas:

MAS
---

``fem_preconditioner="mas"`` is libuipc's multilevel additive Schwarz
preconditioner (its ``linear_system/fem_preconditioner``) on the vertices of
particles, cloth and tetrahedra (bodies keep their 12 x 12 blocks): it inverts
the system exactly on clusters of up to 16 vertices and corrects between them on
coarser levels made of neighbouring clusters. libuipc takes the clusters from
METIS; libuipc-warp's array path makes them itself, by recursive inertial
bisection of every connected cloth, soft body and spring network at its rest
positions. The preconditioner is libuipc's on another partition: the PCG
iterations differ from libuipc's, and the result is the same to the tolerance of
the linear solve. Particles on no element keep their 3 x 3 blocks.

.. _ipc-worlds:

Worlds
------

Particles, bodies and colliding shapes in more than one local world are
simulated as independent worlds, Newton's meaning of worlds: every world takes
the step it would take alone, with its own Newton iterations, line searches and
step lengths, convergence test, linear solves and status, as libuipc-warp's
independent worlds solve them (libuipc-warp's own; libuipc solves the subscenes
of a scene together). Every local world is a libuipc subscene of its own and the
global entities (world -1) one more, which meets every world, as Newton's
collision rules say: no contact pair joins two worlds, also where the worlds
overlap in space, and a global ground or static shape acts in every world. The
worlds share one frame graph, one collision detection and one set of contact
lists: a step runs as many Newton iterations as its slowest world, and a world
that has converged is held where it is while the others iterate. A world takes
the Newton iterations of the same world alone and agrees with it to the rounding
of the engine's sums, which add in another order than in an engine of that world
alone, as two runs of one engine already differ by the rounding of its atomics;
where a step carries rounding far, as a stiff landing does at the default
tolerances, the two part as far as a run alone parts from one started a rounding
error away.

A world that fails (a non-finite value, a contact pair at or inside its
thickness, a CCD fraction that is not positive) fails alone: it is put back at
its state at the step's start, its friction discarded and its BDF2 history
started anew, and the other worlds take their step. ``STATUS`` and
``SOLVER_RESIDUAL`` hold every world's own bits and residual at its entry; the
entries of the global world and of worlds that hold no unknowns (only static or
kinematic content) stay 0. A step that is not captured and fails in some worlds
writes ``state_out``, those worlds at the step's start and the others at their
step, and raises :class:`RuntimeError` naming the worlds and their bits;
:meth:`~newton.solvers.SolverIPC.check_status` names them too. The worlds share
the contact lists, and every world has a quota of them, sized as for that world
alone (libuipc-warp's capacity per world): a world whose contact asks for more
than its quota fails alone in the same way, while the other worlds keep every
contact pair and take their step. An eager step then grows that world's quota
and solves the step again for every world; a captured step sets ``OVERFLOW`` in
that world's entry alone, and :meth:`~newton.solvers.SolverIPC.grow` grows its
quota. A reset with a ``world_mask``, or a new state written for one world,
discards that world's friction and starts its BDF2 history anew and leaves the
other worlds as they were; the mask's global entry selects nothing, since the
global entities are no unknowns. :meth:`~newton.solvers.SolverIPC.frame_stats`
gives the counts of the step's loops: the Newton iterations of its slowest
world. With ``adaptive_kappa=True`` every world takes libuipc's stiffness range
of the same world alone, from the mean mass and the extent of its own vertices
and the global ones (libuipc-warp's stiffness per world): its own lower bound
with ``kappa=None``, a given ``kappa`` clamped into its own range, at every
step's ``dt``; :meth:`~newton.solvers.SolverIPC.contact_stiffness` gives every
world's.

Global entities (world -1) must not move, because every world would push them
and so join the worlds. The solver refuses with :class:`NotImplementedError` a
global free particle (one with mass that is
:attr:`~newton.ParticleFlags.ACTIVE`), a global dynamic body, a global proxy of
a coupled solver, and a global kinematic body that an explicit
``kinematic_motion="aim"`` would move toward an aim (one with mass and colliding
shapes; the default takes the jump for it, as a fixed body). Newton's implicit MPM refuses global colliders backed by dynamic bodies
with ``separate_worlds`` in the same way. A triangle, tetrahedron or spring that
joins two worlds is refused too. Global static shapes and planes, kinematic
particles and kinematic bodies that jump act in every world. Not offered with independent worlds yet, where two worlds or
more move, and refused naming what to use instead:
``contact_constitution="al-ipc"`` and ``fem_preconditioner="mas"`` (its clusters
may join two worlds). The engine's Newton exit check, which independent worlds
do not take either, is not used by the solver.

With ``coupled_worlds=True`` the worlds are one solve instead, with the same
subscenes: one step length (the smallest time of impact over all worlds, and a
line search on the sum of their energies), one Newton convergence test (the
largest displacement over all worlds), one PCG solve and one set of contact
lists. A world's result therefore depends on the others through the iteration
counts and the step lengths, and a failure or an overflow in one world fails the
step in every world, whose entries all get the bits of the one solve. Copies of
one world take the iterations of one copy, and each follows the run of that
world alone to the rounding of the shared sums. Global particles and bodies may
move, as unknowns of the one solve that meet every world, and the options above
are taken. A reset of one world, or a new state written for one world, discards
the friction of that world's contacts and starts its BDF2 history anew, and
leaves every other world's friction and history as they were, bit for bit
(libuipc-warp's requests per subscene; libuipc discards friction for a whole
scene); a reset of the global entities discards the friction of every pair with
a global entity; with ``adaptive_kappa`` the stiffness range is that of the
whole scene.

A model with content in one local world is solved as one world either way. The
engine takes up to 32,768 subscenes, and contact elements times the subscene
count rounded up to a power of two below :math:`2^{31}`; the start check forms
no pair of parts of two worlds.

.. _ipc-kinematic-bodies:

Kinematic bodies
----------------

A kinematic body (:attr:`~newton.BodyFlags.KINEMATIC`) follows the poses that
the caller writes into ``state_in``; forces and contacts do not change its
motion. The solver moves it in one of two ways, chosen by ``kinematic_motion``:
by an aim inside the step, or by a jump before it. The default, ``None``, picks
``"aim"`` on the barrier pipeline when every kinematic body with mass and
colliding shapes can aim, and ``"jump"`` otherwise: under AL-IPC, with
independent worlds when such a body belongs to the global world, and when such
a body has a singular inertia. :attr:`~newton.solvers.SolverIPC.kinematic_motion`
gives the mode a solver took.

With ``kinematic_motion="aim"``, a kinematic body with mass and colliding shapes
moves during the step. Its aim is the pose of ``state_in``
advanced over the step by the twist (``body_qd``) of ``state_in``. The step
pulls the body toward its aim with libuipc's soft transform constraint
(constitution 16), an elastic pull whose strength ratio :math:`\eta` is
``kinematic_strength_ratio`` (default 100, the value libuipc's motors use).
Because the motion is part of the step, collision detection sweeps it, surfaces
stay apart, and friction acts on it: a platform sliding under a box carries the
box. A body that contact blocks stops short of its aim. Under a load :math:`F` a
body of mass :math:`m` lags its aim by about
:math:`F \Delta t^2 / ((1 + \eta) m)`, for example 2.8e-5 m for 10 N on 1 kg at
1/60 s. What the caller writes sets the aim:

- ``body_q`` and ``body_qd`` at every step: the aim is the next pose of that
  motion.
- ``body_q`` only: the aim is the written pose.
- nothing: the body moves on at its twist.

``state_out`` holds the pose the body reached and the twist of ``state_in``,
which the step passes through unchanged, as Newton's other solvers do for
kinematic bodies. Each step starts from the pose the body reached in the step
before, so the move from there to the aim is swept, even when the caller writes
a pose far away. The first step after the solver is built, and the first step
after a :meth:`~newton.solvers.SolverIPC.reset` that selects the body's world,
take the pose of ``state_in`` without sweeping; use them to place a body
elsewhere. A first step that the start check refuses (see :ref:`ipc-start`)
does not count, and the next step is the first again. A body that
``notify_model_changed`` (with ``BODY_PROPERTIES``) makes kinematic ignores the
twist the solver wrote for it while it was dynamic: it stays where it is, as in
Newton's other solvers, until the caller writes a twist. Two shapes that are
each kinematic or static never collide, as Newton's collision pipeline can
filter such pairs (``include_static_kinematic_pairs=False``); the solver checks
its pairs against :attr:`~newton.Model.shape_contact_pairs` without them.

An explicit ``kinematic_motion="aim"`` is refused, naming the body, in the
cases where the default takes the jump:

- with ``contact_constitution="al-ipc"``, because libuipc's AL-IPC frame checks
  animated aims with a test of its own that the solver does not run (see
  :ref:`ipc-al-ipc`);
- with independent worlds, for a body of the global world, which every world
  would push (see :ref:`ipc-worlds`);
- for a body with mass but a singular inertia, which the aim cannot turn.

With ``kinematic_motion="jump"`` a kinematic body jumps: before the step the
solver writes the pose of ``state_in`` into a fixed body. Kinematic particles
always jump, because a soft aim would let a pinned vertex sag. So do kinematic
bodies without colliding shapes, and those of mass 0, such as a link of density
0 that anchors joints, because the aim has no mass to pull. Before the step the
solver sweeps every jump with continuous collision detection, from where the
last step left the body to its new pose (libuipc-warp's
``EngineOptions(sweep_kinematic=True)``). A jump that meets a surface fails the
step with :attr:`~newton.solvers.SolverIPC.StatusBits.INVALID_CCD_STEP` before
the solve runs: an eager step raises, and a captured step sets the bit. Without
this sweep, a box resting on a platform that rose at 0.5 m/s, 120 steps per
second, fell through it. A jump that meets no surface is taken, but it is not
part of the step's motion: the barrier has one step to push away what the jump
brought close, and friction does not act on the jump (a platform sliding under
a box does not carry it). The first step after the solver is built, the step
after a :meth:`~newton.solvers.SolverIPC.reset` that selects the body's world,
and the step after a failed step take the new poses without a sweep. Under
AL-IPC the step's collision-free sweeps carry the jump and push what it meets
(see :ref:`ipc-al-ipc`). Newton's test suite gives the same results with either
mode.

.. _ipc-attachments:

Attachments
-----------

A particle-body attachment
(:meth:`~newton.ModelBuilder.add_attachment_particle_body`) is libuipc's soft
vertex stitch (constitution 22) of rest length 0 between the particle and an
anchor of the body's affine body at ``attachment_particle_body_body_point`` in
Newton's body frame: a point that moves with the body and is no contact
primitive, as the interior vertices of an affine-body mesh are in libuipc;
attachments at one point of one body share it. Its energy in the step's
incremental potential is :math:`\tfrac{1}{2} k \Delta t^2 |x_p - x_a|^2`, the
spring :math:`\tfrac{1}{2} k |x_p - x_a|^2` of implicit Euler with :math:`k` =
``attachment_particle_body_stiffness``, so that the particle and the body take
equal and opposite forces, the body with the torque the anchor's offset gives,
and a disabled attachment (``attachment_particle_body_enabled``) takes none. A
hard attachment (infinite stiffness), which libuipc has no constitution for,
takes the penalty of a ball joint (see :ref:`ipc-joints`): :math:`K/\Delta t^2`
with :math:`K` = ``joint_strength_ratio`` times the masses of the particle, if
it is not fixed, and of the body, if it is dynamic, so that it holds a load
:math:`F` within :math:`F \Delta t^2 / K`. The stiffness and the enabled flags
are read at every step: a captured step takes new values at its next replay. A
spring of rest length 0 is the same stitch between its two particles, with
``spring_stiffness``. Damping is rejected: libuipc's stitch has none, and an
implicit damping cannot be folded into it the way the joint drives fold theirs.
An attached particle does not collide with the body it is attached to, whether
or not the attachment is enabled. So a particle may be attached at a point on
the body's surface, where the start check would otherwise refuse it for
starting in contact (see :ref:`ipc-start`). The edges and triangles of cloth or
a soft body that hold such a particle do not collide with that body either,
because libuipc lets a primitive collide only where each of its vertices may.
The particle collides with every other body and shape as before. Newton's other
solvers keep this contact, which does them no harm.

.. _ipc-rigid-bodies:

Rigid bodies
------------

Each Newton body becomes a libuipc affine body with 12 degrees of freedom
:math:`q = (p, a_1, a_2, a_3)`, the translation and the rows of :math:`A`, in
Newton's body frame: a point with body-frame coordinates :math:`\bar x` sits at
:math:`x = A \bar x + p`, starting from :math:`A = R(q_{body})` and :math:`p` =
the body origin. Its kinetic term uses libuipc's dyadic mass, built from
:attr:`~newton.Model.body_mass` :math:`m`, :attr:`~newton.Model.body_com`
:math:`c` and :attr:`~newton.Model.body_inertia` :math:`I` (which Newton
computes from the shapes' densities): :math:`\sum m \bar x = m c` and
:math:`\sum m \bar x \bar x^T = \tfrac{1}{2}\mathrm{tr}(I)\,1 - I + m c c^T`, so
the affine kinetic energy equals the rigid one for every rigid motion. That
second moment about the centre of mass, :math:`\tfrac{1}{2}\mathrm{tr}(I)\,1 - I`,
is positive definite where the largest principal moment of :math:`I` is below
the sum of the other two. An inertia whose largest moment is at or above that
sum (a mass distribution with no extent across a plane, or none at all, past
the triangle inequality) takes the correction
:meth:`~newton.ModelBuilder.finalize` makes by default past the inequality
(:attr:`~newton.ModelBuilder.balance_inertia`): every principal moment raised by
the difference and 1e-6 kg m², with a :class:`~newton.NewtonWarning`. A model
holds such an inertia only when it was set in place, built without that
correction, or flat. A dynamic body's inertia must be positive definite. The
orthogonality potential :math:`\kappa V \Delta t^2 \|A A^T - I\|_F^2` with the
stiffness :math:`\kappa` (below) and the volume :math:`V` of the body's
collision meshes keeps :math:`A` within a small strain of a rotation; this is
libuipc's near-rigid body, not an exactly rigid one. In libuipc's step
(``body_rotation="linear"``) a turning body moves along a straight line in its
12 coordinates within a step and the orthogonality potential pulls it back onto
the rotations, so under implicit Euler a free spin at :math:`\omega` loses about
:math:`(\omega \Delta t)^2` of its angular velocity per step: 1e-4 at 1 rad/s
and :math:`\Delta t` = 0.01 s, and a body tumbling about no principal axis loses
angular momentum and energy alike. By default under implicit Euler
(``body_rotation="rigid"``, libuipc-warp's ``affine_rotation="rigid"``) every
free body is predicted instead by the flight of a rigid body, which keeps its
angular momentum :math:`L` about the centre of mass and turns it at the implicit
midpoint of the free rigid body, and its velocity after the step carries that
momentum plus the angular impulse of the step's contact and joint forces. A free
body then keeps its angular momentum and kinetic energy, and a spin about a
principal axis, or of a body with equal moments, is the exact motion; a torque
spins a body up by :math:`\tau / I` per second, which libuipc's step falls short
of by the same loss. A pendulum on a hinge away from its centre of mass keeps
implicit Euler's loss of its centre of mass's circular motion, and the Newton
loop's displacement test leaves a little of each turn at the default tolerances
(a 20 cm cube at 2.24 rad/s keeps 0.995 of its spin over 200 steps of 0.01 s,
0.910 in libuipc's step). The body's twist then holds the angular velocity that
carries its angular momentum, :math:`\omega = (R I R^T)^{-1} L`, rather than the
skew part of :math:`\dot A A^{-1}`, which also counts the spin of the body's
strain. Fixed, kinematic and massless bodies keep libuipc's prediction, and a
body whose rows of :math:`A` have no velocity steps bit for bit as in libuipc's
step. Under ``integrator="bdf2"`` the default is ``"linear"``: BDF2 predicts
from the step before as well, which the flight does not turn, and
``body_rotation="rigid"`` with it raises ``ValueError``.

The stiffness is ``body_kappa`` at every step when it is given. By default it is
libuipc's 1e8 Pa, lowered for a body and a step to
:math:`\kappa = 25\, I_{max} / (V \Delta t^2)` where that is lower,
:math:`I_{max}` being the body's largest principal moment of inertia. The
straight line to a turn :math:`\theta` stretches the body by about
:math:`\theta^2 / 2`, and the first Newton iteration of a step takes the turn in
full while that stretch's energy, :math:`2 \theta^4 \kappa V \Delta t^2`, stays
below the inertia term :math:`\theta^2 I / 2` of the incremental potential that
it removes, :math:`I` being the moment of inertia about the axis of the turn.
With the lowered stiffness the first iteration takes a turn of up to 0.1 rad
about the axis of the largest moment and of up to :math:`0.1 \sqrt{I / I_{max}}`
rad about an axis of moment :math:`I`: 0.1 rad about any axis of a compact body,
less about the long axis of a thin one, which keeps its stiffness along its
length. With a stiffer body the line search cuts the iteration short, and the
next Newton directions, held back by the stretched potential, become small
enough to pass the displacement test of the Newton loop long before the turn is
complete: at 1e8 Pa a 1 m cube that a torque turns by 0.06 rad in a 0.1 s step
ends at 31% of its angular velocity, and a 10 cm cube spinning at 0.6 rad/s
keeps 33% of it over six steps of 1/60 s, where the converged solve keeps 99.9%.
The lowered stiffness costs rigidity in proportion to :math:`\Delta t^2` where
it applies: the lowest of five 10 cm cubes stacked on the ground is compressed
by 4e-5 at :math:`\Delta t` = 1/600 s and by 4e-3 at 1/60 s (6e-6 at 1e8 Pa), a
30 cm by 1 cm finger swinging from a hinge stretches by up to 8e-6 and 7e-4, and
a box of unequal sides that a torque starts turning runs ahead of a rigid one by
7e-4 of its turn after one step of 0.01 s and by 1e-5 after ten. Pass
``body_kappa=1e8`` for libuipc's stiffness at every step. Gravity acts on
:math:`p`; with the full dyadic mass this is the exact gravity force on the
affine body. An external force :math:`f` at the centre of mass is the
generalized force :math:`(f, f c^T)` on :math:`(p, A)`, and a torque
:math:`\tau` is :math:`(0, \tfrac{1}{2} [\tau]_\times A^{-T})`. The body's
collision shapes are triangulated (curved shapes with ``shape_segments``
segments around) and act as its contact surface.

After each step the solver writes ``body_q`` and ``body_qd``. :math:`A` is
projected onto its nearest rotation :math:`R` by the polar decomposition
:math:`A = R S` (from the SVD :math:`A = U \Sigma V^T`, :math:`R = U V^T`),
since a quaternion can only hold a rotation; the residual stretch :math:`S - 1`
is of the order of the orthogonality strain. The body origin is placed so that
the centre of mass stays where the affine map puts it,
:math:`x_{com} = A c + p`; the linear velocity is that point's,
:math:`\dot p + \dot A c`, and the angular velocity is the one that carries the
body's angular momentum, :math:`(R I R^T)^{-1} L`, when its rotation is a rigid
body's flight (``body_rotation="rigid"``), and the axial vector of the skew part
of :math:`\dot A A^{-1}` in libuipc's step.

.. _ipc-joints:

Joints
------

Revolute, prismatic, fixed and ball joints are libuipc's affine-body joints
(constitutions 18, 20, 25 and 26), between the affine bodies of their parent and
child, a parent of -1 being a fixed body without mass at the world's origin. A
revolute or prismatic joint holds the two points of its axis, ``joint_X_p`` and
``joint_X_c`` and one metre along ``joint_axis`` from them, together; a
prismatic joint also holds the bodies' relative rotation, a fixed joint the
anchors and the rotation, a ball joint the anchors. The rotation they hold, and
the zero of a joint's coordinate, is where the joint frames coincide, as in
Newton, whatever the bodies' poses when the solver is built.

The joints are penalties, not hard constraints: libuipc's energy
:math:`\tfrac{K}{2}|C|^2` of the joint's violation :math:`C`, a term of the
incremental potential (not multiplied by :math:`\Delta t^2`), with :math:`K` =
``joint_strength_ratio`` times the mass of the dynamic bodies the joint joins
(the world and kinematic bodies count 0). Against a translation of the child, in
a step of length :math:`\Delta t`, a revolute or prismatic joint, which holds
two points, is a spring of :math:`2K/\Delta t^2` [N/m], a fixed or ball joint
one of :math:`K/\Delta t^2`: a load that would move the child by :math:`\delta`
in one step stretches the joint by about :math:`\delta / (2\gamma)` or
:math:`\delta / \gamma`, :math:`\gamma` = ``joint_strength_ratio``. Gravity at
1/60 s and the default 100 stretch a revolute joint by 1.4e-5 m and a ball joint
by 2.7e-5 m. A larger ratio holds the joints closer and makes the linear system
stiffer.

Drives, passive springs and passive damping of a revolute or prismatic joint are
exact for the step's integrator: the joint's
:math:`\tau = k_e (q^* - q) + k_d (\dot q^* - \dot q) - k_s (q - q_r) - c \dot q`
(``joint_target_ke``, ``joint_target_kd``, ``joint_stiffness``,
``joint_rest_q``, ``joint_damping``; targets from
:attr:`~newton.Control.joint_target_q` and ``joint_target_qd``), taken at the
end of the step with :math:`\dot q = (q - q_n)/\Delta t` (BDF2:
:math:`(3 q - 4 q_n + q_{n-1}) / (2 \Delta t)`), is minus the gradient of one
quadratic potential in :math:`q`, which the solver gives libuipc's driving term
every step. The gains act whatever ``joint_target_mode`` says, as in the other
maximal-coordinate solvers.

A revolute or prismatic joint's ``joint_armature`` :math:`a`, which
Featherstone and MuJoCo add to the joint's inertia, is the kinetic term of its
coordinate in the same potential:
:math:`\tfrac{a}{2 \Delta t^2} (q - \tilde q)^2` with
:math:`\tilde q = q_n + \Delta t \dot q_n` (BDF2: 9/4 of it, with
:math:`\tilde q = \tfrac{4 q_n - q_{n-1}}{3} + \tfrac{2 \Delta t}{9} (4 \dot q_n - \dot q_{n-1})`),
as libuipc writes the inertia of every degree of freedom, :math:`\dot q_n` being
the rate the bodies' twists give at the step's start (as ``joint_qd`` is
written). The armature of a ball or free joint, an inertia about three axes or a
mass without weight, has no such potential and is rejected.

The drive's force cannot be bounded inside libuipc's driving term, a quadratic
potential. A ``joint_effort_limit`` :math:`L` below Newton's default 1e6 bounds
the force of the gains as decided at the start of every step: where
:math:`|k_e (q^* - q_n) + k_d (\dot q^* - \dot q_n)| > L`, the step takes the
constant force :math:`\pm L` (libuipc's joint torque or force, with ``joint_f``)
in place of the gains' part of the potential; the passive spring and damping
remain. The bound holds at the start of every step, which is where MuJoCo clamps
its actuators' force; a step that starts below it and ends above it, or the
reverse, takes the bound one step late, which a drive clamped at the end of the
step would not (by 1.6e-4 of the motion for a hinge driven through a step of its
target). With ``enable_joint_effort_limit=False`` such a limit on a driven
degree of freedom is rejected instead.

Limits are libuipc's cubic penalty outside
``[joint_limit_lower, joint_limit_upper]``, not Newton's linear spring: at an
overshoot :math:`\delta` it stores :math:`k_l \delta^3 / (3 \delta_r)` with
:math:`k_l` = ``joint_limit_ke`` and :math:`\delta_r` = 1e-3 (rad or m), so that
it pushes back with the spring's :math:`k_l \delta_r` at :math:`\delta_r` and
harder beyond it; a load :math:`F` rests at :math:`\sqrt{F \delta_r / k_l}` past
the limit. A ``joint_limit_ke`` of 0 holds nothing, and a bound at
:data:`newton.MAXVAL` is no bound. libuipc's limit has no damping; a joint
whose coordinate is past one of its limits at the start of a step takes the
damping
:math:`-k_{ld} \dot q` of Featherstone and SemiImplicit, :math:`k_{ld}` =
``joint_limit_kd``, in the same quadratic potential as ``joint_damping``: exact
for the step's integrator, with the velocity at the end of the step, and decided
at its start, as those solvers decide it (they also switch the drive off past a
limit, which SolverIPC does not, as MuJoCo does not). A joint that reaches its
limit within a step is damped from the next step on; a limit of
``joint_limit_ke`` 0 damps nothing. With ``enable_joint_limit_damping=False``
``joint_limit_kd`` is not read.

Newton's MJCF and USD importers turn a CONNECT or a WELD equality into a BALL or
a FIXED joint outside the articulations (a loop joint), and keep the equality
row, linked to the joint (``model.mujoco.equality_constraint_target_kind`` and
``equality_constraint_target``). SolverIPC simulates the loop joint as any other
joint and takes the row as held by it, by SolverSteel's rule: a CONNECT row
linked to a BALL joint, or a WELD row to a FIXED joint (to a BALL joint when its
``torquescale`` is 0), that is in no articulation and joins the row's two
bodies (or its body and the world). Every other enabled row raises
:class:`NotImplementedError`, when the solver is built and when
:meth:`~newton.solvers.SolverIPC.notify_model_changed` reports
``CONSTRAINT_PROPERTIES`` (a row enabled since); a row's
``equality_constraint_enabled`` does not switch its loop joint,
``joint_enabled`` does.

:attr:`~newton.Control.joint_f` of a revolute or prismatic joint is libuipc's
joint torque or force between the two bodies; of a ball joint a torque in the
parent's joint frame, on the child, the parent taking the reaction; of a free
joint a wrench at the child's centre of mass in the world frame, the parent, if
it is a body, taking the reaction. ``joint_enabled`` is read at every step, and
so are the gains, targets, springs, damping and limits: a captured step takes
new values at its next replay. A change of the joints' types, bodies, frames or
axes rebuilds the solve (``notify_model_changed`` with ``JOINT_PROPERTIES``).

After every step ``joint_q`` and ``joint_qd`` hold the coordinates
:func:`newton.eval_ik` computes from the bodies' state, for every joint, but the
position of a revolute joint, which is libuipc's angle and turns on past
:math:`\pi` as in the generalized-coordinate solvers, and of a prismatic joint,
libuipc's distance. A revolute joint's angle continues from the step before;
where ``state_in.joint_q`` is not what the solver wrote, it picks the turn of
the bodies' angle (the pose decides the angle modulo :math:`2\pi`).

.. _ipc-d6-joints:

D6 joints
~~~~~~~~~

libuipc has no D6 joint (its constitution 28 is only registered). A D6 joint
whose free axes leave locked what one of libuipc's joints locks is that joint,
exactly: no axis is a fixed joint, one
angular axis a revolute joint about it, one linear axis a prismatic joint along
it, with their drives, springs, damping, limits and ``joint_f``; three
orthonormal angular axes are a ball joint, and three orthonormal linear and
three orthonormal angular axes lock nothing, as a free joint. ``joint_q`` and
``joint_qd`` hold Newton's coordinates of the D6 joint: a revolute or prismatic
one's as above, of three angular axes the angles :func:`newton.eval_ik` takes
(the XYZ Euler angles in the basis of the first two axes), of linear axes the
projections of the joint frames' offset. ``joint_f`` of a D6 joint of three
angular axes is Featherstone's generalized force: a torque along the dual basis
of the axes as the coordinates carry them, and with linear axes a force along
them on the child's joint frame, the parent taking the reaction on the same
line. Other D6 joints (two angular axes, which lock one rotation, or linear and
angular axes together but three and three) lock what no libuipc joint locks, and
three axes that are not orthonormal have no coordinates in Newton's
decomposition: both are rejected, and so are drives, springs, damping and limits
on the D6 joints that become a ball or no joint.

.. _ipc-distance-joints:

Distance joints
~~~~~~~~~~~~~~~

libuipc has no distance joint. A distance joint whose distance has a single
value ``L`` (the two bounds of its first degree of freedom
equal and not negative) holds the origins of its joint frames ``L`` apart as
libuipc's soft vertex stitch (constitution 22) of rest length ``L`` between two
anchors of the bodies there (the world's fixed body for a parent of -1): the
penalty of a ball joint, ``K/2 (|x_c - x_p| - L)^2`` with the same ``K``, and
nothing when the joint is disabled. One whose bounds are both negative holds
nothing, as Featherstone and SemiImplicit take every distance joint. Either
takes ``joint_q``, ``joint_qd`` and ``joint_f`` as a free joint does. A range of
distances, or a single bound, is an inequality, which a stitch, a spring about
one rest length, cannot hold: rejected, and so are drives on a distance joint
(SolverXPBD reads its gains as the compliance of the distance).

.. _ipc-refused-joints:

Refused joints
~~~~~~~~~~~~~~

Rejected (:class:`NotImplementedError`), each for the libuipc constitution it
lacks: distance joints and D6 joints but those above, which libuipc does not
have, and rod joints, which no libuipc constitution between two affine bodies
represents; drives, springs, damping and limits on ball and free joints, since
libuipc's driving term and limits act on one hinge angle or one distance and its
ball joint has neither (its ball driving term, constitution 27, is only
registered); a negative ``joint_armature``, and one other than 0 on a ball or
free joint, an inertia about three axes or a mass without weight, which
libuipc's driving term of one coordinate cannot hold; ``joint_friction`` other
than 0, since libuipc's joints have no friction and dry friction, a bounded
force, is no quadratic potential; a negative ``joint_effort_limit`` on a driven
degree of freedom, and with ``enable_joint_effort_limit=False`` one below
Newton's default 1e6, since the quadratic drive cannot bound its force.

.. _ipc-state:

State
-----

The solver keeps positions and velocities in float64 and reads ``state_in`` on
every step. A particle whose ``state_in`` position and velocity are the float32
rounding of the solver's state, and a body whose pose and twist are what the
solver last wrote, keep the float64 state; anything else is taken from
``state_in``, a body as the rigid motion of its pose and twist. Taking a new
state for a particle or body that is not kinematic discards the friction of that
step's contacts (with several worlds, the contacts of its world), as libuipc
does after a state is written and as a solver started from that state does.
:meth:`~newton.solvers.SolverIPC.reset` does the same without a new state.
Output is written to ``state_out`` in float32.

The solve is built at the model's state: :attr:`~newton.Model.body_q` and
``body_qd``, but a body whose pose there is the identity, where
:meth:`~newton.ModelBuilder.add_link` and :meth:`~newton.ModelBuilder.add_body`
put a body given no transform, starts where :func:`newton.eval_fk` puts it from
:attr:`~newton.Model.joint_q` and :attr:`~newton.Model.joint_qd`. As for
Newton's other maximal-coordinate solvers, the caller writes those poses into
the state before the first step, also after building the solver; the model of an
articulation built without poses keeps the identity until then, and its links
may cross each other or the ground there. The start is checked where it is known
(see :ref:`ipc-start`): on the model's state, and for the pairs with a kinematic
body or such a link, whose start the caller writes, on the first step's
``state_in``.

.. _ipc-status:

Status
------

Request :attr:`~newton.solvers.SolverIPC.ObservableKind.STATUS` and
``ObservableKind.SOLVER_RESIDUAL`` (not with ``contact_constitution="al-ipc"``)
with :meth:`~newton.solvers.SolverIPC.observables` and pass the container to
:meth:`~newton.solvers.SolverIPC.step`: the step writes a
:class:`~newton.solvers.SolverIPC.StatusBits` mask and the residual on the
device, also during graph replay. With independent worlds every world that holds
unknowns receives its own (see :ref:`ipc-worlds`); with one world, or
``coupled_worlds``, every world that owns particles or bodies receives the
values of the one solve. The residual is the convergence measure of libuipc's
Newton loop: the largest displacement of the last Newton direction (the rows of
an affine body's :math:`A` scaled by
``newton_velocity_tol / newton_transrate_tol``) over
``newton_velocity_tol * dt``. It is dimensionless, at most one when the loop
ended on its displacement test, may be above one when the semi-implicit test
ended it, and infinite when the step failed on a value in
:class:`~newton.solvers.SolverIPC.StatusBits`. A later libuipc-warp may set bits
of its own above those of :attr:`~newton.solvers.SolverIPC.StatusBits.KNOWN`
(its ``FrameStatusFlags``): such a bit does not fail the step, and
:meth:`~newton.solvers.SolverIPC.check_status` warns, naming it as libuipc-warp
does.

A step that is not captured reads the status itself after the solve, with one
synchronization. When a contact list overflowed (with independent worlds, a
world's quota of the lists) it enlarges the lists and solves the step again;
when the step failed otherwise (a non-finite value, a contact starting at or
inside its thickness, an exhausted BVH traversal), it raises
:class:`RuntimeError` and leaves ``state_out`` unwritten, but where independent
worlds failed alone: there it writes ``state_out`` first, the failed worlds at
the step's start and the others at their step (see :ref:`ipc-worlds`). A
captured step cannot do either: it sets the bit, and nothing else reports the
failure. A caller that captures :meth:`~newton.solvers.SolverIPC.step` must
therefore request ``STATUS``, pass one container to each captured step (a
container holds the bits of the last step that wrote it), and after every replay
call :meth:`~newton.solvers.SolverIPC.check_status` with each container. It
raises :class:`RuntimeError` on a failed step and warns on an iteration limit;
with ``grow=True`` it answers
:attr:`~newton.solvers.SolverIPC.StatusBits.OVERFLOW` with
:meth:`~newton.solvers.SolverIPC.grow` and returns ``True``, and the caller then
restores the state the replayed steps started from, captures again and repeats
them. The PCG loop of a captured step re-reads its stopping test every 8 or 32
iterations, and work whose size only the device knows starts at a guessed size;
both are chosen when the step is captured.
:meth:`~newton.solvers.SolverIPC.graph_outdated`, called after replays, returns
``True`` when the steps replayed since asked for the other length or for other
guesses, and the caller then captures again to run faster (the old graphs stay
valid and give the same results).

.. _ipc-contact-output:

Contact output
--------------

Request ``SolverBase.ObservableKind.CONTACT_F`` with
:meth:`~newton.solvers.SolverIPC.observables`, passing the
:class:`~newton.Contacts` that sizes ``contact_f``, and pass the container and a
``Contacts`` to :meth:`~newton.solvers.SolverIPC.step`: the step writes its own
contacts into that buffer and their forces into ``contact_f``, as a native
collision backend does (SolverMuJoCo with its own collision detection). Do not
run ``CollisionPipeline.collide`` into the buffer; the step overwrites its rows.
Each contact term of the step with a force is one row: a point-triangle,
edge-edge, point-edge, point-point or vertex / half-plane pair of the final
active set (its barrier) or of the step's friction pairs (its lagged friction),
so that a pair in both gives two rows, one normal and one tangential. A term
between two shapes is a rigid row. A term between a particle feature (a
particle, or an edge or a face of cloth or tetrahedra) and a shape is a soft
row, with the particle, edge or face record of
:attr:`~newton.Contacts.soft_contact_indices` and the weights of its contact
point. Terms between particle features (cloth self-contact, particles among
themselves) are not exported, as SolverVBD exports no soft self-contact.

The forces are those at the end of the step: minus the gradient of the term's
energy in the step's incremental potential at the final positions, over
:math:`\Delta t^2`, which the step balanced to the Newton tolerance. A box
resting on the ground reports its weight. On a body of mass ``m`` whose step
ended on the displacement test the forces can be off by up to about
``m * newton_velocity_tol / dt``: a box creeping down a ramp under friction
reports its weight within 1.2% at the default tolerances and within 1e-4 with
``newton_velocity_tol=1e-4``, ``linear_tol_rate=1e-10`` and
``semi_implicit=False``. A rigid row's ``contact_f`` is the force on the body of
shape 0 by the body of shape 1, the sum of the forces on its vertices, with
their torque about that body's centre of mass at the end of the step (about the
world origin for a static shape), in the world frame; a soft row's is the force
on the shape's body by the particle feature, with its torque about that body's
centre of mass. The points are written in the body frames of ``state_out``, so
pass ``state_out`` to a consumer such as :class:`~newton.sensors.SensorContact`.
Each side's point is the force-weighted mean of its vertices, the closest point
of its primitive on the triangulated surface (a half plane's: the point under
the vertex, on the plane's surface). The normal points from shape 0 toward shape
1, a soft row's from the shape toward the particle feature: against the force on
shape 0 for a barrier term, whose force is the gradient of the distance, and
from shape 0's point to shape 1's for a friction term, whose force is
tangential. The margins are the shapes' ``shape_margin``, their contact
thickness, and the offsets the normal times each margin toward the other side,
as Newton's collision pipeline writes them. ``rigid_contact_surface_velocity``
(when allocated) and ``soft_contact_body_vel`` are zero, and
``soft_contact_rigid_indices`` is ``-1``: the solver has no surface velocity,
and it triangulates and welds the shapes' meshes. The counts are true counts:
rows past the buffer's capacity are dropped, as Newton's collision pipeline
drops them, and a ``rigid_contact_count`` or ``soft_contact_count`` above the
capacity tells it; :meth:`~newton.solvers.SolverIPC.get_max_contact_count` and
:meth:`~newton.solvers.SolverIPC.get_max_soft_contact_count` give capacities
that drop none, a worst case several times the usual count. A step that failed
writes no rows. The order of the rows changes from run to run (atomic counters),
their set and values do not. With several worlds the rows of every world share
the buffer, and the shapes tell the worlds apart. With independent worlds a step
in which some worlds failed writes no row of those worlds and the rows of the
others: libuipc-warp's records give the terms of a world that failed no force,
so the feedback forces of a coupled solver's proxies (see :ref:`ipc-coupling`)
in such a world are zero in that step, and those in the other worlds are their
step's. Not with ``contact_constitution="al-ipc"``. The output costs, only when
requested, one pass over the step's contact terms and one over its rows, and 64
bytes per slot of the contact lists.

.. _ipc-coupling:

Coupling
--------

The solver is a :class:`~newton.solvers.experimental.coupled.CouplingInterface`:
an entry of Newton's coupled solvers (``SolverCoupled``,
``SolverCoupledProxy``), built on the entry's ``ModelView``. It reads a view as
a model; three things that only a view holds take their meaning from the
coupler. A body of positive mass whose inverse mass is 0, which the view makes
of a body another entry owns, is a kinematic body: a fixed affine body whose
pose follows ``state_in``, and whose shapes, which the view hides, collide with
nothing. A triangle, tetrahedron or spring whose particles are all disabled
(mass 0 and not :attr:`~newton.ParticleFlags.ACTIVE`), as the view disables
another entry's particles, is that entry's: neither simulated nor a contact
surface here, its particles fixed points that collide with nothing. And the
view's :attr:`~newton.Model.shape_contact_pairs`, which the coupler filters by
the view's collide flags, is the set the solver's rules give (see
:ref:`ipc-contact`). An entry therefore steps its own content as the solver
alone does, to the rounding of its sums. A particle-body attachment whose
particle and body the entry owns is in its view, and the solver applies it (see
:ref:`ipc-attachments`): ``SolverCoupledADMM`` leaves such an attachment to the
entry, a hard one too, as to SolverVBD, since the solver reports
``coupling_supports_particle_body_attachments()``.

A proxy body or particle (:attr:`~newton.BodyFlags.PROXY`,
:attr:`~newton.ParticleFlags.PROXY`), another entry's that this entry moves with
the view's scaled mass, is an unknown of the step. It does not collide with
another proxy, with a static shape or with a body of inverse mass 0, as the
interface's default ``coupling_prepare_proxy_contacts`` drops Newton's contacts
of those pairs (the solver reads no :class:`~newton.Contacts`, and returns them
as they are). The coupler writes a proxy's pose and twist at every step; unlike
any other new state, that write keeps the friction of the step's contacts, where
libuipc discards friction after any write of a state: a proxy would otherwise
never have friction. The solver sweeps a proxy's new pose from where the last
step left it, as it sweeps a kinematic body's jump. A source solver that moves a
proxy into this entry's content fails the step with
:attr:`~newton.solvers.SolverIPC.StatusBits.INVALID_CCD_STEP`, because IPC does
not take a state in which surfaces cross. A proxy's feedback is the force of the step's contact terms on it at
the end of the step, a wrench about its centre of mass for a body
(``coupling_harvest_proxy_wrenches``,
``coupling_harvest_proxy_particle_forces``), as SolverVBD harvests it, rather
than its change of momentum. The barrier's force rises steeply as surfaces
approach, so the feedback is only as accurate as the step's convergence. An
entry with proxies therefore takes converged steps by default:
``newton_velocity_tol`` 5e-4 m/s, ``newton_transrate_tol`` 1e-3 1/s and
``semi_implicit=False``. These keep an impact's feedback within 1% of the
converged feedback, measured step by step for a box and a particle hitting a
box at 0.5 and 2 m/s, with steps of 1/60 and 1/240 s. They take about twice the
Newton iterations of the defaults without proxies (0.05 m/s with the
semi-implicit exit), under which an impact's feedback was up to 27 times the
converged one. The semi-implicit exit alone leaves it up to 2.5 times the
converged one at any tolerance. On cloth, two converged solves of an impact can
end in nearby minima whose feedback differs by up to 2.4%, whatever the
tolerance. Pass the three arguments to choose other values. With
``SolverCoupledProxy``'s lagged transfer, the default,
a proxy resting on this entry's content does not come to rest: the feedback of a
contact this stiff reaches the source one step late and keeps the proxy
bouncing, though its average holds the proxy up; the staggered transfer
(``mode="staggered"``) brings it to rest. Proxies are refused with
``integrator="bdf2"``, whose history the coupler's writes would leave behind,
and with ``contact_constitution="al-ipc"``, which neither sweeps a proxy's
synced pose into its active set nor exports its contact forces. The effective
masses, gravity and the rewinds of the proxies take the interface's defaults.

A coupler that solves a step more than once (a ``SolverCoupledProxy`` or
``SolverCoupledADMM`` with more than one iteration) starts every pass after the
first from the step's start. The solver keeps that start on the device:
libuipc-warp's frame restart (the float64 positions and velocities, the BDF2
history, the active contact set the step takes its friction from, the step's
requests to discard friction and restart the history, the joints' angles) and
what the solver carries from step to step (what it wrote of the bodies and
joints, the kinematic bodies it holds toward their aims, the joints' BDF2
history). The step after a coupler's write of the input state copies the start,
and the step after a restart's notification copies it back: device copies,
inside a captured graph too. Every pass therefore starts from the same float64
state with the same friction, where taking ``state_in``'s float32 rounding would
discard the friction, and a pass with the inputs of the one before repeats it,
bit for bit where a step repeats itself (where sums add with atomics, to their
rounding). With independent worlds every world starts again, also one that
failed in the pass before. An entry with several substeps per coupled step keeps
the start of the first. The copy holds the vertices' state, the active contact
lists and the joints' coordinates, and a solver whose coupler does not restart
allocates none of it. The coupler steps its entries without observables, so the
solver as an entry keeps the status bits of its steps, captured or not, with
independent worlds a word for every world entry, and
:meth:`~newton.solvers.SolverIPC.check_status` without a container reports them,
naming the worlds, and clears them; a restarted pass replaces the bits of the
pass before it, so a coupled step reports its last pass. A reset through the
coupler is :meth:`~newton.solvers.SolverIPC.reset` with the coupler's world
mask. Of a view that the coupler cannot compact, the other entries' joints are
in the view, disabled, and are checked as the solver's own.

.. _ipc-tolerances:

Tolerances
----------

A step ends once the Newton update moves no vertex by more than
``newton_velocity_tol * dt``. When a body should gain less velocity per step
than ``newton_velocity_tol`` (slow sliding against friction, for example), lower
the tolerance, or the body creeps at about ``eps_velocity`` instead.

.. _ipc-d-hat:

The barrier distance d_hat
--------------------------

With a stiff barrier (large ``kappa``) a resting contact settles just inside
``d_hat``, so objects rest up to ``d_hat`` apart; lower ``d_hat`` for tighter
contact. Without a ``d_hat`` argument the solver keeps it at or below a quarter
of the smallest half-size of a colliding shape (a sphere's or capsule's radius,
a box's smallest half extent, half of a mesh's thinnest extent) and of the
smallest radius of a free particle that collides (an active particle on no cloth
or tetrahedron, in a model whose shapes collide with particles or whose particles
collide with each other), and half the median rest length of the edges of the
cloth triangles and tetrahedra, chosen whenever the solve is built: a free
particle of radius 1 mm rests less than 0.25 mm above the ground, where libuipc's
0.01 m would rest it ten radii above. A ``d_hat`` that spans a small shape lets it rest
up to its own size away and puts more contact pairs inside ``d_hat`` than the
contact lists reserve, which a captured step reports as
:attr:`~newton.solvers.SolverIPC.StatusBits.OVERFLOW`; one as long as the edges
of a cloth or soft body puts its non-adjacent primitives inside ``d_hat`` at
rest, and the barrier stretches it. A smaller ``d_hat`` also leaves less room
for a kinematic jump, which fails the step when it meets a surface (see
:ref:`ipc-kinematic-bodies`).

.. _ipc-start:

The start
---------

The barrier needs a gap at the start. When the surfaces of different bodies
cross each other, or are at or inside their contact thickness (up to the
rounding of float32 positions), at the start, the solver raises
:class:`ValueError` and names the closest pair, as libuipc's sanity check
rejects such a scene at init. The start is the state the first step takes. Where
the model holds it, the solver checks it at construction, on the model's state
(with forward kinematics for bodies at the identity, see :ref:`ipc-state`). A
kinematic body, whose pose the caller writes at every step, and a link that the
model leaves at the identity, which the caller places by writing forward
kinematics into the state, start where the caller puts them: the pairs with such
a body are checked by the first step on its ``state_in``, before anything runs
from it. A first step captured in a graph cannot read its ``state_in``: it
raises when such a pair fails in the model's state, and otherwise takes its
start as it is; take the first step outside graph capture to have that start
checked. The thickness of a pair is the sum of its two sides' (see
:ref:`ipc-contact`). Surfaces that never move may touch or cross, and pairs that
do not collide may overlap; the inside of a body is not checked. A gap inside
``d_hat`` is valid, but there the barrier starts compressed and pushes the
surfaces apart in the first steps. The check compares the bodies whose bounding
boxes come that close, every primitive of one against every primitive of the
other, and skips a pair of bodies with more than ``2**31 - 1`` primitive pairs
with a warning. The solver does not run this check on states written after
the first step. Instead, each step sweeps the jumps of kinematic bodies and
particles and the new poses of a coupled solver's proxies (see
:ref:`ipc-kinematic-bodies`), and a step that starts with a pair at or inside
its contact thickness fails
(:attr:`~newton.solvers.SolverIPC.StatusBits.THICKNESS_VIOLATED` or
:attr:`~newton.solvers.SolverIPC.StatusBits.INVALID_CCD_STEP`). Any other state
written into ``state_in``, such as a dynamic body or particle put elsewhere, is
taken as it is. If it puts two surfaces through each other by more than their
contact thickness, neither bit is set, and the barrier, which acts on the
unsigned distance, pushes them further in.
