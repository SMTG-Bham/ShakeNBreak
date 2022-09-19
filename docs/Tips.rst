Miscellaneous Tips & Tricks
============================

Tricky Relaxations
-------------------

If certain relaxations are not converging after multiple continuation calculations (i.e. if :code:`snb-run` keeps
resubmitting certain relaxations), this is likely due to an error in the underlying calculation, and so you should
check the calculation output files to see if this requires fixing. Often this may require changing a specific input file
setting (e.g. in the :code:`INCAR` for :code:`VASP`), and copying the updated input files to other directories for which
relaxations are struggling to converge.

- For :code:`VASP`, a common culprit is :code:`EDWAV` in the output file, which can typically be avoided by reducing
  :code:`NCORE` and/or :code:`KPAR`. Another is :code:`ZPOTRF` which can happen in rare cases and suggests too large
  initial forces, so typically easiest to just ignore this specific distortion.

If the calculation outputs show that the relaxation is proceeding fine, without any errors, just not converging to
completion, then other input settings such as the ionic relaxation algorithm (:code:`IBRION` in :code:`VASP`),
electronic minimisation algorithm (:code:`ALGO` in :code:`VASP`) or real space force projection (:code:`LREAL`
in :code:`VASP`) should be adjusted to aid convergence.


Hard/Ionic Materials
---------------------

The default bond distortion range of -60% to +60%, and the default rattling standard deviation of 0.25 Å, can be too
extreme in the case of hard/ionic/oxide materials which typically yield larger forces in response to bond distortion.
If this is the case for your material, it will manifest in the form of:

- High energies / unconverging calculations for the ±60% endpoints. Here you should adjust the distortion range to
  exclude these points (e.g. :code:`bond_distortions = np.arange(-0.5, 0.501, 0.1)`), or just ignore these calculations.

- If the rattle standard deviation is too large, it may result in high energies for each distorted & rattled structure
  (consistently higher energy than the unperturbed structure). In this case, you will need to reduce the rattle
  standard deviation to 0.15 Å (or 0.075 Å if this still causes higher energies) to avoid this. Typically the largest
  rattle standard deviation for which the relaxations run without issue is best for performance in terms of finding
  groundstate structures.

If you are unsure but suspect this could be an issue for your material, the best strategy is often to begin with the
default settings, and then monitor the results of the first few distortion meshes that run; if the final energies are
less than or equal to the unperturbed relaxation, then no need to change. Otherwise, you should reduce the rattle
:code:`std_dev` as above.


Polarons
---------

For polar/ionic systems, defects are often associated with polarons (region of localised charge formed when charge
carriers interact with the lattice ions). When using ``ShakeNBreak`` to identify these localised solutions, we recommend to:

- **Not** specify initial magnetic moments for each atom (``MAGMOM`` in ``VASP``). Generally, the lattice distortion is
  enough to localise the charge in the defect vicinity, with the advantage of less bias as the user does not have to specify the
  atoms where the charge is localised. If the defect is surrounded by two types of ions, and the charge is expected to preferentially
  localise in one of them, you can use the ``neighbour_elements`` keyword to only distort the specified element (see next section).

- Specify the total magnetic moment (``NUPDOWN`` in ``VASP``). We recommend to use a wider distortion mesh
  (``delta = 0.2``) and run main ``NUPDOWN`` possibilities, e.g. if there are two extra/missing electrons run
  ``NUPDOWN = 0`` (anti-parallel arrangement) and ``NUPDOWN = 2`` (parallel arrangement).

:code:`neighbour_elements` Use Cases
-------------------------------------

When generating atomic distortions with :code:`ShakeNBreak`, the :code:`neighbour_elements` optional parameter can be
particularly useful in certain cases if:

- You have a complex multi-cation / multi-anion system, and believe that the most likely distorting species about
  certain defect sites are not the nearest neighbour atoms. For example, in a rare case you might have two cations (A
  and B), where the nearest neighbours of cation A are cations B, but it is the (second-nearest-neighbour) anions which
  are likely most prone to rearrange upon formation of a A cation vacancy.

- You have 'spectator' ions (e.g. the A-site cation in ABX\ :sub:`3` perovskites) that are nearest neighbours to the
  defect, but unlikely to distort or rebond. This has been seen in studies in our research groups (reference to be
  added when preprinted).

Defect Complexes
------------------

At present, ``ShakeNBreak`` is optimised to work with isolated *point* defects. However, it can also be used with
complex defects (and has been found to be important in these cases as well, e.g. this |chemsci|_), but requires some workarounds as the ``pymatgen`` defect
functions are not natively built for this.
This involves generating the defect *complex* as a ``pymatgen`` ``Defect`` object using one of the *point*
defects as the 'bulk' structure and the other as the 'defect', then feeding this to ``ShakeNBreak`` in order to
generate the distortions. If you are trying to do this and are running into issues, you can contact the developers and
we can share some guidance for this (until improved ``pymatgen``-based functionality comes about for complex defects).

.. _chemsci: https://doi.org/10.1039/D1SC03775G

.. |chemsci| replace:: *Chem Sci* paper


Metastable Defects
--------------------

While the ``ShakeNBreak`` workflow is primarily geared toward ground-state structure identification, it can also be
applicable to finding metastable states, as described in the `method paper <https://arxiv.org/abs/2207.09862>`_.
For this, you can use the optional ``metastable`` argument for ``get_energy_lowering_distortions``;
see `docs here <https://shakenbreak.readthedocs.io/en/latest/shakenbreak.energy_lowering_distortions.html#shakenbreak.energy_lowering_distortions.get_energy_lowering_distortions>`_.

Have any tips for users from using `ShakeNBreak`? Please share it with the developers and we'll add them here!
