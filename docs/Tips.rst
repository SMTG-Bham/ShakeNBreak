Miscellaneous Tips & Tricks
============================

Tricky Relaxations
-------------------

If certain relaxations are not converging after multiple continuation calculations (i.e. if :code:`snb-run` keeps
resubmitting certain relaxations), this is likely due to an error in the underlying calculation, extreme forces and/or
small residual forces which the structure optimisation algorithm is struggling to relax. In most of these cases,
:code:`ShakeNBreak` will automatically detect and handle these calculations with :code:`snb-run`, but some tricky cases
may require manual tuning from the user:

- For :code:`VASP`, a common culprit is :code:`EDWAV` in the output file, which can typically be avoided by reducing
  :code:`NCORE` and/or :code:`KPAR`. Other errors related to forces / unreasonable interatomic distances (like
  :code:`EDDDAV` or :code:`ZPOTRF`), as well as cases with positive energies, are automatically handled by
  :code:`snb-run` (folder is renamed to :code:`Bond_Distortion_X_High_Energy` and subsequently ignored).
    - If some relaxations are still not converging after multiple continuations, you should check the calculation output
      files to see if this requires fixing. Often this may require changing a specific input file setting (e.g. in the
      :code:`INCAR` for :code:`VASP`), and using the updated setting(s) for any other relaxations which are struggling
      to converge.
- For codes other than :code:`VASP`, forces errors or high / positive energies are not automatically handled, and so in
  the rare cases where this occurs, you should rename the folder(s) to :code:`Bond_Distortion_X_High_Energy` and
  :code:`ShakeNBreak` will subsequently ignore them.

- If the calculation outputs show that the relaxation is proceeding fine, without any errors, just not converging to
  completion (i.e. residual forces), then :code:`ShakeNBreak` will consider the calculation converged if the energy is
  changing by <2 meV with >50 ionic steps. Alternatively, convergence of the forces can be aided by:
    - Switching the ionic relaxation algorithm (e.g. change :code:`IBRION` to :code:`1` or :code:`3` in :code:`VASP`)
    - Reducing the ionic step width (e.g. change :code:`POTIM` to :code:`0.02` in :code:`VASP`)
    - Switching the electronic minimisation algorithm (e.g. change :code:`ALGO` to :code:`All` in :code:`VASP`), if
      electronic concergence seems to be causing issues.
    - Tightening/reducing the electronic convergence criterion (e.g. change :code:`EDIFF` to :code:`1e-7` in :code:`VASP`)

In the other rare case where all distortions yield high energies, relative to the :code:`Unperturbed` structure, this is
typically indicative of an unreasonable defect charge state (with the extreme excess charge inducing many false local
minima on the PES). :code:`ShakeNBreak` will print a warning in these cases, with advice on how to proceed if this is
not the case and the charge state is reasonable (see below).


Hard/Ionic/Magnetic Materials
---------------------

The default bond distortion range of -60% to +60%, and rattling standard deviation (:code:`stdev` = 10% of the bulk bond
length) are reasonable choices for most materials, typically giving best performance, but in some rare cases these may
need to be adjusted. This can be the case with extremely hard/ionic/oxide materials which typically yield larger forces
in response to bond distortion. If this issue occurs, it will manifest as:

- If the rattle standard deviation is too large, it may result in high energies for each distorted & rattled structure
  (consistently higher energy than the unperturbed structure). As mentioned in :ref:`Tricky Relaxations` above,
  :code:`ShakeNBreak` will print a warning in these cases, and often it is the result of unreasonable defect charge
  states. If the calculations have finished ok and the defect charge states are reasonable, then you likely need to
  reduce the rattle standard deviation (:code:`stdev`) to 7.5% of the bulk bond length (or 5% if this still causes
  higher energies) to avoid this – if you're unsure of the bulk bond length for your material, just look at the previous
  info messages or output :code:`distortion_metadata.json` files from :code:`ShakeNBreak`, for which the default
  :code:`stdev` will be equal to 10% of the bulk bond length. If this occurs, it often indicates a complex PES with
  multiple energy minima, thus energy-lowering distortions particularly likely, so important to test these cases with
  reduced :code:`stdev`! Typically the largest rattle standard deviation for which the relaxations run without issue is
  best for performance in terms of finding groundstate structures.
    - Note that strongly-correlated / magnetic materials in particular can be extremely sensitive to large structural
      noise, and so these typically require rattle standard deviations (:code:`stdev`) ≤ 0.05 Å.

- High energies / non-converging calculations for the ±60% endpoints. As mentioned in :ref:`Tricky Relaxations` above,
  these are automatically handled by :code:`snb-run` for :code:`VASP` and so no changes are required, but for other
  codes you should rename the folder(s) to :code:`Bond_Distortion_X_High_Energy` and :code:`ShakeNBreak` will
  subsequently ignore them.
.. Here you should adjust the distortion range to exclude these points (e.g. :code:`bond_distortions = np.arange(-0.5, 0.501, 0.1)`), or just ignore these calculations.

If you are unsure but suspect this could be an issue for your material, the best strategy is typically to begin the
calculations with the default settings for one defect, then parse the results and :code:`ShakeNBreak` will warn you if
this issue is occurring – if not, all good!


Polarons
---------

For polar/ionic systems, defects are often associated with polarons (region of localised charge formed when charge
carriers interact with the lattice ions). When using ``ShakeNBreak`` to identify these localised solutions, we recommend
to:

- **Not** specify initial magnetic moments for each atom (``MAGMOM`` in ``VASP``). Generally, the lattice distortion is
  enough to localise the charge in the defect vicinity, with the advantage of less bias as the user does not have to
  specify the atoms where the charge is localised. If the defect is surrounded by two types of ions, and the charge is
  expected to preferentially localise in one of them, you can use the ``neighbour_elements`` keyword to only distort the
  specified element (see next section).

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

Bulk Phase Transformations
------------------

If you perform :code:`ShakeNBreak` calculations with a supercell structure for which a lower energy polymorph exists
(i.e. by using a bulk structure which has imaginary phonon modes), often the symmetry-breaking introduced by
:code:`ShakeNBreak` can cause your defect supercell to relax to this lower energy bulk structure. If this is the case,
it will typically cause *all* distortion calculations to yield a significantly lower energy structure than the
``Unperturbed`` relaxation, for *most* (if not all) defects calculated; as any significant symmetry-breaking in the
supercell is allowing the structure to rearrange to the lower energy polymorph. :code:`ShakeNBreak` will print a cautionary
warning in these cases, and one way to manually check this is to visually compare the relaxed structure from one of the
low energy distortion calculations, with the relaxed ``Unperturbed`` structure, and the bulk supercell, and see how the
regions away from the defect site compare.

Often this is useful information, as it may reveal a previously-unknown low-energy polymorph for your host system.
However, it also means that your original higher energy bulk structure is no longer an appropriate reference structure
for calculating your final defect formation energies, and so you should instead obtain the bulk supercell corresponding
to this lower energy polymorph, and use this as the reference structure for calculating defect formation energies.
The recommended workflow for doing this is to firstly try obtaining this lower energy polymorph in the defect-free host
system using similar atom rattling to that used in the :code:`ShakeNBreak` distorted structure generation:

..  code-block:: python

    from shakenbreak.distortions import rattle, Structure
    bulk = Structure.from_file("path/to/Bulk_Supercell_POSCAR")
    rattled_supercell = rattle(bulk)
    rattled_supercell.to(fmt="POSCAR",
                         filename="Rattled_Bulk_Supercell_POSCAR")

Then calculate the energy of this rattled supercell and compare to the original high-symmetry bulk supercell.
If it's significantly lower energy (similar to the energy difference between your ``Unperturbed`` and distorted defect
relaxations), then this is likely the lower energy polymorph for your host system.

If this does not yield a significantly lower energy polymorph, then it's recommended to calculate the phonon dispersion
of your host material, and check if there are any imaginary phonon modes (indicating the presence of a nearby
lower-symmetry lower-energy polymorph). If this is the case, then you can try to obtain this lower energy polymorph
using a code like `ModeMap <https://github.com/JMSkelton/ModeMap>`_ or similar, to generate the distorted structure
corresponding to this imaginary mode.

This workflow also serves to explicitly test if indeed a phase transformation is occurring in your defect supercell(s).
If this does indeed reveal a significantly lower energy polymorph for your host material, depending on how different this
structure is, you might want to regenerate the defects in this new supercell and possibly re-run :code:`ShakeNBreak` on
these – particularly for any defects that did not show an energy-lowering relative to ``Unperturbed`` in the original
supercell (suggesting that they remained 'stuck' in the original higher-energy arrangement).

In certain cases, it's possible that you witness behaviour similar to the above, despite no lower energy polymorph
existing for your host material. This can be the case in certain high-symmetry materials (e.g. we have witnessed this
in incipient ferroelectrics), where defects introduce strong local symmetry-breaking (i.e. a 'local phase
transformation') which is missed by the standard approach with unperturbed defect relaxations. These are cases where
:code:`ShakeNBreak` is particularly important for your material, you can continue your calculations as normal, ignoring
these warnings.


Defect Complexes
------------------

At present, :code:`ShakeNBreak` is optimised to work with isolated *point* defects. However, it can also be used with
complex defects (and has been found to be important in these cases as well, e.g. this |chemsci|_), but requires some
workarounds as the ``pymatgen`` defect functions are not natively built for this.
This involves generating the defect *complex* as a ``pymatgen`` ``Defect`` object using one of the *point*
defects as the 'bulk' structure and the other as the 'defect', then feeding this to :code:`ShakeNBreak` in order to
generate the distortions. If you are trying to do this and are running into issues, you can contact the developers and
we can share some guidance for this (until improved ``pymatgen``-based functionality comes about for complex defects).

.. _chemsci: https://doi.org/10.1039/D1SC03775G

.. |chemsci| replace:: *Chem Sci* paper


Metastable Defects
--------------------

While the :code:`ShakeNBreak` workflow is primarily geared toward ground-state structure identification, it can also be
applicable to finding metastable states, as described in the `method paper <https://www.nature.com/articles/s41524-023-00973-1>`_.
For this, you can use the optional ``metastable`` argument for ``get_energy_lowering_distortions``;
see `docs here <https://shakenbreak.readthedocs.io/en/latest/shakenbreak.energy_lowering_distortions.html#shakenbreak.energy_lowering_distortions.get_energy_lowering_distortions>`_.


Troubleshooting
-------------------

- A current known issue with ``numpy``/``pymatgen`` is that it might give an error similar to this:

  .. code:: python

      ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject

  This is due to a recent change in the ``numpy`` C API, see `here <https://stackoverflow.com/questions/66060487/valueerror-numpy-ndarray-size-changed-may-indicate-binary-incompatibility-exp>`_.
  It should be fixed by reinstalling ``pymatgen``, so that it is rebuilt with the new ``numpy`` C API:

  .. code:: bash

      pip uninstall pymatgen
      pip install pymatgen


Have any tips for users from using `ShakeNBreak`? Please share it with the developers and we'll add them here!
