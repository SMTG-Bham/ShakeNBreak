---
title: 'ShakeNBreak: Navigating the defect configurational landscape'
tags:
  - point defects
  - structure searching
  - symmetry breaking
  - distortions
  - semiconductors
  - ab initio
authors:
  - name: Irea Mosquera-Lois
    equal-contrib: true
    orcid: 0000-0001-7651-0814
    affiliation: 1
  - name: Seán R. Kavanagh
    equal-contrib: true
    corresponding: true
    affiliation: 1, 2
    orcid: 0000-0003-4577-9647
  - name: Aron Walsh
    affiliation: 2
    orcid: 0000-0001-5460-7033
  - name: David O. Scanlon
    corresponding: true
    affiliation: 1
    orcid: 0000-0001-9174-8601
affiliations:
 - name: Thomas Young Centre and Department of Chemistry, University College London, United Kingdom
   index: 1
 - name: Thomas Young Centre and Department of Materials, Imperial College London, United Kingdom
   index: 2
date: 07 August 2022
bibliography: paper.bib
---

# Summary

Point defects are present in all crystalline solids, controlling the properties and performance
of most functional materials, including thermoelectrics, photovoltaics and catalysts.
However, the standard modelling approach, based on local optimisation of a defect placed on a known crystal site, can miss the true ground state structure.
This structure may lie within a local minimum of the potential energy surface (PES), trapping a gradient-based optimisation algorithm in a metastable arrangement and thus
yielding incorrect defect structures that compromise predicted properties [@Mosquera-Lois:2021]. As such, an efficient way to explore the defect energy landscape and identify low-energy structures is required.

# Statement of need

To tackle this limitation, two approaches have recently been designed. Arrigoni and Madsen [@Arrigoni:2021] developed an evolutionary algorithm combined with a machine learning model to navigate the defect configurational landscape and identify low-energy structures. While ideal to study specific defects, its complexity and computational cost hinders its application to typical defect investigations.
Alternatively, Pickard and Needs [@Pickard:2011] applied random sampling to the atoms near the defect site -- with the limitation that random sampling on a high-dimensional space lowers efficiency and increases computational cost. To improve sampling efficiency, domain knowledge can be used to tailor the sampling structures towards likely energy-lowering distortions. This is the purpose of our package, which aims to serve as a simple, efficient, and affordable tool to identify low-energy defect structures.

# ShakeNBreak

`ShakeNBreak` is a set of Python modules developed to automate the process of defect structure searching.
It makes extensive use of several open-source packages, including Python Materials Genomics (pymatgen) [@pymatgen] and the Atomic Simulation Environment (ase) [@ase]. It supports most common *ab-initio* plane wave codes, including VASP [@vasp], CP2K [@cp2k], Quantum Espresso [@espresso], CASTEP [@castep] and FHI-aims [@fhi_aims].
In combination, these features make `ShakeNBreak` compatible with the majority of defect packages such as PyCDT [@pycdt], pylada [@pylada], DASP [@dasp] and Spinney [@spinney], as well as workflow managers (FireWorks [@fireworks] and AiiDA [@aiida]).
In addition to a Python API, `ShakeNBreak` provides a command line interface, making it user-friendly and readily applicable to defect modelling workflows.

The structure search strategy is based on applying a range of chemically-guided distortions
to the high-symmetry defect configuration, yielding a set of sampling structures which are then geometrically optimised.
Although the distortions can be customised, a set of sensible defaults and informative warnings have been implemented.
Optionally, the relaxation input files can be generated for the desired *ab-initio* code and
organised into a directory structure. These processes are fully automated, requiring only a few lines of code or a single command.
Following the geometry optimisations, the results can be automatically parsed, analysed
and plotted to identify the different low-energy structures, as well as the physico-chemical factors driving the energy-lowering distortions.
Within its analysis toolbox, `ShakeNBreak` includes methods to quantify structural similarity (Figure 1a), compare the defect local environments (Figure 1c,d) and analyse site- and orbital-decomposed magnetisations (Figure 1b).

The distortion procedure, underlying rationale and its application to a wide range of semiconductors have recently been described [@Mosquera-Lois:2022]. In addition, the package has been employed to identify the defect structures reported in several studies [@Kavanagh:2021; @Kavanagh:2022], with the identified configurations having significant impact on predicted behaviour.

![Example analysis for a cadmium vacancy defect in CdTe: a) plot of final energies versus bond distortion factor, with a colorbar quantifying the structural similarity between configurations b) analysis of site magnetisations for the Unperturbed configuration, c) distances between the defect and its nearest neighbours and d) resemblance of the defect environment to difference structural motifs. \label{fig1}](../figures/Figure_joss.png)

# Acknowledgements

I.M.L. thanks La Caixa Foundation for funding a postgraduate scholarship (ID 100010434, fellowship code
LCF/BQ/EU20/11810070). S.R.K. acknowledges the EPSRC Centre for Doctoral Training in the Advanced
Characterisation of Materials (CDTACM)(EP/S023259/1) for funding a PhD studentship. DOS acknowledges
support from the EPSRC (EP/N01572X/1) and from the European Research Council, ERC (Grant No. 758345).

ShakeNBreak has benefitted from feature requests from many members of the Walsh and Scanlon research groups, including Adair Nicolson, Xinwei Wang, Katarina Brlec, Joe Willis, Zhenzhu Li, Jiayi Cen, Lavan Ganeshkumar, Daniel Sykes, Luisa Herring-Rodriguez, Alex Squires, Sabrine Hachmiouane and Chris Savory.

# References
