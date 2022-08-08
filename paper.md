---
title: 'ShakeNBreak: Navigating the defect configurational landscape'
tags:
  - point defects
  - structure searching
  - ab initio
  - symmetry breaking
  - distortions
  - semiconductors
authors:
  - name: Irea Mosquera-Lois
    orcid: 0000-0001-7651-0814
    affiliation: 1
  - name: Seán R. Kavanagh
    affiliation: "1,2" # (Multiple affiliations must be quoted)
    orcid: 0000-0003-4577-9647
  - name: Aron Walsh
    affiliation: 2
  - name: David O. Scanlon
    corresponding: true
    affiliation: 1
affiliations:
 - name: Thomas Young Centre and Department of Chemistry, University
College London, 20 Gordon Street, London, WC1H 0AJ, UK
   index: 1
 - name: Thomas Young Centre and Department of Materials, Imperial
College London, Exhibition Road, London, SW7 2AZ, UK
   index: 2
date: 07 August 2022
bibliography: paper.bib
---

# Summary
Point defects are a universal feature of crystalline solids. They control the properties and performance
of most functional materials, including thermoelectrics, photovoltaics and catalysts. Their dilute concentrations
challenge their characterisation, which is often addressed by combining experimental and computational studies.
The standard modelling approach, based on local optimisation of a defect containing crystal, tends to miss the
true ground state structure, however. The initial defect configuration is often chosen as a defect
on a known crystal site with all other atoms remaining on their typical lattice positions. This structure may lie
within a local minimum or on a saddle point of the potential energy surface (PES), trapping a gradient-based
optimisation algorithm in an unstable or metastable arrangement and thus yielding incorrect configurations.

# Statement of need
To tackle this limitation, two approaches have recently been designed. Arrigoni and Madsen [@Arrigoni:2021]
developed an evolutionary algorithm combined with a machine learning model to navigate the defect configurational
landscape and identify low energy structures. While ideal to study specific defects, its complexity
and computational cost hinders its application to typical defect investigations.
Alternatively, Pickard and Needs [@Pickard:2021] employed random sampling on the atoms near the defect site -- with
the limitation that random sampling on a high-dimensional space lowers efficiency and increases computational
cost. To improve sampling efficiency, domain knowledge can be used to target the sampling structures towards likely
energy lowering distortions. This is the purpose of our package, which aims to serve as an efficient, simple and affordable
tool to identify low energy defect structures.

# ShakeNBreak
`ShakeNBreak` is a set of python modules developed to automatise the process of defect structure searching.
It makes extensive use of several open-source packages, including Python Materials Genomics (`pymatgen`) [@pymatgen]
and the Atomic Simulation Environment (`ase`) [@ase]. It supports most common *ab-initio* plane wave codes,
including VASP, CP2K, Quantum Espresso, CASTEP and FHI-aims. In addition to a Python API, `ShakeNBreak`
provides a command line interface, and is thus trivial to use. Its main functionality entails:
* Automatised application of chemically-guided distortions to all input defects, generating the structures to sample the PES.
  Optionally, it can also generate the calculation input files for the desired code, organising them into a directory structure.
* Automatised parsing, analysis and plotting of the geometry optimisation results, to quickly identify any energy-lowering
distortions as well as the physico-chemical factors driving them.

# Acknowledgements
I.M.L. thanks La Caixa Foundation for funding a postgraduate scholarship (ID 100010434, fellowship code
LCF/BQ/EU20/11810070). S.R.K. acknowledges the EPSRC Centre for Doctoral Training in the Advanced
Characterisation of Materials (CDTACM)(EP/S023259/1) for funding a PhD studentship. DOS acknowledges
support from the EPSRC (EP/N01572X/1) and from the European Research Council, ERC (Grant No. 758345).
Via membership of the UK's HEC Materials Chemistry Consortium, which is funded by the EPSRC
(EP/L000202EP/R029431, EP/T022213), this work used the UK Materials and Molecular Modelling (MMM) Hub
 (Thomas EP/P020194 and Young EP/T022213).

# References