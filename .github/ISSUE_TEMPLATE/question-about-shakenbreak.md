---
name: Question about ShakeNBreak
about: An issue template for general questions about ShakeNBreak
title: ''
labels: question
assignees: ''

---

For general questions related to `ShakeNBreak` and defect structure-searching, we ask that you post these on the 
[`ShakeNBreak` MatSci community forum](https://matsci.org/c/shake-n-break/71).

Before taking developer time, we kindly ask that you first check the following curated resources for answers, as many
questions have been answered before:
- [ ] [`ShakeNBreak` documentation site](https://shakenbreak.readthedocs.io/en/latest/) – tip: use the search bar on the left.
- [ ] [`ShakeNBreak` tutorials](https://shakenbreak.readthedocs.io/en/latest/Tutorials.html)
- [ ] [`ShakeNBreak` Python API docs](https://shakenbreak.readthedocs.io/en/latest/modules.html)
- [ ] [Previous `ShakeNBreak` issues](https://github.com/SMTG-Bham/ShakeNBreak/issues?q=is%3Aissue+is%3Aclosed) where the same question may have been asked – tip: use the search bar.
- [ ] All other `ShakeNBreak` docs pages; such as [Tips](https://shakenbreak.readthedocs.io/en/latest/Tips.html) and [Installation](https://shakenbreak.readthedocs.io/en/latest/Installation.html)
- [ ] [`ShakeNBreak` MatSci community forum](https://matsci.org/c/shake-n-break/71)

If your question is about general defect structure-searching methodology, please refer to the following papers, and other relevant literature:
- [Defect Structure Searching Preview](https://doi.org/10.1016/j.matt.2021.06.003)
- [Defect Structure Searching Main Paper](doi.org/10.1038/s41524-023-00973-1)
- [Free Energies of Defects](https://doi.org/10.1039/D3CS00432E)
- [Guidelines for Robust Defect Simulations](https://doi.org/10.26434/chemrxiv-2025-3lb5k)

FAQ: Why are only `Rattled` and `Unperturbed` generated for some charge states?
> `ShakeNBreak` uses the change in valence electron count (i.e. 'excess charge') at the defect site to dictate the number of bonds to distort, which was found to be the best chemically-motivated strategy; see [the npj theory paper](https://www.nature.com/articles/s41524-023-00973-1). For _fully-ionised_ charge states (e.g. `v_Cu_-1`) the excess charge is zero, and so no bonds are distorted and only rattling to break symmetry is trialled. These defect charge states are typically the most simple, and do not involve charge localisation.
