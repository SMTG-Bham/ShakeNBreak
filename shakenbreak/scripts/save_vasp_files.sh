#!/bin/bash
# Script to save important VASP output files for future reference,
# likely before beginning a continuation job in the same directory.

current_time=$(date +%H_%M_%Son%d_%m_%y)
for i in {CONTCAR,OUTCAR,XDATCAR,POSCAR,INCAR,OSZICAR,vasprun.xml}; do
  cp $i ${i}_${current_time}
done
gzip vasprun.xml_${current_time} # gzip to save file space
