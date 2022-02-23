#!/bin/bash

shopt -s extglob # ensure extended globbing (pattern matching) is enabled

# for each defect, in the directory where your top-level defect folders are
for defect_name in *[0-9]/; # matches each subdirectory with a name ending in a number (charge state)
    do cd ${defect_name}/BDM;
    if [[ -f ${defect_name%?}.txt ]];
        then
        echo "Moving old ${defect_name%?}.txt to ${defect_name%?}_$(date +%H_%M_%Son%d_%m_%y).txt to avoid overwriting"
        mv ${defect_name%?}.txt ${defect_name%?}_$(date +%H_%M_%Son%d_%m_%y).txt
        fi

    for i in ?(*Distortion|*Unperturbed_Defect|*only_rattled)/; # for each BDM distortion
        do if grep -q "required accuracy" ${i}/vasp_gam/OUTCAR; # check calculation fully relaxed and finished
            then echo ${i%?} >> ${defect_name%?}.txt; # add BDM distortion to txt file, "%?" cuts slash at end of string
            grep -a sigma ${i}/vasp_gam/OUTCAR | tail -1 | awk '{print $NF}' >> ${defect_name%?}.txt; #and its energy
            else echo "${i%?} not fully relaxed" 
           fi;
        done;
    cd ../..;
    done
