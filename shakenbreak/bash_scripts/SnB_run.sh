#!/bin/bash
# Optional flags:
single_defect=true
verbose=false
while getopts ":av" opt; do  # Only loop over distortion folders in the current directory (run within a defect folder)
    case $opt in
        a) single_defect=false ;;
        v) verbose=true ;;
        *) echo 'Error in command line parsing' >&2
           exit 1
    esac
done
shift $(( OPTIND - 1 ))

# User Options:
job_submit_command=${1:-"qsub"}  # "sbatch" for Slurm
job_filename=${2:-"job"}  # jobscript name
job_name_option=${3:-"-N"}  # job name option

DIR="$(dirname "${BASH_SOURCE[0]}")"
if [ $single_defect = true ]; then
    defect_name=${PWD##*/}/ # Use current directory name
fi

# Job parsing and running script
if shopt -q extglob
  then shopt_enabled=true
else
  shopt -s extglob  # ensure extended globbing (pattern matching) is enabled
fi

if [ ! -f "$job_filename" ]
  then echo "Job file '$job_filename' not in current directory, so will only submit jobs in folders with '$job_filename' present"
  job_in_cwd=false
fi

SnB_run_loop () {
  for i in ?(*Distortion*|*Unperturbed|*attled)/; # for each distortion
    do
      if [ "$i" == "?(*Distortion*|*Unperturbed|*attled)/" ]
        then echo "No distortion folders found in current directory"
        break  # exit if no distortion folders found
      fi
      if [ ! -f "${i}"/OUTCAR ] || ! grep -q "required accuracy" "${i}"/OUTCAR; # check calculation fully relaxed and finished
        then
        builtin cd "$i" || return
        if [ ! -f "${job_filename}" ] && [ ! "$job_in_cwd" = false ]
          then "cp" ../../"${job_filename}" . 2>/dev/null || "cp" ../"${job_filename}" . || return
        fi
        if [ -f OUTCAR ]  # if OUTCAR exists so rerunning rather than 1st run
          then echo "${i%?} not (fully) relaxed, saving files and rerunning"
          # shellcheck disable=SC2093
          bash "${DIR}"/save_vasp_files.sh
          "cp" CONTCAR POSCAR
          # sed -i 's/IBRION.*/IBRION = 1/g' INCAR # sometimes helps to change IBRION if relaxation not converging
        fi
        if [ -f "${job_filename}" ]
          then echo "Running job for ${i%?}"
          folder_shortname="${i##*_}"
          ${job_submit_command} "${job_name_option}" "${defect_name%?}"_"${folder_shortname%?}" "${job_filename}"
        fi
        builtin cd .. || return
      else
        if [ "$verbose" = true ]
          then echo "${i%?} fully relaxed"
        fi
      fi;
    done
}

if [ "$single_defect" = false ]
  then
  # for each defect, in the directory where your top-level defect folders are
  for defect_name in *[0-9]/;  # matches each subdirectory with a name ending in a number (charge state)
    do
    if [ "$defect_name" == "*[0-9]/" ]
        then echo "No defect folders (with names ending in a number (charge state)) found in current directory"
        break  # exit if no defect folders found
    fi
    echo "Looping through distortion folders for ${defect_name%?}"
    builtin cd "${defect_name}" || return
    SnB_run_loop
    cd .. || return
    done
else
  SnB_run_loop
fi

if [ ! $shopt_enabled = true ]
  then shopt -u extglob  # disable extended globbing (pattern matching) if it was disabled at start
fi