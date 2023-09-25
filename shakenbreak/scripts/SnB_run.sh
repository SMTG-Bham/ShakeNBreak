#!/bin/bash
# Optional flags:
single_defect=true
verbose=false
while getopts ":av" opt; do # Only loop over distortion folders in the current directory (run within a defect folder)
  case $opt in
  a) single_defect=false ;;
  v) verbose=true ;;
  *)
    echo 'Error in command line parsing' >&2
    exit 1
    ;;
  esac
done
shift $((OPTIND - 1))

# User Options:
job_submit_command=${1:-"qsub"}            # "sbatch" for Slurm
job_filepath=${2:-"job"}                   # jobscript path
job_filename=$(basename "${job_filepath}") # jobscript name
job_name_option=${3:-"-N"}                 # job name option

DIR="$(dirname "${BASH_SOURCE[0]}")"
if [ $single_defect = true ]; then
  defect_name=${PWD##*/}/ # Use current directory name
fi

# Job parsing and running script
if shopt -q extglob; then
  shopt_enabled=true
else
  shopt -s extglob # ensure extended globbing (pattern matching) is enabled
fi

if [ ! -f "$job_filepath" ]; then
  echo "Job file '$job_filepath' not found, so will only submit jobs in folders with '$job_filename' present"
  job_in_cwd=false
fi

check_multiple_single_step_outcars() {
    # Counter for files matching the condition
    counter=0

    # Iterate over all OUTCAR*on* files in the current directory
    for file in OUTCAR*on*
    do
        # Check if file is a regular file
        if [[ -f $file ]]; then
            # Count 'entropy=' occurrences
            count=$(grep -c 'entropy=' "$file")

            # Check if count is less than or equal to 1
            if [[ $count -le 1 ]]; then
                ((counter++))  # increment counter
            fi
        fi
    done

    # Return 0 if multiple files matching the condition exist, else return 1
    if [[ $counter -gt 1 ]]; then
        return 0
    else
        return 1
    fi
}


SnB_run_loop() {
  for i in ?(*Distortion*|*Unperturbed*|*attled*)/; do # for each distortion
    if [ "$i" == "?(*Distortion*|*Unperturbed*|*attled*)/" ]; then
      echo "No distortion folders found in current directory"
      break # exit if no distortion folders found
    fi
    # if "_High_Energy" is in the folder name, skip it
    if [[ "$i" == *"_High_Energy"* ]]; then
      continue
    fi
    if [ ! -f "${i}"/OUTCAR ] || ! grep -q "required accuracy" "${i}"/OUTCAR; then # check calculation fully relaxed and finished
      builtin cd "$i" || return
      if [ ! -f "${job_filepath}" ] && [ ! "$job_in_cwd" = false ]; then
        "cp" ../"${job_filepath}" "./${job_filename}" 2>/dev/null  || "cp" ../../"${job_filepath}" "./${job_filename}" 2>/dev/null || return
      fi
      if [ -f OUTCAR ]; then # if OUTCAR exists so rerunning rather than 1st run
        # count number of ionic steps with positive energies, after the first 5 ionic steps
        pos_energies=$(grep entropy= OUTCAR | awk 'FNR>5 && $NF !~ /^-/{print $0}' | wc -l)
        errors=$(grep -Ec "(EDDDAV|ZHEGV|CNORMN|ZPOTRF|ZTRTRI|FEXC)" OUTCAR)
        if ((pos_energies > 0)) || ((errors > 0)); then # if there are positive energies or errors in OUTCAR
          if [[ "$i" == *"Unperturbed"* ]]; then
            # positive energies / errors for Unperturbed structure, indicates pathological defect structure
            echo "Positive energies or forces error encountered for ${i%/}. "
            echo "This typically indicates the initial defect structure supplied to ShakeNBreak is highly unstable, often with bond lengths smaller than the ionic radii."
            echo "Please check this defect structure and/or the relaxation output files."
            continue
          else
            echo "Positive energies or forces error encountered for ${i%/}, ignoring and renaming to ${i%/}_High_Energy"
            builtin cd .. || return
            mv "${i%/}" "${i%/}_High_Energy"
            continue
          fi
        fi

        num_energies=$(grep -c entropy= OUTCAR)
        if ((num_energies > 50)); then
          init_energy=$(grep entropy= OUTCAR | awk '{print $NF}' | head -1)
          fin_energy=$(grep entropy= OUTCAR | awk '{print $NF}' | tail -1)
          energy_diff=$(echo "$init_energy - $fin_energy" | bc)

          # if there are more than 50 ionic steps and the final energy is less than 2 meV lower than the
          # initial energy then calculation is essentially converged, don't rerun
          if (($(echo "${energy_diff#-} < 0.002" | bc -l))); then
            if [ "$verbose" = true ]; then
              echo "${i%?} has some (small) residual forces but energy converged to < 2 meV, considering this converged."
            fi
            echo "ShakeNBreak: At least 50 ionic steps and energy change < 2 meV for this defect, considering this converged." >>OUTCAR
            # sed -i 's/IBRION.*/IBRION = 1/g' INCAR # sometimes helps to change IBRION if relaxation not converging
            builtin cd .. || return
            continue
          fi
        fi

        # if electronic convergence is not being reached, change ALGO to All
        if grep -q "aborting loop EDIFF was not reached" OUTCAR && ! grep -q "aborting loop because EDIFF is reached" OUTCAR; then
          # unconverged electronic loops and no converged electronic loops
          if [ "$verbose" = true ]; then
            echo "${i%?} is showing poor electronic convergence, changing ALGO to All."
          fi
          sed -i 's/ALGO.*/ALGO = All/g' INCAR
        fi

        # check if multiple <=single-step OUTCARs present, and CONTCAR empty/less than 9 lines or same as POSCAR
        if check_multiple_single_step_outcars && { [[ -f "CONTCAR" ]] && [[ $(wc -l < "CONTCAR") -le 9 ]] || [[ -f "CONTCAR" ]] && diff -q "POSCAR" "CONTCAR" >/dev/null || [[ ! -f "CONTCAR" ]]; }; then
            echo "Previous run for ${i%?} did not yield more than one ionic step, and multiple OUTCARs with <=1 ionic "
            echo "steps present, suggesting poor convergence. Recommended to manually check the VASP output files for this!"
        fi

        echo "${i%?} not (fully) relaxed, saving files and rerunning"
        bash "${DIR}"/save_vasp_files.sh

        if [[ -f "CONTCAR" ]] && [[ $(wc -l < "CONTCAR") -ge 9 ]]; then # CONTCAR exists and greater than 9 lines
          "cp" CONTCAR POSCAR
        fi

        # check if calc was spin-polarised and magnetisation below threshold, then switch to ISPIN = 1
        if grep -q "ISPIN  =      2" OUTCAR; then  # spin-polarised calc
          if snb-mag; then
            sed -i 's/ISPIN.*/ISPIN = 1  # atomic magnetization in previous run below threshold/' INCAR
          fi
        fi
      fi
      if [ -f "./${job_filename}" ]; then
        echo "Running job for ${i%?}"
        folder_shortname="${i#*_*_}"
        # Remove % from folder_shortname as messes with some HPC schedulers
        ${job_submit_command} "${job_name_option}" "${defect_name%?}"_"${folder_shortname%?}" "${job_filename}" 2>/dev/null || ${job_submit_command} "${job_name_option}" "${defect_name%?}"_"${folder_shortname%??}" "${job_filename}"
      fi
      builtin cd .. || return
    else
      if [ "$verbose" = true ]; then
        echo "${i%?} fully relaxed"
      fi
      if [ -f "${i}"/DOSCAR ]; then # remove DOSCAR to save space
        rm "${i}"/DOSCAR
      fi
    fi
  done
}

if [ "$single_defect" = false ]; then
  # for each defect, in the directory where your top-level defect folders are
  for defect_name in *[0-9]/; do # matches each subdirectory with a name ending in a number (charge state)
    if [ "$defect_name" == "*[0-9]/" ]; then
      echo "No defect folders (with names ending in a number (charge state)) found in current directory"
      break # exit if no defect folders found
    fi
    echo "Looping through distortion folders for ${defect_name%?}"
    builtin cd "${defect_name}" || return
    SnB_run_loop
    cd .. || return
  done
else
  SnB_run_loop
fi

if [ ! $shopt_enabled = true ]; then
  shopt -u extglob # disable extended globbing (pattern matching) if it was disabled at start
fi
