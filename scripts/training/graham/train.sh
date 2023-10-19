#!/bin/bash

#SBATCH --array=0-25  # Creates 26 jobs, one for each patient
#SBATCH --job-name=patient_gr_conversion_array
#SBATCH --output="${xav}/ssl_epilepsy/jobs/%A_%a.out"
#SBATCH --mail-user=xmootoo@gmail.com
#SBATCH --mail-type=ALL

# Define an array of patients
patients=("jh101" "jh103" "jh108" "pt01" "pt2" "pt3" "pt6" "pt7" "pt8" "pt10" "pt11" "pt12" "pt13" "pt14" "pt15" "pt16" "umf001" "ummc001" "ummc002" "ummc003" "ummc004" "ummc005" "ummc006" "ummc007" "ummc008" "ummc009")

# Use the SLURM_ARRAY_TASK_ID to get the patient
patient="${patients[$SLURM_ARRAY_TASK_ID]}"

# Call the wrapper script
sbatch "patch_wrapper.sh" "${patient}"
