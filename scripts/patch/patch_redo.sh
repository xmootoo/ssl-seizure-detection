#!/bin/bash

# Define an array of patients
patients=("pt2" "pt15")
num_patients=${#patients[@]}  # Get the length of the array

# Decrement num_patients by 1 because array indices start at 0
((num_patients--))

#SBATCH --array=0-${num_patients}  # Dynamically set the array size based on the number of patients
#SBATCH --job-name=patient_gr_conversion_array
#SBATCH --output="${xav}/ssl_epilepsy/jobs/%A_%a.out"
#SBATCH --mail-user=xmootoo@gmail.com
#SBATCH --mail-type=ALL

# Use the SLURM_ARRAY_TASK_ID to get the patient
patient="${patients[$SLURM_ARRAY_TASK_ID]}"

# Call the wrapper script
sbatch "patch_wrapper.sh" "${patient}"