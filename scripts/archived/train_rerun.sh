#!/bin/bash


#SBATCH --array=0-1 # Array index from 0 to 25 (for 26 patients)
#SBATCH --job-name=training_array
#SBATCH --output="${xav}/ssl_epilepsy/jobs/%A_%a.out"
#SBATCH --error=train_wrapper_%A_%a.err
#SBATCH --mail-user=xmootoo@gmail.com
#SBATCH --mail-type=ALL

# Define the list of patients with problematic runs (26 patients).
declare -a patients=("pt13" "ummc006")

# Fixed parameters for each patient
tau_pos=12
tau_neg=90

# Get the patient_id using the array index
patient_id=${patients[$SLURM_ARRAY_TASK_ID]}

# Call the original script for each patient
sbatch train_wrapper.sh $patient_id $tau_pos $tau_neg
