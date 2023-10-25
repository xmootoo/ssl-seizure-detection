#!/bin/bash


#SBATCH --array=0-25 # Array index from 0 to 25 (for 26 patients)
#SBATCH --job-name=training_array
#SBATCH --output="${xav}/ssl_epilepsy/jobs/%A_%a.out"
#SBATCH --error=train_wrapper_%A_%a.err
#SBATCH --mail-user=xmootoo@gmail.com
#SBATCH --mail-type=ALL

# Define the list of patients (26 patients).
declare -a patients=("jh101" "jh103" "jh108" "pt01" "pt2" "pt3" "pt6" "pt7" "pt8" "pt10" "pt11" "pt12" "pt13" "pt14" "pt15" "pt16" "umf001" "ummc001" "ummc002" "ummc003" "ummc004" "ummc005" "ummc006" "ummc007" "ummc008" "ummc009")

# Fixed parameters for each patient
tau_pos=12
tau_neg=90

# Get the patient_id using the array index
patient_id=${patients[$SLURM_ARRAY_TASK_ID]}

# Call the original script for each patient
sbatch train_wrapper.sh $patient_id $tau_pos $tau_neg
