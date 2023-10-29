#!/bin/bash

#SBATCH --array=0-25 # Array index from 0 to 25 (for 26 patients)
#SBATCH --job-name=training_array
#SBATCH --output="${xav}/ssl_epilepsy/jobs/%A_%a.out"
#SBATCH --error=train_wrapper_%A_%a.err
#SBATCH --mail-user=xmootoo@gmail.com
#SBATCH --mail-type=ALL

# The date and time ID fo the pretrained model, which we will extract the pretrained layers from.
pretrained_datetime_id="$1"

# Name of model to be trained using the pretrained layers. Options: downstream1.
model_id="$2"

# Freezes or unfreezes the weights of the pretrained layers. Options: 0 (False) or 1 (True).
frozen="$3"

# Define the list of patients (26 patients).
declare -a patients=("jh101" "jh103" "jh108" "pt01" "pt2" "pt3" "pt6" "pt7" "pt8" "pt10" "pt11" "pt12" "pt13" "pt14" "pt15" "pt16" "umf001" "ummc001" "ummc002" "ummc003" "ummc004" "ummc005" "ummc006" "ummc007" "ummc008" "ummc009")

# Get date time id
datetime_id=$(date "+%Y-%m-%d_%H.%M.%S")

# Get the patient_id using the array index
patient_id=${patients[$SLURM_ARRAY_TASK_ID]}

# Call the original script for each patient
sbatch transfer_wrapper.sh $patient_id $datetime_id $pretrained_datetime_id $model_id $frozen
