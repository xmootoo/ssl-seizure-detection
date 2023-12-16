#!/bin/bash

#SBATCH --array=0-25 # Array index from 0 to 25 (for 26 patients)
#SBATCH --job-name=training_array
#SBATCH --output="${xav}/ssl_epilepsy/jobs/%A_%a.out"
#SBATCH --error=train_wrapper_%A_%a.err
#SBATCH --mail-user=xmootoo@gmail.com
#SBATCH --mail-type=ALL

# Read the tau_pos and tau_neg from the command line arguments
tau_pos="$1"
tau_neg="$2"
split="$3"

# Model selection. It is 1-3 digits, where a 0=supervised, 1=relative positioning, 2=temporal shuffling.
# For example, model_selection=01 means that only the supervised and relative positioning models will be trained.
model_selection="$4"

# Classification task
classify="$5"

# Define the list of patients (26 patients).
declare -a patients=("jh101" "jh103" "jh108" "pt01" "pt2" "pt3" "pt6" "pt7" "pt8" "pt10" "pt11" "pt12" "pt13" "pt14" "pt15" "pt16" "umf001" "ummc001" "ummc002" "ummc003" "ummc004" "ummc005" "ummc006" "ummc007" "ummc008" "ummc009")

# Get date time id
datetime_id=$(date "+%Y-%m-%d_%H.%M.%S")

# Loop over each patient and call the original script
for patient_id in "${patients[@]}"; do
    bash train_wrapper.sh $patient_id $tau_pos $tau_neg $datetime_id $split $model_selection $classify
done