#!/bin/bash

# Define an array of patients
patients=("jh101" "jh103" "jh108" "pt01" "pt2" "pt3" "pt6" "pt7" "pt8" "pt10" "pt11" "pt12" "pt13" "pt14" "pt15" "pt16" "umf001" "ummc001" "ummc002" "ummc003" "ummc004" "ummc005" "ummc006" "ummc007" "ummc008" "ummc009")

# Define an array of number of runs for each patient
#TODO: Update to actual num run information, this is dummy information. This should be the same length as the patients array.
num_runs=("4" "3" "5" "2" "1" "3" "4" "2" "1" "3" "4" "2" "1" "3" "4" "2" "1" "3" "4" "2" "1" "3" "4" "2" "1" "3")

# Calculate the total number of jobs required
total_jobs=0
for runs in "${num_runs[@]}"; do
    total_jobs=$((total_jobs + runs))
done

# Create an array job with the total number of jobs
#SBATCH --array=0-$((total_jobs - 1))
#SBATCH --job-name=training_array
#SBATCH --output="${xav}/ssl_epilepsy/jobs/%A_%a.out"
#SBATCH --mail-user=xmootoo@gmail.com
#SBATCH --mail-type=ALL

# Calculate the patient and run based on the SLURM_ARRAY_TASK_ID
current_id=0
for idx in "${!patients[@]}"; do
    patient="${patients[$idx]}"
    runs="${num_runs[$idx]}"
    for run in $(seq 1 $runs); do
        if [ "$current_id" -eq "$SLURM_ARRAY_TASK_ID" ]; then
            # Call the wrapper script
            sbatch "train_wrapper.sh" "${patient}" "${run}"
            exit 0
        fi
        current_id=$((current_id + 1))
    done
done
