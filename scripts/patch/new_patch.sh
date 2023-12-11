#!/bin/bash
model="$1"
sample_ratio="$2"
time="$3"

# Define the list of patients (26 patients).
declare -a patients=("jh101", "jh103", "jh108", "pt01", "pt2", "pt3", "pt6", "pt7", "pt8", "pt10", "pt11", "pt12", "pt13", "pt14", "pt15", "pt16", "umf001", "ummc001", "ummc002", "ummc003", "ummc004", "ummc005", "ummc006", "ummc007", "ummc008", "ummc009")

# Loop over each patient
for patient_id in "${patients[@]}"; do
    # Select the corresponding model
    if [[ "$model" == "supervised" ]]; then
        bash supervised_patch.sh "$model" "$patient_id" "$sample_ratio" "$time" # Suggested: sample_ratio=1.0, time=00:30:00
    elif [[ "$model" == "relative_positioning" ]]; then
        # Call the script with parameters specific to relative_positioning
        tau_pos="$4"
        tau_neg="$5"
        bash rp_ts_patch.sh "$model" "$patient_id" "$sample_ratio" "$time" "$tau_pos" "$tau_neg" # Suggested: sample_ratio=0.9, time=00:35:00, tau_pos=12, tau_neg=90
    elif [[ "$model" == "temporal_shuffling" ]]; then
        # Call the script with parameters specific to temporal_shuffling
        tau_pos="$4"
        tau_neg="$5"
        bash rp_ts_patch.sh "$model" "$patient_id" "$sample_ratio" "$time" "$tau_pos" "$tau_neg" # Suggested: sample_ratio=0.22, time=00:45:00, tau_pos=12, tau_neg=90
    elif [[ "$model" == "CPC" ]]; then
        # Call the script with parameters specific to CPC
        K="$4"
        N="$5"
        P="$6"
        data_size="$7"
        bash cpc_patch.sh "$model" "$patient_id" "$sample_ratio" "$time" "$K" "$N" "$P" "$data_size"
    elif [[ "$model" == "VICRegT1" ]]; then
        sigma="$4"
        tau="$5"
        # Call the script with parameters specific to VICRegT1
        bash vicregt1_patch.sh "$model" "$patient_id" "$sample_ratio" "$time" "$sigma" "$tau"
    fi
done