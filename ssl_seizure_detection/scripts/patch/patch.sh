#!/bin/bash

model="$1"

# Define the list of patients (26 patients).
declare -a patients=("jh101" "jh103" "jh108" "pt01" "pt2" "pt3" "pt6" "pt7" "pt8" "pt10" "pt11" "pt12" "pt13" "pt14" "pt15" "pt16" "umf001" "ummc001" "ummc002" "ummc003" "ummc004" "ummc005" "ummc006" "ummc007" "ummc008" "ummc009")

for patient_id in "${patients[@]}"; do
    # VICRegT1 Model
    if [ "$model" = "VICRegT1" ]; then
        sigma="$2" # Suggested: sigma=5
        tau="$3" # Suggested: tau=0.05
        bash vicregt1_patch.sh "VICRegT1" "$patient_id" "1.0" "01:00:00" "${sigma}" "${tau}"
    
    # Supervised Model
    elif [ "$model" = "supervised" ]; then
        bash supervised_patch.sh "supervised" "$patient_id" "1.0" "00:30:00"
    fi
done