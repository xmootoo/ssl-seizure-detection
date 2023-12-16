#!/bin/bash

model="$1"
epochs="$2"
project_id="$3"
split="$4"
time="$5"

# Define the list of patients (26 patients).
declare -a patients=("jh101" "jh103" "jh108" "pt01" "pt2" "pt3" "pt6" "pt7" "pt8" "pt10" "pt11" "pt12" "pt13" "pt14" "pt15" "pt16" "umf001" "ummc001" "ummc002" "ummc003" "ummc004" "ummc005" "ummc006" "ummc007" "ummc008" "ummc009")

# Get date time id
datetime_id=$(date "+%Y-%m-%d_%H.%M.%S")



# Loop over each patient and call the original script
for patient_id in "${patients[@]}"; do
    if [ "$model" = "VICRegT1" ]; then
        sigma="$5"
        tau="$6"
        sr="$7"
        bash vicregt1_train.sh "${patient_id}" "VICRegT1" "${datetime_id}" "${sigma}" "${tau}" "${sr}" "${epochs}" "${project_id}" "${split}" "${time}"
    fi
done