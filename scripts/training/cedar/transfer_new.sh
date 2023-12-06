#!/bin/bash


# The date and time ID fo the pretrained model, which we will extract the pretrained layers from.
pretrained_datetime_id="$1"

# Name of model to be trained using the pretrained layers. Options: downstream1, downstream2.
model_id="$2"

# Freezes or unfreezes the weights of the pretrained layers. Options: 0 (False) or 1 (True).
frozen="$3"

# "train,val,test" split. Please put values in quotations
# Option 1: Fix number of examples: "500,300,400" for "train,val,test"
# Option 2: Ratios: "0.7,0.2,0.1" for "train,val,test"
# Option 3: If only two values are indicated, it only corresponds to val and test: "0.2,0.1" gives 0.7 for train
split="$4"

# Define the list of patients (26 patients).
declare -a patients=("jh101" "jh103" "jh108" "pt01" "pt2" "pt3" "pt6" "pt7" "pt8" "pt10" "pt11" "pt12" "pt13" "pt14" "pt15" "pt16" "umf001" "ummc001" "ummc002" "ummc003" "ummc004" "ummc005" "ummc006" "ummc007" "ummc008" "ummc009")

# Get date time id
datetime_id=$(date "+%Y-%m-%d_%H.%M.%S")

# Get the patient_id using the array index
patient_id=${patients[$SLURM_ARRAY_TASK_ID]}

# Call the original script for each patient
bash transfer_wrapper.sh $patient_id $datetime_id $pretrained_datetime_id $model_id $frozen $split
