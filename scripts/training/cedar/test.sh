
# Check if the patient ID is provided
if [ -z "$1" ]; then
    echo "Error: No patient ID provided."
    exit 1
fi

# The patient ID (e.g., jh101)
patient_id="$1"

# SSL Parameters
tau_pos="$2"
tau_neg="$3"

# Date and time ID
datetime_id="$4"

# Train, val, test split
split="$5"

# Model selection. It is 1-3 digits, where a 0 indicates supervised, 1 indicates relative positioning, and 2 indicates temporal shuffling.
# For example, model_selection=01 means that only the supervised and relative positioning models will be trained.
model_selection=${6:-}

# Training arguments
run_types=("combined" "all" "all")
model_ids=("supervised" "relative_positioning" "temporal_shuffling")
times=("00:45:00" "20:00:00" "20:00:00")


# Function to keep elements by index
keep_by_indices() {
    local indices_to_keep=($1)
    local -a _arr=("${!2}")
    local -a new_arr=()
    for index in "${indices_to_keep[@]}"; do
        new_arr+=("${_arr[index]}")
    done
    echo "${new_arr[@]}"
}

# ... rest of your script ...

if [ -n "$model_selection" ]; then
    indices_to_keep=()
    for (( i=0; i<${#model_selection}; i++ )); do
        indices_to_keep+=(${model_selection:$i:1})
    done

    # Now capture the output of keep_by_indices into the arrays
    model_ids=($(keep_by_indices "${indices_to_keep[*]}" "model_ids[@]"))
    run_types=($(keep_by_indices "${indices_to_keep[*]}" "run_types[@]"))
    times=($(keep_by_indices "${indices_to_keep[*]}" "times[@]"))
    echo "Model IDs: ${model_ids[@]}"
    echo "Run types: ${run_types[@]}"
    echo "Times: ${times[@]}"
else
    echo "Model selection is not set. Running all models..."
fi
