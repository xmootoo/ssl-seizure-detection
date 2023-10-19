#!/bin/bash

# Check if the patient ID is provided
if [ -z "$1" ]; then
    echo "Error: No patient ID provided."
    exit 1
fi

# The patient ID (e.g., jh101)
patient_id="$1"

# Training arguments
model_ids=("supervised" "relative_positioning" "temporal_shuffling")
times=("00:30:00" "11:59:00" "20:00:00")
date_and_time=$(date "+%Y-%m-%d_%H.%M.%S")

# Base directory
base_dir="${xav}/ssl_epilepsy/models/${patient_id}"
mkdir -p "${base_dir}"

# Iterate over each model and its corresponding time
for i in "${!model_ids[@]}"; do
    model_id="${model_ids[$i]}"
    time="${times[$i]}"
    job_name="training_${patient_id}_${model_id}_${time}"
    
    # Define the data path
    case "$model_id" in
        "supervised")
            data_path="${xav}/ssl_epilepsy/data/patient_pyg/${patient_id}/${model_id}/${patient_id}_run1.pt"
            ;;
        "relative_positioning")
            data_path="${xav}/ssl_epilepsy/data/patient_pyg/${patient_id}/${model_id}/${patient_id}_run1_12s_90_1.0sr.pt"
            ;;
        "temporal_shuffling")
            data_path="${xav}/ssl_epilepsy/data/patient_pyg/${patient_id}/${model_id}/${patient_id}_run1_12s_90_0.22sr.pt"
            ;;
        *)
            echo "Unknown model_id: $model_id"
            continue
            ;;
    esac
    
    logdir="${base_dir}/${model_id}/${date_and_time}"
    mkdir -p "${logdir}"

    # Create a job for each model + time
    sbatch <<EOT
#!/bin/bash
#SBATCH --ntasks=1              # Number of tasks
#SBATCH --gres=gpu:v100l:1      # Number of Volta 100 GPUs
#SBATCH --cpus-per-task=8       # CPU cores/threads, AKA number of workers (num_workers)
#SBATCH --mem-per-cpu=12G       # memory per CPU core
#SBATCH --time="${time}"        # Time limit for the job  
#SBATCH --job-name="${job_name}"
#SBATCH --output="${xav}/ssl_epilepsy/jobs/training/${job_name}.out"
#SBATCH --error="${xav}/ssl_epilepsy/jobs/training/${job_name}.err"
#SBATCH --mail-user=xmootoo@gmail.com
#SBATCH --mail-type=ALL

cd "${xav}/ssl_epilepsy/ssl-seizure-detection/src"
module load cuda/11.7 cudnn python/3.10
source ~/torch2_cuda11.7/bin/activate

export WANDB_API_KEY="$WANDB_API_KEY"

python main.py "${data_path}" "${logdir}" "${patient_id}" "${model_id}" "${date_and_time}"
EOT
done
