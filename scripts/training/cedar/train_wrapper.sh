#!/bin/bash

# The patient ID (e.g. jh101)
patient_id="$1"

# Training arguments
model_ids=("supervised" "relative_positioning" "temporal_shuffling")
times=("00:30:00" "11:59:00" "20:00:00")
date_and_time=$(date "+%Y-%m-%d_%H.%M.%S")

# Base directory
base_dir="${xav}/ssl_epilepsy/models/${patient_id}"
mkdir -p "${base_dir}"

# Fetch wandb API key from an environment variable
wandb_api_key=$WANDB_API_KEY

# Iterate over each model and its corresponding sample_ratio and time
for i in "${!model_ids[@]}"; do
    model_id="${model_ids[$i]}"
    time="${times[$i]}"
    job_name="training_${patient_id}_${model_id}_${time}"
    
    # Define the data path
    if [ "$model_id" == "supervised" ]; then
        data_path="${xav}/ssl_epilepsy/data/patient_pyg/${patient_id}/${model_id}/${patient_id}_run1.pt"
    elif [ "$model_id" == "relative_positioning" ]; then
        data_path="${xav}/ssl_epilepsy/data/patient_pyg/${patient_id}/${model_id}/${patient_id}_run1_12s_90_1.0sr.pt"
    elif [ "$model_id" == "temporal_shuffling" ]; then
    fi
    
    logdir="${logdir}/${model_id}/${data_and_time}"
    mkdir -p "${logdir}/${model_id}/${data_and_time}"

    # Create a job for each model + sample_ratio + time
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

export WANDB_API_KEY="${wandb_api_key}"

python main.py "${data_path}" "${logdir}" "${patient_id}" "${model_id}"
EOT
done
