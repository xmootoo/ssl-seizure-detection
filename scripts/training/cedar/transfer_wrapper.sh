#!/bin/bash

# Check if the patient ID is provided
if [ -z "$1" ]; then
    echo "Error: No patient ID provided."
    exit 1
fi

# The patient ID (e.g., jh101)
patient_id="$1"

# Date and time ID
datetime_id="$2"

# Model datetime ID of the pretrained model
pretrained_datetime_id="$3"

# Model ID of using the pretrained layers (i.e. new untrained model)
model_id="$4"

# Freezes or unfreezes pretrained layers
frozen="$5"

# Training arguments
run_types="combined"
transfer_ids=("relative_positioning" "temporal_shuffling")
if [ "$model_id" == "downstream1" ]; then
  time="00:45:00"
elif [ "$model_id" == "downstream2" ]; then
  time="00:25:00"
else
  echo "Invalid model_id"
fi



# Base directory
base_dir="${xav}/ssl_epilepsy/models/${patient_id}"
mkdir -p "${base_dir}" || { echo "Error: Cannot create directory ${base_dir}"; exit 1; }



# Iterate over each model and its corresponding time
for i in "${!transfer_ids[@]}"; do
    transfer_id="${transfer_ids[$i]}"
    job_name="transfer_learning_${patient_id}_${model_id}_${transfer_id}_${pretrained_datetime_id}_${datetime_id}"
    data_path="${xav}/ssl_epilepsy/data/patient_pyg/${patient_id}/supervised"
    
    model_path="${xav}/ssl_epilepsy/models/${patient_id}/${transfer_id}/${pretrained_datetime_id}/model/${transfer_id}.pth"
    model_dict_path="${xav}/ssl_epilepsy/models/${patient_id}/${transfer_id}/${pretrained_datetime_id}/model/${transfer_id}_state_dict.pth"
    
    logdir="${base_dir}/${model_id}/${datetime_id}"
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

python main.py "${data_path}" "${logdir}" "${patient_id}" "${model_id}" "${datetime_id}" "${run_type}" "${model_path}" "${model_dict_path}" "${transfer_id}" "${frozen}"
EOT
done
