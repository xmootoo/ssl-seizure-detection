#!/bin/bash

# Job parameters
patient_id="$1"

model_id="$2"
datetime_id="$3"

# Other parameters
epochs="$4"
project_id="$5"

# Train, val, test split
split="$6"

# Classification task (binary or multiclass)
classify="$7"

# Experimental ID
exp_id="$8"

# Pretrainted datetime ID
pretrained_datetime_id="$9"

# Frozen (0) or unfrozen (1)
requires_grad="${10}"

# Transfer ID
transfer_id="${11}"

# Data size
data_size="${12}"

# Echo statements for debugging
echo "Patient ID: ${patient_id}"
echo "Model ID: downstream3"
echo "Datetime ID: ${datetime_id}"
echo "Epochs: ${epochs}"
echo "Project ID: ${project_id}"
echo "Split: ${split}"
echo "Classify: ${classify}"
echo "Experiment ID: ${exp_id}"
echo "Pretrained Datetime ID: ${pretrained_datetime_id}"
echo "Requires Grad: ${requires_grad}"
echo "Transfer ID: ${transfer_id}"
echo "Data Size: ${data_size}"

# Run time
time="00:25:00"

# Directories
base_dir="${xav}/ssl_epilepsy/models/${patient_id}"
mkdir -p "${base_dir}" || { echo "Error: Cannot create directory ${base_dir}"; exit 1; }
logdir="${base_dir}/${model_id}/${datetime_id}"
mkdir -p "${logdir}" || { echo "Error: Cannot create directory ${logdir}"; exit 1; }
data_path="${xav}/ssl_epilepsy/data/patient_pyg/${patient_id}/supervised"
model_path="${xav}/ssl_epilepsy/models/${patient_id}/${transfer_id}/${pretrained_datetime_id}/model/${transfer_id}.pth"
model_dict_path="${xav}/ssl_epilepsy/models/${patient_id}/${transfer_id}/${pretrained_datetime_id}/model/${transfer_id}_state_dict.pth"


# Training arguments
job_name="transfer_anyGPU_${patient_id}_${model_id}_${transfer_id}_${pretrained_datetime_id}_${datetime_id}"
run_type="all"


echo "Preparing to submit downstream3 training job..."
sbatch <<EOT
#!/bin/bash
#SBATCH --ntasks=1              # Number of tasks
#SBATCH --gpus-per-node=1       # Number of Volta 100 GPUs
#SBATCH --cpus-per-task=4      # CPU cores/threads, AKA number of workers (num_workers)
#SBATCH --mem-per-cpu=12G       # memory per CPU core
#SBATCH --time="${time}"        # Time limit for the job  
#SBATCH --job-name="${job_name}"
#SBATCH --output="${xav}/ssl_epilepsy/jobs/training/${job_name}.out"
#SBATCH --error="${xav}/ssl_epilepsy/jobs/training/${job_name}.err"
#SBATCH --mail-user=xmootoo@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account="def-milad777"

cd "${xav}/ssl_epilepsy/ssl-seizure-detection/src"
module load cuda/11.7 cudnn/8.9.5.29 python/3.10
source ~/torch2_cuda11.7/bin/activate

export WANDB_API_KEY="$WANDB_API_KEY"

python main.py "${data_path}" "${logdir}" "${patient_id}" \
"${model_id}" "${datetime_id}" "${run_type}" "${classify}" \
"${split}" "${epochs}" "${project_id}" "${exp_id}" "${data_size}" \
"${model_path}" "${model_dict_path}" "${transfer_id}" "${requires_grad}" 

EOT

if [ $? -ne 0 ]; then
    echo "Error: sbatch submission failed."
    exit 1
fi

echo "Job submission complete."