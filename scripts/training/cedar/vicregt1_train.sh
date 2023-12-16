#!/bin/bash

# Job parameters
patient_id="$1"
model_id="$2"
datetime_id="$3"

# Data parameters
sigma="$4"
tau="$5"
sr="$6"

# Other parameters
epochs="$7"
project_id="$8"

# Train, val, test split
split="$9"
time="00:30:00"

# Directories
base_dir="${xav}/ssl_epilepsy/models/${patient_id}"
mkdir -p "${base_dir}" || { echo "Error: Cannot create directory ${base_dir}"; exit 1; }
logdir="${base_dir}/${model_id}/${datetime_id}"
mkdir -p "${logdir}" || { echo "Error: Cannot create directory ${logdir}"; exit 1; }
data_path="${xav}/ssl_epilepsy/data/patient_pyg/${patient_id}/${model_id}/${sigma}var_${tau}tau_${sr}sr"

# Training arguments
job_name="training_${patient_id}_${model_id}_${time}_${datetime_id}"
run_type="all"
classify="None"

#SBATCH --ntasks=1              # Number of tasks
#SBATCH --gres=gpu:v100l:1      # Number of Volta 100 GPUs
#SBATCH --cpus-per-task=10       # CPU cores/threads, AKA number of workers (num_workers)
#SBATCH --mem-per-cpu=16G       # memory per CPU core
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

python main.py "${data_path}" "${logdir}" "${patient_id}" "${model_id}" "${datetime_id}" "${run_type}" "${classify}" "${split}" "${epochs}" "${project_id}"