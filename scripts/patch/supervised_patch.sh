#!/bin/bash

# Takes the patient, sample_ratio, and time as arguments
model="$1"
patient="$2"
sample_ratio="$3"
time="$4"

# Directories and job name
patient_dir="/home/xmootoo/projects/def-milad777/gr_research/brain-greg/data/ds003029-processed/graph_representation_elements"
logdir="${xav}/ssl_epilepsy/data/patient_pyg"
job_name="patient_gr_conversion_${patient}_${model}"

# Create a job for the model with the given sample_ratio and time
sbatch <<EOT
#!/bin/bash
#SBATCH --ntasks=1              # Number of tasks
#SBATCH --cpus-per-task=1       # CPU cores/threads
#SBATCH --mem-per-cpu=12G       # memory per CPU core
#SBATCH --time="${time}"        # Time limit for the job  
#SBATCH --job-name="${job_name}"
#SBATCH --output="${xav}/ssl_epilepsy/jobs/${job_name}.out"
#SBATCH --mail-user=xmootoo@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account="def-milad777"

cd "${xav}/ssl_epilepsy/ssl-seizure-detection/src"
module load cuda/11.7 cudnn python/3.10
source ~/torch2_cuda11.7/bin/activate

python patch.py "${patient_dir}" "${patient}" "${logdir}" "${model}" "${sample_ratio}"
EOT
