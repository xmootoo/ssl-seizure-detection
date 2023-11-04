#!/bin/bash

# Takes the patient as an argument
patient="$1"

# Other arguments
K="$2"
N="$3"
P="$4"
data_size="$5"

# Directories
patient_dir="${xav}/ssl_epilepsy/data/patient_pyg"
logdir="${xav}/ssl_epilepsy/data/patient_pyg"

# Define arrays for different models, sample_ratios, and times
model="CPC"
time="01:00:00"
job_name="patient_gr_conversion_${patient}_${model}"

# Create a job for each model + sample_ratio + time
sbatch <<EOT
#!/bin/bash
#SBATCH --ntasks=1              # Number of tasks
#SBATCH --cpus-per-task=1       # CPU cores/threads, AKA number of workers (num_workers)
#SBATCH --mem-per-cpu=12G       # memory per CPU core
#SBATCH --time="${time}"        # Time limit for the job  
#SBATCH --job-name="${job_name}"
#SBATCH --output="${xav}/ssl_epilepsy/jobs/${job_name}.out"
#SBATCH --mail-user=xmootoo@gmail.com
#SBATCH --mail-type=ALL

cd "${xav}/ssl_epilepsy/ssl-seizure-detection/src"
module load cuda/11.7 cudnn python/3.10
source ~/torch2_cuda11.7/bin/activate

python patch.py "${patient_dir}" "${patient}" "${logdir}" "${model}" "${K}" "${N}" "${P}" "${data_size}"
EOT
done
