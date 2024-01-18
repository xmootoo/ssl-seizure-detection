#!/bin/bash

# Debugging version of the script

# Takes the model, patient, sample_ratio, and time as arguments
model="$1"
patient="$2"
sample_ratio="$3"
time="$4"
sigma="$5"
tau="$6"

# Ensure that the xav environment variable is set
if [ -z "${xav}" ]; then
    echo "Error: xav variable is not set."
    exit 1
fi

# Directories and job name
patient_dir="/home/xmootoo/projects/def-milad777/gr_research/brain-greg/data/ds003029-processed/graph_representation_elements"
logdir="${xav}/ssl_epilepsy/data/patient_pyg"
job_name="patient_gr_conversion_${patient}_${model}"

# Check if patient_dir exists
if [ ! -d "${patient_dir}" ]; then
    echo "Error: Patient directory ${patient_dir} does not exist."
    exit 1
fi

# Check if log directory exists or create it
if [ ! -d "${logdir}" ]; then
    echo "Creating log directory ${logdir}."
    mkdir -p "${logdir}"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create log directory ${logdir}."
        exit 1
    fi
fi

echo "Submitting a job for the model with the given sample_ratio, time, sigma, and tau..."
sbatch <<EOT
#!/bin/bash
#SBATCH --ntasks=1              # Number of tasks
#SBATCH --cpus-per-task=4       # CPU cores/threads
#SBATCH --mem-per-cpu=12G       # memory per CPU core
#SBATCH --time="${time}"        # Time limit for the job  
#SBATCH --job-name="${job_name}"
#SBATCH --output="${xav}/ssl_epilepsy/jobs/${job_name}.out"
#SBATCH --mail-user=xmootoo@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account="def-milad777"

echo "Starting job ${job_name} on $(date)"
cd "${xav}/ssl_epilepsy/ssl-seizure-detection/src" || { echo "Failed to change directory"; exit 1; }

echo "Loading modules..."
module load cuda/11.7 cudnn python/3.10 || { echo "Module loading failed"; exit 1; }

echo "Activating Python environment..."
source ~/torch2_cuda11.7/bin/activate || { echo "Failed to activate Python environment"; exit 1; }

echo "Running Python script..."
python patch.py "${patient_dir}" "${patient}" "${logdir}" "${model}" "${sample_ratio}" "${sigma}" "${tau}" || { echo "Python script execution failed"; exit 1; }

echo "Job ${job_name} finished on $(date)"
EOT

if [ $? -ne 0 ]; then
    echo "Error: sbatch submission failed."
    exit 1
fi

echo "Job submission complete."
