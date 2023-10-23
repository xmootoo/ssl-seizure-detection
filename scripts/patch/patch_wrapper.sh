#!/bin/bash

# Takes the patient as an argument
patient="$1"

# Directories
patient_dir="/home/xmootoo/projects/def-milad777/gr_research/brain-greg/data/ds003029-processed/graph_representation_elements"
logdir="${xav}/ssl_epilepsy/data/patient_pyg"

# Define arrays for different models, sample_ratios, and times
models=("supervised" "relative_positioning" "temporal_shuffling")
sample_ratios=("1.0" "0.9" "0.22")
times=("00:15:00" "00:30:00" "00:45:00")

# Conditionally change times and sample_ratios if patient is "pt15" or "pt2", as they fail for the above settings.
if [[ "$patient" == "pt15" || "$patient" == "pt2" ]]; then
    times=("00:30:00" "00:50:00" "01:00:00")
    sample_ratios=("1.0" "0.75" "0.13")
fi

# Iterate over each model and its corresponding sample_ratio and time
for i in "${!models[@]}"; do
    model="${models[$i]}"
    sample_ratio="${sample_ratios[$i]}"
    time="${times[$i]}"
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

python patch.py "${patient_dir}" "${patient}" "${logdir}" "${model}" "${sample_ratio}"
EOT
done
