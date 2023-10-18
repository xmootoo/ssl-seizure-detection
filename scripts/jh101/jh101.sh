#!/bin/bash
data_path="$xav/ssl_epilepsy/data/pseudolabeled/jh101/jh101_12s_7min_PairData.pt"
model_path="$xav/ssl_epilepsy/jobs/relative_positioning/jh101/models"
stats_path="$xav/ssl_epilepsy/jobs/relative_positioning/jh101/stats"
model_name="jh101_12s_7min_pyg1"
num_workers="8" 


data_path = sys.argv[1]
    
    # The path to save the model
    logdir = sys.argv[2]
    
    # The path to save the training statistics
    patient_id = sys.argv[3]
    
    # The name of the model
    model_id = sys.argv[4]

sbatch <<EOT
#!/bin/bash
#SBATCH --ntasks=1                # Number of tasks
#SBATCH --gres=gpu:v100:1         # Number of Volta 100 GPUs
#SBATCH --cpus-per-task=8         # CPU cores/threads, AKA number of workers (num_workers)
#SBATCH --mem-per-cpu=12G         # memory per CPU core
#SBATCH --time=00:30:00     
#SBATCH --job-name=$job_name
#SBATCH --output=$xav/ssl_epilepsy/jobs/relative_positioning/jh101/out/$job_name.out
#SBATCH --mail-user=xmootoo@gmail.com
#SBATCH --mail-type=ALL

cd $xav/ssl_epilepsy/ssl-seizure-detection/relative_positioning/pytorch/pyg
module load cuda/11.7 cudnn python/3.10
source ~/torch2_cuda11.7/bin/activate

python main.py $data_path $model_path $stats_path $model_name $num_workers

EOT