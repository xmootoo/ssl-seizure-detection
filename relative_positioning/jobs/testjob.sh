#!/bin/bash
job_name="test5_20_08_2023"
model_logdir="$xav/ssl_epilepsy/jobs/relative_positioning/models/"
stats_logdir="$xav/ssl_epilepsy/jobs/relative_positioning/stats/"


sbatch <<EOT
#!/bin/bash
#SBATCH --ntasks=1               # Number of cores (CPUs, threads)
#SBATCH --gres=gpu:1             # Number of GPU(s) per node
#SBATCH --cpus-per-task=16       # CPU cores/threads
#SBATCH --mem-per-cpu=10G        # memory per CPU core
#SBATCH --time=00:05:00     
#SBATCH --job-name=$job_name
#SBATCH --output=$xav/ssl_epilepsy/jobs/relative_positioning/tests/$job_name.out
#SBATCH --mail-user=xmootoo@gmail.com
#SBATCH --mail-type=ALL

cd $xav/ssl_epilepsy/ssl-seizure-detection/relative_positioning/tensorflow/tests
module load cuda cudnn python/3.10
source ~/tf2.12/bin/activate

python testcc.py $model_logdir $stats_logdir

EOT