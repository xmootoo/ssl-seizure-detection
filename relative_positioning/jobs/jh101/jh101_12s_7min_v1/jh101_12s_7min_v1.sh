#!/bin/bash
job_name="jh101_12s_7min_v1"
model_logdir="$xav/ssl_epilepsy/jobs/relative_positioning/jh101/models"
stats_logdir="$xav/ssl_epilepsy/jobs/relative_positioning/jh101/stats"
pseudolabeled_data="$xav/ssl_epilepsy/data/pseudolabeled/jh101/jh101_12s_7min.pkl"

sbatch <<EOT
#!/bin/bash
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --cpus-per-task=16        # CPU cores/threads
#SBATCH --mem-per-cpu=10G        # memory per CPU core
#SBATCH --time=12:00:00     
#SBATCH --job-name=$job_name
#SBATCH --output=$xav/ssl_epilepsy/jobs/relative_positioning/tests/$job_name.out
#SBATCH --mail-user=xmootoo@gmail.com
#SBATCH --mail-type=ALL

cd $xav/ssl_epilepsy/ssl-seizure-detection/relative_positioning/tensorflow
module load cuda cudnn python/3.10
source ~/tf2.12/bin/activate

python main.py $model_logdir $stats_logdir $pseudolabeled_data

EOT