#!/bin/bash
job_name="test_19_08_2023"

sbatch <<EOT
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=10G
#SBATCH --time=00:05:00
#SBATCH --job-name=$job_name
#SBATCH --output=$xav/ssl_epilepsy/jobs/$job_name.out
#SBATCH --mail-user=xmootoo@gmail.com
#SBATCH --mail-type=ALL

cd $xav/ssl_epilepsy/ssl-seizure-detection/relative_positioning/tensorflow/tests
module load python/3.10
source ~/tf2.12/bin/activate

python testcc.py

EOT