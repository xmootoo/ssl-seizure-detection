import submitit
import os
import wandb

def run_training(x):

    # Environment setup
    # wandb_api_key = os.environ.get("WANDB_API_KEY")
    # if not wandb_api_key:
    #     raise ValueError("WANDB_API_KEY not set in environment variables")
    
    module_load = "module load cuda/11.7 cudnn/8.9.5.29 python/3.10"
    activate_env = "source ~/torch2_cuda11.7/bin/activate"

    # Define the command to run, including module load and environment activation
    command = f"{module_load} && {activate_env} && python test.py {x} " 

    os.system(command)

def submit_job(x):
    # Create an executor for SLURM
    logdir="/home/xmootoo/projects/def-milad777/xmootoo/ssl_epilepsy/jobs"
    executor = submitit.AutoExecutor(folder=logdir)

    # Set SLURM job parameters
    executor.update_parameters(
        gpus_per_node=1,
        tasks_per_node=1,
        cpus_per_task=1,
        mem_gb=4,  # Adjust memory as needed
        time="00:01:00",
        slurm_partition="def-milad777",  # Replace with your SLURM partition
        slurm_array_parallelism=1
    )

    # Submit the job
    job = executor.submit(run_training, x)
    print(f"Job submitted: {job.job_id}")

if __name__ == "__main__":
    submit_job(4)