#!/bin/bash
#SBATCH --job-name=Behave_exp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gl3048@nyu.edu
#SBATCH --open-mode=append
#SBATCH --output=./logs/%A_%a.out
#SBATCH --error=./logs/%A_%a.err
#SBATCH --export=ALL
#SBATCH --time=10:00:00       
#SBATCH --gres=gpu:1          # Request 1 GPU
#SBATCH --mem=16G             # Adjust memory as needed
#SBATCH --cpus-per-task=1
#SBATCH --array=1-10         # CHANGE THIS to the number of rows in your CSV

# 1. Load your environment (NYU Greene specific)
module purge
module load anaconda3/2020.07
# 3. Initialize Conda for this bash script
eval "$(conda shell.bash hook)"

# 4. Activate your environment
conda activate /scratch/gl3048/Behavioral-MFG/jax_gpu
# 2. Get the specific line from the CSV (skipping header)
# This assumes your file is named 'params.csv'
# Task 1 gets line 2, Task 2 gets line 3, etc.
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" experiments.csv)

# 3. Parse CSV into variables (comma-separated)
IFS=',' read -r env_name seed folder_name file_name <<< "$LINE"

# 4. Run the Python script with these arguments
python main.py \
    --env_name "$env_name" \
    --seed "$seed" \
    --folder_name "$folder_name" \
    --file_name "$file_name"    
