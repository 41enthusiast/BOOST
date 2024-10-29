#!/bin/bash

#SBATCH -N 1 # Request a single node
#SBATCH -c 4 # Request four CPU cores
#SBATCH --gres=gpu:turing:1 # Request one gpu
 
#SBATCH -p res-gpu-small # Use the res-gpu-small partition
#SBATCH --qos=long-high-prio # Use the short QOS
#SBATCH -t 7-0 # Set maximum walltime to 1 day
#SBATCH --job-name=ood-paintings# Name of the job
#SBATCH --mem=16G # Request 16Gb of memory

#SBATCH -o program_output1_2.txt
#SBATCH -e whoopsies1_2.txt

# Load the global bash profile
source /etc/profile
module load cuda/11.0

# Load your Python environment
source ../../new_repos/mv_test1/bin/activate

# Run the code

# python3 wikiart_ds.py
# python3 pacs_ds.py

../../new_repos/mv_test1/bin/python3 main.py

# python3 vis_utils.py
# ../../new_repos/mv_test1/bin/python3 clf_vis_utils.py

# python3 gen_geom_feats_ds.py

# cd FastSAM; python3 sam_mask.py

# python3 resnet50_custom.py

# python3 compute_mem_req.py