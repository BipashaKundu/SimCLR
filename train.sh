#!/bin/bash -l

#SBATCH --job-name=gpu_train2		# Name for your job
#SBATCH --comment="Testing Simclr"		# Comment for your job

#SBATCH --account=atrium		# Project account to run your job under
#SBATCH --partition=tier3 		# Partition to run your job on

#SBATCH --output=%x_%j.out		# Output file
#SBATCH --error=%x_%j.err		# Error file

#SBATCH --mail-user=bk7944@g.rit.edu	# Slack username to notify
#SBATCH --mail-type=END			# Type of slack notifications to send

#SBATCH --time=3-00:00:00		# Time limit
#SBATCH --ntasks=1		
#SBATCH --mem=80g		
#SBATCH --gres=gpu:a100:1		



conda activate simclr_gpu

python3 -u run.py -data heart_data/unlabeled/ -a resnet18 -dataset-name cardiac_data -b 32 --log-every-n-steps 100 --epochs 100 --save_dir gpu_train2
