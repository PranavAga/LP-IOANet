#!/bin/bash
#SBATCH -A research
#SBATCH --time=24:00:00
#SBATCH -w gnode041
#SBATCH -n 40
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:4
#SBATCH --output=op_final.txt

set -e

module load u18/cuda/10.2

echo "PWD: $PWD"
WORKING_DIR=/scratch/$USER
if [ -d "$WORKING_DIR" ]; then rm -Rf $WORKING_DIR; fi
mkdir -p $WORKING_DIR
cd $WORKING_DIR
echo "PWD: $PWD"

SECONDS=0
scp -r ada:~/smai/SD7K/train ./
scp -r ada:~/smai/SD7K/test ./
scp -r ada:~/smai/models/shadrem-att.pth ./
duration=$SECONDS
echo "$((duration / 60)) minutes elapsed."

echo "transferd data"

python3 ~/smai/lp_att_model.py
echo "trained model"

scp  ./lpnet_shadrem-att.pth ada:~/smai/models
echo "transfered model"
