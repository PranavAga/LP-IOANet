#!/bin/bash
#SBATCH -A research
#SBATCH -p long
#SBATCH -w gnode040
#SBATCH -n 40
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:4
#SBATCH --output=op_model.txt

set -e

module load u18/cuda/10.2

echo "PWD: $PWD"
WORKING_DIR=/scratch/$USER
if [ -d "$WORKING_DIR" ]; then rm -Rf $WORKING_DIR; fi
mkdir -p $WORKING_DIR
cd $WORKING_DIR
echo "PWD: $PWD"

mkdir -p ./input/
mkdir -p ./target/
echo "list:"
ls

#       local location
scp -r ada:~/smai/lowresSD7K/train/input/* ./input
scp -r ada:~/smai/lowresSD7K/train/target/* ./target

ls ./input | wc -l
ls ./target | wc -l
echo "transferd data"

python3 ~/smai/att_model.py
echo "trained model"

#                        local location
scp  ./shadrem-att.pth ada:~/smai/models
echo "transfered model"
