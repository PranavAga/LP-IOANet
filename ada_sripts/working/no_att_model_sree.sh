#!/bin/bash
#SBATCH -A research
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=sreenivas.bhumireddy@research.iiit.ac.in
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --output=no_att_model_sree_output.txt
#SBATCH --error=no_att_model_sree_error.txt
#SBATCH --job-name="no_att_model_sree"
#SBATCH -n 20
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:2
#SBATCH -w gnode062

set -e

module load u18/cuda/10.2

echo "PWD: $PWD"
WORKING_DIR=/scratch/$USER
if [ -d "$WORKING_DIR" ]; then rm -Rf $WORKING_DIR; fi
mkdir -p $WORKING_DIR
cd $WORKING_DIR
echo "PWD: $PWD"

mkdir -p ./temp_data/
# mkdir -p ./input/
# mkdir -p ./target/
echo "list:"
ls

#       local location
# scp -r ada:~/work/LP-IOANet/data/192,256/train/input/* ./input
# scp -r ada:~/work/LP-IOANet/data/192,256/train/target/* ./target

#  temp data files
scp -r ada:~/work/LP-IOANet/ada_sripts/temp_data/X_test.pth ./temp_data/X_test.pth
scp -r ada:~/work/LP-IOANet/ada_sripts/temp_data/Y_test.pth ./temp_data/Y_test.pth
scp -r ada:~/work/LP-IOANet/ada_sripts/temp_data/X_train.pth ./temp_data/X_train.pth
scp -r ada:~/work/LP-IOANet/ada_sripts/temp_data/Y_train.pth ./temp_data/Y_train.pth

# ls ./input | wc -l
# ls ./target | wc -l

ls ./temp_data | wc -l

echo "transferd data"

python3 ~/work/LP-IOANet/ada_sripts/script_without_att_4000.py
echo "trained model"

#                        local location
scp  ./shadrem-no-att.pth ada:~/smai/models
echo "transfered model"
