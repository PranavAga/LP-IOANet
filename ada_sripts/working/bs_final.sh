#!/bin/bash
#SBATCH -A research
#SBATCH --time=12:00:00
#SBATCH --mail-user=sreenivas.bhumireddy@research.iiit.ac.in
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --output=v2_LPIONET_model_sree_output.txt
#SBATCH --error=v2_LPIONET_model_sree_error.txt
#SBATCH --job-name="v2_LPIONET_model_sree"
#SBATCH -n 40
#SBATCH --mem-per-cpu=2G
#SBATCH --gres=gpu:4
#SBATCH -w gnode076

set -e

module load u18/cuda/10.2

echo "PWD: $PWD"
WORKING_DIR=/scratch/$USER
if [ -d "$WORKING_DIR" ]; then rm -Rf $WORKING_DIR; fi
mkdir -p $WORKING_DIR
cd $WORKING_DIR
echo "PWD: $PWD"

SECONDS=0
scp -r ada:~/work/LP-IOANet/data/train ./
scp -r ada:~/work/LP-IOANet/data/test ./
scp -r ada:~/work/LP-IOANet/models/v2-shadrem-att.pth ./
duration=$SECONDS
echo "$((duration / 60)) minutes elapsed."

echo "transferd data"

python3 ~/work/LP-IOANet/ada_sripts/lp_att_model.py
echo "trained model"

scp  ./v2_lpnet_shadrem-att.pth ada:~/work/LP-IOANet/models
echo "transfered model"
