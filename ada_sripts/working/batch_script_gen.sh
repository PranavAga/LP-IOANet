#!/bin/bash
#SBATCH -A research
#SBATCH -p long
#SBATCH -w gnode040
#SBATCH -n 40
#SBATCH --mem-per-cpu=2G
#SBATCH --output=op_file_gen.txt

set -e

SET=test
TYPE=target

echo $SET"/"$TYPE

rm -rf /scratch/$USER

scp -r ada:/share1/$USER/smai/SD7K/$SET/$TYPE /scratch/$USER
echo "transfered original images"

cd /scratch/$USER/
echo "PWD: $PWD"
python3 ~/smai/gen_lowres.py
echo "generated low resolution images"

scp -r ./lowres/* ada:~/smai/lowresSD7K/$SET/$TYPE