#!/bin/bash -l
# BEGIN SLRUM OPTIONS DECLARATIONS
# Export all environment variables
#SBATCH --export=ALL
#SBATCH --export=TMPDIR=/tmp
#SBATCH --propagate=NONE
#SBATCH --time=2-00:00:00
# Your job name
#SBATCH --job-name=testrun
#
# Use current working directory
#SBATCH --chdir=###
#
# Set queue to use
#SBATCH --partition=ceoas-arm
#
#SBATCH --nodelist=youmu
# 
# Requesting more memory
#SBATCH --mem=32G
#
# Asking for the gpu
#SBATCH --gres=gpu:1
#
# Output files for stdout and stderr
#SBATCH --output=testrun/testrun.o
#SBATCH --error=testrun/testrun.e
#
# Send an email at the beginning or end of the job (b and e), or if it's aborted or rescheduled (a)
#SBATCH --mail-user=###
#SBATCH --mail-type=ALL
#
# END SLURM OPTIONS DECLARATIONS
#
PATH=###
LD_LIBRARY_PATH=###
export PATH
export R_LIBS=###
#
#
#The following auto-generated commands will be run by the execution node.
#We execute your command via /usr/bin/time with a custom format
#so that the memory usage and other stats can be tracked; note that
#GNU time v1.7 has a bug in that it reports 4X too much memory usage
source /fs1/home/ceoas/leendere/miniforge3/etc/profile.d/conda.sh
conda activate tm_arm

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH


echo "  Started on:           " `/bin/hostname -s` 
echo "  Started at:           " `/bin/date` 
/usr/bin/time -f " \\tFull Command:                      %C \\n\\tMemory (kb):                       %M \\n\\t# SWAP  (freq):                    %W \\n\\t# Waits (freq):                    %w \\n\\tCPU (percent):                     %P \\n\\tTime (seconds):                    %e \\n\\tTime (hh:mm:ss.ms):                %E \\n\\tSystem CPU Time (seconds):         %S \\n\\tUser   CPU Time (seconds):         %U " \
python3 Train.py
echo "  Finished at:           " `date` 
