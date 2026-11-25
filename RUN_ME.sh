#!/bin/bash

# Entry point. If a SLURM_JOB.sh exists, the job is submitted to the cluster;
# otherwise LOCAL_JOB.sh runs it on this machine.
#
# SLURM is only one example: any server or scheduler setup can be plugged in
# here in the same way, by adding a job script and the command that submits it.

if [ -f "SLURM_JOB.sh" ]; then
  # Create the output folder if it doesn't exist, the SLURM-setup does
  # not work otherwise.
  mkdir -p "SLURM/outputs"
  sbatch SLURM_JOB.sh
else
  # Local jobs can occupy the terminal.
  bash LOCAL_JOB.sh
fi
