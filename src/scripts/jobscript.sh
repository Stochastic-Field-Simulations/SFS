#!/bin/bash

# Use bash as shell
#$ -S /bin/bash

# Save output files
#$ -o cluster_output/

# Preserve environment variables
#$ -V

# Execute from current working directory
#$ -cwd

#$ -q rostam.q

# Merge standard output and standard error into one file
#$ -j yes

# Standard name of the job (if none is given on the command line)
#$ -N MKJ

# Diagonstics
job_number=$SGE_TASK_ID
echo Started: `date`
echo on `hostname`
echo job_number = $SGE_TASK_ID;
echo -----------------

# launch with argumets from cluster.sh
/usr/ds/bin/julia --project=. -t 1 $job $run $job_number $n $data_dir

# Diagnostics
echo -----------------
echo Stopped: `date`
