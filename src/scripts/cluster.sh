# run number, name, where to save data, job script

run=1
n=100
name=NAME
data_dir="/scratch04.local/mjohnsrud/DATADIR"
job="SCRIPTPATH"

if [ -d "$data_dir" ]; then 
    echo "Directory $data_dir exists."
else
    echo "Directory $data_dir does not exist. Creating now."
    mkdir -p $data_dir
    echo "Directory created."
fi

rm -r ./cluster_output/$name*

qsub -N $name -t 1-$n -v job=$job -v n=$n -v job_dir=$job_dir -v data_dir=$data_dir -v run=$run scripts/jobscript.sh
