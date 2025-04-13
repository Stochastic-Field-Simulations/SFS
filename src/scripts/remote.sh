# SSH into this computer, and run this from ../SFS/
# Will run in the backround, so not interupted if ssh breaks
# Output written to remote.log
nohup sh SFS/src/scripts/run.sh >SFS/src/scripts/remote.log 2>&1 </dev/null &
