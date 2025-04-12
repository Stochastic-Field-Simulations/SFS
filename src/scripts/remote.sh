# SSH into this computer, and run this from NPSTED/
# Will run in the backround, so not interupted if ssh breaks
# Output written to remote.log
nohup sh scripts/run.sh >scripts/remote.log 2>&1 </dev/null &
