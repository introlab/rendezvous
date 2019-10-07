#!/bin/bash

OUTPUT=$(top -c -n 1 -b | grep -w 'python src/app/main.py')

ARRAY=()
PIDS=()
N=0
J=0

for i in $(echo $OUTPUT | tr " " "\n")
do
  if [ "$i" = "src/app/main.py" ] && [ "${ARRAY[${N}-2]}" != "-w" ]; then
    PIDS[${J}]=${ARRAY[${J}]}
    J=${N}+1
  fi
  ARRAY[${N}]=${i}
  N=${N}+1
done

for i in "${PIDS[@]}"
do
   echo ${i}
   sudo kill ${i}
done

