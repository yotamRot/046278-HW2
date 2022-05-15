#!/bin/bash
max_load=$1
max_load_cur=$(( $max_load/10 ))
max_load_max=$(( $max_load*2 ))
step=$(((  $max_load_max-$max_load_cur )/10 ))
while [ $max_load_cur -le $max_load_max ]
do
   echo "$max_load_cur - $max_load_max"
   ./ex2 streams $max_load_cur | grep throughput | cut -d " " -f 3 >> out.txt
   max_load_cur=$(( $max_load_cur+$step ))
done