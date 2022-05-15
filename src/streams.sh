#!/bin/bash
max_load=260000
max_load_cur=$(( $max_load/10 ))
max_load_max=$(( $max_load*2 ))
while [ $max_load_cur -le $max_load_max ]
do
   ./ex2 streams $max_load_cur | grep throughput | cut -d " " -f 3
   max_load_cur=$(( $max_load_cur+10 ))
done