#!/usr/bin/env bash

SECONDS=0
N=200
url="http://10.3.9.222:5003" # add more URLs here

for i in in $( seq 0 $N )
do
   # run the curl job in the background so we can start another job
   # and disable the progress bar (-s)
   echo "fetching $url"
   curl $url -s -F image=@misc/old/NVT.jpg &
done
wait #wait for all background jobs to terminate

duration=$SECONDS
printf "\nThroughput: %.2f req/s" $(echo "scale=2; $N / $duration" | bc)
