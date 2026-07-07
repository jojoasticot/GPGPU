#! /bin/sh

for i in $(seq 10); do 
    echo "BENCHMARK $i"
    gst-launch-1.0 -e -v uridecodebin uri=file://$(pwd)/video03.avi !  videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! fpsdisplaysink video-sink=fakesink sync=false;
    echo "-----------------------"
    # | grep "\(Execution ended\)\|average:";
done > a
