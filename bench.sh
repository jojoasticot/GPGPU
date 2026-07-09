#! /bin/sh

echo "" > a

for file in $(ls test_videos/lil_clown_studio.mp4 test_videos/video03.avi); do
    echo "FILE: $file" >> a
    for i in $(seq 10); do 
        echo "BENCHMARK $i, file $file"
        gst-launch-1.0 -e -v uridecodebin uri=file://$(pwd)/$file !  videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! fpsdisplaysink video-sink=fakesink sync=false >> a
        echo "-----------------------"
        # | grep "\(Execution ended\)\|average:";
    done
done
