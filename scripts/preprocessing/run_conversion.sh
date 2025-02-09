#!/bin/bash

if [ "$1" == "python" ]; then
    shift
    echo "Running python conversion script"
    python /root/catkin_ws/extract_videos.py "$@"
elif [ "$1" == "shell" ]; then
    shift
    echo "Running shell conversion script."
    /root/catkin_ws/extract_videos.sh "$@"
else
    echo "Usage: docker run ros-bag-to-mp4 {python|shell} [args]"
    exit 1
fi
