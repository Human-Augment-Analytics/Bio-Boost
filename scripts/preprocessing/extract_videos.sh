#!/bin/bash

# start roscore
echo "Starting roslaunch server..."
roscore & ROSCORE_PID=$!

# define cleanup script to run at end
cleanup() {
    echo "Terminating roslaunch server..."
    kill $ROSCORE_PID
    exit
}

trap cleanup INT TERM EXIT

# iterate over each .bag file in the directory
TOPIC="/device_0/sensor_1/Color_0/image/data"
for BAG_FILE in "$1/todo/*.bag"; do
    # Extract the filename without extension
    FILE_NAME="${BAG_FILE%.*}"
    echo "Processing $BAG_FILE"

    # start image extraction in background
    rosrun image_view extract_images image:=$TOPIC _filename_format:="$1/done/frames/${FILE_NAME}_frame%04d.jpg" & ROSRUN_PID=$!

    # play the .bag file
    rosbag play --clock --rate=1.0 "$BAG_FILE" # FIX Attempt 1: Play the .bag file with forced rate and clock sync to ensure consistent timing

    sleep 1 # FIX Attempt 2: Add short delay for images to be written 

    # once done playing .bag file, terminate rosrun
    kill $ROSRUN_PID
    wait $ROSRUN_PID 2>/dev/null

    # save frames as .mp4 video with ffmpeg

    # Old: Sampling at forced 15 fps
    # ffmpeg -framerate 15 -i "$1/done/frames/${FILE_NAME}_frame%04d.jpg" -c:v libx264 -pix_fmt yuv420p "$1/done/videos/${FILE_NAME}.mp4"

    # New Attempt 1: Pair -framerate with -vsync vfr which adjusts frame rate based on time intervals (records at variable rate essentially)
    # Note: Not super optimistic about this one because the ChatGPT analysis told us that the timestamps seemed inconsistent 
    ffmpeg -framerate 15 -i "$1/done/frames/${FILE_NAME}_frame%04d.jpg" -vsync vfr -c:v libx264 -pix_fmt yuv420p "$1/done/videos/${FILE_NAME}.mp4" 

    # New Attempt 2: Use filter to resample video stream at a constant 15 fps
    # ffmpeg -framerate 15 -i "$1/done/frames/${FILE_NAME}_frame%04d.jpg" -filter:v "fps=fps=15" -c:v libx264 -pix_fmt yuv420p "$1/done/videos/${FILE_NAME}.mp4"

    # New Attempt 3: Drop the forced framerate from 15 to 10
    # ffmpeg -r 10 -i "$1/done/frames/${FILE_NAME}_frame%04d.jpg" -c:v libx264 -pix_fmt yuv420p "$1/done/videos/${FILE_NAME}.mp4"


    echo "Saved $1/done/videos/${FILE_NAME}.mp4"
done

# cleanup roscore
cleanup
