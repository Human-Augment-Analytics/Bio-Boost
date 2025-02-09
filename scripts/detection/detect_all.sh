#!/bin/bash

# Directory containing the .mp4 files
INPUT_DIR="/home/charlieclark/lindenthal-videos"
OUTPUT_DIR="/home/charlieclark/lindenthal-yolo-preds"
MODEL_PATH="runs/train/exp/weights/best.pt"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Iterate over each .mp4 file in the input directory
for video in "$INPUT_DIR"/*.mp4; do
    echo "Processing video: $video"
    
    # Extract the base name of the video file (without extension)
    base_name=$(basename "$video" .mp4)
    
    # Set the output directory for the current video
    video_output_dir="$OUTPUT_DIR/$base_name"
    mkdir -p "$video_output_dir"
    
    # Run YOLOv5 detection on the video file (frame-by-frame)
    # --source is the video, --project is the output directory, --name is the video name
    python3 detect.py --weights "$MODEL_PATH" --source "$video" --project "$video_output_dir" --name "detections"
    
    echo "Finished processing: $video"
done

echo "All videos processed."
