import os
import sys
import cv2
import rosbag
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import argparse

def extract_video_from_bag(bag_file, output_dir, fps):
    # Initialize CvBridge
    bridge = CvBridge()

    # Get the base name and output MP4 file path
    base_name = os.path.basename(bag_file).replace('.bag', '')
    output_video_file = os.path.join(output_dir, f"{base_name}.mp4")

    # Open the bag file
    print(f"Processing: {bag_file}")
    with rosbag.Bag(bag_file, 'r') as bag:
        # Detect topics with video data
        video_topics = [
            topic for topic, info in bag.get_type_and_topic_info().topics.items()
            if info.msg_type in ["sensor_msgs/CompressedImage", "sensor_msgs/Image"] and 'Color_0' in topic
        ]

        # print(video_topics)

        if not video_topics:
            print(f"No video topics found in {bag_file}. Skipping.")
            return

        # Choose the first video topic
        video_topic = video_topics[0]
        print(f"Using video topic: {video_topic}")

        # Get the frame size and initialize the video writer
        frame_width, frame_height = None, None
        video_writer = None

        for topic, msg, t in bag.read_messages(topics=[video_topic]):
            # Convert ROS image message to OpenCV format
            if msg._type == "sensor_msgs/CompressedImage":
                cv_image = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            elif msg._type == "sensor_msgs/Image":
                cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
            else:
                continue

            if frame_width is None or frame_height is None:
                frame_height, frame_width = cv_image.shape[:2]
                video_writer = cv2.VideoWriter(
                    output_video_file,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,  # Use the specified FPS
                    (frame_width, frame_height)
                )

            video_writer.write(cv_image)

        # Release the video writer
        if video_writer:
            video_writer.release()
            print(f"Saved video to {output_video_file}")
        else:
            print(f"No frames extracted from {bag_file}. Skipping.")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract video data from ROS1 .bag files and save as MP4.")
    parser.add_argument("basedir", help="Directory containing .bag files.")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second for the output video (default: 30).")
    args = parser.parse_args()

    base_dir = args.basedir
    fps = args.fps

    input_dir = os.path.join(base_dir, 'todo')

    # Ensure the input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a valid directory.")
        sys.exit(1)

    # Create the output directory if it doesn't exist
    output_dir = os.path.join(base_dir, 'done', 'videos')
    os.makedirs(output_dir, exist_ok=True)

    # Process each .bag file in the directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.bag'):
            # print(f'{file_name}...')

            bag_file = os.path.join(input_dir, file_name)
            extract_video_from_bag(bag_file, output_dir, fps)

    print("Processing complete.")

if __name__ == "__main__":
    main()
