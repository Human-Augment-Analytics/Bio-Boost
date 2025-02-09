FROM ros:melodic

# Install dependencies
RUN apt-get update && apt-get install -y \
    ros-melodic-cv-bridge \
    ros-melodic-image-view \
    ffmpeg \
    python-opencv \
 && rm -rf /var/lib/apt/lists/*

# Set up workspace directory
WORKDIR /root/catkin_ws

# Copy conversion scripts into the image.
COPY extract_videos.py /root/catkin_ws/
COPY extract_videos.sh /root/catkin_ws/

# Make the shell script executable.
RUN chmod +x /root/catkin_ws/extract_videos.sh

# Copy custom wrapper script to select between the two scripts
COPY run_conversion.sh /root/catkin_ws/
RUN chmod +x /root/catkin_ws/run_conversion.sh

# Source ROS environment by default
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/melodic/setup.bash && exec \"$@\"", "--"]

# Default to running the wrapper script
CMD ["/root/catkin_ws/run_conversion.sh"]