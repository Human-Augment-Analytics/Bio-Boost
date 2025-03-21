import glob
import os
import shutil
import cv2

def process_video(video_path, output_dir, target_size, filename, fps=29):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening: {video_path}")
        return
    
    frame_count = 0
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Scale frame to current training data
        resized_frame = cv2.resize(frame, target_size)

        # Save as jpg
        frame_filename = os.path.join(output_dir, f"{filename}_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, resized_frame)
        frame_count += 1
    
    cap.release()
    print(f"Processed {frame_count} frames from {filename}")

def main():
    """
    Step 1: Filter out videos in the test dataset that also appear in the training dataset
    """
    # Get list of video names from training dataset
    training_dataset = set()
    input_directories = [
        "../YOLOv11/data/train/female",
        "../YOLOv11/data/train/male",
        "../YOLOv11/data/val/female",
        "../YOLOv11/data/val/male"
    ]
    initial_test_directories = [
        "./Female",
        "./Male"
    ]

    # Iterate through training data, and add all videos to set
    for directory in input_directories:
        files = glob.glob(os.path.join(directory, "**", "*.jpg"), recursive=True)

        # Iterate through every file in the female/male directories
        for image in files:
            # Parse file to get video name
            initial_image = os.path.basename(image) # remove directories
            image_name = initial_image.rsplit(".jpg", 1)[0] # remove .jpg
            video_name = image_name.rsplit("__", 1)[0]
            training_dataset.add(video_name) # add to set

    # Initial iteration through test data: remove duplicates
    for directory in initial_test_directories:
        files = glob.glob(os.path.join(directory, "**", "*.mp4"), recursive=True)
        num_duplicates = 0
        class_id = ""

        for video in files:
            base_video = os.path.basename(video) # remove directories
            video_name = base_video.rsplit(".mp4", 1)[0] # remove the .mp4
            clean_name = video_name.rsplit("__", 1)[0] # remove the __Male/__Female
            if "Female" in directory:
                class_id = "Female"
            else:
                class_id = "Male"

            if clean_name in training_dataset:
                num_duplicates += 1
                os.remove(video)
        
        print(f"{num_duplicates} duplicates found in {class_id} directory")


    """
    Step 2: Exclude projects that were not used in training
    """
    valid_projects = set()
    for directory in input_directories:
        files = glob.glob(os.path.join(directory, "**", "*.jpg"), recursive=True)

        for image in files:
            base_image = os.path.basename(image) # remove directories
            project_name = base_image.split("__", 1)[0] # grab project name
            valid_projects.add(project_name)

    for directory in initial_test_directories:
        files = glob.glob(os.path.join(directory, "**", "*.mp4"), recursive=True)
        num_invalid_videos = 0
        num_valid_projects = 0

        for video in files:
            if "Female" in directory:
                class_id = "female"
            else:
                class_id = "male"
            base_video = os.path.basename(video)
            project_name = base_video.split("__", 1)[0]

            if project_name in valid_projects:
                # Keep video in directory
                num_valid_projects += 1
            else:
                os.remove(video)
                num_invalid_videos += 1
        print(f"{num_valid_projects} valid projects found in {class_id} directory")
        print(f"{num_invalid_videos} invalid projects found in {class_id} directory")
    
    """
    Step 3: Ensure proper scaling (compare to training dataset)
    """
    sample_files = glob.glob(os.path.join(input_directories[0], "*.jpg"))
    if not sample_files:
        raise Exception("No files found in training dataset")
    
    sample_img = cv2.imread(sample_files[0])
    if sample_img is None:
        raise Exception(f"Failed to load image: {sample_img}")
    height, width = sample_img.shape[:2]
    scaling = (width, height)
    print(f"Scaling: {scaling}")

    """
    Step 4: Generate frames from valid videos
    """
    valid_data = [
        "./Female",
        "./Male"
    ]

    for directory in valid_data:
        files = glob.glob(os.path.join(directory, "**", "*.mp4"), recursive=True)
        if "Female" in directory:
            print(f"Remaining Female Videos: {len(files)}")
        else:
            print(f"Remaining Male Videos: {len(files)}")
        print("Starting: ", directory)
        i = 1
        size = len(files)

        for video in files:
            base_video = os.path.basename(video)
            video_name = base_video.rsplit("__", 1)[0]
            if "Female" in directory:
                output_dir = f"./test/female"
            else: 
                output_dir = f"./test/male"

            process_video(video, output_dir, scaling, video_name)
            print(f"Processed video {i} / {size}")
            i += 1


if __name__ == "__main__":
    main()