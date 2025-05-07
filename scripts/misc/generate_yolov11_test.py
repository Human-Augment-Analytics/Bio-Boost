"""
From merged CSV generate frames, can only do one video at a time. 
"""

import pandas as pd
import cv2
import os


def count_unique_videos(csv_file, output_txt="vid_downloads.txt"):
    df = pd.read_csv(csv_file)
    if not {'project', 'vid_id'}.issubset(df.columns):
        print("CSV doesn't contain 'project' and 'vid_id' columns.")
        return

    # Drop duplicate rows based on project and vid
    unique_videos = df[['project', 'vid_id']].drop_duplicates()
    count = unique_videos.shape[0]
    
    try:
        with open(output_txt, "w") as f:
            f.write(f"Total unique videos: {count}\n")
            for _, row in unique_videos.iterrows():
                f.write(f"{row['project']}/{row['vid_id']}.mp4\n")
    except Exception as e:
        print(f"Error writing to file {output_txt}: {e}")
        return

    print(f"Counted {count} unique videos")

def sample_frames_from_video(video_path, csv_file):
    df = pd.read_csv(csv_file)
    required_cols = {'project', 'vid_id', 'vid_frame_num', 'track_frame_num', 'filepath', 'is_male'}
    if not required_cols.issubset(df.columns):
        print(f"CSV file must contain the following columns: {required_cols}")
        return

    # Extract project and vid_id from the video_paths
    video_dir, video_file = os.path.split(video_path)
    vid_id = os.path.splitext(video_file)[0]
    project = os.path.basename(video_dir)
    print(f"{project} - {vid_id}")

    # Select rows corresponding to both the current project and vid_id.
    rows = df[(df['project'] == project) & (df['vid_id'] == vid_id)]
    if rows.empty:
        print(f"No CSV entries found for project '{project}' with video id '{vid_id}'.")
        return

    # Open the video file using cv2.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video {video_path} opened. Total frames: {total_frames}")
    
    # Process each row for sampling.
    for idx, row in rows.iterrows():
        try:
            start_frame = int(row['vid_frame_num'])
            offset = int(row['track_frame_num'])
        except ValueError as ve:
            print(f"Invalid frame numbers in row {idx}: {row.to_dict()}. Error: {ve}")
            continue

        frame_to_capture = start_frame + offset

        if frame_to_capture >= total_frames:
            print(f"Error: computed frame {frame_to_capture} exceeds total frames {total_frames} in {video_path}")
            continue

        # Seek to the desired frame.
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_capture)
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"Error reading frame {frame_to_capture} from {video_path}")
            continue

        # Determine destination folder based on is_male.
        is_male_val = str(row['is_male']).strip().lower()
        if is_male_val in ['1', 'true', 'yes']:
            save_dir = os.path.join("test", "male")
        else:
            save_dir = os.path.join("test", "female")

        os.makedirs(save_dir, exist_ok=True)

        # Use the 'filepath' column to name the sampled jpg.
        img_name = str(row['filepath'])
        if not img_name.lower().endswith(".jpg"):
            img_name += ".jpg"
        img_path = os.path.join(save_dir, img_name)

        # Save image to disk.
        try:
            cv2.imwrite(img_path, frame)
            print(f"Saved frame {frame_to_capture} from {video_path} to {img_path}")
        except Exception as e:
            print(f"Error saving image to {img_path}: {e}")

    cap.release()

    # After processing, remove all rows for the current video that match both project and vid_id.
    try:
        updated_df = df[~((df['project'] == project) & (df['vid_id'] == vid_id))]
        updated_df.to_csv(csv_file, index=False)
        print(f"Removed rows for project '{project}' and video id '{vid_id}' from {csv_file}.")
    except Exception as e:
        print(f"Error updating CSV file {csv_file}: {e}")


count_unique_videos("matched_data.csv")
# sample_frames_from_video("MC_singlenuc96_b1_Tk41_081120/0001_vid.mp4", "matched_data.csv") # Change first arg per video