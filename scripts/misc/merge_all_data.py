"""
Match SORT data with testing frames and merge into CSV
"""

import pandas as pd

# Read in all_data and testing set CSVs
all_data = pd.read_csv("all_data.csv")
test_data = pd.read_csv("smaller_expanded_testing_set.csv")
pd.set_option('display.max_colwidth', None) # print out full values

# Drop the directory, .jpg, and frame number from filepath
test_data['track_id'] = test_data['filepath'].str.replace(r'^(male/|female/)|_\d+\.jpg$', '', regex=True)

# Drop just the directory from filepath 
test_data['filepath'] = test_data['filepath'].str.replace(r'^(male/|female/)', '', regex=True)

# Sort the DataFrame by the extracted frame number
test_data['track_frame_num'] = test_data['filepath'].str.extract(r'_(\d+)\.jpg$').astype(int)
test_data = test_data.sort_values(by=['track_id', 'track_frame_num'])

# Drop unnecessary columns
test_data = test_data.drop(columns=["distance", "speed", "acceleration", "outreach_ratio", "sqr_displacement", "mean_turning_angle", "jerk", "rms_velocity"])

# In new dataframe store all track_ids that are the same
matched_data = test_data[test_data['track_id'].isin(all_data['track_id'])]

# For all track_ids in all_data that match track_ids in matched_data - take the corresponding frame_num from all_data
track_id_to_frame = all_data.drop_duplicates('track_id')[['track_id', 'frame_num']]
track_id_to_frame = track_id_to_frame.set_index('track_id')['frame_num']
matched_data['vid_frame_num'] = matched_data['track_id'].map(track_id_to_frame)

# Split track_id into project and vid name for rclone
matched_data['project'] = matched_data['track_id'].str.extract(r'^(.*)__\d+_vid')[0]
matched_data['vid_id'] = matched_data['track_id'].str.extract(r'__(\d+_vid)')[0]

if len(test_data) != len(matched_data):
    print("Error: Could not find all test data in all_data.csv")

matched_data.to_csv("matched_data.csv")

print(matched_data.head(10))
