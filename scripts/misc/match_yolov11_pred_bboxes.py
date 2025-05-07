"""
For each unique_track_id, align the first prediction frame to the first GT frame, and then treat all other prediction frames as relative offsets from that GT starting frame.
"""

import pandas as pd
import numpy as np

# Load predictions and ground truth
pred = pd.read_csv('yolov11_pred.csv')
gt   = pd.read_csv('all_data.csv')

# Create unique_track_id and frame number column for prediction
pred[['unique_track_id', 'frame_num']] = (
    pred['filename']
        .str.replace(r'\.jpg$', '', regex=True)
        .str.rsplit('_', n=1, expand=True)
)
pred['frame_num'] = pred['frame_num'].astype(int) # make integer for later calculations

# For each unique_track id in gt, find the first frame number, so we can use it as offset
gt_start_frames = (
    gt.groupby('unique_track_id')['frame_num']
      .min()
      .rename('gt_start_frame')
)

# For prediction, add the starting frame/offset as a column to each unique frame id, then compute abs_frame to get full frame number
pred = pred.merge(gt_start_frames, on='unique_track_id', how='left')
pred['abs_frame'] = pred['gt_start_frame'] + pred['frame_num']

# There were some duplicates, so only keep the first match - Note: xc and yc did not appear to be different in dupes
gt_unique = gt.drop_duplicates(
    subset=['unique_track_id', 'frame_num'],
    keep='first'
)

# Merge prediction with GT by aligning on unique track id and matched frame number
matched = pred.merge(
    gt_unique,
    left_on=['unique_track_id', 'abs_frame'],
    right_on=['unique_track_id', 'frame_num'],
    how='inner',
    suffixes=('_pred', '_gt')
)

# Drop irrelevant columns
matched.drop(columns=[
    'speed', 'distance_traveled', 'mean_acceleration',
    'outreach_ratio', 'v_dot', 'u_dot', 'yolov5_class_id', 'track_type'
], errors='ignore', inplace=True)


# Calculate euclidean distance between pred & gt boxes and keep lowest calculated euclidean distance
matched["dist"] = np.sqrt((matched["x_center"] - matched["xc"])**2 + (matched["y_center"] - matched["yc"])**2)
matched = (
    matched.sort_values("dist")
        .groupby(['filename', 'abs_frame'], as_index=False)
           .first()
)
matched.to_csv('pred_gt_matched.csv', index=False)

# Print out to get an idea
mean_dist = matched["dist"].mean()
print(f"Mean Euclidean Distance: {mean_dist}")