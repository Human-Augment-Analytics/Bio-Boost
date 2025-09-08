# GENERIC IMPORTS
import numpy as np
import pandas as pd



# DISTANCE TRAVELED: This is the total Euclidean pixel distance travelled in a track.
def calculate_distance_traveled(xc, yc):
    xc = np.array(xc)
    yc = np.array(yc)
    dx = np.diff(xc)
    dy = np.diff(yc)
    distances = np.sqrt(dx**2 + dy**2)
    distance_traveled = np.sum(distances)
    return distance_traveled



# SPEED: This should give us the correct speed values frame-by-frame.
# After the speed is taken for all values in the track, we can find the mean of speeds from the output.
def calculate_speed(df):
    df['speed'] = np.sqrt(df.loc[:, 'u_dot']**2 + df.loc[:, 'v_dot']**2)
    return df

def calculate_speed_new(df):
    df['speed'] = len(df) / len(df)



# ACCELERATION: After the mean is taken accross the track, it should give us mean acceleration.
def calculate_acceleration(u_dot, v_dot, time_intervals):
    u_dot = np.array(u_dot)
    v_dot = np.array(v_dot)
    time_intervals = np.array(time_intervals)
    ax = np.diff(u_dot) / time_intervals[:-1]
    ay = np.diff(v_dot) / time_intervals[:-1]
    acceleration = np.sqrt(ax**2 + ay**2)
    return ax, ay, acceleration



# NORMALIZED MAXIMUM DISPLACEMENT: Similar, but slightly different from, outreach ratio.
def max_displacement(df):
    start = df[['xc', 'yc']].iloc[0].to_numpy()
    coords = df[['xc', 'yc']].to_numpy()
    distances = np.linalg.norm(coords - start, axis=1)
    return np.max(distances)

def path_length(df):
    coords = df[['xc', 'yc']].to_numpy()
    segment_distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    return np.sum(segment_distances)

def calculate_norm_max_displacement(df):
    dmax = max_displacement(df)
    length = path_length(df)
    return dmax / length if length > 0 else np.nan



# AUTOCORRELATION: Autocorrelation looks at the cyclic or repetitiveness of the track over a time period.
# Finding the Euclidean Displacement from Start
### Outputs column of displacement values.
def find_displacement_from_start(df):
    start_x, start_y = df.iloc[0][['xc', 'yc']]
    displacement = np.sqrt((df['xc'] - start_x)**2 + (df['yc'] - start_y)**2)
    return displacement

# Use Cubic Spline Interpolation to Smooth Curve
### Outputs column of smoothed displacement values.
from scipy.interpolate import CubicSpline
def smooth_displacement_curve(df):
    displacement = find_displacement_from_start(df)
    frame_indices = np.arange(len(df))
    cs = CubicSpline(frame_indices, displacement)
    smoothed_displacement = cs(frame_indices) 
    return smoothed_displacement

# Find the Autocorrelation
### Use fast Fourier transform.
### N lags will probably always be 100, because we already filtered for longer tracks.
### Outputs array of values that is the number of lags + 1.
from statsmodels.tsa.stattools import acf
def calculate_autocorrelation(df):
    smoothed_displacement = smooth_displacement_curve(df)
    nlags = min(100, len(df) // 2)
    autocorr_values = acf(smoothed_displacement, nlags=nlags, fft=True)
    return autocorr_values



# CROSS CORRELATION: Cross correlation looks at the straightness (linear vs circular) of the track over time.
### Output is a single value between -1 to 1.
### Closer to -1 or 1 means straight line in different directions.
### Closer to 0 would be circular motion.
def calculate_cross_correlation(df):
    x = df['xc']
    y = df['yc']
    correlation = np.corrcoef(x, y)[0, 1]  # Pearson correlation coefficient
    return correlation

# Median Smoothing
def calculate_median_smoothing(df):
    smoothed_df = df.copy()
    window_size = 21
    smoothed_df['xc'] = df['xc'].rolling(window=window_size, center=True, min_periods=window_size).median()
    smoothed_df['yc'] = df['yc'].rolling(window=window_size, center=True, min_periods=window_size).median()
    smoothed_df.dropna(subset=['xc', 'yc'], inplace=True)
    return smoothed_df

# Combining Cross Correlation from Other Paper with Median Smoothing
def calculate_cross_correlation_with_median_smoothing(df):
    df = calculate_median_smoothing(df)
    x = df['xc']
    y = df['yc']
    correlation = np.corrcoef(x, y)[0, 1]  # Pearson correlation coefficient
    return correlation



# LOCATION-BASED CLUSTERING: Number of Residence Patch Centroids
from sklearn.cluster import DBSCAN
def calculate_num_residence_patch_centroids(df, eps=5, min_samples=10):
    coords = df[['xc', 'yc']].values
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    df['patch_id'] = clustering.labels_
    num_patches = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    return num_patches