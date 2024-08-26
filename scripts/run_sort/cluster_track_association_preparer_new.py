import os, pdb, cv2
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon

class ClusterTrackAssociationPreparer():

	# This class takes in directory information and a logfile containing depth information and performs the following:
	# 1. Identifies tray using manual input
	# 2. Interpolates and smooths depth data
	# 3. Automatically identifies bower location
	# 4. Analyze building, shape, and other pertinent info of the bower

	def __init__(self, fileManager):

		self.__version__ = '1.0.0'
		self.fileManager = fileManager
		self.validateInputData()

	def validateInputData(self):
		
		assert os.path.exists(self.fileManager.localLogfileDir)
		assert os.path.exists(self.fileManager.localOldVideoCropFile)
		assert os.path.exists(self.fileManager.localAllLabeledClustersFile)
		assert os.path.exists(self.fileManager.localMaleFemalesVideosDir)
		return
		for videoIndex in range(len(self.fileManager.lp.movies)):
			videoObj = self.fileManager.returnVideoObject(videoIndex)
			assert os.path.exists(videoObj.localFishTracksFile)
			assert os.path.exists(videoObj.localFishDetectionsFile)

	def summarizeTracks(self, minimum_frame_number = 30):

		# Loop through videos and combine into a single file
		for videoIndex in range(len(self.fileManager.lp.movies)):
			videoObj = self.fileManager.returnVideoObject(videoIndex)

			#Read in individual video detection and tracks files
			video_dt_t = pd.read_csv(videoObj.localFishTracksFile)
			video_dt_t['xc'] = videoObj.width*video_dt_t['xc']
			video_dt_t['yc'] = videoObj.height*video_dt_t['yc']
			video_dt_t['w'] = videoObj.width*video_dt_t['w']
			video_dt_t['h'] = videoObj.height*video_dt_t['h']
			
			video_dt_d = pd.read_csv(videoObj.localFishDetectionsFile)

			# Combine them into a single master pandas DataFrame
			try:
				dt_t = dt_t.append(video_dt_t)
				dt_d = dt_d.append(video_dt_d)
			except NameError:
				dt_t = video_dt_t
				dt_d = video_dt_d

		# Save raw detections file
		dt_d.to_csv(self.fileManager.localAllFishDetectionsFile)

		# Use video_crop to determine if fish is inside or outside the frame
		video_crop = np.load(self.fileManager.localOldVideoCropFile)
		poly = Polygon(video_crop)
		dt_t['InBounds'] = [poly.contains(Point(x, y)) for x,y in zip(dt_t.xc, dt_t.yc)]

		# Determine track lengths (useful for identifing longest tracks for manual annotation)
		track_lengths = dt_t.groupby(['track_id','base_name']).count()['p_value'].rename('track_length').reset_index()
		track_lengths = track_lengths[track_lengths.track_length > minimum_frame_number]
		dt_t = pd.merge(dt_t, track_lengths, left_on = ['track_id','base_name'], right_on = ['track_id','base_name'])
		#dt_t['binned_track_length'] = dt_t.track_length.apply(bin_tracklength)

		dt_t.to_csv(self.fileManager.localAllFishTracksFile, index = False)

		t_dt = dt_t.groupby(['track_id', 'track_length', 'base_name']).mean()[['class_id', 'p_value','InBounds']].rename({'class_id':'Reflection'}, axis = 1).reset_index().sort_values(['base_name','track_id'])
		t_dt.to_csv(self.fileManager.localAllTracksSummaryFile, index = False)

	def associateClustersWithTracks(self):
		c_dt = pd.read_csv(self.fileManager.localAllLabeledClustersFile, index_col = 0)
		c_dt['track_id'] = ''
		t_dt = pd.read_csv(self.fileManager.localAllFishTracksFile, index_col = 0, dtype={'base_name':'category'})

		t_dt['delta'] = 100000

		for i,cluster in c_dt.iterrows():
			if i % 250 == 0:
				print(i)
			if cluster.ClipCreated == 'Yes':

				possible_tracks = t_dt[(t_dt.base_name == cluster.VideoID) & (t_dt.frame.values > (cluster.t-0.5)*29) & (t_dt.frame.values < (cluster.t+0.5)*29) ]
				if len(possible_tracks) == 0:
					continue
				possible_tracks['delta'] = pow(possible_tracks['yc'] - cluster.X,2) + pow(possible_tracks['xc'] - cluster.Y, 2)
				track_id = possible_tracks.loc[possible_tracks['delta'].idxmin(),'track_id']
				c_dt.iloc[i]['track_id'] = track_id
		
		pdb.set_trace()

	def createMaleFemaleAnnotationVideos(self, n_videos = 25, delta_xy = 75):

		caps = {}
		for videoIndex in range(len(self.fileManager.lp.movies)):
			videoObj = self.fileManager.returnVideoObject(videoIndex)
			caps[videoObj.baseName] = cv2.VideoCapture(videoObj.localVideoFile)

		s_dt = pd.read_csv(self.fileManager.localAllTracksSummaryFile)
		t_dt = pd.read_csv(self.fileManager.localAllFishTracksFile)


		min_length = 1800
		final_candidates = []
		while len(final_candidates) < n_videos:
			mtracks = t_dt[t_dt.track_length > min_length].groupby(['base_name','frame']).count()['xc'].reset_index()
			if len(mtracks) == 0:
				min_length -= 300
				continue
			candidate_tracks = pd.merge(mtracks[mtracks.xc > 1], t_dt, on=['base_name','frame']).groupby(['base_name','track_id']).count()['w'].reset_index()
			candidate_tracks = candidate_tracks[candidate_tracks.w > 100]
			final_candidates = pd.merge(candidate_tracks, s_dt[(s_dt.track_length > min_length) & (s_dt.InBounds > 0.75)], on = ['base_name','track_id'])
			min_length -= 300
		
		for i,track in final_candidates.sample(n = n_videos).iterrows():
			outVideoFile = self.fileManager.localMaleFemalesVideosDir + self.fileManager.projectID + '__' + track.base_name + '__' + str(track.track_id) + '.mp4'
			tracks = t_dt[(t_dt.base_name == track.base_name) & (t_dt.track_id == track.track_id)]

			
			outAll = cv2.VideoWriter(outVideoFile , cv2.VideoWriter_fourcc(*"mp4v"), 30, (videoObj.width, videoObj.height))
#			outAll = cv2.VideoWriter(outVideoFile , cv2.VideoWriter_fourcc(*"mp4v"), 30, (2*delta_xy, 2*delta_xy))

			caps[tracks.base_name.min()].set(cv2.CAP_PROP_POS_FRAMES, tracks.frame.min())
			current_frame = tracks.frame.min()
			for j,current_track in tracks.iterrows():
				ret, frame = caps[tracks.base_name.min()].read()
				while current_track.frame != current_frame:
					 ret, frame = caps[tracks.base_name.min()].read()
					 current_frame += 1

				cv2.rectangle(frame, (int(current_track.xc - delta_xy), int(current_track.yc - delta_xy)), (int(current_track.xc + delta_xy), int(current_track.yc + delta_xy)), (255,0,0), 2)
				#cv2.rectangle(frame, (int(current_track.yc - delta_xy), int(current_track.xc - delta_xy)), (int(current_track.yc + delta_xy), int(current_track.xc + delta_xy)), (255,0,0), 2)
				outAll.write(frame)
				#outAll.write(frame[int(current_track.yc - delta_xy):int(current_track.yc + delta_xy), int(current_track.xc - delta_xy):int(current_track.xc + delta_xy)])
				current_frame += 1
			outAll.release()
			self.fileManager.uploadData(self.fileManager.localMaleFemalesVideosDir + self.fileManager.projectID + '__' + track.base_name + '__' + str(track.track_id) + '.mp4')
		# Group data together to single track
