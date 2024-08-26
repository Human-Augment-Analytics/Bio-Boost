import subprocess, os, pdb, datetime
from shapely.geometry import Point, Polygon

class FishTrackingPreparer():
	# This class takes in a yolov5 model + video information and:
	# 1. Detects fish objects and classifies into normal or reflection using a yolov5 object
	# 2. 
	# 3. Automatically identifies bower location
	# 4. Analyze building, shape, and other pertinent info of the bower

	def __init__(self, fileManager, videoIndex):

		self.__version__ = '1.0.0'
		self.fileManager = fileManager
		self.videoObj = self.fileManager.returnVideoObject(videoIndex)
		self.videoIndex = videoIndex
		self.fileManager.downloadData(self.fileManager.localYolov5WeightsFile)

	def validateInputData(self):
		
		assert os.path.exists(self.videoObj.localVideoFile)

		assert os.path.exists(self.fileManager.localTroubleshootingDir)
		assert os.path.exists(self.fileManager.localAnalysisDir)
		assert os.path.exists(self.fileManager.localTempDir)
		assert os.path.exists(self.fileManager.localLogfileDir)
		assert os.path.exists(self.fileManager.localYolov5WeightsFile)

	def runObjectDetectionAnalysis(self, gpu = 0):


		print('Running Object detection on ' + self.videoObj.baseName + ' ' + str(datetime.datetime.now()), flush = True)
		self.annotations_dir = self.fileManager.localTempDir + self.videoObj.localVideoFile.split('/')[-1].replace('.mp4','')

		command = ['python3', 'detect.py']
		command.extend(['--weights', self.fileManager.localYolov5WeightsFile])
		command.extend(['--source', self.videoObj.localVideoFile])
		command.extend(['--device', str(gpu)])
		command.extend(['--project', self.annotations_dir])
		command.extend(['--save-txt', '--nosave', '--save-conf','--agnostic-nms'])

		command = "source " + os.getenv('HOME') + "/anaconda3/etc/profile.d/conda.sh; conda activate yolov5; " + ' '.join(command)

		os.chdir(os.getenv('HOME') + '/yolov5')
		print('bash -c \"' + command + '\"')
		output = subprocess.Popen('bash -c \"' + command + '\"', shell = True, stderr = open(os.getenv('HOME') + '/' + self.videoObj.baseName + '_detectionerrors.txt', 'w'), stdout=subprocess.DEVNULL)
		#os.chdir(os.getenv('HOME') + '/CichlidBowerTracking/cichlid_bower_tracking')
		return output

	def runSORT(self):
		self.annotations_dir = self.fileManager.localTempDir + self.videoObj.localVideoFile.split('/')[-1].replace('.mp4','')

		os.chdir(os.getenv('HOME') + '/CichlidBowerTracking/cichlid_bower_tracking')
		print('Running Sort detection on ' + self.videoObj.baseName + ' ' + str(datetime.datetime.now()), flush = True)

		command = ['python3', 'unit_scripts/sort_detections.py', self.annotations_dir + '/exp/labels/', self.videoObj.localFishDetectionsFile, self.videoObj.localFishTracksFile, self.videoObj.baseName]

		command = "source " + os.getenv('HOME') + "/anaconda3/etc/profile.d/conda.sh; conda activate CichlidSort; " + ' '.join(command)
		#subprocess.run('bash -c \"' + command + '\"', shell = True)

		output = subprocess.Popen('bash -c \"' + command + '\"', shell = True, stderr = open(os.getenv('HOME') + '/' + self.videoObj.baseName + '_trackingerrors.txt', 'w'), stdout=subprocess.DEVNULL)
		return output


