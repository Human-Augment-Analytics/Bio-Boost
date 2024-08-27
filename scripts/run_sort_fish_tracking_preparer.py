"""
SORT

The following class prepares and analyzes fish tracking data using YOLOv5 
and SORT algorithms.
"""

# Imports
import subprocess, os, pdb, datetime
from shapely.geometry import Point, Polygon

# Class to Prepare and Analyze Fish Tracking
class FishTrackingPreparer():
	"""
    This class performs the following tasks:
    1. Detects fish objects and classifies them into normal or reflection using a YOLOv5 model.
    2. Automatically identifies bower locations.
    3. Analyzes the building, shape, and other pertinent information of the bower.

    Attributes:
        fileManager: An instance of FileManager to manage file operations.
        videoObj: An object representing the video being analyzed.
        videoIndex: Index of the video in the dataset.
    """

	def __init__(self, fileManager, videoIndex):
		"""
		Initializes the FishTrackingPreparer class with the file manager and video index.

		Parameters:
			fileManager (FileManager): An instance of the FileManager class.
			videoIndex (int): Index of the video in the dataset.
		"""
		self.__version__ = '1.0.0'
		self.fileManager = fileManager
		self.videoObj = self.fileManager.returnVideoObject(videoIndex)
		self.videoIndex = videoIndex
		self.fileManager.downloadData(self.fileManager.localYolov5WeightsFile)

	def validateInputData(self):
		"""
        Validates the existence of necessary input data files and directories.
        
        Asserts:
            All required files and directories exist in the local file system.
        """
		assert os.path.exists(self.videoObj.localVideoFile)
		assert os.path.exists(self.fileManager.localTroubleshootingDir)
		assert os.path.exists(self.fileManager.localAnalysisDir)
		assert os.path.exists(self.fileManager.localTempDir)
		assert os.path.exists(self.fileManager.localLogfileDir)
		assert os.path.exists(self.fileManager.localYolov5WeightsFile)

	def runObjectDetectionAnalysis(self, gpu = 0):
		"""
		Runs the YOLOv5 object detection analysis on the specified video.

		Parameters:
			gpu (int, optional): GPU device index to use for the analysis. Default is 0.

		Returns:
			subprocess.Popen: The process object for the detection command.
		"""
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
		"""
        Runs the SORT tracking algorithm on the YOLOv5 detection results.

        Returns:
            subprocess.Popen: The process object for the SORT command.
        """
		self.annotations_dir = self.fileManager.localTempDir + self.videoObj.localVideoFile.split('/')[-1].replace('.mp4','')

		os.chdir(os.getenv('HOME') + '/CichlidBowerTracking/cichlid_bower_tracking')
		print('Running Sort detection on ' + self.videoObj.baseName + ' ' + str(datetime.datetime.now()), flush = True)

		command = ['python3', 'unit_scripts/sort_detections.py', self.annotations_dir + '/exp/labels/', self.videoObj.localFishDetectionsFile, self.videoObj.localFishTracksFile, self.videoObj.baseName]

		command = "source " + os.getenv('HOME') + "/anaconda3/etc/profile.d/conda.sh; conda activate CichlidSort; " + ' '.join(command)
		#subprocess.run('bash -c \"' + command + '\"', shell = True)

		output = subprocess.Popen('bash -c \"' + command + '\"', shell = True, stderr = open(os.getenv('HOME') + '/' + self.videoObj.baseName + '_trackingerrors.txt', 'w'), stdout=subprocess.DEVNULL)
		return output