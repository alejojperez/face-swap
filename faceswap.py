import os
import shutil
import cv2
import insightface
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt
import re
from multiprocessing import Pool, Value
from enum import IntEnum
from io import StringIO 
import sys
import time
import subprocess


class CaptureOutput(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


def getFaceModel(printLog=True):
    with CaptureOutput() as output:
        """
        insightface version 0.7 onwards is the onw that
        contains the required capabilities for face swapping
        """
        assert float('.'.join(insightface.__version__.split('.')[:2]))>=float('0.7')

        """
        Create the app that is going to contain the model
        and specify the image size
        """
        detector = FaceAnalysis(name='buffalo_l', root='./')
        detector.prepare(ctx_id=0)

        """
        Get the model using the onnx helper file
        """
        swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)
    
    if printLog:
        print(output)

    return (detector, swapper)


def getSourceFace(sourceImgPath, detector):
    sourceImg = cv2.imread(sourceImgPath)
    sourceFaces = detector.get(sourceImg)

    if len(sourceFaces) < 1:
        raise Exception("The source file provided did not contain any face.")

    return sourceFaces[0]


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)


def swapFaces(targetImgPath, sourceFace, detector, swapper, printLog=True):
    if printLog:
        print("Swaping faces for file: %s" % targetImgPath)

    targetImg = cv2.imread(targetImgPath)
    targetFaces = detector.get(targetImg)

    for face in targetFaces:
        targetImg = swapper.get(targetImg, face, sourceFace, paste_back=True)
        cv2.imwrite(targetImgPath, targetImg)


def swapFacesListInitializer(shared_value):
    global swapFacesListCounter
    swapFacesListCounter = shared_value


def swapFacesList(sourceImgPath, targetImgPathList, progressFilePath, printLog=True):
    detector, swapper = getFaceModel(False)
    sourceFace = getSourceFace(sourceImgPath, detector)

    for f in targetImgPathList:
        swapFaces(f, sourceFace, detector, swapper, printLog)

        with swapFacesListCounter.get_lock():
            swapFacesListCounter.value -= 1
            print("Remaining frames: %d" % swapFacesListCounter.value)
        
    with open(progressFilePath, '+a') as f:
        for line in targetImgPathList:
            f.write(f"{line}\n")


def swapJobHash(sourceImgPath, targetImgPath):
    return "job__" + os.path.basename(sourceImgPath) + "__" + os.path.basename(targetImgPath)


class LogLevel(IntEnum):
    OFF = 0
    MINIMUM = 1
    VERBOSE = 2


class FaceSwap:
    def __init__(self, loglevel=LogLevel.MINIMUM):
        if not loglevel in LogLevel:
            raise Exception("Invalid loglevel provided")

        self.loglevel = loglevel
        self.detector, self.swapper = getFaceModel(self.loglevel > LogLevel.MINIMUM)
        self.sourceImgPath = input("Enter the source image: ")
        self.sourceFace = getSourceFace(self.sourceImgPath, self.detector)


    def imageSwap(self, outputPath):
        targetImg = cv2.imread(input("Enter the target image: "))
        start_time = time.time()

        """
        Iterate through all the faces found and replace
        using the source face
        """
        targetFaces = self.detector.get(targetImg)
        for face in targetFaces:
            targetImg = self.swapper.get(targetImg, face, self.sourceFace, paste_back=True)
        
        """
        Show the results
        """
        plt.imshow(targetImg[:,:,::-1])
        plt.show()

        if self.loglevel > LogLevel.OFF:
            print("--- %s seconds ---" % (time.time() - start_time))

        cv2.imwrite(outputPath, targetImg)

        return True


    def videoSwap(self, outputPath, parallel=True, processes=None):
        videoPath = input("Enter the target video: ")
        targetVid = cv2.VideoCapture(videoPath)

        start_time = time.time()
        tempPath = swapJobHash(self.sourceImgPath, videoPath)
        progressFileName = "progress.txt"
        progressFilePath = tempPath + "/" + progressFileName
        audioFileFileName = "audio.mp3"
        audioFilePath = tempPath + "/" + audioFileFileName
        success = True
        count = 0
        continueWork = True

        """
        Create temporal directory if it does not exist
        """
        if not os.path.exists(tempPath):
            os.makedirs(tempPath)
            continueWork = False
        else:
            continueWork = input("There seems to exist a previous job. Would you like to continue? y/n: ") == "y"

        if not continueWork:
            """
            Delete all existing files
            """
            for f in os.listdir(tempPath):
                os.remove(os.path.join(tempPath, f))

            if self.loglevel > 0:
                print("Starting to read video...")

            while success:
                if self.loglevel > 0:
                    print("Frame read: %d" % count)
                success,image = targetVid.read()

                if success:
                    cv2.imwrite(tempPath + "/" + "frame%d.jpg" % count, image)
                    count += 1
                else:
                    if self.loglevel > 0:
                        print("Finished reading video.")

        imagesListProcessed = list([])
        mode = "+r" if os.path.exists(progressFilePath) else "+a"
        with open(progressFilePath, mode) as f:
            for line in f:
                imagesListProcessed.append(line.rstrip())

        imagesList = list(f for f in os.listdir(tempPath) if f != progressFileName and f != audioFileFileName and (tempPath + "/" + f) not in imagesListProcessed and not re.match(r'^\..+', f))
        imagesList = sorted_alphanumeric(imagesList)
        imagesList = list(map(lambda x: os.path.join(tempPath, x), imagesList))
        print("Remaining frames to be processed: %d" % len(imagesList))

        if parallel:
            sourceImgPath = self.sourceImgPath
            itemsPerChunk = 10
            imagesListChunks = [imagesList[i:i + itemsPerChunk] for i in range(0, len(imagesList), itemsPerChunk)]
            imagesListParallel = list(map(lambda l: (sourceImgPath, l, progressFilePath, self.loglevel > LogLevel.OFF), imagesListChunks))
            counter = Value('i', len(imagesList))

            with Pool(processes=processes, initializer=swapFacesListInitializer, initargs=(counter,)) as pool:
                pool.starmap(swapFacesList, imagesListParallel)
        else:
            for f in imagesList:
                swapFaces(f, self.sourceFace, self.detector, self.swapper, self.loglevel > LogLevel.OFF)
                with open(progressFilePath, '+a') as p:
                    p.write(f"{f}\n")

        imagesList = list(f for f in os.listdir(tempPath) if f != progressFileName and f != audioFileFileName and not re.match(r'^\..+', f))
        imagesList = sorted_alphanumeric(imagesList)
        imagesList = list(map(lambda x: os.path.join(tempPath, x), imagesList))
        images = []
        for f in imagesList:
            images.append(cv2.imread(f))
        
        if self.loglevel > LogLevel.OFF:
            print("Generating video...")

        if len(images) > 0:
            videoWriter = cv2.VideoWriter(outputPath, cv2.VideoWriter_fourcc(*'mp4v'), targetVid.get(cv2.CAP_PROP_FPS), (images[0].shape[1], images[0].shape[0]))
            for image in images:
                videoWriter.write(image)

            videoWriter.release()

        if self.loglevel > LogLevel.OFF:
            print("Video generated.")
            print("--- %s seconds ---" % (time.time() - start_time))
            print("Extracting audio...")

        command = ["ffmpeg", "-i", videoPath, "-f", "mp3", "-ab", "192000", "-vn", audioFilePath]

        if subprocess.run(command).returncode != 0:
            print ("ERROR executing ffmpeg command to extract audio")

        if self.loglevel > LogLevel.OFF:
            print("Audio extracted.")
            print("Combining audio and video into final file...")

        tmpVideoPath = tempPath + "/" + "tmp__video.mp4"
        command = ["ffmpeg", "-i", outputPath, "-i", audioFilePath, "-c:v", "copy", "-map", "0:v", "-map", "1:a", "-y", tmpVideoPath]

        if subprocess.run(command).returncode != 0:
            print ("ERROR executing ffmpeg command to combine audio and video")
        
        shutil.move(tmpVideoPath, outputPath)

        if self.loglevel > LogLevel.OFF:
            print("Audio and video combined.")

        # TODO: remove all the extra files

        return True