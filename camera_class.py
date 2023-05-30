# for logging
import logging.handlers
import os,sys

path_of_script = os.path.dirname(os.path.realpath(sys.argv[0]))
name_of_script = os.path.basename(sys.argv[0])

LOG_FILE_NAME=path_of_script+'/log/camera.log'
LOGGING_LEVEL = logging.INFO

formatter = logging.Formatter('%(asctime)s %(message)s',
                       "%Y-%m-%d %H:%M:%S")
handler = logging.handlers.RotatingFileHandler(LOG_FILE_NAME, mode='a',
                      maxBytes=10000000, backupCount=7)
handler.setFormatter(formatter)
log = logging.getLogger( __name__ )
log.addHandler(handler)
log.setLevel(LOGGING_LEVEL)


try:
   import RPi.GPIO as GPIO
   from picamera.array import PiRGBArray
   from picamera import PiCamera
   from gpiozero import CPUTemperature
   raspicam=True
except ModuleNotFoundError:
   print("raspicam modules not found, assuming v4l2 camera")
   raspicam=False


from collections import deque
import pytz
from datetime import datetime
from threading import Thread
import time
import sys
import cv2 as cv
import numpy as np
import io, gc

class Camera:
    def __init__(self,):
        if raspicam:
            IRPin = 36
            # GPIO Stuff
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(IRPin, GPIO.OUT)
            GPIO.output(IRPin, GPIO.LOW)
        time.sleep(2)

    def fill_queue(self, deque):
        while(1):
            gc.collect()
            if raspicam:
                camera = PiCamera()
                camera.framerate = 3
                camera.vflip = False
                camera.hflip = False
                camera.resolution = (2592, 1944)
                camera.exposure_mode = 'sports'
                stream = io.BytesIO()
                for i, frame in enumerate(camera.capture_continuous(stream, format="jpeg", use_video_port=True)):
                    stream.seek(0)
                    data = np.frombuffer(stream.getvalue(), dtype=np.uint8)
                    image = cv2.imdecode(data, 1)
                    deque.append(
                        (datetime.now(pytz.timezone('Europe/Zurich')).strftime("%Y_%m_%d_%H-%M-%S.%f"), image))
                    #deque.pop()
                    log.info("Quelength: " + str(len(deque)) + "\tStreamsize: " + str(sys.getsizeof(stream)))
                    if i == 60:
                        log.info("Loop ended, starting over.")
                        camera.close()
                        del camera
                        break

            else:
                cap=cv.VideoCapture(0)
                #Set the resolution
                log.info("Setting up camera")
                cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
                cap.read(0)
                #time.sleep(0.001)
                cap.read(0)
                log.info("capturing 60 frames")
                for i in range(59):
                    ret, frame = cap.read()
                    time.sleep(0.3)
                    #print("frame read")
                    if ret:
                        log.info("appending frame:")
                        deque.append(
                        (datetime.now(pytz.timezone('Europe/Zurich')).strftime("%Y_%m_%d_%H-%M-%S.%f"), frame))
                    #deque.pop()
                        log.info("Added "+str(i)+". Quelength: " + str(len(deque)))
                try:
                    log.info("writing test image")
                    cv.imwrite(path_of_script+"/test.jpg", frame)
                except cv.error as e:
                    log.info("writing test frame from camera thread failed.")
                cap.release()
                del cap
                log.info("reset capture device, starting over")
        log.info("Should not reach this point in camera thread??")
