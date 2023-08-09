import os
import sys
import io
import gc
import time
import queue
from collections import deque
from datetime import datetime
from threading import Thread
import logging.handlers
import cv2 as cv
import numpy as np
import pytz

path_of_script = os.path.dirname(os.path.realpath(sys.argv[0]))
name_of_script = os.path.basename(sys.argv[0])

LOG_FILE_NAME = path_of_script + '/log/camera.log'
LOGGING_LEVEL = logging.INFO

formatter = logging.Formatter('%(asctime)s %(message)s', "%Y-%m-%d %H:%M:%S")
handler = logging.handlers.RotatingFileHandler(LOG_FILE_NAME, mode='a', maxBytes=10000000, backupCount=7)
handler.setFormatter(formatter)
log = logging.getLogger(__name__)
log.addHandler(handler)
log.setLevel(LOGGING_LEVEL)

class VideoCapture:
  def __init__(self, name):
    self.cap = cv.VideoCapture(name)
    self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()

raspicam = False
raspicam2 = False
try:
   import RPi.GPIO as GPIO
   from picamera.array import PiRGBArray
   from picamera import PiCamera
   from gpiozero import CPUTemperature

   raspicam = True
except ModuleNotFoundError:
   try:
      from picamera2 import Picamera2
      raspicam2 = True
   except ModuleNotFoundError:
      print("raspicam modules not found, assuming v4l2 camera")

class Camera:
    def __init__(self):
        if raspicam:
            IRPin = 36
            # GPIO Stuff
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(IRPin, GPIO.OUT)
            GPIO.output(IRPin, GPIO.LOW)
        time.sleep(2)

    def fill_queue(self, deque):
        if raspicam2:
            picam2 = Picamera2()
            config = picam2.create_still_configuration()
            picam2.configure(config)
            picam2.set_controls({"AwbEnable": 1,"ColourGains": (0.8, 1)})
            picam2.start()
            i = 0
        elif not raspicam:
            streamURL = os.getenv('STREAM_URL')
            cap = VideoCapture(streamURL)
            # Set the resolution
            log.info("Setting up camera")
            i = 0

        while True:
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
            elif raspicam2:
                frame = picam2.capture_array()
                time.sleep(0.3)
                log.info("appending frame:")
                datestr = datetime.now(pytz.timezone('Europe/Zurich')).strftime("%Y_%m_%d_%H-%M-%S.%f")
                deque.append((datestr, frame))
                log.info("Added " + str(i) + ". Quelength: " + str(len(deque)))
            else:
                ret, frame = cap.read()
                # time.sleep(1/25) # 25fps
                time.sleep(1/3) # 3fps
                if ret:
                    log.info("appending frame:")
                    deque.append((datetime.now(pytz.timezone('Europe/Zurich')).strftime("%Y_%m_%d_%H-%M-%S.%f"), frame))
                    log.info("Added " + str(i) + ". Quelength: " + str(len(deque)))

        log.info("Should not reach this point in camera thread??")
