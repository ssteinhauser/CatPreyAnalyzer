# Introduction
If you own a cat that has the freedom to go outside, then you probably are familliar with the issue of your feline bringing home prey. This leads to a clean up effort that one wants to avoid!
This project aims to perform Cat Prey Detection with Deep Learning on any cat in any environement. For a brief and light intro in what it does, check out the [Raspberry Pi blog post](https://www.raspberrypi.org/blog/deep-learning-cat-prey-detector/) about it. The idea is that you can use the output of this system to trigger your catflap such that it locks out your cat, if it wants to enter with prey.

<img src="/readme_images/lenna_casc_Node1_001557_02_2020_05_24_09-49-35.jpg" width="400">

# Related work
This isn't the first approach at solving the mentioned problem! There have been other equally (if not better) valid approaches such as the [Catcierge](https://github.com/JoakimSoderberg/catcierge) which analyzes the silhouette of the cat a very recent approach of the [AI powered Catflap](https://www.theverge.com/tldr/2019/6/30/19102430/amazon-engineer-ai-powered-catflap-prey-ben-hamm).
The difference of this project however is that it aims to solve *general* cat-prey detection through a vision based approach. Meaning that this should work for any cat! 

# How to use the Code
The code is meant to run on a RPI4 with the [IR JoyIt Camera](https://joy-it.net/de/products/rb-camera-IR_PRO) attached. If you have knowledge regarding Keras, you can also run the models on your own, as the .h5 files can be found in the /models directory (check the input shapes, as they can vary). Nonetheless, I will explain the prerequesites to run this project on the RPI4 with the attached infrared camera:

- Download the whole project and transfer it to your RPI. Make sure to place the folder in your home directory such that its path matches: ```/home/pi/CatPreyAnalyzer```

- Install the tensorflow object detection API as explained in [EdjeElectronics Repositoy](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi), which provides other excellent RPI object detection information.

- Create a Telegram Bot via the [Telegram Bot API](https://core.telegram.org/bots). After doing so your bot will receive a **BOT_TOKEN**, write this down. Next you will have to get your **CHAT_ID** by calling ```https://api.telegram.org/bot<YourBOTToken>/getUpdates``` in your browser, as in [this](https://stackoverflow.com/questions/32423837/telegram-bot-how-to-get-a-group-chat-id). Now you can edit ```cascade.py NodeBot().__init__()``` at line 613 and insert your Telegram credentials: 
  ```
  def __init__(self):
        #Insert Chat ID and Bot Token according to Telegram API
        self.CHAT_ID = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
        self.BOT_TOKEN = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
  ```
  I am working on a environement variable script that will automate this process. In the meanwhile sorry for this.
  
  - If you want your RPI to directly boot into the Cat_Prey_Analyzer then I suggest you use a crontab. To do so, on the RPI type: ```crontab -e``` and enter 
  ```
  @reboot sleep 30 && sudo /home/pi/CatPreyAnalyzer/catCam_starter.sh
  ```
  - Don't forget to make the ```catCam_starter.sh``` executable by performing 
  ```
  chmod +x /home/pi/CatPreyAnalyzer/catCam_starter.sh
  ```
  - Reboot and enjoy!
  
  By following all these steps, you should now be greated by your Bot at startup:

  <img src="/readme_images/bot_good_morning.png" width="400">
  
  The system is now running and you can check out the bot commands via ```/help```. Be aware that you need patience at startup, as the models take up to 5 min to be   completely loaded, as they are very large.
  
# Modifications for running the code on a Rock 3a
If you don't use a Raspberry Pi, but an alternative such as the [Radxa Rock 3a](https://wiki.radxa.com/Rock3/3a), then a few modifications must be made.

### Prerequisites
  
- software: the code was tested on [Debian Bullseye](https://wiki.radxa.com/Rock3/Debian) for the Rock 3a. Download a suitable Debian image [here](https://github.com/radxa-build/rock-3a/releases/latest)).
  
- hardware: the Raspberry Pi camera (v1) can be made to work with the Rock 3a on Linux kernel version 4.19.193 (the default kernel shipped with the Debian image mentioned above) by enabling a [device tree overlay](https://wiki.radxa.com/Device-tree-overlays). On a default Debian install, the device tree overlay for the camera must be activated by editing ```/boot/config.txt```:
  ```
  sudo nano /boot/config.txt
  ```
  Add the following line (at the end)
  ```
  dtoverlay=rock-3ab-rpi-camera-v1p3-ov5647
  ```
  
  Save and exit ```nano``` (```CTRL-X```) and run
  
  ```
  sudo update_extlinux.sh
  ```
  
  Reboot.
  
  To test the camera from the command line, install the ```v4l2-utils``` and ```ffmpeg``` packages:
  ```
  sudo apt install v4l2-utils ffmpeg
  ```
  and run a command like this:
  ```
  v4l2-ctl --device /dev/video0 --stream-mmap=3 --stream-count=1 \
  --stream-skip=10 --stream-to=1920x1080.nv12 \
  --set-fmt-video=width=1920,height=1080,pixelformat=NV12 \
  && ffmpeg -y -f rawvideo -s 1920x1080 -pix_fmt nv12 -i \
  1920x1080.nv12  -pix_fmt rgb24 1920x1080.nv12.png
  ```
  This will record an image (```1920x1080.nv12.png```). Use your favorite image viewer to display this file. The file ```1920x1080.nv12``` should contain a really short video. The video can be viewed using commands like
  ```
  vlc --demux rawvideo --rawvid-fps 25 --rawvid-width 1920 --rawvid-height 1080 --rawvid-chroma I420 1920x1080.nv12
  ```
  or
  ```
  mplayer -demuxer rawvideo -rawvideo w=1920:h=1080:format=nv12 1920x1080.nv12
  ```
- Follow the instructions for installing opencv, python3 packages and tensorflow as laid out above (see [EdjeElectronics Repositoy](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi)), with one major modification: ```python-opencv``` should be installed via package manager, not via pip3:
  ```
  sudo apt install python3-opencv 
  ```
  Additionally, some required packages and modules mentioned in EdjeElectronics Repository are not available in Debian Bullseye (```libjasper-dev```) or have a different name or version (```libpng12-dev``` -> ```libpng-dev```), but the installation works when using the following commands
  ```
  sudo apt install cython3
  sudo apt install python-tk
  sudo apt install libjpeg-dev libtiff5-dev libpng-dev
  sudo apt install libavcodec-dev libavformat-dev
  sudo apt install libswscale-dev libv4l-dev
  sudo apt install libxvidcore-dev libx264-dev
  sudo apt install libatlas-base-dev
  sudo apt install python3-python-telegram-bot python3-tz
  sudo apt install protobuf-compiler python3-pil python3-lxml
  
  sudo pip3 install tensorflow tensorflow-io
  sudo pip3 install pillow lxml jupyter matplotlib
  ```
  (omitting ```libjasper-dev``` and using ```libpng-dev``` instead of ```libpng12-dev```, and using the distribution-specific version of ```cython```, matching the python version.).
  
###  Modifications to the code
  
  - the script for starting the code (```catCam_starter.sh```) was slightly modified to be more device-independent (different ```PYTHONPATH``` and  ```HOME``` locations).
  - Telegram bot info should be entered in ```catCam_starter.sh``` in two environment variables:
    ```
    CHAT_ID="XXXXXXXXX"
    BOT_TOKEN="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    ```
    the Python3 code reads the info from these variables.
  - the Python3 ```picamera``` library is not available on Debian for the Rock 3a, so  ```camera_class.py``` was rewritten to access the camera via the ```V4L2``` backend of ```opencv```. The code for capturing images (frames) in Python3 using ```opencv/V4L2``` is essentially like this:
    ```
    import cv2 as cv
    cap = cv.VideoCapture(0)
    cap.read(0) # helps to initialize the camera
    cap.read(0) # helps to initialize the camera
    ret, frame = cap.read()
    print("Frame actual is {}x{}".format(str(frame.shape[1]), \
    str(frame.shape[0])))
    cv.imwrite("test.png", frame)
    cap.release()
    ```
    (check ```test.png``` for a captured frame)
  
# Firebase messaging
The original project uses telegram as a means of interaction with the software. There are other means of sending messages, which can be much more convenient. For instance, if you plan to use your smartphone, which alerts you of any cat approaching your camera, you could create an app for that. This can be much more custmizable than receiving telegram messages. One option for sending messages to smartphone apps is [Google Firebase Messaging](https://firebase.google.com/docs/cloud-messaging).
There is a python implementation that allows for integrating Firebase Messaging:
   ```
    import firebase_admin
    from firebase_admin import credentials, messaging, storage
   ```
Setting up Firebase is described [here](https://medium.com/@abdelhedihlel/upload-files-to-firebase-storage-using-python-782213060064). In essence, you need a credentials file (JSON) and the name of the storage bucket. The storage bucket must be created during the Firebase setup. The credentials file can be downloaded from Google Firebase. It should be named ```firebasekey.json``` and placed in the same directory as the ```cascade.py``` python code. The storage bucket name should be defined in ```catCam_starter.sh``` in an environment variable:
    ```
    FIREBASE_BUCKET="xxxxx.xxxx.xxx"
    ```
Of course, you would need to write a smartphone app which receives messages via Firebase.
If you don't want to use firebase messaging, comment all lines which contain the method ```sendPushNotification``` and the call to the initialization (```init_firebase_messaging```)
# A word of caution
This project uses deep learning (DL)! Contrary to popular belief DL is **not** black magic (altough close to 😎)! The network perceives image data differently than us humans. It "sees" more abstractly than us. This means a cat in the image lives as an abstract blob deep within the layers of the network. Thus there are going to be instances where the system will produce absurdly wrong statements such as:

 <img src="/readme_images/bot_fail.png" width="400">
 
  This can happen and the reason why is maths... so you have to be aware of it. If this fascinates you as much as it does me and you want a deeper understanding, check out [the deep learning book](http://www.deeplearningbook.org/)!
 
Further this project is based on transfer learning and has had a **very** small training set of only 150 prey images, sampled from the internet and a custom data-gathering network (more info in ```/readme_images/Semesterthesis_Smart_Catflap.pdf```). It works amazingly well *for this small amount of Data*, yet you will realize that there are still a lot of false positives. I am working on a way that we could all collaborate and upload the prey images of our cats, such that we can further train the models and result in a **much** stronger classifier. 

And check the issues section for known issues regarding this project. If you encounter something new, don't hesitate to flag it! For the interested reader, a TLDR of my thesis is continued below.

# Architecture
In this section we will discuss the the most important architectural points of the project.

### Cascade of Neural Nets ###
This project utilises a cascade of Convolutional Neural Networks (CNN) to process images and infer about the Prey/No_Prey state of a cat image. The reason why it uses a cascade is simple: CNN's need data to learn their task, the amount of data is related to the complexity of the problem. For general cat prey detection, a NN would need to first learn what a cat is in general, and find out how their snouts differ with and without prey. This turns out to be quite complex for a machine to learn and we simply don't have enough data of cats with prey (only 150 images to be exact). This is why we use a cascade to break up the complex problem into smaller substages:

- First detect if there is a cat or not. There exists a lot of data on this problem and a lot of complete solutions such for example any COCO trained Object detector such as for example [Tensorflows COCO trained MobileNetV2](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md). We call it the CatFinder stage which utilises the mentioned Tensorflow object detection API and runs the Tensorflow pretrained MobileNetV2 and soley aims to detect a cat in the image.

- Second we detect the snout of the cat within the image section of the first stage. This is done by a combination of different Computer Vision (CV) techniques such as a HAAR-Cascade and a self trained CNN (CNN-BB + FF).

- Lastly we classify the snout-cropped image of the cat with a self trained CNN based on the VGG16 architecture. It was only trained with ~150 Prey-Snout-Images gathered from the internet and personal images. This is the data-critical section; we can find more than enough images of cats but only very few images of cats with prey in their snout. Obviously the tasks complexity of identifying prey in a cropped image of the snout is simpler than classifying so on a full image, hence the extra steps of the cascade.

Here is a brief overview of the cascade:

<img src="/readme_images/cascade.png" width="400">

As depicted in the image, there are four resulting paths that can be taken which yield different runtimes. On an off the shelf Raspberry Pi 4 the runtimes areas follows:

- P1: 507 ms
- P2: 3743 ms
- P3: 2035 ms
- P4: 5481 ms


### Processing Queue ###
Now the runtime numbers are quite high, which is why we use a dynamically adapting queue to adjust the framerate of the system. This part is built specifically for the RPI and its camera system. It is a multithreading process where the camera runs on an own thread and the cascade on a seperate thread. The camera fills a concurrent queue while the cascade pops the queue at a dynamic rate. Sounds fancy and complicated, but it isn't:

<img src="/readme_images/queue.png" width="400">

### Cumuli Points ###
As we are evaluating over multiple images that shall make up an event, we must have the policy, We chose: *A cat must prove that it has no prey*. The cat has to accumulate trust-points. The more points the more we trust our classification, as our threshold value is 0.5 (1: Prey, 0: No_Prey) points above 0.5 count negatively and points below 0.5 count positively towards the trust-points aka cummuli-points. 

<img src="/readme_images/cummuli_approach.png" width="400">

As is revealed in the Results section, we chose a cumuli-treshold of 2.93. Meaning that we classify the cat to have proven that it has no prey as soon as it reaches 2.93 cumuli-points.


# Results

As a cat returns with prey roughly only 3% of the time, we are dealing with an imbalanced problem. To evaluate such problems we can use a Precision-Recall curve, where the "no_skill" threshold is depicted by the dashed line, for further reading on how this works check out this [Scikit Article](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#:~:text=The%20precision%2Drecall%20curve%20shows,a%20low%20false%20negative%20rate.). Next to it the ROC-curve is depicted, as it is a very common method to evaluate NN models, yet more suited for a balanced evaluation. 

As you can see in the ROC plot (using ROC because explaination is more intuitive), we chose the threshold point of 2.93 cummuli points which yields a True Positive Ratio (TPR) of ~93% while showing a False Positive Ratio (FPR) of ~28%. This means that 93% of all prey cases will be cought correctly while the cat is falsely accused of entering with prey 28% of times that it actually does not have prey.

<img src="/readme_images/combined_curve.png" width="700">

Here is the simple confusion matrix (shown for data transparency reasons), with the decison threshold set at 2.93 cummuli points. The confusion matrix has been evaluated on 57 events which results in ~855 images.

<img src="/readme_images/Cummuli Confusion Matrix @ Threshold_ 2.96.png" width="400">

And here we have a less technical "proof" that the cascade actually does what it is supposed to do. On the top are independent images of my cat without prey, while on the bottom the same images have a photoshopped mouse in its snout. You can see that the photoshopped images significantly alter the prediction value of the network.

<img src="/readme_images/merged_prey.png" width="700">


