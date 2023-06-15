import numpy as np
from pathlib import Path
import os, cv2, time, csv, sys, gc
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta, timezone
from collections import deque
from threading import Thread
from multiprocessing import Process
import telegram
from telegram.ext import Updater, CommandHandler, Filters, MessageHandler
import xml.etree.ElementTree as ET
# for firebase messaging
from email.quoprimime import body_decode
import firebase_admin
from firebase_admin import credentials, messaging, storage, exceptions, db

# for logging
import logging.handlers

path_of_script = os.path.dirname(os.path.realpath(sys.argv[0]))
name_of_script = os.path.basename(sys.argv[0])
home = str(Path.home())
sys.path.append(path_of_script)
sys.path.append(home)
from CatPreyAnalyzer.model_stages import PC_Stage, FF_Stage, Eye_Stage, Haar_Stage, CC_MobileNet_Stage
#log.info("model stages imported")
from CatPreyAnalyzer.camera_class import Camera
#log.info("camera class imported")
cat_cam_py = str(Path(os.getcwd()).parents[0])

LOG_FILE_NAME=path_of_script+'/log/catCam.log'
LOGGING_LEVEL = logging.INFO

formatter = logging.Formatter('%(asctime)s %(message)s',
                              "%Y-%m-%d %H:%M:%S")
handler = logging.handlers.RotatingFileHandler(LOG_FILE_NAME, mode='a',
                                               maxBytes=10000000, backupCount=7)
handler.setFormatter(formatter)
log = logging.getLogger(name_of_script)
log.addHandler(handler)
log.setLevel(LOGGING_LEVEL)


class Spec_Event_Handler():
    def __init__(self):
        self.img_dir = os.path.join(cat_cam_py, 'CatPreyAnalyzer/debug/input')
        self.out_dir = os.path.join(cat_cam_py, 'CatPreyAnalyzer/debug/output')

        self.img_list = [x for x in sorted(os.listdir(self.img_dir)) if'.jpg' in x]
        self.base_cascade = Cascade()

    def log_to_csv(self, img_event_obj):
        csv_name = img_event_obj.img_name.split('_')[0] + '_' + img_event_obj.img_name.split('_')[1] + '.csv'
        file_exists = os.path.isfile(os.path.join(self.out_dir, csv_name))
        with open(os.path.join(self.out_dir, csv_name), mode='a') as csv_file:
            headers = ['Img_Name', 'CC_Cat_Bool', 'CC_Time', 'CR_Class', 'CR_Val', 'CR_Time', 'BBS_Time', 'HAAR_Time', 'FF_BBS_Bool', 'FF_BBS_Val', 'FF_BBS_Time', 'Face_Bool', 'PC_Class', 'PC_Val', 'PC_Time', 'Total_Time']
            writer = csv.DictWriter(csv_file, delimiter=',', lineterminator='\n', fieldnames=headers)
            if not file_exists:
                writer.writeheader()

            writer.writerow({'Img_Name':img_event_obj.img_name, 'CC_Cat_Bool':img_event_obj.cc_cat_bool,
                             'CC_Time':img_event_obj.cc_inference_time, 'CR_Class':img_event_obj.cr_class,
                             'CR_Val':img_event_obj.cr_val, 'CR_Time':img_event_obj.cr_inference_time,
                             'BBS_Time':img_event_obj.bbs_inference_time,
                             'HAAR_Time':img_event_obj.haar_inference_time, 'FF_BBS_Bool':img_event_obj.ff_bbs_bool,
                             'FF_BBS_Val':img_event_obj.ff_bbs_val, 'FF_BBS_Time':img_event_obj.ff_bbs_inference_time,
                             'Face_Bool':img_event_obj.face_bool,
                             'PC_Class':img_event_obj.pc_prey_class, 'PC_Val':img_event_obj.pc_prey_val,
                             'PC_Time':img_event_obj.pc_inference_time, 'Total_Time':img_event_obj.total_inference_time})

    def debug(self):
        event_object_list = []
        for event_img in sorted(self.img_list):
            event_object_list.append(Event_Element(img_name=event_img, cc_target_img=cv2.imread(os.path.join(self.img_dir, event_img))))

        for event_obj in event_object_list:
            start_time = time.time()
            single_cascade = self.base_cascade.do_single_cascade(event_img_object=event_obj)
            single_cascade.total_inference_time = sum(filter(None, [
                single_cascade.cc_inference_time,
                single_cascade.cr_inference_time,
                single_cascade.bbs_inference_time,
                single_cascade.haar_inference_time,
                single_cascade.ff_bbs_inference_time,
                single_cascade.ff_haar_inference_time,
                single_cascade.pc_inference_time]))
            log.info("Total Inference Time:"+ str(round(single_cascade.total_inference_time,3))+' s')
            log.info('Total Runtime:'+ str(round(time.time() - start_time,3))+ ' s')

            # Write img to output dir and log csv of each event
            cv2.imwrite(os.path.join(self.out_dir, single_cascade.img_name), single_cascade.output_img)
            #self.log_to_csv(img_event_obj=single_cascade)

class Sequential_Cascade_Feeder():
    def __init__(self):
        self.log_dir = os.path.join(os.getcwd(), 'log')
        log.info('Log Dir:'+ self.log_dir)
        self.event_nr = 0
        self.base_cascade = Cascade()
        self.DEFAULT_FPS_OFFSET = 1
        self.QUEQUE_MAX_THRESHOLD = 30
        self.fps_offset = self.DEFAULT_FPS_OFFSET
        self.MAX_PROCESSES = 5
        self.EVENT_FLAG = False
        self.event_objects = []
        self.patience_counter = 0
        self.PATIENCE_FLAG = False
        self.FACE_FOUND_FLAG = False
        self.event_reset_threshold = 6
        self.event_reset_counter = 0
        self.cumulus_points = 0
        self.cumulus_prey_threshold = -10
        self.cumulus_no_prey_threshold = 2.9603
        self.prey_val_hard_threshold = 0.6
        self.face_counter = 0
        self.PREY_FLAG = None
        self.NO_PREY_FLAG = None
        self.queues_cumuli_in_event = []
        self.bot = NodeBot()
        self.bot.sendPushNotification('catCam start','catCam AI software was restarted','','','1')

        self.processing_pool = []
        #log.info("deque")
        self.main_deque = deque()

    def reset_cumuli_et_al(self):
        self.EVENT_FLAG = False
        self.patience_counter = 0
        self.PATIENCE_FLAG = False
        self.FACE_FOUND_FLAG = False
        self.cumulus_points = 0
        self.fps_offset = self.DEFAULT_FPS_OFFSET
        self.event_reset_counter = 0
        self.face_counter = 0
        self.PREY_FLAG = None
        self.NO_PREY_FLAG = None
        self.cumulus_points = 0

        #Close the node_letin flag
        self.bot.node_let_in_flag = False

        self.event_objects.clear()
        self.queues_cumuli_in_event.clear()
        self.main_deque.clear()

        #terminate processes when pool too large
        if len(self.processing_pool) >= self.MAX_PROCESSES:
            log.info('terminating oldest processes Len:'+ str(len(self.processing_pool)))
            for p in self.processing_pool[0:int(len(self.processing_pool)/2)]:
                p.terminate()
            log.info('Now processes Len:'+ str(len(self.processing_pool)))

    def log_event_to_csv(self, event_obj, queues_cumuli_in_event, event_nr):
        csv_name = 'event_log.csv'
        file_exists = os.path.isfile(os.path.join(self.log_dir, csv_name))
        with open(os.path.join(self.log_dir, csv_name), mode='a') as csv_file:
            headers = ['Event', 'Img_Name', 'Done_Time', 'Queue', 'Cumuli', 'CC_Cat_Bool', 'CC_Time', 'CR_Class', 'CR_Val', 'CR_Time', 'BBS_Time', 'HAAR_Time', 'FF_BBS_Bool', 'FF_BBS_Val', 'FF_BBS_Time', 'Face_Bool', 'PC_Class', 'PC_Val', 'PC_Time', 'Total_Time']
            writer = csv.DictWriter(csv_file, delimiter=',', lineterminator='\n', fieldnames=headers)
            if not file_exists:
                writer.writeheader()

            for i,img_obj in enumerate(event_obj):
                writer.writerow({'Event':event_nr, 'Img_Name':img_obj.img_name, 'Done_Time':queues_cumuli_in_event[i][2],
                                 'Queue':queues_cumuli_in_event[i][0],
                                 'Cumuli':queues_cumuli_in_event[i][1],'CC_Cat_Bool':img_obj.cc_cat_bool,
                                 'CC_Time':img_obj.cc_inference_time, 'CR_Class':img_obj.cr_class,
                                 'CR_Val':img_obj.cr_val, 'CR_Time':img_obj.cr_inference_time,
                                 'BBS_Time':img_obj.bbs_inference_time,
                                 'HAAR_Time':img_obj.haar_inference_time, 'FF_BBS_Bool':img_obj.ff_bbs_bool,
                                 'FF_BBS_Val':img_obj.ff_bbs_val, 'FF_BBS_Time':img_obj.ff_bbs_inference_time,
                                 'Face_Bool':img_obj.face_bool,
                                 'PC_Class':img_obj.pc_prey_class, 'PC_Val':img_obj.pc_prey_val,
                                 'PC_Time':img_obj.pc_inference_time, 'Total_Time':img_obj.total_inference_time})

    def send_prey_message(self, event_objects, cumuli):
        prey_vals = [x.pc_prey_val for x in event_objects]
        max_prey_index = prey_vals.index(max(filter(lambda x: x is not None, prey_vals)))

        event_str = ''
        face_events = [x for x in event_objects if x.face_bool]
        for f_event in face_events:
            log.info('****************')
            log.info('Img_Name:'+ f_event.img_name)
            log.info('PC_Val:'+ str('%.2f' % f_event.pc_prey_val))
            log.info('****************')
            event_str += '\n' + f_event.img_name + ' => PC_Val: ' + str('%.2f' % f_event.pc_prey_val)

        sender_img = event_objects[max_prey_index].output_img
        caption = 'Cumuli: ' + str(cumuli) + ' => PREY DETECTED!' + ' 🐁🐁🐁' + event_str

        self.bot.send_img(img=sender_img, caption=caption)
        self.bot.sendDetectImage("PREY detected", "PREY detected", sender_img)
        return

    def send_no_prey_message(self, event_objects, cumuli):
        prey_vals = [x.pc_prey_val for x in event_objects]
        min_prey_index = prey_vals.index(min(filter(lambda x: x is not None, prey_vals)))

        event_str = ''
        face_events = [x for x in event_objects if x.face_bool]
        for f_event in face_events:
            log.info('****************')
            log.info('Img_Name:'+ f_event.img_name)
            log.info('PC_Val:'+ str('%.2f' % f_event.pc_prey_val))
            log.info('****************')
            event_str += '\n' + f_event.img_name + ' => PC_Val: ' + str('%.2f' % f_event.pc_prey_val)

        sender_img = event_objects[min_prey_index].output_img
        caption = 'Cumuli: ' + str(cumuli) + ' => Cat has no prey...' + ' 🐱' + event_str

        self.bot.send_img(img=sender_img, caption=caption)
        self.bot.sendDetectImage("Cat without prey detected", "Cat without prey detected", sender_img)
        return

    def send_dk_message(self, event_objects, cumuli):
        event_str = ''
        face_events = [x for x in event_objects if x.face_bool]
        for f_event in face_events:
            log.info('****************')
            log.info('Img_Name:'+ f_event.img_name)
            log.info('PC_Val:'+ str('%.2f' % f_event.pc_prey_val))
            log.info('****************')
            event_str += '\n' + f_event.img_name + ' => PC_Val: ' + str('%.2f' % f_event.pc_prey_val)
        try:
            sender_img = face_events[0].output_img
            caption = 'Cumuli: ' + str(cumuli) + " => Can't say for sure..." + ' 🤷‍♀️' + event_str + '\nMaybe use /letin?'
            self.bot.send_img(img=sender_img, caption=caption)
            self.bot.sendDetectImage("Prey detection inconclusive", "not sure", sender_img)
        except Exception as e:
            log.info('Failed to extract sender_img in send_dk_message:')
            log.info(e)
        return

    def get_event_nr(self):
        tree = ET.parse(os.path.join(self.log_dir, 'info.xml'))
        data = tree.getroot()
        imgNr = int(data.find('node').get('imgNr'))
        data.find('node').set('imgNr', str(int(imgNr) + 1))
        tree.write(os.path.join(self.log_dir, 'info.xml'))

        return imgNr

    def queque_worker(self):
        log.info('Working the Queue with len:'+ str(len(self.main_deque)))
        start_time = time.time()
        #Feed the latest image in the Queue through the cascade
        cascade_obj = self.feed(target_img=self.main_deque[self.fps_offset][1], img_name=self.main_deque[self.fps_offset][0])[1]
        #log.info('Runtime:'+ str(time.time() - start_time))
        done_timestamp = datetime.now(tz=ZoneInfo('Europe/Zurich')).strftime("%Y_%m_%d_%H-%M-%S.%f")
        prettytimestamp= datetime.now(tz=ZoneInfo('Europe/Zurich')).strftime("%d.%m.%Y %H:%M:%S")
        log.info('Runtime:'+ str(round(time.time() - start_time,3))+ 's') # , Timestamp:'+ str(done_timestamp))

        overhead = datetime.strptime(done_timestamp, "%Y_%m_%d_%H-%M-%S.%f") - datetime.strptime(self.main_deque[self.fps_offset][0], "%Y_%m_%d_%H-%M-%S.%f")
        log.info('Overhead:'+str(round(overhead.total_seconds(),3))+' s')

        #Add this so that the bot has some info
        self.bot.node_queue_info = len(self.main_deque)
        live_img = self.main_deque[self.fps_offset][1]
        color = (97, 70, 223) # https://colorcodes.io/red/cerise-color-codes/
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 2
        lineType = 3
        #self.input_text(img=live_img, text=done_timestamp, text_pos=(15, 100), color=color)
        cv2.putText(live_img, prettytimestamp,
                    (15, 60),
                    font,
                    fontScale,
                    color,
                    lineType)

        self.bot.node_live_img=live_img
        self.bot.timestamp=prettytimestamp

        self.bot.node_over_head_info = overhead.total_seconds()

        # Always delete the left part
        for i in range(self.fps_offset + 1):
            self.main_deque.popleft()

        if cascade_obj.cc_cat_bool == True:
            #We are inside an event => add event_obj to list
            self.EVENT_FLAG = True
            self.event_nr = self.get_event_nr()
            self.event_objects.append(cascade_obj)
            #self.bot.send_text("Cat found")

            #Last cat pic for bot
            cv2.putText(cascade_obj.output_img, prettytimestamp,
                     (15, 60),
                     font,
                     fontScale,
                     color,
                     lineType)
            self.bot.node_last_casc_img=cascade_obj.output_img
            #self.bot.uploadLastCascImage()
            upload_thread = Thread(target=self.bot.uploadLastCascImage, daemon=True)
            upload_thread.start()

            self.fps_offset = 0
            #If face found add the cumulus points
            if cascade_obj.face_bool:
                self.face_counter += 1
                self.cumulus_points += (50 - int(round(100 * cascade_obj.pc_prey_val)))
                self.FACE_FOUND_FLAG = True
                #self.bot.send_text("Cat face found")
                self.bot.send_img(self.bot.node_last_casc_img,"Cat face found")
                self.bot.sendCascImage()

            log.info('CUMULUS:'+ str(self.cumulus_points))
            self.queues_cumuli_in_event.append((len(self.main_deque),self.cumulus_points, done_timestamp))

            #Check the cumuli points and set flags if necessary
            if self.face_counter > 0 and self.PATIENCE_FLAG:
                if self.cumulus_points / self.face_counter > self.cumulus_no_prey_threshold:
                    self.NO_PREY_FLAG = True
                    log.info('NO PREY DETECTED... YOU CLEAN...')
                    p = Process(target=self.send_no_prey_message, args=(self.event_objects, self.cumulus_points / self.face_counter,), daemon=True)
                    p.start()
                    self.processing_pool.append(p)
                    #self.log_event_to_csv(event_obj=self.event_objects, queues_cumuli_in_event=self.queues_cumuli_in_event, event_nr=self.event_nr)
                    self.reset_cumuli_et_al()
                elif self.cumulus_points / self.face_counter < self.cumulus_prey_threshold:
                    self.PREY_FLAG = True
                    log.info('IT IS A PREY!!!!!')
                    p = Process(target=self.send_prey_message, args=(self.event_objects, self.cumulus_points / self.face_counter,), daemon=True)
                    p.start()
                    self.processing_pool.append(p)
                    #self.log_event_to_csv(event_obj=self.event_objects, queues_cumuli_in_event=self.queues_cumuli_in_event, event_nr=self.event_nr)
                    self.reset_cumuli_et_al()
                else:
                    self.NO_PREY_FLAG = False
                    self.PREY_FLAG = False

            #Cat was found => still belongs to event => acts as dk state
            self.event_reset_counter = 0

        #No cat detected => reset event_counters if necessary
        else:
            log.info('NO CAT FOUND!')
            self.event_reset_counter += 1
            if self.event_reset_counter >= self.event_reset_threshold:
                # If was True => event now over => clear queque
                if self.EVENT_FLAG == True:
                    log.info('CLEARED QUEUE BECAUSE EVENT OVER WITHOUT CONCLUSION...')
                    #TODO QUICK FIX
                    if self.face_counter == 0:
                        self.face_counter = 1
                    p = Process(target=self.send_dk_message, args=(self.event_objects, self.cumulus_points / self.face_counter,), daemon=True)
                    p.start()
                    self.processing_pool.append(p)
                    #self.log_event_to_csv(event_obj=self.event_objects, queues_cumuli_in_event=self.queues_cumuli_in_event, event_nr=self.event_nr)
                self.reset_cumuli_et_al()

        if self.EVENT_FLAG and self.FACE_FOUND_FLAG:
            self.patience_counter += 1
        if self.patience_counter > 2:
            self.PATIENCE_FLAG = True
        if self.face_counter > 1:
            self.PATIENCE_FLAG = True

    def single_debug(self):
        start_time = time.time()
        target_img_name = 'dummy_img.jpg'
        target_img = cv2.imread(os.path.join(cat_cam_py, 'CatPreyAnalyzer/readme_images/lenna_casc_Node1_001557_02_2020_05_24_09-49-35.jpg'))
        #cv2.imwrite(os.path.join(cat_cam_py,'test.jpg'),target_img)
        cascade_obj = self.feed(target_img=target_img, img_name=target_img_name)[1]
        log.info('Runtime:'+ str(round(time.time() - start_time,3))+' s')
        return cascade_obj

    def queque_handler(self):
        # Do this to force run all networks s.t. the network inference time stabilizes
        self.single_debug()
        log.info("initializing camera")
        camera = Camera()
        log.info("starting camera thread")
        camera_thread = Thread(target=camera.fill_queue, args=(self.main_deque,), daemon=True)
        camera_thread.start()
        log.info("camera thread started")
        last_run_time=time.time()

        while(True):
            current_time=time.time()
            if len(self.main_deque) > self.QUEQUE_MAX_THRESHOLD:
                self.main_deque.clear()
                self.reset_cumuli_et_al()
                # Clean up garbage
                gc.collect()
                log.info('DELETING QUEUE BECAUSE IT IS OVERLOADED!')
                self.bot.send_text(message='Too many images to process ... had to kill Queue!')

            elif len(self.main_deque) > self.DEFAULT_FPS_OFFSET:
                self.queque_worker()

            else:
                log.info('Nothing to work with => Queue_length:'+str(len(self.main_deque)))
                time.sleep(0.25)
                # check if thread is still alive (it may die unexpectedly)
                if not camera_thread.is_alive():
                    log.info("camera thread not found - restarting camera")
                    camera = Camera()
                    camera_thread = Thread(target=camera.fill_queue, args=(self.main_deque,), daemon=True)
                    camera_thread.start()
                    log.info("camera thread restarted")
                    self.bot.send_text("camera thread was restarted")


            #Check if user force opens the door
            if self.bot.node_let_in_flag == True:
                self.reset_cumuli_et_al()
                open_time = 5
                self.bot.send_text('Ok door is open for ' + str(open_time) + 's...')
                time.sleep(open_time)
                self.bot.send_text('Door locked again, back to business...')

            # check if we should export a live image
            if (last_run_time+10 < current_time):
                last_run_time=current_time
                liveImageThread=Thread(target=self.bot.sendLiveImage, daemon=True)
                liveImageThread.start()
                #self.bot.sendLiveImage()

    def dummy_queque_handler(self):
        # Do this to force run all networks s.t. the network inference time stabilizes
        self.single_debug()

        dummyque = DummyDQueque()
        dummy_thread = Thread(target=dummyque.dummy_queque_filler, args=(self.main_deque,))
        dummy_thread.start()

        while(True):
            if len(self.main_deque) > self.QUEQUE_MAX_THRESHOLD:
                self.main_deque.clear()
                log.info('DELETING QUEUE BECAUSE IS IS OVERLOADED!')
                self.bot.send_text(message='Too many images to process ... had to kill Queue!')

            elif len(self.main_deque) > self.DEFAULT_FPS_OFFSET:
                self.queque_worker()

            else:
                log.info('Nothing to work with => Queue_length:'+str(len(self.main_deque)))
                time.sleep(0.25)

            #Check if user force opens the door
            if self.bot.node_let_in_flag == True:
                self.reset_cumuli_et_al()
                open_time = 5
                self.bot.send_text('Ok door is open for ' + str(open_time) + 's...')
                time.sleep(open_time)
                self.bot.send_text('Door locked again, back to business...')

    def feed(self, target_img, img_name):
        target_event_obj = Event_Element(img_name=img_name, cc_target_img=target_img)

        start_time = time.time()
        single_cascade = self.base_cascade.do_single_cascade(event_img_object=target_event_obj)
        #log.info("base cascade created")
        single_cascade.total_inference_time = sum(filter(None, [
            single_cascade.cc_inference_time,
            single_cascade.cr_inference_time,
            single_cascade.bbs_inference_time,
            single_cascade.haar_inference_time,
            single_cascade.ff_bbs_inference_time,
            single_cascade.ff_haar_inference_time,
            single_cascade.pc_inference_time]))
        total_runtime = time.time() - start_time
        log.info('Total Runtime:'+str(round(total_runtime,3))+ ' s')

        return total_runtime, single_cascade

class Event_Element():
    def __init__(self, img_name, cc_target_img):
        self.img_name = img_name
        self.cc_target_img = cc_target_img
        self.cc_cat_bool = None
        self.cc_pred_bb = None
        self.cc_inference_time = None
        self.cr_class = None
        self.cr_val = None
        self.cr_inference_time = None
        self.bbs_target_img = None
        self.bbs_pred_bb = None
        self.bbs_inference_time = None
        self.haar_pred_bb = None
        self.haar_inference_time = None
        self.ff_haar_bool = None
        self.ff_haar_val = None
        self.ff_haar_inference_time = None
        self.ff_bbs_bool = None
        self.ff_bbs_val = None
        self.ff_bbs_inference_time = None
        self.face_box = None
        self.face_bool = None
        self.pc_prey_class = None
        self.pc_prey_val = None
        self.pc_inference_time = None
        self.total_inference_time = None
        self.output_img = None

class Cascade:
    def __init__(self):
        # Models
        self.cc_mobile_stage = CC_MobileNet_Stage()
        self.pc_stage = PC_Stage()
        self.ff_stage = FF_Stage()
        self.eyes_stage = Eye_Stage()
        self.haar_stage = Haar_Stage()

    def do_single_cascade(self, event_img_object):
        log.info('img_name:'+event_img_object.img_name)
        cc_target_img = event_img_object.cc_target_img
        original_copy_img = cc_target_img.copy()
        #cv2.imwrite(os.path.join(cat_cam_py,'test.jpg'),original_copy_img)
        #Do CC
        start_time = time.time()
        dk_bool, cat_bool, bbs_target_img, pred_cc_bb_full, cc_inference_time = self.do_cc_mobile_stage(cc_target_img=cc_target_img)
        log.info('CC_Do Time:'+str(round(time.time() - start_time,3))+' s')
        event_img_object.cc_cat_bool = cat_bool
        event_img_object.cc_pred_bb = pred_cc_bb_full
        event_img_object.bbs_target_img = bbs_target_img
        event_img_object.cc_inference_time = cc_inference_time

        if cat_bool and bbs_target_img.size != 0:
            log.info('Cat Detected!')
            rec_img = self.cc_mobile_stage.draw_rectangle(img=original_copy_img, box=pred_cc_bb_full, color=(255, 0, 0), text='CC_Pred')
            #log.info('writing test.jpg: rec_img cat detected')
            height, width = rec_img.shape[:2]
            #log.info("rec_img Size "+ str(width)+ "x"+str(height))
            height, width = bbs_target_img.shape[:2]
            #log.info("bbs_target_img Size"+str(width)+"x"+str(height))
            height, width = cc_target_img.shape[:2]
            #log.info("cc_target_img Size"+str(width)+"x"+str(height))
            try:
                my_resul = cv2.imwrite('rec_img.jpg',rec_img)
            except cv2.error as e:
                log.info("writing rec_img.jpg failed")
            #my_resul = cv2.imwrite('bbs_target_img.jpg',bbs_target_img)
            #my_resul = cv2.imwrite('cc_target_img.jpg',cc_target_img)
            #Do HAAR
            haar_snout_crop, haar_bbs, haar_inference_time, haar_found_bool = self.do_haar_stage(target_img=bbs_target_img, pred_cc_bb_full=pred_cc_bb_full, cc_target_img=cc_target_img)
            rec_img = self.cc_mobile_stage.draw_rectangle(img=rec_img, box=haar_bbs, color=(0, 255, 255), text='HAAR_Pred')
            #log.info('writing test.jpg: rec_img haar')
            #my_resul = cv2.imwrite('rec_img_haar.jpg',rec_img)
            height, width = rec_img.shape[:2]
            #log.info("rec_img Size"+str(width)+"x"+str(height))

            event_img_object.haar_pred_bb = haar_bbs
            event_img_object.haar_inference_time = haar_inference_time

            if haar_found_bool and haar_snout_crop.size != 0 and self.cc_haar_overlap(cc_bbs=pred_cc_bb_full, haar_bbs=haar_bbs) >= 0.1:
                inf_bb = haar_bbs
                face_bool = True
                snout_crop = haar_snout_crop

            else:
                # Do EYES
                bbs_snout_crop, bbs, eye_inference_time = self.do_eyes_stage(eye_target_img=bbs_target_img,
                                                                             cc_pred_bb=pred_cc_bb_full,
                                                                             cc_target_img=cc_target_img)
                rec_img = self.cc_mobile_stage.draw_rectangle(img=rec_img, box=bbs, color=(255, 0, 255), text='BBS_Pred')
                event_img_object.bbs_pred_bb = bbs
                event_img_object.bbs_inference_time = eye_inference_time

                # Do FF for Haar and EYES
                bbs_dk_bool, bbs_face_bool, bbs_ff_conf, bbs_ff_inference_time = self.do_ff_stage(snout_crop=bbs_snout_crop)
                event_img_object.ff_bbs_bool = bbs_face_bool
                event_img_object.ff_bbs_val = bbs_ff_conf
                event_img_object.ff_bbs_inference_time = bbs_ff_inference_time

                inf_bb = bbs
                face_bool = bbs_face_bool
                snout_crop = bbs_snout_crop

            event_img_object.face_bool = face_bool
            event_img_object.face_box = inf_bb

            if face_bool:
                #log.info('writing test.jpg: rec_img')
                #my_resul = cv2.imwrite('/home/rock/test.jpg',rec_img)
                rec_img = self.cc_mobile_stage.draw_rectangle(img=rec_img, box=inf_bb, color=(255, 255, 255), text='INF_Pred')
                log.info('Face Detected!')
                #

                #Do PC
                pred_class, pred_val, inference_time = self.do_pc_stage(pc_target_img=snout_crop)
                log.info('Prey Prediction: ' + str(pred_class))
                log.info('Pred_Val: ' + str('%.2f' % pred_val))
                pc_str = 'PC_Pred: ' + str(pred_class) + ' @ ' + str('%.2f' % pred_val)
                color = (0, 0, 255) if pred_class else (0, 255, 0)
                rec_img = self.input_text(img=rec_img, text=pc_str, text_pos=(15, 120), color=color)
                try:
                     my_resul = cv2.imwrite('preyprediction.jpg',rec_img)
                except cv2.error as e:
                     log.info('writing preyprediction.jpg failed:')
                     log.info(e)
                event_img_object.pc_prey_class = pred_class
                event_img_object.pc_prey_val = pred_val
                event_img_object.pc_inference_time = inference_time

            else:
                log.info('No Face Found...')
                ff_str = 'No_Face'
                rec_img = self.input_text(img=rec_img, text=ff_str, text_pos=(15, 120), color=(255, 255, 0))

        else:
            log.info('No Cat Found...')
            rec_img = self.input_text(img=original_copy_img, text='CC_Pred: NoCat', text_pos=(15, 120), color=(255, 255, 0))

        #log.info('writing test.jpg')
        #my_resul = cv2.imwrite('/home/rock/test.jpg',rec_img)
        #log.info(my_resul)
        #Always save rec_img in event_img object
        event_img_object.output_img = rec_img
        return event_img_object

    def cc_haar_overlap(self, cc_bbs, haar_bbs):
        cc_area = abs(cc_bbs[0][0] - cc_bbs[1][0]) * abs(cc_bbs[0][1] - cc_bbs[1][1])
        haar_area = abs(haar_bbs[0][0] - haar_bbs[1][0]) * abs(haar_bbs[0][1] - haar_bbs[1][1])
        overlap = haar_area / cc_area
        log.info('Overlap: '+str(overlap))
        return overlap

    def infere_snout_crop(self, bbs, haar_bbs, bbs_face_bool, bbs_ff_conf, haar_face_bool, haar_ff_conf, cc_target_img):
        #Combine BBS's if both are faces
        if bbs_face_bool and haar_face_bool:
            xmin = min(bbs[0][0], haar_bbs[0][0])
            ymin = min(bbs[0][1], haar_bbs[0][1])
            xmax = max(bbs[1][0], haar_bbs[1][0])
            ymax = max(bbs[1][1], haar_bbs[1][1])
            inf_bb = np.array([(xmin,ymin), (xmax,ymax)]).reshape((-1, 2))
            snout_crop = cc_target_img[ymin:ymax, xmin:xmax]
            return snout_crop, inf_bb, False, True, (bbs_ff_conf + haar_ff_conf)/2

        #When they are different choose the one that is true, if none is true than there is no face
        else:
            if bbs_face_bool:
                xmin = bbs[0][0]
                ymin = bbs[0][1]
                xmax = bbs[1][0]
                ymax = bbs[1][1]
                inf_bb = np.array([(xmin, ymin), (xmax, ymax)]).reshape((-1, 2))
                snout_crop = cc_target_img[ymin:ymax, xmin:xmax]
                return snout_crop, inf_bb, False, True, bbs_ff_conf
            elif haar_face_bool:
                xmin = haar_bbs[0][0]
                ymin = haar_bbs[0][1]
                xmax = haar_bbs[1][0]
                ymax = haar_bbs[1][1]
                inf_bb = np.array([(xmin, ymin), (xmax, ymax)]).reshape((-1, 2))
                snout_crop = cc_target_img[ymin:ymax, xmin:xmax]
                return snout_crop, inf_bb, False, True, haar_ff_conf
            else:
                ff_conf = (bbs_ff_conf + haar_ff_conf)/2 if haar_face_bool else bbs_ff_conf
                return None, None, False, False, ff_conf

    def calc_iou(self, gt_bbox, pred_bbox):
        (x_topleft_gt, y_topleft_gt), (x_bottomright_gt, y_bottomright_gt) = gt_bbox.tolist()
        (x_topleft_p, y_topleft_p), (x_bottomright_p, y_bottomright_p) = pred_bbox.tolist()

        if (x_topleft_gt > x_bottomright_gt) or (y_topleft_gt > y_bottomright_gt):
            raise AssertionError("Ground Truth Bounding Box is not correct")
        if (x_topleft_p > x_bottomright_p) or (y_topleft_p > y_bottomright_p):
            raise AssertionError("Predicted Bounding Box is not correct", x_topleft_p, x_bottomright_p, y_topleft_p, y_bottomright_gt)

        # if the GT bbox and predcited BBox do not overlap then iou=0
        if (x_bottomright_gt < x_topleft_p):# If bottom right of x-coordinate  GT  bbox is less than or above the top left of x coordinate of  the predicted BBox
            return 0.0
        if (y_bottomright_gt < y_topleft_p):  # If bottom right of y-coordinate  GT  bbox is less than or above the top left of y coordinate of  the predicted BBox
            return 0.0
        if (x_topleft_gt > x_bottomright_p):  # If bottom right of x-coordinate  GT  bbox is greater than or below the bottom right  of x coordinate of  the predcited BBox
            return 0.0
        if (y_topleft_gt > y_bottomright_p):  # If bottom right of y-coordinate  GT  bbox is greater than or below the bottom right  of y coordinate of  the predcited BBox
            return 0.0

        GT_bbox_area = (x_bottomright_gt - x_topleft_gt + 1) * (y_bottomright_gt - y_topleft_gt + 1)
        Pred_bbox_area = (x_bottomright_p - x_topleft_p + 1) * (y_bottomright_p - y_topleft_p + 1)

        x_top_left = np.max([x_topleft_gt, x_topleft_p])
        y_top_left = np.max([y_topleft_gt, y_topleft_p])
        x_bottom_right = np.min([x_bottomright_gt, x_bottomright_p])
        y_bottom_right = np.min([y_bottomright_gt, y_bottomright_p])

        intersection_area = (x_bottom_right - x_top_left + 1) * (y_bottom_right - y_top_left + 1)

        union_area = (GT_bbox_area + Pred_bbox_area - intersection_area)

        return intersection_area / union_area

    def do_cc_mobile_stage(self, cc_target_img):
        pred_cc_bb_full, cat_bool, inference_time = self.cc_mobile_stage.do_cc(target_img=cc_target_img)
        dk_bool = False if cat_bool is True else True
        if cat_bool:
            bbs_xmin = pred_cc_bb_full[0][0]
            bbs_ymin = pred_cc_bb_full[0][1]
            bbs_xmax = pred_cc_bb_full[1][0]
            bbs_ymax = pred_cc_bb_full[1][1]
            bbs_target_img = cc_target_img[bbs_ymin:bbs_ymax, bbs_xmin:bbs_xmax]
            return dk_bool, cat_bool, bbs_target_img, pred_cc_bb_full, inference_time
        else:
            return dk_bool, cat_bool, None, None, inference_time

    def do_eyes_stage(self, eye_target_img, cc_pred_bb, cc_target_img):
        snout_crop, bbs, inference_time = self.eyes_stage.do_eyes(cc_target_img, eye_target_img, cc_pred_bb)
        return snout_crop, bbs, inference_time

    def do_haar_stage(self, target_img, pred_cc_bb_full, cc_target_img):
        haar_bbs, haar_inference_time, haar_found_bool = self.haar_stage.haar_do(target_img=target_img, cc_bbs=pred_cc_bb_full, full_img=cc_target_img)
        pc_xmin = int(haar_bbs[0][0])
        pc_ymin = int(haar_bbs[0][1])
        pc_xmax = int(haar_bbs[1][0])
        pc_ymax = int(haar_bbs[1][1])
        snout_crop = cc_target_img[pc_ymin:pc_ymax, pc_xmin:pc_xmax].copy()

        return snout_crop, haar_bbs, haar_inference_time, haar_found_bool

    def do_ff_stage(self, snout_crop):
        face_bool, ff_conf, ff_inference_time = self.ff_stage.ff_do(target_img=snout_crop)
        dk_bool = False if face_bool is True else True
        return dk_bool, face_bool, ff_conf, ff_inference_time

    def do_pc_stage(self, pc_target_img):
        pred_class, pred_val, inference_time = self.pc_stage.pc_do(target_img=pc_target_img)
        return pred_class, pred_val, inference_time

    def input_text(self, img, text, text_pos, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 2
        lineType = 3

        cv2.putText(img, text,
                    text_pos,
                    font,
                    fontScale,
                    color,
                    lineType)
        return img

class NodeBot():
    def __init__(self):
        # Get environment variables for accessing Telegram API
        self.CHAT_ID= os.getenv('CHAT_ID')
        self.BOT_TOKEN = os.getenv('BOT_TOKEN')

        # Data for firebase messaging
        self.cred = credentials.Certificate("./firebasekey.json")
        self.storageBucket = os.getenv('FIREBASE_BUCKET')
        self.database = os.getenv('FIREBASE_DATABASE')
        self.livePrefix='live_img_'
        self.cascPrefix='last_casc_img_'
        self.timestamp=''

        self.last_msg_id = 0
        self.bot_updater = Updater(token=self.BOT_TOKEN)
        self.bot_dispatcher = self.bot_updater.dispatcher
        self.commands = ['/help', '/nodestatus', '/sendlivepic', '/sendlastdetectpic', '/letin', '/reboot']

        self.node_live_img = None
        self.node_queue_info = None
        self.node_status = None
        self.node_last_casc_img = None
        self.node_over_head_info = None
        self.node_let_in_flag = None

        #Init firebase
        self.init_firebase_messaging()
        #Init the listener
        self.init_bot_listener()

    def init_firebase_messaging(self):
        # initialize firebase messaging

        firebase_admin.initialize_app(self.cred,
            {'storageBucket': self.storageBucket, 'databaseURL':self.database})
        self.bucket = storage.bucket()
        # delete old images from firebase storage
        log.info('Deleting old images from firebase storage')
        cutoff = datetime.now(timezone.utc)- timedelta(days=1)
        blobs = self.bucket.list_blobs(prefix=self.cascPrefix)
        for blob in blobs:
            try:
                log.info(blob)
                updated=blob.updated
                # delete only those older than one day (=older than cutoff)

                if (updated>cutoff):
                    log.info("not yet to be deleted")
                else:
                    blob.delete()
                    log.info("deleted")
            except Exception as e:
                log.info('failed to delete '+blob)
                log.info(e)
        self.dbref=db.reference('/')
        self.images_ref=self.dbref.child('images')


    def init_bot_listener(self):
        telegram.Bot(token=self.BOT_TOKEN).send_message(chat_id=self.CHAT_ID, text='Good Morning, Catcam is online!' + '🤙')
        # Add all commands to handler
        help_handler = CommandHandler('help', self.bot_help_cmd)
        self.bot_dispatcher.add_handler(help_handler)
        node_status_handler = CommandHandler('nodestatus', self.bot_send_status)
        self.bot_dispatcher.add_handler(node_status_handler)
        send_pic_handler = CommandHandler('sendlivepic', self.bot_send_live_pic)
        self.bot_dispatcher.add_handler(send_pic_handler)
        send_last_casc_pic = CommandHandler('sendlastdetectpic', self.bot_send_last_casc_pic)
        self.bot_dispatcher.add_handler(send_last_casc_pic)
        letin = CommandHandler('letin', self.node_let_in)
        self.bot_dispatcher.add_handler(letin)
        reboot = CommandHandler('reboot', self.node_reboot)
        self.bot_dispatcher.add_handler(reboot)

        self.bot_dispatcher.add_error_handler(self.bot_error_handler)

        # Start the polling stuff
        self.bot_updater.start_polling()

    # https://www.askpython.com/python/examples/python-telegram-bot
    def bot_error_handler(self, update, context):
        prettytimestamp= datetime.now(tz=ZoneInfo('Europe/Zurich')).strftime("%d.%m.%Y %H:%M:%S")
        log.info(prettytimestamp+": Telegram bot error")
        log.info(context.error)

    def bot_help_cmd(self, bot, update):
        bot_message = 'The following commands are supported:'
        for command in self.commands:
            bot_message += '\n ' + command
        self.send_text(bot_message)

    def node_let_in(self, bot, update):
        self.node_let_in_flag = True

    def node_reboot(self, bot, update):
        for i in range(5):
            time.sleep(1)
            bot_message = 'Rebooting in ' + str(5-i) + ' seconds...'
            self.send_text(bot_message)
        self.send_text('See ya later Alligator 🐊🐊🐊')
        os.system("sudo reboot")

    def bot_send_last_casc_pic(self, bot, update):
        caption = 'Latest detection result'
        #if self.node_last_casc_img is not None:
        if os.path.isfile('last_casc.jpg'):
        #    self.send_img(self.node_last_casc_img, caption)
            self.send_img_file('last_casc.jpg',caption)
        else:
            if self.node_last_casc_img is not None:
                self.send_img(self.node_last_casc_img, caption)
            else:
                self.send_text('Detection did not happen yet...')

    def bot_send_live_pic(self, bot, update):
        if self.node_live_img is not None:
            try:
                cv2.imwrite('live_img.jpg', self.node_live_img)
            except cv2.error as e:
                log.info("writing live_img.jpg failed")
                self.send_text('could not write live_img.jpg')
                return

            caption = 'Current image'
            self.send_img(self.node_live_img, caption)
        else:
            self.send_text('No image available yet...')

    def bot_send_status(self, bot, update):
        if self.node_queue_info is not None and self.node_over_head_info is not None:
            bot_message = 'Queue length: ' + str(self.node_queue_info) + '\nOverhead: ' + str(self.node_over_head_info) + 's'
        else:
            bot_message = 'No info yet...'
        self.send_text(bot_message)

    def send_text(self, message):
        try:
            telegram.Bot(token=self.BOT_TOKEN).send_message(chat_id=self.CHAT_ID, text=message, parse_mode=telegram.ParseMode.MARKDOWN)
        except Exception as e:
            log.info('failed to send text message to telegram bot')
            log.info('message was '+message)
            log.info(e)

    def send_img_file(self, filename, caption):
        try:
            telegram.Bot(token=self.BOT_TOKEN).send_photo(chat_id=self.CHAT_ID, photo=open(filename, 'rb'), caption=caption)
        except Exception as e:
            log.info('failed to send image to telegram bot:')
            log.info(e)

    def send_img(self, img, caption):
        try:
            cv2.imwrite('degubi.jpg', img)
        except cv2.error as e:
            log.info("writing degubi.jpg failed")
            return
        self.send_img_file('degubi.jpg', caption)
        #try:
        #    telegram.Bot(token=self.BOT_TOKEN).send_photo(chat_id=self.CHAT_ID, photo=open('degubi.jpg', 'rb'), caption=caption)
        #except Exception as e:
        #    log.info('failed to send image to telegram bot:')
        #    log.info(e)

    def send_voice(self, audio, caption):
        #audio_r='86.wav'
        try:
            telegram.Bot(token=self.BOT_TOKEN).send_voice(chat_id=self.CHAT_ID, voice=open(audio, 'rb'), caption=caption)
        except Exception as e:
            log.info('failed to send voice message to telegram bot:')
            log.info(e)

    # upload image to firebase storage bucket
    def uploadImage(self, imagefile):
        blob=self.bucket.blob(imagefile)
        returnUrl=""
        if (os.path.isfile(imagefile)):
            try:
                blob.upload_from_filename(imagefile)
                blob.make_public()
                returnUrl = blob.public_url
            except Exception as e:
                log.info('failed to upload image:')
                log.info(e)
                returnUrl=""
        else:
            log.info('image file for upload to firebase not accessible? '+imagefile)
        return returnUrl

    def addImageToDatabase(self, url, timeString):
        timeStamp=datetime.strptime(timeString, "%d.%m.%Y %H:%M:%S")
        unixTimeStamp=int(timeStamp.timestamp())
        try:
            self.images_ref.update({
                unixTimeStamp : {
                    'timestamp' : timeString,
                    'url' : url
                }
            })
        except Exception as e:
            log.info('failed to update database:')
            log.info(timeString)
            log.info(e)


    # send firebase push notification
    def sendPushNotification(self, title, body, imageSource, time, topic):
        topicAsString = ""
        imgUrl= self.uploadImage(imageSource)
        if (imgUrl==""):
            log.info("not sending message, image upload failed")
            return


        if (int(topic) == 0):
            self.addImageToDatabase(imgUrl,time)
            log.info("Topic is alert")
            topicAsString = "alert"
            message = messaging.Message(
                {"ImageURL": imgUrl,
                 "Type" : topicAsString,
                 "Time" : time},
                notification=messaging.Notification(
                    title=title,
                    body=body,
                    image=imgUrl
                ),
                topic=topicAsString,
                android=messaging.AndroidConfig(
                    notification=messaging.AndroidNotification(
                        channel_id="alert"
                    )
                )
            )
            log.info("Sending message")
            returnString=""
            try:
                returnString = messaging.send(message)
            except Exception as e:
                log.info('failed to send message:')
                log.info(self.timestamp)
                log.info(e)
            log.info(returnString)
        elif (int(topic) == 1):
            log.info("Topic is update")
            topicAsString = "update"
            message = messaging.Message(
                {"ImageURL": imgUrl,
                 "Type" : topicAsString,
                 "Time" : time},
                notification=messaging.Notification(
                    title=title,
                    body=body,
                    image=imgUrl
                ),
                topic=topicAsString,
                android=messaging.AndroidConfig(
                    notification=messaging.AndroidNotification(
                        channel_id="update"
                    )
                )
            )
            try:
                returnString = messaging.send(message)
            except Exception as e:
                log.info('failed to send message:')
                log.info(self.timestamp)
                log.info(e)
            log.info(returnString)

    def sendDetectImage(self, message, text, image):
        timestamp_string=datetime.now(tz=ZoneInfo('Europe/Zurich')).strftime("%d.%m.%Y %H:%M:%S")
        imgfilename=self.cascPrefix+timestamp_string+'_detect.jpg'
        try:
            my_resul = cv2.imwrite(imgfilename,image)
            # save curremt live image for future model training
            if self.node_live_img is not None:
                my_resul = cv2.imwrite("clean_image_detect"+timestamp_string+".jpg",self.node_live_img)
        except cv2.error as e:
            log.info("writing detection image failed:")
            log.info(e)
            return
        self.sendPushNotification(message, text, imgfilename, timestamp_string,"0")

    def sendCascImage(self):
        #font=cv2.FONT_HERSHEY_SIMPLEX
        timestamp_string=datetime.now(tz=ZoneInfo('Europe/Zurich')).strftime("%d.%m.%Y %H:%M:%S")
        last_casc_filename=self.cascPrefix+timestamp_string+'.jpg'
        try:
            my_resul = cv2.imwrite(last_casc_filename,self.node_last_casc_img)
        except cv2.error as e:
            log.info("writing last cascade image failed:")
            log.info(e)
            return
        self.sendPushNotification("Latest detection result", "detection result", last_casc_filename, timestamp_string,"0")

    def sendLiveImage(self):
        log.info("checking for firebase file")
        blob=self.bucket.blob('settings.txt')
        try:
            blob_exists=blob.exists()
        except Exception as e:
            log.info('checking for settings.txt failed:')
            log.info(e)
            return
        if (blob_exists):
            log.info("File exists, we should send image")
            timestamp_string=datetime.now(tz=ZoneInfo('Europe/Zurich')).strftime("%d.%m.%Y %H:%M:%S")

            live_filename=self.livePrefix+timestamp_string+'.jpg'
            blob.download_to_filename("settings.txt")
            f=open('settings.txt')
            firstline = f.readline()
            f.close()
            #if ('{SendCascImage:True}' in firstline):
            #    log.info('text last_casc found in settings.txt')
            #    try:
            #        cv2.imwrite("last_casc_img.jpg", self.last_casc_img)
            #    except cv2.error as e:
            #        log.info("writing last casc img failed:")
            #        log.info(e)
            #else:
            try:
                # delete old images from firebase storage
                log.info('Deleting old images from firebase storage')
                oldblobs = self.bucket.list_blobs(prefix=self.livePrefix)
                for oldblob in oldblobs:
                    log.info(oldblob)
                    oldblob.delete()
                # write live file
                cv2.imwrite(live_filename, self.node_live_img)
                # send it
                self.sendPushNotification("current image","current image",live_filename,timestamp_string,"1")
                blob.delete()
            except cv2.error as e:
                log.info("writing live img failed:")
                log.info(e)

    def uploadLastCascImage(self):
        log.info("Sending last cascade image")
        if self.node_last_casc_img is not None:
            try:
                cv2.imwrite("last_casc.jpg", self.node_last_casc_img)
            except cv2.error as e:
                log.info("writing last_casc.jpg failed:")
                log.info(e)
                return
            time.sleep(1)
            blob = self.bucket.blob("last_casc.jpg")
            #if (blob.exists()):
            #    try:
            #        generation_match_precondition = None
            #        # Optional: set a generation-match precondition to avoid potential race conditions
            #        # and data corruptions. The request to delete is aborted if the object's
            #        # generation number does not match your precondition.
            #        blob.reload()  # Fetch blob metadata to use in generation_match_precondition.
            #        generation_match_precondition = blob.generation
            #        blob.delete(if_generation_match=generation_match_precondition)
            #        log.info("deleted old last_casc.jpg from firebase")
            #    except exceptions.FirebaseError as e:
            #        log.info("deleting old last_casc.jpg from firebase failed")
            #        log.info(e)
            #        return
            #    time.sleep(0.5)
            url= self.uploadImage("last_casc.jpg")
            log.info("uploaded new file last_casc.jpg.")
            log.info("last_casc.jpg:" + url)
        else:
            log.info("detection did not happen yet, cannot write last_casc.jpg")


class DummyDQueque():
    def __init__(self):
        self.target_img = cv2.imread(os.path.join(cat_cam_py, 'CatPreyAnalyzer/readme_images/lenna_casc_Node1_001557_02_2020_05_24_09-49-35.jpg'))

    def dummy_queque_filler(self, main_deque):
        while(True):
            img_name = datetime.now(tz=ZoneInfo('Europe/Zurich')).strftime("%Y_%m_%d_%H-%M-%S.%f")
            main_deque.append((img_name, self.target_img))
            log.info("Took image, que-length:"+str(main_deque.__len__()))
            time.sleep(0.4)

if __name__ == '__main__':

    prettytime=datetime.now(tz=ZoneInfo('Europe/Zurich')).strftime("%d.%m.%Y, %H:%M:%S")
    print(prettytime+":  starting sequential cascade feeder")
    sq_cascade = Sequential_Cascade_Feeder()
    log.info("Sequential_Cascade_Feeder initialized")
    sq_cascade.queque_handler()
