import os
import cv2
from cv2 import dnn
import dlib
import numpy as np
import argparse
import inception_resnet_v1
import tensorflow as tf
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from utils.datasets import get_labels
from keras.models import load_model
from utils.preprocessor import preprocess_input
import threading
import tkinter as tk
from   tkinter import Menu
from   tkinter import Frame
from PIL import ImageTk, Image
import datetime
import addFace as af
import time as t
global count
import xmltodict

import face_recognition

# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
count = 0

class FaceSDK:
  def __init__(self):
    self.main()
  def get_args(self):
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age and gender for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None,
                        help="path to weight file (e.g. weights.18-4.06.hdf5)")
    parser.add_argument("--depth", type=int, default=16,
                        help="depth of network")
    parser.add_argument("--width", type=int, default=8,
                        help="width of network")
    args = parser.parse_args()
    return args
  
  def open_add_face_ui(self):
    global openAddUI
    openAddUI = True
    af.Open_dialog_add_face(self)    
    
  def set_cam(self):
    thread = threading.Thread(target=self.faceDetection, args=())
    thread.start()
    global count
    print(count)
    count +=1
   
  # def is_intersect(self, tuple_a, tuple_b):
  #   return (tuple_a[0] < tuple_b[1] and tuple_a[1] > tuple_b[0] and
  #   tuple_a[2] > tuple_b[3] and tuple_a[3] < tuple_b[2])

  def is_intersect(self, tuple_a, tuple_b):


    rectA = dlib.rectangle(left = int(tuple_a[0]), top = int(tuple_a[2]), right = int(tuple_a[1]), bottom = int(tuple_a[3]))
    rectB = dlib.rectangle(left = int(tuple_b[0]), top = int(tuple_b[2]), right = int(tuple_b[1]), bottom = int(tuple_b[3]))
    interRect = rectA.intersect(rectB)
    return interRect.area() > 0

  def load_model(self):
    global sess, age, gender, train_mode, images_pl
    global fa
    global face_net
    global emotion_labels, emotion_classifier, emotion_target_size
    global parentDir
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", "--M", default="./models", type=str, help="Model Path")
    # args = parser.parse_args()    
    sess, age, gender, train_mode, images_pl = self.load_network(parentDir + "./models")   

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(parentDir + "\\src\\dist\\FaceSDK\\face_recognition_models\\models\\shape_predictor_68_face_landmarks.dat")
    fa = FaceAligner(predictor, desiredFaceWidth=160)   
    
    face_model_txt = parentDir + '/trained_models/face_models/deploy_resnet.prototxt'
    face_model_bin = parentDir + '/trained_models/face_models/res10_300x300_ssd_iter_140000_fp16.caffemodel'
    face_net = dnn.readNetFromCaffe(face_model_txt, face_model_bin) 
    
    # face_net.setPreferableTarget(DNN_TARGET_OPENCL)
      
  def faceDetection(self):
    print('xxxx') 
    global sess, age, gender, train_mode, images_pl
    global label_header
    global face_img_list
    global face_time_list
    global label_face_ubf
    global count
    global openAddUI
    global fa
    global face_net
    global emotion_labels, emotion_classifier, emotion_target_size
    # for face detection
    global doc
    global parentDir
    # load model and weights
    img_size = 160
    
    # capture video
    # cap = cv2.VideoCapture('C:\\Users\\Owner\\Desktop\\Summer-2018\\AI-Internship\\face_classification\\src\\dist\\trained_models\\dinner.mp4')
    cap = cv2.VideoCapture(0)
    
    #face model
    filename = parentDir + '/Deha.xml'
    if os.stat(filename).st_size != 0:
      with open(filename) as fd:
          doc = xmltodict.parse(fd.read())   
          for key_out in doc['root'].keys():
            found = False
            for key in doc['root'][key_out].keys():
              doc['root'][key_out][key] = np.array(doc['root'][key_out][key]['item'], dtype=float)          
    else:
      print('Database is empty')    

    # emotion model
    if count == 1:
      emotion_model_path = parentDir + '/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
      emotion_labels = get_labels('fer2013')
      emotion_classifier = load_model(emotion_model_path, compile=False)
      emotion_target_size = emotion_classifier.input_shape[1:3]      

    inWidth = 300
    inHeight = 300
    confThreshold = 0.5    
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

     # Track model
    trackerType = "KCF"      
    listTrack = []
    list_faces = []
    gender_show = ''
    age_show = ''
    emotion_txt = ''
    knownPeople = ''
    countFrame = 0
    listBox = []
    listDetails = []
    passEmotions = []
    emotion_dict = {'happy':0, 'sad':0,'surprise':0, 'disgust':0, 'angry':0,'fear':0,'neutral':0}
    while True:
      if openAddUI == True:
        print('Stop Camera')
        break    
      # get video frame
      ret, img = cap.read()
      cloneImage = img.copy()
      countFrame += 1

      timer = cv2.getTickCount()

      male_count = 0
      female_count = 0
      list_time = []

       # Track box
      if listTrack != []:
        for track in listTrack:
          ok, track[0] = track[1].update(img)
          if ok == False or track[0][0] <= 0 or track[0][1] >= img.shape[0] or track[0][2] <= 0 or track[0][2] >= img.shape[1]:
            listTrack.remove(track)

      if countFrame % 2 == 0:
        cols = img.shape[1]
        rows = img.shape[0]        
    
        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_h, img_w, _ = np.shape(input_img)      
        #detect face      
        face_net.setInput(dnn.blobFromImage(img, 1.0, (inWidth, inHeight), (104.0, 177.0, 123.0), False, False))
        detections = face_net.forward()
        faces = np.empty((1, img_size, img_size, 3))
        passEmotions = []
        
        list_faces = []
        listBox = []
        countFace = 0
        for i in range(detections.shape[2]):
          confidence = detections[0, 0, i, 2]
          if confidence > confThreshold:
            x1 = int(detections[0, 0, i, 3] * cols)
            y1 = int(detections[0, 0, i, 4] * rows)
            x2 = int(detections[0, 0, i, 5] * cols)
            y2 = int(detections[0, 0, i, 6] * rows)     
            face_image = img[y1:y2, x1:x2]

            if face_image.shape[0] > 0 and face_image.shape[1] > 0: 
              countFace += 1
              list_faces.append(face_image)
              cur_time = datetime.datetime.now()
              list_time.append(cur_time.strftime('%H:%M:%S'))  
            
              box = (x1, x2, y1, y2)
              
              foundTrack = False 
              for index, j in enumerate(listTrack):
                if self.is_intersect(j[0], box):
                  knownPeople = j[2]
                  foundTrack = True
                  j[1].init(img, box)
                  if knownPeople == 'Unknown' and countFrame > 1000:
                    countFrame = 0
                    faceResize = cv2.resize(face_image, (100, 100), interpolation=cv2.INTER_CUBIC)
                    face_RGB = cv2.cvtColor(faceResize,cv2.COLOR_BGR2RGB)
                    face_features = face_recognition.face_encodings(face_RGB)
                    if face_features != []:
                      if os.stat(filename).st_size != 0:
                        for key_out in doc['root'].keys():
                          found = False
                          for key in doc['root'][key_out].keys():
                            face_distances = face_recognition.face_distance(doc['root'][key_out][key], face_features)
                            if min(face_distances) < 0.4:
                              found = True  
                          if found:
                            knownPeople = key_out
                    j[2] = knownPeople
                  break
              if not foundTrack:
                faceResize = cv2.resize(face_image, (100, 100), interpolation=cv2.INTER_CUBIC)
                face_RGB = cv2.cvtColor(faceResize,cv2.COLOR_BGR2RGB)
                face_features = face_recognition.face_encodings(face_RGB)
                knownPeople = 'Unknown'
                if face_features != []:
                  if os.stat(filename).st_size != 0:
                    for key_out in doc['root'].keys():
                      found = False
                      for key in doc['root'][key_out].keys():
                        face_distances = face_recognition.face_distance(doc['root'][key_out][key], face_features)
                        if min(face_distances) < 0.4:
                          found = True
                      if found:
                        knownPeople = key_out
                        
                track = cv2.TrackerKCF_create()
                track.init(img, box) 
                listTrack.append([box, track, knownPeople])
              listBox.append((box, knownPeople, cloneImage))

        if listBox != [] and listDetails != []:
          for index, details in enumerate(listDetails):
            gender_show = details[0]
            age_show = details[1] 
            emotion_txt = details[2]
            if index < len(listBox):
              knownPeople = listBox[index][1]
            passEmotions.append(emotion_txt)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            labelSize, baseLine = cv2.getTextSize('normaler', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - 4 * labelSize[1] - baseLine), (x1 + labelSize[0], y1),(255, 255, 255))
            cv2.putText(img, gender_show, (x1, y1 - baseLine), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            cv2.putText(img, age_show, (x1, y1 - labelSize[1] - baseLine), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0))
            cv2.putText(img, emotion_txt, (x1, y1 - 2 * labelSize[1] - baseLine), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0))
            cv2.putText(img, knownPeople, (x1, y1 - 3 * labelSize[1] - baseLine), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0))
        listDetails = []

      else:
        for index, box in enumerate(listBox):
          x1 = box[0][0]
          x2 = box[0][1]
          y1 = box[0][2]
          y2 = box[0][3]
          knownPeople = box[1]
          passImage = box[2]

          faceR = dlib.rectangle(left = x1, top = y1, right = x2, bottom = y2)
          input_img = cv2.cvtColor(passImage, cv2.COLOR_BGR2RGB)
          gray = cv2.cvtColor(passImage, cv2.COLOR_BGR2GRAY)
          faces[0, :, :, :] = fa.align(input_img, gray, faceR)
          ages,genders = sess.run([age, gender], feed_dict={images_pl: faces, train_mode: False})
          
          if genders[0] == 1:
              gender_show = 'Male'
              male_count+=1
          else:
              gender_show = 'Female' 
              female_count+=1
          
          check_age = int(ages[0])
          if check_age < 8:
              age_show = '0 - 7'
          elif check_age >= 8 and  check_age < 15:
              age_show = '8 - 14'
          elif check_age >= 15 and  check_age < 25:
              age_show = '15 - 24'
          elif check_age >= 25 and  check_age < 38:
              age_show = '25 - 35'
          elif check_age >= 38 and  check_age < 48:
              age_show = '38 - 47'
          elif check_age >= 48 and  check_age < 60:
              age_show = '48 - 59'
          else:
              age_show = '60 - 100'    
          face_image = passImage[y1:y2, x1:x2]
          face_gray_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
          croppedMat = cv2.resize(face_image, (227, 227), interpolation=cv2.INTER_CUBIC)
          input_mat = dnn.blobFromImage(croppedMat, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), False)      
          
          emotion_face_image = cv2.resize(face_gray_image, (emotion_target_size))
          emotion_face_image = preprocess_input(emotion_face_image, True)
          emotion_face_image = np.expand_dims(emotion_face_image, 0)
          emotion_face_image = np.expand_dims(emotion_face_image, -1)
          emotion_label_arg = np.argmax(emotion_classifier.predict(emotion_face_image))
          emotion_txt = emotion_labels[emotion_label_arg]  
          if passEmotions != [] and index < len(passEmotions) and emotion_txt != passEmotions[index]:
            emotion_dict[passEmotions[index]] += 1

          listDetails.append((gender_show, age_show, emotion_txt))
          #draw text and rect
          cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
          labelSize, baseLine = cv2.getTextSize('normaler', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
          cv2.rectangle(img, (x1, y1 - 4 * labelSize[1] - baseLine), (x1 + labelSize[0], y1),(255, 255, 255))
          cv2.putText(img, gender_show, (x1, y1 - baseLine), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
          cv2.putText(img, age_show, (x1, y1 - labelSize[1] - baseLine), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0))
          cv2.putText(img, emotion_txt, (x1, y1 - 2 * labelSize[1] - baseLine), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0))
          cv2.putText(img, knownPeople, (x1, y1 - 3 * labelSize[1] - baseLine), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0))
      self.gender_text(male_count, female_count)
      self.emotion_text(emotion_dict.get('happy'), emotion_dict.get('sad'), emotion_dict.get('surprise'), emotion_dict.get('fear'), emotion_dict.get('angry'), emotion_dict.get('disgust'), emotion_dict.get('neutral'))

      for track in listTrack:
        for box in listBox:
          if not(self.is_intersect(track[0], box[0])):
            if track in listTrack:
             listTrack.remove(track)

      image = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
      src_mat = Image.fromarray(image)
      pic = ImageTk.PhotoImage(src_mat)   
      label_header['image'] = pic   
      img_list = self.convert_mat_to_img(list_faces)

      for index, face_img in enumerate(img_list): 
        face_img_list[index]['image'] = face_img   
      for index, time_text in enumerate(list_time):
        face_time_list[index]['text'] = time_text      
      label_face_ubf['text'] = 'Face: {}'.format(len(list_faces))
    
    print('end camera')
    cap.release()
    openAddUI = False
    
  def convert_mat_to_img(self,list_faces):
    img_list = []
    for index, face_img in enumerate(list_faces):
      face_img = cv2.cvtColor(face_img,cv2.COLOR_BGR2RGB)
      src_mat1 = Image.fromarray(face_img).resize((100, 100), Image.ANTIALIAS)
      pic1 = ImageTk.PhotoImage(src_mat1)      
      img_list.append(pic1)
    return img_list

  def load_network(self,model_path):
      sess = tf.Session()
      images_pl = tf.placeholder(tf.float32, shape=[None, 160, 160, 3], name='input_image')
      images_norm = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), images_pl)
      train_mode = tf.placeholder(tf.bool)
      age_logits, gender_logits, _ = inception_resnet_v1.inference(images_norm, keep_probability=0.8,
                                                                   phase_train=train_mode,
                                                                   weight_decay=1e-5)
      gender = tf.argmax(tf.nn.softmax(gender_logits), 1)
      age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
      age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
      init_op = tf.group(tf.global_variables_initializer(),
                         tf.local_variables_initializer())
      sess.run(init_op)
      saver = tf.train.Saver()
      ckpt = tf.train.get_checkpoint_state(model_path)
      if ckpt and ckpt.model_checkpoint_path:
          saver.restore(sess, ckpt.model_checkpoint_path)
          print("restore model!")
      else:
          pass
      return sess,age,gender,train_mode,images_pl
    
  def close_window(self,window):
      window.destroy()

  def gender_text(self,male_num, female_num):
      global label_num_icon_man
      global label_num_icon_woman
      label_num_icon_man['text'] = str(male_num)
      label_num_icon_woman['text'] = str(female_num)
      
  def emotion_text(self,face_happy_num, face_sad_num, face_surprise_num, face_fear_num, face_angry_num, face_disgust_num, face_neutral_num):
      global label_num_icon_happy
      global label_num_icon_sad
      global label_num_icon_surprise
      global label_num_icon_fear
      global label_num_icon_angry
      global label_num_icon_disgust
      global label_num_icon_neutral   
      
      global label_bar_icon_happy
      global label_bar_icon_sad
      global label_bar_icon_surprise
      global label_bar_icon_fear
      global label_bar_icon_angry
      global label_bar_icon_disgust
      global label_bar_icon_neutral
      
      total = face_happy_num + face_sad_num + face_surprise_num + face_fear_num + face_angry_num + face_disgust_num + face_neutral_num
      if total == 0:
        total = 1
      
      label_num_icon_happy['text'] = face_happy_num
      label_num_icon_sad['text'] = face_sad_num
      label_num_icon_surprise['text'] = face_surprise_num
      label_num_icon_fear['text'] = face_fear_num
      label_num_icon_angry['text'] = face_angry_num
      label_num_icon_disgust['text'] = face_disgust_num
      label_num_icon_neutral['text'] = face_neutral_num
  
      label_bar_icon_happy['width'] = int((face_happy_num / total) * 30)
      label_bar_icon_sad['width'] = int((face_sad_num / total) * 30)
      label_bar_icon_surprise['width'] = int((face_surprise_num / total) * 30)
      label_bar_icon_fear['width'] = int((face_fear_num / total) * 30)
      label_bar_icon_angry['width'] = int((face_angry_num / total) * 30)
      label_bar_icon_disgust['width'] = int((face_disgust_num / total) * 30)
      label_bar_icon_neutral['width'] = int((face_neutral_num / total) * 30)
      
  def do_popup(self, event):
    global popup
    # display the popup menu
    try:
        popup.tk_popup(event.x_root, event.y_root, 0)
    finally:
        # make sure to release the grab (Tk 8.0a1 only)
        popup.grab_release()
    
  def main(self):
      global label_num_icon_man
      global label_num_icon_woman    
      
      global popup
      
      global label_num_icon_happy
      global label_num_icon_sad
      global label_num_icon_surprise
      global label_num_icon_fear
      global label_num_icon_angry
      global label_num_icon_disgust
      global label_num_icon_neutral
      
      global label_bar_icon_happy
      global label_bar_icon_sad
      global label_bar_icon_surprise
      global label_bar_icon_fear
      global label_bar_icon_angry
      global label_bar_icon_disgust
      global label_bar_icon_neutral    
      
      global face_img_list
      global face_time_list
  
      global label_header
      global label_face_ubf
      global window 
      global openAddUI
      openAddUI = False
      global parentDir
      fileDir = os.path.dirname(os.path.abspath(__file__))   # Directory of the Module
      parentDir = os.path.dirname(fileDir)
      
      #root = tk.Tk()
      #root.attributes('-alpha', 0.0) #For icon
      ##root.lower()
      #root.iconify()
      window = tk.Tk()
      window.overrideredirect(1) #Remove border
      w = 1024 # width for the Tk window
      h = 840 # height for the Tk window  
      ws = window.winfo_screenwidth() # width of the screen
      hs = window.winfo_screenheight() # height of the screen
      x = (ws/2) - (w/2)
      y = (hs/2) - (h/2)
      window.geometry('%dx%d+%d+%d' % (w, h, x, y))
      
      header_frame = Frame(window, bg='blue', width=1024, height=49)
      top_frame = Frame(window, bg='white', highlightbackground="white", highlightthickness=20, width=640, height=480)
      
      bottom_frame = Frame(window, bg='white', width=640, highlightthickness=0,highlightbackground="white", height=360)
      right_frame = Frame(window, bg='white', highlightbackground="white", highlightthickness=20, width=384, height=840)
      
      # layout all of the main containers
      header_frame.grid(row=0, column=0, columnspan =2)
      top_frame.grid(row=1, column=0)
      bottom_frame.grid(row=2, column=0)  
      right_frame.grid(row=1, column=1, rowspan=2)
      
      # Header Frame
      header_frame.grid_propagate(0)
      header_frame.grid_rowconfigure(0, weight=1)
      header_frame.grid_columnconfigure(1, weight=1)   
      
      # create the widgets for the header frame
      header_Img = ImageTk.PhotoImage(Image.open(parentDir + "/images/header_label_fix.png").resize((1024, 49), Image.ANTIALIAS)) 
      label_header_img = tk.Label(header_frame, image = header_Img)
      label_header_img.grid(row=0, column=0, columnspan=3, sticky ='wesn')    
      
      label_header_text_1 = tk.Label(header_frame, text='DEHA SMART TECHNOLOGIES',  borderwidth=0, highlightthickness = 0, bg ='#4F86B1', fg='white')
      label_header_text_1.grid(row=0, column=0, sticky ='wns', padx=(20,0))
      
      label_header_text_2 = tk.Label(header_frame, text='SMART FACE', bg='#3E54AF', fg='white', borderwidth=0, highlightthickness = 0)
      label_header_text_2.grid(row=0, column=1, sticky ='ens', padx=(300,300)) 
      
      label_header_text_3 = tk.Label(header_frame, text='Setting', bg='#4F86B1', fg='white')
      label_header_text_3.grid(row=0, column=2, sticky ='ens', padx=(20,20))    
      
      button_Img = ImageTk.PhotoImage(Image.open(parentDir + "/images/button setting.png"))
      btnClose = tk.Button(header_frame, image = button_Img, borderwidth=0, highlightthickness = 0)
      btnClose.grid(row=0,column=2, padx= 70)     
      btnClose.bind("<Button-1>", self.do_popup)
      # Top Frame
      top_frame.grid_propagate(0)
      top_frame.grid_rowconfigure(0, weight=1)
      top_frame.grid_columnconfigure(0, weight=1)  
      
      # create the widgets for the top frame
      top_Img = ImageTk.PhotoImage(Image.open(parentDir + "/images/images.jpg").resize((640, 480), Image.ANTIALIAS))  
      label_header = tk.Label(top_frame, image = top_Img)
      label_header.grid(row=0, column=0, columnspan=1, sticky ='wesn')     
      
      # Right Frame
      right_frame.grid_propagate(0)
      right_frame.grid_rowconfigure(1, weight=1)
      right_frame.grid_columnconfigure(1, weight=1)     
      
      #Create 2 frames inside right frame
      top_right_frame = Frame(right_frame, bg='white', highlightbackground="#589EB2", highlightthickness=2, width=360, height=70)
      top_right_frame.grid(row=0)
      
      top_right_frame.grid_propagate(0)
      top_right_frame.grid_rowconfigure(0, weight=1)
      top_right_frame.grid_columnconfigure(2, weight=1)  
      
      bottom_right_frame = Frame(right_frame, bg='white', highlightbackground="#589EB2", highlightthickness=2, width=360, height=690)    
      bottom_right_frame.grid(row=1, sticky='n')
      bottom_right_frame.grid_propagate(0)
      bottom_right_frame.grid_rowconfigure(8, weight=1)
      bottom_right_frame.grid_columnconfigure(1, weight=1)    
      
      # create the widgets for the top_right_frame
      img_top_right_frame = ImageTk.PhotoImage(Image.open(parentDir + "/images/statistic_dot.jpg")) 
      label_top_right_frame = tk.Label(top_right_frame, image = img_top_right_frame,borderwidth=0, highlightthickness = 0 )
      label_top_right_frame.grid(row=0, column=0, padx=(5,0), sticky='e')   
      
      label_text_trf = tk.Label(top_right_frame, text= 'Statistic', borderwidth=0, highlightthickness = 0, bg='white')
      label_text_trf.grid(row=0, column=1, sticky='w')     
      
      btnImage_top_right_frame= ImageTk.PhotoImage(Image.open(parentDir + "/images/statistic_bnt.jpg")) 
      btn_top_right_frame = tk.Button(top_right_frame, image=btnImage_top_right_frame, borderwidth=0, highlightthickness = 0)
      btn_top_right_frame.grid(row=0,column=2, sticky='e')   
      
      cur_day = datetime.datetime.now()
      label_text_trf_btn = tk.Label(top_right_frame, text=cur_day.strftime('%a, %b %d, %Y'), borderwidth=0, highlightthickness = 0, bg='white')
      label_text_trf_btn.grid(row=0, column=3, padx=5, sticky='w')     

      # create the widgets for the bottom_right_frame
      happy_Img = ImageTk.PhotoImage(Image.open(parentDir + "/images/happy.jpg"))
      sad_Img = ImageTk.PhotoImage(Image.open(parentDir + "/images/sad.jpg"))
      surprise_Img = ImageTk.PhotoImage(Image.open(parentDir + "/images/surprise.jpg"))
      fear_Img = ImageTk.PhotoImage(Image.open(parentDir + "/images/fear.jpg"))
      angry_Img = ImageTk.PhotoImage(Image.open(parentDir + "/images/angry.jpg"))
      disgust_Img = ImageTk.PhotoImage(Image.open(parentDir + "/images/disgust.jpg"))
      nature_Img = ImageTk.PhotoImage(Image.open(parentDir + "/images/nature.jpg"))
      man_Img = ImageTk.PhotoImage(Image.open(parentDir + "/images/man.jpg"))
      woman_Img = ImageTk.PhotoImage(Image.open(parentDir + "/images/woman.jpg"))
      #bar_Img = ImageTk.PhotoImage(Image.open(parentDir + "/images/bar.jpg"))
      
      label_icon_happy = tk.Label(bottom_right_frame, image = happy_Img, borderwidth=0, bg='white', highlightthickness=0, pady=60)
      label_icon_happy.grid(row=0, column=0, pady=(10,20), sticky ='wesn')  
      label_bar_icon_happy = tk.Label(bottom_right_frame, borderwidth=0, bg='#EDAD52', width = 30, highlightthickness=0)
      label_bar_icon_happy.grid(row=0, column=1, pady=(10,20), sticky ='w')
      label_num_icon_happy = tk.Label(bottom_right_frame, borderwidth=0, text='5012', bg='white', highlightthickness=0)
      label_num_icon_happy.grid(row=0, column=2, pady=(10,20), padx=(5,10), sticky ='w')    
      
      
      label_icon_sad = tk.Label(bottom_right_frame, image = sad_Img, borderwidth=0, bg='white', highlightthickness=0, pady=60)
      label_icon_sad.grid(row=1, column=0, pady=(0,20), sticky ='wesn')
      label_bar_icon_sad = tk.Label(bottom_right_frame, borderwidth=0, bg='#EDAD52', width = 20, highlightthickness=0)
      label_bar_icon_sad.grid(row=1, column=1, pady=(10,20), sticky ='w')    
      label_num_icon_sad = tk.Label(bottom_right_frame, borderwidth=0, text='5012', bg='white', highlightthickness=0)
      label_num_icon_sad.grid(row=1, column=2, pady=(10,20), padx=(5,10), sticky ='w')    
      
      label_icon_surprise = tk.Label(bottom_right_frame, image = surprise_Img, borderwidth=0, bg='white', highlightthickness=0, pady=60)
      label_icon_surprise.grid(row=2, column=0, pady=(0,20), sticky ='wesn')
      label_bar_icon_surprise = tk.Label(bottom_right_frame, borderwidth=0, bg='#EDAD52', width = 20, highlightthickness=0)
      label_bar_icon_surprise.grid(row=2, column=1, pady=(10,20), sticky ='w')
      label_num_icon_surprise = tk.Label(bottom_right_frame, borderwidth=0, text='5012', bg='white', highlightthickness=0)
      label_num_icon_surprise.grid(row=2, column=2, pady=(10,20), padx=(5,10), sticky ='w')    
      
      label_icon_fear = tk.Label(bottom_right_frame, image = fear_Img, borderwidth=0, bg='white', highlightthickness=0)
      label_icon_fear.grid(row=3, column=0, pady=(0,20), sticky ='wesn')
      label_bar_icon_fear = tk.Label(bottom_right_frame, borderwidth=0, bg='#EDAD52', width = 20, highlightthickness=0)
      label_bar_icon_fear.grid(row=3, column=1, pady=(10,20), sticky ='w')    
      label_num_icon_fear = tk.Label(bottom_right_frame, borderwidth=0, text='5012', bg='white', highlightthickness=0)
      label_num_icon_fear.grid(row=3, column=2, pady=(10,20), padx=(5,10), sticky ='w')    
      
      label_icon_angry = tk.Label(bottom_right_frame, image = angry_Img, borderwidth=0, bg='white', highlightthickness=0)
      label_icon_angry.grid(row=4, column=0, pady=(0,20), sticky ='wesn')
      label_bar_icon_angry = tk.Label(bottom_right_frame, borderwidth=0, bg='#EDAD52', width = 20, highlightthickness=0)
      label_bar_icon_angry.grid(row=4, column=1, pady=(10,20), sticky ='w')   
      label_num_icon_angry = tk.Label(bottom_right_frame, borderwidth=0, text='5012', bg='white', highlightthickness=0)
      label_num_icon_angry.grid(row=4, column=2, pady=(10,20), padx=(5,10), sticky ='w')    
      
      label_icon_disgust = tk.Label(bottom_right_frame, image = disgust_Img, borderwidth=0, bg='white', highlightthickness=0)
      label_icon_disgust.grid(row=5, column=0, pady=(0,20), sticky ='wesn')   
      label_bar_icon_disgust = tk.Label(bottom_right_frame, borderwidth=0, bg='#EDAD52', width = 20, highlightthickness=0)
      label_bar_icon_disgust.grid(row=5, column=1, pady=(10,20), sticky ='w')    
      label_num_icon_disgust = tk.Label(bottom_right_frame, borderwidth=0, text='5012', bg='white', highlightthickness=0)
      label_num_icon_disgust.grid(row=5, column=2, pady=(10,20), padx=(5,10), sticky ='w')    
      
      label_icon_neutral = tk.Label(bottom_right_frame, image = nature_Img, borderwidth=0, bg='white', highlightthickness=0)
      label_icon_neutral.grid(row=6, column=0, pady=(0,20), sticky ='wesn') 
      label_bar_icon_neutral = tk.Label(bottom_right_frame, borderwidth=0, bg='#EDAD52', width = 20, highlightthickness=0)
      label_bar_icon_neutral.grid(row=6, column=1, pady=(10,20), sticky ='w')   
      label_num_icon_neutral = tk.Label(bottom_right_frame, borderwidth=0, text='5012', bg='white', highlightthickness=0)
      label_num_icon_neutral.grid(row=6, column=2, pady=(10,20), padx=(5,10), sticky ='w')    
      
      label_icon_man = tk.Label(bottom_right_frame, image = man_Img, borderwidth=0, bg='white', highlightthickness=0)
      label_icon_man.grid(row=7, column=0, padx=(10,10), sticky ='w')     
      label_num_icon_man = tk.Label(bottom_right_frame, borderwidth=0, text='5012', bg='white', highlightthickness=0)
      label_num_icon_man.grid(row=7, column=1, pady=(10,20), padx=(5,10), sticky ='w')    
      
      label_icon_woman = tk.Label(bottom_right_frame, image = woman_Img, borderwidth=0, bg='white', highlightthickness=0)
      label_icon_woman.grid(row=8, column=0, padx=(10,10), sticky ='wn')    
      label_num_icon_woman = tk.Label(bottom_right_frame, borderwidth=0, text='5012', bg='white', highlightthickness=0)
      label_num_icon_woman.grid(row=8, column=1, pady=(20,00), padx=(5,10), sticky ='wn')    
      
      #Bottom Frame
      bottom_frame.grid_propagate(0)
      bottom_frame.grid_rowconfigure(1, weight=1)    
      
      #Create 2 frames inside Bottom Frame
      up_bottom_frame = Frame(bottom_frame, bg='white', highlightbackground="#589EB2", highlightthickness=2, width=610, height=40)
      up_bottom_frame.grid(row=0, padx=(20,0))
      up_bottom_frame.grid_propagate(0)
      up_bottom_frame.grid_rowconfigure(0, weight=1)
      up_bottom_frame.grid_columnconfigure(2, weight=1)      
      
      down_bottom_frame = Frame(bottom_frame, bg='white', highlightbackground="#589EB2", highlightthickness=2, width=610, height=260)    
      down_bottom_frame.grid(row=1, sticky='n', padx=(20,0))
      down_bottom_frame.grid_propagate(0)
      down_bottom_frame.grid_rowconfigure(4, weight=1)
      down_bottom_frame.grid_columnconfigure(4, weight=1)      
      
      # create the widgets for the up_bottom_frame   
      icon_up_bottom_frame = ImageTk.PhotoImage(Image.open(parentDir + "/images/bottom_icon.jpg")) 
      label_icon_up_bottom_frame = tk.Label(up_bottom_frame, image = icon_up_bottom_frame,borderwidth=0, highlightthickness = 0 )
      label_icon_up_bottom_frame.grid(row=0, column=0, padx=(5,0), sticky='e')   
      
      label_text_ubf = tk.Label(up_bottom_frame, text= 'Captured', borderwidth=0, highlightthickness = 0, bg='white')
      label_text_ubf.grid(row=0, column=1, sticky='w')      
      
      Image_up_bottom_frame= ImageTk.PhotoImage(Image.open(parentDir + "/images/face_bottom_icon.jpg")) 
      faceImage_up_bottom_frame = tk.Label(up_bottom_frame, image=Image_up_bottom_frame, borderwidth=0, highlightthickness = 0)
      faceImage_up_bottom_frame.grid(row=0,column=2, sticky='e')   
      numsFace = 10240
      label_face_ubf = tk.Label(up_bottom_frame, text= 'Face: {}'.format(numsFace), borderwidth=0, highlightthickness = 0, bg='white')
      label_face_ubf.grid(row=0, column=3, padx=5, sticky='w')   
      
      # create the widgets for the down_bottom_frame 
      lady_face = ImageTk.PhotoImage(Image.open(parentDir + "/images/lady_face.jpg").resize((100, 100), Image.ANTIALIAS)) 
      
      face_time_list = []
      face_img_list = []
      face_img_list.append(tk.Label(down_bottom_frame, borderwidth=0, highlightthickness = 0))
      face_img_list[0].grid(row=0, column=0, sticky='w', padx=(15,0), pady=(10,0)) 
      
      face_img_list.append(tk.Label(down_bottom_frame, borderwidth=0, highlightthickness = 0, bg='white'))
      face_img_list[1].grid(row=0, column=1, sticky='w', padx=(10,0), pady=(10,0)) 
      
      face_img_list.append(tk.Label(down_bottom_frame, borderwidth=0, highlightthickness = 0, bg='white'))
      face_img_list[2].grid(row=0, column=2, sticky='w', padx=(10,0), pady=(10,0))     
      
      face_img_list.append(tk.Label(down_bottom_frame, borderwidth=0, highlightthickness = 0, bg='white'))
      face_img_list[3].grid(row=0, column=3, sticky='w', padx=(10,0), pady=(10,0))  
      
      face_img_list.append(tk.Label(down_bottom_frame, borderwidth=0, highlightthickness = 0, bg='white'))
      face_img_list[4].grid(row=0, column=4, sticky='w', padx=(10,0), pady=(10,0))     
      
      face_time_list.append(tk.Label(down_bottom_frame, borderwidth=0, width=15, highlightthickness = 0, bg='white'))
      face_time_list[0].grid(row=1, column=0, sticky='w', padx=(10,0), pady=(0,0))  
      
      face_time_list.append(tk.Label(down_bottom_frame, borderwidth=0, width=15, highlightthickness = 0, bg='white'))
      face_time_list[1].grid(row=1, column=1, sticky='w', padx=(10,0), pady=(0,0))  
      
      face_time_list.append(tk.Label(down_bottom_frame, borderwidth=0, width=15, highlightthickness = 0, bg='white'))
      face_time_list[2].grid(row=1, column=2, sticky='w', padx=(10,0), pady=(0,0))  
      
      face_time_list.append(tk.Label(down_bottom_frame, borderwidth=0, width=15, highlightthickness = 0, bg='white'))
      face_time_list[3].grid(row=1, column=3, sticky='w', padx=(10,0), pady=(0,0))  
      
      face_time_list.append(tk.Label(down_bottom_frame, borderwidth=0, width=15, highlightthickness = 0, bg='white'))
      face_time_list[4].grid(row=1, column=4, sticky='w', padx=(10,0), pady=(0,0))  
      
      face_img_list.append(tk.Label(down_bottom_frame, borderwidth=0, highlightthickness = 0, bg='white'))
      face_img_list[5].grid(row=2, column=0, sticky='w', padx=(15,0), pady=(5,0)) 
      
      face_img_list.append(tk.Label(down_bottom_frame, borderwidth=0, highlightthickness = 0, bg='white'))
      face_img_list[6].grid(row=2, column=1, sticky='w', padx=(10,0), pady=(5,0)) 
      
      face_img_list.append(tk.Label(down_bottom_frame, borderwidth=0, highlightthickness = 0, bg='white'))
      face_img_list[7].grid(row=2, column=2, sticky='w', padx=(10,0), pady=(5,0))     
      
      face_img_list.append(tk.Label(down_bottom_frame, borderwidth=0, highlightthickness = 0, bg='white'))
      face_img_list[8].grid(row=2, column=3, sticky='w', padx=(10,0), pady=(5,0))  
      
      face_img_list.append(tk.Label(down_bottom_frame, borderwidth=0, highlightthickness = 0, bg='white'))
      face_img_list[9].grid(row=2, column=4, sticky='w', padx=(10,0), pady=(5,0))     
      
      face_time_list.append(tk.Label(down_bottom_frame, borderwidth=0, width=15, highlightthickness = 0, bg='white'))
      face_time_list[5].grid(row=3, column=0, sticky='w', padx=(10,0), pady=(0,0))  
      
      face_time_list.append(tk.Label(down_bottom_frame, borderwidth=0, width=15, highlightthickness = 0, bg='white'))
      face_time_list[6].grid(row=3, column=1, sticky='w', padx=(10,0), pady=(0,0))  
      
      face_time_list.append(tk.Label(down_bottom_frame, borderwidth=0, width=15, highlightthickness = 0, bg='white'))
      face_time_list[7].grid(row=3, column=2, sticky='w', padx=(10,0), pady=(0,0))  
      
      face_time_list.append(tk.Label(down_bottom_frame, borderwidth=0, width=15, highlightthickness = 0, bg='white'))
      face_time_list[8].grid(row=3, column=3, sticky='w', padx=(10,0), pady=(0,0))  
      
      face_time_list.append(tk.Label(down_bottom_frame, borderwidth=0, width=15, highlightthickness = 0, bg='white'))
      face_time_list[9].grid(row=3, column=4, sticky='w', padx=(10,0), pady=(0,0))      
      # create a popup menu
      popup = Menu(window, tearoff=0) 
      popup.add_command(label="Add Face", command=self.open_add_face_ui)
      popup.add_separator()
      popup.add_command(label="Quit", command=lambda: self.close_window(window))
      self.load_model()
      self.set_cam() 
      
      window.mainloop()
      
if __name__ == '__main__':
  app = FaceSDK()
     
   
    
    
