import cv2
import numpy as np
from keras.models import load_model
from statistics import mode
from Emotion.utils.datasets import get_labels
from Emotion.utils.inference import detect_faces
from Emotion.utils.inference import draw_text
from Emotion.utils.inference import draw_bounding_box
from Emotion.utils.inference import apply_offsets
from Emotion.utils.inference import load_detection_model
from Emotion.utils.preprocessor import preprocess_input

from SSRNET_model import SSR_net, SSR_net_general

USE_WEBCAM = False # If false, loads video file source

##############
def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=0.5, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.putText(image, label, point,font , font_scale, (255, 0, 0), thickness)
    

# parameters for loading data and images
emotion_model_path = './Emotion/models/emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_cascade = cv2.CascadeClassifier('./Emotion/models/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(emotion_model_path)

#(*)
weight_file = "pre-trained/morph2/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5"
weight_file_gender = "pre-trained/wiki_gender_models/ssrnet_3_3_3_64_1.0_1.0/ssrnet_3_3_3_64_1.0_1.0.h5"
#(*)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# starting video streaming

cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)

# load model and weights
img_size = 64
stage_num = [3,3,3]
lambda_local = 1
lambda_d = 1
model = SSR_net(img_size,stage_num, lambda_local, lambda_d)()
model.load_weights(weight_file)

model_gender = SSR_net_general(img_size,stage_num, lambda_local, lambda_d)()
model_gender.load_weights(weight_file_gender)

# Select video or webcam feed
cap = None
if (USE_WEBCAM == True):
    cap = cv2.VideoCapture(0) # Webcam source
else:
    cap = cv2.VideoCapture('./Emotion/demo/TGOP.mp4') # Video file source

while cap.isOpened(): # True:
    ret, bgr_image = cap.read()

    #bgr_image = video_capture.read()[1]

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5,
			minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        faces_2 = np.empty((1, img_size, img_size, 3))
        faces_2[0,:,:,:] = cv2.resize(rgb_image[y1:y2 + 1, x1:x2 + 1, :], (img_size, img_size))
        faces_2[0,:,:,:] = cv2.normalize(faces_2[0,:,:,:], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        predicted_ages = model.predict(faces_2)
        predicted_genders = model_gender.predict(faces_2)
        print(str(predicted_ages))
        

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)

        gender_str = 'male'
        if predicted_genders[0]<0.5:
            gender_str = 'female'
        label = "{},{}".format(int(predicted_ages[0]),gender_str)
        draw_label(rgb_image, (x1 + 20, y1 + 25), label)
      
        cv2.putText(rgb_image, emotion_text, (x1 + 20, y1 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
