import tensorflow as tf 
import cv2 
import os
import matplotlib.pyplot as plt 
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.optimizers import adam_v2
from keras.preprocessing.image import ImageDataGenerator

# Set the relative path for the image file
img_path = '.\\train\\0\\Training_3908.jpg'
img_array = cv2.imread(img_path)

img_array.shape

plt.imshow(img_array)

# Set the relative paths for train and test data directories
trainData = '.\\train\\'
testData = '.\\test\\'

# create a variable
Classes = ["0","1","2","3","4","5","6"]

for category in Classes:
    path = os.path.join(trainData, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))
        plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        plt.show()
        break
    break
for category in Classes:
    path = os.path.join(testData, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))
        plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        plt.show()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

img_size = 224
new_array = cv2.resize(img_array, (img_size, img_size))
plt.imshow(cv2.cvtColor(new_array, cv2.COLOR_BGR2RGB))
plt.show()

train_generator = train_datagen.flow_from_directory(
    trainData,
    target_size=(224,224),
    color_mode="rgb",
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    testData,
    target_size=(224,224),
    color_mode="rgb",
    class_mode='categorical')

new_array.shape

training_Data = []
def create_training_Data():
    for category in Classes:
        path = os.path.join(trainData, category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_Data.append([new_array,class_num])
            except Exception as e:
                pass
create_training_Data()

print(len(training_Data))
import random
random.shuffle(training_Data)

X = []
Y = list()
for features, label in training_Data:
    X.append(features)
    Y.append(label)
X = np.array(X).reshape( -1, img_size, img_size, 3) 
Y = np.array(Y)

X = X / 255.0

model = tf.keras.applications.MobileNetV2()  # CHANGE THIS TO TEST DIFFERENT ML MODELS

base_input = model.layers[0].input
base_output = model.layers[-2].output
final_output = layers.Dense(128)(base_output) 
final_output = layers.Activation('relu')(final_output) 
final_output = layers.Dense(64)(final_output)
final_output = layers.Activation('relu')(final_output)
final_output = layers.Dense(7, activation='softmax')(final_output) 

new_model = keras.Model(inputs=base_input, outputs=final_output)

new_model.compile(loss='categorical_crossentropy', 
                  optimizer=adam_v2.Adam(learning_rate=0.0001, decay=1e-7), metrics=['accuracy'])

modelInfo = new_model.fit(
    train_generator,
    steps_per_epoch=28709 // 64,
    epochs=35,
    validation_data=test_generator,
    validation_steps=7178 // 64)

new_model.save('MobTest.h5')


path = ".\\haarcascade_frontalface_default.xml"
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

# set the rectangle background to white
rectangle_bgr = (255, 255, 255)
# make a black image
img = np.zeros((500, 500))
# set some text
text = "Some text "
# get the width and height of the text box
(text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
# set the text start position
text_offset_x = 10
text_offset_y = img.shape[0] - 25
# make the coords of the box with a small padding of two pixels
box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)

faceHelp = cv2.CascadeClassifier('.\\haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(1)
# Check if the webcam is opened correctly
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    for x, y, w, h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        facess = faceCascade.detectMultiScale(roi_gray)
        if len(facess) == 0:
            print("Face not detected")
        else:
            for (ex,ey,ew,eh) in facess:
                face_roi = roi_color
                face_roi = roi_color[ey: ey+eh, ex:ex + ew] ## cropping the face
                final_image = cv2.resize(face_roi, (224,224))
                final_image = np.expand_dims(final_image, axis=0) ## need fourth dimension
                final_image = final_image/255.0

                font = cv2.FONT_HERSHEY_SIMPLEX

                Predictions = new_model.predict(final_image)

                font_scale = 1.5
                font = cv2.FONT_HERSHEY_PLAIN

                if np.argmax(Predictions) == 0:
                    status = "Angry"
                    cv2.putText(frame, status, (100,150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255))
                elif np.argmax(Predictions) == 1:
                    status = "Disgust"
                    cv2.putText(frame, status, (100,150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255))
                elif np.argmax(Predictions) == 2:
                    status = "Fear"
                    cv2.putText(frame, status, (100,150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255))
                elif np.argmax(Predictions) == 3:
                    status = "Happy"
                    cv2.putText(frame, status, (100,150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255))
                elif np.argmax(Predictions) == 4:
                    status = "Sad"
                    cv2.putText(frame, status, (100,150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255))
                elif np.argmax(Predictions) == 5:
                    status = "Surprise"
                    cv2.putText(frame, status, (100,150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255))
                elif np.argmax(Predictions) == 6:
                    status = "Neutral"
                    cv2.putText(frame, status, (100,150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255))

        cv2.imshow('Face Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
