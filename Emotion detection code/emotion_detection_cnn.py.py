import tensorflow as tf 
import cv2 
import os
import matplotlib.pyplot as plt 
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import adam_v2

# Set the relative path for the image file
img_path = '.\\train\\0\\Training_3908.jpg'
img_array = cv2.imread(img_path)

img_array.shape

plt.imshow(img_array)

# Set the relative paths for train and test data directories
trainData = '.\\train\\'
testData = '.\\test\\'

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
        break
    break

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range = 10,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode = 'nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

img_size = 224
new_array = cv2.resize(img_array, (img_size, img_size))
plt.imshow(cv2.cvtColor(new_array, cv2.COLOR_BGR2RGB))
plt.show()

train_generator = train_datagen.flow_from_directory(
        trainData,
        #images in FER-2013 dataset are grayscale and 48x48
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

print (len (training_Data))
import random
random.shuffle(training_Data)

X = []
Y = list()
for features, label in training_Data:
    X.append(features)
    Y.append(label)
X = np.array(X).reshape( -1,img_size, img_size, 3) 
Y = np.array(Y)
# 3 is the channel for RGB


X =X/255.0

# Check if the model file exists before creating a new model
model_save_path = 'MobTest.h5'
if os.path.exists(model_save_path):
    # Load the existing model
    new_model = keras.models.load_model(model_save_path)
else:
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

    # Training the model
    modelInfo = new_model.fit(
        train_generator,
        steps_per_epoch=28709 // 256,
        epochs=35,
        validation_data=test_generator,
        validation_steps=7178 // 64)

    # Save the trained model
    new_model.save(model_save_path)

acc = modelInfo.history['accuracy']
test_acc = modelInfo.history['val_accuracy']

loss = modelInfo.history['loss']
test_loss = modelInfo.history['val_loss']

epochsR = range(len(acc))

plt.figure(figsize = (18, 6))
plt.subplot(1, 2, 1)
plt.plot(epochsR, acc, label = 'Training Accuracy')
plt.plot(epochsR, test_acc, label = 'Test Accuracy')
plt.legend(loc = 'lower right')
plt.title('Training and Test Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochsR, loss, label = 'Training Loss')
plt.plot(epochsR, test_loss, label = 'Test Loss')
plt.legend(loc = 'upper right')
plt.title('Training and Test Loss')
plt.savefig('bestModelGraph.png')
plt.show()
