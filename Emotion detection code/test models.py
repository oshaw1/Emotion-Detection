import os
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import adam_v2
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

#specify the path to the train/test folders
trainData = '\\Emotion detection code\\train/'
testData = '\\Emotion detection code\\test/'
#set image pixels to value of 1 or 0
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range = 10,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode = 'nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

#set image size/color/class for training and validation
train_generator = train_datagen.flow_from_directory(
        trainData,
        #images in FER-2013 dataset are grayscale and 48x48
        target_size=(48,48),
        color_mode="grayscale",
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        testData,
        target_size=(48,48),
        color_mode="grayscale",
        class_mode='categorical')

model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Dropout(0.3))
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7, activation='softmax'))

if os.path.exists("SeqTEST.h5"):
    print("Found existing model.")
    model = load_model('SeqTEST.h5')  # Load the existing model
else:
    print("No existing model found. Training a new one.")
    model = Sequential()

    # Add layers to the model (same as your original code)

    model.compile(loss='categorical_crossentropy', optimizer=adam_v2.Adam(learning_rate=0.0001, decay=1e-7), metrics=['accuracy'])
    model.fit(
        train_generator,
        steps_per_epoch=28709 // 128,
        epochs=2,
        validation_data=test_generator,
        validation_steps=7178 // 64)

    model.save('SeqTEST.h5')  # Save the trained model

acc = model.history['accuracy']
test_acc = model.history['val_accuracy']

loss = model.history['loss']
test_loss = model.history['val_loss']

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
plt.savefig('bestModelGraph1.png')
plt.show()