# SET THIS FOLDER TO YOUR PROJECT FOLDER

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json

# TRAINING IMAGE PROCESSING
print("Load Training Set...")
training_set = tf.keras.utils.image_dataset_from_directory(
    'datasets\\training',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

print("Load Validation set...")
validation_set = tf.keras.utils.image_dataset_from_directory(
    'datasets\\validation',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)

print("Setting the parameters...")
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=[64,64,3]))
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Dropout(0.25))
cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=64,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
cnn.add(tf.keras.layers.Dropout(0.25))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=512,activation='relu'))
cnn.add(tf.keras.layers.Dropout(0.5)) #To avoid overfitting
cnn.add(tf.keras.layers.Dense(units=10,activation='softmax')) #Output Layer

# Compiling and Training Phase
print("Compiling...")
cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
cnn.summary()
print("Start to train...")
training_history = cnn.fit(x=training_set,validation_data=validation_set,epochs=32)

#Training set Accuracy
print("Evaluating Train Accuracy...")
train_loss, train_acc = cnn.evaluate(training_set)
print('Training accuracy:', train_acc)

#Validation set Accuracy
print("Evaluating Validation Accuracy...")
val_loss, val_acc = cnn.evaluate(validation_set)
print('Validation accuracy:', val_acc)

#Saving Training Model
print("Saving Trained Model...")
cnn.save('new_trained_model.h5')
print("Trained Model Saved!")

training_history.history #Return Dictionary of history

#Recording History in json
print("Recording Train History to json...")
with open('training_hist.json','w') as f:
    json.dump(training_history.history,f)
print(training_history.history.keys())

#Calculating Accuracy of Model Achieved on Validation set
print("Calculating Accuracy of Model Achieved on Validation Set...")
print("Validation set Accuracy: {} %".format(training_history.history['val_accuracy'][-1]*100))

#Training Visualization
#training_history.history['accuracy']
print("Visualizing Training Accuracy...")
epochs = [i for i in range(1,33)]
plt.plot(epochs,training_history.history['accuracy'],color='red')
plt.xlabel('No. of Epochs')
plt.ylabel('Traiining Accuracy')
plt.title('Visualization of Training Accuracy Result')
plt.show()

#Validation Accuracy Visualization
print("Visualizing Validation Accuracy...")
plt.plot(epochs,training_history.history['val_accuracy'],color='blue')
plt.xlabel('No. of Epochs')
plt.ylabel('Validation Accuracy')
plt.title('Visualization of Validation Accuracy Result')
plt.show()

#Test Set Evaluation
print("Load Test Set...")
test_set = tf.keras.utils.image_dataset_from_directory(
    'datasets\\testing',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(64, 64),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False
)
print("Calculating Test Accuracy...")
test_loss,test_acc = cnn.evaluate(test_set)
print('Test accuracy:', test_acc)