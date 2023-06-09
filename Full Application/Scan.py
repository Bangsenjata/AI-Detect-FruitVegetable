import pyautogui
import PySimpleGUI as sg
import cv2
import numpy as np
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from pygrabber.dshow_graph import FilterGraph

def getcam():
    devices = FilterGraph().get_input_devices()
    available_cameras = {}
    for device_index, device_name in enumerate(devices):
        available_cameras[device_index] = device_name

    return available_cameras

sg.theme('LightBrown2')

camera_list = getcam()
camlen = len(camera_list)
x = camlen-3
cn = 0
while x>0:
    temp = camera_list[cn]
    camera_list[cn] = camera_list[cn+1]
    camera_list[cn+1] = temp
    cn += 1
    x -= 1
#temp = camera_list[1]
#camera_list[1] = camera_list[0]
#camera_list[0] = temp
cameras = [f'{camera_list[i]}' for i in camera_list]

layout=[[sg.Text('SCANNER', size=(40,1), justification='center', font='Arial 20')],
        [sg.Combo(cameras, default_value = camera_list[0], size=(20, 1), key='camera')],
        [sg.Button('Switch Camera', size=(12,1), font='Roboto 10')],
        [sg.Image(filename='', key='image')],
        [sg.Button('Scan', size=(10,1), font='Roboto 14'),
         sg.Button('Exit', size=(10,1), font='Roboto 14')],
        [sg.Text('', size=(20,1), justification='left', font='Arial 12', key='hasil')]]

window=sg.Window('SCAN FRUIT AND VEGETABLES',layout,location=(800,200))

indeks_kamera = 0
cap=cv2.VideoCapture(indeks_kamera)
recording=True
outputdir='image'
count = 0
test_set = tf.keras.utils.image_dataset_from_directory(
            'image\\testing',
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
cnn = tf.keras.models.load_model('model\\trained_model_final.h5')

while True:
    event, values = window.read(timeout=20)
    if event =='Exit' or event==sg.WIN_CLOSED:
        break
    
    elif event == 'Switch Camera':
        selected_camera = values['camera']
        indeks_kamera = cameras.index(selected_camera)
        cap=cv2.VideoCapture(indeks_kamera)

    if recording:
        ret,frame=cap.read()
        imgbytes=cv2.imencode('.png', frame)[1].tobytes()
        window['image'].update(data=imgbytes)
        if event == 'Scan':
            count += 1
            window['image'].update(data=imgbytes)
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
            output_path = os.path.join(outputdir, 'scan{}.jpg'.format(count))
            cv2.imwrite(output_path, frame)
            image_path = 'image/scan{}.jpg'.format(count)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #Converting BGR to RGB
            image = tf.keras.preprocessing.image.load_img(image_path,target_size=(64,64))
            input_arr = tf.keras.preprocessing.image.img_to_array(image)
            input_arr = np.array([input_arr])  # Convert single image to a batch.
            predictions = cnn.predict(input_arr)
            result_index = np.argmax(predictions)
            hasil_prediksi="Anda membeli {}".format(test_set.class_names[result_index])
            window['hasil'].update(hasil_prediksi)
            if count == 10:
                count = 0
        