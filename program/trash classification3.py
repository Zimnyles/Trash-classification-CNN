import numpy as np
import pathlib
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential
import cv2
import os
from tkinter import * 
from PIL import ImageTk,Image
import tensorflow as tf
import os 
from keras.utils import plot_model
from keras.utils.vis_utils import plot_model    
from importlib import reload
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
from tkinter import ttk
from tkinter import font


def take_picture():
    cap = cv2.VideoCapture(0)
    ret,frame = cap.read()
    if ret:
        cap.release()
        save_path="image-to-test/screen.png"
        cv2.imwrite(save_path, frame)


def delete_previous():
    if os.path.isfile('image-to-test\screen.jpg'):
        os.remove('image-to-test\screen.jpg')
    else: pass

    
def newbutton1():
    start_classification_button = Button(frame, text = 'Начать классификацию',font='Arial 13', fg='#ffffff', bg = '#232323', command=lambda:[start_classification()] )
    start_classification_button.place(x=750, y=350, height=100, width=300)

         
def start_classification():
    image_width, image_height = 128, 128
    pattern = keras.models.load_model('neron_alt')
    model = pattern
    image_file = 'image-to-test/screen.png'
    label_names = ["biological", "glass", "metal", "paper", "plastic"]
    img = Image.open(image_file)
    # изменяем размер
    img = img.resize((image_width, image_height))
    img_arr = np.expand_dims(img, axis=0) / 255.0
    plt.imshow(img)
    t_start = time.time()
    result = model.predict(img_arr)
    # print(result)
    # print("Time:", time.time() - t_start)
    # print("Result: %s" % label_names[result.argmax(axis=1)[0]])
    endtext_indef = ("Result: %s" % label_names[result.argmax(axis=1)[0]])

    endtext_label = Label(frame, text = endtext_indef , bg = '#bbbbbb',font='Arial 13', fg='#ffffff')
    endtext_label.place(x=750, y=500, height=100, width=300)
    



root = Tk()
root.title('Trash-classification')
root.geometry('1280x720')
root.resizable(width=False, height=False)
canvas = Canvas(root,width=1280,height=720 )
canvas.pack()

bg_for_frame = PhotoImage(file='other_images/bgimgv110.png')


frame = Frame(root, bg = 'black')
frame.place(relheight=1,relwidth=1)

bglabel = Label(frame, image = bg_for_frame)
bglabel.place(relheight=1,relwidth=1)



take_picture_text = Label(frame, bg='#232323', font = 40)
take_picture_text.place()
take_picture_button = Button(frame, text = 'Нажмите, чтобы сделать снимок',font='Arial 13', fg='#ffffff', bg = '#232323', command=lambda:[delete_previous(),take_picture(),newbutton1()] )
take_picture_button.place(x=750, y=200, height=100, width=300)

root.mainloop()