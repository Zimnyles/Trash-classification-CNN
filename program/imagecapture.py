import cv2
import os
from tkinter import * 
from PIL import Image, ImageTk
import uuid

def take_picture():
    cap = cv2.VideoCapture(0)
    ret,frame = cap.read()
    if ret:
        cap.release()
        save_path="image-to-test/screen.jpg"
        cv2.imwrite(save_path, frame)

def delete_previous():
    if os.path.isfile('image-to-test\screen.jpg'):
        os.remove('image-to-test\screen.jpg')
    else: pass
         
def newbutton1():
    start_classification_button = Button(frame, text = 'Начать классификацию', bg = '#bbbbbb' )
    start_classification_button.place(x=600, y=200)


    
root = Tk()
root.title('Trash-classification')
root.geometry('1280x720')
root.resizable(width=False, height=False)

canvas = Canvas(root,width=1280,height=720 )
canvas.pack()
frame = Frame(root, bg='#ffffff')
frame.place(relheight=1,relwidth=1)

take_picture_text = Label(frame, text = 'Нажмите, чтобы сделать снимок', bg='#ffffff', font = 40)
take_picture_text.place()
take_picture_button = Button(frame, text = 'жми', bg = '#bbbbbb', command=lambda:[delete_previous(),take_picture(),newbutton1()] )
take_picture_button.place(x=600, y=50)

button = Button()













root.mainloop()