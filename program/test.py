import cv2
import os
from tkinter import * 
from PIL import Image, ImageTk
import uuid


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
take_picture_button = Button(frame, text = 'жми', bg = '#bbbbbb')
take_picture_button.place(x=600, y=50)

button = Button()
