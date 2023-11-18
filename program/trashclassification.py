import numpy as np
import pathlib
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential
import cv2
import os
from tkinter import * 
from PIL import ImageTk,Image


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
    start_classification_button = Button(frame, text = 'Начать классификацию', bg = '#bbbbbb', command=lambda:[start_classification()] )
    start_classification_button.place(x=750, y=350, height=100, width=300)

         
def start_classification():
    dataset_dir = pathlib.Path('garbage_classification_dataset')
    batch_size = 32
    img_width = 180
    img_height = 180

    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split= 0.2,
        subset = 'training',
        seed = 123,
        image_size = (img_height, img_width),
        batch_size = batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split= 0.2,
        subset = 'validation',
        seed = 123,
        image_size = (img_height, img_width),
        batch_size = batch_size
    )

    class_names = train_ds.class_names
    print(f'class names: {class_names}')

    #cache
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    #create model
    num_classes = len(class_names)
    model = Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width,3)),
        
        tf.keras.layers.RandomFlip('horizontal', input_shape=(img_height, img_width, 3)),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.2),    
        

        layers.Conv2D(16,3,padding='same',activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(32,3,padding='same',activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(64,3,padding='same',activation='relu'),
        layers.MaxPooling2D(),
        
        #регуляризация
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])


    #compile model
    model.compile(
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics = ['accuracy']
    )


    model.build(input_shape = ())
    model.load_weights('weights\garbage-sorter.index')

    loss, acc = model.evaluate(train_ds, verbose=2)
    print('Restored model, acc: {:5.2f}%'.format(100*acc))

    img = tf.keras.utils.load_img("image-to-test\screen.png", target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array,0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    endtext_indef = 'Изображено {}({:.2f}% вероятность)'.format(class_names[np.argmax(score)],100*np.max(score))

    endtext_label = Label(frame, text = endtext_indef, bg = '#bbbbbb')
    endtext_label.place(x=750, y=500, height=100, width=300)
    
    



    

    
root = Tk()
root.title('Trash-classification')
root.geometry('1280x720')
root.resizable(width=False, height=False)
canvas = Canvas(root,width=1280,height=720 )
canvas.pack()

bg = PhotoImage(file='other_images/bgimgv100.png')


frame = Frame(root, bg = 'black')
frame.place(relheight=1,relwidth=1)

bglabel = Label(frame, image = bg)
bglabel.place(relheight=1,relwidth=1)

take_picture_text = Label(frame, text = 'Нажмите, чтобы сделать снимок', bg='#ffffff', font = 40)
take_picture_text.place()
take_picture_button = Button(frame, text = 'Нажмите, чтобы сделать снимок', bg = '#bbbbbb', command=lambda:[delete_previous(),take_picture(),newbutton1()] )
take_picture_button.place(x=750, y=200, height=100, width=300)

button = Button()
root.mainloop()