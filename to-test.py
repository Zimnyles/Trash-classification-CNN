import copy
import itertools
import json
import warnings
import weakref
import numpy as np
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_ops
from tensorflow.python.eager import context
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
from keras import backend
from keras import callbacks as callbacks_module
from keras import optimizers
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import compile_utils
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine import input_layer as input_layer_module
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.utils.mode_keys import ModeKeys
###

###
import pathlib
import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
from tensorflow.python import distribute
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.activations import linear
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.losses import MeanSquaredError

#module 'tensorflow.python.distribute.input_lib' has no attribute 'DistributedDatasetInterface'


dataset_dir = pathlib.Path('garbage_classification')
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

#load model
model.build(input_shape = ())
model.load_weights(r'C:\Users\HUAWEI\Desktop\Dataset train\weights\garbage-sorter.index')
#evaluate model
loss, acc = model.evaluate(train_ds, verbose=2)
print('Restored model, acc: {:5.2f}%'.format(100*acc))
#img upload
img = tf.keras.utils.load_img(r"C:\Users\HUAWEI\Desktop\Dataset train\image-to-test\metal-to-test.jpg", target_size=(img_height, img_width))
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array,0)
#make predictions
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
#resul print
print('Изображено {}({:.2f}% вероятность)'.format(
    class_names[np.argmax(score)],
    100*np.max(score)))