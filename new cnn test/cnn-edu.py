from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import layers
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D, SeparableConv2D  
from keras import backend as K
from keras.layers import BatchNormalization
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
import tensorflow as tf
from keras.optimizers import gradient_descent_v2
from keras.callbacks import ModelCheckpoint
import os 
from keras.utils import plot_model
from keras.utils.vis_utils import plot_model    
import keras.utils.vis_utils
from importlib import reload


# sgd = gradient_descent_v2.SGD(...)

# Указываем разрешение для изображений к единому формату
image_width, image_height = 128, 128
# Указываем путь к обучающей выборке train_data_dir 
directory_data_train= 'garbage_classification_dataset' 
# Указываем путь к проверочной выборке validation_data_dir
directory_data_validation= 'image-to-test/validation_ds'  
# Сразу устанавливаем необходимые параметры
train_sample = 9272 
validation_sample = 5647 
# Количество эпох
epochs = 20
# batch_size 
lot_size = 32
# Число классификаций (число папок в directory_data_train)
cls_size = 5

if K.image_data_format() != 'channels_first':
    input_shape = (image_width, image_height, 3)
else:
    input_shape = (3, image_width, image_height)
    

image_size = (image_width, image_height)
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    activation = "softmax"
    units = num_classes
       

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


pattern = make_model(input_shape=image_size + (3,), num_classes=2)
# keras.utils.plot_model(pattern, show_shapes=True)

pattern.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

train_datagen = ImageDataGenerator(
    rescale=1. / 255, # коэффициент масштабирования
    shear_range=0.2, # Интенсивность сдвига
    zoom_range=0.2, # Диапазон случайного увеличения
    horizontal_flip=True) # Произвольный поворот по горизонтали
test_datagen = ImageDataGenerator(rescale=1. / 255)

#Предобработка обучающей выборки
train_processing = train_datagen.flow_from_directory(
    directory_data_train,
    target_size=(image_width, image_height), # Размер изображений
    batch_size=lot_size, #Размер пакетов данных
    class_mode='categorical') # Режим класса

#Предобработка тестовой выборки
validation_processing= test_datagen.flow_from_directory(
    directory_data_validation,
    target_size=(image_width, image_height),
    batch_size=lot_size,
    class_mode='categorical')

filepath="first_model_weights.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Загрузка модели
if  os.path.isdir('neron_4'):
    pattern = keras.models.load_model('neron_4')
    
history = pattern.fit(
    train_processing, # Помещаем обучающую выборку
    steps_per_epoch=train_sample // lot_size, #количество итераций пакета до того, как период обучения считается завершенным
    epochs=epochs, # Указываем количество эпох
    validation_data = validation_processing, # Помещаем проверочную выборку
    validation_steps=validation_sample  // lot_size,  # Количество итерации, но на проверочном пакете данных
callbacks= callbacks_list) #Автосейвы

# if 'loss_values' in locals():
#     loss_values += history.history['loss']
#     epochs = range(1, len(loss_values)+1)
# else:
#     loss_values = history.history['loss']
#     epochs = range(1, len(loss_values)+1)

# plt.plot(epochs, loss_values, label='Training Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# plt.show()

pattern.save_weights('first_model_weights.h5') #Сохранение весов модели
pattern.save('neron_4') #Сохранение модели

pattern = keras.models.load_model('neron_4')
def do_predict(model, image_file):
    label_names = ["cat", "other"]
    img = Image.open(image_file)
    # изменяем размер
    img = img.resize((image_width, image_height))
    img_arr = np.expand_dims(img, axis=0) / 255.0
    plt.imshow(img)
    t_start = time.time()
    result = model.predict(img_arr)
    print(result)
    print("Time:", time.time() - t_start)
    print("Result: %s" % label_names[result.argmax(axis=1)[0]])
    
do_predict(pattern, "image-to-test/validation_ds/paper/cardboard526.jpg")
do_predict(pattern, "image-to-test/validation_ds/paper/cardboard527.jpg")
do_predict(pattern, "image-to-test/validation_ds/paper/cardboard528.jpg")
do_predict(pattern, "image-to-test/validation_ds/paper/cardboard529.jpg")

