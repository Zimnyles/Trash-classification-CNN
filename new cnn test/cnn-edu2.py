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
from keras.optimizers import SGD, Adam

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
epochs = 25
# batch_size 
lot_size = 32
# Число классификаций (число папок в directory_data_train)
cls_size = 5





# Создание модели
pattern = Sequential() 

# Первый слой нейросети
pattern.add(Conv2D(32, (3, 3), input_shape=(image_width, image_height, 3), padding="same"))
pattern.add(Activation("relu"))
pattern.add(BatchNormalization(axis=1))
pattern.add(Activation("relu"))
pattern.add(Conv2D(32, (3, 3), padding="same"))
pattern.add(Activation("relu"))
pattern.add(BatchNormalization(axis=1))
pattern.add(MaxPooling2D(pool_size=(2, 2)))
pattern.add(Activation("relu"))

# Второй слой нейросети
pattern.add(Conv2D(64, (3, 3), padding="same"))
pattern.add(Activation("relu"))
pattern.add(BatchNormalization(axis=1))
pattern.add(Activation("relu"))
pattern.add(Conv2D(64, (3, 3), padding="same"))
pattern.add(Activation("relu"))
pattern.add(BatchNormalization(axis=1))
pattern.add(Activation("relu"))
pattern.add(MaxPooling2D(pool_size=(2, 2)))
pattern.add(Activation("relu"))
pattern.add(Dropout(0.25))

# Третий слой нейросети
pattern.add(Conv2D(256, (3, 3), padding="same"))
pattern.add(Activation("relu"))
pattern.add(BatchNormalization(axis=1))
pattern.add(Activation("relu"))
pattern.add(Conv2D(256, (3, 3), padding="same"))
pattern.add(Activation("relu"))
pattern.add(BatchNormalization(axis=1))
pattern.add(Activation("relu"))
pattern.add(MaxPooling2D(pool_size=(2, 2)))
pattern.add(Activation("relu"))

# Третий слой нейросети
pattern.add(Conv2D(512, (3, 3), padding="same"))
pattern.add(Activation("relu"))
pattern.add(BatchNormalization(axis=1))
pattern.add(Activation("relu"))
pattern.add(Conv2D(512, (3, 3), padding="same"))
pattern.add(Activation("relu"))
pattern.add(BatchNormalization(axis=1))
pattern.add(Activation("relu"))
pattern.add(MaxPooling2D(pool_size=(2, 2)))
pattern.add(Activation("relu"))

# Пятый слой нейросети
pattern.add(SeparableConv2D(1024, (3, 3), padding="same"))
pattern.add(Activation("relu"))
pattern.add(BatchNormalization(axis=1))
pattern.add(Activation("relu"))
pattern.add(GlobalAveragePooling2D())
pattern.add(Activation("relu"))
pattern.add(BatchNormalization(axis=1))
pattern.add(Activation("relu"))

#Aктивация, свертка, объединение, исключение
pattern.add(Dense(1024))
pattern.add(Activation("relu"))
pattern.add(BatchNormalization(axis=1))
pattern.add(Dropout(0.15))
pattern.add(Activation("relu"))
pattern.add(Dense(512))
pattern.add(Activation("relu"))
pattern.add(BatchNormalization(axis=1))
pattern.add(Dense(cls_size))# число классов
pattern.add(Activation('softmax'))

# инициализируем скорость обучения
#INIT_LR = 0.01
#opt = SGD(learning_rate=INIT_LR)
#Cкомпилируем модель с выбранными параметрами. Также укажем метрику для оценки.
pattern.compile(loss='categorical_crossentropy',
             optimizer=Adam(),
              metrics=['accuracy'])


print(pattern.summary())


# from keras.utils.vis_utils import plot_model
# plot_model(pattern, to_file='model_plot_alt.png', show_shapes=True, show_layer_names=True)



# Задаём параметры аугментации
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


filepath="first_model_weights_v1.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]


# Загрузка модели
if  os.path.isdir('neron_alt'):
    pattern = keras.models.load_model('neron_alt')
    
    
    
    
history = pattern.fit(
    train_processing, # Помещаем обучающую выборку
    steps_per_epoch=train_sample // lot_size, #количество итераций пакета до того, как период обучения считается завершенным
    epochs=epochs, # Указываем количество эпох
    validation_data = validation_processing, # Помещаем проверочную выборку
    validation_steps=validation_sample  // lot_size,  # Количество итерации, но на проверочном пакете данных
callbacks= callbacks_list) #Автосейвы


pattern.save_weights('first_model_weights_v1.h5') #Сохранение весов модели
pattern.save('neron_alt') #Сохранение модели

# pattern = keras.models.load_model('neron_alt')
# def do_predict(model, image_file):
#     label_names = ["biological", "glass", "metal", "paper", "plastic"]
#     img = Image.open(image_file)
#     # изменяем размер
#     img = img.resize((image_width, image_height))
#     img_arr = np.expand_dims(img, axis=0) / 255.0
#     plt.imshow(img)
#     t_start = time.time()
#     result = model.predict(img_arr)
#     print(result)
#     print("Time:", time.time() - t_start)
#     print("Result: %s" % label_names[result.argmax(axis=1)[0]])
    
# do_predict(pattern, "image-to-test/validation_ds/paper/cardboard526.jpg")
# do_predict(pattern, "image-to-test/validation_ds/paper/cardboard527.jpg")
# do_predict(pattern, "image-to-test/validation_ds/paper/cardboard528.jpg")
# do_predict(pattern, "image-to-test/validation_ds/paper/cardboard529.jpg")
