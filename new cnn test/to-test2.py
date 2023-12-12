import matplotlib.pyplot as plt
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator

epoche = []
accuracy = []
#Путь к папке с фото
src_dir = 'image-to-test'
# Указываем разрешение для изображений к единому формату
image_width, image_height = 128, 128
# Задаём параметры аугментации
test_data = ImageDataGenerator(rescale=1. / 255)

#Предобработка тестовой выборки
test_processing= test_data.flow_from_directory(
    src_dir,
    target_size=(image_width, image_height),
    batch_size=32,
    class_mode='categorical')


path = 'neron_alt'
pattern = keras.models.load_model(path)
# Model evaluation
scores = pattern.evaluate(test_processing, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
accuracy.append(scores[1]*100)
epoche.append(20)
plt.plot(epoche, accuracy) 