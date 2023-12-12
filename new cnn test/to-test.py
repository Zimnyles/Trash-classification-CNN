from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image

image_width, image_height = 128, 128

pattern = keras.models.load_model('neron_alt')
def do_predict(model, image_file):
    label_names = ["biological", "glass", "metal", "paper", "plastic"]
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
    
do_predict(pattern, "image-to-test/paper.jpg")

