import tensorflow as tf
import numpy as np
from PIL import Image
import os
import shutil

# load most recent model
new_model = tf.keras.models.load_model('model/best_model.h5')

# load 1000 more images from the source file
class_max = 1000

path = '/media/jambobjambo/AELaCie/Datasets/DCTR/intextAI/Source'

path_to_internal = '/media/jambobjambo/AELaCie/Datasets/DCTR/intextAI/Train/Internal'
path_to_external = '/media/jambobjambo/AELaCie/Datasets/DCTR/intextAI/Train/External'

# load images
def load_and_read(path, resize_h=224, resize_w=224):
    class_folders = [f for f in sorted(os.listdir(path)) if not f.startswith('.')]
    images = []
    classes = []
    dataset = []

    for image, class_ in enumerate(class_folders):
        images_per_class = [f for f in sorted(os.listdir(os.path.join(path, class_))) if f.endswith('jpg')]
        # limit number of images to class_max
        if len(images_per_class) > class_max:
            images_per_class = images_per_class[0:class_max]
        image_class = np.zeros(len(class_folders))
        image_class[image] = 1

        for image_i, image_per_class in enumerate(images_per_class):
            images.append(os.path.join(path, class_, image_per_class))
            classes.append(image_class)

    ## can i select them randomly between the two folders?
    ## is there much point including the classes during the loading process?

    for image in images:
        # open image
        data = Image.open(image)
        # convert into numpy array
        data = np.array(data)
        # resize image
        img = tf.image.resize(data, (resize_h, resize_w))
        # cast image to tf.float32                  ## check why this step is important
        img = tf.cast(img, tf.float32)
        # normalise the image
        img = (img / 255.0)
        dataset.append(img)

    return dataset

class_names = ['external', 'internal']

def prediction_and_sort(dataset):
    for image in dataset:
        x = np.array(image)
        x = np.expand_dims(x, axis=0)
        prediction = new_model.predict(x)[0]
        if prediction == 'internal':
            pass                # shutil.move(path, path_to_internal)
        elif prediction == 'external':
            pass                # shutil.move(path, path_to_external)



