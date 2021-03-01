import tensorflow as tf
import numpy as np
from PIL import Image
import os
import shutil
import tensorflow_hub as hub

# load model
new_model = tf.keras.models.load_model('model/g-inexai.h5', custom_objects={'KerasLayer':hub.KerasLayer})

# load 200 models per class
class_max = 200

path = '../../InternalExternalAI/data2/train'

# path to new sorted class_folders
path_to_bath = 'home/Desktop/InternalExternalAI/data3/train/bathroom/'
path_to_bed = 'home/Desktop/InternalExternalAI/data3/train/bedroom/'
path_to_cons = 'home/Desktop/InternalExternalAI/data3/train/conservatory/'
path_to_din = 'home/Desktop/InternalExternalAI/data3/train/dining_room/'
path_to_entr = 'home/Desktop/InternalExternalAI/data3/train/entrance_hall_landing/'
path_to_foh = 'home/Desktop/InternalExternalAI/data3/train/front_of_house/'
path_to_gard = 'home/Desktop/InternalExternalAI/data3/train/garden/'
path_to_sink = 'home/Desktop/InternalExternalAI/data3/train/kitchen/'
path_to_live = 'home/Desktop/InternalExternalAI/data3/train/living_room/'
path_to_pool = 'home/Desktop/InternalExternalAI/data3/train/pool/'
street_path = 'home/Desktop/InternalExternalAI/data3/train/street_scape/'
path_to_stud = 'home/Desktop/InternalExternalAI/data3/train/study_office/'
path_to_util = 'home/Desktop/InternalExternalAI/data3/train/utility_room/'

# load images
def load_and_read(path, resize_h=224, resize_w=224):
    class_folders = [f for f in sorted(os.listdir(path)) if not f.startswith('.')]
    images = []
    classes = []
    dataset = []
    image_name = []

    for image, class_ in enumerate(class_folders):
        images_per_class = [f for f in sorted(os.listdir(os.path.join(path, class_))) if f.endswith('jpg')]
        # limit number of images to class_max
        if len(images_per_class) > class_max:
            images_per_class = images_per_class[0:class_max]
        #image_class = np.zeros(len(class_folders))
        #image_class[image] = 1

        for image_i, image_per_class in enumerate(images_per_class):
            images.append(os.path.join(path, class_, image_per_class))
            classes.append(image)
            image_name.append([class_, image_per_class])

    for image in images:
            # open image
            data = Image.open(image)
            # convert into numpy array
            data = np.array(data)
            # resize image
            img = tf.image.resize(data, (resize_h, resize_w))
            # cast image to tf.float32                  ## check why this step is important
            img = tf.cast(img, tf.float32)
            # normalise the image
            img = (img / 255.0)
            dataset.append(img)

    return dataset, image_name

class_names = ['bathroom','bedroom','conservatory','dining_room','entrance_hall_landing','front_of_house','garden','kitchen','living_room','pool', 'street_scape', 'study_office','utility_room']
file_path = [path_to_bath, path_to_bed, path_to_cons, path_to_din, path_to_entr, path_to_foh, path_to_gard, path_to_sink, path_to_live, path_to_pool, street_path, path_to_stud, path_to_util]

def prediction_and_sort(dataset, image_name):
    max_file_in_folder = 500

    for index, image in enumerate(dataset):
        x = np.array(image)
        x = np.expand_dims(x, axis=0)
        predictions = new_model.predict(x)
        # convert array into binary with 1 as max
        prediction = (predictions.argmax(1)[:,None] == np.arange(predictions.shape[1])).astype(int)
        #prediction = tf.round(new_model.predict(x)[0][0])
        print(prediction, image_name[index])
        for i in enumerate(prediction):
            if prediction[i][1] == 1:
                shutil.move(os.path.join(path, image_name[index][0], image_name[index][1]), os.path.join(file_path[i], image_name[index][1]))

        """if prediction == 1 and internal_num < max_file_in_folder:
            shutil.move(os.path.join(path, image_name[index][0], image_name[index][1]) , os.path.join(path_to_internal, image_name[index][1]))
            internal_num += 1
        elif prediction == 0 and external_num < max_file_in_folder:
            shutil.move(os.path.join(path, image_name[index][0], image_name[index][1]) , os.path.join(path_to_external, image_name[index][1]))
            external_num += 1"""

dataset, image_name = load_and_read(path)
prediction_and_sort(dataset, image_name)
