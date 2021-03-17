import numpy as np
import os
import tensorflow as tf
import random
from PIL import Image
from sklearn.model_selection import train_test_split

import tensorflow_hub as hub

class_max = 1000
batch_size = 64

module_selection = ("mobilenet_v3_large_100_224", 224)
handle_base, pixels = module_selection
MODULE_HANDLE ="https://tfhub.dev/google/imagenet/{}/feature_vector/5".format(handle_base)

external_classes = ['garden', 'front_of_house', 'street_scape']

def dataset_classifcation(path, resize_h, resize_w, train=True, limit=None):

    # list all paths to data classes except DS_Store
    class_folders = [f for f in sorted(os.listdir(path)) if not f.startswith('.')]
    # load images
    images = []
    classes = []

    train_test_split_num = 0.1
    number_of_test = int(class_max * train_test_split_num)
    for i, c in enumerate(class_folders):

        rd_array = np.zeros(13)
        inex_array = np.zeros(2)
        rd_array[i] = 1.

        if c in external_classes:
            inex_array[0] = 1.
        else:
            inex_array[1] = 1.
        #images_per_class = sorted(os.path.join(path, c))
        images_per_class = [f for f in sorted(os.listdir(os.path.join(path, c))) if 'jpg' in f]
        # testing inbalanced class theory so limiting to 800 per class - can remove 20-21 later
        if len(images_per_class) > class_max:
            images_per_class = images_per_class[0:class_max]
        #image_class = np.zeros(len(class_folders))
        #image_class[i] = 1

        for image_i, image_per_class in enumerate(images_per_class):
            images.append(os.path.join(path, c, image_per_class))
            # print(rd_array, inex_array)
            classes.append([rd_array, inex_array])

    train_filenames, val_filenames, train_labels, val_labels = train_test_split(images, classes, train_size=0.9, random_state=420)

    rd_labels_train = [tl[0] for tl in train_labels]
    ie_labels_train = [tl[1] for tl in train_labels]

    rd_labels_test = [tl[0] for tl in val_labels]
    ie_labels_test = [tl[1] for tl in val_labels]

    num_train = len(train_filenames)
    num_val = len(val_filenames)

    @tf.function
    def read_images(image_path, rd_type, ie_type, mirrored=False, train=False):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image)

        h, w, c = image.shape
        if not (h == resize_h and w == resize_w):
            image = tf.image.resize(
            image, [resize_h, resize_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # set all images shape to RGB
            image.set_shape((224, 224, 3))


        if train == True:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            image = tf.image.random_brightness(image, 0.2)
            image = tf.image.random_contrast(image, 0.2, 0.5)
            image = tf.image.random_jpeg_quality(image, 75, 100)

        # change DType of image to float32
        image = tf.cast(image, tf.float32)
        rd_type = tf.cast(rd_type, tf.float32)
        ie_type = tf.cast(ie_type, tf.float32)

        # normalise the image pixels
        image = (image / 255.0)

        return image, rd_type, ie_type


    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_data = tf.data.Dataset.from_tensor_slices((tf.constant(train_filenames), tf.constant(rd_labels_train), tf.constant(ie_labels_train))).map(lambda x,y,y2: read_images(x, y, y2, train=True)).shuffle(1000).batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    val_data = tf.data.Dataset.from_tensor_slices((tf.constant(val_filenames), tf.constant(rd_labels_test), tf.constant(ie_labels_test))).map(read_images).batch(batch_size).prefetch(buffer_size=AUTOTUNE)

    return train_data, val_data, num_train, len(class_folders)

#path = '/Users/georgebrockman/code/georgebrockman/Autoenhance.ai/InternalExternalAI/images/training_data/'
#path = '/media/jambobjambo/AELaCie/Datasets/DCTR/intextAI/Train'
#path = './data2/train'

path = '../../InternalExternalAI/data3/train'
train_data, val_data, num_train, num_classes = dataset_classifcation(path, 224, 224)

IMG_WIDTH, IMG_HEIGHT = 224, 224
IMG_SIZE = IMG_WIDTH, IMG_HEIGHT

IMG_SHAPE = IMG_SIZE + (3,)

data_aug = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.25),
    #tf.keras.layers.experimental.preprocessing.RandomZoom(.5, .2)
])

IMAGE_SIZE = (224, 224)
do_fine_tuning = True
input = tf.keras.layers.Input(shape=IMAGE_SIZE + (3,))
x = hub.KerasLayer(MODULE_HANDLE, trainable=do_fine_tuning)(input)
# rd = room detection
rd = tf.keras.layers.Dropout(rate=0.2)(x)
rd = tf.keras.layers.Dense(13,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                          activation='softmax',
                          name='rd_loss')(rd)
# ie = internal / external
ie = tf.keras.layers.Dropout(rate=0.2)(x)
ie = tf.keras.layers.Dense(2,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                          activation='softmax',
                          name='ie_loss')(ie)
model = tf.keras.Model(inputs=input, outputs=[rd, ie])
#model.build((None,)+IMAGE_SIZE+(3,))
model.summary()

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)

base_learning_rate = 0.0001 #1e-3, decay=1e-4
losses = {"ie_loss": tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
            "rd_loss": tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)}
loss_weights = {"ie_loss": 1., "rd_loss": 1.}
model_metrics = {"ie_loss": 'accuracy', "rd_loss": 'accuracy'}
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
              # Only two linear outputs so use BinaryCrossentropy and logits =True
              loss=losses,
              loss_weights = loss_weights,
              metrics=model_metrics)

checkpoint_path = "./"
checkpoint_dir = os.path.dirname(checkpoint_path)

initial_epochs = 200
steps_per_epoch = round(num_train)//batch_size
val_steps = 20

def generator(dataset):
    for input, output1, output2 in dataset.repeat():
        yield input, [output1, output2]

history = model.fit_generator(generator(train_data),
                    steps_per_epoch = steps_per_epoch,
                    epochs=initial_epochs,
                    callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath='model/best_model_inex.h5', monitor='val_loss', save_best_only=True)],
                    validation_data= generator(val_data),
                    validation_steps=val_steps)

'''history = model.fit(train_data.repeat(),
                    steps_per_epoch = steps_per_epoch,
                    epochs=initial_epochs,
                    callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath='model/best_model_inex.h5', monitor='val_loss', save_best_only=True)],
                    validation_data= val_data.repeat(),
                    validation_steps=val_steps)'''
# save model
model.save('model/g-internalexai.h5')

# print best model score.
best_score = max(history.history['accuracy'])
print(best_score)
