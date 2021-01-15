import numpy as np
import os
import tensorflow as tf
import random
from PIL import Image

class_max = 800

def dataset_classifcation(path, resize_h, resize_w, train=True, limit=None):

    # list all paths to data classes except DS_Store
    class_folders = [f for f in sorted(os.listdir(path)) if not f.startswith('.')]
    # load images
    images = []
    classes = []

    train_test_split = 0.1
    number_of_test = int(class_max * train_test_split)
    for i, c in enumerate(class_folders):
        #images_per_class = sorted(os.path.join(path, c))
        images_per_class = [f for f in sorted(os.listdir(os.path.join(path, c))) if 'jpg' in f]
        # testing inbalanced class theory so limiting to 800 per class - can remove 20-21 later
        if len(images_per_class) > class_max:
            images_per_class = images_per_class[0:class_max]
        image_class = np.zeros(len(class_folders))
        image_class[i] = 1

        for image_i, image_per_class in enumerate(images_per_class):
            if train == False and image_i <= number_of_test:
                images.append(os.path.join(path, c, image_per_class))
                classes.append(image_class)
            elif train == True and image_i > number_of_test:
                images.append(os.path.join(path, c, image_per_class))
                classes.append(image_class)

    random.seed(10)
    shuffle_index = random.sample(list(range(len(images))), len(images))

    images_shuffle = [images[i] for i in shuffle_index]
    classes_shuffle = [classes[i] for i in shuffle_index]

    #print(classes_shuffle[100], images_shuffle[100])

    images_tf = tf.data.Dataset.from_tensor_slices(images)
    classes_tf = tf.data.Dataset.from_tensor_slices(classes)
    # put two arrays together so that each image has its classifying label
    dataset = tf.data.Dataset.zip((images_tf, classes_tf))

    @tf.function
    def read_images(image_path, class_type, mirrored=False):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image)

        h, w, c = image.shape
        if not (h == resize_h and w == resize_w):
            image = tf.image.resize(
            image, [resize_h, resize_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # set all images shape to RGB
            image.set_shape((224, 224, 3))
#             print(image.shape)

        '''if train == True:
            image = tf.image.random_brightness(image, 0.2)
            image = tf.image.random_contrast(image, 0.2, 0.5)
            #image = tf.image.random_jpeg_quality(image, 75, 100)'''

        # change DType of image to float32
        image = tf.cast(image, tf.float32)
        class_type = tf.cast(class_type, tf.float32)

        # normalise the image pixels
        image = (image / 255.0)

        return image, class_type

    dataset = dataset.map(
        read_images,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=False)

    return dataset, len(class_folders)

#path = '/Users/georgebrockman/code/georgebrockman/Autoenhance.ai/InternalExternalAI/images/training_data/'
path = '/media/jambobjambo/AELaCie/Datasets/DCTR/intextAI/Train'
train_dataset, num_classes = dataset_classifcation(path, 224, 224)
test_dataset, num_classes = dataset_classifcation(path, 224, 224, train=False)

IMG_WIDTH, IMG_HEIGHT = 224, 224
IMG_SIZE = IMG_WIDTH, IMG_HEIGHT
#batch_size = 32
batch_size = 64

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).batch(32).prefetch(buffer_size=AUTOTUNE)

test_dataset = test_dataset.cache().batch(32).prefetch(buffer_size=AUTOTUNE)

# import the base model
# instantiate the MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                             include_top=False,
                                             weights='imagenet')
top_layers = base_model.output
#base_model.summary()

image_batch, label_batch = next(iter(train_dataset))
# # global_av_layer experiment
# feature_batch = base_model(image_batch)

# freeze the convolutional base
base_model.trainable=False

fine_tune_at = 110

# freeze all the layers before the tuning - this can be done with a for loop and slicing
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# convert the features to a single 1280-element vector per image
#global_av_layer = tf.keras.layers.GlobalAveragePooling2D() # averages over a 5x5 spatial
#feature_batch_av = global_av_layer(feature_batch)
#print(feature_batch_av.shape)

# pred_layer_1 = tf.keras.layers.Dense(1024, activation = 'relu')
pred_layer = tf.keras.layers.Dense(num_classes, activation='softmax')
#pred_batch = pred_layer(feature_batch_av)
#pred_batch = pred_layer(pred_layer_1)
#pred_batch.shape

# data augmentation
data_aug = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomFlip('vertical'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.25)
    #tf.keras.layers.experimental.preprocessing.RandomZoom(.5, .2)
])

# rescale the pixel values to match the expected values of the MobileNetV2 model
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# # convert the features to a single 1280-element vector per image
# global_av_layer = tf.keras.layers.GlobalAveragePooling2D() # averages over a 5x5 spatial
# feature_batch_av = global_av_layer(feature_batch)

# # apply a dense layer to convert these features into a single prediction per image
# # no activation needed as the prediction will be treated as a logit (mapping of probabilities to Real Numbers)

# # changed to 2 dense layer for global_Av experiment so logits and labels shape matches
# pred_layer = tf.keras.layers.Dense(2)
# pred_batch = pred_layer(feature_batch_av)

# chain together data augmentation, rescaling, base_model and feature extractor layers useing the Keras Functional API

inputs = tf.keras.Input(shape=(224,224,3)) # image size and channels
# data augmentation layer
x = data_aug(inputs)
# preprocess, feed x into and reassign variable
x = preprocess_input(x)
# basemodel, set training =False for the BN layer
base_model = base_model(x, training=False)
# print(x.shape)
#Conv_layer = tf.keras.layers.Conv2D(32, 1)(x)
# commented out to test global_Av_layer in fine tuning tutoral

'''x = tf.keras.layers.Conv2D(64, 1)(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Conv2D(64, 1)(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Conv2D(64, 1)(x)
x = tf.keras.layers.Dropout(0.2)(x)
flatten = tf.keras.layers.Flatten()(x)
pred_layer_1 = tf.keras.layers.Dense(100, activation = 'relu')(flatten)'''

top_layers = tf.keras.layers.GlobalAveragePooling2D()(base_model)
top_layers = tf.keras.layers.Dense(1024, activation='relu')(top_layers)
predictions = tf.keras.layers.Dense(2, activation='softmax')(top_layers)
# feature extraction
# x = global_av_layer(x)
# # add a dropout layer
# x = tf.keras.layers.Dropout(0.2)(x)

# outputs = pred_layer(x)
# model = tf.keras.Model(inputs, outputs)


# commented out for global_av_layer experiment
#outputs = pred_layer(pred_layer_1)
#print(outputs.shape)
model = tf.keras.Model(inputs, predictions)

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25)

base_learning_rate = 1e-3
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate, decay=1e-4),
              # Only two linear outputs so use BinaryCrossentropy and logits =True
              loss='categorical_crossentropy',
              metrics=["accuracy"])
#tf.keras.metrics.BinaryAccuracy()
#checkpoint_path = "/Users/georgebrockman/code/georgebrockman/Autoenhance.ai/InternalExternalAI/checkpoints/cp.ckpt"
checkpoint_path = "./"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                 save_weights_only=True,
#                                                 verbose=1)
initial_epochs = 100
history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    callbacks=[es, tf.keras.callbacks.ModelCheckpoint(filepath='model/best_model.h5', monitor='val_loss', save_best_only=True)],
                    validation_data= test_dataset)
# save model
model.save('model/g-inexai.h5')