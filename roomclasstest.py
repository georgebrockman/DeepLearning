import tensorflow as tf
import numpy as np
from PIL import Image
import tensorflow_hub as hub

# load model
new_model = tf.keras.models.load_model('model/g-inexai.h5', custom_objects={'KerasLayer':hub.KerasLayer})

# Check its architecture
#new_model.summary()

# add all the classes in order
room_classes = ['bathroom','bedroom','conservatory','dining_room','entrance_hall_landing','front_of_house','garden','kitchen','living_room','pool', 'street_scape', 'study_office','utility_room']


data = Image.open('./quicktestimages/street_scape/image9.jpg')
data = np.array(data)
img = tf.image.resize(data,(224, 224))
img = tf.cast(img, tf.float32)
img = (img / 255.0)
# Do the prediction
x = np.array(img)
# trained on batches in four dimensions, so need to expand dimension of input image from three to four
x = np.expand_dims(x, axis=0)
prediction = new_model.predict(x)[0]

print(prediction)

highest_confidence = list(prediction).index(max(prediction))

room_class_response = {
  "highest_confidence": {
    "label": room_classes[highest_confidence],
    "confidence": str(round(max(prediction), 2))
  },
  "all_classes": []
}
for i, confidence in enumerate(prediction):
  room_class_response["all_classes"].append({
    "label": room_classes[i],
    "confidence": str(round(confidence, 2))
  })

print(room_class_response)
