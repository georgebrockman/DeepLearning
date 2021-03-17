import tensorflow as tf
import numpy as np
from PIL import Image
import tensorflow_hub as hub

# load model
new_model = tf.keras.models.load_model('model/best_model_inex.h5', custom_objects={'KerasLayer':hub.KerasLayer})

# Check its architecture
#new_model.summary()

# add all the classes in order
room_classes = ['bathroom','bedroom','conservatory','dining_room','entrance_hall_landing','front_of_house','garden','kitchen','living_room','pool', 'street_scape', 'study_office','utility_room']
inex_classes = ['external', 'internal']

data = Image.open('./quicktestimages/conservatory/test_image9.jpg')
data = np.array(data)
img = tf.image.resize(data,(224, 224))
img = tf.cast(img, tf.float32)
img = (img / 255.0)
# Do the prediction
x = np.array(img)
# trained on batches in four dimensions, so need to expand dimension of input image from three to four
x = np.expand_dims(x, axis=0)
prediction = new_model.predict(x)[0]

pred_1 = new_model.predict(x)[1]

highest_confidence = [(list(prediction[0]).index(max(prediction[0]))), (list(pred_1[0]).index(max(pred_1[0])))]


room_class_response = {
  "highest_confidence": {
    "room_label": room_classes[int(highest_confidence[0])],
    "room_confidence": str(round(max(prediction[0]), 2)),
    "inex_label": inex_classes[int(highest_confidence[1])],
    "inex_confidence": str(round(max(pred_1[0]), 2)),
  },
  "all_classes": []
}
# for i, confidence in enumerate(prediction):
#   room_class_response["all_classes"].append({
#     "label": room_classes[i],
#     "confidence": str(round(confidence, 2))
#   })

print(room_class_response)
