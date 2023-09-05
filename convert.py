import tensorflow as tf

# Convert the model
model = tf.keras.models.load_model("model_snapshot_5.h5")
print('the model summary is:', model.summary)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)