import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds

model = keras.models.load_model("trained_model/saved_model_versions")
def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255
  return image, label

datasets, _ = tfds.load(
  name='fashion_mnist', with_info=True, as_supervised=True)

ds = datasets['test'].map(scale).cache().shuffle(10000).batch(64)

model.predict(ds)