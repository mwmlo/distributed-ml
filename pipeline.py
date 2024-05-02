import tensorflow_datasets as tfds
import tensorflow as tf

from models import *

strategy = tf.distribute.MultiWorkerMirroredStrategy
BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

def make_datasets_unbatched():
  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label
  datasets, _ = tfds.load(name='fashion_mnist',
    with_info=True, as_supervised=True)
  return datasets['train'].map(scale).cache().shuffle(10000)

with strategy.scope():
  # Create repeated batches of data
  ds_train = make_datasets_unbatched().batch(BATCH_SIZE).repeat()
  # Enable automatic data sharding: each workers trains on subset of dataset
  options = tf.data.Options()
  options.experimental_distribute.auto_shard_policy = \
    tf.data.experimental.AutoShardPolicy.DATA
  ds_train = ds_train.with_options(options)
  model = build_and_compile_model()
  