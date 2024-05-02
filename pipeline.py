import tensorflow_datasets as tfds
import tensorflow as tf
import os

from models import *

strategy = tf.distribute.MultiWorkerMirroredStrategy
decay = 0.001
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
  # model = build_and_compile_model()
  
  single_worker_model = build_and_compile_cnn_model()
  checkpoint_prefix = os.path.join("checkpoints", "ckpt_{epoch}")
  
  # Print learning rate at end of every epoch
  class PrintLR(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs=None):
        print('\nLearning rate for epoch {} is {}'.format(
        epoch + 1, single_worker_model.optimizer.lr.numpy()))
  
  callbacks = [
    # Start interactive visualization for training progress
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    # Save model weights for inference later
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                            save_weights_only=True),
    # Decay learning rate at end of every epoch
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR()
  ]
    
  single_worker_model.fit(ds_train,
                          epochs=1,
                          steps_per_epoch=70,
                          callbacks=callbacks)