import argparse
import tensorflow_datasets as tfds
import tensorflow as tf
import os
import json

from models import *

strategy = tf.distribute.MultiWorkerMirroredStrategy(
  communication_options=tf.distribute.experimental.CommunicationOptions(
  implementation=tf.distribute.experimental.CollectiveCommunication.AUTO))

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

def is_chief():
  return TASK_INDEX == 0
  
tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
TASK_INDEX = tf_config['task']['index']

def _preprocess(bytes_inputs):
    decoded = tf.io.decode_jpeg(bytes_inputs, channels=1)
    resized = tf.image.resize(decoded, size=(28, 28))
    return tf.cast(resized, dtype=tf.uint8)

def _get_serve_image_fn(model):
    @tf.function(input_signature=[tf.TensorSpec([None], dtype=tf.string, name='image_bytes')])
    def serve_image_fn(bytes_inputs):
        decoded_images = tf.map_fn(_preprocess, bytes_inputs, dtype=tf.uint8)
        return model(decoded_images)
    return serve_image_fn

def main(args):
  with strategy.scope():
    # Create repeated batches of data
    ds_train = make_datasets_unbatched().batch(BATCH_SIZE).repeat()
    # Enable automatic data sharding: each workers trains on subset of dataset
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = \
      tf.data.experimental.AutoShardPolicy.DATA
    ds_train = ds_train.with_options(options)
    
    if args.model_type == "cnn":
      multi_worker_model = build_and_compile_cnn_model()
    elif args.model_type == "dropout":
      multi_worker_model = build_and_compile_cnn_model_with_dropout()
    elif args.model_type == "batch_norm":
      multi_worker_model = build_and_compile_cnn_model_with_batch_norm()
    else:
      raise Exception("Unsupported model type: %s" % args.model_type)
  
  # Define the checkpoint directory to store the checkpoints
  checkpoint_dir = args.checkpoint_dir
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                        save_weights_only=True),
    tf.keras.callbacks.LearningRateScheduler(decay),
    PrintLR()
  ]

  multi_worker_model.fit(ds_train,
                         epochs=1,
                         steps_per_epoch=70,
                         callbacks=callbacks)
  
  # Save model on chief worker only
  if is_chief():
    model_path = args.saved_model_dir
  else:
    model_path = args.saved_model_dir + '/worker_tmp_' + str(TASK_INDEX)
  
  multi_worker_model.save(model_path)

  # Define input signature
  signatures = {
    "serving_default": _get_serve_image_fn(multi_worker_model).get_concrete_function(
        tf.TensorSpec(shape=[None], dtype=tf.string, name='image_bytes')
    )
  }
  tf.saved_model.save(multi_worker_model, model_path, signatures=signatures)

if __name__ == '__main__':
  tfds.disable_progress_bar()
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--saved_model_dir',
                         type=str,
                         required=True,
                         help='Tensorflow export directory.')
  
  parser.add_argument('--checkpoint_dir',
                         type=str,
                         required=True,
                         help='Tensorflow checkpoint directory.')
  
  parser.add_argument('--model_type',
                         type=str,
                         required=True,
                         help='Type of model to train.')
  
  parsed_args = parser.parse_args()
  main(parsed_args)