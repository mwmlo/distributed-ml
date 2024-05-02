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
    
  multi_worker_model.fit(ds_train, epochs=1, steps_per_epoch=70)
  if is_chief():
    model_path = args.saved_model_dir
  else:
    model_path = args.saved_model_dir + '/worker_tmp_' + str(TASK_INDEX)
    
  multi_worker_model.save(model_path)

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