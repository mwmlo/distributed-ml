import os
from models import *
import tensorflow as tf

def train_local():
    single_worker_model = build_and_compile_cnn_model()
    checkpoint_prefix = os.path.join(args.checkpoint_dir, "ckpt_{epoch}")
    
    class PrintLR(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
                    print('\nLearning rate for epoch {} is {}'.format(
                    epoch + 1, multi_worker_model.optimizer.lr.numpy()))
        callbacks = [
                    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
                    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                            save_weights_only=True),
                    tf.keras.callbacks.LearningRateScheduler(decay),
                    PrintLR()
        ]
    
    single_worker_model.fit(ds_train,
                            epochs=1,
                            steps_per_epoch=70,
                            callbacks=callbacks)