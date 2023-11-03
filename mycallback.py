import tensorflow as tf
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.98):
            print("Training reached 95% let's stop training")
            self.model.stop_training = True