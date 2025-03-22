import sys, os, time, re, gc
from pathlib import Path
from glob import glob
import random, hashlib, json, copy, h5py, pickle

# Select GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_DETERMINISTIC_OPS'] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import resnet

histories = []

os.environ['TF_DETERMINISTIC_OPS'] = '0'
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        start = time.time()
        self.model.evaluate(x_test, y_test,verbose=0,batch_size=128)
        print('Test seconds: ',time.time()-start)

class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        mark = 0
        duration = 0
    def on_epoch_begin(self, epoch, logs=None):
        self.mark = time.time()
    def on_epoch_end(self, epoch, logs=None):
        self.duration = time.time() - self.mark
        print('Train seconds: ',self.duration)

class TestTimeCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(TestTimeCallback, self).__init__()
        self.test_times = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        start = time.time()
        self.model.evaluate(x_test, y_test, verbose=0, batch_size=128)
        test_time = time.time() - start
        self.test_times.append(test_time)
        logs['test_time'] = test_time
        print(f'Test seconds: {test_time:.2f}')

class TrainTimeCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(TrainTimeCallback, self).__init__()
        self.train_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.mark = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_time = time.time() - self.mark
        self.train_times.append(train_time)
        logs['train_time'] = train_time
        print(f'Train seconds: {train_time:.2f}')

def preprocess_data(X, Y):
  X_p = tf.keras.applications.resnet.preprocess_input(X)
  # encode to one-hot
  Y_p = tf.keras.utils.to_categorical(Y, 10) #cifar 10 için 2. parametre 10 olmalı. 100 için 100
  return X_p, Y_p

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.cifar10.load_data() #cifar 10 için cifar10 yaz, 100 için 100

# pre-procces data
x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)
print(x_train.shape,y_train.shape)
print(type(x_train.shape),type(y_train))

from tensorflow.keras.models import clone_model

def top10_trainable(model):
  for layer in model.layers:
    layer.trainable = False
  conv_count=0
  for layer in model.layers[::-1]:
    if (isinstance(layer, tf.keras.layers.Conv2D)):
      conv_count+=1
    if conv_count <= 9:
      layer.trainable = True
      if conv_count == 9:
        break
  return model

resnet50 = tf.keras.applications.ResNet50(include_top=False,
                                          weights='imagenet',
                                          input_shape=(160,160,3))

resnet50_weight = {i.name:i.get_weights() for i in resnet50.layers}

new_input = tf.keras.Input(shape=(32, 32, 3),name="")

# upscale layer
x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x,[160,160]))(new_input)
x = top10_trainable(resnet50)(x)

model = tf.keras.models.Model(inputs=new_input, outputs=x)
out = model.output
out = tf.keras.layers.GlobalAveragePooling2D()(out)
#initializer = tf.keras.initializers.GlorotUniform(seed=SEED)

out = tf.keras.layers.Dense(10, activation='softmax')(out) #cifar 10 için 1. parametre 10, 100 için 100

model = tf.keras.models.Model(inputs=model.input, outputs=out)

my_val_callback = MyCustomCallback()
timetaken = timecallback()

model.layers[-1].trainable = True
#for layer in model.layers[:-2]:#
  #layer.trainable = False

model.summary()

def plot_loss_accuracy(history):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(history.history["loss"],'r-x', label="Train Loss")
    ax.plot(history.history["val_loss"],'b-x', label="Validation Loss")
    ax.legend()
    ax.set_title('cross_entropy loss')
    ax.grid(True)


    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history.history["accuracy"],'r-x', label="Train Accuracy")
    ax.plot(history.history["val_accuracy"],'b-x', label="Validation Accuracy")
    ax.legend()
    ax.set_title('accuracy')
    ax.grid(True)
    plt.show()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
from tensorflow.keras.callbacks import Callback

class ValidationBatchSizeCallback(Callback):
    def on_test_batch_begin(self, batch, logs=None):
        print(f"Validation Batch {batch} started")
validation_batch_callback = ValidationBatchSizeCallback()

train_callback = TrainTimeCallback()
test_callback = TestTimeCallback()

history = model.fit(x=x_train,
          y=y_train,
          batch_size=128,
          epochs=10,
          callbacks=[train_callback, test_callback,validation_batch_callback],verbose=1,
          validation_data=(x_test, y_test))

histories.append(history)

plot_loss_accuracy(history)

def calculate_total_times(train_callback, test_callback):
    total_train_time = sum(train_callback.train_times)
    total_test_time = sum(test_callback.test_times)
    avg_train_time = total_train_time / len(train_callback.train_times)
    avg_test_time = total_test_time / len(test_callback.test_times)

    return {
        'total_train_time': total_train_time,
        'total_test_time': total_test_time,
        'avg_train_time': avg_train_time,
        'avg_test_time': avg_test_time
    }

times = calculate_total_times(train_callback, test_callback)
print(f"\nTotal training time: {times['total_train_time']:.2f} seconds")
print(f"Total test time: {times['total_test_time']:.2f} seconds")
print(f"Average training time per epoch: {times['avg_train_time']:.2f} seconds")
print(f"Average test time per epoch: {times['avg_test_time']:.2f} seconds")
