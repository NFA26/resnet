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
        # Access the model explicitly via the 'model' attribute
        start = time.time()
        # Ensure 'model' is correctly accessed and callable
        if hasattr(self, 'model') and self.model is not None:
            self.model.evaluate(x_test, y_test, verbose=0, batch_size=128)
            print('Test seconds: ', time.time() - start)
        else:
            print('Model is not available for evaluation.')

class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.mark = 0
        self.duration = 0
    def on_epoch_begin(self, epoch, logs=None):
        self.mark = time.time()
    def on_epoch_end(self, epoch, logs=None):
        self.duration = time.time() - self.mark
        print('Train seconds: ', self.duration)

class TestTimeCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(TestTimeCallback, self).__init__()
        self.test_times = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        start = time.time()
        if hasattr(self, 'model') and self.model is not None:
            self.model.evaluate(x_test, y_test, verbose=0, batch_size=128)
            test_time = time.time() - start
            self.test_times.append(test_time)
            logs['test_time'] = test_time
            print(f'Test seconds: {test_time:.2f}')
        else:
            print('Model is not available for evaluation.')

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
    Y_p = tf.keras.utils.to_categorical(Y, 10)  # 10 for CIFAR-10
    return X_p, Y_p

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()  # CIFAR-10

# Pre-process data
x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)

from tensorflow.keras.models import clone_model

def top10_trainable(model):
    for layer in model.layers:
        layer.trainable = False
    conv_count = 0
    for layer in model.layers[::-1]:
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_count += 1
        if conv_count <= 9:
            layer.trainable = True
            if conv_count == 9:
                break
    return model

resnet50 = tf.keras.applications.ResNet50(include_top=False,
                                          weights='imagenet',
                                          input_shape=(160, 160, 3))

resnet50_weight = {i.name: i.get_weights() for i in resnet50.layers}

new_input = tf.keras.Input(shape=(32, 32, 3), name="")

# Upscale layer
x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, [160, 160]))(new_input)
x = top10_trainable(resnet50)(x)

model = tf.keras.models.Model(inputs=new_input, outputs=x)
out = model.output
out = tf.keras.layers.GlobalAveragePooling2D()(out)
out = tf.keras.layers.Dense(10, activation='softmax')(out)  # CIFAR-10

model = tf.keras.models.Model(inputs=model.input, outputs=out)

my_val_callback = MyCustomCallback()
timetaken = timecallback()

model.layers[-1].trainable = True
model.summary()

def plot_loss_accuracy(history):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(history['loss'], 'r-x', label="Train Loss")
    ax.plot(history['val_loss'], 'b-x', label="Validation Loss")
    ax.legend()
    ax.set_title('Cross-Entropy Loss')
    ax.grid(True)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history['accuracy'], 'r-x', label="Train Accuracy")
    ax.plot(history['val_accuracy'], 'b-x', label="Validation Accuracy")
    ax.legend()
    ax.set_title('Accuracy')
    ax.grid(True)
    plt.show()

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

class ValidationBatchSizeCallback(tf.keras.callbacks.Callback):
    def on_test_batch_begin(self, batch, logs=None):
        print(f"Validation Batch {batch} started")
validation_batch_callback = ValidationBatchSizeCallback()

# Custom training loop
EPOCHS = 10
BATCH_SIZE = 128
train_callback = TrainTimeCallback()
test_callback = TestTimeCallback()

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

history_log = {
    'loss': [],
    'accuracy': [],
    'val_loss': [],
    'val_accuracy': [],
    'train_time': [],
    'test_time': []
}

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    
    # ---------- Test Phase ----------
    test_callback.on_epoch_end(epoch)
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    history_log['val_loss'].append(test_loss)
    history_log['val_accuracy'].append(test_accuracy)
    
    # ---------- Train Phase ----------
    train_callback.on_epoch_begin(epoch)
    epoch_loss = []
    epoch_accuracy = []

    for step, (x_batch, y_batch) in enumerate(train_dataset):
        metrics = model.train_on_batch(x_batch, y_batch)
        epoch_loss.append(metrics[0])
        epoch_accuracy.append(metrics[1])
        
    train_callback.on_epoch_end(epoch)
    
    avg_train_loss = np.mean(epoch_loss)
    avg_train_accuracy = np.mean(epoch_accuracy)
    
    history_log['loss'].append(avg_train_loss)
    history_log['accuracy'].append(avg_train_accuracy)
    history_log['train_time'].append(train_callback.train_times[-1])
    history_log['test_time'].append(test_callback.test_times[-1])

    print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}")
    print(f"Train seconds: {train_callback.train_times[-1]:.2f}")
    print(f"Test seconds: {test_callback.test_times[-1]:.2f}")

# Plot loss and accuracy
plot_loss_accuracy(history_log)

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
