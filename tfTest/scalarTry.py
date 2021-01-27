from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime
from tensorflow import keras

import numpy as np

data_size = 1000
train_pct = 0.8


train_size = int(train_pct*data_size)
# Create some input data between -1 and 1 and randomize it.
x  = np.linspace(-1,1,data_size)
np.random.shuffle(x) #打乱
y = 0.5 * x+2+np.random.normal(0,0.05,(data_size,))

x_train,y_train = x[:train_size],y[:train_size]
x_test,y_test = x[train_size:],y[train_size:]
logdir = os.path.join("log")

print(logdir)
if not os.path.exists(logdir):
    os.mkdir(logdir)
# 使用tensorboard
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# 构建模型
model = keras.models.Sequential([
    keras.layers.Dense(16, input_dim=1),
    keras.layers.Dense(1),
])

model.compile(
    loss='mse', # keras.losses.mean_squared_error
    optimizer=keras.optimizers.SGD(lr=0.2),
)

print("Training ... With default parameters, this takes less than 10 seconds.")
training_history = model.fit(
    x_train, # input
    y_train, # output
    batch_size=train_size,
    verbose=0, # Suppress chatty output; use Tensorboard instead
    epochs=100,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_callback],
)

print("Average test loss: ", np.average(training_history.history['loss']))