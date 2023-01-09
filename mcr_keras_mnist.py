import os

# depending on the runing environments, you may not need these two lines
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['KMP_DUPLICATE_LIB_OK']='True'

'''Trains a simple convnet on the MNIST dataset using
[1] Maximal Correlation Regression: https://ieeexplore.ieee.org/abstract/document/8979352

Author: Xiangxiang Xu <xiangxiangxu.thu@gmail.com>

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).

With a comparison between Log-loss and H-score
'''

import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Lambda
from keras import backend as K
import numpy as np

# obtain the normalized feature by subtracting the mean
tilde = lambda x: x - K.mean(x, axis = 0)
# compute the covariance
cov = lambda x: K.dot(K.transpose(x), x) / K.cast(K.shape(x)[0] - 1, dtype = 'float32')

def neg_hscore(x):
    """
    Computation of negative hscore [1, Eq. (7)]
    """
    f = x[0]
    g = x[1]
    # subtract the mean
    f0 = tilde(f)
    g0 = tilde(g)
    # compute correlation
    corr = K.mean(K.sum(f0 * g0, axis = 1))
    # compute covariances 
    cov_f = cov(f)
    cov_g = cov(g)
    # negative H-score
    neg_h = - corr + K.sum(cov_f * cov_g) / 2
    return neg_h



def feature_f(input_x, fdim):
    """
    use a CNN to extract feature f [1, Figure 2]
    """
    conv1 = Conv2D(32, kernel_size=(3, 3),
                 activation='relu')(input_x)
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
    pool = MaxPooling2D(pool_size=(2, 2))(conv2)
    f = Dropout(0.25)(pool)
    f = Flatten()(f)
    f = Dense(fdim)(f)
    return f


"""
Preparation of the data
"""

# input image dimensions
img_rows, img_cols = 28, 28
num_classes = 10

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


# simple normalization
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# obtain one-hot encoded labels
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
Training Parameters
"""

batch_size = 128
epochs = 12


# dimension of features
fdim = 128
gdim = fdim

"""
Construct a baseline model, with softmax layer and log-loss for classification

The same network architecture is used in both log-loss and MCR for feature extraction
"""
input_x = Input(shape = input_shape)
f_log = feature_f(input_x, fdim)
predictions = Dense(num_classes, activation='softmax')(f_log)
model_log = Model(inputs=input_x, outputs=predictions)

model_log.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
print('# of Paras', model_log.count_params())
# train the log-loss model
model_log.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(x_test, y_test))
# accuracy of the log-loss model
acc_log = model_log.evaluate(x_test, y_test, verbose=0)[1]
print('acc_log = ', acc_log)
model_log_f = Model(inputs = model_log.input, outputs = model_log.layers[-1].input)


f_test_log = model_log_f.predict(x_test)


"""
MCR model: training and testing
with the network architecture shown in [1, Figure 6]
"""
input_x = Input(shape = input_shape)
f = feature_f(input_x, fdim)

# one-hot input
input_y = Input(shape = (num_classes, ))

g = Dense(gdim)(input_y) 
# using one embedding layer to generate feature g of y, (weights)
# Note that g should not be activated, as the linear layer already has full capbablity in express any feature of input

loss = Lambda(neg_hscore)([f, g])
model = Model(inputs = [input_x, input_y], outputs = loss)
model.compile(optimizer=keras.optimizers.Adadelta(), loss = lambda y_true,y_pred: y_pred)
# train the MCR model
model.fit([x_train, y_train],
          np.zeros([y_train.shape[0], 1]),
          batch_size = batch_size,
          epochs = epochs,
          validation_data=([x_test, y_test], np.zeros([y_test.shape[0], 1])))#validation_split = 0.2)

# define new models to obtain the trained feature f and g in MCR
model_f = Model(inputs = input_x, outputs = f)
model_g = Model(inputs = input_y, outputs = g)


f_test = model_f.predict(x_test)
# normalize: subtract the mean
f_test0 = f_test - np.mean(f_test, axis = 0)
g_val = model_g.predict(np.eye(10))

# compute py: empirical distribution of y
## you can simply use uniform distribution for balanced dataset
py = np.mean(y_train, axis = 0)
# normalize: subtract the mean
g_val0 = g_val - np.matmul(py, g_val) 

# prediction of MCR [1, Eq. (2)]
pygx = py * (1 + np.matmul(f_test0, g_val0.T))

# accuracy of MCR
acc_mcr = np.mean(np.argmax(pygx, axis = 1) == np.argmax(y_test, axis = 1))
print('acc_mcr = ', acc_mcr)
