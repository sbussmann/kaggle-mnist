
# coding: utf-8

# In[19]:

from __future__ import absolute_import
from __future__ import print_function
#from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np


# In[2]:

batch_size = 32
nb_classes = 10
nb_epoch = 3
data_augmentation = False


# In[3]:

# shape of the image (SHAPE x SHAPE)
shapex, shapey = 28, 28
# number of convolutional filters to use at each layer
nb_filters = [32, 64]
# level of pooling to perform at each layer (POOL x POOL)
nb_pool = [2, 2]
# level of convolution to perform at each layer (CONV x CONV)
nb_conv = [3, 3]
# the MNIST images are greyscale
image_dimensions = 1


# In[10]:

train = pd.read_csv('../Data/train.csv')


# In[16]:

Xcol = train.columns[1:]
ycol = 'label'
features = train[Xcol].values
labels = train[ycol].values


# In[18]:

# the data, shuffled and split between tran and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# In[69]:

# turn 1D feature vector into 3D cube
def makeImage(vectors):
    images = []
    nsamples = len(vectors[:, 0])
    for ivector in range(nsamples):
        vector = vectors[ivector, :]
        npixels = len(vector)
        nx = int(np.sqrt(npixels))
        ny = int(np.sqrt(npixels))
        image = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                image[j, i] = vector[j * nx + i]
        images.append(image)
    images = np.array(images)
    images = np.reshape(images, (nsamples, 1, nx, ny))
    return images


# In[70]:

X_train_image = makeImage(X_train)


# In[72]:

X_test_image = makeImage(X_test)


# In[74]:

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


# In[75]:

y_train.shape, Y_train.shape


# In[76]:

model = Sequential()


# In[77]:

model.add(Convolution2D(nb_filters[0], image_dimensions, nb_conv[0], nb_conv[0], border_mode='full'))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters[0], nb_filters[0], nb_conv[0], nb_conv[0]))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(nb_pool[0], nb_pool[0])))
model.add(Dropout(0.25))


# In[78]:

model.add(Convolution2D(nb_filters[1], nb_filters[0], nb_conv[0], nb_conv[0], border_mode='full'))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters[1], nb_filters[1], nb_conv[1], nb_conv[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(nb_pool[1], nb_pool[1])))
model.add(Dropout(0.25))


# In[79]:

model.add(Flatten())


# In[80]:

# the image dimensions are the original dimensions divided by any pooling
# each pixel has a number of filters, determined by the last Convolution2D layer
model.add(Dense(nb_filters[-1] * (shapex / nb_pool[0] / nb_pool[1]) * (shapey / nb_pool[0] / nb_pool[1]), 512))
model.add(Activation('relu'))
model.add(Dropout(0.5))


# In[81]:

model.add(Dense(512, nb_classes))
model.add(Activation('softmax'))


# In[82]:

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)


# In[ ]:

if not data_augmentation:
    print("Not using data augmentation or normalization")

    X_train_image = X_train_image.astype("float32")
    X_test_image = X_test_image.astype("float32")
    X_train_image /= 255
    X_test_image /= 255
    model.fit(X_train_image, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True)
    score = model.evaluate(X_test_image, Y_test, batch_size=batch_size)
    print('Test score:', score)

else:
    print("Using real time data augmentation")

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    for e in range(nb_epoch):
        print('-'*40)
        print('Epoch', e)
        print('-'*40)
        print("Training...")
        # batch train with realtime data augmentation
        progbar = generic_utils.Progbar(X_train.shape[0])
        for X_batch, Y_batch in datagen.flow(X_train, Y_train):
            loss = model.train_on_batch(X_batch, Y_batch)
            progbar.add(X_batch.shape[0], values=[("train loss", loss)])

        print("Testing...")
        # test time!
        progbar = generic_utils.Progbar(X_test.shape[0])
        for X_batch, Y_batch in datagen.flow(X_test, Y_test):
            score = model.test_on_batch(X_batch, Y_batch)
            progbar.add(X_batch.shape[0], values=[("test loss", score)])


# In[ ]:




test = pd.read_csv('../Data/test.csv')


test_image = makeImage(test.values)

ypredict = model.predict(test_image, batch_size=batch_size)

dfpredict = pd.DataFrame(ypredict)
dfpredict.columns = ['Label']
dfpredict['ImageId'] = np.arange(28000) + 1
dfpredict.to_csv('../Data/predict_CNNbenchmark.csv', index=False)

