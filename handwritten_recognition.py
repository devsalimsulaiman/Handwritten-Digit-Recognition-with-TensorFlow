#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
from matplotlib import pyplot as plt
import numpy as np


# In[4]:


(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = datasets.mnist.load_data()


# In[5]:


print(y_train_raw[0])
print(x_train_raw.shape, y_train_raw.shape)
print(x_test_raw.shape, y_test_raw.shape)


# In[6]:


#Convert the labels into one-hot codes.
num_classes = 10
y_train = keras.utils.to_categorical(y_train_raw, num_classes)
y_test = keras.utils.to_categorical(y_test_raw, num_classes)
print(y_train[0])


# In[24]:


# In the MNIST dataset, the images are a tensor in the shape of [60000, 28, 28]. The first dimension is used to extract images, and the second and third dimensions are used to extract pixels in each image. Each element in this tensor indicates the strength of a pixel in an image. The value ranges from 0 to 255. Label data is converted from scalar to one-hot vectors. In a one-hot vector, one digit is 1, and digits in other dimensions are all 0s. For example, label 1 may be represented as [0,1,0,0,0,0,0,0,0,0,0,0]. Therefore, the labels are a digital matrix of [60000, 10]


# In[11]:


# draw the first 9 images
plt.figure()
for i in range(9):
 plt.subplot(3,3,i+1)
 plt.imshow(x_train_raw[i])
 #plt.ylabel(y[i].numpy())
 plt.axis('off')
plt.show()


# In[12]:


#Convert a 28 x 28 image into a 784 x 1 vector.
x_train = x_train_raw.reshape(60000, 784)
x_test = x_test_raw.reshape(10000, 784)


# In[13]:


#Normalize image pixel values.
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255


# ## Building a DNN Model
# 

# In[15]:


#Create a deep neural network (DNN) model that consists of three fully connected layers and two RELU activation functions.
model = keras.Sequential([
 layers.Dense(512, activation='relu', input_dim = 784),
 layers.Dense(256, activation='relu'),
 layers.Dense(124, activation='relu'),
layers.Dense(num_classes, activation='softmax')])
model.summary()


# In[16]:


# layer.Dense() indicates a fully connected layer, and activation indicates a used activation function.


# In[17]:


# Compiling the DNN Model
Optimizer = optimizers.Adam(0.001)
model.compile(loss=keras.losses.categorical_crossentropy,
 optimizer=Optimizer,
 metrics=['accuracy'])


# In[18]:


# Training the DNN Model


# In[19]:


#Fit the training data to the model by using the fit method.
model.fit(x_train, y_train,
 batch_size=128,
 epochs=10,
 verbose=1)


# In[22]:


#  Evaluating the DNN Model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# ************
print("the evaluation shows that the model accuracy reaches ",score[1]," and 10 training iterations have been performed")


# In[23]:


# Saving the DNN Model.. Create model folder under relative path.
model.save('./model/final_DNN_model.h5')


# In[25]:


# CNN Construction
# The conventional CNN construction method helps you better understand the internal network structure but has a large code volume. Therefore, attempts to construct a CNN by using high-level APIs are made to simplify the network construction process.
import tensorflow as tf
from tensorflow import keras
import numpy as np
model=keras.Sequential() #Create a network sequence.
##Add the first convolutional layer and pooling layer.
model.add(keras.layers.Conv2D(filters=32,kernel_size = 5,strides = (1,1),
 padding = 'same',activation = tf.nn.relu,input_shape = (28,28,1)))
model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides = (2,2), padding = 'valid'))
##Add the second convolutional layer and pooling layer.
model.add(keras.layers.Conv2D(filters=64,kernel_size = 3,strides = (1,1),padding = 'same',activation =
tf.nn.relu))
model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides = (2,2), padding = 'valid'))
##Add a dropout layer to reduce overfitting.
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
##Add two fully connected layers.
model.add(keras.layers.Dense(units=128,activation = tf.nn.relu))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(units=10,activation = tf.nn.softmax))


# In[26]:


# compiling and Training the CNN Model
#Expand data dimensions to adapt to the CNN model.
X_train=x_train.reshape(60000,28,28,1)
X_test=x_test.reshape(10000,28,28,1)
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])
model.fit(x=X_train,y=y_train,epochs=5,batch_size=128)


# In[27]:


# Evaluating the CNN Model
test_loss,test_acc=model.evaluate(x=X_test,y=y_test)
print("Test Accuracy %.2f"%test_acc)

# in real life project, better to check for accuracy % and proceed if better accuracy..


# In[28]:


# saving the CNN model
model.save('./model/final_CNN_model.h5')


# In[29]:


# Loading the CNN Model
from tensorflow.keras.models import load_model
new_model = load_model('./model/final_CNN_model.h5')
new_model.summary()


# In[41]:


# visualize prediction results
#Visualize test set output results.
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
def res_Visual(n):
    final_opt_a=new_model.predict(X_test[0:n])#Perform predictions on the test set by using the model. in lesser version of tensorflow it is model.predict_classes
    fig, ax = plt.subplots(nrows=int(n/5),ncols=5 )
    ax = ax.flatten()
    print('prediction results of the first {} images:'.format(n))
    for i in range(n):
        print(final_opt_a[i],end=',')
        if int((i+1)%5) ==0:
            print('\t')
        #Visualize image display.
        img = X_test[i].reshape((28,28))#Read each row of data in the format of Ndarry.
        plt.axis("off")
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')#Visualization
        ax[i].axis("off")
    print('first {} images in the test set:'.format(n))
    
res_Visual(20)

