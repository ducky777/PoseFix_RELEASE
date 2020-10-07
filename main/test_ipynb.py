#%%
import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Conv2D,\
    MaxPooling2D

import numpy as np

x_img = np.random.uniform(size=(224, 224, 3))
x_pose_heatmap = np.random.uniform(size=(224, 224, 17))

input_img = Input(shape=(None, None, 3))
input_pose = Input(shape=(None, None, 17))
x = Concatenate(axis=-1)([input_img, input_pose])
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = Conv2D(64, 7, strides=2, padding='same')(x)
x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
x = MaxPooling2D([3, 3], strides=2, padding='valid')(x)

model = tf.keras.applications.EfficientNetB0(weights='imagenet',
                                             include_top=False,
                                             input_tensor=x)
# model.trainable = False

# x = Concatenate(axis=-1)([model.output, input_pose])
# model = model([x])
#%%
model.output
#%%
model.layers.pop()
#%%
model = model([x])
#%%

#%%