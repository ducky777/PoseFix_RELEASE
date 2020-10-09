#%%
import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Conv2D,\
    MaxPooling2D, Add

import numpy as np

x_img = np.random.uniform(size=(225, 225, 3))
x_pose_heatmap = np.random.uniform(size=(225, 225, 17))

input_img = Input(shape=(None, None, 3))
input_pose = Input(shape=(None, None, 17))
x = Concatenate(axis=-1)([input_img, input_pose])
# x = tf.keras.applications.efficientnet.preprocess_input(x)
x = Conv2D(64, 7, strides=2, padding='same')(x)
x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
x = MaxPooling2D([3, 3], strides=2, padding='valid')(x)

model = tf.keras.applications.EfficientNetB0(weights='imagenet',
                                             include_top=False)
# model.trainable = False

posefix = Concatenate(axis=-1)([model.input, x])
#%%
posefix_model = tf.keras.Model([input_img, input_pose], [posefix])
#%%
posefix
#%%
posefix
#%%
model.layers.pop()
#%%
model = model([x])
#%%

#%%