import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from model import Model

model = Model()
model.make_network(False)
