import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class generator_Input(tf.keras.Model):
  def __init__(self, shape):
    super(generator_Input, self).__init__()
    self.dense = layers.Dense(shape[0] * shape[1] * shape[2], use_bias=False)
    self.reshape = layers.Reshape(shape)
    self.bn = layers.BatchNormalization(momentum=0.9)
    self.relu = tf.keras.layers.ReLU()
  def call(self, x):
    x = self.dense(x)
    x = self.reshape(x)
    x = self.bn(x)
    x = self.relu(x)
    return x

class generator_Middle(tf.keras.Model):
  def __init__(self, filters, strides):
      super(generator_Middle, self).__init__()
      self.conv = layers.Conv2DTranspose(filters, (5, 5), strides=strides, padding='same', use_bias=False)
      self.bn = layers.BatchNormalization(momentum=0.9)
      self.relu = tf.keras.layers.ReLU()
  def call(self, x):
      x = self.conv(x)
      x = self.bn(x)
      x = self.relu(x)
      return x

class generator_Output(tf.keras.Model):
  def __init__(self, image_depth, strides):
    super(generator_Output, self).__init__()
    self.conv = layers.Conv2DTranspose(image_depth, (5, 5), strides=strides, padding='same', use_bias=False, activation='tanh')
    # self.actv = layers.Activation(activation='tanh')
  def call(self, x):
    x = self.conv(x)
    # x = self.actv(x)
    return x



class discriminator_Input(tf.keras.Model):
  def __init__(self, filters, strides):
    super(discriminator_Input, self).__init__()
    self.conv = keras.layers.Conv2D(filters, kernel_size=5, strides=strides, padding="same")
    self.leakyRelu = keras.layers.LeakyReLU(alpha=0.2)
    self.dropout = keras.layers.Dropout(0.3)

  def call(self, x):
    x = self.conv(x)
    x = self.leakyRelu(x)
    x = self.dropout(x)
    return x

class discriminator_Middle(tf.keras.Model):
  def __init__(self, filters, strides):
      super(discriminator_Middle, self).__init__()
      self.conv = tf.keras.layers.Conv2D(filters, kernel_size=5, strides=strides, padding="same")
      self.bn = tf.keras.layers.BatchNormalization(momentum=0.9)
      self.leakyRelu = tf.keras.layers.LeakyReLU(alpha=0.2)
      self.dropout = tf.keras.layers.Dropout(0.3)

  def call(self, x):
      x = self.conv(x)
      x = self.bn(x)
      x = self.leakyRelu(x)
      x = self.dropout(x)
      return x

class discriminator_Output_channel(tf.keras.Model):
  def __init__(self, filters, with_activation):
      super(discriminator_Output_channel, self).__init__()
      self.conv = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=1, padding="same")
      self.flatten = tf.keras.layers.Flatten()
      if with_activation:
        self.dense = tf.keras.layers.Dense(units=1, activation='sigmoid')
      else:
        self.dense = tf.keras.layers.Dense(1)

  def call(self, x):
      x = self.conv(x)
      x = self.flatten(x)
      x = self.dense(x)
      return x

class discriminator_Output(tf.keras.Model):
  def __init__(self, with_activation):
      super(discriminator_Output, self).__init__()
      self.flatten = tf.keras.layers.Flatten()
      if with_activation:
        self.dense = tf.keras.layers.Dense(units=1, activation='sigmoid')
      else:
        self.dense = tf.keras.layers.Dense(1)

  def call(self, x):
      x = self.flatten(x)
      x = self.dense(x)
      return x







