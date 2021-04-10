# -*- coding: utf-8 -*-
"""
Implement a few models for parameter inference.

This module implements models to be used for analysis of strong
lensing parameters.
"""

from . import manual_keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def build_resnet_50(img_size,num_outputs):
	"""	Build the traditional resnet 50 model.

	Args:
		img_size ((int,int,int)): A tuple with shape (pix,pix,freq) that
			describes the size of the input images.
		num_outputs (int): The number of outputs to predict

	Returns:
		(keras.Model): An instance of the ResNet50 model implemented in
			Keras.
	"""
	# Set some of the parameters for resnet 50
	use_bias = True
	# Initialzie the inputs
	inputs = layers.Input(shape=img_size)

	# Build the first resnet stack
	x = layers.ZeroPadding2D(padding=((3,3),(3,3)),name='conv1_pad')(inputs)
	x = layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)
	x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
	x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

	# Apply the resnet stacks
	x = manual_keras.stack1(x, 64, 3, stride1=1, name='conv2')
	x = manual_keras.stack1(x, 128, 4, name='conv3')
	x = manual_keras.stack1(x, 256, 6, name='conv4')
	x = manual_keras.stack1(x, 512, 3, name='conv5')

	# Pass the output to the fully connected layer
	x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
	outputs = layers.Dense(num_outputs, name='output')(x)

	# Construct the model
	model = Model(inputs=inputs,outputs=outputs)

	return model
